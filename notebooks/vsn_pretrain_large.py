# %% [markdown]
# # VSN Large Pre-Training: 200-700M params, Byte-level, Multi-dataset
#
# **Modelo**: VSN Hybrid Real (~350M params), d=512, 12 layers
# **Tokenización**: Byte-level (256 vocab, sin tokenizer externo)
# **Datos**: Mix de múltiples datasets (1-2M samples)
# **Secuencia**: 2048 bytes
# **Hardware**: 2× T4 (Kaggle), AMP fp16, DataParallel
# **Training**: 2 épocas, LR alto primera media → refinamiento después
# **Memoria infinita**: Estado latente H acumula entre ventanas

# %%
# !pip install vsn-framework datasets tqdm --quiet

# %%
import os, time, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Device: {device} | GPUs: {n_gpus}")
for i in range(n_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB)")

# Byte-level: vocab = 256 bytes + 1 padding token
VOCAB_SIZE = 257  # 0-255 = bytes, 256 = PAD/EOT
PAD_TOKEN = 256
SEQ_LEN = 2048  # bytes per sequence
print(f"Byte-level tokenizer: vocab={VOCAB_SIZE}, seq={SEQ_LEN}")

# %% [markdown]
# ## Dataset: Multi-source Mix (1-2M samples)

# %%
from datasets import load_dataset

def bytes_encode(text: str, max_len: int = SEQ_LEN) -> list:
    """Encode text to bytes (UTF-8). Fast, no external tokenizer."""
    b = list(text.encode('utf-8', errors='replace'))[:max_len]
    return b

def load_and_sample(name, config=None, split="train", n=None, text_field="text"):
    """Load a dataset and sample n examples."""
    try:
        if config:
            ds = load_dataset(name, config, split=split, streaming=True)
        else:
            ds = load_dataset(name, split=split, streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if n and i >= n: break
            text = item.get(text_field, "")
            if len(text) > 50:  # filter too short
                samples.append(text)
        print(f"  ✓ {name}: {len(samples):,} samples")
        return samples
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return []

print("Loading datasets (streaming, sampled)...\n")

all_texts = []

# English general
all_texts += load_and_sample("HuggingFaceFW/fineweb-2", "eng_Latn", n=300000)

# Spanish
all_texts += load_and_sample("HuggingFaceFW/fineweb-2", "spa_Latn", n=200000)

# Code
all_texts += load_and_sample("bigcode/the-stack-smol", "python", n=150000, text_field="content")

# Math
all_texts += load_and_sample("HuggingFaceTB/finemath", "finemath-4plus", n=150000)

# General knowledge / reasoning
all_texts += load_and_sample("HuggingFaceTB/smoltalk", "all", n=200000, text_field="content")

# ── Benchmark/evaluation datasets reformatted for pre-training ──

def format_humaneval(item):
    """HumanEval: prompt + canonical_solution as training text."""
    return item.get("prompt", "") + "\n" + item.get("canonical_solution", "")

def format_mbpp(item):
    """MBPP: text description + code."""
    return f"# {item.get('text', '')}\n{item.get('code', '')}"

def format_apps(item):
    """APPS: problem + solution."""
    q = item.get("question", "")
    sols = item.get("solutions", "")
    if isinstance(sols, str) and sols.startswith("["):
        try:
            import json
            sols = json.loads(sols)[0] if sols != "[]" else ""
        except: pass
    return f"Problem: {q}\nSolution:\n{sols}" if sols else ""

def format_mmlu(item):
    """MMLU: question + choices + answer formatted as text."""
    q = item.get("question", "")
    choices = item.get("choices", [])
    answer = item.get("answer", 0)
    text = f"Question: {q}\n"
    for i, c in enumerate(choices):
        text += f"{'ABCD'[i]}) {c}\n"
    if isinstance(answer, int) and answer < len(choices):
        text += f"Answer: {'ABCD'[answer]}) {choices[answer]}"
    return text

def format_gsm8k(item):
    """GSM8K: question + answer (chain of thought)."""
    return f"Question: {item.get('question', '')}\nAnswer: {item.get('answer', '')}"

# Load benchmark datasets
def load_benchmark(name, config, formatter, n=None, split="train"):
    try:
        ds = load_dataset(name, config, split=split, streaming=True, trust_remote_code=True)
        samples = []
        for i, item in enumerate(ds):
            if n and i >= n: break
            text = formatter(item)
            if len(text) > 50:
                samples.append(text)
        print(f"  ✓ {name}: {len(samples):,} samples (benchmark)")
        return samples
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return []

print("\n── Benchmark datasets (reformatted for pre-training) ──")
all_texts += load_benchmark("openai/openai_humaneval", None, format_humaneval, split="test")
all_texts += load_benchmark("google-research-datasets/mbpp", "full", format_mbpp, n=1000)
all_texts += load_benchmark("codeparrot/apps", "introductory", format_apps, n=5000, split="train")
all_texts += load_benchmark("cais/mmlu", "all", format_mmlu, n=10000, split="auxiliary_train")
all_texts += load_benchmark("openai/gsm8k", "main", format_gsm8k, n=8000, split="train")

# If some datasets failed, fill with more fineweb
if len(all_texts) < 800000:
    print(f"\n  Supplementing... (have {len(all_texts):,})")
    all_texts += load_and_sample("roneneldan/TinyStories", n=500000 - max(0, len(all_texts) - 800000))

random.seed(42)
random.shuffle(all_texts)
all_texts = all_texts[:1500000]  # Cap at 1.5M
print(f"\nTotal: {len(all_texts):,} samples")

# %% [markdown]
# ## Preparar datos (byte chunks)

# %%
class ByteChunkDataset(Dataset):
    """Pre-tokeniza todo a bytes y corta en chunks de SEQ_LEN."""
    
    def __init__(self, texts, seq_len=2048, max_chunks=2000000):
        print("  Encoding to bytes...")
        t0 = time.time()
        
        # Encode all to one big byte array (fast)
        all_bytes = bytearray()
        for i, text in enumerate(texts):
            all_bytes.extend(text.encode('utf-8', errors='replace'))
            all_bytes.append(PAD_TOKEN % 256)  # separator
            if i % 200000 == 0 and i > 0:
                print(f"    {i:,}/{len(texts):,} texts encoded...")
        
        total_bytes = len(all_bytes)
        print(f"  Total bytes: {total_bytes:,} ({total_bytes/1e9:.2f} GB)")
        
        # Cut into chunks of seq_len+1
        n_chunks = min(total_bytes // (seq_len + 1), max_chunks)
        self.data = np.frombuffer(bytes(all_bytes[:n_chunks * (seq_len + 1)]), dtype=np.uint8)
        self.data = self.data.reshape(n_chunks, seq_len + 1)
        
        print(f"  Chunks: {n_chunks:,} (seq_len={seq_len})")
        print(f"  Time: {time.time()-t0:.0f}s")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        chunk = self.data[idx].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

print("Building dataset...")
train_dataset = ByteChunkDataset(all_texts, seq_len=SEQ_LEN)

# Batch size optimizado para 2× T4 (16GB cada una)
# Con d=512, seq=2048, batch=4 per GPU → ~12GB VRAM
BATCH_SIZE = 4 * max(n_gpus, 1)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, drop_last=True, pin_memory=True,
    persistent_workers=True,
)
print(f"Batch: {BATCH_SIZE} | Batches/epoch: {len(train_loader):,}")

# %% [markdown]
# ## Modelo VSN Large (~350M params)

# %%
from vsn.core.vgb_v3 import VGBv3
from vsn.core.rms_norm import RMSNorm

class LatentCompressor(nn.Module):
    """Comprime estado de ventana en H (memoria acumulada)."""
    def __init__(self, d):
        super().__init__()
        self.norm = RMSNorm(d)
        self.compress = nn.Linear(d, d)
        self.gate = nn.Linear(d * 2, d)
    def forward(self, states, H_prev):
        curr = self.compress(self.norm(states.mean(dim=1)))
        gate = torch.sigmoid(self.gate(torch.cat([curr, H_prev], -1)))
        return gate * H_prev + (1 - gate) * curr

class LatentInjector(nn.Module):
    """Inyecta H (contexto acumulado) en embeddings."""
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d)
    def forward(self, h, H):
        return h + self.proj(H).unsqueeze(1)

class VSNLarge(nn.Module):
    """VSN Large: VGB v3 + H acumulador, byte-level, ~350M params.
    
    d=512, 12 layers, window=2048, byte vocab=257
    """
    def __init__(self, vocab_size=257, d=384, n_layers=8, window=2048, dropout=0.05):
        super().__init__()
        self.d, self.window, self.n_layers = d, window, n_layers
        self.vocab_size = vocab_size
        
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.drop = nn.Dropout(dropout)
        
        # Sinusoidal pos (extensible)
        pe = torch.zeros(window, d)
        pos = torch.arange(window).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        self.injector = LatentInjector(d)
        self.layers = nn.ModuleList([VGBv3(d, i, spatial_size=window) for i in range(n_layers)])
        self.compressor = LatentCompressor(d)
        self.ln_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        
        self.apply(self._init)
        n = sum(p.numel() for p in self.parameters())
        print(f"VSN Large: {n:,} params ({n/1e6:.0f}M)")
        print(f"  d={d}, layers={n_layers}, window={window}, vocab={vocab_size}")
    
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, idx, H_prev=None):
        B, T = idx.shape
        if H_prev is None:
            H_prev = torch.zeros(B, self.d, device=idx.device)
        
        h = self.tok_emb(idx) + self.pe[:, :T, :]
        h = self.drop(h)
        h = self.injector(h, H_prev)
        
        p = h.unsqueeze(1)
        m = torch.zeros_like(p)
        for layer in self.layers:
            F_out, _, _, m = layer(p, m)
            p = F_out
        
        out = p.squeeze(1)
        H_new = self.compressor(out, H_prev)
        logits = self.lm_head(self.ln_f(out + h))
        return logits, H_new
    
    @torch.no_grad()
    def generate(self, prompt_bytes, max_new=500, temperature=0.8, top_k=50):
        self.eval()
        idx = torch.tensor([prompt_bytes], device=device, dtype=torch.long)
        B = 1
        H = torch.zeros(B, self.d, device=device)
        
        # Process prompt in windows
        for s in range(0, idx.shape[1], self.window):
            e = min(s + self.window, idx.shape[1])
            _, H = self(idx[:, s:e], H)
        
        # Generate
        ctx = idx[:, -self.window:]
        for _ in range(max_new):
            logits, H = self(ctx, H)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            ctx = torch.cat([ctx, nxt], dim=1)
            if ctx.shape[1] > self.window:
                ctx = ctx[:, -self.window:]
            if nxt.item() == PAD_TOKEN: break
        return idx[0].tolist()

# %%
model = VSNLarge(vocab_size=VOCAB_SIZE, d=384, n_layers=8, window=SEQ_LEN, dropout=0.05)
if n_gpus > 1:
    model = nn.DataParallel(model)
model = model.to(device)
base_model = model.module if hasattr(model, 'module') else model

# %% [markdown]
# ## Entrenamiento: LR alto primera mitad → refinamiento
#
# Estrategia:
# - Primera media época: LR alto constante (aprendizaje forzado rápido)
# - Resto: cosine decay (refinamiento de generación)

# %%
EPOCHS = 2
LR_HIGH = 2e-4      # LR más conservador (era 5e-4 que causaba NaN)
LR_LOW = 5e-5       # Mínimo más alto para que no caiga tanto
WARMUP = 500        # Warmup más largo para estabilidad
TOTAL_STEPS = EPOCHS * len(train_loader)
HALF_EPOCH = len(train_loader) // 2

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_HIGH, weight_decay=0.1, betas=(0.9, 0.95))

def lr_schedule(step):
    """Warmup largo → decay MUY lento (linear, no cosine agresivo)."""
    if step < WARMUP:
        return step / WARMUP
    # Decay lineal suave: de 1.0 a ratio a lo largo de todo el training
    ratio = LR_LOW / LR_HIGH  # 0.25
    progress = (step - WARMUP) / max(TOTAL_STEPS - WARMUP, 1)
    return 1.0 - (1.0 - ratio) * progress  # lineal de 1.0 a 0.25

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
use_amp = torch.cuda.is_available()

print(f"Training plan:")
print(f"  LR: {LR_HIGH} → {LR_LOW} (linear decay, very slow)")
print(f"  Warmup: {WARMUP} steps")
print(f"  NO forced phase — stable throughout")
print(f"  Total steps: {TOTAL_STEPS:,}")

# %%
def train_step(x, y):
    """Single optimized training step with AMP."""
    if use_amp:
        with torch.amp.autocast('cuda'):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=PAD_TOKEN)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=PAD_TOKEN)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    return loss.item()

print("\n" + "="*70)
print("  VSN Large Pre-Training: ~150M params, Byte-level, Multi-dataset")
print("  Infinite memory via Latent H accumulator")
print("="*70 + "\n", flush=True)

from tqdm.auto import tqdm

global_step = 0
t0 = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", 
                dynamic_ncols=True, smoothing=0.1)
    
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        loss = train_step(x, y)
        epoch_loss += loss
        global_step += 1
        
        # Update progress bar
        al = epoch_loss / (step + 1)
        ppl = math.exp(min(al, 20))
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        tok_s = (global_step * BATCH_SIZE * SEQ_LEN) / elapsed
        phase = "TRAIN"
        
        pbar.set_postfix({
            'loss': f'{al:.3f}',
            'ppl': f'{ppl:.1f}',
            'tok/s': f'{tok_s:.0f}',
            'lr': f'{lr:.1e}',
        })
        
        # Save at mid-epoch
        if (step + 1) == len(train_loader) // 2:
            ckpt = f"vsn_large_ep{epoch+1}_mid.pt"
            torch.save({"model": base_model.state_dict(), "step": global_step, 
                        "loss": al}, ckpt)
            pbar.write(f"  💾 Mid-epoch save: {ckpt} (loss={al:.3f})")
    
    pbar.close()
    avg_loss = epoch_loss / len(train_loader)
    ppl = math.exp(min(avg_loss, 20))
    print(f"\n  ✓ Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.3f}, PPL: {ppl:.1f}")
    
    ckpt = f"vsn_large_ep{epoch+1}.pt"
    torch.save({"model": base_model.state_dict(), "step": global_step, "loss": avg_loss}, ckpt)
    print(f"  💾 {ckpt}")
    
    # Quick generation sample
    prompt = "Once upon a time there was"
    prompt_bytes = list(prompt.encode('utf-8'))
    gen = base_model.generate(prompt_bytes, max_new=300, temperature=0.8)
    print(f"  >>> {bytes(gen).decode('utf-8', errors='replace')[:400]}\n", flush=True)

total_time = time.time() - t0
print(f"\nTraining complete! {total_time:.0f}s ({total_time/3600:.1f}h)")

# %% [markdown]
# ## Generación Final

# %%
print("="*70)
print("  Generación Final — VSN Large (350M)")
print("="*70)

prompts_en = [
    "The meaning of life is",
    "In the year 2030, artificial intelligence",
    "def fibonacci(n):\n    ",
    "Once upon a time, a brave knight",
]
prompts_es = [
    "La inteligencia artificial es",
    "En un mundo donde la tecnología",
    "Había una vez un pequeño gato que",
]

base_model.eval()
for p in prompts_en + prompts_es:
    gen = base_model.generate(list(p.encode('utf-8')), max_new=400, temperature=0.7, top_k=50)
    text = bytes(gen).decode('utf-8', errors='replace')
    print(f"\n  \"{p}\"")
    print(f"  {text[:500]}")
    print(f"  {'─'*55}")

# %%
# Final save
save_path = "vsn_large_final.pt"
torch.save({
    "model": base_model.state_dict(),
    "params": sum(p.numel() for p in base_model.parameters()),
    "config": {"d": 512, "layers": 12, "window": SEQ_LEN, "vocab": VOCAB_SIZE},
}, save_path)
print(f"\nFinal: {save_path} ({os.path.getsize(save_path)/1e6:.0f}MB)")
print(f"Params: {sum(p.numel() for p in base_model.parameters()):,}")
