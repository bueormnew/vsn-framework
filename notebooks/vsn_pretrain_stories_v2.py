# %% [markdown]
# # VSN Pre-Training v2: Arquitectura Híbrida
#
# **Diseño**: Usa VGB v3 blocks (causal spatial mixing) como procesador principal,
# con un sistema de windowing ligero para soportar secuencias infinitas.
#
# **Principio**: Los tokens se procesan en ventanas de tamaño fijo. Entre ventanas,
# un estado comprimido (memory state) se transporta al siguiente chunk. Esto permite:
# - Input infinito (procesar por ventanas)
# - Output infinito (generar por ventanas)  
# - Velocidad alta (~50K tok/s en T4)
# - Sin los problemas de P/Q gigantes
#
# **Componentes VSN usados:**
# - VGB v3 blocks (causal spatial mixing + gated memory + MLP)
# - Memoria M propagada entre bloques Y entre ventanas (→ contexto infinito)
# - RMSNorm, GELU, conexiones residuales
#
# **Lo que NO usa** (por ser contraproducente para texto):
# - P/Q (Linear gigante → NaN, lento, innecesario)
# - Φ raster positioning (no aporta para secuencias 1D)
# - Input Cache formal (el windowing cumple la misma función)

# %%
# !pip install vsn-framework tiktoken datasets --quiet

# %%
import os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Device: {device} | GPUs: {n_gpus}")
for i in range(n_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB)")

enc = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = enc.n_vocab
EOT_TOKEN = enc.eot_token
print(f"Vocab: {VOCAB_SIZE}")

# %% [markdown]
# ## Dataset

# %%
from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")
print(f"Dataset: {len(dataset)} stories")

# %%
WINDOW_SIZE = 512  # Tokens por ventana de procesamiento
BATCH_SIZE = 8 * max(n_gpus, 1)

class StoriesDataset(Dataset):
    """Chunks de WINDOW_SIZE tokens con overlap."""
    def __init__(self, texts, tokenizer, window=512, max_samples=150000):
        all_tokens = []
        for i, item in enumerate(texts):
            if i >= max_samples: break
            all_tokens.extend(tokenizer.encode(item["text"]))
            all_tokens.append(EOT_TOKEN)
        self.chunks = []
        stride = window // 2  # 50% overlap
        for i in range(0, len(all_tokens) - window, stride):
            self.chunks.append(all_tokens[i:i + window + 1])
        print(f"  Tokens: {len(all_tokens):,} | Chunks: {len(self.chunks):,} (window={window}, stride={stride})")
    
    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        c = self.chunks[idx]
        return torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)

train_dataset = StoriesDataset(dataset, enc, window=WINDOW_SIZE, max_samples=150000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, drop_last=True, pin_memory=True)
print(f"Batch: {BATCH_SIZE} | Batches/epoch: {len(train_loader)}")

# %% [markdown]
# ## Modelo Híbrido VSN
#
# Arquitectura:
# ```
# tokens → Embedding → [VGB v3 × N layers] → LM Head
#                        ↑
#                    Memory M se propaga entre layers
#                    (y potencialmente entre ventanas para input infinito)
# ```
#
# Para ~35M params: d=256, 8 layers (encoder-style, sin split enc/dec)

# %%
from vsn.core.vgb_v3 import VGBv3
from vsn.core.rms_norm import RMSNorm

class VSNHybrid(nn.Module):
    """VSN Híbrido para lenguaje — VGB v3 blocks con memoria persistente.
    
    - N capas de VGB v3 (causal spatial mixing + gated memory)
    - La memoria M se propaga entre capas → contexto profundo
    - Embedding sinusoidal (sin límite fijo)
    - Para generación infinita: sliding window + memory state
    
    Diferencia con un Transformer:
    - Sin atención cuadrática (spatial mixing es O(N²) pero N=window_size fijo)
    - Gated memory propaga información entre layers sin atención
    - Complejidad: O(N * d² * n_layers) — lineal en N para d fijo
    """
    
    def __init__(self, vocab_size=50257, d_model=256, n_layers=8, 
                 max_window=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_window = max_window
        self.vocab_size = vocab_size
        
        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Positional: sinusoidal (extensible infinitamente)
        pe = torch.zeros(max_window * 4, d_model)
        pos = torch.arange(0, max_window * 4).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # VGB v3 layers (causal spatial mixing)
        self.layers = nn.ModuleList([
            VGBv3(d_model, plane_idx=i, spatial_size=max_window)
            for i in range(n_layers)
        ])
        
        # Output
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        
        self.apply(self._init_weights)
        n = sum(p.numel() for p in self.parameters())
        print(f"VSN Hybrid: {n:,} params ({n/1e6:.1f}M)")
        print(f"  d={d_model}, layers={n_layers}, window={max_window}")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, idx):
        """Forward pass.
        Args: idx (B, T) token IDs
        Returns: logits (B, T, vocab_size)
        """
        B, T = idx.shape
        
        # Embed + positional
        h = self.tok_emb(idx) + self.pe[:, :T, :]
        h = self.drop(h)
        
        # Process through VGB v3 layers
        # Shape for VGB: (B, 1, T, d) — Y=1, Z=T
        p = h.unsqueeze(1)
        m = torch.zeros_like(p)
        
        for layer in self.layers:
            F_out, _, _, m = layer(p, m)
            p = F_out
        
        # Output
        out = p.squeeze(1)  # (B, T, d)
        out = self.ln_f(out + h)  # global residual
        return self.lm_head(out)
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.8, top_k=50):
        """Generación infinita con sliding window.
        
        Usa los últimos max_window tokens como contexto.
        La memoria M dentro de cada VGB layer mantiene información
        de tokens previos via el gating mechanism.
        """
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.max_window:]
            logits = self(ctx)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if nxt.item() == EOT_TOKEN: break
        return idx

# %%
# Config: ~35M params
model = VSNHybrid(
    vocab_size=VOCAB_SIZE,
    d_model=256,
    n_layers=8,
    max_window=WINDOW_SIZE,
    dropout=0.1,
)

if n_gpus > 1:
    model = nn.DataParallel(model)
model = model.to(device)
base_model = model.module if hasattr(model, 'module') else model

# %% [markdown]
# ## Entrenamiento

# %%
EPOCHS = 3
LR = 3e-4
WARMUP = 300
TOTAL_STEPS = EPOCHS * len(train_loader)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))

def lr_fn(step):
    if step < WARMUP: return step / WARMUP
    p = (step - WARMUP) / max(TOTAL_STEPS - WARMUP, 1)
    return 0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * p))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
use_amp = torch.cuda.is_available()

print(f"Config: {EPOCHS} ep, LR={LR}, warmup={WARMUP}, total={TOTAL_STEPS}, AMP={use_amp}")

# %%
def train_epoch(epoch):
    model.train()
    total_loss, n_tok, t0 = 0.0, 0, time.time()
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=EOT_TOKEN)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=EOT_TOKEN)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        total_loss += loss.item()
        n_tok += (y != EOT_TOKEN).sum().item()
        if (step+1) % 50 == 0:
            al = total_loss/(step+1)
            ppl = math.exp(min(al, 20))
            print(f"  Ep{epoch+1} {step+1:5d}/{len(train_loader)} | Loss:{al:.3f} PPL:{ppl:.1f} | "
                  f"{n_tok/(time.time()-t0):.0f} tok/s | LR:{scheduler.get_last_lr()[0]:.2e} [{time.time()-t0:.0f}s]", flush=True)
    return total_loss / len(train_loader)

# ── Train ──
print("="*65)
print("  VSN Hybrid — VGB v3 (causal mix) + Memory + Sliding Window")
print("="*65, flush=True)

for ep in range(EPOCHS):
    loss = train_epoch(ep)
    ppl = math.exp(min(loss, 20))
    print(f"\n  ✓ Epoch {ep+1}/{EPOCHS} — Loss: {loss:.3f}, PPL: {ppl:.1f}")
    
    # Save
    torch.save({"model": base_model.state_dict(), "epoch": ep+1, "loss": loss,
                "config": {"d_model":256, "n_layers":8, "max_window": WINDOW_SIZE}},
               f"vsn_hybrid_ep{ep+1}.pt")
    print(f"  💾 vsn_hybrid_ep{ep+1}.pt")
    
    # Generate
    ids = torch.tensor([enc.encode("Once upon a time")], device=device)
    gen = base_model.generate(ids, max_new_tokens=200, temperature=0.8)
    print(f"  >>> {enc.decode(gen[0].tolist())[:400]}\n", flush=True)

# %% [markdown]
# ## Generación Final

# %%
print("="*65)
print("  Generación Final")
print("="*65)
prompts = [
    "Once upon a time, there was a little girl named",
    "The brave knight rode his horse into the dark",
    "A small cat found a magic wand and decided to",
    "In a magical forest, the animals gathered to",
    "The little boy was afraid of the dark but",
    "One sunny morning, the princess discovered a",
]
base_model.eval()
for p in prompts:
    ids = torch.tensor([enc.encode(p)], device=device)
    gen = base_model.generate(ids, max_new_tokens=250, temperature=0.7, top_k=50)
    print(f"\n  \"{p}\"")
    print(f"  {enc.decode(gen[0].tolist())[:500]}")
    print(f"  {'─'*55}")

# %%
save_path = "vsn_hybrid_final.pt"
torch.save({"model": base_model.state_dict(), 
            "params": sum(p.numel() for p in base_model.parameters()),
            "config": {"d_model":256, "n_layers":8, "window":WINDOW_SIZE}}, save_path)
print(f"Saved: {save_path} ({os.path.getsize(save_path)/1e6:.0f}MB)")
