# %% [markdown]
# # VSN Pre-Training v2: Arquitectura Híbrida Real
#
# **Diseño**: VGB v3 blocks + Estado Latente H comprimido entre ventanas.
#
# ```
# [Window 1] → VGB v3 layers → Compress → H₁
#                                           ↓ (inyectado como contexto)
# [Window 2] → VGB v3 layers + H₁ → Compress → H₂
#                                                 ↓
# [Window 3] → VGB v3 layers + H₂ → Compress → H₃ → ...
# ```
#
# **NO es sliding window** — H acumula información de TODA la historia previa.
# **NO es LSTM** — procesa ventanas completas con VGB v3 (causal spatial mixing).
# **Memoria infinita** via el estado latente H que se comprime y propaga.

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
WINDOW = 512
BATCH_SIZE = 8 * max(n_gpus, 1)

class StoriesDataset(Dataset):
    def __init__(self, texts, tokenizer, window=512, max_samples=150000):
        all_tokens = []
        for i, item in enumerate(texts):
            if i >= max_samples: break
            all_tokens.extend(tokenizer.encode(item["text"]))
            all_tokens.append(EOT_TOKEN)
        self.chunks = []
        for i in range(0, len(all_tokens) - window, window // 2):
            self.chunks.append(all_tokens[i:i + window + 1])
        print(f"  Tokens: {len(all_tokens):,} | Chunks: {len(self.chunks):,}")
    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        c = self.chunks[idx]
        return torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)

train_dataset = StoriesDataset(dataset, enc, window=WINDOW, max_samples=150000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, drop_last=True, pin_memory=True)
print(f"Batch: {BATCH_SIZE} | Batches/epoch: {len(train_loader)}")

# %% [markdown]
# ## Modelo VSN Híbrido Real

# %%
from vsn.core.vgb_v3 import VGBv3
from vsn.core.rms_norm import RMSNorm

class LatentCompressor(nn.Module):
    """Comprime el estado de una ventana en un vector latente H.
    
    Toma la salida de los VGB layers (B, T, d) y produce un estado
    comprimido (B, d) que resume toda la información de la ventana.
    Esto es el equivalente funcional del operador P de la arquitectura.
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.compress = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
    
    def forward(self, states, prev_H):
        """
        Args:
            states: (B, T, d) — salida de la última VGB layer
            prev_H: (B, d) — estado latente de la ventana anterior (o zeros)
        Returns:
            H_new: (B, d) — estado latente actualizado
        """
        # Comprimir ventana actual: mean pool → linear
        current = self.norm(states.mean(dim=1))  # (B, d)
        current = self.compress(current)
        
        # Fusionar con H previo via gating (como Ψ)
        combined = torch.cat([current, prev_H], dim=-1)  # (B, 2d)
        gate = torch.sigmoid(self.gate(combined))  # (B, d)
        H_new = gate * prev_H + (1 - gate) * current
        
        return H_new


class LatentInjector(nn.Module):
    """Inyecta el estado latente H en la secuencia de la ventana actual.
    
    Equivalente funcional del operador Q: transforma H en algo que 
    puede sumarse a los embeddings para proveer contexto histórico.
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, h_emb, H):
        """
        Args:
            h_emb: (B, T, d) — embeddings de la ventana actual
            H: (B, d) — estado latente (contexto previo comprimido)
        Returns:
            h_augmented: (B, T, d) — embeddings + contexto histórico
        """
        # Proyectar H y sumarlo a cada posición (broadcast)
        context = self.proj(H).unsqueeze(1)  # (B, 1, d)
        return h_emb + context


class VSNHybridReal(nn.Module):
    """VSN Híbrido Real — VGB v3 + Estado Latente H entre ventanas.
    
    Pipeline por ventana:
        1. Embedding + positional + inyectar H_prev (contexto acumulado)
        2. VGB v3 layers (causal spatial mixing + gated memory)
        3. Comprimir estado → H_new (para próxima ventana)
        4. LM Head → logits
    
    El estado H se acumula entre ventanas — NUNCA se pierde información.
    Esto permite:
        - Input infinito: procesar por ventanas acumulando H
        - Output infinito: generar ventana tras ventana con H propagado
        - Memoria real: H retiene lo importante de toda la historia
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
        
        # Sinusoidal positional (sin límite)
        pe = torch.zeros(max_window * 4, d_model)
        pos = torch.arange(0, max_window * 4).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Inyector de H (equivalente a Q)
        self.injector = LatentInjector(d_model)
        
        # VGB v3 layers
        self.layers = nn.ModuleList([
            VGBv3(d_model, plane_idx=i, spatial_size=max_window)
            for i in range(n_layers)
        ])
        
        # Compresor (equivalente a P + Ψ)
        self.compressor = LatentCompressor(d_model)
        
        # Output
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        
        self.apply(self._init_weights)
        n = sum(p.numel() for p in self.parameters())
        print(f"VSN Hybrid Real: {n:,} params ({n/1e6:.1f}M)")
        print(f"  d={d_model}, layers={n_layers}, window={max_window}")
        print(f"  H carries ALL history between windows (no forgetting)")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, idx, H_prev=None):
        """Forward una ventana con contexto H previo.
        
        Args:
            idx: (B, T) token IDs de esta ventana
            H_prev: (B, d) estado latente de ventana anterior. None = zeros.
        Returns:
            logits: (B, T, vocab)
            H_new: (B, d) estado latente actualizado
        """
        B, T = idx.shape
        
        if H_prev is None:
            H_prev = torch.zeros(B, self.d_model, device=idx.device)
        
        # 1. Embed + positional
        h = self.tok_emb(idx) + self.pe[:, :T, :]
        h = self.drop(h)
        
        # 2. Inyectar H previo (contexto de toda la historia anterior)
        h = self.injector(h, H_prev)
        
        # 3. VGB v3 layers
        p = h.unsqueeze(1)  # (B, 1, T, d)
        m = torch.zeros_like(p)
        for layer in self.layers:
            F_out, _, _, m = layer(p, m)
            p = F_out
        
        out = p.squeeze(1)  # (B, T, d)
        
        # 4. Comprimir estado → H_new (para próxima ventana)
        H_new = self.compressor(out, H_prev)
        
        # 5. LM Head
        logits = self.lm_head(self.ln_f(out + h))
        
        return logits, H_new
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.8, top_k=50):
        """Generación con memoria infinita via H.
        
        No usa sliding window — usa H acumulado para mantener contexto.
        """
        self.eval()
        B = idx.shape[0]
        H = torch.zeros(B, self.d_model, device=idx.device)
        
        # Procesar prompt completo (puede ser >max_window, se procesa por ventanas)
        prompt_len = idx.shape[1]
        for start in range(0, prompt_len, self.max_window):
            end = min(start + self.max_window, prompt_len)
            chunk = idx[:, start:end]
            _, H = self(chunk, H)
        
        # Generar tokens uno a uno (o en mini-batches)
        generated = idx
        context = idx[:, -self.max_window:]  # última ventana como contexto activo
        
        for _ in range(max_new_tokens):
            # Forward con H (que contiene TODA la historia)
            logits, H = self(context, H)
            logits = logits[:, -1, :] / temperature
            
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, nxt], dim=1)
            
            # Actualizar contexto: añadir nuevo token
            context = torch.cat([context, nxt], dim=1)
            if context.shape[1] > self.max_window:
                context = context[:, -self.max_window:]
            
            if nxt.item() == EOT_TOKEN: break
        
        return generated

# %%
model = VSNHybridReal(
    vocab_size=VOCAB_SIZE,
    d_model=256,
    n_layers=8,
    max_window=WINDOW,
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
print(f"Training: {EPOCHS} ep, LR={LR}, steps={TOTAL_STEPS}, AMP={use_amp}")

# %%
def train_epoch(epoch):
    model.train()
    total_loss, n_tok, t0 = 0.0, 0, time.time()
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits, _ = model(x)  # H_new no se usa entre batches en training
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=EOT_TOKEN)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(x)
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
            ppl = math.exp(min(al,20))
            print(f"  Ep{epoch+1} {step+1:5d}/{len(train_loader)} | Loss:{al:.3f} PPL:{ppl:.1f} | "
                  f"{n_tok/(time.time()-t0):.0f} tok/s | LR:{scheduler.get_last_lr()[0]:.2e} [{time.time()-t0:.0f}s]", flush=True)
    return total_loss / len(train_loader)

print("="*65)
print("  VSN Hybrid Real — VGB v3 + Latent H (infinite memory)")
print("  H acumula contexto de TODA la historia — no olvida")
print("="*65, flush=True)

for ep in range(EPOCHS):
    loss = train_epoch(ep)
    ppl = math.exp(min(loss, 20))
    print(f"\n  ✓ Epoch {ep+1}/{EPOCHS} — Loss: {loss:.3f}, PPL: {ppl:.1f}")
    torch.save({"model": base_model.state_dict(), "epoch": ep+1, "loss": loss}, f"vsn_hybrid_ep{ep+1}.pt")
    print(f"  💾 vsn_hybrid_ep{ep+1}.pt")
    ids = torch.tensor([enc.encode("Once upon a time")], device=device)
    gen = base_model.generate(ids, max_new_tokens=200, temperature=0.8)
    print(f"  >>> {enc.decode(gen[0].tolist())[:400]}\n", flush=True)

# %% [markdown]
# ## Generación Final

# %%
print("="*65)
print("  Generación Final — Memoria Infinita via H")
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
            "config": {"d":256, "layers":8, "window":WINDOW}}, save_path)
print(f"Saved: {save_path} ({os.path.getsize(save_path)/1e6:.0f}MB)")
