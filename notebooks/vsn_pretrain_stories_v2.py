# %% [markdown]
# # VSN Pre-Training v2: 30-50M params, Multi-GPU, Dynamic Sequence Length
#
# **Mejoras sobre v1:**
# - Modelo más grande (~35-50M params): d=256, 6+6 layers
# - Entrenamiento en 2× T4 GPUs con DataParallel
# - Secuencia dinámica: se adapta al texto más largo del batch
# - Más datos del dataset TinyStories
#
# **Arquitectura:** VSN con VGB v3 (causal spatial mixing)

# %% [markdown]
# ## 1. Instalación

# %%
# !pip install vsn-framework tiktoken datasets --quiet

# %% [markdown]
# ## 2. Imports y Setup Multi-GPU

# %%
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Device: {device} | GPUs disponibles: {n_gpus}")
for i in range(n_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name} ({mem:.1f} GB)")

# %% [markdown]
# ## 3. Tokenizer

# %%
import tiktoken

enc = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = enc.n_vocab  # 50257
EOT_TOKEN = enc.eot_token  # 50256

print(f"Vocab: {VOCAB_SIZE} | EOT: {EOT_TOKEN}")

# %% [markdown]
# ## 4. Dataset con longitud dinámica

# %%
from datasets import load_dataset

# Cargar más datos para modelo más grande
dataset = load_dataset("roneneldan/TinyStories", split="train")
print(f"Dataset: {len(dataset)} stories")

# %%
class DynamicStoriesDataset(Dataset):
    """Dataset que preserva la longitud natural de cada historia.
    
    Trunca a max_tokens (1024). Las historias más largas se cortan.
    """
    
    def __init__(self, texts, tokenizer, max_tokens=1024, max_samples=200000):
        self.samples = []
        self.max_tokens = max_tokens
        
        for i, item in enumerate(texts):
            if i >= max_samples:
                break
            tokens = tokenizer.encode(item["text"])
            if len(tokens) >= 10:
                tokens = tokens[:max_tokens]
                self.samples.append(tokens)
        
        lengths = [len(s) for s in self.samples]
        self.max_len = max(lengths)
        print(f"  Samples: {len(self.samples):,}")
        print(f"  Max length: {self.max_len} tokens (cap: {max_tokens})")
        print(f"  Avg length: {sum(lengths)/len(lengths):.0f} tokens")
        print(f"  Total tokens: {sum(lengths):,}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_dynamic(batch: List[List[int]]):
    """Collate que padea al largo máximo del batch (max 1024)."""
    max_len = min(max(len(s) for s in batch) + 1, 1025)
    
    inputs = []
    targets = []
    for tokens in batch:
        padded = tokens + [EOT_TOKEN] * (max_len - len(tokens))
        padded = padded[:max_len]
        inputs.append(padded[:-1])
        targets.append(padded[1:])
    
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


print("Procesando dataset...")
train_dataset = DynamicStoriesDataset(dataset, enc, max_tokens=1024, max_samples=200000)

# DataLoader con collate dinámico
# Batch size por GPU × num_gpus
BATCH_PER_GPU = 4
BATCH_SIZE = BATCH_PER_GPU * max(n_gpus, 1)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_dynamic,
    num_workers=2,
    drop_last=True,
    pin_memory=True,
)
print(f"Batch size: {BATCH_SIZE} ({BATCH_PER_GPU}/GPU × {max(n_gpus,1)} GPUs)")
print(f"Batches per epoch: {len(train_loader)}")

# %% [markdown]
# ## 5. Modelo VSN Grande (30-50M params)

# %%
from vsn.core.vgb_v3 import VGBv3
from vsn.core.rms_norm import RMSNorm

@dataclass
class VSNLargeConfig:
    vocab_size: int = 50257
    d_model: int = 256        # Mayor dimensión → más capacidad
    n_layers: int = 6         # 6 enc + 6 dec = 12 bloques total
    max_seq_len: int = 1024   # Ventana de entrenamiento (el modelo genera infinito)
    dropout: float = 0.1


class VSNLanguageV2(nn.Module):
    """VSN modelo grande para generación de lenguaje.
    
    - VGB v3 con causal spatial mixing
    - Embeddings posicionales sinusoidales (sin límite fijo)
    - Entrena con ventana de max_seq_len tokens
    - Genera texto infinito usando sliding window
    - Compatible con DataParallel para multi-GPU
    """
    
    def __init__(self, config: VSNLargeConfig):
        super().__init__()
        self.config = config
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # VGB v3 blocks (causal) — spatial_size = ventana de entrenamiento
        self.encoder_blocks = nn.ModuleList([
            VGBv3(config.d_model, plane_idx=i, spatial_size=config.max_seq_len)
            for i in range(config.n_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            VGBv3(config.d_model, plane_idx=i, spatial_size=config.max_seq_len)
            for i in range(config.n_layers)
        ])
        
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        
        # Pre-compute sinusoidal positional encoding (extendible a cualquier longitud)
        self._build_sin_pos(config.max_seq_len * 2, config.d_model)
        
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"VSN Language V2: {n_params:,} parameters ({n_params/1e6:.1f}M)")
    
    def _build_sin_pos(self, max_len, d_model):
        """Construye embeddings posicionales sinusoidales (no entrenables, extensibles)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d)
    
    def _get_pos_emb(self, seq_len):
        """Obtiene positional embedding para cualquier longitud."""
        if seq_len <= self.pe.shape[1]:
            return self.pe[:, :seq_len, :]
        # Extender si es necesario (generación más larga que el buffer)
        self._build_sin_pos(seq_len * 2, self.config.d_model)
        self.pe = self.pe.to(next(self.parameters()).device)
        return self.pe[:, :seq_len, :]
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
        B, T = idx.shape
        
        # Token embedding + positional (sinusoidal, sin límite)
        h = self.tok_emb(idx) + self._get_pos_emb(T)
        h = self.drop(h)
        
        # (B, T, d) → (B, 1, T, d) para VGB
        p = h.unsqueeze(1)
        m = torch.zeros_like(p)
        
        for block in self.encoder_blocks:
            F_out, _, _, m = block(p, m)
            p = F_out
        
        for block in self.decoder_blocks:
            F_out, _, _, m = block(p, m)
            p = F_out
        
        out = p.squeeze(1)
        out = self.ln_f(out + h)
        return self.lm_head(out)
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=50):
        """Generación autoregresiva con sliding window (tokens infinitos).
        
        Usa los últimos max_seq_len tokens como contexto en cada paso.
        No hay límite en la longitud total generada.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Sliding window: tomar solo los últimos max_seq_len tokens
            idx_cond = idx[:, -self.config.max_seq_len:]
            
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            
            if next_token.item() == EOT_TOKEN:
                break
        
        return idx

# Crear modelo
config = VSNLargeConfig(
    vocab_size=VOCAB_SIZE,
    d_model=256,
    n_layers=6,
    max_seq_len=1024,
    dropout=0.1,
)
model = VSNLanguageV2(config)

# Multi-GPU con DataParallel
if n_gpus > 1:
    print(f"Usando DataParallel en {n_gpus} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

# Para acceder al modelo base (con o sin DataParallel)
base_model = model.module if hasattr(model, 'module') else model

# %% [markdown]
# ## 6. Entrenamiento Multi-GPU

# %%
EPOCHS = 3
LR = 3e-4
WARMUP_STEPS = 500
TOTAL_STEPS = EPOCHS * len(train_loader)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))

def get_lr(step):
    # Warmup lineal
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    # Cosine decay con mínimo 30% del LR original (no baja a casi cero)
    progress = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
    return 0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

# AMP para velocidad
scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
use_amp = torch.cuda.is_available()

print(f"Config: {EPOCHS} epochs, LR={LR}, warmup={WARMUP_STEPS}")
print(f"Total steps: {TOTAL_STEPS}")
print(f"AMP: {'enabled' if use_amp else 'disabled'}")
print(f"GPUs: {n_gpus}")

# %%
def train_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss = 0
    total_tokens = 0
    t0 = time.time()
    
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Forward con AMP
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
        total_tokens += (y != EOT_TOKEN).sum().item()
        
        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step + 1)
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t0
            tok_s = total_tokens / elapsed
            lr = scheduler.get_last_lr()[0]
            seq_len = x.shape[1]
            print(f"  Ep{epoch+1} Step {step+1:5d}/{len(loader)} | "
                  f"Loss: {avg_loss:.3f} PPL: {ppl:.1f} | "
                  f"Tok/s: {tok_s:.0f} | SeqLen: {seq_len} | "
                  f"LR: {lr:.2e} [{elapsed:.0f}s]", flush=True)
    
    avg_loss = total_loss / len(loader)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl

# ── Training loop ──
print("="*70)
print("  VSN Pre-Training v2: Large Model, Multi-GPU, Dynamic Length")
print("="*70)
print()

for epoch in range(EPOCHS):
    loss, ppl = train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
    print(f"\n  ✓ Epoch {epoch+1}/{EPOCHS} — Loss: {loss:.3f}, PPL: {ppl:.1f}")
    
    # Guardar checkpoint al final de cada época
    ckpt_path = f"vsn_stories_epoch{epoch+1}.pt"
    torch.save({
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch + 1,
        "loss": loss,
        "config": {"vocab_size": config.vocab_size, "d_model": config.d_model, 
                   "n_layers": config.n_layers, "max_seq_len": config.max_seq_len},
    }, ckpt_path)
    print(f"  💾 Checkpoint saved: {ckpt_path}")
    
    # Sample generation
    print(f"  Sample:")
    prompt = "Once upon a time"
    prompt_ids = torch.tensor([enc.encode(prompt)], device=device)
    generated = base_model.generate(prompt_ids, max_new_tokens=150, temperature=0.8)
    text = enc.decode(generated[0].tolist())
    print(f"  >>> {text[:400]}")
    print()

# %% [markdown]
# ## 7. Generación Final

# %%
print("="*70)
print("  Generación Final")
print("="*70)

prompts = [
    "Once upon a time, there was a little girl named",
    "The brave knight rode his horse into the dark",
    "A small cat found a magic wand and",
    "In a magical forest, the animals decided to",
    "The little boy was afraid of the",
    "One sunny morning, the princess woke up and",
]

base_model.eval()
for prompt in prompts:
    prompt_ids = torch.tensor([enc.encode(prompt)], device=device)
    generated = base_model.generate(prompt_ids, max_new_tokens=200, temperature=0.7, top_k=50)
    text = enc.decode(generated[0].tolist())
    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Output: {text[:500]}")
    print(f"  {'─'*60}")

# %% [markdown]
# ## 8. Guardar

# %%
save_path = "vsn_stories_v2_large.pt"
torch.save({
    "model_state_dict": base_model.state_dict(),
    "config": {
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "max_seq_len": config.max_seq_len,
        "dropout": config.dropout,
    },
    "tokenizer": "gpt2",
    "n_params": sum(p.numel() for p in base_model.parameters()),
}, save_path)
size_mb = os.path.getsize(save_path) / 1e6
print(f"Saved: {save_path} ({size_mb:.0f} MB)")
print(f"Params: {sum(p.numel() for p in base_model.parameters()):,}")

# %% [markdown]
# ## Notas
#
# - **Multi-GPU**: Usa `nn.DataParallel` que divide el batch entre GPUs
# - **Secuencia dinámica**: El collate padea al max del batch, no a un fijo global
# - **AMP**: fp16 automático para duplicar velocidad en T4
# - **VGB v3**: Causal mixing permite generación coherente
# - **Modelo**: ~35-50M params (d=256, 12 bloques)
#
# Para escalar más:
# - d=512, n_layers=8 → ~200M params (necesita A100)
# - d=1024, n_layers=12 → ~1B params
