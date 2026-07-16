# %% [markdown]
# # VSN Pre-Training v2: Arquitectura REAL
#
# Pipeline: Input Cache → Φ → Encoder (VGB v3) → P → H → Q → Decoder (VGB v3 + Ψ)
#
# Volumen: X=4, Y=4, Z=4, d=256 → 64 tokens por plano, 256 tokens totales
# Modelo: ~30-50M params | Multi-GPU (2× T4) | AMP fp16

# %%
# !pip install vsn-framework tiktoken datasets --quiet

# %%
import os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List
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
# Volumen config: X=4, Y=4, Z=4 → 64 tokens por plano, 256 por volumen completo
# Entrenamiento: cada sample es un chunk de 256 tokens (cabe exactamente en el volumen)
CHUNK_SIZE = 256  # X * Y * Z = 4 * 4 * 4 * 4 = 256 (pero solo Y*Z=16 por plano, X=4 → 64 total)
# Corrección: con X=4, Y=4, Z=4 → Y*Z=16 tokens por plano, X planos = 64 tokens MAX en volumen
# Necesitamos más: X=16, Y=4, Z=4 → Y*Z=16 tokens/plano × 16 planos = 256 tokens
# O mejor: X=4, Y=8, Z=8 → Y*Z=64 tokens/plano × 4 planos = 256 tokens

# Usamos X=4, Y=8, Z=8: 64 tokens por plano, 256 tokens totales en el volumen
TOKENS_PER_PLANE = 64  # Y * Z = 8 * 8
PLANES = 4  # X
VOLUME_CAPACITY = TOKENS_PER_PLANE * PLANES  # 256 tokens
BATCH_SIZE = 16 * max(n_gpus, 1)

class ChunkedDataset(Dataset):
    """Divide historias en chunks de VOLUME_CAPACITY tokens."""
    def __init__(self, texts, tokenizer, chunk_size=256, max_samples=150000):
        self.chunks = []
        all_tokens = []
        for i, item in enumerate(texts):
            if i >= max_samples: break
            all_tokens.extend(tokenizer.encode(item["text"]))
            all_tokens.append(EOT_TOKEN)
        # Cortar en chunks de chunk_size+1 (input + target shift)
        for i in range(0, len(all_tokens) - chunk_size, chunk_size):
            self.chunks.append(all_tokens[i:i + chunk_size + 1])
        print(f"  Total tokens: {len(all_tokens):,}")
        print(f"  Chunks ({chunk_size} tokens): {len(self.chunks):,}")
    
    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        c = self.chunks[idx]
        return torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)

train_dataset = ChunkedDataset(dataset, enc, chunk_size=VOLUME_CAPACITY, max_samples=150000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, drop_last=True, pin_memory=True)
print(f"Batch size: {BATCH_SIZE} | Batches/epoch: {len(train_loader)}")

# %% [markdown]
# ## Modelo VSN Real

# %%
from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel

class VSNForLanguage(nn.Module):
    """VSN Real para lenguaje: IC → Φ → Encoder → P → H → Q → Decoder(+Ψ) → LM Head.
    
    El volumen X=4, Y=8, Z=8 almacena 256 tokens.
    El decoder produce Y_dec*Z_dec=64 tokens por ventana.
    Con num_windows=4, produce 256 tokens = coincide con input.
    """
    
    def __init__(self, vsn_config: VSNConfig, vocab_size: int = 50257):
        super().__init__()
        self.vsn_config = vsn_config
        self.vocab_size = vocab_size
        self.volume_capacity = vsn_config.X_enc * vsn_config.Y * vsn_config.Z
        self.tokens_per_window = vsn_config.Y_dec * vsn_config.Z_dec
        
        # Embedding
        self.tok_emb = nn.Embedding(vocab_size, vsn_config.d)
        
        # VSN completo
        self.vsn = VSNModel(vsn_config)
        
        # LM Head (weight tied con embedding)
        self.lm_head = nn.Linear(vsn_config.d, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        
        n = sum(p.numel() for p in self.parameters())
        print(f"VSN Real: {n:,} params ({n/1e6:.1f}M)")
        print(f"  Volume: X={vsn_config.X_enc} Y={vsn_config.Y} Z={vsn_config.Z} d={vsn_config.d}")
        print(f"  Capacity: {self.volume_capacity} tokens")
        print(f"  Decoder window: {self.tokens_per_window} tokens")
    
    def forward(self, token_ids: torch.Tensor, num_windows: int = 4):
        """
        Args:
            token_ids: (B, 256) — chunk de tokens
            num_windows: ventanas DGW (4 × 64 = 256 tokens output)
        Returns:
            logits: (B, 256, vocab)
        """
        B, T = token_ids.shape
        
        # Embedding
        emb = self.tok_emb(token_ids)  # (B, T, d)
        
        # VSN forward: produce decoder_states = list of (B, Y_dec, Z_dec, d)
        outputs = self.vsn(emb, num_windows=num_windows)
        dec_states = outputs.states["decoder_states"]
        
        # Aplanar ventanas: cada (B, Y_dec, Z_dec, d) → (B, Y_dec*Z_dec, d)
        flat_states = [s.reshape(B, -1, self.vsn_config.d) for s in dec_states]
        # Concatenar todas las ventanas: (B, num_windows * Y_dec * Z_dec, d)
        combined = torch.cat(flat_states, dim=1)  # (B, 256, d) si 4 windows × 64
        
        # Truncar a T tokens (por si num_windows produce más)
        combined = combined[:, :T, :]
        
        # LM Head
        return self.lm_head(combined)  # (B, T, vocab)
    
    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 0.8, top_k: int = 50):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = token_ids[:, -self.volume_capacity:]
            
            # Pad a volume_capacity si es menor
            B, T = ctx.shape
            if T < self.volume_capacity:
                pad = torch.full((B, self.volume_capacity - T), EOT_TOKEN, device=ctx.device, dtype=ctx.dtype)
                ctx_padded = torch.cat([pad, ctx], dim=1)
            else:
                ctx_padded = ctx
            
            logits = self(ctx_padded, num_windows=4)
            logits = logits[:, -1, :] / temperature
            
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_tok], dim=1)
            
            if next_tok.item() == EOT_TOKEN:
                break
        return token_ids

# %%
# Config: X=4, Y=8, Z=8, d=256 → P/Q son Linear(8*8*256, 8*8*256) = Linear(16384, 16384) = 268M
# Eso es demasiado. Solución: usar d más pequeño para el volumen.
# Config real: X=4, Y=8, Z=8, d=128 → P/Q = Linear(8192, 8192) = 67M — sigue alto
# Mejor: X=4, Y=4, Z=4, d=256 → P/Q = Linear(4096, 4096) = 16M — razonable!

# PERO con Y=4, Z=4 solo caben 16 tokens por plano → 64 tokens totales (muy poco)
# Compromiso: X=16, Y=4, Z=4, d=128 → 16 tokens/plano × 16 planos = 256 tokens
# P/Q = Linear(4*4*128, 4*4*128) = Linear(2048, 2048) = 4M — perfecto

vsn_config = VSNConfig(
    X_enc=16, X_dec=16,
    Y=4, Z=4, d=128,
    ics=16,  # Y*Z = 16
    Y_H=4, Z_H=4, d_H=128,
    p_mode="identity",
    Y_dec=4, Z_dec=4,
    dgw=16,
    head_type="regression",
    vgb_version="v3",
)
# Capacidad: 16 planos × 16 tokens/plano = 256 tokens
# P/Q: Linear(2048, 2048) = 4M params cada uno — razonable!
# Decoder: 16 planos × 16 = produce 16 tokens por ventana, con 16 windows = 256

model = VSNForLanguage(vsn_config, vocab_size=VOCAB_SIZE)

if n_gpus > 1:
    model = nn.DataParallel(model)
model = model.to(device)
base_model = model.module if hasattr(model, 'module') else model

# %% [markdown]
# ## Entrenamiento

# %%
EPOCHS = 3
LR = 3e-4
WARMUP_STEPS = 500
TOTAL_STEPS = EPOCHS * len(train_loader)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))

def get_lr(step):
    if step < WARMUP_STEPS: return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
    return 0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
use_amp = torch.cuda.is_available()

print(f"Training: {EPOCHS} ep, LR={LR}, steps={TOTAL_STEPS}, AMP={use_amp}")

# %%
def train_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss, total_tokens, t0 = 0.0, 0, time.time()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(x, num_windows=16)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=EOT_TOKEN)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x, num_windows=16)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=EOT_TOKEN)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        total_loss += loss.item()
        total_tokens += (y != EOT_TOKEN).sum().item()
        if (step+1) % 100 == 0:
            al = total_loss/(step+1); ppl = math.exp(min(al,20))
            print(f"  Ep{epoch+1} Step {step+1:5d}/{len(loader)} | Loss:{al:.3f} PPL:{ppl:.1f} | "
                  f"Tok/s:{total_tokens/(time.time()-t0):.0f} | LR:{scheduler.get_last_lr()[0]:.2e} [{time.time()-t0:.0f}s]", flush=True)
    al = total_loss/len(loader)
    return al, math.exp(min(al,20))

print("="*70)
print("  VSN REAL: IC → Φ → Encoder(16 planes) → P → H → Q → Decoder(16 planes+Ψ)")
print("="*70, flush=True)

for epoch in range(EPOCHS):
    loss, ppl = train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
    print(f"\n  ✓ Epoch {epoch+1}/{EPOCHS} — Loss: {loss:.3f}, PPL: {ppl:.1f}")
    torch.save({"model": base_model.state_dict(), "epoch": epoch+1, "loss": loss}, f"vsn_real_ep{epoch+1}.pt")
    print(f"  💾 Saved vsn_real_ep{epoch+1}.pt")
    ids = torch.tensor([enc.encode("Once upon a time")], device=device)
    gen = base_model.generate(ids, max_new_tokens=150, temperature=0.8)
    print(f"  >>> {enc.decode(gen[0].tolist())[:300]}\n", flush=True)

# %% [markdown]
# ## Generación Final

# %%
print("="*70)
print("  Generación Final")
print("="*70)
prompts = ["Once upon a time, there was a", "The brave knight went to", "A small cat found a",
           "In a magical forest,", "The little girl loved", "One sunny morning,"]
base_model.eval()
for p in prompts:
    ids = torch.tensor([enc.encode(p)], device=device)
    gen = base_model.generate(ids, max_new_tokens=200, temperature=0.7, top_k=50)
    print(f"\n  \"{p}\"")
    print(f"  {enc.decode(gen[0].tolist())[:400]}")
    print(f"  {'─'*50}")

# %%
save_path = "vsn_real_final.pt"
torch.save({"model": base_model.state_dict(), "config": vars(vsn_config),
            "params": sum(p.numel() for p in base_model.parameters())}, save_path)
print(f"Saved: {save_path} ({os.path.getsize(save_path)/1e6:.0f}MB)")
