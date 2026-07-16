# %% [markdown]
# # VSN Pre-Training v2: Arquitectura REAL (Input Cache → Encoder → P → Q → Decoder + Ψ)
#
# Este notebook usa la arquitectura VSN COMPLETA:
# - **Input Cache**: acumula tokens antes de poblar el volumen
# - **Encoder volumétrico**: posiciona tokens en X×Y×Z con Φ, propaga sobre eje X
# - **Plano latente H**: P proyecta el último plano del encoder
# - **Decoder volumétrico**: Q inicializa desde H, genera por ventanas DGW
# - **Operador Ψ**: transporta estado entre ventanas → generación infinita
#
# NO es un transformer. NO usa atención. Complejidad O(n).

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
# Cada historia se tokeniza y se corta a 1024 tokens (ventana de entrenamiento)
MAX_TOKENS = 1024
BATCH_SIZE = 8 * max(n_gpus, 1)  # 8 por GPU

class StoriesDataset(Dataset):
    def __init__(self, texts, tokenizer, max_tokens=1024, max_samples=150000):
        self.samples = []
        for i, item in enumerate(texts):
            if i >= max_samples: break
            tokens = tokenizer.encode(item["text"])
            if len(tokens) >= 20:
                self.samples.append(tokens[:max_tokens])
        lengths = [len(s) for s in self.samples]
        print(f"  Samples: {len(self.samples):,} | Avg len: {sum(lengths)//len(lengths)} | Total tok: {sum(lengths):,}")
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    max_len = min(max(len(s) for s in batch) + 1, MAX_TOKENS + 1)
    inputs, targets = [], []
    for tokens in batch:
        padded = tokens + [EOT_TOKEN] * (max_len - len(tokens))
        padded = padded[:max_len]
        inputs.append(padded[:-1])
        targets.append(padded[1:])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

train_dataset = StoriesDataset(dataset, enc, max_tokens=MAX_TOKENS, max_samples=150000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=collate_fn, num_workers=2, drop_last=True, pin_memory=True)
print(f"Batches/epoch: {len(train_loader)}")

# %% [markdown]
# ## Modelo: Arquitectura VSN Real
#
# Pipeline real:
# ```
# token_ids → Embedding → VSNModel(Input Cache → Φ → Encoder → P → H → Q → Decoder + Ψ) → LM Head
# ```
# 
# El volumen es X×Y×Z. Los tokens se posicionan en el volumen con Φ (raster).
# El decoder genera ventanas DGW. Ψ permite generación infinita.

# %%
from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel

class VSNForLanguage(nn.Module):
    """Wrapper que conecta la arquitectura VSN REAL con generación de lenguaje.
    
    Componentes internos (todos de la librería vsn):
    - VSNModel: Input Cache → Encoder (VGB v3) → P → H → Q → Decoder (VGB v3 + Ψ)
    - Embedding: token IDs → vectores d-dimensionales
    - LM Head: estados del decoder → logits sobre vocabulario
    
    La generación infinita usa DGW windows:
    - Cada ventana produce Y_dec * Z_dec tokens
    - Ψ transporta estado entre ventanas
    - No hay límite teórico en la longitud generada
    """
    
    def __init__(self, vsn_config: VSNConfig, vocab_size: int = 50257):
        super().__init__()
        self.vsn_config = vsn_config
        self.vocab_size = vocab_size
        self.tokens_per_window = vsn_config.Y_dec * vsn_config.Z_dec
        
        # Embedding: token IDs → (batch, seq, d)
        self.tok_emb = nn.Embedding(vocab_size, vsn_config.d)
        
        # Arquitectura VSN completa
        self.vsn = VSNModel(vsn_config)
        
        # LM Head: (batch, Y_dec, Z_dec, d) → (batch, Y_dec*Z_dec, vocab)
        self.lm_head = nn.Linear(vsn_config.d, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"VSN Language (REAL architecture): {n_params:,} params ({n_params/1e6:.1f}M)")
        print(f"  Volume: X={vsn_config.X_enc}, Y={vsn_config.Y}, Z={vsn_config.Z}, d={vsn_config.d}")
        print(f"  Tokens per plane: {vsn_config.Y * vsn_config.Z}")
        print(f"  Max input tokens: {vsn_config.X_enc * vsn_config.Y * vsn_config.Z}")
        print(f"  DGW window: {self.tokens_per_window} tokens")
        print(f"  VGB version: {vsn_config.vgb_version}")
    
    def forward(self, token_ids: torch.Tensor, num_windows: int = 1):
        """Forward completo con arquitectura VSN real.
        
        Args:
            token_ids: (batch, seq_len) — IDs de tokens
            num_windows: ventanas DGW para el decoder
            
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = token_ids.shape
        
        # 1. Embedding
        embeddings = self.tok_emb(token_ids)  # (B, T, d)
        
        # 2. Pasar por VSNModel completo:
        #    Input Cache → Φ posicionamiento → Encoder → P → H → Q → Decoder(+Ψ)
        #    El VSNModel espera (batch, num_tokens, d) donde num_tokens ≤ X*Y*Z
        max_input = self.vsn_config.X_enc * self.vsn_config.Y * self.vsn_config.Z
        
        # Procesar la secuencia en chunks de max_input tokens
        all_logits = []
        
        for start in range(0, T, max_input):
            end = min(start + max_input, T)
            chunk = embeddings[:, start:end, :]  # (B, chunk_len, d)
            chunk_len = chunk.shape[1]
            
            # Pad a múltiplo de Y*Z si necesario (para que Φ funcione)
            plane_size = self.vsn_config.Y * self.vsn_config.Z
            if chunk_len % plane_size != 0:
                pad_n = plane_size - (chunk_len % plane_size)
                chunk = F.pad(chunk, (0, 0, 0, pad_n))
            
            # Forward VSN real
            outputs = self.vsn(chunk, num_windows=num_windows)
            
            # Decoder states: lista de (B, Y_dec, Z_dec, d) por ventana
            # Aplanar cada ventana: (B, Y_dec*Z_dec, d) y concatenar
            dec_states = outputs.states["decoder_states"]
            
            # Convertir estados del decoder a logits
            window_logits = []
            for state in dec_states:
                flat = state.reshape(B, -1, self.vsn_config.d)  # (B, Y*Z, d)
                logits = self.lm_head(flat)  # (B, Y*Z, vocab)
                window_logits.append(logits)
            
            # Concatenar ventanas y tomar solo chunk_len posiciones
            chunk_logits = torch.cat(window_logits, dim=1)[:, :chunk_len, :]
            all_logits.append(chunk_logits)
        
        return torch.cat(all_logits, dim=1)  # (B, T, vocab)
    
    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 200, 
                 temperature: float = 0.8, top_k: int = 50):
        """Generación autoregresiva usando la arquitectura VSN real.
        
        Usa el decoder con ventanas DGW para generación continuada.
        El contexto se procesa a través del pipeline completo.
        """
        self.eval()
        max_ctx = self.vsn_config.X_enc * self.vsn_config.Y * self.vsn_config.Z
        
        for _ in range(max_new_tokens):
            # Tomar los últimos max_ctx tokens como contexto
            ctx = token_ids[:, -max_ctx:]
            
            # Forward completo
            logits = self(ctx, num_windows=1)
            logits = logits[:, -1, :] / temperature
            
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
            
            if next_token.item() == EOT_TOKEN:
                break
        
        return token_ids

# %%
# Configuración: ~30-50M params
# X=4 planos, Y=16, Z=16, d=128 → 256 tokens por plano, 1024 tokens totales
vsn_config = VSNConfig(
    X_enc=4, X_dec=4,
    Y=16, Z=16, d=128,
    ics=256,  # Y*Z = 256 tokens en el input cache
    Y_H=16, Z_H=16, d_H=128,
    p_mode="identity",
    Y_dec=16, Z_dec=16,
    dgw=4,
    head_type="regression",  # usamos LM head externo
    vgb_version="v3",
)

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
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
    return 0.3 + 0.7 * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
use_amp = torch.cuda.is_available()

print(f"Training: {EPOCHS} epochs, LR={LR}, warmup={WARMUP_STEPS}, total_steps={TOTAL_STEPS}")
print(f"AMP: {use_amp} | GPUs: {n_gpus}")

# %%
def train_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss, total_tokens = 0.0, 0
    t0 = time.time()
    
    for step, (x, y) in enumerate(loader):
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
        total_tokens += (y != EOT_TOKEN).sum().item()
        
        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step + 1)
            ppl = math.exp(min(avg_loss, 20))
            tok_s = total_tokens / (time.time() - t0)
            lr = scheduler.get_last_lr()[0]
            print(f"  Ep{epoch+1} Step {step+1:5d}/{len(loader)} | "
                  f"Loss: {avg_loss:.3f} PPL: {ppl:.1f} | "
                  f"Tok/s: {tok_s:.0f} | LR: {lr:.2e} [{time.time()-t0:.0f}s]", flush=True)
    
    return total_loss / len(loader), math.exp(min(total_loss / len(loader), 20))

# ── Training loop ──
print("="*70)
print("  VSN REAL Architecture — Pre-Training Stories")
print("  Pipeline: Input Cache → Φ → Encoder → P → H → Q → Decoder + Ψ")
print("="*70, flush=True)

for epoch in range(EPOCHS):
    loss, ppl = train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
    print(f"\n  ✓ Epoch {epoch+1}/{EPOCHS} — Loss: {loss:.3f}, PPL: {ppl:.1f}")
    
    # Guardar checkpoint
    ckpt = f"vsn_real_epoch{epoch+1}.pt"
    torch.save({"model_state_dict": base_model.state_dict(), "epoch": epoch+1, 
                "loss": loss, "vsn_config": vars(vsn_config)}, ckpt)
    print(f"  💾 Saved: {ckpt}")
    
    # Generar sample
    prompt = "Once upon a time"
    ids = torch.tensor([enc.encode(prompt)], device=device)
    gen = base_model.generate(ids, max_new_tokens=150, temperature=0.8)
    print(f"  >>> {enc.decode(gen[0].tolist())[:400]}\n", flush=True)

# %% [markdown]
# ## Generación Final

# %%
print("="*70)
print("  Generación Final — Arquitectura VSN Real")
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
    ids = torch.tensor([enc.encode(prompt)], device=device)
    gen = base_model.generate(ids, max_new_tokens=200, temperature=0.7, top_k=50)
    text = enc.decode(gen[0].tolist())
    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Output: {text[:500]}")
    print(f"  {'─'*60}")

# %%
# Guardar modelo final
save_path = "vsn_real_stories_final.pt"
torch.save({
    "model_state_dict": base_model.state_dict(),
    "vsn_config": vars(vsn_config),
    "vocab_size": VOCAB_SIZE,
    "tokenizer": "gpt2",
    "n_params": sum(p.numel() for p in base_model.parameters()),
}, save_path)
print(f"Final model saved: {save_path} ({os.path.getsize(save_path)/1e6:.0f}MB)")
