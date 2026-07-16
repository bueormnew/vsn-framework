# %% [markdown]
# # VSN Pre-Training: Story Generation (GPT-2 style)
# 
# **Objetivo**: Pre-entrenar un modelo VSN "small" (4×4×4, d=64) para generación de texto
# similar a GPT-2, usando el tokenizer de GPT-2 y un dataset de cuentos/historias en inglés.
#
# **Arquitectura**: VSN con VGB v2 (spatial mixing) — complejidad lineal O(n)
#
# **Dataset**: TinyStories (historias cortas en inglés, ideales para modelos pequeños)
#
# **Capacidad teórica**: Secuencias infinitas via DGW (Decoder Generation Windows)

# %% [markdown]
# ## 1. Instalación

# %%
# !pip install vsn-framework tiktoken datasets --quiet

# %% [markdown]
# ## 2. Imports y configuración

# %%
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 3. Tokenizer (GPT-2)

# %%
import tiktoken

enc = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = enc.n_vocab  # 50257
PAD_TOKEN = enc.eot_token  # 50256 — end of text token

print(f"Vocab size: {VOCAB_SIZE}")
print(f"PAD/EOT token: {PAD_TOKEN}")
print(f"Example: '{enc.decode(enc.encode('Once upon a time'))}'")

# %% [markdown]
# ## 4. Dataset: TinyStories

# %%
from datasets import load_dataset

# TinyStories: dataset de historias cortas generadas, ideal para modelos pequeños
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
print(f"Dataset size: {len(dataset)} stories")
print(f"Example: {dataset[0]['text'][:200]}...")

# %% [markdown]
# ## 5. Data Processing

# %%
MAX_SEQ_LEN = 1024  # Tokens por secuencia
BATCH_SIZE = 8      # Ajustar según VRAM disponible

class StoriesDataset(Dataset):
    """Dataset que tokeniza historias y las divide en chunks de MAX_SEQ_LEN."""
    
    def __init__(self, texts, tokenizer, max_len=1024, max_samples=50000):
        self.max_len = max_len
        self.samples = []
        
        # Tokenizar y crear samples
        all_tokens = []
        for i, text in enumerate(texts):
            if i >= max_samples:
                break
            tokens = tokenizer.encode(text["text"])
            all_tokens.extend(tokens)
            all_tokens.append(PAD_TOKEN)  # separador entre historias
        
        # Dividir en chunks de max_len
        for i in range(0, len(all_tokens) - max_len, max_len // 2):  # overlap 50%
            chunk = all_tokens[i:i + max_len + 1]
            if len(chunk) == max_len + 1:
                self.samples.append(chunk)
        
        print(f"  Total tokens: {len(all_tokens):,}")
        print(f"  Samples (chunks): {len(self.samples):,}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # input
        y = torch.tensor(chunk[1:], dtype=torch.long)   # target (shifted)
        return x, y

print("Processing dataset...")
train_dataset = StoriesDataset(dataset, enc, max_len=MAX_SEQ_LEN, max_samples=50000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
print(f"Batches per epoch: {len(train_loader)}")

# %% [markdown]
# ## 6. Modelo VSN para Lenguaje
#
# Arquitectura: Embedding → VGB v2 blocks (encoder) → VGB v2 blocks (decoder) → LM Head
#
# La clave es que el VGB v2 tiene SPATIAL MIXING integrado en cada bloque,
# permitiendo que cada token vea a todos los demás tokens en la secuencia.

# %%
from vsn.core.vgb_v2 import VGBv2
from vsn.core.vgb_v3 import VGBv3
from vsn.core.rms_norm import RMSNorm

@dataclass
class VSNLanguageConfig:
    """Configuración del modelo VSN para lenguaje."""
    vocab_size: int = 50257
    d_model: int = 64         # Dimensión interna
    n_layers: int = 4         # Planos encoder (4 enc + 4 dec = 8 total)
    max_seq_len: int = 1024   # Largo máximo de secuencia
    dropout: float = 0.1
    
    @property
    def total_blocks(self):
        return self.n_layers * 2  # encoder + decoder


class VSNForLanguage(nn.Module):
    """VSN model optimizado para generación de lenguaje.
    
    Usa VGB v2 blocks con spatial mixing para procesamiento de secuencias.
    Compatible con generación autoregresiva via causal masking implícito
    en el entrenamiento teacher-forcing.
    
    Arquitectura:
        Token Embedding + Position Embedding
        → N bloques VGB v2 (encoder, propagación sobre "planos")
        → N bloques VGB v2 (decoder)  
        → LayerNorm + LM Head (Linear → vocab_size)
    """
    
    def __init__(self, config: VSNLanguageConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # VGB v3 encoder blocks (causal spatial mixing)
        self.encoder_blocks = nn.ModuleList([
            VGBv3(config.d_model, plane_idx=i, spatial_size=config.max_seq_len)
            for i in range(config.n_layers)
        ])
        
        # VGB v3 decoder blocks (causal spatial mixing)
        self.decoder_blocks = nn.ModuleList([
            VGBv3(config.d_model, plane_idx=i, spatial_size=config.max_seq_len)
            for i in range(config.n_layers)
        ])
        
        # Output
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: lm_head comparte pesos con tok_emb
        self.lm_head.weight = self.tok_emb.weight
        
        # Init weights
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"VSN Language Model: {n_params:,} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            idx: (batch, seq_len) token indices
            
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Seq len {T} > max {self.config.max_seq_len}"
        
        # Embeddings
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        h = self.tok_emb(idx) + self.pos_emb(pos)
        h = self.drop(h)
        
        # Reshape para VGB: (B, 1, T, d) — Y=1, Z=T
        p = h.unsqueeze(1)
        m = torch.zeros_like(p)
        
        # Encoder blocks
        for block in self.encoder_blocks:
            F_out, _, _, m = block(p, m)
            p = F_out
        
        # Decoder blocks
        for block in self.decoder_blocks:
            F_out, _, _, m = block(p, m)
            p = F_out
        
        # Output: (B, 1, T, d) → (B, T, d)
        out = p.squeeze(1)
        out = self.ln_f(out + h)  # residual global + norm
        logits = self.lm_head(out)
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        """Genera texto autoregressivamente.
        
        Args:
            idx: (1, prompt_len) tensor de token IDs del prompt
            max_new_tokens: cuántos tokens generar
            temperature: creatividad (1.0=normal, <1=más conservador)
            top_k: top-k sampling
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Truncar al max_seq_len
            idx_cond = idx[:, -self.config.max_seq_len:]
            
            # Forward
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # último token
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat([idx, next_token], dim=1)
            
            # Stop en end-of-text
            if next_token.item() == PAD_TOKEN:
                break
        
        return idx

# %% [markdown]
# ## 7. Configuración de Entrenamiento

# %%
# Configuración del modelo
# NOTA: En Kaggle con T4 (16GB VRAM), max_seq_len=1024 con d=64 funciona bien.
# Si tienes OOM, reduce max_seq_len a 512 o batch_size a 4.

config = VSNLanguageConfig(
    vocab_size=VOCAB_SIZE,
    d_model=64,           # "small" — aumentar a 128/256 para más capacidad
    n_layers=4,           # 4 encoder + 4 decoder = 8 VGB v2 blocks total
    max_seq_len=MAX_SEQ_LEN,
    dropout=0.1,
)

model = VSNForLanguage(config).to(device)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
)

# Learning rate scheduler (cosine with warmup)
EPOCHS = 3
WARMUP_STEPS = 100
TOTAL_STEPS = EPOCHS * len(train_loader)

def get_lr(step):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

print(f"\nTraining config:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Seq length: {MAX_SEQ_LEN}")
print(f"  Total steps: {TOTAL_STEPS}")
print(f"  Warmup steps: {WARMUP_STEPS}")

# %% [markdown]
# ## 8. Training Loop

# %%
def train_epoch(model, loader, optimizer, scheduler, epoch, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    t0 = time.time()
    
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        # Forward
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=PAD_TOKEN)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_tokens += (y != PAD_TOKEN).sum().item()
        
        # Log cada 50 steps
        if (step + 1) % 50 == 0:
            avg_loss = total_loss / (step + 1)
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t0
            tok_per_sec = total_tokens / elapsed
            lr = optimizer.param_groups[0]['lr'] * get_lr(epoch * len(loader) + step)
            print(f"  Ep{epoch+1} Step {step+1:4d}/{len(loader)} | "
                  f"Loss: {avg_loss:.3f} | PPL: {ppl:.1f} | "
                  f"Tok/s: {tok_per_sec:.0f} | LR: {lr:.2e} | "
                  f"[{elapsed:.0f}s]", flush=True)
    
    avg_loss = total_loss / len(loader)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


# Entrenar
print("="*60)
print("  VSN Pre-Training: Story Generation")
print("="*60)
print()

for epoch in range(EPOCHS):
    loss, ppl = train_epoch(model, train_loader, optimizer, scheduler, epoch, device)
    print(f"\n  ✓ Epoch {epoch+1}/{EPOCHS} complete — Loss: {loss:.3f}, PPL: {ppl:.1f}")
    
    # Generar sample al final de cada época
    print(f"\n  Sample generation:")
    prompt = "Once upon a time"
    prompt_ids = torch.tensor([enc.encode(prompt)], device=device)
    generated = model.generate(prompt_ids, max_new_tokens=100, temperature=0.8)
    text = enc.decode(generated[0].tolist())
    print(f"  >>> {text[:300]}")
    print()

# %% [markdown]
# ## 9. Evaluación y Generación Final

# %%
print("="*60)
print("  Generación Final — Diferentes Prompts")
print("="*60)

prompts = [
    "Once upon a time, there was a little",
    "The brave knight went to the",
    "A small cat found a",
    "In a magical forest, the animals",
    "The little girl loved to",
]

model.eval()
for prompt in prompts:
    prompt_ids = torch.tensor([enc.encode(prompt)], device=device)
    generated = model.generate(prompt_ids, max_new_tokens=150, temperature=0.8, top_k=40)
    text = enc.decode(generated[0].tolist())
    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Output: {text[:400]}")
    print(f"  {'─'*50}")

# %% [markdown]
# ## 10. Guardar Modelo

# %%
# Guardar checkpoint
save_path = "vsn_stories_model.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "max_seq_len": config.max_seq_len,
        "dropout": config.dropout,
    },
    "optimizer_state_dict": optimizer.state_dict(),
    "tokenizer": "gpt2",
}, save_path)
print(f"Model saved to: {save_path}")
print(f"Size: {os.path.getsize(save_path) / 1e6:.1f} MB")

# %% [markdown]
# ## 11. Notas de Escalado
#
# Este notebook usa la configuración "small" (d=64, 4+4 layers).
# Para mejor calidad de generación:
#
# | Config | d_model | Layers | Params | VRAM (est.) | Calidad |
# |--------|---------|--------|--------|-------------|---------|
# | tiny   | 32      | 2+2    | ~100K  | <1 GB       | Prueba  |
# | small  | 64      | 4+4    | ~2.8M  | ~2 GB       | Básica  |
# | medium | 128     | 6+6    | ~20M   | ~8 GB       | Buena   |
# | base   | 256     | 8+8    | ~140M  | ~16 GB      | Alta    |
#
# La arquitectura VSN escala linealmente en tiempo O(n), así que
# secuencias más largas (2048, 4096, 8192) son viables sin el
# cuello de botella cuadrático de los Transformers.
#
# **Para generar texto infinito**: usar DGW (Decoder Generation Windows)
# que permite generar en ventanas sucesivas, reutilizando el estado
# del decoder entre ventanas via el operador Ψ.
