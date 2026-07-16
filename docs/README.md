# VSN Framework

<p align="center">
  <strong>Volumetric Sequential Network — Arquitectura de complejidad lineal con propagación 3D</strong>
</p>

<p align="center">
  <code>pip install -e ./vsn-framework</code>
</p>

---

## ¿Qué es VSN?

**VSN** es una arquitectura de red neuronal que procesa secuencias en **tiempo lineal O(n)** organizando el cómputo como un volumen 3D (X×Y×Z×d) con propagación exclusiva sobre el eje X. A diferencia de los Transformers (O(n²)), VSN escala linealmente manteniendo alta capacidad de aprendizaje.

```
Tokens → Embedding → [Input Cache] → Encoder (VGB×X) → H → Decoder (VGB×X) → Output
```

### Benchmark: Aritmética

Con solo **203K parámetros** y **10 épocas** de entrenamiento en CPU:

| Época | Token Accuracy (eval) | Estado |
|-------|----------------------|--------|
| 1 | 42.7% | Aprendiendo |
| 3 | 80.1% | Convergiendo rápido |
| 5 | 98.5% | Casi perfecto |
| 8 | 99.6% | Dominado |
| 10 | **99.6%** | ✅ Completo |

El modelo aprende las 4 operaciones (suma, resta, multiplicación, división) con números de 1-99 en **63 segundos** en CPU. En teacher forcing verifica **10/10 respuestas correctas**.

---

## Instalación

```bash
cd "VSN- 3D"

# Opción 1: Instalar todo junto (recomendado)
pip install -e ./vsn-framework

# Opción 2: Instalar componentes individualmente
pip install -e ./VSN    # Core matemático
pip install -e ./VGB    # Framework operativo
```

---

## Uso Rápido

### Crear un modelo en una línea

```python
from vsn_framework import VSN

# Modelo con preset
model = VSN.create("small", task="text", vocab_size=32000)

# Modelo personalizado
model = VSN.create(X=4, Y=4, Z=4, d=64, task="regression")

# Ver resumen
print(model.summary())
# VSN Model (V2)
#   Volume:  4×4×4×64
#   Encoder: 4 planes
#   Decoder: 4 planes (DGW=1)
#   Task:    regression
#   Params:  2,781,056
#   VGB:     v2
```

### Presets disponibles

| Preset | Dimensiones | Parámetros | Uso |
|--------|-------------|-----------|-----|
| `"tiny"` | 2×2×2, d=32 | 104,528 | Pruebas rápidas |
| `"small"` | 4×4×4, d=64 | 2,781,056 | Experimentación |
| `"base"` | 8×8×8, d=128 | 141,745,664 | Producción |
| `"large"` | 16×16×16, d=256 | 8,684,930,048 | Gran escala |

### Entrenamiento con DataRouter (aritmética)

```python
from vsn_framework import VSN, ArithmeticRouter

# 1. Router prepara datos automáticamente
router = ArithmeticRouter()
inputs, targets = router.generate(n=5000, ops=["+", "-", "*", "/"])

# 2. Crear modelo
model = VSN.create("tiny", task="regression", vgb_version="v2")

# 3. Entrenar
history = model.fit(inputs, targets, epochs=10, lr=3e-3)
```

### Uso con texto

```python
from vsn_framework import VSN, TextRouter

router = TextRouter(vocab_size=256, max_seq_len=64)
inputs, targets = router.prepare(["hello world", "foo bar", ...])

model = VSN.create("small", task="text", vocab_size=256)
```

---

## Arquitectura

### VGB v2 — Voxel Gate Block con Spatial Mixing

El bloque fundamental de VSN. Evolución del VGB v1 que añade **comunicación entre posiciones** dentro de cada plano.

```
┌─────────────────────────────────────────────────────┐
│  VGB v2 — 7 pasos por plano                         │
├─────────────────────────────────────────────────────┤
│  1. RMSNorm(x)                                      │
│  2. SPATIAL MIXING: Linear(Y×Z, Y×Z) + residual  ← NUEVO │
│  3. Proyecciones: W_m, W_c, W_g                    │
│  4. Memoria gated: M = σ(g)·M_old + (1-σ(g))·m    │
│  5. MLP: d → 4d → d (GELU)                         │
│  6. Residual: r = x + MLP_out                      │
│  7. Salidas: F=r (→plano x+1), G=W_P2·r (→plano x+2) │
└─────────────────────────────────────────────────────┘
```

**¿Por qué spatial mixing?** Sin él, cada posición [y,z] se transforma independientemente (per-voxel). El mixing permite que **cada posición vea todas las demás** dentro del plano, habilitando razonamiento sobre relaciones entre tokens.

**Resultado**: VGB v2 alcanza 99.6% accuracy en aritmética donde VGB v1 se estanca en 38%.

### Diferencia v1 vs v2

| Aspecto | VGB v1 | VGB v2 |
|---------|--------|--------|
| Procesamiento | Per-voxel (independiente) | Con spatial mixing (cruzado) |
| Comunicación entre posiciones | Solo vía memoria M (débil) | Directa vía Linear(N,N) |
| Aritmética (5K datos, 30 ep) | 38% token acc, 0% exact | **99.6% token acc, 100% TF** |
| Parámetros extra | — | +N² por bloque (mínimo) |
| Pasos | 6 | 7 |

### Propagación sobre el eje X

```
Plano 0 ──F──► Plano 1 ──F──► Plano 2 ──F──► Plano 3 (output)
    │                │                │
    └───────G───────►└───────G───────►│
```

- **F**: Conexión directa (residual) al plano siguiente
- **G**: Conexión skip al plano x+2 (proyección aprendida)
- **M**: Memoria gated que se propaga secuencialmente entre planos

---

## Data Router

El DataRouter prepara datos automáticamente según la tarea, evitando errores de formato.

### ArithmeticRouter

```python
from vsn_framework import ArithmeticRouter

router = ArithmeticRouter(max_seq_len=12)

# Generar dataset automático
inputs, targets = router.generate(
    n=10000,           # Número de ejemplos
    ops=["+", "-"],    # Operaciones a incluir
    max_val=99,        # Valor máximo de operandos
    seed=42,           # Reproducibilidad
)

# O preparar datos propios
data = [("5+5", "10"), ("12-3", "9"), ("4*5", "20")]
inputs, targets = router.prepare(data)

# También acepta formato string
data_str = ["5+5=10", "12-3=9", "4*5=20"]
inputs, targets = router.prepare(data_str)
```

### TextRouter

```python
from vsn_framework import TextRouter

router = TextRouter(vocab_size=256, max_seq_len=64)
inputs, targets = router.prepare(["texto de ejemplo", ...])

# Codificar para inferencia
encoded = router.encode("hello")

# Decodificar output
text = router.decode(output_tensor)
```

### DataRouter genérico

```python
from vsn_framework import DataRouter

router = DataRouter.for_task("arithmetic")  # Factory automático
router = DataRouter.for_task("text", vocab_size=1000)
```

---

## Estructura del Proyecto

```
VSN-3D/
├── vsn-framework/          # Librería wrapper (API simplificada)
│   └── src/vsn_framework/
│       ├── api.py           # VSN.create(), QuickModel
│       └── router/          # DataRouter, ArithmeticRouter, TextRouter
│
├── VSN/                     # Core matemático
│   └── src/vsn/
│       ├── core/            # VGBv1, VGBv2, Encoder, Decoder, P, Q, Ψ, Model
│       ├── heads/           # TextHead, ClassificationHead, Regression, Dense
│       ├── losses/          # CrossEntropy, MSE, L1, MultiTaskLoss
│       ├── io/              # save_model, load_model
│       └── formats/         # Bundle export/import
│
├── VGB/                     # Framework operativo
│   └── src/vgb/
│       ├── config/          # Configuración tipada + YAML loader
│       ├── runtime/         # Bootstrap, AMP, FSDP2, DCP
│       ├── training/        # Trainer, loops, métricas
│       ├── inference/       # Predictor
│       └── cli/             # CLI (train, eval, infer, export, ...)
│
└── docs/                    # Documentación
```

---

## Configuración Avanzada

### Modelo totalmente personalizado

```python
from vsn.core import VSNConfig, VSNModel

config = VSNConfig(
    X_enc=8, X_dec=8,       # 8 planos encoder/decoder
    Y=8, Z=8, d=128,       # Plano 8×8 con d=128
    ics=64,                 # Input Cache Size
    Y_H=8, Z_H=8, d_H=128, # Plano latente H
    p_mode="identity",      # Proyección P: compress/identity/expand
    Y_dec=8, Z_dec=8,
    dgw=4,                  # Decoder Generation Window
    head_type="text",
    vocab_size=32000,
    vgb_version="v2",       # v1 o v2
)

model = VSNModel(config)
```

### Selección de bloque VGB

```python
# VGB v2 (recomendado — con spatial mixing)
model = VSN.create("small", vgb_version="v2")

# VGB v1 (original — per-voxel, para investigación)
model = VSN.create("small", vgb_version="v1")
```

### Guardar y cargar modelos

```python
from vsn.io import save_model, load_model
from vsn.formats import export_bundle, load_bundle

# Checkpoint simple
save_model(model.vsn, "model.pt")
loaded = load_model("model.pt")

# Bundle de inferencia (con checksums SHA-256)
export_bundle(model.vsn, "bundles/my_model/")
serving_model = load_bundle("bundles/my_model/")
```

---

## CLI

```bash
# Entrenar
vsn train --config configs/train/single_gpu.yaml

# Validar config
vsn validate-config --config my_config.yaml

# Exportar bundle
vsn export --config config.yaml --checkpoint ckpts/best.pt --output-dir bundle/

# Inspeccionar checkpoint
vsn inspect-checkpoint ckpts/step_1000.pt
```

---

## Invariantes Arquitectónicos

Estos principios se mantienen en v1 Y v2:

1. Propagación exclusivamente sobre eje X
2. Todos los planos usan el mismo bloque (VGB v1 o v2)
3. Parámetros independientes por plano (θ_x ≠ θ_x')
4. Sin bloques especializados por profundidad
5. H es la única interfaz encoder-decoder
6. Input Cache siempre precede al encoder
7. El VGB puede evolucionar conservando estos principios

---

## Complejidad

| Métrica | VSN | Transformer |
|---------|-----|-------------|
| Tiempo (secuencia) | **O(n)** | O(n²) |
| Memoria atención | **—** (no hay) | O(n²) |
| Escalado | Incrementar X,Y,Z | Rediseñar |

---

## Testing

```bash
# 635 tests totales
pytest VSN/tests/ -q          # 443 tests (core)
pytest VGB/tests/ -q          # 192 tests (framework)

# Verificar VGB v2
python -c "from vsn.core import VGBv2; print('OK')"
```

---

## Benchmark: Aritmética Simple

**Configuración**: VGB-v2 ×4, d=64 | 203K params | CPU | 10 épocas

**Dataset**: ~7000 operaciones únicas (+, -, ×, ÷) con números 1-99

**Resultados**:
```
  Ep 1  TL=1.975  EA=42.7%   ← Aprendiendo distribución
  Ep 3  TL=0.969  EA=80.1%   ← Convergencia rápida  
  Ep 5  TL=0.129  EA=98.5%   ← Casi perfecto
  Ep10  TL=0.020  EA=99.6%   ← Dominado
```

**Verificación teacher forcing** (10/10 correctas):
```
  ✓ 91-56 = 35     ✓ 42+2 = 44     ✓ 84/7 = 12
  ✓ 37-5 = 32      ✓ 98-8 = 90     ✓ 5*12 = 60
  ✓ 2+25 = 27      ✓ 27+38 = 65    ✓ 39-28 = 11
  ✓ 1+29 = 30
```

**Comparación con otras arquitecturas** (misma tarea, mismos datos):

| Modelo | Params | Token Acc (30 ep) | Aprende? |
|--------|--------|-------------------|----------|
| VGB v1 (per-voxel) | 203K | 38% → estancado | ❌ |
| LSTM (2 layers) | 70K | 49% → mejorando | Parcial |
| **VGB v2 (spatial mix)** | 203K | **99.6%** en 10 ep | ✅ Perfecto |

---

## Licencia

MIT

---

*VSN Framework v0.2.0 — Julio 2026*
