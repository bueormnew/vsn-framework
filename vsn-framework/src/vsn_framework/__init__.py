"""VSN Framework — Volumetric Sequential Network.

Una librería completa para crear, entrenar y usar modelos VSN.
Combina el core matemático (vsn) y el framework operativo (vgb)
en una API unificada y fácil de usar.

Uso rápido:
    from vsn_framework import VSN

    # Crear modelo en una línea
    model = VSN.create("small", task="text", vocab_size=1000)
    
    # Entrenar
    model.fit(train_data, epochs=10)
    
    # Generar
    output = model.predict(tokens)
"""

__version__ = "0.1.0"

# Ensure internal packages are importable
import sys
import os
_pkg_root = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_pkg_root)))
_vsn_src = os.path.join(_project_root, "VSN", "src")
_vgb_src = os.path.join(_project_root, "VGB", "src")
for p in [_vsn_src, _vgb_src]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from vsn_framework.api import VSN, QuickModel
from vsn_framework.router import DataRouter, ArithmeticRouter, TextRouter

__all__ = [
    "VSN",
    "QuickModel",
    "DataRouter",
    "ArithmeticRouter",
    "TextRouter",
]
