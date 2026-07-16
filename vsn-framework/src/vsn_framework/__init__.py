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

__version__ = "0.1.5"

from vsn_framework.api import VSN, QuickModel
from vsn_framework.router import DataRouter, ArithmeticRouter, TextRouter

__all__ = [
    "VSN",
    "QuickModel",
    "DataRouter",
    "ArithmeticRouter",
    "TextRouter",
]
