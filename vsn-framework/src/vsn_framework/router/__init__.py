"""Data Router — preparación automática de datos según la tarea.

Transforma datos crudos (texto, números, etc.) en tensores listos para
entrenar un modelo VSN. Evita errores de formato y tokenización.

Uso:
    from vsn_framework import ArithmeticRouter, TextRouter, DataRouter
    
    # Para aritmética
    router = ArithmeticRouter()
    inputs, targets = router.prepare(["5+5=10", "12+8=20", ...])
    
    # Para texto genérico
    router = TextRouter(vocab_size=1000)
    inputs, targets = router.prepare(["hello world", ...])
    
    # Router genérico
    router = DataRouter.for_task("arithmetic")
"""

from vsn_framework.router.base import DataRouter
from vsn_framework.router.arithmetic import ArithmeticRouter
from vsn_framework.router.text import TextRouter

__all__ = ["DataRouter", "ArithmeticRouter", "TextRouter"]
