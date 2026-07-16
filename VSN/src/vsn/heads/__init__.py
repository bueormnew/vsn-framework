"""Output heads O: text, classification, regression, dense.

Exporta:
- BaseHead: ABC para todos los heads de salida.
- TextHead: Head de proyección a vocabulario para modelado de lenguaje.
- ClassificationHead: Head de clasificación (pooling + linear a clases).
- RegressionHead: Head de regresión (pooling + linear continuo).
- DenseHead: Head denso (proyección per-voxel sin colapsar estructura).
- Funciones de agregación: last_token, mean_pool, max_pool, cls_token, build_aggregation.
- Registry: HEAD_REGISTRY, register_head, build_head.
"""

from vsn.heads.base import (
    AGGREGATION_REGISTRY,
    HEAD_REGISTRY,
    BaseHead,
    build_aggregation,
    build_head,
    cls_token,
    last_token,
    max_pool,
    mean_pool,
    register_head,
)
from vsn.heads.classification import ClassificationHead
from vsn.heads.dense import DenseHead
from vsn.heads.regression import RegressionHead
from vsn.heads.text import TextHead

__all__ = [
    "AGGREGATION_REGISTRY",
    "HEAD_REGISTRY",
    "BaseHead",
    "ClassificationHead",
    "DenseHead",
    "RegressionHead",
    "TextHead",
    "build_aggregation",
    "build_head",
    "cls_token",
    "last_token",
    "max_pool",
    "mean_pool",
    "register_head",
]
