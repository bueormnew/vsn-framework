"""RMSNorm — Root Mean Square Layer Normalization.

Implementación según VGB v1, Step 1:
    v̂ = (v / RMS(v)) ⊙ γ
    RMS(v) = √( (1/d) Σ v_k² + ε )

Equivalente a:
    output = x * rsqrt(mean(x², dim=-1, keepdim=True) + eps) * scale
"""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normaliza la última dimensión del tensor usando RMS y aplica
    un parámetro de escala entrenable (γ) de shape (d,).

    Args:
        d: Dimensión de la última axis (tamaño del vector a normalizar).
        eps: Epsilon para estabilidad numérica. Default: 1e-6.
    """

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d = d
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        """Aplica RMSNorm sobre la última dimensión.

        Args:
            x: Tensor de forma arbitraria (..., d) donde la última
               dimensión tiene tamaño d.

        Returns:
            Tensor normalizado de la misma forma que x.
        """
        # x.pow(2).mean(-1, keepdim=True) computa (1/d) Σ x_k²
        rms_inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms_inv * self.scale
