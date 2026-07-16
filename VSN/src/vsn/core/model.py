"""VSNModel — Modelo ensamblado completo.

Integra todos los componentes de la arquitectura VSN en un solo nn.Module:
    Input Cache → Encoder → P → Q → Decoder(+Ψ) → Head → ModelOutputs

El forward ejecuta la cadena completa en orden estricto según Requisito 6.2.
La instanciación valida todos los contratos dimensionales (Requisitos 6.3, 6.5).

Uso:
    config = VSNConfig.small()
    model = VSNModel(config)
    outputs = model(tokens)  # tokens: (batch, num_tokens, d)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

import torch.nn as nn
from torch import Tensor

from vsn.contracts.outputs import ModelOutputs
from vsn.core.config import ConfigurationError, VSNConfig
from vsn.core.decoder import VSNDecoder
from vsn.core.encoder import VSNEncoder
from vsn.core.latent import ProjectorP
from vsn.core.transitions import TransitionQ


class HeadProtocol(Protocol):
    """Protocolo mínimo que debe cumplir un head externo."""

    def __call__(self, decoder_states: List[Tensor], metadata: Dict[str, Any]) -> ModelOutputs:
        ...


class VSNModel(nn.Module):
    """Modelo VSN completo: IC → Encoder → P → Q → Decoder(+Ψ) → Head.

    Ensambla todos los componentes de la arquitectura volumétrica secuencial
    en un único nn.Module con forward que respeta el orden estricto de ejecución.

    Args:
        config: VSNConfig con todas las dimensiones y parámetros del modelo.
        head: Módulo head opcional. Si se provee, se invoca sobre los estados
            del decoder para producir logits/embeddings. Si es None, el forward
            retorna ModelOutputs con los estados del decoder directamente.
        num_windows: Número de ventanas DGW a generar por defecto en el decoder.
            Puede sobreescribirse en el forward.

    Raises:
        ConfigurationError: Si la configuración tiene dimensiones incompatibles.
    """

    def __init__(
        self,
        config: VSNConfig,
        head: Optional[nn.Module] = None,
        num_windows: int = 1,
    ) -> None:
        super().__init__()

        # Paso 1: Validar configuración — fail fast
        config.validate()

        # Almacenar configuración para inspección
        self.config = config
        self.num_windows = num_windows

        # Paso 2: Construir componentes en orden de la cadena

        # Encoder: Input Cache + Φ + VGB blocks
        self.encoder = VSNEncoder(
            X=config.X_enc,
            Y=config.Y,
            Z=config.Z,
            d=config.d,
            ics=config.ics,
            vgb_version=config.vgb_version,
        )

        # P: Proyección V_{X-1} → H
        self.P = ProjectorP(
            Y=config.Y,
            Z=config.Z,
            d=config.d,
            Y_H=config.Y_H,
            Z_H=config.Z_H,
            d_H=config.d_H,
            mode=config.p_mode,
        )

        # Q: Transición H → V^dec_0
        self.Q = TransitionQ(
            Y_H=config.Y_H,
            Z_H=config.Z_H,
            d_H=config.d_H,
            Y_dec=config.Y_dec,
            Z_dec=config.Z_dec,
            d=config.d,
        )

        # Decoder: VGB blocks + Ψ para continuidad entre ventanas
        self.decoder = VSNDecoder(
            X_dec=config.X_dec,
            Y=config.Y_dec,
            Z=config.Z_dec,
            d=config.d,
            dgw=config.dgw,
            vgb_version=config.vgb_version,
        )

        # Head: opcional — si no se provee, retornamos estados directamente
        self.head = head

    def forward(
        self,
        tokens: Tensor,
        num_windows: Optional[int] = None,
    ) -> ModelOutputs:
        """Forward completo: IC → Encoder → P → Q → Decoder(+Ψ) → Head.

        Ejecuta la cadena de procesamiento en orden estricto según la
        especificación formal. Cada componente se invoca exactamente una vez
        y en el orden definido.

        Args:
            tokens: Tensor de shape (batch, num_tokens, d) — tokens embedidos.
            num_windows: Número de ventanas DGW a generar. Si None, usa
                el valor configurado en __init__.

        Returns:
            ModelOutputs con:
                - logits: del head si está presente, None si no
                - states: dict con decoder_states (lista de tensores por ventana)
                          y latent_H (plano latente)
                - metadata: info del config y dimensiones
        """
        windows = num_windows if num_windows is not None else self.num_windows

        # 1. Encoder: tokens → V_{X-1}
        V_last = self.encoder(tokens)  # (batch, Y, Z, d)

        # 2. P: V_{X-1} → H
        H = self.P(V_last)  # (batch, Y_H, Z_H, d_H)

        # 3. Q: H → V^dec_0
        V_dec_0 = self.Q(H)  # (batch, Y_dec, Z_dec, d)

        # 4. Decoder(+Ψ): V^dec_0 → lista de estados por ventana
        decoder_states = self.decoder(V_dec_0, num_windows=windows)
        # decoder_states: List[Tensor], cada uno (batch, Y_dec, Z_dec, d)

        # 5. Head (si existe): estados del decoder → ModelOutputs
        metadata: Dict[str, Any] = {
            "model_family": self.config.model_family,
            "vgb_version": self.config.vgb_version,
            "psi_version": self.config.psi_version,
            "head_type": self.config.head_type,
            "num_windows": windows,
            "X_enc": self.config.X_enc,
            "X_dec": self.config.X_dec,
        }

        states: Dict[str, Any] = {
            "decoder_states": decoder_states,
            "latent_H": H,
        }

        if self.head is not None:
            # Delegar al head — el head produce ModelOutputs completo
            head_output = self.head(decoder_states, metadata)
            # Preservar states si el head no los incluyó
            if head_output.states is None:
                head_output.states = states
            return head_output

        # Sin head: retornar estados directamente
        return ModelOutputs(
            logits=None,
            embeddings=None,
            aux_losses={},
            states=states,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        return (
            f"VSNModel(\n"
            f"  config={self.config.model_family}/"
            f"{self.config.vgb_version},\n"
            f"  encoder=VSNEncoder(X={self.config.X_enc}, "
            f"Y={self.config.Y}, Z={self.config.Z}, d={self.config.d}),\n"
            f"  P={self.config.p_mode}: "
            f"({self.config.Y},{self.config.Z},{self.config.d}) → "
            f"({self.config.Y_H},{self.config.Z_H},{self.config.d_H}),\n"
            f"  Q: ({self.config.Y_H},{self.config.Z_H},{self.config.d_H}) → "
            f"({self.config.Y_dec},{self.config.Z_dec},{self.config.d}),\n"
            f"  decoder=VSNDecoder(X={self.config.X_dec}, "
            f"dgw={self.config.dgw}),\n"
            f"  head={'present' if self.head else 'none'},\n"
            f"  total_params={total_params:,}\n"
            f")"
        )
