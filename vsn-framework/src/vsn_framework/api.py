"""API simplificada para VSN — crear, entrenar y usar modelos en pocas líneas.

Uso:
    from vsn_framework import VSN
    
    # Opción 1: Modelo rápido
    model = VSN.create("small", task="text", vocab_size=32000)
    
    # Opción 2: Modelo personalizado
    model = VSN.create(X=4, Y=4, Z=4, d=64, task="regression")
    
    # Entrenar
    model.fit(train_tokens, train_targets, epochs=10, lr=1e-3)
    
    # Predecir
    outputs = model.predict(tokens)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from vsn.core.config import VSNConfig
from vsn.core.model import VSNModel
from vsn.core.vgb_v2 import VGBv2
from vsn.core.vgb import VGBv1


# ── Presets ──────────────────────────────────────────────────────────

PRESETS = {
    "tiny": {"X": 2, "Y": 2, "Z": 2, "d": 32},
    "small": {"X": 4, "Y": 4, "Z": 4, "d": 64},
    "base": {"X": 8, "Y": 8, "Z": 8, "d": 128},
    "large": {"X": 16, "Y": 16, "Z": 16, "d": 256},
}


class VSN:
    """API principal de VSN Framework.
    
    Provee factory methods para crear modelos de forma simple o personalizada.
    """
    
    @staticmethod
    def create(
        preset: Optional[str] = None,
        *,
        # Dimensiones (override de preset o custom)
        X: Optional[int] = None,
        Y: Optional[int] = None,
        Z: Optional[int] = None,
        d: Optional[int] = None,
        # Tarea
        task: str = "regression",
        vocab_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        # VGB version
        vgb_version: str = "v2",
        # Avanzado
        X_dec: Optional[int] = None,
        dgw: int = 1,
    ) -> "QuickModel":
        """Crea un modelo VSN listo para usar.
        
        Args:
            preset: Nombre del preset ("tiny", "small", "base", "large") o None para custom.
            X, Y, Z, d: Dimensiones del volumen (override del preset).
            task: Tipo de tarea ("text", "classification", "regression", "dense").
            vocab_size: Tamaño del vocabulario (requerido si task="text").
            num_classes: Número de clases (requerido si task="classification").
            vgb_version: Versión del bloque VGB ("v1" o "v2"). Default: "v2".
            X_dec: Profundidad del decoder (default = X).
            dgw: Tamaño de ventana DGW.
            
        Returns:
            QuickModel listo para entrenar y usar.
            
        Examples:
            >>> model = VSN.create("small", task="text", vocab_size=1000)
            >>> model = VSN.create(X=4, Y=4, Z=4, d=64, task="regression")
            >>> model = VSN.create("tiny", task="classification", num_classes=10)
        """
        # Resolver dimensiones
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(f"Preset '{preset}' no válido. Opciones: {list(PRESETS.keys())}")
            dims = PRESETS[preset].copy()
        else:
            dims = {}
        
        # Override con valores explícitos
        if X is not None: dims["X"] = X
        if Y is not None: dims["Y"] = Y
        if Z is not None: dims["Z"] = Z
        if d is not None: dims["d"] = d
        
        # Defaults si no se especificaron
        dims.setdefault("X", 4)
        dims.setdefault("Y", 4)
        dims.setdefault("Z", 4)
        dims.setdefault("d", 64)
        
        _X = dims["X"]
        _Y = dims["Y"]
        _Z = dims["Z"]
        _d = dims["d"]
        _X_dec = X_dec if X_dec is not None else _X
        
        config = VSNConfig(
            X_enc=_X, X_dec=_X_dec,
            Y=_Y, Z=_Z, d=_d,
            ics=_Y * _Z,
            Y_H=_Y, Z_H=_Z, d_H=_d,
            p_mode="identity",
            Y_dec=_Y, Z_dec=_Z,
            dgw=dgw,
            head_type=task,
            vocab_size=vocab_size,
            num_classes=num_classes,
            vgb_version=vgb_version,
        )
        
        return QuickModel(config)


class QuickModel(nn.Module):
    """Modelo VSN con API de alto nivel para entrenar y predecir.
    
    Wrappea VSNModel añadiendo:
    - .fit() para entrenamiento simple
    - .predict() para inferencia
    - .summary() para ver la arquitectura
    - Embedding interno opcional
    """
    
    def __init__(self, config: VSNConfig):
        super().__init__()
        self.config = config
        self.vsn = VSNModel(config)
        self._device = "cpu"
        self._trained = False
    
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def summary(self) -> str:
        """Resumen del modelo."""
        lines = [
            f"VSN Model ({self.config.vgb_version.upper()})",
            f"  Volume:  {self.config.X_enc}×{self.config.Y}×{self.config.Z}×{self.config.d}",
            f"  Encoder: {self.config.X_enc} planes",
            f"  Decoder: {self.config.X_dec} planes (DGW={self.config.dgw})",
            f"  Task:    {self.config.head_type}",
            f"  Params:  {self.num_params:,}",
            f"  VGB:     {self.config.vgb_version}",
        ]
        return "\n".join(lines)
    
    def forward(self, tokens: Tensor, num_windows: int = 1):
        """Forward pass directo."""
        return self.vsn(tokens, num_windows=num_windows)
    
    def fit(
        self,
        inputs: Tensor,
        targets: Tensor,
        *,
        epochs: int = 10,
        lr: float = 3e-3,
        batch_size: int = 128,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Entrena el modelo con datos directamente.
        
        Args:
            inputs: Tensor de inputs (batch, seq_len, d) o (batch, seq_len) si son IDs.
            targets: Tensor de targets para la loss.
            epochs: Número de épocas.
            lr: Learning rate.
            batch_size: Tamaño de batch.
            verbose: Imprimir progreso.
            
        Returns:
            Diccionario con historial de loss por época.
        """
        self.train()
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        
        history = {"loss": [], "time": []}
        t0 = time.time()
        
        for ep in range(epochs):
            ep_loss = 0.0
            for batch_inputs, batch_targets in loader:
                outputs = self.vsn(batch_inputs)
                # Compute loss based on task
                decoder_states = outputs.states["decoder_states"]
                last_state = decoder_states[-1]  # (B, Y, Z, d)
                pooled = last_state.mean(dim=(1, 2))  # (B, d)
                
                loss = F.mse_loss(pooled, batch_targets)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                ep_loss += loss.item()
            
            avg_loss = ep_loss / len(loader)
            history["loss"].append(avg_loss)
            history["time"].append(time.time() - t0)
            
            if verbose:
                print(f"  Epoch {ep+1}/{epochs} — Loss: {avg_loss:.4f} [{time.time()-t0:.0f}s]")
        
        self._trained = True
        return history
    
    def predict(self, tokens: Tensor, num_windows: int = 1):
        """Ejecuta inferencia sin gradientes.
        
        Args:
            tokens: Tensor de input (batch, num_tokens, d).
            num_windows: Número de ventanas DGW a generar.
            
        Returns:
            ModelOutputs con los estados del modelo.
        """
        self.eval()
        with torch.no_grad():
            return self.vsn(tokens, num_windows=num_windows)
    
    def save(self, path: str) -> None:
        """Guarda el modelo."""
        from vsn.io.save_load import save_model
        save_model(self.vsn, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "QuickModel":
        """Carga un modelo guardado."""
        from vsn.io.save_load import load_model
        vsn_model = load_model(path, device=device)
        instance = cls(vsn_model.config)
        instance.vsn = vsn_model
        instance._trained = True
        return instance
