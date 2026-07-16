"""TextRouter — prepara datos de texto para entrenamiento VSN.

Tokenización por caracteres o por palabras para tareas de lenguaje.

Uso:
    router = TextRouter(vocab_size=256)  # char-level
    inputs, targets = router.prepare(["hello world", "foo bar"])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from vsn_framework.router.base import DataRouter


class TextRouter(DataRouter):
    """Router para tareas de texto (char-level tokenization).
    
    Args:
        vocab_size: Tamaño del vocabulario (default 256 para ASCII).
        max_seq_len: Longitud máxima de secuencia.
    """
    
    PAD = 0
    BOS = 1
    EOS = 2
    
    def __init__(self, vocab_size: int = 256, max_seq_len: int = 64):
        self._vocab_size = vocab_size
        self._max_seq_len = max_seq_len
        # Build char-level vocab (ASCII)
        self._char_to_id: Dict[str, int] = {}
        self._id_to_char: Dict[int, str] = {}
        for i in range(min(vocab_size - 3, 253)):
            c = chr(i + 32)  # printable ASCII from space
            self._char_to_id[c] = i + 3
            self._id_to_char[i + 3] = c
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len
    
    def prepare(self, data: List[str], **kwargs) -> Tuple[Tensor, Tensor]:
        """Prepara texto para entrenamiento autoregresivo (next-char prediction).
        
        Args:
            data: Lista de strings de texto.
            
        Returns:
            (inputs, targets) para next-token prediction.
        """
        inputs_list = []
        targets_list = []
        
        for text in data:
            tokens = [self.BOS] + [self._char_to_id.get(c, self.PAD) for c in text] + [self.EOS]
            tokens = tokens[:self._max_seq_len]
            
            inp = tokens[:-1]
            tgt = tokens[1:]
            
            pad_n = self._max_seq_len - 1 - len(inp)
            inp = inp + [self.PAD] * pad_n
            tgt = tgt + [self.PAD] * pad_n
            
            inputs_list.append(inp)
            targets_list.append(tgt)
        
        return torch.tensor(inputs_list, dtype=torch.long), torch.tensor(targets_list, dtype=torch.long)
    
    def encode(self, sample: str) -> Tensor:
        """Codifica texto para inferencia."""
        tokens = [self.BOS] + [self._char_to_id.get(c, self.PAD) for c in sample]
        tokens = tokens[:self._max_seq_len - 1]
        pad_n = self._max_seq_len - 1 - len(tokens)
        tokens = tokens + [self.PAD] * pad_n
        return torch.tensor([tokens], dtype=torch.long)
    
    def decode(self, tensor: Tensor) -> str:
        """Decodifica tensor a texto."""
        chars = []
        ids = tensor.tolist() if isinstance(tensor, Tensor) else tensor
        for t in ids:
            if t == self.EOS:
                break
            if t in self._id_to_char:
                chars.append(self._id_to_char[t])
        return "".join(chars)
