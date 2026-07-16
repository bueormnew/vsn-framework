"""ArithmeticRouter — prepara datos de aritmética para entrenamiento VSN.

Convierte expresiones como "5+5=10" en tensores tokenizados con formato:
    BOS + question + SEP + answer + EOS + PAD...

Genera datasets automáticamente o acepta datos externos.

Uso:
    router = ArithmeticRouter(max_seq_len=12)
    
    # Opción 1: datos propios
    inputs, targets = router.prepare(["5+5=10", "12-3=9", "4*5=20"])
    
    # Opción 2: generar dataset automático
    inputs, targets = router.generate(n=5000, ops=["+", "-", "*", "/"])
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from vsn_framework.router.base import DataRouter


class ArithmeticRouter(DataRouter):
    """Router para tareas de aritmética.
    
    Vocabulario fijo de 18 tokens:
        PAD=0, BOS=1, EOS=2, SEP=3, '0'-'9'=4-13, '+'=14, '-'=15, '*'=16, '/'=17
    
    Args:
        max_seq_len: Longitud máxima de secuencia (default 12).
    """
    
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3
    
    CHAR_TO_ID = {
        '0': 4, '1': 5, '2': 6, '3': 7, '4': 8,
        '5': 9, '6': 10, '7': 11, '8': 12, '9': 13,
        '+': 14, '-': 15, '*': 16, '/': 17,
    }
    ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()}
    VOCAB_SIZE = 18
    
    def __init__(self, max_seq_len: int = 12):
        self._max_seq_len = max_seq_len
    
    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE
    
    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len
    
    @property
    def d_model(self) -> int:
        """Dimensión mínima recomendada para el modelo."""
        return 64
    
    def prepare(self, data: List[str], **kwargs) -> Tuple[Tensor, Tensor]:
        """Prepara datos de aritmética.
        
        Args:
            data: Lista de strings con formato "expr=answer" 
                  (ej: ["5+5=10", "12-3=9"])
                  O lista de tuplas (question, answer):
                  (ej: [("5+5", "10"), ("12-3", "9")])
        
        Returns:
            (inputs, targets) — tensores de shape (N, max_seq_len-1)
        """
        inputs_list = []
        targets_list = []
        
        for item in data:
            if isinstance(item, tuple):
                q, a = item
            elif isinstance(item, str) and "=" in item:
                q, a = item.split("=", 1)
            else:
                raise ValueError(f"Formato no válido: {item}. Use 'expr=answer' o (expr, answer)")
            
            inp, tgt = self._tokenize_pair(q, a)
            inputs_list.append(inp)
            targets_list.append(tgt)
        
        return torch.tensor(inputs_list, dtype=torch.long), torch.tensor(targets_list, dtype=torch.long)
    
    def generate(
        self,
        n: int = 5000,
        ops: List[str] = ["+", "-", "*", "/"],
        max_val: int = 99,
        seed: int = 42,
    ) -> Tuple[Tensor, Tensor]:
        """Genera un dataset de aritmética automáticamente.
        
        Args:
            n: Número de ejemplos a generar.
            ops: Lista de operaciones a incluir.
            max_val: Valor máximo para operandos.
            seed: Seed para reproducibilidad.
            
        Returns:
            (inputs, targets) — tensores listos para entrenamiento.
        """
        random.seed(seed)
        data: List[Tuple[str, str]] = []
        seen: Set[str] = set()
        
        for _ in range(n * 10):
            if len(data) >= n:
                break
            
            op = random.choice(ops)
            
            if op == "+":
                a, b = random.randint(1, max_val), random.randint(1, max_val)
                q, ans = f"{a}+{b}", str(a + b)
            elif op == "-":
                a = random.randint(2, max_val)
                b = random.randint(1, a)
                q, ans = f"{a}-{b}", str(a - b)
            elif op == "*":
                a, b = random.randint(2, min(12, max_val)), random.randint(2, min(12, max_val))
                q, ans = f"{a}*{b}", str(a * b)
            elif op == "/":
                b = random.randint(2, min(12, max_val))
                r = random.randint(1, min(12, max_val))
                q, ans = f"{r*b}/{b}", str(r)
            else:
                continue
            
            if q not in seen:
                seen.add(q)
                data.append((q, ans))
        
        return self.prepare(data)
    
    def encode(self, sample: str) -> Tensor:
        """Codifica una expresión para inferencia (sin respuesta).
        
        Args:
            sample: Expresión como "5+5" (sin =answer).
            
        Returns:
            Tensor de shape (1, max_seq_len-1) con BOS+expr+SEP+PAD...
        """
        tokens = [self.BOS] + [self.CHAR_TO_ID[c] for c in sample if c in self.CHAR_TO_ID] + [self.SEP]
        pad_n = self._max_seq_len - 1 - len(tokens)
        tokens = tokens + [self.PAD] * pad_n
        return torch.tensor([tokens[:self._max_seq_len - 1]], dtype=torch.long)
    
    def decode(self, tensor: Tensor) -> str:
        """Decodifica un tensor de predicción a string.
        
        Args:
            tensor: Tensor 1D de token IDs.
            
        Returns:
            String decodificado.
        """
        chars = []
        for t in tensor.tolist() if isinstance(tensor, Tensor) else tensor:
            if t == self.EOS:
                break
            if t == self.SEP:
                chars.append("=")
            elif t in self.ID_TO_CHAR:
                chars.append(self.ID_TO_CHAR[t])
        return "".join(chars)
    
    def _tokenize_pair(self, q: str, a: str) -> Tuple[List[int], List[int]]:
        """Tokeniza un par pregunta-respuesta."""
        q_tokens = [self.CHAR_TO_ID[c] for c in q if c in self.CHAR_TO_ID]
        a_tokens = [self.CHAR_TO_ID[c] for c in a if c in self.CHAR_TO_ID]
        
        full_seq = [self.BOS] + q_tokens + [self.SEP] + a_tokens + [self.EOS]
        full_seq = full_seq[:self._max_seq_len]
        
        inp = full_seq[:-1]
        tgt = full_seq[1:]
        
        pad_n = self._max_seq_len - 1 - len(inp)
        inp = inp + [self.PAD] * pad_n
        tgt = tgt + [self.PAD] * pad_n
        
        return inp, tgt
