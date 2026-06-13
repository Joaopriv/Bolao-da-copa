"""Splits temporais cronológicos (NUNCA aleatórios) — evita vazamento de futuro.

Para cada torneio de teste, o treino contém SÓ jogos anteriores ao seu início.
Delega o trabalho pesado ao dataset.py (já calcula pesos relativos à data de referência).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset  # noqa: E402


def split_for_tournament(spec: dict, lam: float):
    """Retorna (train_df, test_df) para um torneio de teste.

    - train_df: jogos JOGADOS antes de spec['start'], com pesos (w_t × w_comp).
    - test_df : jogos do torneio (mesmo label + janela de datas).
    O treino para no início do torneio → zero vazamento.
    """
    train = dataset.training_frame(spec["start"], lam=lam)
    test = dataset.get_test_tournament(spec)
    return train, test
