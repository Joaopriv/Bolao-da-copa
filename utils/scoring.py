"""Pontuação do bolão e valor esperado (EV) de palpites de placar.

Regra de pontos (bolão clássico):
  25 = placar exato
  18 = vencedor + gols do vencedor corretos (mas não placar exato)
  15 = vencedor + diferença de gols correta (mas não os gols exatos)
  12 = vencedor + gols do perdedor corretos (mas não os outros critérios)
  10 = vencedor correto (sem nenhum detalhe) OU empate correto sem placar exato
   0 = vencedor/resultado errado
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


def score(palpite_h: int, palpite_a: int, real_h: int, real_a: int) -> int:
    """Pontos do bolão para um palpite dado o resultado real."""
    if palpite_h == real_h and palpite_a == real_a:
        return 25  # placar exato

    palpite_vencedor = 1 if palpite_h > palpite_a else (-1 if palpite_h < palpite_a else 0)
    real_vencedor = 1 if real_h > real_a else (-1 if real_h < real_a else 0)

    if palpite_vencedor != real_vencedor:
        return 0  # resultado errado — sem pontos parciais

    if palpite_vencedor == 0:  # empate correto, placar exato já tratado acima
        return 10

    if palpite_vencedor == 1:  # vitória do mandante
        gols_vencedor_ok = palpite_h == real_h
        diff_ok = (palpite_h - palpite_a) == (real_h - real_a)
        gols_perdedor_ok = palpite_a == real_a
    else:  # vitória do visitante
        gols_vencedor_ok = palpite_a == real_a
        diff_ok = (palpite_a - palpite_h) == (real_a - real_h)
        gols_perdedor_ok = palpite_h == real_h

    if gols_vencedor_ok:
        return 18
    if diff_ok:
        return 15
    if gols_perdedor_ok:
        return 12
    return 10  # só resultado correto


def expected_value(palpite_h: int, palpite_a: int, score_matrix: pd.DataFrame) -> float:
    """EV em pontos esperados de um palpite, dada a matriz P(real_h, real_a)."""
    ev = 0.0
    for real_h in score_matrix.index:
        for real_a in score_matrix.columns:
            prob = float(score_matrix.loc[real_h, real_a])
            ev += prob * score(palpite_h, palpite_a, real_h, real_a)
    return round(ev, 2)


def truncate_and_renormalize(grid: np.ndarray, max_goals: int) -> np.ndarray:
    """Trunca `grid` (P(home_goals, away_goals), shape (M+1, M+1) com M >= max_goals)
    para placares 0..max_goals e renormaliza para somar 1.0.

    [Auditoria M8/P11] Usado tanto por `top_scores_by_ev` (EV de palpites) quanto por
    `predict_2026.generate` (heatmap `score_matrix`) -- garante que o campo `prob` de
    `top_scores` e as células de `score_matrix` se refiram à MESMA distribuição
    renormalizada (antes, só o heatmap renormalizava; `top_scores` usava a grade
    truncada crua, divergindo em pontos de arredondamento).
    """
    g = max_goals + 1
    truncated = np.asarray(grid, dtype=float)[:g, :g]
    total = truncated.sum()
    if total > 0:
        truncated = truncated / total
    return truncated


def top_scores_by_ev(grid: np.ndarray, max_goals: int, n: int) -> list[dict]:
    """Trunca e renormaliza `grid` (P(home_goals, away_goals), shape (M+1, M+1) com
    M >= max_goals) para placares 0..max_goals (ver `truncate_and_renormalize`),
    calcula EV de cada palpite 0..max_goals x 0..max_goals e retorna os `n` palpites
    com maior EV: [{"score": "h-a", "prob": p, "ev": ev}, ...].
    """
    g = max_goals + 1
    truncated = truncate_and_renormalize(grid, max_goals)
    score_matrix = pd.DataFrame(truncated, index=range(g), columns=range(g))

    candidates = []
    for h, a in product(range(g), range(g)):
        candidates.append({
            "score": f"{h}-{a}",
            "prob": round(float(truncated[h, a]), 3),
            "ev": expected_value(h, a, score_matrix),
        })
    candidates.sort(key=lambda c: c["ev"], reverse=True)
    return candidates[:n]
