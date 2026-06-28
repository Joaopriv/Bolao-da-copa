"""[Mercados de aposta] Derivação de probabilidades de mercado a partir da grade
conjunta de placar P(gols_casa, gols_fora) produzida pelos modelos de gols.

PRINCÍPIO: não há um "modelo por mercado". Há UM modelo de gols (Poisson/Dixon-Coles/
bivariado) que produz a distribuição conjunta sobre placares. Todo mercado é um EVENTO
mensurável (uma soma de células) sobre essa mesma distribuição:

  1X2:          P(casa) = soma_{h>a} G[h,a] ; empate = soma_{h=a} ; fora = soma_{h<a}
  total O/U L:  P(over)  = soma_{h+a > L} G[h,a]      (L = linha, ex. 2.5)
  time O/U L:   P(over_casa) = soma_{h > L} G[h,a]
  BTTS:         P(sim)   = soma_{h>=1 e a>=1} G[h,a]
  placar exato: G[h,a]

Como todos derivam da MESMA grade, validar over/under é validar se a MARGINAL de total
de gols do modelo está calibrada -- teste diferente do 1X2 (dá pra acertar o vencedor e
errar o volume de gols). Por isso cada mercado precisa de validação própria
(4_validation/markets_check.py), não herda a credibilidade do 1X2.

Odds NUNCA entram aqui -- este módulo só produz probabilidades do modelo. O cruzamento
com odds de mercado (valor/EV) é feito separadamente.
"""
from __future__ import annotations

import numpy as np

# Linhas de aposta padrão (meias-linhas evitam empate/push na liquidação).
TOTAL_LINES = (1.5, 2.5, 3.5)
TEAM_LINES = (0.5, 1.5, 2.5)


def _normalize(grid: np.ndarray) -> np.ndarray:
    """Grade P(h,a) renormalizada para somar 1 (o truncamento em max_goals corta uma
    cauda minúscula; renormalizar mantém as probabilidades de mercado consistentes)."""
    g = np.asarray(grid, dtype=float)
    s = g.sum()
    return g / s if s > 0 else g


def result_1x2(grid: np.ndarray) -> dict[str, float]:
    """P(vitória casa / empate / vitória fora)."""
    g = _normalize(grid)
    n = g.shape[0]
    home = sum(g[h, a] for h in range(n) for a in range(n) if h > a)
    draw = float(np.trace(g))
    away = sum(g[h, a] for h in range(n) for a in range(n) if h < a)
    return {"home": float(home), "draw": draw, "away": float(away)}


def total_over_under(grid: np.ndarray, lines=TOTAL_LINES) -> dict[float, dict[str, float]]:
    """Para cada linha L: P(total de gols > L) e P(< L). L é meia-linha (sem push)."""
    g = _normalize(grid)
    n = g.shape[0]
    # total[k] = P(gols_casa + gols_fora == k)
    totals = np.zeros(2 * n - 1)
    for h in range(n):
        for a in range(n):
            totals[h + a] += g[h, a]
    out = {}
    for L in lines:
        over = float(totals[int(np.ceil(L)):].sum())
        out[L] = {"over": over, "under": 1.0 - over}
    return out


def team_over_under(grid: np.ndarray, side: str, lines=TEAM_LINES) -> dict[float, dict[str, float]]:
    """P(gols do `side` ('home'/'away') > L) e P(< L), por linha L."""
    g = _normalize(grid)
    marg = g.sum(axis=1) if side == "home" else g.sum(axis=0)  # marginal de gols do time
    out = {}
    for L in lines:
        over = float(marg[int(np.ceil(L)):].sum())
        out[L] = {"over": over, "under": 1.0 - over}
    return out


def btts(grid: np.ndarray) -> dict[str, float]:
    """Ambas as equipes marcam: P(casa>=1 E fora>=1)."""
    g = _normalize(grid)
    yes = float(g[1:, 1:].sum())
    return {"yes": yes, "no": 1.0 - yes}


def exact_score(grid: np.ndarray, h: int, a: int) -> float:
    """P(placar exato == h-a)."""
    g = _normalize(grid)
    if h < g.shape[0] and a < g.shape[1]:
        return float(g[h, a])
    return 0.0


def all_markets(grid: np.ndarray, total_lines=TOTAL_LINES, team_lines=TEAM_LINES) -> dict:
    """Todas as probabilidades de mercado de um confronto, a partir da grade."""
    return {
        "result": result_1x2(grid),
        "total": total_over_under(grid, total_lines),
        "home_goals": team_over_under(grid, "home", team_lines),
        "away_goals": team_over_under(grid, "away", team_lines),
        "btts": btts(grid),
    }
