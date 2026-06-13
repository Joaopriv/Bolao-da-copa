"""Baselines obrigatórios (pisos de sanidade). Todo modelo PRECISA bater os dois;
se não bater, está quebrado.

- NaiveBaseline   : prevê sempre as taxas históricas base de H/D/A (ponderadas).
- RankingBaseline : força = média ponderada do saldo de gols por jogo; mapeia a diferença
                    de força (com vantagem de mando) para 1X2 via logística de coeficientes
                    FIXOS (não ajustados). Substitui o baseline de FIFA-rank (não temos a
                    tabela FIFA; este piso de ranking cumpre o mesmo papel).

Ambos seguem o mesmo "contrato" duck-typed dos modelos: fit / predict_proba /
known_team / supports_scoreline / top_scores.
"""
from __future__ import annotations

import numpy as np

# Coeficientes FIXOS do ranking baseline (deliberadamente não ajustados — é um piso).
_HOME_EDGE = 0.20          # vantagem de mando em "unidades de força" (saldo/jogo)
_SLOPE = 1.1               # inclinação da logística força->prob
_DRAW_PEAK = 0.30          # prob de empate quando os times são equivalentes
_DRAW_DECAY = 1.2          # quão rápido o empate cai com a diferença de força


class NaiveBaseline:
    name = "baseline_naive"
    supports_scoreline = False

    def __init__(self):
        self._p = np.array([0.45, 0.27, 0.28])
        self._teams: set[str] = set()

    def fit(self, df):
        w = df["weight"].to_numpy() if "weight" in df else np.ones(len(df))
        res = df["result"].to_numpy()
        tot = w.sum()
        self._p = np.array([
            w[res == "H"].sum() / tot,
            w[res == "D"].sum() / tot,
            w[res == "A"].sum() / tot,
        ])
        self._teams = set(df["home_team"]) | set(df["away_team"])
        return self

    def known_team(self, team):
        return team in self._teams

    def predict_proba(self, home, away, neutral=False):
        return self._p.copy()

    def top_scores(self, home, away, n=5, neutral=False):
        return []


class RankingBaseline:
    name = "baseline_ranking"
    supports_scoreline = False

    def __init__(self):
        self._strength: dict[str, float] = {}
        self._default = 0.0

    def fit(self, df):
        # Força = média ponderada de (gols marcados - sofridos) por jogo, por time.
        num: dict[str, float] = {}
        den: dict[str, float] = {}
        w = df["weight"].to_numpy() if "weight" in df else np.ones(len(df))
        for wi, h, a, hs, as_ in zip(w, df["home_team"], df["away_team"],
                                     df["home_score"], df["away_score"]):
            num[h] = num.get(h, 0.0) + wi * (hs - as_)
            den[h] = den.get(h, 0.0) + wi
            num[a] = num.get(a, 0.0) + wi * (as_ - hs)
            den[a] = den.get(a, 0.0) + wi
        self._strength = {t: num[t] / den[t] for t in num}
        self._default = float(np.mean(list(self._strength.values()))) if self._strength else 0.0
        return self

    def known_team(self, team):
        return team in self._strength

    def predict_proba(self, home, away, neutral=False):
        sh = self._strength.get(home, self._default)
        sa = self._strength.get(away, self._default)
        d = (sh - sa) + (0.0 if neutral else _HOME_EDGE)
        # Empate: pico quando d~0, decai com |d|.
        p_draw = _DRAW_PEAK * np.exp(-_DRAW_DECAY * abs(d))
        # Reparte o restante entre H/A por logística da diferença de força.
        p_home_rel = 1.0 / (1.0 + np.exp(-_SLOPE * d))
        rest = 1.0 - p_draw
        p = np.array([rest * p_home_rel, p_draw, rest * (1.0 - p_home_rel)])
        return p / p.sum()

    def top_scores(self, home, away, n=5, neutral=False):
        return []


def build_baselines():
    """Lista de baselines a incluir em toda comparação."""
    return [NaiveBaseline(), RankingBaseline()]
