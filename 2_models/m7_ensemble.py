"""Ensemble por pooling log-odds (média geométrica ponderada das probabilidades).

p_ens(resultado) ∝ Π_i p_i(resultado) ^ w_i  (pesos somando 1, normalizado).

Os pesos são otimizados APENAS na validação (torneios antigos), nunca no teste final
— a otimização vive em 4_validation/selection, não aqui. Este modelo só combina.
"""
from __future__ import annotations

import numpy as np

from base_model import BaseModel

_EPS = 1e-9


class EnsembleModel(BaseModel):
    supports_scoreline = True

    def __init__(self, members: list[BaseModel], weights: list[float] | None = None,
                 name="ensemble"):
        self.name = name
        self.members = members
        n = len(members)
        w = np.ones(n) / n if weights is None else np.asarray(weights, dtype=float)
        self.weights = w / w.sum()

    def fit(self, df):
        for m in self.members:
            m.fit(df)
        return self

    def known_team(self, team):
        return any(m.known_team(team) for m in self.members)

    def predict_proba(self, home, away, neutral=False):
        log_acc = np.zeros(3)
        for m, w in zip(self.members, self.weights):
            p = np.clip(m.predict_proba(home, away, neutral), _EPS, 1.0)
            log_acc += w * np.log(p)
        p = np.exp(log_acc)
        return p / p.sum()

    def predict_scoreline(self, home, away, neutral=False):
        # Média ponderada (aritmética) dos grids dos membros que suportam placar.
        grids, ws = [], []
        for m, w in zip(self.members, self.weights):
            g = m.predict_scoreline(home, away, neutral)
            if g is not None:
                grids.append(g)
                ws.append(w)
        if not grids:
            return None
        ws = np.asarray(ws) / np.sum(ws)
        return np.tensordot(ws, np.stack(grids), axes=(0, 0))

    def set_weights(self, weights):
        w = np.asarray(weights, dtype=float)
        self.weights = w / w.sum()
        return self
