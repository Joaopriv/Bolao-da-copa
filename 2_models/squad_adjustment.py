"""[Iteração 2 / D6.2] Ajuste de força de elenco (squad_offset) sobre os modelos base.

Tilting exponencial do grid de placar: para Poisson independente, multiplicar a matriz
de probabilidades por r_home^h * r_away^a (e renormalizar) é EXATAMENTE equivalente a
escalar os parâmetros de ataque/defesa por r_home/r_away -- para Dixon-Coles/Bivariate
Poisson é uma aproximação razoável dado o peso pequeno (squad_offset_weight ~ 0.3).
Abordagem modelo-agnóstica: não depende de acesso aos parâmetros internos do
penaltyblog (não expostos pós-fit).

  squad_offset_home = atk_z[home] + def_z[away]
  squad_offset_away = atk_z[away] + def_z[home]
  r_home = exp(weight * squad_offset_home)
  r_away = exp(weight * squad_offset_away)

Times ausentes de squad_strength (squad_z_map) ou squad_offset_weight==0 -> r=1.0
(no-op transparente, preserva o comportamento da Iteração 1 sem squad_strength).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import db_client  # noqa: E402
from base_model import BaseModel  # noqa: E402


def tilt_grid(grid: np.ndarray, r_home: float, r_away: float) -> np.ndarray:
    """Reescala um grid de placar (g x g, soma 1) por r_home^h * r_away^a e renormaliza."""
    g = grid.shape[0]
    h = np.arange(g).reshape(-1, 1)
    a = np.arange(g).reshape(1, -1)
    out = grid * (r_home ** h) * (r_away ** a)
    total = out.sum()
    return out / total if total > 0 else grid


def grid_to_1x2(grid: np.ndarray) -> np.ndarray:
    """[pH, pD, pA] a partir de um grid de placar (soma 1)."""
    idx = np.indices(grid.shape)
    pH = grid[idx[0] > idx[1]].sum()
    pD = grid[idx[0] == idx[1]].sum()
    pA = grid[idx[0] < idx[1]].sum()
    return np.array([pH, pD, pA])


@lru_cache(maxsize=1)
def squad_z_map() -> dict[str, dict[str, float]]:
    """{team: {"atk_z":..., "def_z":...}} via z-score de attack_adjusted/defense_adjusted
    entre as seleções de squad_strength. {} se a tabela estiver vazia/sem variação --
    SquadAdjustedModel cai no no-op (r=1.0), preservando o comportamento da Iteração 1."""
    rows = db_client.fetch_all("squad_strength") or []
    valid = [r for r in rows
             if r.get("attack_adjusted") is not None and r.get("defense_adjusted") is not None]
    if len(valid) < 2:
        return {}

    attack = np.array([r["attack_adjusted"] for r in valid])
    defense = np.array([r["defense_adjusted"] for r in valid])
    attack_std, defense_std = attack.std(), defense.std()
    if attack_std == 0 or defense_std == 0:
        return {}

    attack_mean, defense_mean = attack.mean(), defense.mean()
    return {
        r["team"]: {
            "atk_z": (r["attack_adjusted"] - attack_mean) / attack_std,
            "def_z": (r["defense_adjusted"] - defense_mean) / defense_std,
        }
        for r in valid
    }


class SquadAdjustedModel(BaseModel):
    """Wrapper genérico: ajusta o grid/1X2 de `inner` pela diferença de força de elenco
    (squad_z_map) com peso `weight` (cfg.squad_strength.squad_offset_weight).

    Times ausentes de squad_z_map ou weight==0 -> r_home=r_away=1.0 (no-op
    transparente). predict_proba: grid_to_1x2(tilt) se `inner` suporta placar; senão
    repassa inner.predict_proba inalterado (cobre o Elo, que não suporta scoreline)."""

    def __init__(self, inner: BaseModel, weight: float):
        self.inner = inner
        self.weight = weight
        self.name = inner.name
        self.supports_scoreline = inner.supports_scoreline

    def fit(self, df):
        self.inner.fit(df)
        return self

    def known_team(self, team: str) -> bool:
        return self.inner.known_team(team)

    def _ratios(self, home: str, away: str) -> tuple[float, float]:
        if self.weight == 0:
            return 1.0, 1.0
        z = squad_z_map()
        h, a = z.get(home), z.get(away)
        if h is None or a is None:
            return 1.0, 1.0
        offset_home = h["atk_z"] - a["def_z"]
        offset_away = a["atk_z"] - h["def_z"]
        return float(np.exp(self.weight * offset_home)), float(np.exp(self.weight * offset_away))

    def predict_scoreline(self, home, away, neutral=False):
        grid = self.inner.predict_scoreline(home, away, neutral)
        if grid is None:
            return None
        r_home, r_away = self._ratios(home, away)
        if r_home == 1.0 and r_away == 1.0:
            return grid
        return tilt_grid(grid, r_home, r_away)

    def predict_proba(self, home, away, neutral=False):
        grid = self.predict_scoreline(home, away, neutral)
        if grid is None:
            return self.inner.predict_proba(home, away, neutral)
        return grid_to_1x2(grid)
