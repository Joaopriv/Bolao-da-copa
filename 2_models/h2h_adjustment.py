"""[Iteração 3 / F7] Ajuste de confronto direto (H2H) sobre os modelos base.

Mesmo mecanismo de tilting do squad_adjustment.py (D6.2): reescala o grid de placar
por r_home^h * r_away^a e renormaliza.

  h2h_factor(home, away, ref_date) = (vitórias_home - vitórias_away) / n * 0.1
      (n = nº de confrontos diretos considerados ANTES de `ref_date`, range [-0.1, +0.1])
  r_home = exp(weight * h2h_factor)
  r_away = exp(-weight * h2h_factor)

Amostra insuficiente (n < h2h.min_matches) ou weight==0 -> h2h_factor=0.0 -> r=1.0
(no-op transparente, preserva o comportamento sem H2H).

[Auditoria P1] `ref_date` é OBRIGATÓRIO para não vazar: sem ele, `_h2h_pairs()`
incluiria confrontos posteriores (ou o próprio) ao jogo sendo previsto no
backtest. `H2HAdjustedModel.fit(df)` fixa `ref_date = max(df["date"])` (o jogo
mais recente do treino, sempre < início do torneio de teste -- ver
dataset.training_frame), e `h2h_factor`/`h2h_count` descartam confrontos
com `date >= ref_date` antes de aplicar `_MAX_H2H`."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset  # noqa: E402
from base_model import BaseModel  # noqa: E402
from squad_adjustment import tilt_grid, grid_to_1x2  # noqa: E402

_MAX_H2H = 10  # nº máximo de confrontos diretos considerados (mais recentes por data)


@lru_cache(maxsize=1)
def _h2h_pairs() -> dict[frozenset, list[tuple]]:
    """{frozenset({home, away}): [(date, home_team, away_team, home_score, away_score), ...]}
    -- todos os confrontos diretos JOGADOS (qualquer ordem casa/fora), ordenados por
    data (mais recente primeiro). O filtro por `ref_date` é aplicado em
    `h2h_count`/`h2h_factor`, não aqui (cache único, independente da data de
    referência)."""
    df = dataset.load_matches()
    played = df[df["played"]].sort_values("date", ascending=False)
    out: dict[frozenset, list[tuple]] = {}
    for _, r in played.iterrows():
        key = frozenset((r["home_team"], r["away_team"]))
        out.setdefault(key, []).append(
            (r["date"], r["home_team"], r["away_team"], r["home_score"], r["away_score"])
        )
    return out


def _matches_before(home: str, away: str, ref_date) -> list[tuple]:
    """Confrontos diretos com `date < ref_date` (sem vazamento), mais recentes
    primeiro. `ref_date=None` -> sem filtro (uso explícito fora de backtest,
    ex. diagnóstico de cobertura com `cfg.data.today`)."""
    matches = _h2h_pairs().get(frozenset((home, away)), [])
    if ref_date is None:
        return matches
    ref = pd.Timestamp(ref_date)
    return [m for m in matches if m[0] < ref]


def h2h_count(home: str, away: str, ref_date=None) -> int:
    """Nº de confrontos diretos (qualquer ordem casa/fora, `date < ref_date`)
    considerados -- até `_MAX_H2H`, os mais recentes por data."""
    matches = _matches_before(home, away, ref_date)
    return min(len(matches), _MAX_H2H)


def h2h_factor(home: str, away: str, min_matches: int = 3, ref_date=None) -> float:
    """(vitórias_home - vitórias_away) / n * 0.1, sobre os `_MAX_H2H` confrontos
    diretos mais recentes com `date < ref_date`. n < min_matches -> 0.0
    (amostra insuficiente)."""
    matches = _matches_before(home, away, ref_date)[:_MAX_H2H]
    n = len(matches)
    if n < min_matches:
        return 0.0
    home_wins = away_wins = 0
    for _, h_team, a_team, h_score, a_score in matches:
        if h_team == home:
            h_for, a_for = h_score, a_score
        else:
            h_for, a_for = a_score, h_score
        if h_for > a_for:
            home_wins += 1
        elif h_for < a_for:
            away_wins += 1
    return (home_wins - away_wins) / n * 0.1


class H2HAdjustedModel(BaseModel):
    """Wrapper genérico: ajusta o grid/1X2 de `inner` pelo fator de confronto direto
    (h2h_factor) com peso `weight` (cfg.h2h.weight).

    weight==0 ou h2h_factor==0 (amostra insuficiente) -> r_home=r_away=1.0 (no-op
    transparente). predict_proba: grid_to_1x2(tilt) se `inner` suporta placar; senão
    repassa inner.predict_proba inalterado (cobre o Elo, que não suporta scoreline).

    [Auditoria P1] `fit(df)` fixa `self._ref_date = max(df["date"])` -- o jogo mais
    recente do treino. `_ratios()` só considera confrontos H2H ANTERIORES a essa
    data, evitando que o tilt veja resultados do próprio torneio de teste (ou
    futuros) durante o backtest."""

    def __init__(self, inner: BaseModel, weight: float, min_matches: int = 3):
        self.inner = inner
        self.weight = weight
        self.min_matches = min_matches
        self.name = inner.name
        self.supports_scoreline = inner.supports_scoreline
        self._ref_date = None

    def fit(self, df):
        self.inner.fit(df)
        if "date" in df.columns and len(df):
            self._ref_date = pd.Timestamp(df["date"].max())
        return self

    def known_team(self, team: str) -> bool:
        return self.inner.known_team(team)

    def _ratios(self, home: str, away: str) -> tuple[float, float]:
        if self.weight == 0:
            return 1.0, 1.0
        factor = h2h_factor(home, away, self.min_matches, ref_date=self._ref_date)
        if factor == 0.0:
            return 1.0, 1.0
        return float(np.exp(self.weight * factor)), float(np.exp(-self.weight * factor))

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
