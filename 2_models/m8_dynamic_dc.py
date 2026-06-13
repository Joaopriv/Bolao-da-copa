"""m8 — Dixon-Coles dinâmico: ataque/defesa por seleção como random walk online.

Implementação PRÓPRIA (penaltyblog não tem modelo estado-de-espaço). A única peça
reaproveitada do penaltyblog é UMA MLE estática de DixonColesGoalModel sobre o dataset
todo, usada apenas para obter valores iniciais de mu, home_advantage, rho e dos estados
a_i(0)/d_i(0) por seleção (init_from_static).

Parametrização (Koopman-Lit):
    lambda_home = exp(mu + gamma_home * (not neutral) + a_home(t) - d_away(t))
    lambda_away = exp(mu                              + a_away(t) - d_home(t))

Conversão da MLE estática do penaltyblog (lambda_pb = exp(home_adv + attack + defense),
com sum(attack_pb) = n_teams) para esta parametrização:
    mu      = mean(attack_pb) + mean(defense_pb)
    gamma   = home_advantage_pb
    rho     = rho_pb
    a_i(0)  = attack_pb_i - mean(attack_pb)
    d_i(0)  = mean(defense_pb) - defense_pb_i

Atualização online (Via A, filtro estocástico de gradiente — "Elo para gols"): para cada
jogo, em ordem cronológica, erro = gols_observados - lambda_esperado; os estados de
ataque/defesa dos dois times são corrigidos na direção do gradiente da log-verossimilhança
de Poisson (d logL / d eta = y - lambda). Sem ponderar por `weight`: a própria caminhada
aleatória já "esquece" jogos antigos.
"""
from __future__ import annotations

import numpy as np
import penaltyblog as pb
from scipy.stats import poisson

from base_model import BaseModel, _FALLBACK_1X2

_STATE_CLIP = 3.0  # folga sobre os bounds [-2.5, 2.5] usados pelo DixonColesGoalModel.fit


class DynamicDixonColesModel(BaseModel):
    """Dixon-Coles com a_i/d_i (ataque/defesa) evoluindo via random walk online."""

    supports_scoreline = True

    def __init__(self, name="dynamic_dc", max_goals=10, eta_attack=0.05,
                 eta_defense=0.05, rho=None, home_advantage=None,
                 init_from_static=True):
        self.name = name
        self.max_goals = max_goals
        self._eta_a = eta_attack
        self._eta_d = eta_defense
        self._rho_override = rho
        self._gamma_override = home_advantage
        self._init_from_static = init_from_static
        self._teams: set[str] = set()
        self._a: dict[str, float] = {}
        self._d: dict[str, float] = {}
        self._mu = 0.0
        self._gamma = 0.0
        self._rho = 0.0

    def fit(self, df):
        df = df.sort_values("date")
        self._teams = set(df["home_team"]) | set(df["away_team"])
        self._a = {}
        self._d = {}

        # 1) MLE estática (penaltyblog) só para mu/gamma/rho e estados iniciais a_i(0)/d_i(0).
        #    Mesmo padrão de arrays contíguos de PenaltyblogGoalModel.fit (base_model.py).
        def _arr(col, dtype):
            return np.ascontiguousarray(df[col].to_numpy(dtype=dtype, copy=True))

        static = pb.models.DixonColesGoalModel(
            goals_home=_arr("home_score", int),
            goals_away=_arr("away_score", int),
            teams_home=_arr("home_team", str),
            teams_away=_arr("away_team", str),
            weights=_arr("weight", float) if "weight" in df else None,
            neutral_venue=_arr("neutral", int) if "neutral" in df else None,
        )
        static.fit()

        n = static.n_teams
        attack_pb = static._params[:n]
        defense_pb = static._params[n:2 * n]
        gamma_pb = static._params[-2]
        rho_pb = static._params[-1]
        mean_a, mean_d = attack_pb.mean(), defense_pb.mean()

        self._mu = float(mean_a + mean_d)
        self._gamma = float(gamma_pb if self._gamma_override is None else self._gamma_override)
        self._rho = float(rho_pb if self._rho_override is None else self._rho_override)

        if self._init_from_static:
            for i, t in enumerate(static.teams):
                self._a[str(t)] = float(attack_pb[i] - mean_a)
                self._d[str(t)] = float(mean_d - defense_pb[i])

        # 2) Random walk online, em ordem cronológica.
        for _, r in df.iterrows():
            h, a = r["home_team"], r["away_team"]
            ah, dh = self._a.get(h, 0.0), self._d.get(h, 0.0)
            aa, da = self._a.get(a, 0.0), self._d.get(a, 0.0)
            gamma_term = 0.0 if r.get("neutral", 0) else self._gamma

            lam_h = np.exp(self._mu + gamma_term + ah - da)
            lam_a = np.exp(self._mu + aa - dh)
            err_h = r["home_score"] - lam_h
            err_a = r["away_score"] - lam_a

            self._a[h] = float(np.clip(ah + self._eta_a * err_h, -_STATE_CLIP, _STATE_CLIP))
            self._d[a] = float(np.clip(da - self._eta_d * err_h, -_STATE_CLIP, _STATE_CLIP))
            self._a[a] = float(np.clip(aa + self._eta_a * err_a, -_STATE_CLIP, _STATE_CLIP))
            self._d[h] = float(np.clip(dh - self._eta_d * err_a, -_STATE_CLIP, _STATE_CLIP))

        return self

    def known_team(self, team: str) -> bool:
        return team in self._teams

    def _grid(self, home, away, neutral):
        ah, dh = self._a[home], self._d[home]
        aa, da = self._a[away], self._d[away]
        gamma_term = 0.0 if neutral else self._gamma

        lam_h = float(np.exp(self._mu + gamma_term + ah - da))
        lam_a = float(np.exp(self._mu + aa - dh))

        g = self.max_goals
        hg = poisson.pmf(np.arange(g), lam_h)
        ag = poisson.pmf(np.arange(g), lam_a)
        grid = np.outer(hg, ag)

        # Correção tau de Dixon-Coles nos placares baixos.
        rho = self._rho
        grid[0, 0] *= 1 - rho * lam_h * lam_a
        grid[1, 0] *= 1 + rho * lam_h
        grid[0, 1] *= 1 + rho * lam_a
        grid[1, 1] *= 1 - rho

        grid = np.clip(grid, 0.0, None)
        return grid / grid.sum()

    def predict_proba(self, home, away, neutral=False):
        if not (self.known_team(home) and self.known_team(away)):
            return _FALLBACK_1X2.copy()
        grid = self._grid(home, away, neutral)
        i, j = np.indices(grid.shape)
        return np.array([grid[i > j].sum(), grid[i == j].sum(), grid[i < j].sum()])

    def predict_scoreline(self, home, away, neutral=False):
        if not (self.known_team(home) and self.known_team(away)):
            return None
        return self._grid(home, away, neutral)


def build(cfg) -> DynamicDixonColesModel:
    c = cfg["models"]["dynamic_dc"]
    return DynamicDixonColesModel(
        name="dynamic_dc",
        max_goals=cfg["models"]["max_goals"],
        eta_attack=c["eta_attack"],
        eta_defense=c["eta_defense"],
        rho=c.get("rho"),
        home_advantage=c.get("home_advantage"),
        init_from_static=c.get("init_from_static", True),
    )
