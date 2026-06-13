"""Interface comum dos modelos + wrappers sobre penaltyblog.

Padroniza .fit() / .predict_proba() / .predict_scoreline() entre TODOS os modelos,
garantindo comparação justa no backtest (mesmas entradas, mesmo formato de saída).

API real do penaltyblog 1.11.0 (confirmada na instalação):
  Model(goals_home, goals_away, teams_home, teams_away, weights, neutral_venue).fit()
  .predict(home, away, max_goals, neutral_venue) -> FootballProbabilityGrid
     .home_draw_away -> [pH, pD, pA]
     .grid           -> matriz (max_goals+1 x max_goals+1) de probabilidade de placar
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import penaltyblog as pb

# Prior neutro 1X2 quando um time é desconhecido pelo modelo (sem histórico no treino).
# Leve viés de mando reflete a base histórica de seleções (~empate técnico em campo neutro).
_FALLBACK_1X2 = np.array([0.40, 0.27, 0.33])


class BaseModel(ABC):
    """Contrato que todo modelo deve cumprir."""

    name: str = "base"
    supports_scoreline: bool = True

    @abstractmethod
    def fit(self, df) -> "BaseModel":
        """Treina com um DataFrame contendo home_team, away_team, home_score,
        away_score, weight, neutral."""

    @abstractmethod
    def predict_proba(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        """Retorna np.array([pH, pD, pA]) somando 1."""

    def predict_scoreline(self, home: str, away: str, neutral: bool = False):
        """Matriz de probabilidade de placar (ndarray) ou None se não suportado."""
        return None

    def top_scores(self, home, away, n=5, neutral=False) -> list[tuple[str, float]]:
        """Top-N placares mais prováveis [(="2-1", prob), ...]. Vazio se não suportado."""
        grid = self.predict_scoreline(home, away, neutral)
        if grid is None:
            return []
        flat = np.argsort(grid, axis=None)[::-1][:n]
        out = []
        for idx in flat:
            h, a = np.unravel_index(idx, grid.shape)
            out.append((f"{h}-{a}", float(grid[h, a])))
        return out

    @abstractmethod
    def known_team(self, team: str) -> bool:
        """True se o modelo viu o time no treino."""


class PenaltyblogGoalModel(BaseModel):
    """Wrapper genérico para os modelos de gols do penaltyblog (Poisson, Dixon-Coles,
    Bivariate Poisson, Bayesian Hierarchical). A classe concreta é injetada."""

    supports_scoreline = True

    def __init__(self, pb_class, name: str, max_goals: int = 10, fit_kwargs: dict | None = None,
                 max_train_matches: int | None = None):
        self._cls = pb_class
        self.name = name
        self.max_goals = max_goals
        self._fit_kwargs = fit_kwargs or {}
        # Limite opcional de jogos de treino (mantém o MCMC do bayesiano tratável).
        # Quando ativo, mantém os jogos de MAIOR peso (mais relevantes).
        self._max_train = max_train_matches
        self._model = None
        self._teams: set[str] = set()

    def fit(self, df):
        self._grid_cache: dict[tuple[str, str, bool], object] = {}
        if self._max_train is not None and len(df) > self._max_train and "weight" in df:
            df = df.nlargest(self._max_train, "weight")
        # penaltyblog usa memoryviews Cython e exige buffers GRAVÁVEIS e contíguos —
        # `.to_numpy()` do pandas pode devolver buffer read-only. Daí as cópias explícitas.
        def _arr(col, dtype):
            return np.ascontiguousarray(df[col].to_numpy(dtype=dtype, copy=True))

        weights = _arr("weight", float) if "weight" in df else None
        neutral = _arr("neutral", int) if "neutral" in df else None
        self._model = self._cls(
            goals_home=_arr("home_score", int),
            goals_away=_arr("away_score", int),
            teams_home=_arr("home_team", str),
            teams_away=_arr("away_team", str),
            weights=weights,
            neutral_venue=neutral,
        )
        self._model.fit(**self._fit_kwargs)
        self._teams = set(df["home_team"]) | set(df["away_team"])
        return self

    def known_team(self, team: str) -> bool:
        return team in self._teams

    def _grid(self, home, away, neutral):
        # Cacheia por (home, away, neutral): predict_proba() e predict_scoreline() pedem
        # o mesmo grid para o mesmo confronto — evita refazer o predict() (caro nos
        # modelos MCMC, que fazem média sobre o trace a cada chamada).
        key = (home, away, bool(neutral))
        if key not in self._grid_cache:
            self._grid_cache[key] = self._model.predict(
                home, away, max_goals=self.max_goals, neutral_venue=bool(neutral)
            )
        return self._grid_cache[key]

    def predict_proba(self, home, away, neutral=False):
        if not (self.known_team(home) and self.known_team(away)):
            return _FALLBACK_1X2.copy()
        try:
            hda = np.asarray(self._grid(home, away, neutral).home_draw_away, dtype=float)
        except (KeyError, ValueError):
            return _FALLBACK_1X2.copy()
        return hda / hda.sum()

    def predict_scoreline(self, home, away, neutral=False):
        if not (self.known_team(home) and self.known_team(away)):
            return None
        try:
            return np.asarray(self._grid(home, away, neutral).grid, dtype=float)
        except (KeyError, ValueError):
            return None


class EloModel(BaseModel):
    """Elo dinâmico (penaltyblog.ratings.Elo). Produz 1X2, NÃO produz placar exato.

    Elo é sequencial: ignora os pesos de competição (recência é intrínseca). Isso é
    declarado abertamente — no backtest ele compete em 1X2/RPS/Brier/LogLoss, mas fica
    de fora da métrica de placar exato (supports_scoreline=False)."""

    supports_scoreline = False

    def __init__(self, name="elo", k=24.0, home_field_advantage=65.0,
                 draw_base=0.30, draw_width=200.0):
        self.name = name
        self._k = k
        self._hfa = home_field_advantage
        self._draw_base = draw_base
        self._draw_width = draw_width
        self._elo = None
        self._teams: set[str] = set()

    def fit(self, df):
        self._elo = pb.ratings.Elo(k=self._k, home_field_advantage=self._hfa)
        # Atualização cronológica. Encoding do penaltyblog: 0 = vitória mandante,
        # 1 = empate, 2 = vitória visitante.
        for _, r in df.sort_values("date").iterrows():
            if r["home_score"] > r["away_score"]:
                res = 0
            elif r["home_score"] < r["away_score"]:
                res = 2
            else:
                res = 1
            self._elo.update_ratings(str(r["home_team"]), str(r["away_team"]), res)
        self._teams = set(df["home_team"]) | set(df["away_team"])
        return self

    def known_team(self, team: str) -> bool:
        return team in self._teams

    def predict_proba(self, home, away, neutral=False):
        if not (self.known_team(home) and self.known_team(away)):
            return _FALLBACK_1X2.copy()
        # [Auditoria P3] pb.ratings.Elo não aceita `neutral_venue` por chamada -- o HFA
        # fica fixo no objeto (self._elo.hfa). Em jogo neutro zeramos hfa
        # temporariamente para a duração da chamada e restauramos depois, igualando o
        # tratamento dado aos demais modelos via `neutral_venue=True`.
        hfa = self._elo.hfa
        if neutral:
            self._elo.hfa = 0.0
        try:
            d = self._elo.calculate_match_probabilities(
                str(home), str(away), draw_base=self._draw_base, draw_width=self._draw_width
            )
        finally:
            self._elo.hfa = hfa
        p = np.array([d["home_win"], d["draw"], d["away_win"]], dtype=float)
        return p / p.sum()
