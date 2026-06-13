"""m3 — Dixon-Coles sobre xG (StatsBomb). Mesma família do m2, mas treinado
no subconjunto de jogos com home_xg/away_xg (arredondados p/ inteiro).

Restrição do penaltyblog 1.11.0: DixonColesGoalModel exige `goals_home`/
`goals_away` inteiros (np.asarray(..., dtype=cython_long_dtype)). xG contínuo
é arredondado para o inteiro mais próximo antes do fit — solução padrão da
literatura. Limitação conhecida: jogos de baixo xG (ex.: 0.3 x 0.4) viram 0x0
no treino, distorcendo o sinal em jogos equilibrados (não é bug).
"""
from __future__ import annotations

import numpy as np
import penaltyblog as pb
from base_model import PenaltyblogGoalModel


class DixonColesXGModel(PenaltyblogGoalModel):
    def __init__(self, max_goals, min_train_matches):
        super().__init__(pb.models.DixonColesGoalModel, name="dixon_coles_xg",
                          max_goals=max_goals)
        self._min_train = min_train_matches

    def fit(self, df):
        self._grid_cache = {}
        sub = df[df["home_xg"].notna() & df["away_xg"].notna()]
        if len(sub) < self._min_train:
            # Dados de xG insuficientes nesta janela de treino: comporta-se
            # como "nenhum time conhecido" -> predict_proba/predict_scoreline
            # caem no fallback 1X2 / None (mesmo padrão do EloModel p/ scoreline).
            self._model = None
            self._teams = set()
            return self

        def _arr(col, dtype):
            return np.ascontiguousarray(sub[col].to_numpy(dtype=dtype, copy=True))

        weights = _arr("weight", float) if "weight" in sub else None
        neutral = _arr("neutral", int) if "neutral" in sub else None
        self._model = self._cls(
            goals_home=np.ascontiguousarray(np.round(sub["home_xg"].to_numpy(float)).astype(int)),
            goals_away=np.ascontiguousarray(np.round(sub["away_xg"].to_numpy(float)).astype(int)),
            teams_home=_arr("home_team", str), teams_away=_arr("away_team", str),
            weights=weights, neutral_venue=neutral,
        )
        self._model.fit()
        self._teams = set(sub["home_team"]) | set(sub["away_team"])
        return self


def build(cfg, mode=None) -> DixonColesXGModel:
    mcfg = cfg["models"].get("dixon_coles_xg", {})
    return DixonColesXGModel(max_goals=cfg["models"]["max_goals"],
                              min_train_matches=mcfg.get("min_train_matches", 30))
