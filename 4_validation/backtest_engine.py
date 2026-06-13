"""Motor de backtest: roda UM modelo em UM torneio, sem vazamento.

Treina com jogos anteriores ao torneio e prevê cada partida. Devolve as previsões
cruas (matriz P de 1X2, resultados y, placares previstos/reais) para o metrics.py
calcular as métricas com IC.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_split import split_for_tournament  # noqa: E402

_RES_IDX = {"H": 0, "D": 1, "A": 2}


def _parse_score(score_str: str):
    h, a = score_str.split("-")
    return int(h), int(a)


def run_model_on_tournament(model, spec: dict, lam: float, train_df=None) -> dict:
    """Treina `model` (instância não treinada) e prevê o torneio `spec`.

    Se `train_df` for passado, reusa-o (evita recomputar o treino quando vários
    modelos compartilham o mesmo split). Retorna dict com P, y, pred_scores, true_scores.
    """
    if train_df is None:
        train_df, test = split_for_tournament(spec, lam)
    else:
        import dataset
        test = dataset.get_test_tournament(spec)

    model.fit(train_df)

    P, y, pred_scores, true_scores = [], [], [], []
    for _, r in test.iterrows():
        home, away = str(r["home_team"]), str(r["away_team"])
        neutral = bool(r["neutral"])
        P.append(model.predict_proba(home, away, neutral))
        y.append(_RES_IDX[r["result"]])
        true_scores.append((int(r["home_score"]), int(r["away_score"])))
        if getattr(model, "supports_scoreline", False):
            ts = model.top_scores(home, away, n=1, neutral=neutral)
            pred_scores.append(_parse_score(ts[0][0]) if ts else None)
        else:
            pred_scores.append(None)

    return {
        "P": np.asarray(P, dtype=float),
        "y": np.asarray(y, dtype=int),
        "pred_scores": pred_scores,
        "true_scores": true_scores,
        "n_matches": len(test),
    }
