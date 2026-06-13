"""Métricas de previsão com BOOTSTRAP de intervalo de confiança.

Princípio #1 do projeto: TODA métrica retorna ponto + IC95% (não só o ponto).
Categorias 1X2 ordenadas como [H, D, A] (índices 0,1,2) — a ordem importa para o RPS.

Métricas "menor é melhor": RPS, Brier, LogLoss.
Métricas "maior é melhor": Acurácia 1X2, Top-1 placar exato.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import scoring  # noqa: E402

H, D, A = 0, 1, 2
_EPS = 1e-12


@dataclass
class Score:
    """Resultado de uma métrica: ponto + IC + n de amostras."""
    name: str
    point: float
    lo: float
    hi: float
    n: int
    lower_is_better: bool

    def __str__(self):
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


# ── Vetores de perda por jogo (cada elemento = perda daquele jogo) ───────────

def rps_vector(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked Probability Score por jogo (categorias ordenadas H,D,A)."""
    P = np.asarray(P, dtype=float)
    O = np.eye(3)[y]
    cum_p = np.cumsum(P, axis=1)
    cum_o = np.cumsum(O, axis=1)
    # RPS = (1/(r-1)) * sum_{i=1}^{r-1} (cumP_i - cumO_i)^2  ; r=3 -> divide por 2
    return ((cum_p[:, :2] - cum_o[:, :2]) ** 2).sum(axis=1) / 2.0


def brier_vector(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Brier multiclasse por jogo: sum_j (p_j - o_j)^2."""
    P = np.asarray(P, dtype=float)
    O = np.eye(3)[y]
    return ((P - O) ** 2).sum(axis=1)


def logloss_vector(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Log-loss por jogo: -log(p_resultado)."""
    P = np.clip(np.asarray(P, dtype=float), _EPS, 1.0)
    return -np.log(P[np.arange(len(y)), y])


def accuracy_vector(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Acerto 1X2 por jogo (1 se argmax bate o resultado)."""
    return (np.asarray(P).argmax(axis=1) == y).astype(float)


def exact_score_vector(pred_scores: list, true_scores: list) -> np.ndarray:
    """Top-1 placar exato por jogo (1 se o placar mais provável == placar real).

    pred_scores: lista de tuplas (h,a) previstas (ou None se modelo não dá placar).
    Jogos com previsão None são marcados NaN e excluídos da agregação.
    """
    out = np.full(len(true_scores), np.nan)
    for i, (pred, true) in enumerate(zip(pred_scores, true_scores)):
        if pred is None:
            continue
        out[i] = 1.0 if tuple(pred) == tuple(true) else 0.0
    return out


def bolao_points_vector(pred_scores: list, true_scores: list) -> np.ndarray:
    """[Auditoria M2/P6] Pontos do bolão (utils.scoring.score) por jogo, usando o
    placar Top-1 previsto (mesmo `pred_scores` do ExactTop1).

    pred_scores: lista de tuplas (h,a) previstas (ou None se modelo não dá placar).
    Jogos com previsão None são marcados NaN e excluídos da agregação. Métrica
    "maior é melhor" -- traduz RPS/Brier/LogLoss para a métrica que importa de fato
    para o usuário do bolão.
    """
    out = np.full(len(true_scores), np.nan)
    for i, (pred, true) in enumerate(zip(pred_scores, true_scores)):
        if pred is None:
            continue
        out[i] = scoring.score(pred[0], pred[1], true[0], true[1])
    return out


# ── Bootstrap ────────────────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray, *, name: str, lower_is_better: bool,
                 iters: int = 1000, percentiles=(2.5, 97.5), seed: int = 42) -> Score:
    """IC do MÉDIA via bootstrap (reamostragem com reposição). Ignora NaN."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return Score(name, float("nan"), float("nan"), float("nan"), 0, lower_is_better)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, percentiles)
    return Score(name, float(v.mean()), float(lo), float(hi), n, lower_is_better)


# Registro de métricas 1X2 (nome -> (função_vetor, menor_é_melhor)).
METRICS_1X2 = {
    "RPS": (rps_vector, True),
    "Brier": (brier_vector, True),
    "LogLoss": (logloss_vector, True),
    "Accuracy": (accuracy_vector, False),
}


def evaluate_all(P, y, *, pred_scores=None, true_scores=None,
                 iters=1000, percentiles=(2.5, 97.5), seed=42) -> dict[str, Score]:
    """Calcula todas as métricas 1X2 (com IC) e, se houver placares, o Top-1 exato."""
    res = {}
    for name, (fn, lower) in METRICS_1X2.items():
        res[name] = bootstrap_ci(fn(P, y), name=name, lower_is_better=lower,
                                 iters=iters, percentiles=percentiles, seed=seed)
    if pred_scores is not None and true_scores is not None:
        res["ExactTop1"] = bootstrap_ci(
            exact_score_vector(pred_scores, true_scores),
            name="ExactTop1", lower_is_better=False,
            iters=iters, percentiles=percentiles, seed=seed,
        )
        res["BolaoPoints"] = bootstrap_ci(
            bolao_points_vector(pred_scores, true_scores),
            name="BolaoPoints", lower_is_better=False,
            iters=iters, percentiles=percentiles, seed=seed,
        )
    return res
