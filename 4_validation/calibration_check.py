"""[Auditoria M4/P8] Diagnóstico de calibração (reliability / ECE).

RPS/Brier/LogLoss/Accuracy medem se o modelo ACERTA o resultado, mas não dizem se as
PROBABILIDADES em si são confiáveis -- ex.: um modelo que diz "70% de chance" mas
acerta apenas 50% das vezes que diz isso está mal calibrado, mesmo com boa acurácia.
Isso importa diretamente para o bolão: o EV de cada palpite (utils/scoring.py) depende
das probabilidades serem realistas, não só do ranking de palpites estar certo.

ECE (Expected Calibration Error) top-label, como em Guo et al. 2017:
  confidence_i = max(P_i)            (probabilidade do resultado MAIS provável)
  correct_i    = 1 se argmax(P_i) == y_i, senão 0
  bins de confidence em [0,1] (largura igual); ECE = soma_b (n_b/N) * |conf_b - acc_b|

ECE baixo (~0) = bem calibrado (quando o modelo diz "70%", acerta ~70% das vezes).
ECE alto = overconfident (conf > acc) ou underconfident (conf < acc).

Apenas DIAGNÓSTICO -- não é gate de IC95% como h2h_check/squad_offset_check: ECE não
tem teste de significância padronizado aqui, e não decide o `chosen_model` (decisão
do funil em significance.py permanece baseada em RPS).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_config  # noqa: E402
from tournament_comparison import run_comparison  # noqa: E402


def compute_ece(P: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """ECE top-label + bins do diagrama de confiabilidade (ver docstring do módulo)."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    n = len(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append({"range": f"[{lo:.1f},{hi:.1f}]", "n": 0,
                          "confidence": None, "accuracy": None})
            continue
        avg_conf = float(conf[mask].mean())
        acc = float(correct[mask].mean())
        ece += (cnt / n) * abs(avg_conf - acc)
        bins.append({"range": f"[{lo:.1f},{hi:.1f}]", "n": cnt,
                      "confidence": avg_conf, "accuracy": acc})
    return {"ece": ece, "n": n, "bins": bins}


def run(model_names: list[str] | None = None, tournament_names: list[str] | None = None,
        n_bins: int = 10, iters: int | None = None, verbose: bool = True) -> dict:
    """Calcula o ECE de cada modelo sobre o pool de jogos de teste (todos os torneios
    configurados por padrão -- mais jogos = bins mais estáveis)."""
    cfg = load_config()
    iters = iters or cfg["validation"]["bootstrap_iterations"]
    comp = run_comparison(cfg=cfg, model_names=model_names, tournament_names=tournament_names,
                           iters=iters, verbose=verbose)

    results = {m: compute_ece(comp["pooled_P"][m], comp["pooled_y"], n_bins=n_bins)
               for m in comp["models"]}

    if verbose:
        print(f"\n  ECE (Expected Calibration Error, top-label, {n_bins} bins, "
              f"{comp['pooled_y'].shape[0]} jogos) -- menor é melhor (0 = perfeitamente calibrado):")
        for m, r in sorted(results.items(), key=lambda kv: kv[1]["ece"]):
            print(f"    {m:18s} ECE={r['ece']:.4f}")
        print()
        chosen = max(results.items(), key=lambda kv: kv[1]["n"])[0]  # qualquer um serve (mesmo n)
        print(f"  diagrama de confiabilidade -- {chosen} (confidence vs. acurácia observada por bin):")
        print(f"    {'bin':>12s} {'n':>5s} {'confidence':>10s} {'acurácia':>10s} {'|diff|':>8s}")
        for b in results[chosen]["bins"]:
            if b["n"] == 0:
                print(f"    {b['range']:>12s} {0:>5d} {'—':>10s} {'—':>10s} {'—':>8s}")
            else:
                diff = abs(b["confidence"] - b["accuracy"])
                print(f"    {b['range']:>12s} {b['n']:>5d} {b['confidence']:>10.3f} "
                      f"{b['accuracy']:>10.3f} {diff:>8.3f}")

    return {"results": results, "tournaments": comp["tournaments"], "n_bins": n_bins}


if __name__ == "__main__":
    run()
