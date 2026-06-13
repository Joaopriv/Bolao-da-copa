"""[Iteração 3 / Prompt F7] H2H (confronto direto) — diagnóstico de cobertura + gate
de IC95%.

`h2h.weight` é lido do `cfg` passado a `build_member` em models_registry.py (mesmo
mecanismo do squad_offset_weight, D7/squad_offset_check.py) -- diferente de
`competition_weights` (F4), aqui o toggle via `cfg` em `run_comparison()` funciona
corretamente dentro do mesmo processo, sem subprocesso.

1. Diagnóstico de cobertura (antes do gate): para os 72 jogos de
   `dataset.get_wc2026_fixtures()`, classifica cada par (home, away) por
   `h2h_adjustment.h2h_count()`:
     - disponível   (>= h2h.min_matches confrontos)
     - insuficiente (1..min_matches-1 confrontos)
     - zero         (sem histórico)
   "insuficiente" e "zero" resultam em h2h_factor=0.0 (no-op), mas são logados
   separadamente para visibilidade.

2. Gate de IC95% (mesma regra do F4), restrito a tournament_names=["WC2018","WC2022"]:
   diff = rps_com - rps_sem (com = h2h.weight do config; sem = 0.0)
     diff_hi < 0 (com melhor)   -> APROVADO
     diff cruza zero            -> DESCARTADO (Occam)
     diff_lo > 0 (com pior)     -> DESCARTADO (pior)
   O ensemble decide a recomendação agregada (config final é decisão do usuário).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "2_models"))

from config_loader import load_config  # noqa: E402
import dataset  # noqa: E402
import models_registry as reg  # noqa: E402
from tournament_comparison import run_comparison  # noqa: E402
from significance import compare_pair  # noqa: E402
from h2h_adjustment import h2h_count  # noqa: E402


def run_h2h_coverage(min_matches: int, ref_date=None, verbose: bool = True) -> dict:
    """Classifica os 72 jogos da Copa 2026 por cobertura de H2H.

    `ref_date` (default `cfg.data.today`, passado por `run()`): cobertura é o que
    estaria disponível HOJE para prever esses jogos -- mesmo cutoff usado pelo
    `H2HAdjustedModel` no treino real (Auditoria P1)."""
    fixtures = dataset.get_wc2026_fixtures()
    disponivel = insuficiente = zero = 0
    for _, r in fixtures.iterrows():
        n = h2h_count(r["home_team"], r["away_team"], ref_date=ref_date)
        if n >= min_matches:
            disponivel += 1
        elif n > 0:
            insuficiente += 1
        else:
            zero += 1
    total = len(fixtures)

    if verbose:
        print(f"  H2H disponível (>={min_matches} confrontos): {disponivel}/{total} jogos")
        print(f"  H2H insuficiente (<{min_matches} confrontos): {insuficiente}/{total} jogos")
        print(f"  H2H zero (sem histórico): {zero}/{total} jogos")

    return {"total": total, "disponivel": disponivel, "insuficiente": insuficiente, "zero": zero}


def run_h2h_comparison(iters: int | None = None, seed: int | None = None,
                        verbose: bool = True) -> dict:
    cfg = load_config()
    vcfg = cfg["validation"]
    iters = iters or vcfg["bootstrap_iterations"]
    seed = seed if seed is not None else vcfg["random_seed"]
    weight_on = cfg["h2h"]["weight"]

    cfg_sem = copy.deepcopy(cfg)
    cfg_sem["h2h"]["weight"] = 0.0
    cfg_com = copy.deepcopy(cfg)
    cfg_com["h2h"]["weight"] = weight_on

    model_names = reg.available_members() + ["ensemble"]
    tournament_names = ["WC2018", "WC2022"]

    if verbose:
        print("  rodando SEM h2h (weight=0.0) ...")
    comp_sem = run_comparison(cfg=cfg_sem, model_names=model_names,
                               tournament_names=tournament_names, iters=iters, verbose=verbose)
    if verbose:
        print(f"  rodando COM h2h (weight={weight_on}) ...")
    comp_com = run_comparison(cfg=cfg_com, model_names=model_names,
                               tournament_names=tournament_names, iters=iters, verbose=verbose)

    results = {}
    for mname in model_names:
        rps_com = comp_com["rps_by_model"][mname]
        rps_sem = comp_sem["rps_by_model"][mname]
        cmp = compare_pair(rps_com, rps_sem, "com_h2h", "sem_h2h", iters=iters, seed=seed)
        if cmp["diff_hi"] < 0:
            decision = "APROVADO"
        elif cmp["diff_lo"] > 0:
            decision = "DESCARTADO (pior)"
        else:
            decision = "DESCARTADO (Occam, IC cruza zero)"
        results[mname] = {
            **cmp, "decision": decision,
            "rps_sem_mean": float(np.nanmean(rps_sem)),
            "rps_com_mean": float(np.nanmean(rps_com)),
        }

    ensemble_decision = results["ensemble"]["decision"]

    if verbose:
        print(f"\n  {'modelo':18s} {'RPS sem':>10s} {'RPS com':>10s} {'IC95% diff (com-sem)':>22s}  decisão")
        for mname, r in results.items():
            print(f"  {mname:18s} {r['rps_sem_mean']:>10.4f} {r['rps_com_mean']:>10.4f} "
                  f"[{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}]      {r['decision']}")
        print(f"\n  Recomendação (ensemble decide o config final): {ensemble_decision}")

    return {"weight_on": weight_on, "iters": iters, "seed": seed,
            "results": results, "ensemble_decision": ensemble_decision}


def run(iters: int | None = None, seed: int | None = None, verbose: bool = True) -> dict:
    cfg = load_config()
    coverage = run_h2h_coverage(cfg["h2h"]["min_matches"], ref_date=cfg["data"]["today"], verbose=verbose)
    if verbose:
        print()
    comparison = run_h2h_comparison(iters=iters, seed=seed, verbose=verbose)
    return {"coverage": coverage, **comparison}


if __name__ == "__main__":
    run()
