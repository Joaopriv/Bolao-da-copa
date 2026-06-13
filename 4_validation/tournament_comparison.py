"""Comparação cruzada: todos os modelos × todos os torneios de teste.

Para cada torneio (split cronológico sem vazamento), roda todos os modelos + baselines,
calcula as métricas com IC por célula (modelo×torneio) e agregadas (pool de todos os
jogos). Também monta os vetores de RPS por jogo ALINHADOS entre modelos, que alimentam
o funil de eliminação em significance.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_config  # noqa: E402
import models_registry as reg  # noqa: E402
from temporal_split import split_for_tournament  # noqa: E402
from backtest_engine import run_model_on_tournament  # noqa: E402
import metrics as M  # noqa: E402
from significance import elimination_funnel  # noqa: E402
import baselines as BL  # noqa: E402


def _model_factory(cfg, include_ensemble=True):
    """Mapa nome -> função que cria uma instância NOVA (não treinada) do modelo."""
    factory = {b.name: (lambda b=b: type(b)()) for b in BL.build_baselines()}
    for name in reg.available_members():
        factory[name] = (lambda n=name: reg.build_member(n, cfg))
    if include_ensemble:
        factory["ensemble"] = (lambda: reg.build_ensemble(cfg=cfg))  # pesos iguais
    return factory


def run_comparison(cfg=None, *, model_names=None, tournament_names=None,
                   lam=None, iters=None, include_ensemble=True, verbose=True) -> dict:
    cfg = cfg or load_config()
    vcfg = cfg["validation"]
    lam = lam if lam is not None else cfg["preprocess"]["temporal_decay_lambda"]
    iters = iters or vcfg["bootstrap_iterations"]
    seed = vcfg["random_seed"]
    pct = tuple(vcfg["ci_percentiles"])

    specs = vcfg["test_tournaments"]
    if tournament_names:
        specs = [s for s in specs if s["name"] in tournament_names]
    factory = _model_factory(cfg, include_ensemble)
    names = model_names or list(factory)
    # Permite `--models` referenciar qualquer modelo registrado em _BUILDERS, mesmo que
    # não esteja em config.models.members (comparação ad-hoc, ex.: candidatos ainda não
    # promovidos). Não afeta o conjunto default (sem --models).
    for n in names:
        if n not in factory:
            factory[n] = (lambda n=n: reg.build_member(n, cfg))

    cell: dict[tuple, dict] = {}          # (model, tourn) -> {metric: Score}
    raw: dict[tuple, dict] = {}           # (model, tourn) -> backtest result
    for spec in specs:
        tname = spec["name"]
        if verbose:
            print(f"  · torneio {tname} ...", flush=True)
        train_df, _ = split_for_tournament(spec, lam)
        for mname in names:
            model = factory[mname]()
            res = run_model_on_tournament(model, spec, lam, train_df=train_df)
            raw[(mname, tname)] = res
            cell[(mname, tname)] = M.evaluate_all(
                res["P"], res["y"], pred_scores=res["pred_scores"],
                true_scores=res["true_scores"], iters=iters, percentiles=pct, seed=seed,
            )
            if verbose:
                print(f"      {mname:18s} RPS={cell[(mname,tname)]['RPS']}", flush=True)

    tourn_names = [s["name"] for s in specs]
    aggregate, rps_by_model, pooled_P = {}, {}, {}
    pooled_y = None
    for mname in names:
        P = np.vstack([raw[(mname, t)]["P"] for t in tourn_names])
        y = np.concatenate([raw[(mname, t)]["y"] for t in tourn_names])
        pred_s = sum((raw[(mname, t)]["pred_scores"] for t in tourn_names), [])
        true_s = sum((raw[(mname, t)]["true_scores"] for t in tourn_names), [])
        aggregate[mname] = M.evaluate_all(P, y, pred_scores=pred_s, true_scores=true_s,
                                          iters=iters, percentiles=pct, seed=seed)
        rps_by_model[mname] = M.rps_vector(P, y)  # alinhado: mesma ordem de jogos p/ todos
        pooled_P[mname] = P
        pooled_y = y

    # Candidatos a "escolhido" = modelos reais (exclui baselines de sanidade).
    baseline_names = {b.name for b in BL.build_baselines()}
    candidates = [n for n in names if n not in baseline_names]
    funnel = elimination_funnel(rps_by_model, candidates=candidates, iters=iters, seed=seed)

    # Raw por (modelo, torneio) aninhado {modelo: {torneio: {...}}} — permite a select()
    # reaproveitar UMA comparação (deriva validação/teste por subconjunto de torneios).
    raw_nested = {m: {t: raw[(m, t)] for t in tourn_names} for m in names}

    return {
        "lambda": lam, "iters": iters,
        "tournaments": tourn_names, "models": names,
        "cell": cell, "aggregate": aggregate,
        "rps_by_model": {k: v.tolist() for k, v in rps_by_model.items()},
        "pooled_P": pooled_P, "pooled_y": pooled_y,
        "raw": raw_nested,
        "funnel": funnel,
    }
