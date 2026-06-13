"""[Iteração 2 / D7] Revalidação com/sem squad_offset — regra do IC.

Roda `run_comparison()` duas vezes sobre TODOS os membros + ensemble: uma com
`squad_strength.squad_offset_weight` forçado a 0.0 ("sem", equivalente à Iteração 1 —
`build_member` não envolve `SquadAdjustedModel`) e outra com o valor do config ("com").
`w_overlap` (D6.1, peso de treino) permanece ATIVO nas duas — só o wrapper de tilting
(D6.2) é alternado.

Regra do IC (por modelo, diff = rps_com - rps_sem; negativo = "com" melhor):
  diff_hi < 0 (não cruza zero, "com" melhor)  -> MANTER  (squad_offset_weight no config)
  diff cruza zero                             -> DESLIGAR (squad_offset_weight = 0)
  diff_lo > 0 (não cruza zero, "com" pior)    -> BUG (provável sinal trocado em
                                                  def_z/tilt_grid) -- investigar antes
                                                  de qualquer decisão.

O ensemble decide a recomendação agregada (config final é decisão do usuário).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_config, path  # noqa: E402
import db_client  # noqa: E402
import models_registry as reg  # noqa: E402
from tournament_comparison import run_comparison  # noqa: E402
from significance import compare_pair  # noqa: E402


def run_squad_offset_comparison(iters: int | None = None, seed: int | None = None,
                                 verbose: bool = True) -> dict:
    cfg = load_config()
    vcfg = cfg["validation"]
    iters = iters or vcfg["bootstrap_iterations"]
    seed = seed if seed is not None else vcfg["random_seed"]
    weight_on = cfg["squad_strength"]["squad_offset_weight"]

    cfg_sem = copy.deepcopy(cfg)
    cfg_sem["squad_strength"]["squad_offset_weight"] = 0.0
    cfg_com = copy.deepcopy(cfg)
    cfg_com["squad_strength"]["squad_offset_weight"] = weight_on

    model_names = reg.available_members() + ["ensemble"]

    if verbose:
        print(f"  rodando SEM squad_offset (weight=0.0) ...")
    comp_sem = run_comparison(cfg=cfg_sem, model_names=model_names, iters=iters, verbose=verbose)
    if verbose:
        print(f"  rodando COM squad_offset (weight={weight_on}) ...")
    comp_com = run_comparison(cfg=cfg_com, model_names=model_names, iters=iters, verbose=verbose)

    results = {}
    for mname in model_names:
        rps_com = comp_com["rps_by_model"][mname]
        rps_sem = comp_sem["rps_by_model"][mname]
        cmp = compare_pair(rps_com, rps_sem, "com_squad", "sem_squad", iters=iters, seed=seed)
        if cmp["diff_hi"] < 0:
            decision = "MANTER"
        elif cmp["diff_lo"] > 0:
            decision = "BUG"
        else:
            decision = "DESLIGAR"
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
        print(f"\n  Recomendação (ensemble decide o config final): {ensemble_decision}"
              f" (squad_offset_weight={'0.0' if ensemble_decision == 'DESLIGAR' else weight_on})")
        if ensemble_decision == "BUG":
            print("  AVISO: ensemble PIOR com squad_offset -- investigar sinal trocado "
                  "em def_z/tilt_grid (squad_adjustment.py) antes de decidir.")

    return {"weight_on": weight_on, "iters": iters, "seed": seed,
            "results": results, "ensemble_decision": ensemble_decision}


def _snapshot_odds(out_path: Path, verbose: bool = True) -> dict | None:
    """Snapshot de `predictions` (modelo escolhido, com market_prob_* já preenchidos
    por `crosscheck_odds`) no mesmo formato de `odds_before.json`, para diff manual."""
    from odds_crosscheck import _load_chosen_model_name

    chosen = _load_chosen_model_name()
    rows = db_client.fetch_all("predictions") or []
    rows = [r for r in rows if r["model_name"] == chosen and r.get("market_prob_home") is not None]
    if not rows:
        if verbose:
            print("  sem odds de mercado em predictions -- pulando snapshot.")
        return None

    games, diffs = [], []
    for r in rows:
        diff_pp = max(abs(r["prob_home"] - r["market_prob_home"]),
                       abs(r["prob_draw"] - r["market_prob_draw"]),
                       abs(r["prob_away"] - r["market_prob_away"]))
        diffs.append(diff_pp)
        games.append({
            "game_id": r["game_id"], "home_team": r["home_team"], "away_team": r["away_team"],
            "model_name": r["model_name"],
            "prob_home": r["prob_home"], "prob_draw": r["prob_draw"], "prob_away": r["prob_away"],
            "market_prob_home": r["market_prob_home"], "market_prob_draw": r["market_prob_draw"],
            "market_prob_away": r["market_prob_away"], "diff_pp": round(diff_pp, 3),
        })

    snapshot = {
        "chosen_model": chosen, "n_games": len(games),
        "n_over_5pp": sum(1 for d in diffs if d > 0.05),
        "mean_diff_pp": round(float(np.mean(diffs)) * 100, 2),
        "games": games,
    }
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"  snapshot de odds -> {out_path.relative_to(ROOT)} "
              f"({snapshot['n_games']} jogos, {snapshot['n_over_5pp']} > 5pp, "
              f"média={snapshot['mean_diff_pp']}pp)")
    return snapshot


def run(iters: int | None = None, seed: int | None = None, verbose: bool = True) -> dict:
    result = run_squad_offset_comparison(iters=iters, seed=seed, verbose=verbose)

    import os
    if os.environ.get("ODDS_API_KEY"):
        from odds_crosscheck import crosscheck_odds
        if verbose:
            print("\n● Cruzando previsões (com squad_offset, config atual) com odds de mercado ...")
        crosscheck_odds(verbose=verbose)
        _snapshot_odds(path("5_outputs", "odds_after.json"), verbose=verbose)
        before = path("5_outputs", "odds_before.json")
        if verbose and before.exists():
            print(f"  diff manual: compare 5_outputs/odds_after.json com {before.relative_to(ROOT)}")
    elif verbose:
        print("\n  ODDS_API_KEY ausente -- pulando crosscheck de odds (odds_after.json).")

    return result


if __name__ == "__main__":
    run()
