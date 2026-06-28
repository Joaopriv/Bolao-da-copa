"""[Validação de mercados de aposta] Cada mercado (1X2, over/under total e por time,
BTTS) derivado da grade de placar (utils/markets.py) é validado nas Copas/torneios de
teste, walk-forward SEM vazamento (treina só com jogos anteriores ao torneio).

Métrica central: BRIER SKILL SCORE (BSS) contra a climatologia (taxa-base histórica do
mercado, calculada SÓ no treino). BSS = 1 - Brier_modelo / Brier_climatologia:
  BSS > 0  -> modelo bate o palpite ingênuo "sempre a taxa-base" (tem skill real)
  BSS ~ 0  -> não acrescenta nada além de saber a frequência histórica
  BSS < 0  -> pior que o palpite ingênuo
IC95% por bootstrap sobre os jogos (Princípio #1: todo número com IC). Se o IC do BSS
cruza zero, o mercado NÃO tem skill comprovado -- não apostar nele com confiança.

Também reporta calibração (ECE) e log-loss vs climatologia. 1X2 entra como acurácia
(o RPS dele já é validado em tournament_comparison.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "utils"))

from config_loader import load_config  # noqa: E402
import dataset  # noqa: E402
import models_registry as reg  # noqa: E402
from temporal_split import split_for_tournament  # noqa: E402
import markets as MK  # noqa: E402

_EPS = 1e-12


def _binary_specs(total_lines, team_lines) -> list[tuple]:
    """[(chave, função(grade)->p_sim, função(hs,as)->resultado_0/1, função(treino_df)->taxa_base)]."""
    specs = []
    for L in total_lines:
        specs.append((
            f"total>{L}",
            lambda g, L=L: MK.total_over_under(g, [L])[L]["over"],
            lambda hs, a, L=L: int((hs + a) > L),
            lambda df, L=L: float(((df["home_score"] + df["away_score"]) > L).mean()),
        ))
    for L in team_lines:
        specs.append((
            f"casa>{L}",
            lambda g, L=L: MK.team_over_under(g, "home", [L])[L]["over"],
            lambda hs, a, L=L: int(hs > L),
            lambda df, L=L: float((df["home_score"] > L).mean()),
        ))
    for L in team_lines:
        specs.append((
            f"fora>{L}",
            lambda g, L=L: MK.team_over_under(g, "away", [L])[L]["over"],
            lambda hs, a, L=L: int(a > L),
            lambda df, L=L: float((df["away_score"] > L).mean()),
        ))
    specs.append((
        "BTTS",
        lambda g: MK.btts(g)["yes"],
        lambda hs, a: int(hs >= 1 and a >= 1),
        lambda df: float(((df["home_score"] >= 1) & (df["away_score"] >= 1)).mean()),
    ))
    return specs


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 5) -> float:
    """Expected Calibration Error binário (|prob prevista - freq observada| por bin)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def _bss_ci(p_model: np.ndarray, p_base: np.ndarray, y: np.ndarray,
            iters: int, seed: int, pct=(2.5, 97.5)) -> dict:
    """Brier Skill Score (1 - BS_modelo/BS_base) + IC95% bootstrap sobre os jogos."""
    bs_m = float(np.mean((p_model - y) ** 2))
    bs_b = float(np.mean((p_base - y) ** 2))
    bss = 1.0 - bs_m / bs_b if bs_b > 0 else float("nan")
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(iters, n))
    bm = ((p_model[idx] - y[idx]) ** 2).mean(axis=1)
    bb = ((p_base[idx] - y[idx]) ** 2).mean(axis=1)
    boot = np.where(bb > 0, 1.0 - bm / bb, np.nan)
    lo, hi = np.nanpercentile(boot, pct)
    return {"brier_model": bs_m, "brier_base": bs_b, "bss": bss,
            "bss_lo": float(lo), "bss_hi": float(hi)}


def run(model_name: str | None = None, iters: int | None = None, verbose: bool = True) -> dict:
    cfg = load_config()
    vcfg = cfg["validation"]
    iters = iters or vcfg["bootstrap_iterations"]
    seed = vcfg["random_seed"]
    lam = cfg["preprocess"]["temporal_decay_lambda"]

    if model_name is None:
        sel_path = ROOT / "5_outputs" / "selected_model.json"
        model_name = (json.loads(sel_path.read_text(encoding="utf-8")).get("chosen_model", "poisson")
                      if sel_path.exists() else "poisson")

    bspecs = _binary_specs(MK.TOTAL_LINES, MK.TEAM_LINES)
    pooled = {k: {"p_model": [], "p_base": [], "y": []} for k, *_ in bspecs}
    res_correct = []  # acurácia 1X2

    for spec in vcfg["test_tournaments"]:
        train_df, test = split_for_tournament(spec, lam)

        model = reg.build_member(model_name, cfg) if model_name != "ensemble" \
            else reg.build_ensemble(cfg=cfg)
        model.fit(train_df)
        # Mercados precisam da grade conjunta; se o escolhido não a produz (ex. elo),
        # usa dixon_coles como provedor de placar (mesma política do predict_2026).
        score_model = model if getattr(model, "supports_scoreline", False) \
            else reg.build_member("dixon_coles", cfg).fit(train_df)

        base_rates = {k: base_fn(train_df) for k, _, _, base_fn in bspecs}

        for _, r in test.iterrows():
            home, away, neutral = str(r["home_team"]), str(r["away_team"]), bool(r["neutral"])
            hs, a = int(r["home_score"]), int(r["away_score"])
            grid = score_model.predict_scoreline(home, away, neutral)
            if grid is None:
                continue
            r1x2 = MK.result_1x2(grid)
            pred_res = max(("home", "draw", "away"), key=lambda o: r1x2[o])
            actual_res = "home" if hs > a else ("away" if a > hs else "draw")
            res_correct.append(int(pred_res == actual_res))

            for k, prob_fn, out_fn, _ in bspecs:
                pooled[k]["p_model"].append(prob_fn(grid))
                pooled[k]["p_base"].append(base_rates[k])
                pooled[k]["y"].append(out_fn(hs, a))

    results = {}
    for k, *_ in bspecs:
        p_model = np.array(pooled[k]["p_model"], dtype=float)
        p_base = np.array(pooled[k]["p_base"], dtype=float)
        y = np.array(pooled[k]["y"], dtype=float)
        stats = _bss_ci(p_model, p_base, y, iters=iters, seed=seed)
        stats["n"] = len(y)
        stats["base_rate"] = float(y.mean())
        stats["ece"] = _ece(p_model, y)
        # log-loss modelo vs climatologia
        pc = np.clip(p_model, _EPS, 1 - _EPS)
        bc = np.clip(p_base, _EPS, 1 - _EPS)
        stats["logloss_model"] = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
        stats["logloss_base"] = float(-np.mean(y * np.log(bc) + (1 - y) * np.log(1 - bc)))
        results[k] = stats

    acc_1x2 = float(np.mean(res_correct)) if res_correct else float("nan")
    n_matches = len(res_correct)

    if verbose:
        print(f"  modelo: {model_name} | {n_matches} jogos | "
              f"{len(vcfg['test_tournaments'])} torneios | walk-forward sem vazamento\n")
        print(f"  acurácia 1X2 (resultado mais provável): {acc_1x2:.1%}\n")
        print(f"  {'mercado':>10s} {'n':>4s} {'taxa-base':>9s} {'Brier mod':>9s} "
              f"{'Brier clim':>10s} {'skill (BSS)':>12s} {'IC95% BSS':>20s} {'ECE':>6s}  veredito")
        for k, *_ in bspecs:
            s = results[k]
            tem_skill = s["bss_lo"] > 0
            veredito = "✅ skill" if tem_skill else (
                "⚠ sem skill (IC cruza 0)" if s["bss_hi"] > 0 else "❌ pior que clim.")
            print(f"  {k:>10s} {s['n']:>4d} {s['base_rate']:>8.1%} {s['brier_model']:>9.4f} "
                  f"{s['brier_base']:>10.4f} {s['bss']:>11.1%} "
                  f"[{s['bss_lo']:>+6.1%},{s['bss_hi']:>+6.1%}] {s['ece']:>6.3f}  {veredito}")
        print("\n  BSS > 0 com IC95% acima de zero = mercado tem skill real (vale modelar/apostar).")
        print("  IC cruzando zero = indistinguível de só usar a frequência histórica.")

    return {"model": model_name, "n_matches": n_matches, "acc_1x2": acc_1x2,
            "markets": results}


if __name__ == "__main__":
    run()
