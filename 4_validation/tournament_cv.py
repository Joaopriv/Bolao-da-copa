"""Leave-one-tournament-out CV — valida a MÁQUINA DE SELEÇÃO num torneio não visto.

Motivação (anti-fator-emocional): o corte único de `selection.py` (validação <= cutoff,
teste > cutoff) depende de QUAL torneio calhou de ser o out-of-sample. Aqui rotacionamos:
cada torneio-fold é retido uma vez, a seleção completa (otimização de pesos do ensemble +
funil de eliminação) roda nos OUTROS torneios, e o modelo escolhido é testado UMA vez no
fold retido. O número honesto é o RPS agregado nos folds retidos — nenhum fold influenciou
a escolha que o avaliou.

NÃO é k-fold aleatório: folds aleatórios vazariam futuro->passado (proibido em
`temporal_split.py`). Cada fold treina só com jogos ANTERIORES ao início do seu torneio
(já garantido por `run_comparison`/`split_for_tournament`).

`fold_kind="world_cup"` (padrão): folds = só as Copas do Mundo (WC2018, WC2022). É o pedido
literal "k-fold entre as Copas", mas com 2 folds o poder estatístico é baixo (IC largo) —
o relatório avisa. `fold_kind="all"`: folds = os 7 torneios (estimador mais robusto).
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
import baselines as BL  # noqa: E402
import metrics as M  # noqa: E402
from tournament_comparison import run_comparison  # noqa: E402
from selection import optimize_ensemble_weights, pool_logodds  # noqa: E402
from significance import elimination_funnel, COMPLEXITY  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent / "tournament_cv_report.txt"


def run_cv(cfg=None, *, fold_kind="world_cup", iters=None, verbose=True,
           write_report=True) -> dict:
    cfg = cfg or load_config()
    vcfg = cfg["validation"]
    iters = iters or vcfg["bootstrap_iterations"]
    seed = vcfg["random_seed"]
    members = reg.available_members()
    specs = vcfg["test_tournaments"]
    all_names = [s["name"] for s in specs]
    kind_of = {s["name"]: s.get("kind") for s in specs}

    if fold_kind == "world_cup":
        fold_names = [n for n in all_names if kind_of[n] == "world_cup"]
    else:
        fold_names = list(all_names)
    if not fold_names:
        raise ValueError(f"Nenhum torneio-fold para fold_kind={fold_kind!r}.")

    if verbose:
        print(f"  folds (retidos um por vez): {fold_names}")
        print(f"  pool de seleção por fold: os demais {len(all_names) - 1} torneios "
              "(otimiza pesos + funil, depois testa no fold retido)")

    # UMA comparação completa (todos os torneios, splits cronológicos sem vazamento).
    comp = run_comparison(cfg, include_ensemble=True, iters=iters, verbose=verbose)
    raw = comp["raw"]                # {modelo: {torneio: {P, y, pred_scores, true_scores}}}
    cell = comp["cell"]             # {(modelo, torneio): {metrica: Score}}  (descritivo, Nível A)
    baseline_names = {b.name for b in BL.build_baselines()}

    def concat_P(model, tnames):
        return np.vstack([raw[model][t]["P"] for t in tnames])

    def concat_y(tnames):
        return np.concatenate([raw[members[0]][t]["y"] for t in tnames])

    folds = []
    heldout_rps_parts, heldout_bolao_parts = [], []
    for t_star in fold_names:
        select_names = [t for t in all_names if t != t_star]
        y_sel = concat_y(select_names)
        member_P_sel = [concat_P(m, select_names) for m in members]
        weights = optimize_ensemble_weights(
            member_P_sel, y_sel, reg=vcfg.get("ensemble_weight_reg", 0.0))
        ens_sel_P = pool_logodds(member_P_sel, weights)

        # Funil de seleção NOS OUTROS torneios (o fold retido não participa).
        rps_sel = {m: M.rps_vector(concat_P(m, select_names), y_sel)
                   for m in comp["models"] if m != "ensemble"}
        rps_sel["ensemble"] = M.rps_vector(ens_sel_P, y_sel)
        candidates = [m for m in rps_sel if m not in baseline_names]
        funnel = elimination_funnel(rps_sel, candidates=candidates, iters=iters, seed=seed)
        chosen = funnel["chosen"]

        # Testa o escolhido UMA vez no fold retido.
        y_star = raw[members[0]][t_star]["y"]
        true_scores = raw[members[0]][t_star]["true_scores"]
        if chosen == "ensemble":
            P_star = pool_logodds([raw[m][t_star]["P"] for m in members], weights)
            pred_scores = None  # ensemble não produz placar exato
        else:
            P_star = raw[chosen][t_star]["P"]
            pred_scores = raw[chosen][t_star]["pred_scores"]

        rps_vec = M.rps_vector(P_star, y_star)
        heldout_rps_parts.append(rps_vec)
        scores = M.evaluate_all(P_star, y_star, pred_scores=pred_scores,
                                true_scores=true_scores, iters=iters, seed=seed)
        if "BolaoPoints" in scores and scores["BolaoPoints"].n > 0:
            heldout_bolao_parts.append(
                M.bolao_points_vector(pred_scores, true_scores))

        folds.append({
            "held_out": t_star,
            "chosen": chosen,
            "ensemble_weights": {m: float(w) for m, w in zip(members, weights)},
            "n_matches": len(y_star),
            "RPS": vars(scores["RPS"]),
            "BolaoPoints": vars(scores["BolaoPoints"]) if "BolaoPoints" in scores else None,
        })
        if verbose:
            print(f"  · fold {t_star:10s} -> escolhido '{chosen}'  "
                  f"RPS retido = {scores['RPS']}")

    # Agregado honesto: pool dos RPS por jogo de TODOS os folds retidos.
    pooled_rps = np.concatenate(heldout_rps_parts)
    agg_rps = M.bootstrap_ci(pooled_rps, name="RPS", lower_is_better=True,
                             iters=iters, seed=seed)
    agg_bolao = None
    if heldout_bolao_parts:
        agg_bolao = M.bootstrap_ci(np.concatenate(heldout_bolao_parts),
                                   name="BolaoPoints", lower_is_better=False,
                                   iters=iters, seed=seed)

    result = {
        "fold_kind": fold_kind,
        "fold_names": fold_names,
        "n_folds": len(fold_names),
        "folds": folds,
        "heldout_aggregate": {
            "RPS": vars(agg_rps),
            "BolaoPoints": vars(agg_bolao) if agg_bolao is not None else None,
        },
        "cell": cell,
        "models": comp["models"],
    }
    report = build_report(result)
    result["report"] = report
    if verbose:
        print("\n" + report)
    if write_report:
        REPORT_PATH.write_text(report, encoding="utf-8")
        if verbose:
            print(f"\n  relatório salvo em {REPORT_PATH.relative_to(ROOT)}")
    return result


def build_report(res: dict) -> str:
    L = []
    L.append("━━ LEAVE-ONE-TOURNAMENT-OUT CV ━━")
    scope = ("só Copas do Mundo" if res["fold_kind"] == "world_cup"
             else "todos os torneios")
    L.append(f"  Escopo dos folds: {scope}  ({res['n_folds']} folds: "
             f"{', '.join(res['fold_names'])})")
    L.append("")
    L.append("  Cada fold: seleção (pesos+funil) nos OUTROS torneios, testado 1x no retido.")
    L.append("  ── Por fold (número honesto: o fold não influenciou a escolha) ──")
    for f in res["folds"]:
        rps = f["RPS"]
        line = (f"    {f['held_out']:10s}  escolhido={f['chosen']:16s}  "
                f"n={f['n_matches']:>3}  RPS={rps['point']:.4f} "
                f"[{rps['lo']:.4f}, {rps['hi']:.4f}]")
        if f["BolaoPoints"]:
            bp = f["BolaoPoints"]
            line += f"  | pts/jogo={bp['point']:.3f}"
        L.append(line)

    agg = res["heldout_aggregate"]["RPS"]
    L.append("")
    L.append(f"  ── Agregado out-of-fold (pool de {agg['n']} jogos retidos) ──")
    L.append(f"    RPS = {agg['point']:.4f} [{agg['lo']:.4f}, {agg['hi']:.4f}]  "
             "← número honesto de generalização")
    if res["heldout_aggregate"]["BolaoPoints"]:
        bp = res["heldout_aggregate"]["BolaoPoints"]
        L.append(f"    Pontos do bolão / jogo = {bp['point']:.3f} "
                 f"[{bp['lo']:.3f}, {bp['hi']:.3f}]")

    # Tabela descritiva (Nível A): RPS de cada modelo em cada fold retido.
    L.append("")
    L.append("  ── Descritivo: RPS de cada modelo em cada fold (sem seleção) ──")
    header = "    " + f"{'modelo':18s}" + "".join(f"{t:>12s}" for t in res["fold_names"])
    L.append(header)
    real_models = [m for m in res["models"]
                   if m not in {b.name for b in BL.build_baselines()}]
    for m in sorted(real_models, key=lambda n: COMPLEXITY.get(n, 50)):
        row = f"    {m:18s}"
        for t in res["fold_names"]:
            sc = res["cell"].get((m, t), {}).get("RPS")
            row += f"{sc.point:>12.4f}" if sc is not None else f"{'-':>12s}"
        L.append(row)

    L.append("")
    if res["n_folds"] < 3:
        L.append(f"  ⚠ AVISO (poder estatístico): só {res['n_folds']} folds. O IC agregado "
                 "vem do bootstrap sobre os JOGOS retidos, mas a variância ENTRE torneios")
        L.append("    (contexto/edição) está mal amostrada com tão poucos folds. Não use "
                 "isto para distinguir modelos próximos — use para sanidade de generalização.")
        L.append("    Para um estimador mais robusto: rode com --cv-scope all (7 folds).")
    L.append("")
    L.append("  Lembrete: 1 rodada da Copa 2026 (~4-8 jogos) é DOMINADA por variância. "
             "Este CV mede generalização do método, não 'consertar' uma rodada ruim — "
             "re-tunar o modelo por causa de 1 rodada é exatamente o fator emocional a evitar.")
    return "\n".join(L)


def run(*, fold_kind="world_cup", iters=None, verbose=True):
    """Entrada usada pelo main.py (--cv-tournaments)."""
    return run_cv(fold_kind=fold_kind, iters=iters, verbose=verbose)
