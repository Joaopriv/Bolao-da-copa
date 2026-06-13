"""Seleção anti-overfitting + otimização dos pesos do ensemble.

Etapa 4g/4h da spec:
- Pesos do ensemble otimizados APENAS na validação (torneios <= cutoff), minimizando RPS.
- Modelo escolhido pelo funil de eliminação na validação (empate -> mais simples).
- O escolhido é testado UMA vez nos torneios > cutoff -> número HONESTO (out-of-sample).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_config  # noqa: E402
import models_registry as reg  # noqa: E402
from tournament_comparison import run_comparison  # noqa: E402
from significance import elimination_funnel, COMPLEXITY  # noqa: E402
import metrics as M  # noqa: E402

_EPS = 1e-9


def pool_logodds(member_P: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    """Pooling log-odds ponderado de previsões 1X2 dos membros -> (N,3) normalizado."""
    stack = np.stack([np.clip(p, _EPS, 1.0) for p in member_P])  # (k, N, 3)
    logp = np.tensordot(w, np.log(stack), axes=(0, 0))
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def optimize_ensemble_weights(member_P: list[np.ndarray], y: np.ndarray,
                               reg: float = 0.0) -> np.ndarray:
    """Acha os pesos (simplex) que minimizam o RPS médio do ensemble na validação.

    [Auditoria P10] `reg` (cfg.validation.ensemble_weight_reg): penalidade L2 em
    direção ao peso uniforme (1/k), somada ao RPS médio. Sem regularização, SLSQP
    tende a soluções de canto do simplex ajustadas a ruído da amostra de validação
    (pequena). `reg=0.0` reproduz o comportamento original.
    """
    k = len(member_P)
    uniform = np.ones(k) / k

    def neg(w):
        rps = M.rps_vector(pool_logodds(member_P, w), y).mean()
        return rps + reg * float(np.sum((w - uniform) ** 2))

    x0 = uniform.copy()
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * k
    res = minimize(neg, x0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-8})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def build_justification(sel: dict) -> str:
    """Texto de justificativa no estilo da spec (PT-BR), guiado pela regra do IC."""
    f = sel["funnel"]
    ins = sel["in_sample"]["RPS"]
    oos = sel["out_of_sample"]["RPS"]
    parts = []
    if f["eliminated"]:
        parts.append("Descartados por IC da diferença longe de zero (claramente piores): "
                     + ", ".join(f["eliminated"]) + ".")
    parts.append("Equivalentes estatisticamente (IC da diferença cruza zero): "
                 + ", ".join(f["equivalent"]) + ".")
    parts.append(f"Entre os equivalentes, escolhido '{sel['chosen']}' por ser o mais "
                 "simples (navalha de Occam).")
    parts.append(f"RPS in-sample (<= {sel['cutoff_year']}) = {ins['point']:.4f} "
                 f"[{ins['lo']:.4f}, {ins['hi']:.4f}]; "
                 f"RPS out-of-sample (> {sel['cutoff_year']}) = {oos['point']:.4f} "
                 f"[{oos['lo']:.4f}, {oos['hi']:.4f}] (este é o número honesto).")
    # [Auditoria P2/P7] "Equivalente" = IC da diferença cruza zero, i.e. a amostra de
    # validação (~169 jogos) não tem poder estatístico para distinguir os modelos --
    # não é prova de igualdade. Além disso, escolher entre os equivalentes usando o
    # mesmo RPS de validação do funil tende a favorecer quem teve mais sorte de amostra
    # (winner's curse), por isso o in-sample acima tende a ser otimista -- o
    # out-of-sample (testado uma única vez, sem reotimizar) é o número confiável.
    parts.append("Nota (P2/P7): 'equivalentes' = indistinguíveis dada a amostra de "
                 "validação (poder estatístico limitado), não 'comprovadamente iguais'; "
                 "e o RPS in-sample do escolhido tende a ser otimista (winner's curse) -- "
                 "use o out-of-sample para expectativas futuras.")
    parts.append("Iteração 1: sem xG/elenco/sequencial — adiados para a Iteração 2 por "
                 "falta de fonte de dados.")
    # [Auditoria M5/P5] o RPS do bayesiano no funil acima vem do modo "fast" (config.
    # models.bayesian.fast: trace menor, max_train_matches=200) -- mais rápido para
    # caber em --compare/--select, mas MAIS RUIDOSO que o modo "full" usado em
    # --predict-2026 (trace maior, max_train_matches=800). Se o bayesiano pesa na
    # decisão, o número validado pode não corresponder ao que rodaria em produção.
    bw = sel["ensemble_weights"].get("bayesian", 0.0)
    if sel["chosen"] == "bayesian" or bw > 0.05:
        parts.append("⚠ Aviso (M5/P5): o bayesiano participa da decisão "
                     f"({'escolhido' if sel['chosen'] == 'bayesian' else f'peso={bw:.2f} no ensemble'}), "
                     "mas seu RPS de validação usa o modo 'fast' (trace menor que o 'full' "
                     "usado em --predict-2026) -- o número validado pode não refletir "
                     "exatamente o comportamento em produção.")
    return " ".join(parts)


def select(cfg=None, *, iters=None, verbose=True) -> dict:
    cfg = cfg or load_config()
    vcfg = cfg["validation"]
    cutoff = vcfg["selection_cutoff_year"]
    iters = iters or vcfg["bootstrap_iterations"]
    seed = vcfg["random_seed"]
    members = reg.available_members()

    tt = vcfg["test_tournaments"]
    val_names = [t["name"] for t in tt if t["year"] <= cutoff]
    test_names = [t["name"] for t in tt if t["year"] > cutoff]

    if verbose:
        print(f"  validação (<= {cutoff}): {val_names}")
        print(f"  teste out-of-sample (> {cutoff}): {test_names}")

    # UMA comparação completa (todos os torneios, sem vazamento por torneio). A validação
    # e o teste out-of-sample são DERIVADOS por subconjunto de torneios — evita refitar tudo
    # três vezes. Os pesos do ensemble são otimizados SÓ na validação.
    comp = run_comparison(cfg, include_ensemble=True, iters=iters, verbose=verbose)
    raw = comp["raw"]  # {modelo: {torneio: {P,y,pred_scores,true_scores}}}

    def concat_P(model, tnames):
        return np.vstack([raw[model][t]["P"] for t in tnames])

    def concat_y(tnames):
        return np.concatenate([raw[members[0]][t]["y"] for t in tnames])

    y_val, y_test = concat_y(val_names), concat_y(test_names)
    member_P_val = [concat_P(m, val_names) for m in members]
    member_P_test = [concat_P(m, test_names) for m in members]

    weights = optimize_ensemble_weights(member_P_val, y_val,
                                         reg=vcfg.get("ensemble_weight_reg", 0.0))
    w_map = {m: float(w) for m, w in zip(members, weights)}
    ens_val_P = pool_logodds(member_P_val, weights)

    # Funil de seleção na VALIDAÇÃO (RPS por modelo nos torneios <= cutoff) com ensemble ótimo.
    import baselines as BL
    baseline_names = {b.name for b in BL.build_baselines()}
    rps_val = {m: M.rps_vector(concat_P(m, val_names), y_val)
               for m in comp["models"] if m != "ensemble"}
    rps_val["ensemble"] = M.rps_vector(ens_val_P, y_val)  # usa pesos ótimos, não pesos iguais
    candidates = [m for m in rps_val if m not in baseline_names]
    funnel = elimination_funnel(rps_val, candidates=candidates, iters=iters, seed=seed)
    chosen = funnel["chosen"]

    # OOS (UMA vez) nos torneios > cutoff.
    if chosen == "ensemble":
        oos_P, ins_P = pool_logodds(member_P_test, weights), ens_val_P
    else:
        oos_P, ins_P = concat_P(chosen, test_names), concat_P(chosen, val_names)

    oos = M.evaluate_all(oos_P, y_test, iters=iters, seed=seed)
    ins = M.evaluate_all(ins_P, y_val, iters=iters, seed=seed)

    return {
        "cutoff_year": cutoff,
        "validation_tournaments": val_names,
        "test_tournaments": test_names,
        "members": members,
        "ensemble_weights": w_map,
        "funnel": funnel,
        "chosen": chosen,
        "in_sample": {k: vars(v) for k, v in ins.items()},
        "out_of_sample": {k: vars(v) for k, v in oos.items()},
        "complexity_rank": {m: COMPLEXITY.get(m, 50) for m in funnel["equivalent"]},
        "comparison": comp,  # reaproveitado pelo relatório (sem nova comparação)
    }
