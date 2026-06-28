"""[Pergunta do usuário 2026-06-27] RPS do modelo escolhido restrito aos jogos de
MATA-MATA dos torneios de teste (vs. fase de grupos) -- diagnóstico para estimar o
que esperar no mata-mata real da Copa 2026. Não entra na seleção/treino, é só leitura.

Mata-mata é identificado SEM hardcode de formato por torneio (nº de grupos/vagas
varia entre WC/Euro/Copa América e mudou de edição pra edição): G = nº de jogos da
fase de grupos por seleção (moda da contagem de jogos por time no torneio); todo jogo
que é o (G+1)-ésimo ou posterior de QUALQUER um dos dois times é mata-mata -- válido
porque a fase de grupos sempre termina, para TODOS os times, antes do mata-mata
começar (nunca se intercalam no calendário).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_config  # noqa: E402
import dataset  # noqa: E402
import models_registry as reg  # noqa: E402
from temporal_split import split_for_tournament  # noqa: E402
import metrics as M  # noqa: E402

_RES_IDX = {"H": 0, "D": 1, "A": 2}


def _stage_mask(test) -> tuple[np.ndarray, int]:
    """True = mata-mata, por linha de `test` (já ordenado por data). Retorna
    também `g` (jogos da fase de grupos por time, p/ log).

    g = MÍNIMO de jogos por time (não moda): o(s) time(s) eliminado(s) logo após a
    fase de grupos sempre jogam exatamente `g` jogos e nenhum time joga menos --
    moda falha quando o nº de eliminados na fase de grupos é igual (ou menor) ao nº
    eliminado na primeira rodada do mata-mata (ex. Copa América com poucos grupos
    e muitas vagas de classificação, visto em CopaAm2019/2021)."""
    counts = Counter()
    for _, r in test.iterrows():
        counts[r["home_team"]] += 1
        counts[r["away_team"]] += 1
    g = min(counts.values())

    played: Counter = Counter()
    mask = []
    for _, r in test.iterrows():
        mask.append(played[r["home_team"]] >= g or played[r["away_team"]] >= g)
        played[r["home_team"]] += 1
        played[r["away_team"]] += 1
    return np.array(mask), g


def _build_model(model_name: str, cfg: dict):
    if model_name == "ensemble":
        return reg.build_ensemble(cfg=cfg)
    return reg.build_member(model_name, cfg)


def run(model_name: str | None = None, iters: int | None = None, verbose: bool = True,
        kinds: tuple[str, ...] | None = None) -> dict:
    """`kinds`: filtra `test_tournaments` por `spec["kind"]` (ex. ("world_cup",) para
    só Copas do Mundo -- "torneios de teste" inclui Euro/Copa América também)."""
    cfg = load_config()
    vcfg = cfg["validation"]
    iters = iters or vcfg["bootstrap_iterations"]
    seed = vcfg["random_seed"]
    pct = tuple(vcfg["ci_percentiles"])
    lam = cfg["preprocess"]["temporal_decay_lambda"]

    if model_name is None:
        sel_path = ROOT / "5_outputs" / "selected_model.json"
        model_name = (json.loads(sel_path.read_text(encoding="utf-8")).get("chosen_model", "poisson")
                      if sel_path.exists() else "poisson")

    specs = vcfg["test_tournaments"]
    if kinds:
        specs = [s for s in specs if s.get("kind") in kinds]
    rps_group_all, rps_ko_all = [], []
    acc_group_all, acc_ko_all = [], []
    per_tournament = []

    for spec in specs:
        train_df, _ = split_for_tournament(spec, lam)
        test = dataset.get_test_tournament(spec)
        ko_mask, g = _stage_mask(test)

        model = _build_model(model_name, cfg)
        model.fit(train_df)

        P, y = [], []
        for _, r in test.iterrows():
            P.append(model.predict_proba(str(r["home_team"]), str(r["away_team"]), bool(r["neutral"])))
            y.append(_RES_IDX[r["result"]])
        P, y = np.asarray(P, dtype=float), np.asarray(y, dtype=int)
        rps = M.rps_vector(P, y)
        acc = M.accuracy_vector(P, y)

        rps_group_all.append(rps[~ko_mask])
        rps_ko_all.append(rps[ko_mask])
        acc_group_all.append(acc[~ko_mask])
        acc_ko_all.append(acc[ko_mask])
        per_tournament.append({
            "name": spec["name"], "g": g,
            "n_group": int((~ko_mask).sum()), "n_knockout": int(ko_mask.sum()),
            "rps_group": float(np.mean(rps[~ko_mask])) if (~ko_mask).any() else float("nan"),
            "rps_knockout": float(np.mean(rps[ko_mask])) if ko_mask.any() else float("nan"),
            "acc_group": float(np.mean(acc[~ko_mask])) if (~ko_mask).any() else float("nan"),
            "acc_knockout": float(np.mean(acc[ko_mask])) if ko_mask.any() else float("nan"),
            "hits_knockout": int(acc[ko_mask].sum()),
        })
        if verbose:
            t = per_tournament[-1]
            print(f"  {t['name']:12s} grupos(G={g}): {t['n_group']:3d} jogos, "
                  f"acerto={t['acc_group']:.1%}   |   mata-mata: {t['n_knockout']:3d} jogos, "
                  f"acerto={t['acc_knockout']:.1%} ({t['hits_knockout']}/{t['n_knockout']})")

    rps_group = np.concatenate(rps_group_all)
    rps_ko = np.concatenate(rps_ko_all)
    acc_group = np.concatenate(acc_group_all)
    acc_ko = np.concatenate(acc_ko_all)
    score_group = M.bootstrap_ci(rps_group, name="RPS_grupos", lower_is_better=True,
                                  iters=iters, percentiles=pct, seed=seed)
    score_ko = M.bootstrap_ci(rps_ko, name="RPS_mata_mata", lower_is_better=True,
                               iters=iters, percentiles=pct, seed=seed)
    acc_score_group = M.bootstrap_ci(acc_group, name="Acerto1X2_grupos", lower_is_better=False,
                                      iters=iters, percentiles=pct, seed=seed)
    acc_score_ko = M.bootstrap_ci(acc_ko, name="Acerto1X2_mata_mata", lower_is_better=False,
                                   iters=iters, percentiles=pct, seed=seed)

    if verbose:
        print(f"\n  modelo avaliado: {model_name}"
              + (f"  (torneios: {', '.join(kinds)})" if kinds else ""))
        print(f"  taxa de acerto (1X2) fase de grupos ({acc_score_group.n} jogos): "
              f"{acc_score_group.point:.1%} [{acc_score_group.lo:.1%}, {acc_score_group.hi:.1%}]")
        print(f"  taxa de acerto (1X2) mata-mata      ({acc_score_ko.n} jogos): "
              f"{acc_score_ko.point:.1%} [{acc_score_ko.lo:.1%}, {acc_score_ko.hi:.1%}]")
        print(f"  RPS fase de grupos ({score_group.n} jogos): {score_group}")
        print(f"  RPS mata-mata      ({score_ko.n} jogos): {score_ko}")

    return {"model": model_name, "per_tournament": per_tournament,
            "rps_group": score_group, "rps_knockout": score_ko,
            "acc_group": acc_score_group, "acc_knockout": acc_score_ko}


if __name__ == "__main__":
    run()
