"""[Pergunta do usuário 2026-06-28] Confiança do modelo (probabilidade do resultado
mais provável) x desempenho real, restrito aos jogos da fase de grupos da Copa 2026
JÁ JOGADOS (rodadas 1-3).

Walk-forward honesto (sem vazamento): cada rodada é prevista com o modelo treinado
SÓ com os jogos anteriores ao início daquela rodada -- o mesmo corte que
`--update-round N` usaria na hora, não o modelo final (pós-rodada-3) com hindsight
de tudo. Cada rodada usa, portanto, um fit diferente do modelo escolhido.

Bins de confiança de 10 em 10% (0-100%). Por bin: nº de jogos, acerto 1X2 (resultado
mais provável bateu: vitória/empate certo), erro 1X2, e placar cravado (top-1 do grid
de placar -- mesmo `score_model` de `predict_2026.py`, com fallback pra dixon_coles
se o escolhido não suportar placar -- bateu o placar real).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "1_data"))

from config_loader import load_config  # noqa: E402
import dataset  # noqa: E402
import db_client  # noqa: E402
import models_registry as reg  # noqa: E402

_RES_IDX = {"H": 0, "D": 1, "A": 2}


def _parse_score(s: str) -> tuple[int, int]:
    h, a = s.split("-")
    return int(h), int(a)


def _group_round_windows() -> list[tuple[int, str, str]]:
    """[(round_n, min_date, max_date)] da fase de grupos, a partir de
    `copa_2026_results` (exclui mata-mata via group_name começando com "Group")."""
    rows = db_client.fetch_all("copa_2026_results") or []
    by_round: dict[int, list[str]] = {}
    for r in rows:
        if not (r.get("group_name") or "").startswith("Group"):
            continue
        by_round.setdefault(r["round"], []).append(r["date"])
    return sorted((n, min(ds), max(ds)) for n, ds in by_round.items())


def run(model_name: str | None = None, n_bins: int = 10, verbose: bool = True) -> dict:
    cfg = load_config()
    lam = cfg["preprocess"]["temporal_decay_lambda"]
    disp = cfg.get("display_names", {})

    if model_name is None:
        sel_path = ROOT / "5_outputs" / "selected_model.json"
        model_name = (json.loads(sel_path.read_text(encoding="utf-8")).get("chosen_model", "poisson")
                      if sel_path.exists() else "poisson")

    windows = _group_round_windows()
    all_matches = dataset.load_matches()

    rows_out = []
    for round_n, dmin, dmax in windows:
        train = dataset.training_frame(dmin, lam)
        round_games = all_matches[
            all_matches["is_wc2026_fixture"] & all_matches["played"]
            & (all_matches["date"] >= pd.Timestamp(dmin)) & (all_matches["date"] <= pd.Timestamp(dmax))
        ]

        model = reg.build_member(model_name, cfg) if model_name != "ensemble" \
            else reg.build_ensemble(cfg=cfg)
        model.fit(train)
        score_model = model if getattr(model, "supports_scoreline", False) else \
            reg.build_member("dixon_coles", cfg).fit(train)

        for _, r in round_games.iterrows():
            home, away = str(r["home_team"]), str(r["away_team"])
            neutral = bool(r["neutral"])
            p = model.predict_proba(home, away, neutral)
            ts = score_model.top_scores(home, away, n=1, neutral=neutral)
            pred_score = _parse_score(ts[0][0]) if ts else None
            rows_out.append({
                "round": round_n, "home": disp.get(home, home), "away": disp.get(away, away),
                "confidence": float(np.max(p)), "correct": bool(np.argmax(p) == _RES_IDX[r["result"]]),
                "exact": bool(pred_score == (int(r["home_score"]), int(r["away_score"])))
                         if pred_score else False,
            })

    df = pd.DataFrame(rows_out)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (df["confidence"] >= lo) & (df["confidence"] < hi if i < n_bins - 1 else df["confidence"] <= hi)
        sub = df[mask]
        n = len(sub)
        bins.append({
            "range": f"{int(lo*100)}-{int(hi*100)}%", "n": n,
            "avg_confidence": float(sub["confidence"].mean()) if n else None,
            "acertos": int(sub["correct"].sum()), "acerto_rate": float(sub["correct"].mean()) if n else None,
            "erros": int((~sub["correct"]).sum()), "erro_rate": float((~sub["correct"]).mean()) if n else None,
            "exatos": int(sub["exact"].sum()), "exato_rate": float(sub["exact"].mean()) if n else None,
        })

    if verbose:
        print(f"  modelo avaliado: {model_name}  ({len(df)} jogos, rodadas {[w[0] for w in windows]}, "
              f"walk-forward sem vazamento)\n")
        print(f"  {'faixa conf.':>12s} {'n':>4s} {'conf. média':>11s} {'acerto vitória':>15s} "
              f"{'erro vitória':>13s} {'placar cravado':>15s}")
        for b in bins:
            if b["n"] == 0:
                print(f"  {b['range']:>12s} {0:>4d}            —               —              —               —")
                continue
            print(f"  {b['range']:>12s} {b['n']:>4d}        {b['avg_confidence']:>5.1%}        "
                  f"{b['acerto_rate']:>5.1%} ({b['acertos']}/{b['n']})      "
                  f"{b['erro_rate']:>5.1%} ({b['erros']}/{b['n']})     "
                  f"{b['exato_rate']:>5.1%} ({b['exatos']}/{b['n']})")

    return {"model": model_name, "n": len(df), "bins": bins, "df": df}


if __name__ == "__main__":
    run()
