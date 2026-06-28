"""Gera predictions_2026.json no schema do frontend (bolao-copa-2026.jsx).

Treina o modelo ESCOLHIDO (selected_model.json) em TODOS os jogos jogados até hoje e
prevê os 72 jogos da fase de grupos da Copa 2026 (que já vêm no dataset martj42).

Campo de xG (xg_context) fica null nesta iteração — sem fonte. squad_note_home/away
(Iteração 3 / F6) vêm de cfg.squad_notes (preenchido manualmente antes de cada rodada).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))
from config_loader import load_config, path  # noqa: E402
import dataset  # noqa: E402
import db_client  # noqa: E402
import models_registry as reg  # noqa: E402
import scoring  # noqa: E402


def _load_chosen(cfg):
    """Lê selected_model.json; se não existir, cai no padrão dixon_coles."""
    sel_path = path("5_outputs", "selected_model.json")
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        return sel.get("chosen_model", "dixon_coles"), sel.get("ensemble_weights")
    print("  (selected_model.json ausente — usando dixon_coles como padrão)")
    return "dixon_coles", None


def _result_probs(p):
    return {"home": round(float(p[0]), 3), "draw": round(float(p[1]), 3),
            "away": round(float(p[2]), 3)}


def generate(round_updated: int | None = None, odds_credits: int | None = None) -> Path:
    cfg = load_config()
    lam = cfg["preprocess"]["temporal_decay_lambda"]
    today = cfg["data"]["today"]
    disp = cfg.get("display_names", {})
    squad_notes = cfg.get("squad_notes", {})
    n_top = cfg["predict_2026"]["top_scores_n"]
    ev_max_goals = cfg["predict_2026"]["ev_max_goals"]
    heatmap_max_goals = cfg["predict_2026"]["heatmap_max_goals"]

    chosen_name, weights = _load_chosen(cfg)
    train = dataset.training_frame(today, lam=lam)
    print(f"  treinando '{chosen_name}' em {len(train):,} jogos até {today} ...")

    # Modelo escolhido (para result_probs). Bayesiano em mode="full" — fit único no
    # dataset completo, custo (~1min) compensa a precisão extra.
    if chosen_name == "ensemble":
        chosen = reg.build_ensemble(weights=weights, cfg=cfg, bayesian_mode="full").fit(train)
    else:
        chosen = reg.build_member(chosen_name, cfg, mode="full").fit(train)

    # Provedor de placar: o próprio escolhido se suportar; senão, dixon_coles.
    if getattr(chosen, "supports_scoreline", False):
        score_model = chosen
    else:
        score_model = reg.build_member("dixon_coles", cfg).fit(train)

    # Membros para o model_breakdown (transparência).
    members = {n: reg.build_member(n, cfg, mode="full" if n == "bayesian" else None).fit(train)
               for n in reg.available_members()}

    odds_rows = db_client.fetch_all("odds_2026") or []
    odds_map = {r["match_id"]: r for r in odds_rows}

    # round/group_name vêm de copa_2026_results (fonte autoritativa) -- evita
    # heurística no frontend, que falha no mata-mata (confrontos cruzam grupos).
    results_rows = db_client.fetch_all("copa_2026_results") or []
    stage_map = {r["game_id"]: (r["round"], r["group_name"]) for r in results_rows}

    fixtures = dataset.get_wc2026_fixtures()
    preds = []
    pred_rows = []  # espelha `preds` em formato tabular p/ Supabase (tabela predictions)
    now_iso = datetime.now(timezone.utc).isoformat()
    for _, r in fixtures.iterrows():
        home, away = str(r["home_team"]), str(r["away_team"])
        game_id = f"{home}-{away}-{r['date'].strftime('%Y%m%d')}"
        # Vantagem de anfitrião (México/Canadá/EUA jogando em casa): usa o `neutral`
        # por jogo de `fixtures` em vez do booleano global de config.
        neutral = bool(r["neutral"])
        p = chosen.predict_proba(home, away, neutral=neutral)
        grid = score_model.predict_scoreline(home, away, neutral)
        rp = _result_probs(p)
        top_scores = (scoring.top_scores_by_ev(grid, max_goals=ev_max_goals, n=n_top)
                       if grid is not None else [])
        if grid is not None:
            # [Auditoria M8/P11] mesma truncagem+renormalização usada em
            # top_scores_by_ev -- "prob" de top_scores e células de score_matrix
            # ficam consistentes (mesma convenção, grades de tamanhos diferentes).
            sm_arr = scoring.truncate_and_renormalize(grid, heatmap_max_goals)
            score_matrix = sm_arr.round(4).tolist()
        else:
            score_matrix = None
        breakdown = {
            n: _result_probs(m.predict_proba(home, away, neutral=neutral))
            for n, m in members.items()
        }
        odds_row = odds_map.get(game_id)
        odds_implied = ({"home": odds_row["implied_home"], "draw": odds_row["implied_draw"],
                          "away": odds_row["implied_away"]} if odds_row else None)
        divergence_alert = bool(odds_row["divergence_alert"]) if odds_row else False
        # True se modelo e mercado apontam favoritos DIFERENTES (1X2 discordante),
        # não só magnitude. False se odds indisponíveis.
        if odds_implied:
            outcomes = ("home", "draw", "away")
            model_fav = max(outcomes, key=lambda o: rp[o])
            market_fav = max(outcomes, key=lambda o: odds_implied[o])
            divergence_direction = model_fav != market_fav
        else:
            divergence_direction = False
        stage_round, stage_name = stage_map.get(game_id, (None, None))
        preds.append({
            "game": f"{disp.get(home, home)} vs {disp.get(away, away)}",
            "home_team": disp.get(home, home),
            "away_team": disp.get(away, away),
            "date": r["date"].strftime("%Y-%m-%d"),
            "round": stage_round,
            "stage": stage_name,
            "confidence": round(float(np.max(p)), 3),  # prob do resultado modal
            "top_scores": top_scores,
            "score_matrix": score_matrix,
            "result_probs": rp,
            "model_breakdown": breakdown,
            "xg_context": None,   # Iteração 2 (sem fonte de xG nesta)
            "squad_note_home": squad_notes.get(home),  # Iteração 3 (F6, cfg.squad_notes)
            "squad_note_away": squad_notes.get(away),
            "odds_implied": odds_implied,
            "divergence_alert": divergence_alert,
            "divergence_direction": divergence_direction,
        })

        # Uma linha por membro do breakdown; se o modelo escolhido for um deles, a MESMA
        # linha carrega confidence/top_scores (evita duas linhas (game_id, model_name)
        # iguais no upsert, que o Postgres rejeita).
        for mname, mp in breakdown.items():
            row = {
                "game_id": game_id, "home_team": home, "away_team": away,
                "model_name": mname,
                "prob_home": mp["home"], "prob_draw": mp["draw"], "prob_away": mp["away"],
                "confidence": None, "top_scores": None,
                "generated_at": now_iso,
            }
            if mname == chosen_name:
                row["confidence"] = round(float(np.max(p)), 3)
                row["top_scores"] = top_scores
            pred_rows.append(row)
        if chosen_name not in breakdown:
            pred_rows.append({
                "game_id": game_id, "home_team": home, "away_team": away,
                "model_name": chosen_name,
                "prob_home": rp["home"], "prob_draw": rp["draw"], "prob_away": rp["away"],
                "confidence": round(float(np.max(p)), 3), "top_scores": top_scores,
                "generated_at": now_iso,
            })

    out = {
        "meta": {
            "model": chosen_name,
            "trained_until": today,
            "lambda": lam,
            "iteration": 1,
            "notes": "Modelagem sobre gols (martj42). xG/elenco/sequencial: Iteração 2.",
            "round_updated": round_updated,
            "odds_api_credits_remaining": odds_credits,
            "model_confidence": round(float(np.mean([p["confidence"] for p in preds])), 3),
        },
        "predictions": preds,
    }
    dest = path("5_outputs", "predictions_2026.json")
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {len(preds)} previsões -> {dest}")

    if db_client.upsert("predictions", pred_rows, on_conflict="game_id,model_name"):
        print(f"  {len(pred_rows)} linhas -> Supabase (tabela predictions)")
    return dest


if __name__ == "__main__":
    generate()
