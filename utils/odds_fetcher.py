"""Busca de odds de mercado (The Odds API) + de-vig + persistência em `odds_2026`.

Odds NUNCA entram nos modelos -- o modelo é 100% independente. Este módulo só busca
odds h2h, calcula probabilidade implícita sem vig (penaltyblog.implied) e compara com a
previsão do modelo escolhido (tabela `predictions`), gravando o snapshot em `odds_2026`
(1 linha por jogo, upsert) para exibição no frontend (régua) e alerta de divergência.

Compartilhado com `4_validation/odds_crosscheck.py` (fetch + agregação h2h + nome do
modelo escolhido), que segue escrevendo em `predictions` (odds_home/market_prob_*).

Requer ODDS_API_KEY no ambiente e Supabase configurado. Sem ODDS_API_KEY ou sem
Supabase: no-op.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from penaltyblog import implied

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_loader import load_config, path  # noqa: E402
import db_client  # noqa: E402

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def load_chosen_model_name() -> str:
    """Lê selected_model.json; se não existir, cai no padrão dixon_coles."""
    sel_path = path("5_outputs", "selected_model.json")
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        return sel.get("chosen_model", "dixon_coles")
    return "dixon_coles"


def fetch_market_odds(cfg) -> tuple[list[dict] | None, int | None]:
    """Busca odds h2h na The Odds API. Retorna (jogos, créditos_restantes), ou
    (None, None) se ODDS_API_KEY não estiver configurada."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("  ODDS_API_KEY ausente — pulando busca de odds.")
        return None, None
    odds_cfg = cfg["odds_api"]
    url = f"{ODDS_API_BASE}/sports/{odds_cfg['sport_key']}/odds/"
    resp = requests.get(url, params={
        "apiKey": key,
        "regions": odds_cfg["regions"],
        "markets": odds_cfg["markets"],
        "oddsFormat": "decimal",
    }, timeout=30)
    resp.raise_for_status()
    credits_remaining = resp.headers.get("x-requests-remaining")
    return resp.json(), (int(credits_remaining) if credits_remaining is not None else None)


def aggregate_h2h(game: dict) -> tuple[float, float, float] | None:
    """Mediana das odds h2h (home, draw, away) entre todos os bookmakers, ou None."""
    home_name, away_name = game["home_team"], game["away_team"]
    odds_h, odds_d, odds_a = [], [], []
    for bk in game.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in mk["outcomes"]}
            if home_name in outcomes and away_name in outcomes and "Draw" in outcomes:
                odds_h.append(outcomes[home_name])
                odds_d.append(outcomes["Draw"])
                odds_a.append(outcomes[away_name])
    if not odds_h:
        return None
    return float(np.median(odds_h)), float(np.median(odds_d)), float(np.median(odds_a))


def fetch_and_store_odds(verbose: bool = True) -> dict:
    """Busca odds de mercado, compara com o modelo escolhido e grava snapshot em
    `odds_2026` (upsert on_conflict=match_id).

    Retorna {"games": int, "alerts": int, "credits_remaining": int | None}.
    """
    cfg = load_config()
    aliases = cfg.get("team_aliases", {})
    devig_method = cfg["odds_api"]["devig_method"]
    threshold = cfg["odds_api"]["divergence_alert_pp"] / 100.0

    market_games, credits = fetch_market_odds(cfg)
    if market_games is None:
        return {"games": 0, "alerts": 0, "credits_remaining": None}

    pred_rows = db_client.fetch_all("predictions")
    if pred_rows is None:
        print("  Supabase não configurado — sem tabela predictions para comparar odds.")
        return {"games": 0, "alerts": 0, "credits_remaining": credits}

    chosen_name = load_chosen_model_name()
    by_pair = {
        (r["home_team"], r["away_team"]): r
        for r in pred_rows if r["model_name"] == chosen_name
    }

    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    n_alerts = 0
    for g in market_games:
        agg = aggregate_h2h(g)
        if agg is None:
            continue
        oh, od, oa = agg
        home = aliases.get(g["home_team"], g["home_team"])
        away = aliases.get(g["away_team"], g["away_team"])

        pred = by_pair.get((home, away))
        swapped = pred is None
        if swapped:
            pred = by_pair.get((away, home))
        if pred is None:
            continue

        market = implied.calculate_implied([oh, od, oa], method=devig_method).probabilities
        m_home, m_draw, m_away = market
        if swapped:
            m_home, m_away = m_away, m_home
            oh, oa = oa, oh

        p_home = pred["prob_home"] or 0.0
        p_draw = pred["prob_draw"] or 0.0
        p_away = pred["prob_away"] or 0.0
        diff_pp = max(abs(p_home - m_home), abs(p_draw - m_draw), abs(p_away - m_away))
        alert = diff_pp > threshold
        if alert:
            n_alerts += 1

        game_id = pred["game_id"]
        rows.append({
            "match_id": game_id,
            "home_team": pred["home_team"], "away_team": pred["away_team"],
            "match_date": datetime.strptime(game_id[-8:], "%Y%m%d").strftime("%Y-%m-%d"),
            "odd_home": round(oh, 3), "odd_draw": round(od, 3), "odd_away": round(oa, 3),
            "implied_home": round(float(m_home), 3),
            "implied_draw": round(float(m_draw), 3),
            "implied_away": round(float(m_away), 3),
            "diff_pp": round(float(diff_pp), 4),
            "divergence_alert": alert,
            "fetched_at": now_iso,
        })
        if verbose and alert:
            print(f"  divergência {diff_pp * 100:.1f}pp em {game_id}: "
                  f"modelo H/D/A={p_home:.2f}/{p_draw:.2f}/{p_away:.2f} "
                  f"mercado H/D/A={m_home:.2f}/{m_draw:.2f}/{m_away:.2f}")

    if rows:
        db_client.upsert("odds_2026", rows, on_conflict="match_id")
    elif verbose:
        print("  nenhum jogo da Copa 2026 com odds h2h disponíveis ainda.")

    return {"games": len(rows), "alerts": n_alerts, "credits_remaining": credits}


if __name__ == "__main__":
    print(fetch_and_store_odds())
