"""Cruzamento com odds de mercado (VALIDAÇÃO apenas) — The Odds API.

Odds NUNCA entram nos modelos — o modelo é 100% independente. Aqui só comparamos:
- prob do modelo escolhido (tabela `predictions`) vs prob de mercado sem vig
  (penaltyblog.implied), jogo a jogo;
- convergência = boa calibração; divergência > `odds_api.divergence_alert_pp`
  (qualquer outcome) é alertada no console.
O mercado é régua, não professor.

Requer ODDS_API_KEY no ambiente e Supabase configurado (lê/escreve a tabela `predictions`).
Sem ODDS_API_KEY ou sem Supabase: no-op (retorna lista vazia).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from penaltyblog import implied

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))
from config_loader import load_config  # noqa: E402
import db_client  # noqa: E402
from odds_fetcher import (  # noqa: E402
    aggregate_h2h as _aggregate_h2h,
    fetch_market_odds as _fetch_market_odds_with_credits,
    load_chosen_model_name as _load_chosen_model_name,
)


def _fetch_market_odds(cfg) -> list[dict] | None:
    games, _credits = _fetch_market_odds_with_credits(cfg)
    return games


def crosscheck_odds(verbose: bool = True) -> list[dict]:
    """Compara probabilidades do modelo escolhido com o mercado (sem vig) jogo a jogo.

    Atualiza odds_home/draw/away e market_prob_home/draw/away na tabela `predictions`
    (linha do modelo escolhido). Retorna a lista de linhas atualizadas.
    """
    cfg = load_config()
    aliases = cfg.get("team_aliases", {})
    devig_method = cfg["odds_api"]["devig_method"]
    threshold = cfg["odds_api"]["divergence_alert_pp"] / 100.0

    market_games = _fetch_market_odds(cfg)
    if market_games is None:
        return []

    pred_rows = db_client.fetch_all("predictions")
    if pred_rows is None:
        print("  Supabase não configurado — sem tabela predictions para crosscheck.")
        return []

    chosen_name = _load_chosen_model_name()
    by_pair = {
        (r["home_team"], r["away_team"]): r
        for r in pred_rows if r["model_name"] == chosen_name
    }

    now_iso = datetime.now(timezone.utc).isoformat()
    updates = []
    for g in market_games:
        agg = _aggregate_h2h(g)
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

        updates.append({
            "game_id": pred["game_id"], "home_team": pred["home_team"],
            "away_team": pred["away_team"], "model_name": chosen_name,
            "odds_home": round(oh, 3), "odds_draw": round(od, 3), "odds_away": round(oa, 3),
            "market_prob_home": round(float(m_home), 3),
            "market_prob_draw": round(float(m_draw), 3),
            "market_prob_away": round(float(m_away), 3),
            "generated_at": now_iso,
        })
        if verbose and diff_pp > threshold:
            print(f"  divergência {diff_pp * 100:.1f}pp em {pred['game_id']}: "
                  f"modelo H/D/A={p_home:.2f}/{p_draw:.2f}/{p_away:.2f} "
                  f"mercado H/D/A={m_home:.2f}/{m_draw:.2f}/{m_away:.2f}")

    if updates:
        db_client.upsert("predictions", updates, on_conflict="game_id,model_name")
        if verbose:
            print(f"  {len(updates)} jogos cruzados com odds (modelo '{chosen_name}') "
                  f"-> Supabase (predictions)")
    elif verbose:
        print("  nenhum jogo da Copa 2026 com odds h2h disponíveis ainda.")
    return updates


if __name__ == "__main__":
    crosscheck_odds()
