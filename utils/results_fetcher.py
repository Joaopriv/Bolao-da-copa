"""[Iteração 2 / --fetch-results] Busca automática de placares reais da Copa 2026.

Tenta, em ordem, três fontes públicas de resultados:
  FONTE 1 -- API-Football v3 (api-sports.io), requer APIFOOTBALL_KEY no ambiente.
  FONTE 2 -- The Odds API /scores (já usada por odds_fetcher.py), requer ODDS_API_KEY.
  FONTE 3 -- scraping estático da página de scores/fixtures da FIFA.com.
Cada fonte ausente/falha cai para a próxima (no-op silencioso, sem erro).

Os jogos retornados (completed=True) são normalizados via config.team_aliases +
fuzzy match (difflib, threshold 0.80) para o nome canônico (martj42) e casados
contra `copa_2026_results` (home_score IS NULL). Os candidatos confirmados são
inseridos via `sequential_backtest.insert_result` (reaproveita upsert + guard de
round -- E1).

Resultados são fatos públicos, não input do modelo: este módulo não toca em
scoring.py nem nos modelos.
"""
from __future__ import annotations

import difflib
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "4_validation"))

from config_loader import load_config  # noqa: E402
import dataset  # noqa: E402
import db_client  # noqa: E402

APIFOOTBALL_BASE = "https://v3.football.api-sports.io"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
FIFA_SCORES_URL = ("https://www.fifa.com/en/tournaments/mens/worldcup/"
                    "canadamexicousa2026/scores-fixtures")
FUZZY_THRESHOLD = 0.80


def _normalize_team(raw: str, aliases: dict, canonical: set[str]) -> tuple[str | None, bool]:
    """Resolve `raw` (nome bruto de uma fonte externa) para o nome canônico
    (martj42) usado em copa_2026_results. Retorna (nome | None, usou_fuzzy)."""
    if raw in canonical:
        return raw, False
    aliased = aliases.get(raw)
    if aliased in canonical:
        return aliased, False
    matches = difflib.get_close_matches(raw, canonical, n=1, cutoff=FUZZY_THRESHOLD)
    if matches:
        return matches[0], True
    return None, False


def _fetch_apifootball(verbose: bool = True) -> list[dict] | None:
    """FONTE 1 -- API-Football v3. None se APIFOOTBALL_KEY ausente ou erro."""
    key = os.environ.get("APIFOOTBALL_KEY")
    if not key:
        if verbose:
            print("  FONTE 1 (API-Football): APIFOOTBALL_KEY ausente -- pulando.")
        return None
    try:
        resp = requests.get(
            f"{APIFOOTBALL_BASE}/fixtures",
            headers={"x-rapidapi-key": key},
            params={"league": 1, "season": 2026},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        if verbose:
            print(f"  FONTE 1 (API-Football) falhou ({e}) -- tentando próxima fonte.")
        return None

    games = []
    for item in data.get("response", []):
        fx, teams, goals = item["fixture"], item["teams"], item["goals"]
        games.append({
            "home_raw": teams["home"]["name"], "away_raw": teams["away"]["name"],
            "home_score": goals["home"], "away_score": goals["away"],
            "completed": fx["status"]["short"] == "FT",
            "date": fx["date"][:10], "source": "API-Football",
        })
    if verbose:
        print(f"  FONTE 1 (API-Football): {len(games)} jogos retornados.")
    return games


def _fetch_oddsapi(cfg: dict, verbose: bool = True) -> list[dict] | None:
    """FONTE 2 -- The Odds API /scores. None se ODDS_API_KEY ausente ou erro."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        if verbose:
            print("  FONTE 2 (The Odds API): ODDS_API_KEY ausente -- pulando.")
        return None
    url = f"{ODDS_API_BASE}/sports/{cfg['odds_api']['sport_key']}/scores/"
    try:
        resp = requests.get(url, params={"apiKey": key, "daysFrom": 3}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        if verbose:
            print(f"  FONTE 2 (The Odds API) falhou ({e}) -- tentando próxima fonte.")
        return None

    games = []
    for g in data:
        scores = {s["name"]: s["score"] for s in (g.get("scores") or [])}
        h = scores.get(g["home_team"])
        a = scores.get(g["away_team"])
        games.append({
            "home_raw": g["home_team"], "away_raw": g["away_team"],
            "home_score": int(h) if h is not None else None,
            "away_score": int(a) if a is not None else None,
            "completed": bool(g.get("completed")) and h is not None and a is not None,
            "date": g["commence_time"][:10], "source": "The Odds API",
        })
    if verbose:
        credits = resp.headers.get("x-requests-remaining")
        print(f"  FONTE 2 (The Odds API): {len(games)} jogos retornados "
              f"(créditos restantes: {credits}).")
    return games


def _fetch_fifa_scrape(verbose: bool = True) -> list[dict] | None:
    """FONTE 3 -- scraping estático da FIFA.com. None se a página não carregar."""
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(FIFA_SCORES_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        if verbose:
            print(f"  FONTE 3 (FIFA.com scraping) falhou ({e}).")
        return None
    if verbose:
        print("  FONTE 3 (FIFA.com scraping): página estática não expõe placares "
              "(conteúdo renderizado via JS) -- 0 jogos.")
    return []


def fetch_and_insert_results(verbose: bool = True) -> dict:
    """Busca placares terminados (FONTE 1 -> 2 -> 3), casa com `copa_2026_results`
    (home_score IS NULL) e insere via `insert_result`. Retorna
    {"inserted": [...], "pending": [...], "unmatched": [...]}."""
    cfg = load_config()
    aliases = cfg.get("team_aliases", {})
    fixtures = dataset.get_wc2026_fixtures()
    canonical = set(fixtures["home_team"]) | set(fixtures["away_team"])

    db_rows = db_client.fetch_all("copa_2026_results") or []
    pending_pairs: dict[tuple[str, str], dict] = {}
    done_pairs: set[tuple[str, str]] = set()
    for r in db_rows:
        pair = (r["home_team"], r["away_team"])
        if r.get("home_score") is None:
            pending_pairs[pair] = r
        else:
            done_pairs.add(pair)

    games = _fetch_apifootball(verbose=verbose)
    if games is None:
        games = _fetch_oddsapi(cfg, verbose=verbose)
    if games is None:
        games = _fetch_fifa_scrape(verbose=verbose)
    if games is None:
        games = []

    from sequential_backtest import insert_result

    inserted: list[tuple[str, str, int, int]] = []
    unmatched: list[str] = []
    matched_ids: set[str] = set()

    for g in games:
        if not g["completed"]:
            continue
        home_norm, home_fuzzy = _normalize_team(g["home_raw"], aliases, canonical)
        away_norm, away_fuzzy = _normalize_team(g["away_raw"], aliases, canonical)
        if home_norm is None or away_norm is None:
            unmatched.append(f"{g['home_raw']} vs {g['away_raw']} ({g['source']})")
            continue
        if home_fuzzy or away_fuzzy:
            print(f"  (fuzzy match >= {FUZZY_THRESHOLD}) '{g['home_raw']}' -> '{home_norm}', "
                  f"'{g['away_raw']}' -> '{away_norm}' -- revisar team_aliases.")

        row = pending_pairs.get((home_norm, away_norm))
        swapped = False
        if row is None:
            row = pending_pairs.get((away_norm, home_norm))
            swapped = row is not None
        if row is None:
            # já inserido anteriormente (idempotência) -- pular silenciosamente.
            if (home_norm, away_norm) in done_pairs or (away_norm, home_norm) in done_pairs:
                continue
            unmatched.append(f"{g['home_raw']} vs {g['away_raw']} ({g['source']})")
            continue

        h_score, a_score = g["home_score"], g["away_score"]
        if swapped:
            h_score, a_score = a_score, h_score

        game_id = insert_result(row["home_team"], row["away_team"], h_score, a_score,
                                 row["date"], verbose=verbose)
        if game_id:
            inserted.append((row["home_team"], row["away_team"], h_score, a_score))
            matched_ids.add(row["game_id"])

    still_pending = [r for r in pending_pairs.values() if r["game_id"] not in matched_ids]

    if verbose:
        print(f"\n✅ Inseridos: {len(inserted)} jogos" + (":" if inserted else ""))
        for h, a, hs, a_s in inserted:
            print(f"    {h} {hs}×{a_s} {a}")
        print(f"⏳ Pendentes (não terminados): {len(still_pending)} jogos" +
              (":" if still_pending else ""))
        for r in still_pending:
            print(f"    {r['home_team']} vs {r['away_team']} ({r['date']})")
        print(f"⚠ Sem match: {len(unmatched)} jogos" + (":" if unmatched else ""))
        for u in unmatched:
            print(f"    {u}")

    return {"inserted": inserted, "pending": still_pending, "unmatched": unmatched}


if __name__ == "__main__":
    fetch_and_insert_results()
