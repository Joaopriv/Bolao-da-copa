"""[Iteração 2 / D1] xG de jogador e de time por temporada (clube) via Understat.

ScraperFC.Understat está QUEBRADO contra o site atual: understat.com não embute mais
`teamsData`/`playersData` em <script> tags (a leitura via BeautifulSoup retorna
IndexError) — os dados agora são carregados via AJAX pelo `js/league.min.js`. Este
módulo chama diretamente os dois endpoints JSON internos usados pelo frontend
(descobertos inspecionando esse arquivo), sem dependências extras:

- POST /main/getPlayersStats/  {league, season} -> stats de jogador na temporada
  (minutos, xG, xA, npxG, xGChain, xGBuildup) -> player_seasons (source='understat').
- GET  /getLeagueData/{league}/{season}        -> histórico jogo-a-jogo de xG/xGA por
  time -> team_seasons (source='understat'), alvo (y) do RAPM-lite (D3).

`cfg['understat']['leagues']` usa os SLUGS de URL do Understat (EPL, La_liga,
Bundesliga, Serie_A, Ligue_1, RFPL); `seasons` é o ano de início (ex. "2024" =
temporada 2024/2025). Rate limit configurável entre requisições.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load_config  # noqa: E402
import db_client  # noqa: E402

BASE = "https://understat.com"


def _headers(league: str, season: str) -> dict:
    return {"X-Requested-With": "XMLHttpRequest", "Referer": f"{BASE}/league/{league}/{season}"}


def _team_season_rows(league: str, season: str) -> list[dict]:
    r = requests.get(f"{BASE}/getLeagueData/{league}/{season}", headers=_headers(league, season), timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = []
    for team in data.get("teams", {}).values():
        history = team.get("history", [])
        if not history:
            continue
        n = len(history)
        rows.append({
            "team": team["title"], "league": league, "season": season,
            "xg90": round(sum(float(h["xG"]) for h in history) / n, 4),
            "xga90": round(sum(float(h["xGA"]) for h in history) / n, 4),
            "games": n, "source": "understat",
        })
    return rows


def _player_rows(league: str, season: str) -> tuple[list[dict], list[dict]]:
    r = requests.post(f"{BASE}/main/getPlayersStats/", data={"league": league, "season": season},
                       headers=_headers(league, season), timeout=30)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("success", True):
        return [], []

    season_rows, players_rows = [], []
    for p in payload.get("players", []):
        minutes = float(p["time"])
        if minutes <= 0:
            continue
        pid = f"understat_{p['id']}"
        season_rows.append({
            "player_id": pid, "team": p["team_title"], "season": season, "competition": league,
            "minutes": minutes, "games": int(p["games"]),
            "xg90": round(float(p["xG"]) / minutes * 90, 4),
            "xa90": round(float(p["xA"]) / minutes * 90, 4),
            "npxg90": round(float(p["npxG"]) / minutes * 90, 4),
            "xgchain90": round(float(p["xGChain"]) / minutes * 90, 4),
            "xgbuildup90": round(float(p["xGBuildup"]) / minutes * 90, 4),
            "source": "understat",
        })
        players_rows.append({"player_id": pid, "name": p["player_name"], "position": p.get("position")})
    return season_rows, players_rows


def scrape_understat_players(verbose: bool = True) -> None:
    cfg = load_config()
    leagues = cfg["understat"]["leagues"]
    seasons = cfg["understat"]["seasons"]
    rate = cfg["understat"].get("rate_limit_seconds", 6)

    team_rows: list[dict] = []
    season_rows: list[dict] = []
    players_rows: dict[str, dict] = {}

    for league in leagues:
        for season in seasons:
            try:
                t_rows = _team_season_rows(league, season)
                time.sleep(rate)
                s_rows, p_rows = _player_rows(league, season)
                time.sleep(rate)
            except Exception as e:
                if verbose:
                    print(f"  {league} {season}: erro ({e}) — pulado")
                continue
            team_rows.extend(t_rows)
            season_rows.extend(s_rows)
            for p in p_rows:
                players_rows[p["player_id"]] = p
            if verbose:
                print(f"  {league} {season}: {len(t_rows)} times, {len(s_rows)} jogadores")

    if players_rows:
        db_client.upsert("players", list(players_rows.values()), on_conflict="player_id")
    if team_rows:
        db_client.upsert("team_seasons", team_rows, on_conflict="team,league,season")
    if season_rows:
        db_client.upsert("player_seasons", season_rows, on_conflict="player_id,team,season,source")

    print(f"  Upsert: {len(players_rows)} players, {len(team_rows)} team_seasons, "
          f"{len(season_rows)} player_seasons (source=understat).")


if __name__ == "__main__":
    scrape_understat_players()
