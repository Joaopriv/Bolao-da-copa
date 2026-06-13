"""[Iteração 2 / Prompt C2] Scraper de xG por jogo via StatsBomb open data.

Fonte: `statsbombpy` (StatsBomb open data, sem autenticação). Para cada
competição/temporada listada em `cfg["xg_scraping"]["competitions"]`, soma
`shot_statsbomb_xg` por time em cada partida (`sb.events`) e casa o resultado
com uma linha de `matches` por (tournament, home_team, away_team) + data
(tolerância ±1 dia, para diferenças de fuso horário).

Idempotente: pula jogos cujo `home_xg` já está preenchido. Ao final faz
upsert em `matches` (on_conflict=date,home_team,away_team,tournament) só com
{date, home_team, away_team, tournament, home_xg, away_xg}.

Sem dados sintéticos: jogos não casados ficam com home_xg/away_xg=NULL
(o m3_dixon_coles_xg simplesmente ignora esses jogos no treino).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from statsbombpy import sb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load_config  # noqa: E402
import db_client  # noqa: E402


def _alias_map(cfg: dict) -> dict:
    aliases = dict(cfg.get("team_aliases", {}))
    aliases.update(cfg.get("statsbomb_team_aliases", {}))
    return aliases


def _match_xg(match_id: int) -> pd.Series | None:
    """Soma shot_statsbomb_xg por time para uma partida. None se não houver xG."""
    events = sb.events(match_id=int(match_id))
    shots = events[events["type"] == "Shot"]
    if "shot_statsbomb_xg" not in shots.columns or shots["shot_statsbomb_xg"].dropna().empty:
        return None
    return shots.groupby("team")["shot_statsbomb_xg"].sum()


def scrape_xg(verbose: bool = True) -> None:
    cfg = load_config()
    aliases = _alias_map(cfg)
    competitions = cfg.get("xg_scraping", {}).get("competitions", [])

    rows = db_client.fetch_all("matches")
    if not rows:
        print("  Supabase não configurado — sem tabela matches para atualizar.")
        return
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    found = 0
    not_found = 0
    updates = []

    for comp in competitions:
        comp_id, season_id, tlabel = comp["competition_id"], comp["season_id"], comp["tournament_label"]
        sb_matches = sb.matches(competition_id=comp_id, season_id=season_id)

        # Sonda a primeira partida: se a competição não tem shot_statsbomb_xg, pula inteira.
        probe_xg = _match_xg(sb_matches.iloc[0]["match_id"])
        if probe_xg is None:
            if verbose:
                print(f"  {tlabel} (competition_id={comp_id}, season_id={season_id}): "
                      f"sem shot_statsbomb_xg — competição pulada.")
            continue

        window = df[df["tournament"] == tlabel]
        for i, m in sb_matches.iterrows():
            sb_home = aliases.get(m["home_team"], m["home_team"])
            sb_away = aliases.get(m["away_team"], m["away_team"])
            mdate = pd.Timestamp(m["match_date"])

            cand = window[
                (window["date"] >= mdate - pd.Timedelta(days=1))
                & (window["date"] <= mdate + pd.Timedelta(days=1))
            ]
            same = cand[(cand["home_team"] == sb_home) & (cand["away_team"] == sb_away)]
            swapped = same.empty
            if swapped:
                same = cand[(cand["home_team"] == sb_away) & (cand["away_team"] == sb_home)]
            if same.empty:
                not_found += 1
                continue

            row = same.iloc[0]
            if pd.notna(row["home_xg"]):
                continue  # já preenchido — idempotente

            xg = probe_xg if i == 0 else _match_xg(m["match_id"])
            if xg is None:
                not_found += 1
                continue

            home_xg = float(xg.get(m["home_team"], 0.0))
            away_xg = float(xg.get(m["away_team"], 0.0))
            if swapped:
                home_xg, away_xg = away_xg, home_xg

            updates.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "tournament": row["tournament"],
                "home_xg": round(home_xg, 4),
                "away_xg": round(away_xg, 4),
            })
            found += 1

    if updates:
        db_client.upsert("matches", updates, on_conflict="date,home_team,away_team,tournament")

    print(f"  xG atualizado: {found} jogos. Não encontrados: {not_found} jogos.")
    if found > 0 and not_found > 0.10 * found:
        print(f"  AVISO: {not_found} não encontrados é mais de 10% de {found} — "
              f"verificar matching de nomes de times (team_aliases/statsbomb_team_aliases).")


if __name__ == "__main__":
    scrape_xg()
