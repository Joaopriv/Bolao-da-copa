"""[Iteração 2 / D2] Matching jogador de clube (Understat) -> seleção nacional (Transfermarkt).

Para cada jogador distinto em `player_seasons` (source='understat'), busca o melhor
candidato em `players.csv` (Transfermarkt, via `scraper_transfermarkt.download_player_scores`)
por similaridade de nome (rapidfuzz.fuzz.WRatio, score_cutoff=cfg.transfermarkt.fuzzy_match_threshold).

`cfg.transfermarkt.name_aliases` cobre casos pontuais onde o nome Understat não bate
mesmo após normalização (ex. "Kylian Mbappe-Lottin" vs "Kylian Mbappé", score=78.8 <
threshold) -- aplicado só na query de busca, não altera `players.name` armazenado.

País de nacionalidade do Transfermarkt (`country_of_citizenship`) é mapeado ao nome
canônico martj42 via `team_aliases` + `transfermarkt_team_aliases`. O pool de candidatos
é restrito a:
- jogadores ativos na janela do Understat (`last_season >= cfg.transfermarkt.min_last_season`)
- nacionalidade canônica entre as 48 seleções da Copa 2026 (`display_names.keys()`) —
  jogadores de outras seleções não entram em `squad_2026` (D2/build_squad_2026) de
  qualquer forma.

Saída:
- `player_national_team` (player_id, national_team, position, transfermarkt_id) —
  reconstruída do zero a cada execução (sem isso, matches de uma rodada anterior
  com seleção diferente/sem match deixariam linhas órfãs).
- `players` (upsert: nationality=seleção canônica, position, market_value; jogadores
  sem match >= threshold recebem nationality/position/market_value=None).
- `1_data/processed/unmatched_players.csv` — jogadores Understat sem match >= threshold
  (revisão manual; sem dado sintético).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_player_impact"))
from config_loader import load_config, path  # noqa: E402
import db_client  # noqa: E402
from sb_match import normalize_name  # noqa: E402
from scrapers.scraper_transfermarkt import download_player_scores  # noqa: E402


def _country_alias_map(cfg: dict) -> dict:
    aliases = dict(cfg.get("team_aliases", {}))
    aliases.update(cfg.get("transfermarkt_team_aliases", {}))
    return aliases


def _load_tm_candidates(cfg: dict) -> pd.DataFrame:
    base = download_player_scores(verbose=True)
    cols = ["player_id", "name", "country_of_citizenship", "position", "sub_position",
            "last_season", "market_value_in_eur"]
    df = pd.read_csv(base / "players.csv", usecols=cols)

    min_season = cfg["transfermarkt"]["min_last_season"]
    df = df[df["last_season"] >= min_season].copy()

    aliases = _country_alias_map(cfg)
    canonical = set(cfg["display_names"].keys())
    df["national_team"] = df["country_of_citizenship"].map(lambda c: aliases.get(c, c))
    df = df[df["national_team"].isin(canonical)].copy()
    df["position"] = df["sub_position"].fillna(df["position"])
    return df.reset_index(drop=True)


def _understat_players(cfg: dict) -> pd.DataFrame:
    seasons = db_client.fetch_all("player_seasons") or []
    understat_ids = sorted({r["player_id"] for r in seasons if r["source"] == "understat"})

    players = db_client.fetch_all("players") or []
    by_id = {r["player_id"]: r for r in players}

    rows = []
    for pid in understat_ids:
        info = by_id.get(pid)
        if info is None or not info.get("name"):
            continue
        rows.append({"player_id": pid, "name": info["name"]})
    return pd.DataFrame(rows)


def match_understat_to_transfermarkt(verbose: bool = True) -> dict:
    cfg = load_config()
    threshold = cfg["transfermarkt"]["fuzzy_match_threshold"]

    understat = _understat_players(cfg)
    tm = _load_tm_candidates(cfg)
    if verbose:
        print(f"  {len(understat)} jogadores Understat distintos | "
              f"{len(tm)} candidatos Transfermarkt (seleções Copa 2026, "
              f"last_season >= {cfg['transfermarkt']['min_last_season']})")

    if understat.empty or tm.empty:
        return {"player_national_team": [], "players": [], "unmatched": []}

    name_aliases = cfg.get("transfermarkt", {}).get("name_aliases", {})
    queries = [normalize_name(name_aliases.get(n, n)) for n in understat["name"]]
    choices = [normalize_name(n) for n in tm["name"]]
    scores = process.cdist(queries, choices, scorer=fuzz.token_sort_ratio, score_cutoff=threshold, workers=-1)
    best_idx = scores.argmax(axis=1)
    best_score = scores.max(axis=1)

    pnt_rows, players_rows, unmatched = [], [], []
    for i, (_, u) in enumerate(understat.iterrows()):
        if best_score[i] < threshold:
            unmatched.append({"player_id": u["player_id"], "name": u["name"]})
            players_rows.append({
                "player_id": u["player_id"], "name": u["name"],
                "nationality": None, "position": None, "market_value": None,
            })
            continue
        m = tm.iloc[best_idx[i]]
        market_value = m["market_value_in_eur"]
        pnt_rows.append({
            "player_id": u["player_id"], "national_team": m["national_team"],
            "position": m["position"], "transfermarkt_id": str(int(m["player_id"])),
        })
        players_rows.append({
            "player_id": u["player_id"], "name": u["name"], "nationality": m["national_team"],
            "position": m["position"],
            "market_value": float(market_value) if pd.notna(market_value) else None,
        })

    # reconstrói player_national_team do zero (sem isso, jogadores que antes batiam
    # com uma seleção errada e agora ficam sem match/com outra seleção deixariam
    # linhas órfãs do matching anterior).
    db_client.delete_all("player_national_team", pk_col="player_id")
    if pnt_rows:
        db_client.upsert("player_national_team", pnt_rows, on_conflict="player_id,national_team")
    if players_rows:
        db_client.upsert("players", players_rows, on_conflict="player_id")

    if unmatched:
        out = path("1_data", "processed", "unmatched_players.csv")
        pd.DataFrame(unmatched).to_csv(out, index=False)

    rate = len(pnt_rows) / len(understat) * 100 if len(understat) else 0.0
    print(f"  Match: {len(pnt_rows)}/{len(understat)} ({rate:.1f}%) | "
          f"sem match: {len(unmatched)} -> 1_data/processed/unmatched_players.csv")
    return {"player_national_team": pnt_rows, "players": players_rows, "unmatched": unmatched}


if __name__ == "__main__":
    match_understat_to_transfermarkt()
