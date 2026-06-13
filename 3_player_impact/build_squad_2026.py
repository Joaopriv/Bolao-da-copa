"""[Iteração 2 / D2] Convocação aproximada da Copa 2026 (`squad_2026`).

Para cada uma das 48 seleções (`dataset.get_wc2026_fixtures()`, união de
home_team/away_team): pega os jogadores em `player_national_team` daquela seleção,
junta com `player_seasons` (source='understat', temporada mais recente disponível por
jogador) para minutos de clube e com `players.market_value` (Transfermarkt).

Ranking: jogadores com minutos >= `cfg.squad_2026_build.min_minutes_for_squad` (i.e.
"jogou de fato na temporada mais recente") são ordenados por market_value desc (depois
minutos desc); o resto (minutos abaixo do piso -- provável lesão/aposentadoria, com
market_value possivelmente desatualizado) entra só para completar o roster, também por
market_value desc. Top `cfg.squad_2026_build.roster_size` (26) vira o elenco aproximado.

Por que market_value e não só minutos: estrelas de clubes grandes (PSG, Real Madrid,
Man City) são rotacionadas entre competições e por isso acumulam MENOS minutos numa
única temporada do que "iron men" de clubes de meio de tabela -- ordenar só por minutos
elegia os segundos e excluía as primeiras (ex. Mbappé/Neymar/Vinícius Jr. ficavam fora
do top-26 de França/Brasil, distorcendo squad_strength/D5).

Aproximação reconhecida: a convocação real só é anunciada ~1 semana antes do torneio;
isto usa minutagem+valor de mercado de clube 2022-2024 como proxy de "jogador relevante
para a seleção". Sem dado sintético: seleções sem jogadores mapeados em
`player_national_team` ficam sem linhas em `squad_2026` (squad_strength tratará via
data_coverage=0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "1_data"))
from config_loader import load_config  # noqa: E402
import db_client  # noqa: E402
from dataset import get_wc2026_fixtures  # noqa: E402


def build_probable_squads(verbose: bool = True) -> list[dict]:
    cfg = load_config()
    roster_size = cfg["squad_2026_build"]["roster_size"]
    min_minutes = cfg["squad_2026_build"]["min_minutes_for_squad"]

    fixtures = get_wc2026_fixtures()
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))

    pnt = db_client.fetch_all("player_national_team") or []
    seasons = db_client.fetch_all("player_seasons") or []
    understat = [r for r in seasons if r["source"] == "understat"]
    players = db_client.fetch_all("players") or []
    market_value = {r["player_id"]: r.get("market_value") for r in players}

    # temporada mais recente por jogador (Understat)
    latest: dict[str, dict] = {}
    for r in understat:
        pid = r["player_id"]
        if pid not in latest or r["season"] > latest[pid]["season"]:
            latest[pid] = r

    pnt_df = pd.DataFrame(pnt)
    if pnt_df.empty:
        print("  player_national_team vazio — rode --match-players antes.")
        return []

    rows: list[dict] = []
    coverage = []
    for team in teams:
        candidates = pnt_df[pnt_df["national_team"] == team]
        scored = []
        for _, c in candidates.iterrows():
            pid = c["player_id"]
            ls = latest.get(pid)
            minutes = (ls["minutes"] if ls else 0.0) or 0.0
            mv = market_value.get(pid) or 0.0
            scored.append({"player_id": pid, "position": c["position"], "minutes": minutes, "market_value": mv})
        active = [r for r in scored if r["minutes"] >= min_minutes]
        inactive = [r for r in scored if r["minutes"] < min_minutes]
        active.sort(key=lambda r: (r["market_value"], r["minutes"]), reverse=True)
        inactive.sort(key=lambda r: (r["market_value"], r["minutes"]), reverse=True)
        top = (active + inactive)[:roster_size]
        coverage.append((team, len(candidates), len(top)))
        for r in top:
            rows.append({
                "team": team, "player_id": r["player_id"], "available": True,
                "position": r["position"],
            })

    # reconstrói squad_2026 do zero (sem isso, jogadores que saíram do top-N de uma
    # seleção numa rodada anterior de --match-players ficariam como linhas órfãs).
    db_client.delete_all("squad_2026", pk_col="team")
    if rows:
        db_client.upsert("squad_2026", rows, on_conflict="team,player_id")

    low = [t for t, n, k in coverage if k < roster_size]
    if verbose:
        print(f"  {len(teams)} seleções | {len(rows)} linhas squad_2026 "
              f"(top {roster_size} por valor de mercado, minutos >= {min_minutes})")
        if low:
            print(f"  seleções com elenco incompleto (<{roster_size}): "
                  + ", ".join(f"{t}({k})" for t, n, k in coverage if k < roster_size))
    return rows


if __name__ == "__main__":
    build_probable_squads()
