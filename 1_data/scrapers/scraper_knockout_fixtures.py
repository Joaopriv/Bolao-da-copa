"""Fixtures do mata-mata da Copa 2026 -> tabelas `matches` (placar NaN, para o modelo
prever) e `copa_2026_results` (round/group_name, para --insert-result/--update-round).

Diferente da fase de grupos (scraper_fixtures.py), o confronto do mata-mata só fica
conhecido depois que a fase de grupos termina (1º/2º de cada grupo + 8 melhores 3os
colocados, conforme classificação oficial). Por isso cada rodada do mata-mata precisa
de uma lista de confrontos curada manualmente (fonte: FIFA/Wikipedia, conferida contra
os 72 resultados reais já gravados em copa_2026_results) e adicionada aqui quando a
rodada anterior termina.

`round`: 4=Round of 32, 5=Round of 16, 6=Quartas, 7=Semis, 8=Final (mesmo esquema de
`round` usado pela fase de grupos para --update-round N).
`group_name`: rótulo do estágio (ex.: "Round of 32"), em vez do grupo A-L.
`neutral`: False só quando o anfitrião (México/Canadá/EUA) joga no próprio país.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import db_client  # noqa: E402

# Round of 32 (round=4) -- confrontos oficiais confirmados após o fim da fase de
# grupos (12 campeões + 12 vices + 8 melhores terceiros), conferidos em 2026-06-28.
ROUND_OF_32: list[dict] = [
    {"home": "South Africa", "away": "Canada", "date": "2026-06-28", "neutral": True},
    {"home": "Brazil", "away": "Japan", "date": "2026-06-29", "neutral": True},
    {"home": "Germany", "away": "Paraguay", "date": "2026-06-29", "neutral": True},
    {"home": "Netherlands", "away": "Morocco", "date": "2026-06-29", "neutral": True},
    {"home": "Ivory Coast", "away": "Norway", "date": "2026-06-30", "neutral": True},
    {"home": "France", "away": "Sweden", "date": "2026-06-30", "neutral": True},
    {"home": "Mexico", "away": "Ecuador", "date": "2026-06-30", "neutral": False},
    {"home": "England", "away": "DR Congo", "date": "2026-07-01", "neutral": True},
    {"home": "Belgium", "away": "Senegal", "date": "2026-07-01", "neutral": True},
    {"home": "United States", "away": "Bosnia and Herzegovina", "date": "2026-07-01", "neutral": False},
    {"home": "Spain", "away": "Austria", "date": "2026-07-02", "neutral": True},
    {"home": "Portugal", "away": "Croatia", "date": "2026-07-02", "neutral": True},
    {"home": "Switzerland", "away": "Algeria", "date": "2026-07-02", "neutral": True},
    {"home": "Australia", "away": "Egypt", "date": "2026-07-03", "neutral": True},
    {"home": "Argentina", "away": "Cape Verde", "date": "2026-07-03", "neutral": True},
    {"home": "Colombia", "away": "Ghana", "date": "2026-07-03", "neutral": True},
]


def scrape_knockout_fixtures(fixtures: list[dict] = ROUND_OF_32, round_n: int = 4,
                              stage_label: str = "Round of 32") -> list[dict]:
    """UPSERT dos confrontos do mata-mata em `matches` (sem placar) e
    `copa_2026_results` (round=round_n, group_name=stage_label, sem placar)."""
    match_rows = [{
        "date": f["date"], "home_team": f["home"], "away_team": f["away"],
        "tournament": "FIFA World Cup", "neutral": f["neutral"],
    } for f in fixtures]

    result_rows = [{
        "game_id": f"{f['home']}-{f['away']}-{f['date'].replace('-', '')}",
        "home_team": f["home"], "away_team": f["away"], "date": f["date"],
        "round": round_n, "group_name": stage_label,
    } for f in fixtures]

    print(f"  {len(fixtures)} jogos ({stage_label}) ...", end=" ", flush=True)
    ok_matches = db_client.upsert("matches", match_rows,
                                   on_conflict="date,home_team,away_team,tournament")
    ok_results = db_client.upsert("copa_2026_results", result_rows, on_conflict="game_id")
    if ok_matches and ok_results:
        print("ok -> Supabase (matches + copa_2026_results)")
    else:
        print("Supabase não configurado — nada gravado.")
    return fixtures


if __name__ == "__main__":
    scrape_knockout_fixtures()
