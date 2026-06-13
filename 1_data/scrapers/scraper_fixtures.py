"""Fixtures da fase de grupos da Copa 2026 (round/group_name) -> tabela copa_2026_results.

As datas/confrontos dos 72 jogos já vêm do dataset martj42 (results.csv, placar NaN —
ver dataset.get_wc2026_fixtures). O que falta é a composição dos 12 grupos (A-L), que
NÃO está no martj42. Em vez de raspar uma tabela grande de 72 jogos via IA (risco de
dado sintético/errado), raspamos só os 48 times -> grupo (Wikipedia, "2026 FIFA World Cup
Group A".."Group L", confirmado em 2026-06-10) e derivamos:
  - group_name: grupo do confronto (home/away sempre no mesmo grupo na fase de grupos);
  - round: rodada 1/2/3, pela ordem cronológica das datas dentro de cada grupo
    (3 rodadas de 2 jogos cada, por construção do calendário da Copa).

Sem placares ainda (home_score/away_score ficam NULL — preenchidos pelo aprendizado
sequencial da Iteração 2 conforme os jogos acontecem).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load_config  # noqa: E402
import dataset  # noqa: E402
import db_client  # noqa: E402

# Composição dos 12 grupos (Wikipedia, "2026 FIFA World Cup Group A".."Group L", 2026-06-10).
# Nomes como aparecem na Wikipedia; normalizados para os nomes canônicos do martj42 via
# config.team_aliases (ex.: "Czechia" -> "Czech Republic").
GROUPS_RAW: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def _team_to_group(aliases: dict[str, str]) -> dict[str, str]:
    mapping = {}
    for group, teams in GROUPS_RAW.items():
        for t in teams:
            mapping[aliases.get(t, t)] = group
    return mapping


def scrape_fixtures() -> list[dict]:
    """Monta os 72 jogos da fase de grupos com group_name/round e faz UPSERT em
    `copa_2026_results` (sem placares). Retorna as linhas geradas."""
    cfg = load_config()
    aliases = cfg.get("team_aliases", {})
    team_to_group = _team_to_group(aliases)

    fixtures = dataset.get_wc2026_fixtures().copy()
    if fixtures.empty:
        print("  nenhum fixture da Copa 2026 encontrado em `matches` (rode --scrape antes).")
        return []

    fixtures["group_name"] = fixtures["home_team"].map(team_to_group)
    missing = fixtures[fixtures["group_name"].isna()]
    if not missing.empty:
        teams = sorted(set(missing["home_team"]) | set(missing["away_team"]))
        print(f"  aviso: {len(missing)} jogo(s) sem grupo mapeado (times: {teams}); "
              f"GROUPS_RAW pode estar desatualizado.")
        fixtures = fixtures[fixtures["group_name"].notna()]

    rows = []
    for group, sub in fixtures.groupby("group_name"):
        sub = sub.sort_values("date").reset_index(drop=True)
        for i, r in sub.iterrows():
            date = r["date"]
            rows.append({
                "game_id": f"{r['home_team']}-{r['away_team']}-{date.strftime('%Y%m%d')}",
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "date": date.strftime("%Y-%m-%d"),
                "round": i // 2 + 1,
                "group_name": f"Group {group}",
            })

    print(f"  {len(rows)} jogos da fase de grupos (12 grupos x 6 jogos) ...", end=" ", flush=True)
    if db_client.upsert("copa_2026_results", rows, on_conflict="game_id"):
        print("ok -> Supabase (copa_2026_results)")
    else:
        print("Supabase não configurado — nada gravado (sem CSV nesta etapa).")
    return rows


if __name__ == "__main__":
    scrape_fixtures()
