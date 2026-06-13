"""[Iteração 2 / D4+D6] Linkagem de nomes StatsBomb (`sb_*`) <-> Understat (`understat_*`).

`player_national_team` (D2) usa `player_id="understat_*"`; a Fase 0 ("Prompt C3") usa
`player_id="sb_*"`. Não há chave direta entre os dois — o link é por NOME, restrito a
jogadores da MESMA seleção (poucas dezenas de candidatos por seleção).

Understat usa nomes curtos/comuns ("Alvaro Morata") e StatsBomb usa nomes legais
completos ("Álvaro Borja Morata Martín"). Após remover acentos, usa-se
`fuzz.token_set_ratio` (score=100 quando TODOS os tokens do nome curto aparecem no nome
longo) — `fuzz.WRatio` causa falsos-positivos massivos aqui: qualquer token isolado em
comum (ex. "Martín") já batia >=85 contra qualquer nome completo que o contivesse,
colapsando dezenas de jogadores diferentes no mesmo sb_pid.

Reutilizado por `club_to_national.py` (D4, blend de impacto) e `squad_strength.py`
(D6.1, w_overlap via lineups StatsBomb).

`normalize_name` também é reaproveitado por `1_data/match_players.py` (D2,
Understat -> Transfermarkt) -- mesmo tipo de bug de acentuação, scorer diferente
(`fuzz.token_sort_ratio`, ver comentário em config.yaml).
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load_config  # noqa: E402
import db_client  # noqa: E402


def normalize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def build_understat_to_sb(player_national_team: list[dict], players: dict[str, dict],
                           threshold: int) -> dict[str, str]:
    """understat_pid -> sb_pid, fuzzy-matched por nome dentro da mesma seleção."""
    sb_by_team: dict[str, list[dict]] = {}
    for pid, info in players.items():
        if not pid.startswith("sb_"):
            continue
        team = info.get("nationality")
        if team:
            sb_by_team.setdefault(team, []).append({"player_id": pid, "name": info["name"]})

    mapping: dict[str, str] = {}
    for r in player_national_team:
        pid, team = r["player_id"], r["national_team"]
        cand = sb_by_team.get(team)
        if not cand:
            continue
        name = players.get(pid, {}).get("name")
        if not name:
            continue
        choices = [normalize_name(c["name"]) for c in cand]
        match = process.extractOne(normalize_name(name), choices,
                                     scorer=fuzz.token_set_ratio, score_cutoff=threshold)
        if match:
            mapping[pid] = cand[match[2]]["player_id"]
    return mapping


def load_understat_to_sb(verbose: bool = True) -> dict[str, str]:
    cfg = load_config()
    threshold = cfg["club_to_national"]["statsbomb_match_threshold"]
    player_national_team = db_client.fetch_all("player_national_team") or []
    players = {r["player_id"]: r for r in (db_client.fetch_all("players") or [])}
    mapping = build_understat_to_sb(player_national_team, players, threshold)
    if verbose:
        print(f"  {len(player_national_team)} jogadores convocáveis | "
              f"{len(mapping)} casados com StatsBomb (Fase 0)")
    return mapping
