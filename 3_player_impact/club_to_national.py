"""[Iteração 2 / D4] Transferência de impacto clube -> seleção.

Combina `player_impact(source='rapm_club')` (D3, força de clube no Understat) com
`player_impact(source='statsbomb_national')` (Fase 0, força na seleção) num único
`player_impact(source='final_2026')` por jogador convocável (`player_national_team`, D2).

Linkagem clube <-> seleção: via `sb_match.load_understat_to_sb` (fuzzy-match por nome,
ver docstring desse módulo para detalhes do scorer).

Confiança (D3 PAUSA — resolução do usuário: nada de 1/std_error² para o lado clube,
pois o bootstrap do Ridge dá std_error artificialmente BAIXO para jogadores de poucos
minutos; std_error do rapm_club continua salvo na tabela só como referência):

  confianca_nacional = min(minutos_selecao, 500) / 500
  confianca_clube    = min(minutos_clube_total, cfg.club_to_national.club_min_minutes) /
                        cfg.club_to_national.club_min_minutes

Blend (pesos de evidência, não uma média que force soma=1 — jogadores com pouca
evidência dos dois lados ficam perto de delta=0, i.e. "sem ajuste", que é o
comportamento honesto para um jogador pouco visto):

  w_nat  = confianca_nacional
  w_club = (1 - confianca_nacional) * confianca_clube
  attack_delta_final  = w_nat * nat.attack  + w_club * club.attack  * league_factor
  defense_delta_final = w_nat * nat.defense + w_club * club.defense * league_factor
  std_error_final     = w_nat * nat.std_error + w_club * club.std_error

`league_factor` (config) ajusta o delta de clube pela força relativa da liga onde o
jogador atuou na temporada Understat mais recente.

Casos sem um dos dois lados: o lado ausente entra com delta=0/std_error=0 nas somas
acima (confianca correspondente também é 0), então o resultado degrada naturalmente
para "só o outro lado, ponderado pela confiança dele" — sem dado sintético.
Nenhum dos dois lados -> jogador SKIP.

Saída: player_impact (source='final_2026').
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load_config  # noqa: E402
import db_client  # noqa: E402
from sb_match import load_understat_to_sb  # noqa: E402


def _latest_understat_season(player_seasons: list[dict]) -> dict[str, dict]:
    """player_id -> linha understat da temporada mais recente (para league_factor)."""
    latest: dict[str, dict] = {}
    for r in player_seasons:
        pid = r["player_id"]
        if pid not in latest or r["season"] > latest[pid]["season"]:
            latest[pid] = r
    return latest


def _club_minutes_total(player_seasons: list[dict]) -> dict[str, float]:
    """player_id -> soma de minutos em todas as temporadas/times Understat (sem
    linhas de transferência em meio à temporada, "," no team)."""
    totals: dict[str, float] = {}
    for r in player_seasons:
        if "," in r["team"]:
            continue
        totals[r["player_id"]] = totals.get(r["player_id"], 0.0) + (r["minutes"] or 0.0)
    return totals


def transfer_impact(verbose: bool = True) -> list[dict]:
    cfg = load_config()
    league_factor = cfg["club_to_national"]["league_factor"]
    club_min_minutes = cfg["club_to_national"]["club_min_minutes"]

    player_national_team = db_client.fetch_all("player_national_team") or []
    if not player_national_team:
        print("  player_national_team vazio — rode --match-players antes.")
        return []

    seasons = db_client.fetch_all("player_seasons") or []
    understat_seasons = [r for r in seasons if r["source"] == "understat"]
    national_seasons = [r for r in seasons if r["source"] == "statsbomb_national"]

    latest_understat = _latest_understat_season(understat_seasons)
    club_minutes_total = _club_minutes_total(understat_seasons)
    national_minutes = {r["player_id"]: (r["minutes"] or 0.0) for r in national_seasons}

    club_impact = {r["player_id"]: r for r in (db_client.fetch_all("player_impact") or [])
                    if r["source"] == "rapm_club"}
    nat_impact = {r["player_id"]: r for r in (db_client.fetch_all("player_impact") or [])
                   if r["source"] == "statsbomb_national"}

    pid_to_sb = load_understat_to_sb(verbose=verbose)

    rows = []
    for r in player_national_team:
        pid = r["player_id"]
        club = club_impact.get(pid)
        sb_pid = pid_to_sb.get(pid)
        nat = nat_impact.get(sb_pid) if sb_pid else None

        if club is None and nat is None:
            continue

        nat_minutes = national_minutes.get(sb_pid, 0.0) if sb_pid else 0.0
        confianca_nacional = min(nat_minutes, 500.0) / 500.0

        club_minutes = club_minutes_total.get(pid, 0.0)
        confianca_clube = min(club_minutes, club_min_minutes) / club_min_minutes

        comp = latest_understat.get(pid, {}).get("competition")
        factor = league_factor.get(comp, 1.0)

        w_nat = confianca_nacional
        w_club = (1.0 - confianca_nacional) * confianca_clube

        nat_attack = nat["attack_delta"] if nat else 0.0
        nat_defense = nat["defense_delta"] if nat else 0.0
        nat_se = nat["std_error"] if nat else 0.0
        club_attack = (club["attack_delta"] if club else 0.0) * factor
        club_defense = (club["defense_delta"] if club else 0.0) * factor
        club_se = club["std_error"] if club else 0.0

        attack_delta = w_nat * nat_attack + w_club * club_attack
        defense_delta = w_nat * nat_defense + w_club * club_defense
        std_error = w_nat * nat_se + w_club * club_se

        rows.append({
            "player_id": pid,
            "attack_delta": round(float(attack_delta), 4),
            "defense_delta": round(float(defense_delta), 4),
            "std_error": round(float(std_error), 4),
            "source": "final_2026",
        })

    return rows


def run(verbose: bool = True) -> None:
    rows = transfer_impact(verbose=verbose)
    if rows:
        db_client.upsert("player_impact", rows, on_conflict="player_id,source")
    print(f"  Upsert: {len(rows)} player_impact (source=final_2026).")


if __name__ == "__main__":
    run()
