"""[Iteração 2 / Fase 0 — "Prompt C3"] Impacto de jogador na SELEÇÃO via StatsBomb.

Reaproveita os 314 jogos de `cfg["xg_scraping"]["competitions"]` (mesmos do Prompt
C2/C4, `1_data/scrapers/scraper_xg.py`) — sem scraping novo. Para cada partida:

- `sb.lineups(match_id)`: titulares/reservas por seleção, com os intervalos `from`/`to`
  (formato "MM:SS") em que cada jogador esteve em campo -> minutos jogados.
- `sb.events(match_id)`: `shot_statsbomb_xg` por chute. O atirador recebe o xG como
  ATAQUE; o xG é também rateado entre os jogadores do time DEFENSOR que estavam em
  campo no minuto do chute (xG / nº de defensores em campo) -> DEFESA.

Saída (agregada por jogador × seleção, todas as competições juntas):
- `player_seasons` (source='statsbomb_national'): minutes, games, xg90 (=ataque/90),
  xa90/npxg90/xgchain90/xgbuildup90 = None (StatsBomb não fornece esses agregados).
- `player_impact` (source='statsbomb_national'): attack_delta, defense_delta
  (positivo = defesa BOA), std_error (bootstrap por partida).
- `players`: upsert de (player_id, name, nationality=seleção, position).

Filtra jogadores com minutos totais < cfg.national_impact.min_minutes_for_impact.
Sem dados sintéticos: jogadores sem chutes a favor/contra apenas ficam com
attack_delta/defense_delta = -/+ a média global (delta = 0 quando exatamente na média).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsbombpy import sb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load_config  # noqa: E402
import db_client  # noqa: E402


def _alias_map(cfg: dict) -> dict:
    aliases = dict(cfg.get("team_aliases", {}))
    aliases.update(cfg.get("statsbomb_team_aliases", {}))
    return aliases


def _parse_clock(s: str) -> float:
    """'MM:SS' -> minutos (float)."""
    mm, ss = s.split(":")
    return int(mm) + int(ss) / 60.0


def _player_minutes(positions: list[dict], match_end: float) -> tuple[float, list[tuple[float, float]], str | None]:
    """Soma (to-from) das janelas de `positions` (`to=None` -> match_end).

    Retorna (minutos_totais, intervalos_em_campo, posição_da_1a_janela)."""
    intervals = []
    total = 0.0
    first_pos = None
    for p in positions:
        if first_pos is None:
            first_pos = p.get("position")
        start = _parse_clock(p["from"])
        end = match_end if p["to"] is None else _parse_clock(p["to"])
        total += max(0.0, end - start)
        intervals.append((start, end))
    return total, intervals, first_pos


def _on_field(intervals: list[tuple[float, float]], minute: float) -> bool:
    return any(s <= minute < e for s, e in intervals)


def _process_match(match_id: int, aliases: dict) -> list[dict]:
    """Retorna registros por (player_id, team_canonico) para esta partida:
    {player_id, name, team, position, minutes, xg_for, xg_against}."""
    lineups = sb.lineups(match_id)
    events = sb.events(match_id=match_id)
    if events.empty:
        return []

    match_end = max(float(events["minute"].max()), 90.0)

    # team_players[raw_team_name][player_id] = {name, intervals, minutes, position}
    team_players: dict[str, dict[int, dict]] = {}
    for raw_team, df in lineups.items():
        tp = {}
        for _, row in df.iterrows():
            minutes, intervals, position = _player_minutes(row["positions"], match_end)
            if minutes <= 0:
                continue
            tp[row["player_id"]] = {
                "name": row["player_name"],
                "intervals": intervals,
                "minutes": minutes,
                "position": position,
            }
        team_players[raw_team] = tp

    # acumuladores desta partida: (player_id, raw_team) -> dict
    match_acc: dict[tuple[int, str], dict] = {}

    def _ensure(pid, raw_team, info):
        key = (pid, raw_team)
        if key not in match_acc:
            match_acc[key] = {
                "player_id": pid, "name": info["name"], "team": raw_team,
                "position": info["position"], "minutes": info["minutes"],
                "xg_for": 0.0, "xg_against": 0.0,
            }
        return match_acc[key]

    for raw_team, players in team_players.items():
        for pid, info in players.items():
            _ensure(pid, raw_team, info)

    shots = events[events["type"] == "Shot"]
    shots = shots[shots["shot_statsbomb_xg"].notna()]
    for _, shot in shots.iterrows():
        shot_team = shot["team"]
        xg = float(shot["shot_statsbomb_xg"])
        second = shot["second"] if "second" in shot.index and pd.notna(shot["second"]) else 0
        minute = float(shot["minute"]) + float(second) / 60.0

        shooter_id = shot.get("player_id")
        if shot_team in team_players and shooter_id in team_players[shot_team]:
            match_acc[(shooter_id, shot_team)]["xg_for"] += xg

        for def_team, players in team_players.items():
            if def_team == shot_team:
                continue
            on_field = [pid for pid, info in players.items() if _on_field(info["intervals"], minute)]
            if not on_field:
                continue
            share = xg / len(on_field)
            for pid in on_field:
                match_acc[(pid, def_team)]["xg_against"] += share

    out = []
    for rec in match_acc.values():
        rec = dict(rec)
        rec["team"] = aliases.get(rec["team"], rec["team"])
        rec["match_id"] = match_id
        out.append(rec)
    return out


def _per90(total: float, minutes: float) -> float:
    return total / minutes * 90.0


def _bootstrap_std_error(sub: pd.DataFrame, iters: int, seed: int) -> float:
    """Bootstrap por partida (resample de linhas de `sub` com reposição).

    Retorna max(se_attack, se_defense) — uma única coluna `std_error` cobre
    ambos os lados (D4 usa este valor como medida de confiança geral)."""
    n = len(sub)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    minutes = sub["minutes"].to_numpy()
    xg_for = sub["xg_for"].to_numpy()
    xg_against = sub["xg_against"].to_numpy()
    attack_samples = np.empty(iters)
    defense_samples = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        m = minutes[idx].sum()
        if m <= 0:
            attack_samples[i] = 0.0
            defense_samples[i] = 0.0
            continue
        attack_samples[i] = _per90(xg_for[idx].sum(), m)
        defense_samples[i] = _per90(xg_against[idx].sum(), m)
    return float(max(attack_samples.std(ddof=1), defense_samples.std(ddof=1)))


def compute_national_impact(verbose: bool = True) -> dict:
    cfg = load_config()
    aliases = _alias_map(cfg)
    competitions = cfg.get("xg_scraping", {}).get("competitions", [])
    min_minutes = cfg.get("national_impact", {}).get("min_minutes_for_impact", 90)
    iters = cfg.get("validation", {}).get("bootstrap_iterations", 1000)
    seed = cfg.get("validation", {}).get("random_seed", 42)

    records: list[dict] = []
    n_matches = 0
    for comp in competitions:
        comp_id, season_id, tlabel = comp["competition_id"], comp["season_id"], comp["tournament_label"]
        sb_matches = sb.matches(competition_id=comp_id, season_id=season_id)
        for _, m in sb_matches.iterrows():
            mid = int(m["match_id"])
            try:
                records.extend(_process_match(mid, aliases))
            except Exception as e:
                if verbose:
                    print(f"  match {mid} ({tlabel}): erro ({e}) — pulado", flush=True)
                continue
            n_matches += 1
            if verbose and n_matches % 25 == 0:
                print(f"  ... {n_matches} jogos processados", flush=True)
        if verbose:
            print(f"  {tlabel} (competition_id={comp_id}, season_id={season_id}): processado.", flush=True)

    if not records:
        print("  Nenhum registro extraído (StatsBomb indisponível?).")
        return {"player_seasons": [], "player_impact": [], "players": []}

    df = pd.DataFrame(records)
    agg = df.groupby(["player_id", "team"]).agg(
        name=("name", "first"),
        position=("position", "first"),
        minutes=("minutes", "sum"),
        games=("match_id", "nunique"),
        xg_for=("xg_for", "sum"),
        xg_against=("xg_against", "sum"),
    ).reset_index()
    agg = agg[agg["minutes"] >= min_minutes].copy()

    if agg.empty:
        print(f"  Nenhum jogador com >= {min_minutes} minutos em {n_matches} jogos.")
        return {"player_seasons": [], "player_impact": [], "players": []}

    agg["xg90_for"] = agg.apply(lambda r: _per90(r["xg_for"], r["minutes"]), axis=1)
    agg["xg90_against"] = agg.apply(lambda r: _per90(r["xg_against"], r["minutes"]), axis=1)
    mean_for = agg["xg90_for"].mean()
    mean_against = agg["xg90_against"].mean()
    agg["attack_delta"] = agg["xg90_for"] - mean_for
    agg["defense_delta"] = mean_against - agg["xg90_against"]

    std_errors = []
    for _, r in agg.iterrows():
        sub = df[(df["player_id"] == r["player_id"]) & (df["team"] == r["team"])]
        std_errors.append(_bootstrap_std_error(sub, iters, seed))
    agg["std_error"] = std_errors

    player_seasons_rows = []
    player_impact_rows = []
    players_rows = []
    for _, r in agg.iterrows():
        pid = f"sb_{int(r['player_id'])}"
        player_seasons_rows.append({
            "player_id": pid, "team": r["team"], "season": "2018-2024",
            "competition": "statsbomb_national_teams", "minutes": round(float(r["minutes"]), 1),
            "games": int(r["games"]), "xg90": round(float(r["xg90_for"]), 4),
            "xa90": None, "npxg90": None, "xgchain90": None, "xgbuildup90": None,
            "source": "statsbomb_national",
        })
        player_impact_rows.append({
            "player_id": pid, "attack_delta": round(float(r["attack_delta"]), 4),
            "defense_delta": round(float(r["defense_delta"]), 4),
            "std_error": round(float(r["std_error"]), 4), "source": "statsbomb_national",
        })
        players_rows.append({
            "player_id": pid, "name": r["name"], "nationality": r["team"],
            "position": r["position"],
        })

    if verbose:
        print(f"  {n_matches} jogos processados | {len(agg)} jogadores com >= {min_minutes}min")
        print(f"  média xG90_for={mean_for:.3f} | média xG90_against={mean_against:.3f}")

    return {"player_seasons": player_seasons_rows, "player_impact": player_impact_rows,
            "players": players_rows}


def run(verbose: bool = True) -> None:
    out = compute_national_impact(verbose=verbose)
    if out["players"]:
        db_client.upsert("players", out["players"], on_conflict="player_id")
    if out["player_seasons"]:
        db_client.upsert("player_seasons", out["player_seasons"],
                          on_conflict="player_id,team,season,source")
    if out["player_impact"]:
        db_client.upsert("player_impact", out["player_impact"], on_conflict="player_id,source")
    print(f"  Upsert: {len(out['players'])} players, {len(out['player_seasons'])} player_seasons, "
          f"{len(out['player_impact'])} player_impact (source=statsbomb_national).")


if __name__ == "__main__":
    run()
