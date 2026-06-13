"""[Iteração 2 / D5 + D6.1] Força de elenco por seleção (`squad_strength`).

D5 — ataque/defesa ajustados
-----------------------------
Para cada uma das 48 seleções, pega o elenco aproximado (`squad_2026`, D2) e ordena
por minutos de clube na temporada Understat mais recente (fallback 0 — jogador sem
dado de clube ainda conta no elenco, só não pesa no ranking). Os top
`cfg.squad_2026_build.starters_count` (11) por minutos são "titulares projetados"
(peso 2); o restante são "reservas" (peso 1).

Para cada jogador, resolve o impacto pela primeira fonte disponível em
`cfg.squad_strength.impact_source_priority` (`final_2026` > `rapm_club` >
`statsbomb_national`) — jogadores sem QUALQUER linha em `player_impact` não entram
na média (sem dado sintético), mas contam no denominador de `data_coverage`.

  attack_adjusted  = média ponderada (pesos titular/reserva) de attack_delta
  defense_adjusted = média ponderada de defense_delta
  attack_z_pct     = percentil (0-100) de attack_adjusted entre as seleções com dado
  xgf90            = média ponderada de player_seasons.xg90 (clube, Understat)
  xga90            = média ponderada de team_seasons.xga90 do clube/temporada de cada
                     jogador (proxy honesto — Understat não dá xGA por jogador)
  data_coverage    = (# jogadores do roster com QUALQUER linha em player_impact)
                     / len(roster)

Seleções sem `squad_2026` (ex.: elenco não mapeado, D2) recebem uma linha com todos os
campos acima = None e `data_coverage = 0.0` — D8 trata graciosamente via `is None`.

D6.1 — w_overlap
-----------------
Para cada um dos 314 jogos StatsBomb (`cfg.xg_scraping.competitions`), `sb.lineups`
dá a escalação titular (intervalo `from="00:00"`) de cada seleção. Mapeia cada jogador
StatsBomb -> `understat_pid` (via `sb_match`) e verifica se está em `squad_2026[team]`:

  overlap_jogo = (# titulares do lineup que estão em squad_2026[team]) / 11
  w_overlap[team] = média de overlap_jogo entre os jogos dessa seleção

Seleções sem cobertura StatsBomb: `w_overlap = cfg.preprocess.squad_overlap_weight`
(1.0 — comportamento atual preservado, D6.2).

Cache + idempotência (antes do D8)
-----------------------------------
- Lineups StatsBomb são cacheados em disco (`.cache/lineups/{match_id}.json`, dados
  históricos imutáveis) -- reruns não rebatem na API para jogos já vistos.
- `w_overlap` é idempotente por seleção: uma seleção cujo `squad_strength.w_overlap`
  já existe no Supabase é pulada (mesmo padrão do `scraper_xg.py` -- "já preenchido").
  Um match só é buscado/processado se pelo menos uma das duas seleções dele ainda
  estiver pendente.
- Persistência incremental: `attack_adjusted`/`defense_adjusted`/`xgf90`/`xga90`/
  `data_coverage`/`attack_z_pct` (rápido, só DB) são upsertados ANTES do laço lento
  de `w_overlap`, preservando o `w_overlap` já existente -- uma interrupção durante o
  laço de lineups não perde esse resultado.

Saída: squad_strength (upsert on_conflict='team', uma linha por seleção).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "lineups"

sys.path.insert(0, str(ROOT))
from config_loader import load_config  # noqa: E402
import db_client  # noqa: E402
from dataset import get_wc2026_fixtures  # noqa: E402
from sb_match import load_understat_to_sb  # noqa: E402


def _alias_map(cfg: dict) -> dict:
    aliases = dict(cfg.get("team_aliases", {}))
    aliases.update(cfg.get("statsbomb_team_aliases", {}))
    return aliases


def _load_lineups(mid: int, sb, cache_file: Path) -> dict[str, pd.DataFrame]:
    """sb.lineups(match_id=mid), cacheado em `cache_file` (jogos passados são imutáveis)."""
    if cache_file.exists():
        with open(cache_file) as f:
            raw = json.load(f)
        return {team: pd.read_json(StringIO(data), orient="records") for team, data in raw.items()}

    lineups = sb.lineups(match_id=mid)
    raw = {team: df.to_json(orient="records") for team, df in lineups.items()}
    with open(cache_file, "w") as f:
        json.dump(raw, f)
    return lineups


def _latest_understat(player_seasons: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for r in player_seasons:
        pid = r["player_id"]
        if pid not in latest or r["season"] > latest[pid]["season"]:
            latest[pid] = r
    return latest


def _impact_maps(player_impact: list[dict]) -> dict[str, dict[str, dict]]:
    by_source: dict[str, dict[str, dict]] = {}
    for r in player_impact:
        by_source.setdefault(r["source"], {})[r["player_id"]] = r
    return by_source


def _resolve_impact(pid: str, by_source: dict, priority: list[str]) -> dict | None:
    for src in priority:
        row = by_source.get(src, {}).get(pid)
        if row is not None:
            return row
    return None


def compute_w_overlap(squad_sets: dict[str, set[str]], sb_to_understat: dict[str, str],
                       existing_w_overlap: dict[str, float | None], cfg: dict,
                       verbose: bool = True) -> tuple[dict[str, float], float, set[str]]:
    """Idempotente por seleção: `pending` = seleções com `squad_2026` cujo
    `squad_strength.w_overlap` ainda não existe no Supabase. Um match só é
    buscado/processado se pelo menos um dos dois times dele estiver em `pending`
    (lineups cacheados em `CACHE_DIR`, ver `_load_lineups`)."""
    from statsbombpy import sb

    default = cfg["preprocess"]["squad_overlap_weight"]
    aliases = _alias_map(cfg)
    all_teams = set(squad_sets.keys())
    pending = {t for t in all_teams if existing_w_overlap.get(t) is None}
    skipped = all_teams - pending

    if verbose:
        print(f"  w_overlap: {len(skipped)} times já calculados (pulados) / "
              f"{len(pending)} times processados agora", flush=True)
    if not pending:
        return {}, default, pending

    competitions = cfg.get("xg_scraping", {}).get("competitions", [])
    overlaps: dict[str, list[float]] = defaultdict(list)
    n_matches = n_cached = n_fetched = 0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for comp in competitions:
        comp_id, season_id, tlabel = comp["competition_id"], comp["season_id"], comp["tournament_label"]
        sb_matches = sb.matches(competition_id=comp_id, season_id=season_id)
        for _, m in sb_matches.iterrows():
            home = aliases.get(m["home_team"], m["home_team"])
            away = aliases.get(m["away_team"], m["away_team"])
            if home not in pending and away not in pending:
                continue
            mid = int(m["match_id"])
            cache_file = CACHE_DIR / f"{mid}.json"
            cached = cache_file.exists()
            try:
                lineups = _load_lineups(mid, sb, cache_file)
            except Exception as e:
                if verbose:
                    print(f"  match {mid} ({tlabel}): erro ({e}) — pulado", flush=True)
                continue
            n_cached += int(cached)
            n_fetched += int(not cached)
            for raw_team, df in lineups.items():
                team = aliases.get(raw_team, raw_team)
                if team not in pending:
                    continue
                squad = squad_sets.get(team)
                if not squad:
                    continue
                matched = 0
                for _, row in df.iterrows():
                    positions = row["positions"]
                    if not positions or positions[0].get("from") != "00:00":
                        continue
                    sb_pid = f"sb_{int(row['player_id'])}"
                    if sb_to_understat.get(sb_pid) in squad:
                        matched += 1
                overlaps[team].append(matched / 11.0)
            n_matches += 1
            if verbose and n_matches % 25 == 0:
                print(f"  ... {n_matches} jogos processados (w_overlap)", flush=True)
        if verbose:
            print(f"  {tlabel} (competition_id={comp_id}, season_id={season_id}): processado.", flush=True)

    if verbose:
        print(f"  lineups: {n_cached} em cache local (.cache/lineups/) / "
              f"{n_fetched} buscados da API StatsBomb agora", flush=True)

    computed = {team: (sum(v) / len(v) if v else default) for team, v in overlaps.items()}
    return computed, default, pending


def compute_squad_strength(verbose: bool = True) -> list[dict]:
    cfg = load_config()
    starters_count = cfg["squad_2026_build"]["starters_count"]
    priority = cfg["squad_strength"]["impact_source_priority"]
    default_overlap = cfg["preprocess"]["squad_overlap_weight"]

    fixtures = get_wc2026_fixtures()
    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))

    existing = {r["team"]: r for r in (db_client.fetch_all("squad_strength") or [])}

    squad_2026 = db_client.fetch_all("squad_2026") or []
    by_team: dict[str, list[dict]] = defaultdict(list)
    for r in squad_2026:
        by_team[r["team"]].append(r)

    seasons = db_client.fetch_all("player_seasons") or []
    understat_seasons = [r for r in seasons if r["source"] == "understat"]
    latest = _latest_understat(understat_seasons)

    team_seasons = {(r["team"], r["league"], r["season"]): r
                     for r in (db_client.fetch_all("team_seasons") or [])}

    by_source = _impact_maps(db_client.fetch_all("player_impact") or [])

    rows = []
    coverage_report = []
    for team in teams:
        roster = by_team.get(team, [])
        if not roster:
            rows.append({
                "team": team, "attack_adjusted": None, "defense_adjusted": None,
                "attack_z_pct": None, "xgf90": None, "xga90": None,
                "w_overlap": round(default_overlap, 4), "data_coverage": 0.0,
            })
            coverage_report.append((team, 0, 0))
            continue

        scored = []
        for r in roster:
            pid = r["player_id"]
            ls = latest.get(pid)
            minutes = (ls["minutes"] or 0.0) if ls else 0.0
            scored.append({"player_id": pid, "minutes": minutes, "season": ls})
        scored.sort(key=lambda r: r["minutes"], reverse=True)

        sum_attack = sum_defense = weight_impact = 0.0
        sum_xgf = weight_xgf = 0.0
        sum_xga = weight_xga = 0.0
        n_with_impact = 0
        for i, s in enumerate(scored):
            w = 2.0 if i < starters_count else 1.0
            imp = _resolve_impact(s["player_id"], by_source, priority)
            if imp is not None:
                sum_attack += w * imp["attack_delta"]
                sum_defense += w * imp["defense_delta"]
                weight_impact += w
                n_with_impact += 1

            ls = s["season"]
            if ls and ls.get("xg90") is not None:
                sum_xgf += w * ls["xg90"]
                weight_xgf += w
                ts = team_seasons.get((ls["team"], ls["competition"], ls["season"]))
                if ts and ts.get("xga90") is not None:
                    sum_xga += w * ts["xga90"]
                    weight_xga += w

        rows.append({
            "team": team,
            "attack_adjusted": round(sum_attack / weight_impact, 4) if weight_impact else None,
            "defense_adjusted": round(sum_defense / weight_impact, 4) if weight_impact else None,
            "attack_z_pct": None,  # preenchido abaixo (precisa de todas as 48 seleções)
            "xgf90": round(sum_xgf / weight_xgf, 4) if weight_xgf else None,
            "xga90": round(sum_xga / weight_xga, 4) if weight_xga else None,
            # preserva w_overlap já existente (upsert incremental, ver abaixo) --
            # None na 1a execução, recalculado em compute_w_overlap se necessário.
            "w_overlap": existing.get(team, {}).get("w_overlap"),
            "data_coverage": round(n_with_impact / len(roster), 4),
        })
        coverage_report.append((team, len(roster), n_with_impact))

    # attack_z_pct: percentil 0-100 de attack_adjusted entre as seleções com dado
    valid = [r for r in rows if r["attack_adjusted"] is not None]
    valid.sort(key=lambda r: r["attack_adjusted"])
    n = len(valid)
    for i, r in enumerate(valid):
        r["attack_z_pct"] = round(i / (n - 1) * 100, 1) if n > 1 else 50.0

    # (a) persiste ataque/defesa/coverage AGORA (rápido, só DB) -- antes do laço lento
    # de w_overlap (D6.1) abaixo, preservando o w_overlap já existente. Uma interrupção
    # durante o laço de lineups não perde este resultado.
    db_client.upsert("squad_strength", rows, on_conflict="team")

    # D6.1: w_overlap via lineups StatsBomb (314 jogos), cache local + skip por seleção
    squad_sets = {team: {r["player_id"] for r in roster} for team, roster in by_team.items()}
    sb_to_understat = {v: k for k, v in load_understat_to_sb(verbose=verbose).items()}
    existing_w_overlap = {team: r.get("w_overlap") for team, r in existing.items()}
    w_overlap_map, _, pending = compute_w_overlap(
        squad_sets, sb_to_understat, existing_w_overlap, cfg, verbose=verbose)
    for r in rows:
        if r["team"] in pending:
            r["w_overlap"] = round(w_overlap_map.get(r["team"], default_overlap), 4)

    if verbose:
        low = [(t, n_imp, n_ros) for t, n_ros, n_imp in coverage_report if n_ros == 0 or n_imp / n_ros < 0.5]
        print(f"  {len(rows)} seleções | data_coverage < 50%: "
              + (", ".join(f"{t}({n_imp}/{n_ros})" for t, n_imp, n_ros in low) if low else "nenhuma"))

    return rows


def run(verbose: bool = True) -> None:
    rows = compute_squad_strength(verbose=verbose)
    if rows:
        db_client.upsert("squad_strength", rows, on_conflict="team")
    print(f"  Upsert: {len(rows)} squad_strength.")


if __name__ == "__main__":
    run()
