"""[Iteração 2 / Prompt E] Inserção de resultados reais + ciclo sequencial da Copa 2026.

`insert_result`: registra o placar real de um jogo (idempotente) em `copa_2026_results`
e `matches` (E1).

`update_round`: PASSO1 busca odds (E2) -> PASSO2 lê resultados da Rodada N ->
PASSO2a marca os jogos como jogados (avança `data.today`, upsert `matches`) ->
PASSO2b/3 re-treina TODOS os membros (inclui `bayesian`) via `selection.select()` sobre
o `training_frame()` expandido -- "sequencial" = re-fit com mais dados reais da Copa, já
que `pb.models.HierarchicalBayesianGoalModel.fit()` não suporta chaining de
prior/posterior -> PASSO4 EV automático (scoring.py, dentro de predict_2026) ->
PASSO5 regera predictions_2026.json (+ odds_implied/divergence_alert/round_updated/
model_confidence) e copia para o frontend -> PASSO6 log final em PT-BR.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "5_outputs"))
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import CONFIG_PATH, load_config, path  # noqa: E402
import dataset  # noqa: E402
import db_client  # noqa: E402
from odds_fetcher import fetch_and_store_odds  # noqa: E402


def _resolve_team(name: str, cfg: dict, canonical: set[str]) -> str | None:
    """Resolve `name` para o nome canônico (EN, martj42). Aceita o nome canônico
    direto ou o nome em português de `display_names` (lookup reverso)."""
    if name in canonical:
        return name
    reverse_display = {pt: en for en, pt in cfg.get("display_names", {}).items()}
    return reverse_display.get(name)


def insert_result(home: str, away: str, home_score: int, away_score: int,
                   date_str: str, verbose: bool = True) -> str | None:
    """Registra o placar real de um confronto da Copa 2026 (idempotente).

    Atualiza `copa_2026_results` (upsert on_conflict=game_id, preserva round/group_name)
    e `matches` (upsert on_conflict=date,home_team,away_team,tournament -- preprocess
    recalcula `played=True` na próxima leitura).
    """
    cfg = load_config()
    disp = cfg.get("display_names", {})
    fixtures = dataset.get_wc2026_fixtures()
    canonical = set(fixtures["home_team"]) | set(fixtures["away_team"])

    home_en = _resolve_team(home, cfg, canonical)
    away_en = _resolve_team(away, cfg, canonical)
    if home_en is None or away_en is None:
        unresolved = [n for n, e in ((home, home_en), (away, away_en)) if e is None]
        print(f"  Time(s) não reconhecido(s): {unresolved}. Use o nome canônico (inglês, "
              f"martj42) ou o nome em português de config.display_names.")
        return None

    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    game_id = f"{home_en}-{away_en}-{d.strftime('%Y%m%d')}"

    db_client.upsert("copa_2026_results", [{
        "game_id": game_id, "home_team": home_en, "away_team": away_en,
        "date": date_str, "home_score": home_score, "away_score": away_score,
        "played_at": datetime.now(timezone.utc).isoformat(),
    }], on_conflict="game_id")

    row = db_client.fetch_one("copa_2026_results", "game_id", game_id, columns="round")
    if row is not None and row.get("round") is None:
        print(f"⚠ Resultado inserido SEM round (game_id={game_id}). "
              f"--update-round N não vai enxergá-lo. Rode --scrape-fixtures antes, "
              f"ou verifique o game_id.")

    db_client.upsert("matches", [{
        "date": date_str, "home_team": home_en, "away_team": away_en,
        "tournament": "FIFA World Cup",
        "home_score": home_score, "away_score": away_score,
    }], on_conflict="date,home_team,away_team,tournament")

    if verbose:
        print(f"Resultado inserido: {disp.get(home_en, home_en)} {home_score}×{away_score} "
              f"{disp.get(away_en, away_en)} ✅")
    return game_id


def _advance_today(new_date: str) -> None:
    """Atualiza `data.today` em config.yaml (substituição direcionada, preserva
    comentários/formatação) e limpa o cache de `load_config`."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    new_text = re.sub(r'(today:\s*)"[\d-]+"', rf'\1"{new_date}"', text, count=1)
    CONFIG_PATH.write_text(new_text, encoding="utf-8")
    load_config.cache_clear()


def update_round(n: int, verbose: bool = True, force: bool = False) -> None:
    """PASSO0-6 do Prompt E: garante os resultados da Rodada N (--fetch-results),
    busca odds, incorpora os resultados, re-treina e re-seleciona, regenera
    previsões/EV e atualiza o frontend."""
    # PASSO0 -- busca automática de placares terminados (FONTE 1 -> 2 -> 3).
    from results_fetcher import fetch_and_insert_results
    fetch_and_insert_results(verbose=verbose)

    # PASSO1 -- odds de mercado.
    odds_info = fetch_and_store_odds(verbose=verbose)

    # PASSO2 -- resultados da Rodada N (proxy de "has_result": home_score IS NOT NULL).
    results = db_client.fetch_all("copa_2026_results") or []
    round_rows = [r for r in results if r.get("round") == n and r.get("home_score") is not None]
    pending_rows = [r for r in results if r.get("round") == n and r.get("home_score") is None]
    if not round_rows and not pending_rows:
        print(f"  Nenhum resultado da Rodada {n} encontrado em copa_2026_results -- "
              f"rode --insert-result para os jogos da Rodada {n} antes de --update-round {n}.")
        return
    if pending_rows and not force:
        print(f"⚠ Rodada {n} tem {len(pending_rows)} jogo(s) sem placar. "
              f"Aguardar término ou forçar com --force-update.")
        for r in pending_rows:
            print(f"    {r['home_team']} vs {r['away_team']} ({r['date']})")
        return

    # PASSO2a -- avança "hoje" (Adaptação 3) e marca os jogos da rodada como jogados.
    _advance_today(date.today().isoformat())
    match_rows = [{
        "date": r["date"], "home_team": r["home_team"], "away_team": r["away_team"],
        "tournament": "FIFA World Cup",
        "home_score": r["home_score"], "away_score": r["away_score"],
    } for r in round_rows]
    db_client.upsert("matches", match_rows, on_conflict="date,home_team,away_team,tournament")
    dataset.load_matches.cache_clear()
    cfg = load_config()

    # PASSO2b/3 -- re-treina TODOS os membros (inclui bayesian) e re-seleciona.
    from selection import select, build_justification
    from report import write_comparison_report, write_selected_model

    sel_path = path("5_outputs", "selected_model.json")
    prev_chosen = None
    if sel_path.exists():
        prev_chosen = json.loads(sel_path.read_text(encoding="utf-8")).get("chosen_model")

    if verbose:
        print(f"● Re-treinando e re-selecionando com os resultados da Rodada {n} ...")
    sel = select(verbose=verbose)
    just = build_justification(sel)
    write_selected_model(sel, just)
    if prev_chosen and prev_chosen != sel["chosen"]:
        print(f"  modelo escolhido mudou: {prev_chosen} -> {sel['chosen']}")
        write_comparison_report(sel["comparison"], selection=sel)

    # PASSO4 -- EV recalculado automaticamente dentro de predict_2026 (scoring.py).

    # PASSO5 -- regenera predictions_2026.json (odds_implied/divergence_alert/
    # round_updated/model_confidence) e copia para o frontend.
    from predict_2026 import generate
    generate(round_updated=n, odds_credits=odds_info["credits_remaining"])

    for fname in ("predictions_2026.json", "selected_model.json"):
        src = path("5_outputs", fname)
        dest = ROOT / "frontend" / "public" / "data" / fname
        if src.exists():
            shutil.copy(src, dest)

    # PASSO6 -- relatório final em PT-BR.
    oos = sel["out_of_sample"]["RPS"]
    lam = cfg["preprocess"]["temporal_decay_lambda"]
    n_treino = len(dataset.training_frame(cfg["data"]["today"], lam))
    print(f"\n━━ RODADA {n} PROCESSADA ━━")
    print(f"  Resultados inseridos: {len(round_rows)} jogos")
    print(f"  Modelo re-treinado: {n_treino} jogos totais no treino")
    print(f"  chosen_model: {sel['chosen']}")
    print(f"  Odds atualizadas: {odds_info['games']} jogos")
    print(f"  RPS acumulado: {oos['point']:.4f}")
    print(f"  Créditos The Odds API restantes: {odds_info['credits_remaining']}")
