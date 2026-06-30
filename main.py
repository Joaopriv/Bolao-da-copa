"""Orquestrador CLI do Bolão Copa 2026 (Iteração 1 — modelagem sobre gols).

Fluxo:  --scrape -> --prep -> --compare -> --select -> --predict-2026
Etapas de xG/elenco/sequencial/odds são da Iteração 2 (stubs documentados).

Exemplos:
  python main.py --scrape
  python main.py --prep
  python main.py --compare                       # todos os modelos × torneios
  python main.py --compare --tournaments WC2018 WC2022   # subconjunto rápido
  python main.py --select                        # seleção anti-overfitting + relatório
  python main.py --predict-2026
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "1_data"))
sys.path.insert(0, str(ROOT / "4_validation"))
sys.path.insert(0, str(ROOT / "5_outputs"))

from config_loader import load_config, path  # noqa: E402


def cmd_scrape(_args):
    from scrapers.scraper_results import download
    print("● Baixando dados do martj42 ...")
    download()


def cmd_prep(_args):
    import preprocess
    print("● Pré-processando (pesos temporal × competição) ...")
    preprocess.build()


def cmd_compare(args):
    from tournament_comparison import run_comparison
    from report import write_comparison_report
    print("● Comparação cruzada (modelo × torneio, com IC bootstrap) ...")
    comp = run_comparison(
        tournament_names=args.tournaments, model_names=args.models,
        lam=args.lam, iters=args.iters, verbose=True,
    )
    f = comp["funnel"]
    print(f"\n  melhor por RPS: {f['best_by_rps']} | descartados: {f['eliminated']} "
          f"| escolhido: {f['chosen']}")
    write_comparison_report(comp)


def cmd_select(args):
    from selection import select, build_justification
    from report import write_comparison_report, write_selected_model
    print("● Seleção anti-overfitting (treina ≤ cutoff, testa 1x nos posteriores) ...")
    sel = select(iters=args.iters, verbose=True)
    just = build_justification(sel)
    write_selected_model(sel, just)
    print("\n  Justificativa:\n  " + just)
    # Relatório usa a MESMA comparação completa que a seleção já calculou (sem refit extra).
    print("\n● Gerando relatório completo ...")
    write_comparison_report(sel["comparison"], selection=sel)


def cmd_predict(_args):
    from predict_2026 import generate
    print("● Gerando previsões da Copa 2026 ...")
    generate()


def cmd_odds_check(_args):
    from odds_crosscheck import crosscheck_odds
    print("● Cruzando previsões com odds de mercado (The Odds API) ...")
    crosscheck_odds(verbose=True)


def cmd_fetch_odds(_args):
    sys.path.insert(0, str(ROOT / "utils"))
    from odds_fetcher import fetch_and_store_odds
    print("● Buscando odds de mercado (The Odds API) ...")
    info = fetch_and_store_odds(verbose=True)
    credits = info["credits_remaining"]
    print(f"Odds atualizadas: {info['games']} jogos. Divergência >5pp: {info['alerts']} jogos. "
          f"Créditos restantes: {credits}")
    if credits is not None and credits < 50:
        print("⚠ Créditos da API restantes < 50 — uso de --fetch-odds deve ser espaçado.")


def cmd_fetch_market_odds(_args):
    sys.path.insert(0, str(ROOT / "utils"))
    from odds_fetcher import fetch_and_store_market_odds
    print("● Buscando odds de mercado EXPANDIDAS (total O/U, BTTS, team totals) "
          "-> odds_2026_markets (isolado do bolão) ...")
    info = fetch_and_store_market_odds(verbose=True)
    credits = info["credits_remaining"]
    print(f"Mercados gravados: {info['rows']} linhas em {info['games']} jogos. "
          f"Com odd: {', '.join(info['markets_found']) or '(nenhum)'}. "
          f"Sem odd: {', '.join(info['markets_missing']) or '(nenhum)'}. "
          f"Créditos restantes: {credits}")
    if credits is not None and credits < 50:
        print("⚠ Créditos da API restantes < 50 — uso de --fetch-market-odds deve ser espaçado.")


def cmd_insert_result(args):
    from sequential_backtest import insert_result
    print("● Registrando resultado real (Copa 2026) ...")
    insert_result(args.home, args.away, args.score[0], args.score[1], args.date, verbose=True)


def cmd_backtest_rounds(_args):
    from round_backtest import run, format_report
    res = run(verbose=True)
    print()
    print(format_report(res))


def cmd_fixtures(_args):
    from scrapers.scraper_fixtures import scrape_fixtures
    print("● Baixando fixtures da fase de grupos (Copa 2026) ...")
    scrape_fixtures()


def cmd_knockout_fixtures(_args):
    from scrapers.scraper_knockout_fixtures import scrape_knockout_fixtures
    print("● Gravando fixtures do mata-mata (Round of 32, Copa 2026) ...")
    scrape_knockout_fixtures()


def cmd_scrape_xg(_args):
    from scrapers.scraper_xg import scrape_xg
    print("● Coletando xG via StatsBomb open data ...")
    scrape_xg()


def cmd_national_impact(_args):
    sys.path.insert(0, str(ROOT / "3_player_impact"))
    from national_impact import run
    print("● Impacto de jogador na seleção (StatsBomb, Fase 0 / Prompt C3) ...")
    run()


def cmd_scrape_understat(_args):
    from scrapers.scraper_players import scrape_understat_players
    print("● Coletando xG de jogador/time por temporada (Understat, D1) ...")
    scrape_understat_players()


def cmd_match_players(_args):
    from match_players import match_understat_to_transfermarkt
    print("● Matching jogador de clube -> seleção (Transfermarkt + rapidfuzz, D2) ...")
    match_understat_to_transfermarkt()


def cmd_build_squads(_args):
    sys.path.insert(0, str(ROOT / "3_player_impact"))
    from build_squad_2026 import build_probable_squads
    print("● Construindo squad_2026 aproximado (top-26 por minutagem de clube, D2) ...")
    build_probable_squads()


def cmd_compute_rapm(_args):
    sys.path.insert(0, str(ROOT / "3_player_impact"))
    from rapm import run
    print("● RAPM-lite (Ridge por temporada, D3) ...")
    run()


def cmd_transfer_impact(_args):
    sys.path.insert(0, str(ROOT / "3_player_impact"))
    from club_to_national import run
    print("● Transferência de impacto clube -> seleção (D4) ...")
    run()


def cmd_compute_squad_strength(_args):
    sys.path.insert(0, str(ROOT / "3_player_impact"))
    from squad_strength import run
    print("● Força de elenco 2026 + w_overlap (D5+D6.1) ...")
    run()


def cmd_squad_offset_check(args):
    from squad_offset_check import run
    print("● Revalidação com/sem squad_offset -- regra do IC (D7) ...")
    run(iters=args.iters)


def cmd_competition_weight_check(args):
    from competition_weight_check import run
    print("● Revalidação dos pesos por torneio (competition_weights) -- regra do IC (F4) ...")
    run(iters=args.iters)


def cmd_montecarlo(args):
    from montecarlo_sim import run, format_report, N_SIMS_DEFAULT, SEED_DEFAULT
    n_sims = args.sims or N_SIMS_DEFAULT
    strategy = args.strategy or "max_ev"
    other_strategy = "modal" if strategy == "max_ev" else "max_ev"
    print(f"● Simulação Monte Carlo (fase de grupos): {n_sims} torneios, "
          f"estratégia '{strategy}' (+ comparação com '{other_strategy}') ...")
    result = run(n_sims=n_sims, strategy=strategy, seed=SEED_DEFAULT)
    other = run(n_sims=n_sims, strategy=other_strategy, seed=SEED_DEFAULT)
    print()
    print(format_report(result, other_strategy_result=other))


def cmd_h2h_check(args):
    from h2h_check import run
    print("● H2H (confronto direto): cobertura nos 72 jogos da Copa 2026 + revalidação "
          "com/sem h2h -- regra do IC (F7) ...")
    run(iters=args.iters)


def cmd_calibration_check(args):
    from calibration_check import run
    print("● Calibração (ECE/reliability) -- as probabilidades batem com a frequência "
          "observada? (Auditoria M4/P8) ...")
    run(tournament_names=args.tournaments, model_names=args.models, iters=args.iters)


def cmd_blend_check(args):
    from market_blend_check import run
    print("● Validação do blend modelo × mercado (RPS modelo vs blend, IC da diferença) ...")
    run(iters=args.iters, verbose=True)


def cmd_build_multiples(args):
    sys.path.insert(0, str(ROOT / "utils"))
    from multiples import build
    print("● Montando múltiplas de maior probabilidade de acerto (próximos jogos) ...")
    build(profile=args.profile, legs=args.legs, round_n=args.round_n_multi)


def cmd_multiples_menu(args):
    from multiples_menu import run
    print("● Cardápio de múltiplas (1X2 + totals + BTTS, EV real) -- ranqueado por faixa "
          "de retorno, jogos da rodada atual (READ-ONLY) ...")
    run(round_n=args.round_n_multi, top=args.menu_top)


def cmd_multiples_backtest(args):
    from multiples_backtest import run
    print("● Backtest da estratégia de múltiplas (quantas de alta prob teriam batido) ...")
    run(profile=args.profile, k=args.legs)


def cmd_multiples_calibration(args):
    from multiples_calibration import run
    print("● Calibração do MOTOR de múltiplas (prob conjunta prevista vs observada) -- "
          "same-game pela grade + cross-game por produto, 3 recortes (READ-ONLY) ...")
    run(profile=args.profile, iters=args.iters)


def cmd_ev_multiples(args):
    sys.path.insert(0, str(ROOT / "utils"))
    from multiples import build_ev
    print("● Montando múltiplas de maior EV (modelo × odds reais) -- próximos jogos ...")
    build_ev(legs=args.legs, round_n=args.round_n_multi)


def cmd_markets_check(args):
    from markets_check import run
    print("● Validação de mercados (over/under total e por time, BTTS, 1X2) -- "
          "Brier Skill Score vs climatologia, walk-forward sem vazamento ...")
    run(model_name=args.models[0] if args.models else None, iters=args.iters)


def cmd_markets_compare(args):
    from markets_check import run_compare
    print("● Comparação de modelos no pool completo (grupos + mata-mata) -- "
          "1X2=RPS, binários=Brier, bootstrap PAREADO vs referência (READ-ONLY) ...")
    run_compare(model_names=args.models or None, iters=args.iters)


def cmd_wc2026_calibration(args):
    from wc2026_calibration import run
    print("● Confiança do modelo x desempenho real -- fase de grupos da Copa 2026 "
          "(walk-forward por rodada, sem vazamento) ...")
    run(model_name=args.models[0] if args.models else None)


def cmd_knockout_check(args):
    from knockout_check import run
    kinds = ("world_cup",) if args.knockout_scope == "wc" else None
    escopo = "só Copas do Mundo" if kinds else "Copas + Euro + Copa América"
    print(f"● Taxa de acerto (1X2) e RPS do modelo escolhido: fase de grupos vs. "
          f"mata-mata ({escopo}) ...")
    run(model_name=args.models[0] if args.models else None, iters=args.iters, kinds=kinds)


def cmd_cv_tournaments(args):
    from tournament_cv import run
    fold_kind = "all" if args.cv_scope == "all" else "world_cup"
    print(f"● Leave-one-tournament-out CV (escopo={fold_kind}): seleção nos outros "
          "torneios, teste no fold retido ...")
    run(fold_kind=fold_kind, iters=args.iters, verbose=True)


def cmd_update_round(args):
    print(f"● --update-round {args.n}: Iteração 2 (aprendizado bayesiano sequencial).")
    from sequential_backtest import update_round
    update_round(args.n, force=args.force)


def cmd_auto_round(args):
    sys.path.insert(0, str(ROOT / "utils"))
    from auto_round import run_auto_round
    print(f"● --auto-round {args.n}: update + Monte Carlo + push (Netlify) ...")
    run_auto_round(args.n, push=not args.no_push, force=args.force_update)


def cmd_fetch_results(_args):
    sys.path.insert(0, str(ROOT / "utils"))
    from results_fetcher import fetch_and_insert_results
    print("● Buscando placares reais da Copa 2026 (FONTE 1 -> 2 -> 3) ...")
    fetch_and_insert_results(verbose=True)


def cmd_check_db(_args):
    import db_client
    client = db_client.get_client()
    if client is None:
        print("  Supabase NÃO configurado (defina SUPABASE_URL/SUPABASE_KEY em .env).")
        print("  Pipeline operando 100% via CSV local (1_data/processed/matches_weighted.csv).")
        return
    print("● Verificando tabelas no Supabase ...")
    tables = ["matches", "players", "player_impact", "squad_2026",
              "predictions", "copa_2026_results",
              "player_seasons", "player_national_team", "team_seasons", "squad_strength"]
    for t in tables:
        try:
            resp = client.table(t).select("*", count="exact").limit(1).execute()
            print(f"  ✓ {t:18s} ({resp.count} linhas)")
        except Exception as e:
            print(f"  ✗ {t:18s} erro: {e}")
            print("    -> rode supabase/schema.sql no SQL Editor do projeto.")


def build_parser():
    p = argparse.ArgumentParser(description="Bolão Copa 2026 — pipeline (Iteração 1)")
    p.add_argument("--scrape", action="store_true", help="baixa dados (martj42)")
    p.add_argument("--prep", action="store_true", help="gera matches_weighted.csv")
    p.add_argument("--compare", action="store_true", help="comparação cruzada + relatório")
    p.add_argument("--select", action="store_true", help="seleção out-of-sample + outputs")
    p.add_argument("--predict-2026", dest="predict", action="store_true",
                   help="previsões da Copa 2026")
    p.add_argument("--update-round", dest="round_n", type=int, metavar="N",
                   help="[Iteração 2] aprendizado sequencial após a rodada N")
    p.add_argument("--force-update", action="store_true",
                   help="ignora jogos pendentes da Rodada N e força --update-round mesmo assim")
    p.add_argument("--auto-round", dest="auto_round_n", type=int, metavar="N",
                   help="[Deploy] update-round N + Monte Carlo + push p/ GitHub (rebuild Netlify)")
    p.add_argument("--no-push", action="store_true",
                   help="desabilita o push do --auto-round (atualiza só localmente)")
    p.add_argument("--fetch-results", action="store_true",
                   help="busca automaticamente placares terminados da Copa 2026 "
                        "(API-Football -> The Odds API -> FIFA.com) e insere via insert_result")
    p.add_argument("--check-db", action="store_true",
                   help="verifica conexão e tabelas do Supabase")
    p.add_argument("--odds-check", action="store_true",
                   help="cruza previsões com odds de mercado (The Odds API)")
    p.add_argument("--fetch-odds", action="store_true",
                   help="[Iteração 2 / Prompt E] busca odds de mercado (The Odds API) -> odds_2026")
    p.add_argument("--fetch-market-odds", action="store_true",
                   help="busca odds EXPANDIDAS (total O/U, BTTS, team totals) com de-vig -> "
                        "tabela odds_2026_markets (ISOLADO: não toca odds_2026 nem o bolão); "
                        "alimenta o gerador de múltiplas com EV real")
    p.add_argument("--insert-result", action="store_true",
                   help="[Iteração 2 / Prompt E] registra o placar real de um jogo da Copa 2026 "
                        "(use com --home --away --score --date)")
    p.add_argument("--home", type=str, default=None, help="time da casa (--insert-result)")
    p.add_argument("--away", type=str, default=None, help="time visitante (--insert-result)")
    p.add_argument("--score", nargs=2, type=int, default=None, metavar=("HOME", "AWAY"),
                   help="placar HOME AWAY (--insert-result)")
    p.add_argument("--date", type=str, default=None, metavar="YYYY-MM-DD",
                   help="data do jogo (--insert-result)")
    p.add_argument("--backtest-rounds", action="store_true",
                   help="[Iteração 2 / Prompt E5] backtest WC2022 por rodada "
                        "(sequencial vs estático) -- PAUSA antes da Copa 2026 ao vivo")
    p.add_argument("--scrape-fixtures", action="store_true",
                   help="baixa fixtures (round/grupo) da fase de grupos da Copa 2026")
    p.add_argument("--scrape-knockout-fixtures", action="store_true",
                   help="grava os confrontos do mata-mata (Round of 32) em matches + "
                        "copa_2026_results")
    p.add_argument("--scrape-xg", action="store_true",
                   help="coleta xG por jogo via StatsBomb open data (matches.home_xg/away_xg)")
    p.add_argument("--national-impact", action="store_true",
                   help="[Iteração 2 / Fase 0] impacto de jogador na seleção via StatsBomb (314 jogos)")
    p.add_argument("--scrape-understat", action="store_true",
                   help="[Iteração 2 / D1] xG de jogador/time por temporada via Understat (6 ligas x 3 temporadas)")
    p.add_argument("--match-players", action="store_true",
                   help="[Iteração 2 / D2] matching jogador de clube -> seleção (Transfermarkt + rapidfuzz)")
    p.add_argument("--build-squads", action="store_true",
                   help="[Iteração 2 / D2] squad_2026 aproximado (top-26 por minutagem de clube)")
    p.add_argument("--compute-rapm", action="store_true",
                   help="[Iteração 2 / D3] RAPM-lite (Ridge por temporada) -> player_impact(rapm_club)")
    p.add_argument("--transfer-impact", action="store_true",
                   help="[Iteração 2 / D4] blend clube+seleção (confiança por minutos) -> player_impact(final_2026)")
    p.add_argument("--compute-squad-strength", action="store_true",
                   help="[Iteração 2 / D5+D6.1] força de elenco 2026 + w_overlap (lineups StatsBomb) -> squad_strength")
    p.add_argument("--squad-offset-check", action="store_true",
                   help="[Iteração 2 / D7] revalidação com/sem squad_offset (regra do IC, todos os membros + ensemble)")
    p.add_argument("--competition-weight-check", action="store_true",
                   help="[Iteração 3 / F4] revalidação dos pesos por torneio competition_weights "
                        "(regra do IC, novo vs antigo, todos os membros + ensemble)")
    p.add_argument("--h2h-check", action="store_true",
                   help="[Iteração 3 / F7] cobertura de H2H nos 72 jogos da Copa 2026 + "
                        "revalidação com/sem h2h (regra do IC, WC2018/WC2022, todos os membros + ensemble)")
    p.add_argument("--calibration-check", action="store_true",
                   help="[Auditoria M4/P8] diagnóstico de calibração (ECE/reliability) por "
                        "modelo -- as probabilidades batem com a frequência observada?")
    p.add_argument("--blend-check", action="store_true",
                   help="[Blend modelo × mercado] valida RPS modelo vs blend (α fixo) nos jogos "
                        "da Copa com resultado real + odds; reporta IC da diferença (regra do IC)")
    p.add_argument("--build-multiples", action="store_true",
                   help="monta múltiplas de maior probabilidade de acerto para os próximos "
                        "jogos (mercados bem calibrados; mostra a prob. conjunta)")
    p.add_argument("--ev-multiples", action="store_true",
                   help="monta múltiplas de maior EV (prob. modelo × odd real, 1X2): combina "
                        "pernas de EV positivo (>=3); mostra EV e chance conjuntos")
    p.add_argument("--multiples-backtest", action="store_true",
                   help="backtest da estratégia de múltiplas nos torneios passados: prob "
                        "conjunta prevista vs taxa real de acerto (usa --profile e --legs)")
    p.add_argument("--multiples-calibration", action="store_true",
                   help="calibração do MOTOR de múltiplas (READ-ONLY): conjunta same-game pela "
                        "grade + cross-game por produto; previsto vs observado com IC (cluster-"
                        "bootstrap por jogo) em 3 recortes -- mata-mata, grupos+KO e por época "
                        "(WC2010/14/18/22). Usa --profile p/ a perna cross-game")
    p.add_argument("--multiples-menu", action="store_true",
                   help="cardápio rico de múltiplas (1X2 + totals + BTTS) com EV real "
                        "(odds dos 3 mercados): duplas/triplas same-game, cross-game e misto, "
                        "ranqueadas POR FAIXA DE RETORNO; usa --round-multi e --menu-top")
    p.add_argument("--menu-top", type=int, default=5, metavar="N",
                   help="nº de múltiplas por faixa de retorno no --multiples-menu (padrão: 5)")
    p.add_argument("--profile", choices=("seguro", "vitoria", "gols"), default="seguro",
                   help="perfil das pernas do --build-multiples: seguro=dupla chance, "
                        "vitoria=vitória seca, gols=favorito marca 1+ (padrão: seguro)")
    p.add_argument("--legs", type=int, default=3, metavar="N",
                   help="nº de pernas por múltipla no --build-multiples (padrão: 3)")
    p.add_argument("--round-multi", dest="round_n_multi", type=int, default=None, metavar="N",
                   help="restringe o --build-multiples aos jogos da rodada/fase N")
    p.add_argument("--markets-check", action="store_true",
                   help="valida mercados (over/under total e por time, BTTS, 1X2) derivados "
                        "da grade de placar: Brier Skill Score vs climatologia + calibração, "
                        "walk-forward sem vazamento nos torneios de teste")
    p.add_argument("--markets-compare", action="store_true",
                   help="compara modelos (padrão: poisson, dixon_coles, bivariate_poisson) no "
                        "pool completo (grupos + mata-mata): 1X2=RPS e binários=Brier por "
                        "mercado, com Δ vs referência (1º modelo) via bootstrap PAREADO + IC95%% "
                        "(READ-ONLY; use --models p/ trocar a lista, 1º = referência)")
    p.add_argument("--wc2026-calibration", action="store_true",
                   help="confiança do modelo (faixas de 10%%) x desempenho real (acerto "
                        "vitória/erro/placar cravado) nos jogos JÁ JOGADOS da fase de "
                        "grupos da Copa 2026, walk-forward por rodada (sem vazamento)")
    p.add_argument("--knockout-check", action="store_true",
                   help="taxa de acerto (1X2) + RPS do modelo escolhido (ou --models, 1º nome) "
                        "restrito aos jogos de mata-mata dos torneios de teste, vs. fase de "
                        "grupos (diagnóstico)")
    p.add_argument("--knockout-scope", choices=("wc", "all"), default="wc",
                   help="--knockout-check: 'wc' = só Copas do Mundo (padrão), 'all' = "
                        "também Euro + Copa América")
    p.add_argument("--cv-tournaments", action="store_true",
                   help="[Validação cruzada] leave-one-tournament-out CV: rotaciona o torneio "
                        "retido, seleciona nos demais e testa no retido (número honesto de "
                        "generalização). Padrão: só Copas (--cv-scope wc).")
    p.add_argument("--cv-scope", choices=("wc", "all"), default="wc",
                   help="folds do --cv-tournaments: 'wc' = só Copas do Mundo (padrão, 2 folds), "
                        "'all' = os 7 torneios (mais robusto)")
    p.add_argument("--montecarlo", action="store_true",
                   help="simulação Monte Carlo da distribuição de pontos/acertos na fase de "
                        "grupos (consome predictions_2026.json + utils/scoring.py)")
    p.add_argument("--sims", type=int, default=None, metavar="N",
                   help="nº de torneios simulados (--montecarlo, padrão 20000)")
    p.add_argument("--strategy", choices=("max_ev", "modal"), default=None,
                   help="estratégia de palpite (--montecarlo, padrão max_ev)")
    # Opções
    p.add_argument("--tournaments", nargs="*", default=None, help="subconjunto de torneios")
    p.add_argument("--models", nargs="*", default=None, help="subconjunto de modelos")
    p.add_argument("--lam", type=float, default=None, help="override do λ de decay temporal")
    p.add_argument("--iters", type=int, default=None, help="override de reamostras bootstrap")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    ran = False
    if args.scrape:
        cmd_scrape(args); ran = True
    if args.prep:
        cmd_prep(args); ran = True
    if args.compare:
        cmd_compare(args); ran = True
    if args.select:
        cmd_select(args); ran = True
    if args.predict:
        cmd_predict(args); ran = True
    if args.round_n is not None:
        class _A: n = args.round_n; force = args.force_update
        cmd_update_round(_A); ran = True
    if args.auto_round_n is not None:
        class _B: n = args.auto_round_n; no_push = args.no_push; force_update = args.force_update
        cmd_auto_round(_B); ran = True
    if args.check_db:
        cmd_check_db(args); ran = True
    if args.odds_check:
        cmd_odds_check(args); ran = True
    if args.fetch_odds:
        cmd_fetch_odds(args); ran = True
    if args.fetch_market_odds:
        cmd_fetch_market_odds(args); ran = True
    if args.insert_result:
        cmd_insert_result(args); ran = True
    if args.fetch_results:
        cmd_fetch_results(args); ran = True
    if args.backtest_rounds:
        cmd_backtest_rounds(args); ran = True
    if args.scrape_fixtures:
        cmd_fixtures(args); ran = True
    if args.scrape_knockout_fixtures:
        cmd_knockout_fixtures(args); ran = True
    if args.scrape_xg:
        cmd_scrape_xg(args); ran = True
    if args.national_impact:
        cmd_national_impact(args); ran = True
    if args.scrape_understat:
        cmd_scrape_understat(args); ran = True
    if args.match_players:
        cmd_match_players(args); ran = True
    if args.build_squads:
        cmd_build_squads(args); ran = True
    if args.compute_rapm:
        cmd_compute_rapm(args); ran = True
    if args.transfer_impact:
        cmd_transfer_impact(args); ran = True
    if args.compute_squad_strength:
        cmd_compute_squad_strength(args); ran = True
    if args.squad_offset_check:
        cmd_squad_offset_check(args); ran = True
    if args.competition_weight_check:
        cmd_competition_weight_check(args); ran = True
    if args.h2h_check:
        cmd_h2h_check(args); ran = True
    if args.calibration_check:
        cmd_calibration_check(args); ran = True
    if args.blend_check:
        cmd_blend_check(args); ran = True
    if args.build_multiples:
        cmd_build_multiples(args); ran = True
    if args.ev_multiples:
        cmd_ev_multiples(args); ran = True
    if args.multiples_menu:
        cmd_multiples_menu(args); ran = True
    if args.multiples_backtest:
        cmd_multiples_backtest(args); ran = True
    if args.multiples_calibration:
        cmd_multiples_calibration(args); ran = True
    if args.markets_check:
        cmd_markets_check(args); ran = True
    if args.markets_compare:
        cmd_markets_compare(args); ran = True
    if args.wc2026_calibration:
        cmd_wc2026_calibration(args); ran = True
    if args.knockout_check:
        cmd_knockout_check(args); ran = True
    if args.cv_tournaments:
        cmd_cv_tournaments(args); ran = True
    if args.montecarlo:
        cmd_montecarlo(args); ran = True
    if not ran:
        build_parser().print_help()


if __name__ == "__main__":
    main()
