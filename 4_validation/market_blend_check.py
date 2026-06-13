"""[Blend modelo × mercado / Etapa 2] Validação out-of-sample do blend de odds.

As odds históricas dos torneios de TREINO não existem no dataset -- só temos odds
dos 72 jogos da Copa 2026. Por isso a validação é feita DIRETAMENTE nos jogos da
Copa que JÁ TÊM resultado real (copa_2026_results.home_score IS NOT NULL) E odds
disponíveis (predictions_2026.json.odds_implied != null), comparando o RPS do
modelo puro vs o RPS do blend (α=0.5 fixo, Occam -- sem otimização na amostra).

Decisão (regra do IC, igual ao resto do projeto):
  - IC95% da diferença (blend - modelo) TODO < 0  -> blend melhor  -> APROVADO
  - IC95% cruza zero                              -> equivalente  -> DESCARTADO (Occam)
  - IC95% TODO > 0                                -> blend pior   -> DESCARTADO

Com N pequeno (Rodada 1 ~ 4 jogos) o IC é largo e o resultado provavelmente
inconclusivo -- isso é esperado e reportado explicitamente. O poder estatístico
cresce a cada rodada (--update-round N -> re-rodar --blend-check).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_config, path  # noqa: E402
import db_client  # noqa: E402
import metrics  # noqa: E402
from market_blend import blend_probabilities  # noqa: E402

_OUTCOMES = ("home", "draw", "away")
REPORT_PATH = ROOT / "4_validation" / "market_blend_report.txt"

# Abaixo deste N, o bootstrap não tem poder para sustentar um veredito: com N=1 o
# IC é DEGENERADO (toda reamostra é idêntica -> largura zero, "APROVADO" espúrio) e
# com N pequeno (2-4) ele é altamente instável -- um IC que por acaso não cruza
# zero seria falso-positivo (o exato overfitting que a regra do IC existe p/ evitar).
# O próprio plano antecipa decisão confiável só com ~30-50 jogos (Rodada 2-3).
MIN_N_FOR_DECISION = 30


def _probs_to_array(d: dict) -> np.ndarray:
    """{home, draw, away} -> np.array([H, D, A]) (ordem que rps_vector espera)."""
    return np.array([d["home"], d["draw"], d["away"]], dtype=float)


def _outcome_index(home_score: int, away_score: int) -> int:
    """Resultado 1X2 como índice (H=0, D=1, A=2)."""
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def _load_predictions() -> list[dict]:
    p = path("5_outputs", "predictions_2026.json")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)["predictions"]


def _build_pred_index(preds: list[dict], disp: dict) -> dict:
    """Indexa as previsões por (home_pt, away_pt, date) -- a chave que dá para
    reconstruir a partir de um resultado em copa_2026_results (nomes EN -> PT)."""
    return {(p["home_team"], p["away_team"], p["date"]): p for p in preds}


def collect(alpha: float | None = None, verbose: bool = True) -> dict:
    """Reúne os jogos com resultado real + odds e calcula RPS modelo vs blend.

    Retorna dict com vetores de RPS por jogo, amostra e contagens de exclusão.
    """
    cfg = load_config()
    if alpha is None:
        alpha = cfg.get("market_blend", {}).get("alpha", 0.5)
    disp = cfg.get("display_names", {})

    preds = _load_predictions()
    pred_index = _build_pred_index(preds, disp)

    results = db_client.fetch_all("copa_2026_results") or []
    played = [r for r in results if r.get("home_score") is not None]

    matched = []          # (game_label, P_model, P_market, y)
    no_odds = []          # jogos com resultado mas SEM odds
    unmatched = []        # resultados sem previsão correspondente (não deveria ocorrer)

    for r in played:
        home_pt = disp.get(r["home_team"], r["home_team"])
        away_pt = disp.get(r["away_team"], r["away_team"])
        date_str = str(r["date"])[:10]
        pred = pred_index.get((home_pt, away_pt, date_str))
        if pred is None:
            unmatched.append(f"{home_pt} vs {away_pt} ({date_str})")
            continue
        label = f"{home_pt} {r['home_score']}×{r['away_score']} {away_pt}"
        if not pred.get("odds_implied"):
            no_odds.append(label)
            continue
        y = _outcome_index(r["home_score"], r["away_score"])
        matched.append((label, pred["result_probs"], pred["odds_implied"], y))

    if not matched:
        return {
            "alpha": alpha, "n": 0, "matched": matched,
            "no_odds": no_odds, "unmatched": unmatched,
            "rps_model_vec": np.array([]), "rps_blend_vec": np.array([]),
        }

    P_model = np.vstack([_probs_to_array(m) for _, m, _, _ in matched])
    P_market = np.vstack([_probs_to_array(k) for _, _, k, _ in matched])
    P_blend = np.vstack([
        _probs_to_array(blend_probabilities(m, k, alpha))
        for _, m, k, _ in matched
    ])
    y = np.array([yy for _, _, _, yy in matched], dtype=int)

    rps_model_vec = metrics.rps_vector(P_model, y)
    rps_blend_vec = metrics.rps_vector(P_blend, y)

    return {
        "alpha": alpha,
        "n": len(matched),
        "matched": matched,
        "no_odds": no_odds,
        "unmatched": unmatched,
        "rps_model_vec": rps_model_vec,
        "rps_blend_vec": rps_blend_vec,
    }


def run(iters: int | None = None, verbose: bool = True) -> dict:
    """Validação completa: RPS modelo vs blend + IC95% bootstrap da diferença.

    Reporta em PT-BR, grava 4_validation/market_blend_report.txt e devolve o dict
    de resultados (inclui a decisão APROVADO/DESCARTADO)."""
    cfg = load_config()
    iters = iters or cfg["validation"]["bootstrap_iterations"]
    pcts = tuple(cfg["validation"]["ci_percentiles"])
    seed = cfg["validation"]["random_seed"]

    data = collect(verbose=verbose)
    lines = _format_report(data, iters=iters, percentiles=pcts, seed=seed)
    report = "\n".join(lines)

    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    if verbose:
        print(report)
        print(f"\n  relatório salvo em {REPORT_PATH.relative_to(ROOT)}")

    # Anexa a decisão estruturada ao dict retornado (caso já calculada).
    data["report_path"] = str(REPORT_PATH)
    if data["n"] > 0:
        diff = data["rps_blend_vec"] - data["rps_model_vec"]
        ci = metrics.bootstrap_ci(diff, name="Δ(blend-modelo)", lower_is_better=True,
                                  iters=iters, percentiles=pcts, seed=seed)
        data["diff_ci"] = ci
        data["decision"] = _decide(ci, data["n"])
    else:
        data["decision"] = "INCONCLUSIVO (0 jogos com resultado + odds)"
    return data


def _decide(ci: "metrics.Score", n: int) -> str:
    """Regra do IC sobre a diferença (blend - modelo), com guarda de amostra mínima.

    Abaixo de MIN_N_FOR_DECISION o IC bootstrap não é confiável (degenerado em N=1,
    instável em N pequeno) -- nunca declaramos APROVADO; o veredito fica suspenso."""
    if n < MIN_N_FOR_DECISION:
        return (f"INCONCLUSIVO (N={n} < {MIN_N_FOR_DECISION} -> amostra insuficiente "
                f"para um IC confiável; aguardar mais rodadas). Mantém enabled=false.")
    if ci.hi < 0:
        return "APROVADO (IC95% todo < 0 -> blend melhora o RPS)"
    if ci.lo > 0:
        return "DESCARTADO (IC95% todo > 0 -> blend PIORA o RPS)"
    return "DESCARTADO por Occam (IC95% cruza zero -> equivalente, sem ganho comprovado)"


def _format_report(data: dict, *, iters: int, percentiles, seed: int) -> list[str]:
    L = []
    L.append("━━ VALIDAÇÃO DO BLEND MODELO × MERCADO — COPA 2026 ━━")
    L.append("")
    L.append(f" Abordagem A (Occam): P_blend = α·P_modelo + (1-α)·P_mercado, "
             f"α={data['alpha']} fixo (sem otimização).")
    L.append(" Teste direto nos jogos da Copa 2026 com resultado real + odds disponíveis.")
    L.append("")

    n = data["n"]
    L.append(f" AMOSTRA")
    L.append(f"   jogos com resultado real + odds .... {n}")
    if data["no_odds"]:
        L.append(f"   excluídos (resultado mas SEM odds) .. {len(data['no_odds'])}:")
        for g in data["no_odds"]:
            L.append(f"       {g}")
    if data["unmatched"]:
        L.append(f"   ⚠ sem previsão correspondente ....... {len(data['unmatched'])}:")
        for g in data["unmatched"]:
            L.append(f"       {g}")
    L.append("")

    if n == 0:
        L.append(" Nenhum jogo com resultado real + odds ainda -- valide após a Rodada 1.")
        return L

    rps_m = data["rps_model_vec"]
    rps_b = data["rps_blend_vec"]
    diff = rps_b - rps_m

    L.append(" RPS POR JOGO (menor é melhor)")
    L.append(f"   {'jogo':<42s} {'modelo':>8s} {'blend':>8s} {'Δ':>8s}")
    for (label, _, _, _), rm, rb in zip(data["matched"], rps_m, rps_b):
        L.append(f"   {label:<42s} {rm:>8.4f} {rb:>8.4f} {rb - rm:>+8.4f}")
    L.append("")

    L.append(" RPS MÉDIO")
    L.append(f"   modelo puro ........ {rps_m.mean():.4f}")
    L.append(f"   blend (α={data['alpha']}) ..... {rps_b.mean():.4f}")
    L.append(f"   diferença média .... {diff.mean():+.4f}  (negativo = blend melhor)")
    L.append("")

    ci = metrics.bootstrap_ci(diff, name="Δ", lower_is_better=True,
                              iters=iters, percentiles=percentiles, seed=seed)
    lo_p, hi_p = percentiles
    L.append(f" IC{hi_p - lo_p:.0f}% BOOTSTRAP DA DIFERENÇA (blend - modelo, {iters} reamostras)")
    L.append(f"   ponto: {ci.point:+.4f}  IC: [{ci.lo:+.4f}, {ci.hi:+.4f}]")
    L.append("")

    L.append(f" DECISÃO: {_decide(ci, n)}")
    L.append("")

    if n < MIN_N_FOR_DECISION:
        L.append(f" ⚠ AVISO HONESTO: N={n} é pequeno demais para um veredito.")
        if n == 1:
            L.append("   Com N=1 o bootstrap é DEGENERADO (toda reamostra é o mesmo jogo ->")
            L.append("   IC de largura zero). O 'IC todo < 0' acima é artefato, NÃO evidência.")
        else:
            L.append("   Com N pequeno o IC é instável -- um IC que não cruza zero aqui")
            L.append("   pode ser falso-positivo (o overfitting que a regra do IC evita).")
        L.append(f"   Decisão suspensa até N >= {MIN_N_FOR_DECISION} (~Rodada 2-3). O teste")
        L.append("   ganha poder a cada rodada: re-rodar --blend-check após --update-round.")
    return L


if __name__ == "__main__":
    run(verbose=True)
