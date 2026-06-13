"""Simulação Monte Carlo da Copa 2026 (fase de grupos) — distribuição de pontos.

NÃO re-treina nada: consome `5_outputs/predictions_2026.json` (score_matrix já
calculada pelo modelo escolhido) e a regra de pontos de `utils/scoring.score()`.

Para cada jogo, `score_matrix[h][a]` é tratada como a probabilidade "verdadeira"
do placar h-a. Em cada torneio simulado, sorteia-se UM placar real por jogo
(np.random.choice sobre as células achatadas) e pontua-se o palpite do bolão
contra esse placar. Repetindo N_SIMS vezes obtém-se a DISTRIBUIÇÃO de pontos e
acertos — não um número único.

Premissas (ver `format_report` para o texto completo ao usuário):
  1. Assume o modelo calibrado (score_matrix = verdade). Mal calibrado -> pior na vida real.
  2. Cobre só os 72 jogos da fase de grupos; mata-mata é extrapolação grosseira.
  3. score_matrix vem truncada (grid 6x6, gols 0-5) — placares mais altos saem da amostragem.
  4. Jogos são tratados como independentes (sem correlação entre rodadas/times).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

from config_loader import path  # noqa: E402
import scoring  # noqa: E402

N_SIMS_DEFAULT = 20000
SEED_DEFAULT = 42
STRATEGIES = ("max_ev", "modal")
MODEL_MAX_GOALS = 10  # grid completo do modelo (config.yaml: predict_2026.max_goals)
KNOCKOUT_GAMES = 104 - 72  # jogos de mata-mata na Copa de 48 seleções


def _load_predictions(predictions_path=None) -> dict:
    p = Path(predictions_path) if predictions_path else path("5_outputs", "predictions_2026.json")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _pick_guess(pred: dict, strategy: str) -> tuple[int, int]:
    """Retorna (palpite_h, palpite_a) segundo a estratégia escolhida."""
    if strategy == "max_ev":
        h, a = pred["top_scores"][0]["score"].split("-")
        return int(h), int(a)
    if strategy == "modal":
        sm = np.asarray(pred["score_matrix"], dtype=float)
        h, a = np.unravel_index(np.argmax(sm), sm.shape)
        return int(h), int(a)
    raise ValueError(f"strategy desconhecida: {strategy!r} (use 'max_ev' ou 'modal')")


def _outcome_tables(g: int, palpite_h: int, palpite_a: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tabelas g x g (pontos, acertou_1X2, placar_exato) para um palpite fixo,
    via `scoring.score()` -- g*g chamadas por jogo, depois indexação vetorizada
    sobre os placares reais sorteados."""
    pontos = np.zeros((g, g), dtype=np.int64)
    placar_exato = np.zeros((g, g), dtype=bool)
    for real_h in range(g):
        for real_a in range(g):
            pts = scoring.score(palpite_h, palpite_a, real_h, real_a)
            pontos[real_h, real_a] = pts
            placar_exato[real_h, real_a] = (palpite_h == real_h and palpite_a == real_a)
    acertou_1x2 = pontos > 0  # na regra do bolão, qualquer resultado certo vale >= 10
    return pontos, acertou_1x2, placar_exato


def _resumo(arr: np.ndarray) -> dict:
    """Distribuição de uma métrica: nunca um número só."""
    return {
        "media": float(np.mean(arr)),
        "desvio": float(np.std(arr)),
        "mediana": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def run(n_sims: int = N_SIMS_DEFAULT, strategy: str = "max_ev", seed: int = SEED_DEFAULT,
        predictions_path: str | None = None) -> dict:
    """Roda N_SIMS torneios simulados da fase de grupos. Retorna vetores brutos,
    resumos por distribuição e checagens de sanidade.

    Para um mesmo `seed`, os placares reais sorteados por jogo são IDÊNTICOS entre
    estratégias (o sorteio depende só de score_matrix, não do palpite) — isso
    permite comparar max_ev vs modal sobre os MESMOS torneios simulados.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy desconhecida: {strategy!r} (use 'max_ev' ou 'modal')")

    data = _load_predictions(predictions_path)
    preds = data["predictions"]
    n_games = len(preds)
    rng = np.random.default_rng(seed)

    pontos_totais = np.zeros(n_sims, dtype=np.int64)
    n_resultados_certos = np.zeros(n_sims, dtype=np.int64)
    n_placares_exatos = np.zeros(n_sims, dtype=np.int64)

    grid_sizes = set()
    soma_score_matrix = []
    max_result_probs = []

    for pred in preds:
        sm = np.asarray(pred["score_matrix"], dtype=float)
        g = sm.shape[0]
        grid_sizes.add(g)

        flat = sm.reshape(-1)
        total = float(flat.sum())
        soma_score_matrix.append(total)
        flat_norm = flat / total  # renormaliza p/ somar exatamente 1 (arredondamento do JSON)

        palpite_h, palpite_a = _pick_guess(pred, strategy)
        pontos_tab, acertou_tab, placar_tab = _outcome_tables(g, palpite_h, palpite_a)

        idx = rng.choice(flat_norm.size, size=n_sims, p=flat_norm)
        real_h, real_a = np.divmod(idx, g)

        pontos_totais += pontos_tab[real_h, real_a]
        n_resultados_certos += acertou_tab[real_h, real_a]
        n_placares_exatos += placar_tab[real_h, real_a]

        rp = pred.get("result_probs")
        if rp:
            max_result_probs.append(max(rp.values()))

    taxa_1x2 = n_resultados_certos / n_games

    taxa_1x2_media_simulada = float(np.mean(taxa_1x2))
    media_max_result_probs = float(np.mean(max_result_probs)) if max_result_probs else None
    diff_pp = (abs(taxa_1x2_media_simulada - media_max_result_probs) * 100
               if media_max_result_probs is not None else None)

    max_goals_grid = min(grid_sizes) - 1 if grid_sizes else None
    truncado = max_goals_grid is not None and max_goals_grid < MODEL_MAX_GOALS

    return {
        "n_sims": n_sims,
        "strategy": strategy,
        "seed": seed,
        "n_games": n_games,
        "vetores": {
            "pontos_totais": pontos_totais,
            "n_resultados_certos": n_resultados_certos,
            "n_placares_exatos": n_placares_exatos,
            "taxa_1x2": taxa_1x2,
        },
        "resumo": {
            "pontos_totais": _resumo(pontos_totais),
            "n_resultados_certos": _resumo(n_resultados_certos),
            "n_placares_exatos": _resumo(n_placares_exatos),
            "taxa_1x2": _resumo(taxa_1x2),
        },
        "sanity_check": {
            "taxa_1x2_media_simulada": taxa_1x2_media_simulada,
            "media_max_result_probs": media_max_result_probs,
            "diff_pp": diff_pp,
            "ok": diff_pp is not None and diff_pp <= 2.0,
        },
        "grid_info": {
            "grid_sizes": sorted(grid_sizes),
            "max_goals_grid": max_goals_grid,
            "truncado": truncado,
            "soma_score_matrix_min": min(soma_score_matrix) if soma_score_matrix else None,
            "soma_score_matrix_max": max(soma_score_matrix) if soma_score_matrix else None,
        },
    }


def _ascii_histogram(arr: np.ndarray, bins: int = 16, width: int = 40) -> str:
    counts, edges = np.histogram(arr, bins=bins)
    max_count = counts.max() if counts.size else 0
    lines = []
    for i, c in enumerate(counts):
        bar_len = int(round(c / max_count * width)) if max_count else 0
        bar = "█" * bar_len
        lines.append(f"  {edges[i]:6.0f} – {edges[i + 1]:6.0f} | {bar} {c}")
    return "\n".join(lines)


def format_report(result: dict, other_strategy_result: dict | None = None) -> str:
    """Relatório PT-BR honesto: sempre faixas/distribuições, nunca um número só."""
    L = []
    res = result["resumo"]
    n_games = result["n_games"]

    L.append("━━ SIMULAÇÃO MONTE CARLO — COPA 2026 (FASE DE GRUPOS) ━━")
    L.append("")
    L.append(f" N_SIMS={result['n_sims']} | estratégia='{result['strategy']}' "
             f"| seed={result['seed']} | jogos={n_games}")
    L.append("")

    # --- Premissas ---
    L.append(" PREMISSAS (leia antes de confiar nos números abaixo)")
    L.append(" 1. Assume-se que o MODELO ESTÁ CALIBRADO, ou seja, que score_matrix")
    L.append("    descreve corretamente as probabilidades reais. Se o modelo estiver")
    L.append("    mal calibrado, as taxas de acerto NA VIDA REAL serão PIORES que")
    L.append("    aqui. Isto é um teto otimista condicional ao modelo, não uma")
    L.append("    previsão da realidade.")
    L.append(f" 2. predictions_2026.json cobre só os {n_games} jogos da FASE DE GRUPOS")
    L.append("    (o mata-mata depende dos classificados). Os números abaixo valem")
    L.append("    para a fase de grupos; uma extrapolação para o mata-mata aparece")
    L.append("    ao final, com ressalva de que jogos eliminatórios costumam ser")
    L.append("    mais parelhos (taxa de acerto MENOR).")
    gi = result["grid_info"]
    if gi["truncado"]:
        L.append(f" 3. score_matrix vem truncada em grid {gi['max_goals_grid'] + 1}x"
                 f"{gi['max_goals_grid'] + 1} (gols 0-{gi['max_goals_grid']}), enquanto o")
        L.append(f"    modelo completo usa até {MODEL_MAX_GOALS} gols. Placares com gols >"
                 f" {gi['max_goals_grid']} para")
        L.append("    qualquer um dos times ficam FORA da amostragem (massa de")
        L.append("    probabilidade desprezível em futebol, mas registrado aqui por")
        L.append("    honestidade). As células foram renormalizadas para somar 1.0")
        L.append(f"    (soma original por jogo entre {gi['soma_score_matrix_min']:.4f} e "
                 f"{gi['soma_score_matrix_max']:.4f}).")
    L.append(" 4. Jogos são tratados como INDEPENDENTES entre si (correlações reais —")
    L.append("    ex. fadiga, motivação, lesões entre rodadas — são ignoradas).")
    L.append("    Razoável como aproximação para a fase de grupos.")
    L.append("")

    # --- Validação de sanidade ---
    sc = result["sanity_check"]
    L.append(" VALIDAÇÃO DE SANIDADE")
    L.append(f"   taxa_1X2 média simulada .......... {sc['taxa_1x2_media_simulada']:.4f} "
             f"({sc['taxa_1x2_media_simulada'] * 100:.1f}%)")
    if sc["media_max_result_probs"] is not None:
        L.append(f"   média de max(result_probs) ...... {sc['media_max_result_probs']:.4f} "
                 f"({sc['media_max_result_probs'] * 100:.1f}%)")
        status = "✓ OK (≤ 2pp)" if sc["ok"] else "⚠ ATENÇÃO (> 2pp — checar amostragem)"
        L.append(f"   diferença ........................ {sc['diff_pp']:.2f}pp  {status}")
    L.append("")

    # --- Distribuições ---
    pt = res["pontos_totais"]
    L.append(f" DISTRIBUIÇÃO — PONTOS TOTAIS ({n_games} jogos)")
    L.append(f"   média ± desvio: {pt['media']:.1f} ± {pt['desvio']:.1f}")
    L.append(f"   mediana: {pt['mediana']:.0f}")
    L.append(f"   percentis -> p5={pt['p5']:.0f} | p25={pt['p25']:.0f} | "
             f"p75={pt['p75']:.0f} | p95={pt['p95']:.0f}")
    L.append(f"   faixa provável (p5-p95): {pt['p5']:.0f} a {pt['p95']:.0f} pontos")
    L.append("")
    L.append("   histograma (pontos totais por torneio simulado):")
    L.append(_ascii_histogram(result["vetores"]["pontos_totais"]))
    L.append("")

    t1 = res["taxa_1x2"]
    nr = res["n_resultados_certos"]
    L.append(" DISTRIBUIÇÃO — TAXA DE ACERTO DE RESULTADO (1X2)")
    L.append(f"   taxa: mediana {t1['mediana'] * 100:.0f}%, faixa provável (p5-p95) "
             f"{t1['p5'] * 100:.0f}%-{t1['p95'] * 100:.0f}%")
    L.append(f"   média ± desvio: {t1['media'] * 100:.1f}% ± {t1['desvio'] * 100:.1f}%")
    L.append(f"   em nº de jogos (de {n_games}): mediana {nr['mediana']:.0f}, "
             f"faixa provável {nr['p5']:.0f} a {nr['p95']:.0f}")
    L.append(f"   -> em metade dos cenários você acerta entre {nr['p25']:.0f} e "
             f"{nr['p75']:.0f} resultados.")
    L.append("")

    pe = res["n_placares_exatos"]
    L.append(" DISTRIBUIÇÃO — PLACARES EXATOS")
    L.append(f"   mediana: {pe['mediana']:.0f} | faixa provável (p5-p95): "
             f"{pe['p5']:.0f} a {pe['p95']:.0f}")
    L.append(f"   média ± desvio: {pe['media']:.2f} ± {pe['desvio']:.2f}")
    L.append("")

    # --- Comparação de estratégias ---
    if other_strategy_result is not None:
        a, b = result, other_strategy_result
        pa = a["resumo"]["pontos_totais"]["media"]
        pb = b["resumo"]["pontos_totais"]["media"]
        L.append(" COMPARAÇÃO DE ESTRATÉGIAS (pontos totais médios)")
        L.append(f"   {a['strategy']:>8s}: {pa:.2f}")
        L.append(f"   {b['strategy']:>8s}: {pb:.2f}")
        melhor = a["strategy"] if pa >= pb else b["strategy"]
        L.append(f"   -> '{melhor}' rende mais pontos esperados nesta simulação.")
        if melhor != "max_ev":
            L.append("   ⚠ esperado era 'max_ev' vencer (é o palpite que maximiza EV "
                     "célula a célula) — investigar.")
        L.append("")

    # --- Extrapolação mata-mata ---
    media_por_jogo_pts = pt["media"] / n_games
    media_por_jogo_taxa = t1["media"]
    proj_pts_mata_mata = media_por_jogo_pts * KNOCKOUT_GAMES
    proj_taxa_mata_mata = media_por_jogo_taxa * KNOCKOUT_GAMES
    L.append(f" EXTRAPOLAÇÃO PARA O MATA-MATA ({KNOCKOUT_GAMES} jogos) — ESTIMATIVA GROSSEIRA")
    L.append(f"   se a mesma taxa média da fase de grupos se mantivesse: "
             f"~{proj_pts_mata_mata:.0f} pontos extras, ~{proj_taxa_mata_mata:.0f} "
             f"resultados certos.")
    L.append("   ⚠ ressalva: jogos eliminatórios tendem a ser MAIS PARELHOS (times de")
    L.append("   nível mais próximo) — a taxa de acerto real no mata-mata é")
    L.append("   provavelmente MENOR que esta extrapolação linear. Tratar como teto,")
    L.append("   não como projeção.")

    return "\n".join(L)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Simulação Monte Carlo — Copa 2026 (fase de grupos)")
    ap.add_argument("--sims", type=int, default=N_SIMS_DEFAULT, help="nº de torneios simulados")
    ap.add_argument("--strategy", choices=STRATEGIES, default="max_ev", help="estratégia de palpite")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT, help="seed (reprodutibilidade)")
    args = ap.parse_args()

    other_strategy = "modal" if args.strategy == "max_ev" else "max_ev"
    print(f"● Simulando {args.sims} torneios ('{args.strategy}' + comparação com "
          f"'{other_strategy}') ...")
    main_result = run(n_sims=args.sims, strategy=args.strategy, seed=args.seed)
    other_result = run(n_sims=args.sims, strategy=other_strategy, seed=args.seed)
    print()
    print(format_report(main_result, other_strategy_result=other_result))
