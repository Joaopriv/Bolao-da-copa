"""Significância estatística — a FERRAMENTA DE DECISÃO do projeto.

- Teste t pareado sobre o diferencial de RPS entre dois modelos (mesmas partidas).
- Bootstrap da DIFERENÇA de RPS entre dois modelos -> IC da diferença.
- Funil de eliminação: classifica cada modelo vs o melhor e escolhe entre os equivalentes
  o MAIS SIMPLES (navalha de Occam).

Regra (sobre o IC da diferença `rps_modelo - rps_melhor`, positivo = modelo pior):
  IC todo > 0 (não cruza zero)  -> modelo claramente PIOR  -> DESCARTADO
  IC cruza zero                 -> empate estatístico       -> EQUIVALENTE
Entre os equivalentes (inclui o melhor), escolhe-se o de menor complexidade.

[Auditoria P2] Limitação de poder estatístico: a amostra de validação (~169 jogos)
é pequena -- "equivalente" significa apenas que o IC da diferença cruza zero, i.e.
os dados não bastam para DISTINGUIR os modelos, não que sejam comprovadamente iguais
(erro tipo II é provável). Mais jogos estreitariam o IC e poderiam eliminar modelos
hoje classificados como equivalentes.

[Auditoria P7] Winner's curse: `chosen` é escolhido com base no MESMO RPS de
validação usado no funil -- mesmo entre "equivalentes", o vencedor tende a ter sido
parcialmente favorecido por ruído da amostra de validação, então `in_sample` tende a
ser otimista. Por isso `select()` testa o escolhido UMA ÚNICA vez nos torneios
out-of-sample (sem reotimizar nada) -- esse número é o que deve guiar expectativas
futuras (ver `build_justification`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import bootstrap_ci  # noqa: E402

# Ordem de complexidade (menor = mais simples). Escolha metodológica explícita:
# o desempate entre modelos equivalentes favorece o mais simples.
COMPLEXITY = {
    "baseline_naive": 0,
    "baseline_ranking": 1,
    "poisson": 2,
    "elo": 3,
    "dixon_coles": 4,
    "bivariate_poisson": 5,
    "dixon_coles_xg": 6,
    "bayesian": 7,
    "ensemble": 9,
}


def paired_t_test(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """[Auditoria P4] Teste t pareado sobre o diferencial de perda (loss_a - loss_b),
    mesmas partidas. Retorna (estatística, p-valor).

    H0: os dois modelos têm perda média igual. Usa t de Student com n-1 g.l. (mais
    conservador para amostra pequena).

    Nota: NÃO é o teste de Diebold-Mariano -- este último usa um estimador de
    variância HAC/Newey-West para corrigir a autocorrelação dos erros de previsão
    (relevante em previsões h-passos-à-frente). Aqui as observações são tratadas
    como independentes; o nome anterior (`dm_test`) sugeria HAC sem implementá-lo.
    Como ferramenta de decisão, o IC bootstrap em `compare_pair`/`elimination_funnel`
    é o critério primário -- este teste t é informativo, não decisório.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3 or np.allclose(d.std(ddof=1), 0):
        return 0.0, 1.0
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


def compare_pair(rps_a, rps_b, name_a, name_b, *, iters=1000, seed=42) -> dict:
    """Compara dois modelos: IC bootstrap da diferença de RPS + DM.

    [Auditoria P15] `seed` é o MESMO (42, default) em todos os pares chamados por
    `elimination_funnel` -- as réplicas de bootstrap reamostram os MESMOS índices de
    jogos em cada par (mesmo n). Isso é "common random numbers": cada IC individual
    continua válido (a reamostragem é sobre os jogos, não sobre os modelos), mas
    comparações entre pares ficam correlacionadas -- o que REDUZ ruído ao comparar
    rankings entre pares (decisão do funil), em vez de adicioná-lo. Decisão (não-ação):
    seeds independentes por par removeriam essa correlação útil sem ganho claro, e
    mudariam os ICs publicados em `selected_model.json` (risco à RESTRIÇÃO de manter
    `chosen='poisson'`) -- mantido como está.
    """
    d = np.asarray(rps_a, float) - np.asarray(rps_b, float)
    sc = bootstrap_ci(d, name=f"{name_a}-{name_b}", lower_is_better=True, iters=iters, seed=seed)
    t_stat, t_p = paired_t_test(rps_a, rps_b)
    crosses_zero = sc.lo <= 0 <= sc.hi
    return {
        "model_a": name_a, "model_b": name_b,
        "diff_mean": sc.point, "diff_lo": sc.lo, "diff_hi": sc.hi,
        "crosses_zero": crosses_zero,
        "t_stat": t_stat, "t_p": t_p,
    }


def elimination_funnel(rps_by_model: dict[str, np.ndarray], *, candidates=None,
                       iters=1000, seed=42) -> dict:
    """Funil de eliminação a partir dos vetores de RPS por jogo (alinhados entre modelos).

    `candidates`: modelos ELEGÍVEIS a serem escolhidos (default: todos). Os baselines NÃO
    devem ser candidatos — são pisos de sanidade, aparecem na tabela e podem ser
    eliminados, mas nunca são o "modelo escolhido".

    O `best_by_rps` é o de menor RPS ENTRE OS CANDIDATOS; todos os modelos são comparados
    a ele. Retorna: best, eliminated[], equivalent[], chosen, e detalhes de cada par.
    """
    names = list(rps_by_model)
    cand = list(candidates) if candidates is not None else names
    means = {n: float(np.nanmean(rps_by_model[n])) for n in names}
    best = min(cand, key=lambda n: means[n])  # melhor entre candidatos (nunca baseline)

    eliminated, equivalent, details = [], [], {}
    for n in names:
        if n == best:
            equivalent.append(n)
            continue
        cmp = compare_pair(rps_by_model[n], rps_by_model[best], n, best, iters=iters, seed=seed)
        details[n] = cmp
        # diff = rps_n - rps_best ; positivo = n pior. Se o IC todo > 0 -> claramente pior.
        if cmp["diff_lo"] > 0:
            eliminated.append(n)
        else:
            equivalent.append(n)

    # Escolhe o mais simples ENTRE OS EQUIVALENTES QUE SÃO CANDIDATOS (exclui baselines).
    eligible = [n for n in equivalent if n in cand]
    chosen = min(eligible, key=lambda n: (COMPLEXITY.get(n, 50), means[n]))

    if "dixon_coles_xg" in eliminated:
        print("  m3 eliminado — possível causa: distorção por arredondamento "
              "xG->int em jogos de baixo xG (esperado).")

    return {
        "means": means,
        "best_by_rps": best,
        "eliminated": eliminated,
        "equivalent": sorted(equivalent, key=lambda n: COMPLEXITY.get(n, 50)),
        "chosen": chosen,
        "details": details,
    }
