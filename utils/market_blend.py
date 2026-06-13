"""Blending das probabilidades 1X2 do modelo com as probabilidades implícitas
do mercado (odds já sem margem da casa).

Princípio (Occam): peso fixo α=0.5 (igual weight), SEM otimização nos dados da
Copa -- otimizar α na própria amostra de teste seria overfitting. O blend só
entra em produção se a validação out-of-sample (4_validation/market_blend_check.py)
confirmar melhora no RPS com IC95% bootstrap todo < 0.

Nota sobre `scoring.truncate_and_renormalize`: aquela função opera sobre a GRADE
de placares (matriz (M+1)x(M+1), P(home_goals, away_goals)) -- aqui lidamos com o
vetor 1X2 ({home, draw, away}, 3 categorias), então a renormalização é feita
localmente (mesma ideia: dividir pela soma para somar 1.0), não há reuso direto.
"""
from __future__ import annotations

_OUTCOMES = ("home", "draw", "away")


def blend_probabilities(
    model_probs: dict,
    market_probs: dict | None,
    alpha: float = 0.5,
) -> dict:
    """Mistura linear das probabilidades 1X2 do modelo com as do mercado.

    P_blend[o] = α × P_modelo[o] + (1-α) × P_mercado[o], para cada outcome
    o ∈ {home, draw, away}, renormalizado para somar exatamente 1.0.

    Args:
        model_probs:  {"home", "draw", "away"} do modelo (poisson etc.).
        market_probs: {"home", "draw", "away"} das odds implícitas (de-vig),
                      ou None se as odds daquele jogo não estão disponíveis.
        alpha:        peso do modelo (1-alpha = peso do mercado). Padrão 0.5.

    Returns:
        {"home", "draw", "away"} normalizado (soma 1.0). Se market_probs é None,
        devolve uma cópia de model_probs SEM alterar (fallback -- não penaliza o
        blend nos ~5/72 jogos sem odds).
    """
    if market_probs is None:
        return {o: float(model_probs[o]) for o in _OUTCOMES}

    blended = {
        o: alpha * float(model_probs[o]) + (1.0 - alpha) * float(market_probs[o])
        for o in _OUTCOMES
    }
    total = sum(blended.values())
    if total > 0:
        blended = {o: v / total for o, v in blended.items()}
    return blended
