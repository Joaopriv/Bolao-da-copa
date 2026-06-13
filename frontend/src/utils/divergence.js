// Divergência entre os modelos do model_breakdown (proxy de "odds divergence" sem
// mercado real) e bandas de divergência pp para a aba Odds Check.

const DIVERGENCE_THRESHOLD = 0.05; // >5pp entre o min e o max dos 5 modelos

// true se, em qualquer outcome (home/draw/away), o spread entre modelos > 5pp.
export function hasDivergence(modelBreakdown) {
  if (!modelBreakdown) return false;
  const outcomes = ["home", "draw", "away"];
  return outcomes.some((outcome) => {
    const values = Object.values(modelBreakdown).map((m) => m[outcome]);
    const max = Math.max(...values);
    const min = Math.min(...values);
    return max - min > DIVERGENCE_THRESHOLD;
  });
}

// Banda de cor para divergência modelo x mercado (em pontos percentuais).
export function oddsDivergenceBand(diffPp) {
  const abs = Math.abs(diffPp);
  if (abs < 3) return "good";
  if (abs <= 7) return "warn";
  return "bad";
}

// Alerta de divergência alta (>10pp) entre modelo e mercado.
export function isHighOddsDivergence(diffPp) {
  return Math.abs(diffPp) > 10;
}
