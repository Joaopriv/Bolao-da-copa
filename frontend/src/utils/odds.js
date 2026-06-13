// Conversão de odds decimais (1X2) para probabilidades implícitas (de-vig
// multiplicativo: normaliza 1/odd para somar 1 — remove o overround).

export function impliedProbs({ home, draw, away }) {
  const raw = [home, draw, away].map((o) => (o > 1 ? 1 / o : null));
  if (raw.some((v) => v === null)) return null;
  const sum = raw.reduce((a, b) => a + b, 0);
  if (sum <= 0) return null;
  return {
    home: raw[0] / sum,
    draw: raw[1] / sum,
    away: raw[2] / sum,
  };
}

// Diferença em pontos percentuais (modelo - mercado), arredondada a 1 casa.
export function divergencePp(modelProb, marketProb) {
  return Math.round((modelProb - marketProb) * 1000) / 10;
}
