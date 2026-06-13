// Bandas de cor (good/warn/bad) usadas em barras de confiança e feedback de probabilidade.

export const BAND_CLASSES = {
  good: { bar: "bg-good", text: "text-good", border: "border-good" },
  warn: { bar: "bg-warn", text: "text-warn", border: "border-warn" },
  bad: { bar: "bg-bad", text: "text-bad", border: "border-bad" },
};

// Confiança do modelo (result_probs / confidence): >70% boa, 50-70% média, <50% baixa.
export function confidenceBand(value) {
  if (value > 0.7) return "good";
  if (value >= 0.5) return "warn";
  return "bad";
}

// Probabilidade de um placar específico (Minha Aposta): >10% boa, 5-10% média, <5% baixa.
export function probBand(value) {
  if (value > 0.1) return "good";
  if (value >= 0.05) return "warn";
  return "bad";
}

// Cor de célula do heatmap: cyan da marca, opacidade proporcional ao valor.
export function heatColor(value, max) {
  if (!max || max <= 0) return "rgba(34,211,238,0)";
  const alpha = Math.max(0, Math.min(1, value / max));
  return `rgba(34,211,238,${alpha.toFixed(3)})`;
}
