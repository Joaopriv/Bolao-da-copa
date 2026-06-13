// Resultado 1X2 (vencedor/empate) a partir de result_probs ou de um placar.

// { home, draw, away } -> "home" | "draw" | "away" (maior probabilidade)
export function argmaxOutcome(resultProbs) {
  const entries = Object.entries(resultProbs);
  return entries.reduce((best, cur) => (cur[1] > best[1] ? cur : best))[0];
}

// placar h x a -> "home" | "draw" | "away"
export function scorelineOutcome(home, away) {
  if (home > away) return "home";
  if (home < away) return "away";
  return "draw";
}
