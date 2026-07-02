// Porta em JS de _RETURN_TIERS/run()/_target_odd_search/_target_prob_search em
// 4_validation/multiples_menu.py -- mantém o mesmo ranking do --multiples-menu/--multiples-target,
// mas calculado no client sobre o `evaluated` já exportado (sem round-trip a Python).

export const RETURN_TIERS = [
  [2.0, 3.0, "2-3x"],
  [3.0, 5.0, "3-5x"],
  [5.0, 10.0, "5-10x"],
  [10.0, Infinity, "10x+"],
];

export function bucketByReturnTier(evaluated) {
  const tiers = Object.fromEntries(RETURN_TIERS.map(([, , label]) => [label, []]));
  for (const m of evaluated) {
    const tier = RETURN_TIERS.find(([lo, hi]) => m.ret >= lo && m.ret < hi);
    if (tier) tiers[tier[2]].push(m);
  }
  for (const label in tiers) tiers[label].sort((a, b) => b.prob - a.prob);
  return tiers;
}

export function targetOddSearch(evaluated, target, band = 0.15) {
  const lo = target * (1 - band);
  const hi = target * (1 + band);
  return evaluated
    .filter((m) => m.ret >= lo && m.ret <= hi)
    .sort((a, b) => b.prob - a.prob || a.n_legs - b.n_legs);
}

export function targetProbSearch(evaluated, minProb) {
  return evaluated
    .filter((m) => m.prob >= minProb)
    .sort((a, b) => b.ret - a.ret || a.n_legs - b.n_legs);
}
