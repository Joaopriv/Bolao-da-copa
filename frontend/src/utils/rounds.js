// Rótulos das fases do mata-mata (round 4-8, vindo de copa_2026_results via predict_2026.py).
export const KNOCKOUT_STAGE_LABELS = {
  4: "32-avos",
  5: "Oitavas",
  6: "Quartas",
  7: "Semis",
  8: "Final",
};

// `round` e `stage` já vêm de predictions_2026.json (autoritativos, de copa_2026_results).
// Deriva só `group` (letra A-L) a partir de `stage` ("Group A" -> "A"); fica null no
// mata-mata, onde os confrontos cruzam grupos diferentes.
export function deriveRoundsAndGroups(predictions) {
  return predictions.map((m) => ({
    ...m,
    group: m.stage?.startsWith("Group ") ? m.stage.slice("Group ".length) : null,
  }));
}

// Agrupa partidas (já com `round`/`group`) em { [round]: { [group]: [matches] } }
export function groupByRoundAndGroup(matches) {
  const result = {};
  for (const m of matches) {
    (result[m.round] ??= {});
    (result[m.round][m.group] ??= []).push(m);
  }
  return result;
}
