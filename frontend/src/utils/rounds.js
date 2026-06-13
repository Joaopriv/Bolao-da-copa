import { GROUPS_PT } from "../data/groups";

// Deriva grupo (A-L) e rodada (1-3) para cada partida. Ordena por `date` dentro de
// cada grupo (não depende da ordem de chegada de `predictions`) e pareia 2 a 2.
export function deriveRoundsAndGroups(predictions) {
  const withGroup = predictions.map((m) => ({
    ...m,
    group: GROUPS_PT[m.home_team] ?? null,
  }));

  const byGroup = {};
  for (const m of withGroup) {
    (byGroup[m.group] ??= []).push(m);
  }

  const result = [];
  for (const group of Object.keys(byGroup)) {
    const sorted = [...byGroup[group]].sort((a, b) => a.date.localeCompare(b.date));
    sorted.forEach((m, idx) => result.push({ ...m, round: Math.floor(idx / 2) + 1 }));
  }
  return result;
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
