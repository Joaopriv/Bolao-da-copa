// Probabilidade de um placar h x a para uma partida: tenta score_matrix (grade 0-5
// x 0-5), senão cai para top_scores (placares fora da grade 0-5, ex. goleadas).
// Retorna null se não houver dado disponível.
export function lookupScoreProb(match, home, away) {
  const { score_matrix: matrix, top_scores: topScores } = match;

  if (
    matrix &&
    home >= 0 &&
    home < matrix.length &&
    away >= 0 &&
    away < matrix[home].length
  ) {
    return matrix[home][away];
  }

  const key = `${home}-${away}`;
  const found = topScores?.find((t) => t.score === key);
  if (found) return found.prob;

  return null;
}
