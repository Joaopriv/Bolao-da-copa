import { scorelineOutcome } from "./outcome";

// Status do palpite em "Minhas apostas": "Pendente" sem resultado real ainda;
// "Correto"/"Errado" comparando 1X2 (vencedor/empate) do palpite vs resultado real
// — não exige placar exato (ver Contexto do plano).
export function pickStatus(pick, result) {
  if (!pick) return null;
  if (!result) return "Pendente";

  const pickOutcome = scorelineOutcome(pick.home, pick.away);
  const resultOutcome = scorelineOutcome(result.home_score, result.away_score);
  return pickOutcome === resultOutcome ? "Correto" : "Errado";
}
