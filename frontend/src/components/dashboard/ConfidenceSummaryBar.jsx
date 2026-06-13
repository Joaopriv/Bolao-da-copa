import ConfidenceBar from "../../shared/ConfidenceBar";

export default function ConfidenceSummaryBar({ matches }) {
  const avg =
    matches.length > 0
      ? matches.reduce((sum, m) => sum + m.confidence, 0) / matches.length
      : 0;

  return (
    <ConfidenceBar value={avg} label="Confiança média do modelo (72 jogos)" className="w-full" />
  );
}
