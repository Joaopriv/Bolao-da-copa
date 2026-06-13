import { confidenceBand, BAND_CLASSES } from "../utils/thresholds";

// Barra horizontal de confiança, colorida por banda (>70% verde, 50-70% amarelo, <50% vermelho).
export default function ConfidenceBar({ value, label, className = "" }) {
  const band = confidenceBand(value);
  const pct = Math.round(value * 100);

  return (
    <div className={className}>
      {label && (
        <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
          <span>{label}</span>
          <span className={`font-semibold ${BAND_CLASSES[band].text}`}>{pct}%</span>
        </div>
      )}
      <div className="h-2 w-full overflow-hidden rounded-full bg-base-700">
        <div
          className={`h-full rounded-full ${BAND_CLASSES[band].bar} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
