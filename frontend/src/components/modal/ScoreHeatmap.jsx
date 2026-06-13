import { heatColor } from "../../utils/thresholds";

// Heatmap 6x6 (placares 0-0 a 5-5) via CSS grid (7x7: 1 header + 6x6).
// Linhas = gols do mandante (0-5), colunas = gols do visitante (0-5).
export default function ScoreHeatmap({ matrix, topScore }) {
  if (!matrix) {
    return (
      <div className="rounded-md border border-base-700 bg-base-800/50 p-4 text-center text-sm text-slate-500">
        Matriz de placares indisponível para esta partida.
      </div>
    );
  }

  const maxProb = Math.max(...matrix.flat());
  const size = matrix.length; // 6

  return (
    <div className="overflow-x-auto">
      <div
        className="inline-grid w-full max-w-full gap-1 text-center text-[11px] sm:text-xs"
        style={{ gridTemplateColumns: `auto repeat(${size}, minmax(0, 1fr))` }}
      >
        <div />
        {Array.from({ length: size }, (_, away) => (
          <div key={`h-${away}`} className="pb-1 font-semibold text-slate-400">
            {away}
          </div>
        ))}

        {matrix.map((row, home) => (
          <div key={`row-${home}`} className="contents">
            <div className="flex items-center justify-center pr-1 font-semibold text-slate-400">
              {home}
            </div>
            {row.map((value, away) => {
              const isTop = topScore === `${home}-${away}`;
              return (
                <div
                  key={`cell-${home}-${away}`}
                  className={`flex aspect-square min-w-0 items-center justify-center rounded-md border border-base-700 font-medium text-slate-100 ${
                    isTop ? "ring-2 ring-brand" : ""
                  }`}
                  style={{ backgroundColor: heatColor(value, maxProb) }}
                  title={`${home}-${away}: ${(value * 100).toFixed(1)}%`}
                >
                  {(value * 100).toFixed(1)}%
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <p className="mt-2 text-[11px] text-slate-500">
        Linhas = gols do mandante · colunas = gols do visitante
      </p>
    </div>
  );
}
