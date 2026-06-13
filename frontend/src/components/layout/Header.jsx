import ConfidenceSummaryBar from "../dashboard/ConfidenceSummaryBar";
import PicksProgress from "../dashboard/PicksProgress";
import { MODEL_LABELS } from "../../data/modelLabels";

export default function Header({ meta, matches, picksCount, totalGames }) {
  const modelLabel = MODEL_LABELS[meta?.model] ?? meta?.model ?? "—";

  return (
    <header className="border-b border-base-700 bg-base-900/80 backdrop-blur sticky top-0 z-20">
      <div className="mx-auto max-w-6xl px-4 py-4">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-slate-50 sm:text-2xl">
              🏆 Bolão Copa 2026
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              Modelo: <span className="font-semibold text-brand">{modelLabel}</span>
              {meta?.trained_until && (
                <>
                  {" "}
                  · gerado a partir de dados até{" "}
                  <span className="font-medium text-slate-300">{meta.trained_until}</span>
                </>
              )}
            </p>
          </div>
          <PicksProgress saved={picksCount} total={totalGames} />
        </div>
        <ConfidenceSummaryBar matches={matches} />
      </div>
    </header>
  );
}
