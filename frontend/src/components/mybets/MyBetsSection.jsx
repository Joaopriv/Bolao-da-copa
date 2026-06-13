import MyBetsTable from "./MyBetsTable";
import { exportPicksCsv } from "../../utils/csv";

export default function MyBetsSection({ matches, picks, results }) {
  const hasPicks = Object.keys(picks).length > 0;

  return (
    <div className="mx-auto max-w-6xl space-y-4 px-4 py-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-slate-100">Minhas apostas</h2>
        <button
          onClick={() => exportPicksCsv(matches, picks, results)}
          disabled={!hasPicks}
          className="rounded-md bg-base-700 px-3 py-2 text-sm font-semibold text-slate-100 transition-colors hover:bg-brand hover:text-base-950 disabled:opacity-40"
        >
          Exportar CSV
        </button>
      </div>
      <MyBetsTable matches={matches} picks={picks} results={results} />
    </div>
  );
}
