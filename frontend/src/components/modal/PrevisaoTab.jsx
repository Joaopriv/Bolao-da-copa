import ResultProbBars from "./ResultProbBars";
import ScoreHeatmap from "./ScoreHeatmap";
import ModelBreakdownChart from "./ModelBreakdownChart";
import { argmaxOutcome } from "../../utils/outcome";

const OUTCOME_TEXT = {
  home: (home) => `o modelo favorece a vitória de ${home}`,
  draw: () => "o modelo aponta o empate como resultado mais provável",
  away: (_, away) => `o modelo favorece a vitória de ${away}`,
};

export default function PrevisaoTab({ match, onSelectScore }) {
  const { home_team, away_team, result_probs, top_scores, score_matrix, confidence, model_breakdown } =
    match;
  const outcome = argmaxOutcome(result_probs);

  return (
    <div className="space-y-6">
      <section>
        <h3 className="mb-2 text-sm font-semibold text-slate-300">Resultado (V/E/D)</h3>
        <ResultProbBars resultProbs={result_probs} homeTeam={home_team} awayTeam={away_team} />
        <p className="mt-1 text-xs text-slate-400">
          Confiança de {Math.round(confidence * 100)}%: {OUTCOME_TEXT[outcome](home_team, away_team)}.
        </p>
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold text-slate-300">Mapa de placares (0-5 x 0-5)</h3>
        <ScoreHeatmap matrix={score_matrix} topScore={top_scores?.[0]?.score} />
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold text-slate-300">
          Top 8 placares (maior valor esperado)
        </h3>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {top_scores?.map((ts) => (
            <button
              key={ts.score}
              onClick={() => onSelectScore(ts.score)}
              className="flex flex-col items-center rounded-md border border-base-700 bg-base-800 px-2 py-2 transition-colors hover:border-brand hover:bg-brand/10"
            >
              <span className="text-lg font-bold text-slate-100">{ts.score}</span>
              <span className="text-xs text-slate-400">{(ts.prob * 100).toFixed(1)}%</span>
              <span className="text-[11px] text-brand">EV {ts.ev.toFixed(2)} pts</span>
            </button>
          ))}
        </div>
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold text-slate-300">Comparação entre modelos</h3>
        <ModelBreakdownChart modelBreakdown={model_breakdown} />
      </section>
    </div>
  );
}
