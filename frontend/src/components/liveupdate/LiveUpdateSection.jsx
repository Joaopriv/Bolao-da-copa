import { useState } from "react";
import RoundResultForm from "./RoundResultForm";
import AccuracyMetric from "./AccuracyMetric";

export default function LiveUpdateSection({ matches, results, setResults, meta }) {
  const [round, setRound] = useState(1);
  const roundMatches = matches.filter((m) => m.round === round);
  const divergenceCount = roundMatches.filter((m) => m.divergence_alert).length;

  const handleChange = (game, value) => {
    setResults((prev) => {
      const next = { ...prev };
      if (value === null) {
        delete next[game];
      } else {
        next[game] = value;
      }
      return next;
    });
  };

  return (
    <div className="mx-auto max-w-3xl space-y-4 px-4 py-6">
      <div>
        <h2 className="text-lg font-bold text-slate-100">Update ao vivo</h2>
        <p className="text-sm text-slate-400">
          Insira os placares reais dos jogos da rodada para acompanhar a acurácia do modelo.
        </p>
      </div>

      {meta && (
        <div className="space-y-1 rounded-md border border-base-700 bg-base-900 p-3 text-sm text-slate-300">
          {meta.model_confidence != null && (
            <p>Confiança média do modelo: {Math.round(meta.model_confidence * 100)}%</p>
          )}
          <p>Última rodada atualizada: {meta.round_updated ?? "—"}</p>
          <p>Jogos com divergência de odds &gt;5pp (rodada atual): {divergenceCount}</p>
          {meta.odds_api_credits_remaining != null && (
            <p>Créditos da API de odds restantes: {meta.odds_api_credits_remaining}</p>
          )}
        </div>
      )}

      <div className="flex gap-2">
        {[1, 2, 3].map((r) => (
          <button
            key={r}
            onClick={() => setRound(r)}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              round === r ? "bg-brand text-base-950" : "bg-base-800 text-slate-300 hover:bg-base-700"
            }`}
          >
            Rodada {r}
          </button>
        ))}
      </div>

      <div className="space-y-2">
        {roundMatches.map((m) => (
          <RoundResultForm
            key={m.game}
            match={m}
            result={results[m.game]}
            onChange={(value) => handleChange(m.game, value)}
          />
        ))}
      </div>

      <AccuracyMetric matches={roundMatches} results={results} round={round} />
    </div>
  );
}
