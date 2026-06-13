import { argmaxOutcome, scorelineOutcome } from "../../utils/outcome";

export default function AccuracyMetric({ matches, results, round }) {
  const withResult = matches.filter((m) => results[m.game]);

  if (withResult.length === 0) {
    return (
      <p className="text-sm text-slate-500">
        Informe os placares reais acima para ver a acurácia do modelo nesta rodada.
      </p>
    );
  }

  const hits = withResult.filter((m) => {
    const predicted = argmaxOutcome(m.result_probs);
    const real = scorelineOutcome(results[m.game].home_score, results[m.game].away_score);
    return predicted === real;
  }).length;

  const pct = Math.round((hits / withResult.length) * 100);

  return (
    <div className="space-y-3 rounded-md border border-brand/30 bg-brand/5 p-4">
      <p className="text-sm text-slate-200">
        <span className="text-2xl font-bold text-brand">{pct}%</span> dos resultados (1X2)
        previstos bateram com o resultado real na Rodada {round} ({hits}/{withResult.length}{" "}
        jogos com placar informado).
      </p>
      <div className="rounded-md border border-base-700 bg-base-900 p-3 font-mono text-xs text-slate-400">
        Para retreinar o modelo com esses resultados, rode no backend:
        <div className="mt-1 text-brand">python main.py --update-round {round}</div>
      </div>
    </div>
  );
}
