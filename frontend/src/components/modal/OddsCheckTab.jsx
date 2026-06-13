import { useState } from "react";
import { impliedProbs, divergencePp } from "../../utils/odds";
import { oddsDivergenceBand, isHighOddsDivergence } from "../../utils/divergence";
import { BAND_CLASSES } from "../../utils/thresholds";

const OUTCOME_LABELS = { home: "Casa", draw: "Empate", away: "Fora" };

export default function OddsCheckTab({ match }) {
  const [odds, setOdds] = useState({ home: "", draw: "", away: "" });

  const hasAutoOdds = Boolean(match.odds_implied);

  const parsed = {
    home: parseFloat(odds.home),
    draw: parseFloat(odds.draw),
    away: parseFloat(odds.away),
  };
  const validOdds = Object.values(parsed).every((v) => !isNaN(v) && v > 1);
  const market = hasAutoOdds ? match.odds_implied : validOdds ? impliedProbs(parsed) : null;

  return (
    <div className="space-y-6">
      {hasAutoOdds ? (
        <section>
          <h3 className="mb-2 text-sm font-semibold text-slate-300">
            Mercado (automático via The Odds API)
          </h3>
          <div className="grid grid-cols-3 gap-3">
            {["home", "draw", "away"].map((key) => (
              <div key={key} className="flex flex-col gap-1 text-xs text-slate-400">
                {OUTCOME_LABELS[key]}
                <div className="rounded-md border border-base-700 bg-base-800 px-2 py-2 text-base font-semibold text-slate-100">
                  {Math.round(match.odds_implied[key] * 100)}%
                </div>
              </div>
            ))}
          </div>
          {match.divergence_alert && (
            <p className="mt-2 text-yellow-400">
              ⚠ Divergência &gt;5pp entre modelo e mercado
            </p>
          )}
        </section>
      ) : (
        <section>
          <h3 className="mb-2 text-sm font-semibold text-slate-300">
            Odds decimais do mercado (1X2)
          </h3>
          <div className="grid grid-cols-3 gap-3">
            {["home", "draw", "away"].map((key) => (
              <label key={key} className="flex flex-col gap-1 text-xs text-slate-400">
                {OUTCOME_LABELS[key]}
                <input
                  type="number"
                  step="0.01"
                  min="1.01"
                  inputMode="decimal"
                  placeholder="ex: 2.50"
                  value={odds[key]}
                  onChange={(e) => setOdds((prev) => ({ ...prev, [key]: e.target.value }))}
                  className="rounded-md border border-base-700 bg-base-800 px-2 py-2 text-base text-slate-100 focus:border-brand focus:outline-none"
                />
              </label>
            ))}
          </div>
          {!validOdds && (odds.home || odds.draw || odds.away) && (
            <p className="mt-2 text-xs text-bad">
              Informe odds decimais válidas (maiores que 1.00) nos 3 campos.
            </p>
          )}
        </section>
      )}

      {market && (
        <section>
          <h3 className="mb-2 text-sm font-semibold text-slate-300">Modelo vs Mercado</h3>
          <table className="w-full overflow-hidden rounded-md border border-base-700 text-sm">
            <thead className="bg-base-800 text-xs uppercase text-slate-400">
              <tr>
                <th className="px-3 py-2 text-left">Resultado</th>
                <th className="px-3 py-2 text-right">Modelo</th>
                <th className="px-3 py-2 text-right">Mercado</th>
                <th className="px-3 py-2 text-right">Divergência</th>
              </tr>
            </thead>
            <tbody>
              {["home", "draw", "away"].map((key) => {
                const modelProb = match.result_probs[key];
                const marketProb = market[key];
                const diff = divergencePp(modelProb, marketProb);
                const band = oddsDivergenceBand(diff);
                return (
                  <tr key={key} className="border-t border-base-700">
                    <td className="px-3 py-2 text-slate-300">{OUTCOME_LABELS[key]}</td>
                    <td className="px-3 py-2 text-right text-slate-100">
                      {(modelProb * 100).toFixed(1)}%
                    </td>
                    <td className="px-3 py-2 text-right text-slate-100">
                      {(marketProb * 100).toFixed(1)}%
                    </td>
                    <td className={`px-3 py-2 text-right font-semibold ${BAND_CLASSES[band].text}`}>
                      {diff > 0 ? "+" : ""}
                      {diff.toFixed(1)}pp
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {["home", "draw", "away"].some((key) =>
            isHighOddsDivergence(divergencePp(match.result_probs[key], market[key]))
          ) && (
            <p className="mt-2 rounded-md border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">
              ⚠ Divergência acima de 10pp entre o modelo e o mercado em pelo menos um
              resultado — confira as odds digitadas.
            </p>
          )}
        </section>
      )}
    </div>
  );
}
