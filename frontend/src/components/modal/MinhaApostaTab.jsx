import { lookupScoreProb } from "../../utils/scoreLookup";
import { probBand, BAND_CLASSES } from "../../utils/thresholds";

export default function MinhaApostaTab({ match, draftScore, onChangeDraft, onSave, pick }) {
  const { home, away } = draftScore;
  const prob = home !== "" && away !== "" ? lookupScoreProb(match, Number(home), Number(away)) : null;
  const band = prob != null ? probBand(prob) : null;

  const handleNumberChange = (field) => (e) => {
    const value = e.target.value;
    if (value === "" || (/^\d{1,2}$/.test(value) && Number(value) <= 20)) {
      onChangeDraft({ ...draftScore, [field]: value });
    }
  };

  return (
    <div className="space-y-6">
      <section>
        <h3 className="mb-3 text-sm font-semibold text-slate-300">Seu palpite</h3>
        <div className="flex items-center justify-center gap-4">
          <input
            type="number"
            inputMode="numeric"
            min="0"
            max="20"
            value={home}
            onChange={handleNumberChange("home")}
            className="h-20 w-20 rounded-lg border border-base-700 bg-base-800 text-center text-4xl font-bold text-slate-100 focus:border-brand focus:outline-none"
          />
          <span className="text-2xl font-bold text-slate-500">×</span>
          <input
            type="number"
            inputMode="numeric"
            min="0"
            max="20"
            value={away}
            onChange={handleNumberChange("away")}
            className="h-20 w-20 rounded-lg border border-base-700 bg-base-800 text-center text-4xl font-bold text-slate-100 focus:border-brand focus:outline-none"
          />
        </div>

        {prob != null ? (
          <p className={`mt-3 text-center text-sm font-semibold ${BAND_CLASSES[band].text}`}>
            Probabilidade do modelo para este placar: {(prob * 100).toFixed(1)}%
          </p>
        ) : (
          home !== "" &&
          away !== "" && (
            <p className="mt-3 text-center text-sm text-slate-500">
              &lt;dados insuficientes para este placar&gt;
            </p>
          )
        )}
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold text-slate-300">Sugestões (maior EV)</h3>
        <div className="flex flex-wrap gap-2">
          {match.top_scores?.slice(0, 5).map((ts) => {
            const [h, a] = ts.score.split("-");
            return (
              <button
                key={ts.score}
                onClick={() => onChangeDraft({ home: h, away: a })}
                className="rounded-md border border-base-700 bg-base-800 px-3 py-1.5 text-sm font-semibold text-slate-100 transition-colors hover:border-brand hover:bg-brand/10"
              >
                {ts.score}
              </button>
            );
          })}
        </div>
      </section>

      <button
        onClick={onSave}
        disabled={home === "" || away === ""}
        className="w-full rounded-md bg-brand px-4 py-3 text-sm font-bold text-base-950 transition-opacity disabled:opacity-40"
      >
        Salvar palpite
      </button>

      {pick && (
        <p className="text-center text-xs text-slate-500">
          Salvo em {new Date(pick.savedAt).toLocaleString("pt-BR")}
        </p>
      )}
    </div>
  );
}
