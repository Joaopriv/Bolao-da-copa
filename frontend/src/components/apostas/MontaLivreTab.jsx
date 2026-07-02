import { useMemo, useState } from "react";
import { targetOddSearch, targetProbSearch } from "../../utils/multiplesTiers";
import ComboCard from "./ComboCard";

function ModeButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
        active ? "bg-brand text-base-950" : "bg-base-800 text-slate-300 hover:bg-base-700"
      }`}
    >
      {children}
    </button>
  );
}

const ALT_COUNT = 2;

// Monta-livre -- fixa odd-alvo ou prob-alvo e resolve o melhor conjunto pro outro eixo.
// Mesma lógica de run_target()/_target_odd_search/_target_prob_search em multiples_menu.py,
// calculada no client sobre o evaluated completo (sem round-trip a Python).
export default function MontaLivreTab({ evaluated }) {
  const [mode, setMode] = useState("odd");
  const [targetOdd, setTargetOdd] = useState(4);
  const [targetProb, setTargetProb] = useState(0.3);

  const candidates = useMemo(() => {
    if (mode === "odd") {
      if (!targetOdd || targetOdd <= 1) return [];
      return targetOddSearch(evaluated, targetOdd);
    }
    if (!targetProb || targetProb <= 0) return [];
    return targetProbSearch(evaluated, targetProb);
  }, [evaluated, mode, targetOdd, targetProb]);

  const best = candidates[0] ?? null;
  const alternates = candidates.slice(1, 1 + ALT_COUNT);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <ModeButton active={mode === "odd"} onClick={() => setMode("odd")}>
          Fixar odd-alvo
        </ModeButton>
        <ModeButton active={mode === "prob"} onClick={() => setMode("prob")}>
          Fixar prob-alvo
        </ModeButton>
      </div>

      {mode === "odd" ? (
        <label className="flex items-center gap-2 text-sm text-slate-300">
          Odd-alvo (busca maior prob conjunta dentro de ±15%)
          <input
            type="number"
            step="0.1"
            min="1.1"
            value={targetOdd}
            onChange={(e) => setTargetOdd(parseFloat(e.target.value))}
            className="w-24 rounded-md border border-base-600 bg-base-800 px-2 py-1 text-slate-100"
          />
        </label>
      ) : (
        <label className="flex items-center gap-2 text-sm text-slate-300">
          Prob-alvo (0-1, busca maior retorno)
          <input
            type="number"
            step="0.05"
            min="0.01"
            max="0.99"
            value={targetProb}
            onChange={(e) => setTargetProb(parseFloat(e.target.value))}
            className="w-24 rounded-md border border-base-600 bg-base-800 px-2 py-1 text-slate-100"
          />
        </label>
      )}

      {!best ? (
        <p className="text-sm text-slate-500">
          Nenhuma combinação encontrada nesse alvo. Tente ampliar a banda/afrouxar o alvo.
        </p>
      ) : (
        <>
          <section>
            <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-400">
              Melhor combinação
            </h3>
            <ComboCard combo={best} />
          </section>
          {alternates.length > 0 && (
            <section>
              <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Alternativas (mesmo alvo)
              </h3>
              <div className="space-y-2">
                {alternates.map((combo, i) => (
                  <ComboCard key={i} combo={combo} rank={i + 2} />
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
