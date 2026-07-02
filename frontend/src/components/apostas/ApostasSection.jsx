import { useState } from "react";
import { useMultiples } from "../../hooks/useMultiples";
import SugeridasTab from "./SugeridasTab";
import MontaLivreTab from "./MontaLivreTab";

const SUBTABS = [
  { id: "sugeridas", label: "Sugeridas" },
  { id: "montalivre", label: "Monta-livre" },
];

// useMultiples() só é chamado aqui dentro -- o fetch de multiplas_2026.json (arquivo grande,
// milhares de combos) só dispara quando o usuário abre a aba Apostas pela primeira vez.
export default function ApostasSection() {
  const [subtab, setSubtab] = useState("sugeridas");
  const { data, loading, error } = useMultiples();

  if (loading) {
    return (
      <div className="mx-auto max-w-6xl px-4 py-6 text-sm text-slate-400">
        Carregando cardápio de múltiplas...
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-6xl px-4 py-6">
        <div className="rounded-xl border border-bad/40 bg-bad/10 p-6 text-center">
          <h2 className="text-lg font-bold text-bad">Erro ao carregar apostas</h2>
          <p className="mt-2 text-sm text-slate-300">{error.message}</p>
          <p className="mt-2 text-xs text-slate-500">
            Rode "python main.py --multiples-export" para gerar multiplas_2026.json em
            public/data/.
          </p>
        </div>
      </div>
    );
  }

  const evaluated = data.evaluated ?? [];

  return (
    <div className="mx-auto max-w-6xl space-y-4 px-4 py-6">
      <div>
        <h2 className="text-lg font-bold text-slate-100">Apostas</h2>
        <p className="text-xs text-slate-500">
          modelo: {data.model} · rodada {data.round ?? "-"} · {data.n_games} jogos com odds ·{" "}
          {evaluated.length} combinações
          {data.sem_odds?.length > 0 && ` · sem odds: ${data.sem_odds.join(", ")}`}
        </p>
      </div>

      <div className="flex gap-1 border-b border-base-700">
        {SUBTABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setSubtab(t.id)}
            className={`border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
              subtab === t.id
                ? "border-brand text-brand"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {evaluated.length === 0 ? (
        <p className="text-sm text-slate-500">
          Nenhuma perna com odd real nesta rodada ainda -- sem cardápio.
        </p>
      ) : subtab === "sugeridas" ? (
        <SugeridasTab evaluated={evaluated} />
      ) : (
        <MontaLivreTab evaluated={evaluated} />
      )}
    </div>
  );
}
