import { useMemo } from "react";
import { RETURN_TIERS, bucketByReturnTier } from "../../utils/multiplesTiers";
import ComboCard from "./ComboCard";

const TOP = 5;

// Ranking por faixa de retorno -- mesma lógica de --multiples-menu (run() em
// 4_validation/multiples_menu.py), calculada no client sobre o evaluated completo.
export default function SugeridasTab({ evaluated }) {
  const tiers = useMemo(() => bucketByReturnTier(evaluated), [evaluated]);

  return (
    <div className="space-y-6">
      {RETURN_TIERS.map(([, , label]) => {
        const combos = tiers[label] ?? [];
        return (
          <section key={label}>
            <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-400">
              Faixa de retorno {label}
            </h3>
            {combos.length === 0 ? (
              <p className="text-sm text-slate-500">Nenhuma múltipla nesta faixa.</p>
            ) : (
              <div className="space-y-2">
                {combos.slice(0, TOP).map((combo, i) => (
                  <ComboCard key={i} combo={combo} rank={i + 1} />
                ))}
              </div>
            )}
          </section>
        );
      })}
    </div>
  );
}
