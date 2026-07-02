import Badge from "../../shared/Badge";

const TIPO_LABELS = { same: "same-game", cross: "cross-game", misto: "misto" };

// Porta visual de _format_combo_lines em 4_validation/multiples_menu.py -- um card por
// combinação: header (prob conjunta / retorno / EV / tipo), pernas, avisos.
export default function ComboCard({ combo, rank }) {
  const isPositiveEv = combo.ev >= 0;

  return (
    <div className="rounded-lg border border-base-700 bg-base-800 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm">
          {rank != null && <span className="font-semibold text-slate-500">#{rank}</span>}
          <span className="font-semibold text-slate-100">{(combo.prob * 100).toFixed(1)}%</span>
          <span className="text-slate-500">conjunta</span>
          <span className="text-slate-600">·</span>
          <span className="font-semibold text-brand">{combo.ret.toFixed(2)}x</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Badge variant={isPositiveEv ? "good" : "bad"}>
            EV {combo.ev >= 0 ? "+" : ""}
            {(combo.ev * 100).toFixed(1)}%
          </Badge>
          <Badge variant="neutral">{TIPO_LABELS[combo.tipo] ?? combo.tipo}</Badge>
          <Badge variant="neutral">{combo.n_legs} pernas</Badge>
        </div>
      </div>

      <ul className="mt-2 space-y-1 text-sm">
        {combo.legs.map((leg, i) => (
          <li key={i} className="flex items-center justify-between gap-2 text-slate-300">
            <span className="truncate">
              {leg.game} — {leg.sel}
              {leg.derived && <span className="text-slate-500"> *</span>}
            </span>
            <span className="shrink-0 text-slate-400">
              {(leg.prob * 100).toFixed(1)}% @ {leg.odd.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>

      {combo.flags.length > 0 && (
        <ul className="mt-2 space-y-0.5 border-t border-base-700 pt-2">
          {combo.flags.map((f, i) => (
            <li key={i} className="text-xs text-warn">
              ⚠ {f}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
