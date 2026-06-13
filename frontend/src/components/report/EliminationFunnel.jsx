import { MODEL_LABELS } from "../../data/modelLabels";

function ModelChip({ name, variant }) {
  const variants = {
    bad: "border-bad/40 bg-bad/10 text-bad line-through",
    neutral: "border-base-600 bg-base-700 text-slate-300",
    good: "border-brand/40 bg-brand/10 text-brand font-bold",
  };
  return (
    <span className={`rounded-full border px-3 py-1 text-xs ${variants[variant]}`}>
      {MODEL_LABELS[name] ?? name}
    </span>
  );
}

export default function EliminationFunnel({ funnel }) {
  const { eliminated = [], equivalent = [], chosen } = funnel ?? {};

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <div className="rounded-md border border-base-700 bg-base-800 p-3">
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Eliminados
        </h4>
        <div className="flex flex-wrap gap-2">
          {eliminated.map((m) => (
            <ModelChip key={m} name={m} variant="bad" />
          ))}
        </div>
      </div>
      <div className="rounded-md border border-base-700 bg-base-800 p-3">
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Estatisticamente equivalentes
        </h4>
        <div className="flex flex-wrap gap-2">
          {equivalent.map((m) => (
            <ModelChip key={m} name={m} variant="neutral" />
          ))}
        </div>
      </div>
      <div className="rounded-md border border-brand/40 bg-brand/5 p-3">
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Escolhido
        </h4>
        <div className="flex flex-wrap gap-2">
          <ModelChip name={chosen} variant="good" />
        </div>
      </div>
    </div>
  );
}
