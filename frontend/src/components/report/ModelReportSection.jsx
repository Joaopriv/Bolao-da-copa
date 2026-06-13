import RpsErrorBarChart from "./RpsErrorBarChart";
import EliminationFunnel from "./EliminationFunnel";
import JustificationCard from "./JustificationCard";
import { MODEL_LABELS } from "../../data/modelLabels";

export default function ModelReportSection({ selectedModel }) {
  const { chosen_model, in_sample_RPS, out_of_sample_RPS, funnel, justification, cutoff_year } =
    selectedModel;

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-6">
      <div>
        <h2 className="text-lg font-bold text-slate-100">Relatório do modelo</h2>
        <p className="text-sm text-slate-400">
          Modelo escolhido: <span className="font-semibold text-brand">{MODEL_LABELS[chosen_model] ?? chosen_model}</span>
          {" "}· seleção anti-overfitting com corte em {cutoff_year}
        </p>
      </div>

      <section className="rounded-md border border-base-700 bg-base-800 p-4">
        <h3 className="mb-2 text-sm font-semibold text-slate-300">
          Performance (RPS, intervalo de confiança 95%)
        </h3>
        <RpsErrorBarChart inSample={in_sample_RPS} outOfSample={out_of_sample_RPS} />
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold text-slate-300">Funil de eliminação</h3>
        <EliminationFunnel funnel={funnel} />
      </section>

      <JustificationCard justification={justification} />
    </div>
  );
}
