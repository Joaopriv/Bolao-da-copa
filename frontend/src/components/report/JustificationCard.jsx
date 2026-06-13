export default function JustificationCard({ justification }) {
  return (
    <div className="rounded-md border border-base-700 bg-base-800 p-4">
      <h3 className="mb-2 text-sm font-semibold text-slate-300">
        Por que confiamos neste modelo
      </h3>
      <p className="whitespace-pre-line text-sm leading-relaxed text-slate-300">
        {justification}
      </p>
    </div>
  );
}
