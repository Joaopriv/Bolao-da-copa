export default function PicksProgress({ saved, total }) {
  const pct = total > 0 ? Math.round((saved / total) * 100) : 0;
  return (
    <div className="flex flex-col items-end gap-1">
      <span className="text-sm font-semibold text-slate-100">
        {saved} / {total} apostas salvas
      </span>
      <div className="h-1.5 w-32 overflow-hidden rounded-full bg-base-700">
        <div
          className="h-full rounded-full bg-brand transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
