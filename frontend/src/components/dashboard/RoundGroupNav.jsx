const ROUNDS = [1, 2, 3];
const GROUPS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"];

function FilterButton({ active, onClick, children }) {
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

export default function RoundGroupNav({ round, group, onRoundChange, onGroupChange }) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Rodada
        </span>
        <FilterButton active={round === "all"} onClick={() => onRoundChange("all")}>
          Todas
        </FilterButton>
        {ROUNDS.map((r) => (
          <FilterButton key={r} active={round === r} onClick={() => onRoundChange(r)}>
            {r}
          </FilterButton>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Grupo
        </span>
        <FilterButton active={group === "all"} onClick={() => onGroupChange("all")}>
          Todos
        </FilterButton>
        {GROUPS.map((g) => (
          <FilterButton key={g} active={group === g} onClick={() => onGroupChange(g)}>
            {g}
          </FilterButton>
        ))}
      </div>
    </div>
  );
}
