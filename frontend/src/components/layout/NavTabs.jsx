const TABS = [
  { id: "dashboard", label: "Dashboard" },
  { id: "mybets", label: "Minhas apostas" },
  { id: "report", label: "Relatório do modelo" },
  { id: "live", label: "Update ao vivo" },
];

export default function NavTabs({ active, onChange }) {
  return (
    <nav className="border-b border-base-700 bg-base-900">
      <div className="mx-auto flex max-w-6xl gap-1 overflow-x-auto px-4">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            className={`whitespace-nowrap border-b-2 px-3 py-3 text-sm font-medium transition-colors ${
              active === tab.id
                ? "border-brand text-brand"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </nav>
  );
}
