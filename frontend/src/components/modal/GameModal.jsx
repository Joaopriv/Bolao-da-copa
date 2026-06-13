import { useState } from "react";
import FlagTeam from "../../shared/FlagTeam";
import PrevisaoTab from "./PrevisaoTab";
import OddsCheckTab from "./OddsCheckTab";
import MinhaApostaTab from "./MinhaApostaTab";

const TABS = [
  { id: "previsao", label: "Previsão" },
  { id: "odds", label: "Odds Check" },
  { id: "aposta", label: "Minha Aposta" },
];

export default function GameModal({ match, pick, onSavePick, onClose }) {
  const [tab, setTab] = useState("previsao");
  const [draftScore, setDraftScore] = useState(() => {
    if (pick) {
      const [home, away] = pick.score.split("-");
      return { home, away };
    }
    if (match.top_scores?.[0]) {
      const [home, away] = match.top_scores[0].score.split("-");
      return { home, away };
    }
    return { home: "", away: "" };
  });

  const handleSelectScore = (score) => {
    const [home, away] = score.split("-");
    setDraftScore({ home, away });
    setTab("aposta");
  };

  const handleSave = () => {
    onSavePick(match.game, `${draftScore.home}-${draftScore.away}`);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-base-950/80 sm:items-center sm:p-4">
      <div className="flex h-full w-full flex-col bg-base-900 sm:h-auto sm:max-h-[90vh] sm:max-w-2xl sm:rounded-xl sm:border sm:border-base-700">
        <div className="flex items-center justify-between border-b border-base-700 px-4 py-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            <FlagTeam name={match.home_team} />
            <span className="text-slate-500">vs</span>
            <FlagTeam name={match.away_team} />
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-slate-400 hover:bg-base-700 hover:text-slate-100"
            aria-label="Fechar"
          >
            ✕
          </button>
        </div>

        <div className="flex border-b border-base-700">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex-1 border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
                tab === t.id
                  ? "border-brand text-brand"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {tab === "previsao" && <PrevisaoTab match={match} onSelectScore={handleSelectScore} />}
          {tab === "odds" && <OddsCheckTab match={match} />}
          {tab === "aposta" && (
            <MinhaApostaTab
              match={match}
              draftScore={draftScore}
              onChangeDraft={setDraftScore}
              onSave={handleSave}
              pick={pick}
            />
          )}
        </div>
      </div>
    </div>
  );
}
