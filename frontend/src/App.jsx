import { lazy, Suspense, useState } from "react";
import { usePredictions } from "./hooks/usePredictions";
import { useSelectedModel } from "./hooks/useSelectedModel";
import { useRoundsAndGroups } from "./hooks/useRoundsAndGroups";
import { useLocalStorageState } from "./hooks/useLocalStorageState";
import Header from "./components/layout/Header";
import NavTabs from "./components/layout/NavTabs";
import Dashboard from "./components/dashboard/Dashboard";
import MyBetsSection from "./components/mybets/MyBetsSection";
import ApostasSection from "./components/apostas/ApostasSection";
import ModelReportSection from "./components/report/ModelReportSection";
import LiveUpdateSection from "./components/liveupdate/LiveUpdateSection";

const GameModal = lazy(() => import("./components/modal/GameModal"));

export default function App() {
  const { data: predictionsData, loading: loadingPreds, error: errorPreds } = usePredictions();
  const { data: selectedModel, loading: loadingModel, error: errorModel } = useSelectedModel();
  const matches = useRoundsAndGroups(predictionsData?.predictions);

  const [picks, setPicks] = useLocalStorageState("bolao_picks_2026", {});
  const [results, setResults] = useLocalStorageState("bolao_results_2026", {});
  const [activeTab, setActiveTab] = useState("dashboard");
  const [selectedMatch, setSelectedMatch] = useState(null);

  const handleSavePick = (game, score) => {
    setPicks((prev) => ({ ...prev, [game]: { score, savedAt: new Date().toISOString() } }));
  };

  const handleOpenAnalysis = (match) => setSelectedMatch(match);
  const handleCloseAnalysis = () => setSelectedMatch(null);

  if (loadingPreds || loadingModel) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-base-950 text-slate-400">
        Carregando previsões...
      </div>
    );
  }

  if (errorPreds || errorModel) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-base-950 px-4">
        <div className="max-w-md rounded-xl border border-bad/40 bg-bad/10 p-6 text-center">
          <h2 className="text-lg font-bold text-bad">Erro ao carregar dados</h2>
          <p className="mt-2 text-sm text-slate-300">
            {errorPreds?.message || errorModel?.message}
          </p>
          <p className="mt-2 text-xs text-slate-500">
            Verifique se predictions_2026.json e selected_model.json estão em public/data/.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-base-950 text-slate-200">
      <Header
        meta={predictionsData.meta}
        matches={matches}
        picksCount={Object.keys(picks).length}
        totalGames={matches.length}
      />
      <NavTabs active={activeTab} onChange={setActiveTab} />

      {activeTab === "dashboard" && (
        <Dashboard matches={matches} picks={picks} onOpenAnalysis={handleOpenAnalysis} />
      )}
      {activeTab === "mybets" && (
        <MyBetsSection matches={matches} picks={picks} results={results} />
      )}
      {activeTab === "apostas" && <ApostasSection />}
      {activeTab === "report" && <ModelReportSection selectedModel={selectedModel} />}
      {activeTab === "live" && (
        <LiveUpdateSection
          matches={matches}
          results={results}
          setResults={setResults}
          meta={predictionsData.meta}
        />
      )}

      {selectedMatch && (
        <Suspense
          fallback={
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-base-950/80 text-slate-300">
              Carregando análise...
            </div>
          }
        >
          <GameModal
            match={selectedMatch}
            pick={picks[selectedMatch.game]}
            onSavePick={handleSavePick}
            onClose={handleCloseAnalysis}
          />
        </Suspense>
      )}
    </div>
  );
}
