import { useMemo, useState } from "react";
import RoundGroupNav from "./RoundGroupNav";
import GameCardGrid from "./GameCardGrid";

export default function Dashboard({ matches, picks, onOpenAnalysis }) {
  const [round, setRound] = useState("all");
  const [group, setGroup] = useState("all");

  const filtered = useMemo(() => {
    return matches.filter((m) => {
      if (round !== "all" && m.round !== round) return false;
      if (group !== "all" && m.group !== group) return false;
      return true;
    });
  }, [matches, round, group]);

  return (
    <div className="mx-auto max-w-6xl space-y-4 px-4 py-6">
      <RoundGroupNav
        round={round}
        group={group}
        onRoundChange={setRound}
        onGroupChange={setGroup}
        matches={matches}
      />
      <p className="text-sm text-slate-500">
        {filtered.length} / {matches.length} jogos
      </p>
      <GameCardGrid matches={filtered} picks={picks} onOpenAnalysis={onOpenAnalysis} />
    </div>
  );
}
