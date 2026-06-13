import GameCard from "./GameCard";

export default function GameCardGrid({ matches, picks, onOpenAnalysis }) {
  if (matches.length === 0) {
    return (
      <p className="py-12 text-center text-sm text-slate-500">
        Nenhum jogo encontrado para esse filtro.
      </p>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {matches.map((match) => (
        <GameCard
          key={match.game}
          match={match}
          pick={picks[match.game]}
          onOpenAnalysis={onOpenAnalysis}
        />
      ))}
    </div>
  );
}
