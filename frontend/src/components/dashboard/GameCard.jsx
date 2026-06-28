import FlagTeam from "../../shared/FlagTeam";
import ConfidenceBar from "../../shared/ConfidenceBar";
import Badge from "../../shared/Badge";
import { hasDivergence } from "../../utils/divergence";
import { KNOCKOUT_STAGE_LABELS } from "../../utils/rounds";

export default function GameCard({ match, pick, onOpenAnalysis }) {
  const {
    home_team, away_team, date, result_probs, top_scores, confidence, model_breakdown,
    odds_implied, divergence_alert, divergence_direction, squad_note_home, squad_note_away,
  } = match;
  const topScore = top_scores?.[0];
  const divergent = hasDivergence(model_breakdown);

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-base-700 bg-base-800 p-4">
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>{date}</span>
        <Badge variant="neutral">
          {match.group
            ? `Grupo ${match.group} · Rodada ${match.round}`
            : KNOCKOUT_STAGE_LABELS[match.round] ?? match.stage}
        </Badge>
      </div>

      <div className="flex items-center justify-between gap-2">
        <FlagTeam name={home_team} className="min-w-0 flex-1" />
        <span className="shrink-0 text-xs font-semibold text-slate-500">vs</span>
        <FlagTeam name={away_team} align="right" className="min-w-0 flex-1" />
      </div>

      {(squad_note_home || squad_note_away) && (
        <div className="flex items-center justify-between gap-2 text-[11px] text-warn">
          <span className="truncate">{squad_note_home}</span>
          <span className="truncate text-right">{squad_note_away}</span>
        </div>
      )}

      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div className="rounded-md bg-base-700/60 px-2 py-1.5">
          <div className="text-slate-400">Casa</div>
          <div className="font-semibold text-slate-100">
            {Math.round(result_probs.home * 100)}%
          </div>
        </div>
        <div className="rounded-md bg-base-700/60 px-2 py-1.5">
          <div className="text-slate-400">Empate</div>
          <div className="font-semibold text-slate-100">
            {Math.round(result_probs.draw * 100)}%
          </div>
        </div>
        <div className="rounded-md bg-base-700/60 px-2 py-1.5">
          <div className="text-slate-400">Fora</div>
          <div className="font-semibold text-slate-100">
            {Math.round(result_probs.away * 100)}%
          </div>
        </div>
      </div>

      {topScore && (
        <div className="flex items-center justify-between rounded-md border border-brand/30 bg-brand/10 px-3 py-2">
          <span className="text-xs text-slate-300">Placar mais provável (maior EV)</span>
          <span className="text-lg font-bold text-brand">{topScore.score}</span>
        </div>
      )}

      <ConfidenceBar value={confidence} label="Confiança do modelo" />

      {odds_implied && (
        <div className="text-xs text-slate-500">
          Mercado: {Math.round(odds_implied.home * 100)}% / {Math.round(odds_implied.draw * 100)}%
          {" "}/ {Math.round(odds_implied.away * 100)}%
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {divergent && <Badge variant="warn">⚠ Divergência</Badge>}
        {divergence_alert && divergence_direction && (
          <Badge variant="bad">⚠ Direções opostas</Badge>
        )}
        {divergence_alert && !divergence_direction && (
          <Badge variant="warn">△ Magnitude diferente</Badge>
        )}
        {pick && <Badge variant="brand">Sua aposta: {pick.score}</Badge>}
      </div>

      <button
        onClick={() => onOpenAnalysis(match)}
        className="mt-1 rounded-md bg-base-700 px-3 py-2 text-sm font-semibold text-slate-100 transition-colors hover:bg-brand hover:text-base-950"
      >
        Ver análise
      </button>
    </div>
  );
}
