import { lookupScoreProb } from "../../utils/scoreLookup";
import { pickStatus } from "../../utils/status";
import Badge from "../../shared/Badge";

const STATUS_VARIANT = { Pendente: "neutral", Correto: "good", Errado: "bad" };

export default function MyBetsTable({ matches, picks, results }) {
  const rows = matches
    .filter((m) => picks[m.game])
    .map((m) => {
      const pick = picks[m.game];
      const [home, away] = pick.score.split("-").map(Number);
      const prob = lookupScoreProb(m, home, away);
      const result = results[m.game];
      const status = pickStatus({ home, away }, result) ?? "Pendente";
      return { match: m, pick, prob, status };
    });

  if (rows.length === 0) {
    return (
      <p className="py-12 text-center text-sm text-slate-500">
        Você ainda não salvou nenhuma aposta. Abra "Ver análise" em um jogo e salve seu palpite
        na aba "Minha Aposta".
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-md border border-base-700">
      <table className="w-full text-sm">
        <thead className="bg-base-800 text-xs uppercase text-slate-400">
          <tr>
            <th className="px-3 py-2 text-left">Jogo</th>
            <th className="px-3 py-2 text-left">Data</th>
            <th className="px-3 py-2 text-center">Aposta</th>
            <th className="px-3 py-2 text-right">Prob. modelo</th>
            <th className="px-3 py-2 text-center">Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ match, pick, prob, status }) => (
            <tr key={match.game} className="border-t border-base-700">
              <td className="px-3 py-2 text-slate-200">{match.game}</td>
              <td className="px-3 py-2 text-slate-400">{match.date}</td>
              <td className="px-3 py-2 text-center font-bold text-brand">{pick.score}</td>
              <td className="px-3 py-2 text-right text-slate-300">
                {prob != null ? `${(prob * 100).toFixed(1)}%` : "-"}
              </td>
              <td className="px-3 py-2 text-center">
                <Badge variant={STATUS_VARIANT[status]}>{status}</Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
