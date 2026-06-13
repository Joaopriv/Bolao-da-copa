import { useEffect, useState } from "react";
import FlagTeam from "../../shared/FlagTeam";

export default function RoundResultForm({ match, result, onChange }) {
  const [home, setHome] = useState(result?.home_score?.toString() ?? "");
  const [away, setAway] = useState(result?.away_score?.toString() ?? "");

  useEffect(() => {
    if (home !== "" && away !== "") {
      onChange({ home_score: Number(home), away_score: Number(away) });
    } else {
      onChange(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [home, away]);

  const handleChange = (setter) => (e) => {
    const raw = e.target.value;
    if (raw === "" || /^\d{1,2}$/.test(raw)) setter(raw);
  };

  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-base-700 bg-base-800 px-3 py-2">
      <FlagTeam name={match.home_team} className="min-w-0 flex-1" />
      <div className="flex items-center gap-2">
        <input
          type="number"
          inputMode="numeric"
          min="0"
          max="20"
          value={home}
          onChange={handleChange(setHome)}
          className="h-10 w-12 rounded-md border border-base-700 bg-base-900 text-center text-lg font-bold text-slate-100 focus:border-brand focus:outline-none"
        />
        <span className="text-slate-500">×</span>
        <input
          type="number"
          inputMode="numeric"
          min="0"
          max="20"
          value={away}
          onChange={handleChange(setAway)}
          className="h-10 w-12 rounded-md border border-base-700 bg-base-900 text-center text-lg font-bold text-slate-100 focus:border-brand focus:outline-none"
        />
      </div>
      <FlagTeam name={match.away_team} align="right" className="min-w-0 flex-1" />
    </div>
  );
}
