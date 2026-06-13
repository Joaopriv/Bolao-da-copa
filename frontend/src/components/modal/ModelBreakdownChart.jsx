import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { MODEL_LABELS } from "../../data/modelLabels";

export default function ModelBreakdownChart({ modelBreakdown }) {
  const data = Object.entries(modelBreakdown).map(([name, probs]) => ({
    name: MODEL_LABELS[name] ?? name,
    Casa: Math.round(probs.home * 100),
    Empate: Math.round(probs.draw * 100),
    Fora: Math.round(probs.away * 100),
  }));

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
          <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip
            formatter={(value) => `${value}%`}
            contentStyle={{ background: "#161e2e", border: "1px solid #232d42", borderRadius: 8 }}
            labelStyle={{ color: "#e2e8f0" }}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: "#cbd5e1" }} />
          <Bar dataKey="Casa" fill="#22d3ee" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Empate" fill="#94a3b8" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Fora" fill="#f472b6" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
