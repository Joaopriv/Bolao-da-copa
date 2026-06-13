import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from "recharts";

const COLORS = { home: "#22d3ee", draw: "#94a3b8", away: "#f472b6" };

export default function ResultProbBars({ resultProbs, homeTeam, awayTeam }) {
  const data = [
    { key: "home", name: `Vitória ${homeTeam}`, value: Math.round(resultProbs.home * 100) },
    { key: "draw", name: "Empate", value: Math.round(resultProbs.draw * 100) },
    { key: "away", name: `Vitória ${awayTeam}`, value: Math.round(resultProbs.away * 100) },
  ];

  return (
    <div className="h-48 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 16, right: 24 }}>
          <XAxis type="number" domain={[0, 100]} hide />
          <YAxis
            type="category"
            dataKey="name"
            width={140}
            tick={{ fill: "#cbd5e1", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            formatter={(value) => `${value}%`}
            contentStyle={{ background: "#161e2e", border: "1px solid #232d42", borderRadius: 8 }}
            labelStyle={{ color: "#e2e8f0" }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive label={{ position: "right", fill: "#e2e8f0", formatter: (v) => `${v}%` }}>
            {data.map((entry) => (
              <Cell key={entry.key} fill={COLORS[entry.key]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
