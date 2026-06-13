import { BarChart, Bar, XAxis, YAxis, Tooltip, ErrorBar, ResponsiveContainer } from "recharts";

export default function RpsErrorBarChart({ inSample, outOfSample }) {
  const data = [
    {
      name: `In-sample (n=${inSample.n})`,
      point: inSample.point,
      error: [inSample.point - inSample.lo, inSample.hi - inSample.point],
    },
    {
      name: `Out-of-sample (n=${outOfSample.n})`,
      point: outOfSample.point,
      error: [outOfSample.point - outOfSample.lo, outOfSample.hi - outOfSample.point],
    },
  ];

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 16, right: 16, left: -8, bottom: 0 }}>
          <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip
            formatter={(value) => value.toFixed(3)}
            contentStyle={{ background: "#161e2e", border: "1px solid #232d42", borderRadius: 8 }}
            labelStyle={{ color: "#e2e8f0" }}
          />
          <Bar dataKey="point" fill="#22d3ee" radius={[4, 4, 0, 0]} barSize={60}>
            <ErrorBar dataKey="error" width={6} strokeWidth={2} stroke="#f59e0b" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-1 text-center text-[11px] text-slate-500">
        RPS — quanto menor, melhor. Barras de erro = IC95% (bootstrap).
      </p>
    </div>
  );
}
