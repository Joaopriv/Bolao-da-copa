const VARIANTS = {
  warn: "bg-warn/15 text-warn border-warn/40",
  good: "bg-good/15 text-good border-good/40",
  bad: "bg-bad/15 text-bad border-bad/40",
  brand: "bg-brand/15 text-brand border-brand/40",
  neutral: "bg-base-700 text-slate-300 border-base-600",
};

export default function Badge({ children, variant = "neutral", className = "" }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${VARIANTS[variant]} ${className}`}
    >
      {children}
    </span>
  );
}
