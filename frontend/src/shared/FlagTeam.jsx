import { getFlag } from "../data/flags";

export default function FlagTeam({ name, align = "left", className = "" }) {
  const reverse = align === "right";
  return (
    <div
      className={`flex items-center gap-2 ${reverse ? "flex-row-reverse text-right" : ""} ${className}`}
    >
      <span className="text-xl leading-none">{getFlag(name)}</span>
      <span className="truncate font-medium">{name}</span>
    </div>
  );
}
