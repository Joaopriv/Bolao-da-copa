import { useMemo } from "react";
import { deriveRoundsAndGroups } from "../utils/rounds";

export function useRoundsAndGroups(predictions) {
  return useMemo(() => {
    if (!predictions) return [];
    const matches = deriveRoundsAndGroups(predictions);
    if (import.meta.env.DEV) {
      const missing = matches.filter((m) => !m.group || !m.round);
      if (missing.length > 0) {
        console.warn(
          "[useRoundsAndGroups] partidas sem group/round:",
          missing.map((m) => m.game)
        );
      }
    }
    return matches;
  }, [predictions]);
}
