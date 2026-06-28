import { useMemo } from "react";
import { deriveRoundsAndGroups } from "../utils/rounds";

export function useRoundsAndGroups(predictions) {
  return useMemo(() => {
    if (!predictions) return [];
    const matches = deriveRoundsAndGroups(predictions);
    if (import.meta.env.DEV) {
      // group é null no mata-mata (confrontos cruzam grupos) -- só round é obrigatório.
      const missing = matches.filter((m) => !m.round);
      if (missing.length > 0) {
        console.warn(
          "[useRoundsAndGroups] partidas sem round:",
          missing.map((m) => m.game)
        );
      }
    }
    return matches;
  }, [predictions]);
}
