import { useEffect, useState } from "react";

export function useLocalStorageState(key, initial) {
  const [state, setState] = useState(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored !== null ? JSON.parse(stored) : initial;
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(state));
    } catch {
      // localStorage indisponível (modo privado, quota) — ignora silenciosamente
    }
  }, [key, state]);

  return [state, setState];
}
