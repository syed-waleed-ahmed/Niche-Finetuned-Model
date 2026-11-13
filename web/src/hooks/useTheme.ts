import { useCallback, useLayoutEffect, useState } from "react";

import { applyTheme, loadTheme, saveTheme, type ThemePreference } from "../state/theme";

/** The theme preference, applied to the root element and persisted on change. */
export function useTheme(): [ThemePreference, (theme: ThemePreference) => void] {
  const [theme, setTheme] = useState<ThemePreference>(() => loadTheme());

  // Layout effect: the attribute is in place before the browser paints.
  useLayoutEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const update = useCallback((next: ThemePreference) => {
    setTheme(next);
    saveTheme(next);
  }, []);

  return [theme, update];
}
