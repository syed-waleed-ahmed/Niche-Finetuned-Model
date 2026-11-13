import { useCallback, useEffect, useState } from "react";

import { loadSettings, saveSettings, type Settings } from "../state/settings";

export function useSettings(): [Settings, (update: Partial<Settings>) => void] {
  const [settings, setSettings] = useState<Settings>(() => loadSettings());

  useEffect(() => {
    saveSettings(settings);
  }, [settings]);

  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((current) => ({ ...current, ...patch }));
  }, []);

  return [settings, update];
}
