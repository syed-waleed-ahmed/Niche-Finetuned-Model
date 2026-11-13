import { THEME_LABELS, nextTheme, type ThemePreference } from "../state/theme";
import { MonitorIcon, MoonIcon, SunIcon } from "./Icons";

interface Props {
  theme: ThemePreference;
  onTheme: (theme: ThemePreference) => void;
}

const ICONS = { system: MonitorIcon, light: SunIcon, dark: MoonIcon } as const;

/** Header button that cycles System, Light and Dark. Settings has the explicit choice. */
export function ThemeToggle({ theme, onTheme }: Props) {
  const next = nextTheme(theme);
  const Icon = ICONS[theme];
  return (
    <button
      type="button"
      className="btn btn--icon btn--ghost"
      onClick={() => onTheme(next)}
      aria-label={`Theme: ${THEME_LABELS[theme]}`}
      title={`Theme: ${THEME_LABELS[theme]}. Switch to ${THEME_LABELS[next]}.`}
    >
      <Icon size={18} />
    </button>
  );
}
