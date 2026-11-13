import { useCallback, useSyncExternalStore } from "react";

function mediaQueryList(query: string): MediaQueryList | null {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return null;
  return window.matchMedia(query);
}

/** Whether a CSS media query matches, kept in sync with the viewport. False without matchMedia. */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      const mql = mediaQueryList(query);
      if (!mql) return () => undefined;
      if (typeof mql.addEventListener === "function") {
        mql.addEventListener("change", onChange);
        return () => mql.removeEventListener("change", onChange);
      }
      // Safari before 14 only has the deprecated listener API.
      mql.addListener(onChange);
      return () => mql.removeListener(onChange);
    },
    [query],
  );
  const getSnapshot = () => mediaQueryList(query)?.matches ?? false;
  return useSyncExternalStore(subscribe, getSnapshot, () => false);
}

/** Below this width the side panel becomes an off-canvas drawer (matches styles.css). */
export const NARROW_QUERY = "(max-width: 899.98px)";
