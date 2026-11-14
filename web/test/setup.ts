import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
  window.localStorage.clear();
  // The conversation is kept in sessionStorage across reloads (#41).
  window.sessionStorage.clear();
  // The theme lives on <html>, outside the rendered tree.
  document.documentElement.removeAttribute("data-theme");
});
