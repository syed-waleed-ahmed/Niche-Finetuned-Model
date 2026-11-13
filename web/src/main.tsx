import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { applyTheme, loadTheme } from "./state/theme";
import "./styles.css";

// Apply a saved light/dark choice before the first render. The CSP forbids inline
// scripts, so this is the earliest point at which it can run.
applyTheme(loadTheme());

// During a rollout the HTML shell and the hashed bundles can come from different
// releases; a failed chunk load is fixed by reloading the page.
window.addEventListener("vite:preloadError", (event) => {
  event.preventDefault();
  window.location.reload();
});

const container = document.getElementById("root");
if (!container) throw new Error("Missing #root element");

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
