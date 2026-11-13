/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the UI is served by Vite and the API by the FastAPI process on
// :8000. Proxying keeps everything same-origin, exactly as in production where the
// edge Worker serves these files next to the API (see deploy/cloudflare-worker).
const API_TARGET = process.env.VITE_DEV_API_TARGET ?? "http://localhost:8000";
const API_PATHS = ["/generate", "/v1", "/health", "/ready", "/openapi.json", "/docs", "/redoc"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_PATHS.map((path) => [path, { target: API_TARGET, changeOrigin: true }]),
    ),
  },
  build: {
    sourcemap: false,
    target: "es2022",
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./test/setup.ts"],
    include: ["test/**/*.test.{ts,tsx}"],
    css: false,
  },
});
