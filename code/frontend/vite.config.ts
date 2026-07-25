import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// The frontend talks to the FastAPI backend at :8000. In dev, Vite proxies
// /api/* to it so the browser only ever sees one origin (FR-016 data layer).
export default defineConfig(({ mode }) => ({
  // GitHub Pages serves the app under /llm-geometry/ (org custom domain); the
  // deploy workflow sets PAGES_BASE. Local dev/build default to "/". (globalThis
  // avoids needing @types/node in this browser-typed project — the config itself
  // always runs under Node.)
  base: (globalThis as { process?: { env?: Record<string, string> } }).process?.env?.PAGES_BASE || "/",
  plugins: [svelte()],
  // Under vitest (mode "test"), resolve Svelte's browser build (client-side
  // mount/effects); without this the "default" condition picks svelte/index-server,
  // whose mount() throws server_context_required in component tests.
  resolve: mode === "test" ? { conditions: ["browser"] } : undefined,
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    include: ["tests/unit/**/*.test.ts"],
    globals: true,
  },
}));
