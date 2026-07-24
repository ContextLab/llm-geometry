import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// The frontend talks to the FastAPI backend at :8000. In dev, Vite proxies
// /api/* to it so the browser only ever sees one origin (FR-016 data layer).
export default defineConfig(({ mode }) => ({
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
