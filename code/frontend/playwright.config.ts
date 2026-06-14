import { defineConfig, devices } from "@playwright/test";

// E2E runs against the REAL backend (uvicorn) + the Vite dev server. Playwright starts
// both. Generous timeouts because the first run downloads/loads a real model.
export default defineConfig({
  testDir: "tests/e2e",
  timeout: 240_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:5173",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "sh -c '. .venv/bin/activate && uvicorn llm_geometry.api.app:app --port 8000'",
      cwd: "../backend",
      url: "http://localhost:8000/api/health",
      timeout: 120_000,
      reuseExistingServer: true,
    },
    {
      command: "npm run dev",
      url: "http://localhost:5173",
      timeout: 120_000,
      reuseExistingServer: true,
    },
  ],
});
