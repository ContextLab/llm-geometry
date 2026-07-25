import { defineConfig, devices } from "@playwright/test";

// Two projects (feature 003, FR-206):
//   chromium — the ORIGINAL suite against the REAL backend (uvicorn) + Vite dev server.
//   static   — tests/e2e/static.spec.ts against the BUILT static site (`npm run
//              preview:static` on :4173, Pages base path, NO Python backend), proving
//              the GitHub Pages build stands alone.
// Playwright's webServer list is global (not per-project), so the servers to boot are
// chosen from the --project filter on the command line: `--project static` must not
// require a backend venv, and `--project chromium` must not pay for a static build.
// With no filter (plain `npx playwright test`) all servers start and BOTH projects run.
const argv: string[] =
  (globalThis as { process?: { argv?: string[] } }).process?.argv ?? [];
const projectFilters: string[] = [];
argv.forEach((a, i) => {
  if (a === "--project" && argv[i + 1]) projectFilters.push(argv[i + 1]);
  else if (a.startsWith("--project=")) projectFilters.push(a.slice("--project=".length));
});
const wants = (name: string): boolean =>
  projectFilters.length === 0 || projectFilters.includes(name);

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 240_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    // Allow software WebGL in headless Chromium so the Three.js manifold renders.
    launchOptions: { args: ["--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], baseURL: "http://localhost:5173" },
      testIgnore: /static\.spec\.ts/,
    },
    {
      name: "static",
      // The site is served under the GitHub Pages base path (/llm-geometry/) so the
      // static suite also exercises FR-205 (BASE_URL-relative assets + deep links).
      use: { ...devices["Desktop Chrome"], baseURL: "http://localhost:4173" },
      testMatch: /static\.spec\.ts/,
    },
  ],
  webServer: [
    ...(wants("chromium")
      ? [
          {
            command:
              "sh -c '. .venv/bin/activate && uvicorn llm_geometry.api.app:app --port 8000'",
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
        ]
      : []),
    ...(wants("static")
      ? [
          {
            // Builds the static bundle first (same flags as the Pages deploy), then
            // previews it — generous timeout for a cold `vite build`.
            command: "npm run preview:static",
            url: "http://localhost:4173/llm-geometry/",
            timeout: 300_000,
            reuseExistingServer: true,
          },
        ]
      : []),
  ],
});
