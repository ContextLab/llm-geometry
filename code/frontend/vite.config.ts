import { defineConfig, type Plugin } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";

/**
 * Serve the ONNX runtime's WASM from OUR origin instead of cdn.jsdelivr.net.
 *
 * transformers.js defaults `env.backends.onnx.wasm.wasmPaths` to a jsdelivr URL, so the
 * deployed site's headline feature — in-browser generation — silently depended on a
 * third party serving a pinned prerelease. This copies the exact same files out of the
 * installed `onnxruntime-web` (byte-identical: the installed version is the one
 * transformers.js pins) and serves them at `<base>ort/`, in dev and in the build.
 */
function selfHostOnnxRuntime(): Plugin {
  // Resolved by path, not require.resolve: onnxruntime-web's `exports` map blocks
  // "./package.json", so resolution throws. Absent (it arrives via
  // @huggingface/transformers) ⇒ skip, and the runtime falls back to its CDN default
  // rather than breaking the build.
  const candidates = [
    path.resolve(process.cwd(), "node_modules/onnxruntime-web/dist"),
    path.resolve(process.cwd(), "node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist"),
  ];
  const distDir = candidates.find((d) => existsSync(d)) ?? "";
  // Only the builds the runtime actually loads: `asyncify` is what the wasm/q8 rung
  // uses and `jsep` what WebGPU uses. Copying every variant (jspi, plain, the whole
  // ort.*.mjs family and their source maps) added 94 MB of files nobody fetches.
  const NEEDED = new Set([
    "ort-wasm-simd-threaded.asyncify.wasm",
    "ort-wasm-simd-threaded.asyncify.mjs",
    "ort-wasm-simd-threaded.jsep.wasm",
    "ort-wasm-simd-threaded.jsep.mjs",
  ]);
  const wanted = (f: string) => NEEDED.has(f);
  return {
    name: "self-host-onnxruntime",
    configureServer(server) {
      if (!distDir) return;
      server.middlewares.use((req, res, next) => {
        const m = /\/ort\/([\w.-]+)$/.exec(req.url ?? "");
        if (!m || !wanted(m[1])) return next();
        try {
          const body = readFileSync(path.join(distDir, m[1]));
          res.setHeader(
            "content-type",
            m[1].endsWith(".wasm") ? "application/wasm" : "text/javascript",
          );
          res.end(body);
        } catch {
          next();
        }
      });
    },
    writeBundle(options) {
      if (!distDir) return;
      const out = path.join(options.dir ?? "dist", "ort");
      mkdirSync(out, { recursive: true });
      for (const f of readdirSync(distDir)) {
        if (wanted(f)) cpSync(path.join(distDir, f), path.join(out, f));
      }
    },
  };
}

// The frontend talks to the FastAPI backend at :8000. In dev, Vite proxies
// /api/* to it so the browser only ever sees one origin (FR-016 data layer).
export default defineConfig(({ mode }) => ({
  // GitHub Pages serves the app under /llm-geometry/ (org custom domain); the
  // deploy workflow sets PAGES_BASE. Local dev/build default to "/". (globalThis
  // avoids needing @types/node in this browser-typed project — the config itself
  // always runs under Node.)
  base: (globalThis as { process?: { env?: Record<string, string> } }).process?.env?.PAGES_BASE || "/",
  plugins: [svelte(), selfHostOnnxRuntime()],
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
