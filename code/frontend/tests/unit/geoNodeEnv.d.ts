/**
 * Minimal ambient typings for the Node built-ins the geoEngine test suites use
 * under vitest. The frontend tsconfig deliberately pins `types` to
 * ["vitest/globals"] (no @types/node), so we declare exactly what we consume —
 * these run only in the Node-hosted test process, never in the browser bundle.
 */

declare module "node:fs" {
  const fs: {
    readFileSync(path: string, encoding: "utf-8"): string;
    existsSync(path: string): boolean;
  };
  export default fs;
}

declare module "node:path" {
  const path: {
    resolve(...parts: string[]): string;
    join(...parts: string[]): string;
    // Consumed by tests/unit/staticTestUtils.ts and tests/e2e/static.spec.ts, both of
    // which resolve a directory from `import.meta.url`. Declared here because this file
    // is the project's only description of node:path — the tsconfig pins
    // `types: ["vitest/globals"]`, so @types/node is deliberately absent.
    dirname(p: string): string;
  };
  export default path;
}

declare const __dirname: string;
