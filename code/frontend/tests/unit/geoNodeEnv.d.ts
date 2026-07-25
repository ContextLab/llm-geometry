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
  };
  export default path;
}

declare const __dirname: string;
