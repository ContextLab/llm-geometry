/**
 * Test seam for the static client: a fetchImpl that serves the REAL exported
 * assets in public/static-data/ from disk (no mock payloads — the same bytes
 * the Pages build ships), optionally passing everything else through to the
 * real network (for the safetensors range-read tests).
 */

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

import type { FetchLike } from "../../src/lib/staticClient/assets";

const STATIC_DATA_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../public/static-data",
);

export function fsStaticFetch(passthrough: boolean = false): FetchLike {
  return async (input: string, init?: RequestInit): Promise<Response> => {
    if (input.startsWith("/static-data/")) {
      const rel = input.slice("/static-data/".length);
      const file = path.join(STATIC_DATA_DIR, rel);
      if (!path.resolve(file).startsWith(STATIC_DATA_DIR)) {
        return new Response("forbidden", { status: 403 });
      }
      try {
        const buf = await readFile(file);
        return new Response(new Uint8Array(buf), { status: 200 });
      } catch {
        return new Response("not found", { status: 404 });
      }
    }
    if (passthrough) return fetch(input, init);
    throw new Error(`unexpected non-static fetch in unit test: ${input}`);
  };
}

export async function readStaticJson<T>(rel: string): Promise<T> {
  const buf = await readFile(path.join(STATIC_DATA_DIR, rel), "utf8");
  return JSON.parse(buf) as T;
}
