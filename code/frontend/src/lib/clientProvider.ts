/**
 * Data-layer switch (feature 003, FR-201): the app-wide `client` is the live
 * backend client by default, or the static (GitHub Pages) client when the
 * build sets VITE_DATA_MODE=static. Views keep importing `client` from
 * dataClient unchanged — dataClient re-exports this module's pick.
 *
 * `import.meta.env.VITE_DATA_MODE` is statically replaced at build time, so
 * production backend builds dead-code-eliminate the whole staticClient graph.
 */
/// <reference types="vite/client" />

import { createClient, type Client } from "./dataClient";
import { createStaticClient } from "./staticClient";

export type DataMode = "backend" | "static";

export const DATA_MODE: DataMode =
  import.meta.env.VITE_DATA_MODE === "static" ? "static" : "backend";

export const client: Client = DATA_MODE === "static" ? createStaticClient() : createClient();
