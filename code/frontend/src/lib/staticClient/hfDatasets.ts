/**
 * Real HuggingFace datasets, in the browser (feature 004, FR-411).
 *
 * The full stack streams datasets with the `datasets` Python package, which the static
 * build obviously cannot do. But HuggingFace runs a public, CORS-enabled REST service
 * over the same data — the "dataset viewer" (datasets-server) — so the static build can
 * read genuine rows from genuine datasets rather than refusing.
 *
 *   GET /splits?dataset=…              → available config/split pairs
 *   GET /rows?dataset=…&config=…&split=…&offset=…&length=…   → up to 100 rows per call
 *
 * Text columns are picked the same way the backend picks them (`text`, else `content`,
 * else the first string-valued column), so both runtimes build the same corpus from
 * the same dataset.
 */

import { computeError, invalidParamError } from "./errors";

const BASE = "https://datasets-server.huggingface.co";
/** The service's own hard cap per request. */
const PAGE = 100;

export interface DatasetSplit {
  dataset: string;
  config: string;
  split: string;
}

export interface DatasetFetchOptions {
  config?: string;
  split?: string;
  /** Rows to pull (the service pages at 100; this fans out as needed). */
  maxSamples?: number;
  onProgress?: (fraction: number, message: string) => void;
  fetchImpl?: typeof fetch;
}

export interface DatasetFetchResult {
  text: string;
  dataset: string;
  config: string;
  split: string;
  column: string;
  rows: number;
}

/**
 * Statuses worth trying again: the service is up but momentarily unable to answer.
 * 429 is rate limiting; 5xx is the service failing on its own side. Everything else
 * (404 unknown dataset, 401/403 gated) is a real answer and must NOT be retried — the
 * user needs to see it immediately.
 */
const TRANSIENT = (status: number): boolean => status === 429 || status >= 500;
const RETRY_DELAYS_MS = [500, 1500, 4000];

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/**
 * GET with retries for transient upstream failures, then FAIL LOUDLY.
 *
 * This is a live third-party service and it does go down: CI observed `HTTP 502` here
 * while every other test passed, and the same run saw the Hub 429-ing the static export.
 * Retrying is not about CI convenience — a reader who pastes a dataset id during a blip
 * should get their data, not an error they can do nothing about. Nothing is faked and
 * nothing is swallowed: after the last attempt the real status is raised.
 */
async function getJson(url: string, fetchImpl: typeof fetch): Promise<unknown> {
  let lastDetail = "";
  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
    let res: Response;
    try {
      res = await fetchImpl(url);
    } catch (e) {
      // A network-level failure is transient too (dropped connection, DNS blip).
      lastDetail = e instanceof Error ? e.message : String(e);
      if (attempt < RETRY_DELAYS_MS.length) {
        await sleep(RETRY_DELAYS_MS[attempt]);
        continue;
      }
      throw computeError(`Could not reach the HuggingFace dataset service: ${lastDetail}`);
    }

    if (res.ok) return res.json();

    // The service returns a useful `error` string for unknown / gated / not-yet-indexed
    // datasets; surface it rather than a bare status code.
    let detail = `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { error?: string };
      if (body?.error) detail = body.error;
    } catch {
      // non-JSON body — the status is all we have
    }

    if (TRANSIENT(res.status) && attempt < RETRY_DELAYS_MS.length) {
      lastDetail = detail;
      await sleep(RETRY_DELAYS_MS[attempt]);
      continue;
    }
    throw invalidParamError(`HuggingFace dataset service: ${detail}`);
  }
  // Unreachable: the loop either returns or throws. Kept so the type is honest.
  throw invalidParamError(`HuggingFace dataset service: ${lastDetail || "unavailable"}`);
}

/** The config/split pairs a dataset actually exposes. */
export async function listSplits(
  dataset: string,
  fetchImpl: typeof fetch = fetch,
): Promise<DatasetSplit[]> {
  const id = dataset.trim();
  if (!id) throw invalidParamError("Enter a HuggingFace dataset id, e.g. roneneldan/TinyStories");
  const data = (await getJson(
    `${BASE}/splits?dataset=${encodeURIComponent(id)}`,
    fetchImpl,
  )) as { splits?: DatasetSplit[] };
  const splits = data?.splits ?? [];
  if (splits.length === 0) {
    throw invalidParamError(`${id} exposes no splits through the HuggingFace dataset viewer.`);
  }
  return splits;
}

/**
 * Mirrors geo/finetune.load_text_from_hf's column choice.
 *
 * Column names come out of a remote JSON document, so every lookup below is own-property
 * only. Python's `dict` has no prototype chain and its half of this pair cannot be fooled;
 * a JavaScript object read with `obj[key]` can be, and the failure would not be a crash —
 * it would be this loader picking a "column" that is really an `Object.prototype` member
 * and training a model on whatever that stringifies to.
 */
function pickTextColumn(rows: Record<string, unknown>[]): string {
  const first = rows.find((r) => r && typeof r === "object");
  if (!first) throw computeError("the dataset returned no usable rows");
  for (const preferred of ["text", "content"]) {
    if (Object.hasOwn(first, preferred) && typeof first[preferred] === "string") return preferred;
  }
  for (const [k, v] of Object.entries(first)) {
    if (typeof v === "string") return k;
  }
  throw invalidParamError(
    "This dataset has no string column to read — pick a text dataset " +
      "(e.g. roneneldan/TinyStories, Salesforce/wikitext).",
  );
}

/**
 * Pull real rows and join them into one corpus, exactly as the backend does
 * ("\n\n" between records).
 */
export async function fetchDatasetText(
  dataset: string,
  opts: DatasetFetchOptions = {},
): Promise<DatasetFetchResult> {
  const fetchImpl = opts.fetchImpl ?? fetch;
  const id = dataset.trim();
  const maxSamples = Math.max(1, Math.trunc(opts.maxSamples ?? 2000));

  let config = opts.config;
  let split = opts.split;
  if (!config || !split) {
    const splits = await listSplits(id, fetchImpl);
    const preferred = splits.find((s) => s.split === (split ?? "train")) ?? splits[0];
    config = config ?? preferred.config;
    split = split ?? preferred.split;
  }

  const texts: string[] = [];
  let column = "";
  for (let offset = 0; offset < maxSamples; offset += PAGE) {
    const length = Math.min(PAGE, maxSamples - offset);
    opts.onProgress?.(
      Math.min(0.95, offset / maxSamples),
      `reading ${id} · ${split} · rows ${offset}–${offset + length}`,
    );
    const url =
      `${BASE}/rows?dataset=${encodeURIComponent(id)}&config=${encodeURIComponent(config)}` +
      `&split=${encodeURIComponent(split)}&offset=${offset}&length=${length}`;
    const data = (await getJson(url, fetchImpl)) as {
      rows?: { row?: Record<string, unknown> }[];
    };
    const rows = (data?.rows ?? []).map((r) => r?.row ?? {});
    if (rows.length === 0) break; // dataset ended before maxSamples — that's fine
    if (!column) column = pickTextColumn(rows);
    for (const row of rows) {
      // Own property only — see `pickTextColumn`. A row that does not carry the column
      // contributes nothing, rather than contributing whatever the prototype chain has
      // under that name.
      const value = Object.hasOwn(row, column) ? row[column] : undefined;
      if (typeof value === "string" && value.trim()) texts.push(value);
    }
  }

  if (texts.length === 0) {
    throw invalidParamError(
      `${id} (${split}) returned no non-empty text in column ${JSON.stringify(column || "?")}.`,
    );
  }
  opts.onProgress?.(1, `read ${texts.length} records from ${id}`);
  return {
    text: texts.join("\n\n"),
    dataset: id,
    config: config!,
    split: split!,
    column,
    rows: texts.length,
  };
}
