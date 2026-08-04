/**
 * Geometry Lab in static mode: every /api/geo/* operation delegates to the
 * golden-tested TS GeoEngine (src/lib/geoEngine/), lazily initialized from the
 * backend-exported checkpoint + vocab. Training resolves instantly (the
 * checkpoint ships precomputed); fine-tuning runs REAL SGD in the finetune
 * worker, surfaced through the same job/subscribe protocol as the backend.
 */

import type {
  CorpusStatsResult,
  GeoFinetuneBody,
  GeoFinetuneResult,
  GeoSpec,
  GeoTokenizeResult,
  GeoTrace,
  GeoTrainResult,
  GeoTrainScratchBody,
  GeoTrainScratchResult,
  GeoVectorFieldData,
  GeoVectorFieldParams,
  GeoWeightsData,
  GeoWeightsParams,
  GeoWeightsPostBody,
  GeoWeightsPostResult,
} from "../dataClient";
import { GeoEngine, runFinetune, type ExportedWeightSet, type GeoModelBundle, type WeightSet } from "../geoEngine";
import {
  buildVocabWords,
  corpusStats,
  runScratchTrain,
  uniformBaselineLoss,
  SCRATCH_DEFAULT_EPOCHS,
  SCRATCH_LEARNED_MARGIN,
  SCRATCH_MAX_EPOCHS,
} from "../geoEngine/scratch";
import { FINETUNE_MAX_UNK_RATE } from "../geoEngine/model";
import type { ScratchWorkerRequest, ScratchWorkerResponse } from "../geoEngine/scratchWorker";
import { GeoTokenizer } from "../geoEngine/tokenizer";
import { fetchDatasetText } from "./hfDatasets";
import type { FinetuneWorkerRequest, FinetuneWorkerResponse } from "../geoEngine";
import type { StaticAssets } from "./assets";
import { computeError, invalidParamError, staticModeError, toApiError } from "./errors";
import type { LocalJobRegistry, ProgressFn } from "./jobs";

// Contract-documented defaults (specs/002 contracts/api.md): steps ≤ 500
// default 100, lr default 1e-2. Kept literal here to avoid importing geoEngine
// internals (model.ts) — geoEngine.test.ts pins the same values.
// Minted weight sets persist across reloads (red-team static finding #3: the
// engine's store is in-memory, so a sessionStorage token would otherwise silently
// self-heal back to "learned" after a reload — the backend build persists edits).
//
// A set trained from scratch or loaded from a file also carries its OWN vocabulary,
// and `exportWeightSet` puts it in the payload: weights alone do not describe such a
// model, and restoring them alone made `Save model` write a file pairing them with the
// shipped word list under a matching `vocab_sha256` — an unrejectable wrong file. The
// backend has always stored the two together (`save_weight_set(..., vocab_json=…)`).
//
// Payloads written before `ownsVocab` existed recorded vocabulary ownership only through
// `setSource`, and a `finetuned`/`edited` payload derived from a scratch model was written
// WITHOUT the word list it needs — restoring one would revive exactly the corruption this
// fix removes, and nothing in the payload can distinguish it from a legitimate fine-tune
// of the shipped model. `GeoEngine.importWeightSet` REFUSES such a payload outright and
// `restorePersistedSets` drops it; the version suffix on the key is belt-and-braces, not
// the defence. (It was the defence once, and a key rename is not one: it hides only the
// payloads this build wrote, not one copied between profiles or restored from a backup.)
const MINTED_SETS_KEY = "llm-geometry:static-weight-sets:v2";
const MINTED_SETS_CAP = 8; // LRU; each entry is ~50 KB of JSON, ~60 KB with a vocabulary

function loadPersistedSets(): Record<string, ExportedWeightSet> {
  try {
    const raw = sessionStorage.getItem(MINTED_SETS_KEY);
    return raw ? (JSON.parse(raw) as Record<string, ExportedWeightSet>) : {};
  } catch {
    return {};
  }
}

function persistMintedSet(engine: GeoEngine, token: string): void {
  try {
    const all = loadPersistedSets();
    delete all[token]; // re-insert -> most recent
    all[token] = engine.exportWeightSet(token);
    const keys = Object.keys(all);
    for (let i = 0; i < keys.length - MINTED_SETS_CAP; i++) delete all[keys[i]];
    sessionStorage.setItem(MINTED_SETS_KEY, JSON.stringify(all));
  } catch {
    // Quota/serialization failure: the edit still works this session; the
    // evicted-token self-heal covers the next reload.
  }
}

function restorePersistedSets(engine: GeoEngine): void {
  const all = loadPersistedSets();
  let changed = false;
  for (const [token, payload] of Object.entries(all)) {
    if (!engine.importWeightSet(token, payload)) {
      delete all[token]; // hash mismatch / corrupted — drop it
      changed = true;
    }
  }
  if (changed) {
    try {
      sessionStorage.setItem(MINTED_SETS_KEY, JSON.stringify(all));
    } catch {
      /* best-effort */
    }
  }
}

const FINETUNE_DEFAULT_STEPS = 100;
const FINETUNE_MAX_STEPS = 500;
const FINETUNE_DEFAULT_LR = 0.01;

interface GeoCheckpointAsset {
  [k: string]: unknown;
}

interface FinetuneRunResult {
  weights: WeightSet;
  lossBefore: number;
  lossAfter: number;
}

/** Blob → text; falls back to FileReader where Blob.text() is missing (jsdom). */
function readBlobText(file: Blob): Promise<string> {
  if (typeof (file as { text?: unknown }).text === "function") return file.text();
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(computeError("could not read the uploaded file"));
    reader.readAsText(file);
  });
}

export class GeoSection {
  private enginePromise: Promise<GeoEngine> | null = null;
  private readonly finetuneCache = new Map<string, GeoFinetuneResult>();
  private readonly scratchCache = new Map<string, GeoTrainScratchResult>();

  constructor(
    private readonly assets: StaticAssets,
    private readonly jobs: LocalJobRegistry,
  ) {}

  /** Lazy engine boot from the exported static assets (checkpoint integrity is
   * verified inside GeoEngine: content hash must equal the recorded id). */
  engine(): Promise<GeoEngine> {
    if (!this.enginePromise) {
      this.enginePromise = (async () => {
        const [checkpoint, vocab, spec] = await Promise.all([
          this.assets.json<GeoCheckpointAsset>("geo/checkpoint.json"),
          this.assets.json<unknown>("geo/vocab.json"),
          this.assets.json<GeoSpec>("geo/spec.json"),
        ]);
        const engine = GeoEngine.fromAssets(checkpoint, vocab);
        // Cross-check the export's own spec: same checkpoint everywhere.
        const specId = spec.checkpoint?.checkpoint_id;
        if (specId && specId !== engine.canonicalToken) {
          throw computeError(
            `static geo assets disagree: spec.json checkpoint_id ${specId} != ` +
              `checkpoint content hash ${engine.canonicalToken}`,
          );
        }
        restorePersistedSets(engine); // reload keeps minted edits/fine-tunes
        return engine;
      })().catch((e) => {
        this.enginePromise = null;
        throw toApiError(e);
      });
    }
    return this.enginePromise;
  }

  private async run<T>(fn: (engine: GeoEngine) => T): Promise<T> {
    const engine = await this.engine();
    try {
      return fn(engine);
    } catch (e) {
      throw toApiError(e);
    }
  }

  getGeoSpec(): Promise<GeoSpec> {
    return this.run((e) => e.spec());
  }

  /** The checkpoint ships precomputed → training is always an instant 200. */
  async geoTrain(seed?: number): Promise<GeoTrainResult> {
    const engine = await this.engine();
    if (seed !== undefined && seed !== engine.meta.seed) {
      throw staticModeError(
        `The static build ships the precomputed seed-${engine.meta.seed} checkpoint; ` +
          "training with a different seed needs the full stack (see the README).",
      );
    }
    return { ready: true, checkpoint_id: engine.canonicalToken, status: "ready" };
  }

  geoTokenize(text: string, weightsToken?: string): Promise<GeoTokenizeResult> {
    // A scratch-trained model has its own vocabulary — tokenizing with the canonical
    // one would mislabel every chip in the token strip.
    return this.run((e) => e.tokenize(text, weightsToken));
  }

  getGeoTrace(prompt: string, weightsToken?: string): Promise<GeoTrace> {
    return this.run((e) => e.trace(prompt, weightsToken));
  }

  getGeoVectorField(params: GeoVectorFieldParams): Promise<GeoVectorFieldData> {
    return this.run((e) => e.vectorField(params));
  }

  getGeoWeights(params: GeoWeightsParams): Promise<GeoWeightsData> {
    return this.run((e) => e.getWeights(params));
  }

  postGeoWeights(body: GeoWeightsPostBody): Promise<GeoWeightsPostResult> {
    return this.run((e) => {
      const result = e.postWeights(body);
      persistMintedSet(e, result.weights_token);
      return result;
    });
  }

  /**
   * Fine-tune: 202-style {ready:false, job_id} + progress via the local job
   * registry; the done payload carries {weights_token, loss_before, loss_after}
   * exactly like the backend's SSE done event. Identical requests are 200-style
   * cache hits. `hf_dataset` needs the full stack's streaming → StaticModeError.
   */
  async geoFinetune(body: GeoFinetuneBody): Promise<GeoFinetuneResult> {
    // HuggingFace datasets DO work here (feature 004): the Hub's public, CORS-enabled
    // dataset-viewer service serves real rows, so the static build reads genuine data
    // instead of refusing. Same column choice as the backend.
    let text = body.text;
    if (body.hf_dataset != null && body.hf_dataset !== "") {
      const pulled = await fetchDatasetText(String(body.hf_dataset), { maxSamples: 500 });
      text = pulled.text;
    }
    if (text == null || text.trim().length === 0) {
      throw invalidParamError("Provide non-empty fine-tuning text (exactly one of text/hf_dataset).");
    }
    const steps = Math.trunc(body.steps ?? FINETUNE_DEFAULT_STEPS);
    if (!(steps >= 1 && steps <= FINETUNE_MAX_STEPS)) {
      throw invalidParamError(`steps must be in 1..${FINETUNE_MAX_STEPS}, got ${body.steps}`);
    }
    const lr = body.lr ?? FINETUNE_DEFAULT_LR;
    if (!(lr > 0)) throw invalidParamError(`lr must be > 0, got ${body.lr}`);
    const base = body.base ?? "learned";

    const engine = await this.engine();
    const baseWeights = this.resolveBaseWeights(engine, base);
    const baseToken = base === "learned" ? engine.canonicalToken : base;
    const cacheKey = JSON.stringify({ base: baseToken, text, steps, lr, seed: 0 });
    const cached = this.finetuneCache.get(cacheKey);
    if (cached) return { ...cached };

    // The ACTIVE model's own vocabulary, not the canonical one (issue #6): encoding a
    // scratch model's fine-tuning corpus with the shipped Alice words turned the whole
    // stream into <unk> while the UI still reported "loss 6.58 → 5.58 on your text".
    const enc = engine.tokenizerFor(base).encode(text, { truncate: false });
    const tokenIds = enc.ids;
    if (tokenIds.length < 2) {
      throw invalidParamError(
        "fine-tuning text is too short after tokenization (need at least 2 tokens)",
      );
    }
    const unkRate = enc.n_unk / tokenIds.length;
    // `>=`, mirroring the backend and the engine: EXACTLY 90 % <unk> is refused, not
    // accepted and reported as a clean loss drop one token below the refusal.
    if (unkRate >= FINETUNE_MAX_UNK_RATE) {
      throw invalidParamError(
        `${enc.n_unk} of ${tokenIds.length} tokens (${(unkRate * 100).toFixed(1)}%, the ` +
          `limit is ${Math.round(FINETUNE_MAX_UNK_RATE * 100)}%) in this text are outside ` +
          "the active model's vocabulary, so fine-tuning on it would mostly teach the " +
          "model to emit <unk> and the loss would say nothing about your words. Use " +
          "'Train a new model' to build a vocabulary from this text instead.",
      );
    }
    const jobId = this.jobs.create(cacheKey, async (report) => {
      const result = await this.runFinetuneAsync(
        { baseWeights, tokenIds, steps, lr, seed: 0 },
        report,
      );
      const token = engine.registerFinetunedWeights(result.weights, base);
      persistMintedSet(engine, token);
      const payload = {
        weights_token: token,
        loss_before: result.lossBefore,
        loss_after: result.lossAfter,
        n_tokens: tokenIds.length,
        n_unk: enc.n_unk,
        unk_rate: unkRate,
      };
      this.finetuneCache.set(cacheKey, { ready: true, ...payload });
      return payload;
    });
    return { ready: false, job_id: jobId };
  }

  // --- from-scratch training + portable models (feature 004) ------------------------

  /** Token / distinct-type counts — the same numbers GET /api/geo/corpus_stats reports. */
  async geoCorpusStats(text: string): Promise<CorpusStatsResult> {
    return corpusStats(text);
  }

  /**
   * Train a BRAND NEW model on the user's own corpus, really, in this browser.
   *
   * Fine-tuning keeps the shipped vocabulary; this rebuilds it from the supplied text
   * and starts from fresh weights, so the result is a different model. The run happens
   * in a Worker (it is minutes of arithmetic, not the fine-tune's fraction of a
   * second) and reports through the same job protocol as the backend's SSE.
   */
  async geoTrainScratch(body: GeoTrainScratchBody): Promise<GeoTrainScratchResult> {
    const epochs = Math.trunc(body.epochs ?? SCRATCH_DEFAULT_EPOCHS);
    if (!(epochs >= 1 && epochs <= SCRATCH_MAX_EPOCHS)) {
      throw invalidParamError(`epochs must be in 1..${SCRATCH_MAX_EPOCHS}, got ${body.epochs}`);
    }
    const sources = [body.text, body.hf_dataset].filter((v) => v != null && String(v).trim() !== "");
    if (sources.length !== 1) {
      throw invalidParamError(
        `exactly one of text / hf_dataset must be provided, got ${sources.length} sources`,
      );
    }

    const engine = await this.engine();
    const cacheKey = JSON.stringify({ scratch: sources[0], epochs });
    const cached = this.scratchCache.get(cacheKey);
    if (cached) return { ...cached };

    const jobId = this.jobs.create(cacheKey, async (report) => {
      let text = body.text ? String(body.text) : "";
      if (body.hf_dataset) {
        const pulled = await fetchDatasetText(String(body.hf_dataset), {
          maxSamples: body.max_samples ?? 2000,
          onProgress: (f, m) => report(0.02 * f, m),
        });
        text = pulled.text;
      }
      const result = await this.runScratchAsync({ text, epochs, seed: 0 }, report);
      const token = engine.registerScratchModel(
        result.weights as WeightSet,
        result.vocabWords,
      );
      persistMintedSet(engine, token);
      const payload = {
        weights_token: token,
        vocab_size: new GeoTokenizer(result.vocabWords).idToText.size,
        final_loss: result.finalLoss,
        // Whether the run actually left the uniform-distribution baseline. Without it
        // a run that learned nothing reads exactly like one that learned something.
        uniform_baseline: uniformBaselineLoss(),
        learned: result.finalLoss < uniformBaselineLoss() - SCRATCH_LEARNED_MARGIN,
        n_tokens: result.nTokens,
        n_distinct: result.nDistinct,
        epochs: result.epochs,
      };
      this.scratchCache.set(cacheKey, { ready: true, ...payload });
      return payload;
    });
    return { ready: false, job_id: jobId };
  }

  /** The active model as one portable file (weights + the vocabulary its ids mean). */
  geoExportModel(weightsToken?: string): Promise<GeoModelBundle> {
    return this.run((e) => e.exportBundle(weightsToken));
  }

  /** Validate and load a saved model file; returns its token. */
  geoImportModel(bundle: unknown): Promise<{ weights_token: string; vocab_size: number }> {
    return this.run((e) => {
      const out = e.importBundle(bundle);
      persistMintedSet(e, out.weights_token);
      return out;
    });
  }

  /** Real training in the scratch worker; synchronous when Workers don't exist
   * (vitest/jsdom) — the same code the worker itself runs. */
  private runScratchAsync(
    req: ScratchWorkerRequest,
    report: ProgressFn,
  ): Promise<Extract<ScratchWorkerResponse, { type: "done" }>> {
    if (typeof Worker === "undefined") {
      return Promise.resolve().then(() => {
        const words = buildVocabWords(req.text);
        const tokenizer = new GeoTokenizer(words);
        const tokenIds = tokenizer.encodeStream(req.text);
        const r = runScratchTrain({
          tokenIds,
          epochs: req.epochs,
          seed: req.seed,
          onProgress: report,
        });
        return {
          type: "done" as const,
          weights: r.weights as Record<string, Float32Array>,
          vocabWords: words,
          finalLoss: r.finalLoss,
          epochs: r.epochs,
          nTokens: tokenIds.length,
          nDistinct: new Set(words).size,
        };
      });
    }
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL("../geoEngine/scratchWorker.ts", import.meta.url), {
        type: "module",
      });
      worker.onmessage = (ev: MessageEvent<ScratchWorkerResponse>) => {
        const msg = ev.data;
        if (msg.type === "progress") {
          report(msg.fraction, msg.message);
          return;
        }
        worker.terminate();
        if (msg.type === "done") resolve(msg);
        else reject(toApiError(Object.assign(new Error(msg.message), { type: msg.errorType })));
      };
      worker.onerror = (ev) => {
        worker.terminate();
        reject(computeError(`training worker failed: ${ev.message || "unknown error"}`));
      };
      worker.postMessage(req);
    });
  }

  async geoFinetuneFile(
    file: Blob,
    filename: string,
    options: Omit<GeoFinetuneBody, "text" | "hf_dataset"> = {},
  ): Promise<GeoFinetuneResult> {
    if (!/\.(txt|md)$/i.test(filename)) {
      throw invalidParamError(`Only .txt/.md files are accepted, got '${filename}'.`);
    }
    const text = await readBlobText(file);
    return this.geoFinetune({ ...options, text });
  }

  /**
   * Raw base weights for the worker. The canonical set is public; minted sets
   * live in the engine's private store — reached via its (runtime-public)
   * resolveWeightSet so unknown tokens fail with the engine's own
   * NotFoundError. A unit test pins this method's existence.
   */
  private resolveBaseWeights(engine: GeoEngine, base: string): WeightSet {
    if (base === "learned" || base === engine.canonicalToken) return engine.canonical;
    const resolver = (engine as unknown as { resolveWeightSet?: (t: string) => WeightSet })
      .resolveWeightSet;
    if (typeof resolver !== "function") {
      throw computeError(
        "GeoEngine.resolveWeightSet is missing — the staticClient fine-tune bridge " +
          "must be updated to match the engine",
      );
    }
    try {
      return resolver.call(engine, base);
    } catch (e) {
      throw toApiError(e);
    }
  }

  /** Real SGD in the finetune worker; synchronous engine math when Workers
   * don't exist (vitest/jsdom) — same code path the worker itself runs. */
  private runFinetuneAsync(
    req: FinetuneWorkerRequest,
    report: ProgressFn,
  ): Promise<FinetuneRunResult> {
    if (typeof Worker === "undefined") {
      return Promise.resolve().then(() => {
        const r = runFinetune({
          baseWeights: req.baseWeights as WeightSet,
          tokenIds: req.tokenIds,
          steps: req.steps,
          lr: req.lr,
          seed: req.seed,
          onProgress: report,
        });
        return { weights: r.weights, lossBefore: r.lossBefore, lossAfter: r.lossAfter };
      });
    }
    return new Promise<FinetuneRunResult>((resolve, reject) => {
      const worker = new Worker(new URL("../geoEngine/finetuneWorker.ts", import.meta.url), {
        type: "module",
      });
      const done = (fn: () => void): void => {
        worker.terminate();
        fn();
      };
      worker.onmessage = (ev: MessageEvent<FinetuneWorkerResponse>) => {
        const msg = ev.data;
        if (msg.type === "progress") report(msg.fraction, msg.message);
        else if (msg.type === "done") {
          done(() =>
            resolve({
              weights: msg.weights as WeightSet,
              lossBefore: msg.lossBefore,
              lossAfter: msg.lossAfter,
            }),
          );
        } else {
          done(() => reject(toApiError(Object.assign(new Error(msg.message), { type: msg.errorType }))));
        }
      };
      worker.onerror = (ev) => {
        done(() => reject(computeError(`fine-tune worker failed: ${ev.message || "unknown error"}`)));
      };
      worker.postMessage(req);
    });
  }
}
