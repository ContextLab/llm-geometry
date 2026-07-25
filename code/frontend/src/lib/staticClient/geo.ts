/**
 * Geometry Lab in static mode: every /api/geo/* operation delegates to the
 * golden-tested TS GeoEngine (src/lib/geoEngine/), lazily initialized from the
 * backend-exported checkpoint + vocab. Training resolves instantly (the
 * checkpoint ships precomputed); fine-tuning runs REAL SGD in the finetune
 * worker, surfaced through the same job/subscribe protocol as the backend.
 */

import type {
  GeoFinetuneBody,
  GeoFinetuneResult,
  GeoSpec,
  GeoTokenizeResult,
  GeoTrace,
  GeoTrainResult,
  GeoVectorFieldData,
  GeoVectorFieldParams,
  GeoWeightsData,
  GeoWeightsParams,
  GeoWeightsPostBody,
  GeoWeightsPostResult,
} from "../dataClient";
import { GeoEngine, runFinetune, type WeightSet } from "../geoEngine";
import type { FinetuneWorkerRequest, FinetuneWorkerResponse } from "../geoEngine";
import type { StaticAssets } from "./assets";
import { computeError, invalidParamError, staticModeError, toApiError } from "./errors";
import type { LocalJobRegistry, ProgressFn } from "./jobs";

// Contract-documented defaults (specs/002 contracts/api.md): steps ≤ 500
// default 100, lr default 1e-2. Kept literal here to avoid importing geoEngine
// internals (model.ts) — geoEngine.test.ts pins the same values.
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

  geoTokenize(text: string): Promise<GeoTokenizeResult> {
    return this.run((e) => e.tokenize(text));
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
    return this.run((e) => e.postWeights(body));
  }

  /**
   * Fine-tune: 202-style {ready:false, job_id} + progress via the local job
   * registry; the done payload carries {weights_token, loss_before, loss_after}
   * exactly like the backend's SSE done event. Identical requests are 200-style
   * cache hits. `hf_dataset` needs the full stack's streaming → StaticModeError.
   */
  async geoFinetune(body: GeoFinetuneBody): Promise<GeoFinetuneResult> {
    if (body.hf_dataset != null && body.hf_dataset !== "") {
      throw staticModeError(
        "Fine-tuning on a Hugging Face dataset needs the full stack's streaming " +
          "download — the static demo can't fetch datasets. Paste text or upload " +
          "a .txt/.md file instead, or run the full stack (see the README).",
      );
    }
    const text = body.text;
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

    const tokenIds = engine.tokenizer.encodeStream(text);
    const jobId = this.jobs.create(cacheKey, async (report) => {
      const result = await this.runFinetuneAsync(
        { baseWeights, tokenIds, steps, lr, seed: 0 },
        report,
      );
      const token = engine.registerFinetunedWeights(result.weights);
      const payload = {
        weights_token: token,
        loss_before: result.lossBefore,
        loss_after: result.lossAfter,
      };
      this.finetuneCache.set(cacheKey, { ready: true, ...payload });
      return payload;
    });
    return { ready: false, job_id: jobId };
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
