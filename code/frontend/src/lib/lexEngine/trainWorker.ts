/**
 * Web Worker wrapper around `runTraining`, so a real from-scratch run never freezes the
 * UI thread (FR-617). Spawn with:
 *
 *   new Worker(new URL("./trainWorker.ts", import.meta.url), { type: "module" })
 *
 * Protocol (the geoEngine's, extended with live samples): postMessage(LexTrainRequest)
 * -> a stream of {type:"progress"} and {type:"sample"} messages, then exactly one
 * {type:"done"} or {type:"error"}. Everything that crosses the boundary is
 * structured-cloneable: weights travel as plain Float32Arrays, the vocabulary as
 * string[], so the main thread can rebuild a LexModel without a second training run.
 *
 * The worker owns the whole pipeline — build the budget, encode the corpus, train — so
 * the caller cannot accidentally pair a model with a vocabulary it was not trained on
 * (feature 004's issue #6).
 */

import { GeoEngineError } from "../geoEngine/errors";
import { generate, type GenerateResult } from "./generate";
import { LexModel, defaultConfig, type LexConfig, type WeightSet } from "./model";
import { DEFAULT_STEPS, assertTrainable, runTraining, tokenStream, type TrainPoint } from "./train";
import { LexVocab, buildVocab, type Coverage } from "./vocab";

export interface LexTrainRequest {
  /** Raw corpus text. The worker encodes it with the vocabulary below. */
  text: string;
  /** "dolch" | "frequency". */
  budgetSource: string;
  /** A graded Dolch name ("full", "service", ...); also sizes a frequency budget. */
  budget: string;
  /** Explicit |V| for a frequency budget; defaults to the matching Dolch size. */
  budgetSize?: number;
  /**
   * An EXPLICIT budget, which wins over `budgetSource`/`budget`. Fine-tuning must pass
   * this: the model keeps the vocabulary it was trained on even when the new text would
   * have produced a different one (feature 004's issue #6).
   */
  vocabWords?: string[];
  /** Continue from these weights rather than a fresh init (fine-tuning, FR-619). */
  initialWeights?: WeightSet;
  /** Model dimensions. Anything omitted takes the documented default. */
  model?: Partial<Omit<LexConfig, "vocabRows">>;
  steps?: number;
  lr?: number;
  batchSize?: number;
  weightDecay?: number;
  seed?: number;
  sampleEvery?: number;
  /** Prompt used for the live samples and the final one. */
  samplePrompt?: string;
  sampleTemperature?: number;
  sampleMaxNewTokens?: number;
}

export type LexTrainResponse =
  | { type: "progress"; fraction: number; message: string; point: TrainPoint; totalSteps: number }
  | { type: "sample"; step: number; text: string; words: string[] }
  | {
      type: "done";
      weights: WeightSet;
      config: LexConfig;
      vocabWords: string[];
      budgetSource: string;
      budgetName: string;
      coverage: Coverage;
      history: TrainPoint[];
      initialTrainLoss: number;
      finalTrainLoss: number;
      valLoss: number;
      nTokens: number;
      sample: { text: string; words: string[] };
    }
  | { type: "error"; errorType: string; message: string };

interface WorkerScope {
  onmessage: ((event: MessageEvent<LexTrainRequest>) => void) | null;
  postMessage: (message: LexTrainResponse) => void;
}

/**
 * True only inside a real worker: `self` exists there and in a window, but `document`
 * exists only in a window and neither exists under Node. Without this guard, merely
 * importing the module (as `index.ts` and the unit tests do, for `runTrainingJob` and
 * the response types) would install a handler on the page — or throw under vitest.
 */
const inWorker = typeof self !== "undefined" && typeof document === "undefined";
const workerScope = inWorker ? (self as unknown as WorkerScope) : null;

/** Everything the worker does, factored out so a test can drive it without a Worker. */
export function runTrainingJob(
  req: LexTrainRequest,
  emit: (message: LexTrainResponse) => void,
): void {
  const vocab: LexVocab = req.vocabWords
    ? new LexVocab(req.vocabWords, req.budgetSource === "frequency" ? "frequency" : "dolch", req.budget)
    : buildVocab(req.budgetSource, req.budget, req.text, req.budgetSize);
  const cfg = defaultConfig(vocab.rows, req.model ?? {});
  // `tokenStream`, never `encodeText`: the contract's stream closes every non-blank line
  // with `<eos>`, which is what lets the model learn where a line of verse ends. This is
  // the single place training data is built, for both from-scratch runs and fine-tunes.
  const tokens = tokenStream(req.text, vocab);
  assertTrainable(tokens.length, cfg.ctx);
  const steps = req.steps ?? DEFAULT_STEPS;

  const sampleOf = (model: LexModel, seed: number): GenerateResult =>
    generate(model, vocab, {
      prompt: req.samplePrompt,
      temperature: req.sampleTemperature,
      maxNewTokens: req.sampleMaxNewTokens,
      seed,
    });

  const result = runTraining({
    cfg,
    tokens,
    steps,
    lr: req.lr,
    batchSize: req.batchSize,
    weightDecay: req.weightDecay,
    seed: req.seed,
    initialWeights: req.initialWeights,
    sampleEvery: req.sampleEvery,
    onProgress: (fraction, point) =>
      emit({
        type: "progress",
        fraction,
        message: `step ${point.step} · loss ${point.loss.toFixed(3)} · lr ${point.lr.toExponential(2)}`,
        point,
        totalSteps: steps,
      }),
    onSample: (step, model) => {
      const s = sampleOf(model, step);
      emit({ type: "sample", step, text: s.text, words: s.words });
    },
  });

  const model = new LexModel(cfg, result.weights);
  const final = sampleOf(model, result.steps);
  emit({
    type: "done",
    weights: result.weights,
    config: cfg,
    vocabWords: [...vocab.words],
    budgetSource: vocab.source,
    budgetName: vocab.budgetName,
    coverage: vocab.coverage(req.text),
    history: result.history,
    initialTrainLoss: result.initialTrainLoss,
    finalTrainLoss: result.finalTrainLoss,
    valLoss: result.valLoss,
    nTokens: result.nTokens,
    sample: { text: final.text, words: final.words },
  });
}

if (workerScope) {
  workerScope.onmessage = (event: MessageEvent<LexTrainRequest>) => {
    try {
      runTrainingJob(event.data, (message) => workerScope.postMessage(message));
    } catch (e) {
      const err = e as GeoEngineError;
      workerScope.postMessage({
        type: "error",
        errorType: err?.type ?? "ComputeError",
        message: err?.message ?? String(e),
      });
    }
  };
}
