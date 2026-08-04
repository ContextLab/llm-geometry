/**
 * The Lexicon Lab browser engine (feature 006) — the one import the tab needs.
 *
 * A from-scratch TypeScript implementation of the model, training recipe, generation and
 * spectrum specified by `specs/006-lexicon-lab-tiny/architecture.md`, mirroring
 * `code/backend/src/llm_geometry/lex/`. It has no npm dependencies of its own; the only
 * things it borrows are the geoEngine's dense-tensor kernels and error taxonomy, which
 * are engine-agnostic and already golden-tested against PyTorch.
 *
 * Typical use, end to end:
 *
 *   const vocab  = buildVocab("dolch", "full", corpusText);
 *   const cfg    = defaultConfig(vocab.rows);            // d=64, L=2, H=2, ctx=64, tied
 *   const result = runTraining({ cfg, tokens: tokenStream(corpusText, vocab) });
 *   const model  = new LexModel(cfg, result.weights);
 *   generate(model, vocab, { prompt: "the little" });    // in budget by construction
 *   spectrum(model.weights.embed, cfg.vocabRows, cfg.dModel);
 *
 * For anything longer than a demo, drive `trainWorker.ts` instead of calling
 * `runTraining` on the UI thread.
 */

export {
  DOLCH_BUDGETS,
  DOLCH_ORDER,
  FIRST,
  NOUNS,
  PRE_PRIMER,
  PRIMER,
  SECOND,
  THIRD,
  dolchBudget,
  dolchSizes,
  isDolchBudgetName,
  type DolchBudgetName,
} from "./dolch";

export {
  BOS_ID,
  BUDGET_SOURCES,
  DEFAULT_BUDGET,
  DEFAULT_BUDGET_SOURCE,
  EOS_ID,
  GENERATION_BANNED_IDS,
  LexVocab,
  PAD_ID,
  SPECIAL_TOKENS,
  UNK_ID,
  WORD_RE,
  buildVocab,
  frequencyBudget,
  hasWord,
  splitLines,
  tokenize,
  type BudgetSource,
  type Coverage,
} from "./vocab";

export {
  CTX_CHOICES,
  DEFAULT_CTX,
  DEFAULT_DROPOUT,
  DEFAULT_D_MODEL,
  DEFAULT_N_HEADS,
  DEFAULT_N_LAYERS,
  DEFAULT_TIED,
  D_MODEL_CHOICES,
  LAYER_NORM_EPS,
  LexModel,
  MLP_RATIO,
  N_HEAD_CHOICES,
  N_LAYER_CHOICES,
  backward,
  cloneWeights,
  configParamCount,
  crossEntropy,
  decayedWeightNames,
  defaultConfig,
  initWeights,
  nonFiniteWeightNames,
  paramCount,
  sfc32,
  validateConfig,
  weightNames,
  weightSizes,
  zeroGrads,
  type Activations,
  type GradSet,
  type LexConfig,
  type LossResult,
  type WeightSet,
} from "./model";

export {
  ADAM_BETA1,
  ADAM_BETA2,
  ADAM_EPS,
  DEFAULT_BATCH,
  DEFAULT_LR,
  DEFAULT_SAMPLE_EVERY,
  DEFAULT_SEED,
  DEFAULT_STEPS,
  DEFAULT_WEIGHT_DECAY,
  GRAD_CLIP_NORM,
  MAX_STEPS,
  ONECYCLE_DIV_FACTOR,
  ONECYCLE_FINAL_DIV_FACTOR,
  ONECYCLE_PCT_START,
  VAL_FRACTION,
  assertTrainable,
  evalLoss,
  oneCycleLr,
  pyRound,
  runTraining,
  sampleBatch,
  splitTokens,
  tilingBatches,
  tokenStream,
  type Batch,
  type TrainOptions,
  type TrainPoint,
  type TrainResult,
} from "./train";

export {
  DEFAULT_MAX_NEW_TOKENS,
  DEFAULT_TEMPERATURE,
  MAX_NEW_TOKENS,
  generate,
  type GenerateOptions,
  type GenerateResult,
} from "./generate";

export {
  DEFAULT_TRACE_TOPK,
  residualNorms,
  topKFromLogits,
  traceForward,
  type LensReadout,
  type LexTrace,
  type StageKind,
  type TraceOptions,
  type TraceStage,
  type TraceToken,
} from "./trace";

export {
  PCA_COMPONENTS,
  SPECTRUM_DISPLAY_K,
  centreColumns,
  gram,
  jacobiEigen,
  randomBaselineSpectrum,
  spectrum,
  spectrumStats,
  type Eigen,
  type SpectrumResult,
  type SpectrumStats,
} from "./spectrum";

export { runTrainingJob, type LexTrainRequest, type LexTrainResponse } from "./trainWorker";

export {
  LEX_BUNDLE_FORMAT,
  LEX_BUNDLE_SUFFIX,
  LEX_BUNDLE_VERSION,
  exportLexBundle,
  importLexBundle,
  lexCanonicalConfig,
  lexCanonicalVocab,
  lexEngineShapes,
  lexModelToken,
  lexVocabDigest,
  lexWeightsToken,
  lexWeightsTokenOf,
  lexWireShapes,
  toEngineName,
  toWireConfig,
  toWireName,
  type LexBundleInput,
  type LexBundleLoad,
  type LexModelBundle,
  type LexWireConfig,
} from "./bundle";

// --- the one call a UI panel needs ---------------------------------------------------

import { LexModel, defaultConfig, type LexConfig, type WeightSet } from "./model";
import { runTrainingJob, type LexTrainRequest, type LexTrainResponse } from "./trainWorker";
import { LexVocab, type BudgetSource, type Coverage } from "./vocab";
import { type TrainPoint } from "./train";

/**
 * A trained model as it crosses the worker boundary: plain data, so the vocabulary can
 * never be separated from the weights that were trained on it (feature 004's issue #6).
 */
export interface LexTrainedModel {
  config: LexConfig;
  weights: WeightSet;
  vocabWords: string[];
  budgetSource: BudgetSource;
  budgetName: string;
  coverage: Coverage;
  history: TrainPoint[];
  initialLoss: number;
  finalLoss: number;
  valLoss: number;
  nTokens: number;
  sample: { text: string; words: string[] };
}

/** Rebuild the live objects. Both take the bundle, so they cannot be mismatched. */
export const modelOf = (trained: LexTrainedModel): LexModel => new LexModel(trained.config, trained.weights);
export const vocabOf = (trained: LexTrainedModel): LexVocab =>
  new LexVocab(trained.vocabWords, trained.budgetSource, trained.budgetName);

export interface TrainProgress {
  step: number;
  totalSteps: number;
  loss: number;
  valLoss?: number;
  lr: number;
  elapsedMs: number;
  /** Present only on the steps a live sample was generated. */
  sample?: string;
}

export interface TrainInWorkerOptions extends Partial<Omit<LexConfig, "vocabRows">> {
  /** The vocabulary to train against. Passing it (rather than a budget name) is what
   *  makes a fine-tune keep the model's own budget. */
  vocab: { words: readonly string[]; source?: string; budgetName?: string };
  text: string;
  steps?: number;
  lr?: number;
  batch?: number;
  weightDecay?: number;
  sampleEvery?: number;
  seed?: number;
  samplePrompt?: string;
  sampleTemperature?: number;
  sampleMaxNewTokens?: number;
  /** Fine-tune from an existing model instead of a fresh init (FR-619). */
  initFrom?: LexTrainedModel | null;
  signal?: AbortSignal;
}

export interface TrainInWorkerResult {
  model: LexTrainedModel;
  steps: number;
  finalLoss: number;
  valLoss: number;
  elapsedMs: number;
  nTokens: number;
}

class TrainAborted extends Error {
  readonly type = "AbortError";
  constructor() {
    super("training was cancelled");
  }
}

/**
 * Train in a Web Worker so the UI thread stays responsive (FR-617), resolving with the
 * finished bundle. Where `Worker` does not exist — Node, and therefore the unit tests —
 * it runs the SAME job function inline rather than pretending to be asynchronous: one
 * code path, no mock.
 */
export function trainInWorker(
  opts: TrainInWorkerOptions,
  onProgress?: (p: TrainProgress) => void,
): Promise<TrainInWorkerResult> {
  const started = Date.now();
  const words = [...opts.vocab.words];
  const source = opts.vocab.source === "frequency" ? "frequency" : "dolch";
  const cfg = defaultConfig(words.length + 4, {
    dModel: opts.dModel,
    nLayers: opts.nLayers,
    nHeads: opts.nHeads,
    ctx: opts.ctx,
    tied: opts.tied,
    dropout: opts.dropout,
  });
  if (opts.initFrom && opts.initFrom.config.vocabRows !== cfg.vocabRows) {
    return Promise.reject(
      new Error(
        `cannot fine-tune a model with ${opts.initFrom.config.vocabRows} embedding rows against a ` +
          `${cfg.vocabRows}-row vocabulary — retrain from scratch, or keep the original budget`,
      ),
    );
  }

  const request: LexTrainRequest = {
    text: opts.text,
    budgetSource: source,
    budget: opts.vocab.budgetName ?? "custom",
    vocabWords: words,
    model: { ...cfg },
    steps: opts.steps,
    lr: opts.lr,
    batchSize: opts.batch,
    weightDecay: opts.weightDecay,
    seed: opts.seed,
    sampleEvery: opts.sampleEvery,
    samplePrompt: opts.samplePrompt,
    sampleTemperature: opts.sampleTemperature,
    sampleMaxNewTokens: opts.sampleMaxNewTokens,
    initialWeights: opts.initFrom ? opts.initFrom.weights : undefined,
  };

  return new Promise<TrainInWorkerResult>((resolve, reject) => {
    let last: TrainPoint | null = null;
    let totalSteps = opts.steps ?? 0;

    const handle = (message: LexTrainResponse): void => {
      if (message.type === "progress") {
        last = message.point;
        totalSteps = message.totalSteps;
        onProgress?.({
          step: message.point.step,
          totalSteps,
          loss: message.point.loss,
          valLoss: message.point.valLoss,
          lr: message.point.lr,
          elapsedMs: Date.now() - started,
        });
      } else if (message.type === "sample") {
        onProgress?.({
          step: message.step,
          totalSteps,
          loss: last?.loss ?? NaN,
          valLoss: last?.valLoss,
          lr: last?.lr ?? NaN,
          elapsedMs: Date.now() - started,
          sample: message.text,
        });
      } else if (message.type === "done") {
        resolve({
          model: {
            config: message.config,
            weights: message.weights,
            vocabWords: message.vocabWords,
            budgetSource: message.budgetSource === "frequency" ? "frequency" : "dolch",
            budgetName: message.budgetName,
            coverage: message.coverage,
            history: message.history,
            initialLoss: message.initialTrainLoss,
            finalLoss: message.finalTrainLoss,
            valLoss: message.valLoss,
            nTokens: message.nTokens,
            sample: message.sample,
          },
          steps: message.history.length,
          finalLoss: message.finalTrainLoss,
          valLoss: message.valLoss,
          elapsedMs: Date.now() - started,
          nTokens: message.nTokens,
        });
      } else {
        reject(new Error(message.message));
      }
    };

    if (typeof Worker === "undefined") {
      try {
        if (opts.signal?.aborted) throw new TrainAborted();
        runTrainingJob(request, handle);
      } catch (e) {
        reject(e);
      }
      return;
    }

    const worker = new Worker(new URL("./trainWorker.ts", import.meta.url), { type: "module" });
    const finish = (): void => {
      opts.signal?.removeEventListener("abort", onAbort);
      worker.terminate();
    };
    const onAbort = (): void => {
      finish();
      reject(new TrainAborted());
    };
    worker.onmessage = (event: MessageEvent<LexTrainResponse>) => {
      const message = event.data;
      handle(message);
      if (message.type === "done" || message.type === "error") finish();
    };
    worker.onerror = (event) => {
      finish();
      reject(new Error(event.message || "the training worker failed"));
    };
    opts.signal?.addEventListener("abort", onAbort, { once: true });
    if (opts.signal?.aborted) {
      onAbort();
      return;
    }
    worker.postMessage(request);
  });
}
