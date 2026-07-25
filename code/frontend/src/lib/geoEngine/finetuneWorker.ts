/**
 * Web Worker wrapper around runFinetune so static-mode fine-tuning never blocks
 * the UI thread. Spawn with:
 *
 *   new Worker(new URL("./finetuneWorker.ts", import.meta.url), { type: "module" })
 *
 * Protocol: postMessage(FinetuneWorkerRequest) -> a stream of
 * {type:"progress"} messages followed by one {type:"done"} or {type:"error"}.
 * Weight tensors travel as plain Float32Arrays (structured-cloneable).
 */

import { GeoEngineError } from "./errors";
import type { WeightSet } from "./model";
import { runFinetune } from "./finetune";

export interface FinetuneWorkerRequest {
  baseWeights: Record<string, Float32Array>;
  tokenIds: number[];
  steps: number;
  lr: number;
  seed?: number;
}

export type FinetuneWorkerResponse =
  | { type: "progress"; fraction: number; message: string }
  | { type: "done"; weights: Record<string, Float32Array>; lossBefore: number; lossAfter: number }
  | { type: "error"; errorType: string; message: string };

// Minimal worker-scope typing (the DOM lib doesn't ship DedicatedWorkerGlobalScope).
const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<FinetuneWorkerRequest>) => void) | null;
  postMessage: (message: FinetuneWorkerResponse) => void;
};

workerScope.onmessage = (event: MessageEvent<FinetuneWorkerRequest>) => {
  const { baseWeights, tokenIds, steps, lr, seed } = event.data;
  try {
    const result = runFinetune({
      baseWeights: baseWeights as WeightSet,
      tokenIds,
      steps,
      lr,
      seed,
      onProgress: (fraction, message) => workerScope.postMessage({ type: "progress", fraction, message }),
    });
    workerScope.postMessage({
      type: "done",
      weights: result.weights,
      lossBefore: result.lossBefore,
      lossAfter: result.lossAfter,
    });
  } catch (err) {
    workerScope.postMessage({
      type: "error",
      errorType: err instanceof GeoEngineError ? err.type : "ComputeError",
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
