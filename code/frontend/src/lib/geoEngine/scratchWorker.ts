/**
 * Web Worker wrapper around runScratchTrain, so training a brand-new model never
 * freezes the UI thread (a from-scratch run is minutes of arithmetic, not the
 * fine-tune's fraction of a second). Spawn with:
 *
 *   new Worker(new URL("./scratchWorker.ts", import.meta.url), { type: "module" })
 *
 * Protocol: postMessage(ScratchWorkerRequest) -> a stream of {type:"progress"}
 * messages followed by one {type:"done"} or {type:"error"}. Weight tensors travel as
 * plain Float32Arrays (structured-cloneable).
 */

import { GeoEngineError } from "./errors";
import { buildVocabWords, runScratchTrain } from "./scratch";
import { GeoTokenizer } from "./tokenizer";

export interface ScratchWorkerRequest {
  /** The raw corpus. The worker builds the vocabulary AND the token stream from it. */
  text: string;
  epochs?: number;
  seed?: number;
}

export type ScratchWorkerResponse =
  | { type: "progress"; fraction: number; message: string }
  | {
      type: "done";
      weights: Record<string, Float32Array>;
      vocabWords: string[];
      finalLoss: number;
      epochs: number;
      nTokens: number;
      nDistinct: number;
    }
  | { type: "error"; errorType: string; message: string };

const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<ScratchWorkerRequest>) => void) | null;
  postMessage: (message: ScratchWorkerResponse) => void;
};

workerScope.onmessage = (event: MessageEvent<ScratchWorkerRequest>) => {
  const { text, epochs, seed } = event.data;
  try {
    workerScope.postMessage({
      type: "progress",
      fraction: 0.02,
      message: "building a vocabulary from your text",
    });
    // buildVocabWords throws the plain-language "not enough distinct types" error.
    const words = buildVocabWords(text);
    const tokenizer = new GeoTokenizer(words);
    const tokenIds = tokenizer.encodeStream(text);

    const result = runScratchTrain({
      tokenIds,
      epochs,
      seed,
      onProgress: (fraction, message) =>
        workerScope.postMessage({
          type: "progress",
          fraction: 0.05 + 0.93 * fraction,
          message,
        }),
    });

    workerScope.postMessage({
      type: "done",
      weights: result.weights as Record<string, Float32Array>,
      vocabWords: words,
      finalLoss: result.finalLoss,
      epochs: result.epochs,
      nTokens: tokenIds.length,
      nDistinct: new Set(words).size,
    });
  } catch (e) {
    const err = e as GeoEngineError;
    workerScope.postMessage({
      type: "error",
      errorType: err?.type ?? "ComputeError",
      message: err?.message ?? String(e),
    });
  }
};
