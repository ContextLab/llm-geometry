/**
 * Typed errors for the static (GitHub Pages) data client.
 *
 * `StaticModeError` is the honest "this build can't do that" signal (FR-203):
 * anything the static site cannot compute for real — arbitrary prompts on the
 * 001 views, HF-dataset fine-tunes, non-curated models — is refused with a
 * plain-language message instead of fabricated output. Views can branch on
 * `error.type === "StaticModeError"` to show their "run the full stack"
 * affordance.
 *
 * NOTE (module-evaluation order): this module is part of an import cycle
 * (dataClient → clientProvider → staticClient → dataClient), so `ApiError` must
 * only ever be touched *inside functions* — never at module top level.
 */

import { ApiError } from "../dataClient";

export const STATIC_MODE_ERROR = "StaticModeError";

export function staticModeError(message: string): ApiError {
  return new ApiError(STATIC_MODE_ERROR, message);
}

export function notFoundError(message: string): ApiError {
  return new ApiError("NotFoundError", message);
}

export function invalidParamError(message: string): ApiError {
  return new ApiError("InvalidParamError", message);
}

export function computeError(message: string): ApiError {
  return new ApiError("ComputeError", message);
}

export function networkError(message: string): ApiError {
  return new ApiError("NetworkError", message);
}

/** Map any thrown value onto the contract's typed error envelope. */
export function toApiError(e: unknown): ApiError {
  if (e instanceof ApiError) return e;
  // GeoEngineError (and the finetune worker's error messages) carry the same
  // contract `type` strings; duck-type instead of importing the class to keep
  // this module free of geoEngine coupling.
  if (e instanceof Error && typeof (e as { type?: unknown }).type === "string") {
    return new ApiError((e as unknown as { type: string }).type, e.message);
  }
  return new ApiError("ComputeError", e instanceof Error ? e.message : String(e));
}
