/**
 * Error types for the TS geoEngine, mirroring the backend's error taxonomy
 * (llm_geometry.errors) so a future staticClient can map them onto the frozen
 * contract's error envelope {type, message} with the same `type` strings.
 */

export type GeoErrorType =
  | "InvalidParamError" // 400 in the backend
  | "InvalidWeightEditError" // 422
  | "NotFoundError" // 404
  | "ComputeError"; // 500

export class GeoEngineError extends Error {
  readonly type: GeoErrorType;

  constructor(type: GeoErrorType, message: string) {
    super(message);
    this.name = type;
    this.type = type;
  }
}

export const invalidParam = (message: string): GeoEngineError =>
  new GeoEngineError("InvalidParamError", message);

export const invalidWeightEdit = (message: string): GeoEngineError =>
  new GeoEngineError("InvalidWeightEditError", message);

export const notFound = (message: string): GeoEngineError =>
  new GeoEngineError("NotFoundError", message);

export const computeError = (message: string): GeoEngineError =>
  new GeoEngineError("ComputeError", message);
