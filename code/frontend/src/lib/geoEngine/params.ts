/**
 * The build-wide number rule (`lib/params.ts`), bound to the engine's `GeoEngineError`.
 *
 * Nothing is decided here — see `lib/params.ts` for the rule and why it is one rule.
 */
import { makeNumberParams, show } from "../params";
import { invalidParam } from "./errors";

const params = makeNumberParams(invalidParam);

export const asInt = params.asInt;
export const asFloat = params.asFloat;
export const asBool = params.asBool;
export { show };
