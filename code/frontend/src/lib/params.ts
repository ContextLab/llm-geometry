/**
 * What a number is, decided ONCE for this whole build.
 *
 * The mirror of the backend's `llm_geometry/api/params.py`, and it exists for the same
 * reason: the rule was a private copy inside `staticClient/lex.ts`, `staticClient/geo.ts`
 * had none at all, and `geoEngine.finetune` had a third answer. So the public site
 * truncated `steps: 7.5` to 7 and accepted `lr: Infinity` (`Infinity > 0` is `true`) on
 * the Geometry Lab while `POST /api/geo/finetune` answered the same body with a typed
 * 400 — one request, two different runs, depending on which build served it. A run that
 * is not the run you asked for, reported as though it were, is the defect class this
 * whole campaign is about.
 *
 * The rule, matching `api/params.py` value for value:
 *
 * - only `number` is accepted; strings are refused rather than parsed, because
 *   `Number("٧")` is `NaN` while Python's `float("٧")` is `7.0`, so parsing would let one
 *   body mean two different numbers;
 * - `NaN` and `±Infinity` are refused — `Infinity` passes `> 0` and `NaN` fails every
 *   comparison, so both survive range guards and diverge later with nothing thrown;
 * - a non-integer is refused, never truncated: a number that is not the number you asked
 *   for is worse than a refusal;
 * - `undefined` (an absent field) takes the caller's default, which is what a default is.
 *
 * The two error taxonomies in this app (`geoEngine/errors`' `GeoEngineError` and
 * `staticClient/errors`' `ApiError`) are the only thing that differs between callers, so
 * they are the only thing parameterized — see `staticClient/params.ts` and
 * `geoEngine/params.ts`, each of which is a two-line binding of this file.
 */

/**
 * How a refused value is quoted back. `JSON.stringify` is not usable on its own here:
 * it renders `NaN`, `Infinity` and `-Infinity` all as the string `null`, so the three
 * numbers hardest to notice would be reported as the one value nobody sent.
 */
export function show(value: unknown): string {
  if (typeof value === "number") return String(value);
  if (typeof value === "bigint") return `${value}n`;
  if (value === undefined) return "undefined";
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

export interface NumberParams {
  /** A JSON integer, or a typed refusal — never a coercion. */
  asInt(value: unknown, name: string, fallback: number): number;
  /** A finite JSON number, or a typed refusal. `asInt`'s rule, one type down. */
  asFloat(value: unknown, name: string, fallback: number): number;
  /** A JSON boolean, or the four strings a form field can spell one with. */
  asBool(value: unknown, name: string, fallback: boolean): boolean;
}

/** Bind the rule to one error taxonomy. `fail` must produce a THROWABLE typed error. */
export function makeNumberParams(fail: (message: string) => Error): NumberParams {
  return {
    asInt(value: unknown, name: string, fallback: number): number {
      if (value === undefined) return fallback;
      if (typeof value !== "number") {
        throw fail(`${name} must be an integer, got ${show(value)}`);
      }
      if (!Number.isFinite(value)) {
        throw fail(`${name} must be a finite integer, got ${show(value)}`);
      }
      if (!Number.isInteger(value)) {
        throw fail(
          `${name} must be an integer, got ${show(value)} — it is not rounded or truncated, ` +
            "because a number that is not the number you asked for is worse than a refusal",
        );
      }
      return value;
    },
    asFloat(value: unknown, name: string, fallback: number): number {
      if (value === undefined) return fallback;
      if (typeof value !== "number") {
        throw fail(`${name} must be a number, got ${show(value)}`);
      }
      if (!Number.isFinite(value)) {
        throw fail(`${name} must be a finite number, got ${show(value)}`);
      }
      return value;
    },
    asBool(value: unknown, name: string, fallback: boolean): boolean {
      if (value === undefined) return fallback;
      if (typeof value === "boolean") return value;
      if (typeof value === "string" && ["true", "false", "1", "0"].includes(value.toLowerCase())) {
        return value.toLowerCase() === "true" || value === "1";
      }
      throw fail(`${name} must be a boolean, got ${show(value)}`);
    },
  };
}
