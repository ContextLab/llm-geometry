/**
 * What the weights on screen ACTUALLY are.
 *
 * The tab used to ask one question — `trained !== null` — and every panel wrote its prose
 * from the answer. That is wrong the moment the Weight Lab is used, because an edit sits
 * IN FRONT of the base model rather than replacing it: with nothing trained, one click on
 * a preset leaves `trained === null` while the active weights are no longer the random
 * initialization either. Five sentences then described a model that was not on screen.
 *
 * So the tab reports provenance, not a boolean. Two independent facts — where the base
 * weights came from, and whether an edit is active — give the states below, and each
 * panel's sentence is written for all of them rather than for two.
 *
 * ## The third origin: `unrecorded`
 *
 * A model can also arrive from a `.llmlex.json` file, and a file may not say what its
 * weights are. `GET /api/lex/model` writes a `metrics` block of losses and step counts,
 * and versions of it predating this module wrote no provenance at all. There is no honest
 * way to turn that into "trained" or "untrained": the file's weights verify, and what they
 * ARE is simply not recorded. That is a third origin, not a default — red-team finding F1
 * was precisely the tab picking the flattering one, so that a file whose own `metrics`
 * read `"provenance":"untrained","trained":false` cleared every untrained warning on the
 * page and SamplePanel offered to "generate from the model you trained".
 *
 * ## What a `metrics` block can and cannot establish
 *
 * `metrics` is outside all three digests (see `lexEngine/bundle.ts`), so what it says is a
 * CLAIM BY THE FILE, not a verified fact — `provenanceFromMetrics` reads it, and every
 * surface that repeats it must attribute it (ModelFile's load line does). What the digests
 * do prove is that the weights and the word list are the ones the file was written with,
 * which is a different and narrower statement.
 */

/** Where the base weights came from, independent of any edit sitting in front of them. */
export type BaseOrigin = "untrained" | "trained" | "unrecorded";

/** `BaseOrigin` × "is an edit active" — the state every panel's prose is written for. */
export type Provenance =
  | "untrained"
  | "trained"
  | "unrecorded"
  | "edited-untrained"
  | "edited-trained"
  | "edited-unrecorded";

/** The single place the two facts become one state. */
export function provenanceOf(origin: BaseOrigin, hasEdit: boolean): Provenance {
  if (!hasEdit) return origin;
  return origin === "trained"
    ? "edited-trained"
    : origin === "untrained"
      ? "edited-untrained"
      : "edited-unrecorded";
}

/** The origin an active state departs from — the inverse of `provenanceOf`'s first argument. */
export function originOf(p: Provenance): BaseOrigin {
  switch (p) {
    case "trained":
    case "edited-trained":
      return "trained";
    case "untrained":
    case "edited-untrained":
      return "untrained";
    case "unrecorded":
    case "edited-unrecorded":
      return "unrecorded";
  }
}

/** True when hand-edited weights are what everything below is running. */
export function isEdited(p: Provenance): boolean {
  return p === "edited-untrained" || p === "edited-trained" || p === "edited-unrecorded";
}

/**
 * True when a training run produced the weights an edit departs from. NOT "these weights
 * were trained" — under `edited-trained` they are a trained model plus somebody's edits —
 * and NOT the negation of "untrained": under `unrecorded` nobody knows, and `false` here
 * means "this page cannot say it was trained", never "it was not".
 */
export function hasTrainedBase(p: Provenance): boolean {
  return p === "trained" || p === "edited-trained";
}

/**
 * The `trained` flag a saved file should carry: `null` where the answer is unknown.
 *
 * A `false` there would be a claim, and the whole point of `unrecorded` is that nobody on
 * this page is in a position to make it.
 */
export function trainedFlagOf(p: Provenance): boolean | null {
  const origin = originOf(p);
  return origin === "unrecorded" ? null : origin === "trained";
}

/**
 * How the Weight Lab names the model an edit departs from and returns to. Takes the BASE
 * model's own provenance, not the active state: after loading a file that was already
 * hand-edited, restore returns to the file's weights, and calling those "trained" would
 * name a model this tab never had.
 */
export function baseLabelOf(p: Provenance): string {
  switch (p) {
    case "trained":
      return "trained";
    case "untrained":
      return "random init";
    case "unrecorded":
      return "the loaded file";
    case "edited-trained":
      return "the loaded file (trained, then hand-edited)";
    case "edited-untrained":
      return "the loaded file (random init, then hand-edited)";
    case "edited-unrecorded":
      return "the loaded file (hand-edited)";
  }
}

/**
 * A noun phrase for the active weights, true in every state and usable mid-sentence.
 * Panels that need more than a phrase branch on the state itself.
 */
export function activeWeightsNoun(p: Provenance): string {
  switch (p) {
    case "trained":
      return "the model you trained";
    case "untrained":
      return "the random-init model";
    case "unrecorded":
      return "the model loaded from a file, which does not record whether it was trained";
    case "edited-trained":
      return "the model you trained, with your weight edits applied";
    case "edited-untrained":
      return "the random initialization, with your weight edits applied";
    case "edited-unrecorded":
      return "the model loaded from a file, with your weight edits applied";
  }
}

/** Every value `metrics.provenance` may carry, for validating a file's claim. */
const PROVENANCES: Record<Provenance, true> = {
  untrained: true,
  trained: true,
  unrecorded: true,
  "edited-untrained": true,
  "edited-trained": true,
  "edited-unrecorded": true,
};

export interface FileProvenance {
  /** What the file says its weights are — `"unrecorded"` when it says nothing usable. */
  provenance: Provenance;
  /** Did the file actually record this, or is it the honest "nobody wrote it down"? */
  declared: boolean;
}

/**
 * What a loaded bundle's `metrics` block says its weights are.
 *
 * Read in three steps, most specific first, because files come from three writers:
 *
 *  1. `metrics.provenance` — what this tab writes, and the only field that distinguishes
 *     all six states.
 *  2. `metrics.trained` / `metrics.edited` — the two booleans written beside it, so a file
 *     that carried only them (or that a future writer emits) is still understood.
 *  3. anything else, including a backend bundle full of real losses and no provenance
 *     field: `unrecorded`, with `declared: false`.
 *
 * Step 3 is the one that matters. Defaulting it to `trained` is red-team finding F1;
 * defaulting it to `untrained` would be the same defect wearing the opposite sign, since
 * a backend-trained model really was trained. Neither is knowable from the file, so the
 * tab says so instead of choosing.
 */
export function provenanceFromMetrics(metrics: Record<string, unknown>): FileProvenance {
  const declared = metrics.provenance;
  if (typeof declared === "string" && declared in PROVENANCES) {
    return { provenance: declared as Provenance, declared: true };
  }
  if (typeof metrics.trained === "boolean") {
    return {
      provenance: provenanceOf(metrics.trained ? "trained" : "untrained", metrics.edited === true),
      declared: true,
    };
  }
  return { provenance: "unrecorded", declared: false };
}
