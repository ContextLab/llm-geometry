/**
 * What the weights on screen ACTUALLY are.
 *
 * The tab used to ask one question — `trained !== null` — and every panel wrote its prose
 * from the answer. That is wrong the moment the Weight Lab is used, because an edit sits
 * IN FRONT of the base model rather than replacing it: with nothing trained, one click on
 * a preset leaves `trained === null` while the active weights are no longer the random
 * initialization either. Five sentences then described a model that was not on screen.
 *
 * So the tab reports provenance, not a boolean. Two independent facts — has anything been
 * trained, and is an edit active — give four states, and each panel's sentence is written
 * for all four rather than for two.
 */
export type Provenance = "untrained" | "trained" | "edited-untrained" | "edited-trained";

/** The single place the two facts become one state. */
export function provenanceOf(hasTrained: boolean, hasEdit: boolean): Provenance {
  if (hasEdit) return hasTrained ? "edited-trained" : "edited-untrained";
  return hasTrained ? "trained" : "untrained";
}

/** True when hand-edited weights are what everything below is running. */
export function isEdited(p: Provenance): boolean {
  return p === "edited-untrained" || p === "edited-trained";
}

/**
 * True when a training run produced the weights an edit departs from. NOT "these weights
 * were trained" — under `edited-trained` they are a trained model plus somebody's edits.
 */
export function hasTrainedBase(p: Provenance): boolean {
  return p === "trained" || p === "edited-trained";
}

/** How the Weight Lab names the model an edit departs from and returns to. */
export function baseLabelOf(p: Provenance): string {
  return hasTrainedBase(p) ? "trained" : "random init";
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
    case "edited-trained":
      return "the model you trained, with your weight edits applied";
    case "edited-untrained":
      return "the random initialization, with your weight edits applied";
  }
}
