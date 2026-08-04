/**
 * The Weight Lab's `randomize` seed box: what it says it accepts, and what it does with
 * the rest.
 *
 * The control was `bind:value={seed}` on `<input type="number">`, which hands Svelte back
 * `null` for an empty box and a `number` for anything the element will parse. So:
 *
 *   - clearing the box re-drew the tensor at seed 0 and wrote a note reading
 *     "re-drawn from its initializer at seed null";
 *   - `2.5` and `1e3` were passed straight to `initWeights`, which expects an integer in
 *     the declared 0..9999;
 *   - `max="9999"` on the element blocked nothing, exactly as `max="9999"` on the Vacancy
 *     panel's seed box blocked nothing before red-team finding F4.
 *
 * `randomize` is reproducible ONLY by its seed, and this panel's note is the sole record
 * of which draw the active weights came from. A seed that is not the one on screen makes
 * that note a false statement about weights that everything downstream — the spectrum, the
 * token cloud, the sampler — is then measuring. Nothing throws; the numbers are simply of
 * a different model than the one described.
 *
 * These tests mount the REAL panel over a REAL `LexModel` and read the answer off the DOM
 * and off the `onEdited` callback — the two things the tab and the reader actually see.
 */
import { describe, expect, it } from "vitest";
import { flushSync, mount, unmount } from "svelte";

import LexWeightLab from "../../src/viz/lex/LexWeightLab.svelte";
import {
  LexModel,
  LexVocab,
  defaultConfig,
  initWeights,
  lexWeightsTokenOf,
  type WeightSet,
} from "../../src/lib/lexEngine";

const VOCAB_WORDS = ["a", "and", "big", "cat", "dog", "the"];

interface Mounted {
  edits: { token: string; note: string; weights: WeightSet }[];
  seedInput: HTMLInputElement;
  apply: HTMLButtonElement;
  error: () => HTMLElement | null;
  dispose: () => void;
}

function mountLab(): Mounted {
  const vocab = new LexVocab(VOCAB_WORDS, "frequency", "full");
  const cfg = defaultConfig(vocab.rows, { dModel: 8, nLayers: 1, nHeads: 2, ctx: 8, dropout: 0 });
  const base = new LexModel(cfg, initWeights(cfg, 1));

  const target = document.createElement("div");
  document.body.appendChild(target);
  const edits: { token: string; note: string; weights: WeightSet }[] = [];
  const app = mount(LexWeightLab, {
    target,
    props: {
      base,
      baseLabel: "trained",
      vocab,
      edited: null,
      onEdited: (model: LexModel, token: string, note: string) =>
        edits.push({ token, note, weights: model.weights }),
      onRestore: () => {},
    },
  });
  flushSync();

  // Choose a preset the seed applies to. `randomize` is the only one that reads it.
  const preset = target.querySelector<HTMLSelectElement>('[data-testid="lex-weight-preset"]');
  if (preset === null) throw new Error("the preset select did not render");
  preset.value = "randomize";
  preset.dispatchEvent(new Event("change", { bubbles: true }));
  flushSync();

  const seedInput = target.querySelector<HTMLInputElement>('[data-testid="lex-weight-seed"]');
  if (seedInput === null) throw new Error("the seed input did not render");
  const apply = target.querySelector<HTMLButtonElement>('[data-testid="lex-weight-apply"]');
  if (apply === null) throw new Error("the apply button did not render");

  return {
    edits,
    seedInput,
    apply,
    error: () => target.querySelector<HTMLElement>('[data-testid="lex-weight-error"]'),
    dispose: () => {
      unmount(app);
      target.remove();
    },
  };
}

/** Type a value the way a person does, then press Apply. */
function typeAndApply(m: Mounted, raw: string): void {
  m.seedInput.value = raw;
  m.seedInput.dispatchEvent(new Event("input", { bubbles: true }));
  flushSync();
  m.apply.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  flushSync();
}

describe("the Weight Lab seed box's declared range is its accepted range", () => {
  it.each([
    ["the empty box, which re-drew at seed 0 and said 'seed null'", ""],
    ["whitespace only", "   "],
    ["a fraction", "2.5"],
    ["exponent notation", "1e3"],
    ["hexadecimal", "0x10"],
    ["a leading plus", "+7"],
    ["a negative seed", "-1"],
    ["one above the declared maximum", "10000"],
    ["not a number at all", "abc"],
  ])("refuses %s rather than drawing with a different seed", (_what, raw) => {
    const m = mountLab();
    try {
      typeAndApply(m, raw);
      expect(m.edits).toEqual([]);
      expect(m.error()?.textContent ?? "").toContain("whole number from 0 to 9999");
    } finally {
      m.dispose();
    }
  });

  it.each(["0", "7", "9999"])("draws with the seed %s exactly as typed", (raw) => {
    const m = mountLab();
    try {
      typeAndApply(m, raw);
      expect(m.edits).toHaveLength(1);
      // The note is the panel's only record of which draw these weights are.
      expect(m.edits[0].note).toContain(`at seed ${raw}`);
    } finally {
      m.dispose();
    }
  });

  it("the weights it produces really are that seed's draw, not another's", () => {
    const m = mountLab();
    try {
      typeAndApply(m, "7");
      expect(m.edits).toHaveLength(1);
      const vocab = new LexVocab(VOCAB_WORDS, "frequency", "full");
      const cfg = defaultConfig(vocab.rows, {
        dModel: 8,
        nLayers: 1,
        nHeads: 2,
        ctx: 8,
        dropout: 0,
      });
      const expected = { ...initWeights(cfg, 1) };
      expected.embed = initWeights(cfg, 7).embed;
      expect(m.edits[0].token).toBe(lexWeightsTokenOf(cfg, expected));
      // And NOT the draw the old code would have produced from a cleared box.
      const atZero = { ...initWeights(cfg, 1) };
      atZero.embed = initWeights(cfg, 0).embed;
      expect(m.edits[0].token).not.toBe(lexWeightsTokenOf(cfg, atZero));
    } finally {
      m.dispose();
    }
  });

  it("clears the refusal once a usable seed is typed", () => {
    const m = mountLab();
    try {
      typeAndApply(m, "2.5");
      expect(m.edits).toEqual([]);
      expect(m.error()).not.toBeNull();
      typeAndApply(m, "3");
      expect(m.edits).toHaveLength(1);
      expect(m.error()).toBeNull();
    } finally {
      m.dispose();
    }
  });

  it("declares the same maximum it enforces", () => {
    const m = mountLab();
    try {
      expect(m.seedInput.getAttribute("max")).toBe("9999");
      expect(m.seedInput.getAttribute("min")).toBe("0");
    } finally {
      m.dispose();
    }
  });
});
