/**
 * The traced forward pass (US-7).
 *
 * The point of these tests is that the trace cannot lie. A visualization of a forward
 * pass is worthless — worse than worthless — if it is a plausible-looking picture of
 * something the model did not compute, so:
 *
 *   * the traced logits are held to the logits of an INDEPENDENTLY-run `forward` at 1e-9
 *     (they are in fact bit-identical, since the trace runs the same code);
 *   * the last stage's readout is held to a softmax of those logits;
 *   * every attention row sums to 1 and is exactly 0 above the diagonal — the causal mask
 *     is asserted, not assumed.
 *
 * Real weights throughout: `LexModel.fresh` runs the model's real initializer, and the
 * trained case really trains. No mocks, no stubs, no hand-written matrices.
 */
import { describe, expect, it } from "vitest";
import { flushSync, mount, unmount } from "svelte";

import ForwardPassPanel from "../../src/viz/lex/ForwardPassPanel.svelte";
import {
  baseLabelOf,
  hasTrainedBase,
  isEdited,
  originOf,
  provenanceFromMetrics,
  provenanceOf,
  trainedFlagOf,
  type Provenance,
} from "../../src/viz/lex/provenance";
import { LexModel, defaultConfig } from "../../src/lib/lexEngine/model";
import { runTraining } from "../../src/lib/lexEngine/train";
import { LexVocab, BOS_ID, UNK_ID } from "../../src/lib/lexEngine/vocab";
import { residualNorms, topKFromLogits, traceForward } from "../../src/lib/lexEngine/trace";

/** A small real vocabulary: the words below plus the four specials. */
const WORDS = [
  "the",
  "little",
  "cat",
  "sat",
  "on",
  "a",
  "mat",
  "and",
  "ran",
  "away",
  "down",
  "up",
];
const vocab = new LexVocab(WORDS, "dolch", "test");

function freshModel(over: Partial<ReturnType<typeof defaultConfig>> = {}): LexModel {
  const cfg = defaultConfig(vocab.rows, { dModel: 16, nLayers: 2, nHeads: 2, ctx: 32, ...over });
  return LexModel.fresh(cfg, 7);
}

describe("traceForward — the trace is the model", () => {
  it("final logits equal an independently-run forward pass to 1e-9", () => {
    const model = freshModel();
    const prompt = "the little cat sat on a mat";
    const trace = traceForward(model, vocab, { prompt });

    // The same input, assembled independently of the trace: <bos> then the prompt ids.
    const ids = Int32Array.from([BOS_ID, ...vocab.encodeText(prompt)]);
    expect(trace.T).toBe(ids.length);
    expect(Array.from(trace.tokens.map((t) => t.id))).toEqual(Array.from(ids));

    const acts = model.forward(ids, 1, ids.length, {});
    expect(trace.logits.length).toBe(acts.logits.length);
    let worst = 0;
    for (let i = 0; i < acts.logits.length; i++) {
      worst = Math.max(worst, Math.abs(trace.logits[i] - acts.logits[i]));
    }
    expect(worst).toBeLessThan(1e-9);
  });

  it("holds at 1e-9 for a model that really trained, untied, with 4 heads", () => {
    const cfg = defaultConfig(vocab.rows, {
      dModel: 16,
      nLayers: 3,
      nHeads: 4,
      ctx: 16,
      tied: false,
    });
    const corpus = "the little cat sat on a mat and ran away down the little mat ".repeat(30);
    const trained = runTraining({
      cfg,
      tokens: vocab.encodeText(corpus),
      steps: 6,
      batchSize: 4,
      seed: 3,
    });
    const model = new LexModel(cfg, trained.weights);

    const prompt = "the little cat";
    const trace = traceForward(model, vocab, { prompt });
    const ids = Int32Array.from([BOS_ID, ...vocab.encodeText(prompt)]);
    const acts = model.forward(ids, 1, ids.length, {});
    for (let i = 0; i < acts.logits.length; i++) {
      expect(Math.abs(trace.logits[i] - acts.logits[i])).toBeLessThan(1e-9);
    }
    // The readout stage IS the model's distribution, so it must equal a softmax of the
    // real last-position logits — not merely resemble one.
    const V = cfg.vocabRows;
    const expected = topKFromLogits(
      acts.logits.slice((ids.length - 1) * V, ids.length * V),
      vocab,
      8,
      true,
    );
    const last = trace.stages[trace.stages.length - 1];
    expect(last.kind).toBe("readout");
    expect(last.lens.ids).toEqual(expected.ids);
    for (let i = 0; i < expected.probs.length; i++) {
      expect(Math.abs(last.lens.probs[i] - expected.probs[i])).toBeLessThan(1e-12);
    }
  });

  it("the last layer's output reads out exactly; every earlier stage is marked approximate", () => {
    const model = freshModel({ nLayers: 3 });
    const trace = traceForward(model, vocab, { prompt: "the cat sat" });

    // embed + (attention, mlp) per layer + readout
    expect(trace.stages.length).toBe(2 * 3 + 2);
    expect(trace.stages.map((s) => s.kind)).toEqual([
      "embed",
      "attention",
      "mlp",
      "attention",
      "mlp",
      "attention",
      "mlp",
      "readout",
    ]);

    const exact = trace.stages.filter((s) => s.lens.exact);
    // Exactly two: the final layer's output (which IS the final LayerNorm's input) and
    // the readout itself. Everything earlier is a logit lens and must say so.
    expect(exact.map((s) => s.index)).toEqual([6, 7]);
    expect(exact[0].lens.ids).toEqual(exact[1].lens.ids);
    for (let i = 0; i < exact[0].lens.probs.length; i++) {
      expect(Math.abs(exact[0].lens.probs[i] - exact[1].lens.probs[i])).toBeLessThan(1e-12);
    }

    for (const s of trace.stages.slice(0, 6)) {
      expect(s.lens.exact).toBe(false);
      // A readout is still a distribution: it must be usable, just not authoritative.
      for (const p of s.lens.probs) expect(p).toBeGreaterThanOrEqual(0);
    }
  });
});

describe("traceForward — attention is causal and normalized", () => {
  it("every row sums to 1 and nothing attends to the future", () => {
    const model = freshModel({ nLayers: 2, nHeads: 2 });
    const trace = traceForward(model, vocab, { prompt: "the little cat sat on a mat and ran" });
    const T = trace.T;
    expect(T).toBeGreaterThan(3);

    const attnStages = trace.stages.filter((s) => s.kind === "attention");
    expect(attnStages.length).toBe(trace.nLayers);

    for (const s of attnStages) {
      expect(s.attention).not.toBeNull();
      expect(s.attention?.length).toBe(trace.nHeads);
      for (const head of s.attention ?? []) {
        expect(head.length).toBe(T);
        for (let i = 0; i < T; i++) {
          expect(head[i].length).toBe(T);
          let sum = 0;
          for (let j = 0; j < T; j++) {
            const a = head[i][j];
            expect(Number.isFinite(a)).toBe(true);
            expect(a).toBeGreaterThanOrEqual(0);
            // The causal mask, asserted: strictly-future positions are EXACTLY zero.
            if (j > i) expect(a).toBe(0);
            sum += a;
          }
          expect(Math.abs(sum - 1)).toBeLessThan(1e-12);
        }
      }
    }
  });

  it("non-attention stages carry no attention, so nothing can be drawn where none exists", () => {
    const trace = traceForward(freshModel(), vocab, { prompt: "the cat" });
    for (const s of trace.stages) {
      if (s.kind === "attention") expect(s.attention).not.toBeNull();
      else expect(s.attention).toBeNull();
    }
  });
});

describe("traceForward — what the panel reads", () => {
  it("reports every residual-stream norm, positive and finite, at every stage", () => {
    const trace = traceForward(freshModel(), vocab, { prompt: "the little cat sat" });
    for (const s of trace.stages) {
      expect(s.residualNorm.length).toBe(trace.T);
      for (const v of s.residualNorm) {
        expect(Number.isFinite(v)).toBe(true);
        expect(v).toBeGreaterThan(0);
      }
    }
  });

  it("makes out-of-budget prompt words visible rather than silently dropping them", () => {
    const trace = traceForward(freshModel(), vocab, { prompt: "the sarsaparilla cat quixotic" });
    expect(trace.unkCount).toBe(2);
    expect(trace.unkWords).toEqual(["sarsaparilla", "quixotic"]);

    const oov = trace.tokens.filter((t) => !t.inBudget);
    expect(oov.map((t) => t.word)).toEqual(["sarsaparilla", "quixotic"]);
    for (const t of oov) expect(t.id).toBe(UNK_ID);
    // The in-budget words are untouched, and <bos> leads the sequence the model sees.
    expect(trace.tokens[0].word).toBe("<bos>");
    expect(trace.tokens[0].special).toBe(true);
    expect(trace.tokens.filter((t) => t.inBudget).map((t) => t.word)).toEqual([
      "<bos>",
      "the",
      "cat",
    ]);
  });

  it("counts a repeated out-of-budget word once in unkWords and every time in unkCount", () => {
    const trace = traceForward(freshModel(), vocab, { prompt: "zzz cat zzz zzz" });
    expect(trace.unkCount).toBe(3);
    expect(trace.unkWords).toEqual(["zzz"]);
  });

  it("keeps the LAST ctx tokens when the prompt overflows, and says how many it dropped", () => {
    const model = freshModel({ ctx: 8 });
    const long = "the cat sat on a mat and ran away down up the little cat";
    const trace = traceForward(model, vocab, { prompt: long });
    expect(trace.T).toBe(8);
    expect(trace.truncated).toBe(true);
    // The kept window is the TAIL of <bos> + prompt, and the drop count says how much.
    const all = [BOS_ID, ...vocab.encodeText(long)];
    expect(trace.droppedTokens).toBe(all.length - 8);
    expect(trace.tokens.map((t) => t.id)).toEqual(all.slice(all.length - 8));
    // And it still traces a real pass at that length.
    const acts = model.forward(Int32Array.from(all.slice(all.length - 8)), 1, 8, {});
    for (let i = 0; i < acts.logits.length; i++) {
      expect(Math.abs(trace.logits[i] - acts.logits[i])).toBeLessThan(1e-9);
    }
  });

  it("traces a bare <bos> when the prompt is empty", () => {
    const trace = traceForward(freshModel(), vocab, { prompt: "" });
    expect(trace.T).toBe(1);
    expect(trace.tokens[0].id).toBe(BOS_ID);
    expect(trace.truncated).toBe(false);
    for (const s of trace.stages.filter((x) => x.kind === "attention")) {
      for (const head of s.attention ?? []) expect(head).toEqual([[1]]);
    }
  });

  it("refuses a vocabulary the model was not built for", () => {
    const model = freshModel();
    const other = new LexVocab(WORDS.slice(0, 5), "dolch", "test");
    expect(() => traceForward(model, other, {})).toThrow(/rows/);
  });

  it("topK is honoured, ordered by probability, and sums to at most 1", () => {
    const trace = traceForward(freshModel(), vocab, { prompt: "the cat", topK: 5 });
    for (const s of trace.stages) {
      expect(s.lens.ids.length).toBe(5);
      expect(s.lens.words.length).toBe(5);
      for (let i = 1; i < 5; i++) expect(s.lens.probs[i]).toBeLessThanOrEqual(s.lens.probs[i - 1]);
      expect(s.lens.probs.reduce((a, b) => a + b, 0)).toBeLessThanOrEqual(1 + 1e-12);
      expect(s.lens.words).toEqual(vocab.decode(s.lens.ids));
    }
  });
});

/**
 * The panel itself, really mounted (Svelte 5 `mount`, jsdom). Canvas 2D does not exist in
 * jsdom, which `MatrixHeatmap` already guards, so these cover structure and interaction —
 * that the controls exist, are labelled, and move the pass — not pixels.
 */
describe("ForwardPassPanel", () => {
  function render(props: {
    model: LexModel | null;
    vocab: LexVocab | null;
    provenance: Provenance;
  }) {
    const target = document.createElement("div");
    document.body.appendChild(target);
    const app = mount(ForwardPassPanel, { target, props });
    flushSync();
    return {
      target,
      done: () => {
        unmount(app);
        target.remove();
      },
      q: <T extends Element>(sel: string) => target.querySelector<T>(sel),
      all: (sel: string) => Array.from(target.querySelectorAll(sel)),
    };
  }

  it("says there is nothing to trace rather than drawing a pattern, when no model exists", () => {
    const r = render({ model: null, vocab: null, provenance: "untrained" });
    expect(r.q('[data-testid="lex-forward-empty"]')?.textContent).toMatch(/no model to trace/);
    expect(r.q('[data-testid="lex-forward-playhead"]')).toBeNull();
    r.done();
  });

  it("labels an untrained model's trace as an untrained model's", () => {
    const r = render({ model: freshModel(), vocab, provenance: "untrained" });
    expect(r.q('[data-testid="lex-forward-untrained"]')?.textContent).toMatch(/random-init/);
    r.done();
  });

  it("steps the pass, and marks the intermediate readouts approximate", () => {
    const model = freshModel({ nLayers: 2 });
    const r = render({ model, vocab, provenance: "trained" });

    const head = () => r.q('[data-testid="lex-forward-playhead"]')!.textContent ?? "";
    expect(head()).toMatch(/stage 1\/6/);
    expect(head()).toMatch(/embed \+ position/);
    expect(r.q('[data-testid="lex-forward-lens-label"]')?.textContent).toMatch(/approximate/);

    const next = r.q<HTMLButtonElement>('[data-testid="lex-forward-next"]')!;
    for (let i = 0; i < 5; i++) {
      next.click();
      flushSync();
    }
    expect(head()).toMatch(/stage 6\/6/);
    // The last stage is the model's real readout, and must not be hedged.
    const label = r.q('[data-testid="lex-forward-lens-label"]')?.textContent ?? "";
    expect(label).toMatch(/exact/);
    expect(label).not.toMatch(/approximate/);
    expect(next.disabled).toBe(true);

    r.q<HTMLButtonElement>('[data-testid="lex-forward-prev"]')!.click();
    flushSync();
    expect(head()).toMatch(/stage 5\/6/);
    r.done();
  });

  it("uses radiogroups (never tablists) for its segmented pickers", () => {
    const r = render({ model: freshModel({ nLayers: 2 }), vocab, provenance: "trained" });
    expect(r.all('[role="tablist"]').length).toBe(0);

    for (const testid of ["lex-forward-speed", "lex-forward-layer"]) {
      const group = r.q(`[data-testid="${testid}"]`)!;
      expect(group.getAttribute("role")).toBe("radiogroup");
      const radios = Array.from(group.querySelectorAll('[role="radio"]'));
      expect(radios.length).toBeGreaterThan(1);
      // Exactly one checked, and it is the only one in the tab order.
      expect(radios.filter((b) => b.getAttribute("aria-checked") === "true").length).toBe(1);
      expect(radios.filter((b) => b.getAttribute("tabindex") === "0").length).toBe(1);
    }
    r.done();
  });

  it("keyboard-selects a speed with the arrow keys", () => {
    const r = render({ model: freshModel(), vocab, provenance: "trained" });
    const group = r.q('[data-testid="lex-forward-speed"]')!;
    const checked = () =>
      group.querySelector('[role="radio"][aria-checked="true"]')?.textContent?.trim();
    expect(checked()).toBe("1×");
    group.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    flushSync();
    expect(checked()).toBe("2×");
    group.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }));
    flushSync();
    expect(checked()).toBe("1×");
    r.done();
  });

  it("choosing a layer turns OFF follow-the-playhead (ArchTracePanel's lesson)", () => {
    const r = render({ model: freshModel({ nLayers: 3 }), vocab, provenance: "trained" });
    const follow = r.q<HTMLInputElement>('[data-testid="lex-forward-follow"]')!;
    expect(follow.checked).toBe(true);

    const layer2 = r
      .all('[data-testid="lex-forward-layer"] [role="radio"]')
      .find((b) => b.getAttribute("data-value") === "2") as HTMLButtonElement;
    layer2.click();
    flushSync();
    expect(follow.checked).toBe(false);
    expect(layer2.getAttribute("aria-checked")).toBe("true");

    // …and stepping the pass no longer drags the selection away from layer 2.
    const next = r.q<HTMLButtonElement>('[data-testid="lex-forward-next"]')!;
    next.click();
    flushSync();
    next.click();
    flushSync();
    expect(layer2.getAttribute("aria-checked")).toBe("true");
    r.done();
  });

  it("shows out-of-budget prompt words instead of swallowing them", () => {
    const r = render({ model: freshModel(), vocab, provenance: "trained" });
    const input = r.q<HTMLInputElement>('[data-testid="lex-forward-prompt"]')!;
    expect(r.q('[data-testid="lex-forward-oov"]')).toBeNull();

    input.value = "the sarsaparilla cat";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    flushSync();

    const note = r.q('[data-testid="lex-forward-oov"]')?.textContent ?? "";
    expect(note).toMatch(/sarsaparilla/);
    expect(note).toMatch(/<unk>/);
    expect(r.all('[data-testid="lex-forward-strip"] .oov').length).toBe(1);
    r.done();
  });

  it("renders one attention tile per head and lets a head be picked", () => {
    const r = render({ model: freshModel({ nHeads: 4, dModel: 16 }), vocab, provenance: "trained" });
    const tiles = r.all('[data-testid="lex-forward-heads"] button');
    expect(tiles.length).toBe(4);
    expect(tiles.filter((b) => b.getAttribute("aria-pressed") === "true").length).toBe(1);
    (tiles[2] as HTMLButtonElement).click();
    flushSync();
    expect(tiles[2].getAttribute("aria-pressed")).toBe("true");
    r.done();
  });

  // H-1: one Weight Lab click on an untrained model used to leave five panels claiming
  // the random initialization was on screen. The panel must name the weights it is
  // actually tracing in all four provenance states, not in two.
  it("names hand-edited weights over an untrained model as exactly that", () => {
    const r = render({ model: freshModel(), vocab, provenance: "edited-untrained" });
    const note = r.q('[data-testid="lex-forward-untrained"]')?.textContent ?? "";
    expect(note).toMatch(/hand-edited/);
    expect(note).toMatch(/Nothing has been trained yet/);
    r.done();
  });

  it("does not call an edited trained model 'random-init', and does not stay silent", () => {
    const r = render({ model: freshModel(), vocab, provenance: "edited-trained" });
    expect(r.q('[data-testid="lex-forward-untrained"]')).toBeNull();
    const note = r.q('[data-testid="lex-forward-edited"]')?.textContent ?? "";
    expect(note).toMatch(/hand-edited/);
    expect(note).not.toMatch(/random-init/);
    r.done();
  });

  it("a trained, unedited model gets no provenance warning at all", () => {
    const r = render({ model: freshModel(), vocab, provenance: "trained" });
    expect(r.q('[data-testid="lex-forward-untrained"]')).toBeNull();
    expect(r.q('[data-testid="lex-forward-edited"]')).toBeNull();
    r.done();
  });

  it("plays and pauses without leaving the button ambiguous", () => {
    const r = render({ model: freshModel(), vocab, provenance: "trained" });
    const play = r.q<HTMLButtonElement>('[data-testid="lex-forward-play"]')!;
    expect(play.getAttribute("aria-label")).toMatch(/play/);
    play.click();
    flushSync();
    expect(play.getAttribute("aria-label")).toMatch(/pause/);
    play.click();
    flushSync();
    expect(play.getAttribute("aria-label")).toMatch(/play/);
    r.done();
  });
});

describe("residualNorms", () => {
  it("is the plain L2 norm of each row", () => {
    const h = Float64Array.from([3, 4, 0, 0, 0, 5]);
    expect(residualNorms(h, 2, 3)).toEqual([5, 5]);
  });
});

/**
 * The state machine every lex panel's prose now reads from. `trained !== null` was the old
 * question and it has a hole: an edit sits IN FRONT of the base model, so with nothing
 * trained the active weights are neither the trained model nor the random initialization.
 */
describe("provenance", () => {
  it("distinguishes every state, including the one that used to be missing", () => {
    expect(provenanceOf("untrained", false)).toBe("untrained");
    expect(provenanceOf("trained", false)).toBe("trained");
    expect(provenanceOf("untrained", true)).toBe("edited-untrained");
    expect(provenanceOf("trained", true)).toBe("edited-trained");
    // The third origin (feature 007 red-team F1): a file whose `metrics` block does not
    // say what its weights are. Not "untrained", which would be a claim of its own.
    expect(provenanceOf("unrecorded", false)).toBe("unrecorded");
    expect(provenanceOf("unrecorded", true)).toBe("edited-unrecorded");
  });

  it("round-trips through the origin it was built from", () => {
    for (const origin of ["untrained", "trained", "unrecorded"] as const) {
      for (const edit of [false, true]) {
        expect(originOf(provenanceOf(origin, edit))).toBe(origin);
      }
    }
  });

  it("separates 'came from training' from 'is what training produced'", () => {
    expect(hasTrainedBase("edited-trained")).toBe(true);
    expect(isEdited("edited-trained")).toBe(true);
    expect(hasTrainedBase("edited-untrained")).toBe(false);
    expect(isEdited("trained")).toBe(false);
    expect(isEdited("edited-unrecorded")).toBe(true);
    // `false` here means "this page cannot say it was trained", never "it was not" — which
    // is why the saved flag is `null` rather than `false` in that state.
    expect(hasTrainedBase("unrecorded")).toBe(false);
    expect(trainedFlagOf("unrecorded")).toBeNull();
    expect(trainedFlagOf("edited-unrecorded")).toBeNull();
    expect(trainedFlagOf("untrained")).toBe(false);
    expect(trainedFlagOf("edited-trained")).toBe(true);
  });

  it("names the model an edit returns to, which is never the edit itself", () => {
    const labels: Record<Provenance, string> = {
      untrained: baseLabelOf("untrained"),
      trained: baseLabelOf("trained"),
      unrecorded: baseLabelOf("unrecorded"),
      "edited-untrained": baseLabelOf("edited-untrained"),
      "edited-trained": baseLabelOf("edited-trained"),
      "edited-unrecorded": baseLabelOf("edited-unrecorded"),
    };
    expect(labels.untrained).toBe("random init");
    expect(labels.trained).toBe("trained");
    expect(labels.unrecorded).toBe("the loaded file");
    // A base that arrived already edited is the FILE's weights, not a model this tab
    // trained — the restore button returns to the file, and the label says so.
    expect(labels["edited-untrained"]).toContain("loaded file");
    expect(labels["edited-trained"]).toContain("loaded file");
    expect(labels["edited-unrecorded"]).toContain("loaded file");
  });

  /**
   * The `metrics` block is a CLAIM, not a fact — it is outside all three digests. What is
   * not negotiable is that an absent claim must not become a flattering one.
   */
  it("reads a file's own account of its weights, and refuses to invent one", () => {
    expect(provenanceFromMetrics({ provenance: "untrained", trained: false })).toEqual({
      provenance: "untrained",
      declared: true,
    });
    expect(provenanceFromMetrics({ provenance: "edited-trained" }).provenance).toBe(
      "edited-trained",
    );
    // Booleans alone, the shape a writer that knows nothing of `provenance` would emit.
    expect(provenanceFromMetrics({ trained: true, edited: true }).provenance).toBe(
      "edited-trained",
    );
    expect(provenanceFromMetrics({ trained: false }).provenance).toBe("untrained");
    // A backend bundle: real losses, no provenance field. Unknown, and said to be unknown.
    expect(provenanceFromMetrics({ final_loss: 2.26, steps: 400 })).toEqual({
      provenance: "unrecorded",
      declared: false,
    });
    expect(provenanceFromMetrics({}).provenance).toBe("unrecorded");
    // Garbage is not a state either, and must not fall through to "trained".
    expect(provenanceFromMetrics({ provenance: "excellent" }).provenance).toBe("unrecorded");
    expect(provenanceFromMetrics({ trained: "yes" }).provenance).toBe("unrecorded");
  });
});
