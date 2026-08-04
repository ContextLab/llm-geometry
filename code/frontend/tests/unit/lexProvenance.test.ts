/**
 * What the Lexicon Lab says about weights that arrived in a file.
 *
 * Red-team finding F1 (`notes/agent-reports/redteam-007-lex.md`): loading ANY
 * `.llmlex.json` made the whole tab claim the weights were trained. The observed file
 *
 *     SAVED bundle metrics: {"note":"untrained random initialization",
 *                            "provenance":"untrained","trained":false,"edited":false}
 *
 * loaded back with
 *
 *     "saveUntrainedNote": null
 *     "spectrumUntrained": null
 *     "sampleHeader": "Generate from the model you trained  PROMPT  TEMPERATURE 0.90 …"
 *     CONSOLE ERRORS: []
 *
 * — no error anywhere, and the success line ended in the word "verified".
 *
 * These tests run the real tab in jsdom over the real committed corpus, save a real bundle
 * through the real `exportLexBundle`, and read the answer off the DOM. Nothing here is a
 * stand-in for the thing being tested: the only shim is `URL.createObjectURL`, which jsdom
 * does not implement, and it is used to READ the bytes the save button really wrote.
 */
import { beforeAll, describe, expect, it } from "vitest";
import { flushSync, mount, unmount } from "svelte";
// jsdom's Blob/File carry no `.text()`, so a real bundle could neither be written nor read
// back inside this environment. Node's are the standard implementations of the same
// interface — the bytes that travel through them are the component's own.
import { Blob as NodeBlob, File as NodeFile } from "node:buffer";

import LexiconLab from "../../src/viz/lex/LexiconLab.svelte";
import ModelFile from "../../src/viz/lex/ModelFile.svelte";
import {
  LexModel,
  LexVocab,
  buildVocab,
  defaultConfig,
  exportLexBundle,
  type LexModelBundle,
} from "../../src/lib/lexEngine";
import { fsStaticFetch, readStaticJson } from "./staticTestUtils";

/** The real committed corpus, exactly as the tab fetches it. */
let corpusText = "";
let vocab: LexVocab;

beforeAll(async () => {
  const asset = await readStaticJson<{ text: string }>("lex/corpus.json");
  corpusText = asset.text;
  // The budget the tab can resolve from its own controls, so loading a file built on it
  // does not also change the shape — this test is about provenance, nothing else.
  vocab = buildVocab("dolch", "pre_primer", corpusText);
  // LexiconLab fetches its corpus from the same asset path the Pages build serves.
  (globalThis as { fetch?: unknown }).fetch = fsStaticFetch();
  (globalThis as { Blob?: unknown }).Blob = NodeBlob;
});

function freshModel(): LexModel {
  return LexModel.fresh(
    defaultConfig(vocab.rows, { dModel: 16, nLayers: 1, nHeads: 2, ctx: 32 }),
    11,
  );
}

/** A bundle written the way the tab writes one, with `metrics` supplied by the caller. */
function bundleWith(metrics: Record<string, unknown>): LexModelBundle {
  const model = freshModel();
  return exportLexBundle({
    config: model.cfg,
    weights: model.weights,
    vocabWords: vocab.words,
    budgetSource: vocab.source,
    budgetName: vocab.budgetName,
    metrics,
  });
}

// --- the save button really writes the provenance -------------------------------------

describe("a saved file records what the weights it holds actually are", () => {
  it("writes the untrained state into `metrics`, not a bare note", async () => {
    const written: InstanceType<typeof NodeBlob>[] = [];
    const url = globalThis.URL as unknown as {
      createObjectURL?: (b: InstanceType<typeof NodeBlob>) => string;
      revokeObjectURL?: (u: string) => void;
    };
    // jsdom implements neither; the Blob they are handed is the real saved file.
    url.createObjectURL = (b: InstanceType<typeof NodeBlob>) => {
      written.push(b);
      return "blob:test";
    };
    url.revokeObjectURL = () => {};

    const target = document.createElement("div");
    document.body.appendChild(target);
    const app = mount(ModelFile, {
      target,
      props: {
        model: freshModel(),
        vocab,
        provenance: "untrained" as const,
        note: "",
        onLoaded: () => {},
      },
    });
    flushSync();
    target.querySelector<HTMLButtonElement>('[data-testid="lex-save-model"]')!.click();
    flushSync();

    expect(written).toHaveLength(1);
    const saved = JSON.parse(await written[0].text()) as LexModelBundle;
    expect(saved.metrics).toMatchObject({
      provenance: "untrained",
      trained: false,
      edited: false,
    });
    expect(String(saved.metrics.note)).toContain("untrained random initialization");
    unmount(app);
    target.remove();
  });
});

// --- and the tab believes the file, not itself ----------------------------------------

interface Tab {
  root: HTMLElement;
  text: () => string;
  testid: (id: string) => HTMLElement | null;
  load: (bundle: LexModelBundle) => Promise<void>;
  dispose: () => void;
}

async function openTab(settle: boolean = true): Promise<Tab> {
  const target = document.createElement("div");
  document.body.appendChild(target);
  const app = mount(LexiconLab, { target, props: {} });
  flushSync();
  // The corpus fetch is a real async read; give it a turn to land. `settle: false` is the
  // race a reader hits by clicking "Load model" the instant the tab opens.
  if (settle) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  }
  return {
    root: target,
    text: () => target.textContent ?? "",
    testid: (id: string) => target.querySelector<HTMLElement>(`[data-testid="${id}"]`),
    load: async (bundle: LexModelBundle) => {
      const input = target.querySelector<HTMLInputElement>(
        '[data-testid="lex-load-model-input"]',
      );
      if (input === null) throw new Error("the load-model input did not render");
      const file = new NodeFile([JSON.stringify(bundle)], "model.llmlex.json", {
        type: "application/json",
      });
      Object.defineProperty(input, "files", { value: [file], configurable: true });
      input.dispatchEvent(new Event("change", { bubbles: true }));
      // `load()` is async (File.text()), and it races the corpus fetch this tab started on
      // mount. Wait for the panel's own verdict rather than for a fixed number of ticks.
      for (let i = 0; i < 200; i++) {
        await new Promise((r) => setTimeout(r, 0));
        flushSync();
        if (
          target.querySelector('[data-testid="lex-file-ok"]') !== null ||
          target.querySelector('[data-testid="lex-file-error"]') !== null
        ) {
          break;
        }
      }
      flushSync();
    },
    dispose: () => {
      unmount(app);
      target.remove();
    },
  };
}

describe("loading a model file", () => {
  it("keeps the untrained warnings for a file that records `trained: false`", async () => {
    const tab = await openTab();
    try {
      // Verbatim from the red-team report's `SAVED bundle metrics`.
      await tab.load(
        bundleWith({
          note: "untrained random initialization",
          provenance: "untrained",
          trained: false,
          edited: false,
        }),
      );
      expect(tab.testid("lex-file-error")).toBeNull();
      // F1: every one of these was cleared by the load, and the sampler switched to
      // "Generate from the model you trained".
      expect(tab.testid("lex-save-untrained")).not.toBeNull();
      expect(tab.testid("lex-spectrum-untrained")).not.toBeNull();
      expect(tab.testid("lex-forward-untrained")).not.toBeNull();
      expect(tab.text()).toContain("has not been trained");
      expect(tab.text()).not.toContain("from the model you trained");
      // The success line claims only what was actually checked here, in this browser.
      expect(tab.testid("lex-file-ok")?.textContent ?? "").toContain(
        "weights + vocabulary verified",
      );
    } finally {
      tab.dispose();
    }
  });

  it("says the history is unknown for a file that records no provenance", async () => {
    const tab = await openTab();
    try {
      // The shape `GET /api/lex/model` emits for a backend-trained model: real losses,
      // no provenance field. It was trained — but nothing in the file says so, and F6's
      // forged `final_loss` shows why this block cannot be promoted to evidence.
      await tab.load(bundleWith({ first_loss: 3.8, final_loss: 2.26, steps: 400, seed: 0 }));
      expect(tab.testid("lex-file-error")).toBeNull();
      expect(tab.testid("lex-spectrum-unrecorded")).not.toBeNull();
      expect(tab.testid("lex-forward-unrecorded")).not.toBeNull();
      expect(tab.testid("lex-save-unrecorded")).not.toBeNull();
      expect(tab.text()).toContain("does not record whether");
      expect(tab.text()).not.toContain("from the model you trained");
      // Never the opposite lie either: it is not asserted to be untrained.
      expect(tab.testid("lex-spectrum-untrained")).toBeNull();
    } finally {
      tab.dispose();
    }
  });

  it("does show the trained wording for a file that records a trained model", async () => {
    const tab = await openTab();
    try {
      await tab.load(
        bundleWith({
          note: "trained in the Lexicon Lab",
          provenance: "trained",
          trained: true,
          edited: false,
        }),
      );
      expect(tab.testid("lex-file-error")).toBeNull();
      expect(tab.text()).toContain("from the model you trained");
      expect(tab.testid("lex-spectrum-untrained")).toBeNull();
      expect(tab.testid("lex-spectrum-unrecorded")).toBeNull();
      expect(tab.testid("lex-save-untrained")).toBeNull();
    } finally {
      tab.dispose();
    }
  });

  it("attributes the claim to the file rather than vouching for it", async () => {
    const tab = await openTab();
    try {
      await tab.load(bundleWith({ provenance: "trained", trained: true, edited: false }));
      const claim = tab.testid("lex-file-claim")?.textContent ?? "";
      // The digests cover the weights and the word list. They do not cover `metrics`, and
      // the line that repeats `metrics` has to say so — F6's justification for leaving the
      // block unhashed was that nothing load-bearing is read from it, which F1 disproved.
      expect(claim).toContain("the file's own label");
      expect(claim).toContain("metrics");
      expect(tab.testid("lex-file-ok")?.textContent ?? "").not.toContain("verified.");
    } finally {
      tab.dispose();
    }
  });

  /**
   * Found while writing the tests above, and of the same family as F1: the load line said
   * the file verified and the tab then threw the model away without a word.
   *
   * `shapeKey` contains `vocab?.rows ?? 0`, and the retirement effect compared it against a
   * snapshot taken when the model was adopted. Load a file before the corpus fetch lands
   * and the snapshot is taken against `rows = 0`; the fetch then "changes the shape" and
   * the just-loaded model is discarded, leaving the untrained warnings up.
   */
  it("keeps a model loaded before the corpus fetch landed", async () => {
    const tab = await openTab(false);
    try {
      await tab.load(
        bundleWith({ provenance: "trained", trained: true, edited: false }),
      );
      // Let the corpus fetch land AFTER the load, which is what used to retire the model.
      for (let i = 0; i < 50; i++) {
        await new Promise((r) => setTimeout(r, 0));
        flushSync();
      }
      expect(tab.testid("lex-file-ok")?.textContent ?? "").toContain("verified");
      expect(tab.testid("lex-spectrum-untrained")).toBeNull();
      expect(tab.testid("lex-active-model")?.textContent ?? "").toContain("loaded from");
      expect(tab.text()).toContain("from the model you trained");
    } finally {
      tab.dispose();
    }
  });

  it("carries a hand-edited file's edited state, which no in-tab edit is holding", async () => {
    const tab = await openTab();
    try {
      await tab.load(
        bundleWith({ provenance: "edited-trained", trained: true, edited: true }),
      );
      expect(tab.testid("lex-file-error")).toBeNull();
      expect(tab.testid("lex-spectrum-edited")).not.toBeNull();
      expect(tab.testid("lex-save-edited")).not.toBeNull();
      expect(tab.text()).toContain("hand-edited");
    } finally {
      tab.dispose();
    }
  });
});
