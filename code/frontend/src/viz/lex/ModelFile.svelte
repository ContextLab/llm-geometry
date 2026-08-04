<script lang="ts">
  /**
   * US-8 — keep it. Save the active model to a file, and load one back.
   *
   * A model trained here lives in this tab and nowhere else: there is no account, no
   * server-side checkpoint, and the content-hash token that identifies an edited weight
   * set means nothing to any other browser. Closing the page ends the model. A
   * `.llmlex.json` bundle is the portable form.
   *
   * The file carries the weights AND the whole vocabulary, because in this model the
   * vocabulary IS the budget: token id 17 is a different word in every model on this page.
   * It carries three digests — the backend's joint `model_token`, a `weights_token` over
   * the weights alone, and `vocab_sha256` over the word list — and ALL THREE are mandatory
   * on load. A file missing any of them is refused rather than loaded leniently, which is
   * the specific hole feature 004 shipped in the Geometry Lab: a missing digest was read
   * as "nothing to check", so tampered weights loaded cleanly the moment you deleted the
   * field instead of editing it.
   *
   * This is ONE format, not a browser dialect: it is the payload of `GET|POST
   * /api/lex/model` in `specs/006-lexicon-lab-tiny/contracts/api-lex.md`, with the
   * Python model's tensor names on the wire, so a file saved here loads into the full
   * stack and back. The format and every check live in `lib/lexEngine/bundle.ts`, tested
   * by `tests/unit/lexBundle.test.ts` — including the deletion attack, and including a
   * `model_token` pinned against one the real Python produced.
   */
  import {
    LEX_BUNDLE_SUFFIX,
    LexModel,
    LexVocab,
    exportLexBundle,
    importLexBundle,
    type BudgetSource,
  } from "../../lib/lexEngine";
  import Explain from "../../lib/Explain.svelte";
  import { hasTrainedBase, isEdited, type Provenance } from "./provenance";

  interface Props {
    /** The model currently driving the page — trained, edited, or the random init. */
    model: LexModel | null;
    vocab: LexVocab | null;
    /**
     * What these weights are. Saving is allowed in all four states — but the file's own
     * `metrics` block and the note beside the button must say WHICH of the four, or a
     * `.llmlex.json` full of hand-edited weights arrives somewhere else labelled "trained".
     */
    provenance: Provenance;
    /** Provenance line for the active model, written into the file. */
    note: string;
    onLoaded: (model: LexModel, vocab: LexVocab, note: string) => void;
  }
  let { model, vocab, provenance, note, onLoaded }: Props = $props();

  /** What the file says about itself when the tab has no note of its own to carry. */
  const DEFAULT_NOTES: Record<Provenance, string> = {
    trained: "trained in the Lexicon Lab",
    untrained: "untrained random initialization",
    "edited-trained": "trained in the Lexicon Lab, then hand-edited in the Weight Lab",
    "edited-untrained": "untrained random initialization, hand-edited in the Weight Lab",
  };

  let fileInput: HTMLInputElement | undefined = $state();
  let error = $state("");
  let ok = $state("");

  const ready = $derived(model !== null && vocab !== null && vocab.rows === model.cfg.vocabRows);

  function save(): void {
    error = "";
    ok = "";
    if (!model || !vocab) return;
    try {
      const bundle = exportLexBundle({
        config: model.cfg,
        weights: model.weights,
        vocabWords: vocab.words,
        budgetSource: vocab.source,
        budgetName: vocab.budgetName,
        // `metrics` is the contract's free provenance block — outside every digest, and
        // the only block that cannot mislabel a token.
        metrics: {
          // The state ALWAYS leads, and the tab's own note (a training summary, or the
          // edit that was applied) follows it — "embed scaled by 0.5" alone would not tell
          // a later reader whether anything had ever been trained.
          note: note ? `${DEFAULT_NOTES[provenance]} · ${note}` : DEFAULT_NOTES[provenance],
          // `trained` describes where the weights CAME FROM; `edited` says whether they
          // are still what training produced. One flag could not tell those apart, and a
          // reader downstream would have to guess.
          provenance,
          trained: hasTrainedBase(provenance),
          edited: isEdited(provenance),
        },
      });
      const name = `lexicon-${vocab.source}-${vocab.budgetName}-${bundle.model_token.slice(0, 8)}${LEX_BUNDLE_SUFFIX}`;
      const url = URL.createObjectURL(new Blob([JSON.stringify(bundle)], { type: "application/json" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      a.click();
      URL.revokeObjectURL(url);
      ok = `saved ${name} · ${vocab.budgetSize} words + ${vocab.rows - vocab.budgetSize} specials`;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  async function load(event: Event): Promise<void> {
    error = "";
    ok = "";
    const input = event.currentTarget as HTMLInputElement;
    const f = input.files?.[0];
    input.value = ""; // so re-picking the same file fires change again
    if (!f) return;
    try {
      const parsed: unknown = JSON.parse(await f.text());
      const loaded = importLexBundle(parsed); // throws on ANY integrity failure
      const nextVocab = new LexVocab(
        loaded.vocabWords,
        loaded.budgetSource as BudgetSource,
        loaded.budgetName,
      );
      const nextModel = new LexModel(loaded.config, loaded.weights);
      const carried = typeof loaded.metrics.note === "string" ? loaded.metrics.note : "";
      onLoaded(
        nextModel,
        nextVocab,
        carried ? `loaded from ${f.name} · ${carried}` : `loaded from ${f.name}`,
      );
      ok =
        `loaded ${f.name} · ${nextVocab.budgetSize}-word ${loaded.budgetSource} budget ` +
        `(${loaded.config.vocabRows} rows) · model ${loaded.modelToken.slice(0, 12)}… verified`;
    } catch (e) {
      error =
        e instanceof SyntaxError
          ? `${f.name} is not valid JSON, so there is nothing to verify.`
          : e instanceof Error
            ? e.message
            : String(e);
    }
  }
</script>

<div class="panel-body" data-testid="lex-model-file">
  <div class="head">
    <h3>Save / load</h3>
    <span class="hint">this model lives in this tab — save it to keep it</span>
  </div>

  <p class="panel-note">
    A <code>{LEX_BUNDLE_SUFFIX}</code> file carries the weights <b>and</b> the whole
    vocabulary, under three checksums: one over the two together, one over the weights
    alone, one over the word list. All three are required when a file is read and a
    mismatch is fatal — a file with genuine weights beside a substituted word list would
    relabel every token on this page while looking perfectly healthy. It is the same file
    the backend's <code>/api/lex/model</code> reads and writes, so a model saved here is
    not trapped in this browser.
  </p>

  <div class="io-row">
    <button data-testid="lex-save-model" disabled={!ready} onclick={save}>↓ Save model</button>
    <button data-testid="lex-load-model" onclick={() => fileInput?.click()}>↑ Load model</button>
    <input
      bind:this={fileInput}
      type="file"
      accept=".json,application/json"
      class="hidden-file"
      data-testid="lex-load-model-input"
      onchange={(e) => void load(e)}
    />
  </div>

  {#if provenance === "untrained"}
    <p class="hint small" data-testid="lex-save-untrained">
      Nothing has been trained yet, so a save right now writes the <b>random
      initialization</b> — a real model file, of a model that has learned nothing.
    </p>
  {:else if provenance === "edited-untrained"}
    <p class="hint small" data-testid="lex-save-untrained">
      Nothing has been trained yet and the weights have been hand-edited, so a save right
      now writes <b>the random initialization with your edit applied</b> — a real model
      file, of a model that has learned nothing and is no longer its own initializer's
      output either. The file says so in its <code>metrics</code> block.
    </p>
  {:else if provenance === "edited-trained"}
    <p class="hint small" data-testid="lex-save-edited">
      The weights have been hand-edited, so a save right now writes <b>the edited
      weights</b>, not the ones training produced. Restore in the Weight Lab first if the
      trained model is what you meant to keep.
    </p>
  {/if}

  {#if error}<div class="err" data-testid="lex-file-error">{error}</div>{/if}
  {#if ok}<div class="ok" data-testid="lex-file-ok">{ok}</div>{/if}

  <Explain
    title="What is inside the file, and what is checked"
    hint="three digests, all mandatory, all fatal on mismatch"
    testid="lex-explain-file"
  >
    <p>
      The bundle is JSON: a <code>config</code> block fixing the architecture, a
      <code>vocab</code> block holding the word list and the four specials, every weight
      tensor as base64 little-endian float32 with its declared shape, a free
      <code>metrics</code> block, and three digests. The tensors are named the way the
      <b>PyTorch</b> model names them (<code>blocks.N.*</code>), not the way the browser
      engine does (<code>layers.N.*</code>) — the translation happens at the file
      boundary, which is what makes one format serve both runtimes.
    </p>
    <ul>
      <li>
        <code>model_token</code> — sha256 over the canonical config, then the word list,
        then every tensor's name, shape and float32 bytes; first 32 hex characters. This
        is the backend's own hash, reproduced byte-for-byte, and it is the one that proves
        the weights and the vocabulary belong <i>together</i>.
      </li>
      <li>
        <code>weights_token</code> — the same construction over the weights alone. It is
        the id the Weight Lab mints for an edited weight set, and it tells you two files
        hold the same weights under different word lists.
      </li>
      <li>
        <code>vocab_sha256</code> — sha256 of the canonical vocabulary block, so a file
        with intact weights and an invented word list cannot load.
      </li>
    </ul>
    <p>
      No digest is optional and a missing one is not "nothing to check" — it is a refusal,
      with its own message. The loader also checks the <b>join</b> structurally: the word
      list plus the four specials must equal the model's embedding row count, and the
      tensor names must be exactly those the <code>tied</code> flag implies — a tied file
      carrying a readout, or an untied one missing it, is refused rather than quietly
      reloaded as a different model.
    </p>
    <p>
      Loading a file replaces the active model <i>and</i> its vocabulary together, and the
      controls follow it, because a model's ids are only meaningful against the budget it
      was trained on.
    </p>
  </Explain>
</div>

<style>
  .panel-body {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    min-width: 0;
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  h3 {
    margin: 0;
    font-size: 0.95rem;
  }
  .hint {
    margin: 0;
    font-size: 0.72rem;
    color: var(--text-dim);
    line-height: 1.45;
  }
  .hint.small {
    font-size: 0.7rem;
  }
  .hint b {
    color: var(--text);
  }
  .panel-note {
    margin: 0.1rem 0 0;
    font-size: 0.74rem;
    line-height: 1.55;
    color: var(--text-dim);
  }
  .panel-note b {
    color: var(--text);
  }
  .panel-note code {
    font-family: var(--mono);
    color: var(--accent);
  }
  .io-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .io-row button {
    padding: 0.35rem 0.8rem;
    font-size: 0.78rem;
  }
  .io-row button:disabled {
    opacity: 0.45;
    cursor: default;
  }
  .hidden-file {
    display: none;
  }
  .err {
    background: rgba(255, 122, 144, 0.1);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 9px;
    padding: 0.45rem 0.6rem;
    font-size: 0.76rem;
    line-height: 1.45;
  }
  .ok {
    background: rgba(91, 224, 176, 0.08);
    color: var(--good);
    border: 1px solid rgba(91, 224, 176, 0.28);
    border-radius: 9px;
    padding: 0.45rem 0.6rem;
    font-size: 0.76rem;
    line-height: 1.45;
    overflow-wrap: anywhere;
  }
</style>
