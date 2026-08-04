/**
 * Generation for the Lexicon Lab.
 *
 * Greedy at `temperature = 0`, otherwise a sample from `softmax(logits[-1] / T)` after
 * setting the logits of `GENERATION_BANNED_IDS = (<unk>, <bos>, <pad>)` to -inf. `<eos>`
 * IS sampleable — it is how the model ends a line.
 *
 * **In-budget output is guaranteed by construction**, because the vocabulary IS the
 * budget: there are no rows for words outside it, so there is nothing to filter. No
 * trie, no post-filter, and none of the source project's `" hameat"` defect (its trie
 * re-opened the root after a completed word). FR-605 / SC-602.
 *
 * "By construction" holds only while the arithmetic does, so it is CHECKED, not assumed.
 * A model whose weights have left float32 range (a `×2` preset applied often enough, or
 * one cell edit of `1e40`, which is finite in JS and `Infinity` once stored) produces NaN
 * logits, and `NaN > -Infinity` is false — so the greedy argmax used to fall through to
 * its initialiser `next = 0`, which is `<unk>`, bypassing the -inf ban entirely, and the
 * sampler fell through to `V - 1`. Both were then hidden by stripping specials from
 * `words`. Non-finite weights and non-finite logits now raise `ComputeError`, exactly as
 * `lex/generate.py` does; nothing is clamped and nothing is sanitised.
 */

import { computeError, invalidParam } from "../geoEngine/errors";
import { LexModel, nonFiniteWeightNames, sfc32 } from "./model";
import { BOS_ID, EOS_ID, GENERATION_BANNED_IDS, LexVocab, SPECIAL_TOKENS, tokenize } from "./vocab";

export const DEFAULT_TEMPERATURE = 0.9;
export const DEFAULT_MAX_NEW_TOKENS = 40;
export const MAX_NEW_TOKENS = 200;

export interface GenerateOptions {
  /** Free text; tokenized with the model's own vocabulary. Out-of-budget words become
   *  `<unk>` and are reported in `promptOov` rather than silently dropped. */
  prompt?: string;
  maxNewTokens?: number;
  /** 0 = greedy (argmax). Above 0, sample from the tempered distribution. */
  temperature?: number;
  seed?: number;
  /**
   * What `<eos>` means. Defaults to `false`, matching the backend's
   * `generate(..., stop_at_eos=False)`: the model runs for the full `maxNewTokens` and
   * `<eos>` renders as a LINE BREAK, which is how a nursery-rhyme model produces verse
   * rather than one long line. `true` stops at the first `<eos>` for a single line.
   *
   * This existed only as the `true` behaviour, which made multi-line output reachable on
   * the full stack and not in the browser — the two runtimes have to mean the same thing
   * by the same name.
   */
  stopAtEos?: boolean;
}

export interface GenerateResult {
  /** The newly generated ids only, `<eos>` included when the model chose to stop. */
  ids: number[];
  /**
   * The generated words as a FLAT list, with the line breaks dropped.
   *
   * `<eos>` is the only token this can remove: `<unk>`, `<bos>` and `<pad>` are masked to
   * -inf before every draw and their appearance is treated as a bug (it raises), so this
   * is a rendering choice — the same ids as `text` without its newlines — and not a
   * budget filter. Every entry is in budget because the vocabulary is the budget.
   */
  words: string[];
  text: string;
  promptIds: number[];
  /** Prompt words the budget cannot express (they entered the model as `<unk>`). */
  promptOov: string[];
  stoppedOnEos: boolean;
}

const SPECIALS = new Set<string>(SPECIAL_TOKENS);
const BANNED = new Set<number>(GENERATION_BANNED_IDS);

/**
 * The masked logits must be usable before anything reads them: at least one row still
 * finite (the backend's `torch.isfinite(logits).any()`), and no NaN or +Infinity anywhere
 * (which the backend catches as non-finite probabilities, and which no comparison-based
 * argmax can survive). Both raise; neither is repaired.
 */
function assertUsableLogits(logits: Float64Array, V: number, step: number): void {
  let anyFinite = false;
  for (let i = 0; i < V; i++) {
    const x = logits[i];
    if (Number.isNaN(x) || x === Infinity) {
      throw computeError(
        `the model produced non-finite logits at step ${step + 1}, so no continuation is ` +
          "defined. This means the weights have left float32 range — most likely a weight " +
          "edit (values above ~3.4e38 become Infinity when stored) or a diverged run. " +
          "Refusing to generate rather than emit an arbitrary token.",
      );
    }
    if (Number.isFinite(x)) anyFinite = true;
  }
  if (!anyFinite) {
    throw computeError(
      `all logits are -inf at step ${step + 1}; generation cannot continue`,
    );
  }
}

export function generate(model: LexModel, vocab: LexVocab, opts: GenerateOptions = {}): GenerateResult {
  const maxNew = Math.trunc(opts.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS);
  const temperature = opts.temperature ?? DEFAULT_TEMPERATURE;
  const stopAtEos = opts.stopAtEos ?? false; // backend default; see GenerateOptions
  if (!(maxNew >= 1 && maxNew <= MAX_NEW_TOKENS)) {
    throw invalidParam(`max_new_tokens must be in 1..${MAX_NEW_TOKENS}, got ${opts.maxNewTokens}`);
  }
  if (!(temperature >= 0 && Number.isFinite(temperature))) {
    throw invalidParam(`temperature must be a finite number >= 0, got ${opts.temperature}`);
  }
  if (vocab.rows !== model.cfg.vocabRows) {
    throw invalidParam(
      `this vocabulary has ${vocab.rows} rows but the model has ${model.cfg.vocabRows} — ` +
        "a model must be generated with the vocabulary it was trained on",
    );
  }

  // Refuse a broken model up front, where the cause is still nameable. `lex/generate.py`
  // reaches the same refusal one step later, from its logits.
  const badWeights = nonFiniteWeightNames(model.cfg, model.weights);
  if (badWeights.length > 0) {
    throw computeError(
      `this model's weights are not finite: ${badWeights.join(", ")} contain NaN or ±Infinity, ` +
        "most likely from a weight edit that pushed them out of float32 range (values above " +
        "~3.4e38 become Infinity when stored) or from a diverged training run. Generation " +
        "would be meaningless, so it is refused rather than approximated.",
    );
  }

  const promptTokens = tokenize(opts.prompt ?? "");
  const promptOov = promptTokens.filter((t) => !vocab.has(t));
  const promptIds = vocab.encode(promptTokens);

  const rand = sfc32(Math.trunc(opts.seed ?? 0));
  const V = model.cfg.vocabRows;
  const ctx = model.cfg.ctx;
  const context: number[] = [BOS_ID, ...promptIds];
  const out: number[] = [];
  let stoppedOnEos = false;

  for (let step = 0; step < maxNew; step++) {
    const window = context.slice(Math.max(0, context.length - ctx));
    const logits = model.lastLogits(window);
    for (const banned of GENERATION_BANNED_IDS) logits[banned] = -Infinity;
    assertUsableLogits(logits, V, step);

    let next: number;
    if (temperature === 0) {
      next = 0;
      let best = -Infinity;
      for (let i = 0; i < V; i++) {
        if (logits[i] > best) {
          best = logits[i];
          next = i;
        }
      }
    } else {
      let max = -Infinity;
      for (let i = 0; i < V; i++) if (logits[i] > max) max = logits[i];
      let sum = 0;
      const probs = new Float64Array(V);
      for (let i = 0; i < V; i++) {
        const e = Number.isFinite(logits[i]) ? Math.exp((logits[i] - max) / temperature) : 0;
        probs[i] = e;
        sum += e;
      }
      const u = rand() * sum;
      let acc = 0;
      next = V - 1;
      for (let i = 0; i < V; i++) {
        acc += probs[i];
        if (u < acc) {
          next = i;
          break;
        }
      }
    }

    // Structurally unreachable: every banned row is -inf and the logits are known finite,
    // so both branches must have chosen a real row. Asserted rather than filtered away —
    // this is the failure the `words` post-filter used to hide.
    if (BANNED.has(next)) {
      throw computeError(
        `generation selected the banned token ${SPECIAL_TOKENS[next] ?? next} at step ${step + 1}, ` +
          "which the -inf mask makes impossible; the model's arithmetic is not sound.",
      );
    }

    out.push(next);
    if (next === EOS_ID && stopAtEos) {
      stoppedOnEos = true;
      break;
    }
    context.push(next);
  }

  // One pass for both renderings, so they can never disagree. With stopAtEos=false,
  // `<eos>` is a line break rather than a terminator — the same rendering the backend
  // does, so both runtimes produce the same shape of verse. `words` is that same
  // sequence flattened; the `SPECIALS` test can only ever match `<eos>`, because every
  // other special is masked to -inf and its selection raises above.
  const words: string[] = [];
  const lines: string[][] = [[]];
  for (const w of vocab.decode(out)) {
    if (w === "<eos>") lines.push([]);
    else if (!SPECIALS.has(w)) {
      lines[lines.length - 1].push(w);
      words.push(w);
    }
  }
  const text = lines
    .filter((l) => l.length > 0)
    .map((l) => l.join(" "))
    .join("\n");
  return { ids: out, words, text, promptIds, promptOov, stoppedOnEos };
}
