/**
 * The README's numbers, measured against the artefacts they describe.
 *
 * `README.md:16` said "1003-word vocab", which is two different quantities run together:
 * `geo/config.py` declares `VOCAB_WORDS = 1000  # word/punctuation types drawn from the
 * corpus` and `VOCAB_SIZE = 1003  # VOCAB_WORDS + the three specials below`. The app's own
 * prose keeps them apart — `GeometryLab.svelte` says "1000-word vocab" and "Its 1003 token
 * embeddings"; `TokenStrip.svelte` says "not in the 1000-word vocabulary" — and the change
 * that introduced "1003-word" was made as a CORRECTION of "~1000-word", citing an e2e
 * assertion on `spec.model.vocab_size`, which is the ROW count.
 *
 * Both numbers are read here from the real exported vocabulary the Pages build serves, so
 * the sentence is checked against the model rather than against another sentence.
 */
import fs from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const REPO = path.resolve(__dirname, "../../../..");
const README = fs.readFileSync(path.join(REPO, "README.md"), "utf-8");

interface GeoVocabAsset {
  vocab_size: number;
  specials: Record<string, number>;
  tokens: string[];
}

const VOCAB: GeoVocabAsset = JSON.parse(
  fs.readFileSync(
    path.resolve(__dirname, "../../public/static-data/geo/vocab.json"),
    "utf-8",
  ),
) as GeoVocabAsset;

describe("the README describes the GeoTransformer's vocabulary in the right units", () => {
  it("counts the rows and the words separately, from the exported vocabulary", () => {
    const rows = VOCAB.tokens.length;
    const specials = Object.keys(VOCAB.specials).length;
    const words = rows - specials;
    // The export is self-consistent first: three specials at the front, words after.
    expect(VOCAB.vocab_size).toBe(rows);
    expect(specials).toBe(3);
    expect(words).toBe(1000);
    expect(rows).toBe(1003);

    // …and the README says exactly that, rather than calling the row count a word count.
    expect(README).toContain(`${words}-word vocab in ${rows} rows`);
    expect(README, "the row count must not be quoted as a word count").not.toMatch(
      new RegExp(`${rows}-word`),
    );
  });
});
