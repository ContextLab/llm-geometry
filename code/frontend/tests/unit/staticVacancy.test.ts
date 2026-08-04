/**
 * `POST /api/lex/vacancy` in the STATIC build — and the parity that makes it worth having.
 *
 * The Lexicon Lab computes in the browser in both modes, so the vacancy transform is not
 * something the Pages build refuses or approximates: it runs the real
 * `lexEngine/vacancy.ts` over the real committed corpus and answers the same request the
 * real FastAPI route answers. FR-722 says the static build "serves the same capability";
 * this file is the measurement of that sentence rather than a restatement of it.
 *
 * THE CENTRAL TEST is `matches the live backend field for field`. Its fixture,
 * `tests/fixtures/vacancy-api-golden.json`, was produced by
 *
 *     python scripts/export_vacancy_api_golden.py
 *
 * running the REAL app through FastAPI's TestClient on the REAL corpus — the responses in
 * it are transcripts, not expectations someone typed. `test_api_lex.py::
 * test_vacancy_matches_the_static_client_fixture` asserts the live route still returns
 * exactly the same file, so the two stacks are pinned to ONE document: if either drifts,
 * one of the two tests fails and names the field it drifted on.
 *
 * What that comparison actually covers, per case: every §10 statistic, the resolved
 * vocabulary word for word, the budget's measured coverage of the vacated corpus, the
 * preview text character for character, and `vacated_sha256` — the sha256 of the WHOLE
 * 86 kB rewrite, which pins every byte the preview does not show. Floats need no tolerance
 * because both stacks round to 6 significant digits before serving (`jsonable_6sig` and
 * `sig6`), so the wire values are identical, not merely close.
 *
 * The remaining tests cover what a single-request fixture cannot: the properties that only
 * exist ACROSS requests (nesting, stability, the invariance theorem's effect on coverage)
 * and the parameter errors, which have no fixture because they have no body.
 */

import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";

import { createStaticClient, type StaticClient } from "../../src/lib/staticClient";
import type { FetchLike } from "../../src/lib/staticClient/assets";
import {
  VACANCY_PREVIEW_CHARS,
  VACANCY_PREVIEW_MAX,
  type LexVacancyBody,
  type LexVacancyResult,
} from "../../src/lib/staticClient/lex";
import { fsStaticFetch } from "./staticTestUtils";

function client(fetchImpl: FetchLike = fsStaticFetch()): StaticClient {
  return createStaticClient({ baseUrl: "/", fetchImpl });
}

// --- the parity fixture ---------------------------------------------------------------

interface ApiGoldenCase {
  label: string;
  request: LexVacancyBody;
  response: LexVacancyResult;
}

interface ApiGolden {
  format: string;
  git_sha: string;
  command: string;
  contract: string;
  tolerance: number;
  endpoint: string;
  defaults: { preview_chars: number; preview_max: number };
  corpus: { sha256: string; chars: number };
  cases: ApiGoldenCase[];
}

const GOLDEN: ApiGolden = JSON.parse(
  fs.readFileSync(
    path.resolve(__dirname, "../fixtures/vacancy-api-golden.json"),
    "utf-8",
  ),
) as ApiGolden;

describe("the API-parity fixture is the real route's own output", () => {
  it("declares its format, its generator and the contract it pins", () => {
    expect(GOLDEN.format).toBe("vacancy-api-golden-v1");
    expect(GOLDEN.command).toBe("python scripts/export_vacancy_api_golden.py");
    expect(GOLDEN.endpoint).toBe("/api/lex/vacancy");
    expect(GOLDEN.contract).toBe("specs/002-interactive-model-explorer/contracts/api.md");
    expect(GOLDEN.git_sha).toMatch(/^[0-9a-f]{7,40}$/);
    expect(GOLDEN.cases.length).toBeGreaterThanOrEqual(6);
  });

  /**
   * The two stacks each hold their own copy of the preview defaults. A fixture generated
   * with one and consumed by the other would paper over a mismatch precisely where it
   * matters — the excerpt length — so the numbers are compared directly as well.
   */
  it("was generated with the preview defaults this build uses", () => {
    expect(GOLDEN.defaults.preview_chars).toBe(VACANCY_PREVIEW_CHARS);
    expect(GOLDEN.defaults.preview_max).toBe(VACANCY_PREVIEW_MAX);
    const usingDefault = GOLDEN.cases.filter((c) => c.request.preview_chars === undefined);
    expect(usingDefault.length).toBeGreaterThan(0);
    for (const c of usingDefault) {
      expect(c.response.preview_chars).toBe(VACANCY_PREVIEW_CHARS);
    }
  });
});

describe("the static build answers /api/lex/vacancy exactly as the backend does", () => {
  for (const testCase of GOLDEN.cases) {
    it(`matches the live backend field for field — ${testCase.label}`, async () => {
      const got = await client().lexVacancy(testCase.request);
      const want = testCase.response;

      // The digest first and by name: it is the one assertion that covers all 86 kB, and
      // a failure here means the two stacks vacated the corpus DIFFERENTLY, which is a
      // different (and worse) bug than a statistic being reported differently.
      expect(got.vacated_sha256).toBe(want.vacated_sha256);
      expect(got.original_sha256).toBe(want.original_sha256);
      expect(got.vacated_chars).toBe(want.vacated_chars);

      expect(got.vacancy_stats).toEqual(want.vacancy_stats);
      expect(got.vocabulary_rule).toBe(want.vocabulary_rule);
      expect(got.words).toEqual(want.words);
      expect(got.budget).toEqual(want.budget);
      expect(got.corpus).toEqual(want.corpus);
      expect(got.preview).toBe(want.preview);
      expect(got.original_preview).toBe(want.original_preview);

      // …and then the whole object, so a field added on one side and not the other is a
      // failure rather than something the assertions above happen not to look at.
      expect(got).toEqual(want);
    });
  }
});

describe("what the response says about itself", () => {
  it("returns an excerpt plus a digest, never the whole corpus", async () => {
    const res = await client().lexVacancy({ p: 1, seed: 0 });
    expect(res.preview_chars).toBe(VACANCY_PREVIEW_CHARS);
    expect(res.preview.length).toBe(VACANCY_PREVIEW_CHARS);
    expect(res.truncated).toBe(true);
    // The corpus is ~86 kB; the excerpt is a fortieth of it and the digest covers the rest.
    expect(res.vacated_chars).toBeGreaterThan(VACANCY_PREVIEW_CHARS * 10);
    expect(res.vacated_sha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it("serves a longer excerpt on request, up to the ceiling", async () => {
    const res = await client().lexVacancy({ p: 1, seed: 0, preview_chars: 10 });
    expect(res.preview.length).toBe(10);
    expect(res.truncated).toBe(true);
    await expect(
      client().lexVacancy({ p: 1, preview_chars: VACANCY_PREVIEW_MAX + 1 }),
    ).rejects.toMatchObject({ type: "InvalidParamError" });
  });

  it("is the identity at p = 0, digest included", async () => {
    const res = await client().lexVacancy({ p: 0, seed: 0 });
    // `u ∈ [0, 1)`, so `u < 0` is never true and nothing can vacate.
    expect(res.vacated_sha256).toBe(res.original_sha256);
    expect(res.preview).toBe(res.original_preview);
    expect(res.vacancy_stats.corpusTypesVacated).toBe(0);
    expect(res.vacancy_stats.tokensVacated).toBe(0);
    expect(res.vacancy_stats.stemsVacated).toBe(0);
  });
});

describe("the properties a single request cannot show", () => {
  /**
   * SC-703 through the API surface. Under the mapped condition the transform is a pure
   * relabelling of the vocabulary, so a budget's measured coverage of the VACATED corpus
   * must equal its coverage of the English one — same tokens in budget, same `<unk>` in
   * the same places. This is the invariance theorem stated in the units the panel shows.
   */
  it("leaves coverage bit-identical under the mapped vocabulary, at every p", async () => {
    const c = client();
    const english = await c.lexCoverage({ source: "dolch", budget: "primer" });
    for (const p of [0, 0.25, 0.5, 0.75, 1]) {
      for (const seed of [0, 7]) {
        const res = await c.lexVacancy({ p, seed, source: "dolch", budget: "primer" });
        expect(res.vocabulary_rule).toBe("mapped");
        expect(res.budget.coverage).toEqual(english.coverage);
        expect(res.budget.rows).toBe(english.rows);
        expect(res.corpus.n_tokens).toBe(english.corpus.n_tokens);
        expect(res.corpus.n_distinct).toBe(english.corpus.n_distinct);
        expect(res.corpus.n_lines).toBe(english.corpus.n_lines);
      }
    }
  });

  /** SC-705: the controls BREAK it, and the break is what the panel measures. */
  it("collapses coverage under both control conditions, at the same p", async () => {
    const c = client();
    const mapped = await c.lexVacancy({ p: 0.5, seed: 0, source: "dolch", budget: "primer" });
    const inconsistent = await c.lexVacancy({
      p: 0.5,
      seed: 0,
      consistent: false,
      source: "dolch",
      budget: "primer",
    });
    const revealed = await c.lexVacancy({
      p: 0.5,
      seed: 0,
      reveal_after: 2,
      source: "dolch",
      budget: "primer",
    });
    expect(mapped.vocabulary_rule).toBe("mapped");
    expect(inconsistent.vocabulary_rule).toBe("rebuilt");
    expect(revealed.vocabulary_rule).toBe("rebuilt");
    expect(inconsistent.budget.coverage.unk_rate).toBeGreaterThan(mapped.budget.coverage.unk_rate);
    expect(revealed.budget.coverage.unk_rate).toBeGreaterThan(mapped.budget.coverage.unk_rate);
  });

  /**
   * SC-701 / SC-702 as the API reports them: `stemsVacated` never falls as `p` rises, and
   * a stem minted at a low `p` is byte-identical at every higher one. The second half is
   * checked through the mapped word list, which is the map restricted to the budget.
   */
  it("nests and stays stable as p rises", async () => {
    const c = client();
    const seen: { p: number; stems: number; words: string[] }[] = [];
    for (const p of [0, 0.25, 0.5, 0.75, 1]) {
      const res = await c.lexVacancy({ p, seed: 0, source: "dolch", budget: "primer" });
      seen.push({ p, stems: res.vacancy_stats.stemsVacated, words: res.words });
    }
    for (let i = 1; i < seen.length; i++) {
      expect(seen[i].stems).toBeGreaterThanOrEqual(seen[i - 1].stems);
      // Stability: any word already rewritten at the lower `p` is unchanged at the higher.
      for (let w = 0; w < seen[i].words.length; w++) {
        const lower = seen[i - 1].words[w];
        const higher = seen[i].words[w];
        if (lower !== seen[0].words[w]) expect(higher).toBe(lower);
      }
    }
    // The two ends are identities, not observations: `u ∈ [0, 1)`, so nothing vacates at
    // `p = 0` and EVERY eligible stem vacates at `p = 1` (§10).
    const last = await client().lexVacancy({ p: 1, seed: 0, source: "dolch", budget: "primer" });
    expect(seen[0].stems).toBe(0);
    expect(last.vacancy_stats.stemsVacated).toBe(last.vacancy_stats.stemsTotal);
    expect(last.vacancy_stats.corpusTypesVacated).toBe(last.vacancy_stats.corpusTypesEligible);
  });

  /** FR-716: vacancy composes with a user's own text, not only with the shipped corpus. */
  it("transforms pasted text as readily as the shipped corpus", async () => {
    const text = "The little brown squirrel ate the pretty acorn.\nThe squirrel ran away.\n";
    const res = await client().lexVacancy({ text, p: 1, seed: 0, preview_chars: 200 });
    expect(res.original_preview).toBe(text);
    expect(res.preview).not.toBe(text);
    // §1: only WORD_RE matches are replaced — punctuation and line breaks pass through.
    expect(res.preview.split("\n").length).toBe(text.split("\n").length);
    expect((res.preview.match(/\./g) ?? []).length).toBe(2);
    expect(res.vacancy_stats.tokensTotal).toBe(12);
    // Three of the twelve are `the`/`The`, which §2.1 preserves. The other nine —
    // little, brown, squirrel, ate, pretty, acorn, squirrel, ran, away — are open class,
    // and at `p = 1` every one of them moves.
    expect(res.vacancy_stats.tokensVacated).toBe(9);
    expect(res.vacancy_stats.corpusTypesVacated).toBe(8); // `squirrel` occurs twice
  });
});

describe("bad vacancy parameters are refused in the shared envelope", () => {
  const bad: [string, LexVacancyBody][] = [
    ["p above 1", { p: 1.5 }],
    ["p below 0", { p: -0.1 }],
    ["a negative reveal_after", { reveal_after: -1 }],
    ["preview_chars above the ceiling", { preview_chars: VACANCY_PREVIEW_MAX + 1 }],
    ["preview_chars below zero", { preview_chars: -1 }],
    ["a bare string for keep", { keep: "little" as unknown as string[] }],
    ["size on a Dolch budget", { source: "dolch", size: 50 }],
    ["an unknown budget", { budget: "not-a-budget" }],
  ];
  for (const [what, body] of bad) {
    it(`rejects ${what}`, async () => {
      await expect(client().lexVacancy(body)).rejects.toMatchObject({
        type: "InvalidParamError",
      });
    });
  }
});

describe("training on a vacated corpus", () => {
  /**
   * FR-713 / SC-703's corollary, run for real in the browser engine: with the mapped
   * vocabulary the token id stream is unchanged, so a real training run at `p = 0.5` must
   * produce BIT-IDENTICAL losses to the same run on the English corpus. Not "close" —
   * identical, because `runTraining` sees the same integers either way.
   */
  it("is bit-identical to training on the English corpus under the mapped vocabulary", async () => {
    const c = client();
    const shape = {
      source: "dolch",
      budget: "pre_primer",
      steps: 12,
      d_model: 16,
      n_layers: 1,
      n_heads: 1,
      ctx: 32,
      batch_size: 8,
      seed: 3,
    } as const;
    const english = await trainToCompletion(c, { ...shape });
    const vacated = await trainToCompletion(c, {
      ...shape,
      vacancy: { p: 0.5, seed: 0 },
    });
    expect(vacated.final_loss).toBe(english.final_loss);
    expect(vacated.first_loss).toBe(english.first_loss);
    expect(vacated.val_loss).toBe(english.val_loss);
    expect(vacated.n_tokens).toBe(english.n_tokens);
    expect(vacated.vocab_rows).toBe(english.vocab_rows);
    // Same numbers, DIFFERENT words: the model is blind to the relabelling, which is the
    // finding rather than a caveat about it.
    expect(vacated.model_token).not.toBe(english.model_token);
  }, 120_000);

  it("rejects a vacancy block that is not an object", async () => {
    await expect(
      client().lexTrain({ vacancy: 0.5 as unknown as Record<string, never> }),
    ).rejects.toMatchObject({ type: "InvalidParamError" });
  });
});

/** Start a training run and wait for its result, exactly as a view would. */
async function trainToCompletion(
  c: StaticClient,
  body: Parameters<StaticClient["lexTrain"]>[0],
): Promise<Record<string, number | string> & { model_token: string }> {
  const started = await c.lexTrain(body);
  if (started.ready) {
    return started as unknown as Record<string, number | string> & { model_token: string };
  }
  const done = await new Promise<Record<string, unknown>>((resolve, reject) => {
    c.subscribeProgress(started.job_id, {
      onDone: (data) => resolve(data ?? {}),
      onError: (type, message) => reject(new Error(`${type}: ${message}`)),
    });
  });
  return done as Record<string, number | string> & { model_token: string };
}
