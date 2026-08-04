/**
 * The Dolch sight-word lists (Dolch, 1936) — the *prescribed* vocabulary budget.
 *
 * A word-for-word transcription of
 * `code/backend/src/llm_geometry/lex/dolch.py`, which is itself a CORRECTED
 * transcription of the `tiny-models` source project. Both corrections are load-bearing
 * and both are pinned by `tests/unit/lexEngine.test.ts`, which re-reads the Python file
 * and compares every list element:
 *
 * 1. First grade has `going`, NOT `giving`. The published list has `going`; the source's
 *    `giving` was a transcription slip (the list already contains `give` separately).
 * 2. `Santa Claus` is absent from the nouns. It contains a space, so a word-level
 *    tokenizer can never match it as one token — in the source it silently made the
 *    "315" budget a 314-word budget. We drop it and report the measured count.
 *
 * NOT ported, exactly as in Python: the source's `MASK_50`. See the Python module's
 * docstring for why (it is not "the commonest words in English").
 *
 * PROVENANCE, precisely. Dolch's 1936 article (ESJ 36(6):456-460) prints the 220 service
 * words (pp. 458-459, grouped by PART OF SPEECH) and the 95 nouns (p. 460). It does NOT
 * contain the pre-primer / primer / first-grade split the three blocks below encode; that
 * is the conventional grading which has travelled with the list since, and where it first
 * appeared has not been established here. "The published list has `going`" above refers to
 * that conventional graded list, not to the 1936 article.
 *
 * Sizes are MEASURED by `dolchSizes()`, never quoted as literals in code or prose.
 */

const words = (block: string): string[] => block.split(/\s+/).filter((w) => w.length > 0);

// --- the graded service words --------------------------------------------------------

export const PRE_PRIMER = words(`a and away big blue can come down find for funny go help here I in is
it jump little look make me my not one play red run said see the three to two up we
where yellow you`);

export const PRIMER = words(`all am are at ate be black brown but came did do eat four get good have he
into like must new no now on our out please pretty ran ride saw say she so soon that
there they this too under want was well went what white who will with yes`);

// `going`, NOT `giving` — see the module docstring.
export const FIRST = words(`after again an any as ask by could every fly from give going had has her him
his how just know let live may of old once open over put round some stop take thank
them then think walk were when`);

export const SECOND = words(`always around because been before best both buy call cold does don't fast
first five found gave goes green its made many off or pull read right sing sit sleep
tell their these those upon us use very wash which why wish work would write
your`);

export const THIRD = words(`about better bring carry clean cut done draw drink eight fall far full got
grow hold hot hurt if keep kind laugh light long much myself never only own pick seven
shall show six small start ten today together try warm`);

// --- the nouns -----------------------------------------------------------------------
// `Santa Claus` is deliberately absent: see the module docstring.

export const NOUNS = words(`apple baby back ball bear bed bell bird birthday boat box boy bread brother
cake car cat chair chicken children Christmas coat corn cow day dog doll door duck egg
eye farm farmer father feet fire fish floor flower game garden girl good-bye grass
ground hand head hill home horse house kitty leg letter man men milk money morning
mother name nest night paper party picture pig rabbit rain ring robin school seed sheep
shoe sister snow song squirrel stick street sun table thing time top toy tree watch
water way wind window wood`);

// --- cumulative budgets --------------------------------------------------------------
// Each budget NESTS in the next, which is what makes |V| a clean independent variable:
// growing the budget only ever ADDS words, so a comparison across budgets is not
// confounded by words leaving.

const SERVICE = [...PRE_PRIMER, ...PRIMER, ...FIRST, ...SECOND, ...THIRD];

export type DolchBudgetName = "pre_primer" | "primer" | "first" | "service" | "full";

export const DOLCH_BUDGETS: Record<DolchBudgetName, string[]> = {
  pre_primer: PRE_PRIMER,
  primer: [...PRE_PRIMER, ...PRIMER],
  first: [...PRE_PRIMER, ...PRIMER, ...FIRST],
  service: SERVICE,
  full: [...SERVICE, ...NOUNS],
};

/** Ascending order of size. Labels fill their number in from the measured length. */
export const DOLCH_ORDER: DolchBudgetName[] = ["pre_primer", "primer", "first", "service", "full"];

export const isDolchBudgetName = (name: string): name is DolchBudgetName =>
  (DOLCH_ORDER as string[]).includes(name);

/** The lower-cased, de-duplicated word list for a named budget, in a stable order. */
export function dolchBudget(name: string): string[] {
  if (!isDolchBudgetName(name)) {
    throw new Error(`unknown Dolch budget ${JSON.stringify(name)}; expected one of ${DOLCH_ORDER.join(", ")}`);
  }
  const seen = new Set<string>();
  const out: string[] = [];
  for (const w of DOLCH_BUDGETS[name]) {
    const lw = w.toLowerCase();
    if (!seen.has(lw)) {
      seen.add(lw);
      out.push(lw);
    }
  }
  return out;
}

/** Measured size of every budget. The UI reads this instead of quoting literals. */
export function dolchSizes(): Record<DolchBudgetName, number> {
  const out = {} as Record<DolchBudgetName, number>;
  for (const name of DOLCH_ORDER) out[name] = dolchBudget(name).length;
  return out;
}
