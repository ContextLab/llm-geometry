/**
 * The sentences the vacancy transform says NO in, and the corpus counts inside them —
 * declared once, imported by everything that refuses.
 *
 * A refusal is a user-facing answer with numbers in it, and these numbers were typed by
 * hand into four files. The transform rewrite corrected three: `lex/vacancy.py`,
 * `lexEngine/vacancy.ts` and the Info tab moved to `1676` / `8125`, while
 * `staticClient/lex.ts` — the wire boundary the DEPLOYED site runs — went on telling
 * readers the corpus has `1680` open-class stems against `8202` vacated tokens. Nothing
 * failed, because the API-parity fixture had seven cases and all seven were 200s.
 *
 * So: one declaration, imported by both the engine and the static client, plus two tests
 * that make a fifth hand-copy impossible to introduce silently —
 *
 *  * `staticVacancy.test.ts` re-measures `stemsTotal` and `tokensVacated` at `p = 1`
 *    through the real engine over the real committed corpus and asserts they equal
 *    `SWAP_SUPPLY`, and that the refusal a caller receives is this string;
 *  * the same file replays the parity fixture's `rejects` — real 400s transcribed from the
 *    live FastAPI route — and compares the numeric literals in each stack's message as a
 *    multiset, which is precisely the assertion `1680` / `8202` would have failed.
 *
 * The two stacks are NOT required to word a refusal identically (`consistent=True` against
 * `consistent = true`: each names its own language's literal). They are required to quote
 * the same corpus.
 */

/**
 * The shipped corpus's supply of real open-class words, and the demand the inconsistent
 * control makes of it.
 *
 * Both are measurements of `code/backend/src/llm_geometry/lex/data/real-mother-goose.txt`
 * through the transform itself, re-derived by `staticVacancy.test.ts` on every run and
 * pinned against the live backend by `test_api_lex.py` (`stemsTotal == 1676`,
 * `tokensVacated == 8125`).
 */
export const SWAP_SUPPLY: { readonly stems: number; readonly vacatedTokens: number } = {
  /** `|VacancyMap.stems|` — one replacement type is available per vacatable stem. */
  stems: 1676,
  /** Tokens vacated at `p = 1`: what a fresh type per OCCURRENCE would have to supply. */
  vacatedTokens: 8125,
};

/**
 * Why `mint = "swap"` cannot run under `consistent = false` (§8.3).
 *
 * Written without the `vacancy: ` prefix the engine's `Error`s carry, so the same sentence
 * can be thrown by the engine and served as an `InvalidParamError` by the static client.
 */
export const SWAP_INCONSISTENT_REFUSAL =
  "mint = 'swap' requires consistent = true — the inconsistent control needs a fresh " +
  `type per occurrence and the corpus has ${SWAP_SUPPLY.stems} open-class stems against ` +
  `${SWAP_SUPPLY.vacatedTokens} vacated tokens, so there is no supply of real words ` +
  "(architecture.md §8.3)";

/**
 * Why a map whose images are domain types has no mapped vocabulary at intermediate `p`
 * (§5.2a). A theorem, not a defect: no `p`-stable swap avoids the collision.
 */
export function noMappedVocabularyRefusal(mint: string, p: number): string {
  return (
    `mint = '${mint}' has no mapped vocabulary at p = ${p}: its replacements ` +
    `are domain types, so a vacated type can collide with an un-vacated one and the map ` +
    `is injective only at full vacancy (architecture.md §5.2a). Use p = 0 or p = 1, or ` +
    `rebuild the budget from the vacated corpus`
  );
}
