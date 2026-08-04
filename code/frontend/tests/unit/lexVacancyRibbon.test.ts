/**
 * The Lexicon Lab's nesting-and-stability ribbon, under BOTH minting strategies.
 *
 * The ribbon exists to show two properties of the transform (FR-711): a word is vacated
 * iff `u(stem) < p`, and its replacement is the same string at every `p` where it is
 * vacated. It reads its rows off `map.mapping.keys()` — and the KEYS DEPEND ON THE MINT
 * (`VacancyMap.mapping`'s own docstring says so):
 *
 *   * `nonce` — `lower(stem) -> nonce`;
 *   * `swap`  — `lower(type) -> lower(type)`, whole type to whole type.
 *
 * The ribbon treated them as stems unconditionally. Under `swap` that put a TYPE through
 * `vacancyU`, so a type whose stem differs (`flowers` -> `flower`) was shown a `u` that is
 * not the number the transform used, and therefore a vacated/not-vacated pattern that need
 * not match the corpus rendered six inches above it. It also filtered the rows against a
 * set of STEMS, so most inflected types were silently dropped from a table whose caption
 * calls its rows "the eligible stems of this corpus".
 *
 * Real transform, real map, real component in jsdom. Nothing here is a fixture.
 */
import { describe, expect, it } from "vitest";
import { flushSync, mount, unmount } from "svelte";

import VacancyPanel from "../../src/viz/lex/VacancyPanel.svelte";
import { LexVocab, tokenize } from "../../src/lib/lexEngine";
import {
  buildVacancyMap,
  stemAndSuffix,
  typeCounts,
  vacancyDomain,
  vacancyParams,
  vacancyU,
  vacateText,
  type VacancyMap,
} from "../../src/lib/lexEngine/vacancy";

/**
 * Long enough for every suffix class the swap builder needs (a class of one is folded into
 * the bare class, so a two-word corpus would test the merge rather than the ribbon), and
 * deliberately full of INFLECTED types — `flowers`, `singing`, `wanted` — because a type
 * that equals its own stem cannot tell the two readings apart.
 */
const TEXT = [
  // The first six words are the function words `stemAndSuffix` breaks open — `after` into
  // `aft + er`, `does` into `doe + s` — which is what the closed-class test below needs.
  "After this, always during a storm, does having shelter matter unless it rains?",
  "The little children ran through the garden and picked the flowers.",
  "The gardener planted seeds and watered them every morning.",
  "Singing birds landed on the branches while the kittens were sleeping.",
  "The farmer wanted apples, pears and berries from the orchard.",
  "Mother baked bread, cakes and puddings for the hungry travellers.",
  "The horses pulled wagons along the dusty roads towards the village.",
  "Older sailors told stories about storms, whales and distant harbours.",
  "The candles flickered as the wind rattled the shutters and windows.",
].join("\n");

const SEED = 5;
const P = 0.5;

interface Mounted {
  root: HTMLElement;
  map: VacancyMap;
  dispose: () => void;
}

function mountPanel(mint: "nonce" | "swap"): Mounted {
  const params = vacancyParams({ p: P, seed: SEED, mint });
  const tokens = tokenize(TEXT);
  const map = buildVacancyMap(
    vacancyDomain(tokens),
    vacancyParams({ seed: SEED, mint }),
    mint === "swap" ? typeCounts(tokens) : undefined,
  );
  const vacated = vacateText(TEXT, map, params);
  const words = [...new Set(tokens)].sort();
  const vocab = new LexVocab(words, "frequency", "full");

  const target = document.createElement("div");
  document.body.appendChild(target);
  const app = mount(VacancyPanel, {
    target,
    props: {
      corpusText: TEXT,
      vacatedText: vacated,
      map,
      params,
      baseVocab: vocab,
      vocab,
      condition: "consistent",
      revealAfter: 1,
      mint,
      refusal: "",
      onP: () => {},
      onSeed: () => {},
      onCondition: () => {},
      onRevealAfter: () => {},
      onProsody: () => {},
      onMint: () => {},
    },
  });
  flushSync();
  return {
    root: target,
    map,
    dispose: () => {
      unmount(app);
      target.remove();
    },
  };
}

/** The ribbon as a reader sees it: label, the printed `u`, and the five `p` cells. */
function readRibbon(root: HTMLElement): { label: string; u: string; cells: string[] }[] {
  const table = root.querySelector('[data-testid="lex-vacancy-ribbon"]');
  if (table === null) throw new Error("the ribbon did not render");
  return [...table.querySelectorAll("tbody tr")].map((tr) => {
    const cells = [...tr.querySelectorAll("td")];
    return {
      label: tr.querySelector("th")?.textContent?.trim() ?? "",
      u: cells[0]?.textContent?.trim() ?? "",
      cells: cells.slice(1).map((td) => td.textContent?.trim() ?? ""),
    };
  });
}

const P_CELLS = [0, 0.25, 0.5, 0.75, 1];

/**
 * The string the transform hashes for a map key, which is the whole point of this file.
 * Under `nonce` the key ALREADY IS the stem and must not be split again (`water` would
 * become `wat`); under `swap` the key is a type, and the vacancy decision is taken on the
 * stem the splitter takes off it.
 */
function hashedStem(key: string, mint: "nonce" | "swap"): string {
  return mint === "nonce" ? key : stemAndSuffix(key)[0];
}

describe("the nesting ribbon reads its keys under the mint that produced them", () => {
  for (const mint of ["nonce", "swap"] as const) {
    it(`prints the u the transform actually used, under mint = "${mint}"`, () => {
      const m = mountPanel(mint);
      try {
        const rows = readRibbon(m.root);
        expect(rows.length).toBeGreaterThan(1);
        for (const row of rows) {
          // `u` is a hash of the STEM under both strategies — the vacancy decision is
          // `u(stem(key)) < p` — so a swap row keyed by a type must still hash its stem.
          const stem = hashedStem(row.label.toLowerCase(), mint);
          expect(row.u, `u printed for '${row.label}'`).toBe(vacancyU(stem, SEED).toFixed(3));
        }
      } finally {
        m.dispose();
      }
    });

    it(`shows each row's own replacement and nests it, under mint = "${mint}"`, () => {
      const m = mountPanel(mint);
      try {
        for (const row of readRibbon(m.root)) {
          const key = row.label.toLowerCase();
          const image = m.map.mapping.get(key);
          expect(image, `'${key}' is not a key of the ${mint} map`).toBeDefined();
          const u = vacancyU(hashedStem(key, mint), SEED);
          row.cells.forEach((shown, i) => {
            expect(shown, `'${key}' at p = ${P_CELLS[i]}`).toBe(u < P_CELLS[i] ? image : key);
          });
        }
      } finally {
        m.dispose();
      }
    });
  }

  it("keeps inflected types as rows under swap instead of dropping them", () => {
    // The old filter tested swap's TYPE keys for membership in a set of STEMS, so
    // `flowers` (stem `flower`) could never be a row. The ribbon then claimed to span the
    // `u` range while showing only the types that happen to equal their own stem.
    const m = mountPanel("swap");
    try {
      const labels = readRibbon(m.root).map((r) => r.label.toLowerCase());
      const inflected = labels.filter((l) => stemAndSuffix(l)[1] !== "");
      expect(
        inflected.length,
        `no inflected type among ${JSON.stringify(labels)}`,
      ).toBeGreaterThan(0);
    } finally {
      m.dispose();
    }
  });

  it("paints a function word the splitter breaks open as preserved, not as open class", () => {
    // §2.2 test 1. `after -> aft + er` and `does -> doe + s`: neither stem is a function
    // word, so a stem-level classification coloured seven function words "open class, not
    // yet vacated" — words this transform will never touch at any `p`, in a panel whose
    // legend says the colours come from the real map.
    const m = mountPanel("nonce");
    try {
      const spans = [...m.root.querySelectorAll('[data-testid="lex-vacancy-corpus"] span')];
      // Svelte adds a scoping class, so the semantic class is picked out by name.
      const classOf = (word: string): string[] =>
        spans
          .filter((s) => s.textContent?.toLowerCase() === word)
          .map(
            (s) =>
              ["kept", "open", "minted", "gap"].find((c) => s.classList.contains(c)) ?? "none",
          );
      for (const word of ["after", "does", "this", "always", "during", "having", "unless"]) {
        for (const cls of classOf(word)) {
          expect(cls, `'${word}' is painted ${cls}`).toBe("kept");
        }
      }
      // The fixture must actually contain them, or the loop above proves nothing.
      expect(classOf("after").length, "the fixture's `After` is not on screen").toBe(1);
      expect(classOf("does").length, "the fixture's `does` is not on screen").toBe(1);
    } finally {
      m.dispose();
    }
  });

  it("names its rows for what they are: stems under nonce, types under swap", () => {
    // The header is a claim about the column beneath it, and under swap "stem" was wrong.
    for (const [mint, expected] of [
      ["nonce", "stem"],
      ["swap", "type"],
    ] as const) {
      const m = mountPanel(mint);
      try {
        const head = m.root
          .querySelector('[data-testid="lex-vacancy-ribbon"] thead th')
          ?.textContent?.trim();
        expect(head, `the ribbon's first column header under ${mint}`).toBe(expected);
      } finally {
        m.dispose();
      }
    }
  });
});
