# 006 — Sourcing a public-domain "Seuss-like" children's-verse corpus

Agent report. All figures below were produced by **actually fetching** each Project Gutenberg
plain-text file and counting; nothing here is estimated or recalled. Fetches performed
2026-08-04 (UTC). Dr. Seuss's own works are under copyright and are **not** used anywhere.

---

## 0. The constraint that actually decides this

`code/backend/src/llm_geometry/geo/config.py` sets:

```
VOCAB_WORDS = 1000  # word/punctuation types drawn from the corpus by frequency
VOCAB_SIZE  = 1003
```

and `tokenizer.py:106-110` **hard-fails** if the corpus has fewer than 1000 distinct types:

```python
words = [w for w, _ in ranked[:VOCAB_WORDS]]
if len(words) < VOCAB_WORDS:
    raise InvalidParamError(f"Corpus has only {len(words)} distinct token types; ...")
```

The split is `tokenizer.py:53`:

```python
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|[0-9]+|[^\sa-z0-9]")
```

— punctuation marks are **separate tokens**. So two things follow:

1. Any candidate must have **≥ 1000 distinct types**. This immediately eliminates the
   short Lear books (`13646` A Book of Nonsense: 988 types) and Struwwelpeter
   (`12116`: 738 types), which are otherwise the most "Seussy" texts in existence.
2. The number that matters operationally is not top-315 coverage but **top-1000
   coverage** — the fraction of corpus tokens the model can represent at all. Everything
   outside it becomes `<unk>`. I report both.

I computed every statistic under **two** tokenizations: `W` = words only
(`[a-z]+(?:'[a-z]+)*`, what "word types" normally means) and `G` = the project's exact
GeoTokenizer regex above (punctuation included).

---

## 1. Verified candidates

Every URL below was fetched and returned **HTTP 200**. No 404s were encountered among the
recommended set. `bytes` = full file including the Gutenberg header/footer.

| # | Title | Author / editor | Pub. date (source) | URL (verified 200) | bytes |
|-|-|-|-|-|-|
| 10607 | The Real Mother Goose | traditional/anon.; illus. Blanche Fisher Wright | **1916** (printed on the title page inside the file) | `https://www.gutenberg.org/ebooks/10607.txt.utf-8` | 110,445 |
| 38562 | The Big Book of Nursery Rhymes | ed. Walter Jerrold (1865–1929); illus. Charles Robinson (1870–1937) | **1903** (no year in the file; Internet Archive source scan `bigbookofnursery00jerr2`, "London : Blackie", year 1903) | `https://www.gutenberg.org/ebooks/38562.txt.utf-8` | 137,835 |
| 39784 | Mother Goose's Nursery Rhymes: Alphabets, Rhymes, Tales, and Jingles | various; illus. Gilbert/Tenniel/Crane | **1877** (title page inside the file: "London GEORGE ROUTLEDGE AND SONS … 1877") | `https://www.gutenberg.org/cache/epub/39784/pg39784.txt` | 143,226 |
| 20511 | The Little Mother Goose | illus. Jessie Willcox Smith | 1912/1914/1918 (copyright notices inside the file) | `https://www.gutenberg.org/cache/epub/20511/pg20511.txt` | 105,760* |
| 77549 | Rhymes for the Nursery | Ann Taylor (1782–1866) & Jane Taylor (1783–1824) | **1831**, 23rd edn (title page inside the file: "PRINTED FOR HARVEY AND DARTON … 1831") | `https://www.gutenberg.org/ebooks/77549.txt.utf-8` | 115,068 |
| 25609 | A Child's Garden of Verses | Robert Louis Stevenson (1850–**1894**) | first pub. 1885 | `https://www.gutenberg.org/ebooks/25609.txt.utf-8` | 82,339 |
| 19722 | A Child's Garden of Verses | Robert Louis Stevenson (1850–**1894**) | first pub. 1885 | `https://www.gutenberg.org/cache/epub/19722/pg19722.txt` | 61,127 |
| 13650 | Nonsense Books (Lear omnibus) | Edward Lear (1812–**1888**) | 1846–1877 collected | `https://www.gutenberg.org/cache/epub/13650/pg13650.txt` | 214,931 |
| 76703 | Sing-Song: A Nursery Rhyme Book | Christina Rossetti (1830–**1894**) | first pub. 1872 | `https://www.gutenberg.org/cache/epub/76703/pg76703.txt` | 82,569* |
| 32415 | The Nursery Rhymes of England | ed. J. O. Halliwell-Phillipps (1820–1889) | 1840s | `https://www.gutenberg.org/cache/epub/32415/pg32415.txt` | 298,295* |
| 36685 | National Rhymes of the Nursery | intro. George Saintsbury (1845–1933) | c. 1895 | `https://www.gutenberg.org/cache/epub/36685/pg36685.txt` | 189,071* |
| 22014 | The Peter Patter Book of Nursery Rhymes | Leroy F. Jackson (1881–1958) | 1918 | `https://www.gutenberg.org/cache/epub/22014/pg22014.txt` | 59,507* |
| 75578 | Poems of Childhood | Eugene Field (1850–**1895**) | 1904 collected | `https://www.gutenberg.org/cache/epub/75578/pg75578.txt` | 151,445* |
| 12116 | Struwwelpeter | Heinrich Hoffmann (1809–**1894**) | 1845 (Eng. trans.) | `https://www.gutenberg.org/cache/epub/12116/pg12116.txt` | 14,661* |

\* These sizes are **body bytes after stripping the Gutenberg header/footer**, taken from my
first pass; the two recommended entries (10607, 38562) plus 77549 and 25609 have full-file
byte counts and sha256s recorded below, which is what a `CORPUS_SHA256` pin needs.

### URL-variant warning (real, and it bit me)

Project Gutenberg serves **different bytes** at the three URL shapes for older ebooks:

```
https://www.gutenberg.org/files/10607/10607-0.txt      200  90,797 bytes  sha256 eb4c54f9…  (2004-era file, 20,217 G-tokens in body)
https://www.gutenberg.org/ebooks/10607.txt.utf-8       200 110,445 bytes  sha256 d514f0fd…  (regenerated, 20,201 G-tokens in body)
https://www.gutenberg.org/cache/epub/10607/pg10607.txt 200 110,445 bytes  sha256 d514f0fd…  (identical to the above)
```

`config.py` currently lists `files/11/11-0.txt` first. For **10607** the `files/…-0.txt`
variant is a *different file* with a slightly different body. For **38562** the `files/`
variant is `38562-8.txt` (ISO-8859-1, 137,680 bytes) — again not byte-identical to the
UTF-8 one. Pin `ebooks/<id>.txt.utf-8` with `cache/epub/<id>/pg<id>.txt` as the fallback
(byte-identical), and record the sha256 of that pair — not the `files/` one.

Recorded 2026-08-04:

```
10607  ebooks/10607.txt.utf-8  110,445 B  sha256 d514f0fd2cd40967eb6cf35b140a6cddc11200126e07d76603fae3f88bf1e0ab
38562  ebooks/38562.txt.utf-8  137,835 B  sha256 0aa1f82c8a96fe3e8f04cd4e24c5de7efe20f07eb02bc9f63d27b2c7dfb9d8cd
77549  ebooks/77549.txt.utf-8  115,068 B  sha256 3ad9b9e416a70f0ba6c9bef4076ad9f9dde9d6801c85ced220c70936bcb1a953
25609  ebooks/25609.txt.utf-8   82,339 B  sha256 d00c04f2c07ee5cbe14076058b4ce4853983d57d91ecd9155af475f94289fafc
```

PG regenerates these cache files occasionally, so the sha256 pins the *committed* file (as
the Alice pattern already does — the download path is only a fallback).

---

## 2. Type/token statistics (measured, not estimated)

Header/footer stripped at `*** START OF THE PROJECT GUTENBERG` / `*** END OF …`, exactly as
`corpus.py:76-91` does. `W…` = words-only tokenization, `G…` = project GeoTokenizer.

| id | title | W tokens | W types | W top-315 | W top-1000 | G tokens | G types | G top-315 | **G top-1000** |
|-|-|-|-|-|-|-|-|-|-|
| **11** | *Alice (current baseline)* | 26,776 | 2,635 | 0.772 | 0.917 | 34,646 | 2,654 | 0.818 | **0.935** |
| **10607** | **The Real Mother Goose** | 16,302 | 2,172 | 0.710 | 0.907 | **20,201** | **2,196** | 0.760 | **0.923** |
| 76703 | Sing-Song (Rossetti) | 6,025 | 1,447 | 0.724 | 0.926 | 7,289 | 1,478 | 0.761 | 0.934 |
| 77549 | Rhymes for the Nursery | 12,328 | 1,977 | **0.755** | 0.921 | 15,284 | 2,081 | 0.787 | 0.929 |
| 19722 | A Child's Garden of Verses | 6,589 | 1,587 | 0.746 | 0.911 | 7,929 | 1,609 | 0.782 | 0.923 |
| **38562** | **The Big Book of Nursery Rhymes** | 18,500 | 2,540 | 0.709 | 0.889 | **23,896** | **2,560** | 0.769 | **0.912** |
| 39784 | Mother Goose's Nursery Rhymes | 18,092 | 2,529 | 0.717 | 0.891 | 23,737 | 2,702 | 0.768 | 0.905 |
| 25609 | A Child's Garden of Verses | 9,100 | 1,986 | 0.727 | 0.892 | 11,077 | 2,074 | 0.761 | 0.903 |
| 13647 | Nonsense Songs (Lear) | 11,476 | 2,371 | 0.713 | 0.878 | 14,486 | 2,398 | 0.764 | 0.900 |
| 20511 | The Little Mother Goose | 15,816 | 2,389 | 0.702 | 0.887 | 20,538 | 2,570 | 0.745 | 0.897 |
| 22014 | Peter Patter Book | 9,120 | 2,062 | 0.688 | 0.884 | 11,178 | 2,149 | 0.727 | 0.896 |
| 2670 | Love-Songs of Childhood (Field) | 10,795 | 2,290 | 0.716 | 0.881 | 13,302 | 2,304 | 0.764 | 0.902 |
| 36685 | National Rhymes of the Nursery | 27,807 | 3,802 | 0.676 | 0.846 | 37,620 | 3,843 | 0.752 | 0.883 |
| 75578 | Poems of Childhood (Field) | 23,003 | 3,794 | 0.681 | 0.842 | 28,580 | 3,913 | 0.734 | 0.867 |
| 13650 | Nonsense Books (Lear omnibus) | 29,682 | 4,635 | 0.682 | 0.822 | 37,472 | 4,683 | 0.741 | 0.856 |
| 32415 | The Nursery Rhymes of England | 45,356 | 5,726 | 0.668 | 0.821 | 59,450 | 6,068 | 0.725 | 0.848 |

**Honest finding that cuts against the premise:** *no* children's-verse corpus beats Alice on
small-vocabulary coverage at the 315-type level (Alice W-315 = 0.772; the best verse book is
77549 at 0.755). Prose repeats function words (`said`, `she`, `it`, `the`) far more densely
than verse does; verse packs content words into every line. What verse **does** win on is
*structural* repetition — whole repeated lines and refrains — which is the property that
actually makes a Seuss-like model interesting:

| id | non-blank lines | distinct lines | **exact-repeat line fraction** |
|-|-|-|-|
| 11 Alice | 2,494 | 2,474 | **0.008** |
| 10607 Real Mother Goose | 3,075 | 2,795 | **0.091** |
| 25609 A Child's Garden of Verses | 1,593 | 1,439 | 0.097 |
| 39784 Mother Goose's NR | 3,204 | 2,787 | 0.130 |
| 13650 Lear Nonsense Books | 4,547 | 3,796 | 0.165 |
| 38562 Big Book of NR | 3,317 | 2,747 | 0.172† |

† 38562's figure is partly inflated by its alphabetical index of first lines.

10607's most-repeated lines are cumulative-rhyme refrains — exactly the Seuss device:
`('When she came back', 11)`, `('That lay in the house that Jack built.', 10)`,
`('That ate the malt', 9)`, `('Dance over my Lady Lee;', 9)`, `('That killed the rat,', 8)`,
`('That worried the cat,', 7)`, `('Hot-cross Buns!', 6)`.
Alice, for comparison, has essentially no repeated lines (0.8%).

### Text cleanliness (matters — the file is committed verbatim)

| id | `[Illustration` | `\|` (ASCII tables) | "Transcriber" notes | `_` italics |
|-|-|-|-|-|
| **10607** | **0** | **0** | **0** | 60 |
| 38562 | 0 | 165 | 0 | 34 |
| 11 (baseline) | 1 | 0 | 0 | 440 |
| 39784 | 304 | — | yes | — |
| 20511 | 81 | — | yes | — |
| 36685 | 67 | yes (publisher ad boxes) | — | — |

**10607 is the only large candidate with zero illustration markers, zero ASCII tables and
zero transcriber notes.** 38562's 165 pipes are a single ASCII-art grid for "A Apple Pie" —
enough to push `|` into its top-25 token types, which would waste a vocabulary slot.

---

## 3. Public-domain reasoning

US public-domain status as of **2026-08-03**: every work *published* in the US before
**1931** is in the public domain (the 95-year term; 1930 publications entered the public
domain on 2026-01-01). Separately, works whose author died before 1956 are PD in life+70
jurisdictions.

- **10607, The Real Mother Goose (1916)** — published 1916, 15 years before the 1931 cutoff.
  The rhymes themselves are anonymous traditional verse, centuries old. Blanche Fisher
  Wright contributed only illustrations, which are not part of the plain-text file.
  Project Gutenberg's own record states **"Copyright — Public domain in the USA."**
- **38562, The Big Book of Nursery Rhymes (Blackie, 1903)** — underlying scan
  `bigbookofnursery00jerr2`, publisher "London : Blackie", year **1903**, per the Internet
  Archive advancedsearch API (there is also an `…00jerr` printing dated 1920). Editor Walter
  Jerrold d. 1929, illustrator Charles Robinson d. 1937 — both >70 years dead. PG record:
  **"Public domain in the USA."** Caveat I must flag: the transcription itself carries **no
  printed year**; the 1903 date comes from the IA source record, not from the text.
- **77549, Rhymes for the Nursery (1831)** — the file's own title page reads "PRINTED FOR
  HARVEY AND DARTON … 1831", 23rd edition. Authors Ann Taylor d. 1866, Jane Taylor d. 1824.
  PG record: "Public domain in the USA."
- **25609 / 19722, A Child's Garden of Verses** — Stevenson d. 1894; first published 1885.
- **13650, Nonsense Books** — Lear d. 1888.
- **76703, Sing-Song** — Rossetti d. 1894.

All of these are PD in the US on both the publication-date and the author-death test. The
only residual risk anywhere is an *editorial apparatus* copyright on a 20th-century
compilation (introductions, arrangement) — which is why I prefer 10607, whose only
non-traditional content is a list of rhyme titles.

---

## 4. Recommendation

### PRIMARY: Project Gutenberg #10607 — *The Real Mother Goose* (1916)

```
corpus id : gutenberg-10607-the-real-mother-goose
url       : https://www.gutenberg.org/ebooks/10607.txt.utf-8
fallback  : https://www.gutenberg.org/cache/epub/10607/pg10607.txt   (byte-identical)
bytes     : 110,445        sha256 d514f0fd2cd40967eb6cf35b140a6cddc11200126e07d76603fae3f88bf1e0ab
body      : 20,201 GeoTokenizer tokens / 2,196 distinct types
coverage  : top-1000 types cover 92.3% of tokens (Alice baseline: 93.5%)
            top-315  types cover 76.0%
repetition: 9.1% of non-blank lines are exact repeats (Alice: 0.8%)
```

Reasoning:

1. **It passes the hard gate with room to spare** — 2,196 types ≥ 1,000, and top-1000
   coverage of 92.3% is within 1.2 points of Alice, i.e. the UNK rate barely moves. Every
   other verse corpus with >20k tokens is worse on this metric.
2. **Highest structural repetition per unit of vocabulary.** The cumulative rhymes ("The
   House That Jack Built", "London Bridge", "Hot-Cross Buns") give the tiny `d_model=3`
   model something it can genuinely learn and a viewer can genuinely see in the vector field.
3. **Strong regular meter throughout** — nursery rhyme is the closest public-domain relative
   of Seuss's anapestic tetrameter and refrain-driven structure.
4. **Cleanest text of every candidate examined**: 0 `[Illustration]` markers, 0 ASCII tables,
   0 transcriber notes. Ships verbatim without a cleaning step, exactly as Alice does.
5. **Right size.** 20,201 tokens clears the >20k bar. (3,147 of those are the front "A LIST
   OF THE RHYMES" — which is itself rhyme first-lines, so it reinforces rather than dilutes;
   Alice likewise keeps its Contents block after the marker strip. Body-only is 17,054.)

### BACKUP: Project Gutenberg #38562 — *The Big Book of Nursery Rhymes* (Blackie, 1903)

```
url    : https://www.gutenberg.org/ebooks/38562.txt.utf-8
bytes  : 137,835   sha256 0aa1f82c8a96fe3e8f04cd4e24c5de7efe20f07eb02bc9f63d27b2c7dfb9d8cd
body   : 23,896 GeoTokenizer tokens / 2,560 types, top-1000 coverage 91.2%
```

Reasoning: 18% more tokens than 10607 at only 1.1 points lower top-1000 coverage, the same
traditional-rhyme register, and the second-cleanest text (no illustration markers, no
transcriber notes). Two known blemishes: a ~780-token prose introduction by Walter Jerrold,
and the 165-pipe "A Apple Pie" ASCII grid, which would occupy a vocabulary slot with `|`. A
three-line strip (drop lines matching `^\s*[+|]`) fixes the second; the intro is harmless.

### If more data is wanted later

Concatenating **10607 + 38562** gives **44,097 tokens / 3,041 types / top-1000 coverage
0.9046**. The two books share 1,715 types (78% of 10607's vocabulary), because they contain
many of the *same traditional rhymes in variant wordings* — near-duplicate text, which is
excellent signal for a tiny model. Adding a third Mother Goose (20511) pushes it to 64,635
tokens but drops coverage to 0.892; I would not go past two.

### Rejected, with reasons

| Candidate | Why not |
|-|-|
| Lear, *A Book of Nonsense* (13646) | 988 types — **fails the ≥1000 gate**. Also only 4,688 tokens. |
| Lear, *Nonsense Books* omnibus (13650) | 37,472 tokens but 4,683 types; top-1000 coverage only 0.856 → ~14% UNK. Every limerick names a new place, so the vocabulary tail is enormous. |
| Stevenson, *A Child's Garden of Verses* (25609 / 19722 / 136) | Best-in-class meter, but the largest edition is 11,077 tokens — half the target. Use as a *fine-tuning* corpus, not a training one. |
| Rossetti, *Sing-Song* (76703) | Excellent coverage (0.934) and register, but only 7,289 tokens. |
| Taylor, *Rhymes for the Nursery* (77549) | **Best word-level small-vocab coverage of any verse candidate** (W-315 = 0.755, W-1000 = 0.921) and coherent single-authorial voice, but 15,284 tokens — below the >20k bar — and carries `[Illustration]` markers. Strongest runner-up if the size bar is relaxed. |
| *Mother Goose's Nursery Rhymes* (39784) | Good size/coverage, but 304 `[Illustration]` markers, transcriber's notes, a long page-numbered index, and prose *Tales* mixed with the verse. |
| *The Nursery Rhymes of England* (32415), *National Rhymes* (36685) | Largest, but scholarly apparatus and publisher ad-pages; coverage collapses to 0.848 / 0.883. |
| Struwwelpeter (12116) | 738 types — fails the gate. |
| Carroll, *Through the Looking-Glass* (12) | 31k tokens, but it is prose with embedded verse — same register as the Alice we already ship, so it adds nothing distinctive. |

---

## 5. Gutenberg licence / attribution for redistributing the text in this repo

Direct quotes from the licence block in the fetched `pg10607.txt`:

> **1.C.** … "Nearly all the individual works in the collection are in the public domain in
> the United States. If an individual work is unprotected by copyright law in the United
> States and you are located in the United States, we do not claim a right to prevent you
> from copying, distributing, performing, displaying or creating derivative works based on
> the work **as long as all references to Project Gutenberg are removed**."

> **1.E.2.** "If an individual Project Gutenberg electronic work is derived from texts not
> protected by U.S. copyright law … the work can be copied and distributed to anyone in the
> United States without paying any fees or charges. **If you are redistributing or providing
> access to a work with the phrase "Project Gutenberg" associated with or appearing on the
> work, you must comply either with the requirements of paragraphs 1.E.1 through 1.E.7** or
> obtain permission …"

> **1.E.1.** "The following sentence … must appear prominently whenever any copy of a
> Project Gutenberg work … is accessed, displayed, performed, viewed, copied or
> distributed: *This eBook is for the use of anyone anywhere in the United States and most
> other parts of the world at no cost and with almost no restrictions whatsoever. You may
> copy it, give it away or re-use it under the terms of the Project Gutenberg™ License
> included with this eBook or online at www.gutenberg.org.*"

> **1.E.4.** "Do not unlink or detach or remove the full Project Gutenberg License terms
> from this work, or any files containing a part of this work …"

> **1.E.7.** "Do not charge a fee for access to, viewing, displaying, performing, copying or
> distributing any Project Gutenberg works unless you comply with paragraph 1.E.8 or 1.E.9."

Project Gutenberg's catalogue record for #10607 and #38562 both state, verbatim:
**"Copyright — Public domain in the USA."**

**Practical consequence for this repository.** There are exactly two compliant options, and
the project already uses the second:

1. **Commit the file whole, header and licence footer intact.** Then 1.E.1/1.E.4 are
   satisfied by the file itself. This is what `code/backend/src/llm_geometry/geo/data/
   alice-in-wonderland.txt` does today — the raw download is committed, and
   `corpus.py:load_corpus_text()` strips the markers *at read time*, not on disk. **Keep
   doing this.** It is the lowest-risk path and needs no new attribution machinery.
2. Strip the PG header/footer on disk. Then, per 1.C, all references to "Project Gutenberg"
   must be removed from the redistributed text — and at that point the underlying 1916 work
   is unencumbered public domain and PG imposes nothing further. But you must then *not*
   label the committed file "Project Gutenberg", which conflicts with wanting a traceable
   provenance comment.

Either way the project must not charge for access to a file bearing the PG trademark
(1.E.7 — not an issue here) and should not silently modify the text.

**Recommended attribution text** (mirroring what `config.py` already does for Alice):

```python
CORPUS_ID = "gutenberg-10607-the-real-mother-goose"
# The Real Mother Goose (1916), traditional English nursery rhymes, illustrated by
# Blanche Fisher Wright. Project Gutenberg ebook #10607 — public domain in the USA
# (published 1916; all US works published before 1931 are in the public domain).
# The raw download is committed VERBATIM, Project Gutenberg header and licence footer
# intact, to satisfy PG licence §§1.E.1/1.E.4; the markers are stripped at read time by
# geo/corpus.py::load_corpus_text().
CORPUS_URLS = (
    "https://www.gutenberg.org/ebooks/10607.txt.utf-8",
    "https://www.gutenberg.org/cache/epub/10607/pg10607.txt",
)
CORPUS_SHA256 = "d514f0fd2cd40967eb6cf35b140a6cddc11200126e07d76603fae3f88bf1e0ab"
```

Note the URL order differs from Alice's: for 10607 the `files/10607/10607-0.txt` form serves
**different bytes** (90,797 B, sha256 `eb4c54f9…`) and would fail the integrity check, so it
must not appear in `CORPUS_URLS`.

---

## 6. Reproducing these numbers

Working scripts were kept out of the repo (scratchpad only). The measurement is four lines:

```python
_N   = str.maketrans({"’":"'", "‘":"'", "“":'"', "”":'"', "\xa0":" "})
GEO  = re.compile(r"[a-z]+(?:'[a-z]+)*|[0-9]+|[^\sa-z0-9]")   # == tokenizer.py:53
body = raw[raw.find("\n", raw.find("*** START OF THE PROJECT GUTENBERG"))+1
           : raw.find("*** END OF THE PROJECT GUTENBERG")].strip()
c    = collections.Counter(GEO.findall(body.lower().translate(_N)))
```

then `sum(v for _, v in c.most_common(1000)) / sum(c.values())` is the top-1000 coverage
column. If this corpus swap lands, that assertion belongs in a real test — the same way
`tests/e2e/docs.spec.ts` pins documented constants.
