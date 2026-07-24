# Repo staleness audit — llm-geometry (branch 002-interactive-model-explorer)

Date: 2026-07-24. Read-only audit; no code changes made. Grounded in actual file
reads/greps; every reference below verified with file:line.

## 1. Top-level inventory + verdicts

| Entry | What it is / used by | Verdict |
|-|-|-|
| `.claude/` | 36 tracked speckit skill files + ignored `settings.local.json` | KEEP |
| `.devservers/` | pid/log dir written by `scripts/dev.sh`; gitignored, untracked | KEEP (local) |
| `.dockerignore` | Docker context excludes; line 17 `paper/` | REWRITE (drop paper line after removal) |
| `.github/workflows/ci.yml` | Real CI (backend/frontend/e2e); caches `data/processed/cache` at :43 and :104 | KEEP (update on data migration) |
| `.gitignore` | Mostly current; latex-junk block :135–143; `data/processed/cache/` :152 | REWRITE (drop latex block with paper; update cache path if migrated) |
| `.gitmodules` | Only entry = `paper/CDL-bibliography` (:1–3) | REMOVE (with submodule) |
| `.omc/` | oh-my-claudecode local state; gitignored | KEEP (local) |
| `.specify/` | Spec Kit config; `feature.json` → specs/002; constitution FILLED (v2.0.0, real principles — no placeholders); templates are stock speckit | KEEP |
| `CLAUDE.md` | Project instructions; multiple stale lines (see §2/§4) | REWRITE |
| `Dockerfile` | CURRENT: `python:3.11-slim` + Node 20, `pip install -e code/backend[test]`, `npm run build`, CMD uvicorn :8000. NOT the old 3.7 conda image (CLAUDE.md:107 claim is stale, not the file). Gap: never COPYs `data/raw/` → geo corpus falls back to Gutenberg download at runtime in containers | KEEP (fix corpus gap via migration) |
| `LICENSE` | MIT | KEEP |
| `README.md` | Mostly accurate; 3 stale spots (§3) | REWRITE |
| `code/backend/`, `code/frontend/` | The app | KEEP |
| `code/notebooks/` | Only `.gitkeep`; `demo.ipynb` already deleted historically | REMOVE |
| `code/README.md` | 1-line template about notebooks/paper figures | REMOVE (or rewrite for backend/frontend) |
| `data/raw/alice-in-wonderland.txt` | REAL input: `geo/config.py:40` `CORPUS_PATH`, sha256-checked by `geo/corpus.py`; tests+CI depend | MIGRATE-THEN-REMOVE (into package data) |
| `data/processed/cache/` | Runtime cache: `config.py:27` default (env override `LLM_GEOMETRY_CACHE_DIR` :24); CI cache path | MIGRATE-THEN-REMOVE (new default dir) |
| `data/{,raw,processed}/README.md` | Template placeholders (§3) | REMOVE with data/ |
| `docs/screenshots/issue-1/` | 4 real feature-002 screenshots | KEEP |
| `environment.yml` | Accurate dual-stack conda env (py3.11+node20, `-e ./code/backend[test]`) | KEEP |
| `notes/` | Session notes (required by user global instructions) | KEEP |
| `paper/` | Template: `main.tex:20` `\title{Template paper}`, `supplement.tex:19` same; `figs/source/` only `.gitkeep`; `admin/` template cover letter; tracked `main.pdf`/`supplement.pdf`; ignored-untracked `main.bbl`/`supplement.bbl` on disk | REMOVE (incl. submodule) |
| `paper/CDL-bibliography` | Submodule, initialized at b339d2a; used only by `main.tex:37`, `supplement.tex:51` | REMOVE |
| `project_description.md` | Source of truth per CLAUDE.md | KEEP |
| `scripts/dev.sh` | Real dev-stack launcher (ports 8000/5173, health checks) | KEEP |
| `setup.sh` | Only pulls the bibliography submodule (`cd paper/CDL-bibliography` :4) | REMOVE |
| `specs/001…, 002…` | Feature history; mention Dockerfile/data paths as historical record | KEEP (do not retro-edit) |

## 2. Reference lists (file:line) — mechanical cleanup targets

**paper/**: `.gitmodules:1-3`; `.dockerignore:17`; `README.md:115`, `README.md:122-127` (whole "Building the paper" section); `CLAUDE.md:50` (main.tex boilerplate), `:74` ("one per paper figure"), `:76` (paper layout bullet), `:100-101` (compile commands); `setup.sh:4`.

**CDL-bibliography**: `.gitmodules:1-3`; `setup.sh:4`; `README.md:125`; `CLAUDE.md:76,81-82`; `paper/main.tex:37` `\bibliography{CDL-bibliography/cdl}`; `paper/supplement.tex:51` same.

**setup.sh**: `README.md:125`; `CLAUDE.md:82` (the file itself: whole thing is submodule setup).

**compile.sh**: `README.md:126`; `CLAUDE.md:101`; files `paper/compile.sh`, `paper/admin/compile.sh` (removed with paper/).

**data/raw**: `code/backend/src/llm_geometry/geo/config.py:40` (`CORPUS_PATH = REPO_ROOT / "data" / "raw" / "alice-in-wonderland.txt"`); `code/backend/src/llm_geometry/geo/corpus.py:4` (docstring path); `CLAUDE.md:75`; `data/raw/README.md`. Corpus id string `gutenberg-11-alice-in-wonderland` also at `geo/config.py:39`, `tests/contract/test_api_geo.py:109`, `frontend/src/viz/geo/vocab.ts:8` (comment) — id can stay after migration.

**data/processed**: `code/backend/src/llm_geometry/config.py:27` (default cache dir; env override at :24); `.gitignore:152`; `.dockerignore:9`; `.github/workflows/ci.yml:43,104` (cache path; key at :44 hashes `geo/config.py`); `README.md:114,119`; `CLAUDE.md:75,104-105`; historical: `specs/001.../plan.md:36,165,171-172`, `tasks.md:28,39,49,223`, `data-model.md:103`, `specs/002.../plan.md:15`.

**code/notebooks**: `CLAUDE.md:51,74`; `README.md:112`; `code/README.md` (entire); tracked file `code/notebooks/.gitkeep`.

**Dockerfile**: `README.md:83-89`; `CLAUDE.md:94-98` (docker commands, accurate) and `:107` (STALE — claims "Python 3.7, conda"); `environment.yml:2` (comment); `code/backend/requirements.txt:3` (comment); historical: `specs/001.../quickstart.md:9,128`, `plan.md:47,83,84`, `research.md:137,145`, `tasks.md:39,153`; generic mentions in `.claude/skills/speckit-implement/SKILL.md:111,139`.

**environment.yml**: `README.md:77`; `CLAUDE.md:96`; `specs/001.../quickstart.md:30,121`, `research.md:135`. All accurate — KEEP.

**demo.ipynb / trig / sin.pdf / cos.pdf**: files already deleted from the repo (`code/notebooks/` holds only `.gitkeep`; `paper/figs/source/` only `.gitkeep`). Remaining references are doc-only: `CLAUDE.md:50-51` (demo.ipynb, sin/cos figure), `CLAUDE.md:74` (demo.ipynb = Figure 1), `CLAUDE.md:76` (trig.pdf → source/sin.pdf/cos.pdf note).

## 3. README inventory

| README | Verdict | Evidence |
|-|-|-|
| `/README.md` | mostly accurate; REWRITE 3 spots | `:112` "notebooks/ # Jupyter notebooks (paper figures)"; `:115` "paper/ # LaTeX sources, figures, bibliography submodule"; `:122-127` "## Building the paper … sh setup.sh … cd paper && sh compile.sh". Minor: `:96` "39 tests" count likely stale. |
| `/code/README.md` | 100% stale template | Entire content: "The notebooks folder contains code for replicating the analyses and figures reported in the paper. One notebook per figure will be added here." |
| `/data/README.md` | stale template | "This folder contains the raw and preprocessed data analyzed in our paper. Our experiment code and stimuli may be found [here](link)." — dead `[here](link)` placeholder |
| `/data/raw/README.md` | stale template (dir is real) | "Placeholder text: describe how raw data files are organized…" |
| `/data/processed/README.md` | stale template (dir is real) | "Placeholder text: describe how processed data files are organized." |
| `.specify/extensions/agent-context/README.md` | tooling doc | KEEP |
| `code/backend/.pytest_cache/README.md` | pytest artifact, untracked | ignore |

## 4. .specify / .gitignore / misc checks

- Constitution: `.specify/memory/constitution.md` is genuinely filled (v2.0.0, five real principles, sync-impact report) — NOT a template placeholder. No action.
- `.specify/feature.json` = `specs/002-interactive-model-explorer` (current). Templates under `.specify/templates/` are stock Spec Kit — keep.
- `.gitignore`: latex-junk block `:135-143` becomes dead with paper/ (note `*.log` there also usefully matches dev logs — keep `*.log`, drop the latex-specific ones: `*.cb*`, `*.dvi`, `*.blg`, `*.aux`, `*.fff`, `*.synctex.gz`, `*.bbl`). `data/processed/cache/` `:152` must track the cache migration.
- `paper/figs/` contains only `source/.gitkeep`; `paper/admin/` has template `cover_letter.{tex,pdf}` + `compile.sh`. `paper/main.bbl`/`supplement.bbl` exist on disk but are gitignored (confirmed via `git check-ignore`).
- No tracked `.DS_Store`/logs/pids (`git ls-files` grep clean); `code/.DS_Store` exists on disk but ignored. No symlinks anywhere (so no dead figure links). `git status` clean.
- Dockerfile gap (real bug to fix during migration): it never COPYs `data/`, so a container that exercises the Geometry tab re-downloads the corpus from Gutenberg at runtime (allowed by `corpus.py` fallback, but violates the "committed so CI never touches the network" intent). Moving the corpus into package data fixes Docker + pip installs simultaneously.
- CLAUDE.md `:107` Dockerfile note ("pins an old scientific-Python stack (Python 3.7, conda)") is factually wrong today — the Dockerfile was rewritten (git log: commit series through Jun 14). Rewrite that paragraph.
- CLAUDE.md `<!-- SPECKIT START/END -->` block (`:109-118`) is auto-managed — edit only above it.

## 5. Recommended cleanup sequence (pytest/CI green at every step)

1. **Corpus migration (first).** `git mv data/raw/alice-in-wonderland.txt code/backend/src/llm_geometry/geo/data/alice-in-wonderland.txt`; declare it package data in `code/backend/pyproject.toml`; update `geo/config.py:40` (`CORPUS_PATH = Path(__file__).parent / "data" / …`) and the `corpus.py:4` docstring. Optionally COPY nothing extra in Dockerfile — package data now ships with the install. Run `pytest -q` (geo tokenizer/corpus/contract tests). CI cache key (`ci.yml:44`) hashes `geo/config.py`, so it self-invalidates.
2. **Cache-dir migration.** Change `config.py:27` default (e.g. `REPO_ROOT/.cache/llm-geometry` or keep env override as primary); update `.gitignore:152`, `.dockerignore:9`, `ci.yml:43` and `:104` (both jobs), `README.md:114,119`, `CLAUDE.md:75,104-105`. Run backend pytest + frontend e2e (cache-dependent).
3. **Delete data/** (now empty of real inputs): `git rm data/README.md data/raw/README.md data/processed/README.md` (+ dir). No code references remain after 1–2.
4. **Remove paper/ + submodule + setup.sh**: `git submodule deinit -f paper/CDL-bibliography && git rm paper/CDL-bibliography && rm -rf .git/modules/paper` ; `git rm -r paper` ; `git rm setup.sh .gitmodules`; drop `.dockerignore:17`; trim `.gitignore:135-143` latex entries (keep `*.log`). No code/test touches paper → pytest untouched.
5. **Remove code/notebooks/**: `git rm code/notebooks/.gitkeep`; delete or rewrite `code/README.md` to describe backend/frontend.
6. **Doc rewrite**: `README.md` (drop :112, :115, :122-127; recheck ":96 39 tests" count) and `CLAUDE.md` human sections (:50-51, :74-76, :81-82, :94-107 — fix stale Dockerfile note; keep SPECKIT block untouched).
7. **Full verification**: backend `ruff check src/ tests/`, `black --check src/`, `pytest -q`; frontend `npm run check && npm run test && npm run build`; `npm run test:e2e`; `docker build -t llm-geometry .`; re-run all after any fix (per project policy).

Notes: specs/001 & 002 artifacts mention old paths as historical records — leave them; they are not executed. `environment.yml`, `scripts/dev.sh`, `docs/screenshots/`, `.specify/`, constitution: all current, keep as-is.
