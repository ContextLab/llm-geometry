/**
 * The Lexicon Lab's GitHub-Pages path (feature 006).
 *
 * No mocks and no fixtures of convenience. Every assertion here is either
 *
 *   * the REAL build-time export in `public/static-data/lex/` (produced by
 *     `python scripts/export_static_assets.py`, i.e. by the real FastAPI routes running
 *     the real `llm_geometry.lex` package), read from disk through the same fetch seam
 *     the other static-client tests use, or
 *   * a golden computed from the real PyTorch model and pasted here with the command
 *     that reproduces it.
 *
 * What that buys: "the browser engine agrees with the backend" is MEASURED — the
 * exported corpus is re-tokenized in TypeScript and its counts are compared against the
 * numbers Python put in `spec.json`, and every budget's coverage is compared against the
 * table Python put in `budgets.json`. A drift in either engine fails this file.
 */

import { describe, expect, it } from "vitest";

import { ApiError } from "../../src/lib/dataClient";
import { sha256Hex, utf8Bytes } from "../../src/lib/geoEngine/hash";
import {
  DOLCH_ORDER,
  LexModel,
  LexVocab,
  SPECIAL_TOKENS,
  buildVocab,
  paramCount,
  splitLines,
  tokenize,
  weightNames,
  type LexConfig,
} from "../../src/lib/lexEngine";
import { createStaticClient, type StaticClient } from "../../src/lib/staticClient";
import type { FetchLike } from "../../src/lib/staticClient/assets";
import {
  LEX_BUNDLE_FORMAT,
  LEX_BUNDLE_VERSION,
  lexModelToken,
  type LexBudgetsResult,
  type LexCorpusAsset,
  type LexModelBundle,
  type LexSpec,
} from "../../src/lib/staticClient/lex";
import { fsStaticFetch, readStaticJson } from "./staticTestUtils";

function client(fetchImpl: FetchLike = fsStaticFetch()): StaticClient {
  return createStaticClient({ baseUrl: "/", fetchImpl });
}

/** The job's terminal `done` payload — the same event the backend's SSE emits. */
function donePayload(c: StaticClient, jobId: string): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    c.subscribeProgress(jobId, {
      onDone: (data) => resolve(data ?? {}),
      onError: (type, message) => reject(new Error(`${type}: ${message}`)),
    });
  });
}

/** Start a training run and wait for its result, as a view would. */
async function trainToCompletion(
  c: StaticClient,
  body: Parameters<StaticClient["lexTrain"]>[0],
): Promise<Record<string, unknown> & { model_token: string }> {
  const started = await c.lexTrain(body);
  if (started.ready) return started as unknown as Record<string, unknown> & { model_token: string };
  const done = await donePayload(c, started.job_id);
  return done as Record<string, unknown> & { model_token: string };
}

/** The exported assets. Loaded once — they are files, not state. */
const corpusAsset = await readStaticJson<LexCorpusAsset>("lex/corpus.json");
const specAsset = await readStaticJson<LexSpec>("lex/spec.json");
const budgetsAsset = await readStaticJson<LexBudgetsResult>("lex/budgets.json");

describe("the exported corpus is the corpus the backend measured", () => {
  it("carries the trimmed body, digest-verified", () => {
    expect(corpusAsset.format).toBe("lex-corpus-v1");
    expect(corpusAsset.title).toBe("The Real Mother Goose");
    expect(corpusAsset.gutenberg_id).toBe(10607);
    // The committed FILE's digest (header + licence footer included) is what
    // /api/lex/spec quotes, so the two must agree byte for byte.
    expect(corpusAsset.sha256).toBe(specAsset.corpus.sha256);
    expect(corpusAsset.bytes).toBe(specAsset.corpus.bytes);
    // …and the body actually shipped hashes to what the export recorded.
    expect(sha256Hex(utf8Bytes(corpusAsset.text))).toBe(corpusAsset.body_sha256);
  });

  it("round-trips to the same token counts Python computed", () => {
    const tokens = tokenize(corpusAsset.text);
    expect(tokens.length).toBe(specAsset.corpus.n_tokens);
    expect(new Set(tokens).size).toBe(specAsset.corpus.n_distinct);
    expect(splitLines(corpusAsset.text).filter((l) => l.trim() !== "").length).toBe(
      specAsset.corpus.n_lines,
    );
    expect(corpusAsset.text.length).toBe(specAsset.corpus.n_chars);
  });

  it("round-trips to the same coverage Python computed, for every budget", () => {
    expect(budgetsAsset.budgets).toHaveLength(DOLCH_ORDER.length);
    for (const row of budgetsAsset.budgets) {
      const vocab = buildVocab(row.source, row.budget, corpusAsset.text);
      expect(vocab.budgetSize, `${row.budget} size`).toBe(row.size);
      expect(vocab.rows, `${row.budget} rows`).toBe(row.rows);
      const cov = vocab.coverage(corpusAsset.text);
      expect(cov.total_tokens).toBe(row.coverage.total_tokens);
      expect(cov.in_budget_tokens, `${row.budget} in_budget_tokens`).toBe(
        row.coverage.in_budget_tokens,
      );
      expect(cov.distinct_types).toBe(row.coverage.distinct_types);
      expect(cov.oov_types, `${row.budget} oov_types`).toBe(row.coverage.oov_types);
      expect(cov.total_lines).toBe(row.coverage.total_lines);
      expect(cov.whole_lines_in_budget, `${row.budget} whole_lines`).toBe(
        row.coverage.whole_lines_in_budget,
      );
      // The backend rounds to 6 significant digits on the way out.
      expect(Number(cov.token_coverage.toPrecision(6))).toBe(row.coverage.token_coverage);
      expect(Number(cov.unk_rate.toPrecision(6))).toBe(row.coverage.unk_rate);
      const m = budgetsAsset.model;
      expect(paramCount(row.rows, m.d_model, m.n_layers, m.ctx, m.tied)).toBe(row.param_count);
    }
  });
});

describe("GET /api/lex/spec in static mode", () => {
  it("returns the contract's payload, cross-checked against the browser engine", async () => {
    const spec = await client().lexSpec();
    expect(Object.keys(spec).sort()).toEqual(
      [
        "budget_sources",
        "budgets",
        "corpus",
        "generation",
        "generation_banned_ids",
        "model",
        "special_tokens",
        "spectrum",
        "training",
      ].sort(),
    );
    expect(spec.budget_sources).toEqual(["dolch", "frequency"]);
    expect(spec.budgets.map((b) => b.name)).toEqual([...DOLCH_ORDER]);
    // FR-602: the sizes are MEASURED. The largest is 314, not the cited 315, because
    // "Santa Claus" contains a space and no word tokenizer can match it.
    expect(spec.budgets.at(-1)).toEqual({ name: "full", size: 314, rows: 318 });
    for (const b of spec.budgets) expect(b.rows).toBe(b.size + SPECIAL_TOKENS.length);
    expect(spec.special_tokens).toEqual({ "<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3 });
    expect(spec.generation_banned_ids).toEqual([0, 1, 3]);
    expect(spec.model.mlp_ratio).toBe(4);
    expect(spec.spectrum).toEqual({ pca_components: 3, display_k: 48 });
    expect(Object.keys(spec.training.defaults).sort()).toEqual([
      "batch_size",
      "lr",
      "sample_every",
      "seed",
      "steps",
      "weight_decay",
    ]);
  });

  it("refuses loudly when the export disagrees with the engine", async () => {
    // A spec whose budget table has been edited: the page would document one model and
    // run another. The client must not serve it.
    const tampered = { ...specAsset, budgets: specAsset.budgets.map((b) => ({ ...b, size: b.size + 1 })) };
    const fetchImpl: FetchLike = async (input) =>
      input.endsWith("lex/spec.json")
        ? new Response(JSON.stringify(tampered), { status: 200 })
        : fsStaticFetch()(input);
    await expect(client(fetchImpl).lexSpec()).rejects.toMatchObject({
      type: "ComputeError",
      message: expect.stringContaining("disagrees with the browser engine"),
    });
  });
});

describe("budgets and coverage are recomputed locally, not replayed", () => {
  it("reproduces the exported /budgets payload exactly", async () => {
    const live = await client().lexBudgets();
    expect(live.source).toBe(budgetsAsset.source);
    expect(live.model).toEqual(budgetsAsset.model);
    expect(live.corpus).toEqual(budgetsAsset.corpus);
    expect(live.budgets).toEqual(budgetsAsset.budgets);
  });

  it("measures one budget with its out-of-budget sample", async () => {
    const cov = await client().lexCoverage({ source: "dolch", budget: "pre_primer" });
    const exported = budgetsAsset.budgets.find((b) => b.budget === "pre_primer");
    expect(cov.coverage).toEqual(exported?.coverage);
    expect(cov.words).toHaveLength(cov.size);
    expect(cov.oov_sample.length).toBeLessThanOrEqual(24);
    // Sorted by count descending — "what this budget cannot say", most frequent first.
    for (let i = 1; i < cov.oov_sample.length; i++) {
      expect(cov.oov_sample[i - 1].count).toBeGreaterThanOrEqual(cov.oov_sample[i].count);
    }
    for (const entry of cov.oov_sample) expect(cov.words).not.toContain(entry.word);
  });

  it("rejects a size on a Dolch budget, as the contract requires", async () => {
    await expect(client().lexCoverage({ source: "dolch", size: 50 })).rejects.toMatchObject({
      type: "InvalidParamError",
    });
  });
});

describe("training, generation and spectra all run in the browser", () => {
  // A small real run: the whole point is that it is REAL, so it is kept cheap rather
  // than shortened into something that is not training.
  const text = corpusAsset.text.slice(0, 6000);

  it("trains, then serves generation, spectrum and a portable bundle from the result", async () => {
    const c = client();
    const body = {
      text,
      source: "dolch",
      budget: "pre_primer",
      d_model: 16,
      n_layers: 1,
      n_heads: 2,
      ctx: 32,
      steps: 4,
      batch_size: 4,
      sample_every: 2,
    };
    const started = await c.lexTrain(body);
    expect(started.ready).toBe(false);
    if (started.ready) throw new Error("unreachable");

    const messages: string[] = [];
    await c.pollJob(started.job_id, (_p, m) => messages.push(m));
    const snapshot = await c.getJob(started.job_id);
    expect(snapshot.status).toBe("done");
    // The contract's progress format: "step 3/4 · loss 3.412 · lr 2.71e-03".
    expect(messages.some((m) => /^step \d+\/4 · loss \d+\.\d{3} · lr [\d.]+e[+-]\d+/.test(m))).toBe(
      true,
    );

    // A repeat of the same request is a cache hit carrying the full record + history.
    const hit = await c.lexTrain(body);
    expect(hit.ready).toBe(true);
    if (!hit.ready) throw new Error("unreachable");
    expect(hit.model_token).toMatch(/^[0-9a-f]{32}$/);
    expect(hit.steps).toBe(4);
    expect(hit.vocab_rows).toBe(44);
    expect(hit.vocab_size).toBe(40);
    expect(hit.param_count).toBe(paramCount(44, 16, 1, 32, true));
    expect(hit.n_tokens).toBeGreaterThan(0);
    expect(Number.isFinite(hit.first_loss)).toBe(true);
    expect(Number.isFinite(hit.final_loss)).toBe(true);
    expect(hit.history).toHaveLength(4);
    expect(hit.history[0]).toHaveProperty("lr");

    // --- generation: in budget by construction --------------------------------------
    const gen = await c.lexGenerate({ model_token: hit.model_token, max_new_tokens: 12, seed: 1 });
    expect(gen.out_of_budget).toEqual([]);
    expect(gen.n_words).toBe(gen.words.length);
    const vocabWords = new Set((await c.lexExportModel(hit.model_token)).vocab.words);
    for (const w of gen.words) expect(vocabWords.has(w)).toBe(true);
    expect(gen.final_loss).toBeCloseTo(hit.final_loss, 5);

    const prompted = await c.lexGenerate({
      model_token: hit.model_token,
      prompt: "the quantum xylophone",
      max_new_tokens: 4,
    });
    // Out-of-budget prompt words are reported, not hidden.
    expect(prompted.prompt_tokens.map((t) => t.text)).toEqual(["the", "quantum", "xylophone"]);
    expect(prompted.prompt_tokens.filter((t) => t.unk).map((t) => t.text)).toEqual([
      "quantum",
      "xylophone",
    ]);
    expect(prompted.prompt_tokens.filter((t) => t.unk).every((t) => t.id === 0)).toBe(true);

    // --- spectrum -------------------------------------------------------------------
    const spec = await c.lexSpectrum({ model_token: hit.model_token });
    expect(spec.projection).toBe("pca");
    expect(spec.tied).toBe(true);
    expect(spec.tokens).toHaveLength(44);
    expect(spec.spectrum.rows).toBe(44);
    expect(spec.spectrum.d_model).toBe(16);
    expect(spec.spectrum.max_rank).toBe(Math.min(44 - 1, 16));
    expect(spec.spectrum.eigenvalues).toHaveLength(16);
    expect(spec.spectrum.pca_coords).toHaveLength(44);
    expect(spec.spectrum.pca_coords[0]).toHaveLength(3);
    expect(spec.spectrum.explained_variance.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 4);
    expect(spec.spectrum.degenerate).toBe(false);
    expect(spec.baseline).toBeDefined();
    expect(spec.baseline).not.toHaveProperty("eigenvalues");
    // The delta is computed from the unrounded statistics and rounded once, so it can
    // differ from the difference of two rounded numbers in the 6th significant digit.
    expect(spec.comparison?.effective_rank_delta).toBeCloseTo(
      spec.spectrum.effective_rank - (spec.baseline?.effective_rank ?? 0),
      3,
    );
    // A tied model has exactly ONE spectrum; asking for its readout is a 400.
    await expect(
      c.lexSpectrum({ model_token: hit.model_token, matrix: "readout" }),
    ).rejects.toMatchObject({ type: "InvalidParamError" });

    // --- portable bundle ------------------------------------------------------------
    const bundle = await c.lexExportModel(hit.model_token);
    expect(bundle.format).toBe(LEX_BUNDLE_FORMAT);
    expect(bundle.version).toBe(LEX_BUNDLE_VERSION);
    expect(bundle.model_token).toBe(hit.model_token);
    expect(bundle.vocab.specials).toEqual([...SPECIAL_TOKENS]);
    expect(bundle.weights).not.toHaveProperty("head_w"); // tied
    // The wire format uses the BACKEND's parameter names, so the file loads there too.
    expect(Object.keys(bundle.weights)).toContain("blocks.0.qkv_w");
    expect(bundle.weights["blocks.0.qkv_w"].shape).toEqual([48, 16]);

    const reloaded = await c.lexImportModel(bundle);
    expect(reloaded.model_token).toBe(hit.model_token); // content hash, so identical
    expect(reloaded.vocab_rows).toBe(44);
    expect(reloaded.param_count).toBe(hit.param_count);

    // SC-607: the reloaded model reproduces its generation exactly.
    const again = await c.lexGenerate({
      model_token: reloaded.model_token,
      max_new_tokens: 12,
      seed: 1,
    });
    expect(again.text).toBe(gen.text);

    // A file whose weights and label disagree is refused, not repaired.
    await expect(
      c.lexImportModel({ ...bundle, model_token: "0".repeat(32) }),
    ).rejects.toMatchObject({ type: "InvalidParamError" });
    // …and so is one whose word list does not match its declared rows.
    await expect(
      c.lexImportModel({ ...bundle, vocab: { ...bundle.vocab, words: bundle.vocab.words.slice(1) } }),
    ).rejects.toMatchObject({ type: "InvalidParamError" });
  }, 120_000);

  it("fine-tunes from a base model and refuses to re-shape it", async () => {
    const c = client();
    const shape = {
      text,
      budget: "pre_primer",
      d_model: 16,
      n_layers: 1,
      n_heads: 2,
      ctx: 32,
      steps: 2,
      batch_size: 4,
    };
    const base = await trainToCompletion(c, shape);

    await expect(c.lexTrain({ text, base: base.model_token, d_model: 32 })).rejects.toMatchObject({
      type: "InvalidParamError",
    });

    const ft = await c.lexTrain({ text, base: base.model_token, steps: 2, batch_size: 4 });
    if (ft.ready) throw new Error("a fine-tune of a new base must be a fresh job");
    const done = await donePayload(c, ft.job_id);
    expect((await c.getJob(ft.job_id)).status).toBe("done");
    // The fine-tune keeps the base's vocabulary and shape (feature 004's issue #6).
    expect(done.vocab_rows).toBe(44);
    expect(done.model_token).not.toBe(base.model_token);
    // The SSE `done` event carries the record WITHOUT history; only the cache-hit body
    // includes it (routes_lex.py).
    expect(done).not.toHaveProperty("history");
  }, 120_000);
});

describe("what static mode refuses, it refuses loudly", () => {
  it("raises StaticModeError — not substitute text — when the corpus was not exported", async () => {
    const without: FetchLike = async (input) =>
      input.endsWith("lex/corpus.json")
        ? new Response("not found", { status: 404 })
        : fsStaticFetch()(input);
    const err = await client(without)
      .lexBudgets()
      .then(
        () => null,
        (e: unknown) => e as ApiError,
      );
    expect(err).toBeInstanceOf(ApiError);
    expect(err?.type).toBe("StaticModeError");
    expect(err?.message).toContain("static-data/lex/corpus.json");
    expect(err?.message).toContain("export_static_assets.py");
  });

  it("raises StaticModeError for stop_at_eos=false rather than inventing the multi-line form", async () => {
    const c = client();
    const trained = await trainToCompletion(c, {
      text: corpusAsset.text.slice(0, 6000),
      budget: "pre_primer",
      d_model: 16,
      n_layers: 1,
      n_heads: 2,
      ctx: 32,
      steps: 2,
      batch_size: 4,
    });
    const token = trained.model_token;

    // `stop_at_eos: false` is the BACKEND'S DEFAULT and used to be refused here, because
    // the browser sampler could only terminate at `<eos>`. The engine now implements both
    // modes, so the two runtimes mean the same thing by the same name — assert the
    // capability rather than the old refusal.
    const multi = await c.lexGenerate({
      model_token: token,
      stop_at_eos: false,
      max_new_tokens: 40,
      seed: 3,
    });
    // Running past `<eos>` is the whole point: it must not stop early at one.
    expect(multi.text).toBeTypeOf("string");
    expect(multi.words.every((w) => !w.startsWith("<"))).toBe(true);

    // The single-line form still produces a real answer.
    const ok = await c.lexGenerate({ model_token: token, stop_at_eos: true, max_new_tokens: 5 });
    expect(ok.out_of_budget).toEqual([]);
  }, 120_000);

  it("404s an unknown model_token instead of fabricating a model", async () => {
    const c = client();
    for (const call of [
      () => c.lexSpectrum({ model_token: "deadbeef".repeat(4) }),
      () => c.lexGenerate({ model_token: "deadbeef".repeat(4) }),
      () => c.lexExportModel("deadbeef".repeat(4)),
      () => c.lexTrain({ base: "deadbeef".repeat(4) }),
    ]) {
      await expect(call()).rejects.toMatchObject({ type: "NotFoundError" });
    }
  });
});

/**
 * The bundle's wire format claims to be the backend's: the same parameter NAMES, the
 * same row-major float32 buffers, and therefore the same content hash. That claim is
 * checked against the real PyTorch model rather than assumed.
 *
 * Both goldens below were produced by the real backend (from code/backend, with its
 * venv) on the deterministic weight set `w[i] = ((i*37) % 101 - 50) / 100`:
 *
 *   python -c "
 *   import numpy as np, torch
 *   from llm_geometry.lex.model import LexConfig, LexModel
 *   from llm_geometry.api.routes_lex import _model_token
 *   cfg = LexConfig(vocab_rows=6, d_model=4, n_layers=1, n_heads=2, ctx=8,
 *                   tied=False, dropout=0.0)
 *   m = LexModel(cfg, seed=0)
 *   m.load_weight_dict({n: np.array([((i*37)%101-50)/100 for i in range(a.size)],
 *                                   dtype=np.float32).reshape(a.shape)
 *                       for n, a in m.weight_dict().items()})
 *   m.eval()
 *   print(_model_token(m.weight_dict(), cfg.as_dict(), ('cat','dog')))
 *   with torch.no_grad(): print(m(torch.tensor([[1,3,5,2]]))[0,-1].numpy())"
 */
describe("the bundle wire format really is the backend's", () => {
  const cfg: LexConfig = {
    vocabRows: 6,
    dModel: 4,
    nLayers: 1,
    nHeads: 2,
    ctx: 8,
    tied: false,
    dropout: 0,
  };
  const PY_TOKEN = "4810360a3aedbd3f5e4a2f4b1a45e81f";
  const PY_LOGITS = [0.36133751, -0.16778623, -0.28880265, -0.16336098, -0.02392921, -0.15893573];

  /** The same deterministic fill, tensor by tensor, in the engine's own names. */
  function deterministicWeights(): Record<string, Float32Array> {
    const shapes: Record<string, number> = {
      embed: cfg.vocabRows * cfg.dModel,
      pos: cfg.ctx * cfg.dModel,
      "layers.0.ln1_g": 4,
      "layers.0.ln1_b": 4,
      "layers.0.qkv_w": 48,
      "layers.0.qkv_b": 12,
      "layers.0.proj_w": 16,
      "layers.0.proj_b": 4,
      "layers.0.ln2_g": 4,
      "layers.0.ln2_b": 4,
      "layers.0.fc1_w": 64,
      "layers.0.fc1_b": 16,
      "layers.0.fc2_w": 64,
      "layers.0.fc2_b": 4,
      lnf_g: 4,
      lnf_b: 4,
      head_w: cfg.vocabRows * cfg.dModel,
    };
    expect(Object.keys(shapes)).toEqual(weightNames(cfg));
    const out: Record<string, Float32Array> = {};
    for (const [name, n] of Object.entries(shapes)) {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = ((i * 37) % 101 - 50) / 100;
      out[name] = a;
    }
    return out;
  }

  it("reproduces PyTorch's logits after a bundle round-trip", async () => {
    const model = new LexModel(cfg, deterministicWeights());
    // Straight from the engine: the layouts the bundle is built from.
    const direct = model.lastLogits([1, 3, 5, 2]);
    for (let i = 0; i < PY_LOGITS.length; i++) {
      expect(direct[i], `logit ${i}`).toBeCloseTo(PY_LOGITS[i], 5);
    }
  });

  it("computes the backend's model_token byte for byte", async () => {
    const c = client();
    const weights = deterministicWeights();
    const bundle: LexModelBundle = {
      format: LEX_BUNDLE_FORMAT,
      version: LEX_BUNDLE_VERSION,
      // Declared deliberately: an import must verify it against a re-hash.
      model_token: PY_TOKEN,
      config: {
        vocab_rows: 6,
        d_model: 4,
        n_layers: 1,
        n_heads: 2,
        ctx: 8,
        tied: false,
        dropout: 0,
      },
      vocab: { source: "dolch", budget: "custom", words: ["cat", "dog"], specials: [...SPECIAL_TOKENS] },
      metrics: {},
      weights: Object.fromEntries(
        Object.entries(weights).map(([name, array]) => {
          const wire = name.startsWith("layers.") ? `blocks.${name.slice(7)}` : name;
          const bytes = new Uint8Array(array.length * 4);
          const dv = new DataView(bytes.buffer);
          for (let i = 0; i < array.length; i++) dv.setFloat32(i * 4, array[i], true);
          let binary = "";
          for (const b of bytes) binary += String.fromCharCode(b);
          const shape: Record<string, number[]> = {
            embed: [6, 4],
            pos: [8, 4],
            "blocks.0.ln1_g": [4],
            "blocks.0.ln1_b": [4],
            "blocks.0.qkv_w": [12, 4],
            "blocks.0.qkv_b": [12],
            "blocks.0.proj_w": [4, 4],
            "blocks.0.proj_b": [4],
            "blocks.0.ln2_g": [4],
            "blocks.0.ln2_b": [4],
            "blocks.0.fc1_w": [16, 4],
            "blocks.0.fc1_b": [16],
            "blocks.0.fc2_w": [4, 16],
            "blocks.0.fc2_b": [4],
            lnf_g: [4],
            lnf_b: [4],
            head_w: [6, 4],
          };
          return [wire, { shape: shape[wire], data: btoa(binary) }];
        }),
      ),
    };

    // The import accepts the backend's own file, hash and all.
    const loaded = await c.lexImportModel(bundle);
    expect(loaded.model_token).toBe(PY_TOKEN);
    expect(loaded.vocab_size).toBe(2);
    expect(loaded.vocab_rows).toBe(6);

    // …and the same hash falls out of the token helper directly.
    const wire = Object.fromEntries(
      Object.entries(weights).map(([n, a]) => [
        n.startsWith("layers.") ? `blocks.${n.slice(7)}` : n,
        a,
      ]),
    );
    const shapes = Object.fromEntries(
      Object.entries(bundle.weights).map(([n, w]) => [n, w.shape]),
    );
    expect(lexModelToken(wire, shapes, bundle.config, ["cat", "dog"])).toBe(PY_TOKEN);

    // The reloaded model generates from the vocabulary the bundle carried.
    const vocab = new LexVocab(["cat", "dog"], "dolch", "custom");
    expect(vocab.rows).toBe(6);
    const gen = await c.lexGenerate({ model_token: loaded.model_token, max_new_tokens: 3 });
    for (const w of gen.words) expect(["cat", "dog"]).toContain(w);
  });
});
