/**
 * The two Geometry Lab vector-field modes — TypeScript port of
 * code/backend/src/llm_geometry/geo/fields.py, returning objects shaped exactly
 * like the frozen /api/geo/vector_field contract (numbers unrounded; the backend's
 * 6-significant-digit JSON rounding is a transport concern, not applied here).
 *
 * next_next: for every vocab token v hypothetically appended to the prompt, arrows
 * from points[v] toward the embedding(s) of the following-token prediction. The
 * backend runs one batched forward over all 1003 (prompt + v) sequences; because
 * the prompt prefix is identical across the batch and attention is causal, the
 * prefix hidden states / keys / values are identical too — we compute them once
 * and run only the appended position per vocab token (same math, ~50x faster).
 *
 * force: per-point field W_V·z over all vocab points (antisymmetrize uses
 * (W_V - W_V^T)/2, exactly tangent to the sphere), plus per-sequence-position
 * aggregate forces sum_{j<=i} softmax(<K z_j, Q z_i>) V z_j — literally the
 * model's attention @ v rows — with normal residual magnitudes |<f_i, z_hat_i>|.
 * The antisymmetrize toggle applies to the per-point field ONLY; aggregate forces
 * always use the real W_V. layer="full" is invalid for force mode.
 */

import { invalidParam } from "./errors";
import {
  CONTEXT_WINDOW,
  D_MODEL,
  GeoModel,
  MLP_HIDDEN,
  N_LAYERS,
  VOCAB_SIZE,
  resolveLayer,
  type LayerParam,
} from "./model";
import { argmax, argsortDesc, dot, gelu, softmaxInPlace, toNested2 } from "./tensor";

export interface FieldArrow {
  origin_index: number;
  vec: number[];
  weight: number;
}

export interface SequenceForce {
  position: number;
  vec: number[];
  normal_residual: number;
}

export interface VectorFieldResult {
  mode: "next_next" | "force";
  layer: LayerParam;
  points: number[][];
  token_ids: number[];
  arrows: FieldArrow[];
  sequence_forces: SequenceForce[] | null;
  tangent_exact: boolean;
}

/** Keep the most recent tokens, leaving `room` positions free (LM conditioning). */
export function clipPrompt(promptIds: number[], room: number): number[] {
  const maxLen = CONTEXT_WINDOW - room;
  return promptIds.length > maxLen ? promptIds.slice(-maxLen) : [...promptIds];
}

const allTokenIds = (): number[] => Array.from({ length: VOCAB_SIZE }, (_, i) => i);

interface PrefixLayer {
  k: Float64Array; // (P,3) prefix keys at this layer
  v: Float64Array; // (P,3) prefix values at this layer
  T: number; // P
}

/**
 * Hidden state of the LAST position (the appended vocab token) after layers
 * 0..upTo-1, given cached prefix keys/values per layer. Exactly the math of the
 * backend's batched _run restricted to the final causal row.
 */
function appendedHidden(
  model: GeoModel,
  prefix: PrefixLayer[],
  tokenId: number,
  position: number,
  upTo: number,
): Float64Array {
  const E = model.embedding;
  const pos = model.ws["pos_embedding"];
  const h = new Float64Array(D_MODEL);
  for (let c = 0; c < D_MODEL; c++) h[c] = E[tokenId * D_MODEL + c] + pos[position * D_MODEL + c];

  for (let l = 0; l < upTo; l++) {
    const WQ = model.layerParam(l, "W_Q");
    const WK = model.layerParam(l, "W_K");
    const WV = model.layerParam(l, "W_V");
    const WO = model.layerParam(l, "W_O");
    const q = new Float64Array(D_MODEL);
    const k = new Float64Array(D_MODEL);
    const v = new Float64Array(D_MODEL);
    for (let a = 0; a < D_MODEL; a++) {
      for (let b = 0; b < D_MODEL; b++) {
        q[a] += WQ[a * D_MODEL + b] * h[b];
        k[a] += WK[a * D_MODEL + b] * h[b];
        v[a] += WV[a * D_MODEL + b] * h[b];
      }
    }
    const P = prefix[l].T;
    const pk = prefix[l].k;
    const pv = prefix[l].v;
    const scores = new Float64Array(P + 1);
    for (let j = 0; j < P; j++) scores[j] = dot(q, pk, D_MODEL, 0, j * D_MODEL);
    scores[P] = dot(q, k, D_MODEL);
    softmaxInPlace(scores);
    const ctx = new Float64Array(D_MODEL);
    for (let j = 0; j < P; j++) {
      const a = scores[j];
      for (let c = 0; c < D_MODEL; c++) ctx[c] += a * pv[j * D_MODEL + c];
    }
    for (let c = 0; c < D_MODEL; c++) ctx[c] += scores[P] * v[c];
    for (let a = 0; a < D_MODEL; a++) {
      let s = 0;
      for (let b = 0; b < D_MODEL; b++) s += WO[a * D_MODEL + b] * ctx[b];
      h[a] += s;
    }
    // MLP block: h += gelu(h W_in + b_in) W_out + b_out
    const WIn = model.layerParam(l, "W_in");
    const bIn = model.layerParam(l, "b_in");
    const WOut = model.layerParam(l, "W_out");
    const bOut = model.layerParam(l, "b_out");
    const mlpOut = new Float64Array(D_MODEL);
    for (let j = 0; j < MLP_HIDDEN; j++) {
      let pre = bIn[j];
      for (let b = 0; b < D_MODEL; b++) pre += h[b] * WIn[b * MLP_HIDDEN + j];
      const g = gelu(pre);
      for (let c = 0; c < D_MODEL; c++) mlpOut[c] += g * WOut[j * D_MODEL + c];
    }
    for (let c = 0; c < D_MODEL; c++) h[c] += mlpOut[c] + bOut[c];
  }
  return h;
}

export function nextNextField(
  model: GeoModel,
  promptIds: number[],
  layer: LayerParam = "full",
  temperature = 0,
  topM = 1,
): VectorFieldResult {
  if (temperature === null || temperature === undefined || Number(temperature) < 0) {
    throw invalidParam(`temperature must be >= 0, got ${JSON.stringify(temperature)}`);
  }
  if (Math.trunc(Number(topM)) < 1) {
    throw invalidParam(`top_m must be >= 1, got ${JSON.stringify(topM)}`);
  }
  const layerIdx = resolveLayer(layer);
  const upTo = layerIdx + 1;

  const prompt = clipPrompt(promptIds.map((t) => Math.trunc(t)), 1);
  const E = model.embedding;
  const points = toNested2(E, VOCAB_SIZE, D_MODEL);

  // Prefix cache: run the prompt once, keep per-layer keys/values (identical for
  // every appended vocab token by causality).
  let prefix: PrefixLayer[];
  if (prompt.length > 0) {
    const acts = model.forwardSeq(prompt, upTo);
    prefix = acts.layers.map((la) => ({ k: la.k, v: la.v, T: acts.T }));
  } else {
    prefix = Array.from({ length: upTo }, () => ({
      k: new Float64Array(0),
      v: new Float64Array(0),
      T: 0,
    }));
  }
  const position = prompt.length; // the appended token's position index

  const arrows: FieldArrow[] = [];
  const T0 = Number(temperature) === 0;
  const m = Math.min(Math.trunc(Number(topM)), VOCAB_SIZE);

  for (let vTok = 0; vTok < VOCAB_SIZE; vTok++) {
    const h = appendedHidden(model, prefix, vTok, position, upTo);
    const logits = model.readoutOne(h);
    if (T0) {
      const target = argmax(logits);
      arrows.push({
        origin_index: vTok,
        vec: [
          E[target * D_MODEL] - E[vTok * D_MODEL],
          E[target * D_MODEL + 1] - E[vTok * D_MODEL + 1],
          E[target * D_MODEL + 2] - E[vTok * D_MODEL + 2],
        ],
        weight: 1,
      });
    } else {
      const t = Number(temperature);
      for (let i = 0; i < logits.length; i++) logits[i] /= t;
      softmaxInPlace(logits);
      const order = argsortDesc(logits);
      for (let rank = 0; rank < m; rank++) {
        const target = order[rank];
        // Quantize to float32 like the backend's f32 softmax before the >0 check,
        // so sub-float32 tail probabilities are dropped identically.
        const weight = Math.fround(logits[target]);
        if (weight <= 0) continue;
        arrows.push({
          origin_index: vTok,
          vec: [
            E[target * D_MODEL] - E[vTok * D_MODEL],
            E[target * D_MODEL + 1] - E[vTok * D_MODEL + 1],
            E[target * D_MODEL + 2] - E[vTok * D_MODEL + 2],
          ],
          weight,
        });
      }
    }
  }

  return {
    mode: "next_next",
    layer: layer === "full" ? "full" : layerIdx,
    points,
    token_ids: allTokenIds(),
    arrows,
    sequence_forces: null,
    tangent_exact: false,
  };
}

export function forceField(
  model: GeoModel,
  promptIds: number[],
  layer: LayerParam,
  antisymmetrize = false,
): VectorFieldResult {
  if (layer === "full") {
    throw invalidParam(
      'layer="full" is invalid for force mode: the attention force is per-layer by definition; ' +
        "choose a layer 0..3",
    );
  }
  const layerIdx = resolveLayer(layer);
  if (layerIdx < 0 || layerIdx >= N_LAYERS) {
    throw invalidParam(`layer must be in 0..${N_LAYERS - 1}`);
  }

  const E = model.embedding;
  const points = toNested2(E, VOCAB_SIZE, D_MODEL);
  const wV = model.layerParam(layerIdx, "W_V");
  const wEff = new Float64Array(9);
  if (antisymmetrize) {
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) wEff[a * 3 + b] = 0.5 * (wV[a * 3 + b] - wV[b * 3 + a]);
    }
  } else {
    for (let i = 0; i < 9; i++) wEff[i] = wV[i];
  }

  // Per-point field W_eff · z over all vocab points; weights = |vec| / max |vec|.
  const vecs = new Float64Array(VOCAB_SIZE * 3);
  const mags = new Float64Array(VOCAB_SIZE);
  let maxMag = 0;
  for (let vTok = 0; vTok < VOCAB_SIZE; vTok++) {
    for (let a = 0; a < 3; a++) {
      let s = 0;
      for (let b = 0; b < 3; b++) s += wEff[a * 3 + b] * E[vTok * 3 + b];
      vecs[vTok * 3 + a] = s;
    }
    const mag = Math.hypot(vecs[vTok * 3], vecs[vTok * 3 + 1], vecs[vTok * 3 + 2]);
    mags[vTok] = mag;
    if (mag > maxMag) maxMag = mag;
  }
  const arrows: FieldArrow[] = [];
  for (let vTok = 0; vTok < VOCAB_SIZE; vTok++) {
    arrows.push({
      origin_index: vTok,
      vec: [vecs[vTok * 3], vecs[vTok * 3 + 1], vecs[vTok * 3 + 2]],
      weight: maxMag > 0 ? mags[vTok] / maxMag : 0,
    });
  }

  // Per-sequence-position aggregate forces (always the REAL W_V, never antisymmetrized).
  //
  // Mirrors geo/fields.py: the sum is drawn anchored at the prompt token's EMBEDDING,
  // so it is projected onto the tangent plane there and the removed radial magnitude is
  // reported as normal_residual. Antisymmetrizing W_V would not help — each term
  // W_V z_j is tangent at z_j, not at the anchor.
  const sequenceForces: SequenceForce[] = [];
  const prompt = clipPrompt(promptIds.map((t) => Math.trunc(t)), 0);
  if (prompt.length > 0) {
    const acts = model.forwardSeq(prompt, layerIdx + 1);
    const tr = acts.layers[layerIdx];
    const T = acts.T;
    for (let i = 0; i < T; i++) {
      const force = [0, 0, 0];
      for (let j = 0; j <= i; j++) {
        const a = tr.attention[i * T + j];
        for (let c = 0; c < 3; c++) force[c] += a * tr.v[j * 3 + c];
      }
      // Project at the point the arrow is DRAWN at — the prompt token's embedding,
      // which is the unit-norm sphere point the client anchors to. Using the layer's
      // residual stream (hiddenIn) put the "tangent" arrows up to 59° out of plane.
      const a = prompt[i] * 3;
      const z0 = model.embedding[a];
      const z1 = model.embedding[a + 1];
      const z2 = model.embedding[a + 2];
      const zNorm = Math.hypot(z0, z1, z2);
      let radial = 0;
      if (zNorm > 1e-12) {
        const n0 = z0 / zNorm;
        const n1 = z1 / zNorm;
        const n2 = z2 / zNorm;
        radial = force[0] * n0 + force[1] * n1 + force[2] * n2;
        force[0] -= radial * n0;
        force[1] -= radial * n1;
        force[2] -= radial * n2;
      }
      sequenceForces.push({ position: i, vec: force, normal_residual: Math.abs(radial) });
    }
  }

  return {
    mode: "force",
    layer: layerIdx,
    points,
    token_ids: allTokenIds(),
    arrows,
    sequence_forces: sequenceForces,
    tangent_exact: Boolean(antisymmetrize),
  };
}
