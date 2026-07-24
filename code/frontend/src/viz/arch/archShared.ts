// Shared helpers for the Architecture Explorer (src/viz/arch/**).
import { ApiError, client, type ArchGraph, type ArchNode, type ArchNodeKind } from "../../lib/dataClient";

// ---------------------------------------------------------------------------
// Graph cache — /api/arch/graph is expensive on a cold model (download + traced
// forward pass, ~10–60 s), so resolved graphs are memoized per model id for the
// whole session (module scope survives view switches). Failures are never cached.
// ---------------------------------------------------------------------------

const graphCache = new Map<string, Promise<ArchGraph>>();

export function fetchArchGraph(modelId: string): Promise<ArchGraph> {
  let p = graphCache.get(modelId);
  if (!p) {
    p = client.getArchGraph(modelId).catch((e) => {
      graphCache.delete(modelId); // don't memoize failures
      throw e;
    });
    graphCache.set(modelId, p);
  }
  return p;
}

export function evictArchGraph(modelId: string): void {
  graphCache.delete(modelId);
}

// ---------------------------------------------------------------------------
// Designed, plain-language error copy (FR-107): every failure class the arch
// endpoints can produce maps to a sentence a student can act on.
// ---------------------------------------------------------------------------

export function plainError(e: unknown): string {
  if (e instanceof ApiError) {
    switch (e.type) {
      case "ModelTooLargeError":
        return `Too big for live tracing — ${e.message}`;
      case "UnsupportedModelError":
        return `That model can't be explored here — ${e.message} Pick an open-weights causal language model instead.`;
      case "NetworkError":
        return "The backend isn't reachable. Start it (sh scripts/dev.sh) and retry.";
      default:
        return e.message;
    }
  }
  return e instanceof Error ? e.message : String(e);
}

export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

export function paramCount(n: ArchNode): number {
  let total = 0;
  for (const p of n.params) total += p.shape.reduce((a, b) => a * b, 1);
  return total;
}

// One-line explainer per node kind — parameterless (functional) ops get these
// instead of a heatmap; parameterized modules show them under the header.
export const KIND_EXPLAINER: Record<ArchNodeKind, string> = {
  embedding:
    "Token embedding — a lookup table mapping each vocabulary id to its vector in the residual stream.",
  linear:
    "Learned linear projection — multiplies incoming activations by this weight matrix (plus optional bias).",
  layernorm:
    "Layer normalization — rescales the residual stream to unit scale before the next block; the learned vector is a per-channel gain.",
  rmsnorm:
    "RMS normalization — divides the residual stream by its root-mean-square; the learned vector is a per-channel gain.",
  rope:
    "Rotary position embedding — rotates each query/key pair by a position-dependent angle, so attention scores depend on relative position. No learned weights.",
  attention_softmax:
    "Attention softmax — turns the scaled Q·K scores into a causal, row-stochastic mixing distribution over earlier tokens. No learned weights.",
  residual_add:
    "Residual add — adds this block's output back into the residual stream, so every layer edits one shared running representation. No learned weights.",
  activation:
    "Elementwise nonlinearity (e.g. SiLU/GELU) inside the MLP — bends the space so stacked linear maps can compute non-linear functions. No learned weights.",
  mlp: "Feed-forward block — expands, gates, and re-projects the residual stream position-by-position.",
  lm_head:
    "Language-model head — projects the final residual stream onto the vocabulary to produce next-token logits.",
  other: "An op captured from the traced forward pass.",
};
