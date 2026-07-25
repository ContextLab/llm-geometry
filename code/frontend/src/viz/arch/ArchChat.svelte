<script lang="ts">
  import { onDestroy } from "svelte";
  import { get } from "svelte/store";
  import {
    archModelId,
    archPrompt,
    archSystemPrompt,
    archTemperature,
    archMaxNewTokens,
  } from "../../lib/explorerStores";
  import { client, type ArchGenerateResult, type ArchGeneratedToken } from "../../lib/dataClient";
  import { showTip, hideTip } from "../../lib/tooltip";
  import { plainError } from "./archShared";
  import StaticRuntimeBadge from "../../lib/StaticRuntimeBadge.svelte";
  import { STATIC_MODE } from "../../lib/staticUx";

  // Single-turn chat (contract v1): POST /api/arch/generate with the current prompt +
  // optional system prompt; the reply renders token-by-token with per-token probability
  // + top-5 alternatives on hover. "Re-run" simply samples again (temperature > 0 ⇒ a
  // genuinely different draw from the real model).
  onDestroy(hideTip);

  let busy = $state(false);
  let error = $state("");
  let result = $state<ArchGenerateResult | null>(null);
  let forModel = ""; // replies are model-specific — drop them when the model changes

  $effect(() => {
    const m = $archModelId;
    if (result && forModel !== m) {
      result = null;
      error = "";
    }
  });

  let seq = 0;
  async function generate(): Promise<void> {
    const my = ++seq;
    const m = get(archModelId);
    busy = true;
    error = "";
    try {
      const sys = get(archSystemPrompt).trim();
      const r = await client.archGenerate({
        model_id: m,
        prompt: get(archPrompt),
        system_prompt: sys ? sys : null,
        temperature: get(archTemperature),
        max_new_tokens: get(archMaxNewTokens),
      });
      if (my !== seq) return;
      result = r;
      forModel = m;
    } catch (e) {
      if (my !== seq) return;
      error = plainError(e);
    } finally {
      if (my === seq) busy = false;
    }
  }

  // Special tokens (e.g. <|im_end|>, <|endoftext|>) are template control markers,
  // not model prose — render them as a subtle small pill (keeping the probability
  // tooltip) instead of leaking the raw marker into the reply text (F5).
  const SPECIAL_TOKEN_RE = /^<\|[^|]+\|>$/;
  function isSpecial(text: string): boolean {
    return SPECIAL_TOKEN_RE.test(text);
  }
  function specialLabel(text: string): string {
    return text.slice(2, -2); // "<|im_end|>" -> "im_end"
  }

  function tokenTip(t: ArchGeneratedToken): string {
    const alts = t.topk.texts
      .map((s, i) => `${JSON.stringify(s)} ${(t.topk.probs[i] * 100).toFixed(1)}%`)
      .join(" · ");
    const note = t.note ? ` · ⚠ ${t.note}` : "";
    return `p = ${(t.prob * 100).toFixed(1)}% · top-5: ${alts}${note}`;
  }

  // Surprise coloring: confident tokens underline cool (--accent), unlikely draws warm
  // (--bad) — the reply doubles as a per-token probability readout.
  const BAD = [255, 122, 144];
  const ACCENT = [110, 168, 254];
  function probColor(p: number): string {
    const t = Math.max(0, Math.min(1, p));
    const c = BAD.map((v, i) => Math.round(v + (ACCENT[i] - v) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }
</script>

<div class="chat">
  {#if STATIC_MODE}
    <!-- Static build: generation is fully live via transformers.js — this badge reports
         the real device/dtype ladder (webgpu·q4f16 → wasm·q8) as it loads. -->
    <StaticRuntimeBadge />
  {/if}
  <div class="knobs">
    <label class="knob">
      <span class="klabel">temperature <b>{$archTemperature.toFixed(2)}</b></span>
      <input type="range" min="0" max="1.6" step="0.05" bind:value={$archTemperature} />
    </label>
    <label class="knob">
      <span class="klabel">max tokens <b>{$archMaxNewTokens}</b></span>
      <input type="range" min="8" max="128" step="4" bind:value={$archMaxNewTokens} />
    </label>
  </div>
  <button
    class="gen"
    data-testid="arch-generate"
    disabled={busy || !$archPrompt.trim()}
    onclick={() => void generate()}
  >
    {busy ? "generating…" : result ? "↻ Re-run" : "▸ Generate reply"}
  </button>
  {#if busy}
    <div class="busyline">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      sampling from the real model…
    </div>
  {/if}
  {#if error}
    <div class="err" data-testid="arch-error">
      <span>{error}</span>
      <button class="retry" onclick={() => void generate()}>Retry</button>
    </div>
  {/if}
  {#if result}
    <div class="reply" data-testid="arch-reply">
      {#each result.tokens as t, i (i)}<span
          class="tok"
          class:special={isSpecial(t.text)}
          role="note"
          aria-label={tokenTip(t)}
          style:border-bottom-color={isSpecial(t.text) ? undefined : probColor(t.prob)}
          onmousemove={(e) => showTip(e, tokenTip(t))}
          onmouseleave={hideTip}>{isSpecial(t.text) ? specialLabel(t.text) : t.text}</span>{/each}
    </div>
    <div class="replymeta">
      {result.tokens.length} tokens · finish: {result.finish_reason} · hover a token for its
      probability + alternatives
    </div>
  {/if}
</div>

<style>
  .chat {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .knobs {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
  }
  .knob {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .klabel {
    font-size: 0.76rem;
    color: var(--text-dim);
  }
  .klabel b {
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .gen {
    width: 100%;
  }
  .gen:disabled {
    opacity: 0.45;
    cursor: default;
  }
  .busyline {
    display: flex;
    align-items: center;
    gap: 0.28rem;
    font-size: 0.76rem;
    color: var(--text-dim);
    font-family: var(--mono);
  }
  .dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1s ease-in-out infinite;
  }
  .dot:nth-child(2) {
    animation-delay: 0.18s;
  }
  .dot:nth-child(3) {
    animation-delay: 0.36s;
  }
  @keyframes pulse {
    0%,
    100% {
      opacity: 0.25;
      transform: scale(0.8);
    }
    50% {
      opacity: 1;
      transform: scale(1.15);
    }
  }
  .err {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    background: rgba(255, 122, 144, 0.1);
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.3);
    border-radius: 10px;
    padding: 0.55rem 0.7rem;
    font-size: 0.78rem;
    line-height: 1.4;
  }
  .retry {
    align-self: flex-start;
    background: transparent;
    color: var(--bad);
    border: 1px solid rgba(255, 122, 144, 0.5);
    border-radius: 8px;
    padding: 0.25rem 0.7rem;
    font-size: 0.74rem;
  }
  .reply {
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 0.7rem;
    font-size: 0.84rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 240px;
    overflow-y: auto;
  }
  .tok {
    border-bottom: 2px solid transparent;
    border-radius: 2px;
    transition: background 0.12s ease;
    cursor: help;
  }
  .tok:hover {
    background: rgba(110, 168, 254, 0.16);
  }
  .tok.special {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--text-dim);
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.06rem 0.42rem;
    margin: 0 0.18rem;
    vertical-align: middle;
    white-space: nowrap;
  }
  .tok.special:hover {
    border-color: var(--accent);
    background: rgba(110, 168, 254, 0.1);
  }
  .replymeta {
    font-size: 0.7rem;
    color: var(--text-dim);
    font-family: var(--mono);
  }
</style>
