<script lang="ts">
  import { onMount } from "svelte";

  import type { RuntimeGenerationInfo } from "./staticClient/runtimeTypes";
  import { staticExtras } from "./staticUx";

  // Live status badge for the in-browser generation runtime (transformers.js):
  // idle → loading → webgpu·q8 / wasm·q8 → (or error). Polled — the runtime
  // reports a plain snapshot and loading happens inside a lazy chunk, so a light
  // interval is the simplest honest "live update".
  //
  // The device/dtype actually in use is NAMED here, and a rung the runtime rejected
  // (thrown error, or the load-time non-degeneracy check) shows as "· fallback" with
  // the rejected rungs in the tooltip. A user on a fallback path must be able to tell:
  // the q4f16 defect was bad because it was silent, not because it fell back.
  const sc = staticExtras();
  let info = $state<RuntimeGenerationInfo | null>(sc ? sc.staticRuntimeInfo().generation : null);

  onMount(() => {
    if (!sc) return;
    const t = setInterval(() => {
      info = sc.staticRuntimeInfo().generation;
    }, 400);
    return () => clearInterval(t);
  });

  const status = $derived(info?.status ?? "idle");
  const rejected = $derived(info?.rejected ?? []);
  const text = $derived(
    status === "ready" && info?.device && info?.dtype
      ? `in-browser · ${info.device} · ${info.dtype}${rejected.length ? " · fallback" : ""}`
      : status === "loading"
        ? "in-browser · loading model…"
        : status === "error"
          ? "in-browser runtime error"
          : "in-browser · model loads on first use",
  );
  const rejectedNote = $derived(
    rejected.length
      ? ` ${rejected.join(", ")} ${rejected.length === 1 ? "was" : "were"} rejected first — ` +
        "either the session failed to build, or its output did not depend on its input " +
        "(the load-time check); see the browser console for the exact reason."
      : "",
  );
  const tip = $derived(
    status === "error"
      ? (info?.error ?? "the in-browser runtime failed to load")
      : status === "ready"
        ? `generation runs locally via transformers.js (${info?.onnx_repo ?? "ONNX export"}) on ` +
          `${info?.device}/${info?.dtype}, which passed a load-time check that its logits ` +
          `actually depend on the input — real logits, no server.${rejectedNote}`
        : "generation runs locally in your browser via transformers.js — the ONNX model downloads on first Generate",
  );
</script>

<span class="rt {status}" data-testid="static-runtime-badge" title={tip}>
  <span class="dot" aria-hidden="true"></span>{text}
</span>

<style>
  .rt {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    align-self: flex-start;
    font-size: 0.66rem;
    font-family: var(--mono);
    color: var(--text-dim);
    background: var(--bg-elev-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.14rem 0.55rem;
    cursor: help;
    white-space: nowrap;
  }
  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-dim);
    flex-shrink: 0;
  }
  .rt.ready {
    color: var(--good);
    border-color: rgba(91, 224, 176, 0.35);
  }
  .rt.ready .dot {
    background: var(--good);
    box-shadow: 0 0 6px rgba(91, 224, 176, 0.6);
  }
  .rt.loading {
    color: var(--accent);
    border-color: rgba(110, 168, 254, 0.4);
  }
  .rt.loading .dot {
    background: var(--accent);
    animation: pulse 1s ease-in-out infinite;
  }
  .rt.error {
    color: var(--bad);
    border-color: rgba(255, 122, 144, 0.4);
  }
  .rt.error .dot {
    background: var(--bad);
  }
  @keyframes pulse {
    50% {
      opacity: 0.35;
    }
  }
</style>
