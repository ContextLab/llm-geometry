<script lang="ts">
  import { README_URL } from "./staticUx";

  // The designed "not precomputed in the static demo" affordance (FR-203): shown
  // wherever an interaction is honest-but-unavailable in the static build. It is
  // deliberately NOT styled like an error — nothing is broken; this build just
  // doesn't carry that computation. `message` is usually the data layer's
  // StaticModeError text (plain language, names the available presets).
  interface Props {
    message: string;
    testid?: string;
    /** Smaller inline variant for sidebars / control rails. */
    compact?: boolean;
  }
  let { message, testid = "static-note", compact = false }: Props = $props();
</script>

<div class="notice" class:compact data-testid={testid}>
  <span class="chip">static demo</span>
  <span class="msg">{message}</span>
  <a class="link" href={README_URL} target="_blank" rel="noopener noreferrer">
    run the full stack ↗
  </a>
</div>

<style>
  .notice {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.45rem 0.6rem;
    background: rgba(110, 168, 254, 0.07);
    border: 1px dashed rgba(110, 168, 254, 0.45);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
    line-height: 1.5;
    color: var(--text-dim);
  }
  .notice.compact {
    padding: 0.45rem 0.6rem;
    font-size: 0.74rem;
    border-radius: 8px;
  }
  .chip {
    font-size: 0.62rem;
    font-family: var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent);
    background: rgba(110, 168, 254, 0.12);
    border: 1px solid rgba(110, 168, 254, 0.35);
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .msg {
    flex: 1 1 16rem;
    min-width: 0;
  }
  .link {
    color: var(--accent);
    font-size: 0.74rem;
    white-space: nowrap;
    text-decoration: none;
  }
  .link:hover {
    text-decoration: underline;
  }
</style>
