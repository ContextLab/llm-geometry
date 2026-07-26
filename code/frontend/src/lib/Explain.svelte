<script lang="ts">
  import type { Snippet } from "svelte";

  /**
   * Collapsible explainer used across both tabs.
   *
   * Built on <details> so it works without JS and is keyboard-reachable. (Chromium can also
   * find collapsed content via in-page search; whether other engines do is not something
   * this component should assume.) The default is collapsed: the tabs stay uncluttered for
   * people who already know what they are looking at, and open for people who don't.
   */
  interface Props {
    /** Short question the reader is actually asking, e.g. "What am I looking at?" */
    title: string;
    /** Optional one-line teaser rendered next to the title while collapsed. */
    hint?: string;
    open?: boolean;
    testid?: string;
    children: Snippet;
  }
  let { title, hint = "", open = false, testid, children }: Props = $props();
</script>

<details class="explain" {open} data-testid={testid}>
  <summary>
    <span class="marker" aria-hidden="true"></span>
    <!-- A heading, so screen-reader users can find these by heading navigation — the
         standard way to skim. Without it the outline skips the explainers entirely. -->
    <span class="title" role="heading" aria-level="3">{title}</span>
    {#if hint}<span class="hint">{hint}</span>{/if}
  </summary>
  <div class="body">
    {@render children()}
  </div>
</details>

<style>
  .explain {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: linear-gradient(180deg, rgba(110, 168, 254, 0.045), transparent 60%);
  }
  summary {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    padding: 0.5rem 0.8rem;
    cursor: pointer;
    list-style: none;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text);
    border-radius: 12px;
  }
  summary::-webkit-details-marker {
    display: none;
  }
  summary:hover .title {
    color: var(--accent);
  }
  summary:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .marker {
    width: 0;
    height: 0;
    border-left: 5px solid var(--accent);
    border-top: 4px solid transparent;
    border-bottom: 4px solid transparent;
    transition: transform 0.16s ease;
    flex-shrink: 0;
    transform-origin: 30% 50%;
    align-self: center;
  }
  .explain[open] .marker {
    transform: rotate(90deg);
  }
  .hint {
    font-weight: 400;
    font-size: 0.74rem;
    color: var(--text-dim);
    min-width: 0;
  }
  /* On a phone the title and its hint share ~290px and both shrink into a ragged
     two-column block. Give the title its natural width and drop the hint below. */
  @media (max-width: 560px) {
    summary {
      flex-wrap: wrap;
    }
    .title {
      flex: 0 0 auto;
    }
    .hint {
      flex: 1 1 100%;
    }
  }
  .body {
    padding: 0 0.9rem 0.85rem;
    font-size: 0.8rem;
    line-height: 1.6;
    color: var(--text-dim);
  }
  /* Styles for the prose the caller passes in. :global is required — the content is a
     snippet compiled in the PARENT component, so Svelte's scoping hashes do not reach
     it and unscoped rules here would otherwise be pruned as unused. */
  .body :global(p) {
    margin: 0 0 0.6rem;
  }
  .body :global(p:last-child) {
    margin-bottom: 0;
  }
  .body :global(b),
  .body :global(strong) {
    color: var(--text);
    font-weight: 600;
  }
  .body :global(code) {
    font-family: var(--mono);
    font-size: 0.94em;
    color: var(--accent);
    background: var(--bg-elev-2);
    border-radius: 5px;
    padding: 0.05em 0.35em;
  }
  .body :global(ul) {
    margin: 0 0 0.6rem;
    padding-left: 1.1rem;
  }
  .body :global(li) {
    margin: 0.18rem 0;
  }
  .body :global(li::marker) {
    color: var(--accent);
  }
  .body :global(.eq) {
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--text);
    background: var(--bg-elev-2);
    border-left: 2px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 0.75rem;
    margin: 0.5rem 0 0.7rem;
    /* See InfoTab: overflow-x only scrolls if the line refuses to wrap. */
    white-space: nowrap;
    overflow-x: auto;
    line-height: 1.75;
  }
  .body :global(.eq:focus-visible) {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .body :global(.eq .note) {
    color: var(--text-dim);
  }
</style>
