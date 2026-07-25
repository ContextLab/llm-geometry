<script lang="ts">
  import { README_URL } from "./staticUx";

  // Masthead badge for the static (GitHub Pages) build: a quiet, always-visible
  // statement of what this build is, linking to the README's full-stack
  // instructions. Rendered by App.svelte only when DATA_MODE === "static", so
  // backend builds have zero layout shift.
</script>

<a
  class="static-badge"
  data-testid="static-badge"
  href={README_URL}
  target="_blank"
  rel="noopener noreferrer"
  title="This is the static demo build: everything shown is computed live in your browser or precomputed by the real backend — nothing is faked. Click for instructions to run the full stack (Python backend + frontend) locally."
>
  <span class="dot" aria-hidden="true"></span>
  static demo — computations run in your browser
</a>

<style>
  .static-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.32rem 0.8rem;
    font-size: 0.72rem;
    font-family: var(--mono);
    color: var(--text-dim);
    text-decoration: none;
    white-space: nowrap;
    transition: color 0.15s ease, border-color 0.15s ease;
  }
  .static-badge:hover {
    color: var(--text);
    border-color: var(--accent);
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent-grad);
    box-shadow: 0 0 8px rgba(110, 168, 254, 0.6);
    flex-shrink: 0;
  }
  @media (max-width: 720px) {
    .static-badge {
      white-space: normal;
      text-align: left;
      line-height: 1.35;
    }
  }
</style>
