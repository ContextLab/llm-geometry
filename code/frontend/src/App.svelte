<script lang="ts">
  import ArchitectureExplorer from "./viz/arch/ArchitectureExplorer.svelte";
  import GeometryLab from "./viz/geo/GeometryLab.svelte";
  import InfoTab from "./viz/info/InfoTab.svelte";
  import Tooltip from "./lib/Tooltip.svelte";
  import StaticBadge from "./lib/StaticBadge.svelte";
  import { STATIC_MODE } from "./lib/staticUx";
  import { view, type View } from "./lib/stores";

  // Two tabs, each owning its own controls (feature 004 removed the three
  // embedding-geometry views and the shared control sidebar they needed).
  const tabs: { id: View; label: string; blurb: string }[] = [
    {
      id: "architecture",
      label: "Architecture",
      blurb: "Watch a real open-weights transformer run your prompt, op by op.",
    },
    {
      id: "geometry",
      label: "Geometry",
      blurb: "A 3-D transformer you can read, edit, and train on the sphere it lives on.",
    },
    {
      id: "info",
      label: "Info",
      blurb:
        "What each view shows, the mathematics behind it, what you can change — and what is not claimed.",
    },
  ];

  const active = $derived(tabs.find((t) => t.id === $view) ?? tabs[0]);

  // Whether this browser has ever opened the Info tab. Persisted, because the pointer
  // is for first-time visitors and would be nagging on every later visit. A blocked or
  // full localStorage must not break the shell, so both accesses are guarded.
  const SEEN_INFO_KEY = "llm-geometry:seen-info";
  let seenInfo = $state(read());

  function read(): boolean {
    try {
      return localStorage.getItem(SEEN_INFO_KEY) === "1";
    } catch {
      return false;
    }
  }

  $effect(() => {
    if ($view !== "info" || seenInfo) return;
    seenInfo = true;
    try {
      localStorage.setItem(SEEN_INFO_KEY, "1");
    } catch {
      // private mode / quota — the pointer simply returns next session
    }
  });
</script>

<div class="app">
  <header class="masthead">
    <div class="brand">
      <span class="logo"></span>
      <div>
        <h1>llm-geometry</h1>
        <p>Explore the geometry of a transformer's embedding space.</p>
      </div>
    </div>
    {#if STATIC_MODE}<StaticBadge />{/if}
  </header>

  <main class="main">
    <nav class="tabs" data-testid="view-tabs">
      {#each tabs as t (t.id)}
        <button
          class:active={$view === t.id}
          onclick={() => view.set(t.id)}
          data-testid={`tab-${t.id}`}
        >{t.label}</button>
      {/each}
    </nav>
    <div class="blurbrow">
      <p class="blurb">{active.blurb}</p>
      <!-- The documentation is worthless if the person who needs it never finds it. The
           in-tab explainers are collapsed and "Info" is one generic word in a tab strip,
           so a first-time visitor gets one unmissable pointer — which retires itself the
           moment they have actually read the Info tab. -->
      {#if $view !== "info" && !seenInfo}
        <button class="firsttime" data-testid="info-pointer" onclick={() => view.set("info")}>
          New here? Start with <b>Info</b> →
        </button>
      {/if}
    </div>

    {#if $view === "architecture"}
      <ArchitectureExplorer />
    {:else if $view === "geometry"}
      <GeometryLab />
    {:else}
      <InfoTab />
    {/if}
  </main>

  <Tooltip />
</div>

<style>
  .app { max-width: 1180px; margin: 0 auto; padding: 1.6rem; }
  .masthead { display: flex; align-items: center; justify-content: space-between; gap: 0.9rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
  .brand { display: flex; align-items: center; gap: 0.9rem; }
  .logo { width: 38px; height: 38px; border-radius: 11px; background: var(--accent-grad); box-shadow: 0 0 22px rgba(110,168,254,0.5); }
  h1 { margin: 0; font-size: 1.4rem; letter-spacing: -0.01em; }
  .brand p { margin: 0.1rem 0 0; color: var(--text-dim); font-size: 0.86rem; }
  .main { display: flex; flex-direction: column; gap: 0.75rem; min-width: 0; }
  .blurb { margin: 0; color: var(--text-dim); font-size: 0.82rem; }
  .blurbrow {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.8rem;
    flex-wrap: wrap;
  }
  .firsttime {
    flex-shrink: 0;
    background: rgba(110, 168, 254, 0.1);
    border: 1px solid rgba(110, 168, 254, 0.45);
    color: var(--accent);
    border-radius: 999px;
    padding: 0.22rem 0.75rem;
    font-size: 0.75rem;
  }
  .firsttime b { font-weight: 600; }
  .firsttime:hover { background: var(--accent-grad); color: #0b0e14; border-color: transparent; }
  /* flex-wrap + max-width keep every pill reachable on narrow phones (a tab
     red-team measured 95px+ of horizontal page overflow at 390-420px). */
  .tabs { display: inline-flex; flex-wrap: wrap; gap: 0.3rem; padding: 0.3rem; background: var(--bg-elev); border: 1px solid var(--border); border-radius: 999px; align-self: flex-start; max-width: 100%; }
  .tabs button {
    background: transparent;
    color: var(--text-dim);
    border: none;
    border-radius: 999px;
    padding: 0.4rem 1.15rem;
    font-size: 0.86rem;
    font-weight: 500;
    transition: color 0.15s ease, background 0.15s ease;
  }
  .tabs button:hover:not(.active) { color: var(--text); }
  .tabs button.active {
    background: var(--accent-grad);
    color: #0b0e14;
    font-weight: 600;
    box-shadow: 0 2px 10px rgba(110,168,254,0.25);
  }
</style>
