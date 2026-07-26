<script lang="ts">
  import { onDestroy, untrack } from "svelte";
  import type { ArchNode, ArchEdge } from "./dataClient";
  import { showTip, hideTip } from "./tooltip";

  // The tooltip is a global singleton — clear it if this component unmounts mid-hover.
  onDestroy(hideTip);

  // Generic SVG diagram of an /api/arch/graph architecture graph. Vertical flow
  // grouped by node.group (stem → layer_0..layer_{n-1} → head). Every traced op is
  // drawn — param-carrying modules (embedding/linear/lm_head/…) as rectangles sized
  // by (log) parameter magnitude; parameterless functional ops (RoPE, softmax,
  // residual adds, activations) as smaller pills, so the 1-to-1 invariant holds
  // visually. Layer groups collapse (default-collapsed beyond the first layer for
  // large models); edges into a collapsed group route to its placeholder. Wheel
  // zooms and drag pans the SVG viewBox.
  interface Props {
    nodes: ArchNode[];
    edges: ArchEdge[];
    selected?: string | null;
    /**
     * Node the view should keep visible (the trace playhead). Large models collapse
     * every layer past the first and open zoomed all the way out, so without this the
     * "animation" highlighted a node that was inside a collapsed group, off-screen, and
     * a few pixels tall — the playhead advanced but nothing visibly moved.
     */
    focus?: string | null;
    onSelect?: (nodeId: string) => void;
  }
  let { nodes, edges, selected = null, focus = null, onSelect }: Props = $props();

  const DIAG_W = 640; // content width in viewBox units
  const CX = DIAG_W / 2;
  const NODE_GAP = 12;
  const HEADER_H = 26;
  const GROUP_GAP = 18;

  // --- collapsible layer groups -------------------------------------------------
  // User toggles are overrides on top of the default policy: for large models
  // (more than 2 layer groups), every layer beyond the first starts collapsed.
  let expandOverrides = $state<Record<string, boolean>>({});

  const layerGroupCount = $derived(
    new Set(nodes.map((n) => n.group).filter((g) => /^layer_\d+$/.test(g))).size,
  );

  function isCollapsed(group: string): boolean {
    if (group in expandOverrides) return !expandOverrides[group];
    const m = /^layer_(\d+)$/.exec(group);
    if (!m) return false; // stem/head never collapse
    return layerGroupCount > 2 && Number(m[1]) >= 1;
  }

  function toggleGroup(group: string) {
    expandOverrides = { ...expandOverrides, [group]: isCollapsed(group) };
  }

  // Reset overrides when a different graph arrives.
  let lastGraphKey = "";
  $effect(() => {
    // Full id list: count+endpoints alone can collide for same-architecture models.
    const key = nodes.map((n) => n.id).join("|");
    if (key !== lastGraphKey) {
      lastGraphKey = key;
      untrack(() => {
        expandOverrides = {};
        viewInited = false;
      });
    }
  });

  // --- layout --------------------------------------------------------------------
  interface PlacedNode {
    node: ArchNode;
    x: number; // center
    y: number; // center
    w: number;
    h: number;
    pill: boolean;
  }
  interface GroupHeader {
    group: string;
    label: string;
    y: number;
    collapsible: boolean;
    collapsed: boolean;
    count: number;
  }
  interface PlacedEdge {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    path: string;
  }

  function paramCount(n: ArchNode): number {
    let total = 0;
    for (const p of n.params) total += p.shape.reduce((a, b) => a * b, 1);
    return total;
  }

  function nodeSize(n: ArchNode): { w: number; h: number; pill: boolean } {
    if (n.op === "functional" || n.params.length === 0) {
      return { w: Math.max(72, n.label.length * 6.4 + 26), h: 22, pill: true };
    }
    // Rectangle width grows with log10(#params) so q_proj vs embed_tokens read differently.
    const w = Math.min(280, 96 + 24 * Math.log10(paramCount(n) + 10));
    return { w, h: 32, pill: false };
  }

  const layout = $derived.by(() => {
    const placed: PlacedNode[] = [];
    const headers: GroupHeader[] = [];
    const collapsedBoxes: { group: string; x: number; y: number; w: number; h: number; count: number }[] = [];
    const pos = new Map<string, { x: number; y: number; top: number; bottom: number }>();

    // Preserve first-appearance order of groups (backend emits trace order).
    const groupOrder: string[] = [];
    const byGroup = new Map<string, ArchNode[]>();
    for (const n of nodes) {
      if (!byGroup.has(n.group)) {
        byGroup.set(n.group, []);
        groupOrder.push(n.group);
      }
      byGroup.get(n.group)!.push(n);
    }

    let y = 8;
    for (const group of groupOrder) {
      const members = byGroup.get(group)!;
      const collapsible = /^layer_\d+$/.test(group);
      const collapsed = collapsible && isCollapsed(group);
      headers.push({
        group,
        label: group.replace("_", " "),
        y: y + HEADER_H / 2,
        collapsible,
        collapsed,
        count: members.length,
      });
      y += HEADER_H;
      if (collapsed) {
        const w = 260;
        const h = 34;
        const cy = y + h / 2;
        collapsedBoxes.push({ group, x: CX, y: cy, w, h, count: members.length });
        for (const n of members) pos.set(n.id, { x: CX, y: cy, top: cy - h / 2, bottom: cy + h / 2 });
        y += h + GROUP_GAP;
        continue;
      }
      for (const n of members) {
        const { w, h, pill } = nodeSize(n);
        const cy = y + h / 2;
        placed.push({ node: n, x: CX, y: cy, w, h, pill });
        pos.set(n.id, { x: CX, y: cy, top: cy - h / 2, bottom: cy + h / 2 });
        y += h + NODE_GAP;
      }
      y += GROUP_GAP - NODE_GAP;
    }

    // Edges: anchor bottom-of-source → top-of-target; after collapse-mapping,
    // drop intra-placeholder self-edges and de-duplicate. Long skips (residual
    // connections) bow out to the right so they stay visible.
    const placedEdges: PlacedEdge[] = [];
    const seen = new Set<string>();
    for (const e of edges) {
      const a = pos.get(e.from);
      const b = pos.get(e.to);
      if (!a || !b) continue;
      if (a.x === b.x && a.y === b.y) continue; // both inside one collapsed group
      const key = `${a.x},${a.y}->${b.x},${b.y}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const y1 = a.bottom;
      const y2 = b.top;
      const dy = y2 - y1;
      let path: string;
      if (dy > 0 && dy <= NODE_GAP + HEADER_H + GROUP_GAP + 2) {
        path = `M ${a.x} ${y1} L ${b.x} ${y2}`; // adjacent: straight
      } else {
        const bow = Math.min(200, 46 + Math.abs(dy) * 0.06); // skip/residual: arc right
        const mx = CX + DIAG_W / 2 - 300 + bow;
        path = `M ${a.x} ${y1} C ${mx} ${y1 + dy * 0.2}, ${mx} ${y2 - dy * 0.2}, ${b.x} ${y2}`;
      }
      placedEdges.push({ x1: a.x, y1, x2: b.x, y2, path });
    }

    return { placed, headers, collapsedBoxes, edges: placedEdges, pos, totalH: y + 8 };
  });

  // --- viewBox pan/zoom ------------------------------------------------------------
  let svgEl: SVGSVGElement | undefined = $state();
  let vb = $state({ x: -20, y: -10, w: DIAG_W + 40, h: 600 });
  let viewInited = false;

  // Fit the view once per graph (not on every collapse toggle).
  $effect(() => {
    const h = layout.totalH;
    if (!viewInited && nodes.length > 0) {
      viewInited = true;
      untrack(() => {
        // Fitting the WHOLE model height (2658 units for a 24-layer model in a 520 px
        // SVG) rendered 32-unit nodes ~6 px tall and their labels ~2 px — a column of
        // unreadable slivers, against a caption inviting you to click one. Cap the
        // initial height so the stem is legible; the user can zoom out from there.
        const fit = Math.max(h + 20, 240);
        vb = { x: -20, y: -10, w: DIAG_W + 40, h: Math.min(fit, INITIAL_MAX_H) };
      });
    }
  });

  // --- follow the playhead ----------------------------------------------------------
  // Two steps, in this order: make the focused node EXIST in the layout (its group may
  // be collapsed), then bring it into view. The auto-expanded group is remembered so the
  // next one can close it again — otherwise a 30-layer model ends up fully expanded and
  // the view scrolls through hundreds of nodes.
  let autoExpanded: string | null = null;

  $effect(() => {
    const id = focus;
    if (!id) return;
    const node = nodes.find((n) => n.id === id);
    if (!node) return;
    untrack(() => {
      if (node.group === autoExpanded) return;
      const next = { ...expandOverrides };
      if (autoExpanded && autoExpanded !== node.group) delete next[autoExpanded];
      if (/^layer_\d+$/.test(node.group)) {
        next[node.group] = true;
        autoExpanded = node.group;
      } else {
        autoExpanded = null;
      }
      expandOverrides = next;
    });
  });

  // Reset the "zoom in once" latch whenever following stops, so a later playback run
  // gets a readable zoom again without stealing zoom mid-run.
  let followZoomed = false;
  $effect(() => {
    if (focus === null) followZoomed = false;
  });

  const INITIAL_MAX_H = 900; // ≈ 20 nodes: readable at 520 px tall
  const FOLLOW_H = 520; // viewBox units ≈ a dozen nodes: readable, not claustrophobic

  $effect(() => {
    const id = focus;
    if (!id) return;
    const p = layout.pos.get(id);
    if (!p) return;
    untrack(() => {
      let { x, y, w, h } = vb;
      // First frame of a run: if the whole model is in view, nothing is legible — zoom
      // to a readable height once, then respect whatever zoom the user chooses after.
      if (!followZoomed && h > FOLLOW_H * 1.35) {
        const s = FOLLOW_H / h;
        w *= s;
        h = FOLLOW_H;
        x = CX - w / 2;
        followZoomed = true;
      }
      // Keep the node inside the middle band; only scroll when it leaves it.
      const margin = h * 0.22;
      if (p.y < y + margin || p.y > y + h - margin) y = p.y - h / 2;
      vb = { x, y, w, h };
    });
  });

  // Wheel zoom about the cursor. Registered manually so preventDefault works
  // (browsers treat declarative wheel listeners as passive).
  $effect(() => {
    const el = svgEl;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx = vb.x + ((e.clientX - rect.left) / rect.width) * vb.w;
      const my = vb.y + ((e.clientY - rect.top) / rect.height) * vb.h;
      const f = Math.exp(e.deltaY * 0.0015);
      const w = Math.min(Math.max(vb.w * f, 120), 6000);
      const s = w / vb.w;
      vb = { x: mx - (mx - vb.x) * s, y: my - (my - vb.y) * s, w, h: vb.h * s };
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  });

  let panning = $state<{ px: number; py: number } | null>(null);
  // Capturing at pointerdown retargets the browser's derived click to the SVG, so
  // node/group click handlers never fire from a real pointer. Instead, arm on
  // pointerdown and only capture once movement crosses a drag threshold — taps stay
  // ordinary clicks, drags still pan smoothly.
  let armedPan: { px: number; py: number; id: number } | null = null;
  const DRAG_THRESHOLD_PX = 4;

  function onPointerDown(e: PointerEvent) {
    armedPan = { px: e.clientX, py: e.clientY, id: e.pointerId };
  }

  function onPointerMove(e: PointerEvent) {
    if (!svgEl) return;
    if (!panning) {
      if (!armedPan) return;
      if (
        Math.hypot(e.clientX - armedPan.px, e.clientY - armedPan.py) < DRAG_THRESHOLD_PX
      ) {
        return;
      }
      panning = { px: armedPan.px, py: armedPan.py };
      try {
        svgEl.setPointerCapture(armedPan.id);
      } catch {
        // pointer may already be gone (e.g. pen leaving the digitizer) — pan uncaptured
      }
    }
    const rect = svgEl.getBoundingClientRect();
    const dx = ((e.clientX - panning.px) / rect.width) * vb.w;
    const dy = ((e.clientY - panning.py) / rect.height) * vb.h;
    panning = { px: e.clientX, py: e.clientY };
    vb = { ...vb, x: vb.x - dx, y: vb.y - dy };
  }

  function onPointerUp() {
    panning = null;
    armedPan = null;
  }

  function kindClass(n: ArchNode): string {
    if (n.kind === "embedding" || n.kind === "lm_head") return "k-embed";
    if (n.kind === "linear" || n.kind === "mlp") return "k-linear";
    if (n.kind === "layernorm" || n.kind === "rmsnorm") return "k-norm";
    return "k-fn";
  }

  function nodeTip(n: ArchNode): string {
    const shapes = n.params.map((p) => `${p.name} [${p.shape.join("×")}]`).join(", ");
    const tied = n.params.find((p) => p.tied_to);
    return (
      `${n.id} · ${n.kind}` +
      (shapes ? ` · ${shapes}` : " · parameterless") +
      (tied ? ` · tied to ${tied.tied_to}` : "")
    );
  }
</script>

<div class="diagram-wrap">
  <svg
    bind:this={svgEl}
    viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`}
    data-testid="pipeline-diagram"
    role="application"
    aria-label="model architecture diagram"
    class:panning={panning !== null}
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    onpointercancel={onPointerUp}
  >
    <!-- edges under nodes -->
    {#each layout.edges as e (e.path)}
      <path class="edge" d={e.path} />
    {/each}

    <!-- group headers -->
    {#each layout.headers as h (h.group)}
      {#if h.collapsible}
        <g
          class="group-head"
          role="button"
          tabindex="0"
          aria-label={`toggle ${h.label}`}
          data-testid={`diagram-group-${h.group}`}
          onclick={() => toggleGroup(h.group)}
          onkeydown={(e) => (e.key === "Enter" || e.key === " ") && toggleGroup(h.group)}
        >
          <text class="group-label" x={16} y={h.y + 4}>
            {h.collapsed ? "▸" : "▾"} {h.label}
          </text>
        </g>
      {:else}
        <text class="group-label static" x={16} y={h.y + 4}>{h.label}</text>
      {/if}
    {/each}

    <!-- collapsed layer placeholders -->
    {#each layout.collapsedBoxes as b (b.group)}
      <g
        class="collapsed-box"
        role="button"
        tabindex="0"
        aria-label={`expand ${b.group}`}
        data-testid={`diagram-collapsed-${b.group}`}
        onclick={() => toggleGroup(b.group)}
        onkeydown={(e) => (e.key === "Enter" || e.key === " ") && toggleGroup(b.group)}
      >
        <rect x={b.x - b.w / 2} y={b.y - b.h / 2} width={b.w} height={b.h} rx={9} />
        <text x={b.x} y={b.y + 4}>{b.group.replace("_", " ")} · {b.count} ops</text>
      </g>
    {/each}

    <!-- nodes: rects for parameterized modules, pills for functional ops -->
    {#each layout.placed as p (p.node.id)}
      <g
        class={`node ${kindClass(p.node)}`}
        class:selected={selected === p.node.id}
        role="button"
        tabindex="0"
        aria-label={p.node.label}
        data-testid={`diagram-node-${p.node.id}`}
        onclick={() => onSelect?.(p.node.id)}
        onkeydown={(e) => (e.key === "Enter" || e.key === " ") && onSelect?.(p.node.id)}
        onmousemove={(e) => showTip(e, nodeTip(p.node))}
        onmouseleave={hideTip}
      >
        <rect
          x={p.x - p.w / 2}
          y={p.y - p.h / 2}
          width={p.w}
          height={p.h}
          rx={p.pill ? p.h / 2 : 7}
        />
        <text x={p.x} y={p.y + (p.pill ? 3.5 : 4)}>{p.node.label}</text>
      </g>
    {/each}
  </svg>
</div>

<style>
  .diagram-wrap {
    width: 100%;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-elev);
    overflow: hidden;
  }
  svg {
    display: block;
    width: 100%;
    height: 520px;
    cursor: grab;
    touch-action: none;
  }
  svg.panning {
    cursor: grabbing;
  }
  .edge {
    fill: none;
    stroke: var(--border);
    stroke-width: 1.4;
  }
  .group-label {
    fill: var(--text-dim);
    font-size: 11px;
    font-family: var(--mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .group-head {
    cursor: pointer;
    outline: none;
  }
  .group-head:hover .group-label,
  .group-head:focus-visible .group-label {
    fill: var(--accent);
  }
  .collapsed-box {
    cursor: pointer;
    outline: none;
  }
  .collapsed-box rect {
    fill: var(--bg-elev-2);
    stroke: var(--border);
    stroke-dasharray: 4 3;
  }
  .collapsed-box:hover rect,
  .collapsed-box:focus-visible rect {
    stroke: var(--accent);
  }
  .collapsed-box text {
    fill: var(--text-dim);
    font-size: 11px;
    text-anchor: middle;
    font-family: var(--mono);
  }
  .node {
    cursor: pointer;
    outline: none;
  }
  .node rect {
    stroke-width: 1.2;
    transition: filter 0.12s ease;
  }
  .node text {
    fill: var(--text);
    font-size: 11px;
    text-anchor: middle;
    pointer-events: none;
  }
  .node:hover rect,
  .node:focus-visible rect {
    filter: brightness(1.25);
  }
  .node.selected rect {
    stroke: var(--accent);
    stroke-width: 2.4;
    filter: drop-shadow(0 0 6px rgba(110, 168, 254, 0.55));
  }
  .k-embed rect {
    fill: rgba(183, 148, 246, 0.18);
    stroke: var(--accent-2);
  }
  .k-linear rect {
    fill: rgba(110, 168, 254, 0.16);
    stroke: var(--accent);
  }
  .k-norm rect {
    fill: rgba(154, 166, 192, 0.12);
    stroke: var(--text-dim);
  }
  .k-fn rect {
    fill: var(--bg-elev-2);
    stroke: var(--border);
  }
  .k-fn text {
    fill: var(--text-dim);
    font-size: 10px;
  }
</style>
