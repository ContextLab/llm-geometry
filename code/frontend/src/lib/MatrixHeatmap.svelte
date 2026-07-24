<script lang="ts">
  import { onDestroy } from "svelte";

  import { showTip, hideTip } from "./tooltip";

  // Reusable canvas heatmap for weight/attention matrices (Architecture + Geometry
  // tabs). Values arrive already server-downsampled (≤ ~64×64 grids per the frozen
  // contract), so drawing each cell as a filled rect is cheap. Diverging color scale
  // centered at 0, matching the app's tokens: --bad (negative) → --bg-elev-2 (zero)
  // → --accent (positive). Row/col labels, when given, appear in the hover tooltip.
  interface Props {
    values: number[][] | number[]; // 1-D input renders as a single column (C=1)
    rowLabels?: string[];
    colLabels?: string[];
    editable?: boolean;
    onCellEdit?: (row: number, col: number, value: number) => void;
    maxCanvasPx?: number; // cap on the longer CSS dimension
  }
  let {
    values,
    rowLabels,
    colLabels,
    editable = false,
    onCellEdit,
    maxCanvasPx = 560,
  }: Props = $props();

  let canvasEl: HTMLCanvasElement | undefined = $state();
  // The input's text lives in its own $state (not on the editor object): binding to
  // `editor.text` races the input's teardown when `editor` is nulled on commit,
  // firing an async `null.text` read.
  let editor = $state<{ row: number; col: number } | null>(null);
  let editorText = $state("");

  // Normalize to number[][] (a 1-D vector is a single-column matrix).
  const grid = $derived(
    values.length === 0
      ? ([] as number[][])
      : Array.isArray(values[0])
        ? (values as number[][])
        : (values as number[]).map((v) => [v]),
  );
  const rows = $derived(grid.length);
  const cols = $derived(rows > 0 ? grid[0].length : 0);
  // Uniform square cells, capped so the whole matrix fits within maxCanvasPx.
  const cell = $derived(
    rows > 0 && cols > 0
      ? Math.max(4, Math.min(36, Math.floor(maxCanvasPx / Math.max(rows, cols))))
      : 0,
  );
  const cssW = $derived(cols * cell);
  const cssH = $derived(rows * cell);
  const absMax = $derived.by(() => {
    let m = 0;
    for (const row of grid) for (const v of row) m = Math.max(m, Math.abs(v));
    return m > 0 ? m : 1;
  });

  // Design tokens from styles/app.css, as RGB triples for interpolation.
  const ZERO_RGB = [27, 34, 53]; // --bg-elev-2
  const POS_RGB = [110, 168, 254]; // --accent
  const NEG_RGB = [255, 122, 144]; // --bad

  function cellColor(v: number): string {
    const t = Math.max(-1, Math.min(1, v / absMax));
    const to = t >= 0 ? POS_RGB : NEG_RGB;
    const a = Math.abs(t);
    const r = Math.round(ZERO_RGB[0] + (to[0] - ZERO_RGB[0]) * a);
    const g = Math.round(ZERO_RGB[1] + (to[1] - ZERO_RGB[1]) * a);
    const b = Math.round(ZERO_RGB[2] + (to[2] - ZERO_RGB[2]) * a);
    return `rgb(${r},${g},${b})`;
  }

  // The tooltip is a global singleton — clear it if this component unmounts mid-hover.
  onDestroy(hideTip);

  // Track devicePixelRatio reactively (a matchMedia "resolution" query fires when the
  // window moves to a different-DPI display) so the canvas re-renders crisply there.
  let dpr = $state(typeof devicePixelRatio === "number" ? devicePixelRatio : 1);
  $effect(() => {
    if (typeof matchMedia !== "function") return;
    const mq = matchMedia(`(resolution: ${dpr}dppx)`);
    const update = () => {
      dpr = typeof devicePixelRatio === "number" ? devicePixelRatio : 1;
    };
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  });

  // Redraw whenever the data or geometry changes; devicePixelRatio-aware so cells
  // stay crisp on retina displays.
  $effect(() => {
    const el = canvasEl;
    if (!el || rows === 0 || cols === 0) return;
    el.width = Math.max(1, Math.round(cssW * dpr));
    el.height = Math.max(1, Math.round(cssH * dpr));
    let ctx: CanvasRenderingContext2D | null = null;
    try {
      ctx = el.getContext("2d");
    } catch {
      ctx = null; // some environments (jsdom) throw instead of returning null
    }
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const gap = cell >= 8 ? 1 : 0; // hairline grid when cells are big enough
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        ctx.fillStyle = cellColor(grid[r][c]);
        ctx.fillRect(c * cell, r * cell, cell - gap, cell - gap);
      }
    }
  });

  function hitCell(e: MouseEvent): { row: number; col: number } | null {
    if (!canvasEl || cell === 0) return null;
    const rect = canvasEl.getBoundingClientRect();
    const col = Math.floor((e.clientX - rect.left) / cell);
    const row = Math.floor((e.clientY - rect.top) / cell);
    if (row < 0 || row >= rows || col < 0 || col >= cols) return null;
    return { row, col };
  }

  function formatValue(v: number): string {
    return String(Number(v.toPrecision(6)));
  }

  function onMove(e: MouseEvent) {
    const hit = hitCell(e);
    if (!hit) {
      hideTip();
      return;
    }
    const rl = rowLabels?.[hit.row] ?? `row ${hit.row}`;
    const cl = colLabels?.[hit.col] ?? `col ${hit.col}`;
    showTip(e, `${rl} · ${cl} · ${formatValue(grid[hit.row][hit.col])}`);
  }

  function onClick(e: MouseEvent) {
    if (!editable) return;
    const hit = hitCell(e);
    if (!hit) return;
    hideTip();
    editor = { row: hit.row, col: hit.col };
    editorText = formatValue(grid[hit.row][hit.col]);
  }

  function commitEdit() {
    if (!editor) return;
    const v = Number(editorText);
    if (Number.isFinite(v)) onCellEdit?.(editor.row, editor.col, v);
    editor = null;
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === "Escape" && editor) {
      e.stopPropagation();
      editor = null; // close without committing
    }
  }

  function onEditorKeydown(e: KeyboardEvent) {
    if (e.key === "Enter") {
      e.preventDefault();
      commitEdit();
    } else if (e.key === "Escape") {
      e.stopPropagation();
      editor = null;
    }
  }

  function autofocus(node: HTMLInputElement) {
    node.focus();
    node.select();
  }
</script>

<div class="heatmap" style:width={`${cssW}px`} style:height={`${cssH}px`}>
  <canvas
    bind:this={canvasEl}
    style:width={`${cssW}px`}
    style:height={`${cssH}px`}
    class:editable
    role="grid"
    aria-label={`matrix heatmap, ${rows} × ${cols}`}
    tabindex="0"
    data-testid="matrix-heatmap"
    onmousemove={onMove}
    onmouseleave={hideTip}
    onclick={onClick}
    onkeydown={onKeydown}
  ></canvas>
  {#if editor}
    <input
      class="cell-editor"
      type="number"
      step="any"
      data-testid="heatmap-cell-editor"
      style:left={`${Math.min(editor.col * cell, Math.max(0, cssW - 76))}px`}
      style:top={`${editor.row * cell}px`}
      bind:value={editorText}
      use:autofocus
      onkeydown={onEditorKeydown}
      onblur={commitEdit}
    />
  {/if}
</div>

<style>
  .heatmap {
    position: relative;
    display: inline-block;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: var(--bg-elev);
  }
  canvas {
    display: block;
    outline: none;
  }
  canvas.editable {
    cursor: pointer;
  }
  canvas:focus-visible {
    box-shadow: inset 0 0 0 2px var(--accent);
  }
  .cell-editor {
    position: absolute;
    width: 76px;
    padding: 0.15rem 0.3rem;
    font-family: var(--mono);
    font-size: 0.74rem;
    color: var(--text);
    background: var(--bg-elev-2);
    border: 1px solid var(--accent);
    border-radius: 6px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
  }
</style>
