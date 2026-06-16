<script lang="ts">
  import { exportSVG, exportSVGtoPDF, exportCanvasPNG, canvasPNG, exportGIF, exportMP4, mp4Supported, svgToCanvas, webglToCanvas } from "../lib/exportFigure";

  // Export toolbar shown on each figure. SVG views get vector SVG + PDF; the WebGL manifold
  // gets a high-res PNG; any view with an `anim` driver gets an animated GIF.
  let { name, svg, webglCanvas, anim }: {
    name: string;
    svg?: () => SVGSVGElement | undefined;
    webglCanvas?: () => HTMLCanvasElement | undefined;
    anim?: { total: () => number; renderFrame: (i: number) => Promise<void>; restore: () => Promise<void> };
  } = $props();

  let busy = $state(false);
  let msg = $state("");
  const fname = (ext: string) => `${name}.${ext}`;

  async function run(label: string, fn: () => Promise<void> | void) {
    if (busy) return;
    busy = true;
    const animated = label === "GIF" || label === "MP4";
    msg = animated ? `rendering ${label}…` : "";
    try {
      await fn();
    } catch (e: any) {
      msg = `export failed: ${e?.message ?? e}`;
      setTimeout(() => (msg = ""), 5000);
    } finally {
      busy = false;
      if (msg.startsWith("rendering")) msg = "";
    }
  }

  const capture = async () => {
    const s = svg?.();
    if (s) return svgToCanvas(s, 1.4);
    const c = webglCanvas?.();
    if (c) return webglToCanvas(c);
    throw new Error("no figure to capture");
  };

  async function gif() {
    if (!anim) return;
    const total = Math.max(1, anim.total());
    await exportGIF({ total: total + 1, renderFrame: (i) => anim.renderFrame(Math.min(i, total)), capture, filename: fname("gif"), fps: 2.5 });
    await anim.restore();
  }

  async function mp4() {
    if (!anim) return;
    const total = Math.max(1, anim.total());
    await exportMP4({ total: total + 1, renderFrame: (i) => anim.renderFrame(Math.min(i, total)), capture, filename: fname("mp4"), fps: 2.5 });
    await anim.restore();
  }
</script>

<div class="export" data-testid="export-bar">
  <span class="lbl">Export</span>
  {#if svg}
    <button disabled={busy} onclick={() => run("SVG", () => exportSVG(svg()!, fname("svg")))} title="Vector SVG">SVG</button>
    <button disabled={busy} onclick={() => run("PDF", () => exportSVGtoPDF(svg()!, fname("pdf")))} title="Vector PDF">PDF</button>
    <button disabled={busy} onclick={() => run("PNG", () => canvasPNG(svg()!, fname("png"), 2))} title="High-res PNG">PNG</button>
  {/if}
  {#if webglCanvas}
    <button disabled={busy} onclick={() => run("PNG", () => exportCanvasPNG(webglCanvas()!, fname("png")))} title="High-res PNG">PNG</button>
  {/if}
  {#if anim}
    <button disabled={busy} onclick={() => run("GIF", gif)} title="Animated GIF" data-testid="export-gif">GIF</button>
    {#if mp4Supported()}
      <button disabled={busy} onclick={() => run("MP4", mp4)} title="H.264 MP4 video" data-testid="export-mp4">MP4</button>
    {/if}
  {/if}
  {#if msg}<span class="msg">{msg}</span>{/if}
</div>

<style>
  .export { display: flex; align-items: center; gap: 0.35rem; flex-wrap: wrap; }
  .lbl { font-size: 0.72rem; color: var(--text-dim); margin-right: 0.1rem; }
  .export button {
    background: var(--bg-elev); color: var(--text-dim); border: 1px solid var(--border);
    border-radius: 7px; padding: 0.18rem 0.5rem; font-size: 0.74rem; font-family: var(--mono); cursor: pointer;
  }
  .export button:hover:not(:disabled) { color: var(--text); border-color: var(--accent); }
  .export button:disabled { opacity: 0.5; cursor: default; }
  .msg { font-size: 0.72rem; color: var(--text-dim); }
</style>
