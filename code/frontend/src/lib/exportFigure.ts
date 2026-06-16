// Figure export: vector SVG / PDF for the SVG views, high-res PNG for the WebGL manifold,
// and animated GIFs that step the response/sequence frame-by-frame.
import { GIFEncoder, quantize, applyPalette } from "gifenc";

const BG = "#0b0e14";
const SVG_NS = "http://www.w3.org/2000/svg";

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function dims(svg: SVGSVGElement): [number, number] {
  const vb = svg.viewBox?.baseVal;
  return [vb?.width || svg.clientWidth || 800, vb?.height || svg.clientHeight || 520];
}

// Clone an SVG and inline computed paint styles + a background, so it renders standalone
// (no CSS variables or external stylesheet needed by a viewer / rasteriser).
function standalone(svg: SVGSVGElement): SVGSVGElement {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const src = svg.querySelectorAll<SVGElement>("*");
  const dst = clone.querySelectorAll<SVGElement>("*");
  const props = ["fill", "stroke", "stroke-width", "opacity", "stroke-opacity", "fill-opacity", "font-size", "font-family"];
  src.forEach((s, i) => {
    const cs = getComputedStyle(s);
    const d = dst[i];
    if (!d) return;
    for (const p of props) {
      const v = cs.getPropertyValue(p);
      if (v) d.style.setProperty(p, v);
    }
  });
  const [w, h] = dims(svg);
  const rect = document.createElementNS(SVG_NS, "rect");
  rect.setAttribute("width", String(w));
  rect.setAttribute("height", String(h));
  rect.setAttribute("fill", BG);
  clone.insertBefore(rect, clone.firstChild);
  clone.setAttribute("xmlns", SVG_NS);
  clone.setAttribute("width", String(w));
  clone.setAttribute("height", String(h));
  return clone;
}

export function exportSVG(svg: SVGSVGElement, filename: string) {
  const s = new XMLSerializer().serializeToString(standalone(svg));
  downloadBlob(new Blob([s], { type: "image/svg+xml" }), filename);
}

export async function exportSVGtoPDF(svg: SVGSVGElement, filename: string) {
  const { jsPDF } = await import("jspdf");
  await import("svg2pdf.js");
  const [w, h] = dims(svg);
  const pdf = new jsPDF({ orientation: w > h ? "landscape" : "portrait", unit: "pt", format: [w, h] });
  await (pdf as any).svg(standalone(svg), { x: 0, y: 0, width: w, height: h });
  pdf.save(filename);
}

export function exportCanvasPNG(canvas: HTMLCanvasElement, filename: string) {
  canvas.toBlob((blob) => blob && downloadBlob(blob, filename), "image/png");
}

// Copy a (WebGL) canvas into a 2D canvas so its pixels can be read for GIF frames.
export function webglToCanvas(gl: HTMLCanvasElement): HTMLCanvasElement {
  const cv = document.createElement("canvas");
  cv.width = gl.width;
  cv.height = gl.height;
  const ctx = cv.getContext("2d")!;
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.drawImage(gl, 0, 0);
  return cv;
}

// Rasterise an SVG element to a <canvas> at `scale` (for GIF frames or a PNG).
export async function svgToCanvas(svg: SVGSVGElement, scale = 1.5): Promise<HTMLCanvasElement> {
  const [w, h] = dims(svg);
  const s = new XMLSerializer().serializeToString(standalone(svg));
  const url = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(s);
  const img = new Image();
  await new Promise<void>((res, rej) => {
    img.onload = () => res();
    img.onerror = () => rej(new Error("svg rasterise failed"));
    img.src = url;
  });
  const cv = document.createElement("canvas");
  cv.width = Math.round(w * scale);
  cv.height = Math.round(h * scale);
  const ctx = cv.getContext("2d")!;
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  return cv;
}

export async function canvasPNG(svg: SVGSVGElement, filename: string, scale = 2) {
  exportCanvasPNG(await svgToCanvas(svg, scale), filename);
}

// Build an animated GIF by advancing `total` frames; `renderFrame(i)` must set the figure
// to frame i and resolve once it has drawn, and `capture()` returns the current frame canvas.
export async function exportGIF(opts: {
  total: number;
  renderFrame: (i: number) => Promise<void>;
  capture: () => Promise<HTMLCanvasElement>;
  filename: string;
  fps?: number;
}) {
  const gif = GIFEncoder();
  const delay = Math.round(1000 / (opts.fps ?? 2.5));
  for (let i = 0; i < opts.total; i++) {
    await opts.renderFrame(i);
    const cv = await opts.capture();
    const { data, width, height } = cv.getContext("2d")!.getImageData(0, 0, cv.width, cv.height);
    const palette = quantize(data, 256);
    const index = applyPalette(data, palette);
    gif.writeFrame(index, width, height, { palette, delay });
  }
  gif.finish();
  const bytes = new Uint8Array(gif.bytesView()); // fresh ArrayBuffer-backed copy for Blob typing
  downloadBlob(new Blob([bytes], { type: "image/gif" }), opts.filename);
}
