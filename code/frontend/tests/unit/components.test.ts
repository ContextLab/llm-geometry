// Render tests for the shared feature-002 components (Svelte 5 mount API, jsdom).
// Canvas 2D is unavailable in jsdom (getContext returns null) — MatrixHeatmap guards
// that, so these tests cover DOM structure/interaction, not pixel output.
import { describe, expect, it } from "vitest";
import { flushSync, mount, unmount } from "svelte";

import MatrixHeatmap from "../../src/lib/MatrixHeatmap.svelte";
import PipelineDiagram from "../../src/lib/PipelineDiagram.svelte";
import type { ArchEdge, ArchNode } from "../../src/lib/dataClient";

function el<T extends Element>(root: Element, sel: string): T | null {
  return root.querySelector<T>(sel);
}

describe("MatrixHeatmap", () => {
  it("mounts with a focusable canvas grid and unmounts cleanly", () => {
    const target = document.createElement("div");
    document.body.appendChild(target);
    const app = mount(MatrixHeatmap, {
      target,
      props: { values: [[1, -1], [0.5, 0]] },
    });
    flushSync();
    const host = el<HTMLElement>(target, '[data-testid="matrix-heatmap"]');
    expect(host).not.toBeNull();
    const canvas = target.querySelector("canvas");
    expect(canvas).not.toBeNull();
    expect(canvas!.getAttribute("tabindex")).toBe("0");
    unmount(app);
    expect(target.querySelector('[data-testid="matrix-heatmap"]')).toBeNull();
    target.remove();
  });

  it("accepts 1-D input (single-column contract shape) without throwing", () => {
    const target = document.createElement("div");
    document.body.appendChild(target);
    const app = mount(MatrixHeatmap, { target, props: { values: [0.1, -0.2, 0.3] } });
    flushSync();
    expect(el(target, '[data-testid="matrix-heatmap"]')).not.toBeNull();
    unmount(app);
    target.remove();
  });

  it("opens the cell editor on click when editable and commits via Enter", () => {
    const target = document.createElement("div");
    document.body.appendChild(target);
    const edits: Array<[number, number, number]> = [];
    const app = mount(MatrixHeatmap, {
      target,
      props: {
        values: [[1, 2], [3, 4]],
        editable: true,
        onCellEdit: (r: number, c: number, v: number) => edits.push([r, c, v]),
      },
    });
    flushSync();
    const canvas = target.querySelector("canvas")!;
    // jsdom rects are 0-sized, so (0,0) lands on cell (0,0) deterministically.
    canvas.dispatchEvent(new MouseEvent("click", { bubbles: true, clientX: 0, clientY: 0 }));
    flushSync();
    const editor = el<HTMLInputElement>(target, '[data-testid="heatmap-cell-editor"]');
    expect(editor).not.toBeNull();
    editor!.value = "7.5";
    editor!.dispatchEvent(new Event("input", { bubbles: true }));
    editor!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    flushSync();
    expect(edits).toEqual([[0, 0, 7.5]]);
    unmount(app);
    target.remove();
  });
});

function smallGraph(): { nodes: ArchNode[]; edges: ArchEdge[] } {
  const mk = (
    id: string,
    kind: ArchNode["kind"],
    op: ArchNode["op"],
    group: string,
    layer: number | null,
  ): ArchNode => ({ id, kind, op, label: id, layer, group, params: [] });
  const nodes = [
    mk("model.embed_tokens", "embedding", "module", "stem", null),
    mk("model.layers.0.q_proj", "linear", "module", "layer_0", 0),
    mk("model.layers.0.attention_softmax", "attention_softmax", "functional", "layer_0", 0),
    mk("model.layers.1.q_proj", "linear", "module", "layer_1", 1),
    mk("model.layers.2.q_proj", "linear", "module", "layer_2", 2),
    mk("lm_head", "lm_head", "module", "head", null),
  ];
  const edges: ArchEdge[] = nodes.slice(0, -1).map((n, i) => ({
    from: n.id,
    to: nodes[i + 1].id,
    tensor_shape: ["T", 8],
  }));
  return { nodes, edges };
}

describe("PipelineDiagram", () => {
  it("renders module nodes and functional pills; collapses layers >= 1 by default", () => {
    const target = document.createElement("div");
    document.body.appendChild(target);
    const { nodes, edges } = smallGraph();
    const app = mount(PipelineDiagram, { target, props: { nodes, edges } });
    flushSync();
    // Visible: stem + layer_0 (incl. the functional softmax pill — 1-to-1 rule).
    expect(el(target, '[data-testid="diagram-node-model.embed_tokens"]')).not.toBeNull();
    expect(
      el(target, '[data-testid="diagram-node-model.layers.0.attention_softmax"]'),
    ).not.toBeNull();
    // layer_1 / layer_2 collapsed (3 layer groups > 2): placeholder instead of nodes.
    expect(el(target, '[data-testid="diagram-collapsed-layer_1"]')).not.toBeNull();
    expect(el(target, '[data-testid="diagram-node-model.layers.1.q_proj"]')).toBeNull();
    unmount(app);
    target.remove();
  });

  it("expands a collapsed group on toggle and reports node selection", () => {
    const target = document.createElement("div");
    document.body.appendChild(target);
    const { nodes, edges } = smallGraph();
    const selected: string[] = [];
    const app = mount(PipelineDiagram, {
      target,
      props: { nodes, edges, onSelect: (id: string) => selected.push(id) },
    });
    flushSync();
    el<SVGElement>(target, '[data-testid="diagram-collapsed-layer_1"]')!.dispatchEvent(
      new MouseEvent("click", { bubbles: true }),
    );
    flushSync();
    expect(el(target, '[data-testid="diagram-node-model.layers.1.q_proj"]')).not.toBeNull();
    el<SVGElement>(target, '[data-testid="diagram-node-model.embed_tokens"]')!.dispatchEvent(
      new MouseEvent("click", { bubbles: true }),
    );
    flushSync();
    expect(selected).toEqual(["model.embed_tokens"]);
    unmount(app);
    target.remove();
  });
});
