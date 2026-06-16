<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { get } from "svelte/store";
  import * as THREE from "three";
  import { OrbitControls } from "three/addons/controls/OrbitControls.js";
  import { modelId, prefixText, temperature, responseText, responseStep, responseTokenCount, refreshNonce } from "../lib/stores";
  import { client, type ManifoldData } from "../lib/dataClient";
  import { showTip, hideTip } from "../lib/tooltip";
  import Progress from "../lib/Progress.svelte";
  import ExportBar from "../controls/ExportBar.svelte";

  // Visualization 3 — reachable "thoughts" as a sphere warped (RBF) toward likely next
  // tokens (project_description.md §3).
  let loading = $state(false);
  let progress = $state(0);
  let progressMsg = $state("");
  let error = $state("");
  let data = $state<ManifoldData | null>(null);
  let containerEl: HTMLDivElement | undefined;

  const SEED = 0;

  let renderer: THREE.WebGLRenderer | undefined;
  let scene: THREE.Scene | undefined;
  let camera: THREE.PerspectiveCamera | undefined;
  let controls: OrbitControls | undefined;
  let mesh: THREE.Mesh | undefined;
  let points: THREE.Points | undefined;
  let traj: THREE.Group | undefined;
  let raf = 0;
  let resizeObs: ResizeObserver | undefined;
  let debounce: ReturnType<typeof setTimeout> | undefined;
  let runId = 0;
  let lastRefresh = 0;

  const LOW = new THREE.Color("#2a3a6e");
  const HIGH = new THREE.Color("#b794f6");

  onMount(() => {
    initThree();
    if (data) buildMesh(data);
    return () => teardown();
  });
  onDestroy(() => teardown());

  function initThree() {
    if (!containerEl) return;
    const w = containerEl.clientWidth || 640;
    const h = 480;
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
    camera.position.set(0, 0, 4.5);
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    } catch (e) {
      error = "WebGL is unavailable in this browser/environment.";
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(w, h);
    containerEl.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0x6677aa, 1.1));
    const dir = new THREE.DirectionalLight(0xffffff, 1.4);
    dir.position.set(3, 4, 5);
    scene.add(dir);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.7;

    // Interactive hover: raycast the token points to reveal the token under the cursor.
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.1 };
    const ndc = new THREE.Vector2();
    renderer.domElement.addEventListener("pointermove", (ev: PointerEvent) => {
      if (!points || !camera || !renderer || !data) return;
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(ndc, camera);
      const hits = raycaster.intersectObject(points);
      if (hits.length && hits[0].index != null) {
        const i = hits[0].index;
        showTip(ev, `${data.token_strs?.[i] ?? ""}   emission ${((data.token_emis?.[i] ?? 0) * 100).toFixed(1)}%`);
      } else {
        hideTip();
      }
    });
    renderer.domElement.addEventListener("pointerleave", () => hideTip());

    const animate = () => {
      raf = requestAnimationFrame(animate);
      controls?.update();
      if (renderer && scene && camera) renderer.render(scene, camera);
    };
    animate();

    resizeObs = new ResizeObserver(() => {
      if (!containerEl || !renderer || !camera) return;
      const cw = containerEl.clientWidth || w;
      renderer.setSize(cw, h);
      camera.aspect = cw / h;
      camera.updateProjectionMatrix();
      const mat = points?.material as THREE.ShaderMaterial | undefined;
      if (mat?.uniforms?.uScale) mat.uniforms.uScale.value = pointScale();
    });
    resizeObs.observe(containerEl);
  }

  function buildMesh(d: ManifoldData, revealStep: number = get(responseStep)) {
    if (!scene) return;
    if (mesh) {
      scene.remove(mesh);
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
      mesh = undefined;
    }
    if (points) {
      scene.remove(points);
      points.geometry.dispose();
      (points.material as THREE.Material).dispose();
      points = undefined;
    }
    if (traj) {
      scene.remove(traj);
      traj.traverse((o: any) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
      traj = undefined;
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(d.vertices.flat()), 3));
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(d.faces.flat()), 1));
    const colors = new Float32Array(d.vertices.length * 3);
    for (let i = 0; i < d.vertices.length; i++) {
      const c = LOW.clone().lerp(HIGH, Math.max(0, Math.min(1, d.warp[i] ?? 0)));
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geom.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.15 });
    mesh = new THREE.Mesh(geom, mat);
    scene.add(mesh);

    // Tokens on the radius-2 sphere — the coordinates the manifold reaches toward. Each
    // marker's emission probability is encoded as a SUBTLE size change and an OBVIOUS
    // transparency change (per-point shader: likely tokens are larger and opaque, unlikely
    // ones tiny and nearly invisible).
    if (d.token_points?.length) {
      const n = d.token_points.length;
      const pgeom = new THREE.BufferGeometry();
      pgeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(d.token_points.flat()), 3));
      const pcolors = new Float32Array(n * 3);
      const aEmis = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const e = Math.max(0, Math.min(1, d.token_emis?.[i] ?? 0));
        const c = LOW.clone().lerp(HIGH, e);
        pcolors[i * 3] = c.r; pcolors[i * 3 + 1] = c.g; pcolors[i * 3 + 2] = c.b;
        aEmis[i] = e;
      }
      pgeom.setAttribute("aColor", new THREE.BufferAttribute(pcolors, 3));
      pgeom.setAttribute("aEmis", new THREE.BufferAttribute(aEmis, 1));
      const pmat = new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        uniforms: { uSize: { value: 0.05 }, uScale: { value: pointScale() } },
        vertexShader: `
          attribute vec3 aColor;
          attribute float aEmis;
          uniform float uSize; uniform float uScale;
          varying vec3 vColor; varying float vAlpha;
          void main() {
            vColor = aColor;
            vAlpha = 0.10 + 0.90 * aEmis;              // obvious transparency change
            vec4 mv = modelViewMatrix * vec4(position, 1.0);
            gl_PointSize = uSize * (0.55 + 0.9 * aEmis) * (uScale / -mv.z); // subtle size change
            gl_Position = projectionMatrix * mv;
          }`,
        fragmentShader: `
          varying vec3 vColor; varying float vAlpha;
          void main() {
            vec2 uv = gl_PointCoord - 0.5;
            if (dot(uv, uv) > 0.25) discard;           // round markers
            gl_FragColor = vec4(vColor, vAlpha);
          }`,
      });
      points = new THREE.Points(pgeom, pmat);
      scene.add(points);
    }

    // Response trajectory: a bright GEODESIC line across the radius-2 sphere through the
    // response tokens (great-circle arcs that stay on the surface, not chords through the
    // interior), revealed up to the current step (▶ Play grows it as the manifold re-warps).
    if (d.traj_points?.length) {
      const tpts = d.traj_points;
      const shown = Math.min(revealStep, tpts.length); // reveal one token per frame
      const g = new THREE.Group();
      const raw = tpts.slice(0, shown).map((p) => new THREE.Vector3(p[0], p[1], p[2]));
      const linePts: THREE.Vector3[] = [];
      for (let i = 0; i < raw.length - 1; i++) {
        const seg = geodesic(raw[i], raw[i + 1], 28).map((v) => v.multiplyScalar(1.01)); // sit just above the surface
        linePts.push(...(i > 0 ? seg.slice(1) : seg));
      }
      if (linePts.length >= 2) {
        const lgeom = new THREE.BufferGeometry().setFromPoints(linePts);
        g.add(new THREE.Line(lgeom, new THREE.LineBasicMaterial({ color: 0x5be0b0, transparent: true, opacity: 0.95 })));
      }
      raw.forEach((p, i) => {
        const isCur = i === shown - 1;
        const m = new THREE.Mesh(
          new THREE.SphereGeometry(isCur ? 0.08 : 0.05, 14, 14),
          new THREE.MeshBasicMaterial({ color: isCur ? 0xffffff : 0x5be0b0 }),
        );
        m.position.copy(p.clone().multiplyScalar(1.01));
        g.add(m);
      });
      traj = g;
      scene.add(g);
    }
  }

  // Great-circle arc between two points on a sphere (slerp at their shared radius).
  function geodesic(a: THREE.Vector3, b: THREE.Vector3, segments: number): THREE.Vector3[] {
    const r = a.length();
    const va = a.clone().normalize();
    const vb = b.clone().normalize();
    const omega = Math.acos(Math.max(-1, Math.min(1, va.dot(vb))));
    if (omega < 1e-5) return [a.clone(), b.clone()];
    const sin = Math.sin(omega);
    const out: THREE.Vector3[] = [];
    for (let s = 0; s <= segments; s++) {
      const t = s / segments;
      const k1 = Math.sin((1 - t) * omega) / sin;
      const k2 = Math.sin(t * omega) / sin;
      out.push(va.clone().multiplyScalar(k1).add(vb.clone().multiplyScalar(k2)).multiplyScalar(r));
    }
    return out;
  }

  // Perspective point-size attenuation scale (matches three's PointsMaterial convention).
  function pointScale(): number {
    return renderer ? renderer.getDrawingBufferSize(new THREE.Vector2()).height * 0.5 : 240;
  }

  function teardown() {
    cancelAnimationFrame(raf);
    resizeObs?.disconnect();
    controls?.dispose();
    if (renderer) {
      renderer.domElement.remove();
      renderer.dispose();
    }
    renderer = scene = camera = controls = mesh = points = traj = undefined;
  }

  $effect(() => {
    const m = $modelId;
    const pfx = $prefixText;
    const temp = $temperature;
    const resp = $responseText;
    const step = $responseStep;
    const rn = $refreshNonce;
    const force = rn !== lastRefresh;
    lastRefresh = rn;
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(() => void load(m, pfx, temp, resp, step, force), force ? 0 : 350);
    return () => {
      if (debounce) clearTimeout(debounce);
    };
  });

  async function load(m: string, pfx: string, temp: number, resp: string, step: number, force = false) {
    const my = ++runId;
    error = "";
    loading = true;
    progress = 0;
    progressMsg = force ? "recomputing…" : "starting…";
    try {
      const params = { temperature: temp, seed: SEED }; // full vocab (a dot per token)
      const inputs = { prefix_text: pfx, response_text: resp, response_step: step };
      if (!force) {
        await client.ensureArtifact("manifold", m, params, inputs, (p, msg) => {
          if (my === runId) {
            progress = p;
            progressMsg = msg;
          }
        });
      }
      if (my !== runId) return;
      data = await client.getManifold(m, { prefix_text: pfx, response_text: resp, response_step: step, ...params, ...(force ? { force: true } : {}) });
      if (my !== runId) return;
      buildMesh(data, step);
    } catch (e: any) {
      if (my === runId) error = `${e.type ?? "Error"}: ${e.message ?? e}`;
    } finally {
      if (my === runId) loading = false;
    }
  }

  // Export: drive the response animation frame-by-frame (for a GIF) + a frame settle so the
  // WebGL buffer is captured after it draws.
  async function renderFrame(i: number) {
    await load(get(modelId), get(prefixText), get(temperature), get(responseText), i);
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  }
  const exportAnim = {
    total: () => get(responseTokenCount),
    renderFrame,
    restore: () => renderFrame(get(responseStep)),
  };
</script>

<section class="viz panel" data-testid="viz-manifold" data-ready={data ? 1 : 0}>
  <header>
    <div>
      <h2>Reachable "thoughts" as a manifold</h2>
      <p class="sub">A unit sphere warped (RBF + ARAP) toward likely next tokens. <b>Bulges = high emission probability; dots = tokens on the radius-2 sphere.</b> Drag to rotate, scroll to zoom, hover a dot for its token. The response traces a geodesic across the sphere.</p>
    </div>
    <ExportBar name="manifold" webglCanvas={() => renderer?.domElement} anim={exportAnim} />
  </header>
  {#if loading}<div class="loading"><Progress {progress} message={progressMsg} /></div>{/if}
  {#if error}<div class="error" data-testid="viz-manifold-error">{error}</div>{/if}
  <div bind:this={containerEl} class="canvas" data-testid="manifold-canvas"></div>
  {#if data}
    <p class="caption">
      {data.vertices.length} mesh vertices · bulging toward: {data.top_tokens.slice(0, 5).map((t) => t.token_str.trim() || "∅").join(", ")}
    </p>
  {/if}
</section>

<style>
  .viz { padding: 1.2rem 1.4rem; display: flex; flex-direction: column; gap: 0.8rem; }
  header { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
  header > div { min-width: 0; }
  header h2 { margin: 0; font-size: 1.1rem; }
  .sub { margin: 0.2rem 0 0; color: var(--text-dim); font-size: 0.82rem; }
  .loading { padding: 0.3rem 0; }
  .error { background: rgba(255,122,144,0.12); color: var(--bad); border: 1px solid rgba(255,122,144,0.3); border-radius: 10px; padding: 0.6rem 0.8rem; font-family: var(--mono); font-size: 0.85rem; }
  .canvas { width: 100%; height: 480px; display: block; background: radial-gradient(circle at 50% 40%, rgba(110,168,254,0.08), transparent 70%); border-radius: 12px; overflow: hidden; }
  .caption { margin: 0; color: var(--text-dim); font-size: 0.76rem; }
</style>
