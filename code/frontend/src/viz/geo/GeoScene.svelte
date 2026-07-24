<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import * as THREE from "three";
  import { OrbitControls } from "three/addons/controls/OrbitControls.js";

  import type { GeoToken, GeoVectorFieldData } from "../../lib/dataClient";
  import { showTip, hideTip } from "../../lib/tooltip";

  // The Geometry Lab's sphere: 1003 unit-norm token embeddings as points on a softly
  // shaded unit sphere, the selected vector field as ONE instanced arrow class
  // (shaft + head InstancedMesh pair, weight -> color/size), and — in force mode —
  // the prompt's per-position aggregate forces as a second, brighter arrow class
  // anchored at the prompt tokens, plus the prompt's token path as a geodesic
  // polyline. Field changes are animated (300 ms lerp old->new, rAF, no per-frame
  // allocations) so edits to the weights feel like the geometry itself moving.
  interface Props {
    field: GeoVectorFieldData | null;
    /** Prompt-token input embeddings (T,3) — anchors for path + sequence forces. */
    traceEmbeddings: number[][] | null;
    traceTokens: GeoToken[] | null;
    /** id -> display text for hover tooltips. */
    label: (id: number) => string;
  }
  let { field, traceEmbeddings, traceTokens, label }: Props = $props();

  let containerEl: HTMLDivElement | undefined;
  let webglError = $state("");

  const HEIGHT = 520;
  const MAX_ARROWS = 5015; // 1003 points x top_m<=5
  const MAX_FORCES = 50; // context window

  const COL_DIM = new THREE.Color("#2a3a6e");
  const COL_HI = new THREE.Color("#b794f6");
  const COL_ACC = new THREE.Color("#6ea8fe");
  const COL_FORCE = new THREE.Color("#ffb454");
  const COL_PATH = new THREE.Color("#5be0b0");

  let renderer: THREE.WebGLRenderer | undefined;
  let scene: THREE.Scene | undefined;
  let camera: THREE.PerspectiveCamera | undefined;
  let controls: OrbitControls | undefined;
  let points: THREE.Points | undefined;
  let shaft: THREE.InstancedMesh | undefined;
  let head: THREE.InstancedMesh | undefined;
  let fShaft: THREE.InstancedMesh | undefined;
  let fHead: THREE.InstancedMesh | undefined;
  let path: THREE.Group | undefined;
  let raf = 0;
  let resizeObs: ResizeObserver | undefined;

  // ---- arrow animation state (preallocated; no allocation inside the rAF loop) ----
  let count = 0; // active vocab arrows
  const srcO = new Float32Array(MAX_ARROWS * 3);
  const srcV = new Float32Array(MAX_ARROWS * 3);
  const tgtO = new Float32Array(MAX_ARROWS * 3);
  const tgtV = new Float32Array(MAX_ARROWS * 3);
  const curO = new Float32Array(MAX_ARROWS * 3);
  const curV = new Float32Array(MAX_ARROWS * 3);
  const srcW = new Float32Array(MAX_ARROWS);
  const tgtW = new Float32Array(MAX_ARROWS);
  const curW = new Float32Array(MAX_ARROWS);
  let fCount = 0; // active sequence-force arrows
  const fSrcO = new Float32Array(MAX_FORCES * 3);
  const fSrcV = new Float32Array(MAX_FORCES * 3);
  const fTgtO = new Float32Array(MAX_FORCES * 3);
  const fTgtV = new Float32Array(MAX_FORCES * 3);
  const fCurO = new Float32Array(MAX_FORCES * 3);
  const fCurV = new Float32Array(MAX_FORCES * 3);
  let animT = 1; // 0..1 lerp progress (1 = settled)
  let animStart = 0;
  const ANIM_MS = 300;

  // scratch objects reused every instance write
  const tmpDir = new THREE.Vector3();
  const tmpPos = new THREE.Vector3();
  const tmpQ = new THREE.Quaternion();
  const tmpS = new THREE.Vector3();
  const tmpM = new THREE.Matrix4();
  const tmpC = new THREE.Color();
  const UP = new THREE.Vector3(0, 1, 0);

  let pointIds: number[] = []; // token id per point (for hover)
  let pointPos: number[][] = []; // current point coords (for anchoring arrows)

  onMount(() => {
    initThree();
    return () => teardown();
  });
  onDestroy(() => teardown());

  function initThree() {
    if (!containerEl) return;
    const w = containerEl.clientWidth || 760;
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(42, w / HEIGHT, 0.1, 100);
    camera.position.set(0.6, 0.7, 3.0);
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    } catch {
      webglError = "WebGL is unavailable in this browser/environment.";
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(w, HEIGHT);
    // If the browser reclaims our context anyway (limit pressure), surface the
    // designed error instead of a silently blank canvas.
    renderer.domElement.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      webglError = "The 3-D view lost its graphics context — reload the page to restore it.";
    });
    containerEl.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0x6677aa, 1.0));
    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(3, 4, 5);
    scene.add(dir);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;

    // The manifold the embeddings live on: a soft-shaded unit sphere + faint wireframe.
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.985, 48, 32),
      new THREE.MeshStandardMaterial({
        color: 0x1d2745,
        roughness: 0.7,
        metalness: 0.25,
        transparent: true,
        opacity: 0.92,
        depthWrite: true, // occlude the far hemisphere so the field reads as a surface
      }),
    );
    scene.add(sphere);
    const wire = new THREE.LineSegments(
      new THREE.WireframeGeometry(new THREE.SphereGeometry(0.99, 24, 16)),
      new THREE.LineBasicMaterial({ color: 0x6ea8fe, transparent: true, opacity: 0.05, depthWrite: false }),
    );
    scene.add(wire);

    // Instanced arrow classes: one InstancedMesh per part (2 draw calls for the whole
    // vocab field), a second brighter/thicker pair for the prompt's sequence forces.
    const shaftGeom = new THREE.CylinderGeometry(1, 1, 1, 5, 1);
    const headGeom = new THREE.ConeGeometry(1, 1, 8);
    const arrowMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.7, toneMapped: false });
    shaft = new THREE.InstancedMesh(shaftGeom, arrowMat, MAX_ARROWS);
    head = new THREE.InstancedMesh(headGeom.clone(), arrowMat.clone(), MAX_ARROWS);
    const forceMat = new THREE.MeshBasicMaterial({ color: COL_FORCE, transparent: true, opacity: 0.98, toneMapped: false, depthTest: false });
    fShaft = new THREE.InstancedMesh(shaftGeom.clone(), forceMat, MAX_FORCES);
    fHead = new THREE.InstancedMesh(headGeom.clone(), forceMat.clone(), MAX_FORCES);
    for (const m of [shaft, head, fShaft, fHead]) {
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      m.count = 0;
      m.frustumCulled = false;
      scene.add(m);
    }
    fShaft.renderOrder = 8;
    fHead.renderOrder = 8;

    // Hover: raycast the token points (Manifold pattern).
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.03 };
    const ndc = new THREE.Vector2();
    renderer.domElement.addEventListener("pointermove", (ev: PointerEvent) => {
      if (!points || !camera || !renderer) return;
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(ndc, camera);
      if (path) {
        const ph = raycaster.intersectObjects(path.children, false).find((h) => h.object.userData?.tok);
        if (ph) {
          showTip(ev, `prompt token ${ph.object.userData.pos + 1}: "${ph.object.userData.tok}"`);
          return;
        }
      }
      const hits = raycaster.intersectObject(points);
      if (hits.length && hits[0].index != null) {
        const id = pointIds[hits[0].index];
        showTip(ev, `"${label(id)}" · token ${id}`);
      } else {
        hideTip();
      }
    });
    renderer.domElement.addEventListener("pointerleave", () => hideTip());

    const animate = (ts: number) => {
      raf = requestAnimationFrame(animate);
      controls?.update();
      if (animT < 1) {
        animT = Math.min(1, (ts - animStart) / ANIM_MS);
        applyLerp(easeOut(animT));
      }
      if (renderer && scene && camera) renderer.render(scene, camera);
    };
    raf = requestAnimationFrame(animate);

    resizeObs = new ResizeObserver(() => {
      if (!containerEl || !renderer || !camera) return;
      const cw = containerEl.clientWidth || w;
      renderer.setSize(cw, HEIGHT);
      camera.aspect = cw / HEIGHT;
      camera.updateProjectionMatrix();
      const mat = points?.material as THREE.ShaderMaterial | undefined;
      if (mat?.uniforms?.uScale) mat.uniforms.uScale.value = pointScale();
    });
    resizeObs.observe(containerEl);
  }

  function easeOut(x: number): number {
    return 1 - (1 - x) * (1 - x);
  }

  function pointScale(): number {
    return renderer ? renderer.getDrawingBufferSize(new THREE.Vector2()).height * 0.5 : 260;
  }

  // ---- token points -----------------------------------------------------------------

  function buildPoints(coords: number[][], ids: number[]) {
    if (!scene) return;
    if (points) {
      scene.remove(points);
      points.geometry.dispose();
      (points.material as THREE.Material).dispose();
      points = undefined;
    }
    pointIds = ids;
    pointPos = coords;
    const n = coords.length;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(coords.flat()), 3));
    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: { uSize: { value: 0.038 }, uScale: { value: pointScale() }, uColor: { value: COL_ACC } },
      vertexShader: `
        uniform float uSize; uniform float uScale;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = uSize * (uScale / -mv.z);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        uniform vec3 uColor;
        void main() {
          vec2 uv = gl_PointCoord - 0.5;
          float d2 = dot(uv, uv);
          if (d2 > 0.25) discard;
          gl_FragColor = vec4(uColor, 1.0 - 1.3 * d2);
        }`,
    });
    points = new THREE.Points(geom, mat);
    scene.add(points);
    void n;
  }

  // ---- vocab arrow field ------------------------------------------------------------

  // Display length: preserve relative magnitude but normalize per dataset by the 90th
  // percentile so one giant vector can't flatten everything else.
  function lengthScale(mags: number[]): number {
    if (mags.length === 0) return 1;
    const sorted = Array.from(mags).sort((a, b) => a - b);
    const p90 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))] || 1;
    return 0.34 / Math.max(p90, 1e-6);
  }

  function retarget(f: GeoVectorFieldData) {
    // snapshot the currently displayed state as the lerp source
    srcO.set(curO.subarray(0, count * 3));
    srcV.set(curV.subarray(0, count * 3));
    srcW.set(curW.subarray(0, count));
    const oldCount = count;
    const arrows = f.arrows;
    const n = Math.min(arrows.length, MAX_ARROWS);
    const s = lengthScale(arrows.map((a) => Math.hypot(a.vec[0], a.vec[1], a.vec[2])));
    for (let i = 0; i < n; i++) {
      const a = arrows[i];
      const o = f.points[a.origin_index];
      const m = Math.hypot(a.vec[0], a.vec[1], a.vec[2]);
      const len = m < 1e-9 ? 0 : Math.min(0.55, m * s);
      const k = m < 1e-9 ? 0 : len / m;
      tgtO[i * 3] = o[0];
      tgtO[i * 3 + 1] = o[1];
      tgtO[i * 3 + 2] = o[2];
      tgtV[i * 3] = a.vec[0] * k;
      tgtV[i * 3 + 1] = a.vec[1] * k;
      tgtV[i * 3 + 2] = a.vec[2] * k;
      tgtW[i] = Math.max(0, Math.min(1, a.weight));
      if (i >= oldCount) {
        // new arrows grow out of their origin instead of teleporting in
        srcO[i * 3] = o[0];
        srcO[i * 3 + 1] = o[1];
        srcO[i * 3 + 2] = o[2];
        srcV[i * 3] = srcV[i * 3 + 1] = srcV[i * 3 + 2] = 0;
        srcW[i] = tgtW[i];
      }
    }
    count = n;
    if (shaft) shaft.count = n;
    if (head) head.count = n;
    animStart = performance.now();
    animT = 0; // the render loop drives the lerp
  }

  function retargetForces(forces: { position: number; vec: number[] }[] | null, anchors: number[][] | null) {
    fSrcO.set(fCurO.subarray(0, fCount * 3));
    fSrcV.set(fCurV.subarray(0, fCount * 3));
    const oldCount = fCount;
    if (!forces || !anchors || anchors.length === 0) {
      fCount = 0;
      if (fShaft) fShaft.count = 0;
      if (fHead) fHead.count = 0;
      return;
    }
    const n = Math.min(forces.length, MAX_FORCES);
    const s = lengthScale(forces.map((f) => Math.hypot(f.vec[0], f.vec[1], f.vec[2])));
    let m = 0;
    for (let i = 0; i < n; i++) {
      const f = forces[i];
      const o = anchors[Math.min(f.position, anchors.length - 1)];
      if (!o) continue;
      const mag = Math.hypot(f.vec[0], f.vec[1], f.vec[2]);
      const len = mag < 1e-9 ? 0 : Math.min(0.6, mag * s);
      const k = mag < 1e-9 ? 0 : len / mag;
      fTgtO[m * 3] = o[0];
      fTgtO[m * 3 + 1] = o[1];
      fTgtO[m * 3 + 2] = o[2];
      fTgtV[m * 3] = f.vec[0] * k;
      fTgtV[m * 3 + 1] = f.vec[1] * k;
      fTgtV[m * 3 + 2] = f.vec[2] * k;
      if (m >= oldCount) {
        fSrcO[m * 3] = o[0];
        fSrcO[m * 3 + 1] = o[1];
        fSrcO[m * 3 + 2] = o[2];
        fSrcV[m * 3] = fSrcV[m * 3 + 1] = fSrcV[m * 3 + 2] = 0;
      }
      m++;
    }
    fCount = m;
    if (fShaft) fShaft.count = m;
    if (fHead) fHead.count = m;
    animStart = performance.now();
    animT = 0;
  }

  // Write instance matrices for the lerped state. Called from the render loop while
  // animT < 1; everything here reuses the scratch objects above.
  function applyLerp(e: number) {
    if (!shaft || !head) return;
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      curO[i3] = srcO[i3] + (tgtO[i3] - srcO[i3]) * e;
      curO[i3 + 1] = srcO[i3 + 1] + (tgtO[i3 + 1] - srcO[i3 + 1]) * e;
      curO[i3 + 2] = srcO[i3 + 2] + (tgtO[i3 + 2] - srcO[i3 + 2]) * e;
      curV[i3] = srcV[i3] + (tgtV[i3] - srcV[i3]) * e;
      curV[i3 + 1] = srcV[i3 + 1] + (tgtV[i3 + 1] - srcV[i3 + 1]) * e;
      curV[i3 + 2] = srcV[i3 + 2] + (tgtV[i3 + 2] - srcV[i3 + 2]) * e;
      curW[i] = srcW[i] + (tgtW[i] - srcW[i]) * e;
      writeArrow(shaft, head, i, curO, curV, 0.0028 + 0.0034 * curW[i], true);
    }
    shaft.instanceMatrix.needsUpdate = true;
    head.instanceMatrix.needsUpdate = true;
    if (shaft.instanceColor) shaft.instanceColor.needsUpdate = true;
    if (head.instanceColor) head.instanceColor.needsUpdate = true;
    if (fShaft && fHead) {
      for (let i = 0; i < fCount; i++) {
        const i3 = i * 3;
        fCurO[i3] = fSrcO[i3] + (fTgtO[i3] - fSrcO[i3]) * e;
        fCurO[i3 + 1] = fSrcO[i3 + 1] + (fTgtO[i3 + 1] - fSrcO[i3 + 1]) * e;
        fCurO[i3 + 2] = fSrcO[i3 + 2] + (fTgtO[i3 + 2] - fSrcO[i3 + 2]) * e;
        fCurV[i3] = fSrcV[i3] + (fTgtV[i3] - fSrcV[i3]) * e;
        fCurV[i3 + 1] = fSrcV[i3 + 1] + (fTgtV[i3 + 1] - fSrcV[i3 + 1]) * e;
        fCurV[i3 + 2] = fSrcV[i3 + 2] + (fTgtV[i3 + 2] - fSrcV[i3 + 2]) * e;
        writeArrow(fShaft, fHead, i, fCurO, fCurV, 0.012, false);
      }
      fShaft.instanceMatrix.needsUpdate = true;
      fHead.instanceMatrix.needsUpdate = true;
    }
  }

  function writeArrow(
    sh: THREE.InstancedMesh,
    hd: THREE.InstancedMesh,
    i: number,
    O: Float32Array,
    V: Float32Array,
    radius: number,
    colorByWeight: boolean,
  ) {
    const i3 = i * 3;
    const len = Math.hypot(V[i3], V[i3 + 1], V[i3 + 2]);
    if (len < 1e-6) {
      tmpS.set(0, 0, 0);
      tmpM.compose(tmpPos.set(O[i3], O[i3 + 1], O[i3 + 2]), tmpQ.identity(), tmpS);
      sh.setMatrixAt(i, tmpM);
      hd.setMatrixAt(i, tmpM);
      return;
    }
    tmpDir.set(V[i3] / len, V[i3 + 1] / len, V[i3 + 2] / len);
    tmpQ.setFromUnitVectors(UP, tmpDir);
    const shaftLen = len * 0.7;
    tmpPos.set(O[i3] + tmpDir.x * shaftLen * 0.5, O[i3 + 1] + tmpDir.y * shaftLen * 0.5, O[i3 + 2] + tmpDir.z * shaftLen * 0.5);
    tmpS.set(radius, shaftLen, radius);
    tmpM.compose(tmpPos, tmpQ, tmpS);
    sh.setMatrixAt(i, tmpM);
    const headLen = len * 0.3;
    tmpPos.set(O[i3] + tmpDir.x * (shaftLen + headLen * 0.5), O[i3 + 1] + tmpDir.y * (shaftLen + headLen * 0.5), O[i3 + 2] + tmpDir.z * (shaftLen + headLen * 0.5));
    tmpS.set(radius * 2.6, headLen, radius * 2.6);
    tmpM.compose(tmpPos, tmpQ, tmpS);
    hd.setMatrixAt(i, tmpM);
    if (colorByWeight) {
      tmpC.copy(COL_DIM).lerp(COL_HI, Math.sqrt(Math.max(0, curW[i]))); // sqrt: low-weight fans stay readable
      sh.setColorAt(i, tmpC);
      hd.setColorAt(i, tmpC);
    }
  }

  // ---- prompt path ------------------------------------------------------------------

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

  function buildPath(embeds: number[][] | null, tokens: GeoToken[] | null, visible: boolean) {
    if (!scene) return;
    if (path) {
      scene.remove(path);
      path.traverse((o: any) => {
        o.geometry?.dispose?.();
        o.material?.dispose?.();
      });
      path = undefined;
    }
    if (!visible || !embeds || embeds.length === 0) return;
    const g = new THREE.Group();
    const pts = embeds.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    const line: THREE.Vector3[] = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const seg = geodesic(pts[i], pts[i + 1], 22).map((v) => v.multiplyScalar(1.012));
      line.push(...(i > 0 ? seg.slice(1) : seg));
    }
    if (line.length >= 2) {
      g.add(
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(line),
          new THREE.LineBasicMaterial({ color: COL_PATH, transparent: true, opacity: 0.9 }),
        ),
      );
    }
    pts.forEach((p, i) => {
      const isLast = i === pts.length - 1;
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(isLast ? 0.028 : 0.018, 12, 12),
        new THREE.MeshBasicMaterial({ color: isLast ? 0xffffff : COL_PATH }),
      );
      dot.position.copy(p.clone().multiplyScalar(1.012));
      dot.userData = { tok: tokens?.[i]?.text ?? "", pos: i };
      g.add(dot);
    });
    g.renderOrder = 10;
    g.traverse((o: any) => {
      if (o.material) {
        o.material.depthTest = false;
        o.material.transparent = true;
      }
    });
    path = g;
    scene.add(g);
  }

  function teardown() {
    cancelAnimationFrame(raf);
    resizeObs?.disconnect();
    controls?.dispose();
    scene?.traverse((o: any) => {
      o.geometry?.dispose?.();
      if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m: THREE.Material) => m.dispose());
    });
    if (renderer) {
      renderer.domElement.remove();
      renderer.dispose();
      // dispose() alone keeps the GL context alive until GC — repeated tab switches
      // then exhaust the browser's context limit and silently blank the oldest canvas.
      renderer.forceContextLoss();
    }
    renderer = scene = camera = controls = points = shaft = head = fShaft = fHead = path = undefined;
  }

  // ---- reactivity -------------------------------------------------------------------

  let lastPointsRef: number[][] | null = null;
  $effect(() => {
    const f = field;
    if (!f || !scene) return;
    if (f.points !== lastPointsRef) {
      lastPointsRef = f.points;
      buildPoints(f.points, f.token_ids);
    }
    retarget(f);
    retargetForces(f.sequence_forces, traceEmbeddings);
  });

  $effect(() => {
    const showPath = field?.mode === "force";
    buildPath(traceEmbeddings, traceTokens, showPath);
    if (showPath) retargetForces(field?.sequence_forces ?? null, traceEmbeddings);
  });
</script>

<div bind:this={containerEl} class="canvas" data-testid="geo-canvas">
  {#if webglError}<div class="webgl-error" data-testid="geo-error">{webglError}</div>{/if}
</div>

<style>
  .canvas {
    position: relative;
    width: 100%;
    height: 520px;
    display: block;
    background: radial-gradient(circle at 50% 42%, rgba(110, 168, 254, 0.09), transparent 68%);
    border-radius: 12px;
    overflow: hidden;
  }
  .webgl-error {
    position: absolute;
    inset: 0;
    display: grid;
    place-content: center;
    color: var(--bad);
    font-family: var(--mono);
    font-size: 0.85rem;
  }
</style>
