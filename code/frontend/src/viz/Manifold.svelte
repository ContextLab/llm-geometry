<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { get } from "svelte/store";
  import * as THREE from "three";
  import { OrbitControls } from "three/addons/controls/OrbitControls.js";
  import { modelId, prefixText, temperature, responseText, responseStep, responseTokenCount, rbfWidth, showSurface, refreshNonce } from "../lib/stores";
  import { client, type ManifoldData, type ManifoldAnimation } from "../lib/dataClient";
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
  let manim = $state<ManifoldAnimation | null>(null); // precomputed key frames (response present)
  let animTime = 0; // continuous key-frame index; tweened toward responseStep
  let tweenRaf = 0;
  let emisNow: Float32Array | null = null; // current interpolated per-token emission (for hover)
  let containerEl: HTMLDivElement | undefined;

  const SEED = 0;
  const MARKERS = 2000; // bounded token-marker set (keeps payloads small + loads snappy)

  let renderer: THREE.WebGLRenderer | undefined;
  let scene: THREE.Scene | undefined;
  let camera: THREE.PerspectiveCamera | undefined;
  let controls: OrbitControls | undefined;
  let mesh: THREE.Mesh | undefined;
  let points: THREE.Points | undefined;
  let traj: THREE.Group | undefined;
  let surf: THREE.Group | undefined; // surface flow field (toggle)
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
      if (!points || !camera || !renderer) return;
      const strs = data?.token_strs ?? manim?.token_strs;
      if (!strs) return;
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(ndc, camera);
      // surface-field arrowheads take hover priority when shown
      if (surf) {
        const sh = raycaster.intersectObjects(surf.children, false).find((h) => h.object.userData?.dst);
        if (sh) {
          const u = sh.object.userData;
          showTip(ev, `"${u.src}" → "${u.dst}"   next-token P ${(u.p * 100).toFixed(1)}%`);
          return;
        }
      }
      const hits = raycaster.intersectObject(points);
      if (hits.length && hits[0].index != null) {
        const i = hits[0].index;
        const emis = data ? (data.token_emis?.[i] ?? 0) : (emisNow?.[i] ?? 0);
        showTip(ev, `${strs[i] ?? ""}   emission ${(emis * 100).toFixed(1)}%`);
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
    clearSurface();

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

    drawSurface(d);
  }

  function clearSurface() {
    if (surf && scene) {
      scene.remove(surf);
      surf.traverse((o: any) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
    }
    surf = undefined;
  }

  // Surface flow field: for the most likely tokens, a geodesic arrow from that token's marker
  // toward the marker of the token the model would emit NEXT — "from here, the model goes there".
  function drawSurface(d: ManifoldData) {
    clearSurface();
    if (!scene || !get(showSurface) || !d.surface_src?.length) return;
    const g = new THREE.Group();
    const amber = new THREE.Color("#ffb454");
    for (let i = 0; i < d.surface_src.length; i++) {
      const s = d.surface_src[i], t = d.surface_dst[i];
      const a = new THREE.Vector3(s[0], s[1], s[2]);
      const b = new THREE.Vector3(t[0], t[1], t[2]);
      if (a.distanceTo(b) < 1e-3) continue; // self-prediction → no arrow
      const arc = geodesic(a, b, 24).map((v) => v.multiplyScalar(1.02)); // ride just above the surface
      const lgeom = new THREE.BufferGeometry().setFromPoints(arc);
      const line = new THREE.Line(lgeom, new THREE.LineBasicMaterial({ color: amber, transparent: true, opacity: 0.85 }));
      g.add(line);
      // a cone arrowhead at the destination, oriented along the final segment
      const tip = arc[arc.length - 1], prev = arc[arc.length - 2];
      const dir = tip.clone().sub(prev).normalize();
      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(0.035, 0.11, 10),
        new THREE.MeshBasicMaterial({ color: amber }),
      );
      cone.position.copy(tip);
      cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      cone.userData = { src: d.surface_src_strs[i], dst: d.surface_dst_strs[i], p: d.surface_probs[i] };
      g.add(cone);
    }
    surf = g;
    scene.add(g);
  }

  // --- Smooth animation (response present): precomputed key frames, morphed continuously. ---
  // Build the mesh + token markers ONCE; setAnimFrame() then lerps vertex positions, warp
  // colors, and per-token emission in place every frame so the surface morphs gradually and
  // the trajectory builds in one dot at a time (no per-frame re-fetch / re-allocation).
  function buildManifoldAnim(a: ManifoldAnimation) {
    if (!scene) return;
    // clear any prior objects (static or anim)
    for (const o of [mesh, points] as (THREE.Mesh | THREE.Points | undefined)[]) {
      if (o) { scene.remove(o); o.geometry.dispose(); (o.material as THREE.Material).dispose(); }
    }
    mesh = points = undefined;
    if (traj) { scene.remove(traj); traj.traverse((o: any) => { o.geometry?.dispose?.(); o.material?.dispose?.(); }); traj = undefined; }

    const nv = a.n_vertices;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(nv * 3), 3));
    geom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(nv * 3), 3));
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(a.faces.flat()), 1));
    mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.15 }));
    scene.add(mesh);

    // Token markers (static radius-2 positions). Color is derived from aEmis IN the shader so a
    // frame update only has to touch the single aEmis attribute (cheap even at full vocab).
    const n = a.token_points.length;
    emisNow = new Float32Array(n);
    const pgeom = new THREE.BufferGeometry();
    pgeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(a.token_points.flat()), 3));
    pgeom.setAttribute("aEmis", new THREE.BufferAttribute(emisNow, 1));
    const pmat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        uSize: { value: 0.05 }, uScale: { value: pointScale() },
        uLow: { value: new THREE.Color(LOW) }, uHigh: { value: new THREE.Color(HIGH) },
      },
      vertexShader: `
        attribute float aEmis;
        uniform float uSize; uniform float uScale; uniform vec3 uLow; uniform vec3 uHigh;
        varying vec3 vColor; varying float vAlpha;
        void main() {
          vColor = mix(uLow, uHigh, aEmis);
          vAlpha = 0.10 + 0.90 * aEmis;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = uSize * (0.55 + 0.9 * aEmis) * (uScale / -mv.z);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        varying vec3 vColor; varying float vAlpha;
        void main() {
          vec2 uv = gl_PointCoord - 0.5;
          if (dot(uv, uv) > 0.25) discard;
          gl_FragColor = vec4(vColor, vAlpha);
        }`,
    });
    points = new THREE.Points(pgeom, pmat);
    scene.add(points);

    setAnimFrame(animTime);
  }

  // Lerp the surface + markers between key frames f0/f1 at continuous index t, and lay the
  // trajectory down dot-by-dot (each token reached once t passes its key frame).
  function setAnimFrame(t: number) {
    const a = manim;
    if (!a || !mesh || !points) return;
    const last = a.n_frames - 1;
    t = Math.max(0, Math.min(last, t));
    const f0 = Math.floor(t);
    const f1 = Math.min(f0 + 1, last);
    const fr = t - f0;
    const V0 = a.vertices[f0], V1 = a.vertices[f1];
    const W0 = a.warp[f0], W1 = a.warp[f1];
    const pos = (mesh.geometry.attributes.position as THREE.BufferAttribute).array as Float32Array;
    const col = (mesh.geometry.attributes.color as THREE.BufferAttribute).array as Float32Array;
    const c = new THREE.Color();
    for (let i = 0; i < a.n_vertices; i++) {
      const p0 = V0[i], p1 = V1[i];
      pos[i * 3] = p0[0] + (p1[0] - p0[0]) * fr;
      pos[i * 3 + 1] = p0[1] + (p1[1] - p0[1]) * fr;
      pos[i * 3 + 2] = p0[2] + (p1[2] - p0[2]) * fr;
      const w = (W0[i] + (W1[i] - W0[i]) * fr);
      c.copy(LOW).lerp(HIGH, Math.max(0, Math.min(1, w)));
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    }
    (mesh.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (mesh.geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    mesh.geometry.computeVertexNormals();

    const E0 = a.token_emis[f0], E1 = a.token_emis[f1];
    const ae = emisNow!;
    for (let i = 0; i < ae.length; i++) ae[i] = E0[i] + (E1[i] - E0[i]) * fr;
    (points.geometry.attributes.aEmis as THREE.BufferAttribute).needsUpdate = true;

    rebuildAnimTrajectory(t);
  }

  // Geodesic trajectory that grows continuously: `whole` completed dots plus a partial arc to
  // the next dot (so a dot is "laid down" exactly when t reaches its key frame).
  function rebuildAnimTrajectory(t: number) {
    if (!scene || !manim) return;
    if (traj) { scene.remove(traj); traj.traverse((o: any) => { o.geometry?.dispose?.(); o.material?.dispose?.(); }); traj = undefined; }
    const tpts = manim.traj_points;
    if (!tpts.length) return;
    const whole = Math.min(Math.floor(t), tpts.length); // dots fully laid down
    const fr = Math.min(t, tpts.length) - whole;        // growth toward the next dot
    const pts = tpts.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    const g = new THREE.Group();
    const line: THREE.Vector3[] = [];
    // completed great-circle segments between laid-down dots
    for (let i = 0; i < whole - 1; i++) {
      const seg = geodesic(pts[i], pts[i + 1], 28).map((v) => v.multiplyScalar(1.01));
      line.push(...(i > 0 ? seg.slice(1) : seg));
    }
    // partial arc growing toward the next (not-yet-reached) dot
    if (whole >= 1 && whole < tpts.length && fr > 0) {
      const seg = geodesic(pts[whole - 1], pts[whole], 28).map((v) => v.multiplyScalar(1.01));
      const partial = seg.slice(0, Math.max(2, Math.ceil(seg.length * fr)));
      line.push(...(whole - 1 > 0 ? partial.slice(1) : partial));
    }
    if (line.length >= 2) {
      const lgeom = new THREE.BufferGeometry().setFromPoints(line);
      g.add(new THREE.Line(lgeom, new THREE.LineBasicMaterial({ color: 0x5be0b0, transparent: true, opacity: 0.95 })));
    }
    for (let i = 0; i < whole; i++) {
      const isCur = i === whole - 1;
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(isCur ? 0.08 : 0.05, 14, 14),
        new THREE.MeshBasicMaterial({ color: isCur ? 0xffffff : 0x5be0b0 }),
      );
      m.position.copy(pts[i].clone().multiplyScalar(1.01));
      g.add(m);
    }
    traj = g;
    scene.add(g);
  }

  // Ease the continuous frame index toward a target key frame (smooth scrub / ▶ Play).
  function tweenTo(target: number) {
    cancelAnimationFrame(tweenRaf);
    const from = animTime;
    const dur = 650;
    let start = -1;
    const easeInOutQuad = (x: number) => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2);
    const step = (ts: number) => {
      if (start < 0) start = ts;
      const k = Math.min(1, (ts - start) / dur);
      animTime = from + (target - from) * easeInOutQuad(k);
      setAnimFrame(animTime);
      if (k < 1) tweenRaf = requestAnimationFrame(step);
    };
    tweenRaf = requestAnimationFrame(step);
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
    cancelAnimationFrame(tweenRaf);
    resizeObs?.disconnect();
    controls?.dispose();
    if (renderer) {
      renderer.domElement.remove();
      renderer.dispose();
    }
    renderer = scene = camera = controls = mesh = points = traj = surf = undefined;
  }

  // Reload whenever the model / context / temperature / response text changes (NOT on step —
  // a response step just scrubs the already-loaded animation; see the tween effect below).
  $effect(() => {
    const m = $modelId;
    const pfx = $prefixText;
    const temp = $temperature;
    const resp = $responseText;
    const width = $rbfWidth;
    const rn = $refreshNonce;
    const force = rn !== lastRefresh;
    lastRefresh = rn;
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(() => void load(m, pfx, temp, resp, width, force), force ? 0 : 350);
    return () => {
      if (debounce) clearTimeout(debounce);
    };
  });

  // A response step change tweens the loaded key-frame animation smoothly to that frame.
  $effect(() => {
    const step = $responseStep;
    if (manim) tweenTo(Math.min(step, manim.n_frames - 1));
  });

  // Toggle the surface flow field overlay without recomputing (static manifold only).
  $effect(() => {
    const on = $showSurface;
    void on;
    if (data && !manim) drawSurface(data);
  });

  async function load(m: string, pfx: string, temp: number, resp: string, width: number, force = false) {
    const my = ++runId;
    error = "";
    loading = true;
    progress = 0;
    progressMsg = force ? "recomputing…" : "starting…";
    try {
      // A bounded reference set (the warp only uses the top tokens anyway): thousands of
      // markers, not the whole vocab — far smaller payloads, so loads/refreshes stay snappy.
      const params = { temperature: temp, seed: SEED, reference_set_size: MARKERS, width };
      const hasResp = resp.trim().length > 0;
      const artifact = hasResp ? "manifold_animation" : "manifold";
      const inputs: Record<string, unknown> = { prefix_text: pfx };
      if (hasResp) inputs.response_text = resp;
      if (!force) {
        await client.ensureArtifact(artifact, m, params, inputs, (p, msg) => {
          if (my === runId) { progress = p; progressMsg = msg; }
        });
      }
      if (my !== runId) return;
      const forceArg = force ? { force: true } : {};
      if (hasResp) {
        // Precomputed key frames → smooth morph + gradual trajectory build-in.
        manim = await client.getManifoldAnimation(m, { prefix_text: pfx, response_text: resp, ...params, ...forceArg });
        if (my !== runId) return;
        data = null;
        animTime = Math.min(get(responseStep), manim.n_frames - 1);
        buildManifoldAnim(manim);
      } else {
        manim = null;
        data = await client.getManifold(m, { prefix_text: pfx, ...params, ...forceArg });
        if (my !== runId) return;
        buildMesh(data, 0);
      }
    } catch (e: any) {
      if (my === runId) error = `${e.type ?? "Error"}: ${e.message ?? e}`;
    } finally {
      if (my === runId) loading = false;
    }
  }

  // Export: interpolate SUB sub-frames per key-frame transition for a watchable morph; settle a
  // couple of rAFs so the WebGL buffer is captured after it actually draws.
  const SUB = 12;
  async function renderFrame(i: number) {
    if (manim) {
      animTime = Math.min(i / SUB, manim.n_frames - 1);
      setAnimFrame(animTime);
    }
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  }
  const exportAnim = {
    total: () => (manim ? (manim.n_frames - 1) * SUB : 0),
    fps: 24,
    renderFrame,
    restore: () => renderFrame(Math.min(get(responseStep), manim ? manim.n_frames - 1 : 0) * SUB),
  };
</script>

<section class="viz panel" data-testid="viz-manifold" data-ready={data || manim ? 1 : 0}>
  <header>
    <div>
      <h2>Reachable "thoughts" as a manifold</h2>
      <p class="sub">A unit sphere warped (RBF + ARAP) toward likely next tokens. <b>Bulges = high emission probability; dots = tokens on the radius-2 sphere.</b> Tune the bump width, or enable the <b>surface flow field</b> to see where each likely token would lead next. Drag to rotate, scroll to zoom, hover a dot or arrow. The response traces a geodesic across the sphere.</p>
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
  {:else if manim}
    <p class="caption">
      {manim.n_vertices} mesh vertices · {manim.n_frames} key frames morphing as the response unfolds: {manim.trajectory_token_strs.map((t) => t.trim() || "∅").join(" → ")}
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
