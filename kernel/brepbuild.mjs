// BUILD A HALF-EDGE B-REP FROM A PLAIN SET OF FACE BOUNDARY LOOPS
// ================================================================
// kernel/brep.mjs is the topology LAYER — entities, Euler operators,
// validity, the Euler-Poincaré invariant. It deliberately never looks at
// geometry and has no way in from an ordinary pile of faces. This module
// is that way in: given N face boundary loops as plain 3D polygons (the
// shape every panel container and PolySurface in this app can already
// produce from its own corner points), it welds coincident corners into
// shared VERTICES, matches directed edge-uses into shared EDGES, and
// emits a real half-edge solid in exactly the shape kernel/brep.mjs
// documents — so `validateBrep`/`eulerCharacteristic` can then be run
// against it as a genuinely independent check.
//
// WHY THIS IS WORTH HAVING AT ALL, over the geometric edge matcher this
// app already ships (findPolySurfaceEdges + curvesCoincident): that
// matcher answers ONE question per edge pair — "do these two curves
// occupy the same space?" — and stops. It cannot answer "is this a
// closed solid", "is this orientable", "is any edge shared by three
// faces", or "what is the genus", because none of those are properties
// of an edge pair; they are properties of the whole welded topology.
// Half-edge + Euler-Poincaré answers all four, and answers them by
// counting, not by hoping. A student who joins five faces of a box and
// is told "closed solid" by a tool that only ever compared edges
// pairwise has been misled; V - E + F is what actually knows.
//
// ORIENTATION IS SOLVED HERE, NOT REQUIRED FROM THE CALLER. Face loops
// arriving from a primitive's own panel list are wound however that
// primitive's construction happened to wind them, which is not
// guaranteed consistent (a Box's own six bilinear panels are authored
// corner-by-corner, never checked against each other). A half-edge
// solid REQUIRES consistent winding — each edge traversed once in each
// direction — so this module propagates an orientation outward from the
// first face by breadth-first search, reversing whichever loops
// disagree, and reports how many it had to flip. A face set that cannot
// be consistently oriented at all (a Möbius-like configuration) is
// refused BY NAME rather than silently producing a solid whose winding
// lies.
//
// SCOPE, named rather than silently missing:
//   - Faces are SIMPLE loops. A face with an inner ring (a hole) is not
//     expressible here; `boundaryLoops` will report the extra loop and
//     `buildBrepSolid` refuses. Rings are a real kernel/brep.mjs
//     capability (KEMR/MEKR) that this builder does not reach.
//   - Geometry is carried only as vertex POSITIONS. Face surfaces and
//     per-half-edge pcurves are left null for the caller to attach via
//     kernel/brep.mjs's own `attachSurface`/`assignPlanarPcurves`.
//   - Genus is DERIVED, not tracked: for a closed orientable manifold
//     with S shells, V - E + F = 2(S - G), so G = S - (V-E+F)/2. That is
//     a real measurement of the built topology, not an assumption — and
//     it is what makes `validateBrep`'s own invariant check meaningful
//     rather than circular (an operator-built solid tracks genus as it
//     goes; a welded one must measure it).
//
// OUTWARDNESS IS A SEPARATE FACT FROM CONSISTENCY, and is settled here
// too. `orientLoops` above guarantees only that neighbors AGREE; which
// way the whole component ends up facing is decided by whichever face
// happened to seed its own traversal, so a shell can come back globally
// inside-out with every check above still passing. Chi, the naked-edge
// count and even a volume MAGNITUDE all read perfectly on an inside-out
// shell, while every downstream consumer — a renderer, an exporter, a
// Thicken, a mass-property measurement — reads its normals as if they
// meant something. So each shell is classified and, where it disagrees,
// reversed: see SHELL ORIENTATION below.

const DEFAULT_WELD_TOL = 1e-6;

// ---------------------------------------------------------------------
// WELDING
// ---------------------------------------------------------------------
// Coincident corners become ONE vertex. A plain spatial hash on the
// tolerance grid, probing the 27 neighboring cells so a pair straddling
// a cell boundary still welds — the same technique kernel/flatten.mjs's
// own weld already uses, restated here rather than imported because that
// one welds a triangulation's own index arrays, not free point lists.
export function weldPoints(pointLists, tol = DEFAULT_WELD_TOL) {
  const cell = Math.max(tol, 1e-12) * 2;
  const buckets = new Map();
  const points = [];
  const key = (i, j, k) => `${i},${j},${k}`;
  const indexOf = (p) => {
    const ci = Math.floor(p[0] / cell), cj = Math.floor(p[1] / cell), ck = Math.floor(p[2] / cell);
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
      const bucket = buckets.get(key(ci + a, cj + b, ck + c));
      if (!bucket) continue;
      for (const idx of bucket) {
        const q = points[idx];
        if (Math.abs(q[0] - p[0]) <= tol && Math.abs(q[1] - p[1]) <= tol && Math.abs(q[2] - p[2]) <= tol) return idx;
      }
    }
    const idx = points.length;
    points.push([p[0], p[1], p[2]]);
    const k0 = key(ci, cj, ck);
    if (!buckets.has(k0)) buckets.set(k0, []);
    buckets.get(k0).push(idx);
    return idx;
  };
  const loops = pointLists.map((list) => {
    const out = [];
    for (const p of list) {
      const idx = indexOf(p);
      if (out.length && out[out.length - 1] === idx) continue; // consecutive duplicate — a degenerate zero-length edge, dropped
      out.push(idx);
    }
    while (out.length > 1 && out[0] === out[out.length - 1]) out.pop(); // an explicitly-closed input loop repeats its first point; the cycle is implicit here
    return out;
  });
  return { points, loops };
}

// ---------------------------------------------------------------------
// ORIENTATION
// ---------------------------------------------------------------------
// Two faces sharing an edge are consistently oriented when they traverse
// that edge in OPPOSITE directions — the combinatorial statement of
// "their normals point the same way out". This propagates that from the
// first face of each connected component outward.
//
// Returns `{ ok, loops, flipped, components, reason }`. `loops` is a NEW
// array (inputs are never mutated). `components` labels each face with
// its own connected component index, which is what later becomes a SHELL.
export function orientLoops(loops) {
  const n = loops.length;
  const out = loops.map((l) => l.slice());
  const flippedFlag = new Array(n).fill(false);
  // undirected edge -> [{face, from, to}, ...]
  const uses = new Map();
  const ukey = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
  for (let f = 0; f < n; f++) {
    const l = out[f];
    for (let i = 0; i < l.length; i++) {
      const a = l[i], b = l[(i + 1) % l.length];
      const k = ukey(a, b);
      if (!uses.has(k)) uses.set(k, []);
      uses.get(k).push({ face: f, a, b });
    }
  }
  const comp = new Array(n).fill(-1);
  let componentCount = 0;
  let flipped = 0;
  for (let seed = 0; seed < n; seed++) {
    if (comp[seed] >= 0) continue;
    const c = componentCount++;
    comp[seed] = c;
    const queue = [seed];
    while (queue.length) {
      const f = queue.shift();
      const l = out[f];
      for (let i = 0; i < l.length; i++) {
        const a = l[i], b = l[(i + 1) % l.length];
        for (const u of uses.get(ukey(a, b))) {
          if (u.face === f) continue;
          // `u` was recorded against the face's ORIGINAL winding, so read
          // the neighbor's CURRENT direction off its live loop instead.
          const nb = u.face;
          const dir = loopEdgeDirection(out[nb], a, b);
          if (dir === 0) continue; // that edge is no longer on this neighbor (shouldn't happen — defensive)
          if (comp[nb] < 0) {
            comp[nb] = c;
            if (dir === +1) { out[nb].reverse(); flippedFlag[nb] = true; flipped++; } // same direction as us: disagrees, flip it
            queue.push(nb);
          } else if (dir === +1) {
            return { ok: false, loops: out, flipped, components: comp, componentCount, reason: 'non-orientable' };
          }
        }
      }
    }
  }
  return { ok: true, loops: out, flipped, flippedFaces: flippedFlag, components: comp, componentCount, reason: null };
}

// +1 if `loop` traverses a->b, -1 if it traverses b->a, 0 if neither.
function loopEdgeDirection(loop, a, b) {
  for (let i = 0; i < loop.length; i++) {
    const x = loop[i], y = loop[(i + 1) % loop.length];
    if (x === a && y === b) return +1;
    if (x === b && y === a) return -1;
  }
  return 0;
}

// ---------------------------------------------------------------------
// SHELL ORIENTATION
// ---------------------------------------------------------------------
// A closed shell's OUTWARDNESS is decided by CONTAINMENT, never by size.
// The standard B-rep convention: a shell bounding material from outside
// faces outward (positive signed volume, by the divergence theorem); a
// VOID shell inside that material faces INTO the void (negative), which
// is exactly what makes a hollow solid's total volume come out as the
// plain sum of its shells rather than needing a per-shell sign table.
//
// A LARGEST-VOLUME RULE WOULD BE WRONG, and the counter-example is
// ordinary rather than exotic: a boolean difference where a slab passes
// clean through a prism genuinely leaves two SEPARATE solids in one
// result — both outer, neither a void — and the smaller one would be
// flipped into a void by any size-based rule. Meanwhile a Shell's own
// inner wall is a real void whose bbox sits strictly inside its outer
// wall's. Only containment tells those two apart.
//
// NESTING DEPTH decides it in general, not just one level: a shell
// contained by an EVEN number of other shells bounds material (outward),
// an ODD number means it bounds a void (inward). That handles a solid
// sitting inside a cavity of another solid without a special case.

// Signed volume of one shell, by the divergence theorem over tetrahedra
// fanned from the ORIGIN. The origin needs no relationship to the shell:
// a tetrahedron behind it contributes negatively and the far face's own
// contribution cancels it, so the total is the enclosed volume wherever
// the shell sits.
function shellSignedVolume(loops, faceIdx, points) {
  let v = 0;
  for (const f of faceIdx) {
    const l = loops[f];
    const a = points[l[0]];
    for (let i = 1; i + 1 < l.length; i++) {
      const b = points[l[i]], c = points[l[i + 1]];
      v += (a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
    }
  }
  return v;
}

// Deliberately irrational-ish directions, so a ray is unlikely to run
// along an edge or through a vertex of an axis-aligned shell — the one
// configuration a crossing count cannot resolve. Several are tried, and
// a near-degenerate hit disqualifies that direction's answer entirely
// rather than being rounded into a guess.
const SHELL_RAY_DIRS = [
  [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
  [0.8017837257372732, 0.5345224838248488, 0.2672612419124244],
  [-0.3713906763541037, 0.7427813527082074, 0.5570860145311556],
  [0.4242640687119285, -0.5656854249492380, 0.7071067811865476],
  [-0.6396021490668313, -0.2558408596267325, 0.7248824356090421],
];

// SIGNED crossings, not parity. A face loop is fan-triangulated, and a
// fan over a non-convex (or non-planar) loop genuinely produces
// triangles that escape the loop — a plain even/odd count would be
// corrupted by them. Signed crossings are not: an escaped triangle is
// traversed the other way round by its neighbors and cancels exactly,
// so the sum is the winding number regardless of how the loop was
// triangulated. Inside reads +/-1 (sign following the shell's own
// current orientation), outside reads 0.
function shellWindingAt(point, loops, faceIdx, points, dir, tol) {
  let winding = 0;
  for (const f of faceIdx) {
    const l = loops[f];
    const a = points[l[0]];
    for (let i = 1; i + 1 < l.length; i++) {
      const b = points[l[i]], c = points[l[i + 1]];
      const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const pv = [
        dir[1] * e2[2] - dir[2] * e2[1],
        dir[2] * e2[0] - dir[0] * e2[2],
        dir[0] * e2[1] - dir[1] * e2[0],
      ];
      const det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
      const scale = Math.hypot(...e1) * Math.hypot(...e2);
      if (!(Math.abs(det) > 1e-12 * Math.max(scale, 1e-12))) continue; // parallel
      const inv = 1 / det;
      const tv = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];
      const u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * inv;
      if (u < -1e-9 || u > 1 + 1e-9) continue;
      const qv = [
        tv[1] * e1[2] - tv[2] * e1[1],
        tv[2] * e1[0] - tv[0] * e1[2],
        tv[0] * e1[1] - tv[1] * e1[0],
      ];
      const v = (dir[0] * qv[0] + dir[1] * qv[1] + dir[2] * qv[2]) * inv;
      if (v < -1e-9 || u + v > 1 + 1e-9) continue;
      const t = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv;
      if (t <= tol) continue; // behind the point, or the point is on this face
      // Grazing an edge or a vertex, or a hit essentially at the point
      // itself: this direction cannot be trusted, so refuse it outright
      // rather than count a crossing that may or may not be real.
      if (u < 1e-7 || v < 1e-7 || u + v > 1 - 1e-7) return null;
      // det = -dot(dir, normal), so a face the ray EXITS through has det < 0.
      winding += det > 0 ? -1 : 1;
    }
  }
  return winding;
}

// Is `point` inside this shell? Returns true/false, or null when every
// tried direction grazed geometry — undecidable, reported rather than
// guessed at.
function pointInsideShell(point, loops, faceIdx, points, tol) {
  for (const dir of SHELL_RAY_DIRS) {
    const w = shellWindingAt(point, loops, faceIdx, points, dir, tol);
    if (w === null) continue;
    return Math.abs(w) >= 0.5;
  }
  return null;
}

/**
 * Reverse whichever shells face the wrong way, so an outer shell faces
 * outward and a void faces into its own cavity.
 *
 * `loops` is mutated in place (it is already `orientLoops`' own fresh
 * copy by the time this runs). Returns `{ shellKinds, flippedShells,
 * uncertain }` — `shellKinds[c]` is 'outer' | 'void' | null, the null
 * meaning containment could not be decided for that shell, in which case
 * its winding is left EXACTLY as it arrived rather than flipped on a
 * guess. `uncertain` counts those.
 */
export function orientShellsOutward(loops, components, componentCount, points, tol = DEFAULT_WELD_TOL) {
  const faceIdx = Array.from({ length: componentCount }, () => []);
  for (let f = 0; f < components.length; f++) faceIdx[components[f]].push(f);

  const shellKinds = new Array(componentCount).fill(null);
  let flippedShells = 0, uncertain = 0;

  for (let c = 0; c < componentCount; c++) {
    if (!faceIdx[c].length) continue;
    // A vertex OF this shell is the representative point. It sits on this
    // shell's own surface, which is fine: every containment test below is
    // against a DIFFERENT shell, and two distinct shells of a valid solid
    // do not touch.
    const rep = points[loops[faceIdx[c][0]][0]];
    let depth = 0, decided = true;
    for (let o = 0; o < componentCount; o++) {
      if (o === c || !faceIdx[o].length) continue;
      const inside = pointInsideShell(rep, loops, faceIdx[o], points, tol);
      if (inside === null) { decided = false; break; }
      if (inside) depth++;
    }
    if (!decided) { uncertain++; continue; }

    const isVoid = (depth % 2) === 1;
    shellKinds[c] = isVoid ? 'void' : 'outer';
    const vol = shellSignedVolume(loops, faceIdx[c], points);
    const wantsPositive = !isVoid;
    if (Math.abs(vol) < 1e-15) { shellKinds[c] = null; uncertain++; continue; }
    if ((vol > 0) !== wantsPositive) {
      for (const f of faceIdx[c]) loops[f].reverse();
      flippedShells++;
    }
  }
  return { shellKinds, flippedShells, uncertain };
}

// ---------------------------------------------------------------------
// BOUNDARY LOOPS OF A FACE SET
// ---------------------------------------------------------------------
// Every directed edge-use that has no opposite-direction partner is a
// NAKED boundary; chained head-to-tail these are the outline of the face
// set. Used two ways: to report an open shell honestly, and — by the app
// layer — to find the single outer boundary of a group of coplanar
// panels so they can be merged into one real face.
export function boundaryLoops(loops) {
  const dir = new Map(); // "a|b" -> count
  const dkey = (a, b) => `${a}|${b}`;
  for (const l of loops) for (let i = 0; i < l.length; i++) {
    const k = dkey(l[i], l[(i + 1) % l.length]);
    dir.set(k, (dir.get(k) ?? 0) + 1);
  }
  const naked = [];
  for (const [k, count] of dir) {
    const [a, b] = k.split('|').map(Number);
    const back = dir.get(dkey(b, a)) ?? 0;
    for (let i = 0; i < count - back; i++) naked.push([a, b]);
  }
  const outFrom = new Map();
  naked.forEach(([a], i) => { if (!outFrom.has(a)) outFrom.set(a, []); outFrom.get(a).push(i); });
  const used = new Array(naked.length).fill(false);
  const chains = [];
  for (let i = 0; i < naked.length; i++) {
    if (used[i]) continue;
    used[i] = true;
    const start = naked[i][0];
    const vertices = [start];
    let cur = naked[i][1];
    let closed = false;
    for (let guard = 0; guard <= naked.length; guard++) {
      if (cur === start) { closed = true; break; }
      const cands = outFrom.get(cur) || [];
      const nextIdx = cands.find((j) => !used[j]);
      if (nextIdx === undefined) break;
      used[nextIdx] = true;
      vertices.push(cur);
      cur = naked[nextIdx][1];
    }
    chains.push({ closed, vertices });
  }
  return { nakedEdgeCount: naked.length, nakedEdges: naked, loops: chains };
}

// ---------------------------------------------------------------------
// THE BUILDER
// ---------------------------------------------------------------------
// Emits a solid in exactly kernel/brep.mjs's own documented entity shape
// (see its ENTITIES section) so that module's `validateBrep`,
// `eulerCharacteristic` and `brepFingerprint` all apply unchanged. This
// module deliberately does NOT validate its own output — an independent
// checker is the whole point.
export function buildBrepSolid(faceLoops, opts = {}) {
  const tol = opts.tolerance ?? DEFAULT_WELD_TOL;
  const welded = weldPoints(faceLoops, tol);
  const degenerate = [];
  const cleanLoops = [];
  for (let i = 0; i < welded.loops.length; i++) {
    const l = welded.loops[i];
    if (l.length < 3) { degenerate.push(i); continue; }
    if (new Set(l).size !== l.length) {
      return report(false, 'repeated-vertex-in-face', { faceIndex: i, points: welded.points });
    }
    cleanLoops.push(l);
  }
  if (degenerate.length) {
    return report(false, 'degenerate-face', { degenerateFaces: degenerate, points: welded.points });
  }
  if (!cleanLoops.length) return report(false, 'no-faces', { points: welded.points });

  // Edge census FIRST, on the raw loops. Deliberately before orientation:
  // an edge shared by three or more faces makes "which way should the
  // neighbor be wound" ill-posed, so an unfiltered orientation pass
  // would report a confusing 'non-orientable' for what is really a
  // non-manifold edge. Undirected counts are orientation-independent, so
  // this ordering costs nothing and names the real fault.
  const ukey = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
  const census = new Map();
  for (let f = 0; f < cleanLoops.length; f++) {
    const l = cleanLoops[f];
    for (let i = 0; i < l.length; i++) {
      const a = l[i], b = l[(i + 1) % l.length];
      const k = ukey(a, b);
      if (!census.has(k)) census.set(k, []);
      census.get(k).push({ face: f, a, b });
    }
  }
  const nonManifold = [];
  const naked = [];
  for (const [k, list] of census) {
    if (list.length > 2) nonManifold.push({ key: k, uses: list.length, a: list[0].a, b: list[0].b });
    else if (list.length === 1) naked.push({ key: k, a: list[0].a, b: list[0].b });
  }
  const V = welded.points.length, E = census.size, F = cleanLoops.length;
  const chi = V - E + F;
  const stats = {
    V, E, F, chi,
    nakedEdgeCount: naked.length,
    nonManifoldEdgeCount: nonManifold.length,
    flippedFaces: 0,
    shellCount: 0,
  };
  if (nonManifold.length) {
    return report(false, 'non-manifold-edge', { points: welded.points, stats, nonManifold, naked });
  }

  const oriented = orientLoops(cleanLoops);
  stats.flippedFaces = oriented.flipped;
  stats.shellCount = oriented.componentCount;
  if (!oriented.ok) {
    return report(false, oriented.reason, { points: welded.points, stats });
  }
  const loops = oriented.loops;

  if (naked.length) {
    return report(false, 'open-shell', { points: welded.points, stats, naked, nakedEdgePoints: naked.map((e) => [welded.points[e.a], welded.points[e.b]]) });
  }

  // Every shell is closed by this point, which is what makes the
  // containment test below meaningful — an open shell has no inside.
  const shellOrient = orientShellsOutward(loops, oriented.components, oriented.componentCount, welded.points, tol);
  stats.flippedShells = shellOrient.flippedShells;
  stats.shellOrientationUncertain = shellOrient.uncertain;

  // --- assemble the half-edge structure -------------------------------
  const solid = { id: 0, kind: 'solid', nextId: 0, vertices: [], edges: [], shells: [], genus: 0, name: opts.name ?? null };
  const newId = () => solid.nextId++;
  solid.id = newId();
  const shells = [];
  for (let c = 0; c < oriented.componentCount; c++) {
    const sh = { id: newId(), kind: 'shell', solid, faces: [], shellKind: shellOrient.shellKinds[c] };
    solid.shells.push(sh);
    shells.push(sh);
  }
  const verts = welded.points.map((p) => {
    const v = { id: newId(), kind: 'vertex', solid, point: [p[0], p[1], p[2]], halfEdge: null };
    solid.vertices.push(v);
    return v;
  });
  const edgeByKey = new Map();
  const faceObjs = [];
  for (let f = 0; f < loops.length; f++) {
    const shell = shells[oriented.components[f]];
    const face = { id: newId(), kind: 'face', solid, shell, loops: [], surface: null };
    shell.faces.push(face);
    const loop = { id: newId(), kind: 'loop', solid, face, halfEdge: null };
    face.loops.push(loop);
    const l = loops[f];
    const hes = [];
    for (let i = 0; i < l.length; i++) {
      const a = l[i], b = l[(i + 1) % l.length];
      const k = ukey(a, b);
      let edge = edgeByKey.get(k);
      if (!edge) {
        edge = { id: newId(), kind: 'edge', solid, halfEdges: [], curve3d: null, tolerance: tol };
        edgeByKey.set(k, edge);
        solid.edges.push(edge);
      }
      const he = { id: newId(), kind: 'halfedge', solid, vertex: verts[a], edge, twin: null, next: null, prev: null, loop, pcurve: null };
      edge.halfEdges.push(he);
      hes.push(he);
      if (!verts[a].halfEdge) verts[a].halfEdge = he;
    }
    for (let i = 0; i < hes.length; i++) {
      hes[i].next = hes[(i + 1) % hes.length];
      hes[(i + 1) % hes.length].prev = hes[i];
    }
    loop.halfEdge = hes[0];
    faceObjs.push(face);
  }
  for (const edge of solid.edges) {
    const [h1, h2] = edge.halfEdges;
    h1.twin = h2; h2.twin = h1;
  }
  // DERIVED, not assumed (see the header note): with S shells and no
  // rings, V - E + F = 2(S - G) fixes G. Stored so kernel/brep.mjs's own
  // invariant check is a real cross-check of the counts rather than a
  // tautology against a genus nobody measured.
  const genusTimesTwo = 2 * solid.shells.length - chi;
  solid.genus = genusTimesTwo / 2;
  stats.genus = solid.genus;
  return report(true, null, { points: welded.points, stats, solid, faces: faceObjs, loops });
}

function report(ok, reason, extra) {
  return { ok, reason, ...extra };
}

// A one-line, student-readable verdict for a `buildBrepSolid` result —
// kept here rather than in the app so the wording of a topology failure
// lives beside the code that decides it.
export function brepVerdict(res) {
  if (res.ok) {
    const s = res.stats;
    return `CLOSED SOLID — Euler characteristic chi = V - E + F = ${s.V} - ${s.E} + ${s.F} = ${s.chi}` +
      `${s.shellCount > 1 ? `, ${s.shellCount} shells` : ''}${s.genus ? `, genus ${s.genus}` : ''}, 0 naked edges`;
  }
  switch (res.reason) {
    case 'open-shell':
      return `OPEN SHELL — not a closed solid: ${res.stats.nakedEdgeCount} naked edge${res.stats.nakedEdgeCount === 1 ? '' : 's'} ` +
        `(V ${res.stats.V}, E ${res.stats.E}, F ${res.stats.F}, chi ${res.stats.chi})`;
    case 'non-manifold-edge':
      return `NON-MANIFOLD — ${res.stats.nonManifoldEdgeCount} edge${res.stats.nonManifoldEdgeCount === 1 ? ' is' : 's are'} shared by more than two faces, so this is not a solid`;
    case 'non-orientable':
      return 'NOT ORIENTABLE — these faces cannot all be wound consistently, so no inside/outside is defined';
    case 'repeated-vertex-in-face':
      return `DEGENERATE — face ${res.faceIndex} visits the same corner twice`;
    case 'degenerate-face':
      return `DEGENERATE — ${res.degenerateFaces.length} face${res.degenerateFaces.length === 1 ? '' : 's'} collapsed to fewer than 3 distinct corners`;
    case 'no-faces':
      return 'NO FACES — nothing to build a topology from';
    default:
      return `TOPOLOGY UNAVAILABLE — ${res.reason}`;
  }
}
