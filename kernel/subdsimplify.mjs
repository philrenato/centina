// SUPERB SIMPLIFY — taking a Catmull-Clark control cage DOWN a level, the
// direction subdivideCatmullClark (kernel/subd.mjs) does not go. KERNEL ONLY:
// plain cage in, plain cage out, exactly like every other kernel/*.mjs.
//
// A CAGE is the same plain data subd.mjs defines:
//   { vertices: [[x,y,z], ...], faces: [[a,b,c,d], ...], creases: { "i_j": w } }
//
// TWO TIERS, because "coarsen this cage" is two different questions:
//
//   TIER 1 — UN-SUBDIVIDE (unsubdivideCage). A cage that IS one Catmull-Clark
//     pass of some coarser cage is taken back to that coarser cage EXACTLY,
//     creases included. Every candidate answer is checked by re-subdividing it
//     with the shipped subdivider and comparing to the input, so the tier
//     either returns a cage whose subdivision is the input to floating-point
//     tolerance, or it REFUSES BY NAME. It never returns a near-miss.
//
//   TIER 2 — POLYCHORD COLLAPSE (polychordsOf / collapsePolychord). For every
//     other quad cage: a polychord is the maximal strip of quads reached by
//     walking through opposite edges (Daniels, Silva, Shepherd & Cohen,
//     "Quadrilateral Mesh Simplification", ACM TOG 27(5), SIGGRAPH Asia 2008).
//     Collapsing one welds the two ends of every edge the strip crosses and
//     deletes the strip's faces; every neighbouring quad keeps four corners by
//     construction, so the result is still all quads.
//
// THE INVARIANT BOTH TIERS ARE HELD TO — checkSimplifyInvariants below, and
// it is the reason this module exists rather than a face-count-driven
// decimator: the result must be ALL QUADS, CLOSED IF THE INPUT WAS CLOSED,
// FREE OF INTERIOR VERTICES BELOW VALENCE 3 (a boundary vertex at valence 2 is
// the ordinary corner of an open grid, not a pinch — the two populations carry
// different floors and cageInvariants reports them apart), and of the SAME
// EULER CHARACTERISTIC. A cage
// that fails any of those is one subdToPatches (kernel/subdlimit.mjs) cannot
// turn into a surface, so producing it is worse than refusing.
//
// WHY EXACTNESS IS ACHIEVABLE AT ALL, and where it stops:
//
//   Catmull-Clark is LINEAR in the vertex positions once the topology and the
//   crease weights are fixed (every branch weight in computeEdgePoint /
//   computeVertexPoint / computeFacePoint is position-independent), so the
//   coarse positions are the solution of a linear system. This module never
//   forms that system as a matrix. It uses the two rules whose right-hand
//   sides are already known from the fine cage alone:
//     - a FACE POINT is a vertex of the fine cage, so every coarse face
//       centroid is known outright;
//     - an EDGE POINT therefore gives the SUM of the two coarse vertices it
//       came from: (Pa+Pb) = 2e for a sharp/boundary edge, and 4e - f0 - f1
//       for a smooth one.
//   Propagating those pair sums along a spanning tree of the coarse edge graph
//   pins every coarse vertex up to ONE unknown 3-vector x per component, with
//   alternating sign: P_i = c_i + sigma_i x. An odd cycle in the coarse graph
//   pins x outright. When the coarse graph is bipartite, x is fitted by
//   re-subdividing three unit displacements with the shipped subdivider — no
//   second implementation of the subdivision rules exists in this file.
//
//   ⚠ THAT FIT IS SINGULAR FOR A CAGE WHOSE COARSE GRAPH IS BIPARTITE AND ALL
//   VALENCE 3 — which is exactly the 8-vertex cube. The null direction is
//   REAL, not ill-conditioning: displacing the four corners of one inscribed
//   tetrahedron by +d and the other four by -d leaves the subdivided cage
//   identical to floating point, because the vertex rule's own coefficient on
//   P is (n-3)/n, which is zero at valence 3, and the two colours cancel in
//   every other rule. No solver recovers a unique answer there, because there
//   is not one; test/subdsimplify.test.mjs measures the two cages and their
//   one subdivision. The family is reported through `unique: false`, and the
//   member returned is the one closest to the fine cage's own vertex points,
//   which is the true cube for a symmetric one.
//
//   The other hard limit is a CREASE OF WEIGHT <= 1. subdivideCatmullClark
//   decrements every crease weight by 1 and drops it at 0, so a coarse weight
//   of 1 or less leaves NO entry in the fine cage's creases map. A weight of
//   exactly 1 is still recoverable — it is fully sharp, so its geometry is
//   distinguishable, and this module retries with the smooth-vs-sharp
//   hypothesis inverted for the edges the first solve could not explain. A
//   weight strictly between 0 and 1 is a partial blend that leaves no trace at
//   all and is REFUSED, by name, rather than returned as a smooth cage.

import { subdivideCatmullClark, buildTopology, edgeKey, creaseWeight } from './subd.mjs';
import { vertexLimitPosition } from './subdlimit.mjs';
import { add, sub, scale, dot } from './vec3.mjs';

// ---------------------------------------------------------------------------
// SHARED SMALL PARTS
// ---------------------------------------------------------------------------

const ZERO = [0, 0, 0];

function refuse(reason, message, extra = {}) {
  return { ok: false, reason, message, ...extra };
}

// Bounding-box diagonal — the scale every relative tolerance in this module is
// measured against. A degenerate (single-point) cage reports 1 so a relative
// tolerance never divides by zero.
export function cageExtent(cage) {
  if (!cage.vertices.length) return 1;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const v of cage.vertices) {
    for (let k = 0; k < 3; k++) {
      if (v[k] < lo[k]) lo[k] = v[k];
      if (v[k] > hi[k]) hi[k] = v[k];
    }
  }
  const d = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  return d > 0 ? d : 1;
}

// STRUCTURAL REPORT — the numbers every gate in this module and its tests
// assert on. `euler` counts only vertices some face actually uses, so a cage
// carrying stray unreferenced points still reports the characteristic of the
// surface it describes (the stray points are reported separately).
export function cageInvariants(cage) {
  const edges = new Map();
  let degenerateFaceCount = 0;
  const faceSizes = {};
  cage.faces.forEach((f, fi) => {
    faceSizes[f.length] = (faceSizes[f.length] || 0) + 1;
    if (new Set(f).size !== f.length) degenerateFaceCount++;
    for (let k = 0; k < f.length; k++) {
      const a = f[k];
      const b = f[(k + 1) % f.length];
      if (a === b) continue;
      const key = edgeKey(a, b);
      let rec = edges.get(key);
      if (!rec) { rec = { v0: Math.min(a, b), v1: Math.max(a, b), faces: [] }; edges.set(key, rec); }
      rec.faces.push(fi);
    }
  });
  const valence = new Map();
  for (const rec of edges.values()) {
    valence.set(rec.v0, (valence.get(rec.v0) || 0) + 1);
    valence.set(rec.v1, (valence.get(rec.v1) || 0) + 1);
  }
  const used = new Set();
  for (const f of cage.faces) for (const v of f) used.add(v);
  let boundaryEdgeCount = 0;
  let nonManifoldEdgeCount = 0;
  for (const rec of edges.values()) {
    if (rec.faces.length === 1) boundaryEdgeCount++;
    else if (rec.faces.length > 2) nonManifoldEdgeCount++;
  }
  // ⚠ A VALENCE FLOOR IS NOT THE SAME NUMBER ON BOTH SIDES OF A BOUNDARY. A
  // valence-2 vertex in the INTERIOR is a doublet — a pinch, with two faces
  // meeting along two edges, which is what the NURBS conversion refuses. A
  // valence-2 vertex ON A BOUNDARY is the ordinary corner of an open grid:
  // every corner of a plane cage is one. So the floor is reported separately
  // for the two populations rather than as one number that condemns every
  // open cage.
  const onBoundary = new Set();
  for (const rec of edges.values()) if (rec.faces.length !== 2) { onBoundary.add(rec.v0); onBoundary.add(rec.v1); }
  const valenceHistogram = {};
  let minValence = used.size ? Infinity : 0;
  let minInteriorValence = Infinity;
  let minBoundaryValence = Infinity;
  for (const v of used) {
    const d = valence.get(v) || 0;
    valenceHistogram[d] = (valenceHistogram[d] || 0) + 1;
    if (d < minValence) minValence = d;
    if (onBoundary.has(v)) { if (d < minBoundaryValence) minBoundaryValence = d; }
    else if (d < minInteriorValence) minInteriorValence = d;
  }
  if (minInteriorValence === Infinity) minInteriorValence = 0;
  if (minBoundaryValence === Infinity) minBoundaryValence = 0;
  return {
    vertexCount: cage.vertices.length,
    usedVertexCount: used.size,
    unusedVertexCount: cage.vertices.length - used.size,
    edgeCount: edges.size,
    faceCount: cage.faces.length,
    euler: used.size - edges.size + cage.faces.length,
    allQuads: cage.faces.every((f) => f.length === 4),
    faceSizes,
    closed: boundaryEdgeCount === 0 && nonManifoldEdgeCount === 0,
    boundaryEdgeCount,
    nonManifoldEdgeCount,
    degenerateFaceCount,
    valenceHistogram,
    minValence,
    minInteriorValence,
    minBoundaryValence,
    boundaryVertexCount: onBoundary.size,
    creaseCount: Object.keys(cage.creases || {}).filter((k) => cage.creases[k] > 0).length,
  };
}

// THE GATE BOTH TIERS RUN BEFORE HANDING A CAGE BACK. Returns the problems as
// sentences; an empty list is the only acceptable result for a shipped
// simplification.
export function checkSimplifyInvariants(before, after) {
  const a = cageInvariants(before);
  const b = cageInvariants(after);
  const problems = [];
  if (!b.allQuads) {
    const sizes = Object.keys(b.faceSizes).filter((s) => s !== '4').join(', ');
    problems.push(`result is not all quads (face sizes present: ${sizes})`);
  }
  if (b.degenerateFaceCount) problems.push(`result has ${b.degenerateFaceCount} face(s) that use a vertex twice`);
  if (b.nonManifoldEdgeCount) problems.push(`result has ${b.nonManifoldEdgeCount} edge(s) shared by more than two faces`);
  if (a.closed && !b.closed) problems.push('input was closed and the result is not');
  if (b.usedVertexCount && b.minInteriorValence < 3) problems.push(`result has an interior vertex of valence ${b.minInteriorValence} (a pinch the NURBS conversion refuses)`);
  if (b.boundaryVertexCount && b.minBoundaryValence < 2) problems.push(`result has a boundary vertex of valence ${b.minBoundaryValence}`);
  if (b.euler !== a.euler) problems.push(`Euler characteristic changed ${a.euler} -> ${b.euler}`);
  if (b.unusedVertexCount) problems.push(`result carries ${b.unusedVertexCount} vertex(es) no face uses`);
  return { ok: problems.length === 0, problems, before: a, after: b };
}

// Connected components of the cage's own vertex/face graph, each returned as a
// standalone cage plus the index map back into the input.
function splitComponents(cage) {
  const n = cage.vertices.length;
  const comp = new Array(n).fill(-1);
  const adj = cage.vertices.map(() => []);
  for (const f of cage.faces) {
    for (let k = 0; k < f.length; k++) {
      const a = f[k];
      const b = f[(k + 1) % f.length];
      adj[a].push(b);
      adj[b].push(a);
    }
  }
  let nc = 0;
  for (let i = 0; i < n; i++) {
    if (comp[i] !== -1) continue;
    const stack = [i];
    comp[i] = nc;
    while (stack.length) {
      const v = stack.pop();
      for (const w of adj[v]) if (comp[w] === -1) { comp[w] = nc; stack.push(w); }
    }
    nc++;
  }
  const parts = [];
  for (let c = 0; c < nc; c++) {
    const verts = [];
    const map = new Map();
    for (let i = 0; i < n; i++) if (comp[i] === c) { map.set(i, verts.length); verts.push(cage.vertices[i].slice()); }
    const faces = cage.faces.filter((f) => comp[f[0]] === c).map((f) => f.map((v) => map.get(v)));
    const creases = {};
    for (const [key, w] of Object.entries(cage.creases || {})) {
      if (!(w > 0)) continue;
      const [a, b] = key.split('_').map(Number);
      if (comp[a] !== c) continue;
      creases[edgeKey(map.get(a), map.get(b))] = w;
    }
    parts.push({ cage: { vertices: verts, faces, creases }, map });
  }
  return parts;
}

function mergeCages(list) {
  const vertices = [];
  const faces = [];
  const creases = {};
  for (const c of list) {
    const off = vertices.length;
    for (const v of c.vertices) vertices.push(v.slice());
    for (const f of c.faces) faces.push(f.map((i) => i + off));
    for (const [key, w] of Object.entries(c.creases || {})) {
      if (!(w > 0)) continue;
      const [a, b] = key.split('_').map(Number);
      creases[edgeKey(a + off, b + off)] = w;
    }
  }
  return { vertices, faces, creases };
}

// ---------------------------------------------------------------------------
// TIER 1 — EXACT UN-SUBDIVIDE
// ---------------------------------------------------------------------------

const LABEL_V = 0; // a moved original vertex (a "vertex point")
const LABEL_E = 1; // an edge point
const LABEL_P = 2; // a face point

// The structural signature of one Catmull-Clark pass, with no geometry in it:
// all quads, manifold, and a vertex two-colouring in which one colour is the
// edge points. Reports what the un-subdivider would find without solving for
// any positions.
export function subdivisionSignature(cage) {
  const basic = basicCageCheck(cage);
  if (basic) return { ok: false, reason: basic.reason, message: basic.message };
  const parts = splitComponents(cage);
  const labelingsPerShell = [];
  for (const part of parts) {
    const topo = buildTopology(part.cage);
    const colour = twoColourVertices(part.cage, topo);
    if (!colour) return { ok: false, reason: 'NOT_A_SUBDIVISION', message: NOT_SUB(BIPARTITE_DETAIL) };
    const labelings = candidateLabelings(part.cage, topo, colour);
    if (!labelings.ok) return { ok: false, reason: 'NOT_A_SUBDIVISION', message: NOT_SUB(labelings.detail) };
    labelingsPerShell.push(labelings.list.length);
  }
  return { ok: true, shells: labelingsPerShell.length, labelings: labelingsPerShell };
}

export function isSubdivisionOfSomething(cage) {
  return unsubdivideCage(cage).ok;
}

const BIPARTITE_DETAIL = 'its vertices cannot be two-coloured with every edge joining the two colours, and one subdivision pass always can be';

function NOT_SUB(detail) {
  return `This cage is not the Catmull-Clark subdivision of a coarser cage, so there is no coarser cage to go back to: ${detail}.`;
}

function basicCageCheck(cage) {
  if (!cage || !Array.isArray(cage.vertices) || !Array.isArray(cage.faces) || !cage.faces.length) {
    return refuse('EMPTY_CAGE', 'This is not a cage: it has no faces.');
  }
  const inv = cageInvariants(cage);
  if (!inv.allQuads) {
    const sizes = Object.keys(inv.faceSizes).filter((s) => s !== '4').map(Number).sort((x, y) => x - y);
    return refuse('NOT_ALL_QUADS', `This cage has a ${sizes[0]}-sided face. One Catmull-Clark pass leaves nothing but quads, so this cage is not the subdivision of anything, and a strip of quads is not something an n-gon belongs to either. Merge Faces and Delete on an edge both leave n-gons behind.`);
  }
  if (inv.degenerateFaceCount) {
    return refuse('DEGENERATE_FACE', `This cage has ${inv.degenerateFaceCount} face(s) that use the same vertex twice.`);
  }
  if (inv.nonManifoldEdgeCount) {
    return refuse('NON_MANIFOLD', `This cage has ${inv.nonManifoldEdgeCount} edge(s) shared by three or more faces. Simplify works on a surface, and this cage is not one.`);
  }
  if (inv.unusedVertexCount) {
    return refuse('UNUSED_VERTICES', `This cage carries ${inv.unusedVertexCount} vertex(es) no face uses.`);
  }
  return null;
}

function twoColourVertices(cage, topo) {
  const n = cage.vertices.length;
  const colour = new Array(n).fill(-1);
  for (let s = 0; s < n; s++) {
    if (colour[s] !== -1) continue;
    colour[s] = 0;
    const stack = [s];
    while (stack.length) {
      const v = stack.pop();
      for (const e of topo.vertexEdges[v]) {
        const w = e.v0 === v ? e.v1 : e.v0;
        if (colour[w] === -1) { colour[w] = 1 - colour[v]; stack.push(w); }
        else if (colour[w] === colour[v]) return null;
      }
    }
  }
  return colour;
}

// An edge point of a closed region has valence 4 and sits on four fine faces
// (its two originals and its two face points); one on the boundary has
// valence 3 and sits on two. Anything else on the candidate edge-point colour
// rules that colour out immediately, which is what makes a puff cage — eight
// interior valence-3 vertices by construction — refuse in constant time.
function edgeSideProblem(cage, topo, colour, eSide) {
  for (let v = 0; v < cage.vertices.length; v++) {
    if (colour[v] !== eSide) continue;
    const deg = topo.vertexEdges[v].length;
    const nf = topo.vertexFaces[v].length;
    const onBoundary = topo.vertexEdges[v].some((e) => e.faces.length === 1);
    if (onBoundary) {
      if (deg !== 3 || nf !== 2) return 'an edge-point candidate on the boundary does not have valence 3 on two faces';
    } else if (deg !== 4 || nf !== 4) {
      return 'an edge-point candidate does not have valence 4';
    }
  }
  return null;
}

// Every fine face is [originalVertex, edgePoint, facePoint, edgePoint], so the
// two non-edge-point corners of a face are DIAGONAL and one of them is the
// face point. Two-colouring that diagonal relation splits the non-edge-point
// vertices into the originals and the face points; the graph it runs on is the
// coarse cage's own vertex/face incidence graph, so on a connected cage there
// are exactly two ways round, and geometry has to break the tie.
function candidateLabelings(cage, topo, colour) {
  const list = [];
  let detail = null;
  for (const eSide of [0, 1]) {
    const problem = edgeSideProblem(cage, topo, colour, eSide);
    if (problem) { detail = detail || problem; continue; }
    const diag = diagonalColouring(cage, colour, eSide);
    if (!diag.ok) { detail = detail || diag.detail; continue; }
    for (const pClass of diag.pClasses) {
      const label = new Array(cage.vertices.length);
      for (let v = 0; v < cage.vertices.length; v++) {
        if (colour[v] === eSide) label[v] = LABEL_E;
        else label[v] = diag.dcol[v] === pClass ? LABEL_P : LABEL_V;
      }
      list.push(label);
    }
  }
  if (!list.length) return { ok: false, detail: detail || 'neither colour can be its edge points' };
  return { ok: true, list };
}

function diagonalColouring(cage, colour, eSide) {
  const n = cage.vertices.length;
  const adj = cage.vertices.map(() => []);
  for (const f of cage.faces) {
    const at = [0, 1, 2, 3].filter((k) => colour[f[k]] !== eSide);
    if (at.length !== 2 || at[1] - at[0] !== 2) {
      return { ok: false, detail: 'a face does not alternate edge points with its other two corners' };
    }
    adj[f[at[0]]].push(f[at[1]]);
    adj[f[at[1]]].push(f[at[0]]);
  }
  const dcol = new Array(n).fill(-1);
  for (let s = 0; s < n; s++) {
    if (colour[s] === eSide || dcol[s] !== -1) continue;
    dcol[s] = 0;
    const stack = [s];
    while (stack.length) {
      const v = stack.pop();
      for (const w of adj[v]) {
        if (dcol[w] === -1) { dcol[w] = 1 - dcol[v]; stack.push(w); }
        else if (dcol[w] === dcol[v]) return { ok: false, detail: 'the two non-edge-point corners of its faces cannot be split into originals and face points' };
      }
    }
  }
  // A face point is a face's own centroid, so it is never on a boundary. On an
  // open cage that fixes which class is which; on a closed one both remain.
  const boundaryVertex = new Set();
  const edgeUse = new Map();
  for (const f of cage.faces) {
    for (let k = 0; k < 4; k++) {
      const key = edgeKey(f[k], f[(k + 1) % 4]);
      edgeUse.set(key, (edgeUse.get(key) || 0) + 1);
    }
  }
  for (const f of cage.faces) {
    for (let k = 0; k < 4; k++) {
      const a = f[k];
      const b = f[(k + 1) % 4];
      if (edgeUse.get(edgeKey(a, b)) === 1) { boundaryVertex.add(a); boundaryVertex.add(b); }
    }
  }
  const blocked = new Set();
  for (const v of boundaryVertex) if (colour[v] !== eSide) blocked.add(dcol[v]);
  const pClasses = [0, 1].filter((c) => !blocked.has(c));
  if (!pClasses.length) return { ok: false, detail: 'both candidate face-point classes reach the boundary, and a face point never does' };
  return { ok: true, dcol, pClasses };
}

// Turns one labeling into the coarse cage's TOPOLOGY, walking the wheel of
// fine faces around each face point to recover that coarse face in its
// original winding order.
function buildCoarseTopology(cage, topo, label) {
  const vList = [];
  const pList = [];
  const coarseOf = new Array(cage.vertices.length).fill(-1);
  for (let v = 0; v < cage.vertices.length; v++) {
    if (label[v] === LABEL_V) { coarseOf[v] = vList.length; vList.push(v); }
    else if (label[v] === LABEL_P) pList.push(v);
  }
  const faces = [];
  const coarseEdgeFineE = new Map();
  for (const p of pList) {
    const inc = topo.vertexFaces[p];
    const start = inc[0];
    const loop = [];
    const edgePts = [];
    let f = start;
    for (let step = 0; step <= inc.length; step++) {
      const face = cage.faces[f];
      const i = face.indexOf(p);
      if (i < 0) return { ok: false, detail: 'a face-point candidate is not a corner of one of its own faces' };
      loop.push(coarseOf[face[(i + 2) % 4]]);
      const eNext = face[(i + 3) % 4];
      if (label[face[(i + 2) % 4]] !== LABEL_V || label[eNext] !== LABEL_E) {
        return { ok: false, detail: 'a face around a face-point candidate is not [original, edge point, face point, edge point]' };
      }
      edgePts.push(eNext);
      const rec = topo.edgeMap.get(edgeKey(p, eNext));
      if (!rec || rec.faces.length !== 2) return { ok: false, detail: 'a face-point candidate sits on the boundary, and a face point never does' };
      f = rec.faces[0] === f ? rec.faces[1] : rec.faces[0];
      if (f === start) break;
    }
    if (f !== start || loop.length !== inc.length) {
      return { ok: false, detail: 'the faces around a face-point candidate do not close into one ring' };
    }
    if (loop.length < 3) return { ok: false, detail: 'a coarse face would have fewer than three sides' };
    if (loop.some((x) => x < 0) || new Set(loop).size !== loop.length) {
      return { ok: false, detail: 'a coarse face would use the same vertex twice' };
    }
    faces.push(loop);
    for (let k = 0; k < loop.length; k++) {
      const key = edgeKey(loop[k], loop[(k + 1) % loop.length]);
      const prev = coarseEdgeFineE.get(key);
      if (prev !== undefined && prev !== edgePts[k]) {
        return { ok: false, detail: 'two coarse faces disagree about which vertex is one edge’s edge point' };
      }
      coarseEdgeFineE.set(key, edgePts[k]);
    }
  }
  let eCount = 0;
  for (let v = 0; v < cage.vertices.length; v++) if (label[v] === LABEL_E) eCount++;
  if (coarseEdgeFineE.size !== eCount) {
    return { ok: false, detail: 'the edge-point candidates do not correspond one-to-one with the coarse cage’s edges' };
  }
  return { ok: true, vList, pList, coarseOf, faces, coarseEdgeFineE };
}

// The coarse crease map, read straight off the fine one: subdivideCatmullClark
// gives both children of a creased edge weight w-1, so the coarse weight is
// the child's weight plus one. A crease anywhere else — on a spoke from an
// edge point to a face point — is something one subdivision pass never
// originates, so it rules the cage out.
function recoverCreases(cage, topo, label, coarse) {
  const creases = {};
  const seen = new Set();
  for (const [key, rec] of topo.edgeMap) {
    const w = creaseWeight(cage, rec.v0, rec.v1);
    if (!(w > 0)) continue;
    const la = label[rec.v0];
    const lb = label[rec.v1];
    const pair = (la === LABEL_E ? lb : la);
    if ((la === LABEL_E) === (lb === LABEL_E) || pair === LABEL_P) {
      return { ok: false, stage: 2, detail: 'a crease sits on an edge one subdivision pass would have created smooth' };
    }
    seen.add(key);
  }
  for (const [ckey, e] of coarse.coarseEdgeFineE) {
    const [ca, cb] = ckey.split('_').map(Number);
    const a = coarse.vList[ca];
    const b = coarse.vList[cb];
    const wa = creaseWeight(cage, a, e);
    const wb = creaseWeight(cage, b, e);
    if (wa <= 0 && wb <= 0) continue;
    if (Math.abs(wa - wb) > 1e-9) {
      return { ok: false, stage: 3, detail: 'the two halves of a creased edge carry different weights, and one subdivision pass always gives them the same one' };
    }
    creases[ckey] = wa + 1;
  }
  return { ok: true, creases };
}

// The pair sum (Pa+Pb) an edge point hands back, per computeEdgePoint's own
// two branches. `sharp` covers a boundary edge and any crease at weight >= 1,
// which is every crease that leaves a trace in the fine cage.
function pairSums(cage, topo, label, coarse, sharpKeys) {
  const sums = new Map();
  for (const [ckey, e] of coarse.coarseEdgeFineE) {
    const ePos = cage.vertices[e];
    const nbrFacePts = [];
    let boundary = false;
    for (const edge of topo.vertexEdges[e]) {
      const other = edge.v0 === e ? edge.v1 : edge.v0;
      if (label[other] === LABEL_P) nbrFacePts.push(cage.vertices[other]);
      if (edge.faces.length === 1) boundary = true;
    }
    const sharp = boundary || sharpKeys.has(ckey);
    if (sharp) { sums.set(ckey, scale(ePos, 2)); continue; }
    if (nbrFacePts.length !== 2) return { ok: false, detail: 'a smooth edge point does not have exactly two face points beside it' };
    sums.set(ckey, sub(scale(ePos, 4), add(nbrFacePts[0], nbrFacePts[1])));
  }
  return { ok: true, sums };
}

// Propagate the pair sums along a spanning tree: P_i = c_i + sigma_i * x, one
// unknown 3-vector x per connected component. A non-tree edge joining two
// vertices of the SAME sign is an odd cycle and pins x outright.
function propagate(nCoarse, faces, sums) {
  const c = new Array(nCoarse).fill(null);
  const sigma = new Array(nCoarse).fill(0);
  const comp = new Array(nCoarse).fill(-1);
  const adj = new Array(nCoarse).fill(null).map(() => []);
  for (const f of faces) {
    for (let k = 0; k < f.length; k++) {
      const a = f[k];
      const b = f[(k + 1) % f.length];
      adj[a].push(b);
      adj[b].push(a);
    }
  }
  const pins = [];
  let nc = 0;
  for (let s = 0; s < nCoarse; s++) {
    if (comp[s] !== -1) continue;
    const id = nc++;
    comp[s] = id; c[s] = ZERO.slice(); sigma[s] = 1;
    pins.push(null);
    const stack = [s];
    while (stack.length) {
      const v = stack.pop();
      for (const w of adj[v]) {
        const T = sums.get(edgeKey(v, w));
        if (!T) continue;
        if (comp[w] === -1) {
          comp[w] = id;
          sigma[w] = -sigma[v];
          c[w] = sub(T, c[v]);
          stack.push(w);
        } else if (sigma[w] === sigma[v]) {
          // 2*sigma*x = T - c[v] - c[w]
          const x = scale(sub(sub(T, c[v]), c[w]), sigma[v] / 2);
          if (!pins[id]) pins[id] = x;
        }
      }
    }
  }
  return { c, sigma, comp, componentCount: nc, pins };
}

function solve3(M, rhs) {
  const A = [[M[0][0], M[0][1], M[0][2], rhs[0]], [M[1][0], M[1][1], M[1][2], rhs[1]], [M[2][0], M[2][1], M[2][2], rhs[2]]];
  let scaleRef = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) scaleRef = Math.max(scaleRef, Math.abs(M[i][j]));
  if (scaleRef === 0) return null;
  for (let col = 0; col < 3; col++) {
    let best = col;
    for (let r = col + 1; r < 3; r++) if (Math.abs(A[r][col]) > Math.abs(A[best][col])) best = r;
    if (Math.abs(A[best][col]) < 1e-12 * scaleRef) return null;
    const tmp = A[col]; A[col] = A[best]; A[best] = tmp;
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const m = A[r][col] / A[col][col];
      for (let k = col; k < 4; k++) A[r][k] -= m * A[col][k];
    }
  }
  return [A[0][3] / A[0][0], A[1][3] / A[1][1], A[2][3] / A[2][2]];
}

// One candidate labeling, solved and then CHECKED by re-subdividing. The check
// is the only thing that decides; nothing here is accepted on structure alone.
function trySolve(cage, topo, label, tolerance) {
  const coarse = buildCoarseTopology(cage, topo, label);
  if (!coarse.ok) return { ok: false, stage: 1, detail: coarse.detail };
  const cr = recoverCreases(cage, topo, label, coarse);
  if (!cr.ok) return { ok: false, stage: cr.stage ?? 2, detail: cr.detail };

  const sharpFromCreases = new Set(Object.keys(cr.creases));
  // Pass 1 assumes every edge with no crease entry in the fine cage was
  // smooth. Pass 2 is the WEIGHT-ONE RETRY: a coarse crease of exactly 1 is
  // fully sharp and yet leaves no entry behind, so the edges the face
  // equations say cannot have been smooth are re-tried as sharp. A wrong
  // guess simply fails the same verification, so this can turn a refusal into
  // an exact answer and can never turn one into a wrong one.
  let best = null;
  const hypotheses = [sharpFromCreases];
  const retry = sharpFromFaceEquations(cage, coarse, sharpFromCreases, 1e-9 * cageExtent(cage));
  if (retry) hypotheses.push(retry);
  for (let pass = 0; pass < hypotheses.length; pass++) {
    // The hypothesised sharp edges have to go into the CREASE MAP the coarse
    // cage carries, not just into the pair-sum arithmetic — the verification
    // re-subdivides that cage, and a cage whose creases say "smooth" produces
    // smooth edge points however the solve was set up.
    const creases = { ...cr.creases };
    if (pass > 0) for (const key of hypotheses[pass]) if (!(creases[key] > 0)) creases[key] = 1;
    const attempt = solveAndVerify(cage, topo, label, coarse, creases, hypotheses[pass], tolerance, pass > 0);
    if (attempt.ok) return attempt;
    if (!best || attempt.residual < best.residual) best = attempt;
  }
  return best || { ok: false, detail: 'no solution' };
}

// WHICH COARSE EDGES CANNOT HAVE BEEN SMOOTH, read off the face-centroid
// equations. A coarse quad's four corners sum to four times its face point, so
// the pair sums of two OPPOSITE edges must add to exactly that. Each opposite
// pair therefore has four smooth/sharp combinations and usually only one of
// them balances, which names the sharp edges without solving anything.
function sharpFromFaceEquations(cage, coarse, already, tol) {
  const facesAtEdge = new Map();
  coarse.faces.forEach((f, fi) => {
    for (let k = 0; k < f.length; k++) {
      const key = edgeKey(f[k], f[(k + 1) % f.length]);
      if (!facesAtEdge.has(key)) facesAtEdge.set(key, []);
      facesAtEdge.get(key).push(fi);
    }
  });
  const pPos = coarse.pList.map((p) => cage.vertices[p]);
  const ePos = (key) => cage.vertices[coarse.coarseEdgeFineE.get(key)];
  const sums = (key) => {
    const out = [null, scale(ePos(key), 2)];
    const fs = facesAtEdge.get(key);
    if (fs.length === 2) out[0] = sub(scale(ePos(key), 4), add(pPos[fs[0]], pPos[fs[1]]));
    return out;
  };
  const verdict = new Map();
  let conflict = false;
  coarse.faces.forEach((f, fi) => {
    if (f.length !== 4) return;
    for (let i = 0; i < 2; i++) {
      const ka = edgeKey(f[i], f[i + 1]);
      const kb = edgeKey(f[i + 2], f[(i + 3) % 4]);
      const sa = sums(ka);
      const sb = sums(kb);
      const want = scale(pPos[fi], 4);
      const feasible = [];
      for (let a = 0; a < 2; a++) {
        for (let b = 0; b < 2; b++) {
          if (!sa[a] || !sb[b]) continue;
          const d = sub(add(sa[a], sb[b]), want);
          if (Math.sqrt(dot(d, d)) <= tol) feasible.push([a, b]);
        }
      }
      if (feasible.length !== 1) continue;
      const [a, b] = feasible[0];
      for (const [key, isSharp] of [[ka, a === 1], [kb, b === 1]]) {
        if (verdict.has(key) && verdict.get(key) !== isSharp) conflict = true;
        verdict.set(key, isSharp);
      }
    }
  });
  if (conflict) return null;
  const out = new Set(already);
  let added = false;
  for (const [key, isSharp] of verdict) if (isSharp && !out.has(key)) { out.add(key); added = true; }
  return added ? out : null;
}

function solveAndVerify(cage, topo, label, coarse, creases, sharpKeys, tolerance, isRetry) {
  const ps = pairSums(cage, topo, label, coarse, sharpKeys);
  if (!ps.ok) return { ok: false, stage: 4, detail: ps.detail, residual: Infinity };
  const nCoarse = coarse.vList.length;
  const prop = propagate(nCoarse, coarse.faces, ps.sums);
  const xs = new Array(prop.componentCount).fill(null).map((_, i) => (prop.pins[i] ? prop.pins[i].slice() : ZERO.slice()));
  const freeComps = [];
  for (let i = 0; i < prop.componentCount; i++) if (!prop.pins[i]) freeComps.push(i);

  const buildVerts = (xsUse) => prop.c.map((ci, j) => add(ci, scale(xsUse[prop.comp[j]], prop.sigma[j])));
  const makeCage = (xsUse) => ({ vertices: buildVerts(xsUse), faces: coarse.faces, creases });

  const target = fineTargetOrder(cage, coarse, creases);
  if (!target.ok) return { ok: false, stage: 4, detail: target.detail, residual: Infinity };

  let unique = true;
  if (freeComps.length) {
    const base = subdivideCatmullClark(makeCage(xs));
    for (const cIdx of freeComps) {
      const cols = [];
      for (let axis = 0; axis < 3; axis++) {
        const xsB = xs.map((x) => x.slice());
        xsB[cIdx] = [axis === 0 ? 1 : 0, axis === 1 ? 1 : 0, axis === 2 ? 1 : 0];
        const s = subdivideCatmullClark(makeCage(xsB));
        cols.push(s.vertices.map((v, i) => sub(v, base.vertices[i])));
      }
      const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      const rhs = [0, 0, 0];
      for (let i = 0; i < base.vertices.length; i++) {
        const b = sub(target.positions[i], base.vertices[i]);
        for (let a = 0; a < 3; a++) {
          rhs[a] += dot(cols[a][i], b);
          for (let d = 0; d < 3; d++) M[a][d] += dot(cols[a][i], cols[d][i]);
        }
      }
      const x = solve3(M, rhs);
      if (x) xs[cIdx] = x;
      else {
        // No unique preimage exists in this direction. Pick the member of the
        // family closest to the fine cage's own vertex points.
        unique = false;
        let acc = ZERO.slice();
        let n = 0;
        for (let j = 0; j < nCoarse; j++) {
          if (prop.comp[j] !== cIdx) continue;
          acc = add(acc, scale(sub(cage.vertices[coarse.vList[j]], prop.c[j]), prop.sigma[j]));
          n++;
        }
        xs[cIdx] = n ? scale(acc, 1 / n) : ZERO.slice();
      }
    }
  }

  const coarseCage = makeCage(xs);
  const sub2 = subdivideCatmullClark(coarseCage);
  const extent = cageExtent(cage);
  let worst = 0;
  for (let i = 0; i < sub2.vertices.length; i++) {
    const d = sub(sub2.vertices[i], target.positions[i]);
    worst = Math.max(worst, Math.sqrt(dot(d, d)));
  }
  const residual = worst / extent;
  const creaseMismatch = compareCreases(sub2.creases, target.creases);
  if (residual > tolerance) {
    return { ok: false, residual, detail: 'positions', coarse: coarseCage, isRetry };
  }
  if (creaseMismatch) {
    return { ok: false, residual, detail: `creases: ${creaseMismatch}`, coarse: coarseCage, isRetry };
  }
  return { ok: true, cage: coarseCage, residual, unique, recoveredWeightOneCrease: isRetry };
}

// The input cage's own vertices, reordered into the order
// subdivideCatmullClark emits them for the recovered coarse cage: the coarse
// vertices first, then the edge points in buildTopology's edgeMap order, then
// the face points in face order.
function fineTargetOrder(cage, coarse, creases) {
  const n = coarse.vList.length;
  const tmp = { vertices: new Array(n).fill(ZERO), faces: coarse.faces, creases };
  const ctopo = buildTopology(tmp);
  const fineIdx = new Array(n);
  for (let j = 0; j < n; j++) fineIdx[j] = coarse.vList[j];
  for (const [key] of ctopo.edgeMap) {
    const e = coarse.coarseEdgeFineE.get(key);
    if (e === undefined) return { ok: false, detail: 'a coarse edge has no edge point' };
    fineIdx.push(e);
  }
  for (const p of coarse.pList) fineIdx.push(p);
  if (fineIdx.length !== cage.vertices.length) {
    return { ok: false, detail: 'the vertex counts do not add up to one subdivision pass' };
  }
  const positions = fineIdx.map((i) => cage.vertices[i]);
  const cr = {};
  for (const [key, w] of Object.entries(cage.creases || {})) {
    if (!(w > 0)) continue;
    const [a, b] = key.split('_').map(Number);
    const ia = fineIdx.indexOf(a);
    const ib = fineIdx.indexOf(b);
    if (ia < 0 || ib < 0) return { ok: false, detail: 'a crease names a vertex the cage does not have' };
    cr[edgeKey(ia, ib)] = w;
  }
  return { ok: true, positions, creases: cr, fineIdx };
}

function compareCreases(got, want) {
  const gk = Object.keys(got).filter((k) => got[k] > 0);
  const wk = Object.keys(want).filter((k) => want[k] > 0);
  if (gk.length !== wk.length) return `${gk.length} recovered against ${wk.length} in the cage`;
  for (const k of wk) {
    if (!(Math.abs((got[k] || 0) - want[k]) <= 1e-9)) return `edge ${k} came back at ${got[k] ?? 0} instead of ${want[k]}`;
  }
  return null;
}

// THE TIER-1 ENTRY POINT.
//   { ok: true,  cage, residual, unique, tier: 'unsubdivide', ... }
//   { ok: false, reason, message }
// `unique: false` means the input has a whole family of preimages and the one
// returned is the family member nearest the input's own vertex points.
export function unsubdivideCage(cage, opts = {}) {
  const tolerance = opts.tolerance ?? 1e-9;
  const basic = basicCageCheck(cage);
  if (basic) return basic;
  const parts = splitComponents(cage);
  const results = [];
  for (const part of parts) {
    const topo = buildTopology(part.cage);
    const colour = twoColourVertices(part.cage, topo);
    if (!colour) return refuse('NOT_A_SUBDIVISION', NOT_SUB(BIPARTITE_DETAIL));
    const cand = candidateLabelings(part.cage, topo, colour);
    if (!cand.ok) return refuse('NOT_A_SUBDIVISION', NOT_SUB(cand.detail));
    const verified = [];
    let nearest = null;
    let structural = null; // the detail from the labeling that got FURTHEST
    for (const label of cand.list) {
      const r = trySolve(part.cage, topo, label, tolerance);
      if (r.ok) verified.push(r);
      else if (r.detail === 'positions' || r.detail?.startsWith('creases')) {
        if (!nearest || r.residual < nearest.residual) nearest = r;
      } else if (!structural || (r.stage ?? 0) > structural.stage) structural = { stage: r.stage ?? 0, detail: r.detail };
    }
    if (!verified.length) {
      if (nearest) {
        const pct = (nearest.residual * 100).toFixed(4);
        return refuse('POSITIONS_DO_NOT_MATCH',
          `This cage has the shape of a subdivision but not its positions: subdividing the cage it would have come from misses this one by ${pct}% of its size. Its points have been moved since it was subdivided, or it carried a crease of weight between 0 and 1, which one subdivision pass erases without trace.`,
          { residual: nearest.residual });
      }
      return refuse('NOT_A_SUBDIVISION', NOT_SUB(structural ? structural.detail : 'no labelling of its vertices survives'));
    }
    results.push({ ...verified[0], ambiguousLabelings: verified.length });
  }
  const merged = mergeCages(results.map((r) => r.cage));
  const inv = cageInvariants(merged);
  const out = {
    ok: true,
    tier: 'unsubdivide',
    cage: merged,
    residual: Math.max(...results.map((r) => r.residual)),
    unique: results.every((r) => r.unique) && results.every((r) => r.ambiguousLabelings === 1),
    recoveredWeightOneCrease: results.some((r) => r.recoveredWeightOneCrease),
    shells: results.length,
    invariants: inv,
    message: `exact — this cage was a subdivision, and this is the cage it came from: ${cage.faces.length} → ${merged.faces.length} faces`,
  };
  if (!out.unique) {
    out.message += '. Its preimage is not unique: the cage returned is the one closest to the points you have';
  }
  if (opts.allowNgons !== true && !inv.allQuads) {
    const sizes = Object.keys(inv.faceSizes).filter((s) => s !== '4').map(Number).sort((x, y) => x - y);
    return refuse('PREIMAGE_NOT_ALL_QUADS',
      `The cage this one was subdivided from has a ${sizes[0]}-sided face. Going back to it would hand you a cage the NURBS conversion cannot turn into a surface, so Simplify stops here.`,
      { cage: merged, invariants: inv });
  }
  if (opts.allowNgons !== true && inv.minInteriorValence < 3) {
    return refuse('PREIMAGE_PINCHED',
      `The cage this one was subdivided from has an interior vertex of valence ${inv.minInteriorValence}. That is a pinch, not a surface, so Simplify stops here.`,
      { cage: merged, invariants: inv });
  }
  return out;
}

// ---------------------------------------------------------------------------
// TIER 2 — POLYCHORD COLLAPSE
// ---------------------------------------------------------------------------

function faceEdgeKey(face, k) { return edgeKey(face[k], face[(k + 1) % 4]); }

// Walk one way from (face, exit edge index), crossing to the neighbouring quad
// through the opposite edge each step.
function walkChord(cage, edgeMap, f0, j0) {
  const faces = [];
  const parities = [];
  const exits = [];
  const seen = new Set();
  let f = f0;
  let j = j0;
  let ended = 'boundary';
  for (;;) {
    const stateKey = `${f}:${j % 2}`;
    if (seen.has(stateKey)) { ended = 'self'; break; }
    seen.add(stateKey);
    faces.push(f);
    parities.push(j % 2);
    const key = faceEdgeKey(cage.faces[f], j);
    exits.push(key);
    const rec = edgeMap.get(key);
    if (!rec || rec.faces.length !== 2) { ended = 'boundary'; break; }
    const g = rec.faces[0] === f ? rec.faces[1] : rec.faces[0];
    const gf = cage.faces[g];
    let jg = -1;
    for (let k = 0; k < 4; k++) if (faceEdgeKey(gf, k) === key) { jg = k; break; }
    if (jg < 0) { ended = 'boundary'; break; }
    f = g;
    j = (jg + 2) % 4;
    if (f === f0 && j === j0) { ended = 'closed'; break; }
  }
  return { faces, parities, exits, ended };
}

// EVERY POLYCHORD OF A QUAD CAGE. Each face lies on exactly two of them, so
// the chords are a complete, non-overlapping decomposition of what a collapse
// can remove. Returns chords whether or not they are collapsible, each
// carrying its own refusal when it is not.
export function polychordsOf(cage, opts = {}) {
  const basic = basicCageCheck(cage);
  if (basic) return { ok: false, reason: basic.reason, message: basic.message, chords: [] };
  const topo = buildTopology(cage);
  const edgeMap = topo.edgeMap;
  const visited = new Set();
  const chords = [];
  for (let f = 0; f < cage.faces.length; f++) {
    for (let p = 0; p < 2; p++) {
      if (visited.has(`${f}:${p}`)) continue;
      const fwd = walkChord(cage, edgeMap, f, p);
      let faces;
      let parities;
      let rungKeys;
      let closed = false;
      let selfTouching = false;
      if (fwd.ended === 'closed') {
        faces = fwd.faces;
        parities = fwd.parities;
        rungKeys = fwd.exits;
        closed = true;
      } else if (fwd.ended === 'self') {
        faces = fwd.faces;
        parities = fwd.parities;
        rungKeys = fwd.exits;
        selfTouching = true;
      } else {
        const back = walkChord(cage, edgeMap, f, (p + 2) % 4);
        if (back.ended === 'self') selfTouching = true;
        faces = back.faces.slice(1).reverse().concat(fwd.faces);
        parities = back.parities.slice(1).reverse().concat(fwd.parities);
        rungKeys = back.exits.slice().reverse().concat(fwd.exits);
      }
      for (let k = 0; k < faces.length; k++) visited.add(`${faces[k]}:${parities[k]}`);
      const rungs = rungKeys.map((k) => { const [a, b] = k.split('_').map(Number); return [a, b]; });
      const chord = {
        id: chords.length,
        faces,
        rungKeys,
        rungs,
        closed,
        selfTouching,
        length: faces.length,
      };
      chord.refusal = chordRefusalAndPreview(cage, chord, opts).refusal || null;
      chord.collapsible = !chord.refusal;
      chords.push(chord);
    }
  }
  return { ok: true, chords };
}

const MIN_FACES_DEFAULT = 6;

// The rails of a chord face: its two edges that are not rungs. They are the
// edges the collapse WELDS TOGETHER into one, which is why a disagreement
// between their crease weights has no answer.
function chordRails(cage, chord) {
  const rungSet = new Set(chord.rungKeys);
  const rails = [];
  for (const f of chord.faces) {
    const face = cage.faces[f];
    const own = [];
    for (let k = 0; k < 4; k++) {
      const key = faceEdgeKey(face, k);
      if (!rungSet.has(key)) own.push(key);
    }
    if (own.length !== 2) return null;
    rails.push(own);
  }
  return rails;
}

// A CHORD IS ONLY COLLAPSIBLE IF THE COLLAPSE ITSELF HOLDS THE INVARIANT, so
// this does the collapse to find out and hands the result back rather than
// throwing it away — the caller that accepts the chord reuses it, and there is
// no second, unreachable copy of the same check downstream.
function chordRefusalAndPreview(cage, chord, opts = {}) {
  const minFaces = opts.minFaces ?? MIN_FACES_DEFAULT;
  if (chord.selfTouching) {
    return { refusal: { reason: 'SELF_TOUCHING', message: 'That strip meets itself; collapsing it would fold the cage onto itself.' } };
  }
  if (!chord.closed && !opts.allowOpenChords) {
    return { refusal: { reason: 'OPEN_CHORD', message: 'That strip runs off an open edge of the cage. Removing it would move the boundary, which is not a simplification — it is a different shape.' } };
  }
  const touched = new Set();
  for (const [a, b] of chord.rungs) {
    if (touched.has(a) || touched.has(b)) {
      return { refusal: { reason: 'SELF_TOUCHING', message: 'That strip crosses itself at a vertex; collapsing it would weld more than two points together.' } };
    }
    touched.add(a); touched.add(b);
  }
  for (const key of chord.rungKeys) {
    const w = (cage.creases || {})[key];
    if (w > 0) {
      return { refusal: { reason: 'CHORD_CREASE', message: 'That strip runs through a creased edge; removing it would delete the crease. Remove the crease first if you meant to.' } };
    }
  }
  const rails = chordRails(cage, chord);
  if (!rails) {
    return { refusal: { reason: 'MALFORMED_CHORD', message: 'That strip does not cross two opposite edges of one of its own faces.' } };
  }
  for (const [r0, r1] of rails) {
    const w0 = (cage.creases || {})[r0] || 0;
    const w1 = (cage.creases || {})[r1] || 0;
    if (Math.abs(w0 - w1) > 1e-12) {
      return { refusal: { reason: 'CREASE_COLLISION', message: 'The two sides of that strip carry different crease weights and the collapse would merge them into one edge, which has no right weight to be.' } };
    }
  }
  if (cage.faces.length - chord.faces.length < minFaces) {
    return { refusal: { reason: 'BELOW_FLOOR', message: `Collapsing that strip would leave ${cage.faces.length - chord.faces.length} faces, below the ${minFaces}-face floor this cage needs to still describe a surface.` } };
  }
  const preview = applyCollapse(cage, chord);
  const check = checkSimplifyInvariants(cage, preview.cage);
  if (!check.ok) {
    return { refusal: { reason: 'DEGENERATE_RESULT', message: `Collapsing that strip would leave a cage that is not a surface: ${check.problems.join('; ')}.` } };
  }
  return { preview, invariants: check.after };
}

// The collapse itself, with no judgement in it: weld the two ends of every
// rung, drop the strip's faces, and let every neighbouring quad keep its four
// corners because its two welded corners land on their partners.
function applyCollapse(cage, chord) {
  const n = cage.vertices.length;
  const parent = new Array(n).fill(0).map((_, i) => i);
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  for (const [a, b] of chord.rungs) { const ra = find(a); const rb = find(b); if (ra !== rb) parent[rb] = ra; }
  const groups = new Map();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(i);
  }
  const drop = new Set(chord.faces);
  const usedRoots = new Set();
  const keptFaces = [];
  cage.faces.forEach((f, fi) => { if (!drop.has(fi)) keptFaces.push(f); });
  for (const f of keptFaces) for (const v of f) usedRoots.add(find(v));
  const newIdx = new Map();
  const vertices = [];
  const vertexSources = [];
  for (const [root, members] of groups) {
    if (!usedRoots.has(root)) continue;
    newIdx.set(root, vertices.length);
    let acc = ZERO.slice();
    for (const m of members) acc = add(acc, cage.vertices[m]);
    vertices.push(scale(acc, 1 / members.length));
    vertexSources.push(members.slice());
  }
  const faces = keptFaces.map((f) => {
    const mapped = f.map((v) => newIdx.get(find(v)));
    const out = [];
    for (let k = 0; k < mapped.length; k++) if (mapped[k] !== mapped[(k + 1) % mapped.length]) out.push(mapped[k]);
    return out;
  });
  const creases = {};
  for (const [key, w] of Object.entries(cage.creases || {})) {
    if (!(w > 0)) continue;
    const [a, b] = key.split('_').map(Number);
    const ra = newIdx.get(find(a));
    const rb = newIdx.get(find(b));
    if (ra === undefined || rb === undefined || ra === rb) continue;
    const k2 = edgeKey(ra, rb);
    creases[k2] = Math.max(creases[k2] || 0, w);
  }
  return { cage: { vertices, faces, creases }, vertexSources };
}

// COLLAPSE ONE POLYCHORD.
//   { ok: true, cage, vertexSources, ... } | { ok: false, reason, message }
// `vertexSources[i]` lists the input vertices the result's vertex i came from,
// which is what a refit needs to know where each surviving point used to be.
export function collapsePolychord(cage, chord, opts = {}) {
  const judged = chordRefusalAndPreview(cage, chord, opts);
  if (judged.refusal) return { ok: false, ...judged.refusal };
  return { ok: true, cage: judged.preview.cage, vertexSources: judged.preview.vertexSources, invariants: judged.invariants, removedFaces: chord.faces.length };
}

// ---------------------------------------------------------------------------
// SHAPE: LIMIT-POSITION REFIT AND THE CHORD RANKING
// ---------------------------------------------------------------------------

// A vertex whose limit position vertexLimitPosition can actually speak for:
// that function carries the SMOOTH interior mask only, so a boundary or
// creased vertex is held still rather than refitted against a mask that does
// not describe it.
function smoothInteriorVertices(cage, topo) {
  const out = [];
  for (let v = 0; v < cage.vertices.length; v++) {
    const edges = topo.vertexEdges[v];
    if (!edges.length) continue;
    let ok = true;
    for (const e of edges) {
      if (e.faces.length !== 2) { ok = false; break; }
      if (creaseWeight(cage, e.v0, e.v1) > 0) { ok = false; break; }
    }
    if (ok) out.push(v);
  }
  return out;
}

// The puff's own technique (kernel/puff.mjs), generalised: move control points
// until their LIMIT positions land on a target. Each pass is a correction, not
// a search. The best iterate is kept, so a target the cage cannot reach leaves
// the cage no worse than the pass that got closest.
export function refitCageToLimitTargets(cage, targets, opts = {}) {
  const passes = opts.passes ?? 12;
  const topo = buildTopology(cage);
  const movable = smoothInteriorVertices(cage, topo).filter((v) => targets[v]);
  let cur = cage.vertices.map((v) => v.slice());
  let best = cur;
  let bestError = Infinity;
  for (let p = 0; p <= passes; p++) {
    const work = { vertices: cur, faces: cage.faces, creases: cage.creases };
    const ctx = buildTopology(work);
    let err = 0;
    const next = cur.map((v) => v.slice());
    for (const i of movable) {
      const L = vertexLimitPosition(work, i, ctx);
      const d = sub(targets[i], L);
      const m = Math.sqrt(dot(d, d));
      if (m > err) err = m;
      next[i] = add(cur[i], d);
    }
    if (err < bestError) { bestError = err; best = cur; }
    if (p === passes) break;
    cur = next;
  }
  return { vertices: best, maxError: bestError, movedCount: movable.length };
}

// LIMIT DRIFT AT THE SURVIVING VERTICES — the ranking metric, and it is a
// PROXY, stated as one: it measures how far the limit surface moved at the
// points that are still there, and says nothing about the band the collapsed
// strip used to occupy. It is the cheap half of the plan's own "rank by
// measured drift", chosen because ranking by chord LENGTH gives the opposite
// order on the two chord families of a rotationally-built cage.
export function polychordDrift(cage, chord, opts = {}) {
  const res = collapsePolychord(cage, chord, opts);
  if (!res.ok) return { ok: false, ...res };
  const beforeTopo = buildTopology(cage);
  const beforeLimit = cage.vertices.map((_, i) => vertexLimitPosition(cage, i, beforeTopo));
  const afterTopo = buildTopology(res.cage);
  const extent = cageExtent(cage);
  let worst = 0;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < res.cage.vertices.length; i++) {
    const src = res.vertexSources[i];
    let want = ZERO.slice();
    for (const s of src) want = add(want, beforeLimit[s]);
    want = scale(want, 1 / src.length);
    const got = vertexLimitPosition(res.cage, i, afterTopo);
    const d = sub(got, want);
    const m = Math.sqrt(dot(d, d));
    if (m > worst) worst = m;
    sum += m;
    count++;
  }
  return { ok: true, worst: worst / extent, mean: (count ? sum / count : 0) / extent, result: res, targets: beforeLimit };
}

// Collapsible chords, cheapest first by the drift proxy above.
export function rankPolychords(cage, opts = {}) {
  const all = polychordsOf(cage, opts);
  if (!all.ok) return all;
  const ranked = [];
  for (const chord of all.chords) {
    if (!chord.collapsible) continue;
    const d = polychordDrift(cage, chord, opts);
    if (!d.ok) continue;
    ranked.push({ chord, worst: d.worst, mean: d.mean });
  }
  ranked.sort((a, b) => a.worst - b.worst || a.chord.length - b.chord.length || a.chord.id - b.chord.id);
  return { ok: true, ranked, chords: all.chords };
}

// ---------------------------------------------------------------------------
// THE COMMAND
// ---------------------------------------------------------------------------

// SIMPLIFY ONE SUPERB CAGE. Tier 1 first, because it is exact; Tier 2 only
// where Tier 1 refuses. The tier used is named in the result so a caller can
// say which answer it got.
//
//   opts.amount           how many steps (un-subdivide levels, or chords). 1.
//   opts.tier             'auto' | 'unsubdivide' | 'polychord'.
//   opts.allowOpenChords  let a chord that ends on a boundary collapse. false.
//   opts.minFaces         the floor. 6.
//   opts.refit            refit the limit surface after each collapse. true.
export function simplifySubD(cage, opts = {}) {
  const amount = Math.max(1, Math.floor(opts.amount ?? 1));
  const tier = opts.tier ?? 'auto';
  const steps = [];
  let cur = cage;

  if (tier === 'auto' || tier === 'unsubdivide') {
    let levels = 0;
    let firstRefusal = null;
    for (let i = 0; i < amount; i++) {
      const r = unsubdivideCage(cur, opts);
      if (!r.ok) { if (i === 0) firstRefusal = r; break; }
      cur = r.cage;
      levels++;
      steps.push({ tier: 'unsubdivide', faces: cur.faces.length, residual: r.residual, unique: r.unique, recoveredWeightOneCrease: r.recoveredWeightOneCrease });
    }
    if (levels > 0) {
      return {
        ok: true,
        tier: 'unsubdivide',
        cage: cur,
        steps,
        invariants: checkSimplifyInvariants(cage, cur),
        message: `exact — this cage was a subdivision, and this is the cage it came from: ${cage.faces.length} → ${cur.faces.length} faces${levels > 1 ? ` over ${levels} levels` : ''}`,
      };
    }
    if (tier === 'unsubdivide') return firstRefusal;
    steps.push({ tier: 'unsubdivide', refused: firstRefusal.reason, message: firstRefusal.message });
  }

  let removed = 0;
  let lastRefusal = null;
  for (let i = 0; i < amount; i++) {
    const ranked = rankPolychords(cur, opts);
    if (!ranked.ok) { lastRefusal = ranked; break; }
    if (!ranked.ranked.length) {
      const reasons = new Map();
      for (const c of ranked.chords) if (c.refusal) reasons.set(c.refusal.reason, c.refusal.message);
      const first = reasons.entries().next().value;
      lastRefusal = refuse(first ? first[0] : 'NO_CHORD', first ? first[1] : 'This cage has no strip of quads that can be removed.');
      break;
    }
    const pick = ranked.ranked[0];
    const res = collapsePolychord(cur, pick.chord, opts);
    if (!res.ok) { lastRefusal = res; break; }
    let next = res.cage;
    let drift = pick.worst;
    if (opts.refit !== false) {
      const beforeTopo = buildTopology(cur);
      const beforeLimit = cur.vertices.map((_, k) => vertexLimitPosition(cur, k, beforeTopo));
      const targets = res.vertexSources.map((src) => {
        let want = ZERO.slice();
        for (const s of src) want = add(want, beforeLimit[s]);
        return scale(want, 1 / src.length);
      });
      const fit = refitCageToLimitTargets(next, targets, opts);
      next = { vertices: fit.vertices, faces: next.faces, creases: next.creases };
      drift = fit.maxError / cageExtent(cur);
    }
    steps.push({ tier: 'polychord', chord: pick.chord.id, removedFaces: pick.chord.length, faces: next.faces.length, drift });
    cur = next;
    removed++;
  }
  if (!removed) {
    return lastRefusal || refuse('NO_CHORD', 'This cage has no strip of quads that can be removed.');
  }
  const check = checkSimplifyInvariants(cage, cur);
  const drift = steps.filter((s) => s.tier === 'polychord').reduce((m, s) => Math.max(m, s.drift), 0);
  return {
    ok: true,
    tier: 'polychord',
    cage: cur,
    steps,
    invariants: check,
    message: `${cage.faces.length} → ${cur.faces.length} faces, limit surface within ${(drift * 100).toFixed(2)}% everywhere`,
  };
}
