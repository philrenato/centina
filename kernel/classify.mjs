// CLASSIFICATION — Phase 7 of the boolean pipeline, and the
// phase where booleans usually die. For every fragment the
// face split produced: is it inside the other solid, outside it, or on its
// boundary. Requicha & Voelcker 1985's boundary evaluation and merging is the
// reference for the STRATEGY; the test itself is written fresh here — that
// source is a REFERENCE, never a source of transcribed code.
//
// THREE STATES, NOT TWO. 'boundary' is a real answer and is checked FIRST,
// before any ray is cast — a fragment lying ON the other solid's surface is
// exactly the case a two-state test has to guess about, and guessing there is
// how a boolean produces a shell with a hole in it. This mirrors
// kernel/trim.mjs's own pointInUVPolygon, which reports 'boundary' separately
// for the same reason one dimension down.
//
// A MAJORITY OF CLEAN RAYS, NOT ONE RAY WITH A RETRY. The obvious approach is
// retrying with a fresh direction when a degenerate hit is detected. That is
// necessary but not sufficient: a single clean-looking cast against a mesh
// that is subtly open answers confidently and wrongly. So several independent
// directions are cast, every DEGENERATE cast is discarded outright (never
// fudged), and the surviving clean casts must AGREE. Disagreement means the
// mesh is not the closed manifold this test assumes, and that refuses by name
// rather than returning the more popular answer — the wrong answer here is
// silent, and a silent wrong answer three phases from the user is precisely
// what this whole plan exists to avoid.
//
// Degeneracy is detected, not tolerated: a ray passing within `baryEps` of a
// triangle's edge or vertex has hit a shared boundary between two triangles,
// where parity counting is ill-defined (the hit is either counted twice or
// not at all). A near-parallel grazing hit is the same problem.

import { add, sub, scale, dot, cross, length, normalize } from './vec3.mjs';
import { surfacePoint } from './surface.mjs';
import { representativeInteriorPoint } from './facesplit.mjs';

// Twice a triangle's area, divided by the square of its own longest edge —
// a scale-free measure of how collapsed it is. Normalizing against the
// LONGEST edge is what makes this hold for both ways a triangle degenerates:
// two edges turning parallel, and a single edge shrinking to nothing. An
// earlier form of this test divided by |e1||e2| and silently missed the
// second case, because a vanishing edge shrinks the divisor at the same
// rate as the area. Deliberately generous — a real triangle in any
// tessellation this app produces clears it by six orders of magnitude,
// while a revolve's own pole sliver sits at ~1e-15. See castRay.
export const DEGENERATE_TRIANGLE_AREA_RATIO = 1e-9;

// Deterministic, well-spread directions on the sphere — a golden-angle
// spiral, the standard construction. Deterministic on purpose: this kernel
// never uses Math.random (kernel/noise.mjs and kernel/curvegen.mjs both make
// the same choice), so the same solid and the same point always classify the
// same way, and a disagreement is reproducible rather than a coin flip.
export function spiralDirections(count) {
  const out = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const z = count === 1 ? 0 : 1 - (2 * i) / (count - 1) * 0.98 - 0.01;
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const theta = golden * i;
    out.push([r * Math.cos(theta), r * Math.sin(theta), z]);
  }
  return out;
}

/* ── TRIANGLE-MESH ACCELERATION for classification ────────────────────────
   Classifying a point by scanning EVERY triangle once for the boundary
   check and once per ray is O(rays × triangles) per fragment. On a real
   student file (two SubD-derived polysurfaces, 359,424 classification
   triangles each, ~660 fragments and faces to classify) that is ~63 SECONDS
   of the boolean's own time. A mid-split AABB tree over the triangles turns
   both scans into log-time queries — the identical per-triangle mathematics
   runs on the triangles that survive the box tests, so the answers do not
   change, only the time.

   Built once per triangle ARRAY and cached by identity (WeakMap): one
   boolean run reuses one operand's triangle array across every face and
   fragment it classifies against that operand, which is exactly when the
   build pays for itself. The cache keeps nothing alive — when the run drops
   the array, the tree goes with it.

   ⚠ ONE DELIBERATE SEMANTIC REFINEMENT, stated rather than hidden: an
   exhaustive castRay declares a cast degenerate when the ray lies in ANY
   triangle's plane — including a triangle nowhere near the ray's path,
   whose parity contribution is provably zero. The tree only visits
   triangles whose (padded) box the ray's forward half-line crosses, so a
   far-away coplanar triangle does not poison the whole cast. Every
   triangle the ray could genuinely hit or graze is still visited and still
   gets the full degeneracy treatment, so the votes that remain are cast by
   the identical test — this can only turn a discarded vote into a correct
   one, never the reverse. */
const TRI_ACCEL_LEAF = 8;
const TRI_ACCEL = new WeakMap();
function triAccelFor(triangles) {
  let acc = TRI_ACCEL.get(triangles);
  if (!acc) { acc = buildTriAccel(triangles); TRI_ACCEL.set(triangles, acc); }
  return acc;
}
function buildTriAccel(triangles) {
  const n = triangles.length;
  const tlo = new Float64Array(3 * n), thi = new Float64Array(3 * n), cen = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const t = triangles[i];
    for (let k = 0; k < 3; k++) {
      const l = Math.min(t[0][k], t[1][k], t[2][k]), h = Math.max(t[0][k], t[1][k], t[2][k]);
      tlo[3 * i + k] = l; thi[3 * i + k] = h; cen[3 * i + k] = (l + h) / 2;
    }
  }
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  // A node is { lo, hi, a, b, leaf }: a leaf owns order[a..b); an interior
  // node's children are nodes[a] and nodes[b]. Split at the midpoint of the
  // widest centroid axis, halving when every centroid lands on one side
  // (coincident triangles), which bounds the depth at log2(n).
  const nodes = [];
  const buildRange = (s, e) => {
    const nl = [Infinity, Infinity, Infinity], nh = [-Infinity, -Infinity, -Infinity];
    for (let i = s; i < e; i++) {
      const t = 3 * order[i];
      for (let k = 0; k < 3; k++) {
        if (tlo[t + k] < nl[k]) nl[k] = tlo[t + k];
        if (thi[t + k] > nh[k]) nh[k] = thi[t + k];
      }
    }
    const self = nodes.length;
    nodes.push({ lo: nl, hi: nh, a: s, b: e, leaf: true });
    if (e - s <= TRI_ACCEL_LEAF) return self;
    let axis = 0, widest = -1;
    const cl = [Infinity, Infinity, Infinity], ch = [-Infinity, -Infinity, -Infinity];
    for (let i = s; i < e; i++) {
      const t = 3 * order[i];
      for (let k = 0; k < 3; k++) { const c = cen[t + k]; if (c < cl[k]) cl[k] = c; if (c > ch[k]) ch[k] = c; }
    }
    for (let k = 0; k < 3; k++) if (ch[k] - cl[k] > widest) { widest = ch[k] - cl[k]; axis = k; }
    const mid = (cl[axis] + ch[axis]) / 2;
    let p = s;
    for (let i = s; i < e; i++) {
      if (cen[3 * order[i] + axis] < mid) { const t = order[p]; order[p] = order[i]; order[i] = t; p++; }
    }
    if (p === s || p === e) p = (s + e) >> 1;
    const node = nodes[self];
    node.leaf = false;
    node.a = buildRange(s, p);
    node.b = buildRange(p, e);
    return self;
  };
  if (n) buildRange(0, n);
  // The root box diagonal IS the old meshScale — same points, same measure.
  const d = n ? Math.hypot(nodes[0].hi[0] - nodes[0].lo[0], nodes[0].hi[1] - nodes[0].lo[1], nodes[0].hi[2] - nodes[0].lo[2]) : 0;
  return { nodes, order, scaleLen: Number.isFinite(d) && d > 0 ? d : 1 };
}

// Ericson, Real-Time Collision Detection §5.1.5 — the same primitive
// kernel/trim.mjs's closestPointOnTriangleMesh carries. Local copy on
// purpose: this block is kept textually identical between the kernel module
// and the app's inline twin, so it imports nothing.
function closestPtOnTri(a, b, c, p) {
  const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return a;
  const bp = sub(p, b);
  const d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return b;
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return add(a, scale(ab, d1 / (d1 - d3)));
  const cp = sub(p, c);
  const d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return c;
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return add(a, scale(ac, d2 / (d2 - d6)));
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return add(b, scale(sub(c, b), w));
  }
  const denom = 1 / (va + vb + vc);
  return add(a, add(scale(ab, vb * denom), scale(ac, vc * denom)));
}

// Exact nearest point on the soup — branch-and-bound over the tree, never a
// cutoff, so it returns byte-for-byte what the linear scan returned.
function nearestOnTriangleMesh(triangles, accel, p) {
  const { nodes, order } = accel;
  if (!nodes.length) return null;
  let best = null, bestSq = Infinity;
  const boxDistSq = (node) => {
    let s = 0;
    for (let k = 0; k < 3; k++) {
      const d = Math.max(node.lo[k] - p[k], 0, p[k] - node.hi[k]);
      s += d * d;
    }
    return s;
  };
  const visit = (ni) => {
    const node = nodes[ni];
    if (node.leaf) {
      for (let i = node.a; i < node.b; i++) {
        const t = triangles[order[i]];
        const q = closestPtOnTri(t[0], t[1], t[2], p);
        const dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
        const dq = dx * dx + dy * dy + dz * dz;
        if (dq < bestSq) { bestSq = dq; best = q; }
      }
      return;
    }
    // Nearer child first, so the far child usually prunes away whole.
    const da = boxDistSq(nodes[node.a]), db = boxDistSq(nodes[node.b]);
    if (da <= db) {
      if (da < bestSq) visit(node.a);
      if (db < bestSq) visit(node.b);
    } else {
      if (db < bestSq) visit(node.b);
      if (da < bestSq) visit(node.a);
    }
  };
  visit(0);
  return best ? { point: best, distance: Math.sqrt(bestSq) } : null;
}

/**
 * One ray cast. Returns { crossings } for a clean cast, or { degenerate,
 * why } when the result cannot be trusted — never a best guess.
 */
function castRay(triangles, origin, dir, scaleLen, accel) {
  const parallelEps = 1e-12 * scaleLen;
  const baryEps = 1e-7;
  const tEps = 1e-9 * scaleLen;
  // Boxes are padded so a hit that grazes a leaf's own wall (the very case
  // the baryEps test must see) still visits that leaf. Padding only ever
  // ADDS visited triangles, and the per-triangle test below is unchanged.
  const boxEps = 1e-7 * scaleLen;
  const { nodes, order } = accel;
  let crossings = 0;
  const stack = nodes.length ? [0] : [];
  while (stack.length) {
    const node = stack.pop();
    const nd = nodes[node];
    // Slab test of the forward half-line against the padded box; an
    // axis-parallel ray is handled by its own branch so 0 × Infinity can
    // never turn a real overlap into NaN.
    let t0 = -Infinity, t1 = Infinity, out = false;
    for (let k = 0; k < 3; k++) {
      const lo = nd.lo[k] - boxEps, hi = nd.hi[k] + boxEps;
      if (dir[k] === 0) {
        if (origin[k] < lo || origin[k] > hi) { out = true; break; }
        continue;
      }
      const ia = (lo - origin[k]) / dir[k], ib = (hi - origin[k]) / dir[k];
      const lo2 = Math.min(ia, ib), hi2 = Math.max(ia, ib);
      if (lo2 > t0) t0 = lo2;
      if (hi2 < t1) t1 = hi2;
    }
    if (out || t0 > t1 || t1 < 0) continue;
    if (!nd.leaf) { stack.push(nd.a, nd.b); continue; }
    for (let ti = nd.a; ti < nd.b; ti++) {
      const [a, b, c] = triangles[order[ti]];
      const e1 = sub(b, a), e2 = sub(c, a);
      const n = cross(e1, e2);
      const nLen = length(n);
      // A COLLAPSED TRIANGLE IS SKIPPED, NOT TREATED AS A DEGENERATE RAY.
      // `nLen` is twice the area, so this is a scale-free test on the
      // triangle's own shape: below the threshold it encloses no area a ray
      // could cross, and — the part that actually bites — its normal
      // DIRECTION is pure roundoff. Any ray is "parallel" to a normal made of
      // noise, so such a triangle would otherwise satisfy the in-plane test
      // below for essentially every direction at once and report the whole
      // cast untrustworthy. A revolve collapses its entire pole row into
      // exactly this, so ONE of them anywhere in a mesh was enough to make
      // every direction refuse, leaving no point classifiable against any
      // solid of revolution at all. Skipping is safe for the same reason the
      // exactly-zero case was already skipped: a genuine crossing of a
      // zero-area region is measure-zero, and in a closed mesh the neighbors
      // sharing its edges still register the real crossing.
      const maxEdge = Math.max(length(e1), length(e2), length(sub(c, b)));
      if (nLen <= DEGENERATE_TRIANGLE_AREA_RATIO * maxEdge * maxEdge) continue;
      const pv = cross(dir, e2);
      const det = dot(e1, pv);
      const tv = sub(origin, a);
      const qv = cross(tv, e1);
      if (Math.abs(det) < parallelEps) {
        // Parallel to this triangle's plane. Harmless unless the ray actually
        // lies IN that plane and passes through the triangle, which parity
        // cannot count. Detected via the ray's distance from the plane. The
        // normal is trustworthy here — a collapsed triangle already left.
        if (Math.abs(dot(tv, n)) / nLen < tEps) {
          return { degenerate: true, why: 'the ray lies in a triangle\'s own plane, where a crossing cannot be counted' };
      }
        continue;
      }
      const inv = 1 / det;
      const u = dot(tv, pv) * inv;
      const v = dot(dir, qv) * inv;
      const w = 1 - u - v;
      if (u < -baryEps || v < -baryEps || w < -baryEps) continue;
      const t = dot(e2, qv) * inv;
      if (t <= tEps) continue;
      // On or within baryEps of an edge or vertex: this hit is shared with the
      // neighboring triangle, so it would be counted twice or zero times.
      if (u < baryEps || v < baryEps || w < baryEps) {
        return { degenerate: true, why: 'the ray grazes a shared triangle edge or vertex, where parity counting is ill-defined' };
      }
      crossings++;
      }
  }
  return { crossings };
}

/**
 * Classify a point against a closed triangle mesh.
 *
 * `triangles` — flat array of [a,b,c] vertex triples in world space, the
 *   already-tessellated boundary of the solid being tested against.
 *
 * Returns { region: 'inside'|'outside'|'boundary', distance, raysUsed } on
 * success, or { region: null, reason } when no trustworthy answer exists.
 */
export function classifyPointInSolid(triangles, point, opts = {}) {
  if (!triangles || !triangles.length) {
    return { region: null, reason: 'the solid has no boundary geometry to classify against' };
  }
  const accel = triAccelFor(triangles);
  // The tree's root box diagonal — the same measure meshScale took, cached
  // with the tree so 660 fragments do not each rescan 359k triangles for it.
  const scaleLen = accel.scaleLen;
  // Relative by default, so this behaves the same on a 1mm part and a 1m one.
  const boundaryTol = opts.boundaryTolerance ?? scaleLen * 1e-7;

  // BOUNDARY FIRST. A point on the other solid's surface has no honest
  // inside/outside answer, and every ray cast from it would be grazing.
  const near = nearestOnTriangleMesh(triangles, accel, point);
  if (near && near.distance <= boundaryTol) {
    return { region: 'boundary', distance: near.distance, raysUsed: 0 };
  }

  const dirs = (opts.directions || spiralDirections(opts.rayCount ?? 7)).map(normalize);
  const votes = [];
  const degenerate = [];
  for (const dir of dirs) {
    const r = castRay(triangles, point, dir, scaleLen, accel);
    if (r.degenerate) { degenerate.push(r.why); continue; }
    votes.push(r.crossings % 2 === 1 ? 'inside' : 'outside');
  }
  // EVERY direction is cast, deliberately — stopping as soon as two agreed
  // would defeat the disagreement check below, which is the only thing that
  // catches an open mesh. For a genuinely closed mesh unanimity holds by
  // construction, so this costs time and never a false refusal.
  if (!votes.length) {
    return {
      region: null, raysUsed: 0,
      reason: `every ray from this point hit the boundary degenerately (${degenerate[0] || 'unknown'}) — no direction gave a countable crossing`,
    };
  }
  if (!votes.every((x) => x === votes[0])) {
    return {
      region: null, raysUsed: votes.length,
      reason: `independent rays disagree (${votes.join(', ')}) — the boundary being tested against is not the closed manifold this test requires, so neither answer can be trusted`,
    };
  }
  return { region: votes[0], distance: near ? near.distance : Infinity, raysUsed: votes.length };
}

/**
 * Classify one FRAGMENT of a split face against the other solid — the shape
 * Phase 8 actually consumes, since a boolean keeps or drops whole fragments,
 * never individual points.
 *
 * `fragment` is one entry from splitFaceByCurves' own output: {outer, holes}
 * in this surface's own (u,v) knot domain (kernel/trim.mjs's own trim-loop
 * convention — real domain values, not normalized fractions, per
 * trivialTrimLoop). The probe point is the fragment's own representative
 * INTERIOR point, not a centroid: a centroid can sit outside a non-convex
 * fragment entirely, or inside one of its holes, and a boolean that classifies
 * one fragment by a point belonging to a different region is exactly the
 * silent wrong answer this whole phase exists to avoid.
 *
 * ONE point decides the whole fragment, and that is sound only because the
 * split already ran: every intersection curve is a fragment BOUNDARY by
 * construction, so no fragment straddles the other solid's surface — its
 * interior is entirely inside or entirely outside. If that precondition is
 * ever broken (a missed intersection curve), this returns a confident wrong
 * answer, which is why splitFaceByCurves refuses rather than approximating.
 */
export function classifyFragment(srf, fragment, otherTriangles, opts = {}) {
  const uv = representativeInteriorPoint(fragment.outer, fragment.holes || []);
  if (!uv) {
    return { region: null, reason: 'the fragment has no findable interior point, so there is nothing to classify' };
  }
  const point = surfacePoint(srf, uv[0], uv[1]);
  if (!point.every((c) => Number.isFinite(c))) {
    return { region: null, uv, reason: 'the surface does not evaluate to a finite point at the fragment\'s own interior' };
  }
  const result = classifyPointInSolid(otherTriangles, point, opts);
  return { ...result, uv, point };
}

/**
 * Turn a triangle soup into a single closed-solid predicate the keep-rules of
 * Phase 8 can be written against. Union/Difference/Intersect differ ONLY in
 * which regions they keep, so they all consume exactly this.
 */
export function keepRuleFor(operation) {
  switch (operation) {
    // A of B's outside, plus B of A's outside.
    case 'union': return (region) => region === 'outside';
    // The overlap: each operand's part that lies within the other.
    case 'intersect': return (region) => region === 'inside';
    // Handled by the caller reversing the second operand, so the rule itself
    // is Intersect's — named here so the equivalence is stated, not implied.
    case 'difference': return (region) => region === 'outside';
    default: return null;
  }
}
