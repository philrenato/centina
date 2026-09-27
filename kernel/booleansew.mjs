// Sew — Phase 8 of the boolean pipeline, first half. Phase 6
// splits every face into fragments and Phase 7 decides which to keep; this
// assembles the kept ones back into one closed, manifold solid, or refuses
// by name.
//
// No new topology engine. kernel/brepbuild.mjs's `buildBrepSolid` welds
// boundary loops into a half-edge solid, checks manifoldness and
// orientability, orients every face consistently, and reports Euler
// characteristic. A boolean's sew is that same function fed the right
// loops, not a second implementation of it. What this module owns is the
// step in between: turning a fragment's (u,v) trim loops into the 3D
// boundary polygon that function expects.
//
// What makes a sew hold, and why it is tolerance-bound. Two fragments that
// came from the same surface share an edge exactly — the arrangement gave
// both the identical PSLG vertices, so both evaluate the same (u,v) and land
// on the same 3D point bit for bit. Two fragments from different surfaces
// share the intersection curve, and there the two surfaces agree only to the
// marcher's own residual: each face evaluates its own surface at its own
// parameters, and those two points are close, never identical. So the weld
// tolerance is what closes the shell across an intersection edge, and a
// residual larger than it leaves a hairline crack that shows up as naked
// edges rather than silently. The residual is reported alongside the
// tolerance so a caller can say which one was at fault instead of guessing.
//
// Boundaries are polygonal in the topology, exact in the geometry. A
// fragment's boundary is taken at its own trim-loop vertices, and each
// face carries its real trimmed NURBS surface through `attachSurface`, so
// the polygon is the topology and the surface is the shape — the same
// convention `buildBrepSolid` and every consumer of it follow.
//
// A shared edge is never densified, and a face's own domain boundary
// always is. The two halves of that rule pull opposite ways and both are
// load-bearing. An intersection edge is sampled by the marcher and its
// samples are the same 3-D points on both faces; adding intermediate
// points there would sample uniform fractions of one surface's (u,v),
// which are not uniform fractions of the other's, so the extra points on
// either side would not weld and a shell about to close would come apart.
// A domain-boundary span is the opposite: nothing cut it, so it arrives as
// its two endpoints, and where that boundary is curved in 3-D the polygon
// carries its chord. That is not a coarse topology but a wrong one, since
// every test below — the weld, the T-junction pass, the naked/paired edge
// census — is a polyline comparison at the sew tolerance. `refineUVLoop`
// therefore bisects domain-boundary spans alone, until each is within
// tolerance of its own geometry.

import { surfacePoint, surfaceClosure } from './surface.mjs';
import { buildBrepSolid, brepVerdict, weldPoints } from './brepbuild.mjs';
import { splitAnnulusTwoBridges } from './trimtess.mjs';
import { add, dot, length, scale, sub } from './vec3.mjs';
import { classifyFragment, keepRuleFor } from './classify.mjs';

/**
 * Apply one operator's keep-rule to a face's fragments — the step that makes
 * "Union, Difference and Intersect are the same machine" literally true:
 * everything before this is shared, and the only thing that differs is which
 * region survives.
 *
 * `operand` says which solid these fragments came from. Difference is
 * Intersect against a reversed second operand, so B's own fragments keep the
 * opposite region from A's — encoded here rather than left for each caller to
 * remember, because getting it backwards produces a plausible solid that is
 * silently the wrong one.
 *
 * A 'boundary' fragment refuses. It means this face lies on the other solid's
 * surface, which is the coincident/tangential-face case this pipeline puts
 * out of scope by name — keeping it would double the shell there and dropping
 * it would leave a hole.
 */
export function keepFragments(srf, fragments, otherTriangles, operation, opts = {}) {
  const operand = opts.operand ?? 'a';
  const effective = operation === 'difference' && operand === 'b' ? 'intersect' : operation;
  const rule = keepRuleFor(effective);
  if (!rule) return { ok: false, reason: `unknown boolean operation "${operation}"` };

  const kept = [];
  const classifications = [];
  for (let i = 0; i < fragments.length; i++) {
    const r = classifyFragment(srf, fragments[i], otherTriangles, opts);
    classifications.push(r);
    if (!r.region) {
      return { ok: false, reason: `fragment ${i} could not be classified — ${r.reason}`, classifications };
    }
    if (r.region === 'boundary') {
      return {
        ok: false,
        reason: `fragment ${i} lies ON the other solid's surface — coincident and tangential faces are out of scope, so this refuses rather than guessing whether to keep it`,
        classifications,
      };
    }
    if (rule(r.region)) kept.push({ srf, ...fragments[i] });
  }
  return { ok: true, kept, classifications };
}

/**
 * One kept fragment's (u,v) trim loops, evaluated onto its own surface.
 * Returns { ok, loops } — a fragment can legitimately need more than one
 * face loop, so this returns a list rather than a single boundary.
 *
 * A fragment with a hole is split, not slit. `mergeLoopsKeyhole` is the
 * standard answer for tessellation, and it is the wrong one here: its
 * bridge necessarily visits its own corners twice, and a repeated vertex
 * in a face loop is exactly what `buildBrepSolid` rejects — correctly,
 * since a face boundary traverses each corner once.
 * `splitAnnulusTwoBridges` cuts the annulus into two simple
 * faces sharing both bridge edges instead.
 *
 * A degenerate outer loop carries no real edge, and this is ordinary on
 * organic geometry rather than exotic. A closed revolve's own domain
 * rectangle has poles at two of its corners — an entire (u,v) row
 * collapses to one 3D point — and its two v-ends are the same seam. So
 * the outer loop of an uncut face on such a surface can evaluate to as
 * few as two distinct 3D points: no area, no boundary, nothing for the
 * welder to sew. When that happens and the fragment has exactly one hole
 * (the cut curve), the hole is the face's only real boundary, reversed to
 * carry the outward winding the vanished outer loop would have had. The
 * pole and seam stay interior to the face, where they belong.
 */
// The two copies of a seam must carry the same points, by construction.
//
// A closed direction's domain edges at aMin and aMax are the same 3D curve.
// After splitAtSeam divides a fragment, one piece ends on aMin and another
// begins at aMax, and each keeps only the points its own chain happened to
// leave there — 4 points on one side and 10 on the other, say. Two different
// point sets along one curve cannot weld.
//
// densifyOnCutLines makes exactly this argument for cut lines: both pieces
// take their points from the same endpoints by the same rule, so they weld
// by construction. A domain-boundary edge is not a cut line, so this extends
// the argument to it: take the union of the other-axis parameters appearing
// on either side, and give both sides all of them.
//
// Why not leave this to the T-junction pass, which exists for this class of
// mismatch: it cannot do it. insertTJunctionVertices tests a foreign vertex
// against an edge's straight chord, and a seam on a curved surface is a
// curve — a seam edge 53-55 units long can deviate 1.3-2.5 units from its
// chord. Widening that tolerance to reach them would admit points 2 units
// off a chord and start welding distinct geometry.
function harmonizeSeamAxis(loops, ai, knots) {
  const lo = knots[0], hi = knots[knots.length - 1];
  const eps = Math.max((hi - lo) * 1e-9, 1e-12);
  const oi = 1 - ai;
  const onSide = (p, side) => Math.abs(p[ai] - side) <= eps;
  // Every other-axis value either copy places on the seam, both sides pooled.
  const vals = [];
  for (const loop of loops) {
    for (const p of loop) if (onSide(p, lo) || onSide(p, hi)) vals.push(p[oi]);
  }
  if (!vals.length) return loops;
  vals.sort((a, b) => a - b);
  return loops.map((loop) => {
    const original = loop;
    const out = [];
    for (let i = 0; i < loop.length; i++) {
      const a = loop[i], b = loop[(i + 1) % loop.length];
      out.push(a);
      // Only an edge running along one side of the seam gets extra points; an
      // edge merely touching it at one end is an ordinary interior edge.
      const side = (onSide(a, lo) && onSide(b, lo)) ? lo : (onSide(a, hi) && onSide(b, hi)) ? hi : null;
      if (side === null) continue;
      const from = a[oi], to = b[oi];
      if (Math.abs(to - from) <= eps) continue;
      const step = to > from ? 1 : -1;
      const between = vals.filter((v) => (v - from) * step > eps && (to - v) * step > eps);
      if (step < 0) between.reverse();
      let last = from;
      for (const v of between) {
        if (Math.abs(v - last) <= eps) continue;
        out.push(ai === 0 ? [side, v] : [v, side]);
        last = v;
      }
    }
    // Same do-no-harm rule the cut lines carry: a loop that would visit one
    // (u,v) twice keeps the points it already had. Reachable because the pool
    // spans every fragment on this surface rather than one fragment's own.
    return repeatsAPoint(out, eps) ? original : out;
  });
}
function harmonizeDomainSeams(uvLoops, srf) {
  const { closedU, closedV } = surfaceClosure(srf);
  let out = uvLoops;
  if (closedU) out = harmonizeSeamAxis(out, 0, srf.knotsU);
  if (closedV) out = harmonizeSeamAxis(out, 1, srf.knotsV);
  return out;
}

// An out-and-back along one line is not a boundary, and it is what survives
// every other cleanup here. A clip bridge can overshoot the crossing it was
// heading for and return — e.g. a loop that walks the u = 1/3 cut up to
// v = 0.6471 and comes straight back down to v = 0.6085: three collinear
// points enclosing no area at all. `splitSelfTouching` cannot see it because
// the two traversals share no vertex — the way out is densified and the way
// back is a single segment — and `hasRepeatedPoint` cannot see it because no
// point is repeated exactly. The neighboring piece's boundary stops at
// 0.6085, so the overshoot has no partner and reads as naked edges.
//
// Only a reversal is dropped, never a merely collinear point: three collinear
// (u,v) samples generally map to a curved 3D polyline, so dropping the middle
// one would move the boundary. A reversal is different — the excursion covers
// the same stretch of the same curve twice, so removing its tip removes both
// copies and changes neither the area nor the 3D path.
function dropReversalSpikes(loop) {
  let cur = loop;
  for (;;) {
    const n = cur.length;
    if (n < 3) return cur;
    let tip = -1;
    for (let i = 0; i < n; i++) {
      const a = cur[(i - 1 + n) % n], b = cur[i], c = cur[(i + 1) % n];
      const d1 = [b[0] - a[0], b[1] - a[1]], d2 = [c[0] - b[0], c[1] - b[1]];
      const m1 = Math.hypot(d1[0], d1[1]), m2 = Math.hypot(d2[0], d2[1]);
      if (m1 <= 0 || m2 <= 0) continue;
      const cross = d1[0] * d2[1] - d1[1] * d2[0];
      const dot = d1[0] * d2[0] + d1[1] * d2[1];
      if (Math.abs(cross) <= 1e-12 * m1 * m2 && dot < 0) { tip = i; break; }
    }
    if (tip < 0) return cur;
    cur = cur.slice(0, tip).concat(cur.slice(tip + 1));
  }
}

// The maximal stretches of a loop that lie along one line of constant `ai`,
// as vertex index ranges. A single vertex merely touching the line is a
// crossing, not a stretch along it, so a run needs at least two.
function lineRuns(loop, ai, value, eps) {
  const n = loop.length;
  const on = loop.map((p) => Math.abs(p[ai] - value) <= eps);
  const runs = [];
  if (on.every((x) => x)) {
    runs.push({ idx: loop.map((_, i) => i) });
  } else {
    for (let s = 0; s < n; s++) {
      if (!on[s] || on[(s - 1 + n) % n]) continue; // maximal runs only
      const idx = [];
      for (let i = s; on[i] && idx.length <= n; i = (i + 1) % n) idx.push(i);
      if (idx.length >= 2) runs.push({ idx });
    }
  }
  const oi = 1 - ai;
  for (const r of runs) {
    r.params = r.idx.map((i) => loop[i][oi]);
    r.lo = Math.min(...r.params);
    r.hi = Math.max(...r.params);
  }
  return runs;
}

// Per run, not per line.
//
// splitAtSeam's own cut lines are shared by the sibling pieces it just made,
// and densifyOnCutLines gives each piece its points from that piece's own two
// endpoints. That welds by construction only while both sides were cut the
// same way. They are not always: a piece spanning a second closed direction
// is divided again, so one side of a cut carries two stretches where the
// other carries one, and the trim boundary leaves the line at a different
// parameter on each side — e.g. one stretch of u = 1/3 holding 14 points
// against 8 + 8 on the other side, with four parameters present on one side
// only: T-junctions.
//
// Pooling every parameter onto a whole cut line, as a domain seam does, is
// wrong here. A domain end is visited once by a piece's loop, but an interior
// line can be traversed several times by the same loop, and a value belonging
// to one traverse inserted into another repeats a (u,v), which is exactly what
// makes a fragment read as not-a-disk. So each run takes only the parameters
// that fall strictly inside its own stretch, and only from runs on other
// loops.
//
// Guard: if this does produce a repeated (u,v), the harmonized loops are
// dropped and the originals stand.
function harmonizeCutLine(loops, ai, value, srf) {
  const oi = 1 - ai;
  const aKnots = ai === 0 ? srf.knotsU : srf.knotsV;
  const oKnots = oi === 0 ? srf.knotsU : srf.knotsV;
  const aEps = Math.max((aKnots[aKnots.length - 1] - aKnots[0]) * 1e-9, 1e-12);
  const oEps = Math.max((oKnots[oKnots.length - 1] - oKnots[0]) * 1e-9, 1e-12);

  const perLoop = loops.map((loop) => lineRuns(loop, ai, value, aEps));
  const all = [];
  perLoop.forEach((runs, li) => runs.forEach((r) => all.push({ li, ...r })));
  if (all.length < 2) return loops;

  const out = loops.map((loop, li) => {
    const runs = perLoop[li];
    if (!runs.length) return loop;
    // Which run, if any, owns each edge — an edge belongs to a run when the
    // run holds both its endpoints consecutively.
    const owner = new Map();
    runs.forEach((r, ri) => {
      for (let k = 0; k + 1 < r.idx.length; k++) owner.set(r.idx[k], ri);
    });
    const pools = runs.map((r) => {
      const vals = [];
      for (const other of all) {
        if (other.li === li) continue;
        if (Math.min(r.hi, other.hi) - Math.max(r.lo, other.lo) <= oEps) continue;
        for (const v of other.params) if (v > r.lo + oEps && v < r.hi - oEps) vals.push(v);
      }
      return vals.sort((x, y) => x - y);
    });
    const rebuilt = [];
    for (let i = 0; i < loop.length; i++) {
      const a = loop[i], b = loop[(i + 1) % loop.length];
      rebuilt.push(a);
      const ri = owner.get(i);
      if (ri === undefined) continue;
      const from = a[oi], to = b[oi];
      if (Math.abs(to - from) <= oEps) continue;
      const step = to > from ? 1 : -1;
      const between = pools[ri].filter((v) => (v - from) * step > oEps && (to - v) * step > oEps);
      if (step < 0) between.reverse();
      let last = from;
      for (const v of between) {
        if (Math.abs(v - last) <= oEps) continue;
        rebuilt.push(ai === 0 ? [value, v] : [v, value]);
        last = v;
      }
    }
    return rebuilt;
  });

  return out.map((loop, i) => (repeatsAPoint(loop, oEps) ? loops[i] : loop));
}

// A harmonized loop that visits one (u,v) twice reads as not-a-disk later and
// takes its whole fragment down with it, so it is dropped in favor of the loop
// it came from. Per loop, not per line: one loop that cannot take the extra
// points is no reason to leave every other loop on that line unwelded.
function repeatsAPoint(loop, eps) {
  for (let i = 0; i < loop.length; i++) {
    for (let j = i + 1; j < loop.length; j++) {
      if (Math.abs(loop[i][0] - loop[j][0]) <= eps && Math.abs(loop[i][1] - loop[j][1]) <= eps) return true;
    }
  }
  return false;
}

function harmonizeCutLines(uvLoops, cuts, srf) {
  let out = uvLoops;
  for (const { ai, value } of cuts) out = harmonizeCutLine(out, ai, value, srf);
  return out;
}

/**
 * A fragment's boundary in the surface's own (u,v), split into simple faces
 * but not yet harmonized against anything.
 *
 * Separated from the 3D evaluation because harmonizing a seam is a question
 * about a whole surface, not about one fragment: the two copies of a closed
 * direction's domain edge can belong to different fragments, and a fragment
 * that only ever sees its own loops cannot give them matching points — e.g.
 * a small lens fragment crossing the seam with one straight edge where the
 * large fragment's pieces carry a vertex partway along it.
 */
export function fragmentUVLoops(srf, fragment) {
  const outer = stripClosingPoint(fragment.outer || []);
  if (outer.length < 3) return { ok: false, reason: 'fewer than three distinct corners' };
  const holes = (fragment.holes || []).map(stripClosingPoint).filter((h) => h.length >= 3);

  let uvLoops;
  let cutLines = [];
  if (!holes.length) {
    const wrapped = splitAtSeam(outer, srf);
    if (wrapped && !wrapped.ok) return wrapped;
    uvLoops = wrapped ? wrapped.loops : [outer];
    if (wrapped) cutLines = wrapped.cuts;
  } else if (holes.length === 1 && collapsesIn3D(outer, srf)) {
    uvLoops = [holes[0].slice().reverse()];
  } else if (holes.length === 1) {
    const split = splitAnnulusTwoBridges(outer, holes[0]);
    if (!split) {
      return { ok: false, reason: 'a holed fragment has no pair of independent, non-crossing bridges to split it into two simple faces' };
    }
    uvLoops = split;
  } else {
    return { ok: false, reason: `a fragment with ${holes.length} holes cannot yet be split into simple faces — only one hole is handled` };
  }

  uvLoops = uvLoops.map(dropReversalSpikes).filter((l) => l.length >= 3);
  if (!uvLoops.length) return { ok: false, reason: 'a fragment boundary carries no stretch that encloses area' };
  return { ok: true, uvLoops, cutLines };
}

// The 3D half: evaluate the surface at each (u,v) and check that what comes
// back is a face rather than a collapsed or self-touching polygon.
//
// Alongside each 3D loop it returns the (u,v) each surviving point came from,
// index for index. That correspondence is what lets a later pass ask the one
// question a bare 3D polygon cannot answer: what curve does this edge actually
// follow between its two corners? (An edge's true path is the image of the
// straight (u,v) segment through the face's own surface — the module-header
// convention "polygonal in the topology, exact in the geometry" made
// addressable.)
/* A trim loop's edge is a straight line in (u,v), and its 3-D image is not.
 * Everything downstream of here compares polylines at the sew tolerance —
 * welding, T-junction insertion, the naked/paired edge census — so an edge
 * whose polyline misses its own geometry by more than that tolerance is not a
 * coarse topology, it is a wrong one.
 *
 * Only a span lying along the face's own domain boundary is refined, and the
 * restriction is the whole of what makes this safe. A span of a cut curve is
 * already sampled by the march, and its samples are the same 3-D points on
 * both of the faces that curve separates; bisecting it would give each face
 * its own intermediate points in its own parameters, and the two sides would
 * stop pairing. A domain-boundary span is the opposite case: nothing cut it,
 * so it arrives as its two endpoints alone. On a face bounded by straight
 * lines that is exact. On a curved boundary — a disc's rim, a tube's end
 * circle — it is the chord, and two things go wrong at once. The chord across
 * a circular segment's arc is that segment's own cut chord, so the loop
 * retraces itself and the builder refuses it as a repeated corner; and where
 * the polygon does survive, the face meeting it along that rim carries its own
 * chord, and two chords of one arc do not pair.
 *
 * Such a span is bisected in (u,v) until the 3-D midpoint sits within `tol` of
 * its own chord. Bisection of the parameter interval is deterministic, so two
 * faces meeting along one surface boundary refine it to the same points.
 * Deviation-driven, therefore self-limiting: a boundary that is already
 * straight is left exactly as it was.
 */
// The finest the lattice below is allowed to get: a knot span is cut into at
// most 2^6 pieces however curved it is, so a pathological boundary costs a
// bounded number of points rather than growing without limit.
const UV_REFINE_MAX_LEVEL = 6;

// One sample set per (surface, domain edge, tolerance). Cached because every
// fragment of a surface asks the same question and the answer is a property of
// the surface, not of the fragment.
const domainEdgeSampleCache = new WeakMap();

function pointToSegmentDistance(p, a, b) {
  const ab = sub(b, a);
  const len2 = dot(ab, ab);
  if (!(len2 > 0)) return length(sub(p, a));
  let t = dot(sub(p, a), ab) / len2;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return length(sub(p, add(a, scale(ab, t))));
}

function edgePoint(srf, varyAxis, fixed, t) {
  return varyAxis === 0 ? surfacePoint(srf, t, fixed) : surfacePoint(srf, fixed, t);
}

/* Where one domain edge has to be sampled, as a property of that edge alone.
 *
 * Two faces meeting along one surface boundary must place their interior
 * points at identical parameters or the two sides pair nowhere along the edge,
 * so the sample set is derived from the surface and the tolerance and from
 * nothing about the fragment being refined.
 *
 * It is the distinct knots of the varying direction, each span then cut into
 * 2^L equal pieces at one level chosen for the whole edge. The knots come
 * first because they are where the curve's own breaks live: a degree-1
 * boundary — a polyline profile's rim — is exact at its knots and nowhere
 * else, and no subdivision of a span that straddles a corner ever reaches it.
 * The lattice inside a span is a power of two so a coarser level's samples are
 * a subset of a finer one's, and the level is common to the edge so that where
 * the level is decided cannot depend on which fragment asked.
 */
function domainEdgeSamples(srf, varyAxis, fixed, tol) {
  let perSurface = domainEdgeSampleCache.get(srf);
  if (!perSurface) { perSurface = new Map(); domainEdgeSampleCache.set(srf, perSurface); }
  const key = `${varyAxis}|${fixed}|${tol}`;
  const cached = perSurface.get(key);
  if (cached) return cached;

  // Some constructed faces already own a shared topological boundary. Its
  // parameters are authoritative: independently refining the same curved rim
  // on the neighboring face creates two incompatible edge vertex sequences.
  const declared = Array.isArray(srf.boundarySampleParams)
    ? srf.boundarySampleParams.find((e) => e && e.varyAxis === varyAxis
      && Math.abs(e.fixed - fixed) <= Math.max(1, Math.abs(fixed)) * 1e-12)
    : null;
  if (declared && Array.isArray(declared.params) && declared.params.length >= 2) {
    const out = declared.params.slice();
    perSurface.set(key, out);
    return out;
  }

  const knots = varyAxis === 0 ? srf.knotsU : srf.knotsV;
  const lo = knots[0], hi = knots[knots.length - 1];
  const eps = Math.abs(hi - lo) * 1e-9;
  const breaks = [];
  for (const k of knots) {
    if (k < lo - eps || k > hi + eps) continue;
    if (!breaks.length || Math.abs(k - breaks[breaks.length - 1]) > eps) breaks.push(k);
  }
  if (breaks.length < 2) { perSurface.set(key, [lo, hi]); return [lo, hi]; }

  let level = 0;
  for (; level < UV_REFINE_MAX_LEVEL; level++) {
    const n = 1 << level;
    let within = true;
    for (let s = 0; s + 1 < breaks.length && within; s++) {
      for (let i = 0; i < n && within; i++) {
        const ta = breaks[s] + ((breaks[s + 1] - breaks[s]) * i) / n;
        const tb = breaks[s] + ((breaks[s + 1] - breaks[s]) * (i + 1)) / n;
        const pa = edgePoint(srf, varyAxis, fixed, ta);
        const pb = edgePoint(srf, varyAxis, fixed, tb);
        const pm = edgePoint(srf, varyAxis, fixed, (ta + tb) / 2);
        if (!pa.every(Number.isFinite) || !pb.every(Number.isFinite) || !pm.every(Number.isFinite)) continue;
        if (pointToSegmentDistance(pm, pa, pb) > tol) within = false;
      }
    }
    if (within) break;
  }

  const n = 1 << level;
  const out = [breaks[0]];
  for (let s = 0; s + 1 < breaks.length; s++) {
    for (let i = 1; i <= n; i++) out.push(breaks[s] + ((breaks[s + 1] - breaks[s]) * i) / n);
  }
  perSurface.set(key, out);
  return out;
}

// The sample parameters strictly inside (a, b), in the direction of travel.
function edgeSamplesBetween(samples, a, b) {
  const from = Math.min(a, b), to = Math.max(a, b);
  const eps = Math.abs(to - from) * 1e-9;
  const out = [];
  for (const t of samples) { if (t > from + eps && t < to - eps) out.push(t); }
  return a <= b ? out : out.reverse();
}

function refineUVLoop(srf, uv, tol) {
  if (!(tol > 0) || uv.length < 3) return uv;
  const uDom = [srf.knotsU[0], srf.knotsU[srf.knotsU.length - 1]];
  const vDom = [srf.knotsV[0], srf.knotsV[srf.knotsV.length - 1]];
  const uEps = Math.abs(uDom[1] - uDom[0]) * 1e-9;
  const vEps = Math.abs(vDom[1] - vDom[0]) * 1e-9;
  // Which domain edge, if any, a span runs along: the axis that stays fixed
  // decides, and the other one is the one the samples live on.
  const edgeOf = (a, b) => {
    for (const end of [0, 1]) {
      if (Math.abs(a[0] - uDom[end]) <= uEps && Math.abs(b[0] - uDom[end]) <= uEps) return { varyAxis: 1, fixed: uDom[end], at: 1 };
    }
    for (const end of [0, 1]) {
      if (Math.abs(a[1] - vDom[end]) <= vEps && Math.abs(b[1] - vDom[end]) <= vEps) return { varyAxis: 0, fixed: vDom[end], at: 0 };
    }
    return null;
  };
  const out = [];
  let grew = false;
  for (let i = 0; i < uv.length; i++) {
    const a = uv[i], b = uv[(i + 1) % uv.length];
    out.push(a);
    const edge = edgeOf(a, b);
    if (!edge) continue;
    const samples = domainEdgeSamples(srf, edge.varyAxis, edge.fixed, tol);
    const ts = edgeSamplesBetween(samples, a[edge.at], b[edge.at]);
    if (!ts.length) continue;
    grew = true;
    for (const t of ts) out.push(edge.at === 0 ? [t, edge.fixed] : [edge.fixed, t]);
  }
  return grew ? out : uv;
}

function evaluateUVLoops(srf, uvLoops, tol = 0) {
  const loops = [];
  const uvs = [];
  for (const raw of uvLoops) {
    const uv = refineUVLoop(srf, raw, tol);
    const pts = uv.map(([u, v]) => surfacePoint(srf, u, v));
    if (!pts.every((p) => p.every((c) => Number.isFinite(c)))) {
      return { ok: false, reason: 'the surface does not evaluate finitely on this fragment' };
    }
    // A pole is a whole (u,v) row collapsed to one point, so the two domain
    // edges leaving it evaluate to the same 3D vertex twice in a row. That
    // is a zero-length edge, not a real one — dropping it turns a
    // degenerate-looking loop into the triangle/quad it is, and is the reason
    // a revolved cap can be sewn at all.
    const welded = weldConsecutivePaired(pts, uv);
    if (welded.pts.length < 3) {
      return { ok: false, reason: 'a fragment boundary collapses to fewer than three distinct points on this surface' };
    }
    if (hasRepeatedPoint(welded.pts)) {
      return { ok: false, reason: 'a fragment boundary visits the same 3D point twice without being a closed band this can split — the face is not a disk' };
    }
    loops.push(welded.pts);
    uvs.push(welded.uvs);
  }
  return { ok: true, loops, uvs };
}
// One fragment on its own. `sewFragments` harmonizes across every fragment of
// a surface instead, which is strictly better where a seam is shared; this
// serves a caller holding a single fragment and nothing to share with.
export function fragmentBoundaries3D(srf, fragment, opts = {}) {
  const prepared = fragmentUVLoops(srf, fragment);
  if (!prepared.ok) return prepared;
  let uvLoops = harmonizeDomainSeams(prepared.uvLoops, srf);
  uvLoops = harmonizeCutLines(uvLoops, prepared.cutLines, srf);
  return evaluateUVLoops(srf, uvLoops, opts.tolerance ?? 0);
}

// A face whose domain spans a closed direction end to end is a tube, not a
// disk: its two v-ends are the same physical place, so its boundary polygon
// visits that seam twice and `buildBrepSolid` rejects it — correctly, since
// a face boundary traverses each corner once. Cutting the domain across
// that direction gives disks sharing the seam edges, the same "split rather
// than slit" move a holed fragment gets from `splitAnnulusTwoBridges`.
//
// Returns null when the fragment does not span a closed direction (the
// ordinary case, left untouched), a refusal when it does but cannot be
// split, and the two halves otherwise.
function splitAtSeam(loop, srf) {
  const { closedU, closedV } = surfaceClosure(srf);
  const doms = [
    [srf.knotsU[0], srf.knotsU[srf.knotsU.length - 1]],
    [srf.knotsV[0], srf.knotsV[srf.knotsV.length - 1]],
  ];
  for (let ai = 0; ai < 2; ai++) {
    if (!(ai === 0 ? closedU : closedV)) continue;
    const [aMin, aMax] = doms[ai];
    const eps = (aMax - aMin) * 1e-9;
    let lo = Infinity, hi = -Infinity;
    for (const p of loop) { lo = Math.min(lo, p[ai]); hi = Math.max(hi, p[ai]); }
    if (!(lo <= aMin + eps && hi >= aMax - eps)) continue;
    // Thirds, not halves. Halving a polar cap leaves each half's rim chord
    // running from one side of the circle to the other — a diameter, which
    // passes exactly through the pole the cap already carries as its own
    // vertex. The face's boundary polygon is then three collinear points with
    // no area at all, and the T-junction pass splits that chord at the pole
    // and reports a repeated corner. A third of the way round subtends 120
    // degrees, so no piece's chord can pass through the center.
    const span = aMax - aMin;
    const cuts = [aMin + span / 3, aMin + (2 * span) / 3];
    const pieces = [
      clipHalfPlane(loop, ai, cuts[0], true),
      clipHalfPlane(clipHalfPlane(loop, ai, cuts[0], false), ai, cuts[1], true),
      clipHalfPlane(loop, ai, cuts[1], false),
    ];
    if (pieces.some((p) => p.length < 3)) {
      return { ok: false, reason: `a fragment wrapping the surface's own closed ${ai === 0 ? 'u' : 'v'} direction did not divide into simple faces` };
    }
    // The cut itself carries no samples, and on a pole-bearing surface that
    // is the difference between a face and nothing at all. A piece of a
    // sphere between two cut meridians is a lune: its other two edges are the
    // pole rows, each of which collapses to a single 3D point, so the four
    // corners the clip produced evaluate to just two distinct places and the
    // piece reads as degenerate. Giving the cut its own interior samples
    // turns it back into the closed boundary it is — pole, meridian,
    // pole, meridian. On a surface with no pole (a cylinder wall) the four
    // corners are already distinct and this only refines them.
    //
    // This is not the densification the header warns against. That trap is
    // inventing points on an edge shared with another surface, where the two
    // sides parametrize differently and the invented points land nowhere on
    // each other. A cut line here is shared only by the two pieces this
    // function just made, both on the same surface, and both take their
    // points from the same two endpoints by the same rule — so they weld by
    // construction.
    const oi = 1 - ai;
    const divided = [];
    for (const p of pieces) {
      for (const simple of splitSelfTouching(densifyOnCutLines(p, ai, oi, cuts, srf))) divided.push(simple);
    }
    // One cut is enough only for a surface closed in one direction. A torus is
    // closed in both, so a fragment covering its whole domain is a punctured
    // torus rather than a punctured sphere: dividing it across u leaves three
    // pieces that each still run the full v range, and a piece whose two
    // v-ends are the same physical place is a tube, which `hasRepeatedPoint`
    // rejects for the same correct reason the undivided fragment was rejected.
    // So each piece is offered the same division again for the direction it
    // still spans. This terminates because a piece is strictly inside the cuts
    // made for `ai` and so can never span `ai` a second time, leaving at most
    // one closed direction to handle per level.
    // The cut lines travel back out with the pieces: the sew has to harmonize
    // the two sides of each one, and only this function knows where they are.
    const madeCuts = cuts.map((value) => ({ ai, value }));
    const out = [];
    for (const piece of divided) {
      const again = splitAtSeam(piece, srf);
      if (again && !again.ok) return again;
      if (again) {
        out.push(...again.loops);
        for (const c of again.cuts) {
          if (!madeCuts.some((m) => m.ai === c.ai && Math.abs(m.value - c.value) <= eps)) madeCuts.push(c);
        }
      } else out.push(piece);
    }
    return { ok: true, loops: out, cuts: madeCuts };
  }
  return null;
}

// A Sutherland-Hodgman clip returns one polygon even when the region is two,
// and that is what this undoes. Clipping a non-convex subject against a half
// plane is exact in the region it keeps but not in the polygon it writes: when
// the subject crosses the plane more than twice, the separate surviving
// regions come back joined by a zero-width bridge running along the clip line,
// traversed once in each direction. That bridge makes the boundary visit the
// same point twice, which `hasRepeatedPoint` rejects — correctly, since a
// face boundary traverses each corner once.
//
// Splitting at the repeated vertex separates the excursion from the loop that
// carries it. A bridge of fewer than three points encloses no area and is
// dropped; anything larger is a second region and becomes its own
// face. The match is on (u,v) rather than 3D because both copies of a bridge
// vertex are written from the same clip arithmetic and so agree exactly, while
// two points that merely evaluate near each other in 3D (the seam) are a
// different question this must not answer.
function signedAreaUV(loop) {
  let a = 0;
  for (let i = 0; i < loop.length; i++) {
    const p = loop[i], q = loop[(i + 1) % loop.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a / 2;
}

function splitSelfTouching(loop) {
  // Both sides of a split go back on the worklist. A subject crossing the clip
  // line many times carries one bridge per crossing pair, and the piece cut
  // away at the first repeat can hold every remaining one — so a pass that
  // banks the excursion and keeps examining only the remainder returns a loop
  // that still visits a point twice. Neither side is finished until it is
  // examined and found clean.
  //
  // Area, not point count, decides whether a finished piece is a face. A bridge
  // that ran along the cut line and back carries one vertex per densified
  // sample, so counting points keeps a collinear zero-area sliver as though it
  // enclosed something.
  const out = [];
  const pending = [loop];
  while (pending.length) {
    const cur = pending.pop();
    if (cur.length < 3) continue;
    let cut = null;
    search: for (let i = 0; i < cur.length; i++) {
      for (let j = i + 1; j < cur.length; j++) {
        if (Math.abs(cur[i][0] - cur[j][0]) < 1e-12 && Math.abs(cur[i][1] - cur[j][1]) < 1e-12) { cut = [i, j]; break search; }
      }
    }
    if (!cut) {
      if (Math.abs(signedAreaUV(cur)) > 1e-18) out.push(cur);
      continue;
    }
    pending.push(cur.slice(cut[0], cut[1]));
    pending.push(cur.slice(0, cut[0]).concat(cur.slice(cut[1])));
  }
  return out.length ? out : [loop];
}

// Interior samples for every edge lying along one of the cut lines. Corners
// land on the surface's own knots in the crossed direction first — that is
// where its basis can change, so it is where a boundary polygon most needs a
// corner — with a uniform floor so a single-span surface still gets a real
// edge rather than a bare chord.
const CUT_EDGE_MIN_SEGMENTS = 4;
function densifyOnCutLines(loop, ai, oi, cuts, srf) {
  const knots = oi === 0 ? srf.knotsU : srf.knotsV;
  const eps = 1e-9;
  const out = [];
  for (let i = 0; i < loop.length; i++) {
    const a = loop[i], b = loop[(i + 1) % loop.length];
    out.push(a);
    const onCut = cuts.some((c) => Math.abs(a[ai] - c) < eps && Math.abs(b[ai] - c) < eps);
    const dO = b[oi] - a[oi];
    if (!onCut || Math.abs(dO) < eps) continue;
    const ts = new Set();
    for (const k of knots) {
      const t = (k - a[oi]) / dO;
      if (t > eps && t < 1 - eps) ts.add(t);
    }
    for (let s = 1; s < CUT_EDGE_MIN_SEGMENTS; s++) ts.add(s / CUT_EDGE_MIN_SEGMENTS);
    for (const t of [...ts].sort((x, y) => x - y)) {
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
  }
  return out;
}

// Sutherland-Hodgman against one axis-aligned half-plane. A single plane is
// the whole clip window here, so this stays exact for a non-convex subject
// too — the case a general convex-window clip could not be trusted with.
function clipHalfPlane(loop, axis, value, keepBelow) {
  const inside = (p) => (keepBelow ? p[axis] <= value : p[axis] >= value);
  const out = [];
  for (let i = 0; i < loop.length; i++) {
    const a = loop[i], b = loop[(i + 1) % loop.length];
    const ain = inside(a), bin = inside(b);
    if (ain) out.push(a.slice());
    if (ain !== bin) {
      const da = a[axis] - value, db = b[axis] - value;
      const t = da / (da - db);
      const cut = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
      cut[axis] = value;
      out.push(cut);
    }
  }
  return stripClosingPoint(dedupeConsecutive2D(out));
}

function dedupeConsecutive2D(loop) {
  const out = [];
  for (const p of loop) {
    const q = out[out.length - 1];
    if (q && Math.abs(p[0] - q[0]) < 1e-12 && Math.abs(p[1] - q[1]) < 1e-12) continue;
    out.push(p);
  }
  return out;
}

// weldConsecutive with a parallel array carried along: whatever survives keeps
// its own companion entry, index for index. It keeps each 3D boundary
// point paired with the (u,v) it was evaluated at through the pole-collapse
// weld, so the pairing cannot silently shift by one when a point is dropped.
function weldConsecutivePaired(pts, companions, tolerance = 1e-9) {
  const out = [];
  const kept = [];
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const q = out[out.length - 1];
    if (q && Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]) < tolerance) continue;
    out.push(p);
    kept.push(companions[i]);
  }
  while (out.length > 1) {
    const a = out[0], b = out[out.length - 1];
    if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) >= tolerance) break;
    out.pop();
    kept.pop();
  }
  return { pts: out, uvs: kept };
}

function hasRepeatedPoint(pts, tolerance = 1e-9) {
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const a = pts[i], b = pts[j];
      if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < tolerance) return true;
    }
  }
  return false;
}

// Does this (u,v) loop enclose real 3D area, or has the surface collapsed
// it? Counted on distinct evaluated points rather than area, because the
// pole case is not a near-zero area to threshold against — it is a loop
// whose corners are literally the same point, so a distinct-point count
// says what is happening without needing a tolerance on area.
function collapsesIn3D(uvLoop, srf, tolerance = 1e-6) {
  const seen = new Set();
  for (const [u, v] of uvLoop) {
    const p = surfacePoint(srf, u, v);
    if (!p.every((c) => Number.isFinite(c))) return false;
    seen.add(p.map((c) => Math.round(c / tolerance)).join('|'));
    if (seen.size >= 3) return false;
  }
  return true;
}

// A trim loop may or may not repeat its first point at the end — both forms
// exist in this kernel. Normalizing to the unclosed form here is not what
// protects the welder: `weldPoints` already pops an explicitly-closed loop's
// repeated first point itself, and says so in its own comment. It earns its
// place for the two things upstream of the welder that do care — the
// three-corner minimum below is only meaningful counted on distinct points,
// and `mergeLoopsKeyhole` should be handed a clean ring rather than one
// carrying a duplicate its bridge search could pick.
function stripClosingPoint(loop) {
  if (loop.length > 1) {
    const a = loop[0], b = loop[loop.length - 1];
    if (Math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-12) return loop.slice(0, -1);
  }
  return loop.slice();
}

/**
 * Split every boundary edge at any vertex another fragment already placed in
 * its interior — the T-junction the sew has to resolve before welding.
 *
 * This is not the densification the header warns against. That trap is
 * inventing intermediate points at uniform fractions of one surface's own
 * (u,v), which land nowhere on the other's. This adds no new point anywhere:
 * it only inserts a point that already exists on the far side of the same
 * edge, at the position that fragment already uses, so the two sides come out
 * with the identical vertex list and weld exactly.
 *
 * A T-junction is not a defect upstream — it is ordinary. Two faces sharing a
 * cut curve are split independently, and each face's own arrangement paves
 * that curve at its crossings, so one side can legitimately carry a vertex
 * the other has no reason to. Every vertex still matches to machine precision
 * where both sides have one; only the subdivision differs, which is exactly
 * what leaves an edge with one incident face on each side of the split point
 * and reads downstream as a naked edge.
 *
 * The inserted point is the far side's own coordinate, not the projection of
 * it onto this edge — a projection would sit up to `tolerance` away from the
 * thing it is meant to weld to, which is the one place that distance is not
 * affordable.
 *
 * "On this edge" is a tighter question than "is this the same point", and
 * the two do not share a tolerance. The weld tolerance is a document-scale
 * number that has to absorb the marcher's own residual across an
 * intersection curve. Reused here it is far too generous: a vertex from an
 * unrelated face merely passing within that distance would be spliced into
 * this edge, bending a straight boundary toward a face it has no
 * relationship with and welding the two together into a vertex-level pinch
 * that the edge-based manifold checks downstream cannot see. So the off-edge
 * test uses its own, much tighter constant.
 *
 * The two failure directions are not symmetric, which is what decides the
 * value. Too loose admits a wrong vertex and produces a plausible solid that
 * is quietly non-manifold. Too tight misses a true T-junction, which leaves
 * a naked edge and comes back as an open-shell verdict naming itself. This
 * kernel prefers the second every time. True T-junctions on a marched
 * boolean sit ~1e-14 off the edge, five orders inside this floor; a false
 * positive at half the weld tolerance sits five orders outside it.
 *
 * A curved shared edge needs a second, structural route, and this is the one
 * case the tight floor above cannot serve. Both sides of an intersection
 * curve carry the same marched samples, so an ordinary T-junction between
 * them sits at machine precision. But a fragment wrapping a closed direction
 * is cut into pieces, and that cut lands where it lands — generally between
 * two consecutive samples, at a point that is on the true curve and
 * therefore off the neighbor's straight chord by that chord's own sagitta:
 * 1.7e-3mm on a 0.41mm chord of a radius-12 circle, for a box unioned with a
 * cylinder, six orders outside the floor. Loosening the floor to admit it
 * would also admit the false positive above, which is the bind.
 *
 * So the second route does not ask "is it close enough" at all — it asks for
 * corroboration: a point is spliced into edge (a,b) only if the already-welded
 * edge set contains a chain running a -> ... -> p -> ... -> b through the
 * candidates. A chain rather than a single (a,p)+(p,b) hop because a coarsely
 * sampled curve can take two cuts inside one gap; see corroboratedSubdivision.
 * That is the literal definition of the T-junction being repaired, and it is
 * evidence a merely-passing vertex from an unrelated face cannot manufacture:
 * to reach the chain at all it has to be an edge-neighbor of something
 * already on the path to both of this edge's endpoints. The proximity bound is
 * then only a sanity guard, and stays relative to the edge's own length (a
 * chord's sagitta scales as length/8R, so a fixed absolute number would be
 * wrong at both ends of the size range).
 *
 * The edge set is flat across every loop, and the chain may therefore be
 * assembled from hops contributed by different faces — no single face has to
 * subdivide this edge exactly that way. Several fragments commonly meet along
 * one shared curve, so the subdivision a given edge needs is spread across
 * them, and demanding one face carry the whole chain would miss the ordinary
 * case. It does mean the evidence is weaker than "some other face already did
 * exactly this" — it is "the loop set collectively already runs a path of
 * real edges from a to b through this point".
 *
 * The route is additive — the absolute floor is checked first and unchanged,
 * so every T-junction it catches is still caught the same way. Known
 * limitation: faces that between them supply a chain from a to b through a
 * point that is not actually on this edge — a sliver sharing both endpoints
 * is the single-face version, and the cross-face version needs only that each
 * hop exist somewhere. Both would have to fall inside the relative bound
 * above. Nothing in this kernel is known to produce either, and a
 * vertex-level pinch from one would still have to pass the manifold checks.
 */
const T_JUNCTION_OFF_EDGE_TOLERANCE = 1e-9;
// As a fraction of the edge's own length. A chord subtending a circle of
// radius R deviates by length/(8R) of its length, so this covers a shared
// curve discretized as coarsely as one sample per 0.4 radian and still stays
// far tighter than any face that is not essentially collinear with the edge.
const T_JUNCTION_CORROBORATED_OFF_EDGE_FRAC = 0.05;
// The third route measures against the edge's own curve, not its chord.
//
// Both routes above corroborate a T-vertex against the straight chord between
// an edge's two corners. That is the right question for an edge whose two
// sides sample the same polyline, and the wrong one for an edge whose real
// path is curved: this module's own header convention is that a boundary is
// "polygonal in the topology, exact in the geometry", so the true edge
// between two corners is the image of the straight (u,v) segment through the
// face's own surface, and a true T-vertex sits on that curve — off the chord
// by the chord's own full sagitta.
//
// The case that forces this is not exotic geometry but a primitive: ToNURBS
// on a SuperB emits adjacent patches at different isolation levels (a face is
// converted at the first level it becomes regular, its neighbor a level
// later), so a coarse patch's edge legitimately carries the fine neighbors'
// shared corner at its middle — a hanging node whose position is exactly on
// the coarse edge's curve (both are the same limit surface, agreeing to
// ~1e-15) and 1–11 mm off its chord on a rounded box a few hundred mm across.
// Widening the chord fraction to reach them would also reach the false
// positive above; measuring against the right curve reaches them at the weld
// tolerance, five orders tighter than the chord bound.
//
// The route is additive and doubly gated, weaker nowhere:
//   * a candidate must lie within `tolerance` — the weld tolerance itself,
//     the number that already defines "these two points are the same point"
//     downstream — of the edge's own curve, found by minimizing the distance
//     to surfacePoint along the edge's (u,v) segment; and
//   * it must still carry the same chain corroboration the second route
//     demands: the already-welded edge set must run a -> ... -> p -> ... -> b
//     through it. Being on the curve alone inserts nothing.
// The search band that feeds it is the curve's own measured deviation from
// the chord, sampled along the edge — derived from the edge's actual
// geometry, never a constant that could quietly grow. The band is a search
// prefilter, not an admission test: everything inside it still faces the
// exact on-curve measurement and the chain, so its only failure direction is
// missing a true T-vertex. The sampled maximum under-reads the true one
// (0.4% low on a rounded-box blend edge, enough to lose a vertex sitting
// exactly at the peak), so the band carries a deviation-proportional margin
// far beyond any plausible sampling error, plus weld-tolerance slack for the
// near-flat case where the deviation itself is negligible.
const T_JUNCTION_CURVE_SAMPLES = 8;
const T_JUNCTION_CURVE_BAND_MARGIN = 1.25;
function uvSegmentPoint(srf, uvA, uvB, s) {
  return surfacePoint(srf, uvA[0] + (uvB[0] - uvA[0]) * s, uvA[1] + (uvB[1] - uvA[1]) * s);
}
// Largest measured deviation of the edge's own curve from its chord. Sampled,
// so it can only under-report — which fails toward a missed insertion and a
// naked edge, never toward admitting a point the exact test below would not
// have to pass anyway.
function chordDeviationOfEdge(srf, uvA, uvB, a, ab, len2) {
  let dev = 0;
  for (let k = 1; k < T_JUNCTION_CURVE_SAMPLES; k++) {
    const q = uvSegmentPoint(srf, uvA, uvB, k / T_JUNCTION_CURVE_SAMPLES);
    const t = Math.min(1, Math.max(0, dot(sub(q, a), ab) / len2));
    const d = length(sub(q, add(a, scale(ab, t))));
    if (d > dev) dev = d;
  }
  return dev;
}
// Distance from `p` to the curve the edge actually follows, with the curve
// parameter of the closest point. Coarse scan, then ternary refinement — the
// scan pins the right basin, the refinement takes it to well under any
// tolerance this is compared against.
function distanceToEdgeCurve(srf, uvA, uvB, p) {
  const f = (s) => length(sub(uvSegmentPoint(srf, uvA, uvB, s), p));
  let bestS = 0, bestD = Infinity;
  for (let k = 0; k <= 16; k++) {
    const s = k / 16;
    const d = f(s);
    if (d < bestD) { bestD = d; bestS = s; }
  }
  let lo = Math.max(0, bestS - 1 / 16), hi = Math.min(1, bestS + 1 / 16);
  for (let it = 0; it < 32; it++) {
    const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3;
    if (f(m1) <= f(m2)) hi = m2; else lo = m1;
  }
  const s = (lo + hi) / 2;
  return { d: f(s), s };
}
export function insertTJunctionVertices(loops, tolerance, loopGeom = null) {
  const verts = [];
  for (const loop of loops) for (const p of loop) verts.push(p);

  // A spatial hash, because the naive scan is edges x vertices — quadratic —
  // and this is the one pass whose input grows with both solids at once: a
  // marched boolean on organic geometry pushes the count up, since every
  // intersection curve contributes its own samples to a fragment boundary.
  //
  // The cell is sized to the mean edge length rather than the tolerance: a
  // tolerance-sized cell (1e-9 here) would make a single ordinary edge cross
  // millions of them. At this size an edge touches a couple of cells and a
  // cell holds a handful of points, which is what makes the walk bounded.
  const grid = buildPointGrid(verts, meanEdgeLength(loops));

  // The corroboration index. Every boundary point collapsed to one welded id
  // and every edge any loop already carries, keyed on those ids — the same
  // weld the solid builder itself is about to run, reused rather than
  // approximated, so "this edge already exists" means exactly what it will
  // mean downstream.
  const welded = weldPoints(loops, tolerance);
  const weldedIdOfPoint = new Map();
  welded.points.forEach((p, i) => weldedIdOfPoint.set(p, i));
  const weldGrid = buildPointGrid(welded.points, Math.max(tolerance, 1e-12) * 2);
  const idOf = (p) => {
    for (const q of gridPointsNearSegment(weldGrid, p, p, 0, tolerance)) {
      if (Math.abs(q[0] - p[0]) <= tolerance && Math.abs(q[1] - p[1]) <= tolerance && Math.abs(q[2] - p[2]) <= tolerance) {
        return weldedIdOfPoint.get(q);
      }
    }
    return -1;
  };
  const edgeSet = new Set();
  for (const l of welded.loops) {
    for (let i = 0; i < l.length; i++) {
      const x = l[i], y = l[(i + 1) % l.length];
      if (x !== y) edgeSet.add(edgeKey(x, y));
    }
  }

  return loops.map((loop, loopIndex) => {
    const own = new Set(welded.loops[loopIndex] ?? []);
    const geom = loopGeom ? loopGeom[loopIndex] : null;
    const out = [];
    for (let i = 0; i < loop.length; i++) {
      const a = loop[i], b = loop[(i + 1) % loop.length];
      out.push(a);
      const ab = sub(b, a);
      const len2 = dot(ab, ab);
      if (!(len2 > 0)) continue;
      const len = Math.sqrt(len2);
      // An edge at most the tolerance long has no interior to speak of.
      if (len <= tolerance * 2) continue;

      // The curve route's own search band: how far the edge's real path
      // strays from the chord, measured on this edge, plus weld-tolerance
      // slack. Zero (route off) without the (u,v) correspondence.
      const uvA = geom && geom.uv ? geom.uv[i] : null;
      const uvB = geom && geom.uv ? geom.uv[(i + 1) % loop.length] : null;
      const curveBand = uvA && uvB
        ? chordDeviationOfEdge(geom.srf, uvA, uvB, a, ab, len2) * T_JUNCTION_CURVE_BAND_MARGIN + tolerance * 4
        : 0;

      const hits = [];
      const candidates = [];
      // The widest tolerance any candidate below is tested against, so the
      // walk covers everything the tests could accept.
      const searchTol = Math.max(T_JUNCTION_OFF_EDGE_TOLERANCE, len * T_JUNCTION_CORROBORATED_OFF_EDGE_FRAC, curveBand);
      for (const p of gridPointsNearSegment(grid, a, b, len, searchTol)) {
        const t = dot(sub(p, a), ab) / len2;
        // Strictly interior, measured in real distance along the edge rather
        // than in the parameter, so a long edge and a short one are held to
        // the same standard.
        if (!(t * len > tolerance) || !((1 - t) * len > tolerance)) continue;
        const off = length(sub(p, add(a, scale(ab, t))));
        if (off <= T_JUNCTION_OFF_EDGE_TOLERANCE) { hits.push({ t, p }); continue; }
        // Too far off the chord to be accepted on proximity alone, but close
        // enough to be worth asking whether another face already subdivides
        // this edge through it.
        if (off <= len * T_JUNCTION_CORROBORATED_OFF_EDGE_FRAC) { candidates.push({ t, p }); continue; }
        // Beyond the chord bound but inside the curve's own measured band:
        // admitted as a candidate only if it sits on the edge's actual curve
        // to the weld tolerance. Still chain-gated below like every other
        // candidate — the exact test earns it consideration, not insertion.
        if (off <= curveBand) {
          const on = distanceToEdgeCurve(geom.srf, uvA, uvB, p);
          if (on.d <= tolerance) candidates.push({ t, p });
        }
      }
      if (candidates.length) {
        for (const h of corroboratedSubdivision(a, b, candidates, idOf, edgeSet, own)) hits.push(h);
      }
      if (!hits.length) continue;

      hits.sort((x, y) => x.t - y.t);
      // Several fragments can each carry their own copy of the same
      // T-junction; inserting it once is what the welder needs. The test for
      // "the same one again" is the 3D distance between the two points,
      // because that is literally the question being asked — will the welder
      // treat these as one vertex — where a gap measured along the edge only
      // approximates it, and can drop a hit the welder would have kept apart.
      // Euclidean here against the welder's own per-axis comparison is the
      // stricter of the two, so anything skipped is a point the welder would
      // certainly have merged.
      let lastP = a;
      for (const h of hits) {
        if (length(sub(h.p, lastP)) <= tolerance) continue;
        out.push(h.p);
        lastP = h.p;
      }
    }
    return out;
  });
}

function edgeKey(i, j) {
  return i < j ? `${i}|${j}` : `${j}|${i}`;
}

// The chain of already-existing edges that runs from `a` to `b` through the
// candidates, or nothing.
//
// A single corroborated point is the ordinary case — one cut landing between
// two samples of a shared curve — but a coarsely sampled curve can take two
// cuts inside one gap, and then the neighbor subdivides the edge as
// a -> p1 -> p2 -> b with no (a,p2) or (p1,b) edge anywhere. So this asks for
// a path rather than a single hop, walking the candidates in order along the
// edge: the answer either way is the exact vertex list the other side already
// uses, which is what makes both sides weld.
//
// The longest such chain wins. A shorter one would leave the leftover
// subdivision points still unmatched, which is the same naked edge one level
// smaller.
function corroboratedSubdivision(a, b, candidates, idOf, edgeSet, ownIds) {
  const ia = idOf(a), ib = idOf(b);
  if (ia < 0 || ib < 0) return [];
  const pts = candidates
    .slice()
    .sort((x, y) => x.t - y.t)
    .map((c) => ({ t: c.t, p: c.p, id: idOf(c.p) }))
    // A point this loop already carries must not be spliced in a second time:
    // the duplicate would be a vertex-level pinch, and the only shape that
    // reaches here is a sliver whose own third corner is nearly collinear
    // with an edge it already owns.
    .filter((c) => c.id >= 0 && c.id !== ia && c.id !== ib && !ownIds.has(c.id));
  if (!pts.length) return [];

  const prev = new Array(pts.length).fill(-2); // -2 unreachable, -1 straight from `a`
  for (let i = 0; i < pts.length; i++) {
    if (edgeSet.has(edgeKey(ia, pts[i].id))) { prev[i] = -1; continue; }
    for (let j = i - 1; j >= 0; j--) {
      if (prev[j] !== -2 && edgeSet.has(edgeKey(pts[j].id, pts[i].id))) { prev[i] = j; break; }
    }
  }
  for (let i = pts.length - 1; i >= 0; i--) {
    if (prev[i] === -2 || !edgeSet.has(edgeKey(pts[i].id, ib))) continue;
    const chain = [];
    for (let k = i; k >= 0; k = prev[k]) {
      chain.push(pts[k]);
      if (prev[k] === -1) break;
    }
    return chain.reverse();
  }
  return [];
}

// The average boundary edge, which sizes the T-junction grid. A cell near
// this length is the balance point: much smaller and one edge walks a huge
// number of cells, much larger and each cell holds most of the points and
// the scan is quadratic again by another name.
function meanEdgeLength(loops) {
  let total = 0;
  let count = 0;
  for (const loop of loops) {
    for (let i = 0; i < loop.length; i++) {
      const a = loop[i];
      const b = loop[(i + 1) % loop.length];
      total += length(sub(b, a));
      count++;
    }
  }
  return count ? total / count : 1;
}

function gridKey(p, size) {
  return `${Math.floor(p[0] / size)}|${Math.floor(p[1] / size)}|${Math.floor(p[2] / size)}`;
}

function buildPointGrid(points, cell) {
  const size = Math.max(cell, 1e-12);
  const map = new Map();
  for (const p of points) {
    const key = gridKey(p, size);
    let bucket = map.get(key);
    if (!bucket) map.set(key, (bucket = []));
    bucket.push(p);
  }
  return { size, map };
}

// Every point that could possibly lie within `searchTol` of this segment,
// found by walking the cells the segment passes through.
//
// Exhaustive only if the neighborhood is sized to the tolerance, which is
// why the caller has to say what its tolerance is. The step is half a cell,
// so any point of the segment is within a quarter cell of some sample; a
// candidate lying up to `searchTol` off the segment is therefore within
// `searchTol + size/4` of that sample, and a displacement of d can move a
// floor() index by at most ceil(d / size) per axis. So the radius below is
// that ceiling, never a fixed 1.
//
// A fixed 27-cell probe is only the special case searchTol +
// size/4 <= size, and it is exceeded here: the corroborated off-edge
// tolerance is a fraction of the edge's own length while the cell is the
// mean edge length, so an edge an order of magnitude longer than the mean
// reaches past its own sample's neighbors. Missing a candidate there only
// costs a T-junction insertion — which degrades into a naked edge and an
// open-shell refusal rather than a wrong solid — but the claim of
// exhaustiveness would be false, and the radius is cheap to get right.
function gridPointsNearSegment(grid, a, b, len, searchTol = 0) {
  const { size, map } = grid;
  const steps = Math.max(1, Math.ceil(len / (size * 0.5)));
  // A zero-length segment has one sample sitting exactly on the query point,
  // so it carries no quarter-cell sampling error to allow for.
  const reach = searchTol + (len > 0 ? size * 0.25 : 0);
  const r = Math.max(1, Math.ceil(reach / size));
  const seen = new Set();
  const out = [];
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    const cx = Math.floor((a[0] + (b[0] - a[0]) * t) / size);
    const cy = Math.floor((a[1] + (b[1] - a[1]) * t) / size);
    const cz = Math.floor((a[2] + (b[2] - a[2]) * t) / size);
    for (let dx = -r; dx <= r; dx++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dz = -r; dz <= r; dz++) {
          const key = `${cx + dx}|${cy + dy}|${cz + dz}`;
          if (seen.has(key)) continue;
          seen.add(key);
          const bucket = map.get(key);
          if (bucket) for (const p of bucket) out.push(p);
        }
      }
    }
  }
  return out;
}

/**
 * Sew kept fragments into one solid.
 *
 * `pieces` — [{ srf, outer, holes? }, ...], each a Phase 6 fragment paired
 *   with the surface its (u,v) belongs to.
 *
 * Returns { ok, solid, stats, verdict, reason?, worstSharedGap, tolerance }.
 * `worstSharedGap` is the largest distance between two boundary points that
 * welded together — the measured evidence for whether the tolerance was
 * doing real work or barely reaching, which is the number a caller needs
 * when a sew comes back open.
 */
export function sewFragments(pieces, opts = {}) {
  if (!pieces || !pieces.length) {
    return { ok: false, reason: 'no kept fragments to sew, so there is no solid to build', verdict: 'NO FACES' };
  }
  const tolerance = opts.tolerance ?? 1e-4;
  const built = sewBoundaries3D(pieces, { tolerance });
  if (!built.ok) return built;
  const loops = built.loops;

  const sewLoops = insertTJunctionVertices(loops, tolerance, built.loopGeom);
  const res = buildBrepSolid(sewLoops, { tolerance, name: opts.name ?? null });
  const worstSharedGap = measureWorstWeldGap(sewLoops, tolerance);

  if (!res.ok) {
    return {
      ok: false,
      reason: res.reason,
      verdict: brepVerdict(res),
      stats: res.stats ?? null,
      // Which edges did not pair, not just how many. The builder measures them,
      // and "12 naked edges" is the same sentence whether one face is missing or
      // two rims are sampled differently, which are opposite repairs.
      nakedEdgePoints: res.nakedEdgePoints ?? null,
      nonManifoldEdgePoints: Array.isArray(res.nonManifold) && Array.isArray(res.points)
        ? res.nonManifold.map((e) => [res.points[e.a], res.points[e.b], e.uses])
        : null,
      nonManifoldEdgeUses: Array.isArray(res.nonManifold)
        ? res.nonManifold.map((e) => (e.faces || []).map((face) => {
          const pieceIndex = built.loopOwner?.[face];
          const piece = Number.isInteger(pieceIndex) ? pieces[pieceIndex] : null;
          return { face, piece: pieceIndex, label: piece?.booleanFaceLabel ?? null };
        }))
        : null,
      worstSharedGap,
      tolerance,
    };
  }
  // Attach each face's own surface. `buildBrepSolid` works in 3-D loops alone,
  // so kernel/brep.mjs's declared `face.surface` slot comes back null from it
  // — which leaves a sewn solid unable to say what any of its faces is, and
  // blocks every downstream question asked in a face's own parameters (a
  // pcurve, a trim, a .3dm face).
  //
  // `sewBoundaries3D` returns `loopOwner`, naming the piece each loop came
  // from, and a piece carries its srf. Loops enter buildBrepSolid in that
  // order and come back as faces in the same order, so the attribution is
  // positional — asserted by count rather than assumed, and skipped (leaving
  // the null) if the counts disagree, because a face wearing the wrong
  // surface is worse than a face wearing none.
  const faces = res.solid.shells.flatMap((sh) => sh.faces || []);
  let surfacesAttached = 0;
  if (built.loopOwner && faces.length === built.loopOwner.length) {
    faces.forEach((f, i) => {
      const piece = pieces[built.loopOwner[i]];
      if (piece && piece.srf) { f.surface = piece.srf; surfacesAttached++; }
    });
  }
  return {
    ok: true,
    solid: res.solid,
    stats: res.stats,
    verdict: brepVerdict(res),
    worstSharedGap,
    tolerance,
    surfacesAttached,
    loopOwner: built.loopOwner ?? null,
  };
}

/**
 * Every kept fragment's boundary as 3D face loops, harmonized across the whole
 * surface — everything `sewFragments` does before the weld.
 *
 * Returns { ok, loops, loopOwner } where `loopOwner[k]` is the index in
 * `pieces` the k-th loop came from, or a refusal in `sewFragments`' own shape.
 * Separate from `sewFragments` because the provenance is what makes a naked
 * edge attributable to a fragment and a surface, and so there is one copy of
 * this pooling.
 */
export function sewBoundaries3D(pieces, opts = {}) {
  // A seam belongs to the surface, not to one fragment. The two copies of a
  // closed direction's domain edge can end up in different fragments — a small
  // lens crossing the seam on one side, the large remainder on the other — and
  // each fragment harmonized alone gives its own copies matching points while
  // leaving the two fragments to disagree. So the (u,v) loops of every fragment
  // sharing a surface are pooled, harmonized together, and only then evaluated.
  const prepared = [];
  for (let i = 0; i < pieces.length; i++) {
    const prep = fragmentUVLoops(pieces[i].srf, pieces[i]);
    if (!prep.ok) {
      return {
        ok: false,
        reason: `fragment ${i} has no usable boundary — ${prep.reason}`,
        verdict: 'DEGENERATE',
      };
    }
    prepared.push(prep);
  }

  const bySurface = new Map();
  pieces.forEach((p, i) => {
    if (!bySurface.has(p.srf)) bySurface.set(p.srf, []);
    bySurface.get(p.srf).push(i);
  });
  for (const [srf, group] of bySurface) {
    const flat = [], from = [], cuts = [];
    for (const i of group) {
      for (const loop of prepared[i].uvLoops) { flat.push(loop); from.push(i); }
      for (const c of prepared[i].cutLines) {
        if (!cuts.some((m) => m.ai === c.ai && Math.abs(m.value - c.value) <= 1e-12)) cuts.push(c);
      }
    }
    let harmonized = harmonizeDomainSeams(flat, srf);
    harmonized = harmonizeCutLines(harmonized, cuts, srf);
    for (const i of group) prepared[i].uvLoops = [];
    harmonized.forEach((loop, k) => prepared[from[k]].uvLoops.push(loop));
  }

  const loops = [];
  const loopOwner = [];
  const loopGeom = [];
  for (let i = 0; i < pieces.length; i++) {
    const evaluated = evaluateUVLoops(pieces[i].srf, prepared[i].uvLoops, opts.tolerance ?? 0);
    if (!evaluated.ok) {
      return {
        ok: false,
        reason: `fragment ${i} has no usable boundary — ${evaluated.reason}`,
        verdict: 'DEGENERATE',
      };
    }
    evaluated.loops.forEach((loop, k) => {
      loops.push(loop);
      loopOwner.push(i);
      loopGeom.push({ srf: pieces[i].srf, uv: evaluated.uvs[k] });
    });
  }
  return { ok: true, loops, loopOwner, loopGeom };
}

// The largest gap between two points that the welder treated as one. This is
// measured independently rather than read out of the welder: what a caller
// wants to know when a sew comes back open is whether the surviving gaps
// were just past the tolerance (a residual problem, fixable by a finer
// march) or nowhere near it (a missing fragment), and only a measurement
// distinguishes those.
//
// Also spatially hashed, and for the same reason: it runs over the same
// enlarged loop set the T-junction pass just produced, so leaving it as an
// all-pairs scan would put the quadratic cost straight back one function
// later. Here the cell is twice the tolerance, which is exactly the sizing
// `weldPoints` itself uses — a pair closer than the tolerance can differ by
// at most one cell index per axis, so the 27-cell probe is exhaustive.
function measureWorstWeldGap(loops, tolerance) {
  const all = [];
  for (const l of loops) for (const p of l) all.push(p);
  const size = Math.max(tolerance, 1e-12) * 2;
  const map = new Map();
  for (let i = 0; i < all.length; i++) {
    const key = gridKey(all[i], size);
    let bucket = map.get(key);
    if (!bucket) map.set(key, (bucket = []));
    bucket.push(i);
  }
  let worst = 0;
  for (let i = 0; i < all.length; i++) {
    const p = all[i];
    const cx = Math.floor(p[0] / size);
    const cy = Math.floor(p[1] / size);
    const cz = Math.floor(p[2] / size);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const bucket = map.get(`${cx + dx}|${cy + dy}|${cz + dz}`);
          if (!bucket) continue;
          for (const j of bucket) {
            // Each unordered pair once.
            if (j >= i) continue;
            const d = length(sub(p, all[j]));
            if (d > 0 && d <= tolerance && d > worst) worst = d;
          }
        }
      }
    }
  }
  return worst;
}
