// TRIMMED SURFACE DATA MODEL — the trimmed-surface spec,
// built for the first time: "A TRIMMED SURFACE is a
// NurbsSrf (unchanged) plus a set of TRIM LOOPS — each loop a closed 2D
// curve living in that surface's own (u,v) parameter space... Trim curves
// are POLYLINES (degree-1 UV polylines), not interpolated smooth NURBS
// curves." Everything here operates on that exact shape: a "loop" is a
// plain array of [u,v] pairs, implicitly closed (the segment from the last
// point back to the first is part of the loop).
//
// Also builds the OTHER real prerequisite for Tier A/B:
// projecting an ordinary 3D curve onto a surface's own parameter space (P&T
// 6.1 point inversion walked along a curve, warm-started sample to sample
// so a near-seam/near-pole curve doesn't jump to the wrong (u,v) preimage
// mid-walk) — the real, materially SIMPLER sub-case named in the
// own task brief: "split a surface by a CURVE... may be a materially
// simpler, more tractable sub-case worth building FIRST."

import { surfacePointAndPartials, closestPointOnSurface, surfaceClosure, wrapParam, escapeDegeneratePoint, nakedEdgeCount } from './surface.mjs';
import { tessellateCurve } from './project.mjs';
import { sub, dot, length, add, scale } from './vec3.mjs';

// A single warm-started Gauss-Newton refinement step against ONE surface —
// the exact 2-unknown recipe closestPointOnSurface's own stage 2 already
// uses (minimize |S(u,v)-P|^2 via the real first partials), factored out
// here so a caller with an ALREADY-GOOD seed (the previous sample along a
// curve being walked) can skip closestPointOnSurface's own coarse grid
// search entirely — re-running that coarse search at every single sample
// along a dense curve would be wasteful AND could jump to a genuinely
// different (u,v) preimage between adjacent samples near a seam/pole,
// exactly the failure mode this function is built to avoid. Wraps at a
// closed direction's own seam and backtracks a worsening step instead of
// rejecting it outright — see closestPointOnSurface's own header comment
// in surface.mjs for the full reasoning (both functions must apply this
// identically, since a curve walked sample-to-sample only ever gets one
// real answer from whichever of the two the caller happens to invoke).
export function refineClosestPointOnSurface(srf, targetPt, u0, v0) {
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const { closedU, closedV } = surfaceClosure(srf);
  const distSq = (p) => { const d = sub(p, targetPt); return dot(d, d); };
  let u = wrapParam(u0, uMin, uMax, closedU), v = wrapParam(v0, vMin, vMax, closedV);
  let curDistSq = distSq(surfacePointAndPartials(srf, u, v).point);
  for (let iter = 0; iter < 20; iter++) {
    const { point, su, sv } = surfacePointAndPartials(srf, u, v);
    const r = sub(point, targetPt);
    const Juu = dot(su, su), Juv = dot(su, sv), Jvv = dot(sv, sv);
    const bu = dot(su, r), bv = dot(sv, r);
    const det = Juu * Jvv - Juv * Juv;
    if (Math.abs(det) < 1e-14) {
      // A pole is rank-1, and leaving it means choosing the collapsed
      // parameter as well as stepping along the live one — see
      // escapeDegeneratePoint's own comment in surface.mjs. Both loops apply
      // this identically, as their headers require.
      const esc = escapeDegeneratePoint(srf, targetPt, u, v, curDistSq, Juu, Jvv, closedU, closedV);
      if (!esc) break;
      u = esc.u; v = esc.v; curDistSq = esc.distSq;
      continue;
    }
    const du = -(Jvv * bu - Juv * bv) / det;
    const dv = -(Juu * bv - Juv * bu) / det;
    let accepted = false, didConverge = false;
    for (let bt = 0, step = 1; bt < 5; bt++, step *= 0.5) {
      const uNext = wrapParam(u + du * step, uMin, uMax, closedU);
      const vNext = wrapParam(v + dv * step, vMin, vMax, closedV);
      const nextPt = surfacePointAndPartials(srf, uNext, vNext).point;
      const nextDistSq = distSq(nextPt);
      if (nextDistSq <= curDistSq + 1e-12) {
        didConverge = Math.abs(uNext - u) < 1e-10 && Math.abs(vNext - v) < 1e-10;
        u = uNext; v = vNext; curDistSq = nextDistSq;
        accepted = true;
        break;
      }
    }
    if (!accepted || didConverge) break;
  }
  return { u, v, distance: Math.sqrt(curDistSq) };
}

// Classic point-in-triangle closest-point test (Ericson, Real-Time
// Collision Detection §5.1.5) — the primitive behind projectCurveToMesh
// below, for any target with no explicit NURBS `srf` at all (a SubD/SuperB
// limit surface, or any panel/mesh-based container — RuledLoft/
// PolySurface/MultiPipe/Split). There is no analytic (u,v) domain to run
// Gauss-Newton against there, so this is a brute-force nearest point over
// the target's own already-tessellated triangle soup instead — no
// iteration at all, so unlike the NURBS path above it structurally cannot
// diverge, get stuck at a seam, or need a tolerance/backtrack tradeoff.
function closestPointOnTriangle(a, b, c, p) {
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
  const v = vb * denom, w = vc * denom;
  return add(a, add(scale(ab, v), scale(ac, w)));
}

// The nearest point on a whole triangle SOUP (a flat array of [a,b,c]
// vertex triples, world space) to an arbitrary 3D point — a plain linear
// scan, tracking the global closest across every triangle. O(triangle
// count) per query; fine at this app's own real mesh densities for a
// one-shot command (no BVH — a curve-projection command runs once, not
// per-frame).
export function closestPointOnTriangleMesh(triangles, targetPt) {
  let best = null, bestDistSq = Infinity;
  for (const [a, b, c] of triangles) {
    const q = closestPointOnTriangle(a, b, c, targetPt);
    const d = sub(q, targetPt);
    const dq = dot(d, d);
    if (dq < bestDistSq) { bestDistSq = dq; best = q; }
  }
  if (!best) return null;
  return { point: best, distance: Math.sqrt(bestDistSq) };
}

// The mesh-target sibling of projectCurveToSurfaceUV below — same
// "sample the curve, pull each sample onto the target, refuse honestly
// past tolerance" contract, but returns real 3D POINTS directly rather
// than a (u,v) pair, since a triangle soup has no continuous parameter
// domain to report one against.
export function projectCurveToMesh(crv, triangles, opts = {}) {
  const samples = opts.samples ?? 64;
  const tolerance = opts.tolerance ?? 1e-3;
  if (!triangles || !triangles.length) {
    return { ok: false, reason: 'the target has no real surface geometry to project onto' };
  }
  const pts = tessellateCurve(crv, samples);
  const points = [];
  for (let i = 0; i < pts.length; i++) {
    const hit = closestPointOnTriangleMesh(triangles, pts[i]);
    if (hit.distance > tolerance) {
      return { ok: false, reason: `curve is not on the surface — sample ${i} is ${hit.distance.toFixed(6)} away, above tolerance ${tolerance}` };
    }
    points.push(hit.point);
  }
  return { ok: true, points };
}

// Project a real 3D curve believed to lie ON (or very near) a surface into
// that surface's own UV parameter space, as a dense polyline — exactly the
// "trim curves are polylines" data model. Honestly refuses
// (rather than silently returning a UV curve wandering off the real
// surface) if any sample's true 3D distance from the surface exceeds
// `tolerance` — a real, checkable validity gate, not an assumption that the
// caller already got this right.
export function projectCurveToSurfaceUV(crv, srf, opts = {}) {
  const samples = opts.samples ?? 64;
  return projectPointsToSurfaceUV(tessellateCurve(crv, samples), srf, opts);
}

// The same walk, entered from an already-sampled point chain rather than a
// NurbsCrv. An intersection curve arrives from the marcher as samples in the
// first place, so making it synthesize a curve just to be re-tessellated back
// into points would lose accuracy for no reason. `projectCurveToSurfaceUV` is
// now this function plus one tessellation step, so both paths share one
// definition of "warm-started, refuses when it leaves the surface".
export function projectPointsToSurfaceUV(pts, srf, opts = {}) {
  const tolerance = opts.tolerance ?? 1e-3;
  if (!pts || pts.length < 2) {
    return { ok: false, reason: 'need at least two points to project onto a surface' };
  }
  const uv = [];
  let seed = closestPointOnSurface(srf, pts[0]);
  if (seed.distance > tolerance) {
    return { ok: false, reason: `curve is not on the surface — sample 0 is ${seed.distance.toFixed(6)} away, above tolerance ${tolerance}` };
  }
  uv.push([seed.u, seed.v]);
  let prevU = seed.u, prevV = seed.v;
  for (let i = 1; i < pts.length; i++) {
    let refined = refineClosestPointOnSurface(srf, pts[i], prevU, prevV);
    // ⚠⚠ A WARM START CAN BE A TRAP, AND A DEGENERATE ONE ALWAYS IS. Refining
    // from the previous sample is what makes a long chain cheap and keeps it on
    // one branch of a closed surface — but if the previous sample landed on a
    // POLE, the surface's partials collapse there, Gauss-Newton has no
    // direction to move in, and every later sample stays pinned at the pole.
    //
    // Measured on a cylinder cap (a revolved disc, whose u=0 edge IS its
    // center, collapsed to a point): rim -> pole -> rim refuses with the sample
    // after the pole "12.000000 away" — exactly the disc's radius, the distance
    // from the center to the rim it should have found. Every one of those points
    // projects EXACTLY when seeded cold, so nothing is off the surface and the
    // refusal was about the seed, not the geometry.
    //
    // So a miss re-seeds from scratch before it is allowed to become a refusal.
    // The cost falls only on the failing sample, the fast path is untouched, and
    // a point that is genuinely off the surface still refuses — `closestPointOnSurface`
    // is a global search, so if it cannot get there, nothing about this chain can.
    if (refined.distance > tolerance) {
      const cold = closestPointOnSurface(srf, pts[i]);
      if (cold.distance <= tolerance) refined = cold;
    }
    if (refined.distance > tolerance) {
      return { ok: false, reason: `curve is not on the surface — sample ${i} is ${refined.distance.toFixed(6)} away, above tolerance ${tolerance} (re-seeded from scratch and it still missed, so this is the geometry and not the search)` };
    }
    uv.push([refined.u, refined.v]);
    prevU = refined.u; prevV = refined.v;
  }
  return { ok: true, uv };
}

// WHERE A CHAIN MEETS THIS SURFACE'S SEAM, as the crossing's own segment index
// and the UV point on the domain edge — the raw material every routine above
// consumes, exposed on its own for the one caller that needs the LOCATION
// rather than the re-expressed chain.
//
// That caller is the boolean's seam-point sharing pass. A seam crossing is
// interpolated from the two samples straddling it IN THIS SURFACE'S OWN
// PARAMETERS, so two faces cut by the same 3D curve, whose seams happen to
// pass through the same place, each derive their own answer and land apart by
// roughly one sample spacing — far outside any weld tolerance. Sharing the
// point requires knowing it before any chain has been built, which is what
// this returns and what neither seamCrossingSpine nor seamStraddleChains can
// hand back.
//
// `before` is deliberately the copy reported: it is the crossing expressed on
// the edge sample `seg` sits against, and both copies evaluate to the same 3D
// point anyway, since aMin and aMax are the same place on a closed surface.
//
// An empty list is the honest answer for every non-crossing case, including
// the topologies seamJumps refuses by name — a caller asking only "where does
// this touch the seam" has nothing to do about a double wrap that it would not
// also do about a chain that never approaches the seam at all.
//
// ⚠ ORDERING CONSTRAINT — IT SITS HERE, AWAY FROM THE SEAM FAMILY IT BELONGS
// TO, AND MUST STAY HERE. Everything boolean.mjs calls out of this file has to
// be DEFINED INSIDE THE SPAN that runs from `projectPointsToSurfaceUV` down to
// the shoelace comment below: embedders that inline the boolean path take that
// span and nothing else out of this file. A definition placed outside it is
// still CALLED from inside it, so the embedded copy references a function it
// never defines — a runtime ReferenceError in that build, invisible to a parse
// or a byte-comparison of this file. Moving it back down to its siblings
// breaks such a build silently. Its siblings (seamCrossingSpine and the rest)
// resolve there only because they are separately present at that build's top level.
export function seamCrossingUVPoints(chainUV, srf, cyclic = true) {
  const j = seamJumps(chainUV, srf, cyclic);
  if (!j.ok) return [];
  const { ai, oi, crossings, aMin, aMax } = j;
  return crossings.map((k) => ({ seg: k, axisIndex: ai, uv: seamPointAt(chainUV, k, ai, oi, aMin, aMax).before }));
}

// Shoelace signed area — the standard winding-direction test the
// "outer loops one winding direction, hole loops the other" rule needs.
// Positive = counter-clockwise in (u,v).
export function signedArea2D(loop) {
  let a = 0;
  const n = loop.length;
  for (let i = 0; i < n; i++) {
    const [u0, v0] = loop[i], [u1, v1] = loop[(i + 1) % n];
    a += u0 * v1 - u1 * v0;
  }
  return a / 2;
}

// Standard ray-casting point-in-polygon test (even-odd rule) against a UV
// trim loop, implicitly closed. A point exactly ON the boundary is reported
// separately (within `tol`) rather than forced arbitrarily inside/outside —
// callers doing tessellation classification (the "fully-inside /
// fully-outside / boundary-crossing" cell test) need that third state
// honestly, not a coin-flip.
export function pointInUVPolygon(loop, u, v, tol = 1e-9) {
  const n = loop.length;
  for (let i = 0; i < n; i++) {
    const [u0, v0] = loop[i], [u1, v1] = loop[(i + 1) % n];
    const du = u1 - u0, dv = v1 - v0;
    const segLenSq = du * du + dv * dv;
    if (segLenSq < 1e-18) continue;
    const t = ((u - u0) * du + (v - v0) * dv) / segLenSq;
    if (t >= -1e-12 && t <= 1 + 1e-12) {
      const px = u0 + du * t, py = v0 + dv * t;
      if (Math.hypot(px - u, py - v) < tol) return 'boundary';
    }
  }
  let inside = false;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const [ui, vi] = loop[i], [uj, vj] = loop[j];
    const crosses = (vi > v) !== (vj > v);
    if (crosses) {
      const uAtV = ui + (v - vi) * (uj - ui) / (vj - vi);
      if (u < uAtV) inside = !inside;
    }
  }
  return inside ? 'inside' : 'outside';
}

// Exported so trimtess.mjs's own keyhole-bridge merge (for
// trim HOLES) can reuse this exact strict-crossing test rather than a
// second copy — its own strict `>0`/`<0` comparisons already correctly
// treat a shared-endpoint touch (the bridge's own two ends) as NOT a
// crossing, which is exactly what a valid bridge-adjacency check needs.
export function segmentsIntersect(p0, p1, p2, p3) {
  // Standard 2D segment-segment intersection via cross-product orientation
  // tests (Cormen et al.), returns true for a genuine crossing OR overlap —
  // shared-endpoint touches (adjacent polyline segments) are excluded by
  // the caller skipping adjacent pairs, not handled specially here.
  const cross = (ax, ay, bx, by) => ax * by - ay * bx;
  const d1 = cross(p3[0] - p2[0], p3[1] - p2[1], p0[0] - p2[0], p0[1] - p2[1]);
  const d2 = cross(p3[0] - p2[0], p3[1] - p2[1], p1[0] - p2[0], p1[1] - p2[1]);
  const d3 = cross(p1[0] - p0[0], p1[1] - p0[1], p2[0] - p0[0], p2[1] - p0[1]);
  const d4 = cross(p1[0] - p0[0], p1[1] - p0[1], p3[0] - p0[0], p3[1] - p0[1]);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
  return false; // collinear/touching edge cases are a real, separate robustness concern — not silently claimed handled here
}

// Whether a UV trim loop (implicitly closed) self-intersects — the real,
// bounded, elementary 2D geometry primitive the "TWO DESIGN
// DECISIONS" ruling calls for ("Trim curves are POLYLINES... a smooth
// global-interpolation fit... can overshoot... a wiggle in a trim boundary
// is a geometry error"), deliberately NOT the harder general NURBS
// curve-curve intersection, still unbuilt —
// plain polyline segment intersection is elementary and does not need that
// machinery. O(n^2) pairwise segment test, skipping segments that already
// share an endpoint (adjacent polyline segments always "touch" there by
// construction, not a real self-intersection).
export function polylineSelfIntersects(loop) {
  const n = loop.length;
  if (n < 3) return false;
  for (let i = 0; i < n; i++) {
    const a0 = loop[i], a1 = loop[(i + 1) % n];
    for (let j = i + 1; j < n; j++) {
      if (j === i) continue;
      const shareEndpoint = j === (i + 1) % n || i === (j + 1) % n || i === j;
      if (shareEndpoint) continue;
      const b0 = loop[j], b1 = loop[(j + 1) % n];
      if (segmentsIntersect(a0, a1, b0, b1)) return true;
    }
  }
  return false;
}

// The real, honest validity gate required before a trim loop enters
// the document: every loop must be non-self-intersecting, and (for 2+
// loops) exactly the outer/hole winding convention must hold — the
// LARGEST-area loop (by absolute value) is the outer boundary and must wind
// one way, every other loop is a hole and must wind the OTHER way. Returns
// {ok:true} or {ok:false, reason} — never silently accepts a malformed set.
export function trimLoopsValid(loops) {
  if (!loops || loops.length === 0) return { ok: false, reason: 'a trimmed surface needs at least one trim loop' };
  for (let i = 0; i < loops.length; i++) {
    const loop = loops[i];
    if (loop.length < 3) return { ok: false, reason: `loop ${i} has fewer than 3 points — not a real closed region` };
    if (polylineSelfIntersects(loop)) return { ok: false, reason: `loop ${i} self-intersects` };
  }
  if (loops.length === 1) return { ok: true };
  const areas = loops.map(signedArea2D);
  let outerIdx = 0;
  for (let i = 1; i < loops.length; i++) if (Math.abs(areas[i]) > Math.abs(areas[outerIdx])) outerIdx = i;
  const outerSign = Math.sign(areas[outerIdx]);
  for (let i = 0; i < loops.length; i++) {
    if (i === outerIdx) continue;
    if (Math.sign(areas[i]) === outerSign) {
      return { ok: false, reason: `loop ${i} (a hole) winds the SAME direction as the outer loop ${outerIdx} — holes must wind the opposite way` };
    }
  }
  return { ok: true, outerIdx };
}

// The trimmed-surface object itself — a NurbsSrf plus trim loops, per doc
// 20's data model exactly: each loop segment carries slots for both its 2D
// UV curve (built here) and an optional 3D edge curve + tolerance (left
// null until an SSI-produced shared edge needs it — a nullable field from
// day one, on the explicit "cheap now, expensive to retrofit"
// reasoning, not a speculative addition).
export function makeTrimmedSurface(srf, loops, opts = {}) {
  const wrapped = loops.map((uv) => ({ uv, edge3d: null, tolerance: opts.tolerance ?? null }));
  const validity = trimLoopsValid(loops);
  return { srf, loops: wrapped, valid: validity.ok, invalidReason: validity.ok ? null : validity.reason, outerIdx: validity.outerIdx ?? 0 };
}

// The trivial, always-valid UNTRIMMED case, named explicitly: "the
// trivial single loop tracing the full parametric rectangle... existing
// objects need no migration." Real, exact rectangle corners from the
// surface's own knot domain — CCW winding in (u,v) by construction (this is
// the convention trimLoopsValid's outer/hole check assumes for a
// single-loop trim too, so anything built through THIS function on an
// untrimmed surface is consistent with a later hole added to it).
export function trivialTrimLoop(srf) {
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  return [[uMin, vMin], [uMax, vMin], [uMax, vMax], [uMin, vMax]];
}

// TRIM-AWARE NAKED-EDGE COUNT — the long-named
// "APP-SIDE RIPPLE" gap, named honestly in
// the app itself since the day Trim shipped (Properties' own "Trimmed"
// row: "Closed U/V and Open Edges above describe the full UNTRIMMED
// surface — trim-aware naked-edge logic isn't built yet"): plain
// `nakedEdgeCount` (surface.mjs) is CONTROL-NET-based — it only ever
// describes the surface's own untrimmed parametric rectangle, which is
// the wrong answer the instant a real trim loop exists. "Naked" needs to
// mean "a trim boundary not shared with anything else," not "one of the
// 4 domain sides," per the APP-SIDE RIPPLE wording exactly.
//
// The two real trimmed-surface shapes this app actually produces (a
// Trim's own single interior loop; a Split/IntersectSplit/SplitByCurve
// piece's identical `{trimLoop, trimHoles}` shape) cover every case
// without needing a stored "which kind of piece is this" flag: an
// INTERIOR trim loop (not the full untrimmed rectangle) discards every
// one of the original 4 sides entirely — the kept region's WHOLE
// boundary is the outer loop itself, always exactly 1 regardless of the
// surface's own U/V closure, plus one more per hole cut into it. The
// EXTERIOR piece of a Split-family cut (trimLoop IS the full untrimmed
// rectangle, with a hole where the other piece was removed) keeps every
// side that survives closure exactly as plain `nakedEdgeCount` already
// reports, plus one more naked loop per hole.
//
// Distinguishing the two is a direct geometric comparison against
// `trivialTrimLoop(srf)` (the exact untrimmed rectangle THIS surface
// would produce) rather than a stored flag — robust to a cloned/
// snapshotted trimLoop array (Copy/Paste, undo/redo, the debug-export
// field all clone via `.map((p) => [...p])`, never keep the same
// reference) with zero extra bookkeeping needed anywhere a trim
// container is built. `trimLoop == null` (the ordinary untrimmed case,
// every ordinary surface in the app today) is byte-identical to plain
// `nakedEdgeCount(srf)` — this function is a strict superset, not a
// replacement, and every existing `nakedEdgeCount` call site/test is
// untouched.
export function trimmedNakedEdgeCount(srf, trimLoop, trimHoles = []) {
  if (!trimLoop) return nakedEdgeCount(srf);
  const holeCount = trimHoles ? trimHoles.length : 0;
  const rect = trivialTrimLoop(srf);
  const isFullRect = trimLoop.length === rect.length
    && trimLoop.every((p, i) => Math.hypot(p[0] - rect[i][0], p[1] - rect[i][1]) < 1e-9);
  return (isFullRect ? nakedEdgeCount(srf) : 1) + holeCount;
}

// SEAM-CROSSING ("WRAPPED") INTERSECTION LOOP -> TWO BAND LOOPS.
//
// The "SEAM/POLE PROBLEM" names
// this as the wall that stops IntersectSplit on the recommended
// reference fixture (two perpendicular offset-axis cylinders, R2 < R1):
// the smaller cylinder's intersection curve necessarily wraps its full
// closed circumference, because R2*cos(theta) < R1 holds at EVERY theta.
// A wrapped curve is not a loop bounding a patch, so the interior/
// exterior-with-a-hole model IntersectSplit is built on cannot express
// it, and v1 detected the wrap and refused by name.
//
// THE KEY OBSERVATION, and the reason this is tractable rather than the
// "periodic-domain-aware clip" rewrite it was assumed to need: a curve
// that crosses the seam EXACTLY ONCE does not need a periodic clipper at
// all. It divides the parametric rectangle into exactly TWO regions, and
// each one is an ORDINARY SIMPLE POLYGON once it is closed through the
// domain boundary — walk the curve from one edge of the wrap axis to the
// other, then return along the far side of the rectangle. The two bands
// are plain trim loops of exactly the shape the existing tessellator
// already handles; nothing downstream needs to learn about periodicity.
// A trim edge lying exactly ON the parametric boundary is already
// exercised and proven by every exterior piece IntersectSplit builds
// today (trivialTrimLoop is that same shape).
//
// The two bands PARTITION the rectangle, so their areas must sum to the
// full domain area — a real, checkable invariant, asserted in the tests
// rather than assumed.
//
// HONESTLY REFUSED, each by its own name rather than one generic wall:
// a curve crossing the seam an EVEN number of times is not a wrap at all
// (it straddles the seam and comes back — an ordinary hole that happens
// to sit on the seam, which genuinely does need periodic clipping);
// crossing more than once in the same direction, or wrapping BOTH closed
// directions at once (a torus-like double wrap), are separate topologies
// this does not attempt.
export function seamBandLoops(loopUV, srf) {
  const cut = seamCrossingSpine(loopUV, srf);
  if (!cut.ok) return cut;
  const { spine, axisIndex: ai, oMin, oMax, aMin, aMax } = cut;
  const oi = 1 - ai;
  const at = (aVal, oVal) => { const p = [0, 0]; p[ai] = aVal; p[oi] = oVal; return p; };
  const bandLow = [...spine, at(aMax, oMin), at(aMin, oMin)];
  const bandHigh = [...spine, at(aMax, oMax), at(aMin, oMax)];
  for (const [name, band] of [['first', bandLow], ['second', bandHigh]]) {
    const v = trimLoopsValid([band]);
    if (!v.ok) return { ok: false, reason: `the ${name} band's own loop ${v.reason}` };
  }
  // Both bands wound the same way trivialTrimLoop is, so every downstream
  // consumer sees one consistent convention (the same normalization
  // computeIntersectSplitPieces already applies to its own two pieces).
  const outerSign = Math.sign(signedArea2D(trivialTrimLoop(srf)));
  const orient = (band) => (Math.sign(signedArea2D(band)) === outerSign ? band : band.slice().reverse());
  return { ok: true, axis: cut.axis, loops: [orient(bandLow), orient(bandHigh)] };
}

// The half of the above that a FACE SPLIT wants on its own: the wrapped
// loop re-expressed as an OPEN CHAIN running from one edge of the wrap axis
// to the other, both ends landing exactly ON the domain boundary. A boolean
// splitting a face hands its cut curves to the planar arrangement rather
// than clipping two bands itself, and the arrangement needs a cut that
// genuinely reaches the rectangle's edge — a chain stopping a sample short
// of it is a dangling spur, which Stage 3 correctly prunes away, leaving the
// face unsplit and the boolean quietly building from one fragment where
// there should be several. Extracted rather than reimplemented so the wrap
// arithmetic has exactly one definition.
export function seamCrossingSpine(loopUV, srf) {
  const j = seamJumps(loopUV, srf);
  if (!j.ok) return j;
  const { ai, oi, crossings, aMin, aMax, oMin, oMax } = j;
  const n = loopUV.length;
  // The crossing COUNT rides on the refusal as a real field, not only
  // inside the prose. A caller that wants to say something of its own
  // about this case must not have to parse a sentence back apart to do it
  // — the same reason `code` exists.
  if (crossings.length % 2 === 0) {
    return { ok: false, code: 'seam-straddle', crossings: crossings.length, reason: `crosses the seam ${crossings.length} times and returns — an ordinary loop sitting ON the seam rather than a wrap. seamStraddleChains handles it; this function does not` };
  }
  if (crossings.length !== 1) {
    return { ok: false, code: 'multi-wrap', crossings: crossings.length, reason: `crosses the seam ${crossings.length} times — only a single once-around wrap is handled` };
  }
  const k = crossings[0];
  // Rotate so the list begins immediately AFTER the seam crossing, making
  // the samples a genuinely OPEN chain running from one edge of the wrap
  // axis to the other instead of a cyclic list with a discontinuity in it.
  const rot = [];
  for (let i = 0; i < n; i++) rot.push(loopUV[(k + 1 + i) % n].slice());
  const ascending = rot[n - 1][ai] > rot[0][ai];
  const chain = ascending ? rot : rot.slice().reverse();
  // Both ends of the chain land on the SAME physical point, because a=aMin
  // and a=aMax are the same place on a closed surface — so the chain
  // genuinely closes in 3D even though its two endpoints differ in UV.
  const seam = seamPointAt(loopUV, k, ai, oi, aMin, aMax);
  const startPt = ascending ? seam.after : seam.before;
  const endPt = ascending ? seam.before : seam.after;
  const spine = closeChainOnSeam(chain, startPt, endPt);
  return { ok: true, axis: ai === 0 ? 'u' : 'v', axisIndex: ai, spine, aMin, aMax, oMin, oMax };
}

// The EVEN-crossing case seamCrossingSpine refuses: a loop that crosses the
// seam and comes back is not a wrap at all, it is an ordinary closed region
// that happens to straddle the seam. In the surface's own parameters that
// region is genuinely in SEVERAL pieces — the domain rectangle cuts it — so
// there is no single chain to return, and no re-expression of it as one that
// would not be a lie.
//
// What there IS, and what a face split actually needs, is one open chain per
// piece, each running between two points ON the domain boundary. Split the
// cyclic sample list at every crossing; each run between two consecutive
// crossings lies on one continuous side of the seam, and gets the crossing's
// own interpolated seam point attached at each end, on whichever edge that
// end's own neighboring sample sits against. Every chain then reaches the
// rectangle's edge at both ends, which is exactly the property the
// arrangement needs and the property a raw wrapped chain lacks.
//
// The two copies of a crossing point — (aMin, o) and (aMax, o) — are the same
// place in 3D, so the fragments either side of the seam weld to each other
// downstream for free, with no seam-specific sewing step.
//
// A once-around wrap is refused here by name rather than handled: it is
// seamCrossingSpine's own case, and a caller that let the two blur would be
// choosing between two genuinely different topologies by accident.
export function seamStraddleChains(loopUV, srf, forceAxis = -1) {
  const j = seamJumps(loopUV, srf, true, forceAxis);
  if (!j.ok) return j;
  const { ai, oi, crossings, aMin, aMax, oMin, oMax } = j;
  const n = loopUV.length;
  if (crossings.length % 2 === 1) {
    return { ok: false, code: 'not-a-straddle', reason: `crosses the seam ${crossings.length} times — an odd count is a wrap, which seamCrossingSpine handles` };
  }
  const m = crossings.length;
  const seams = crossings.map((k) => seamPointAt(loopUV, k, ai, oi, aMin, aMax));
  const chains = [];
  for (let jx = 0; jx < m; jx++) {
    const from = crossings[jx], to = crossings[(jx + 1) % m];
    // The run strictly between the two crossings: it begins at the sample
    // just past `from`'s jump and ends at the sample just before `to`'s.
    const body = [];
    for (let i = (from + 1) % n; ; i = (i + 1) % n) {
      body.push(loopUV[i].slice());
      if (i === to) break;
    }
    const chain = closeChainOnSeam(body, seams[jx].after, seams[(jx + 1) % m].before);
    // Two crossings landing on the same seam point with nothing between them
    // describe no region at all; carrying it would hand the arrangement a
    // zero-length cut.
    if (chain.length >= 2 && !samePt2(chain[0], chain[chain.length - 1])) chains.push(chain);
  }
  if (!chains.length) {
    return { ok: false, code: 'degenerate-straddle', reason: 'every subchain between seam crossings was degenerate' };
  }
  return { ok: true, axis: ai === 0 ? 'u' : 'v', axisIndex: ai, chains, aMin, aMax, oMin, oMax };
}

// THE CORNER CASE, LITERALLY: a region sitting where a doubly-closed
// surface's two seams meet.
//
// A torus is closed in u and in v, so u = uMin/uMax and v = vMin/vMax are two
// seams that cross at the domain's corners. A loop straddling that corner
// crosses each seam twice and winds around neither — contractible, ordinary
// geometry, and it was refused as a "double wrap" purely because both
// directions had jumps in them. Nothing about it is a wrap.
//
// It is handled by composing the two splits that already exist rather than by
// a third piece of seam arithmetic: split the loop on the first seam, which
// yields open chains reaching that seam's edges, then offer each chain to the
// open-chain splitter for whatever it still crosses. A chain that crosses
// nothing more is already a finished cut and passes through unchanged.
//
// The result is one chain per piece of the region, each reaching the domain
// boundary at both ends — the same contract seamStraddleChains and
// seamOpenChains each meet for their own case, which is what the arrangement
// downstream requires.
export function seamDoubleStraddleChains(loopUV, srf) {
  const j = seamJumps(loopUV, srf);
  if (j.ok || j.code !== 'double-straddle') {
    return { ok: false, code: 'not-a-double-straddle', reason: 'this loop does not cross both seams and return along each — the single-seam routines own it' };
  }
  const first = seamStraddleChains(loopUV, srf, 0);
  if (!first.ok) return first;
  const chains = [];
  for (const chain of first.chains) {
    const split = seamOpenChains(chain, srf);
    if (split.ok) { chains.push(...split.chains); continue; }
    // Nothing left to cross is the ordinary outcome for a piece that lies
    // clear of the second seam; any other refusal is real and is passed on.
    if (split.code === 'no-seam-crossing' || split.code === 'too-few-points') { chains.push(chain); continue; }
    return split;
  }
  if (!chains.length) {
    return { ok: false, code: 'degenerate-double-straddle', reason: 'every subchain between seam crossings was degenerate' };
  }
  return { ok: true, axis: 'uv', chains };
}

// The OPEN-CHAIN sibling of seamStraddleChains, and the only seam path in
// this module that handles a chain with two real endpoints.
//
// Two closed surfaces always meet in a closed curve, so the closed cases
// above cover only a pair that is closed on both sides. A sphere meeting a
// BOX PANEL meets it in an open ARC — the arc runs off the panel's own
// boundary rather than closing — and that arc can cross the sphere's seam
// meridian in its MIDDLE. In the surface's own parameters that arc is then
// genuinely two pieces, exactly as a straddling loop is, but it is NOT a
// loop: it has two real endpoints of its own that are not seam crossings
// and must not be closed along a domain edge.
//
// Split at every crossing; each run between crossings gets the crossing's
// own interpolated seam point attached at whichever end IS a crossing, and
// keeps its real terminal sample at whichever end is not. Every piece
// therefore reaches the rectangle's edge exactly where it genuinely leaves
// the domain and nowhere else — which is the property the arrangement
// needs, and the property the raw chain lacks: without this, the raw chain
// carries a phantom chord straight across the face between the two sides
// of the seam, the arrangement splits along that phantom, and the boolean
// builds from fragments that are not real.
//
// The two UV copies of a crossing — (aMin, o) and (aMax, o) — are the same
// place in 3D, so the fragments either side weld to each other downstream
// for free, with no seam-specific sewing step, exactly as in the closed
// case.
export function seamOpenChains(chainUV, srf) {
  const j = seamJumps(chainUV, srf, false);
  if (!j.ok) return j;
  const { ai, oi, crossings, aMin, aMax, oMin, oMax } = j;
  const n = chainUV.length;
  const seams = crossings.map((k) => seamPointAt(chainUV, k, ai, oi, aMin, aMax));
  const chains = [];
  // One more piece than there are crossings: [0..c0], (c0..c1], ..., (cLast..n-1].
  let startIdx = 0;
  for (let jx = 0; jx <= crossings.length; jx++) {
    const endIdx = jx < crossings.length ? crossings[jx] : n - 1;
    const body = [];
    for (let i = startIdx; i <= endIdx; i++) body.push(chainUV[i].slice());
    // A piece opens at the PREVIOUS crossing's far-side copy and closes at
    // THIS crossing's near-side copy; the first and last pieces keep the
    // curve's own real endpoint on their outer end instead.
    const startPt = jx > 0 ? seams[jx - 1].after : null;
    const endPt = jx < crossings.length ? seams[jx].before : null;
    const chain = closeChainOnSeam(body, startPt, endPt);
    // Two crossings landing on the same seam point with nothing between
    // them describe no cut at all; carrying it would hand the arrangement
    // a zero-length edge.
    if (chain.length >= 2 && !samePt2(chain[0], chain[chain.length - 1])) chains.push(chain);
    startIdx = endIdx + 1;
  }
  if (!chains.length) {
    return { ok: false, code: 'degenerate-open-straddle', reason: 'every subchain between seam crossings was degenerate' };
  }
  return { ok: true, axis: ai === 0 ? 'u' : 'v', axisIndex: ai, chains, aMin, aMax, oMin, oMax };
}

// The shared front half of every seam question: which closed direction this
// loop jumps across, and at which sample indices. Extracted so the wrap case
// and the straddle case cannot disagree about what a seam crossing even is —
// two copies of this test drifting apart would let one function see a wrap
// where the other saw a straddle.
//
// Every refusal carries a machine-readable `code` alongside its prose,
// because a CALLER has to tell two very different failures apart and must
// not do it by matching on the sentence. `no-seam-crossing` and
// `too-few-points` mean the loop is not entangled with a seam at all, so the
// raw UV chain is continuous and safe to use as-is. Every other code means
// the chain genuinely DOES jump across the domain — a caller that uses it
// raw anyway is splitting a face along a phantom chord. See unwrapSeamCut in
// boolean.mjs.
// `cyclic` is what distinguishes a CLOSED loop from an OPEN chain, and it
// is the whole difference between the two callers. A closed loop's last
// sample genuinely steps back to its first, so that step is a real edge to
// test. An OPEN chain's last→first step does not exist — testing it counts
// a fake crossing, which turns an open arc crossing the seam in its middle
// into an even-crossing straddle and gets it refused as degenerate.
function seamJumps(loopUV, srf, cyclic = true, forceAxis = -1) {
  const { closedU, closedV } = surfaceClosure(srf);
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const dom = [[uMin, uMax], [vMin, vMax]];
  const n = loopUV.length;
  // A closed loop needs 3 points to bound anything; an open chain needs
  // only 2 to have a crossing between them.
  const minPts = cyclic ? 3 : 2;
  if (n < minPts) return { ok: false, code: 'too-few-points', reason: `the intersection ${cyclic ? 'loop' : 'chain'} has fewer than ${minPts} points` };
  // A "jump" is a step longer than half the closed direction's own span —
  // the same test loopWrapsClosedDirection already uses to DETECT a wrap,
  // reused here to LOCATE it rather than only report that one exists.
  //
  // The SIGN of each jump is kept, because presence and winding are different
  // questions and only the second one distinguishes a wrap from a straddle. A
  // step that falls from near aMax to near aMin left the domain forwards and
  // re-entered at the bottom, so it counts +1; the reverse counts -1. A loop
  // that goes round the direction once nets +/-1; a loop that merely crosses
  // the seam and comes back nets 0, whatever its crossing count.
  const jumps = [[], []];
  const net = [0, 0];
  const steps = cyclic ? n : n - 1;
  for (let i = 0; i < steps; i++) {
    const a = loopUV[i], b = loopUV[(i + 1) % n];
    if (closedU && Math.abs(b[0] - a[0]) > (uMax - uMin) / 2) { jumps[0].push(i); net[0] += b[0] < a[0] ? 1 : -1; }
    if (closedV && Math.abs(b[1] - a[1]) > (vMax - vMin) / 2) { jumps[1].push(i); net[1] += b[1] < a[1] ? 1 : -1; }
  }
  // BOTH DIRECTIONS TOUCHED IS NOT THE SAME AS BOTH DIRECTIONS WRAPPED, and
  // calling it one was a false statement about the topology. A torus's two
  // seams MEET at a corner of the domain, so a small contractible loop sitting
  // over that corner crosses each seam twice and winds around neither — it is
  // an ordinary region in exotic parameters, not an exotic region. Only a
  // genuine wrap in both directions is the topology this cannot express.
  if (forceAxis < 0 && jumps[0].length && jumps[1].length) {
    if (net[0] !== 0 && net[1] !== 0) {
      return { ok: false, code: 'double-wrap', net, reason: `wraps BOTH closed directions at once (net winding ${net[0]} in u, ${net[1]} in v) — a separate topology this does not handle` };
    }
    if (net[0] === 0 && net[1] === 0) {
      return { ok: false, code: 'double-straddle', net, reason: 'crosses BOTH seams and returns along each — a contractible region sitting over the seams\' own corner. seamDoubleStraddleChains handles it; this function does not' };
    }
    return { ok: false, code: 'wrap-and-straddle', net, reason: `wraps one closed direction (net ${net[0] !== 0 ? net[0] + ' in u' : net[1] + ' in v'}) while straddling the other — a mixed topology this does not handle` };
  }
  const ai = forceAxis >= 0 ? forceAxis : (jumps[0].length ? 0 : (jumps[1].length ? 1 : -1));
  if (ai < 0 || !jumps[ai].length) return { ok: false, code: 'no-seam-crossing', reason: 'does not cross a closed direction\'s seam' };
  const oi = 1 - ai;
  return { ok: true, ai, oi, crossings: jumps[ai], net: net[ai], aMin: dom[ai][0], aMax: dom[ai][1], oMin: dom[oi][0], oMax: dom[oi][1] };
}

// One seam crossing, as the two UV points that describe it. The jump runs
// from sample k to sample k+1; the curve leaves the domain past one edge and
// re-enters at the other, so the crossing has a copy on each edge, sharing
// the same value on the other axis. `before` is the copy on the side sample k
// sits against, `after` the copy on sample k+1's side — so a chain arriving
// at the jump closes with `before` and a chain leaving it opens with `after`.
//
// The other-axis value is interpolated across the jump by how far through it
// the edge falls, which is the curve's own crossing value rather than either
// neighboring sample's.
function seamPointAt(loopUV, k, ai, oi, aMin, aMax) {
  const n = loopUV.length;
  const p = loopUV[k], q = loopUV[(k + 1) % n];
  const pHigh = p[ai] > q[ai];
  const gap = pHigh ? (aMax - p[ai]) + (q[ai] - aMin) : (p[ai] - aMin) + (aMax - q[ai]);
  const frac = gap > 1e-12 ? (pHigh ? (aMax - p[ai]) : (p[ai] - aMin)) / gap : 0.5;
  const oSeam = p[oi] + (q[oi] - p[oi]) * frac;
  const at = (aVal) => { const r = [0, 0]; r[ai] = aVal; r[oi] = oSeam; return r; };
  return { before: at(pHigh ? aMax : aMin), after: at(pHigh ? aMin : aMax) };
}

function samePt2(p, q) {
  return Math.abs(p[0] - q[0]) < 1e-9 && Math.abs(p[1] - q[1]) < 1e-9;
}

// Attach a seam point to each end of a chain, dropping a terminal sample that
// already coincides with the point being added — otherwise the polygon built
// from it carries a zero-length opening segment for the self-intersection
// check to trip over.
//
// Either end may be null, which means "this end is not a seam crossing,
// leave the chain's own terminal sample alone" — the case an OPEN chain's
// first and last sub-chains have (they end at the curve's real endpoints,
// not at a seam). With both ends supplied this is byte-identical to the
// closed-loop behavior it was written for.
function closeChainOnSeam(chain, startPt, endPt) {
  const body = chain.filter((p, i) => !(i === 0 && startPt && samePt2(p, startPt)) && !(i === chain.length - 1 && endPt && samePt2(p, endPt)));
  return [...(startPt ? [startPt] : []), ...body, ...(endPt ? [endPt] : [])];
}
