// Blend surface — a new NURBS surface between an edge of one surface and an
// edge of another, meeting each with G1 or G2 continuity, as Rhino's BlendSrf
// does. Nothing existing moves: MatchEdge (matchedge.mjs) bends a surface's own
// boundary rows to meet a neighbor; this builds the surface between.
//
// The construction is the curve blend in blend.mjs (a Bezier of degree 2k+1
// from the beta-constraint characterization of G^k), run once per station
// along the two edges and skinned. At a station the
// neighbor arriving at edge A is A's own cross-boundary isocurve, traveling
// toward the edge (extractIsocurveU/V, derivatives to third order, the odd
// ones negated when the edge is at the start of that isocurve's domain — the
// same reversal identity geometricBlend uses for its far end). So the blend
// leaves A along A's own surface and arrives at B along B's, exactly at every
// station; between stations the skin interpolates the sections' control
// points with a degree-3 curve (chord-length parameters, averaging knots — the
// loft's own skinning), and the continuity there is that interpolant's.
//
// Bulge is geometricBlend's tangent magnitude as a multiple of the station's
// chord: 1 is the shape the curve blend takes by default, less pulls the
// blend straight, more lets it swell out along the neighbor before turning.
//
// Correspondence: both edges are walked by arc length, B in whichever
// direction pairs its ends nearer to A's — the bridge's rule. Two closed
// rims have no ends: B's direction and the point paired with A's seam are
// the ones that minimize the summed distance between paired points. A
// twisted pair of open edges is a valid shape a user can want; it is not
// searched for.
//
// The result's V runs along the edges (degree 3) and its U across the blend
// (degree 2k+1, a single Bezier span): U = 0 is edge A, U = 1 is edge B.
//
// Coordinates: one frame. A caller with placed surfaces carries their nets
// into the world first (blendSurfaceInputs in the app) and the result is
// placed at the identity.

import { extractIsocurveU, extractIsocurveV } from './isocurve.mjs';
import { curvePoint, rationalCurveDerivs } from './curve.mjs';
import { geometricBlend } from './blend.mjs';
import { chordLengthParams, averagingKnotVector, interpAtParams } from './interpolate.mjs';
import { surfaceDerivs2 } from './curvature.mjs';
import { fitPatch, fitPatchToTolerance } from './patch.mjs';
import { findSpan, basisFuns } from './basis.mjs';
import { nSidedTangentPatch } from './tangentpatch.mjs';
const blendSrfDerivs2 = surfaceDerivs2;

const bsSub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const bsLen = (a) => Math.hypot(a[0], a[1], a[2]);

// The edge as a curve of its own parameter and the cross-boundary neighbor
// frame at a point of it. `t` is the edge's own parameter fraction, 0..1 in
// the surface's along-edge domain. `frame(t, onward)`: geometricBlend's
// convention is the curve blend's — the start frame travels out of its own
// surface into the gap, the end frame travels out of the gap onward into its
// surface — so the frame is read with the odd derivatives signed for travel
// toward the edge (start) or away from it (end).
function blendSrfEdgeAccess(srf, edge) {
  const u0 = srf.knotsU[srf.degU], u1 = srf.knotsU[srf.knotsU.length - 1 - srf.degU];
  const v0 = srf.knotsV[srf.degV], v1 = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  if (edge === 'u0' || edge === 'u1') {
    const uFix = edge === 'u0' ? u0 : u1;
    const alongCrv = extractIsocurveU(srf, uFix); // runs along V
    return {
      alongCrv,
      point: (t) => curvePoint(alongCrv, v0 + t * (v1 - v0)),
      // The cross isocurve at this station runs along U; the edge sits at its start (u0) or end (u1).
      frame: (t, onward = false) => {
        const cross = extractIsocurveV(srf, v0 + t * (v1 - v0));
        const [C0, C1, C2, C3] = rationalCurveDerivs(cross, uFix, 3);
        // Traveling toward the edge from the interior: at u0 that is decreasing u; onward is the reverse.
        const s = (edge === 'u0' ? -1 : 1) * (onward ? -1 : 1);
        return { point: C0, d1: C1.map((x) => x * s), d2: C2, d3: C3.map((x) => x * s) };
      },
    };
  }
  const vFix = edge === 'v0' ? v0 : v1;
  const alongCrv = extractIsocurveV(srf, vFix); // runs along U
  return {
    alongCrv,
    point: (t) => curvePoint(alongCrv, u0 + t * (u1 - u0)),
    frame: (t, onward = false) => {
      const cross = extractIsocurveU(srf, u0 + t * (u1 - u0));
      const [C0, C1, C2, C3] = rationalCurveDerivs(cross, vFix, 3);
      const s = (edge === 'v0' ? -1 : 1) * (onward ? -1 : 1);
      return { point: C0, d1: C1.map((x) => x * s), d2: C2, d3: C3.map((x) => x * s) };
    },
  };
}

// Arc-length fractions → parameter fractions, from a dense sample.
function blendSrfArcParam(pointAt, samples = 128) {
  const cum = [0];
  let prev = pointAt(0);
  for (let i = 1; i <= samples; i++) { const p = pointAt(i / samples); cum.push(cum[i - 1] + bsLen(bsSub(p, prev))); prev = p; }
  const total = cum[samples];
  return (s) => {
    if (!(total > 0)) return s;
    const target = s * total;
    let i = 0; while (i < samples && cum[i + 1] < target) i++;
    const seg = cum[i + 1] - cum[i];
    return (i + (seg > 0 ? (target - cum[i]) / seg : 0)) / samples;
  };
}

export const BLEND_SURFACE_EDGES = ['u0', 'u1', 'v0', 'v1'];

// The surface's own normal (Su x Sv, unnormalized) at a parameter fraction t
// of one of its edges: the side a face built against that edge must agree
// with, or the joined shell turns inside out across the seam.
function blendSrfEdgeNormal(srf, edge, t) {
  const u0 = srf.knotsU[srf.degU], u1 = srf.knotsU[srf.knotsU.length - 1 - srf.degU];
  const v0 = srf.knotsV[srf.degV], v1 = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  const u = edge === 'u0' ? u0 : edge === 'u1' ? u1 : u0 + t * (u1 - u0);
  const v = edge === 'v0' ? v0 : edge === 'v1' ? v1 : v0 + t * (v1 - v0);
  const P = surfacePointAndPartials(srf, u, v);
  return bsCross(P.su, P.sv);
}
// Samples of a face, on an n x n grid inside its domain, whose Jacobian turns
// against the face's own summed normal: a fold.
function blendSrfFaceReversals(srf, n = 12) {
  const u0 = srf.knotsU[srf.degU], u1 = srf.knotsU[srf.knotsU.length - 1 - srf.degU];
  const v0 = srf.knotsV[srf.degV], v1 = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  const J = [];
  for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) {
    const P = surfacePointAndPartials(srf, u0 + (u1 - u0) * (0.002 + 0.996 * i / n), v0 + (v1 - v0) * (0.002 + 0.996 * j / n));
    J.push(bsCross(P.su, P.sv));
  }
  const sum = J.reduce((acc, x) => bsAdd(acc, x), [0, 0, 0]);
  return J.filter((x) => bsDot(x, sum) < 0).length;
}
// A surface with its V run reversed: the same sheet, facing the other way.
function blendSrfFlipV(srf) {
  const K = srf.knotsV, a = K[0], b = K[K.length - 1];
  return { ...srf, knotsV: K.map((x) => a + b - x).reverse(), ctrlNet: srf.ctrlNet.map((row) => row.slice().reverse()) };
}

/**
 * @param A {srf, edge}  the first surface (a NurbsSrf {degU, degV, knotsU, knotsV, ctrlNet}) and which of its four edges
 * @param B {srf, edge}  the second
 * @param opts continuity 1|2 (default 2), bulgeA/bulgeB (default 1, multiples of the chord), stations (default 24)
 * @returns {ok, srf, stations, chordMin, chordMax} or {ok:false, reason}
 */
export function blendSurfaceBetweenEdges(A, B, opts = {}) {
  const continuity = opts.continuity ?? 2;
  if (continuity !== 1 && continuity !== 2 && continuity !== 3) return { ok: false, reason: 'a surface blend is G1, G2 or G3' };
  for (const [name, X] of [['first', A], ['second', B]]) {
    if (!X || !X.srf || !X.srf.ctrlNet || !BLEND_SURFACE_EDGES.includes(X.edge)) return { ok: false, reason: `the ${name} surface and one of its edges (u0, u1, v0, v1) are needed` };
  }
  let stations = Math.max(4, Math.round(opts.stations ?? 24));
  const bulgeA = opts.bulgeA ?? 1, bulgeB = opts.bulgeB ?? 1;
  if (!(bulgeA > 0) || !(bulgeB > 0)) return { ok: false, reason: 'bulge must be positive' };

  const ea = blendSrfEdgeAccess(A.srf, A.edge), eb = blendSrfEdgeAccess(B.srf, B.edge);
  const arcA = blendSrfArcParam(ea.point), arcB = blendSrfArcParam(eb.point);
  // B's direction: the one that pairs the ends nearer.
  const a0 = ea.point(0), a1 = ea.point(1), b0 = eb.point(0), b1 = eb.point(1);
  const straight = bsLen(bsSub(a0, b0)) + bsLen(bsSub(a1, b1)), flipped = bsLen(bsSub(a0, b1)) + bsLen(bsSub(a1, b0));
  // opts.flip = [a, b]: each reverses that edge's run against the pairing the
  // gap chooses (the ends that lie nearer each other).
  const flip = Array.isArray(opts.flip) ? opts.flip : [];
  const flipOne = !!flip[0] !== !!flip[1];
  let reverseB = (flipped < straight) !== flipOne, shiftB = 0;
  // Two closed rims have no ends to pair: their end distances tie and the
  // seams can sit anywhere round them. B's direction and the point of B
  // paired with A's seam are the ones that minimize the summed distance
  // between paired points round the loop; any other pairing twists the blend
  // through its own axis (two coaxial tubes run opposite ways pinch to a point).
  const closedPair = blendSrfEdgeClosed(ea) && blendSrfEdgeClosed(eb);
  if (closedPair) {
    const n = 64, m = 256, wrap = (x) => x - Math.floor(x);
    const PA = Array.from({ length: n }, (_, i) => ea.point(arcA(i / n)));
    const PB = Array.from({ length: m }, (_, j) => eb.point(arcB(j / m)));
    const best = [null, null]; // [forward, reversed]: { cost, k }
    for (let d = 0; d < 2; d++) for (let k = 0; k < m; k++) {
      let cost = 0;
      for (let i = 0; i < n; i++) cost += bsLen(bsSub(PA[i], PB[((k + (d ? -1 : 1) * i * (m / n)) % m + m) % m]));
      if (!best[d] || cost < best[d].cost) best[d] = { cost, k };
    }
    const d = (best[1].cost < best[0].cost) !== flipOne ? 1 : 0, sgn = d ? -1 : 1;
    const costAt = (sh) => { let c = 0; for (let i = 0; i < n; i++) c += bsLen(bsSub(PA[i], eb.point(arcB(wrap(sh + sgn * i / n))))); return c; };
    let lo = (best[d].k - 1) / m, hi = (best[d].k + 1) / m;
    for (let it = 0; it < 40; it++) { const a = lo + (hi - lo) / 3, b = hi - (hi - lo) / 3; if (costAt(a) < costAt(b)) hi = b; else lo = a; }
    reverseB = d === 1; shiftB = wrap((lo + hi) / 2);
  }
  const tB = closedPair
    ? (s) => { const x = shiftB + (reverseB ? -s : s); return arcB(x - Math.floor(x)); }
    : (s) => (reverseB ? arcB(1 - s) : arcB(s));

  const sectionAt = (s) => {
    const fa = ea.frame(arcA(s)), fb = eb.frame(tB(s), true), chord = bsLen(bsSub(fb.point, fa.point));
    if (!(chord > 1e-9)) return { ok: false, reason: 'the two edges touch — there is no gap to blend across' };
    const r = geometricBlend(fa, fb, { continuity, startMagnitude: bulgeA * chord, endMagnitude: bulgeB * chord });
    return r.ok ? { ok: true, crv: r.crv, chord } : r;
  };
  const build = (count) => {
    const sections = [], sGrid = [];
    let chordMin = Infinity, chordMax = 0;
    for (let i = 0; i < count; i++) {
      const s = i / (count - 1), r = sectionAt(s);
      if (!r.ok) return { ok: false, reason: `station ${i + 1} of ${count}: ${r.reason}` };
      sections.push(r.crv); sGrid.push(s); chordMin = Math.min(chordMin, r.chord); chordMax = Math.max(chordMax, r.chord);
    }
    // Skin: the sections share degree and knots (one Bezier span), so their
    // control points skin column by column across the stations.
    const degU = sections[0].degree, knotsU = sections[0].knots.slice(), nU = sections[0].ctrlPts.length;
    const vbarSum = new Array(count).fill(0);
    for (let i = 0; i < nU; i++) {
      const v = chordLengthParams(sections.map((c) => c.ctrlPts[i]));
      for (let j = 0; j < count; j++) vbarSum[j] += v[j];
    }
    const vbar = vbarSum.map((x) => x / nU), degV = Math.min(3, count - 1), knotsV = averagingKnotVector(vbar, degV), ctrlNet = [];
    for (let i = 0; i < nU; i++) ctrlNet.push(interpAtParams(sections.map((c) => c.ctrlPts[i]), degV, vbar, knotsV));
    return { ok: true, srf: { degU, knotsU, degV, knotsV, ctrlNet }, sections, sGrid, vbar, chordMin, chordMax };
  };
  const vAt = (built, s) => {
    let i = Math.min(built.sGrid.length - 2, Math.floor(s * (built.sGrid.length - 1)));
    while (i + 1 < built.sGrid.length - 1 && built.sGrid[i + 1] < s) i++;
    const f = (s - built.sGrid[i]) / (built.sGrid[i + 1] - built.sGrid[i]);
    return built.vbar[i] + f * (built.vbar[i + 1] - built.vbar[i]);
  };
  // How far the skin lies from the section family between stations: the
  // exact section at s against the skin's isocurve at the v whose two ends
  // lie nearest the section's (the chord-length v of s slides along the
  // edges, which is no distance off the shape). The edges are measured
  // apart: the skin's boundary rows against the edges themselves, which a
  // join must meet.
  const deviationOf = (built) => {
    let worst = 0, edge = 0;
    // The boundary rows as curves along V: a curve evaluation where a surface
    // one would rebuild the whole net per call (the search below makes ~130).
    const sk = built.srf, uLo = sk.knotsU[sk.degU], uHi = sk.knotsU[sk.knotsU.length - 1 - sk.degU];
    const rowA = extractIsocurveU(sk, uLo), rowB = extractIsocurveU(sk, uHi);
    for (let i = 0; i + 1 < built.sGrid.length; i++) for (const f of [0.25, 0.5, 0.75]) {
      const s = built.sGrid[i] + f * (built.sGrid[i + 1] - built.sGrid[i]), exact = sectionAt(s);
      if (!exact.ok) return { worst: Infinity, edge: Infinity };
      const u0 = exact.crv.knots[exact.crv.degree], u1 = exact.crv.knots[exact.crv.knots.length - 1 - exact.crv.degree];
      const e0 = curvePoint(exact.crv, u0), e1 = curvePoint(exact.crv, u1);
      const miss = (v) => bsLen(bsSub(curvePoint(rowA, v), e0)) + bsLen(bsSub(curvePoint(rowB, v), e1));
      let lo = built.vbar[i], hi = built.vbar[i + 1];
      for (let it = 0; it < 32; it++) { const a = lo + (hi - lo) / 3, b = hi - (hi - lo) / 3; if (miss(a) < miss(b)) hi = b; else lo = a; }
      const v = (lo + hi) / 2, across = extractIsocurveV(sk, v);
      edge = Math.max(edge, bsLen(bsSub(curvePoint(rowA, v), e0)), bsLen(bsSub(curvePoint(rowB, v), e1)));
      for (const u of [0, 0.25, 0.5, 0.75, 1]) worst = Math.max(worst, bsLen(bsSub(curvePoint(across, uLo + u * (uHi - uLo)), curvePoint(exact.crv, u0 + u * (u1 - u0)))));
    }
    return { worst, edge };
  };

  let built = build(stations);
  if (!built.ok) return built;
  let measured = deviationOf(built), refined = false;
  // edgeTolerance: how near the boundary rows must lie to the edges (a join's
  // tolerance); by default the shape's own.
  const tolerance = opts.tolerance ?? Math.max(1e-4, 1e-4 * built.chordMax), maxStations = Math.max(stations, Math.round(opts.maxStations ?? 97));
  const edgeTolerance = Math.min(tolerance, opts.edgeTolerance ?? tolerance);
  const met = (m) => m.worst <= tolerance && m.edge <= edgeTolerance;
  if (opts.adaptive === true) while (!met(measured) && stations < maxStations) {
    stations = Math.min(maxStations, 2 * (stations - 1) + 1);
    built = build(stations); if (!built.ok) return built;
    measured = deviationOf(built); refined = true;
  }
  const deviation = measured.worst, edgeDeviation = measured.edge;
  // The blend faces the way A does along their shared edge.
  let facing = 0;
  for (const s of [0.25, 0.5, 0.75]) {
    const P = surfacePointAndPartials(built.srf, built.srf.knotsU[built.srf.degU], vAt(built, s));
    facing += Math.sign(bsDot(bsCross(P.su, P.sv), blendSrfEdgeNormal(A.srf, A.edge, arcA(s))));
  }
  return { ok: true, srf: facing < 0 ? blendSrfFlipV(built.srf) : built.srf, stations, chordMin: built.chordMin, chordMax: built.chordMax, reverseB, continuity, deviation, edgeDeviation, tolerance, edgeTolerance, refined, toleranceMet: met(measured) };
}

// A junction of three or more edges
//
// Two kinds, told apart by how the sheets lie to the junction plane. Sheets
// edge-on to it (three walls meeting at a corner post) grow arms — a blend from
// each edge to a port, the edge carried `reach` of the way across the junction
// at its full width — and ribbons from the ports to a common spine, Coons
// patches (coons.mjs, exact on their boundaries) with open air between them,
// G0 across the spine. Sheets lying in the plane (three wings meeting flat)
// take the guided cap below, bounded by their own side edges and smooth
// throughout. Every seam is one shared curve: no fit, no trim.
//
// Arms round one plane only: the ports are ordered by angle about the plane
// the edge midpoints best share, as the SubD bridge orders its arms.
import { surfacePoint, surfacePointAndPartials } from './surface.mjs';
import { coonsPatch, coonsCompatible } from './coons.mjs';
import { insertKnot, insertKnotOnce, degreeElevateCurve, rescaleCurveDomain, extractSubCurve, concatTwoC0 } from './knots.mjs';

const bsAdd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const bsScale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const bsCross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const bsDot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const bsNorm = (a) => { const L = bsLen(a); return L > 1e-12 ? bsScale(a, 1 / L) : [0, 0, 0]; };

// A skin between a start station family and an end station family — the
// two-edge blend's own skin, shared with the arms.
function blendSrfSkin(frameA, frameB, stations, continuity, bulgeA, bulgeB) {
  const sections = [];
  let chordMin = Infinity, chordMax = 0;
  for (let i = 0; i < stations; i++) {
    const s = i / (stations - 1);
    const fa = frameA(s), fb = frameB(s);
    const chord = bsLen(bsSub(fb.point, fa.point));
    if (!(chord > 1e-9)) return { ok: false, reason: 'the two edges touch — there is no gap to blend across' };
    chordMin = Math.min(chordMin, chord); chordMax = Math.max(chordMax, chord);
    const r = geometricBlend(fa, fb, { continuity, startMagnitude: bulgeA * chord, endMagnitude: bulgeB * chord });
    if (!r.ok) return { ok: false, reason: `station ${i + 1} of ${stations}: ${r.reason}` };
    sections.push(r.crv);
  }
  const degU = sections[0].degree, knotsU = sections[0].knots.slice();
  const nU = sections[0].ctrlPts.length;
  const vbarSum = new Array(stations).fill(0);
  for (let i = 0; i < nU; i++) {
    const v = chordLengthParams(sections.map((c) => c.ctrlPts[i]));
    for (let j = 0; j < stations; j++) vbarSum[j] += v[j];
  }
  const vbar = vbarSum.map((x) => x / nU);
  const degV = Math.min(3, stations - 1);
  const knotsV = averagingKnotVector(vbar, degV);
  const ctrlNet = [];
  for (let i = 0; i < nU; i++) ctrlNet.push(interpAtParams(sections.map((c) => c.ctrlPts[i]), degV, vbar, knotsV));
  return { ok: true, srf: { degU, knotsU, degV, knotsV, ctrlNet }, chordMin, chordMax };
}

// The guided cap
//
// Sheets meeting flat are bounded by their own side edges, and those guide
// the junction: the side edge of one sheet is blended (a curve blend, at the
// continuity asked) into the side edge of the next, and the web between the
// sheet edges and those free edges is filled at that continuity to every
// sheet and G2 along the interior of its own seams (the extraordinary end
// vertices retain the G1 fan condition). The fill is 2N quads about center C,
// fitted from the sheets' inward directions: one spoke from every sheet edge's parameter midpoint
// to C (a blend leaving the sheet) and one from every free edge's midpoint
// (leaving it square), so each quad is half a sheet edge, half a free edge
// and two spokes.
//
// A quad is the Boolean sum P_v + P_u - P_u P_v of two Hermite projectors:
// across from the sheet (position and the first k cross derivatives at the
// sheet, position and cross derivative at the far spoke) and along it
// (position and cross derivative at both side curves). Every cross field is a
// curve chosen so the data agree at the corners — a side leaves the sheet at
// the ribbon's own derivatives, a field's end value is the neighboring
// side's tangent, the twists match — and then the sum interpolates all of it
// exactly: the quad is its four curves, G^k to the sheet, and its cross
// derivative along each spoke is a prescribed field.
//
// Across a spoke the two quads share that field up to a multiple of the
// spoke's own tangent, α(v)·E'(v), which keeps the tangent plane (G1) and is
// what makes the center solvable: the 2N spokes arrive at C in one tangent
// plane, each sheet's along its own line to C and each free spoke between
// the corners beside it, all with zero second derivative, and the cross
// fields there are the neighboring spokes' tangents — for which speeds
// chosen round the fan give m_{i-1} + m_i ∥ e_i and α(1) per spoke (2cos(π/N)
// on an even fan). A bounded third-row solve then equalizes geometric
// transverse normal curvature along each seam without moving its boundary
// or tangent plane. The twists are zero at C and at the free-edge splits. A
// speed ratio g per sheet (the blend's cross speed over the sheet's own),
// eased to each corner's own share, keeps a corner's curve and the ribbon's
// cross derivative the same vector; the free edges are built at 2g because a
// half of one, on [0,1], runs at half its speed.
const bsGreville = (knots, p) => Array.from({ length: knots.length - p - 1 }, (_, i) => { let a = 0; for (let k = 1; k <= p; k++) a += knots[i + k]; return a / p; });
const bsBinom = (n, k) => { let r = 1; for (let i = 1; i <= k; i++) r = r * (n - k + i) / i; return r; };
// Bezier arithmetic on arrays of 3-vectors.
const bzElevate = (pts, target) => { let P = pts; while (P.length - 1 < target) { const n = P.length - 1, Q = [P[0]]; for (let i = 1; i <= n; i++) Q.push(bsAdd(bsScale(P[i - 1], i / (n + 1)), bsScale(P[i], 1 - i / (n + 1)))); Q.push(P[n]); P = Q; } return P; };
const bzHodo = (pts) => { const n = pts.length - 1; return Array.from({ length: n }, (_, i) => bsScale(bsSub(pts[i + 1], pts[i]), n)); };
const bzScalarTimes = (s, pts) => { const a = s.length - 1, b = pts.length - 1, out = []; for (let k = 0; k <= a + b; k++) { let acc = [0, 0, 0]; for (let i = Math.max(0, k - b); i <= Math.min(a, k); i++) acc = bsAdd(acc, bsScale(pts[k - i], bsBinom(a, i) * bsBinom(b, k - i) / bsBinom(a + b, k) * s[i])); out.push(acc); } return out; };
const bzAdd = (A, B) => { const d = Math.max(A.length, B.length) - 1; const a = bzElevate(A, d), b = bzElevate(B, d); return a.map((p, i) => bsAdd(p, b[i])); };
const bzScale = (A, s) => A.map((p) => bsScale(p, s));
// A Bezier from derivative conditions at t = 0 (c0 = [p, d1, d2, …]) and t = 1 (c1 likewise).
const bzHermite = (c0, c1) => {
  const n = c0.length + c1.length - 1, b = new Array(n + 1);
  const ff = (m) => { let r = 1; for (let i = 0; i < m; i++) r *= (n - i); return r; };
  b[0] = c0[0];
  if (c0.length > 1) b[1] = bsAdd(b[0], bsScale(c0[1], 1 / ff(1)));
  if (c0.length > 2) b[2] = bsAdd(bsSub(bsScale(b[1], 2), b[0]), bsScale(c0[2], 1 / ff(2)));
  if (c0.length > 3) b[3] = bsAdd(bsAdd(bsSub(bsScale(b[2], 3), bsScale(b[1], 3)), b[0]), bsScale(c0[3], 1 / ff(3)));
  b[n] = c1[0];
  if (c1.length > 1) b[n - 1] = bsSub(b[n], bsScale(c1[1], 1 / ff(1)));
  if (c1.length > 2) b[n - 2] = bsAdd(bsSub(bsScale(b[n - 1], 2), b[n]), bsScale(c1[2], 1 / ff(2)));
  if (c1.length > 3) b[n - 3] = bsAdd(bsAdd(bsSub(bsScale(b[n - 2], 3), bsScale(b[n - 1], 3)), b[n]), bsScale(c1[3], -1 / ff(3)));
  return b;
};
// The end derivatives of a clamped B-spline row (3-vectors) on knots U, degree p.
const bsRowD0 = (row, U, p) => bsScale(bsSub(row[1], row[0]), p / (U[p + 1] - U[1]));
const bsRowD1 = (row, U, p) => { const n = row.length - 1; return bsScale(bsSub(row[n], row[n - 1]), p / (U[n + p] - U[n])); };
const asCrv = (pts, degree, knots) => ({ degree, knots: knots.slice(), ctrlPts: pts.map((q) => [q[0], q[1], q[2], 1]) });
const xyz = (crv) => crv.ctrlPts.map((q) => [q[0], q[1], q[2]]);
const bzOf = (crv) => xyz(crv); // a single-span curve's control points as Bezier points
const crvDeriv = (crv, t, order) => rationalCurveDerivs(crv, t, order)[order];
// h: a quintic rising 0 → 1 with zero first, second and third derivative at 0 and zero first derivative at 1.
const H_RISE = [0, 0, 0, 0, 1, 1]; // v⁴(5 - 4v) as Bezier coefficients

// A quad of the cap. edge = { crv on [0,1], frameAt(u) }, k the continuity;
// side0 / side1 run v 0 → 1 from the sheet to the far spoke with cross fields
// fieldA / fieldB (Bezier arrays); far runs u 0 → 1 with its cross field fieldF.
function blendSrfBooleanQuad(q) {
  const { k, side0, side1, far, fieldA, fieldB, fieldF } = q;
  const { rows, p, U } = q.uRows;
  // P_v: Hermite columns across, k+1 conditions at the sheet, two at the far spoke.
  const farRow = xyz(far.inBasis), cfRow = xyz(fieldF.inBasis);
  const nV = k + 2;
  const Pv = rows[0].map((_, i) => bzHermite(rows.map((r) => r[i]), [farRow[i], cfRow[i]]));
  // P_u: cubic Hermite along, on the sides' common Bezier degree.
  const qDeg = Math.max(side0.length, side1.length, fieldA.length, fieldB.length) - 1;
  const c3 = bzElevate(side0, qDeg), c1 = bzElevate(side1, qDeg), A = bzElevate(fieldA, qDeg), B = bzElevate(fieldB, qDeg);
  const Pu = [c3, c3.map((pt, j) => bsAdd(pt, bsScale(A[j], 1 / 3))), c1.map((pt, j) => bsSub(pt, bsScale(B[j], 1 / 3))), c1];
  // P_u P_v: the same projector applied to P_v's own side data.
  const nU = Pv.length;
  const d0 = Pv[0].map((_, j) => bsRowD0(Pv.map((col) => col[j]), U, p));
  const d1 = Pv[0].map((_, j) => bsRowD1(Pv.map((col) => col[j]), U, p));
  const Puv = [Pv[0], Pv[0].map((pt, j) => bsAdd(pt, bsScale(d0[j], 1 / 3))), Pv[nU - 1].map((pt, j) => bsSub(pt, bsScale(d1[j], 1 / 3))), Pv[nU - 1]];
  // One basis: v to the larger Bezier degree, u to (p, U).
  const Q = Math.max(nV, qDeg);
  const PvQ = Pv.map((col) => bzElevate(col, Q)), PuQ = Pu.map((col) => bzElevate(col, Q)), PuvQ = Puv.map((col) => bzElevate(col, Q));
  const toU = (cols) => { // 4 columns (cubic Bezier in u) → nU columns on (p, U)
    const out = Array.from({ length: nU }, () => new Array(Q + 1));
    for (let j = 0; j <= Q; j++) {
      let c = asCrv(cols.map((col) => col[j]), 3, [0, 0, 0, 0, 1, 1, 1, 1]);
      c = degreeElevateCurve(c, p);
      for (const t of U.slice(p + 1, U.length - p - 1)) c = insertKnotOnce(c, t);
      const pts = xyz(c);
      for (let i = 0; i < nU; i++) out[i][j] = pts[i];
    }
    return out;
  };
  const PuU = toU(PuQ), PuvU = toU(PuvQ);
  const ctrlNet = [];
  for (let i = 0; i < nU; i++) { const row = []; for (let j = 0; j <= Q; j++) { const s = bsSub(bsAdd(PvQ[i][j], PuU[i][j]), PuvU[i][j]); row.push([s[0], s[1], s[2], 1]); } ctrlNet.push(row); }
  const V = []; for (let j = 0; j <= Q; j++) V.push(0); for (let j = 0; j <= Q; j++) V.push(1);
  return { degU: p, knotsU: U.slice(), degV: Q, knotsV: V, ctrlNet };
}

// The u basis of a quad and its ribbon rows: position and the k cross
// derivative fields of the sheet, sampled at the Greville abscissae and
// interpolated in the basis the edge half and the far spoke share.
function blendSrfQuadRows(edge, far, fieldF, g, k) {
  let [row0, rowFar] = coonsCompatible(edge.crv, far);
  // Nine gives the later G2 solve interior control points to move while the
  // edge jet and the already-solved neighboring seam remain untouched.
  if (row0.degree < 9) { row0 = degreeElevateCurve(row0, 9); rowFar = degreeElevateCurve(rowFar, 9); }
  let cf = asCrv(fieldF, fieldF.length - 1, [...fieldF.map(() => 0), ...fieldF.map(() => 1)]);
  [row0, cf] = coonsCompatible(row0, cf); [rowFar, cf] = coonsCompatible(rowFar, cf); [row0, rowFar] = coonsCompatible(row0, rowFar);
  while (row0.ctrlPts.length < BLEND_CAP_ROWS_U) {
    const K = row0.knots; let bi = -1, bw = 0;
    for (let i = 0; i + 1 < K.length; i++) if (K[i + 1] - K[i] > bw) { bw = K[i + 1] - K[i]; bi = i; }
    const u = (K[bi] + K[bi + 1]) / 2;
    row0 = insertKnot(row0, u); rowFar = insertKnot(rowFar, u); cf = insertKnot(cf, u);
  }
  const p = row0.degree, U = row0.knots, xi = bsGreville(U, p);
  const samples = Array.from({ length: k + 1 }, () => []);
  for (const u of xi) {
    const f = edge.frameAt(u), gu = typeof g === 'function' ? g(u) : g;
    samples[0].push(f.point); samples[1].push(bsScale(f.d1, gu));
    if (k >= 2) samples[2].push(bsScale(f.d2, gu * gu));
    if (k >= 3) samples[3].push(bsScale(f.d3, gu * gu * gu));
  }
  const rows = samples.map((S) => interpAtParams(S, p, xi, U).map((q) => [q[0], q[1], q[2]]));
  return { rows, p, U, xi, farInBasis: rowFar, cfInBasis: cf };
}

function blendSrfTransverseCurvature(srf, edge, s) {
  const u0 = srf.knotsU[srf.degU], u1 = srf.knotsU[srf.knotsU.length - 1 - srf.degU];
  const v0 = srf.knotsV[srf.degV], v1 = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  const alongV = edge === 'u0' || edge === 'u1';
  const u = edge === 'u0' ? u0 : edge === 'u1' ? u1 : u0 + s * (u1 - u0);
  const v = edge === 'v0' ? v0 : edge === 'v1' ? v1 : v0 + s * (v1 - v0);
  const d = blendSrfDerivs2(srf, u, v);
  const T = alongV ? d.Sv : d.Su, X = alongV ? d.Su : d.Sv;
  const XX = alongV ? d.Suu : d.Svv, XT = d.Suv, TT = alongV ? d.Svv : d.Suu;
  const tt = bsDot(T, T), q = tt > 1e-18 ? -bsDot(X, T) / tt : 0;
  const V = bsAdd(X, bsScale(T, q));
  const A = bsAdd(XX, bsAdd(bsScale(XT, 2 * q), bsScale(TT, q * q)));
  const N = bsNorm(bsCross(d.Su, d.Sv)), vv = bsDot(V, V);
  return vv > 1e-18 && bsLen(N) > 0 ? { kn: bsDot(A, N) / vv, N } : null;
}

// Match one internal seam's normal curvature by moving only the third control
// row. Position and tangent plane use the first two rows and therefore cannot
// move. The caller reserves controls belonging to the sheet/free boundaries or
// to the seam solved immediately before this one. Least squares over three
// samples per movable control, so the seam matches between the controls'
// Greville points as well as at them (damped Gauss-Newton; each control moves
// along the surface normal at its own Greville point).
function blendSrfSolveG2Edge(base, baseEdge, target, targetEdge, reversed, first, last) {
  const alongV = baseEdge === 'u0' || baseEdge === 'u1';
  const count = alongV ? base.ctrlNet[0].length : base.ctrlNet.length;
  const knots = alongV ? base.knotsV : base.knotsU, degree = alongV ? base.degV : base.degU;
  const greville = bsGreville(knots, degree);
  const lo = knots[degree], hi = knots[knots.length - 1 - degree];
  const net = base.ctrlNet.map((row) => row.map((point) => point.slice()));
  const at = (index) => {
    if (baseEdge === 'u0') return [2, index];
    if (baseEdge === 'u1') return [net.length - 3, index];
    if (baseEdge === 'v0') return [index, 2];
    return [index, net[0].length - 3];
  };
  const current = () => ({ ...base, ctrlNet: net });
  const sOf = (index) => (hi > lo ? Math.max(0, Math.min(1, (greville[index] - lo) / (hi - lo))) : 0.5);
  const movable = [];
  for (let index = first; index <= Math.min(last, count - 1); index++) movable.push(index);
  if (!movable.length) return base;
  const dirs = movable.map((index) => { const c = blendSrfTransverseCurvature(base, baseEdge, sOf(index)); return c ? c.N : null; });
  // The seam's ends are an extraordinary vertex and a reserved boundary jet,
  // where curvature cannot be matched; samples there would pull the interior.
  const sLo = Math.max(0.08, sOf(Math.max(0, movable[0] - 1))), sHi = Math.min(0.92, sOf(Math.min(count - 1, movable[movable.length - 1] + 1)));
  const M = 3 * movable.length;
  const samples = Array.from({ length: M }, (_, m) => sLo + (sHi - sLo) * (m + 0.5) / M);
  const residual = () => samples.map((sv) => {
    const have = blendSrfTransverseCurvature(current(), baseEdge, sv);
    const want = blendSrfTransverseCurvature(target, targetEdge, reversed ? 1 - sv : sv);
    if (!have || !want) return 0;
    return (bsDot(have.N, want.N) < 0 ? -want.kn : want.kn) - have.kn;
  });
  const n = movable.length;
  let r = residual();
  for (let iter = 0; iter < 6; iter++) {
    const J = Array.from({ length: M }, () => new Array(n).fill(0));
    for (let j = 0; j < n; j++) {
      if (!dirs[j]) continue;
      const [a, b] = at(movable[j]), keep = net[a][b].slice();
      net[a][b] = [...bsAdd(keep.slice(0, 3), dirs[j]), keep[3] ?? 1];
      const r1 = residual();
      net[a][b] = keep;
      for (let m = 0; m < M; m++) J[m][j] = r[m] - r1[m];
    }
    const A = Array.from({ length: n }, () => new Array(n).fill(0)), g = new Array(n).fill(0);
    for (let m = 0; m < M; m++) for (let j = 0; j < n; j++) {
      if (!J[m][j]) continue;
      g[j] += J[m][j] * r[m];
      for (let l = 0; l < n; l++) A[j][l] += J[m][j] * J[m][l];
    }
    const damp = 1e-9 * Math.max(1e-30, ...A.map((row, j) => row[j]));
    for (let j = 0; j < n; j++) A[j][j] += damp;
    const d = bsSolveDense(A, g);
    if (!d) break;
    for (let j = 0; j < n; j++) {
      if (!dirs[j] || !Number.isFinite(d[j])) continue;
      const [a, b] = at(movable[j]), keep = net[a][b];
      net[a][b] = [...bsAdd(keep.slice(0, 3), bsScale(dirs[j], d[j])), keep[3] ?? 1];
    }
    const worstBefore = Math.max(...r.map(Math.abs));
    r = residual();
    const worst = Math.max(...r.map(Math.abs));
    if (worst < 1e-12 || worst > 0.9 * worstBefore) break;
  }
  return { ...base, ctrlNet: net };
}
// Gaussian elimination with partial pivoting; null when singular.
function bsSolveDense(A0, b0) {
  const n = b0.length, A = A0.map((row, i) => [...row, b0[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (!(Math.abs(A[p][c]) > 1e-300)) return null;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = c + 1; r < n; r++) { const f = A[r][c] / A[c][c]; if (f) for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let v = A[r][n]; for (let k = r + 1; k < n; k++) v -= A[r][k] * x[k]; x[r] = v / A[r][r]; }
  return x;
}

// Control points the cap's G2 solve moves along each seam. Refined by knot
// insertion, never degree elevation: an exchange file holds degree 11 at most.
const BLEND_CAP_ROWS_U = 30, BLEND_CAP_ROWS_V = 26;
function blendSrfRefineV(srf, count) {
  let s = srf;
  while (s.ctrlNet[0].length < count) {
    const K = s.knotsV; let bi = -1, bw = 0;
    for (let i = 0; i + 1 < K.length; i++) if (K[i + 1] - K[i] > bw) { bw = K[i + 1] - K[i]; bi = i; }
    const v = (K[bi] + K[bi + 1]) / 2;
    const rows = s.ctrlNet.map((row) => insertKnotOnce({ degree: s.degV, knots: K, ctrlPts: row }, v));
    s = { ...s, knotsV: rows[0].knots, ctrlNet: rows.map((c) => c.ctrlPts) };
  }
  return s;
}
function blendSrfCapG2(faces, N, continuity) {
  // One continuity per sheet, in ring order; a lone number is every sheet's.
  const kAt = (i) => (Array.isArray(continuity) ? continuity[i] : continuity);
  if (Math.max(...Array.from({ length: N }, (_, i) => kAt(i))) < 2) return faces;
  const out = faces.map((f) => blendSrfRefineV(f, BLEND_CAP_ROWS_V));
  // Sheet-midpoint spokes first. Preserve the requested sheet-edge jet at v=0
  // and the position/tangent of the far spoke at v=1.
  for (let i = 0; i < N; i++) {
    const start = 2 * i, end = start + 1, nv = out[start].ctrlNet[0].length;
    out[start] = blendSrfSolveG2Edge(out[start], 'u1', out[end], 'u0', false, Math.max(2, kAt(i)) + 1, nv - 3);
  }
  // Free-midpoint spokes second. Their solve may use the interior of the free
  // side, but not its boundary point or the three rows that carry the G2 seam
  // just established at u=1.
  for (let i = 0; i < N; i++) {
    const end = 2 * i + 1, nextStart = 2 * ((i + 1) % N), nu = out[nextStart].ctrlNet.length;
    out[nextStart] = blendSrfSolveG2Edge(out[nextStart], 'v1', out[end], 'v1', true, 1, nu - 4);
  }
  return out;
}

function blendSrfGuidedCap(access, arc, order, bulge, k, nrm, flips = [], square, ks = null, waist = null) {
  const N = order.length;
  // Each sheet's own continuity: 0 is a crease (the sheet read as running
  // straight on to the center), 1–3 its tangent, curvature, and curvature's rate.
  const kOf = (kx) => (ks && ks[kx] != null ? ks[kx] : k);
  const kRows = (kx) => Math.max(1, kOf(kx));
  access = access.slice();
  const pt = (kx, t) => access[kx].point(t);
  const frameAt = (kx, t, onward = false) => access[kx].frame(t, onward);
  // Every edge runs the same way round the ring as the ring order; a corner is
  // at t = 0 or 1, the midpoint at t = 0.5. Read from the rotation about the
  // center, not from which end lies nearer the last edge's end: wide sheets
  // with a narrow gap put the wrong end nearer, and that edge ran backwards,
  // twisting its quads.
  const rev = new Array(N).fill(false);
  {
    const c0 = bsScale(order.reduce((acc, kx) => bsAdd(acc, pt(kx, 0.5)), [0, 0, 0]), 1 / N);
    const turn = (a, b) => bsDot(bsCross(bsSub(a, c0), bsSub(b, c0)), nrm);
    const sense = Math.sign(turn(pt(order[0], 0.5), pt(order[1], 0.5))) || 1;
    for (const kx of order) rev[kx] = turn(pt(kx, 0), pt(kx, 1)) * sense < 0;
  }
  for (let kx = 0; kx < N; kx++) if (flips[kx]) rev[kx] = !rev[kx];
  const tOf = (kx, chainT) => (rev[kx] ? 1 - chainT : chainT);
  const corners = [];
  for (const kx of order) corners.push(pt(kx, tOf(kx, 0)), pt(kx, tOf(kx, 1)));
  const cornerMean = bsScale(corners.reduce((a, q) => bsAdd(a, q), [0, 0, 0]), 1 / corners.length);
  const reverseCrv = (c) => ({ ...c, ctrlPts: [...c.ctrlPts].reverse(), knots: c.knots.map((t) => c.knots[0] + c.knots[c.knots.length - 1] - t).reverse() });
  const zero = [0, 0, 0];
  const blend = (a, b, o) => { const r = geometricBlend(a, b, o); return r.ok ? r.crv : null; };
  const mids = order.map((kx) => frameAt(kx, 0.5));
  // The cap belongs where the sheets point, not at the arithmetic mean of
  // their corners. Fit the intersection of their inward midpoint tangents in
  // the junction plane. A long sheet then has no extra vote merely because
  // its boundary sits farther away. Parallel/ill-conditioned lines retain the
  // bounded corner mean rather than manufacturing a remote center.
  let C = cornerMean;
  const seed = mids.map((m) => {
    const d = bsSub(m.d1, bsScale(nrm, bsDot(m.d1, nrm)));
    return bsLen(d) > 1e-12 ? bsNorm(d) : null;
  }).find(Boolean);
  if (seed) {
    const ex = seed, ey = bsCross(nrm, ex);
    let a = 0, b = 0, c = 0, r0 = 0, r1 = 0;
    for (const m of mids) {
      const d3 = bsSub(m.d1, bsScale(nrm, bsDot(m.d1, nrm))), dl = bsLen(d3);
      if (!(dl > 1e-12)) continue;
      const dx = bsDot(d3, ex) / dl, dy = bsDot(d3, ey) / dl;
      const nx = -dy, ny = dx, q = bsSub(m.point, cornerMean);
      const px = bsDot(q, ex), py = bsDot(q, ey), rhs = nx * px + ny * py;
      a += nx * nx; b += nx * ny; c += ny * ny; r0 += nx * rhs; r1 += ny * rhs;
    }
    const det = a * c - b * b;
    if (Math.abs(det) > 1e-10) {
      const x = (c * r0 - b * r1) / det, y = (a * r1 - b * r0) / det;
      const candidate = bsAdd(cornerMean, bsAdd(bsScale(ex, x), bsScale(ey, y)));
      const radius = Math.max(...corners.map((q) => bsLen(bsSub(q, cornerMean))));
      if (Number.isFinite(x) && Number.isFinite(y) && bsLen(bsSub(candidate, cornerMean)) <= 2 * Math.max(radius, 1e-9)) C = candidate;
    }
  }
  // The center rises with the sheets. Held in the junction plane, sheets that
  // slope away from it leave a dimple: each spoke climbs off its sheet's
  // tangent and falls back to C (2.5 at 19 degrees, 4.5 at 35, on 40 mm
  // sheets). C moves along the normal by half of each sheet's tangent-line
  // gain to C, averaged: the height a cubic leaving the edge's slope reaches
  // arriving level, without overshoot. A slope is read at most 60 degrees.
  {
    let lift = 0, n = 0;
    for (const m of mids) {
      const t = bsNorm(m.d1), along = bsDot(t, nrm), flat = bsSub(t, bsScale(nrm, along)), fl = bsLen(flat);
      if (!(fl > 1e-9)) continue;
      const q = bsSub(C, m.point), dist = bsLen(bsSub(q, bsScale(nrm, bsDot(q, nrm))));
      const slope = Math.max(-Math.tan(Math.PI / 3), Math.min(Math.tan(Math.PI / 3), along / fl)); // a standing sheet reads as 60 degrees, not a spike
      lift += 0.5 * slope * dist - bsDot(q, nrm); n++;
    }
    if (n) C = bsAdd(C, bsScale(nrm, lift / n));
  }
  // G0: the sheet's cross direction becomes its chord to the center, at the
  // sheet's own speed, with no curvature — the cap meets it at an angle.
  for (const kx of order) {
    if (kOf(kx) !== 0) continue;
    const own = access[kx].frame;
    access[kx] = { ...access[kx], frame: (t, onward = false) => {
      const f = own(t, onward), toC = bsNorm(bsSub(C, f.point)), sp = bsLen(f.d1);
      return { point: f.point, d1: bsScale(toC, onward ? -sp : sp), d2: [0, 0, 0], d3: [0, 0, 0] };
    } };
  }
  for (let i = 0; i < N; i++) mids[i] = frameAt(order[i], 0.5);
  const g = new Array(N);
  order.forEach((kx, i) => { g[kx] = bulge * bsLen(bsSub(C, mids[i].point)) / bsLen(mids[i].d1); });
  // A free edge joins two side edges the way a fillet would: it leaves one
  // side edge and meets the next tangentially, turning one way. Its handles
  // start at 2g times the sheet's cross speed and are sized per corner (each
  // quad corner takes the speed its half of the edge has, cornerG below).
  //  - The side lines meet ahead of both corners: each handle is held to 0.8
  //    of the distance from its corner to the meeting point, in the junction
  //    plane, so the control polygon turns one way and so does the edge
  //    (past it the edge hooks back on itself: two curvature reversals, a
  //    0.3 mm radius at Bulge 1.4 on 40 mm sheets). Bulge above 1 cannot pass it.
  //  - They meet behind a corner (a sheet set far closer in than its
  //    neighbor), or at one: no edge leaving both side edges tangentially
  //    can turn one way, since the chord lies outside the turn (at a corner,
  //    only with no radius at all). The edge takes the one
  //    reversal it must, with the handles that give its tightest bend the
  //    largest radius.
  const freeHandles = new Array(N);
  const flatOf = (v) => bsSub(v, bsScale(nrm, bsDot(v, nrm)));
  const bendOf = (crv) => { // reversals of the edge's turn about nrm, and its tightest radius
    const K = crv.knots, lo = K[0], hi = K[K.length - 1], ks = [];
    for (let j = 0; j <= 96; j++) { const [, d1, d2] = rationalCurveDerivs(crv, lo + (hi - lo) * j / 96, 2), sp = bsLen(d1); ks.push(sp > 0 ? bsDot(bsCross(d1, d2), nrm) / (sp * sp * sp) : 0); }
    const kmax = Math.max(...ks.map(Math.abs));
    let flips = 0, prev = 0; for (const x of ks) { if (!(Math.abs(x) > 1e-3 * kmax)) continue; const sg = Math.sign(x); if (prev && sg !== prev) flips++; prev = sg; }
    return { flips, radius: kmax > 0 ? 1 / kmax : Infinity };
  };
  for (let i = 0; i < N; i++) {
    const kx = order[i], kn = order[(i + 1) % N];
    const fe = frameAt(kx, tOf(kx, 1)), fs = frameAt(kn, tOf(kn, 0), true);
    const kk = Math.max(kRows(kx), kRows(kn)), sp = bsLen(fe.d1), sq = bsLen(fs.d1);
    let mA = 2 * g[kx] * sp, mB = 2 * g[kn] * sq;
    const uA = bsNorm(flatOf(fe.d1)), uB = bsNorm(flatOf(fs.d1)), w = flatOf(bsSub(fs.point, fe.point));
    const cr = bsDot(bsCross(uA, uB), nrm), chord = bsLen(bsSub(fs.point, fe.point));
    if (Math.abs(cr) > 1e-6) {
      const sA = bsDot(bsCross(w, uB), nrm) / cr, tB = bsDot(bsCross(w, uA), nrm) / cr; // X = A + sA uA = B + tB uB
      if (sA > 0.02 * chord && -tB > 0.02 * chord) {
        mA = Math.max(1e-3 * chord, Math.min(mA, 1.6 * Math.min(bulge, 1) * sA));
        mB = Math.max(1e-3 * chord, Math.min(mB, 1.6 * Math.min(bulge, 1) * -tB));
      } else {
        let best = null;
        for (const fa of [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.05, 1.35]) for (const fb of [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.05, 1.35]) {
          const c = blend({ point: fe.point, d1: fe.d1, d2: fe.d2, d3: fe.d3 }, { point: fs.point, d1: fs.d1, d2: fs.d2, d3: fs.d3 }, { continuity: kk, startMagnitude: fa * chord, endMagnitude: fb * chord });
          if (!c) continue;
          const b = bendOf(c), score = (b.flips <= 1 ? 0 : -1e9 * b.flips) + b.radius;
          if (!best || score > best.score) best = { score, mA: fa * chord, mB: fb * chord };
        }
        if (best) { mA = best.mA; mB = best.mB; }
      }
    }
    freeHandles[i] = [mA, mB];
  }
  // The sheet's cross speed is held to its slower corner's: a corner eased
  // from a fast midpoint down to a tight fillet folds the quad beside it.
  for (let i = 0; i < N; i++) { const kx = order[i], kn = order[(i + 1) % N]; g[kx] = Math.min(g[kx], freeHandles[i][0] / (2 * bsLen(frameAt(kx, tOf(kx, 1)).d1))); g[kn] = Math.min(g[kn], freeHandles[i][1] / (2 * bsLen(frameAt(kn, tOf(kn, 0), true).d1))); }
  // Square to the plane. Where every corner lies on one plane, each sheet
  // meets it square (its edge's tangent at the corner along the plane's
  // normal) and leaves its corners along it, the free edges lie in that plane
  // and the cap is built to meet it at 90 degrees: the spoke from each free
  // edge's midpoint leaves along the normal, and every cross field along a
  // free edge then runs between that and the sheets' own vertical tangents. A
  // mirror of the cap in the plane joins it G1, and G2 by symmetry.
  let squareNormal = null, squareFound = false;
  {
    const cm = cornerMean;
    let cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const q of corners) { const d = bsSub(q, cm); for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) cov[a][b] += d[a] * d[b]; }
    // The plane's normal: the smallest eigenvector, by inverse iteration from the junction normal.
    let nv = nrm.slice();
    const shift = 1e-12 * (cov[0][0] + cov[1][1] + cov[2][2] + 1);
    const Mx = cov.map((row, a) => row.map((x, b) => x + (a === b ? shift : 0)));
    const solve3 = (A, bv) => { const det = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]); const D = det(A); if (!(Math.abs(D) > 0)) return null; return [0, 1, 2].map((c) => det(A.map((row, i) => row.map((x, j) => (j === c ? bv[i] : x)))) / D); };
    for (let it = 0; it < 30; it++) { const x = solve3(Mx, nv); if (!x || !(bsLen(x) > 0)) break; nv = bsNorm(x); }
    const size = Math.max(...corners.map((q) => bsLen(bsSub(q, cm))), 1e-9);
    const flatEnough = corners.every((q) => Math.abs(bsDot(bsSub(q, cm), nv)) <= 1e-4 * size);
    const cos1 = Math.cos(Math.PI / 180);
    let meets = flatEnough;
    for (const kx of order) {
      if (!meets) break;
      for (const tt of [0, 1]) {
        const ea = access[kx], crv = ea.alongCrv, lo = crv.knots[crv.degree], hi = crv.knots[crv.knots.length - 1 - crv.degree];
        const tan = rationalCurveDerivs(crv, lo + tOf(kx, tt) * (hi - lo), 1)[1];
        const f = frameAt(kx, tOf(kx, tt));
        if (!(bsLen(tan) > 0) || Math.abs(bsDot(bsNorm(tan), nv)) < cos1 || Math.abs(bsDot(bsNorm(f.d1), nv)) > 1 - cos1) { meets = false; break; }
      }
    }
    if (meets) {
      squareFound = true;
      if (bsDot(bsSub(C, cm), nv) < 0) nv = bsScale(nv, -1);
      if (square !== false) squareNormal = nv;
    }
  }
  // The fan at C: 2N directions in the junction plane, in ring order.
  const dC = bsSub(mids[0].point, C);
  const e1 = bsNorm(bsSub(dC, bsScale(nrm, bsDot(dC, nrm)))), e2 = bsCross(nrm, e1);
  const angleOf = (q) => { const d = bsSub(q, C); return Math.atan2(bsDot(d, e2), bsDot(d, e1)); };
  // The ring's sense about C, from every consecutive pair: one sheet straight
  // across from the first reads as +180 or -180 and cannot decide it alone.
  let turnSum = 0;
  for (let i = 0; i < N; i++) turnSum += Math.sin(angleOf(mids[(i + 1) % N].point) - angleOf(mids[i].point));
  const sense = turnSum < 0 ? -1 : 1;
  // Each sheet's spoke arrives along its own line to C, and each free spoke
  // through the window between the two corners beside it, as seen from C. A
  // fan pressed regular under uneven arms turns a spoke as much as 30 degrees
  // off its sheet (a T), and a free spoke on the bisector of the sheets
  // passes behind the corner of a sheet set close in; either folds the quads.
  //
  // G1 across spoke j at C needs L_{j-1}d_{j-1} + L_{j+1}d_{j+1} = α_j L_j d_j.
  // With x_i, y_i the angles from sheet spoke i to free spoke i and on to
  // sheet spoke i+1, the speeds solve it round the ring exactly when
  // Π sin x_i = Π sin y_i (the even and odd chains give the same condition):
  // the free spokes are moved off their window centers by one shared
  // multiplier until it holds. Even arms give the regular fan, α = 2cos(π/N).
  const TAU = 2 * Math.PI;
  const rel = (q, base) => { const g = sense * (angleOf(q) - base); return g - TAU * Math.floor(g / TAU); };
  const theta = mids.map((m) => angleOf(m.point));
  let gaps = theta.map((t, i) => rel(mids[(i + 1) % N].point, t));
  if (!(Math.abs(gaps.reduce((a, b) => a + b, 0) - TAU) < 1e-6 && gaps.every((x) => x > 1e-3))) gaps = gaps.map(() => TAU / N);
  const xWant = gaps.map((gi, i) => {
    const kx = order[i], kn = order[(i + 1) % N];
    const a = rel(pt(kx, tOf(kx, 1)), theta[i]), b = rel(pt(kn, tOf(kn, 0)), theta[i]);
    return a < b && a > 0 && b < gi ? (a + b) / 2 : gi / 2;
  });
  const xAt = (lam) => xWant.map((x, i) => Math.max(0.05 * gaps[i], Math.min(0.95 * gaps[i], x + lam * gaps[i] * Math.sign(Math.sin(gaps[i])))));
  const balance = (xs) => xs.reduce((acc, x, i) => acc + Math.log(Math.sin(x)) - Math.log(Math.sin(gaps[i] - x)), 0);
  let xs = xAt(0);
  if (Math.abs(balance(xs)) > 1e-12) {
    let lo = -1, hi = 1;
    for (let it = 0; it < 80; it++) { const m = (lo + hi) / 2; if (balance(xAt(m)) < 0) lo = m; else hi = m; }
    xs = xAt((lo + hi) / 2);
  }
  const ys = xs.map((x, i) => gaps[i] - x);
  const phi = new Array(2 * N);
  for (let i = 0, a = 0; i < N; a += gaps[i], i++) { phi[2 * i] = sense * a; phi[2 * i + 1] = sense * (a + xs[i]); }
  const dir = (j) => bsAdd(bsScale(e1, Math.cos(phi[j])), bsScale(e2, Math.sin(phi[j])));
  // Free edges at 2g, each split at its point M nearest C (its parameter
  // midpoint on a symmetric pair). A free edge from a sheet set close in to
  // one set far out turns back on itself; split at the parameter midpoint,
  // the half beside the near sheet ran out past its own corner and the quads
  // either side folded over it. A half on [0,1] leaves its corner at 2g times
  // its share of the edge's parameter, so the sheet's cross speed runs from g
  // at its midpoint to that at each corner (cornerG, smoothstep along the
  // half) and every quad corner stays compatible.
  const subOn = (crv, a, b) => rescaleCurveDomain(extractSubCurve(crv, a, b), 0, 1);
  const free = new Array(N), M = new Array(N), Mt = new Array(N), split = new Array(N), cornerG = order.map(() => [0, 0]);
  for (let i = 0; i < N; i++) {
    const kx = order[i], kn = order[(i + 1) % N];
    const fe = frameAt(kx, tOf(kx, 1)), fs = frameAt(kn, tOf(kn, 0), true);
    // A waist shared with another cap takes that cap's free edge through these
    // two corners, run this cap's way, so the two meet along one curve.
    const shared = waist && waist.rails ? waist.rails.find((c) => {
      const a = curvePoint(c, c.knots[0]), b = curvePoint(c, c.knots[c.knots.length - 1]), tol = 1e-6 * Math.max(1, bsLen(bsSub(fs.point, fe.point)));
      return (bsLen(bsSub(a, fe.point)) < tol && bsLen(bsSub(b, fs.point)) < tol) || (bsLen(bsSub(b, fe.point)) < tol && bsLen(bsSub(a, fs.point)) < tol);
    }) : null;
    if (shared) {
      const a = curvePoint(shared, shared.knots[0]);
      free[i] = bsLen(bsSub(a, fe.point)) < bsLen(bsSub(a, fs.point)) ? shared : reverseCrv(shared);
    } else free[i] = blend({ point: fe.point, d1: fe.d1, d2: fe.d2, d3: fe.d3 }, { point: fs.point, d1: fs.d1, d2: fs.d2, d3: fs.d3 },
      { continuity: Math.max(kRows(kx), kRows(kn)), startMagnitude: freeHandles[i][0], endMagnitude: freeHandles[i][1] });
    if (!free[i]) return { ok: false, reason: `no edge could join the corners of edges ${kx + 1} and ${kn + 1}` };
    const crv = free[i], dC = (t) => bsLen(bsSub(curvePoint(crv, t), C));
    let tm = 0.5, dm = dC(0.5);
    for (let k = 0; k <= 64; k++) { const t = 0.15 + 0.7 * k / 64, d = dC(t); if (d < dm - 1e-9 * dm) { dm = d; tm = t; } }
    if (tm !== 0.5) { let lo = Math.max(0.15, tm - 0.7 / 64), hi = Math.min(0.85, tm + 0.7 / 64); for (let it = 0; it < 80; it++) { const a = lo + (hi - lo) / 3, b = hi - (hi - lo) / 3; if (dC(a) < dC(b)) hi = b; else lo = a; } tm = (lo + hi) / 2; }
    split[i] = tm;
    const [Mp, T] = rationalCurveDerivs(crv, tm, 1);
    M[i] = Mp; Mt[i] = T;
    // The edge's own speed at each corner, over the sheet's frame speed, as a share of its half.
    cornerG[i][1] = tm * bsLen(crvDeriv(crv, 0, 1)) / bsLen(fe.d1);
    cornerG[(i + 1) % N][0] = (1 - tm) * bsLen(crvDeriv(crv, 1, 1)) / bsLen(fs.d1);
  }
  // Relative speeds round the two chains, then one scale for each: the mean
  // chord, held so no spoke arrives faster than its own chord to C — a
  // quintic arriving at speed s with no second derivative puts its last two
  // controls 0.2s and 0.4s back, and a short spoke at the fan's mean speed
  // doubles back past its start.
  const rE = [1], rM = [1];
  for (let i = 0; i + 1 < N; i++) rE.push(rE[i] * Math.sin(xs[i]) / Math.sin(ys[i]));
  for (let i = 1; i < N; i++) rM.push(rM[i - 1] * Math.sin(ys[i - 1]) / Math.sin(xs[i]));
  let Lmean = 0; for (let i = 0; i < N; i++) Lmean += bsLen(bsSub(C, mids[i].point)) + bsLen(bsSub(C, M[i])); Lmean /= 2 * N;
  const mean = (v) => v.reduce((acc, x) => acc + x, 0) / v.length;
  const sE = Math.min(Lmean / mean(rE), ...mids.map((m, i) => bsLen(bsSub(C, m.point)) / rE[i]));
  const sM = Math.min(Lmean / mean(rM), ...M.map((q, i) => bsLen(bsSub(C, q)) / rM[i]));
  const LE = rE.map((r) => r * sE), LM = rM.map((r) => r * sM);
  const alphaE = LE.map((l, i) => { const ip = (i + N - 1) % N; return (LM[i] * Math.cos(xs[i]) + LM[ip] * Math.cos(ys[ip])) / l; });
  const alphaM = LM.map((l, i) => (LE[(i + 1) % N] * Math.cos(ys[i]) + LE[i] * Math.cos(xs[i])) / l);
  const speed = (j) => (j % 2 ? LM[(j - 1) / 2] : LE[j / 2]);
  const arrive = (j) => bsScale(dir(j), -speed(j)); // the tangent a spoke arrives at C with
  const spokeE = new Array(N), spokeM = new Array(N);
  for (let i = 0; i < N; i++) {
    const kx = order[i], f = mids[i];
    spokeE[i] = blend({ point: f.point, d1: f.d1, d2: f.d2, d3: f.d3 }, { point: C, d1: arrive(2 * i), d2: zero, d3: zero },
      { continuity: Math.max(2, kRows(kx)), startMagnitude: g[kx] * bsLen(f.d1), endMagnitude: LE[i] });
    if (!spokeE[i]) return { ok: false, reason: `no spoke could reach the center from edge ${kx + 1}` };
    let n = bsSub(C, M[i]); const T = bsNorm(Mt[i]); n = bsSub(n, bsScale(T, bsDot(n, T)));
    if (!(bsLen(n) > 1e-9)) n = bsSub(C, M[i]);
    if (squareNormal) n = squareNormal;
    // At a waist each free edge is left along the direction the two sheets' own
    // edges run at its corners (their rims crossing the waist), on this cap's
    // side; the other cap leaves it the opposite way, so the two are tangent.
    if (waist) {
      const kn = order[(i + 1) % N];
      const edgeDir = (kk, tt) => { const ea = access[kk], crv = ea.alongCrv, lo = crv.knots[crv.degree], hi = crv.knots[crv.knots.length - 1 - crv.degree]; return bsNorm(rationalCurveDerivs(crv, lo + tOf(kk, tt) * (hi - lo), 1)[1]); };
      const toCap = (d, at) => (bsDot(d, bsSub(C, at)) < 0 ? bsScale(d, -1) : d);
      const dA = toCap(edgeDir(kx, 1), frameAt(kx, tOf(kx, 1)).point), dB = toCap(edgeDir(kn, 0), frameAt(kn, tOf(kn, 0)).point);
      const w = bsNorm(bsAdd(dA, dB));
      if (bsLen(w) > 0) n = w;
    }
    spokeM[i] = blend({ point: M[i], d1: n, d2: zero, d3: zero }, { point: C, d1: arrive(2 * i + 1), d2: zero, d3: zero },
      { continuity: Math.max(2, kRows(kx), kRows(order[(i + 1) % N])), startMagnitude: bsLen(bsSub(C, M[i])), endMagnitude: LM[i] });
    if (!spokeM[i]) return { ok: false, reason: `no spoke could reach the center from the edge between ${kx + 1} and ${order[(i + 1) % N] + 1}` };
  }
  // Halves: a sheet edge from chain t a to b as a curve on [0,1] with its frames; a free edge's halves.
  const half = (kx, a, b) => {
    const ea = access[kx], crv = ea.alongCrv, lo = crv.knots[crv.degree], hi = crv.knots[crv.knots.length - 1 - crv.degree];
    const ta = tOf(kx, a), tb = tOf(kx, b);
    let sub = extractSubCurve(crv, lo + Math.min(ta, tb) * (hi - lo), lo + Math.max(ta, tb) * (hi - lo));
    if (tb < ta) sub = reverseCrv(sub);
    sub = rescaleCurveDomain(sub, 0, 1);
    return { crv: sub, frameAt: (u) => ea.frame(ta + u * (tb - ta), false) };
  };
  const endTangent = (crv, atEnd) => crvDeriv(crv, atEnd ? crv.knots[crv.knots.length - 1] : crv.knots[0], 1);
  // Pass 1: the quads' rows. End quad of sheet i: mid → end corner; start quad: start corner → mid.
  const quads = [];
  for (let i = 0; i < N; i++) {
    const kx = order[i], ip = (i + N - 1) % N;
    const e_i = endTangent(spokeE[i], true), m_i = endTangent(spokeM[i], true), m_ip = endTangent(spokeM[ip], true);
    const firstHalf = subOn(free[i], 0, split[i]), secondHalfRev = reverseCrv(subOn(free[ip], split[ip], 1));
    // The end quad's far field, C → M_i: from the spoke's tangent at C to the free half's tangent at M, zero twists.
    const cfEnd = bzHermite([e_i, zero], [endTangent(firstHalf, true), zero]);
    const kq = kRows(kx);
    const gS = cornerG[i][0], gE = cornerG[i][1];
    const start = { kind: 'start', kx, i, edge: half(kx, 0, 0.5), side0: secondHalfRev, side1: spokeE[i], far: spokeM[ip], g: gS === g[kx] ? g[kx] : (u) => g[kx] + (gS - g[kx]) * (1 - u) * (1 - u) * (1 + 2 * u), k: kq };
    const end = { kind: 'end', kx, i, edge: half(kx, 0.5, 1), side0: spokeE[i], side1: firstHalf, far: reverseCrv(spokeM[i]), g: gE === g[kx] ? g[kx] : (u) => g[kx] + (gE - g[kx]) * u * u * (3 - 2 * u), k: kq, cf: cfEnd, e_i, m_i, m_ip };
    quads.push(start, end);
  }
  // The start quad's far field is the previous end quad's, reversed, plus β·h·M' (β1 = α1).
  for (let i = 0; i < N; i++) {
    const start = quads[2 * i], ip = (i + N - 1) % N, endPrev = quads[2 * ip + 1];
    const revCf = [...endPrev.cf].reverse().map((v) => bsScale(v, -1));
    // At M the two halves of the free edge may run at different speeds; the
    // reversed field is scaled to the start quad's own half there, by a
    // factor falling to 1 at C, which keeps it in the plane of the end
    // quad's field and the spoke (G1) and leaves the fan untouched.
    const w = (1 - split[ip]) / split[ip];
    start.cf = bzAdd(bzScalarTimes([w, w, 1, 1, 1, 1], revCf), bzScalarTimes(H_RISE, bzScale(bzHodo(bzOf(spokeM[ip])), alphaM[ip])));
  }
  for (const q of quads) {
    try { q.uRows = blendSrfQuadRows(q.edge, q.far, q.cf, q.g, q.k); }
    catch (e) { return { ok: false, reason: `the web would not fill at edge ${q.kx + 1}: ${e.message}` }; }
  }
  // Pass 2: the side fields. A_i along spokeE_i (the end quad's u = 0), shared with the start quad as A + α·h·E'.
  const sheetConds = (q, atU1) => { // the ribbon's own tangent and cross-field derivatives at a sheet corner
    const { rows, p, U } = q.uRows, D = atU1 ? bsRowD1 : bsRowD0;
    return rows.map((r) => D(r, U, p));
  };
  for (let i = 0; i < N; i++) {
    const start = quads[2 * i], end = quads[2 * i + 1];
    const A = bzHermite(sheetConds(end, false), [bsScale(end.m_i, -1), zero]);
    end.fieldA = A;
    end.fieldB = bzHermite(sheetConds(end, true), [endTangent(end.far, true), zero]);
    end.fieldF = end.cf;
    start.fieldB = bzAdd(A, bzScalarTimes(H_RISE, bzScale(bzHodo(bzOf(spokeE[i])), alphaE[i])));
    start.fieldA = bzHermite(sheetConds(start, false), [endTangent(start.far, false), zero]);
    start.fieldF = start.cf;
  }
  let faces = [];
  for (const q of quads) {
    try {
      faces.push(blendSrfBooleanQuad({ k: q.k, edge: q.edge, side0: bzOf(q.side0), side1: bzOf(q.side1), far: { inBasis: q.uRows.farInBasis }, fieldA: q.fieldA, fieldB: q.fieldB, fieldF: { inBasis: q.uRows.cfInBasis }, g: q.g, uRows: q.uRows }));
    } catch (e) { return { ok: false, reason: `the web would not fill at edge ${q.kx + 1}: ${e.message}` }; }
  }
  faces = blendSrfCapG2(faces, N, order.map(kRows));
  return { ok: true, faces, free, rev, C, M, squareFound, square: !!squareNormal };
}

export const JUNCTION_REACH = 0.25;

// A closed edge: the rim of a tube.
function blendSrfEdgeClosed(ea) {
  const a = ea.point(0), b = ea.point(1), m = ea.point(0.5);
  return bsLen(bsSub(a, b)) <= 1e-6 * Math.max(bsLen(bsSub(a, m)), 1e-9);
}

// A piece of a surface between two parameters of one direction, exact: every
// row of the net cut as a curve. `a > b` on a closed direction takes the
// piece that runs through the seam.
function blendSrfSubSurface(srf, dir, a, b) {
  const rows = dir === 'v'
    ? srf.ctrlNet.map((row) => ({ degree: srf.degV, knots: srf.knotsV.slice(), ctrlPts: row.map((p) => p.slice()) }))
    : srf.ctrlNet[0].map((_, j) => ({ degree: srf.degU, knots: srf.knotsU.slice(), ctrlPts: srf.ctrlNet.map((r) => r[j].slice()) }));
  const K = rows[0].knots, lo = K[0], hi = K[K.length - 1];
  const cut = rows.map((c) => {
    if (a < b) return rescaleCurveDomain(extractSubCurve(c, a, b), 0, 1);
    const A = extractSubCurve(c, a, hi), B = rescaleCurveDomain(extractSubCurve(c, lo, b), hi, hi + (b - lo));
    return rescaleCurveDomain(concatTwoC0(A, B, c.degree), 0, 1);
  });
  if (dir === 'v') return { ...srf, degV: cut[0].degree, knotsV: cut[0].knots.slice(), ctrlNet: cut.map((c) => c.ctrlPts) };
  return { ...srf, degU: cut[0].degree, knotsU: cut[0].knots.slice(), ctrlNet: cut[0].ctrlPts.map((_, i) => cut.map((c) => c.ctrlPts[i])) };
}

// Tubes meet: every edge is a closed rim. The tubes' axes share a plane; each
// tube is cut in two where its rim crosses that plane, the upper halves take
// one cap and the lower halves another, and both meet the plane square, so
// they share their free edges and join smoothly round the waist. A tube whose
// axis leaves the others' plane has no such waist and is refused.
function blendSurfaceTubeJunction(parts, access, opts) {
  const N = parts.length;
  const S = 256;
  const rims = access.map((ea) => Array.from({ length: S }, (_, i) => ea.point(i / S)));
  const centres = rims.map((pts) => bsScale(pts.reduce((acc, q) => bsAdd(acc, q), [0, 0, 0]), 1 / S));
  const axes = access.map((ea) => { let d = [0, 0, 0]; for (let i = 0; i < 16; i++) d = bsAdd(d, bsNorm(ea.frame(i / 16, false).d1)); return bsNorm(d); });
  const size = Math.max(...rims.map((pts, k) => Math.max(...pts.map((q) => bsLen(bsSub(q, centres[k]))))), 1e-9);
  // The plane through every center and along every axis: the least-squares
  // normal of the centers and a point one rim-size along each axis.
  const pts = [...centres, ...centres.map((c, k) => bsAdd(c, bsScale(axes[k], size)))];
  const mean = bsScale(pts.reduce((acc, q) => bsAdd(acc, q), [0, 0, 0]), 1 / pts.length);
  const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const q of pts) { const d = bsSub(q, mean); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] += d[i] * d[j]; }
  let nrm = bsNorm(bsCross(axes[0], axes[1 % N]));
  if (!(bsLen(nrm) > 0.5)) nrm = bsNorm(bsCross(axes[0], bsSub(centres[1 % N], centres[0])));
  if (!(bsLen(nrm) > 0.5)) return { ok: false, reason: 'the tubes lie on one line — there is no plane for them to meet on' };
  const det3 = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const tr = cov[0][0] + cov[1][1] + cov[2][2];
  const A = cov.map((row, i) => row.map((x, j) => x + (i === j ? 1e-12 * (tr + 1) : 0)));
  for (let it = 0; it < 40; it++) {
    const D = det3(A); if (!(Math.abs(D) > 0)) break;
    const x = [0, 1, 2].map((c) => det3(A.map((row, i) => row.map((v, j) => (j === c ? nrm[i] : v)))) / D);
    if (!(bsLen(x) > 0)) break; nrm = bsNorm(x);
  }
  // Each tube is cut by the plane through its own axis that lies nearest the
  // junction plane, so a tube raised or turned out of the others' plane still
  // has an upper and a lower half; only a tube standing across the plane
  // (its axis along the plane's normal) has no such cut.
  const cutNormal = axes.map((a) => { const t = bsSub(nrm, bsScale(a, bsDot(nrm, a))); return bsLen(t) > 0.2 ? bsNorm(t) : null; });
  for (let k = 0; k < N; k++) if (!cutNormal[k]) return { ok: false, reason: `tube ${k + 1} stands across the others' plane — turn it toward them` };
  // Each rim's two crossings of the plane, by bisection on its own parameter.
  const halves = { up: [], down: [] };
  for (let k = 0; k < N; k++) {
    // Cyclic, with values within a hair of the plane read as on it: a rim
    // that starts on the plane crosses it at its own seam.
    const ea = access[k], eps = 1e-9 * size, f = (t) => { const d = bsDot(bsSub(ea.point(t), centres[k]), cutNormal[k]); return Math.abs(d) < eps ? 0 : d; };
    const g = Array.from({ length: S }, (_, i) => f(i / S));
    const cross = [];
    for (let i = 0; i < S; i++) {
      const j = (i + 1) % S;
      if (g[i] === 0) { if (g[(i + S - 1) % S] * g[j] < 0 || (g[(i + S - 1) % S] !== 0 && g[j] !== 0)) cross.push(i / S); continue; }
      if (g[j] === 0 || g[i] * g[j] > 0) continue;
      let a = i / S, b = (i + 1) / S, fa = g[i];
      for (let it = 0; it < 60; it++) { const m = (a + b) / 2, fm = f(m); if (fa * fm <= 0) b = m; else { a = m; fa = fm; } }
      cross.push((a + b) / 2);
    }
    cross.sort((x, y) => x - y);
    if (cross.length !== 2) return { ok: false, reason: `the rim of tube ${k + 1} crosses the tubes' plane ${cross.length} times, not twice` };
    const { srf, edge } = parts[k];
    const dir = edge === 'u0' || edge === 'u1' ? 'v' : 'u';
    const K = dir === 'v' ? srf.knotsV : srf.knotsU, deg = dir === 'v' ? srf.degV : srf.degU;
    const lo = K[deg], hi = K[K.length - 1 - deg], P = (t) => lo + t * (hi - lo);
    const [t1, t2] = cross;
    const inner = blendSrfSubSurface(srf, dir, P(t1), P(t2));
    const outer = t1 === 0 ? blendSrfSubSurface(srf, dir, P(t2), hi) : blendSrfSubSurface(srf, dir, P(t2), P(t1));
    const up = f((t1 + t2) / 2) > 0;
    halves.up.push({ srf: up ? inner : outer, edge });
    halves.down.push({ srf: up ? outer : inner, edge });
  }
  const capOpts = { ...opts, square: false, waist: {} };
  const top = blendSurfaceJunction(halves.up, capOpts);
  if (!top.ok) return top;
  const bottom = blendSurfaceJunction(halves.down, { ...capOpts, waist: { rails: top.rails } });
  if (!bottom.ok) return bottom;
  return {
    ok: true, arms: [], order: top.order, reversed: top.reversed, ports: [], rails: [...top.rails, ...bottom.rails],
    centre: [...top.centre, ...bottom.centre], centreTrims: [...(top.centreTrims || top.centre.map(() => null)), ...(bottom.centreTrims || bottom.centre.map(() => null))],
    spine: top.spine, faceUp: true, tube: true,
    squareFound: false, square: !!(top.square && bottom.square), halves,
    patched: top.patched && bottom.patched ? { edgeGap: Math.max(top.patched.edgeGap, bottom.patched.edgeGap), edgeAngle: Math.max(top.patched.edgeAngle, bottom.patched.edgeAngle), trimGap: top.patched.trimGap == null || bottom.patched.trimGap == null ? null : Math.max(top.patched.trimGap, bottom.patched.trimGap) } : null,
  };
}

/**
 * @param parts [{srf, edge}] three or more
 * @param opts continuity 1|2|3, bulge (at the edges, default 1), reach (0.05..0.95, ribbons only, default JUNCTION_REACH), stations (ribbons only, default 16),
 *   flips (per part: reverse that edge's run against the one the ring chooses),
 *   square (cap only: false keeps the free edges' own lean where the sheets stand square on one plane),
 *   continuities (cap only, per part: 0 a crease, 1–3 as continuity; null is the Continuity above),
 *   onePatch (cap only: one trimmed, fitted surface over the gap in place of the exact web; centreTrims carries its trim)
 * @returns {ok, arms: [srf] (ribbons only), centre: [srf] in ring order, spine, order, reversed, ports, rails (the free edges of a cap), faceUp, squareFound, square} or {ok:false, reason}
 */
export function blendSurfaceJunction(parts, opts = {}) {
  const continuity = opts.continuity ?? 2;
  if (continuity !== 1 && continuity !== 2 && continuity !== 3) return { ok: false, reason: 'a surface blend is G1, G2 or G3' };
  if (!Array.isArray(parts) || parts.length < 3) return { ok: false, reason: 'a junction needs three or more edges — two is a blend' };
  for (const X of parts) if (!X || !X.srf || !X.srf.ctrlNet || !BLEND_SURFACE_EDGES.includes(X.edge)) return { ok: false, reason: 'each part is a surface and one of its edges (u0, u1, v0, v1)' };
  const reach = opts.reach ?? JUNCTION_REACH;
  if (!(reach >= 0.05 && reach <= 0.95)) return { ok: false, reason: 'reach must be between 0.05 and 0.95 of the way to the junction center' };
  const bulge = opts.bulge ?? 1;
  if (!(bulge > 0)) return { ok: false, reason: 'bulge must be positive' };
  const stations = Math.max(4, Math.round(opts.stations ?? 16));
  const N = parts.length;
  const flips = Array.isArray(opts.flips) ? opts.flips.map(Boolean) : [];

  const access = parts.map((X) => blendSrfEdgeAccess(X.srf, X.edge));
  if (access.every(blendSrfEdgeClosed)) return blendSurfaceTubeJunction(parts, access, opts);
  const arc = access.map((ea) => blendSrfArcParam(ea.point));
  const mids = access.map((ea, k) => ea.point(arc[k](0.5)));
  const centre = bsScale(mids.reduce((acc, p) => bsAdd(acc, p), [0, 0, 0]), 1 / N);
  // The junction plane from the most spread pair of arms, and the ring order
  // by angle about it.
  let nrm = [0, 0, 0], nrmLen = 0;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const c = bsCross(bsSub(mids[i], centre), bsSub(mids[j], centre)); const L = bsLen(c); if (L > nrmLen) { nrm = c; nrmLen = L; } }
  if (!(nrmLen > 1e-12)) return { ok: false, reason: 'the edges are in a line about their center — there is no junction plane' };
  nrm = bsNorm(nrm);
  const ringOrder = (n) => {
    const d0 = bsSub(mids[0], centre);
    const e1 = bsNorm(bsSub(d0, bsScale(n, bsDot(d0, n)))), e2 = bsCross(n, e1);
    return mids.map((_, i) => i).sort((a, b) => { const da = bsSub(mids[a], centre), db = bsSub(mids[b], centre); return Math.atan2(bsDot(da, e2), bsDot(da, e1)) - Math.atan2(bsDot(db, e2), bsDot(db, e1)); });
  };
  // Every face of the junction faces the way the sheets do: a cap turns with
  // its ring, so the ring runs counterclockwise about the sheets' own side.
  // The sheets' normals summed at their edge midpoints say which side that is.
  {
    let side = [0, 0, 0];
    for (let k = 0; k < N; k++) side = bsAdd(side, bsNorm(blendSrfEdgeNormal(parts[k].srf, parts[k].edge, arc[k](0.5))));
    if (bsDot(side, nrm) < 0) nrm = bsScale(nrm, -1);
  }
  let order = ringOrder(nrm);

  // Two kinds of junction, told apart by how the sheets lie to the junction
  // plane. Sheets edge-on to it (three walls meeting at a corner post) grow
  // arms to ports and ribbons from the ports to a common spine, with nothing
  // between adjacent ribbons — that is open air, as it is between the walls.
  // Sheets lying in the plane (three wings meeting flat) take the guided cap
  // below: their own side edges, blended into free edges, bound one web.
  const sheetNormal = (k) => { const ea = access[k]; const f = ea.frame(arc[k](0.5), false); const [, T] = rationalCurveDerivs(ea.alongCrv, ea.alongCrv.knots[ea.alongCrv.degree] + arc[k](0.5) * (ea.alongCrv.knots[ea.alongCrv.knots.length - 1 - ea.alongCrv.degree] - ea.alongCrv.knots[ea.alongCrv.degree]), 1); return bsNorm(bsCross(T, f.d1)); };
  const faceUp = parts.reduce((acc, _, k) => acc + Math.abs(bsDot(sheetNormal(k), nrm)), 0) / N > 0.5;

  const arms = new Array(N), portPoint = new Array(N);
  if (!faceUp) {
    // Arms (ribbons only): edge frames to the port, which arrives straight. The port is the
    // edge carried `reach` of the way to the center across the edge only — the
    // component along the edge's own run is dropped — so a port keeps the
    // edge's full width and an arm's sides run straight in, so three tall
    // sheets meeting at a spine keep their full height.
    const edgeDir = access.map((ea, k) => bsNorm(bsSub(ea.point(arc[k](1)), ea.point(arc[k](0)))));
    for (let k = 0; k < N; k++) {
      const ea = access[k], ak = arc[k];
      portPoint[k] = (s) => { const p = ea.point(ak(s)); let d = bsSub(centre, p); d = bsSub(d, bsScale(edgeDir[k], bsDot(d, edgeDir[k]))); return bsAdd(p, bsScale(d, reach)); };
      const skin = blendSrfSkin(
        (s) => ea.frame(ak(s), false),
        (s) => { const p = ea.point(ak(s)); const q = portPoint[k](s); const d = bsSub(q, p); return { point: q, d1: d, d2: [0, 0, 0], d3: [0, 0, 0] }; },
        stations, continuity, bulge, 1);
      if (!skin.ok) return { ok: false, reason: `edge ${k + 1}: ${skin.reason}` };
      // The arm faces the way its sheet does at the edge they share.
      const A0 = skin.srf, mid = A0.knotsV[A0.degV] + 0.5 * (A0.knotsV[A0.knotsV.length - 1 - A0.degV] - A0.knotsV[A0.degV]);
      const P = surfacePointAndPartials(A0, A0.knotsU[A0.degU], mid);
      arms[k] = bsDot(bsCross(P.su, P.sv), blendSrfEdgeNormal(parts[k].srf, parts[k].edge, ak(0.5))) < 0 ? blendSrfFlipV(A0) : A0;
    }
  }
  const armEnd = (k, rev, s) => surfacePointAndPartials(arms[k], 1, rev ? 1 - s : s); // U = 1 is the port
  const reverseCrv = (c) => ({ ...c, ctrlPts: [...c.ctrlPts].reverse(), knots: c.knots.map((t) => c.knots[0] + c.knots[c.knots.length - 1] - t).reverse() });
  const portCurve = (k, rv) => { const a = extractIsocurveU(arms[k], 1); return rv ? reverseCrv(a) : a; };
  const railTo = (corner, dir, target, targetDir) => {
    const r = geometricBlend({ point: corner, d1: dir, d2: [0, 0, 0], d3: [0, 0, 0] }, { point: target, d1: targetDir || bsSub(target, corner), d2: [0, 0, 0], d3: [0, 0, 0] }, { continuity: 1 });
    return r.ok ? r.crv : null;
  };
  const line = (A, B) => ({ degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[...A, 1], [...B, 1]] });
  const meanOf = (pts) => bsScale(pts.reduce((acc, p) => bsAdd(acc, p), [0, 0, 0]), 1 / pts.length);
  const rev = new Array(N).fill(false);
  const centreFaces = [], centreTrims = [], rails = [];
  let spine = null, squareFound = false, squared = false, patched = null;
  if (!faceUp) {
    // Ribbons. The ports run the same way, so their starts share one end of
    // the spine and their ends the other.
    const dirOf = (k) => bsSub(armEnd(k, false, 1).point, armEnd(k, false, 0).point);
    // The ribbons run the spine one way or the other; the way that faces
    // them as their arms face is kept.
    for (const turn of [false, true]) {
      const d0dir = bsScale(dirOf(order[0]), turn ? -1 : 1);
      for (let i = 0; i < N; i++) { const k = order[i]; rev[k] = bsDot(dirOf(k), d0dir) < 0; }
      for (let k = 0; k < N; k++) if (flips[k]) rev[k] = !rev[k];
      const ends = order.map((k) => ({ k, start: armEnd(k, rev[k], 0), end: armEnd(k, rev[k], 1) }));
      const spineA = meanOf(ends.map((p) => p.start.point)), spineB = meanOf(ends.map((p) => p.end.point));
      spine = [spineA, spineB];
      const spineCrv = line(spineA, spineB);
      centreFaces.length = 0; rails.length = 0;
      for (const p of ends) {
        const up = railTo(p.end.point, p.end.su, spineB), down = railTo(p.start.point, p.start.su, spineA);
        if (!up || !down) return { ok: false, reason: `no rail could reach the spine from edge ${p.k + 1}` };
        rails.push(up, down);
        try { centreFaces.push(coonsPatch(portCurve(p.k, rev[p.k]), up, reverseCrv(spineCrv), reverseCrv(down))); }
        catch (e) { return { ok: false, reason: `the center would not fill at edge ${p.k + 1}: ${e.message}` }; }
      }
      const f = centreFaces[0], k0 = order[0], A0 = arms[k0];
      const Pf = surfacePointAndPartials(f, (f.knotsU[f.degU] + f.knotsU[f.knotsU.length - 1 - f.degU]) / 2, (f.knotsV[f.degV] + f.knotsV[f.knotsV.length - 1 - f.degV]) / 2);
      const Pa = surfacePointAndPartials(A0, A0.knotsU[A0.knotsU.length - 1 - A0.degU], (A0.knotsV[A0.degV] + A0.knotsV[A0.knotsV.length - 1 - A0.degV]) / 2);
      if (bsDot(bsCross(Pf.su, Pf.sv), bsCross(Pa.su, Pa.sv)) >= 0) break;
    }
  } else if (opts.onePatch) {
    // One surface: the gap the edges bound, patched and trimmed, in place of
    // the web of quads — fitted rather than exact.
    // Fitted from the exact web (the n-sided fill sags toward the plane its
    // corners share: a half-tube cross 8 mm off the web on 10 mm tubes).
    const pr = patchFromSurfaceEdges(parts, { maxCount: opts.patchCount, bulge, web: true, seatRails: true, edgeTolerance: opts.edgeTolerance, webOpts: { square: opts.square, waist: opts.waist || null, continuities: opts.continuities, flips: opts.flips } });
    if (!pr.ok) return pr;
    centreFaces.push(pr.srf); centreTrims.push(pr.trimLoop); rails.push(...pr.rails);
    spine = [centre, centre];
    patched = { edgeGap: pr.edgeGap, edgeAngle: pr.edgeAngleInner, trimGap: pr.trimGap };
  } else {
    const ks = Array.isArray(opts.continuities) ? parts.map((_, kx) => { const c = opts.continuities[kx]; return c === 0 || c === 1 || c === 2 || c === 3 ? c : null; }) : null;
    let cap = blendSrfGuidedCap(access, arc, order, bulge, continuity, nrm, flips, opts.square, ks, opts.waist || null);
    if (!cap.ok) return cap;
    // The first quad meets its sheet along the first half of its edge; a cap
    // facing against it is built again round the ring the other way.
    const q0 = cap.faces[0], k0 = order[0], t0 = cap.rev[k0] ? 0.75 : 0.25;
    const P0 = surfacePointAndPartials(q0, (q0.knotsU[q0.degU] + q0.knotsU[q0.knotsU.length - 1 - q0.degU]) / 2, q0.knotsV[q0.degV]);
    if (bsDot(bsCross(P0.su, P0.sv), blendSrfEdgeNormal(parts[k0].srf, parts[k0].edge, t0)) < 0) {
      nrm = bsScale(nrm, -1); order = ringOrder(nrm);
      cap = blendSrfGuidedCap(access, arc, order, bulge, continuity, nrm, flips, opts.square, ks, opts.waist || null);
      if (!cap.ok) return cap;
    }
    // Sheets that overlap where they meet (a corner of one past the side of
    // the next, as seen from C) leave the quads between them nowhere to go
    // but over themselves. A folded cap is refused, never shipped. A Flip is
    // the one exception: an edge run against the ring twists its quads by
    // request, and the flipped cap is what was asked for.
    const folded = flips.some(Boolean) ? -1 : cap.faces.findIndex((f) => blendSrfFaceReversals(f) > 0);
    if (folded >= 0) {
      const i = folded >> 1, j = (folded % 2 ? i + 1 : i + N - 1) % N, [a, b] = [order[i] + 1, order[j] + 1].sort((x, y) => x - y);
      return { ok: false, reason: `edges ${a} and ${b} overlap at the junction — move them apart` };
    }
    for (let k = 0; k < N; k++) rev[k] = cap.rev[k];
    centreFaces.push(...cap.faces); rails.push(...cap.free); spine = [cap.C, cap.C];
    squareFound = cap.squareFound; squared = cap.square;
  }
  return {
    ok: true,
    arms: faceUp ? [] : arms, order, reversed: rev, ports: faceUp ? [] : portPoint.map((f) => Array.from({ length: 9 }, (_, i) => f(i / 8))), rails,
    centre: centreFaces, centreTrims: centreFaces.map((_, i) => centreTrims[i] || null), spine, faceUp, squareFound, square: squared, patched,
  };
}

// Patch on surface edges — one surface over the gap the edges bound, where
// BlendSrf builds an exact web of faces. Each surface gives its edge facing
// the others; the edges are put in ring order about their center, each run the
// ring's way, and the gap from one edge's end to the next edge's start is
// closed by a curve blend leaving both tangentially, as a cap's free edges are.
// The loop is filled exactly by the n-sided tangent fill (tangentpatch.mjs):
// each surface edge with its surface's own outward direction, each free edge
// with the direction turning from one neighbor's edge to the other's, so every
// corner lies in both neighbors' plane. The fill is not a NURBS surface, so it
// is sampled densely over its own polygon domain and fitted (kernel/patch.mjs,
// given parameters), and the polygon is the trim. A fill that folds is retried
// with shorter cross-tangents; a hole it folds over still is fitted from the
// web Blend builds over the same edges, over their plane. A closed rim is
// refused: it has no ends to close a boundary with, and whole tubes meet
// through Blend.
// Returns { ok, srf, trimLoop (u,v), rails (the free edges), edgeGap and
// edgeAngle / edgeAngleInner (measured along the surface edges) } or a refusal.
// The fill's cross-tangent length, as a share of each side's chord to the far
// corners. Measured on a half-tube cross: 0.25 turns within 2% of the domain
// (a fitted net met the tubes at 15 degrees), 0.8 turns over the whole cap
// (2 degrees at 32 x 32), 1.2 folds.
export const PATCH_EDGE_TANGENT_SCALE = 0.8;
// Seating the trim (patchFromSurfaceEdges): a trim vertex counts this many
// times an interior sample, squared, in the refit; at most this many rounds of
// refit and halving; the refit's net count, half as many again after the first
// round (measured: a cross seats at 64 to 0.00015 mm, a T needs 96).
export const PATCH_SEAT_WEIGHT = 30;
export const PATCH_SEAT_ROUNDS = 3;
export const PATCH_SEAT_COUNT = 64;
export function patchFromSurfaceEdges(parts, opts = {}) {
  if (!Array.isArray(parts) || parts.length < 2) return { ok: false, reason: 'Patch on surfaces needs two or more surface edges' };
  const N = parts.length, bulge = opts.bulge ?? 1;
  if (!(bulge > 0)) return { ok: false, reason: 'bulge must be positive' };
  const access = parts.map((X) => blendSrfEdgeAccess(X.srf, X.edge));
  if (access.some(blendSrfEdgeClosed)) return { ok: false, reason: 'a whole tube has no ends to close a patch with — Blend joins whole tubes' };
  const arc = access.map((ea) => blendSrfArcParam(ea.point));
  const mids = access.map((ea, k) => ea.point(arc[k](0.5)));
  const centre = bsScale(mids.reduce((a, q) => bsAdd(a, q), [0, 0, 0]), 1 / N);
  let nrm = [0, 0, 0], nrmLen = 0;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const c = bsCross(bsSub(mids[i], centre), bsSub(mids[j], centre)); if (bsLen(c) > nrmLen) { nrm = c; nrmLen = bsLen(c); } }
  if (N === 2 || !(nrmLen > 1e-12)) { const run = bsSub(access[0].point(1), access[0].point(0)); nrm = bsCross(bsSub(mids[1], mids[0]), run); nrmLen = bsLen(nrm); }
  if (!(nrmLen > 1e-12)) return { ok: false, reason: 'the edges lie on one line — there is no gap between them to patch' };
  nrm = bsNorm(nrm);
  const e1 = bsNorm(bsSub(bsSub(mids[0], centre), bsScale(nrm, bsDot(bsSub(mids[0], centre), nrm))));
  const e2 = bsCross(nrm, e1);
  const ang = (q) => { const d = bsSub(q, centre); return Math.atan2(bsDot(d, e2), bsDot(d, e1)); };
  const order = mids.map((_, i) => i).sort((a, b) => ang(mids[a]) - ang(mids[b]));
  // Each edge runs the ring's way: its end lies ahead of its start about the
  // center. Two edges need not have the center between them (two concentric
  // arcs lie on one side of it, both turning the same way about it): the
  // second runs from the end nearer the first one's end.
  const rev = access.map((ea) => bsDot(bsCross(bsSub(ea.point(0), centre), bsSub(ea.point(1), centre)), nrm) < 0);
  if (N === 2) {
    const [ka, kb] = order, aEnd = access[ka].point(rev[ka] ? 0 : 1);
    rev[kb] = bsLen(bsSub(aEnd, access[kb].point(1))) < bsLen(bsSub(aEnd, access[kb].point(0)));
  }
  const at = (kx, t) => access[kx].frame(arc[kx](rev[kx] ? 1 - t : t), false);
  const edgePoint = (kx) => (t) => access[kx].point(arc[kx](rev[kx] ? 1 - t : t));
  // The loop's sides in ring order: each surface edge, then the free edge that
  // closes the gap to the next. A side is a curve on [0,1] and a cross-tangent
  // field: a surface edge's is the surface's own outward direction; a free
  // edge's turns from the edge direction of the surface it leaves to that of
  // the surface it reaches, so both its corners share their neighbor's plane.
  const edgeTan = (kx, t) => {
    const K = access[kx].alongCrv.knots, dg = access[kx].alongCrv.degree;
    const tt = arc[kx](rev[kx] ? 1 - t : t);
    const d = rationalCurveDerivs(access[kx].alongCrv, K[dg] + tt * (K[K.length - 1 - dg] - K[dg]), 1)[1];
    return bsScale(bsNorm(d), rev[kx] ? -1 : 1);
  };
  const sides = [], rails = [];
  for (let r = 0; r < N; r++) {
    const kx = order[r], kn = order[(r + 1) % N];
    sides.push({ kx, point: (t) => at(kx, t).point, tangent: (t) => at(kx, t).d1 });
    const fe = at(kx, 1), fs = access[kn].frame(arc[kn](rev[kn] ? 1 : 0), true);
    const gap = bsLen(bsSub(fs.point, fe.point));
    if (!(gap > 1e-9)) continue;
    const m = 0.6 * gap * bulge;
    const r2 = geometricBlend({ point: fe.point, d1: fe.d1, d2: [0, 0, 0], d3: [0, 0, 0] }, { point: fs.point, d1: fs.d1, d2: [0, 0, 0], d3: [0, 0, 0] }, { continuity: 1, startMagnitude: m, endMagnitude: m });
    if (!r2.ok) return { ok: false, reason: `no edge could close the gap between edges ${kx + 1} and ${kn + 1}` };
    rails.push(r2.crv);
    const K = r2.crv.knots, a = K[0], b = K[K.length - 1];
    const t0 = bsScale(edgeTan(kx, 1), -1), t1 = edgeTan(kn, 0);
    const t1s = bsDot(t0, t1) < 0 ? bsScale(t1, -1) : t1;
    sides.push({ kx: null, point: (t) => curvePoint(r2.crv, a + (b - a) * t), tangent: (t) => bsAdd(bsScale(t0, 1 - t), bsScale(t1s, t)) });
  }
  // A fill that folds is tried again with shorter cross-tangents (a flatter
  // shoulder) before the gap is called too distorted for one surface.
  // Bulge scales the cross-tangents (fuller or flatter over the gap), and the
  // free edges' handles with them; a fuller fill that folds falls back to
  // the plain ladder.
  // Each step down the ladder turns the patch faster off its edges (a fitted
  // net then meets them less squarely: 2.3 degrees at 0.75 on five tubes,
  // 3.4 at 0.55), so it steps down in small rungs.
  const ladder = [PATCH_EDGE_TANGENT_SCALE, 0.75, 0.7, 0.65, 0.55, 0.45, 0.35, 0.2];
  let fill = null;
  for (const ts of opts.web ? [] : opts.tangentScale != null ? [opts.tangentScale] : [...ladder.map((x) => x * bulge), ...(bulge > 1 ? ladder : [])]) {
    fill = nSidedTangentPatch({ boundary: sides.map((sd) => sd.point), tangent: sides.map((sd) => sd.tangent), tangentScale: ts, oriented: true });
    if (fill.ok) break;
  }
  const S = Math.max(8, Math.round(opts.samplesPerEdge ?? 32));
  // trimSides: the loop's sides in order, each its exact model-space curve q(t)
  // and the curve's place on the patch uv(t), t in [0, 1]; `seat`: the side is
  // one a neighbor joins along (every surface edge, and the free edges when
  // opts.seatRails says another face shares them).
  const targets = [], params = [], trimSides = [], edgeSamples = [];
  const pad = 0.03;
  // Lays a box of the plane (lx..hx, ly..hy) in the patch's square, padded.
  const square = (lx, ly, hx, hy) => { const sc = (1 - 2 * pad) / Math.max(hx - lx, hy - ly); return (x, y) => [pad + (x - lx) * sc + (1 - 2 * pad - (hx - lx) * sc) / 2, pad + (y - ly) * sc + (1 - 2 * pad - (hy - ly) * sc) / 2]; };
  if (fill && fill.ok) {
    // The fill is exact but not a NURBS surface: it is sampled densely over its
    // own polygon domain, laid in the patch's square, and fitted.
    const V = fill.domain.vertices;
    let lx = Infinity, ly = Infinity, hx = -Infinity, hy = -Infinity;
    for (const [x, y] of V) { lx = Math.min(lx, x); ly = Math.min(ly, y); hx = Math.max(hx, x); hy = Math.max(hy, y); }
    const toUV = square(lx, ly, hx, hy);
    const G = Math.max(24, Math.round(opts.grid ?? 96));
    for (let i = 0; i <= G; i++) for (let j = 0; j <= G; j++) {
      const x = lx + (hx - lx) * i / G, y = ly + (hy - ly) * j / G;
      const q = fill.evaluateXY(x, y);
      if (q) { targets.push(q); params.push(toUV(x, y)); }
    }
    sides.forEach((sd, i) => {
      const A = V[i], B = V[(i + 1) % V.length];
      trimSides.push({ seat: sd.kx != null || !!opts.seatRails, q: sd.kx == null ? sd.point : edgePoint(sd.kx), uv: (t) => toUV(A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t) });
      for (let k = 0; k < S; k++) {
        const t = k / S, uv = trimSides[i].uv(t);
        const q = sd.point(t);
        // Twice: the boundary is worth more than any one interior sample.
        targets.push(q, q); params.push(uv, uv);
        if (sd.kx != null) edgeSamples.push({ point: q, kx: sd.kx, t, uv });
      }
    });
  } else {
    // A hole the fill folds over (a flat star: the side-vertex rays cross where
    // the free edges bow in), or one asked for with opts.web, takes the web
    // Blend builds over the same edges — the guided cap, or between two edges
    // the two-edge blend — sampled and fitted, closed by the web's own free
    // edges. opts.webOpts reaches the cap (a tube junction's waist).
    let faces = null, webRails = null;
    if (N >= 3) {
      const j = blendSurfaceJunction(parts, { continuity: 2, bulge, ...(opts.webOpts || {}) });
      if (j.ok && j.faceUp && !j.tube) { faces = j.centre; webRails = j.rails; }
    } else {
      const j = blendSurfaceBetweenEdges(parts[0], parts[1], { continuity: 2, bulgeA: bulge, bulgeB: bulge, adaptive: true, tolerance: 1e-3, maxStations: 97 });
      if (j.ok) { const K = j.srf.knotsV, d = j.srf.degV; faces = [j.srf]; webRails = [extractIsocurveV(j.srf, K[d]), extractIsocurveV(j.srf, K[K.length - 1 - d])]; }
    }
    if (!faces) return { ok: false, reason: fill ? fill.reason : 'the gap has no web to fit' };
    // Parameters by central projection from a point 0.4 of the ring radius
    // below the web, on the side away from it (the walls spread most there of
    // 0.25 to 1.5: rims met within 1.5 degrees on a T, a cross, five tubes): a flat web projects as its own plane,
    // and a cap standing square to the plane at a tube junction's waist still
    // spreads over the domain, where a projection along the normal would
    // crush its walls to a line. The web must be a graph from that point.
    const webSamples = [], webNormals = [];
    for (const f of faces) {
      const u0 = f.knotsU[f.degU], u1 = f.knotsU[f.knotsU.length - 1 - f.degU], v0 = f.knotsV[f.degV], v1 = f.knotsV[f.knotsV.length - 1 - f.degV];
      for (let i = 0; i <= 32; i++) for (let k = 0; k <= 32; k++) {
        const P = surfacePointAndPartials(f, u0 + (u1 - u0) * i / 32, v0 + (v1 - v0) * k / 32);
        webSamples.push(P.point); webNormals.push(bsCross(P.su, P.sv));
      }
    }
    // Measured from the corners' mean, where a cap over half tubes stands on its waist.
    const ends = sides.flatMap((sd) => [sd.point(0), sd.point(1)]), base = bsScale(ends.reduce((acc, q) => bsAdd(acc, q), [0, 0, 0]), 1 / ends.length);
    const rise = webSamples.reduce((acc, q) => acc + bsDot(bsSub(q, base), nrm), 0), up3 = rise < 0 ? bsScale(nrm, -1) : nrm;
    const h = 0.4 * Math.max(1e-9, mids.reduce((acc, q) => acc + bsLen(bsSub(q, centre)), 0) / N), eye = bsSub(base, bsScale(up3, h));
    const proj = (q) => { const d = bsSub(q, eye), z = bsDot(d, up3); return [h * bsDot(d, e1) / z, h * bsDot(d, e2) / z]; };
    let up = 0, down = 0;
    webSamples.forEach((q, i) => { const n = bsDot(webNormals[i], bsSub(q, eye)); if (n > 0) up++; else if (n < 0) down++; });
    if ((up && down) || webSamples.some((q) => !(bsDot(bsSub(q, eye), up3) > 1e-9 * h))) return { ok: false, reason: fill ? fill.reason : 'the web is not a graph to fit over' };
    // The loop: each surface edge in ring order, then the web's free edge from
    // its end to the next edge's start.
    const loop = [];
    const tol = 1e-6 * Math.max(1, ...mids.map((q) => bsLen(bsSub(q, centre))));
    for (let r = 0; r < N; r++) {
      const sd = sides.find((x) => x.kx === order[r]), kn = order[(r + 1) % N];
      trimSides.push({ seat: true, q: edgePoint(sd.kx) });
      for (let k = 0; k < S; k++) loop.push({ q: sd.point(k / S), kx: sd.kx, t: k / S });
      const from = sd.point(1), to = sides.find((x) => x.kx === kn).point(0);
      if (!(bsLen(bsSub(to, from)) > tol)) continue;
      const c = webRails.find((cv) => { const a = curvePoint(cv, cv.knots[0]), b = curvePoint(cv, cv.knots[cv.knots.length - 1]); return (bsLen(bsSub(a, from)) < 1e3 * tol && bsLen(bsSub(b, to)) < 1e3 * tol) || (bsLen(bsSub(b, from)) < 1e3 * tol && bsLen(bsSub(a, to)) < 1e3 * tol); });
      if (!c) return { ok: false, reason: fill.reason };
      const ka = c.knots[0], kb = c.knots[c.knots.length - 1], fwd = bsLen(bsSub(curvePoint(c, ka), from)) < bsLen(bsSub(curvePoint(c, kb), from));
      const railAt = (t) => curvePoint(c, fwd ? ka + (kb - ka) * t : kb - (kb - ka) * t);
      trimSides.push({ seat: !!opts.seatRails, q: railAt });
      for (let k = 0; k < S; k++) loop.push({ q: railAt(k / S), kx: null });
    }
    let lx = Infinity, ly = Infinity, hx = -Infinity, hy = -Infinity;
    for (const L of loop) { const [x, y] = proj(L.q); lx = Math.min(lx, x); ly = Math.min(ly, y); hx = Math.max(hx, x); hy = Math.max(hy, y); }
    const toUV = square(lx, ly, hx, hy);
    for (const sd of trimSides) { const seen = new Map(); sd.uv = (t) => { let w = seen.get(t); if (!w) { w = toUV(...proj(sd.q(t))); seen.set(t, w); } return w; }; }
    for (const q of webSamples) { targets.push(q); params.push(toUV(...proj(q))); }
    for (const L of loop) {
      const uv = toUV(...proj(L.q));
      targets.push(L.q, L.q); params.push(uv, uv);
      if (L.kx != null) edgeSamples.push({ point: L.q, kx: L.kx, t: L.t, uv });
    }
    rails.length = 0; rails.push(...webRails);
  }
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const q of targets) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], q[k]); hi[k] = Math.max(hi[k], q[k]); }
  const span = bsLen(bsSub(hi, lo));
  const fit = fitPatchToTolerance(targets, { tolerance: opts.tolerance ?? 1e-3 * span, maxCount: opts.maxCount ?? 48, stiffness: opts.stiffness ?? 0.01, degree: 3, uCount: opts.uCount ?? Math.min(32, opts.maxCount ?? 48), params });
  if (!fit.ok) return fit;
  // The trim on its edges. The fit meets the loop only as closely as its net
  // allows, and a polygon trim through samples of a curved side cuts across
  // its chords; either leaves the trimmed boundary hundredths of a millimetre
  // off the edges it was built from, too far to join them. So the trim is
  // first made dense enough that no chord bows across its side's own path by
  // more than half of edgeTolerance, then the net is fitted again
  // (PATCH_SEAT_COUNT, every trim vertex weighted PATCH_SEAT_WEIGHT) until
  // every vertex and chord midpoint lies within edgeTolerance of its side's
  // exact curve: the surface edges, and the free edges a neighbor shares (a
  // free edge nothing shares is held at its corners only).
  // Measured on the four-tube cross: 0.0003 mm on the trim, 0.27 degrees at
  // the rims, 0.010 mm off the web it fits (0.031 before the refit).
  const edgeTol = opts.edgeTolerance ?? 3e-4;
  const ts = trimSides.map(() => Array.from({ length: S }, (_, k) => k / S));
  const cache = trimSides.map(() => new Map());
  // A vertex the refit still misses counts three times as heavily next round, to nine.
  const heavier = trimSides.map(() => new Map());
  const Q = (i, t) => { let p = cache[i].get(t); if (!p) { p = trimSides[i].q(t); cache[i].set(t, p); } return p; };
  // A fitted net is polynomial (every weight 1): its point is the basis sum.
  const pointOn = (f, u, v) => {
    const nu = f.ctrlNet.length - 1, nv = f.ctrlNet[0].length - 1, su = findSpan(nu, f.degU, u, f.knotsU), sv = findSpan(nv, f.degV, v, f.knotsV);
    const Nu = basisFuns(su, u, f.degU, f.knotsU), Nv = basisFuns(sv, v, f.degV, f.knotsV), out = [0, 0, 0];
    for (let a = 0; a <= f.degU; a++) { const row = f.ctrlNet[su - f.degU + a]; for (let b = 0; b <= f.degV; b++) { const w = Nu[a] * Nv[b], P = row[sv - f.degV + b]; out[0] += w * P[0]; out[1] += w * P[1]; out[2] += w * P[2]; } }
    return out;
  };
  const segDist = (p, A, B) => { const d = bsSub(B, A), L2 = bsDot(d, d); const t = L2 > 0 ? Math.max(0, Math.min(1, bsDot(bsSub(p, A), d) / L2)) : 0; return bsLen(bsSub(bsAdd(A, bsScale(d, t)), p)); };
  // One pass over every chord on surface `on`: the largest distance of a
  // vertex or chord midpoint from its side, and (with `halve`) each chord split
  // in two whose midpoint lies off its side by more than edgeTol (`bowOnly`:
  // whose polygon bows across the side's own path on the patch by more than
  // edgeTol / 2 in model space, the patch's partials carrying it there).
  const sweep = (on, halve, bowOnly) => {
    let worst = 0, halved = false;
    trimSides.forEach((sd, i) => {
      const list = ts[i], next = [];
      for (let k = 0; k < list.length; k++) {
        const a = list[k], b = k + 1 < list.length ? list[k + 1] : 1, m = (a + b) / 2, ua = sd.uv(a), ub = sd.uv(b);
        next.push(a);
        // A free side is held only at its corner, where it meets a seated side.
        if (!sd.seat) { if (k === 0 && !bowOnly) { const qa = Q(i, a), pa = pointOn(on, ua[0], ua[1]), ga = bsLen(bsSub(pa, qa)); worst = Math.max(worst, ga); if (halve && ga > edgeTol) heavier[i].set(a, Math.min(9, 3 * (heavier[i].get(a) || 1))); } continue; }
        let split;
        if (bowOnly) {
          const um = sd.uv(m), P = surfacePointAndPartials(on, um[0], um[1]), du = (ua[0] + ub[0]) / 2 - um[0], dv = (ua[1] + ub[1]) / 2 - um[1];
          const off = bsAdd(bsScale(P.su, du), bsScale(P.sv, dv)), along = bsNorm(bsSub(Q(i, b), Q(i, a)));
          split = bsLen(bsSub(off, bsScale(along, bsDot(off, along)))) > edgeTol / 2;
        } else {
          const pa = pointOn(on, ua[0], ua[1]), pm = pointOn(on, (ua[0] + ub[0]) / 2, (ua[1] + ub[1]) / 2), qa = Q(i, a), qb = Q(i, b), qm = Q(i, m);
          const dm = Math.min(segDist(pm, qa, qm), segDist(pm, qm, qb));
          const ga = bsLen(bsSub(pa, qa));
          worst = Math.max(worst, ga, dm);
          if (halve && ga > edgeTol) heavier[i].set(a, Math.min(9, 3 * (heavier[i].get(a) || 1)));
          split = dm > edgeTol;
        }
        if (halve && split) { next.push(m); halved = true; }
      }
      ts[i] = next;
    });
    return { worst, halved };
  };
  for (let k = 0; k < 10 && sweep(fit.srf, true, true).halved; k++);
  let seated = fit.srf, trimGap = null;
  for (let round = 0; round < PATCH_SEAT_ROUNDS; round++) {
    const want = round ? Math.round(PATCH_SEAT_COUNT * 1.5) : PATCH_SEAT_COUNT;
    let count = Math.max(fit.uCount, want);
    // At least two trim vertices to a knot span along a seated side: sparser,
    // the net bows between them where nothing inside holds it (a T's open side
    // bowed 0.008 mm between vertices that sat to a micron).
    const span = 1 / Math.max(1, count - 3);
    trimSides.forEach((sd, i) => {
      if (!sd.seat) return;
      for (let pass = 0; pass < 8; pass++) {
        const list = ts[i], next = [];
        let split = false;
        for (let k = 0; k < list.length; k++) {
          const a = list[k], b = k + 1 < list.length ? list[k + 1] : 1, ua = sd.uv(a), ub = sd.uv(b);
          next.push(a);
          if (Math.hypot(ub[0] - ua[0], ub[1] - ua[1]) > span / 2) { next.push((a + b) / 2); split = true; }
        }
        ts[i] = next;
        if (!split) break;
      }
    });
    const T = targets.slice(), Pm = params.slice(), W = targets.map(() => 1);
    trimSides.forEach((sd, i) => { for (const t of sd.seat ? ts[i] : [0]) { T.push(Q(i, t)); Pm.push(sd.uv(t)); W.push(PATCH_SEAT_WEIGHT * (heavier[i].get(t) || 1)); } });
    count = Math.min(count, Math.max(fit.uCount, Math.floor(Math.sqrt(T.length))));
    const refit = fitPatch(T, { degree: 3, uCount: count, vCount: count, stiffness: opts.stiffness ?? 0.01, params: Pm, weights: W });
    if (!refit.ok) break;
    seated = refit.srf;
    const r = sweep(seated, true, false);
    trimGap = r.worst;
    if (trimGap <= edgeTol || !r.halved) break;
  }
  // A fill the net cannot seat (its corners are cones the polynomial net cannot
  // follow: a millimetre off on four half tubes) gives way to the web fitted
  // over the same edges, when the web seats closer.
  if (fill && fill.ok && !opts.web && !(trimGap <= edgeTol)) {
    const viaWeb = patchFromSurfaceEdges(parts, { ...opts, web: true });
    if (viaWeb.ok && viaWeb.trimGap != null && (trimGap == null || viaWeb.trimGap < trimGap)) return viaWeb;
  }
  const trimLoop = trimSides.flatMap((sd, i) => ts[i].map((t) => sd.uv(t)));
  // How well the edges were met: position, and the angle between the patch and
  // each surface along its edge (the fitted normal against the surface's own).
  // The corners are G0 by construction (the fill's own limit), so the angle is
  // also reported along each edge's interior, a tenth in from its ends.
  let gapMax = 0, angMax = 0, angInner = 0, facing = 0;
  for (const e of edgeSamples) {
    const P = surfacePointAndPartials(seated, e.uv[0], e.uv[1]);
    gapMax = Math.max(gapMax, bsLen(bsSub(P.point, e.point)));
    const pn = bsNorm(bsCross(P.su, P.sv));
    facing += Math.sign(bsDot(pn, blendSrfEdgeNormal(parts[e.kx].srf, parts[e.kx].edge, arc[e.kx](rev[e.kx] ? 1 - e.t : e.t))));
    const sn = bsNorm(bsCross(edgeTan(e.kx, e.t), at(e.kx, e.t).d1));
    const a = Math.acos(Math.min(1, Math.abs(bsDot(pn, sn)))) * 180 / Math.PI;
    angMax = Math.max(angMax, a);
    if (e.t >= 0.1 && e.t <= 0.9) angInner = Math.max(angInner, a);
  }
  // The patch faces the way its surfaces do along their edges: a patch facing
  // against them is transposed, its trim with it (run backward, so the loop
  // keeps its winding).
  let srf = seated, loop = trimLoop;
  if (facing < 0) {
    srf = { ...srf, degU: srf.degV, degV: srf.degU, knotsU: srf.knotsV.slice(), knotsV: srf.knotsU.slice(), ctrlNet: srf.ctrlNet[0].map((_, j) => srf.ctrlNet.map((row) => row[j].slice())) };
    loop = trimLoop.map(([u, v]) => [v, u]).reverse();
  }
  return { ok: true, srf, trimLoop: loop, rails: rails.filter(Boolean), order, edgeGap: gapMax, edgeAngle: angMax, edgeAngleInner: angInner, trimGap, maxDeviation: fit.maxDeviation, count: fit.uCount, tried: fit.tried };
}
