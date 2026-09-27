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
// direction pairs its ends nearer to A's — the bridge's rule. A twisted pair
// of edges is a valid shape a user can want; it is not searched for.
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
import { fitPatchToTolerance } from './patch.mjs';
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
  const reverseB = (flipped < straight) !== (!!flip[0] !== !!flip[1]);
  const tB = (s) => (reverseB ? arcB(1 - s) : arcB(s));

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
  const deviationOf = (built) => {
    let worst = 0;
    for (let i = 0; i + 1 < built.sGrid.length; i++) for (const f of [0.25, 0.5, 0.75]) {
      const s = built.sGrid[i] + f * (built.sGrid[i + 1] - built.sGrid[i]), exact = sectionAt(s);
      if (!exact.ok) return Infinity;
      const v = vAt(built, s), u0 = exact.crv.knots[exact.crv.degree], u1 = exact.crv.knots[exact.crv.knots.length - 1 - exact.crv.degree];
      for (const u of [0, 0.25, 0.5, 0.75, 1]) worst = Math.max(worst, bsLen(bsSub(surfacePoint(built.srf, u, v), curvePoint(exact.crv, u0 + u * (u1 - u0)))));
    }
    return worst;
  };

  let built = build(stations);
  if (!built.ok) return built;
  let deviation = deviationOf(built), refined = false;
  const tolerance = opts.tolerance ?? Math.max(1e-4, 1e-4 * built.chordMax), maxStations = Math.max(stations, Math.round(opts.maxStations ?? 97));
  if (opts.adaptive === true) while (deviation > tolerance && stations < maxStations) {
    stations = Math.min(maxStations, 2 * (stations - 1) + 1);
    built = build(stations); if (!built.ok) return built;
    deviation = deviationOf(built); refined = true;
  }
  if (opts.adaptive === true && deviation > tolerance) return { ok: false, reason: `the blend skin is ${deviation.toPrecision(3)}mm from its section family after ${stations} stations; it needs a looser tolerance than ${tolerance}mm` };
  return { ok: true, srf: built.srf, stations, chordMin: built.chordMin, chordMax: built.chordMax, reverseB, continuity, deviation, tolerance, refined, toleranceMet: deviation <= tolerance };
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
// what makes the center solvable: the 2N spokes arrive at C in a regular fan
// in one tangent plane, all with zero second derivative and speed L, and the
// cross fields there are the neighboring spokes' tangents — for which the
// fan gives m_{i-1} + m_i ∥ e_i, so α(1) = 2 cos(π/N). A bounded third-row
// solve then equalizes geometric transverse normal curvature along each seam
// without moving its boundary or tangent plane. The twists are zero at C and
// at the free-edge midpoints. One speed ratio g per sheet (the blend's
// cross speed over the sheet's own) keeps a corner's curve and the ribbon's
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
    const f = edge.frameAt(u);
    samples[0].push(f.point); samples[1].push(bsScale(f.d1, g));
    if (k >= 2) samples[2].push(bsScale(f.d2, g * g));
    if (k >= 3) samples[3].push(bsScale(f.d3, g * g * g));
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
  // A free edge joins two side edges the way a fillet would. Its handles are
  // 2g times the sheet's cross speed; sized from the center alone they can
  // pass the point where the two side edges' lines meet, and the edge hooks
  // back on itself (two curvature reversals, a 0.3 mm radius at Bulge 1.4 on
  // 40 mm sheets). So g is also held to 0.8 of the distance from each of the
  // sheet's corners to that meeting point, in the junction plane; Bulge above 1
  // cannot pass it.
  for (let i = 0; i < N; i++) {
    const kx = order[i], kn = order[(i + 1) % N];
    const fe = frameAt(kx, tOf(kx, 1)), fs = frameAt(kn, tOf(kn, 0), true);
    const flat = (v) => bsSub(v, bsScale(nrm, bsDot(v, nrm)));
    const uA = bsNorm(flat(fe.d1)), uB = bsNorm(flat(fs.d1)), w = flat(bsSub(fs.point, fe.point));
    const cr = bsDot(bsCross(uA, uB), nrm);
    if (!(Math.abs(cr) > 1e-6)) continue;
    const sA = bsDot(bsCross(w, uB), nrm) / cr, tB = bsDot(bsCross(w, uA), nrm) / cr; // X = A + sA uA = B + tB uB
    if (!(sA > 0) || !(tB < 0)) continue; // the lines meet ahead of both corners, or no fillet is implied
    g[kx] = Math.min(g[kx], 0.8 * Math.min(bulge, 1) * sA / bsLen(fe.d1));
    g[kn] = Math.min(g[kn], 0.8 * Math.min(bulge, 1) * -tB / bsLen(fs.d1));
  }
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
  // Free edges at 2g, split at their parameter midpoint.
  const free = new Array(N), M = new Array(N), Mt = new Array(N);
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
      { continuity: Math.max(kRows(kx), kRows(kn)), startMagnitude: 2 * g[kx] * bsLen(fe.d1), endMagnitude: 2 * g[kn] * bsLen(fs.d1) });
    if (!free[i]) return { ok: false, reason: `no edge could join the corners of edges ${kx + 1} and ${kn + 1}` };
    const [Mp, T] = rationalCurveDerivs(free[i], 0.5, 1);
    M[i] = Mp; Mt[i] = T;
  }
  // The fan at C: 2N directions in the junction plane, in ring order, at one speed L.
  const dC = bsSub(mids[0].point, C);
  const e1 = bsNorm(bsSub(dC, bsScale(nrm, bsDot(dC, nrm)))), e2 = bsCross(nrm, e1);
  const angleOf = (q) => { const d = bsSub(q, C); return Math.atan2(bsDot(d, e2), bsDot(d, e1)); };
  const sense = angleOf(mids[1 % N].point) > 0 ? 1 : -1;
  const delta = Math.PI / N;
  const dir = (j) => bsAdd(bsScale(e1, Math.cos(sense * j * delta)), bsScale(e2, Math.sin(sense * j * delta)));
  let L = 0; for (let i = 0; i < N; i++) L += bsLen(bsSub(C, mids[i].point)) + bsLen(bsSub(C, M[i])); L /= 2 * N;
  const arrive = (j) => bsScale(dir(j), -L); // the tangent a spoke arrives at C with
  const spokeE = new Array(N), spokeM = new Array(N);
  for (let i = 0; i < N; i++) {
    const kx = order[i], f = mids[i];
    spokeE[i] = blend({ point: f.point, d1: f.d1, d2: f.d2, d3: f.d3 }, { point: C, d1: arrive(2 * i), d2: zero, d3: zero },
      { continuity: Math.max(2, kRows(kx)), startMagnitude: g[kx] * bsLen(f.d1), endMagnitude: L });
    if (!spokeE[i]) return { ok: false, reason: `no spoke could reach the centre from edge ${kx + 1}` };
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
      { continuity: Math.max(2, kRows(kx), kRows(order[(i + 1) % N])), startMagnitude: bsLen(bsSub(C, M[i])), endMagnitude: L });
    if (!spokeM[i]) return { ok: false, reason: `no spoke could reach the centre from the edge between ${kx + 1} and ${order[(i + 1) % N] + 1}` };
  }
  const alpha1 = 2 * Math.cos(delta);
  // Halves: a sheet edge from chain t a to b as a curve on [0,1] with its frames; a free edge's halves.
  const half = (kx, a, b) => {
    const ea = access[kx], crv = ea.alongCrv, lo = crv.knots[crv.degree], hi = crv.knots[crv.knots.length - 1 - crv.degree];
    const ta = tOf(kx, a), tb = tOf(kx, b);
    let sub = extractSubCurve(crv, lo + Math.min(ta, tb) * (hi - lo), lo + Math.max(ta, tb) * (hi - lo));
    if (tb < ta) sub = reverseCrv(sub);
    sub = rescaleCurveDomain(sub, 0, 1);
    return { crv: sub, frameAt: (u) => ea.frame(ta + u * (tb - ta), false) };
  };
  const subOn = (crv, a, b) => rescaleCurveDomain(extractSubCurve(crv, a, b), 0, 1);
  const endTangent = (crv, atEnd) => crvDeriv(crv, atEnd ? crv.knots[crv.knots.length - 1] : crv.knots[0], 1);
  // Pass 1: the quads' rows. End quad of sheet i: mid → end corner; start quad: start corner → mid.
  const quads = [];
  for (let i = 0; i < N; i++) {
    const kx = order[i], ip = (i + N - 1) % N;
    const e_i = endTangent(spokeE[i], true), m_i = endTangent(spokeM[i], true), m_ip = endTangent(spokeM[ip], true);
    const firstHalf = subOn(free[i], 0, 0.5), secondHalfRev = reverseCrv(subOn(free[ip], 0.5, 1));
    // The end quad's far field, C → M_i: from the spoke's tangent at C to the free half's tangent at M, zero twists.
    const cfEnd = bzHermite([e_i, zero], [endTangent(firstHalf, true), zero]);
    const kq = kRows(kx);
    const start = { kind: 'start', kx, i, edge: half(kx, 0, 0.5), side0: secondHalfRev, side1: spokeE[i], far: spokeM[ip], g: g[kx], k: kq };
    const end = { kind: 'end', kx, i, edge: half(kx, 0.5, 1), side0: spokeE[i], side1: firstHalf, far: reverseCrv(spokeM[i]), g: g[kx], k: kq, cf: cfEnd, e_i, m_i, m_ip };
    quads.push(start, end);
  }
  // The start quad's far field is the previous end quad's, reversed, plus β·h·M' (β1 = α1).
  for (let i = 0; i < N; i++) {
    const start = quads[2 * i], ip = (i + N - 1) % N, endPrev = quads[2 * ip + 1];
    const revCf = [...endPrev.cf].reverse().map((v) => bsScale(v, -1));
    start.cf = bzAdd(revCf, bzScalarTimes(H_RISE, bzScale(bzHodo(bzOf(spokeM[ip])), alpha1)));
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
    start.fieldB = bzAdd(A, bzScalarTimes(H_RISE, bzScale(bzHodo(bzOf(spokeE[i])), alpha1)));
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
    patched: top.patched && bottom.patched ? { edgeGap: Math.max(top.patched.edgeGap, bottom.patched.edgeGap), edgeAngle: Math.max(top.patched.edgeAngle, bottom.patched.edgeAngle) } : null,
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
  if (!(reach >= 0.05 && reach <= 0.95)) return { ok: false, reason: 'reach must be between 0.05 and 0.95 of the way to the junction centre' };
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
  if (!(nrmLen > 1e-12)) return { ok: false, reason: 'the edges are in a line about their centre — there is no junction plane' };
  nrm = bsNorm(nrm);
  const d0 = bsSub(mids[0], centre);
  const e1 = bsNorm(bsSub(d0, bsScale(nrm, bsDot(d0, nrm)))), e2 = bsCross(nrm, e1);
  const order = mids.map((_, i) => i).sort((a, b) => { const da = bsSub(mids[a], centre), db = bsSub(mids[b], centre); return Math.atan2(bsDot(da, e2), bsDot(da, e1)) - Math.atan2(bsDot(db, e2), bsDot(db, e1)); });

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
      arms[k] = skin.srf;
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
    const d0dir = dirOf(order[0]);
    for (let i = 1; i < N; i++) { const k = order[i]; rev[k] = bsDot(dirOf(k), d0dir) < 0; }
    for (let k = 0; k < N; k++) if (flips[k]) rev[k] = !rev[k];
    const ends = order.map((k) => ({ k, start: armEnd(k, rev[k], 0), end: armEnd(k, rev[k], 1) }));
    const spineA = meanOf(ends.map((p) => p.start.point)), spineB = meanOf(ends.map((p) => p.end.point));
    spine = [spineA, spineB];
    const spineCrv = line(spineA, spineB);
    for (const p of ends) {
      const up = railTo(p.end.point, p.end.su, spineB), down = railTo(p.start.point, p.start.su, spineA);
      if (!up || !down) return { ok: false, reason: `no rail could reach the spine from edge ${p.k + 1}` };
      rails.push(up, down);
      try { centreFaces.push(coonsPatch(portCurve(p.k, rev[p.k]), up, reverseCrv(spineCrv), reverseCrv(down))); }
      catch (e) { return { ok: false, reason: `the centre would not fill at edge ${p.k + 1}: ${e.message}` }; }
    }
  } else if (opts.onePatch) {
    // One surface: the gap the edges bound, patched and trimmed, in place of
    // the web of quads — fitted rather than exact.
    const pr = patchFromSurfaceEdges(parts, { maxCount: opts.patchCount });
    if (!pr.ok) return pr;
    centreFaces.push(pr.srf); centreTrims.push(pr.trimLoop); rails.push(...pr.rails);
    spine = [centre, centre];
    patched = { edgeGap: pr.edgeGap, edgeAngle: pr.edgeAngleInner };
  } else {
    const ks = Array.isArray(opts.continuities) ? parts.map((_, kx) => { const c = opts.continuities[kx]; return c === 0 || c === 1 || c === 2 || c === 3 ? c : null; }) : null;
    const cap = blendSrfGuidedCap(access, arc, order, bulge, continuity, nrm, flips, opts.square, ks, opts.waist || null);
    if (!cap.ok) return cap;
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
// with shorter cross-tangents. A closed rim is refused: it has no ends to close
// a boundary with, and whole tubes meet through Blend.
// Returns { ok, srf, trimLoop (u,v), rails (the free edges), edgeGap and
// edgeAngle / edgeAngleInner (measured along the surface edges) } or a refusal.
// The fill's cross-tangent length, as a share of each side's chord to the far
// corners. Measured on a half-tube cross: 0.25 turns within 2% of the domain
// (a fitted net met the tubes at 15 degrees), 0.8 turns over the whole cap
// (2 degrees at 32 x 32), 1.2 folds.
export const PATCH_EDGE_TANGENT_SCALE = 0.8;
export function patchFromSurfaceEdges(parts, opts = {}) {
  if (!Array.isArray(parts) || parts.length < 2) return { ok: false, reason: 'Patch on surfaces needs two or more surface edges' };
  const N = parts.length;
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
  // Each edge runs the ring's way: its end lies ahead of its start about the center.
  const rev = access.map((ea) => bsDot(bsCross(bsSub(ea.point(0), centre), bsSub(ea.point(1), centre)), nrm) < 0);
  const at = (kx, t) => access[kx].frame(arc[kx](rev[kx] ? 1 - t : t), false);
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
    const m = 0.6 * gap * (opts.bulge ?? 1);
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
  let fill = null;
  for (const ts of opts.tangentScale != null ? [opts.tangentScale] : [PATCH_EDGE_TANGENT_SCALE, 0.55, 0.35, 0.2]) {
    fill = nSidedTangentPatch({ boundary: sides.map((sd) => sd.point), tangent: sides.map((sd) => sd.tangent), tangentScale: ts, oriented: true });
    if (fill.ok) break;
  }
  if (!fill.ok) return { ok: false, reason: fill.reason };
  // The fill is exact but not a NURBS surface: it is sampled densely over its
  // own polygon domain, laid in the patch's square, and fitted.
  const V = fill.domain.vertices, pad = 0.03;
  let lx = Infinity, ly = Infinity, hx = -Infinity, hy = -Infinity;
  for (const [x, y] of V) { lx = Math.min(lx, x); ly = Math.min(ly, y); hx = Math.max(hx, x); hy = Math.max(hy, y); }
  const sc = (1 - 2 * pad) / Math.max(hx - lx, hy - ly);
  const toUV = (x, y) => [pad + (x - lx) * sc + (1 - 2 * pad - (hx - lx) * sc) / 2, pad + (y - ly) * sc + (1 - 2 * pad - (hy - ly) * sc) / 2];
  const targets = [], params = [];
  const G = Math.max(24, Math.round(opts.grid ?? 96));
  for (let i = 0; i <= G; i++) for (let j = 0; j <= G; j++) {
    const x = lx + (hx - lx) * i / G, y = ly + (hy - ly) * j / G;
    const q = fill.evaluateXY(x, y);
    if (q) { targets.push(q); params.push(toUV(x, y)); }
  }
  const S = Math.max(8, Math.round(opts.samplesPerEdge ?? 32));
  const trimLoop = [], edgeSamples = [];
  sides.forEach((sd, i) => {
    const A = V[i], B = V[(i + 1) % V.length];
    for (let k = 0; k < S; k++) {
      const t = k / S, x = A[0] + (B[0] - A[0]) * t, y = A[1] + (B[1] - A[1]) * t, uv = toUV(x, y);
      const q = sd.point(t);
      // Twice: the boundary is worth more than any one interior sample.
      targets.push(q, q); params.push(uv, uv);
      trimLoop.push(uv);
      if (sd.kx != null) edgeSamples.push({ point: q, kx: sd.kx, t, uv });
    }
  });
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const q of targets) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], q[k]); hi[k] = Math.max(hi[k], q[k]); }
  const span = bsLen(bsSub(hi, lo));
  const fit = fitPatchToTolerance(targets, { tolerance: opts.tolerance ?? 1e-3 * span, maxCount: opts.maxCount ?? 48, stiffness: opts.stiffness ?? 0.01, degree: 3, uCount: opts.uCount ?? Math.min(32, opts.maxCount ?? 48), params });
  if (!fit.ok) return fit;
  // How well the edges were met: position, and the angle between the patch and
  // each surface along its edge (the fitted normal against the surface's own).
  // The corners are G0 by construction (the fill's own limit), so the angle is
  // also reported along each edge's interior, a tenth in from its ends.
  let gapMax = 0, angMax = 0, angInner = 0;
  for (const e of edgeSamples) {
    const P = surfacePointAndPartials(fit.srf, e.uv[0], e.uv[1]);
    gapMax = Math.max(gapMax, bsLen(bsSub(P.point, e.point)));
    const pn = bsNorm(bsCross(P.su, P.sv));
    const sn = bsNorm(bsCross(edgeTan(e.kx, e.t), at(e.kx, e.t).d1));
    const a = Math.acos(Math.min(1, Math.abs(bsDot(pn, sn)))) * 180 / Math.PI;
    angMax = Math.max(angMax, a);
    if (e.t >= 0.1 && e.t <= 0.9) angInner = Math.max(angInner, a);
  }
  return { ok: true, srf: fit.srf, trimLoop, rails: rails.filter(Boolean), order, edgeGap: gapMax, edgeAngle: angMax, edgeAngleInner: angInner, maxDeviation: fit.maxDeviation, count: fit.uCount, tried: fit.tried };
}
