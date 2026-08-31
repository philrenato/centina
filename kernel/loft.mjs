// LOFT — the spec: "ordered section curves; seam/direction
// arrows shown at each closed section with click-to-flip. Options:
// Style=Normal/Straight/Loose, Closed=Yes/No." This is v1: a genuine
// SMOOTH tensor-product NURBS surface (P&T 9.2.5 "Global Surface
// Interpolation," the 2D analog of curve.mjs's own Global Curve
// Interpolation, A9.1) skinned through N ordered section curves — but with
// an honest, stated simplification in place of the real thing's seam-
// alignment UI and the Normal/Straight/Loose style choices: cross-sections
// correspond by RELATIVE PARAMETER FRACTION (uniform samples of each
// curve's own domain), not by a user-adjustable seam pick, and there is
// only ONE style (not three). Closed=Yes/No (looping the last section back
// to the first) is also not built yet. All stated honestly, not silently
// dropped — matching this kernel's other v1-scoped simplifications (SubD's
// deferred ToNURBS, Boolean's SDF-mesh fallback).
//
// The correspondence-by-fraction rule means sections with very different
// shapes/point distributions can loft oddly at the seam — a real, known
// limitation, not a bug; Rhino's own seam-alignment arrows exist precisely
// because this correspondence problem has no single universally-right
// answer. A future round can let the user drag each section's own seam
// marker; ships the honest default.

import { curvePoint, closestPointOnCurve, reverseCurve } from './curve.mjs';
import { chordLengthParams, averagingKnotVector, interpAtParams } from './interpolate.mjs';
import { surfacePoint } from './surface.mjs';
import { extractSubCurve } from './knots.mjs';

// SURFACE REBUILD — surfaces need to
// be rebuildable just like curves, in both directions with varying
// degrees. resettable" — the surface analog of SketchCurve's own
// already-shipped point-count/degree Rebuild. Unlike a curve's own Rebuild
// (which decimates/inserts among a stored working POINT SET, kernel/
// simplify.mjs), a surface has no equivalent "list of picks" to resample —
// so this refits the surface's OWN CURRENT SHAPE (whatever it is right
// now, from any construction) through a fresh uCount x vCount parameter
// grid, reusing networkCorrectionSurface's own already-proven two-pass
// global-interpolation machinery wholesale rather than inventing a second
// grid-fit routine. Exact at every one of the uCount*vCount grid stations
// (the same "a global interpolation reproduces its own data exactly"
// guarantee this file already relies on for loft()/gordonNetworkSurface),
// a close, smooth approximation of the true surface in between — an
// honest "high-density resample," not a claim of exact shape preservation
// everywhere (Circle's own Rebuild is exact everywhere specifically
// because a circle's closed-form construction allows it; a general
// Revolve/Extrude/Loft/Sweep1 result has no such shortcut).

// sectionCurves: ordered array of 2+ NurbsCrv ({degree, knots, ctrlPts}),
// the loft's own V direction, in pick order. uSampleCount: how densely
// each section is resampled to build the shared U (profile) direction —
// higher is smoother/costlier, no different in spirit from Revolve's own
// uRes/vRes display-tessellation knobs, except this one bakes INTO the
// stored control net itself (there is no separate "exact analytic profile"
// to fall back on for an arbitrary hand-picked curve the way Revolve has
// its own exact profile.degree/knots to reuse directly).
export function loft(sectionCurves, uSampleCount = 24, degU = 3, degV = 3) {
  if (!Array.isArray(sectionCurves) || sectionCurves.length < 2) throw new Error('loft: expected an array of at least two section curves');
  const n = sectionCurves.length;
  if (n < 2) throw new Error('loft needs at least 2 section curves');
  const dV = Math.min(degV, n - 1);
  const dU = Math.min(degU, uSampleCount - 1);

  // grid[i][j] = a sample point on section j, at the SAME relative
  // parameter fraction i/(uSampleCount-1) of section j's own domain.
  const grid = [];
  for (let i = 0; i < uSampleCount; i++) {
    const t = i / (uSampleCount - 1);
    grid.push(sectionCurves.map((crv) => {
      const u0 = crv.knots[0], u1 = crv.knots[crv.knots.length - 1];
      return curvePoint(crv, u0 + t * (u1 - u0));
    }));
  }

  // Shared U parametrization (P&T 9.2.5): average the chord-length
  // parameters of every V-row (one row per section), so every row shares
  // the SAME knotsU — a real tensor-product surface, not n independent
  // curves that happen to sit side by side.
  const ubarSum = new Array(uSampleCount).fill(0);
  for (let j = 0; j < n; j++) {
    const col = grid.map((row) => [...row[j], 1]);
    const u = chordLengthParams(col);
    for (let i = 0; i < uSampleCount; i++) ubarSum[i] += u[i];
  }
  const ubar = ubarSum.map((s) => s / n);
  const knotsU = averagingKnotVector(ubar, dU);

  // Shared V parametrization: the same idea, across the sections, for
  // every U sample row.
  const vbarSum = new Array(n).fill(0);
  for (let i = 0; i < uSampleCount; i++) {
    const v = chordLengthParams(grid[i].map((p) => [...p, 1]));
    for (let j = 0; j < n; j++) vbarSum[j] += v[j];
  }
  const vbar = vbarSum.map((s) => s / uSampleCount);
  const knotsV = averagingKnotVector(vbar, dV);

  // Pass 1 — per section (column j): interpolate its uSampleCount points
  // along U at the shared ubar/knotsU.
  const R = Array.from({ length: uSampleCount }, () => new Array(n));
  for (let j = 0; j < n; j++) {
    const col = grid.map((row) => [...row[j], 1]);
    const ctrl = interpAtParams(col, dU, ubar, knotsU);
    for (let i = 0; i < uSampleCount; i++) R[i][j] = ctrl[i];
  }
  // Pass 2 — per U row i: interpolate its n R points along V at the
  // shared vbar/knotsV. Final control net.
  const ctrlNet = R.map((row) => interpAtParams(row, dV, vbar, knotsV));

  return { degU: dU, knotsU, degV: dV, knotsV, ctrlNet };
}

// RULED LOFT — lofting between two same-vertex-count
// polygons (e.g. two Star-mode Polygons) with loft() below is precisely
// the wrong tool whenever the two curves' own sharp corners ARE the
// point: loft() is exact for the reason it exists (a genuine smooth
// NURBS skin, honestly approximating cross-section correspondence by
// relative parameter fraction), but a smooth tensor-product fit rounds
// every sharp corner over instead of preserving it.
//
// For two SAME-vertex-count polygons (open or closed, matched 1:1 by
// index — no relative-parameter-fraction guessing needed here, unlike
// loft()'s own honest simplification, since polygon vertices already
// correspond exactly by construction), the honest, EXACT construction is
// N independent ruled (bilinear, degree-1 x degree-1) panels, one per
// corresponding edge pair — flat by construction (2 straight lines),
// reproducing every input vertex and every input edge exactly, never
// smoothing a single corner. This is the SAME ruled-surface identity
// extrude() (primitives.mjs) already uses (a straight line between two
// corresponding points, degree-1 in the ruled direction) — applied per
// polygon EDGE here instead of per whole-profile translate.
//
// pointsA/pointsB: ordered arrays of N world points ([x,y,z], plain
// arrays — same convention as every other kernel entry point), each
// either both OPEN or both CLOSED (the caller's own job to have already
// matched — see the app layer's loftRuledEligible). closed=true: edge i
// connects pointsA[i] to pointsA[(i+1)%N] (the last edge wraps back to
// vertex 0); closed=false: edge i connects i to i+1, for i in 0..N-2 (no
// wraparound edge). Re-validates its own inputs defensively (never
// actually reachable with mismatched length from the app layer's own
// eligibility gate, but a kernel function should refuse honestly on its
// own, not just trust its caller — this project's own standing rule).
export function ruledLoftPanels(pointsA, pointsB, closed) {
  const n = pointsA.length;
  if (pointsB.length !== n) throw new Error(`ruledLoftPanels: pointsA (${n} points) and pointsB (${pointsB.length} points) must have the same vertex count`);
  if (n < 2) throw new Error('ruledLoftPanels needs at least 2 vertices per profile');
  const edgeCount = closed ? n : n - 1;
  const panels = [];
  for (let i = 0; i < edgeCount; i++) {
    const j = (i + 1) % n;
    panels.push({
      degU: 1, knotsU: [0, 0, 1, 1],
      degV: 1, knotsV: [0, 0, 1, 1],
      ctrlNet: [
        [[...pointsA[i], 1], [...pointsA[j], 1]],
        [[...pointsB[i], 1], [...pointsB[j], 1]],
      ],
    });
  }
  return panels;
}

// SUPER SWEEP TIER C — the real Gordon/network-surface construction,
// fully scoped: S(u,v) =
// Lu(u,v) + Lv(u,v) - T(u,v), where Lu = loft(rails) (the U-family),
// Lv = loft(profiles) evaluated with u/v swapped (the V-family), and T is
// a small correction surface built from the SAME two-pass interpAtParams
// technique loft() itself uses, fed the n-by-m grid of curve-family
// station "near-intersections" directly instead of resampling curves.
//
// networkCorrectionSurface is that correction-surface builder — genuinely
// no new algorithmic risk, just loft()'s own Pass-1/Pass-2 machinery
// generalized from "resample n curves into a grid" to "here is the grid
// already, at EXPLICIT parameters, not internally re-derived by chord
// length." `grid[j][i]` = a 3D point ([x,y,z]) at (uParams[j], vParams[i])
// — j indexes the U-family stations (0..m-1, m = uParams.length), i
// indexes the V-family stations (0..n-1, n = vParams.length) — matching
// loft()'s own ctrlNet[uIndex][vIndex] convention exactly.
export function networkCorrectionSurface(grid, uParams, vParams, degU = 3, degV = 3) {
  const m = uParams.length, n = vParams.length;
  if (m < 2 || n < 2) throw new Error('networkCorrectionSurface needs at least 2 stations in each direction');
  const dU = Math.min(degU, m - 1);
  const dV = Math.min(degV, n - 1);
  const knotsU = averagingKnotVector(uParams, dU);
  const knotsV = averagingKnotVector(vParams, dV);
  // Pass 1 — for each V-station i, interpolate the m U-stations' own data
  // at the shared uParams/knotsU.
  const R = Array.from({ length: m }, () => new Array(n));
  for (let i = 0; i < n; i++) {
    const col = grid.map((row) => [...row[i], 1]);
    const ctrl = interpAtParams(col, dU, uParams, knotsU);
    for (let j = 0; j < m; j++) R[j][i] = ctrl[j];
  }
  // Pass 2 — for each U-control-index j, interpolate across the n
  // V-stations at the shared vParams/knotsV. Final control net.
  const ctrlNet = R.map((row) => interpAtParams(row, dV, vParams, knotsV));
  return { degU: dU, knotsU, degV: dV, knotsV, ctrlNet };
}

// refitSurfaceUV — the actual Surface Rebuild entry point (see the header
// comment above this file's own import block). Samples `srf` (ANY valid
// NurbsSrf — rational or not, any degree/knots) at a uniform uCount x
// vCount grid of DOMAIN-FRACTION parameters (0..1 in each direction,
// mapped onto srf's own real knot domain) via the already-proven
// `surfacePoint`, then feeds that grid straight into
// networkCorrectionSurface — no different, mechanically, from feeding it
// a Gordon surface's own P_ij near-intersection grid, except every grid
// point here is already an EXACT point of the real input surface (not a
// near-intersection stand-in), so the exactness guarantee is stronger:
// the returned surface reproduces `srf`'s own true value at all
// uCount*vCount grid stations exactly, not just approximately.
// A REBUILT CLOSED SURFACE USED TO COME BACK CREASED. The plain path below
// interpolates the sample grid with a CLAMPED knot vector in each direction,
// which makes the first and last rows independent of one another. On a closed
// surface the sample at fraction 1 is the same point as the sample at 0, so
// closure survives as a coincidence of position and nothing at all constrains
// the tangent across it: the rebuilt surface is C0 at its seam. Measured on a
// natively revolved sphere rebuilt to 16x16 degree 3 — the seam turned 4.91
// degrees against an interior control of 0.0007, and the un-rebuilt original
// read 0.0002 at the same place. So it was Rebuild that introduced the crease,
// not closure itself.
//
// The cure is the surface analog of closedCurveInterp, and the same one:
// sample the closed direction at DISTINCT stations (the endpoint duplicate is
// dropped), wrap-pad by the degree at both ends so the solver sees a genuinely
// periodic sequence, interpolate, then keep the middle. What comes back is
// tangent-continuous across the seam because the interpolation never saw a
// boundary there.
//
// Extraction is exact rather than resampled. splitSurface is the wrong tool
// twice over: it refuses a closed direction by name, and it works by
// re-sampling, which would reintroduce approximation error into the very thing
// being corrected. Knot insertion to degree+1 is exact on a CLAMPED curve, and
// the padded interpolation is clamped at its own extended ends while the range
// being cut out sits strictly inside — so the cut moves nothing.
function seamClosedIn(srf, dir) {
  const net = srf.ctrlNet;
  const same = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-9;
  if (dir === 'u') return net[0].every((p, j) => same(p, net[net.length - 1][j]));
  return net.every((row) => same(row[0], row[row.length - 1]));
}
// Cut [a,b] out of one direction by running the curve-side extraction along
// every control row that shares that direction's knot vector. Each row yields
// the identical knot vector (insertion depends only on knots and parameters),
// so the rows reassemble into a surface without further reconciliation.
// THE CUT MUST LAND ON THE KNOT IT IS MEANT TO BE, BIT FOR BIT. Both sub-range
// boundaries are station parameters, and every station is already a knot — but
// the boundary is computed as pad/span while the knot came out of averaging the
// same parameters, and those two routes to the same real number do not always
// produce the same double. Measured on a torus: the v cut wanted
// 0.800000000000000044 while the knot present was 0.800000000000000155, one ulp
// apart. insertKnotToMultiplicity matches within 1e-9, so it inserts at the
// requested value anyway and leaves a span 1.1e-16 wide between the two — and
// the pair of control points spanning a sliver collapses onto each other. That
// is the whole defect: |dS/dv| fell from 218 to 0.013 at that end, the surface
// still positionally correct but with a stalled parameterisation. The u
// direction escaped only because its own arithmetic happened to land bit-exact.
function snapCutToKnot(knots, t) {
  const scale = Math.max(1, Math.abs(knots[knots.length - 1] - knots[0]));
  let best = t, bestD = Infinity;
  for (const k of knots) { const dd = Math.abs(k - t); if (dd < bestD) { bestD = dd; best = k; } }
  return bestD <= 1e-9 * scale ? best : t;
}
function extractSurfaceRange(srf, dir, a, b) {
  const rows = dir === 'v' ? srf.ctrlNet : srf.ctrlNet[0].map((_, j) => srf.ctrlNet.map((r) => r[j]));
  const degree = dir === 'v' ? srf.degV : srf.degU;
  const knots = dir === 'v' ? srf.knotsV : srf.knotsU;
  a = snapCutToKnot(knots, a); b = snapCutToKnot(knots, b);
  let outKnots = null;
  const outRows = rows.map((row) => {
    const sub = extractSubCurve({ degree, knots, ctrlPts: row.map((p) => p.slice()) }, a, b);
    outKnots = sub.knots;
    return sub.ctrlPts;
  });
  if (dir === 'v') return { ...srf, knotsV: outKnots, ctrlNet: outRows };
  return { ...srf, knotsU: outKnots, ctrlNet: outRows[0].map((_, j) => outRows.map((r) => r[j])) };
}
// Make a closed direction's two end rows literally identical, at their average.
function tieSeam(srf, dir) {
  const net = srf.ctrlNet.map((r) => r.map((p) => p.slice()));
  if (dir === 'u') {
    const a = net[0], b = net[net.length - 1];
    for (let j = 0; j < a.length; j++) {
      for (let k = 0; k < a[j].length; k++) { const m = (a[j][k] + b[j][k]) / 2; a[j][k] = m; b[j][k] = m; }
    }
  } else {
    for (const row of net) {
      const a = row[0], b = row[row.length - 1];
      for (let k = 0; k < a.length; k++) { const m = (a[k] + b[k]) / 2; a[k] = m; b[k] = m; }
    }
  }
  return { ...srf, ctrlNet: net };
}
export function refitSurfaceUV(srf, uCount, vCount, degU = 3, degV = 3) {
  if (uCount < 2 || vCount < 2) throw new Error('refitSurfaceUV needs at least 2 points in each direction');
  const dU = Math.min(degU, uCount - 1);
  const dV = Math.min(degV, vCount - 1);
  const u0 = srf.knotsU[0], u1 = srf.knotsU[srf.knotsU.length - 1];
  const v0 = srf.knotsV[0], v1 = srf.knotsV[srf.knotsV.length - 1];
  const closedU = seamClosedIn(srf, 'u'), closedV = seamClosedIn(srf, 'v');
  // Stations in a closed direction are i/n over n DISTINCT samples and are then
  // padded outward by the degree; in an open one they are the ordinary
  // i/(count-1) spanning the domain end to end.
  // interpAtParams/averagingKnotVector both require a [0,1]-normalized params
  // array, so the padded stations are normalized across the WHOLE padded span
  // rather than left running from -deg/n to (n+deg)/n. The real surface is then
  // the sub-range covering stations 0..n of that span, which is where lo/hi
  // come from. Sampling still uses the wrapped fraction in the ORIGINAL
  // surface's own domain, which normalization does not touch.
  const stations = (count, deg, closed) => {
    if (!closed) return { params: Array.from({ length: count }, (_, i) => i / (count - 1)), pick: (i) => i / (count - 1) };
    // PADDED BY 2*degree, AND THE PAD IS A TOLERANCE, NOT A THRESHOLD. The wrap
    // makes the solver see a periodic sequence; what remains is the clamped
    // padded END leaking inward. That leak is not confined to degree+1 control
    // points — interpolation is global — it decays geometrically through the
    // inverse at the Euler-Frobenius rate, which for cubics is 2-sqrt(3) ~
    // 0.268 per station. That predicts 4.91 deg (no wrap) * 0.268^3 = 0.094 at
    // a pad of degree, and it is what was measured; at twice the pad the
    // predicted residual falls below the noise floor, measured 0.0011, equal to
    // the interior control. So no finite pad is ever exact — and since higher
    // degrees decay SLOWER, 2*degree is not guaranteed to suffice forever. The
    // exact construction is a genuine periodic solve (uniform unclamped knots,
    // a cyclic banded system, control points wrapped rather than duplicated,
    // then clamped for storage) — what OpenNURBS and Open CASCADE do. That is
    // the right eventual replacement for this whole padding machinery.
    const pad = 2 * deg, n = count, span = n + 2 * pad;
    const params = [], pick = [];
    for (let i = -pad; i <= n + pad; i++) { params.push((i + pad) / span); pick.push((((i % n) + n) % n) / n); }
    return { params, pick: (i) => pick[i], lo: pad / span, hi: (n + pad) / span, padded: true };
  };
  const su = stations(uCount, dU, closedU), sv = stations(vCount, dV, closedV);
  const grid = su.params.map((_, i) => sv.params.map((__, j) =>
    surfacePoint(srf, u0 + su.pick(i) * (u1 - u0), v0 + sv.pick(j) * (v1 - v0))));
  let out = networkCorrectionSurface(grid, su.params, sv.params, dU, dV);
  if (su.padded) out = extractSurfaceRange(out, 'u', su.lo, su.hi);
  if (sv.padded) out = extractSurfaceRange(out, 'v', sv.lo, sv.hi);
  // WHERE THIS SURFACE IS EXACT, as fractions of its own domain — exposed
  // because the answer differs by direction and a caller cannot infer it.
  // An open direction is exact at i/(count-1), spanning end to end. A CLOSED
  // one is exact at i/count, because the station at fraction 1 is the same
  // point as the one at 0 and spending a station on the duplicate is what
  // creased the seam in the first place. Verified: a rebuilt radius-20
  // cylinder is exact to 1.4e-14 at i/count and 1.0e-2 off it, so a caller
  // assuming the open convention on a closed surface measures BETWEEN
  // stations and reads a real-looking error that is only its own sampling.
  const fractions = (count, closed) => (closed
    ? Array.from({ length: count + 1 }, (_, i) => i / count)
    : Array.from({ length: count }, (_, i) => i / (count - 1)));
  out.stationFractions = { u: fractions(uCount, closedU), v: fractions(vCount, closedV) };
  return out;
}

// GORDON NETWORK SURFACE — the full Tier C construction, orchestrating
// Lu/Lv/T and the doc's own "dense-sample combine, then ONE final global
// interpolation" recipe.
//
// STATIONING (v1, with no real curve-curve
// intersection in this kernel): each PROFILE's rail-direction station
// u_j is found via closestPointOnCurve against a REPRESENTATIVE rail
// (rails[0]) — the exact technique sweepNProfiles already uses to
// station its own cross-sections, expressed as a relative fraction of
// rails[0]'s own domain (matching loft()'s own "relative parameter
// fraction" correspondence convention). Each RAIL's own V-station v_i is
// simply its uniform index fraction i/(n-1) — rails have no independent
// "closest point against a curve" mechanism to station them by (only
// profiles do); this matches loft()'s own V direction being fundamentally
// ORDER-based across a small discrete family, not geometrically searched.
//
// THE GRID / EXACTNESS, made concrete: at
// each station (u_j, v_i), define P_ij as the MIDPOINT of what the two
// independently-built families already give there — Lu(u_j,v_i) and
// Lv(u_j,v_i) — an honest "near-intersection" target since this kernel
// has no real curve-curve intersection (a TRUE Gordon surface would have
// Lu(u_j,v_i) = Lv(u_j,v_i) = P_ij already, because P_ij is a genuine
// shared point both curve families actually pass through; ours only
// station NEAR each other, so Lu(u_j,v_i) and Lv(u_j,v_i) are merely
// close, and the midpoint is the honest v1 stand-in for their true
// meeting point). T_srf is built (networkCorrectionSurface above) FROM
// this exact P_ij grid at these exact (u_j,v_i) parameters — so, by
// interpAtParams' own global-interpolation guarantee, T(u_j,v_i) = P_ij
// EXACTLY, at every station, unconditionally. Algebraically this makes
// S_sample(u_j,v_i) = Lu(u_j,v_i) + Lv(u_j,v_i) - T(u_j,v_i) =
// Lu(u_j,v_i) + Lv(u_j,v_i) - [Lu(u_j,v_i)+Lv(u_j,v_i)-P_ij] = P_ij,
// EXACTLY, at every one of the n*m grid stations — REGARDLESS of what
// Lu(u_j,v_i) and Lv(u_j,v_i) individually happen to be. This is the same
// exactness this kernel already proves for loft()/sweepNProfiles (a
// global interpolation reproduces its own data exactly at its own data
// parameters), applied one level higher: T's own data here IS the
// station grid, not resampled curve points.
//
// THE FINAL SURFACE: a DENSE (u,v) sample grid — every station u_j/v_i
// exactly, plus `interiorSamplesPerSpan` genuinely in-between samples per
// gap (sweepNProfiles' own established pattern, reused not reinvented) —
// is built, S_sample(u,v) = Lu(u,v) + Lv(u,v) - T(u,v) (a plain vector
// add/subtract of 3D points, never combined as control nets) is evaluated
// at every dense sample, and ONE final global interpolation (the same
// two-pass technique, a third time) through this dense grid produces the
// returned NurbsSrf. Because every station u_j/v_i is included in the
// dense grid exactly, and S_sample is proven exact AT those stations
// above, the final surface reproduces every one of the n*m grid stations
// exactly too (the same "final re-interpolation reproduces every dense
// sample exactly" guarantee loft()/sweepNProfiles already rely on) — NOT
// the input curves' full continuous extent between samples, the same
// honest limitation loft()/sweepNProfiles already state for their own
// correspondence-by-sampling.
const NETWORK_INTERIOR_SAMPLES_PER_SPAN = 6;
function curveCentroid(crv) {
  const sum = crv.ctrlPts.reduce((acc, [x, y, z]) => [acc[0] + x, acc[1] + y, acc[2] + z], [0, 0, 0]);
  return sum.map((s) => s / crv.ctrlPts.length);
}
export function denseWithInterior(stations, interiorSamplesPerSpan) {
  const dense = [];
  for (let k = 0; k < stations.length; k++) {
    dense.push(stations[k]);
    if (k < stations.length - 1) {
      const a = stations[k], b = stations[k + 1];
      for (let s = 1; s <= interiorSamplesPerSpan; s++) {
        const t = s / (interiorSamplesPerSpan + 1);
        dense.push(a + (b - a) * t);
      }
    }
  }
  return dense;
}
export function gordonNetworkSurface(rails, profiles, opts = {}) {
  const n = rails.length, m = profiles.length;
  if (n < 2) throw new Error('gordonNetworkSurface needs at least 2 rail curves (use loft() for 0 profiles, sweep1Rigid/sweepNProfiles for 1 rail)');
  if (m < 2) throw new Error('gordonNetworkSurface needs at least 2 profile curves: a Gordon surface is defined by two transverse curve FAMILIES, and one profile is not a family — use loft() or sweep1Rigid() for a single section');
  const uSampleCount = opts.uSampleCount ?? 24;
  const vSampleCount = opts.vSampleCount ?? 24;
  const degU = opts.degU ?? 3, degV = opts.degV ?? 3;
  const interiorSamplesPerSpan = opts.interiorSamplesPerSpan ?? NETWORK_INTERIOR_SAMPLES_PER_SPAN;

  const Lu_srf = loft(rails, uSampleCount, degU, degV);
  const Lv_srf = loft(profiles, vSampleCount, degU, degV);

  // Rail stations — order-based, matching loft()'s own V-direction
  // convention (no independent curve to search against).
  const vStations = n === 1 ? [0] : rails.map((_, i) => i / (n - 1));

  let uStations, profileOrder;
  if (opts.uStations) {
    // EXPLICIT STATION OVERRIDE (boundSurfaceFromLoop's own use)
    // — for a caller that ALREADY KNOWS the true correspondence (profiles
    // genuinely touching a rail at known parameter fractions — an EdgeSrf-
    // style closed 4-curve loop — not merely stationed NEAR it), this skips
    // the closest-point search below entirely. That search stations by a
    // profile's own CENTROID (curveCentroid), which is a real, deliberate
    // choice for the "profile merely runs near the rail" case this
    // function's own header comment scopes — but checked directly, not
    // assumed, a profile's centroid is generically NOT at its own touching
    // endpoint, so it would silently produce an approximate, not exact,
    // station even when the two curves genuinely touch exactly at a known
    // parameter. `profiles` must already be given in the SAME order as
    // `opts.uStations` (unlike the auto-search path below, which re-sorts
    // by inferred station regardless of pick order) — the caller owns that
    // correspondence here, this override trusts it rather than re-deriving
    // it a second, redundant way.
    if (opts.uStations.length !== m) throw new Error(`gordonNetworkSurface: opts.uStations must have exactly one entry per profile (${m}), got ${opts.uStations.length}`);
    uStations = opts.uStations.slice();
    profileOrder = profiles.map((_, idx) => idx);
  } else {
    // Profile stations — closestPointOnCurve against a representative rail
    // (rails[0]), expressed as a relative fraction of its domain.
    const r0 = rails[0];
    const r0u0 = r0.knots[0], r0u1 = r0.knots[r0.knots.length - 1];
    const uStationsRaw = profiles.map((profile, idx) => {
      const hit = closestPointOnCurve(r0, curveCentroid(profile));
      if (hit.ambiguous) {
        throw new Error(`gordonNetworkSurface: profile ${idx + 1}'s closest point on rail 1 is AMBIGUOUS (nearly equidistant from rail parameters ${hit.u.toFixed(6)} and ${hit.ambiguousWith.toFixed(6)}) — rail 1 passes near itself more than once, so there is no single honest station for this profile`);
      }
      return { idx, frac: Math.max(0, Math.min(1, (hit.u - r0u0) / (r0u1 - r0u0))) };
    });
    // Sort by station (a user can plausibly pick profiles out of rail order,
    // matching sweepNProfiles' own precedent) and refuse a near-duplicate.
    const ordered = uStationsRaw.slice().sort((a, b) => a.frac - b.frac);
    for (let i = 1; i < ordered.length; i++) {
      if (ordered[i].frac - ordered[i - 1].frac < 1e-6) {
        throw new Error(`gordonNetworkSurface: profile ${ordered[i - 1].idx + 1} and profile ${ordered[i].idx + 1} both station at (nearly) the same rail-direction position — two cross-sections can't occupy the same station; reposition one of them`);
      }
    }
    uStations = ordered.map((s) => s.frac);
    profileOrder = ordered.map((s) => s.idx); // original profile index, in station order
  }

  // The P_ij grid (honest "near-intersection" midpoint — see header
  // comment) and the correction surface T built from it.
  const grid = uStations.map((u) => vStations.map((v) => {
    const pu = surfacePoint(Lu_srf, u, v);
    const pv = surfacePoint(Lv_srf, v, u);
    return [(pu[0] + pv[0]) / 2, (pu[1] + pv[1]) / 2, (pu[2] + pv[2]) / 2];
  }));
  const T_srf = networkCorrectionSurface(grid, uStations, vStations, degU, degV);

  // Dense (u,v) sample grid — every station exactly, plus interior samples.
  const uDense = denseWithInterior(uStations, interiorSamplesPerSpan);
  const vDense = denseWithInterior(vStations, interiorSamplesPerSpan);
  const dU2 = Math.min(degU, uDense.length - 1);
  const dV2 = Math.min(degV, vDense.length - 1);
  const knotsU2 = averagingKnotVector(uDense, dU2);
  const knotsV2 = averagingKnotVector(vDense, dV2);

  const sampleGrid = uDense.map((u) => vDense.map((v) => {
    const lu = surfacePoint(Lu_srf, u, v);
    const lv = surfacePoint(Lv_srf, v, u);
    const t = surfacePoint(T_srf, u, v);
    return [lu[0] + lv[0] - t[0], lu[1] + lv[1] - t[1], lu[2] + lv[2] - t[2]];
  }));
  const R2 = Array.from({ length: uDense.length }, () => new Array(vDense.length));
  for (let b = 0; b < vDense.length; b++) {
    const col = sampleGrid.map((row) => [...row[b], 1]);
    const ctrl = interpAtParams(col, dU2, uDense, knotsU2);
    for (let a = 0; a < uDense.length; a++) R2[a][b] = ctrl[a];
  }
  const ctrlNet = R2.map((row) => interpAtParams(row, dV2, vDense, knotsV2));

  return {
    degU: dU2, knotsU: knotsU2, degV: dV2, knotsV: knotsV2, ctrlNet,
    // Exposed for verification/ghost display — mirrors sweepNProfiles' own
    // `stations`/`ubar` exposure.
    uStations, vStations, profileOrder,
    Lu_srf, Lv_srf, T_srf,
  };
}

// BOUND SURFACE (Rhino: EdgeSrf; our own name "Bound Surface"). It is a
// Coons patch built from N boundary curves, and it shares the N-rail Sweep
// family's machinery rather than carrying its own: a surface bounded by exactly 4
// curves forming a closed loop, in order (c0.end==c1.start==...==
// c3.end==c0.start, within CLOSE_LOOP_TOL) — the classic Rhino EdgeSrf
// "golden path." v1 SCOPE, stated honestly: exactly 4 edges only. A
// 2-edge "bound surface" is already just `loft([c0,c1])`, not a distinct
// feature; a 3-edge triangular patch needs a genuinely different
// degenerate-corner construction (one side collapsing to a point) this
// round doesn't attempt — a real, separate follow-up if ever wanted.
//
// THE CONSTRUCTION IS LITERALLY gordonNetworkSurface WITH 2 RAILS + 2
// PROFILES, opposite edges paired and reversed so both members of each
// pair run the SAME direction (loft()'s own V-direction convention,
// same reasoning `sweep1Rigid`'s frame-continuity already relies on):
// rails = [c0, reverse(c2)] (c0 runs c0.start->c0.end; c2 runs backward
// around the loop, so reversing it makes it run the SAME way, start-
// side to end-side); profiles = [reverse(c3), c1] (c3 connects c2's end
// back to c0's start, so reversed it runs c0-side to c2-side, matching
// c1 which already runs c0-side to c2-side directly).
//
// A REAL, PROVABLE EXACTNESS RESULT FOR THIS SPECIFIC 2-RAIL/2-PROFILE
// CASE, worked out by hand THEN checked by direct numerical probe before
// trusting it (a first version of this module's own test compared the
// wrong thing and looked like a real bug until diagnosed — see below).
// With only 2 rails, loft(rails)'s own V-direction is degree-1 (a
// straight ruled blend), so at the v=0 station (rail 0's own row),
// Lu(u,0) = c0(u) EXACTLY for every u (loft's own domain-end exactness,
// not just at v-station corners) — the same is true of Lv(0,u)
// restricted to that same edge: with only 2 profiles, profiles' own
// "which-profile" direction is ALSO degree-1, so Lv(0,u) traces the
// exact STRAIGHT CHORD between the loop's two corner points at that
// edge as u sweeps 0..1. T(u,0) is built from a station grid whose own
// u-direction is likewise only 2 points wide, so T(u,0) is ALSO the
// same straight chord between the identical two corner points. Lv(0,u)
// and T(u,0) are therefore the SAME degree-1 curve through the SAME two
// points — they cancel EXACTLY, leaving the CONTINUOUS combination
// S_sample(u,0) = Lu(u,0) + 0 = c0(u) exactly, for every u.
//
// THE ACTUAL RETURNED SURFACE, stated honestly, promises SHAPE exactness,
// not parametrization exactness — a real, found-via-testing distinction,
// not a rounding error. gordonNetworkSurface's own FINAL surface is a
// fresh global re-interpolation THROUGH a dense (u,v) sample grid of
// S_sample (this kernel's own already-documented "reproduces its data
// points exactly, approximates continuously in between" limitation,
// stated in this same function's header above) — so the RETURNED edge
// curve traces the identical 3D PATH as c0 (confirmed directly: every
// sampled point on the returned edge lands within ~1e-4mm of c0 via
// closestPointOnCurve, at any density), but its own internal parameter
// no longer runs at exactly c0's own original speed (a denser/different-
// degree re-interpolation has no reason to reproduce the SAME arc-length-
// vs-parameter mapping, only the same shape). A first draft of this
// module's own test compared "same raw parameter fraction" on both
// curves and saw a spurious ~0.5mm gap that LOOKED like a real bug —
// re-diagnosed via closestPointOnCurve before trusting it, which showed
// the true shape-match error is actually ~1e-4mm; fixed the test to
// compare shape (closest point), not parametrization speed.
const CLOSE_LOOP_TOL = 0.001; // mm — matches this app's own JOIN_TOLERANCE value (not imported; this module has no app-layer dependency)
export function boundSurfaceFromLoop(c0, c1, c2, c3) {
  const loop = [c0, c1, c2, c3];
  for (let i = 0; i < 4; i++) {
    const cur = loop[i], next = loop[(i + 1) % 4];
    const curEnd = curvePoint(cur, cur.knots[cur.knots.length - 1]);
    const nextStart = curvePoint(next, next.knots[0]);
    const gap = Math.hypot(curEnd[0] - nextStart[0], curEnd[1] - nextStart[1], curEnd[2] - nextStart[2]);
    if (gap > CLOSE_LOOP_TOL) {
      throw new Error(`boundSurfaceFromLoop: edge ${i + 1} and edge ${(i + 1) % 4 + 1} don't share an endpoint (${gap.toFixed(4)}mm apart) — the 4 curves must form a closed loop, in order`);
    }
  }
  const rails = [c0, reverseCurve(c2)];
  const profiles = [reverseCurve(c3), c1];
  // uStations FORCED to the exact [0,1] this construction's own exactness
  // proof (above) depends on — gordonNetworkSurface's own default auto-
  // stationing (closestPointOnCurve against a profile's CENTROID) does NOT
  // give exactly 0/1 even for genuinely touching curves (checked directly,
  // not assumed — an earlier draft of this function's own test caught a
  // real ~0.5mm deviation from relying on the default path instead).
  return gordonNetworkSurface(rails, profiles, { uStations: [0, 1] });
}
