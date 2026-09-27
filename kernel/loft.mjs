// Loft: a smooth tensor-product NURBS surface skinned through N ordered
// section curves by global surface interpolation (P&T 9.2.5 "Global Surface
// Interpolation," the 2D analog of curve.mjs's Global Curve Interpolation,
// A9.1).
//
// Known limitations: cross-sections correspond by relative parameter
// fraction (uniform samples of each curve's own domain), not by a
// user-adjustable seam pick; there is one style, not Normal/Straight/Loose;
// Closed (looping the last section back to the first) is not built. Sections
// with very different shapes or point distributions can loft oddly at the
// seam, because correspondence by fraction has no single right answer.

import { curvePoint, closestPointOnCurve, reverseCurve } from './curve.mjs';
import { chordLengthParams, averagingKnotVector, interpAtParams } from './interpolate.mjs';
import { surfacePoint } from './surface.mjs';
import { extractSubCurve } from './knots.mjs';

// Surface rebuild (refitSurfaceUV, below), in both directions with any
// degree. A curve's Rebuild resamples a stored working point set
// (kernel/simplify.mjs); a surface has no such set, so this refits the
// surface's current shape, from any construction, through a fresh
// uCount x vCount parameter grid, using networkCorrectionSurface's two-pass
// global interpolation. The result is exact at every one of the
// uCount*vCount grid stations (a global interpolation reproduces its own
// data) and a close, smooth approximation in between; it does not preserve
// shape exactly everywhere.

// sectionCurves: ordered array of 2+ NurbsCrv ({degree, knots, ctrlPts}),
// the loft's V direction, in pick order. uSampleCount: how densely each
// section is resampled to build the shared U (profile) direction; higher is
// smoother and costlier. The samples are baked into the control net, since
// an arbitrary curve has no exact analytic profile to reuse the way
// Revolve reuses profile.degree/knots.
export function loft(sectionCurves, uSampleCount = 24, degU = 3, degV = 3) {
  if (!Array.isArray(sectionCurves) || sectionCurves.length < 2) throw new Error('loft: expected an array of at least two section curves');
  const n = sectionCurves.length;
  if (n < 2) throw new Error('loft needs at least 2 section curves');
  const dV = Math.min(degV, n - 1);
  const dU = Math.min(degU, uSampleCount - 1);

  // grid[i][j] = a sample point on section j, at the same relative
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
  // the same knotsU and the result is one tensor-product surface.
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

// Ruled loft. loft() rounds over every sharp corner, because a smooth
// tensor-product fit cannot hold one. For two polygons with the same vertex
// count (open or closed, matched 1:1 by index, so no correspondence by
// parameter fraction is needed), the exact construction is N ruled
// (bilinear, degree-1 x degree-1) panels, one per corresponding edge pair,
// reproducing every input vertex and edge exactly. This is the ruled-surface
// identity extrude() (primitives.mjs) uses, applied per polygon edge instead
// of per whole-profile translate.
//
// pointsA/pointsB: ordered arrays of N world points ([x,y,z]), both open or
// both closed (the caller matches them; see the app layer's
// loftRuledEligible). closed=true: edge i connects pointsA[i] to
// pointsA[(i+1)%N]; closed=false: edge i connects i to i+1, for i in
// 0..N-2. Inputs are re-validated here rather than trusted to the caller.
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

// Network (Gordon) surface: S(u,v) = Lu(u,v) + Lv(u,v) - T(u,v), where
// Lu = loft(rails) (the U family), Lv = loft(profiles) evaluated with u/v
// swapped (the V family), and T is a correction surface built from the n-by-m
// grid of curve-family station near-intersections.
//
// networkCorrectionSurface builds T with loft()'s two-pass interpolation,
// from a grid given at explicit parameters instead of one resampled from
// curves and parametrized by chord length. `grid[j][i]` = a 3D point
// ([x,y,z]) at (uParams[j], vParams[i]) — j indexes the U-family stations
// (0..m-1, m = uParams.length), i indexes the V-family stations (0..n-1,
// n = vParams.length) — matching loft()'s ctrlNet[uIndex][vIndex] convention.
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

// refitSurfaceUV: the Surface Rebuild entry point (see the note above the
// imports). Samples `srf` (any valid NurbsSrf, rational or not, any
// degree/knots) at a uniform uCount x vCount grid of domain fractions, mapped
// onto srf's knot domain, and interpolates that grid with
// networkCorrectionSurface. Every grid point is a true point of `srf`, so the
// result reproduces `srf` exactly at all uCount*vCount stations.
//
// Closed directions. Interpolating with a clamped knot vector makes the first
// and last rows independent of one another. On a closed surface the sample at
// fraction 1 is the same point as the sample at 0, so closure survives only as
// a coincidence of position and nothing constrains the tangent across it: the
// result is C0 at its seam (a revolved sphere rebuilt to 16x16 degree 3 turns
// 4.91 degrees there, against 0.0007 in the interior).
//
// The remedy is the surface analog of closedCurveInterp: sample the closed
// direction at distinct stations (the endpoint duplicate is dropped),
// wrap-pad by the degree at both ends so the solver sees a periodic sequence,
// interpolate, then keep the middle. The result is tangent-continuous across
// the seam because the interpolation never saw a boundary there.
//
// Extraction is exact rather than resampled. splitSurface is the wrong tool
// twice over: it refuses a closed direction by name, and it works by
// re-sampling, which would reintroduce approximation error into the very thing
// being corrected. Knot insertion to degree+1 is exact on a clamped curve, and
// the padded interpolation is clamped at its own extended ends while the range
// being cut out sits strictly inside, so the cut moves nothing.
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
// The cut must land exactly on the knot it targets. Both sub-range
// boundaries are station parameters, and every station is already a knot, but
// the boundary is computed as pad/span while the knot comes from averaging the
// same parameters, and the two routes can differ by one ulp (on a torus,
// 0.800000000000000044 against a knot of 0.800000000000000155).
// insertKnotToMultiplicity matches within 1e-9, so it would insert at the
// requested value and leave a span 1.1e-16 wide; the control points spanning
// that sliver collapse onto each other and the parameterization stalls at that
// end (|dS/dv| 218 → 0.013) while position stays correct. So the cut snaps to
// any knot within 1e-9 of the domain scale.
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
export function refitSurfaceUV(srf, uCount, vCount, degU = 3, degV = 3) {
  if (uCount < 2 || vCount < 2) throw new Error('refitSurfaceUV needs at least 2 points in each direction');
  const dU = Math.min(degU, uCount - 1);
  const dV = Math.min(degV, vCount - 1);
  const u0 = srf.knotsU[0], u1 = srf.knotsU[srf.knotsU.length - 1];
  const v0 = srf.knotsV[0], v1 = srf.knotsV[srf.knotsV.length - 1];
  const closedU = seamClosedIn(srf, 'u'), closedV = seamClosedIn(srf, 'v');
  // Stations in a closed direction are i/n over n distinct samples and are then
  // padded outward by the degree; in an open one they are the ordinary
  // i/(count-1) spanning the domain end to end.
  // interpAtParams/averagingKnotVector both require a [0,1]-normalized params
  // array, so the padded stations are normalized across the whole padded span
  // rather than left running from -deg/n to (n+deg)/n. The real surface is then
  // the sub-range covering stations 0..n of that span, which is where lo/hi
  // come from. Sampling still uses the wrapped fraction in the original
  // surface's own domain, which normalization does not touch.
  const stations = (count, deg, closed) => {
    if (!closed) return { params: Array.from({ length: count }, (_, i) => i / (count - 1)), pick: (i) => i / (count - 1) };
    // The pad is 2*degree, and it is a tolerance, not a threshold. The wrap
    // makes the solver see a periodic sequence; what remains is the clamped
    // padded end leaking inward. Interpolation is global, so the leak is not
    // confined to degree+1 control points: it decays geometrically through the
    // inverse at the Euler-Frobenius rate, 2-sqrt(3) ~ 0.268 per station for
    // cubics. A pad of degree leaves 4.91 deg * 0.268^3 = 0.094 deg at the seam;
    // a pad of 2*degree leaves 0.0011, equal to the interior control. No finite
    // pad is exact, and higher degrees decay more slowly, so 2*degree is not
    // guaranteed to suffice for them. Known limitation: the exact construction
    // is a periodic solve (uniform unclamped knots, a cyclic banded system,
    // control points wrapped rather than duplicated, then clamped for storage),
    // as OpenNURBS and Open CASCADE do; it would replace this padding.
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
  // Where this surface is exact, as fractions of its own domain, exposed
  // because the answer differs by direction and a caller cannot infer it. An
  // open direction is exact at i/(count-1), end to end. A closed one is exact
  // at i/count, because the station at fraction 1 is the same point as the one
  // at 0 and spending a station on the duplicate creases the seam. A caller
  // assuming the open convention on a closed surface measures between stations
  // (a rebuilt radius-20 cylinder is exact to 1.4e-14 at i/count and 1.0e-2
  // off it).
  const fractions = (count, closed) => (closed
    ? Array.from({ length: count + 1 }, (_, i) => i / count)
    : Array.from({ length: count }, (_, i) => i / (count - 1)));
  out.stationFractions = { u: fractions(uCount, closedU), v: fractions(vCount, closedV) };
  return out;
}

// Gordon network surface: builds Lu, Lv and T, combines them on a dense
// sample grid, then runs one final global interpolation.
//
// Stationing is by closest point, not by curve-curve intersection. Each
// profile's rail-direction station u_j is closestPointOnCurve of the
// profile's centroid against a representative rail (rails[0]), as a fraction
// of rails[0]'s domain — the technique sweepNProfiles uses to station its
// cross-sections. Each rail's station v_i is its index fraction i/(n-1):
// rails have no curve to be searched against, and loft()'s V direction is
// order-based across a small discrete family.
//
// The grid: at each station (u_j, v_i), P_ij is the midpoint of Lu(u_j,v_i)
// and Lv(u_j,v_i). In a true Gordon surface both families pass through a
// shared P_ij; here they only station near each other, and the midpoint
// stands in for their meeting point. T_srf (networkCorrectionSurface) is
// interpolated from this P_ij grid at these (u_j,v_i), so T(u_j,v_i) = P_ij
// exactly, and S_sample(u_j,v_i) = Lu(u_j,v_i) + Lv(u_j,v_i) - P_ij = P_ij,
// since P_ij is their midpoint, at every one of the n*m stations.
//
// The final surface: S_sample(u,v) = Lu(u,v) + Lv(u,v) - T(u,v) (a vector
// sum of evaluated 3D points, never of control nets) is evaluated on a dense
// (u,v) grid containing every station u_j/v_i plus `interiorSamplesPerSpan`
// samples per gap (the sweepNProfiles pattern), and one final two-pass global
// interpolation through that grid is the returned NurbsSrf. It reproduces
// every station exactly; between samples it approximates the input curves'
// continuous extent, as loft() and sweepNProfiles do.
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
    // Explicit stations (boundSurfaceFromLoop). For a caller that knows the
    // true correspondence (profiles touching a rail at known parameter
    // fractions, as in a closed 4-curve loop), this skips the closest-point
    // search below. That search stations by a profile's centroid
    // (curveCentroid), which is generically not at its touching endpoint, so it
    // gives an approximate station even when the curves touch exactly.
    // `profiles` must be in the same order as `opts.uStations`; unlike the
    // search path, which re-sorts by station, this path trusts the caller's
    // order.
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

  // The P_ij grid (the near-intersection midpoint; see the header comment)
  // and the correction surface T built from it.
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

// Bound surface (Rhino: EdgeSrf). A Coons patch from 4 boundary curves
// forming a closed loop, in order (c0.end==c1.start==...==c3.end==c0.start,
// within CLOSE_LOOP_TOL), built with the N-rail Sweep family's machinery.
// Known limitation: exactly 4 edges. A 2-edge bound surface is
// `loft([c0,c1])`; a 3-edge triangular patch needs a degenerate-corner
// construction (one side collapsing to a point) that is not built.
//
// The construction is gordonNetworkSurface with 2 rails and 2 profiles,
// opposite edges paired and reversed so both members of each pair run the
// same direction (loft()'s V-direction convention): rails = [c0, reverse(c2)]
// (c2 runs backward around the loop, so reversed it runs start-side to
// end-side like c0); profiles = [reverse(c3), c1] (c3 connects c2's end back
// to c0's start, so reversed it runs c0-side to c2-side, as c1 does).
//
// Edge exactness for this 2-rail/2-profile case. With 2 rails, loft(rails) is
// degree 1 in V, so at v=0, Lu(u,0) = c0(u) for every u (loft's domain-end
// exactness). With 2 profiles, Lv's which-profile direction is also degree 1,
// so Lv(0,u) is the straight chord between the loop's two corner points on
// that edge. T(u,0), built from a station grid 2 points wide in u, is the same
// chord between the same two points. They cancel, so
// S_sample(u,0) = c0(u) exactly, for every u.
//
// The returned surface is exact in shape, not in parameterization.
// gordonNetworkSurface's final surface is a fresh global interpolation
// through a dense (u,v) sample grid of S_sample, so its edge traces c0's path
// (every sampled point within ~1e-4 mm of c0 by closestPointOnCurve, at any
// density) but does not run at c0's parameter speed. Compare such edges by
// closest point, not by equal parameter fraction, which reads a spurious
// ~0.5 mm gap.
const CLOSE_LOOP_TOL = 0.001; // mm — equals the app's JOIN_TOLERANCE (not imported: the kernel has no app-layer dependency)
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
  // uStations forced to [0, 1], which the edge-exactness argument above
  // depends on: the default search stations by a profile's centroid and does
  // not give exactly 0 and 1 even for curves that touch.
  return gordonNetworkSurface(rails, profiles, { uStations: [0, 1] });
}
