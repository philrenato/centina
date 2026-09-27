// Split — cut a surface into two independent pieces along one of its own
// isocurves (a fixed u or v parameter), rather than along an arbitrary trim
// curve. Each resulting piece tessellates over its own rectangular parameter
// sub-domain, like every other Surface this kernel produces, so no
// boundary-conforming triangulation is needed.
//
// Uses loft.mjs's networkCorrectionSurface (the "dense grid of explicit
// (param, point) samples -> one global-interpolation solve" technique of
// Gordon/loft) rather than knot insertion (P&T A5.1/A5.3, in knots.mjs). Each
// half is therefore a re-derivation, not a literal knot-inserted sub-piece of
// the original control net: exact at every one of its dense sample stations
// (the guarantee loft()/gordonNetworkSurface() give), not necessarily
// identical to what knot insertion would produce between samples.
//
// Shared-boundary exactness (the property specific to splitting): both halves
// are built from the identical cross-direction params/degree/knots array and
// the identical boundary grid row (both evaluated via surfacePoint(srf,
// ...splitParam...) at the same cross samples). networkCorrectionSurface's
// cross-direction interpolation is a deterministic linear solve of (data,
// degree, params, knots), and a clamped B-spline's end control points/rows
// coincide exactly with its end data points — so both halves' shared boundary
// control-point row is byte-identical by construction: no gap. Tested in
// test/split.test.mjs.
import { surfacePoint, surfaceClosure } from './surface.mjs';
import { networkCorrectionSurface } from './loft.mjs';
import { extractSubCurve } from './knots.mjs';

function linspace(a, b, n) {
  if (n === 1) return [a];
  const out = [];
  for (let i = 0; i < n; i++) out.push(a + (b - a) * (i / (n - 1)));
  return out;
}

// Precondition: `interpolate.mjs`'s `averagingKnotVector` hardcodes its
// clamped boundary knots to the values 0 and 1 (P&T Eq 9.8, correct only when
// the `ubar` array is normalized to [0,1] — true of its other callers:
// chordLengthParams returns [0,1], and gordonNetworkSurface's
// uStations/vStations are fractions in [0,1]). This function works in a
// surface's native domain, which need not be [0,1] (e.g. a revolve's
// V-direction spans [0, narcs], an arc-span count). Passing that directly to
// networkCorrectionSurface/interpAtParams gives a knot vector whose end knots
// (1) do not match the data parameters (which can run past 1), corrupting
// `findSpan` and producing a singular system (NaN control points) at higher
// sample counts. So every params array is normalized to [0,1] before calling
// networkCorrectionSurface, and the returned knot vector is affinely rescaled
// back into the native domain afterward (a B-spline's shape is invariant under
// an affine reparametrization of its knot vector). The piece's domain then
// means what the original surface's did, which consumers that read domain
// values directly (ExtractIsocurve, ghost-preview stationing) rely on.
function rescaleKnots(knots, oldMin, oldMax, newMin, newMax) {
  const span = oldMax - oldMin;
  return knots.map((k) => newMin + ((k - oldMin) / span) * (newMax - newMin));
}

// direction: 'u' or 'v'. splitParam: the parameter value (in the surface's
// own domain, not a 0-1 fraction) to cut at — must be strictly interior (a
// split at an existing boundary is a no-op, refused rather than returning a
// zero-extent piece). A closed direction (a seam, per surfaceClosure) is also
// refused — cutting a closed loop at one parameter unrolls it into one open
// piece, not two; that case needs two split parameters and is not
// implemented.
//
// opts.sampleCount: dense samples per piece along the split direction
// (default 12). opts.crossSampleCount: dense samples along the other
// (unsplit) direction (default 16, as in Gordon/loft) — shared, unchanged,
// between both resulting pieces, which the shared-boundary exactness above
// depends on.
export function splitSurface(srf, direction, splitParam, opts = {}) {
  if (direction !== 'u' && direction !== 'v') throw new Error(`splitSurface: direction must be 'u' or 'v', got ${direction}`);
  const knots = direction === 'u' ? srf.knotsU : srf.knotsV;
  const domainMin = knots[0], domainMax = knots[knots.length - 1];
  const eps = (domainMax - domainMin) * 1e-6;
  if (!Number.isFinite(splitParam) || splitParam <= domainMin + eps || splitParam >= domainMax - eps) {
    throw new Error(`splitSurface: split parameter ${splitParam} must be strictly interior to the ${direction}-direction domain [${domainMin}, ${domainMax}]`);
  }
  const { closedU, closedV } = surfaceClosure(srf);
  if ((direction === 'u' && closedU) || (direction === 'v' && closedV)) {
    throw new Error(`splitSurface: the ${direction}-direction is CLOSED (a seam, not a free edge) — splitting a closed direction at one parameter would unroll it into one open piece, not two; not supported yet`);
  }

  const sampleCount = opts.sampleCount ?? 12;
  const crossSampleCount = opts.crossSampleCount ?? 16;
  const degSplit = Math.min(opts.degSplit ?? 3, sampleCount - 1);
  const degCross = Math.min(opts.degCross ?? 3, crossSampleCount - 1);

  const crossKnots = direction === 'u' ? srf.knotsV : srf.knotsU;
  const crossMin = crossKnots[0], crossMax = crossKnots[crossKnots.length - 1];
  const crossParams = linspace(crossMin, crossMax, crossSampleCount);
  // See the rescaleKnots comment above: interpAtParams/averagingKnotVector
  // require a [0,1]-normalized params array — this is shared, unchanged,
  // between both halves (same reasoning as crossParams itself).
  const crossParamsNorm = crossParams.map((c) => (c - crossMin) / (crossMax - crossMin));

  // Builds one half by sampling the original surface over [loParam,
  // hiParam] in the split direction, calling networkCorrectionSurface
  // with its (uParams,vParams) argument order matched to this surface's
  // U/V roles directly — no transpose needed, since
  // networkCorrectionSurface's returned {degU,knotsU,degV,knotsV,ctrlNet}
  // is keyed by whichever params array was passed as uParams/vParams, in
  // that order. Both the split-direction and cross-direction params are
  // normalized to [0,1] before the solve (see rescaleKnots above), then the
  // returned knot vectors are rescaled back into this piece's domain.
  // `stations` on each returned half is the domain (u,v) grid this piece is
  // exact at, by construction — the same "exact at the stations, a smooth
  // approximation in between" limitation loft()/gordonNetworkSurface()
  // state and test; exposed (as their `stations`/`ubar` are) so a caller can
  // check exactness without guessing the internal sample grid.
  function buildHalf(loParam, hiParam) {
    const splitParams = linspace(loParam, hiParam, sampleCount);
    const splitParamsNorm = splitParams.map((s) => (s - loParam) / (hiParam - loParam));
    let built;
    if (direction === 'u') {
      const grid = splitParams.map((u) => crossParams.map((v) => surfacePoint(srf, u, v)));
      built = networkCorrectionSurface(grid, splitParamsNorm, crossParamsNorm, degSplit, degCross);
      return {
        degU: built.degU, knotsU: rescaleKnots(built.knotsU, 0, 1, loParam, hiParam),
        degV: built.degV, knotsV: rescaleKnots(built.knotsV, 0, 1, crossMin, crossMax),
        ctrlNet: built.ctrlNet, uStations: splitParams, vStations: crossParams,
      };
    }
    const grid = crossParams.map((u) => splitParams.map((v) => surfacePoint(srf, u, v)));
    built = networkCorrectionSurface(grid, crossParamsNorm, splitParamsNorm, degCross, degSplit);
    return {
      degU: built.degU, knotsU: rescaleKnots(built.knotsU, 0, 1, crossMin, crossMax),
      degV: built.degV, knotsV: rescaleKnots(built.knotsV, 0, 1, loParam, hiParam),
      ctrlNet: built.ctrlNet, uStations: crossParams, vStations: splitParams,
    };
  }

  const first = buildHalf(domainMin, splitParam);
  const second = buildHalf(splitParam, domainMax);
  return { first, second, direction, splitParam, crossParams };
}

/* Splitting at a crease the surface already has.

   Everything above re-derives each half by sampling and re-fitting, which
   suits an arbitrary isoparametric cut. It is the wrong answer for the cut
   this section makes, and splitSurface refuses the case that needs it.

   A knot whose multiplicity reaches the degree is a C0 line: the surface is
   only positionally continuous across it, the control net already carries a
   full row of coincident points there, and the two sides are separate patches
   stored in one array. Cutting there needs no fitting — knot insertion to
   degree+1 isolates the sub-range exactly (see extractSubCurve's note on why
   that step is exact for clamped input), so each piece is a literal sub-net of
   the original rather than a re-derivation.

   It also handles the case splitSurface refuses. An extruded closed profile is
   closed in u, and cutting a closed direction at one parameter would unroll it
   into a single open piece rather than two. Cutting it at all of its creases
   is a different operation with a well-defined answer: N creases give N open
   pieces, none of which is closed.

   Purpose: an extrude produces one side surface wrapping the whole profile, so
   a hexagonal prism's six vertical edges are creases inside a single face.
   Nothing downstream that reasons about a pair of faces — a dihedral angle, a
   rolling-ball fillet, an edge classification — can see them. Splitting here
   turns them into edges. */

/** The interior knot values whose multiplicity reaches `degree` — the creases.
 *  A knot at multiplicity below the degree is a smooth (C1 or better) join and
 *  is not reported: splitting there would manufacture an edge where the
 *  surface has none. */
export function c0KnotParams(knots, degree, tol = 1e-9) {
  const out = [];
  const last = knots.length - degree - 1;
  let i = degree + 1;
  while (i < last) {
    const u = knots[i];
    let m = 0;
    while (i + m < knots.length && Math.abs(knots[i + m] - u) <= tol) m++;
    if (m >= degree) out.push(u);
    i += m;
  }
  return out;
}

/** Full multiplicity is a candidate, not a crease. A knot at multiplicity ==
 *  degree makes the basis only C0 there; whether the surface kinks depends on
 *  the control net. The standard NURBS circle is the counterexample: degree 2
 *  with knots at multiplicity 2 at every quarter point, and smooth across all
 *  of them, because the control legs meeting at each junction are collinear.
 *  Trusting multiplicity alone would split every cylinder into four faces with
 *  eight edges that do not exist.
 *
 *  So the combinatorial candidates are filtered by measuring the tangent break
 *  at several stations across the surface — a crease can be sharp at one end
 *  of an edge and fade to nothing at the other, which one sample in the middle
 *  would miss. */
export function surfaceCreaseParams(srf, direction, opts = {}) {
  const alongU = direction === 'u';
  const deg = alongU ? srf.degU : srf.degV;
  const knots = alongU ? srf.knotsU : srf.knotsV;
  const angleTol = opts.angleTolRad ?? (0.5 * Math.PI / 180);
  const stations = opts.stations ?? 5;
  const cand = c0KnotParams(knots, deg, opts.tol ?? 1e-9);
  if (!cand.length) return [];
  const crossKnots = alongU ? srf.knotsV : srf.knotsU;
  const c0 = crossKnots[0], c1 = crossKnots[crossKnots.length - 1];
  const span = knots[knots.length - 1] - knots[0];
  const step = span * 1e-6;
  // The surface's own size, so the noise floor below is scale-free: the same
  // shape modeled in millimeters and in meters must answer identically.
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const row of srf.ctrlNet) for (const pt of row) for (let d = 0; d < 3; d++) { if (pt[d] < lo[d]) lo[d] = pt[d]; if (pt[d] > hi[d]) hi[d] = pt[d]; }
  const scale = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), Math.hypot(lo[0], lo[1], lo[2]), 1e-9);
  const floor = scale * 1e-12;
  const out = [];
  for (const w of cand) {
    let broke = false;
    for (let k = 0; k < stations && !broke; k++) {
      const c = c0 + (c1 - c0) * (stations === 1 ? 0.5 : k / (stations - 1));
      const at = (t) => (alongU ? surfacePoint(srf, t, c) : surfacePoint(srf, c, t));
      // A one-sided difference on each side: the derivative is discontinuous at
      // the knot itself, so both samples must stay strictly off it.
      const before = at(w - step), beforeIn = at(w - 2 * step);
      const after = at(w + step), afterOut = at(w + 2 * step);
      const dIn = [before[0] - beforeIn[0], before[1] - beforeIn[1], before[2] - beforeIn[2]];
      const dOut = [afterOut[0] - after[0], afterOut[1] - after[1], afterOut[2] - after[2]];
      const lIn = Math.hypot(dIn[0], dIn[1], dIn[2]), lOut = Math.hypot(dOut[0], dOut[1], dOut[2]);
      /* A pole makes noise look like a tangent, and `> 0` does not catch it.
         A flat circular cap collapses a whole control row to its center, so
         the one-sided differences there are not a direction. Built at the
         origin they come out exactly 0 and a `> 0` test skips them; built at
         x = 400 the same degenerate row differs by float noise at that
         magnitude, the test passes, and the angle between two noise vectors
         is random — spurious creases that depend only on where the surface
         was built. A tangent step is ~scale*1e-6 here and noise is
         ~scale*1e-16, so a floor three decades above the noise separates
         them. */
      if (!(lIn > floor) || !(lOut > floor)) continue;
      const cosA = (dIn[0] * dOut[0] + dIn[1] * dOut[1] + dIn[2] * dOut[2]) / (lIn * lOut);
      if (Math.acos(Math.max(-1, Math.min(1, cosA))) > angleTol) broke = true;
    }
    if (broke) out.push(w);
  }
  return out;
}

/** Split `srf` at every C0 line in one direction. Returns the pieces in
 *  parameter order — exactly one element (the input, untouched) when the
 *  surface has no crease in that direction, so a caller can apply it
 *  unconditionally. Exact: knot insertion only, no sampling and no fitting. */
export function splitSurfaceAtC0Lines(srf, direction, opts = {}) {
  if (direction !== 'u' && direction !== 'v') throw new Error(`splitSurfaceAtC0Lines: direction must be 'u' or 'v', got ${direction}`);
  const tol = opts.tol ?? 1e-9;
  const alongU = direction === 'u';
  const deg = alongU ? srf.degU : srf.degV;
  const knots = alongU ? srf.knotsU : srf.knotsV;
  const params = opts.allCandidates ? c0KnotParams(knots, deg, tol) : surfaceCreaseParams(srf, direction, opts);
  if (!params.length) return [srf];
  const cuts = [knots[0], ...params, knots[knots.length - 1]];

  // Each control-net line running along the split direction is an ordinary
  // curve over the same knot vector, so the surface is cut by cutting every one
  // of them over the same sub-range. They all receive the identical sequence of
  // insertions, so they come back sharing one knot vector — asserted, because a
  // disagreement here would build a net whose rows mean different things.
  const net = srf.ctrlNet;
  const nCross = alongU ? net[0].length : net.length;
  const lineAt = (c) => (alongU ? net.map((row) => row[c]) : net[c].slice());

  const pieces = [];
  for (let s = 0; s + 1 < cuts.length; s++) {
    const a = cuts[s], b = cuts[s + 1];
    let sharedKnots = null;
    const lines = [];
    for (let c = 0; c < nCross; c++) {
      const sub = extractSubCurve({ degree: deg, knots, ctrlPts: lineAt(c) }, a, b);
      if (!sharedKnots) sharedKnots = sub.knots;
      else if (sub.knots.length !== sharedKnots.length) throw new Error('splitSurfaceAtC0Lines: control lines disagreed on the extracted knot vector');
      lines.push(sub.ctrlPts);
    }
    const ctrlNet = alongU
      ? lines[0].map((_, i) => lines.map((ln) => ln[i].slice()))
      : lines.map((ln) => ln.map((pt) => pt.slice()));
    pieces.push(alongU
      ? { degU: deg, knotsU: sharedKnots, degV: srf.degV, knotsV: srf.knotsV.slice(), ctrlNet }
      : { degU: srf.degU, knotsU: srf.knotsU.slice(), degV: deg, knotsV: sharedKnots, ctrlNet });
  }
  return pieces;
}
