// SPLIT — the first, narrowly-scoped piece of the trimmed-surface work:
// the recommended "no new tessellation machinery" next
// step: cut a surface into two independent pieces along one of its own
// ISOCURVES (a fixed u or v parameter), rather than an arbitrary hand-
// picked trim curve. This deliberately sidesteps the hard, still-unbuilt
// general trimmed-surface tessellation problem
// ("a real, separate, serious robustness project") — each resulting piece
// still tessellates over its own ordinary RECTANGULAR parameter sub-
// domain, exactly like every other Surface this kernel already produces;
// no boundary-conforming triangulation is needed at all.
//
// Reuses loft.mjs's networkCorrectionSurface wholesale (the SAME "dense
// grid of explicit (param, point) samples -> one global-interpolation
// solve" technique Gordon/loft already established) rather than real
// knot-insertion/refinement (P&T A5.1/A5.3 — still genuinely unbuilt in
// this kernel, named as deferred since the very first Circle-rebuild
// round). Each resulting half is therefore an HONEST re-derivation, not
// a literal knot-inserted sub-piece of the original control net — exact
// AT every one of its own dense sample stations (the same guarantee
// loft()/gordonNetworkSurface() already prove and rely on), not
// necessarily bit-identical to what real knot insertion would produce
// between samples. Stated plainly, not silently assumed.
//
// THE SHARED-BOUNDARY EXACTNESS PROOF (the one property specific to
// splitting, not inherited for free from Gordon/loft): both halves are
// built from the IDENTICAL cross-direction params/degree/knots array and
// the IDENTICAL boundary grid row (both evaluated via surfacePoint(srf,
// ...splitParam...) at the SAME cross samples). networkCorrectionSurface's
// own cross-direction interpolation is a deterministic linear solve
// purely of (data, degree, params, knots), and a clamped B-spline's own
// END control points/rows coincide exactly with their own end DATA points
// (the standard clamped-knot-vector property this kernel already relies
// on throughout) — so both halves' shared boundary control-point ROW is
// byte-identical by construction, a real "no gap" guarantee, not just
// "looks close." Verified directly in test/split.test.mjs.
import { surfacePoint, surfaceClosure } from './surface.mjs';
import { networkCorrectionSurface } from './loft.mjs';
import { extractSubCurve } from './knots.mjs';

function linspace(a, b, n) {
  if (n === 1) return [a];
  const out = [];
  for (let i = 0; i < n; i++) out.push(a + (b - a) * (i / (n - 1)));
  return out;
}

// A REAL, PREVIOUSLY-LATENT PRECONDITION, found while building this
// function, not previously documented anywhere: `interpolate.mjs`'s
// `averagingKnotVector` hardcodes its own CLAMPED boundary knots to the
// literal values 0 and 1 (P&T Eq 9.8's own textbook formula, which is
// only ever correct when the fed `ubar` array is ALREADY normalized to
// [0,1] — true of every existing caller in this kernel: globalCurveInterp/
// loft's own chordLengthParams always returns [0,1]; gordonNetworkSurface's
// own uStations/vStations are explicitly built as "relative fractions" in
// [0,1]). This function is the FIRST caller to want a surface's own REAL,
// non-[0,1] native domain (e.g. a revolve's V-direction spans [0, narcs],
// arc-span count, not radians and not a fraction) — passing that directly
// into networkCorrectionSurface/interpAtParams silently produced a knot
// vector whose own end knots (hardcoded to 1) didn't match the actual
// data parameters (which can run well past 1), corrupting `findSpan`'s
// span lookup and producing a singular linear system (NaN control points)
// at higher sample counts — reproduced and root-caused directly (isolated
// down to `interpAtParams`+`averagingKnotVector` alone, confirmed the SAME
// point data with [0,1]-domain params solves cleanly at any sample count
// tested). Not a bug in interpolate.mjs itself (its own real callers never
// violate this precondition) — fixed here instead: normalize every params
// array to [0,1] before calling networkCorrectionSurface, then AFFINELY
// RESCALE the returned knot vector's numeric values back into the real
// domain afterward (a B-spline's shape is invariant under an affine
// reparametrization of its own knot vector — only the numbers labeling
// each parameter change, never the control points/geometry), so the
// resulting piece's own domain still means the same real thing the
// original surface's did, matching this app's general expectation that a
// Surface's stored knots reflect meaningful, pickable parameter values
// (ExtractIsocurve, ghost-preview stationing, etc. all read real domain
// values directly).
function rescaleKnots(knots, oldMin, oldMax, newMin, newMax) {
  const span = oldMax - oldMin;
  return knots.map((k) => newMin + ((k - oldMin) / span) * (newMax - newMin));
}

// direction: 'u' or 'v'. splitParam: the parameter VALUE (in the
// surface's own domain, not a 0-1 fraction) to cut at — must be strictly
// interior, not at or beyond either end (a split exactly at an existing
// boundary is a no-op, refused honestly rather than silently returning a
// degenerate zero-extent piece). A direction that's CLOSED (a seam, per
// surfaceClosure) is also refused honestly — cutting a closed loop at one
// parameter unrolls it into one open piece, not two; a real, separate,
// not-yet-built case, needing two split parameters.
//
// opts.sampleCount: dense samples PER PIECE along the split direction
// (default 12). opts.crossSampleCount: dense samples along the OTHER
// (unsplit) direction (default 16, matching Gordon/loft's own defaults)
// — shared, unchanged, between both resulting pieces, which is exactly
// what the shared-boundary exactness proof above depends on.
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

  // Builds one half by sampling the ORIGINAL surface over [loParam,
  // hiParam] in the split direction, calling networkCorrectionSurface
  // with its (uParams,vParams) argument order matched to this surface's
  // OWN real U/V roles directly — no post-hoc transpose needed, since
  // networkCorrectionSurface's returned {degU,knotsU,degV,knotsV,ctrlNet}
  // is keyed exactly by whichever params array was passed as uParams/
  // vParams, in that same order. Both the split-direction and cross-
  // direction params are normalized to [0,1] before the solve (see
  // rescaleKnots above), then the returned knot vectors are rescaled back
  // into this piece's own real domain afterward.
  // `stations` on each returned half is the real domain (u,v) grid this
  // piece is EXACT at, by construction — the same "exact at the true
  // stations, an honest smooth approximation in between" limitation
  // loft()/gordonNetworkSurface() already state and test for; exposed
  // here (mirroring their own `stations`/`ubar` exposure) so a caller —
  // or a verify script — never has to guess the internal sample grid to
  // check exactness correctly.
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

/* ─────────────────────────────────────────────────────────────────────────────
   SPLITTING AT A CREASE THE SURFACE ALREADY HAS.

   Everything above re-derives each half by sampling and re-fitting, which is
   the honest answer for an arbitrary isoparametric cut. It is the WRONG answer
   for the cut this section makes, and it also refuses the case that needs it.

   A knot whose multiplicity reaches the degree is a C0 line: the surface is
   only positionally continuous across it, the control net already carries a
   full row of coincident points there, and the two sides are genuinely separate
   patches that happen to be stored in one array. Cutting there needs no fitting
   at all — knot insertion to degree+1 isolates the sub-range EXACTLY (see
   extractSubCurve's own note on why that step is exact for clamped input), so
   each piece is a literal sub-net of the original rather than a re-derivation.

   It also handles the case splitSurface above refuses outright. An extruded
   closed profile is CLOSED in u, and cutting a closed direction at ONE
   parameter would unroll it into a single open piece rather than two. Cutting
   it at ALL of its creases is a different operation with a well-defined answer:
   N creases give N open pieces, none of which is closed.

   Why this exists: an extrude produces ONE side surface wrapping the whole
   profile, so a hexagonal prism's six vertical edges are creases INSIDE a
   single face. Nothing downstream that reasons about a pair of faces — a
   dihedral angle, a rolling-ball fillet, an edge classification — can see them
   at all. Splitting here is what turns them into edges.
   ───────────────────────────────────────────────────────────────────────────── */

/** The interior knot values whose multiplicity reaches `degree` — the creases.
 *  A knot at multiplicity below the degree is a smooth (C1 or better) join and
 *  is deliberately NOT reported: splitting there would manufacture an edge
 *  where the surface has none. */
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

/** ⚠ FULL MULTIPLICITY IS A CANDIDATE, NOT A CREASE. A knot at multiplicity ==
 *  degree makes the BASIS only C0 there; whether the SURFACE actually kinks
 *  depends on the control net. The standard NURBS circle is the counterexample
 *  that matters: degree 2 with knots at multiplicity 2 at every quarter point,
 *  and perfectly smooth across all of them, because the control legs meeting at
 *  each junction are collinear. Trusting multiplicity alone shatters every
 *  cylinder into four faces and invents eight edges that do not exist.
 *
 *  So the combinatorial candidates are filtered by MEASURING the tangent break,
 *  at several stations across the surface — a crease can be sharp at one end of
 *  an edge and fade to nothing at the other, and one sample in the middle would
 *  miss exactly that case. */
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
      /* ⚠ A POLE MAKES NOISE LOOK LIKE A TANGENT, and `> 0` does not catch it.
         A flat circular cap collapses a whole control row to its center, so the
         one-sided differences there are not a direction at all. Built at the
         ORIGIN they come out exactly 0 and a `> 0` test skips them; built at
         x = 400 the same degenerate row differs by float noise at that
         magnitude, the test passes, and the angle between two noise vectors is
         uniformly random — so every cap sprouted three creases and a cylinder
         reported 24 edges instead of 2, depending only on WHERE it was built.
         A real tangent step is ~scale*1e-6 here and noise is ~scale*1e-16, so
         a floor three decades above the noise separates them cleanly. */
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

  // Each control-net LINE running along the split direction is an ordinary
  // curve over the same knot vector, so the surface is cut by cutting every one
  // of them over the same sub-range. They all receive the identical sequence of
  // insertions, so they come back sharing one knot vector — asserted rather
  // than assumed, because a silent disagreement here would build a net whose
  // rows mean different things.
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
