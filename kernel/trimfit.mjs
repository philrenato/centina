// THE EXACT FORM OF A TRIMMED FACE'S BOUNDARY
// ================================================================
// Turns one face's (u,v) trim loop — a polyline, which is what marching and
// splitting produce — into the runs of real NURBS curves a B-rep trim loop is
// made of, each with the 3-D edge curve that goes with it.
//
// WHY THIS IS NOT JUST `fitCurveToPoints` CALLED ONCE. A single smooth curve
// cannot represent a boundary that has a corner in it. Asked for a square at a
// 1e-3 bound the fitter refuses and says so — closest 3.1e-1 at its full
// control-point ceiling — which is the right answer, because no spline of any
// control-point count rounds a right angle to within a thousandth. A loop must
// therefore be SPLIT at its corners first, one run per smooth stretch. That is
// also the edge structure a B-rep is supposed to have: a corner of a face
// boundary is a vertex, and the stretches between vertices are its edges.
//
// ⚠ A SMOOTH JOIN IS NOT A CORNER, and this deliberately does not split at one.
// A stadium — two straight sides closed by two semicircular ends — is a single
// tangent-continuous loop, and it comes back as ONE run, correctly: there is no
// vertex there for a B-rep to have. It costs control points (35 for a 50-point
// loop) and buys a boundary with no false vertices in it. Splitting at
// CURVATURE discontinuities as well would give the tighter description; it is
// not done here because a false vertex is a topology claim, and this module
// would be making it up.
//
// ⚠⚠ THE PCURVE AND THE EDGE ARE TWO INDEPENDENT APPROXIMATIONS OF ONE
// BOUNDARY, and they cannot be made to agree exactly: the surface image of a
// NURBS curve in (u,v) is not itself a NURBS curve, for the same reason a
// surface's isocurve along an arbitrary parameter path is not one. The
// disagreement between them is precisely what an edge's TOLERANCE means in a
// B-rep, so it is MEASURED here (`consistency`) rather than assumed, and
// handed on to whoever writes the file. Every Brep Rhino writes carries the
// same pair and the same measurement.
import { fitCurveToPoints, maxDeviationFromCurve } from './fitcurve.mjs';
import { surfacePoint } from './surface.mjs';
import { curvePoint } from './curve.mjs';

// Above this turn between the incoming and outgoing segment, a loop vertex is
// a CORNER. In radians, and it is a shape question rather than a tolerance
// question — parameter space has no units, so no distance bound could serve.
//
// ⚠ THE VALUE IS A DISCRIMINATOR, NOT A TUNED CONSTANT. It has to sit above
// the turn a smooth loop's own sampling shows and below the shallowest genuine
// corner. A closed loop of N samples turns 360/N degrees per step, so 25
// degrees admits any smooth loop sampled at 15 points or more, while the
// shallowest corner it can miss is a 155-degree one. Sampling far coarser than
// that is not a smooth loop being under-described, it is a polygon.
const DEFAULT_CORNER_ANGLE = 25 * Math.PI / 180;

function unitStep(a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const L = Math.hypot(dx, dy);
  return L > 0 ? [dx / L, dy / L] : null;
}

// The loop is stored WITHOUT its repeated first point, the convention
// everywhere in this project, so index arithmetic wraps.
// Returns { runs, cornerIndices } where each run is a (u,v) point list that
// STARTS at one corner and ENDS at the next — adjacent runs therefore SHARE
// their meeting point, which is what makes the fitted curves meet exactly.
// A loop with no corner comes back as one run carrying the repeated first
// point, so it closes.
export function splitLoopAtCorners(uv, opts = {}) {
  const cornerAngle = opts.cornerAngle ?? DEFAULT_CORNER_ANGLE;
  const n = uv.length;
  if (n < 3) return { runs: [], cornerIndices: [] };

  const cornerIndices = [];
  for (let i = 0; i < n; i++) {
    const into = unitStep(uv[(i - 1 + n) % n], uv[i]);
    const outOf = unitStep(uv[i], uv[(i + 1) % n]);
    // A repeated point has no direction and cannot be judged a corner; it is
    // not a corner, it is a duplicate, and the fitter drops it either way.
    if (!into || !outOf) continue;
    const cos = Math.min(1, Math.max(-1, into[0] * outOf[0] + into[1] * outOf[1]));
    if (Math.acos(cos) > cornerAngle) cornerIndices.push(i);
  }

  // ONE corner is not a split. A loop with a single corner is still one run —
  // it just happens to have a known place to start and end, which is exactly
  // where a closed run wants its seam anyway.
  if (cornerIndices.length < 2) {
    const start = cornerIndices.length === 1 ? cornerIndices[0] : 0;
    const run = [];
    for (let k = 0; k <= n; k++) run.push(uv[(start + k) % n]);
    return { runs: [run], cornerIndices };
  }

  const runs = [];
  for (let k = 0; k < cornerIndices.length; k++) {
    const a = cornerIndices[k], b = cornerIndices[(k + 1) % cornerIndices.length];
    const run = [];
    let i = a;
    for (;;) {
      run.push(uv[i]);
      if (i === b) break;
      i = (i + 1) % n;
    }
    runs.push(run);
  }
  return { runs, cornerIndices };
}

// How far the SURFACE IMAGE of the fitted pcurve sits from the fitted edge
// curve — the number an ON_Edge's tolerance is supposed to be. Sampled along
// the pcurve's own domain and measured against the edge curve by the same
// conservative rule every other deviation here uses, so it can only
// over-report.
function pcurveEdgeConsistency(srf, pcurve, edge, samples = 64) {
  const U = pcurve.knots, p = pcurve.degree;
  const t0 = U[p], t1 = U[U.length - 1 - p];
  const imaged = [];
  for (let i = 0; i <= samples; i++) {
    const [u, v] = curvePoint(pcurve, t0 + (t1 - t0) * (i / samples));
    imaged.push(surfacePoint(srf, u, v));
  }
  return maxDeviationFromCurve(imaged, edge);
}

// opts: { tolerance, cornerAngle, consistencySamples }
// Returns { ok, runs, stats } where each run is
//   { pcurve, edge, tolerance, kind, pcurveDeviation, edgeDeviation, consistency }
// or { ok: false, reason } — never a boundary that missed its bound. A refusal
// is a real outcome here and the caller is expected to keep the polyline,
// which describes the same boundary and is never wrong, only faceted.
export function fitTrimLoop(srf, uv, opts = {}) {
  const tolerance = opts.tolerance ?? 1e-3;
  if (!srf || !srf.ctrlNet) return { ok: false, reason: 'fitTrimLoop needs a surface' };
  if (!Array.isArray(uv) || uv.length < 3) return { ok: false, reason: 'fitTrimLoop needs a loop of at least 3 points' };
  if (!(tolerance > 0)) return { ok: false, reason: 'fitTrimLoop needs a positive tolerance' };

  const { runs: runsUV, cornerIndices } = splitLoopAtCorners(uv, opts);
  if (!runsUV.length) return { ok: false, reason: 'the loop has no run to fit' };

  const runs = [];
  let worstPcurve = 0, worstEdge = 0, worstConsistency = 0, exactRuns = 0;
  for (const runUV of runsUV) {
    if (runUV.length < 2) continue;
    // ⚠ EXACT ENDPOINTS, ALWAYS. A fitted run has to start and end exactly
    // where its neighbours do, or the loop does not close and OpenNURBS
    // rejects it for not joining — a gap at a shared corner is a naked edge
    // wherever it appears. The fitter fixes its end control points to the
    // first and last input point for exactly this, at the cost of a marginally
    // higher residual, and adjacent runs are handed the SAME shared point.
    const pcurveFit = fitCurveToPoints(runUV.map(([u, v]) => [u, v, 0]), { tolerance, closed: false, exactEndpoints: true });
    if (!pcurveFit.ok) return { ok: false, reason: `the (u,v) boundary did not fit: ${pcurveFit.reason}` };

    const points3d = runUV.map(([u, v]) => surfacePoint(srf, u, v));
    const edgeFit = fitCurveToPoints(points3d, { tolerance, closed: false, exactEndpoints: true });
    if (!edgeFit.ok) return { ok: false, reason: `the 3-D edge did not fit: ${edgeFit.reason}` };

    const consistency = pcurveEdgeConsistency(srf, pcurveFit.curve, edgeFit.curve, opts.consistencySamples);
    if (!Number.isFinite(consistency)) return { ok: false, reason: 'the pcurve and the edge curve could not be compared' };

    if (pcurveFit.kind === 'line' || pcurveFit.kind === 'circle' || pcurveFit.kind === 'ellipse') exactRuns++;
    worstPcurve = Math.max(worstPcurve, pcurveFit.maxDeviation);
    worstEdge = Math.max(worstEdge, edgeFit.maxDeviation);
    worstConsistency = Math.max(worstConsistency, consistency);

    runs.push({
      pcurve: pcurveFit.curve,
      edge: edgeFit.curve,
      // The edge's tolerance is the measured disagreement, never below the
      // bound the curves were fitted to — a tolerance tighter than the fit it
      // describes would be a claim the geometry does not support.
      tolerance: Math.max(consistency, tolerance),
      kind: pcurveFit.kind,
      pcurveDeviation: pcurveFit.maxDeviation,
      edgeDeviation: edgeFit.maxDeviation,
      consistency,
    });
  }
  if (!runs.length) return { ok: false, reason: 'every run of the loop was degenerate' };

  return {
    ok: true,
    runs,
    stats: {
      inputPoints: uv.length,
      corners: cornerIndices.length,
      runs: runs.length,
      exactRuns,
      controlPoints: runs.reduce((s, r) => s + r.pcurve.ctrlPts.length, 0),
      worstPcurveDeviation: worstPcurve,
      worstEdgeDeviation: worstEdge,
      worstConsistency,
    },
  };
}
