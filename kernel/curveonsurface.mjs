// A CURVE THAT LIVES IN A SURFACE'S PARAMETER SPACE, so it can be re-evaluated
// whenever the surface changes and follow it. The curve is stored as (u, v)
// stations, never as 3-D points: 3-D points would have to be re-projected after
// every host edit, and a projection is a search with its own failure modes,
// while an evaluation is exact and cannot land off the surface.
//
// ⚠ THE UNWRAPPED PATH IS THE WHOLE TRICK, and it is what makes "pencil on a
// sphere" work. Between two stations either side of a seam the naive route is
// the long way around the entire surface, because 0.95 -> 0.05 reads as a
// journey of -0.9 rather than +0.1. Interpolating in RAW parameters therefore
// draws a stroke that crosses the whole model to reach a point two millimetres
// away. So consecutive stations are unwrapped first — each step takes the
// SHORTER of the two ways round a closed direction, accumulating into a
// continuous coordinate that may run outside the domain — and only the final
// sample is wrapped back for evaluation. A direction the surface is not closed
// in is clamped instead, since there is no other side to reach.
import { surfacePoint, surfaceClosure, wrapParam } from './surface.mjs';

export const DEFAULT_SAMPLES_PER_SPAN = 24;

function domainOf(knots, degree) {
  return [knots[degree], knots[knots.length - 1 - degree]];
}

// The shorter signed step from a to b, given a direction that may wrap.
function shortestDelta(a, b, min, max, closed) {
  const raw = b - a;
  if (!closed) return raw;
  const span = max - min;
  if (span <= 0) return raw;
  let d = raw % span;
  if (d > span / 2) d -= span;
  if (d < -span / 2) d += span;
  return d;
}

// Turn stations into a continuous path by accumulating shortest steps. The
// result is deliberately allowed to leave the domain; wrapping happens once,
// at evaluation, so the interpolation in between never sees a discontinuity.
export function unwrapStations(srf, uvPoints, opts = {}) {
  const closure = opts.closure || surfaceClosure(srf);
  const [uMin, uMax] = domainOf(srf.knotsU, srf.degU);
  const [vMin, vMax] = domainOf(srf.knotsV, srf.degV);
  const out = [];
  for (let i = 0; i < uvPoints.length; i++) {
    const [u, v] = uvPoints[i];
    if (i === 0) { out.push([u, v]); continue; }
    const prev = out[i - 1];
    out.push([
      prev[0] + shortestDelta(wrapParam(prev[0], uMin, uMax, closure.closedU), u, uMin, uMax, closure.closedU),
      prev[1] + shortestDelta(wrapParam(prev[1], vMin, vMax, closure.closedV), v, vMin, vMax, closure.closedV),
    ]);
  }
  return out;
}

// Catmull-Rom through the stations, in UNWRAPPED space. A spline is what makes
// a drawn stroke read as a curve rather than a chain of straight hops, and
// doing it here rather than in 3-D is what keeps every sample exactly on the
// surface — a 3-D spline through on-surface points leaves the surface between
// them wherever it is curved, which on a sphere is everywhere.
function catmullRom(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}

// The (u, v) stations a stroke passes through, densified. `degree` 1 keeps the
// straight hops (a polyline on the surface); anything higher smooths them.
export function curveOnSurfaceUV(srf, uvPoints, opts = {}) {
  const pts = (uvPoints || []).filter((p) => Array.isArray(p) && Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (pts.length < 2) return { ok: false, reason: 'a curve on a surface needs at least two points', uv: [] };
  const closure = opts.closure || surfaceClosure(srf);
  const closed = !!opts.closed;
  const perSpan = Math.max(2, Math.round(opts.samplesPerSpan ?? DEFAULT_SAMPLES_PER_SPAN));
  const smooth = (opts.degree ?? 3) > 1 && pts.length > 2;
  const path = unwrapStations(srf, closed ? [...pts, pts[0]] : pts, { closure });
  const [uMin, uMax] = domainOf(srf.knotsU, srf.degU);
  const [vMin, vMax] = domainOf(srf.knotsV, srf.degV);
  const uv = [];
  const at = (i) => path[Math.max(0, Math.min(path.length - 1, i))];
  for (let i = 0; i < path.length - 1; i++) {
    const p0 = at(i - 1), p1 = at(i), p2 = at(i + 1), p3 = at(i + 2);
    for (let s = 0; s < perSpan; s++) {
      const t = s / perSpan;
      const u = smooth ? catmullRom(p0[0], p1[0], p2[0], p3[0], t) : p1[0] + (p2[0] - p1[0]) * t;
      const v = smooth ? catmullRom(p0[1], p1[1], p2[1], p3[1], t) : p1[1] + (p2[1] - p1[1]) * t;
      uv.push([
        wrapParam(u, uMin, uMax, closure.closedU),
        wrapParam(v, vMin, vMax, closure.closedV),
      ]);
    }
  }
  const last = path[path.length - 1];
  uv.push([
    wrapParam(last[0], uMin, uMax, closure.closedU),
    wrapParam(last[1], vMin, vMax, closure.closedV),
  ]);
  return { ok: true, uv };
}

// The drawn stroke as 3-D points on THIS surface. Called again with the same
// stations after the host changes — that call is the whole reflow.
export function curveOnSurfacePoints(srf, uvPoints, opts = {}) {
  const res = curveOnSurfaceUV(srf, uvPoints, opts);
  if (!res.ok) return { ok: false, reason: res.reason, points: [], uv: [] };
  const points = res.uv.map(([u, v]) => {
    const p = surfacePoint(srf, u, v);
    return [p[0], p[1], p[2]];
  });
  return { ok: true, points, uv: res.uv };
}

// How far the drawn polyline strays from the surface BETWEEN its samples. The
// samples themselves are on the surface by construction, so measuring them
// proves nothing; the honest question is whether the chords between them are
// close enough that the stroke reads as lying on the surface rather than
// cutting corners across it. This is what "adds points automatically to stay
// parallel" is measured by, and what a caller raises samplesPerSpan against.
export function chordDeviation(srf, uvSamples) {
  let worst = 0;
  for (let i = 0; i < uvSamples.length - 1; i++) {
    const a = uvSamples[i], b = uvSamples[i + 1];
    const pa = surfacePoint(srf, a[0], a[1]);
    const pb = surfacePoint(srf, b[0], b[1]);
    const mid = surfacePoint(srf, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
    const cx = (pa[0] + pb[0]) / 2, cy = (pa[1] + pb[1]) / 2, cz = (pa[2] + pb[2]) / 2;
    worst = Math.max(worst, Math.hypot(mid[0] - cx, mid[1] - cy, mid[2] - cz));
  }
  return worst;
}
