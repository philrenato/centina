// EXTEND — CARRY A CURVE PAST ITS OWN END.
//
// Three kinds, and they are three different promises rather than three settings:
//
//   · LINE   — leaves along the real end tangent. Meets the original at G1.
//   · ARC    — leaves along the tangent on the osculating circle, so the
//              curvature at the join matches too: G2.
//   · SMOOTH — continues the curve's OWN polynomial. Not an approximation of the
//              original's shape, the same function evaluated further along, so
//              every derivative matches and the original portion is unchanged to
//              the last bit.
//
// ⚠⚠ AND NONE OF THEM EVER MAKES AN UNCLAMPED CURVE, which is the trap this
// whole file is arranged to avoid. `kernel/knots.mjs` says it plainly: its
// insertKnot machinery assumes clamped input, `extractSubCurve` silently returns
// a DIFFERENT curve on unclamped input (measured there at 0.37 units on a curve
// spanning 3), and `decomposeToBezier` returns NaN pieces outright. A "just
// evaluate past the end" implementation walks straight into that.
//
// So SMOOTH works on the last BEZIER PIECE instead. De Casteljau is arithmetic on
// control points and is valid at any parameter, including t > 1, where it returns
// the control polygon of the same polynomial over the longer interval. Two splits
// give the extension as its own ordinary CLAMPED Bezier, which the existing
// concatenation understands. No unclamped curve is ever constructed, and nothing
// here has to reason about a knot vector that does not exist.
//
// ⚠ A POLYNOMIAL EXTRAPOLATES FAST AND BADLY. Past a modest fraction of the piece
// it came from, a cubic's continuation is not a plausible reading of the
// designer's intent — it is what the algebra says, which diverges. The distance
// is capped against the source piece's own size and refused beyond it, by name,
// rather than returning a spectacular curve and calling it an extension.
import { curvePointAndTangent, rationalCurveDerivs, reverseCurve } from './curve.mjs';
import { decomposeToBezier, degreeElevateCurve, concatTwoC0, rescaleCurveDomain } from './knots.mjs';

const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul3 = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len3 = (a) => Math.hypot(a[0], a[1], a[2]);
const unit3 = (a) => { const L = len3(a); return L > 1e-14 ? mul3(a, 1 / L) : [0, 0, 0]; };

export const EXTEND_REFUSAL = Object.freeze({
  BAD_INPUT: 'bad-input',
  BAD_LENGTH: 'bad-length',
  DEGENERATE_END: 'degenerate-end',
  TOO_FAR: 'too-far',
  STRAIGHT: 'straight-has-no-arc',
  FAILED: 'failed',
});
const refuse = (kind, reason) => ({ ok: false, kind, reason });

// How far a SMOOTH extension may run, as a multiple of the source Bezier piece's
// own control-polygon length. Beyond this the continuation stops being a reading
// of the curve and becomes a reading of its leading coefficient.
const SMOOTH_REACH = 1.0;

// ── de Casteljau, in homogeneous space so rational curves work too ───────────
// Splitting at parameter t returns both halves' control points. Valid for t
// outside [0,1]: the "left" array is then the control polygon of the SAME
// polynomial over the longer interval, which is exactly what extrapolation is.
function deCasteljau(cps, t) {
  const n = cps.length;
  const rows = [cps.map((c) => c.slice())];
  for (let r = 1; r < n; r++) {
    const prev = rows[r - 1], row = [];
    for (let i = 0; i + 1 < prev.length; i++) {
      const a = prev[i], b = prev[i + 1];
      row.push(a.map((v, k) => v + (b[k] - v) * t));
    }
    rows.push(row);
  }
  /* ⚠ THE RIGHT HALF IS ALREADY IN ORDER — DO NOT REVERSE IT. In the de Casteljau
     triangle b_i^k, the two halves are left[k] = b_0^k and right[k] = b_{k}^{n-1-k}:
     right[0] is the split point and right[n-1] is the original last control
     point, so the array already runs start-to-end. Reversing it produced an
     extension that ran BACKWARDS from the far end to the join — which still left
     the original portion of the joined curve bit-perfect, so the exactness test
     passed and only the curvature across the join gave it away. */
  const left = rows.map((row) => row[0]);
  const right = rows.map((row, r) => rows[n - 1 - r][r]);
  return { left, right };
}

const homog = (cp) => { const w = cp[3] === undefined ? 1 : cp[3]; return [cp[0] * w, cp[1] * w, cp[2] * w, w]; };
const euclid = (cp) => { const w = cp[3] === undefined ? 1 : cp[3]; return Math.abs(w) > 1e-15 ? [cp[0] / w, cp[1] / w, cp[2] / w, w] : [cp[0], cp[1], cp[2], w]; };

function bezierCurve(ctrlPts, degree) {
  const knots = [];
  for (let i = 0; i <= degree; i++) knots.push(0);
  for (let i = 0; i <= degree; i++) knots.push(1);
  return { degree, knots, ctrlPts };
}

// Curvature and the osculating frame at one end, from the curve's own derivatives.
function endFrame(crv, atEnd) {
  const u = atEnd ? crv.knots[crv.knots.length - 1] : crv.knots[0];
  const d = rationalCurveDerivs(crv, u, 2);
  const P = d[0], D1 = d[1], D2 = d[2];
  const speed = len3(D1);
  if (!(speed > 1e-12)) return null;
  const T = mul3(D1, 1 / speed);
  const perp = sub3(D2, mul3(T, dot3(D2, T)));
  const kappa = len3(perp) / (speed * speed);
  return { P, T, N: unit3(perp), kappa, speed };
}

/* THE ENTRY POINT.
 *   crv     — a clamped NURBS curve
 *   opts.at — 'end' (default) or 'start'
 *   opts.kind — 'line' | 'arc' | 'smooth'
 *   opts.length — how far to reach, in model units along the extension
 * Returns { ok, crv, kind, length, ... } or { ok:false, kind, reason }.
 *
 * ⚠ THE START CASE IS THE END CASE ON A REVERSED CURVE, reversed back. Writing
 * it twice is how the two ends drift apart, and reverseCurve is exact and its own
 * inverse.
 */
export function extendCurve(crv, opts = {}) {
  if (!crv || !Array.isArray(crv.ctrlPts) || crv.ctrlPts.length < 2 || !Array.isArray(crv.knots)) {
    return refuse(EXTEND_REFUSAL.BAD_INPUT, 'there is no curve to extend');
  }
  const kind = opts.kind || 'line';
  const length = opts.length;
  if (!Number.isFinite(length) || length <= 0) {
    return refuse(EXTEND_REFUSAL.BAD_LENGTH, `an extension of ${length} is not a positive distance`);
  }
  if (opts.at === 'start') {
    const flipped = extendCurve(reverseCurve(crv), { ...opts, at: 'end' });
    return flipped.ok ? { ...flipped, crv: reverseCurve(flipped.crv) } : flipped;
  }
  const frame = endFrame(crv, true);
  if (!frame) return refuse(EXTEND_REFUSAL.DEGENERATE_END, 'the curve has no direction at that end to continue');

  let piece;
  if (kind === 'line') piece = lineExtension(frame, length, crv.degree);
  else if (kind === 'arc') piece = arcExtension(frame, length, crv.degree);
  else if (kind === 'smooth') piece = smoothExtension(crv, length);
  else return refuse(EXTEND_REFUSAL.BAD_INPUT, `"${kind}" is not one of line, arc or smooth`);
  if (!piece.ok) return piece;

  // Both halves must be the same degree before they can be joined, and the JOIN
  // is C0 — the continuity the extension actually has comes from how it was
  // BUILT (along the real tangent, on the osculating circle, or as the same
  // polynomial), never from the concatenation.
  const deg = Math.max(crv.degree, piece.crv.degree);
  let A = crv.degree === deg ? crv : degreeElevateCurve(crv, deg);
  let B = piece.crv.degree === deg ? piece.crv : degreeElevateCurve(piece.crv, deg);
  const aEnd = A.knots[A.knots.length - 1];
  B = rescaleCurveDomain(B, aEnd, aEnd + (B.knots[B.knots.length - 1] - B.knots[0]));
  let joined;
  try { joined = concatTwoC0(A, B, deg); } catch (e) { return refuse(EXTEND_REFUSAL.FAILED, `the extension could not be joined on: ${e && e.message}`); }
  if (!joined || !Array.isArray(joined.ctrlPts) || joined.ctrlPts.some((c) => !c.every(Number.isFinite))) {
    return refuse(EXTEND_REFUSAL.FAILED, 'the joined curve did not come back finite');
  }
  return { ok: true, crv: joined, kind, length, added: piece.crv.ctrlPts.length };
}

function lineExtension(frame, length, degree) {
  const start = frame.P, end = add3(frame.P, mul3(frame.T, length));
  const seg = bezierCurve([[...start, 1], [...end, 1]], 1);
  return { ok: true, crv: degree > 1 ? degreeElevateCurve(seg, Math.min(degree, 3)) : seg };
}

/* AN ARC ON THE OSCULATING CIRCLE. Radius 1/kappa, center on the normal side, so
 * the extension leaves with the curve's own tangent AND its own curvature: G2 at
 * the join by construction rather than by fitting. A straight end has no
 * osculating circle — infinite radius — and is refused by name rather than
 * silently handed back a line, because a control that quietly does the other
 * thing is how someone ends up believing an arc extension worked. */
function arcExtension(frame, length, degree) {
  if (!(frame.kappa > 1e-9)) {
    return refuse(EXTEND_REFUSAL.STRAIGHT, 'this curve is straight at that end, so it has no arc to continue — use a line extension');
  }
  const r = 1 / frame.kappa;
  const centre = add3(frame.P, mul3(frame.N, r));
  const sweep = length / r;
  if (!(sweep > 0) || sweep > Math.PI) {
    return refuse(EXTEND_REFUSAL.TOO_FAR, `an arc extension of ${length} would sweep ${(sweep * 180 / Math.PI).toFixed(0)}° around a radius of ${r.toFixed(3)}, which doubles back on itself`);
  }
  // A rational quadratic Bezier arc: the standard three-point form, exact for any
  // sweep below a half turn.
  const axis = unit3(cross3(frame.T, frame.N));
  const rot = (v, ang) => {
    const c = Math.cos(ang), s = Math.sin(ang);
    return add3(add3(mul3(v, c), mul3(cross3(axis, v), s)), mul3(axis, dot3(axis, v) * (1 - c)));
  };
  const v0 = sub3(frame.P, centre);
  const P0 = frame.P;
  const P2 = add3(centre, rot(v0, sweep));
  const half = sweep / 2;
  const w = Math.cos(half);
  const mid = add3(centre, mul3(unit3(rot(v0, half)), r / w));
  const seg = bezierCurve([[...P0, 1], [...mid, w], [...P2, 1]], 2);
  return { ok: true, crv: degree > 2 ? degreeElevateCurve(seg, Math.min(degree, 3)) : seg };
}

/* THE SAME POLYNOMIAL, FURTHER ALONG.
 * Take the LAST Bezier piece of the clamped curve, run de Casteljau at t = 1 + s
 * to get that polynomial's control polygon over the longer interval, then split
 * THAT at 1/(1+s) to isolate the part beyond the original end. The result is an
 * ordinary clamped Bezier of the same degree, so nothing downstream ever meets an
 * unclamped knot vector. Every derivative matches at the join because it is not a
 * match — it is the same function. */
function smoothExtension(crv, length) {
  let pieces;
  try { pieces = decomposeToBezier(crv); } catch (e) { return refuse(EXTEND_REFUSAL.FAILED, `this curve could not be read as Bezier pieces: ${e && e.message}`); }
  if (!pieces || !pieces.length) return refuse(EXTEND_REFUSAL.FAILED, 'this curve has no Bezier pieces to continue');
  const last = pieces[pieces.length - 1];
  const cps = last.ctrlPts.map(homog);
  let polyLen = 0;
  for (let i = 1; i < cps.length; i++) polyLen += len3(sub3(euclid(cps[i]), euclid(cps[i - 1])));
  if (!(polyLen > 1e-12)) return refuse(EXTEND_REFUSAL.DEGENERATE_END, 'the last span of this curve is degenerate');
  if (length > polyLen * SMOOTH_REACH) {
    return refuse(EXTEND_REFUSAL.TOO_FAR,
      `a smooth extension continues this curve's own polynomial, and past about ${(polyLen * SMOOTH_REACH).toFixed(3)} it stops describing the curve and starts describing its leading term — asked for ${length}. Extend by less, or use a line or arc extension`);
  }
  // Solve for the parameter overshoot that yields the requested arc length,
  // by bisection on the extension's own control-polygon length. Cheap, monotone,
  // and it keeps the units the ones the person typed.
  const extAt = (s) => {
    const { left } = deCasteljau(cps, 1 + s);
    const { right } = deCasteljau(left, 1 / (1 + s));
    return right;
  };
  const polyLenOf = (pts) => { let L = 0; for (let i = 1; i < pts.length; i++) L += len3(sub3(euclid(pts[i]), euclid(pts[i - 1]))); return L; };
  let lo = 0, hi = 1;
  for (let i = 0; i < 40 && polyLenOf(extAt(hi)) < length; i++) hi *= 1.5;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (polyLenOf(extAt(mid)) < length) lo = mid; else hi = mid;
  }
  const ext = extAt((lo + hi) / 2).map(euclid);
  if (!ext.every((c) => c.every(Number.isFinite))) return refuse(EXTEND_REFUSAL.FAILED, 'the continuation did not come back finite');
  return { ok: true, crv: bezierCurve(ext, last.degree ?? crv.degree) };
}
