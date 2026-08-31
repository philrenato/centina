// FIBERS FROM ROOT FRAMES — the interchange type every fiber/hair generator
// in this kernel is built on.
//
// The thing worth generalizing was never any one generator. It is what such a
// generator needs from whatever it grows on, which is only ever this:
//
//   RootFrame = { hostParam, position, normal, tangentU, tangentV, weight }
//
// A host is anything that can answer "give me frames". Nothing below asks what
// produced them, so there is no list of legal host type names anywhere in this
// file — a curve, a surface, a polysurface face, a SuperB limit region and a
// point container are all the same input once they are frames. Adding a host
// is a new emitter, never a new branch in the fiber code.
//
// Pure math over plain arrays: no THREE, no DOM, per the standing kernel rule.

import { add, sub, scale, dot, cross, length, normalize } from './vec3.mjs';
import { surfacePointAndPartials, surfaceClosure } from './surface.mjs';
import { divideByArcLength, curvePointAndTangent, isCurveClosed } from './curve.mjs';

// A partial derivative this short means the surface has no usable tangent
// plane here — a revolve pole, where one whole family of isocurves collapses
// to a point. Emitting a fiber from it would produce NaN directions rather
// than a wrong-but-finite result, so those roots are refused and counted.
const DEGENERATE = 1e-9;

// ---------------------------------------------------------------------------
// SURFACE HOST
// ---------------------------------------------------------------------------

// Root frames on a (u,v) grid over a NURBS surface.
//
// Returns { frames, skippedPoles, seamCollapsed } — the two skip counts are
// returned rather than logged because a caller that cannot say "3 roots were
// refused at poles" in its own status line would be hiding a real fact about
// the result. Silence and zero are different answers.
export function surfaceRootFrames(srf, uCount, vCount, opts = {}) {
  const nU = Math.max(1, Math.round(uCount));
  const nV = Math.max(1, Math.round(vCount));
  // The parameter domain comes from the surface's own knot vectors, never
  // assumed to be [0,1] — a revolved surface's domain is whatever its
  // construction produced (commonly [0,4] in the revolved direction), and
  // sampling [0,1] over it silently emits every root into one corner.
  const du = knotDomain(srf.knotsU, srf.degU);
  const dv = knotDomain(srf.knotsV, srf.degV);
  const { uMin = du[0], uMax = du[1], vMin = dv[0], vMax = dv[1] } = opts;

  // On a closed direction the first and last grid lines are the SAME physical
  // place, so sampling both lays a double-thick row of fibers down the seam.
  // Drop the duplicate end line instead of the honest interior samples.
  // surfaceClosure reports `closedU` / `closedV`. Reading any other spelling
  // silently disables the seam handling below and the only visible symptom is
  // a double-thick row of fibers down one isocurve.
  const closure = opts.closure || safeClosure(srf);
  const closedU = !!(closure && closure.closedU);
  const closedV = !!(closure && closure.closedV);

  const frames = [];
  let skippedPoles = 0;
  let seamCollapsed = 0;

  // On a CLOSED direction the parameter wraps, so dividing by n (not n-1) and
  // stopping one short means the last sample sits one step before the seam
  // rather than back on top of the first. On an OPEN direction both ends are
  // real, distinct edges and must both be sampled.
  for (let i = 0; i < nU; i++) {
    const uDen = closedU ? nU : Math.max(1, nU - 1);
    const u = uMin + (uMax - uMin) * (nU === 1 ? 0.5 : i / uDen);

    for (let j = 0; j < nV; j++) {
      const vDen = closedV ? nV : Math.max(1, nV - 1);
      const v = vMin + (vMax - vMin) * (nV === 1 ? 0.5 : j / vDen);

      const frame = surfaceFrameAt(srf, u, v);
      if (!frame) { skippedPoles++; continue; }
      frames.push(frame);
    }
  }
  if (closedU) seamCollapsed += nV;
  if (closedV) seamCollapsed += nU;
  return { frames, skippedPoles, seamCollapsed };
}

// A clamped B-spline is only defined over [knots[deg], knots[len-1-deg]];
// outside that the basis functions are not a partition of unity.
export function knotDomain(knots, deg) {
  if (!Array.isArray(knots) || !knots.length) return [0, 1];
  const d = Number.isFinite(deg) ? deg : 0;
  const lo = knots[d], hi = knots[knots.length - 1 - d];
  return Number.isFinite(lo) && Number.isFinite(hi) && hi > lo ? [lo, hi] : [0, 1];
}

function safeClosure(srf) {
  try { return surfaceClosure(srf); } catch { return null; }
}

// One frame at one (u,v). Returns null at a degenerate point rather than a
// frame full of NaN — refusing is a result, a NaN fiber is a bug that surfaces
// three steps downstream.
export function surfaceFrameAt(srf, u, v) {
  let ev;
  try { ev = surfacePointAndPartials(srf, u, v); } catch { return null; }
  if (!ev || !ev.point) return null;
  const { point, su, sv } = ev;
  if (!su || !sv) return null;
  if (length(su) < DEGENERATE || length(sv) < DEGENERATE) return null;

  const n = cross(su, sv);
  // su and sv can both be long yet parallel (a crease collapsing the tangent
  // plane); the cross product is the only thing that actually proves a plane.
  if (length(n) < DEGENERATE) return null;

  const normal = normalize(n);
  const tangentU = normalize(su);
  // Re-derived rather than taken from sv, so the basis is genuinely
  // orthonormal even where the surface's own isocurves are not perpendicular.
  const tangentV = cross(normal, tangentU);
  if (!isFinite3(point) || !isFinite3(normal) || !isFinite3(tangentU) || !isFinite3(tangentV)) return null;

  return { hostParam: { u, v }, position: point, normal, tangentU, tangentV, weight: 1 };
}

function isFinite3(p) {
  return Array.isArray(p) && p.length >= 3 && Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]);
}

// ---------------------------------------------------------------------------
// AIM — two angles, not a tangent/perpendicular toggle
// ---------------------------------------------------------------------------

// lift  0..90   0 = flat in the tangent plane, 90 = straight out along normal
// sweep 0..360  rotation within the tangent plane, 0 = along tangentU
//
// "Tangent" and "perpendicular" are two PRESETS of this one control rather
// than two modes, which is why a flat 2D curve needs no special case: lift
// going 0 -> 90 is literally the fibers rising out of the plane.
export function aimDirection(frame, liftDeg, sweepDeg) {
  const lift = (liftDeg * Math.PI) / 180;
  const sweep = (sweepDeg * Math.PI) / 180;
  const inPlane = add(scale(frame.tangentU, Math.cos(sweep)), scale(frame.tangentV, Math.sin(sweep)));
  return normalize(add(scale(frame.normal, Math.sin(lift)), scale(inPlane, Math.cos(lift))));
}

// T1 — LAUNCH HEMISPHERE CLAMP. Free, and always on: a fiber may
// not leave INTO the thing it grows on. Rotating the direction toward the
// normal preserves its sweep, where simply projecting it would collapse a
// fiber aimed straight down onto an arbitrary in-plane heading.
export function clampToHemisphere(dir, normal, minLiftDeg = 2) {
  const minDot = Math.sin((minLiftDeg * Math.PI) / 180);
  const d = dot(dir, normal);
  if (d >= minDot) return dir;
  const tangential = sub(dir, scale(normal, d));
  const tLen = length(tangential);
  if (tLen < DEGENERATE) return normal.slice();
  const keep = Math.sqrt(Math.max(0, 1 - minDot * minDot));
  return normalize(add(scale(normal, minDot), scale(scale(tangential, 1 / tLen), keep)));
}

// ---------------------------------------------------------------------------
// THE FIBER
// ---------------------------------------------------------------------------

// Sampled points for one fiber. The droop/recoil construction is carried over
// from the shipped strand builder deliberately and unchanged: droop ramps as
// an ANGLE through sin(), with recoil extending that angle past PI/2 so the
// weight genuinely peaks partway along and swings back. A plain power curve
// cannot do this — it is monotonic for any exponent, so no amount of tuning
// curls a tip back, which was the original bug. Do not "simplify" it.
export function fiberPoints(frame, opts = {}) {
  const {
    cvs = 8,
    lengthValue = 10,
    liftDeg = 90,
    sweepDeg = 0,
    droopAmount = 0.35,
    droopExponent = 1.5,
    recoilAmount = 0.4,
    intensity = 1,
    droopDir = null,        // null => lie down against the host (-normal)
    minLiftDeg = 2,
    tangentPlaneGuard = true, // T2
  } = opts;

  const n = Math.max(2, Math.round(cvs));
  const aimed = clampToHemisphere(aimDirection(frame, liftDeg, sweepDeg), frame.normal, minLiftDeg);
  // Default droop is -normal ("lie down against the host"), not world -Z.
  // World -Z is gravity, which is right for hair on a head and points straight
  // into the wall on the inside of a cup — the single default that separates
  // "hair" from "fur" without the student learning a new word.
  const dd = droopDir ? normalize(droopDir) : scale(frame.normal, -1);

  const pts = [];
  for (let k = 0; k < n; k++) {
    const t = k / (n - 1);
    const mainRamp = Math.pow(t, droopExponent);
    const droopWeight = Math.sin(mainRamp * (Math.PI / 2) * (1 + recoilAmount));
    let p = add(frame.position, scale(aimed, t * lengthValue));
    p = add(p, scale(dd, droopAmount * lengthValue * intensity * droopWeight));
    pts.push(tangentPlaneGuard ? liftAboveTangentPlane(p, frame, minLiftDeg) : p);
  }
  return pts;
}

// T2 — ROOT TANGENT-PLANE HALF-SPACE CLAMP. Every deform
// displacement is clamped so no sample crosses below its own root's tangent
// plane. This is the tier that actually makes a concave host work, because
// droop points INTO the wall on the inside of any cup.
export function liftAboveTangentPlane(p, frame, minLiftDeg = 0) {
  const rel = sub(p, frame.position);
  const along = dot(rel, frame.normal);
  const floor = Math.max(0, Math.sin((minLiftDeg * Math.PI) / 180) * 0);
  if (along >= floor) return p;
  return add(p, scale(frame.normal, floor - along));
}

// ---------------------------------------------------------------------------
// CURVE HOST
// ---------------------------------------------------------------------------

// The plane a curve lies in, as { normal, origin, axisU, axisV }, or null if it
// is genuinely non-planar or a straight line.
//
// A STRAIGHT LINE RETURNS null DELIBERATELY. It lies in infinitely many planes,
// so there is no outward direction to pick and no honest way to choose one; the
// caller falls back to a transported frame instead of inventing a side.
//
// Fitted to the CONTROL POINTS, which is exact rather than sampled: a NURBS
// curve lies in the plane of its control points by the convex-hull property, so
// a plane containing the net contains the curve. Sampling could miss a bulge
// between samples.
export function curvePlane(crv, tolScale = 1e-6) {
  const pts = (crv.ctrlPts || []).map((cp) => [cp[0], cp[1], cp[2]]);
  if (pts.length < 3) return null;
  const origin = pts[0];

  // The farthest point from the origin gives the most numerically stable first
  // axis; a near-coincident pair would make every later cross product noise.
  let axisU = null, bestU = 0;
  for (const p of pts) {
    const d = sub(p, origin), l = length(d);
    if (l > bestU) { bestU = l; axisU = d; }
  }
  if (!axisU || bestU < 1e-12) return null;
  axisU = scale(axisU, 1 / bestU);

  // Then the point furthest OFF that axis fixes the plane. If nothing is off it,
  // the net is collinear and this is a line.
  let normal = null, bestN = 0;
  for (const p of pts) {
    const c = cross(sub(p, origin), axisU), l = length(c);
    if (l > bestN) { bestN = l; normal = c; }
  }
  if (!normal || bestN < bestU * 1e-9) return null;
  normal = scale(normal, 1 / length(normal));

  // Planarity is judged RELATIVE to the curve's own size — a 0.01mm deviation
  // is planar on a 1000mm curve and is not on a 0.1mm one.
  const tol = Math.max(bestU * tolScale, 1e-12);
  for (const p of pts) if (Math.abs(dot(sub(p, origin), normal)) > tol) return null;

  return { normal, origin, axisU, axisV: cross(normal, axisU) };
}

// Signed area of a closed planar point loop, measured in the plane's own basis.
// Its SIGN is the whole point: it says whether the loop winds counter-clockwise
// about `normal`, which is what makes "outward" mean outward everywhere on the
// loop instead of outward for half of a star's points.
export function planarLoopSignedArea(points, plane) {
  let a = 0;
  for (let i = 0; i < points.length; i++) {
    const p = sub(points[i], plane.origin), q = sub(points[(i + 1) % points.length], plane.origin);
    a += dot(p, plane.axisU) * dot(q, plane.axisV) - dot(q, plane.axisU) * dot(p, plane.axisV);
  }
  return a / 2;
}

// Root frames along a curve, at equal REAL ARC LENGTH — not equal parameter,
// which bunches roots at a polygon's corners.
//
// PLANAR: normal is the in-plane OUTWARD perpendicular and tangentV is the plane
// normal, so lift 0 lies flat in the plane and raising it lifts the fibers out
// of it. On a closed curve the outward sign comes from the loop's own winding,
// so a 5-point star's reflex vertices point outward like every other root.
//
// NON-PLANAR: there is no outward, so a rotation-minimizing frame is carried
// along the curve by double reflection (Wang, Jüttler, Zheng & Liu, ACM TOG
// 27(1), 2008). Picking a fresh arbitrary perpendicular per point instead is a
// silent-wrong-result bug rather than a crash: it visibly twists along a helix.
//
// ⚠ A CLOSED NON-PLANAR CURVE DOES NOT CLOSE ITS FRAME. Transport around a loop
// accumulates real holonomy, so the last frame does not meet the first and the
// fibers step at the seam. That is a genuine property of the curve, not a bug
// here, and it is the same wall the closed-rail sweep already measured; it is
// not worked around on this path.
export function curveRootFrames(crv, count, opts = {}) {
  const n = Math.max(1, Math.round(count));
  const stations = divideByArcLength(crv, n, opts.tolerance);
  const plane = opts.plane !== undefined ? opts.plane : curvePlane(crv);
  const closed = isCurveClosed(crv);

  const raw = [];
  let skippedDegenerate = 0;
  for (const st of stations) {
    let ev;
    try { ev = curvePointAndTangent(crv, st.u); } catch { skippedDegenerate++; continue; }
    if (!ev || !isFinite3(ev.point) || !isFinite3(ev.tangent) || length(ev.tangent) < DEGENERATE) { skippedDegenerate++; continue; }
    raw.push({ u: st.u, point: ev.point, tangent: normalize(ev.tangent) });
  }
  if (!raw.length) return { frames: [], skippedDegenerate, planar: !!plane, closed };

  let sign = 1;
  if (plane && closed && raw.length >= 3) {
    // Measured on the root loop itself, which is the same loop the frames sit
    // on, so the sign cannot describe a different polygon than the one in hand.
    sign = planarLoopSignedArea(raw.map((r) => r.point), plane) < 0 ? -1 : 1;
  }

  const frames = [];
  let carried = null;
  for (let i = 0; i < raw.length; i++) {
    const { point, tangent } = raw[i];
    let normal, tangentV;
    if (plane) {
      // cross(tangent, planeNormal) is the outward side for a loop winding
      // counter-clockwise about that normal; `sign` flips it for the other
      // winding. Verified on a circle: at (r,0,0) with a CCW tangent (0,1,0)
      // and normal +Z this returns (1,0,0), pointing away from the centre.
      const outward = cross(tangent, plane.normal);
      if (length(outward) < DEGENERATE) { skippedDegenerate++; continue; }
      normal = scale(normalize(outward), sign);
      tangentV = plane.normal.slice();
    } else {
      if (!carried) {
        carried = seedPerpendicular(tangent);
      } else {
        carried = transportByDoubleReflection(raw[i - 1].point, raw[i - 1].tangent, carried, point, tangent);
      }
      // Re-orthogonalized against the tangent every step. Double reflection is
      // exact in theory and drifts in floating point over many stations.
      const proj = sub(carried, scale(tangent, dot(carried, tangent)));
      if (length(proj) < DEGENERATE) { skippedDegenerate++; continue; }
      carried = normalize(proj);
      normal = carried.slice();
      tangentV = cross(normal, tangent);
    }
    frames.push({ hostParam: { u: raw[i].u }, position: point, normal, tangentU: tangent.slice(), tangentV, weight: 1 });
  }
  return { frames, skippedDegenerate, planar: !!plane, closed };
}

// Any unit vector perpendicular to `t`. Only ever used to SEED a transported
// frame — every later frame comes from the one before it, which is what stops
// the twist an arbitrary per-point perpendicular produces.
function seedPerpendicular(t) {
  const ref = Math.abs(t[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  return normalize(cross(t, ref));
}

// One step of the double-reflection rotation-minimizing frame. Two reflections
// compose into a rotation, which is why this needs no trigonometry and no
// arbitrary axis: the first reflection carries the frame across the chord
// between the two points, the second corrects it onto the new tangent.
function transportByDoubleReflection(p0, t0, r0, p1, t1) {
  const v1 = sub(p1, p0);
  const c1 = dot(v1, v1);
  if (c1 < 1e-24) return r0.slice(); // coincident stations: nothing to transport across
  const rL = sub(r0, scale(v1, (2 / c1) * dot(v1, r0)));
  const tL = sub(t0, scale(v1, (2 / c1) * dot(v1, t0)));
  const v2 = sub(t1, tL);
  const c2 = dot(v2, v2);
  if (c2 < 1e-24) return rL; // tangent unchanged: the first reflection already landed it
  return sub(rL, scale(v2, (2 / c2) * dot(v2, rL)));
}
