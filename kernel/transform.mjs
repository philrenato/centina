// Transform — the per-point mapping math behind the app's Transform
// node. Two mathematically distinct classes, both named in the UI:
//
//   Affine  — rotate, scale, shear. Exact. A NURBS curve/surface point is an
//     affine combination of its own control points (the rational basis
//     functions sum to 1, which is the literal definition of an affine
//     combination), and affine maps commute with affine combinations
//     exactly: T(sum(R_i P_i)) = sum(R_i T(P_i)) for any fixed affine T. So
//     applying one of these three maps directly to every control point
//     reproduces the transformed curve/surface exactly, not an
//     approximation — the same guarantee Move/Mirror rely on.
//
//   Space morph — taper, twist, bend, charybdis. Approximate. Each of these
//     is a different map for every distinct point (taper's cross-section
//     factor varies with axial position; twist/bend/charybdis's rotation
//     angle varies with position), so there is no single fixed affine T —
//     nudging each control point by its own point's map does not reproduce
//     the true continuously-deformed shape, only a control-point-level
//     approximation of it (exact only in the degenerate case of a degree-1
//     curve, where every point is a control point). Close at a reasonable
//     control-point density; never reported as exact.
//
// Every function below takes and returns a plain [x, y, z] array (this
// kernel's own convention, kernel/vec3.mjs) and a plain-data axis frame
// (already-normalized direction vectors, never a THREE.Vector3) — the same
// contract every other kernel module holds, so this has no browser or
// three.js dependency.

import { add, sub, scale, dot, cross, normalize } from './vec3.mjs';
import { evalFalloffRamp } from './falloff.mjs';

// The 7 transform types, and which of the two mathematical classes each
// one belongs to — the single source of truth for the classification shown
// in the UI (Properties/UnWire), never re-derived at the app layer.
export const TRANSFORM_TYPES = ['rotate', 'scale', 'shear', 'taper', 'twist', 'bend', 'charybdis'];
export const TRANSFORM_CLASS = {
  rotate: 'affine', scale: 'affine', shear: 'affine',
  taper: 'morph', twist: 'morph', bend: 'morph', charybdis: 'morph',
};

// Rotate — Rodrigues' rotation formula about `center`, axis `axisDir`
// (already unit length), by `angleRad`. Exact affine map (one fixed
// rotation matrix for every point).
export function rotatePoint(p, center, axisDir, angleRad) {
  const rel = sub(p, center);
  const cosA = Math.cos(angleRad), sinA = Math.sin(angleRad);
  const k = axisDir;
  const kxv = cross(k, rel);
  const kdv = dot(k, rel);
  // v_rot = v*cos + (k x v)*sin + k*(k.v)*(1-cos) — standard Rodrigues form.
  const rotated = add(add(scale(rel, cosA), scale(kxv, sinA)), scale(k, kdv * (1 - cosA)));
  return add(center, rotated);
}

// Scale — a per-axis stretch (not a uniform 3D scale, matching the app's
// Gimbal/Scale semantics): the component of (p-center) along
// axisDir is multiplied by `factor`; the perpendicular remainder is
// untouched. Exact affine map (one fixed linear map + fixed center).
export function scalePoint(p, center, axisDir, factor) {
  const rel = sub(p, center);
  const axisComp = dot(rel, axisDir);
  const perp = sub(rel, scale(axisDir, axisComp));
  return add(add(center, perp), scale(axisDir, axisComp * factor));
}

// Shear — a shear matrix (identity + factor*outer(shearDir, measureAxis)):
// every point moves along `shearDir` by `factor` times its own signed
// distance from `center` measured along `measureAxis`. Exact affine map.
export function shearPoint(p, center, measureAxis, shearDir, factor) {
  const rel = sub(p, center);
  const dist = dot(rel, measureAxis);
  return add(p, scale(shearDir, factor * dist));
}

// Twist — a screw motion whose angle varies linearly with position along
// axisDir (angle(t) = totalAngleRad * axisComp/span), extrapolated linearly
// outside [0, span]. Not one fixed affine map (see module header), so
// approximate at the control-point level.
export function twistPoint(p, center, axisDir, span, totalAngleRad) {
  const axisComp = dot(sub(p, center), axisDir);
  const angle = totalAngleRad * (axisComp / span);
  return rotatePoint(p, center, axisDir, angle);
}

// Taper — a cross-section scale factor that varies linearly with position
// along axisDir (factor(t) = lerp(1, endFactor, t/span)), extrapolated
// outside [0, span]. Not scalePoint (which stretches the axis-parallel
// component by one fixed factor) — Taper stretches the perpendicular
// component by a factor that changes per point. Approximate, same reason
// as Twist.
export function taperPoint(p, center, axisDir, span, endFactor) {
  const rel = sub(p, center);
  const axisComp = dot(rel, axisDir);
  const perp = sub(rel, scale(axisDir, axisComp));
  const factor = 1 + (endFactor - 1) * (axisComp / span);
  return add(add(center, scale(axisDir, axisComp)), scale(perp, factor));
}

// Bend — a circular-arc bend of a "spine" running from `p1` along `axisDir`
// for `span` mm, through `angleRad` total, in the plane spanned by axisDir
// and `planeNormal` (planeNormal must already be unit length and
// perpendicular to axisDir — the caller's job, matching Shear's own
// measureAxis/shearDir contract). Rhino: Bend (BendSpaceMorph).
//
// Derivation: treat every
// point's own cross-section (its offset from the spine, decomposed into an
// in-plane component `h` along planeNormal and an out-of-plane component
// `w` along thirdDir = axisDir x planeNormal) as a rigid disc that rotates
// together as the spine sweeps through the arc — the same "plane sections
// remain plane" idealization a physically bent beam is drawn with. A point
// sitting on the spine (h=0, w=0) traces the spine's own circular
// arc of radius R = span/angleRad exactly; a point offset by h traces a
// circle of effective radius (R - h) in the same plane (offset "outward"
// from the bend shrinks its own radius, offset "inward" grows it — the
// inner/outer fiber of a bent beam), while w (perpendicular to the
// bend plane entirely) is carried along unchanged, since the rigid-disc
// rotation is about thirdDir itself.
//
// Reduces to the identity at angleRad === 0 (returned directly, since
// R = span/angleRad is undefined there — the same "typed 0 is a no-op"
// convention as Rotate/Twist). Approximate for the same control-point-level reason
// as Twist/Taper (a different rigid-disc rotation per point, not one fixed
// affine map).
export function bendPoint(p, p1, axisDir, span, planeNormal, angleRad) {
  if (angleRad === 0) return [...p];
  const thirdDir = normalize(cross(axisDir, planeNormal));
  const R = span / angleRad;
  const C = add(p1, scale(planeNormal, R));
  const rel = sub(p, p1);
  const axisComp = dot(rel, axisDir);
  const h = dot(rel, planeNormal);
  const w = dot(rel, thirdDir);
  const theta = angleRad * (axisComp / span);
  const Reff = R - h;
  const cosT = Math.cos(theta), sinT = Math.sin(theta);
  return add(
    add(sub(C, scale(planeNormal, Reff * cosT)), scale(axisDir, Reff * sinT)),
    scale(thirdDir, w),
  );
}

// Charybdis — our name; Rhino's own reference is Maelstrom
// (MaelstromSpaceMorph), a radial twist with falloff (where Twist is an
// axial one). Following McNeel's description:
// inside `innerRadius` points rotate fully by `angleRad`; between
// `innerRadius` and `outerRadius` the rotation falls off; outside
// `outerRadius` nothing moves.
//
// `axisDir` (unit) is the rotation axis through `center` — the maelstrom's
// own plane is perpendicular to it. `falloff` is an optional ramp
// (kernel/falloff.mjs); omitted or null, it evaluates as `1 - smoothstep(t)`,
// cage.mjs's zero-derivative-at-the-boundary curve, the same one the
// Sculpt/Cage/Noise brushes use: a hard cutoff at outerRadius would
// leave a visible crease where the deformation stops. An edited curve
// keeps that property — a smooth-interp ramp has zero derivative at every
// stop, not just its two ends.
//
// Approximate, same class as Twist/Taper/Bend — a per-point rotation angle,
// not one fixed affine map.
export function charybdisPoint(p, center, axisDir, innerRadius, outerRadius, angleRad, falloff) {
  const rel = sub(p, center);
  const axisComp = dot(rel, axisDir);
  const planar = sub(rel, scale(axisDir, axisComp));
  const r = Math.sqrt(dot(planar, planar));
  // All three cases (inside innerRadius, outside outerRadius, and between)
  // route through one ramp evaluation: evalFalloffRamp clamps t itself, and
  // the default ramp is exactly 1 at t=0 and exactly 0 at t=1, so the end
  // cases fall out of it and an edited ramp's own endpoints are respected.
  const span = outerRadius - innerRadius;
  // A degenerate (zero or inverted) band has no interior to interpolate
  // across; split hard at innerRadius rather than dividing by zero. Reachable only through a direct kernel call — the
  // app's own registry refuses outerRadius <= innerRadius at commit time.
  const t = span > 0
    ? (r - innerRadius) / span
    : (r <= innerRadius ? 0 : 1);
  const weight = evalFalloffRamp(falloff, t);
  if (weight === 0) return [...p];
  const rotatedPlanar = rotatePoint(planar, [0, 0, 0], axisDir, angleRad * weight);
  return add(add(center, scale(axisDir, axisComp)), rotatedPlanar);
}

// One dispatcher, matching the app's `transformPointMapper`
// shape — `frame` is the plain-data axis frame every
// type shares (center/axisDir/span, plus planeNormal for shear/bend and
// innerRadius/outerRadius plus an optional `falloff` ramp for charybdis),
// `value` is the type's own single
// typed/dragged number (angleRad for rotate/twist/bend/charybdis, factor
// for scale/shear/taper). Returns null for an unrecognized type; callers
// refuse that before reaching here.
export function transformPoint(type, frame, value, p) {
  if (type === 'rotate') return rotatePoint(p, frame.center, frame.axisDir, value);
  if (type === 'scale') return scalePoint(p, frame.center, frame.axisDir, value);
  if (type === 'shear') return shearPoint(p, frame.center, frame.axisDir, frame.planeNormal, value);
  if (type === 'taper') return taperPoint(p, frame.center, frame.axisDir, frame.span, value);
  if (type === 'twist') return twistPoint(p, frame.center, frame.axisDir, frame.span, value);
  if (type === 'bend') return bendPoint(p, frame.center, frame.axisDir, frame.span, frame.planeNormal, value);
  if (type === 'charybdis') return charybdisPoint(p, frame.center, frame.axisDir, frame.innerRadius, frame.outerRadius, value, frame.falloff);
  return null;
}
