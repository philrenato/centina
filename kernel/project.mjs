// "Project this object's edges to a plane at a scale" — the primitive
// the dimensioning work names explicitly, so the
// Spec Sheet room (v2 target) slots in later without kernel rework. A plane
// is { origin, uAxis, vAxis, scale } — uAxis/vAxis orthonormal in-plane
// basis vectors (e.g. a Spec Sheet view's Top/Front/Right/Iso projection).
//
// Tessellation here is uniform-parameter sampling, not the adaptive
// chord-tolerance scheme the viewport display pipeline uses —
// that's a P1 concern tied to view-dependent redraw; this primitive only
// needs to produce a faithful enough polyline to project and dimension.

import { sub, dot } from './vec3.mjs';
import { curvePoint, assertCurve } from './curve.mjs';

export function projectPointToPlane(p, plane) {
  const rel = sub(p, plane.origin);
  const u = dot(rel, plane.uAxis);
  const v = dot(rel, plane.vAxis);
  return [u * plane.scale, v * plane.scale];
}

export function projectPolylineToPlane(points, plane) {
  return points.map((p) => projectPointToPlane(p, plane));
}

export function tessellateCurve(crv, samples = 64) {
  assertCurve(crv, 'tessellateCurve');
  const { degree: p, knots: U } = crv;
  const uMin = U[p], uMax = U[U.length - 1 - p];
  const pts = [];
  for (let i = 0; i <= samples; i++) {
    const u = uMin + (i / samples) * (uMax - uMin);
    pts.push(curvePoint(crv, u));
  }
  return pts;
}

export function projectCurveToPlane(crv, plane, samples = 64) {
  return projectPolylineToPlane(tessellateCurve(crv, samples), plane);
}
