// Projects an object's edges to a plane at a scale, the primitive a
// dimensioned drawing view is built on. A plane is
// { origin, uAxis, vAxis, scale } — uAxis/vAxis orthonormal in-plane
// basis vectors (e.g. a Top/Front/Right/Iso view).
//
// Tessellation here is uniform-parameter sampling, not the adaptive
// chord-tolerance scheme the viewport display pipeline uses; this primitive
// only needs a polyline faithful enough to project and dimension.

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
