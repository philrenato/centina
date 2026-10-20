// Pipe sections that are not round. A section is a closed curve in its frame's
// plane (local x, y; z = 0), counter-clockwise about +z, carried along the rail
// by the sweep. Kinds:
//
//   round     makeCircle, the section Pipe has always had.
//   teardrop  a circle with two tangent lines meeting at a 90-degree tip on +x,
//             at radius * sqrt(2) from the center: every point of the outline is
//             at most 45 degrees from vertical when +x points up, so a channel
//             built along the build direction needs no support inside. Degree 2,
//             exact: a 270-degree arc and two lines. A teardrop of radius r - w
//             is the inward offset of one of radius r by w (the lines stay
//             tangent and the tip stays a corner), so a thick teardrop's wall is
//             even.
//   lobes     r (1 + depth cos(n theta)), n lobes, one on +x; cubic, interpolated
//             with matching end tangents so the start is not a seam.
//   drawn     a closed planar curve, centered on its area centroid and turned
//             into the frame's plane (pipeDrawnSectionLocal).
//
// The ring a SuperB pipe builds its cage from is the section sampled at even
// arc length from its start (pipeSectionRing). Caps for any section are tensor
// products of the section with a meridian curve drawn in (t, a): t blends each
// outer control point toward its inner partner (or the center), a lifts along
// the cap axis (pipeSectionCapSurface). For a circle this is exactly a revolve.

import { makeArc, makeCircle } from './primitives.mjs';
import { globalCurveInterpWithEndDerivs } from './interpolate.mjs';
import { curvePoint, buildArcLengthTable, paramAtArcLength } from './curve.mjs';
import { curvePlane } from './fiber.mjs';
import { offsetCurve2D } from './offsetcurve.mjs';
import { pipeBlendCapProfile } from './sweep.mjs';

export const PIPE_SECTION_KINDS = ['round', 'teardrop', 'lobes', 'drawn'];
export const PIPE_SECTION_LOBES_MIN = 2, PIPE_SECTION_LOBES_MAX = 12;
export const PIPE_SECTION_DEPTH_MIN = 0.05, PIPE_SECTION_DEPTH_MAX = 0.6;

export function pipeSectionKind(spec) {
  const k = spec && spec.kind;
  return PIPE_SECTION_KINDS.includes(k) ? k : 'round';
}

// The section of a given kind and radius, in local coordinates. 'drawn' has no
// radius of its own: pipeDrawnSectionLocal makes it.
export function pipeSectionCurve(spec, radius) {
  if (!(radius > 0)) throw new Error(`a pipe section needs a positive radius (got ${radius})`);
  const kind = pipeSectionKind(spec);
  if (kind === 'teardrop') return teardropSection(radius);
  if (kind === 'lobes') {
    const n = Math.round(clamp(spec.lobes ?? 3, PIPE_SECTION_LOBES_MIN, PIPE_SECTION_LOBES_MAX));
    return lobedSection(radius, n, clamp(spec.depth ?? 0.25, PIPE_SECTION_DEPTH_MIN, PIPE_SECTION_DEPTH_MAX));
  }
  return makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], radius);
}

function teardropSection(r) {
  const s = Math.SQRT1_2;
  const tip = [r * Math.SQRT2, 0, 0];
  const a = [r * s, r * s, 0], d = [r * s, -r * s, 0];
  const arc = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], r, Math.PI / 4, (3 * Math.PI) / 2, 3);
  const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, 0, 1];
  const ctrlPts = [[...tip, 1], mid(tip, a), ...arc.ctrlPts, mid(d, tip), [...tip, 1]];
  return { degree: 2, knots: [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5], ctrlPts };
}

function lobedSection(r, n, depth) {
  const m = Math.max(32, 12 * n);
  const pts = [];
  for (let k = 0; k <= m; k++) {
    const th = (2 * Math.PI * k) / m;
    const rr = r * (1 + depth * Math.cos(n * th));
    pts.push([rr * Math.cos(th), rr * Math.sin(th), 0]);
  }
  pts[m] = pts[0].slice();
  // The tangent at theta = 0 (where the lobe sits, dR/dtheta = 0) is +y; the
  // end derivative is that direction times the chord length, the length of
  // the normalized-chord parameter's derivative.
  let L = 0;
  for (let k = 1; k <= m; k++) L += Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]);
  const crv = globalCurveInterpWithEndDerivs(pts, [0, L, 0], [0, L, 0]);
  crv.ctrlPts = crv.ctrlPts.map((p) => [p[0], p[1], 0, p.length > 3 ? p[3] : 1]);
  return crv;
}

// A closed planar world curve as a section in `frame`: its area centroid on the
// frame origin, its plane turned onto the frame's plane by the smallest
// rotation (so it keeps the orientation it was drawn in), counter-clockwise
// about +z. Its own size is kept. Null when the curve is not closed and planar.
export function pipeDrawnSectionLocal(crv, frame) {
  const plane = curvePlane(crv);
  if (!plane) return null;
  let n = plane.normal;
  if (psDot(n, frame.zAxis) < 0) n = psScale(n, -1);
  const poly = sampleClosed(crv, 256);
  const c = polygonCentroid(poly, n);
  const toLocal = (p) => {
    const q = rotateMinimally3(psSub(p, c), n, frame.zAxis);
    return [psDot(q, frame.xAxis), psDot(q, frame.yAxis), 0];
  };
  let local = { degree: crv.degree, knots: crv.knots.slice(), ctrlPts: crv.ctrlPts.map((p) => [...toLocal(p), p.length > 3 ? p[3] : 1]) };
  if (psSignedArea(sampleClosed(local, 256)) < 0) local = psReverse(local);
  return local;
}

// The mean distance of a local section from its center: the radius a dome or a
// fitted cage reads for a section that has no radius parameter.
export function pipeSectionMeanRadius(local) {
  const pts = sampleClosed(local, 256);
  return pts.reduce((s, p) => s + Math.hypot(p[0], p[1]), 0) / pts.length;
}

// `facets` points on the section at even arc length from its start, as [x, y].
// For a teardrop the first is the tip.
export function pipeSectionRing(local, facets) {
  const tbl = buildArcLengthTable(local, local.knots[0], local.knots[local.knots.length - 1]);
  const out = [];
  for (let i = 0; i < facets; i++) {
    const u = paramAtArcLength(tbl, (tbl.total * i) / facets);
    const p = curvePoint(local, u);
    out.push([p[0], p[1]]);
  }
  return out;
}

// The inner wall of a thick section: a teardrop or circle of the smaller radius
// (an exact offset), any other kind offset inward along its normals.
export function pipeSectionInner(local, spec, radius, wall) {
  const kind = pipeSectionKind(spec);
  if ((kind === 'round' || kind === 'teardrop') && radius - wall > 0) return pipeSectionCurve(spec, radius - wall);
  const inner = offsetCurve2D(local, -wall, [0, 0, 1]);
  inner.ctrlPts = inner.ctrlPts.map((p) => [p[0], p[1], 0, p.length > 3 ? p[3] : 1]);
  return inner;
}

// The frame options a section is swept with. Round keeps Pipe's own frames;
// any other section is carried on a rotation-minimizing frame, and a teardrop's
// tip is held toward `up`.
export function pipeSectionFrameOpts(spec) {
  const kind = pipeSectionKind(spec);
  if (kind === 'round') return {};
  if (kind === 'teardrop') return { transport: 'rmf', up: unitOr(spec.up, [0, 0, 1]) };
  return { transport: 'rmf' };
}

// A cap on a section's end. `outer` and `inner` are local sections with the
// same knots (inner null: the cap closes to the center). `meridian` is a curve
// whose control points are [t, a, 0, w]: t = 0 is the outer wall, t = 1 the
// inner one (or the center), a the height along `axisDir`.
export function pipeSectionCapSurface(frame, outer, inner, meridian, axisDir) {
  if (inner && inner.ctrlPts.length !== outer.ctrlPts.length) throw new Error('a section cap needs inner and outer sections with the same control points');
  const ctrlNet = outer.ctrlPts.map((o, i) => {
    const q = inner ? inner.ctrlPts[i] : [0, 0, 0, o[3]];
    return meridian.ctrlPts.map(([t, a, , wm]) => {
      const x = o[0] * (1 - t) + q[0] * t, y = o[1] * (1 - t) + q[1] * t;
      const p = psAdd(frame.origin, psAdd(psAdd(psScale(frame.xAxis, x), psScale(frame.yAxis, y)), psScale(axisDir, a)));
      return [...p, (o[3] ?? 1) * (wm ?? 1)];
    });
  });
  return { degU: outer.degree, knotsU: outer.knots.slice(), degV: meridian.degree, knotsV: meridian.knots.slice(), ctrlNet };
}

// A quarter-circle meridian: the dome over a section, `height` tall.
export function pipeDomeMeridian(height) {
  return { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[0, 0, 0, 1], [0, height, 0, Math.SQRT1_2], [1, height, 0, 1]] };
}

// The Blend cap's meridian between an outer and an inner wall, with the same
// fillets pipeBlendCapSurface revolves: its profile drawn in (r, a), then r
// mapped to t across the wall.
export function pipeBlendMeridian(outerRadius, innerRadius, outerFilletRadius, innerFilletRadius) {
  const wall = outerRadius - innerRadius;
  if (!(wall > 0)) throw new Error('a blend cap needs an inner wall inside the outer one');
  const res = pipeBlendCapProfile({ origin: [0, 0, 0], xAxis: [1, 0, 0] }, outerRadius, innerRadius, [0, 0, 1], outerFilletRadius, innerFilletRadius);
  const meridian = { degree: res.profile.degree, knots: res.profile.knots.slice(), ctrlPts: res.profile.ctrlPts.map((p) => [(outerRadius - p[0]) / wall, p[2], 0, p[3] ?? 1]) };
  return { meridian, appliedOuterFilletRadius: res.appliedOuterFilletRadius, appliedInnerFilletRadius: res.appliedInnerFilletRadius };
}

function sampleClosed(crv, n) {
  const u0 = crv.knots[0], u1 = crv.knots[crv.knots.length - 1];
  const out = [];
  for (let i = 0; i < n; i++) out.push(curvePoint(crv, u0 + ((u1 - u0) * i) / n));
  return out;
}
function psSignedArea(pts) {
  let s = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) s += pts[j][0] * pts[i][1] - pts[i][0] * pts[j][1];
  return s / 2;
}
// The area centroid of a planar polygon with normal n (fan from the first point).
function polygonCentroid(pts, n) {
  let A = 0, c = [0, 0, 0];
  for (let i = 1; i + 1 < pts.length; i++) {
    const a = psDot(psCross(psSub(pts[i], pts[0]), psSub(pts[i + 1], pts[0])), n) / 2;
    const g = psScale(psAdd(psAdd(pts[0], pts[i]), pts[i + 1]), 1 / 3);
    A += a; c = psAdd(c, psScale(g, a));
  }
  return Math.abs(A) > 1e-12 ? psScale(c, 1 / A) : psScale(pts.reduce((s, p) => psAdd(s, p), [0, 0, 0]), 1 / pts.length);
}
function psReverse(crv) {
  const k = crv.knots, a = k[0], b = k[k.length - 1];
  return { degree: crv.degree, knots: k.map((x) => a + b - x).reverse(), ctrlPts: crv.ctrlPts.slice().reverse() };
}
function rotateMinimally3(r, a, b) {
  const k = psCross(a, b), s2 = psDot(k, k), c = psDot(a, b);
  if (s2 < 1e-24) return r;
  return psAdd(psAdd(psScale(r, c), psCross(k, r)), psScale(k, (psDot(k, r) * (1 - c)) / s2));
}
function unitOr(v, fallback) {
  if (!Array.isArray(v) || v.length < 3 || !v.every(Number.isFinite)) return fallback;
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 1e-12 ? [v[0] / l, v[1] / l, v[2] / l] : fallback;
}
function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, Number(x))); }
function psAdd(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function psSub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function psScale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function psDot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function psCross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
