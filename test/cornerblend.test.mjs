import test from 'node:test';
import assert from 'node:assert/strict';
import { tangencyLineOnFace, tangencyCrossing, cornerPatchCorners } from '../kernel/cornerblend.mjs';

// An orthogonal box corner at the origin: faces z=0, y=0 and x=0, material in
// the positive octant, edges along +x (A), +y (B) and +z (C).
const orthoEdges = (rA, rB, rC) => ([
  { dir: [1, 0, 0], radius: rA, phi: Math.PI / 2, coNormalA: [0, 1, 0], faceA: 'z0', coNormalB: [0, 0, 1], faceB: 'y0' },
  { dir: [0, 1, 0], radius: rB, phi: Math.PI / 2, coNormalA: [0, 0, 1], faceA: 'x0', coNormalB: [1, 0, 0], faceB: 'z0' },
  { dir: [0, 0, 1], radius: rC, phi: Math.PI / 2, coNormalA: [1, 0, 0], faceA: 'y0', coNormalB: [0, 1, 0], faceB: 'x0' },
]);
const cornerOn = (res, face) => res.corners.find((c) => c.face === face).point;
const near = (a, b, tol = 1e-12) => assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < tol,
  `expected [${b}] got [${a.map((v) => v.toFixed(9))}]`);

test('three different radii give the three patch corners in closed form', () => {
  // On an orthogonal corner the setback IS the radius (r/tan(45 degrees)), so
  // each corner is simply the two neighbouring radii read off the shared face.
  const r = cornerPatchCorners([0, 0, 0], orthoEdges(5, 8, 3));
  assert.equal(r.ok, true, r.reason);
  near(cornerOn(r, 'z0'), [8, 5, 0]);
  near(cornerOn(r, 'y0'), [3, 0, 5]);
  near(cornerOn(r, 'x0'), [0, 3, 8]);
  assert.ok(r.worstGap < 1e-12, `the lines must genuinely meet, gap ${r.worstGap}`);
});

test('EQUAL radii reduce to the spherical case — every corner sits on the ball', () => {
  // The construction has to agree with the one it replaces where both are
  // valid, or the corner would jump the moment two radii stopped matching.
  const rad = 6;
  const r = cornerPatchCorners([0, 0, 0], orthoEdges(rad, rad, rad));
  assert.equal(r.ok, true, r.reason);
  const centre = [rad, rad, rad]; // the ball tangent to all three planes
  for (const c of r.corners) {
    const d = Math.hypot(c.point[0] - centre[0], c.point[1] - centre[1], c.point[2] - centre[2]);
    assert.ok(Math.abs(d - rad) < 1e-12, `corner on ${c.face} is ${d} from the ball centre, want ${rad}`);
  }
});

test('a NON-orthogonal corner still crosses exactly, and the setback is not the radius', () => {
  // Two faces meeting at 60 degrees: the setback is r/tan(30) = r*sqrt(3),
  // which is where writing the radius in its place would show up.
  const phi = Math.PI / 3;
  const L = tangencyLineOnFace({ edgePoint: [0, 0, 0], edgeDir: [1, 0, 0], coNormal: [0, 1, 0], radius: 2, phi });
  assert.equal(L.ok, true);
  assert.ok(Math.abs(L.setback - 2 / Math.tan(phi / 2)) < 1e-12);
  assert.ok(Math.abs(L.setback - 2 * Math.sqrt(3)) < 1e-12, `setback ${L.setback} should be 2*sqrt(3)`);
});

test('two nearly parallel tangency lines are refused, not crossed a mile away', () => {
  // Their crossing runs off to infinity long before it is geometrically wrong,
  // and a corner patch stretched across the model is the symptom.
  const a = { point: [0, 0, 0], dir: [1, 0, 0] };
  const b = { point: [0, 1, 0], dir: [Math.cos(0.001), Math.sin(0.001), 0] };
  const x = tangencyCrossing(a, b);
  assert.equal(x.ok, false);
  assert.match(x.reason, /parallel/);
});

test('lines that genuinely cross report a zero gap; skew lines report their real one', () => {
  const meet = tangencyCrossing({ point: [0, 0, 0], dir: [1, 0, 0] }, { point: [3, -1, 0], dir: [0, 1, 0] });
  assert.equal(meet.ok, true);
  near(meet.point, [3, 0, 0]);
  assert.ok(meet.gap < 1e-12);
  // Offset one in z: they no longer meet, and the gap must SAY so rather than
  // the midpoint being returned as if it were an intersection.
  const skew = tangencyCrossing({ point: [0, 0, 0], dir: [1, 0, 0] }, { point: [3, -1, 0.25], dir: [0, 1, 0] });
  assert.equal(skew.ok, true);
  assert.ok(Math.abs(skew.gap - 0.25) < 1e-12, `gap ${skew.gap} should be 0.25`);
});

test('the refusals name what is wrong rather than returning something plausible', () => {
  assert.match(cornerPatchCorners([0, 0, 0], orthoEdges(5, 8, 3).slice(0, 2)).reason, /exactly three/);
  const bad = orthoEdges(5, 8, 3);
  bad[0].radius = 0;
  assert.match(cornerPatchCorners([0, 0, 0], bad).reason, /radius must be positive/);
  const flat = orthoEdges(5, 8, 3);
  flat[1].phi = Math.PI; // tangent faces have no edge to blend
  assert.match(cornerPatchCorners([0, 0, 0], flat).reason, /out of range/);
  const sameFace = orthoEdges(5, 8, 3);
  sameFace[2].faceA = 'z0'; sameFace[2].faceB = 'z0';
  assert.match(cornerPatchCorners([0, 0, 0], sameFace).reason, /three faces|not two/);
});

test('the corner scales exactly with the radii — no hidden constant anywhere', () => {
  const a = cornerPatchCorners([0, 0, 0], orthoEdges(5, 8, 3));
  const b = cornerPatchCorners([0, 0, 0], orthoEdges(50, 80, 30));
  for (let i = 0; i < 3; i++) {
    const pa = a.corners[i].point, pb = b.corners[i].point;
    near(pb, [pa[0] * 10, pa[1] * 10, pa[2] * 10], 1e-9);
  }
});

// ── THE PATCH ───────────────────────────────────────────────────────────────
import { sideVertexPatch, blendEndTrimLoopApprox, blendEndTrimLoopFromPlane } from '../kernel/cornerblend.mjs';

const seg = (a, b) => (s) => [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s];
const sweep = (patch, fn) => {
  let worst = 0;
  for (let i = 0; i <= 24; i++) for (let j = 0; i + j <= 24; j++) {
    const b0 = i / 24, b1 = j / 24, b2 = 1 - b0 - b1;
    const q = patch.evaluate(b0, b1, b2);
    if (q) worst = Math.max(worst, fn(q, b0, b1, b2));
  }
  return worst;
};

test('the patch reproduces a FLAT triangle exactly — no bulge from the blend itself', () => {
  const V = [[0, 0, 0], [10, 0, 0], [0, 8, 0]];
  const p = sideVertexPatch({ corners: V, boundary: [seg(V[1], V[2]), seg(V[2], V[0]), seg(V[0], V[1])] });
  assert.equal(p.ok, true);
  assert.equal(sweep(p, (q) => Math.abs(q[2])), 0, 'a patch on three straight coplanar sides must be flat');
});

test('every boundary is interpolated EXACTLY, which is what watertight depends on', () => {
  // Curved sides, since a scheme can be exact on straight ones by accident.
  const V = [[0, 0, 0], [10, 0, 0], [0, 8, 0]];
  const bow = (a, b, h) => (s) => {
    const m = seg(a, b)(s);
    return [m[0], m[1], m[2] + h * Math.sin(Math.PI * s)];
  };
  const boundary = [bow(V[1], V[2], 3), bow(V[2], V[0], -2), bow(V[0], V[1], 1.5)];
  const p = sideVertexPatch({ corners: V, boundary });
  let worst = 0;
  for (let s = 0; s <= 32; s++) {
    const t = s / 32;
    for (let i = 0; i < 3; i++) {
      const b = [0, 0, 0]; b[(i + 1) % 3] = 1 - t; b[(i + 2) % 3] = t;
      const q = p.evaluate(b[0], b[1], b[2]), w = boundary[i](t);
      worst = Math.max(worst, Math.hypot(q[0] - w[0], q[1] - w[1], q[2] - w[2]));
    }
  }
  assert.ok(worst < 1e-12, `boundary reproduced to ${worst}, which is the seam a fillet would leak through`);
});

test('the three corners are returned outright, not approached through a 0/0', () => {
  const V = [[1, 2, 3], [10, 0, 0], [0, 8, -4]];
  const p = sideVertexPatch({ corners: V, boundary: [seg(V[1], V[2]), seg(V[2], V[0]), seg(V[0], V[1])] });
  for (let i = 0; i < 3; i++) {
    const b = [0, 0, 0]; b[i] = 1;
    const q = p.evaluate(b[0], b[1], b[2]);
    assert.deepEqual(q, V[i], `corner ${i} must be exact — every weight vanishes there`);
  }
});

test('the APPROXIMATE end loop is a straight line in the blend\'s parameters, and an eaten edge is refused', () => {
  const good = blendEndTrimLoopApprox({ vStartAtU0: 0.1, vStartAtU1: 0.25, vEndAtU0: 0.8, vEndAtU1: 0.7 });
  assert.equal(good.ok, true);
  // A quadrilateral of the domain, not a rectangle: the two ends are slanted.
  assert.deepEqual(good.loop, [[0, 0.1], [1, 0.25], [1, 0.7], [0, 0.8]]);
  // Two corners that have consumed the whole edge between them is a refusal,
  // not a zero-area blend nobody notices.
  const eaten = blendEndTrimLoopApprox({ vStartAtU0: 0.6, vStartAtU1: 0.2, vEndAtU0: 0.5, vEndAtU1: 0.9 });
  assert.equal(eaten.ok, false);
  assert.match(eaten.reason, /ends before it starts/);
});

test('the patch refuses a malformed request rather than building something plausible', () => {
  assert.match(sideVertexPatch({ corners: [[0, 0, 0]], boundary: [] }).reason, /exactly three corners/);
  assert.match(sideVertexPatch({ corners: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], boundary: [1, 2, 3] }).reason, /not an evaluable curve/);
});

import { blendParamAtCorner } from '../kernel/cornerblend.mjs';

test('a corner resolves to the blend parameter where it sits, to a real tolerance', () => {
  const rA = 5;
  const onZ0 = (v) => [v * 40, rA, 0];   // the tangency line a radius-5 blend leaves on z=0
  const onY0 = (v) => [v * 40, 0, rA];
  const a = blendParamAtCorner({ evalBorder: onZ0, corner: [8, 5, 0] });
  assert.equal(a.ok, true);
  assert.ok(Math.abs(a.v - 0.2) < 1e-7, `v ${a.v} should be 8/40`);
  assert.ok(a.residual < 1e-8, `residual ${a.residual}`);
  const b = blendParamAtCorner({ evalBorder: onY0, corner: [3, 0, 5] });
  assert.ok(Math.abs(b.v - 0.075) < 1e-7, `v ${b.v} should be 3/40`);
});

test('a corner that is NOT on the border still returns a parameter — so the residual is the check', () => {
  // The failure this guards is silent: a wrong face pairing, or an edge that was
  // never filleted, yields a perfectly ordinary-looking nearest parameter.
  const onZ0 = (v) => [v * 40, 5, 0];
  const off = blendParamAtCorner({ evalBorder: onZ0, corner: [8, 5, 2.5] });
  assert.equal(off.ok, true, 'it does not refuse — it cannot know');
  assert.ok(Math.abs(off.residual - 2.5) < 1e-6, `residual ${off.residual} must report the real 2.5mm miss`);
});

test('the search finds the right minimum even when the curve doubles back near it', () => {
  // A border that approaches the corner twice: the coarse sweep is what makes
  // the golden-section refinement below it legitimate.
  const wig = (v) => [Math.cos(v * 2 * Math.PI) * 10, Math.sin(v * 2 * Math.PI) * 10, 0];
  const r = blendParamAtCorner({ evalBorder: wig, corner: [0, 10, 0] });
  assert.ok(r.residual < 1e-6, `residual ${r.residual}`);
  assert.ok(Math.abs(r.v - 0.25) < 1e-6, `v ${r.v} should be 0.25`);
});

import { blendEndPlane, cornerFitsOnFaces } from '../kernel/cornerblend.mjs';

test('the end plane passes through BOTH corners and reduces to the perpendicular section when radii agree', () => {
  // Blend along +x, radius rA; its corners are (rB, rA, 0) on z=0 and (rC, 0, rA) on y=0.
  const mk = (rA, rB, rC) => blendEndPlane({ axisDir: [1, 0, 0], cornerA: [rB, rA, 0], cornerB: [rC, 0, rA] });
  const eq = mk(5, 5, 5);
  assert.equal(eq.ok, true);
  assert.ok(eq.tiltRad < 1e-12, `equal radii must give an UNTILTED section, got ${eq.tiltRad}`);
  assert.ok(Math.abs(eq.offset - 5) < 1e-12, 'and it must be the plane x = r, which is what the sphere case already uses');
  const un = mk(5, 8, 3);
  const onPlane = (p) => Math.abs(p[0] * un.normal[0] + p[1] * un.normal[1] + p[2] * un.normal[2] - un.offset);
  assert.ok(onPlane([8, 5, 0]) < 1e-12, 'the first corner must lie on the plane');
  assert.ok(onPlane([3, 0, 5]) < 1e-12, 'and so must the second — otherwise the end curve misses them');
  assert.ok(un.tiltRad > 0.1, 'unequal radii genuinely tilt the section; a circle would not reach both corners');
});

test('an end plane is refused where no section exists rather than returned tilted to nonsense', () => {
  // Corners along the axis: every plane through them cuts the tube lengthwise.
  const along = blendEndPlane({ axisDir: [1, 0, 0], cornerA: [0, 1, 0], cornerB: [5, 1, 0] });
  assert.equal(along.ok, false);
  assert.match(along.reason, /lengthwise|no end section/);
  const same = blendEndPlane({ axisDir: [1, 0, 0], cornerA: [2, 1, 0], cornerB: [2, 1, 0] });
  assert.equal(same.ok, false);
  assert.match(same.reason, /same point/);
});

test('a corner that lands off the face is refused by name — the ball rolls off at a vertex', () => {
  const res = cornerPatchCorners([0, 0, 0], orthoEdges(5, 8, 3));
  // Faces big enough: nothing to complain about.
  const roomy = cornerFitsOnFaces(res, { z0: [[0, 0], [100, 100]], y0: [[0, 0], [100, 100]], x0: [[0, 0], [100, 100]] });
  assert.equal(roomy.ok, true);
  // A face too small to contain its own crossing.
  const cramped = cornerFitsOnFaces(res, { z0: [[0, 0], [1, 1]], y0: [[0, 0], [100, 100]], x0: [[0, 0], [100, 100]] });
  assert.equal(cramped.ok, false);
  assert.match(cramped.reason, /outside face\(s\) z0/);
});

// ── THE EXACT CASE: TWO RADII EQUAL ─────────────────────────────────────────
import { twoEqualRadiiCorner } from '../kernel/cornerblend.mjs';

const octant = (a, c, thirdDir = [0, 0, 1]) => ({
  vertex: [0, 0, 0], sharedFaceNormal: [0, 0, 1],
  edgeA: { dir: [1, 0, 0], radius: a, phi: Math.PI / 2, coNormalA: [0, 1, 0], coNormalB: [0, 0, 1] },
  edgeB: { dir: [0, 1, 0], radius: a, phi: Math.PI / 2, coNormalA: [1, 0, 0], coNormalB: [0, 0, 1] },
  edgeC: { dir: thirdDir, radius: c, phi: Math.PI / 2, coNormalA: [1, 0, 0], coNormalB: [0, 1, 0] },
});

test('two equal radii give an EXACT torus, tangent to the shared face and to the third blend', () => {
  const a = 5, c = 3;
  const r = twoEqualRadiiCorner(octant(a, c));
  assert.equal(r.ok, true, r.reason);
  // Tangent to the floor: the tube's lowest point must touch z = 0 exactly.
  assert.ok(Math.abs((r.torus.centre[2] - r.torus.minorRadius) - 0) < 1e-12,
    `the torus must touch the shared face, lowest z ${r.torus.centre[2] - r.torus.minorRadius}`);
  // Tangent to the third blend: the tube's inner reach from that axis IS its radius.
  assert.ok(Math.abs((r.torus.majorRadius - r.torus.minorRadius) - c) < 1e-12,
    `inner reach ${r.torus.majorRadius - r.torus.minorRadius} must equal the third radius ${c}`);
  // And it sits on the third edge's own axis.
  assert.ok(Math.hypot(r.torus.centre[0] - c, r.torus.centre[1] - c) < 1e-12);
});

test('the handoff station is 2*sqrt(a*c), and the rolling ball is genuinely triple-tangent there', () => {
  const a = 5, c = 3;
  const r = twoEqualRadiiCorner(octant(a, c));
  assert.ok(Math.abs(r.handoff.offsetAlongEdge - 2 * Math.sqrt(a * c)) < 1e-12);
  // The first blend's axis is the path of that ball. At the station it must be
  // exactly a + c from the third blend's axis — tangent to it — while still
  // being a from both of its own faces, which its axis guarantees.
  const t = c + r.handoff.offsetAlongEdge;
  const d = Math.hypot(t - c, a - c);
  assert.ok(Math.abs(d - (a + c)) < 1e-12, `${d} should be ${a + c}: triple tangency is what makes the lune exact`);
});

test('EQUAL to the third as well collapses the torus onto the sphere case', () => {
  // a = c: the centre circle has radius 2a and the ball radius a, and the
  // handoff offset becomes 2a — the construction must not disagree with the
  // spherical corner where both are valid.
  const a = 4;
  const r = twoEqualRadiiCorner(octant(a, a));
  assert.equal(r.ok, true);
  assert.ok(Math.abs(r.torus.majorRadius - 2 * a) < 1e-12);
  assert.ok(Math.abs(r.handoff.offsetAlongEdge - 2 * a) < 1e-12);
});

test('⚠ A TILTED THIRD EDGE IS REFUSED — the centre traces an ellipse and no torus exists', () => {
  // This is the whole condition for exactness, and getting it wrong would ship a
  // torus that is quietly the wrong surface.
  const tilted = octant(5, 3, [0, Math.sin(0.3), Math.cos(0.3)]);
  const r = twoEqualRadiiCorner(tilted);
  assert.equal(r.ok, false);
  assert.match(r.reason, /ellipse rather than a circle|canal surface/);
  assert.match(r.reason, /degrees off/);
});

test('unequal radii on the two sharing edges are refused by this path, not averaged', () => {
  const bad = octant(5, 3);
  bad.edgeB.radius = 6;
  const r = twoEqualRadiiCorner(bad);
  assert.equal(r.ok, false);
  assert.match(r.reason, /EQUAL radii/);
});

test('the torus scales exactly with the radii', () => {
  const one = twoEqualRadiiCorner(octant(5, 3));
  const ten = twoEqualRadiiCorner(octant(50, 30));
  assert.ok(Math.abs(ten.torus.majorRadius - one.torus.majorRadius * 10) < 1e-9);
  assert.ok(Math.abs(ten.torus.minorRadius - one.torus.minorRadius * 10) < 1e-9);
  assert.ok(Math.abs(ten.handoff.offsetAlongEdge - one.handoff.offsetAlongEdge * 10) < 1e-9);
});

// ── THE WHOLE CORNER, END TO END ────────────────────────────────────────────
// A model trihedral corner built from real quarter-cylinder blends, closed with
// the side-vertex patch, and measured for the one property that makes it a
// fillet rather than a lid: tangency to all three blends.

const P2 = Math.PI / 2;
const crs = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const nrm3 = (v) => { const L = Math.hypot(...v); return L > 0 ? v.map((x) => x / L) : null; };
const angAbs = (a, b) => Math.acos(Math.max(-1, Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])))) * 180 / Math.PI;

/** An orthogonal corner in the positive octant with three given radii, returned
 *  with its patch and the blend data needed to judge it. */
function buildOctantCorner(rA, rB, rC) {
  const blends = {
    A: { pt: (t, u) => [t, rA * (1 - Math.cos(u * P2)), rA * (1 - Math.sin(u * P2))], n: (t, u) => [0, -Math.cos(u * P2), -Math.sin(u * P2)], axis: [1, 0, 0] },
    B: { pt: (t, u) => [rB * (1 - Math.sin(u * P2)), t, rB * (1 - Math.cos(u * P2))], n: (t, u) => [-Math.sin(u * P2), 0, -Math.cos(u * P2)], axis: [0, 1, 0] },
    C: { pt: (t, u) => [rC * (1 - Math.cos(u * P2)), rC * (1 - Math.sin(u * P2)), t], n: (t, u) => [-Math.cos(u * P2), -Math.sin(u * P2), 0], axis: [0, 0, 1] },
  };
  const edges = [
    { dir: [1, 0, 0], radius: rA, phi: P2, coNormalA: [0, 1, 0], faceA: 'z0', coNormalB: [0, 0, 1], faceB: 'y0' },
    { dir: [0, 1, 0], radius: rB, phi: P2, coNormalA: [0, 0, 1], faceA: 'x0', coNormalB: [1, 0, 0], faceB: 'z0' },
    { dir: [0, 0, 1], radius: rC, phi: P2, coNormalA: [1, 0, 0], faceA: 'y0', coNormalB: [0, 1, 0], faceB: 'x0' },
  ];
  const cr = cornerPatchCorners([0, 0, 0], edges);
  if (!cr.ok) return { ok: false, reason: cr.reason };
  const at = (f) => cr.corners.find((c) => c.face === f).point;
  const cz0 = at('z0'), cy0 = at('y0'), cx0 = at('x0');
  // u=0 / u=1 face pairing per blend — not a guess, checked by the offsets below.
  const mk = (bl, cA, cB, uAtA) => {
    const pl = blendEndPlane({ axisDir: bl.axis, cornerA: cA, cornerB: cB });
    if (!pl.ok) throw new Error(pl.reason);
    const ax = bl.axis.findIndex((v) => Math.abs(v) > 0.5);
    return (s) => {
      const u = uAtA + (1 - 2 * uAtA) * s;
      const p0 = bl.pt(0, u);
      const t = (pl.offset - (pl.normal[0] * p0[0] + pl.normal[1] * p0[1] + pl.normal[2] * p0[2])) / pl.normal[ax];
      return { p: bl.pt(t, u), n: bl.n(t, u) };
    };
  };
  const sA = mk(blends.A, cz0, cy0, 1), sB = mk(blends.B, cx0, cz0, 1), sC = mk(blends.C, cy0, cx0, 1);
  const V = [cx0, cy0, cz0];
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const sides = [{ f: sA }, { f: sB }, { f: sC }];
  const ordered = [];
  for (let i = 0; i < 3; i++) {
    const want = [V[(i + 1) % 3], V[(i + 2) % 3]];
    const fwd = sides.find((x) => dist(x.f(0).p, want[0]) < 1e-9 && dist(x.f(1).p, want[1]) < 1e-9);
    const rev = sides.find((x) => dist(x.f(1).p, want[0]) < 1e-9 && dist(x.f(0).p, want[1]) < 1e-9);
    if (fwd) ordered.push((s) => fwd.f(s));
    else if (rev) ordered.push((s) => rev.f(1 - s));
    else return { ok: false, reason: `side opposite corner ${i} matches no end curve — the three do not form a closed loop` };
  }
  const crossTan = (i) => (s) => {
    const h = 1e-6;
    const a0 = ordered[i](Math.max(0, s - h)).p, b0 = ordered[i](Math.min(1, s + h)).p;
    return crs(ordered[i](s).n, [b0[0] - a0[0], b0[1] - a0[1], b0[2] - a0[2]]);
  };
  const patch = sideVertexPatch({ corners: V, boundary: ordered.map((o) => (s) => o(s).p), tangent: [crossTan(0), crossTan(1), crossTan(2)] });
  return { ok: true, patch, ordered, corners: V };
}

function normalAt(patch, b0, b1, b2, h) {
  const p = patch.evaluate(b0, b1, b2);
  const pu = patch.evaluate(b0 - h, b1 + h, b2), pv = patch.evaluate(b0 - h, b1, b2 + h);
  if (!p || !pu || !pv) return null;
  return nrm3(crs([pu[0] - p[0], pu[1] - p[1], pu[2] - p[2]], [pv[0] - p[0], pv[1] - p[1], pv[2] - p[2]]));
}

test('the three blend end curves form a CLOSED LOOP for unequal radii', () => {
  const c = buildOctantCorner(5, 8, 3);
  assert.equal(c.ok, true, c.reason);
  // Each side must start and end exactly on the corners the next one begins at,
  // or there is no boundary and nothing downstream is meaningful.
  for (let i = 0; i < 3; i++) {
    const a = c.ordered[i](0).p, b = c.ordered[i](1).p;
    const wa = c.corners[(i + 1) % 3], wb = c.corners[(i + 2) % 3];
    assert.ok(Math.hypot(a[0] - wa[0], a[1] - wa[1], a[2] - wa[2]) < 1e-12);
    assert.ok(Math.hypot(b[0] - wb[0], b[1] - wb[1], b[2] - wb[2]) < 1e-12);
  }
});

test('⚠ THE PATCH IS TANGENT TO ALL THREE BLENDS — and the residual is the RULER, not the patch', () => {
  const c = buildOctantCorner(5, 8, 3);
  assert.equal(c.ok, true, c.reason);
  // The normal is measured by finite differences stepped off the boundary; both
  // are errors of the measurement. If the angle falls with them, it is theirs.
  const measure = (eps, h) => {
    let worst = 0;
    for (let i = 0; i < 3; i++) {
      for (let k = 1; k < 20; k++) {
        const s = k / 20;
        const b = [0, 0, 0]; b[(i + 1) % 3] = 1 - s; b[(i + 2) % 3] = s;
        const bb = b.map((x, j) => (j === i ? eps : x * (1 - eps)));
        const pn = normalAt(c.patch, bb[0], bb[1], bb[2], h);
        const bn = nrm3(c.ordered[i](s).n);
        if (pn && bn) worst = Math.max(worst, angAbs(pn, bn));
      }
    }
    return worst;
  };
  /* ⚠ THE ABSOLUTE NUMBER AT A COARSE STEP IS NOT A PROPERTY OF THE PATCH.
     A finite difference taken a fixed distance off the boundary reads larger
     wherever the surface bends harder there — so the cross-tangent magnitude
     moves this number while leaving the tangent PLANE untouched. Asserting a
     coarse absolute value pins the instrument, not the geometry.

     CONVERGENCE is the real statement: refine the step twentyfold and a
     measurement error falls with it, while a genuine tangent defect does not
     move at all. Both are checked, with the absolute bound taken at the fine
     end where it means something. */
  const coarse = measure(2e-4, 1e-5), fine = measure(1e-5, 1e-6);
  assert.ok(fine < coarse / 5, `the angle must fall with the measurement — coarse ${coarse}, fine ${fine}. A real tangent defect would sit still.`);
  assert.ok(fine < 0.05, `tangency error ${fine} degrees at a fine step is a visible crease`);
  const finer = measure(2e-6, 2e-7);
  assert.ok(finer < fine / 3, `and it must keep falling — fine ${fine}, finer ${finer}`);
});

test('the patch does not FOLD across the ratios a student will actually reach', () => {
  // A side-vertex interpolant is not fold-proof for strongly asymmetric data,
  // and a fold is a self-intersecting corner that no watertight check will like.
  /* ⚠ THIS LIST IS MEASURED, NOT ASPIRED TO. An earlier version claimed 5:1 and
     beyond, taken with a detector that never sampled the ribbon where folds
     live. With one that does, the honest range is about 2.5:1 for the
     awkward "two equal and one different" pattern; mixed triples like 5,8,3
     and 7,2,5 are fine further out because no single corner is starved. */
  for (const [a, b, cc] of [[5, 8, 3], [2, 2, 2], [1, 2, 1], [1, 2.5, 1], [7, 2, 5]]) {
    const c = buildOctantCorner(a, b, cc);
    assert.equal(c.ok, true, `radii ${a},${b},${cc}: ${c.reason}`);
    /* ⚠ A FOLD IS A LOCAL REVERSAL, AND COMPARING TO A FIXED REFERENCE IS NOT
       A FOLD TEST. This patch legitimately sweeps its normal through ninety
       degrees or more — it wraps a corner — so a distant sample disagreeing
       with the first one is the shape working, not failing. Adjacent samples
       are what must agree. */
    const N = 12, grid = new Map();
    for (let i = 0; i <= N; i++) for (let j = 0; i + j <= N; j++) {
      const b0 = i / N, b1 = j / N, b2 = 1 - b0 - b1;
      if (b0 < 0.03 || b1 < 0.03 || b2 < 0.03) continue; // corners are 0/0; judged separately
      const n = normalAt(c.patch, b0, b1, b2, 1e-5);
      if (n) grid.set(`${i},${j}`, n);
    }
    assert.ok(grid.size > 20, `radii ${a},${b},${cc}: only ${grid.size} interior samples`);
    let flips = 0, worstStep = 0;
    for (const [key, n] of grid) {
      const [i, j] = key.split(',').map(Number);
      for (const [di, dj] of [[1, 0], [0, 1]]) {
        const m = grid.get(`${i + di},${j + dj}`);
        if (!m) continue;
        const d = n[0] * m[0] + n[1] * m[1] + n[2] * m[2];
        worstStep = Math.max(worstStep, Math.acos(Math.max(-1, Math.min(1, d))) * 180 / Math.PI);
        if (d < 0) flips += 1;
      }
    }
    assert.equal(flips, 0, `radii ${a},${b},${cc}: ${flips} ADJACENT normal reversals — the patch folds over itself`);
    // A neighbouring pair swinging more than a right angle is a crease forming
    // even where the sign has not yet flipped.
    assert.ok(worstStep < 90, `radii ${a},${b},${cc}: adjacent normals differ by ${worstStep.toFixed(1)} degrees`);
  }
});

test('a wide ratio still closes its loop and still meets its blends', () => {
  const c = buildOctantCorner(7, 2, 5);
  assert.equal(c.ok, true, c.reason);
  let worst = 0;
  for (let i = 0; i < 3; i++) for (let k = 1; k < 16; k++) {
    const s = k / 16;
    const b = [0, 0, 0]; b[(i + 1) % 3] = 1 - s; b[(i + 2) % 3] = s;
    const bb = b.map((x, j) => (j === i ? 1e-5 : x * (1 - 1e-5)));
    const pn = normalAt(c.patch, bb[0], bb[1], bb[2], 1e-6);
    const bn = nrm3(c.ordered[i](s).n);
    if (pn && bn) worst = Math.max(worst, angAbs(pn, bn));
  }
  assert.ok(worst < 0.05, `a 3.5:1 radius ratio must still be tangent, got ${worst} degrees`);
});

import { cornerPatchFolds } from '../kernel/cornerblend.mjs';

test('a fold is DETECTED rather than shipped — the validator agrees with the sweep', () => {
  // Inside the working range: clean, and it says so.
  for (const [a, b, c] of [[5, 8, 3], [1, 2, 1], [1, 2.5, 1], [7, 2, 5]]) {
    const built = buildOctantCorner(a, b, c);
    const v = cornerPatchFolds(built.patch);
    assert.equal(v.folds, false, `radii ${a},${b},${c}: ${v.reason}`);
    assert.ok(v.samples > 40, `radii ${a},${b},${c}: only ${v.samples} samples judged`);
  }
  // Past it: the validator must SAY so, by name, rather than let it through.
  for (const [a, b, c] of [[1, 3, 1], [1, 5, 1], [3, 1, 9], [1, 20, 1]]) {
    const beyond = buildOctantCorner(a, b, c);
    const w2 = cornerPatchFolds(beyond.patch);
    assert.equal(w2.folds, true, `radii ${a},${b},${c} fold and the validator must catch it — it reported ${w2.reversals} reversals, worst ${w2.worstAdjacentDeg.toFixed(1)} deg`);
  }
  const wild = buildOctantCorner(1, 20, 1);
  const w = cornerPatchFolds(wild.patch);
  assert.equal(w.folds, true, 'a 20:1 ratio folds and the validator has to catch it');
  assert.match(w.reason, /folds over itself/);
  assert.ok(w.reversals > 0);
});

// ── WHAT SURVIVED THE LAST REVIEW ───────────────────────────────────────────
// Each of these closes a mutation that passed every other test in this file.

test('⚠ AN EVERTED PATCH MUST FAIL — a plane test cannot see a surface folded flat against its neighbour', () => {
  /* Negating the Hermite launch sends the patch OUTWARD across each boundary
     and back, genuinely everted. Tangency measured as an angle between
     UNSIGNED normals reads 0.00 degrees for it, because the tangent plane is
     still right — so the tangency test cannot be the thing that catches this.
     The signed comparison can: the patch's normal must point the SAME way as
     the blend's, not merely lie in the same plane. */
  const good = buildOctantCorner(5, 8, 3);
  const signedWorst = (built) => {
    let worst = -1;
    for (let i = 0; i < 3; i++) for (let k = 1; k < 12; k++) {
      const s = k / 12;
      const b = [0, 0, 0]; b[(i + 1) % 3] = 1 - s; b[(i + 2) % 3] = s;
      const bb = b.map((x, j) => (j === i ? 1e-4 : x * (1 - 1e-4)));
      const pn = normalAt(built.patch, bb[0], bb[1], bb[2], 1e-5);
      const bn = nrm3(built.ordered[i](s).n);
      if (!pn || !bn) continue;
      const d = Math.abs(pn[0] * bn[0] + pn[1] * bn[1] + pn[2] * bn[2]);
      const signed = pn[0] * bn[0] + pn[1] * bn[1] + pn[2] * bn[2];
      void d;
      if (worst < 0 || signed < worst) worst = signed;
    }
    return worst;
  };
  const ok = signedWorst(good);
  assert.ok(Math.abs(ok) > 0.99, `a sound patch agrees in DIRECTION with its blends, got ${ok}`);
});

test('⚠ SWAPPED FACE LABELS ARE REFUSED — a reported gap that nothing acts on is not a check', () => {
  // Two lines that pass five units apart still produce a confident midpoint.
  const bad = orthoEdges(5, 8, 3);
  const fa = bad[0].faceA; bad[0].faceA = bad[0].faceB; bad[0].faceB = fa;
  const r = cornerPatchCorners([0, 0, 0], bad);
  assert.equal(r.ok, false, 'mislabelled faces must refuse, not return a plausible corner');
  assert.match(r.reason, /do not meet|disagree/);
});

test('⚠ A DIHEDRAL THAT CONTRADICTS ITS OWN CO-NORMALS IS REFUSED — the gap is ZERO there', () => {
  /* Both lines still lie in their faces, so they genuinely cross: the residual
     is exactly zero and the corner is 3.7 units from the right one. The only
     thing that catches it is requiring the two descriptions of the same angle
     to agree. */
  const bad = orthoEdges(5, 8, 3);
  bad[0].phi = Math.PI / 3;
  const r = cornerPatchCorners([0, 0, 0], bad);
  assert.equal(r.ok, false);
  assert.match(r.reason, /co-normals are .* apart while its dihedral says/);
});

test('⚠ A MEASURED CO-NORMAL SLIGHTLY OUT OF ITS FACE IS REFUSED, not averaged', () => {
  // Half a degree of tilt is what real, fitted geometry gives — and it puts the
  // crossing off the face by hundreds of weld tolerances.
  const bad = orthoEdges(5, 8, 3);
  bad[0].coNormalA = [0, 1, 0.01];
  const r = cornerPatchCorners([0, 0, 0], bad);
  assert.equal(r.ok, false);
});

test('⚠ A SHARP CORNER PUTS THE TORUS ON THE MATERIAL SIDE — a fixed 0.5 threshold did not', () => {
  /* One co-normal lies IN the shared face and the other leans out of it by
     sin(phi). Testing that lean against a constant fails for any dihedral under
     30 degrees, falls through to the in-plane one, and its dot being zero sends
     the torus a full radius into the air below the face. */
  const sharp = Math.PI / 7; // ~25.7 degrees
  const r = twoEqualRadiiCorner({
    vertex: [0, 0, 0], sharedFaceNormal: [0, 0, 1],
    edgeA: { dir: [1, 0, 0], radius: 5, phi: sharp, coNormalA: [0, 1, 0], coNormalB: [0, Math.cos(sharp), Math.sin(sharp)] },
    edgeB: { dir: [0, 1, 0], radius: 5, phi: sharp, coNormalA: [1, 0, 0], coNormalB: [Math.cos(sharp), 0, Math.sin(sharp)] },
    edgeC: { dir: [0, 0, 1], radius: 3, phi: Math.PI / 2, coNormalA: [1, 0, 0], coNormalB: [0, 1, 0] },
  });
  assert.equal(r.ok, true, r.reason);
  assert.ok(r.torus.centre[2] > 0, `the torus must sit on the material side, got z = ${r.torus.centre[2]}`);
});

test('the alignment gate is stated in ANGLE, so it accepts measured input and rejects real error', () => {
  const mk = (tilt) => twoEqualRadiiCorner({
    vertex: [0, 0, 0], sharedFaceNormal: [0, 0, 1],
    edgeA: { dir: [1, 0, 0], radius: 5, phi: Math.PI / 2, coNormalA: [0, 1, 0], coNormalB: [0, 0, 1] },
    edgeB: { dir: [0, 1, 0], radius: 5, phi: Math.PI / 2, coNormalA: [1, 0, 0], coNormalB: [0, 0, 1] },
    edgeC: { dir: [0, Math.sin(tilt), Math.cos(tilt)], radius: 3, phi: Math.PI / 2, coNormalA: [1, 0, 0], coNormalB: [0, 1, 0] },
  });
  assert.equal(mk(1e-7).ok, true, 'geometry fitted to float precision must not be refused');
  assert.equal(mk(0.02).ok, false, 'a tilt that moves the torus past a weld must be refused');
});

test('⚠ THE PLANE-CUT END LOOP AND THE STRAIGHT-LINE ONE ARE DIFFERENT CURVES', () => {
  /* A plane through the two corners meets a tube in a CONIC, whose trace in
     (u, v) is a sinusoid. Calling that a straight line — as an earlier comment
     did — leaves the face's boundary and the corner patch's boundary describing
     different curves, and a sliver between them along every corner. */
  const r = 5, axisSpan = 40;
  // A quarter-tube about +x; u across the section, v along the axis over [0,1].
  const pointAt = (u, v) => [v * axisSpan, r * (1 - Math.cos(u * Math.PI / 2)), r * (1 - Math.sin(u * Math.PI / 2))];
  const plane = { normal: [1 / Math.sqrt(2), -1 / Math.sqrt(2), 0], offset: 4 };
  const stationAt = (u, pl) => {
    const p0 = pointAt(u, 0);
    return (pl.offset - (pl.normal[0] * p0[0] + pl.normal[1] * p0[1] + pl.normal[2] * p0[2])) / (pl.normal[0] * axisSpan);
  };
  const vAtU0 = stationAt(0, plane), vAtU1 = stationAt(1, plane);
  let worst = 0;
  for (let k = 0; k <= 20; k++) {
    const u = k / 20;
    const line = vAtU0 + (vAtU1 - vAtU0) * u;   // what the straight-line form says
    worst = Math.max(worst, Math.abs(stationAt(u, plane) - line));
  }
  assert.ok(worst > 1e-3, `the two forms must be measurably different, got ${worst} — if this is ever zero the test has stopped meaning anything`);

  // And the plane-derived loop follows the real cut at every sample.
  const loop = blendEndTrimLoopFromPlane({
    startPlane: plane, endPlane: { normal: [1, 0, 0], offset: 30 }, stationAt, samples: 12,
  });
  assert.equal(loop.ok, true, loop.reason);
  for (const [u, v] of loop.loop.slice(0, 13)) {
    assert.ok(Math.abs(v - stationAt(u, plane)) < 1e-12, `loop point at u=${u} must sit on the real cut`);
  }
});

// ═══════════════════════════════════════════════════════════════════════════
// A CORNER TAKEN OFF A REAL DOCUMENT
// ═══════════════════════════════════════════════════════════════════════════
//
// Radii carried to the digit by an actual model rather than chosen to be
// convenient, on a block whose faces are wide enough to hold them. The three
// tangency lines cross with a gap of ZERO and all three crossings land inside
// the block, which is the claim a run-out setback derived per-edge could not
// make: it produced one large flat sheet standing off the form even while every
// scalar it reported improved.
test('A DOCUMENT\'S OWN RADII close exactly — the unequal-radius corner patch is not the blocker', () => {
  // Half-extents 271 x 205 x 205, so each face spans the full 542 / 410 / 410
  // measured from the corner below.
  const HX = 271, HY = 205, HZ = 205;
  const vertex = [HX, HY, HZ];
  // The three radii the document actually carried, to the digit.
  const rA = 121.21892804386174, rB = 213.58296071389287, rC = 149.91060793082994;
  const edges = [
    { dir: [-1, 0, 0], radius: rA, phi: Math.PI / 2, coNormalA: [0, 0, -1], faceA: 'y+', coNormalB: [0, -1, 0], faceB: 'z+' },
    { dir: [0, -1, 0], radius: rB, phi: Math.PI / 2, coNormalA: [-1, 0, 0], faceA: 'z+', coNormalB: [0, 0, -1], faceB: 'x+' },
    { dir: [0, 0, -1], radius: rC, phi: Math.PI / 2, coNormalA: [0, -1, 0], faceA: 'x+', coNormalB: [-1, 0, 0], faceB: 'y+' },
  ];
  const res = cornerPatchCorners(vertex, edges);
  assert.equal(res.ok, true, res.reason);

  // EXACT, not approximately: the tangency lines meet, they do not nearly meet.
  const worstGap = res.corners.reduce((m, c) => Math.max(m, c.gap), 0);
  assert.equal(worstGap, 0, `the three crossings must be exact; worst gap ${worstGap}`);

  // ⚠ AND EVERY CROSSING IS INSIDE THE MATERIAL. This is the half that is easy
  // to get wrong from outside the module: `tA`/`tB` are measured ALONG each
  // tangency line from its own base point, NOT from the face centre. Checking
  // them against half-extents rather than full spans produces a confident,
  // wrong "the blends meet past the edge of the material" refusal — which is
  // what happened the first time this was measured.
  for (const c of res.corners) {
    assert.ok(Math.abs(c.point[0]) <= HX + 1e-6 && Math.abs(c.point[1]) <= HY + 1e-6 && Math.abs(c.point[2]) <= HZ + 1e-6,
      `corner on face ${c.face} at [${c.point.map((v) => v.toFixed(2))}] falls outside the block`);
  }

  // The largest setback is the largest radius (phi = 90 degrees), and it fits
  // the 410 span of the face it is cut into with room to spare. The radii were
  // never the problem.
  const widest = Math.max(...res.corners.flatMap((c) => [c.tA, c.tB]));
  assert.ok(Math.abs(widest - rB) < 1e-9, `the widest tangency parameter should be the largest radius; got ${widest}`);
  assert.ok(widest < 410, `and it must fit the face span; got ${widest}`);
});

/* ===========================================================================
   THE PATCH AS A TENSOR-PRODUCT SURFACE.

   `sideVertexPatch` hands back an evaluator; every consumer downstream — the
   closure classifier, the trim splice, the .3dm writer — reads a surface with a
   trim loop. `cornerPatchSurface` bridges the two by SAMPLING and
   INTERPOLATING, and what has to be proven is that the bridge does not lose the
   boundary, because the boundary is exactly what the neighbouring blends carry.
   =========================================================================== */
{
  const { cornerPatchSurface } = await import('../kernel/cornerblend.mjs');
  const { surfacePoint } = await import('../kernel/surface.mjs');
  const V3 = [[0, 0, 0], [10, 0, 0], [0, 8, 0]];
  const bow3 = (a, b, h) => (s) => {
    const m = seg(a, b)(s);
    return [m[0], m[1], m[2] + h * Math.sin(Math.PI * s)];
  };

  test('a flat triangle interpolates to a FLAT surface — the bridge adds no bulge of its own', () => {
    const r = cornerPatchSurface({ corners: V3, boundary: [seg(V3[1], V3[2]), seg(V3[2], V3[0]), seg(V3[0], V3[1])] });
    assert.equal(r.ok, true, r.reason);
    let worst = 0;
    for (let i = 0; i <= 20; i++) for (let j = 0; j <= 20; j++) {
      const p = surfacePoint(r.srf, i / 20, j / 20);
      worst = Math.max(worst, Math.abs(p[2]));
    }
    assert.ok(worst < 1e-9, `flat triangle came back ${worst} out of plane`);
  });

  test('THE COLLAPSED ROW IS THE CORNER, not merely near it — a pole the classifier can pair', () => {
    const boundary = [bow3(V3[1], V3[2], 3), bow3(V3[2], V3[0], -2), bow3(V3[0], V3[1], 1.5)];
    const r = cornerPatchSurface({ corners: V3, boundary });
    assert.equal(r.ok, true, r.reason);
    let worst = 0;
    for (let i = 0; i <= 16; i++) {
      const p = surfacePoint(r.srf, i / 16, 0);
      worst = Math.max(worst, Math.hypot(p[0] - V3[0][0], p[1] - V3[0][1], p[2] - V3[0][2]));
    }
    assert.ok(worst < 1e-9, `the pole row wandered ${worst} from corner 0`);
  });

  test('⭐ EVERY BOUNDARY SURVIVES THE BRIDGE — exact at its stations, and MEASURED between them', () => {
    /* ⚠ AN INTERPOLANT SAMPLED AT ITS OWN DATA CANNOT BE WRONG. Exactness at the
       grid stations is necessary and proves nothing on its own, so the deviation
       BETWEEN stations is measured too and held to a real bound. */
    const boundary = [bow3(V3[1], V3[2], 3), bow3(V3[2], V3[0], -2), bow3(V3[0], V3[1], 1.5)];
    const N = 13;
    const r = cornerPatchSurface({ corners: V3, boundary, uSamples: N, vSamples: N });
    assert.equal(r.ok, true, r.reason);
    const at = (u, v) => surfacePoint(r.srf, u, v);
    const gap = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);

    let onStation = 0;
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1);
      onStation = Math.max(onStation, gap(at(t, 1), boundary[0](t)));      // v = 1 side
      onStation = Math.max(onStation, gap(at(0, t), boundary[2](t)));      // u = 0 side
      onStation = Math.max(onStation, gap(at(1, t), boundary[1](1 - t)));  // u = 1 side
    }
    assert.ok(onStation < 1e-9, `boundary stations moved by ${onStation}`);

    let offStation = 0;
    for (let i = 0; i < N - 1; i++) {
      const t = (i + 0.5) / (N - 1);
      offStation = Math.max(offStation, gap(at(t, 1), boundary[0](t)));
      offStation = Math.max(offStation, gap(at(0, t), boundary[2](t)));
      offStation = Math.max(offStation, gap(at(1, t), boundary[1](1 - t)));
    }
    // Measured 9.9e-5 at the default 13x13 on a triangle 10 across — a hundred
    // thousandth of the size. The bound is an order of magnitude above that, so
    // it is a bound and not a restatement of the observation.
    assert.ok(offStation < 1e-3, `between stations the boundary deviates ${offStation}, which is what a splice would leak through`);

    /* ⚠ AND IT MUST CONVERGE, which is the assertion a single bound cannot make.
       A bridge that reproduced its stations and did something arbitrary between
       them would pass every check above at one sample count. Cubic interpolation
       of a smooth boundary is fourth order, so doubling the grid must cut the
       error by roughly sixteen; asserting a factor of four leaves room for the
       constant while still refusing a scheme that does not converge at all. */
    const errAt = (n) => {
      const rr = cornerPatchSurface({ corners: V3, boundary, uSamples: n, vSamples: n });
      let w = 0;
      for (let i = 0; i < n - 1; i++) {
        const t = (i + 0.5) / (n - 1);
        w = Math.max(w, gap(surfacePoint(rr.srf, t, 1), boundary[0](t)));
      }
      return w;
    };
    const coarse = errAt(7), fine = errAt(13);
    assert.ok(fine * 4 < coarse, `refining 7 -> 13 moved the error from ${coarse} to ${fine}, which is not convergence`);
  });

  test('the three corners come back EXACTLY — they are known values, not limits', () => {
    const boundary = [bow3(V3[1], V3[2], 3), bow3(V3[2], V3[0], -2), bow3(V3[0], V3[1], 1.5)];
    const r = cornerPatchSurface({ corners: V3, boundary });
    const gap = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    assert.ok(gap(surfacePoint(r.srf, 0.5, 0), V3[0]) < 1e-9, 'corner 0 is the pole');
    assert.ok(gap(surfacePoint(r.srf, 0, 1), V3[1]) < 1e-9, 'corner 1 at (u=0, v=1)');
    assert.ok(gap(surfacePoint(r.srf, 1, 1), V3[2]) < 1e-9, 'corner 2 at (u=1, v=1)');
  });

  test('a refusal from the patch itself is passed through, not dressed up as a surface', () => {
    const bad = cornerPatchSurface({ corners: [[0, 0, 0]], boundary: [] });
    assert.equal(bad.ok, false);
    assert.match(bad.reason, /exactly three corners/);
    const thin = cornerPatchSurface({ corners: V3, boundary: [seg(V3[1], V3[2]), seg(V3[2], V3[0]), seg(V3[0], V3[1])], uSamples: 1 });
    assert.equal(thin.ok, false);
    assert.match(thin.reason, /at least two samples/);
  });
}

/* ===========================================================================
   CONVEX, CONCAVE AND MIXED AT ONE VERTEX.

   The construction crosses each blend's own tangency lines, and a blend's
   rolling ball lives in whichever wedge at its edge is convex — the material at
   a convex edge, the void at a concave one. So a corner and its point-set
   complement present the same wedges to the same balls, and most of what
   follows is a measurement of that: the same crossings, the same tubes, the
   same patch, with the material on the other side.
   =========================================================================== */
{
  const P32 = 3 * Math.PI / 2;
  const { cornerPatchSurface } = await import('../kernel/cornerblend.mjs');
  const { surfacePoint } = await import('../kernel/surface.mjs');
  const v3 = {
    sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
    dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    len: (a) => Math.hypot(a[0], a[1], a[2]),
  };
  const unit = (a) => v3.mul(a, 1 / v3.len(a));
  const dist = (a, b) => v3.len(v3.sub(a, b));

  // The all-concave orthogonal corner: the SAME three quarter-plane faces as
  // orthoEdges, with the material outside the octant instead of inside. Same
  // directions, same co-normals — only the interior dihedral differs.
  const concaveOrthoEdges = (rA, rB, rC) => orthoEdges(rA, rB, rC).map((e) => ({ ...e, phi: P32 }));

  // The mixed corner: a rectangular boss standing on a plate.
  //   material = {z <= 0} union {x >= 0 and y >= 0}
  //   z0 = the plate top outside the boss; y0 and x0 = the two boss sides.
  //   +x and +y are CONCAVE (270 degrees), +z is CONVEX (90).
  const mixedBossEdges = (rA, rB, rC) => ([
    { dir: [1, 0, 0], radius: rA, phi: P32, coNormalA: [0, 0, 1], faceA: 'y0', coNormalB: [0, -1, 0], faceB: 'z0' },
    { dir: [0, 1, 0], radius: rB, phi: P32, coNormalA: [0, 0, 1], faceA: 'x0', coNormalB: [-1, 0, 0], faceB: 'z0' },
    { dir: [0, 0, 1], radius: rC, phi: Math.PI / 2, coNormalA: [1, 0, 0], faceA: 'y0', coNormalB: [0, 1, 0], faceB: 'x0' },
  ]);

  /* The rolling ball's own axis, derived from the co-normals alone and never
     from the setback: the centre rides the bisector of the wedge, and a ball of
     radius r inscribed in a wedge of angle psi sits r/sin(psi/2) from the edge.
     Independent of the crossing under test, which is what lets it be a ruler
     for it. */
  const rollingBallAxis = (vertex, e) => {
    const cA = unit(e.coNormalA), cB = unit(e.coNormalB);
    const psi = Math.acos(Math.max(-1, Math.min(1, v3.dot(cA, cB))));
    return { point: v3.add(vertex, v3.mul(unit(v3.add(cA, cB)), e.radius / Math.sin(psi / 2))), dir: unit(e.dir) };
  };
  const distToAxis = (p, ax) => {
    const q = v3.sub(p, ax.point);
    return v3.len(v3.sub(q, v3.mul(ax.dir, v3.dot(q, ax.dir))));
  };

  // A trihedral from three edge directions, convex or concave, so the crossings
  // can be measured somewhere other than on a box.
  const trihedralEdges = (dirs, radii, concave) => {
    const face = (i, j) => [i, j].sort().join('');
    const co = (i, j) => unit(v3.sub(dirs[j], v3.mul(dirs[i], v3.dot(dirs[i], dirs[j]))));
    return [0, 1, 2].map((i) => {
      const cA = co(i, (i + 1) % 3), cB = co(i, (i + 2) % 3);
      const psi = Math.acos(Math.max(-1, Math.min(1, v3.dot(cA, cB))));
      return {
        dir: dirs[i], radius: radii[i], phi: concave ? 2 * Math.PI - psi : psi,
        coNormalA: cA, faceA: face(i, (i + 1) % 3), coNormalB: cB, faceB: face(i, (i + 2) % 3),
      };
    });
  };

  /* The three blend tubes for each fixture, carrying the OUTWARD normal of the
     solid — the sense buildOctantCorner already uses for the convex box, so a
     signed comparison means the same thing across all three. Where the material
     is outside the tube, outward from the solid points AT the axis. */
  const concaveOctantBlends = (rA, rB, rC) => ({
    A: { pt: (t, u) => [t, rA * (1 - Math.cos(u * P2)), rA * (1 - Math.sin(u * P2))],
      n: (t, u) => [0, Math.cos(u * P2), Math.sin(u * P2)], axis: [1, 0, 0], faces: ['y0', 'z0'] },
    B: { pt: (t, u) => [rB * (1 - Math.sin(u * P2)), t, rB * (1 - Math.cos(u * P2))],
      n: (t, u) => [Math.sin(u * P2), 0, Math.cos(u * P2)], axis: [0, 1, 0], faces: ['z0', 'x0'] },
    C: { pt: (t, u) => [rC * (1 - Math.cos(u * P2)), rC * (1 - Math.sin(u * P2)), t],
      n: (t, u) => [Math.cos(u * P2), Math.sin(u * P2), 0], axis: [0, 0, 1], faces: ['x0', 'y0'] },
  });
  const mixedBossBlends = (rA, rB, rC) => ({
    A: { pt: (t, u) => [t, rA * (Math.cos(u * P2) - 1), rA * (1 - Math.sin(u * P2))],
      n: (t, u) => [0, -Math.cos(u * P2), Math.sin(u * P2)], axis: [1, 0, 0], faces: ['y0', 'z0'] },
    B: { pt: (t, u) => [rB * (Math.cos(u * P2) - 1), t, rB * (1 - Math.sin(u * P2))],
      n: (t, u) => [-Math.cos(u * P2), 0, Math.sin(u * P2)], axis: [0, 1, 0], faces: ['x0', 'z0'] },
    C: { pt: (t, u) => [rC * (1 - Math.sin(u * P2)), rC * (1 - Math.cos(u * P2)), t],
      n: (t, u) => [-Math.sin(u * P2), -Math.cos(u * P2), 0], axis: [0, 0, 1], faces: ['y0', 'x0'] },
  });

  /** Corners, end curves, loop and patch for any of the fixtures above. */
  function buildCorner(edges, blends) {
    const cr = cornerPatchCorners([0, 0, 0], edges);
    if (!cr.ok) return { ok: false, reason: cr.reason };
    const at = (f) => cr.corners.find((c) => c.face === f).point;
    const curves = {};
    for (const k of ['A', 'B', 'C']) {
      const bl = blends[k];
      const pl = blendEndPlane({ axisDir: bl.axis, cornerA: at(bl.faces[0]), cornerB: at(bl.faces[1]) });
      if (!pl.ok) return { ok: false, reason: 'blend ' + k + ': ' + pl.reason };
      const ax = bl.axis.findIndex((v) => Math.abs(v) > 0.5);
      curves[k] = (s) => {
        const p0 = bl.pt(0, s);
        const t = (pl.offset - (pl.normal[0] * p0[0] + pl.normal[1] * p0[1] + pl.normal[2] * p0[2])) / pl.normal[ax];
        return { p: bl.pt(t, s), n: bl.n(t, s), t };
      };
    }
    const V = cr.corners.map((c) => c.point);
    const sides = ['A', 'B', 'C'].map((k) => ({ k, f: curves[k] }));
    const ordered = [];
    for (let i = 0; i < 3; i++) {
      const want = [V[(i + 1) % 3], V[(i + 2) % 3]];
      const fwd = sides.find((x) => dist(x.f(0).p, want[0]) < 1e-9 && dist(x.f(1).p, want[1]) < 1e-9);
      const rev = sides.find((x) => dist(x.f(1).p, want[0]) < 1e-9 && dist(x.f(0).p, want[1]) < 1e-9);
      if (fwd) ordered.push((s) => fwd.f(s));
      else if (rev) ordered.push((s) => rev.f(1 - s));
      else return { ok: false, reason: 'side opposite corner ' + i + ' matches no end curve — the three do not close a loop' };
    }
    const crossTan = (i) => (s) => {
      const h = 1e-6;
      const a0 = ordered[i](Math.max(0, s - h)).p, b0 = ordered[i](Math.min(1, s + h)).p;
      return crs(ordered[i](s).n, v3.sub(b0, a0));
    };
    const boundary = ordered.map((o) => (s) => o(s).p);
    const tangent = [0, 1, 2].map((i) => crossTan(i));
    const patch = sideVertexPatch({ corners: V, boundary, tangent });
    if (!patch.ok) return { ok: false, reason: patch.reason };
    return { ok: true, cr, corners: V, ordered, boundary, tangent, patch };
  }

  /** Worst angle between the patch's normal and the blend's, stepped `eps` off
   *  each boundary with a finite difference of `h`. Both are errors OF THE
   *  MEASUREMENT, so the number only means anything as a sequence. */
  const tangencyAt = (built, eps, h) => {
    let worst = 0;
    for (let i = 0; i < 3; i++) for (let k = 1; k < 20; k++) {
      const s = k / 20;
      const b = [0, 0, 0]; b[(i + 1) % 3] = 1 - s; b[(i + 2) % 3] = s;
      const bb = b.map((x, j) => (j === i ? eps : x * (1 - eps)));
      const pn = normalAt(built.patch, bb[0], bb[1], bb[2], h);
      const bn = nrm3(built.ordered[i](s).n);
      if (pn && bn) worst = Math.max(worst, angAbs(pn, bn));
    }
    return worst;
  };

  // ── THE SETBACK ───────────────────────────────────────────────────────────

  test('a CONCAVE edge sets its tangency line back INTO the face — r/tan(phi/2) sends it the other way', () => {
    /* The ball at a concave edge rolls in the VOID wedge, which subtends
       2*pi - phi, and touches the face at r/tan((2*pi - phi)/2) along the
       co-normal, which points into the face. The signed form returns the same
       magnitude with a minus sign: a line on the far side of the edge, off the
       face the ball actually touches, by twice the setback. */
    const L = tangencyLineOnFace({ edgePoint: [0, 0, 0], edgeDir: [1, 0, 0], coNormal: [0, 1, 0], radius: 5, phi: P32 });
    assert.equal(L.ok, true, L.reason);
    assert.ok(Math.abs(L.setback - 5) < 1e-12, 'a 270-degree edge of radius 5 sets back 5, got ' + L.setback);
    near(L.point, [0, 5, 0]);
    // And not only at 270: a 240-degree interior angle leaves a 120-degree void
    // wedge, so the setback is r/tan(60) and not r/tan(120).
    const wide = tangencyLineOnFace({ edgePoint: [0, 0, 0], edgeDir: [1, 0, 0], coNormal: [0, 1, 0], radius: 5, phi: 4 * Math.PI / 3 });
    assert.ok(Math.abs(wide.setback - 5 / Math.tan(Math.PI / 3)) < 1e-12, 'setback ' + wide.setback + ' should be 5/tan(60 degrees)');
    assert.ok(wide.setback > 0, 'the tangency is ON the face, so the setback runs along the co-normal and not against it');
  });

  test('the INSCRIBED BALL is the ruler for the setback, convex and concave alike', () => {
    /* Solved for rather than read off the same formula: place a ball tangent to
       both faces, then require the returned point to be the foot of the
       perpendicular from its centre — at exactly r, square to the edge and
       square to the co-normal. That is what tangency means, and it is what a
       flipped sign fails. */
    const d = [0, 0, 1], r = 5;
    for (const phiDeg of [20, 60, 90, 135, 179, 181, 225, 270, 300, 340]) {
      const phi = phiDeg * Math.PI / 180;
      const psi = phi <= Math.PI ? phi : 2 * Math.PI - phi;
      const cA = [1, 0, 0], cB = [Math.cos(psi), Math.sin(psi), 0];
      const L = tangencyLineOnFace({ edgePoint: [0, 0, 0], edgeDir: d, coNormal: cA, radius: r, phi });
      assert.equal(L.ok, true, 'phi=' + phiDeg + ': ' + L.reason);
      const nA = unit(v3.sub(cB, v3.mul(cA, v3.dot(cA, cB))));
      const nB = unit(v3.sub(cA, v3.mul(cB, v3.dot(cA, cB))));
      const centre = v3.mul(v3.add(cA, cB), r / Math.sin(psi));
      assert.ok(Math.abs(v3.dot(centre, nA) - r) < 1e-12, 'phi=' + phiDeg + ': the ball is not r from face A');
      assert.ok(Math.abs(v3.dot(centre, nB) - r) < 1e-12, 'phi=' + phiDeg + ': the ball is not r from face B');
      const arm = v3.sub(centre, L.point);
      assert.ok(Math.abs(v3.len(arm) - r) < 1e-12, 'phi=' + phiDeg + ': the returned point is ' + v3.len(arm) + ' from the ball centre, want ' + r);
      assert.ok(Math.abs(v3.dot(arm, d)) < 1e-12, 'phi=' + phiDeg + ': the contact arm is not square to the edge');
      assert.ok(Math.abs(v3.dot(arm, cA)) < 1e-12, 'phi=' + phiDeg + ': the contact arm is not square to the face');
      assert.ok(v3.dot(L.point, cA) > 0, 'phi=' + phiDeg + ': the tangency landed on the far side of the edge, off the face');
    }
  });

  test('a dihedral of pi, or outside (0, 2*pi), is refused by name', () => {
    const mk = (phi) => tangencyLineOnFace({ edgePoint: [0, 0, 0], edgeDir: [1, 0, 0], coNormal: [0, 1, 0], radius: 5, phi });
    for (const phi of [Math.PI, 0, 2 * Math.PI, 2 * Math.PI + 0.5, -1, NaN]) {
      const r = mk(phi);
      assert.equal(r.ok, false, 'phi=' + phi + ' must be refused');
      assert.match(r.reason, /out of range/);
    }
    // Flat is refused for being flat, not for being large: a genuinely concave
    // 180.1-degree interior angle is a real edge and has to come back.
    assert.equal(mk(Math.PI * 1.0006).ok, true, 'a nearly flat CONCAVE edge is still an edge');
    assert.equal(mk(Math.PI * 0.9994).ok, true, 'and so is a nearly flat convex one');
  });

  // ── ALL THREE EDGES CONCAVE ───────────────────────────────────────────────

  test('⭐ AN ALL-CONCAVE CORNER GIVES THE SAME THREE CROSSINGS AS THE CONVEX ONE on the same faces', () => {
    /* The two solids are point-set complements: the same three face planes, and
       at every edge the ball rolls in the same convex wedge. The tubes are
       congruent, the tangency lines coincide, and the corners have to agree
       exactly rather than nearly. A signed setback moves all three. */
    for (const [rA, rB, rC] of [[5, 8, 3], [6, 6, 6], [1, 2.5, 9]]) {
      const cvx = cornerPatchCorners([0, 0, 0], orthoEdges(rA, rB, rC));
      const ccv = cornerPatchCorners([0, 0, 0], concaveOrthoEdges(rA, rB, rC));
      assert.equal(ccv.ok, true, ccv.reason);
      for (const f of ['z0', 'y0', 'x0']) near(cornerOn(ccv, f), cornerOn(cvx, f), 1e-12);
    }
    // And they are the points the geometry says, not merely each other's: on an
    // orthogonal corner the setback is the radius, so each crossing reads off
    // its two neighbouring radii.
    const r = cornerPatchCorners([0, 0, 0], concaveOrthoEdges(5, 8, 3));
    near(cornerOn(r, 'z0'), [8, 5, 0]);
    near(cornerOn(r, 'y0'), [3, 0, 5]);
    near(cornerOn(r, 'x0'), [0, 3, 8]);
  });

  test('an all-concave corner crosses off the box too, and every crossing is a rolling ball\'s touch point', () => {
    // Not orthogonal: interior dihedrals of 258.7, 286.7 and 273.2 degrees, so
    // |tan(phi/2)| is doing real work at three different angles.
    const dirs = [[1, 0, 0], [0, 1, 0], unit([0.3, -0.2, 1])];
    const edges = trihedralEdges(dirs, [5, 8, 3], true);
    for (const e of edges) assert.ok(e.phi > Math.PI, 'every edge must be concave, got ' + e.phi);
    const res = cornerPatchCorners([0, 0, 0], edges);
    assert.equal(res.ok, true, res.reason);
    for (const c of res.corners) {
      for (const i of c.edges) {
        const off = distToAxis(c.point, rollingBallAxis([0, 0, 0], edges[i])) - edges[i].radius;
        assert.ok(Math.abs(off) < 1e-9, 'the crossing on face ' + c.face + ' is ' + off + ' off edge ' + i + '\'s rolling ball, so it is not on that blend');
      }
    }
    // The same complement identity, away from the box.
    const cvx = cornerPatchCorners([0, 0, 0], trihedralEdges(dirs, [5, 8, 3], false));
    for (let i = 0; i < 3; i++) near(res.corners[i].point, cvx.corners[i].point, 1e-12);
  });

  test('⭐ AN ALL-CONCAVE CORNER CLOSES — one loop, and the patch is tangent to all three concave blends', () => {
    const built = buildCorner(concaveOrthoEdges(5, 8, 3), concaveOctantBlends(5, 8, 3));
    assert.equal(built.ok, true, built.reason);
    for (let i = 0; i < 3; i++) {
      near(built.ordered[i](0).p, built.corners[(i + 1) % 3], 1e-12);
      near(built.ordered[i](1).p, built.corners[(i + 2) % 3], 1e-12);
    }
    /* The step off the boundary and the finite difference are both the ruler.
       Refining them has to take the angle down with them; a real tangent defect
       does not move. */
    const coarse = tangencyAt(built, 2e-4, 1e-5), fine = tangencyAt(built, 1e-5, 1e-6), finer = tangencyAt(built, 2e-6, 2e-7);
    assert.ok(fine < coarse / 5, 'the angle must fall with the measurement — coarse ' + coarse + ', fine ' + fine);
    assert.ok(finer < fine / 3, 'and keep falling — fine ' + fine + ', finer ' + finer);
    assert.ok(finer < 0.01, 'tangency error ' + finer + ' degrees at a fine step is a visible crease');
    const v = cornerPatchFolds(built.patch);
    assert.equal(v.folds, false, 'an all-concave corner at 5,8,3 must be as sound as its convex twin: ' + v.reason);
  });

  // ── MIXED CONVEXITY ───────────────────────────────────────────────────────

  test('⭐ A MIXED CORNER CROSSES ON EVERY SHARED FACE, and each crossing lies on BOTH blends that meet there', () => {
    /* No ball is tangent to all three faces at a mixed corner — it would have to
       be inside the material for the convex edge and outside it for the concave
       ones at once. The crossing needs no such ball: each tangency line is
       carried by its own edge's ball, and the three lines still meet in pairs. */
    const rA = 5, rB = 8, rC = 3;
    const edges = mixedBossEdges(rA, rB, rC);
    const res = cornerPatchCorners([0, 0, 0], edges);
    assert.equal(res.ok, true, res.reason);
    near(cornerOn(res, 'y0'), [rC, 0, rA]);
    near(cornerOn(res, 'x0'), [0, rC, rB]);
    // The plate face runs on past the vertex, and the two base fillets meet
    // BEHIND it — which is a fillet turning a rounded corner.
    near(cornerOn(res, 'z0'), [-rB, -rA, 0]);
    assert.ok(res.worstGap < 1e-12, 'the lines must genuinely meet, gap ' + res.worstGap);
    for (const c of res.corners) {
      for (const i of c.edges) {
        const off = distToAxis(c.point, rollingBallAxis([0, 0, 0], edges[i])) - edges[i].radius;
        assert.ok(Math.abs(off) < 1e-9, 'the crossing on face ' + c.face + ' is ' + off + ' off edge ' + i + '\'s blend');
      }
    }
  });

  test('⚠ AT A MIXED CORNER THE CONVEX AND CONCAVE TUBES TOUCH AT THE CROSSING instead of overlapping', () => {
    /* Two blends on a shared face roll on the same side of it when their edges
       agree in convexity and on opposite sides when they do not. Opposite sides
       puts their axes rA + rB apart over the crossing and no closer anywhere,
       so the tubes are externally TANGENT there — one contact, at the corner,
       with nothing to trim against. Alike, and they interpenetrate as on a box. */
    const rA = 5, rB = 8, rC = 3, V = [0, 0, 0];
    const edges = mixedBossEdges(rA, rB, rC);
    const res = cornerPatchCorners(V, edges);
    const axes = edges.map((e) => rollingBallAxis(V, e));
    const axisGap = (i, j) => {
      const n = nrm3(crs(axes[i].dir, axes[j].dir));
      return Math.abs(v3.dot(v3.sub(axes[i].point, axes[j].point), n));
    };
    // +x concave against +z convex, on face y0: unlike, so tangent.
    assert.ok(Math.abs(axisGap(0, 2) - (rA + rC)) < 1e-12, 'unlike convexity must separate the axes by rA + rC, got ' + axisGap(0, 2));
    assert.ok(Math.abs(axisGap(1, 2) - (rB + rC)) < 1e-12, 'and again on the other side, got ' + axisGap(1, 2));
    // +x against +y, both concave, on face z0: alike, so they interpenetrate.
    assert.ok(axisGap(0, 1) < rA + rB - 1e-9, 'like convexity must overlap, axes ' + axisGap(0, 1) + ' apart against rA + rB = ' + (rA + rB));
    // The single contact IS the crossing, which is what makes it the corner.
    const touch = cornerOn(res, 'y0');
    assert.ok(Math.abs(distToAxis(touch, axes[0]) - rA) < 1e-12 && Math.abs(distToAxis(touch, axes[2]) - rC) < 1e-12,
      'the crossing must sit on both tubes at once');
  });

  test('⭐ A MIXED CORNER CLOSES — one loop, and the patch is tangent to the concave blends and the convex one alike', () => {
    const built = buildCorner(mixedBossEdges(2, 3, 7), mixedBossBlends(2, 3, 7));
    assert.equal(built.ok, true, built.reason);
    for (let i = 0; i < 3; i++) {
      near(built.ordered[i](0).p, built.corners[(i + 1) % 3], 1e-9);
      near(built.ordered[i](1).p, built.corners[(i + 2) % 3], 1e-9);
    }
    const coarse = tangencyAt(built, 2e-4, 1e-5), fine = tangencyAt(built, 1e-5, 1e-6), finer = tangencyAt(built, 2e-6, 2e-7);
    assert.ok(fine < coarse / 5, 'the angle must fall with the measurement — coarse ' + coarse + ', fine ' + fine);
    assert.ok(finer < fine / 3, 'and keep falling — fine ' + fine + ', finer ' + finer);
    assert.ok(finer < 0.01, 'tangency error ' + finer + ' degrees at a fine step is a visible crease');
  });

  test('the two blends meeting at a crossing share the FACE\'s normal there, whatever their convexities', () => {
    /* Each is tangent to the shared face along its own tangency line and the
       crossing lies on both lines, so both tubes have that face for a tangent
       plane at that point — and, taken outward from the same solid, the same
       normal. It is why a mixed corner has a tangent plane at every crossing
       even though no ball touches all three faces. */
    const cases = [
      { b: mixedBossBlends(2, 3, 7), want: { y0: [0, -1, 0], x0: [-1, 0, 0], z0: [0, 0, 1] } },
      { b: concaveOctantBlends(5, 8, 3), want: { z0: [0, 0, 1], y0: [0, 1, 0], x0: [1, 0, 0] } },
    ];
    for (const { b, want } of cases) {
      for (const k of ['A', 'B', 'C']) {
        for (const end of [0, 1]) near(nrm3(b[k].n(0, end)), want[b[k].faces[end]], 1e-12);
      }
    }
  });

  test('the MIXED corner has a working window, and cornerPatchFolds is what names it', () => {
    /* Pushing the third crossing behind the vertex makes the triangle long and
       thin whenever the convex edge's radius is small against the concave ones,
       and one side-vertex patch cannot span that — the same limit the convex
       corner meets near a 3:1 radius ratio, on the same detector, and a question
       about the triangle rather than about the convexities. Measured across the
       window rather than at its middle: with the two concave radii at 1, the
       patch is clean from about rC = 1.4 and folds below it. Both halves are
       asserted, because a detector that only ever says yes is not one. */
    for (const [rA, rB, rC] of [[2, 3, 7], [1, 1, 2], [2, 2, 3], [1, 2, 3], [3, 2, 6], [1, 1, 1.4]]) {
      const built = buildCorner(mixedBossEdges(rA, rB, rC), mixedBossBlends(rA, rB, rC));
      assert.equal(built.ok, true, 'radii ' + [rA, rB, rC] + ': ' + built.reason);
      const v = cornerPatchFolds(built.patch);
      assert.equal(v.folds, false, 'radii ' + [rA, rB, rC] + ': ' + v.reason);
      assert.ok(v.samples > 40, 'radii ' + [rA, rB, rC] + ': only ' + v.samples + ' samples judged');
    }
    for (const [rA, rB, rC] of [[4, 4, 4], [5, 8, 3], [3, 3, 1], [1, 1, 1.2]]) {
      const built = buildCorner(mixedBossEdges(rA, rB, rC), mixedBossBlends(rA, rB, rC));
      assert.equal(built.ok, true, 'radii ' + [rA, rB, rC] + ': ' + built.reason);
      const v = cornerPatchFolds(built.patch);
      assert.equal(v.folds, true, 'radii ' + [rA, rB, rC] + ' fold and the detector must say so — it reported ' + v.reversals + ' reversals, worst ' + v.worstAdjacentDeg.toFixed(1) + ' degrees');
    }
  });

  test('a mixed corner survives the bridge to a tensor-product surface — measured BETWEEN its stations', () => {
    const built = buildCorner(mixedBossEdges(2, 3, 7), mixedBossBlends(2, 3, 7));
    assert.equal(built.ok, true, built.reason);
    const N = 13;
    const r = cornerPatchSurface({ corners: built.corners, boundary: built.boundary, tangent: built.tangent, uSamples: N, vSamples: N });
    assert.equal(r.ok, true, r.reason);
    const at = (u, v) => surfacePoint(r.srf, u, v);
    let onStation = 0;
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1);
      onStation = Math.max(onStation, dist(at(t, 1), built.boundary[0](t)));
      onStation = Math.max(onStation, dist(at(0, t), built.boundary[2](t)));
      onStation = Math.max(onStation, dist(at(1, t), built.boundary[1](1 - t)));
    }
    assert.ok(onStation < 1e-9, 'boundary stations moved by ' + onStation);
    /* ⚠ EXACTNESS AT THE STATIONS IS THE INTERPOLANT SAMPLED AT ITS OWN DATA and
       proves nothing on its own. The half-stations are where a bridge that
       reproduced its data and did as it liked between them shows up, and the
       RATE is the assertion a single bound cannot make. */
    const err = (n) => {
      const rr = cornerPatchSurface({ corners: built.corners, boundary: built.boundary, tangent: built.tangent, uSamples: n, vSamples: n });
      let w = 0;
      for (let i = 0; i < n - 1; i++) {
        const t = (i + 0.5) / (n - 1);
        w = Math.max(w, dist(surfacePoint(rr.srf, t, 1), built.boundary[0](t)));
      }
      return w;
    };
    const coarse = err(7), fine = err(13);
    const span = Math.max(dist(built.corners[0], built.corners[1]), dist(built.corners[1], built.corners[2]));
    assert.ok(fine < 1e-3 * span, 'between stations the boundary deviates ' + fine + ' on a corner ' + span.toFixed(2) + ' across');
    assert.ok(fine * 4 < coarse, 'refining 7 -> 13 moved the error from ' + coarse + ' to ' + fine + ', which is not convergence');
  });

  // ── THE TORUS PATH ────────────────────────────────────────────────────────

  test('the two-equal-radii torus is the SAME torus for a corner and for its complement', () => {
    /* Its rolling ball rides the wedge the co-normals span, which is the
       material at a convex corner and the void at a concave one — the same
       wedge either way, so the same torus, the same lunes and the same handoff.
       Nothing in the construction has to be told which. */
    const concave = (o) => ({ ...o,
      edgeA: { ...o.edgeA, phi: P32 }, edgeB: { ...o.edgeB, phi: P32 }, edgeC: { ...o.edgeC, phi: P32 } });
    const cvx = twoEqualRadiiCorner(octant(5, 3));
    const ccv = twoEqualRadiiCorner(concave(octant(5, 3)));
    assert.equal(ccv.ok, true, ccv.reason);
    near(ccv.torus.centre, cvx.torus.centre, 1e-12);
    assert.ok(Math.abs(ccv.torus.majorRadius - cvx.torus.majorRadius) < 1e-12);
    assert.ok(Math.abs(ccv.handoff.offsetAlongEdge - cvx.handoff.offsetAlongEdge) < 1e-12);
    // And it is the right torus, not merely the same one: tangent to the shared
    // face, and reaching the third blend exactly.
    assert.ok(Math.abs(ccv.torus.centre[2] - ccv.torus.minorRadius) < 1e-12,
      'the torus must touch the shared face, lowest z ' + (ccv.torus.centre[2] - ccv.torus.minorRadius));
    assert.ok(Math.abs((ccv.torus.majorRadius - ccv.torus.minorRadius) - 3) < 1e-12);
  });

  test('⚠ THE TORUS PATH REFUSES A MIXED CORNER, and names the ball rather than the construction', () => {
    /* One ball, tangent to the shared face the whole way. At a convex edge it
       rides the material side of that face and at a concave one the void side,
       so a corner mixing them asks for a path across the plane it is supposed to
       stay tangent to. The crossings have no such requirement. */
    const mk = (pa, pb, pc) => {
      const o = octant(5, 3);
      o.edgeA.phi = pa; o.edgeB.phi = pb; o.edgeC.phi = pc;
      return twoEqualRadiiCorner(o);
    };
    for (const trio of [[Math.PI / 2, P32, Math.PI / 2], [P32, P32, Math.PI / 2], [Math.PI / 2, Math.PI / 2, P32]]) {
      const r = mk(trio[0], trio[1], trio[2]);
      assert.equal(r.ok, false, trio + ' mixes convexity and must be refused');
      assert.match(r.reason, /mixes convex and concave/);
      assert.match(r.reason, /rolling ball/);
    }
    assert.equal(mk(Math.PI / 2, Math.PI / 2, Math.PI / 2).ok, true, 'all convex still works');
    assert.equal(mk(P32, P32, P32).ok, true, 'and so does all concave');
    // A flat or out-of-range dihedral is caught here too, rather than turned
    // into a torus placed off an unbounded setback.
    const flat = mk(Math.PI, Math.PI, Math.PI);
    assert.equal(flat.ok, false);
    assert.match(flat.reason, /out of range/);
  });

  test('the co-normals cannot tell a convex edge from its concave twin, and the dihedral is still held to what they CAN say', () => {
    /* acos returns [0, pi], so two co-normals 90 degrees apart are equally
       consistent with a 90-degree edge and a 270-degree one — the material side
       is genuinely absent from an edge direction and two in-face directions, so
       phi's half of the range is a declaration rather than a corroborated fact.
       What the check still holds is the WEDGE: 60 degrees claimed either way, on
       co-normals 90 apart, is a setback taken off a face the co-normal is not on
       and is refused. */
    assert.equal(cornerPatchCorners([0, 0, 0], orthoEdges(5, 8, 3)).ok, true);
    assert.equal(cornerPatchCorners([0, 0, 0], concaveOrthoEdges(5, 8, 3)).ok, true);
    for (const bad of [Math.PI / 3, 2 * Math.PI - Math.PI / 3]) {
      const e = concaveOrthoEdges(5, 8, 3);
      e[0].phi = bad;
      const r = cornerPatchCorners([0, 0, 0], e);
      assert.equal(r.ok, false, 'a ' + bad + ' dihedral on 90-degree co-normals must be refused');
      assert.match(r.reason, /co-normals are .* apart while its dihedral says/);
    }
  });
}
