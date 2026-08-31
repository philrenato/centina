import test from 'node:test';
import assert from 'node:assert';
import { booleanSolids } from '../kernel/boolean.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { revolve } from '../kernel/primitives.mjs';
import { trivialTrimLoop } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import { surfacePoint, closestPointOnSurface } from '../kernel/surface.mjs';

// ---------------------------------------------------------------------------
// THE FIXTURE: two revolved balls offset ALONG THEIR OWN SEAM MERIDIANS, so
// the cut circle crosses both operands' seams at one and the same 3D point.
//
// That coincidence is the whole case. A seam crossing is a CORNER of a face's
// boundary, and each face used to derive it by interpolating linearly between
// the two curve samples straddling its own domain edge, in its own parameters.
// Two faces deriving the same corner that way get two different answers — 1.3e-2
// apart here, against a 1e-4 weld — and the shell comes back with a four-edge
// sliver quad joining the two corners through the samples either side.
//
// It is not exotic. Two solids of revolution translated apart put each one's
// seam meridian straight through the other, and this is simply the alignment
// where both happen at the same place. Swept over azimuth, a ball pair closes
// at eleven of thirteen and fails at the two where a seam runs through the
// cut.
//
// REFINING THE MARCH DOES NOT FIX IT, which is what makes it a defect in the
// construction rather than a tolerance to be tuned. Swept from stepLen 1 down
// to 0.0625 the sliver shrinks linearly with the sample spacing and the naked
// count stays at exactly 4 — two independent interpolations do not converge on
// each other at any density.
// ---------------------------------------------------------------------------

const R = 15;
const CENTRE_A = [0, 0, 20];
// Purely in the y = 0 half-plane the seams live in, and short enough that the
// two balls genuinely interpenetrate. Both are measured below.
const OFFSET = [Math.hypot(9, 3), 0, 5];

function tess(faces, res) {
  const tris = [];
  for (const f of faces) {
    for (const t of tessellateTrimmedSurface(f.srf, f.trimLoop ?? trivialTrimLoop(f.srf), res, res, f.trimHoles ?? [])) {
      tris.push(t.map((v) => v.position));
    }
  }
  return tris;
}

// TRANSLATE WITHOUT THE WEIGHT. The control net stores EUCLIDEAN xyz beside a
// separate weight, so a rigid shift is `x + cx`. The homogeneous spelling
// `x + cx*w` moves a revolve's rational columns (w = sqrt(2)/2) only 0.707 as
// far, and a "radius 15 ball" then measures as something else entirely. The
// first test below is what catches it.
function ball(offset) {
  const pts = [];
  for (let i = 0; i < 7; i++) {
    const th = -Math.PI / 2 + (Math.PI * i) / 6;
    pts.push([R * Math.cos(th), 0, CENTRE_A[2] + R * Math.sin(th)]);
  }
  const s = revolve(globalCurveInterp(pts, 3), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
  const srf = { ...s, ctrlNet: s.ctrlNet.map((row) => row.map(([x, y, z, w]) => [x + offset[0], y + offset[1], z + offset[2], w])) };
  return { faces: [{ srf }], triangles: tess([{ srf }], 40), srf, centre: CENTRE_A.map((c, i) => c + offset[i]) };
}

function cutCurves(A, B) {
  const out = [];
  let r;
  try { r = intersectSurfacesComplete(A.faces[0].srf, B.faces[0].srf); } catch { return out; }
  if (!r?.ok) return out;
  for (const c of r.components) if (c.samples.length >= 2) out.push({ samples: c.samples.map((s) => s.point), faceA: 0, faceB: 0 });
  return out;
}

const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
const domV = (srf) => [srf.knotsV[0], srf.knotsV[srf.knotsV.length - 1]];

test('the operands are what this file claims: two radius-15 balls that overlap, with both seams in the same half-plane', () => {
  const A = ball([0, 0, 0]), B = ball(OFFSET);
  // Read off the SURFACE over a (u,v) grid, about each body's OWN center — the
  // only reading that tells a real ball from a distorted control net.
  for (const X of [A, B]) {
    let lo = Infinity, hi = -Infinity;
    const uK = X.srf.knotsU, vK = X.srf.knotsV;
    for (let i = 0; i <= 24; i++) {
      for (let j = 0; j <= 60; j++) {
        const p = surfacePoint(X.srf, uK[0] + (uK[uK.length - 1] - uK[0]) * (i / 24), vK[0] + (vK[vK.length - 1] - vK[0]) * (j / 60));
        const d = dist(p, X.centre);
        lo = Math.min(lo, d); hi = Math.max(hi, d);
      }
    }
    // A degree-3 interpolation of a 7-point profile is not an exact circle, so
    // the band is the interpolant's own error and not a slack allowance — the
    // weight-scaled translate this guards against misses by more than a unit.
    assert.ok(lo > R - 0.05 && hi < R + 0.05, `ball at [${X.centre}] spans radius ${lo.toFixed(4)}..${hi.toFixed(4)}`);
  }
  const apart = dist(A.centre, B.centre);
  assert.ok(apart > 0 && apart < 2 * R, `centers ${apart.toFixed(4)} apart — the balls must genuinely interpenetrate`);

  // BOTH SEAMS IN THE PLANE y = 0. A revolve started at angle 0 puts its seam
  // meridian on +x, and B is translated within that plane, so B's seam is a
  // copy of A's shifted along it. Measured on the surfaces rather than argued
  // from the construction.
  for (const X of [A, B]) {
    const [vMin, vMax] = domV(X.srf);
    const uK = X.srf.knotsU;
    for (let i = 0; i <= 12; i++) {
      const u = uK[0] + (uK[uK.length - 1] - uK[0]) * (i / 12);
      for (const v of [vMin, vMax]) {
        const p = surfacePoint(X.srf, u, v);
        assert.ok(Math.abs(p[1]) < 1e-9, `seam point (${p}) is not in the y = 0 plane`);
      }
    }
  }

  const curves = cutCurves(A, B);
  assert.equal(curves.length, 1, 'expected exactly one intersection circle');
  assert.ok(curves[0].samples.length > 50, 'the cut must be a well-sampled curve, not a graze');
});

test('the cut really does cross BOTH operands\' seams at one and the same place — the coincidence this file is about', () => {
  const A = ball([0, 0, 0]), B = ball(OFFSET);
  const curves = cutCurves(A, B);
  const samples = curves[0].samples;

  // ASSERTED AT THE DATA'S OWN RESOLUTION, and that is the point rather than a
  // compromise. Interpolating the crossing off the chord between two samples
  // reproduces exactly the error this file exists to describe — measured at
  // 1.8e-3 here — so an assertion resting on such a point would be testing the
  // defect with the defect. What the curve genuinely knows is WHICH SEGMENT it
  // crosses each seam in, and the claim "both seams, one place" is precisely
  // the claim that it is the SAME segment.
  const seamSegments = (X) => {
    const [vMin, vMax] = domV(X.srf);
    const span = vMax - vMin;
    const vs = samples.map((p) => closestPointOnSurface(X.srf, p).v);
    const segs = [];
    for (let i = 1; i < vs.length; i++) if (Math.abs(vs[i] - vs[i - 1]) > span / 2) segs.push(i - 1);
    return segs;
  };
  const segA = seamSegments(A), segB = seamSegments(B);
  assert.equal(segA.length, 1, `the cut should cross A's seam exactly once, it crosses ${segA.length} times`);
  assert.equal(segB.length, 1, `the cut should cross B's seam exactly once, it crosses ${segB.length} times`);
  assert.equal(segA[0], segB[0], `the two seams are crossed in different segments (${segA[0]} vs ${segB[0]}) — they do not coincide, and this fixture has no coincidence to test`);

  // And that shared segment straddles the y = 0 half-plane both seams live in,
  // on the far side of B's center where BOTH meridians run — so the coincidence
  // is the geometric one claimed and not two unrelated crossings that happen to
  // fall in one sample step.
  const a = samples[segA[0]], b = samples[segA[0] + 1];
  assert.ok((a[1] < 0) !== (b[1] < 0) || a[1] === 0 || b[1] === 0, `the shared segment (y = ${a[1]} to ${b[1]}) does not cross the seams' own plane`);
  assert.ok(a[0] > OFFSET[0] && b[0] > OFFSET[0], `the shared segment sits at x = ${a[0].toFixed(4)}..${b[0].toFixed(4)}, not beyond x = ${OFFSET[0].toFixed(4)} where both meridians run`);

  // The samples either side are on both surfaces, which is what makes them a
  // meaningful bracket rather than two arbitrary points.
  for (const p of [a, b]) {
    for (const X of [A, B]) {
      assert.ok(closestPointOnSurface(X.srf, p).distance < 1e-6, 'a cut sample is not on a surface it was marched onto');
    }
  }
});

test('coincident seams: the pair UNIONS into one closed solid, and the shared corner is what closes it', () => {
  const A = ball([0, 0, 0]), B = ball(OFFSET);
  const curves = cutCurves(A, B);
  const res = booleanSolids(A, B, curves, 'union');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.stats.nakedEdgeCount, 0, 'every shared edge welded');
  assert.equal(res.stats.nonManifoldEdgeCount, 0);
  assert.equal(res.stats.chi, 2, 'a sphere, topologically');
  assert.equal(res.stats.shellCount, 1);
  // NOT VACUOUS: the sharing pass has to have done something here, or the test
  // would pass just as well with the machinery removed. Two corners — one per
  // operand's seam — landing on the same place.
  assert.ok(res.seamPoints > 0, `expected the seam-crossing pass to insert a corner, it inserted ${res.seamPoints}`);
});

test('coincident seams: INTERSECT and DIFFERENCE close on the same pair', () => {
  for (const op of ['intersect', 'difference']) {
    const A = ball([0, 0, 0]), B = ball(OFFSET);
    const res = booleanSolids(A, B, cutCurves(A, B), op);
    assert.ok(res.ok, `${op}: ${res.reason || res.verdict}`);
    assert.equal(res.stats.nakedEdgeCount, 0, `${op} left naked edges`);
    assert.equal(res.stats.chi, 2, `${op} is not a topological sphere`);
  }
});

test('the SAME pair with its seams rotated off the cut still closes, and the pass leaves an untouched face alone', () => {
  // The control, and it is doing two jobs. It says the fix did not buy the
  // coincident case at the expense of the ordinary one; and because a revolve's
  // seam always points somewhere, it pins that "somewhere" as the only variable
  // between this and the test above.
  const rot = (deg) => {
    const a = (deg * Math.PI) / 180;
    return [Math.hypot(9, 3) * Math.cos(a), Math.hypot(9, 3) * Math.sin(a), 5];
  };
  for (const deg of [30, 60, 90, 150]) {
    const A = ball([0, 0, 0]), B = ball(rot(deg));
    const curves = cutCurves(A, B);
    assert.equal(curves.length, 1, `azimuth ${deg}: expected one intersection circle`);
    const res = booleanSolids(A, B, curves, 'union');
    assert.ok(res.ok, `azimuth ${deg}: ${res.reason || res.verdict}`);
    assert.equal(res.stats.nakedEdgeCount, 0, `azimuth ${deg} left naked edges`);
    assert.equal(res.stats.chi, 2, `azimuth ${deg} is not a topological sphere`);
  }
});

test('a pair with NO closed direction anywhere is untouched by the seam pass — it cannot have a seam to share', () => {
  // Two boxes. Nothing here is closed in u or v, so the pass must decline
  // before it projects anything, and the boolean must be exactly what it always
  // was. This is the guard on the early-out being a real skip rather than a
  // silent behavior change for every planar face in the suite.
  const quad = (p00, p10, p11, p01) => {
    const w = (p) => [p[0], p[1], p[2], 1];
    return { degU: 1, degV: 1, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1], ctrlNet: [[w(p00), w(p01)], [w(p10), w(p11)]] };
  };
  const boxFaces = (lo, hi) => [
    { srf: quad([lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]]) },
    { srf: quad([lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]) },
    { srf: quad([lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]], [lo[0], lo[1], hi[2]]) },
    { srf: quad([lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]) },
    { srf: quad([lo[0], lo[1], lo[2]], [lo[0], hi[1], lo[2]], [lo[0], hi[1], hi[2]], [lo[0], lo[1], hi[2]]) },
    { srf: quad([hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [hi[0], hi[1], hi[2]], [hi[0], lo[1], hi[2]]) },
  ];
  const mk = (lo, hi) => { const faces = boxFaces(lo, hi); return { faces, triangles: tess(faces, 2) }; };
  const P = mk([0, 0, 0], [10, 10, 10]), Q = mk([5, 5, 5], [15, 15, 15]);
  const curves = [];
  for (let i = 0; i < P.faces.length; i++) {
    for (let j = 0; j < Q.faces.length; j++) {
      let r;
      try { r = intersectSurfacesComplete(P.faces[i].srf, Q.faces[j].srf); } catch { continue; }
      if (!r?.ok) continue;
      for (const c of r.components) if (c.samples.length >= 2) curves.push({ samples: c.samples.map((s) => s.point), faceA: i, faceB: j });
    }
  }
  assert.ok(curves.length > 0, 'the two boxes must genuinely cut each other');
  const res = booleanSolids(P, Q, curves, 'union');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.seamPoints, 0, 'a pair with no closed direction must contribute no seam corners');
  assert.equal(res.stats.nakedEdgeCount, 0);
});
