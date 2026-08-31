// AN OPEN CUT CURVE CROSSING A CLOSED SURFACE'S SEAM.
//
// Every seam path in this kernel used to assume a CLOSED cut curve, and no
// fixture could reach the open case: two closed surfaces always meet in a
// closed circle. A sphere meets a box PANEL in an open ARC — and that arc
// can cross the sphere's own seam meridian in its middle. unwrapSeamCut's
// closed-in-3D gate correctly declined to run the cyclic wrap test on it,
// then fell through to the RAW uv, which is only valid for a curve that
// never touches the seam at all. The result was a phantom chord across
// almost the whole face, an arrangement split along that phantom, and a
// boolean built from fragments that are not real.
//
// The unit tests below pin the split itself. The end-to-end test at the
// bottom is the one that would actually have caught the bug: a real sphere
// unioned with a real box straddling its seam, against the SAME pair with
// the box moved clear of the seam as a control — so anything that differs
// between the two runs is attributable to the seam and nothing else.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { seamOpenChains, seamStraddleChains, trivialTrimLoop } from '../kernel/trim.mjs';
import { booleanSolids } from '../kernel/boolean.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { revolve } from '../kernel/primitives.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';

// A surface closed in V over the domain [0,4] — the shape a full revolve
// has, which is what puts a seam in the way in the first place.
function closedInV() {
  const R = 10;
  const pts = [];
  for (let i = 0; i < 9; i++) {
    const th = -Math.PI / 2 + (Math.PI * i) / 8;
    pts.push([R * Math.cos(th), 0, R * Math.sin(th)]);
  }
  return revolve(globalCurveInterp(pts, 3), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
}

const V_MAX = 4;

test('SEAM OPEN CHAIN: an open arc crossing the seam splits into two pieces, each reaching the domain edge', () => {
  const srf = closedInV();
  // An open chain walking UP in v, jumping the seam once between samples 2
  // and 3 — the shape a real arc has where it crosses the meridian.
  const chain = [
    [1.0, 3.4], [1.1, 3.7], [1.2, 3.95],
    [1.3, 0.05], [1.4, 0.3], [1.5, 0.6],
  ];
  const r = seamOpenChains(chain, srf);
  assert.equal(r.ok, true, r.reason);
  assert.equal(r.axis, 'v');
  assert.equal(r.chains.length, 2, 'one crossing splits an open chain into exactly two pieces');

  const [first, second] = r.chains;
  // The first piece keeps the curve's OWN start and ends ON the seam it
  // leaves through; the second opens on the other copy of that same seam
  // point and keeps the curve's own end. Neither is closed along a domain
  // edge — that is the closed-loop case, and doing it here would invent a
  // cut the geometry does not have.
  assert.deepEqual(first[0], chain[0], 'the first piece keeps the curve\'s own real start point');
  assert.equal(first[first.length - 1][1], V_MAX, 'the first piece ends exactly on the seam');
  assert.equal(second[0][1], 0, 'the second piece opens on the other copy of that same seam point');
  assert.deepEqual(second[second.length - 1], chain[chain.length - 1], 'the second piece keeps the curve\'s own real end point');

  // The two copies (v=vMin and v=vMax) are the SAME physical place on a
  // closed surface, so they must agree on the other axis — that agreement
  // is what lets the fragments either side weld downstream with no
  // seam-specific sewing step.
  assert.ok(
    Math.abs(first[first.length - 1][0] - second[0][0]) < 1e-12,
    'both copies of the crossing carry the same u — they are one point in 3D',
  );
});

test('SEAM OPEN CHAIN: the crossing u is interpolated across the jump, not snapped to a neighboring sample', () => {
  const srf = closedInV();
  // Deliberately asymmetric: the jump runs 3.9 -> 0.1, so the crossing sits
  // 3/4 of the way through it. A crossing that merely reused a neighboring
  // sample's u would land on 2.0 or 3.0 and pass a weaker test.
  const chain = [[1.0, 3.5], [2.0, 3.9], [3.0, 0.1], [3.5, 0.5]];
  const r = seamOpenChains(chain, srf);
  assert.equal(r.ok, true, r.reason);
  const uSeam = r.chains[0][r.chains[0].length - 1][0];
  const gap = (V_MAX - 3.9) + 0.1; // 0.2 of v spanned by the jump
  const frac = (V_MAX - 3.9) / gap; // 0.5
  assert.ok(Math.abs(uSeam - (2.0 + (3.0 - 2.0) * frac)) < 1e-9, `expected the interpolated crossing u, got ${uSeam}`);
});

test('SEAM OPEN CHAIN: two crossings give three pieces', () => {
  const srf = closedInV();
  const chain = [
    [1.0, 3.8], [1.1, 3.95],
    [1.2, 0.1], [1.3, 0.2], [1.4, 0.05],
    [1.5, 3.9], [1.6, 3.7],
  ];
  const r = seamOpenChains(chain, srf);
  assert.equal(r.ok, true, r.reason);
  assert.equal(r.chains.length, 3, 'n crossings split an open chain into n+1 pieces');
  // Only the two INTERIOR ends are seam points; the outer two are the
  // curve's own real endpoints.
  assert.deepEqual(r.chains[0][0], chain[0]);
  assert.deepEqual(r.chains[2][r.chains[2].length - 1], chain[chain.length - 1]);
});

test('SEAM OPEN CHAIN: a chain nowhere near the seam is reported as not entangled at all', () => {
  const srf = closedInV();
  const chain = [[1.0, 1.0], [1.1, 1.2], [1.2, 1.4], [1.3, 1.6]];
  const r = seamOpenChains(chain, srf);
  assert.equal(r.ok, false);
  // The CODE is what the caller branches on — a benign refusal means the raw
  // chain is exactly right and must pass through untouched. Matching on the
  // prose instead is how two very different failures get conflated.
  assert.equal(r.code, 'no-seam-crossing');
});

test('SEAM OPEN CHAIN: reading the same open chain CYCLICALLY misreads it — which is why this sibling exists', () => {
  const srf = closedInV();
  // One real crossing in the middle. Read cyclically, the fake last->first
  // step counts as a second crossing, so the loop reader sees an even count
  // and reports a straddle — then finds its subchains degenerate.
  const chain = [
    [1.0, 3.4], [1.1, 3.7], [1.2, 3.95],
    [1.3, 0.05], [1.4, 0.3], [1.5, 0.6],
  ];
  const open = seamOpenChains(chain, srf);
  assert.equal(open.ok, true, 'the open reading splits it correctly');
  assert.equal(open.chains.length, 2);

  const cyclic = seamStraddleChains(chain, srf);
  // The cyclic reading is not merely different, it is unusable here: it
  // either refuses outright or returns a piece count that does not describe
  // this curve. Pinned so a future "simplification" that drops the open
  // sibling and reuses the loop reader fails loudly instead of silently
  // handing the arrangement a phantom chord again.
  const cyclicIsWrong = !cyclic.ok || cyclic.chains.length !== 2;
  assert.ok(cyclicIsWrong, 'the cyclic reader must NOT quietly produce the right answer for an open chain');
});

// ---------------------------------------------------------------------------
// END TO END — the test that would have caught the reported bug.
// ---------------------------------------------------------------------------

const R = 170.5;
const BH = [142, 113, 113];
const CX = Math.hypot(197, 161);

function tess(faces, res) {
  const tris = [];
  for (const f of faces) {
    for (const t of tessellateTrimmedSurface(f.srf, f.trimLoop ?? trivialTrimLoop(f.srf), res, res, f.trimHoles ?? [])) {
      tris.push(t.map((v) => v.position));
    }
  }
  return tris;
}

function sphereSolid() {
  const pts = [];
  for (let i = 0; i < 9; i++) {
    const th = -Math.PI / 2 + (Math.PI * i) / 8;
    pts.push([R * Math.cos(th), 0, R * Math.sin(th)]);
  }
  const srf = revolve(globalCurveInterp(pts, 3), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
  const faces = [{ srf }];
  return { faces, triangles: tess(faces, 40) };
}

// A degree-1 x degree-1 patch's control net IS its geometry, so the four
// corners are exact with nothing fitted — the app's own box-panel shape.
function panel(p00, p10, p11, p01) {
  return {
    degU: 1, degV: 1,
    knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[[...p00, 1], [...p01, 1]], [[...p10, 1], [...p11, 1]]],
  };
}

function boxSolid(cx, cy, cz) {
  const [hx, hy, hz] = BH;
  const x0 = cx - hx, x1 = cx + hx, y0 = cy - hy, y1 = cy + hy, z0 = cz - hz, z1 = cz + hz;
  const faces = [
    { srf: panel([x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]) },
    { srf: panel([x0, y0, z1], [x0, y1, z1], [x1, y1, z1], [x1, y0, z1]) },
    { srf: panel([x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]) },
    { srf: panel([x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0]) },
    { srf: panel([x0, y0, z0], [x0, y0, z1], [x1, y0, z1], [x1, y0, z0]) },
    { srf: panel([x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]) },
  ];
  return { faces, triangles: tess(faces, 6) };
}

function cutCurves(A, B) {
  const out = [];
  for (let i = 0; i < A.faces.length; i++) {
    for (let j = 0; j < B.faces.length; j++) {
      const r = intersectSurfacesComplete(A.faces[i].srf, B.faces[j].srf);
      if (!r || !r.ok) continue;
      for (const comp of r.components) {
        if (comp.samples.length < 2) continue;
        out.push({ samples: comp.samples.map((s) => s.point), faceA: i, faceB: j });
      }
    }
  }
  return out;
}

test('SEAM OPEN CHAIN: a box containing the sphere\'s seam meridian still unions into a closed solid', () => {
  // makeCircle starts at +X, so the sphere's seam is the half-plane y=0,
  // x>0. A box on the +X side with |dy| under its own Y half-extent (113)
  // therefore CONTAINS that seam; dy=180 puts it clear. Only the Y offset
  // differs between the two runs.
  const A = sphereSolid();
  const straddling = booleanSolids(A, boxSolid(CX, 0, 0), cutCurves(A, boxSolid(CX, 0, 0)), 'union');
  assert.equal(straddling.ok, true, `the straddling pair must close: ${straddling.reason || ''}`);
  assert.equal(straddling.stats.nakedEdgeCount, 0, 'no naked edges — the seam-straddling arc was split, not chorded across');
  assert.equal(straddling.stats.nonManifoldEdgeCount, 0, 'no edge shared by more than two faces');

  const clear = booleanSolids(A, boxSolid(CX, 180, 0), cutCurves(A, boxSolid(CX, 180, 0)), 'union');
  assert.equal(clear.ok, true, `the control must still close: ${clear.reason || ''}`);
  assert.equal(clear.stats.nakedEdgeCount, 0);
  assert.equal(clear.stats.nonManifoldEdgeCount, 0);
});
