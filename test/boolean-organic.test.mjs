import test from 'node:test';
import assert from 'node:assert';
import { booleanSolids } from '../kernel/boolean.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { revolve } from '../kernel/primitives.mjs';
import { trivialTrimLoop } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import { surfacePoint } from '../kernel/surface.mjs';

// ---------------------------------------------------------------------------
// THE FIXTURE: a WAVY REVOLVE — a blob, not a primitive on a primitive. Its
// profile is a real interpolated curve that bulges in and out, revolved a
// full turn, so the resulting surface is closed in its sweep direction AND
// carries a POLE at each end where the profile touches the axis.
//
// That combination is what this file exists for. A box fixture cannot reach
// it: a pole collapses an entire (u,v) row to ONE point, and a closed sweep
// makes the domain's two v-ends the same seam — so the face's own domain
// rectangle evaluates to as few as two distinct points in 3D. Its boundary
// carries no length, no area, and nothing for a welder to sew, while the
// only REAL boundary the cut leaves is the intersection curve itself.
// ---------------------------------------------------------------------------

function quadSurface(p00, p10, p11, p01) {
  const w = (p) => [p[0], p[1], p[2], 1];
  return {
    degU: 1, degV: 1,
    knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[w(p00), w(p01)], [w(p10), w(p11)]],
  };
}

function boxFaces(lo, hi) {
  const c = (x, y, z) => [x, y, z];
  return [
    { srf: quadSurface(c(lo[0], lo[1], lo[2]), c(hi[0], lo[1], lo[2]), c(hi[0], hi[1], lo[2]), c(lo[0], hi[1], lo[2])) },
    { srf: quadSurface(c(lo[0], lo[1], hi[2]), c(hi[0], lo[1], hi[2]), c(hi[0], hi[1], hi[2]), c(lo[0], hi[1], hi[2])) },
    { srf: quadSurface(c(lo[0], lo[1], lo[2]), c(hi[0], lo[1], lo[2]), c(hi[0], lo[1], hi[2]), c(lo[0], lo[1], hi[2])) },
    { srf: quadSurface(c(lo[0], hi[1], lo[2]), c(hi[0], hi[1], lo[2]), c(hi[0], hi[1], hi[2]), c(lo[0], hi[1], hi[2])) },
    { srf: quadSurface(c(lo[0], lo[1], lo[2]), c(lo[0], hi[1], lo[2]), c(lo[0], hi[1], hi[2]), c(lo[0], lo[1], hi[2])) },
    { srf: quadSurface(c(hi[0], lo[1], lo[2]), c(hi[0], hi[1], lo[2]), c(hi[0], hi[1], hi[2]), c(hi[0], lo[1], hi[2])) },
  ];
}

function tessellate(faces, res) {
  const tris = [];
  for (const f of faces) {
    for (const tri of tessellateTrimmedSurface(f.srf, f.trimLoop ?? trivialTrimLoop(f.srf), res, res, f.trimHoles ?? [])) {
      tris.push(tri.map((v) => v.position));
    }
  }
  return tris;
}

// The curves come from a REAL face-pair SSI sweep, not hand-derived — on a
// wavy revolve there is no closed form to hand-derive them from, which is
// exactly why this fixture is worth having.
function buildFixture() {
  const profile = globalCurveInterp([[0, 0, 0], [12, 0, 6], [8, 0, 14], [15, 0, 24], [6, 0, 34], [0, 0, 40]], 3);
  const blobSrf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
  const blob = { faces: [{ srf: blobSrf }] };
  blob.triangles = tessellate(blob.faces, 40);

  const slabFaces = boxFaces([-30, -30, 18], [4, 30, 30]);
  const slab = { faces: slabFaces, triangles: tessellate(slabFaces, 2) };

  const curves = [];
  for (let j = 0; j < slab.faces.length; j++) {
    let r;
    try { r = intersectSurfacesComplete(blobSrf, slab.faces[j].srf); } catch { continue; }
    if (!r.ok) continue;
    for (const comp of r.components) curves.push({ samples: comp.samples.map((s) => s.point), faceA: 0, faceB: j });
  }
  return { blob, slab, curves };
}

test('the fixture really is the hard case: a closed revolve with poles at both ends', () => {
  const { blob, curves } = buildFixture();
  const srf = blob.faces[0].srf;
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];

  // Asserted on EVALUATION, not on control-net indexing, because evaluation
  // is what the sew actually reads — a face's boundary is its trim loop put
  // through `surfacePoint`, so "does this domain edge carry any length" is
  // the real question, and asking the control net instead would be testing
  // a different thing that happens to correlate.
  for (const u of [uMin, uMax]) {
    const at = (v) => surfacePoint(srf, u, v);
    const first = at(vMin);
    for (let k = 0; k <= 8; k++) {
      const p = at(vMin + (vMax - vMin) * (k / 8));
      const d = Math.hypot(p[0] - first[0], p[1] - first[1], p[2] - first[2]);
      assert.ok(d < 1e-9, `the domain edge u=${u} should collapse to one point, found a spread of ${d}`);
    }
  }

  // And the sweep is genuinely closed: the two v-ends are the same seam.
  const uMid = (uMin + uMax) / 2;
  const a = surfacePoint(srf, uMid, vMin), b = surfacePoint(srf, uMid, vMax);
  assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-9, 'v-ends are the same seam');

  assert.ok(curves.length > 0, 'the slab genuinely cuts the blob');
});

test('an organic blob cut by a slab INTERSECTS into one closed solid', () => {
  const { blob, slab, curves } = buildFixture();
  const res = booleanSolids(blob, slab, curves, 'intersect');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.stats.nakedEdgeCount, 0, 'every shared edge welded');
  assert.equal(res.stats.chi, 2, 'a sphere, topologically');
  assert.equal(res.stats.shellCount, 1);
});

test('the same pair UNIONS into one closed solid — the pole/seam case', () => {
  // This is the operation the degenerate-outer-loop handling exists for: the
  // blob's own kept fragment is the whole surface minus the cut region, whose
  // outer loop is the collapsed domain rectangle and whose only real edge is
  // the hole.
  const { blob, slab, curves } = buildFixture();
  const res = booleanSolids(blob, slab, curves, 'union');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.chi, 2);
  assert.equal(res.stats.shellCount, 1);
});

test('the same pair DIFFERENCES into one closed solid', () => {
  // The slab cuts a bite out of one side rather than passing clean through,
  // so the result stays a single connected solid — unlike the star prism,
  // where a slab through the middle correctly leaves two.
  const { blob, slab, curves } = buildFixture();
  const res = booleanSolids(blob, slab, curves, 'difference');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.chi, 2);
  assert.equal(res.stats.shellCount, 1);
});
