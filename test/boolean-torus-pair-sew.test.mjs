// THE TORUS PAIR THAT WOULD NOT SEW — the banked .3dm, run through the real
// boolean, with its own operands measured first.
//
// Two tori that BOTH straddle their seams reach a state no sphere-pair fixture
// can: each surface is closed in u AND v, so a fragment covering the whole
// domain is a punctured torus rather than a punctured sphere. The union sewed
// to DEGENERATE, then to an OPEN SHELL with 35 naked edges, then 28, and this
// file is what keeps those numbers from drifting back up unnoticed.
//
// ALL THREE OPERATORS NOW CLOSE, and the Euler characteristic each one lands
// on is asserted rather than accepted. chi is the check that catches a shell
// which closed by welding the wrong things together: a count of naked edges
// alone cannot tell a correct solid from one that sewed itself into the wrong
// topology.
//
// THE OPERANDS ARE MEASURED BEFORE ANYTHING IS ASSERTED ABOUT THEM. A fixture
// nobody measures is a free variable in every result resting on it — this
// project has already shipped a "radius 20 ball" that spanned 17.73 to 22.42
// and passed five assertions on it.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import rhino3dmFactory from 'rhino3dm';
import { surfaceFromRhino } from '../io3dm.mjs';
import { surfacePoint, surfaceClosure } from '../kernel/surface.mjs';
import { trivialTrimLoop } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { booleanSolids } from '../kernel/boolean.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = path.join(HERE, 'fixtures', 'two_tori_seam_straddle_sew_open.3dm');

const rhino = await rhino3dmFactory();
const doc = rhino.File3dm.fromByteArray(new Uint8Array(fs.readFileSync(FIXTURE)));
const objs = doc.objects();
const solids = [];
for (let i = 0; i < objs.count; i++) {
  const g = objs.get(i).geometry();
  const ns = g instanceof rhino.NurbsSurface ? g : (g.toNurbsSurface ? g.toNurbsSurface() : null);
  if (!ns) continue;
  const srf = surfaceFromRhino(ns);
  const tris = tessellateTrimmedSurface(srf, trivialTrimLoop(srf), 60, 60, []).map((t) => t.map((v) => v.position));
  solids.push({ name: objs.get(i).attributes().name || `#${i}`, faces: [{ srf }], triangles: tris });
}

function evaluatedBox(srf, n = 40) {
  const uMax = srf.knotsU[srf.knotsU.length - 1], vMax = srf.knotsV[srf.knotsV.length - 1];
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) {
    const p = surfacePoint(srf, (i / n) * uMax, (j / n) * vMax);
    for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
  }
  return { lo, hi, size: hi.map((v, k) => v - lo[k]) };
}

test('THE FIXTURE ITSELF: two tori, each closed in BOTH directions, at the size the file says', () => {
  assert.equal(solids.length, 2, 'the fixture holds exactly two NURBS surfaces');
  for (const s of solids) {
    const { srf } = s.faces[0];
    const closure = surfaceClosure(srf);
    assert.ok(closure.closedU && closure.closedV, `${s.name} is closed in both u and v — the whole point of this fixture`);
    assert.equal(srf.degU, 3);
    assert.equal(srf.degV, 3);
    const box = evaluatedBox(srf);
    // A torus is as wide as it is deep, and much flatter than either. These
    // bounds are wide enough to survive an honest re-export and tight enough
    // to catch an operand silently deformed by a bad conversion.
    assert.ok(box.size[0] > 400 && box.size[0] < 520, `${s.name} spans ${box.size[0].toFixed(1)} in x`);
    assert.ok(box.size[1] > 400 && box.size[1] < 520, `${s.name} spans ${box.size[1].toFixed(1)} in y`);
    assert.ok(box.size[2] > 90 && box.size[2] < 180, `${s.name} spans ${box.size[2].toFixed(1)} in z`);
    assert.ok(Math.abs(box.size[0] - box.size[1]) < 1, `${s.name} is as wide as it is deep`);
  }
});

const [A, B] = solids;
const ssi = intersectSurfacesComplete(A.faces[0].srf, B.faces[0].srf);
const curves = ssi && ssi.ok
  ? ssi.components.map((c) => ({ samples: c.samples.map((s) => s.point), faceA: 0, faceB: 0 }))
  : [];

test('THE INPUT LANDS: SSI finds the single intersection component the boolean is fed', () => {
  // An SSI that refused looks exactly like a boolean that failed, three stages
  // downstream. Nothing below means anything without this.
  assert.ok(ssi && ssi.ok, `SSI refused: ${ssi ? ssi.reason : 'null'}`);
  assert.equal(curves.length, 1);
  assert.ok(curves[0].samples.length > 100, 'the component is sampled densely enough to trim with');
});

// EACH chi IS DERIVED, NOT OBSERVED. A solid torus is a handlebody of genus 1,
// so chi(solid) = 1 - g = 0 for each operand, and their overlap here is a
// single ball, chi = 1. Inclusion-exclusion gives chi(A u B) = 0 + 0 - 1 = -1,
// a genus-2 handlebody, whose BOUNDARY surface is chi = 2 - 2g = -2. Intersect
// is that ball, boundary chi = 2. Difference is A with a dent that does not
// reach through its tube, so it stays genus 1 and its boundary stays chi = 0.
for (const [op, chi, faces] of [['union', -2, 18], ['intersect', 2, 6], ['difference', 0, 12]]) {
  test(`${op.toUpperCase()} closes into a manifold solid of chi ${chi}`, () => {
    const res = booleanSolids(A, B, curves, op);
    assert.ok(res.ok, `${op} refused: ${res.reason}`);
    assert.equal(res.stats.nakedEdgeCount, 0, `${op} leaves no naked edge`);
    assert.equal(res.stats.nonManifoldEdgeCount, 0, `${op} produces no non-manifold edge`);
    assert.equal(res.stats.F, faces, `${op} builds ${faces} valid faces`);
    assert.equal(res.stats.chi, chi, `${op} lands on the topology inclusion-exclusion predicts`);
  });
}
