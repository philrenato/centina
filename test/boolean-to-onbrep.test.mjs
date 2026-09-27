// A boolean result, assembled into an ON_Brep and checked by OpenNURBS.
//
// This is the end of the .3dm write path: our own sewn solid -> merged edges with fitted curves -> pcurves per
// half-edge -> ON_Brep through the authoring bindings -> ON_Brep::IsValid.
//
// The verdict here comes from an independent validator. Every other check in
// this suite is the kernel checking against its own conventions — naked-edge
// counts, Euler characteristics, deviations we defined and measured.
// OpenNURBS's validator knows none of those conventions, so it tells a valid
// solid from a merely closed-looking one.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { surfaceFromRhino } from '../io3dm.mjs';
import { trivialTrimLoop } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { booleanSolids } from '../kernel/boolean.mjs';
import { fitSolidEdgeCurves, fitHalfEdgePcurves, fitFaceLoops } from '../kernel/brepfit.mjs';
import { surfaceToRhino, curveToRhino } from '../io3dm.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
// The vendored build, not the npm package: the released rhino3dm has no
// authoring at all, so `import 'rhino3dm'` here would test nothing. The .cjs
// is the same build the browser loads; the ES module beside it takes
// emscripten's ENVIRONMENT_IS_NODE branch and calls a bare `require`, which an
// ES module does not have.
const VENDOR = path.join(HERE, '..', 'vendor', 'rhino3dm');
const rhino = await require(path.join(VENDOR, 'rhino3dm.cjs'))({ locateFile: (p) => path.join(VENDOR, p) });

const TOL = 0.01;
const doc = rhino.File3dm.fromByteArray(new Uint8Array(fs.readFileSync(
  path.join(HERE, 'fixtures', 'two_tori_seam_straddle_sew_open.3dm'))));
const operands = [];
{
  const objs = doc.objects();
  for (let i = 0; i < objs.count; i++) {
    const g = objs.get(i).geometry();
    const ns = g instanceof rhino.NurbsSurface ? g : (g.toNurbsSurface ? g.toNurbsSurface() : null);
    if (!ns) continue;
    const srf = surfaceFromRhino(ns);
    operands.push({ faces: [{ srf }], triangles: tessellateTrimmedSurface(srf, trivialTrimLoop(srf), 60, 60, []).map((t) => t.map((v) => v.position)) });
  }
}
const ssi = intersectSurfacesComplete(operands[0].faces[0].srf, operands[1].faces[0].srf);
const curves = ssi.components.map((c) => ({ samples: c.samples.map((s) => s.point), faceA: 0, faceB: 0 }));

// Assemble one ON_Brep from our own solid. Faces share their edges, which is
// how a multi-face solid is built — ON_Brep has no join, and needs none.
function assemble(solid, fitted, pcurves, loops) {
  const brep = new rhino.Brep();
  const faces = solid.shells.flatMap((sh) => sh.faces || []);

  // Surfaces, one per face, in face order.
  const faceSurfaceIndex = new Map();
  for (const f of faces) {
    const si = brep.addSurface(surfaceToRhino(rhino, f.surface));
    faceSurfaceIndex.set(f, si);
  }
  // Vertices: only the corners survive the merge, and a fitted chain names its
  // own endpoints, so vertices are minted per distinct chain endpoint.
  const vertexIndex = new Map();
  const keyOf = (p) => `${p[0].toFixed(6)},${p[1].toFixed(6)},${p[2].toFixed(6)}`;
  const vertexFor = (p) => {
    const k = keyOf(p);
    if (!vertexIndex.has(k)) vertexIndex.set(k, brep.newVertex(p, TOL));
    return vertexIndex.get(k);
  };
  // Edges, one per fitted chain.
  const edgeIndex = new Map();
  for (const [i, e] of fitted.edges.entries()) {
    if (!e.curve) continue;
    const c3 = brep.addEdgeCurve(curveToRhino(rhino, e.curve));
    const a = vertexFor(e.points[0]);
    const b = e.closed ? a : vertexFor(e.points[e.points.length - 1]);
    edgeIndex.set(i, brep.newEdge(a, b, c3, TOL));
  }
  // Faces, loops and trims.
  const bySideFace = new Map(); // faceId -> [{edgeIndex, pieces}]
  for (const p of pcurves.pcurves) {
    if (!p.pieces) continue;
    if (!bySideFace.has(p.faceId)) bySideFace.set(p.faceId, []);
    bySideFace.get(p.faceId).push(p);
  }
  // Trims in loop-traversal order, each with its own orientation; OpenNURBS
  // rejects trims emitted in edge order.
  let trimCount = 0, skippedFaces = 0;
  for (const rec of loops.faces) {
    const f = faces.find((x) => x.id === rec.faceId);
    if (!f || !rec.loops || !rec.loops.length) { skippedFaces++; continue; }
    const fi = brep.newFace(faceSurfaceIndex.get(f));
    for (const loop of rec.loops) {
      const li = brep.newLoop(fi, loop.loopType === 'inner' ? 2 : 1);
      for (const t of loop.trims) {
        const ei = edgeIndex.get(t.chainIndex);
        if (ei == null) continue;
        const c2 = brep.addTrimCurve(curveToRhino(rhino, t.curve));
        if (brep.newTrim(li, ei, !!t.reversed, c2, TOL) >= 0) trimCount++;
      }
    }
  }
  brep.setTrimIsoFlags();
  brep.setVertexTolerances(true);
  brep.setEdgeTolerances(true);
  brep.setTrimTolerances(true);
  brep.compact();
  const [valid, log] = brep.isValidWithLog;
  return { brep, valid, log, counts: { faces: faces.length, vertices: vertexIndex.size, edges: edgeIndex.size, trims: trimCount, skippedFaces } };
}

const results = {};
for (const op of ['union', 'intersect', 'difference']) {
  const r = booleanSolids(operands[0], operands[1], curves, op);
  if (!r.ok) { results[op] = { refused: r.reason }; continue; }
  const fitted = fitSolidEdgeCurves(r.solid, { tolerance: TOL });
  const pcurves = fitHalfEdgePcurves(r.solid, fitted, { tolerance: TOL });
  const loops = fitFaceLoops(r.solid, fitted, { tolerance: TOL });
  results[op] = { boolean: r, fitted, pcurves, loops, ...assemble(r.solid, fitted, pcurves, loops) };
}

test('input: all three booleans close, fit and project', () => {
  for (const op of ['union', 'intersect', 'difference']) {
    const r = results[op];
    assert.ok(!r.refused, `${op} refused: ${r.refused}`);
    assert.equal(r.boolean.stats.nakedEdgeCount, 0, `${op} leaves no naked edge`);
    assert.ok(r.fitted.ok, `${op} did not fit`);
    assert.ok(r.pcurves.ok, `${op} did not project`);
  }
});

test('the vendored build is reachable from node, with its authoring intact', () => {
  // Without this the whole file would be testing the released package, which
  // has no authoring and would fail for a reason that says nothing.
  assert.equal(typeof rhino.Brep.prototype.newTrim, 'function');
  assert.equal(typeof rhino.Brep.prototype.addTrimCurve, 'function');
});

test('every boolean assembles into an ON_Brep — no call refused mid-build', () => {
  for (const op of ['union', 'intersect', 'difference']) {
    const c = results[op].counts;
    assert.ok(c.faces > 0 && c.edges > 0 && c.trims > 0,
      `${op} assembled ${JSON.stringify(c)}`);
    assert.equal(c.skippedFaces, 0, `${op} left ${c.skippedFaces} faces with no side data`);
  }
});

test('OpenNURBS returns a legible verdict on each', () => {
  // Asserts that a verdict exists and is legible, not that it is favorable;
  // the next test asserts validity.
  for (const op of ['union', 'intersect', 'difference']) {
    const r = results[op];
    assert.equal(typeof r.valid, 'boolean');
    assert.equal(typeof r.log, 'string');
    if (!r.valid) assert.ok(r.log.length > 0, 'an invalid brep must say why');
  }
});

test('OpenNURBS reports all three valid', () => {
  // ON_Brep::IsValid knows none of this kernel's conventions. Assembly
  // defects it rejects:
  //   1. trims emitted in fitted-edge order, not loop-traversal order
  //   2. no per-trim orientation (half of them run against their edge)
  //   3. a primitive fit moving a pcurve endpoint outside the domain
  //   4. a whole-period shift computed with round() instead of floor(),
  //      which pushes a run out of the rectangle so the clamp squashes a
  //      391-unit 3-D run onto a single parameter
  for (const op of ['union', 'intersect', 'difference']) {
    const r = results[op];
    assert.equal(r.valid, true, `${op} rejected: ${r.log}`);
    assert.equal(r.log, '', `${op} must validate with an empty log, not a quiet pass`);
  }
});

test('a valid brep stays valid through a .3dm round trip', () => {
  // Valid in memory is not the deliverable; valid after being written and read
  // back is. This is the write path's claim.
  for (const op of ['union', 'intersect', 'difference']) {
    const doc3 = new rhino.File3dm();
    doc3.settings().modelAbsoluteTolerance = TOL;
    doc3.objects().add(results[op].brep, null);
    const bytes = doc3.toByteArray();
    assert.ok(bytes.length > 1000, `${op} wrote real bytes (${bytes.length})`);
    const back = rhino.File3dm.fromByteArray(bytes).objects().get(0).geometry();
    assert.equal(back.constructor.name, 'Brep');
    assert.equal(back.faces().count, results[op].counts.faces, `${op} kept its face count`);
    const [v, log] = back.isValidWithLog;
    assert.equal(v, true, `${op} invalid after a round trip: ${log}`);
  }
});

test('each boolean result passes the kernel\'s own closed, manifold and Euler checks', () => {
  // Each result satisfies every check the kernel makes: closed, manifold,
  // and on the Euler characteristic inclusion-exclusion predicts. A validator
  // rejection therefore points at assembly, not geometry.
  for (const [op, chi] of [['union', -2], ['intersect', 2], ['difference', 0]]) {
    const s = results[op].boolean.stats;
    assert.equal(s.nakedEdgeCount, 0, `${op} closed`);
    assert.equal(s.nonManifoldEdgeCount, 0, `${op} manifold`);
    assert.equal(s.chi, chi, `${op} on the predicted topology`);
  }
});
