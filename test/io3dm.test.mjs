import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import rhino3dmFactory from 'rhino3dm';
import { makeArc, revolve } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { curvePoint } from '../kernel/curve.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { buildTopology, subdivideCatmullClark } from '../kernel/subd.mjs';
import {
  ourKnotsToRhino, rhinoKnotsToOurs,
  curveToRhino, curveFromRhino,
  surfaceToRhino, surfaceFromRhino,
  exportDocument, importDocument,
} from '../io3dm.mjs';

const rhino = await rhino3dmFactory();

function worstCurveDeviation(ours, rhinoCrv, samples = 25) {
  const uMin = ours.knots[ours.degree];
  const uMax = ours.knots[ours.knots.length - 1 - ours.degree];
  let worst = 0;
  for (let i = 0; i <= samples; i++) {
    const u = uMin + (uMax - uMin) * i / samples;
    const a = curvePoint(ours, u);
    const b = rhinoCrv.pointAt(u);
    worst = Math.max(worst, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
  }
  return worst;
}

function worstSurfaceDeviation(ours, rhinoSrf, samples = 8) {
  const uMin = ours.knotsU[ours.degU], uMax = ours.knotsU[ours.knotsU.length - 1 - ours.degU];
  const vMin = ours.knotsV[ours.degV], vMax = ours.knotsV[ours.knotsV.length - 1 - ours.degV];
  let worst = 0;
  for (let i = 0; i <= samples; i++) {
    for (let j = 0; j <= samples; j++) {
      const u = uMin + (uMax - uMin) * i / samples;
      const v = vMin + (vMax - vMin) * j / samples;
      const a = surfacePoint(ours, u, v);
      const b = rhinoSrf.pointAt(u, v);
      worst = Math.max(worst, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
    }
  }
  return worst;
}

test('ourKnotsToRhino / rhinoKnotsToOurs round-trip exactly', () => {
  const ours = [0, 0, 0, 1, 2, 3, 3, 3];
  const reduced = ourKnotsToRhino(ours);
  assert.deepEqual(reduced, [0, 0, 1, 2, 3, 3]);
  assert.deepEqual(rhinoKnotsToOurs(reduced), ours);
});

test('curveToRhino: a real rational arc matches rhino3dm pointAt in-memory', () => {
  const arc = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 0, Math.PI / 2, 1);
  const nc = curveToRhino(rhino, arc);
  assert.equal(nc.points().count, arc.ctrlPts.length);
  const worst = worstCurveDeviation(arc, nc);
  assert.ok(worst < 1e-9, `worst curve deviation ${worst}`);
});

test('curveToRhino: survives a full binary File3dm write/read round trip', () => {
  const arc = makeArc([3, -2, 5], [1, 0, 0], [0, 1, 0], 12.5, 0, Math.PI / 2, 1);
  const nc = curveToRhino(rhino, arc);
  const doc = new rhino.File3dm();
  doc.objects().addCurve(nc, new rhino.ObjectAttributes());
  const bytes = doc.toByteArray();
  const doc2 = rhino.File3dm.fromByteArray(bytes);
  const geo = doc2.objects().get(0).geometry();
  const worst = worstCurveDeviation(arc, geo);
  assert.ok(worst < 1e-9, `worst curve deviation after binary round trip: ${worst}`);
});

test('curveFromRhino: inverts curveToRhino exactly (round trip our-shape -> rhino -> our-shape)', () => {
  const points = [[0, 0, 0], [10, 4, 0], [16, 12, 6], [8, 20, 2]];
  const arc = globalCurveInterp(points, 3);
  const nc = curveToRhino(rhino, arc);
  const back = curveFromRhino(nc);
  assert.equal(back.degree, arc.degree);
  assert.equal(back.knots.length, arc.knots.length);
  for (let i = 0; i < arc.knots.length; i++) {
    assert.ok(Math.abs(back.knots[i] - arc.knots[i]) < 1e-9, `knot ${i} mismatch`);
  }
  assert.equal(back.ctrlPts.length, arc.ctrlPts.length);
  for (let i = 0; i < arc.ctrlPts.length; i++) {
    for (let k = 0; k < 4; k++) {
      assert.ok(Math.abs(back.ctrlPts[i][k] - arc.ctrlPts[i][k]) < 1e-9, `ctrlPt ${i}[${k}] mismatch`);
    }
  }
  // and the reconstructed curve still evaluates to the same real shape
  const worst = worstCurveDeviation(back, nc);
  assert.ok(worst < 1e-9, `worst deviation of reconstructed curve ${worst}`);
});

test('surfaceToRhino: a real revolved (rational) cylinder matches rhino3dm pointAt in-memory', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[5, 0, 0, 1], [5, 0, 20, 1]] };
  const cyl = revolve(profile, [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
  const ns = surfaceToRhino(rhino, cyl);
  const worst = worstSurfaceDeviation(cyl, ns);
  assert.ok(worst < 1e-9, `worst surface deviation ${worst}`);
});

test('surfaceToRhino: survives a full binary File3dm write/read round trip', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[6, 0, 0, 1], [6, 0, 30, 1]] };
  const cyl = revolve(profile, [1, 2, 0], [0, 0, 1], 0, Math.PI * 2);
  const ns = surfaceToRhino(rhino, cyl);
  const doc = new rhino.File3dm();
  doc.objects().addSurface(ns, new rhino.ObjectAttributes());
  const bytes = doc.toByteArray();
  const doc2 = rhino.File3dm.fromByteArray(bytes);
  const geo = doc2.objects().get(0).geometry();
  const worst = worstSurfaceDeviation(cyl, geo);
  assert.ok(worst < 1e-9, `worst surface deviation after binary round trip: ${worst}`);
});

test('surfaceFromRhino: inverts surfaceToRhino exactly', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[4, 0, 0, 1], [4, 0, 15, 1]] };
  const cyl = revolve(profile, [0, 0, 0], [0, 0, 1], 0, Math.PI * 1.5);
  const ns = surfaceToRhino(rhino, cyl);
  const back = surfaceFromRhino(ns);
  assert.equal(back.degU, cyl.degU);
  assert.equal(back.degV, cyl.degV);
  assert.equal(back.knotsU.length, cyl.knotsU.length);
  assert.equal(back.knotsV.length, cyl.knotsV.length);
  assert.equal(back.ctrlNet.length, cyl.ctrlNet.length);
  assert.equal(back.ctrlNet[0].length, cyl.ctrlNet[0].length);
  const worst = worstSurfaceDeviation(back, ns);
  assert.ok(worst < 1e-9, `worst deviation of reconstructed surface ${worst}`);
});

test('curveToRhino: a non-rational degree-1 curve (all weights 1) still round-trips', () => {
  const line = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [10, 10, 10, 1]] };
  const nc = curveToRhino(rhino, line);
  const worst = worstCurveDeviation(line, nc);
  assert.ok(worst < 1e-9, `worst deviation ${worst}`);
  const back = curveFromRhino(nc);
  assert.deepEqual(back.ctrlPts.map(p => p.map(x => Math.round(x * 1e6) / 1e6)),
    line.ctrlPts.map(p => p.map(x => Math.round(x * 1e6) / 1e6)));
});

// ---- document-level export/import ----

test('exportDocument + importDocument: a real multi-object document round-trips through a genuine .3dm', () => {
  const arc = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 8, 0, Math.PI / 2, 1);
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[4, 0, 0, 1], [4, 0, 12, 1]] };
  const cyl = revolve(profile, [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);

  const payload = {
    tolerance: 0.001,
    layers: [
      { id: 0, name: 'Default', color: { r: 140, g: 141, b: 144 }, parentId: null },
      { id: 1, name: 'Curves', color: { r: 60, g: 120, b: 220 }, parentId: 0 },
    ],
    objects: [
      { kind: 'point', layerId: 0, name: 'Point01', point: [3, 4, 5] },
      { kind: 'curve', layerId: 1, name: 'Arc01', ...arc },
      { kind: 'surface', layerId: 0, name: 'Cyl01', ...cyl },
    ],
  };

  const { bytes, skipped: exportSkipped } = exportDocument(rhino, payload);
  assert.equal(exportSkipped.length, 0);
  assert.ok(bytes.length > 0);

  const result = importDocument(rhino, bytes);
  assert.ok(Math.abs(result.tolerance - 0.001) < 1e-9);
  assert.equal(result.layers.length, 2);
  const defaultLayer = result.layers.find(l => l.name === 'Default');
  const curvesLayer = result.layers.find(l => l.name === 'Curves');
  assert.ok(defaultLayer);
  assert.ok(curvesLayer);
  assert.equal(curvesLayer.parentId, defaultLayer.id);

  assert.equal(result.objects.length, 3);
  const pt = result.objects.find(o => o.kind === 'point');
  assert.deepEqual(pt.point.map(x => Math.round(x * 1e6) / 1e6), [3, 4, 5]);
  assert.equal(pt.layerId, defaultLayer.id);

  const crv = result.objects.find(o => o.kind === 'curve');
  assert.equal(crv.layerId, curvesLayer.id);
  assert.equal(crv.ctrlPts.length, arc.ctrlPts.length);

  const srf = result.objects.find(o => o.kind === 'surface');
  assert.equal(srf.ctrlNet.length, cyl.ctrlNet.length);
  assert.equal(srf.ctrlNet[0].length, cyl.ctrlNet[0].length);

  assert.equal(result.skipped.length, 0);
});

test('exportDocument: a multi-panel container (N separate surface entries) round-trips as N objects, no special-casing needed', () => {
  const panelA = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [1, 0, 0, 1]] };
  const panelASrf = {
    degU: 1, degV: 1, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[[0, 0, 0, 1], [0, 1, 0, 1]], [[1, 0, 0, 1], [1, 1, 0, 1]]],
  };
  const panelBSrf = {
    degU: 1, degV: 1, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[[1, 0, 0, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 1, 1]]],
  };
  const payload = {
    tolerance: 0.001,
    layers: [{ id: 0, name: 'Default', color: { r: 140, g: 141, b: 144 }, parentId: null }],
    objects: [
      { kind: 'surface', layerId: 0, name: 'RuledLoft04 panel 1', ...panelASrf },
      { kind: 'surface', layerId: 0, name: 'RuledLoft04 panel 2', ...panelBSrf },
    ],
  };
  const { bytes, skipped } = exportDocument(rhino, payload);
  assert.equal(skipped.length, 0);
  const result = importDocument(rhino, bytes);
  assert.equal(result.objects.length, 2);
  assert.ok(result.objects.every(o => o.kind === 'surface'));
  assert.ok(result.objects.some(o => o.name === 'RuledLoft04 panel 1'));
  assert.ok(result.objects.some(o => o.name === 'RuledLoft04 panel 2'));
});

test('importDocument: a genuine multi-face Brep imports as one untrimmed surface panel per face, honestly named, never silently dropped', () => {
  const doc = new rhino.File3dm();
  const bbox = new rhino.BoundingBox([0, 0, 0], [10, 10, 10]);
  const brep = rhino.Brep.createFromBoundingBox(bbox);
  assert.ok(brep, 'expected a real 6-face box brep from rhino3dm itself');
  assert.equal(brep.isSurface, false, 'a 6-face box brep is not a single trivially-trimmed face');
  const attrs = new rhino.ObjectAttributes();
  attrs.name = 'Box01';
  doc.objects().addBrep(brep, attrs);
  const bytes = doc.toByteArray();

  const result = importDocument(rhino, bytes);
  assert.equal(result.skipped.length, 0);
  assert.equal(result.objects.length, 6);
  assert.ok(result.objects.every(o => o.kind === 'surface'));
  assert.ok(result.objects.some(o => o.name === 'Box01 face 1'));
  assert.ok(result.objects.some(o => o.name === 'Box01 face 6'));
});

test('importDocument: a single-face (trivially-trimmed) Brep imports as a plain editable surface, not skipped', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[4, 0, 0, 1], [4, 0, 10, 1]] };
  const cyl = revolve(profile, [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
  const ns = surfaceToRhino(rhino, cyl);
  const brep = rhino.Brep.createFromSurface(ns);
  assert.ok(brep);
  const doc = new rhino.File3dm();
  doc.objects().addBrep(brep, new rhino.ObjectAttributes());
  const bytes = doc.toByteArray();
  const result = importDocument(rhino, bytes);
  assert.equal(result.skipped.length, 0);
  assert.equal(result.objects.length, 1);
  assert.equal(result.objects[0].kind, 'surface');
});

// SUBD IMPORT (the import half of native-SubD interchange).
// The fixture is a genuinely Rhino-authored SubD — it has to be, because
// nothing in this toolchain can author one: rhino3dm's JS binding exposes
// SubD as an opaque handle with no constructor and no cage accessors, in
// 8.17.0 and 8.32.0 alike. That is exactly why this test reads a checked-in
// .3dm rather than round-tripping through a synthetic one.
test('importDocument: a real Rhino SubD arrives as its own welded control net, not a refined mesh', () => {
  const bytes = new Uint8Array(readFileSync(new URL('./fixtures/rhino_subd.3dm', import.meta.url)));
  const result = importDocument(rhino, bytes);
  assert.equal(result.skipped.length, 0, 'a SubD must not be skipped anymore');
  assert.equal(result.objects.length, 1);
  const o = result.objects[0];
  assert.equal(o.kind, 'subd');
  const { vertices, faces } = o.cage;

  // WELDED, not per-face. createFromSubDControlNet's second argument
  // decides this and its own .d.ts declares the function taking no
  // arguments at all; getting it wrong returns 4 vertices per quad, which
  // still LOOKS like a cage and subdivides into confetti because no two
  // faces share a vertex. The discriminating check is not the count but
  // the sharing: a closed cage must have far fewer vertices than 4x its
  // face count, and every vertex must be used by more than one face.
  assert.ok(vertices.length < faces.length * 4 * 0.5, `expected a welded net, got ${vertices.length} vertices for ${faces.length} faces`);
  const uses = new Array(vertices.length).fill(0);
  for (const f of faces) for (const v of f) uses[v]++;
  assert.ok(uses.every((n) => n >= 2), 'every cage vertex must be shared by at least two faces in a welded net');

  for (const f of faces) {
    assert.ok(f.length === 3 || f.length === 4, `cage faces must be tris or quads, got ${f.length}`);
    assert.equal(new Set(f).size, f.length, 'a cage face must not repeat a vertex');
    for (const v of f) assert.ok(Number.isInteger(v) && v >= 0 && v < vertices.length, 'face index in range');
  }
  for (const v of vertices) assert.ok(v.length === 3 && v.every(Number.isFinite));
  assert.equal(o.creasesLost, true, 'the crease loss must be reported, not silent');
});

test('importDocument: the imported SubD cage is a genuine manifold that really subdivides', () => {
  const bytes = new Uint8Array(readFileSync(new URL('./fixtures/rhino_subd.3dm', import.meta.url)));
  const { cage } = importDocument(rhino, bytes).objects[0];

  // The real test of a cage is not that it parses — it is that this app's
  // own Catmull-Clark machinery accepts it. Every edge shared by exactly
  // 2 faces (closed and manifold), and a refinement pass that quadruples
  // the face count without producing anything degenerate.
  const ctx = buildTopology(cage);
  for (const e of ctx.edgeMap.values()) assert.equal(e.faces.length, 2, 'every edge of a closed cage has exactly 2 faces');
  const V = cage.vertices.length, E = ctx.edgeMap.size, F = cage.faces.length;
  assert.equal(V - E + F, 2, `expected a closed genus-0 cage, got chi = ${V - E + F}`);

  const refined = subdivideCatmullClark(cage);
  assert.equal(refined.faces.length, cage.faces.reduce((n, f) => n + f.length, 0));
  for (const f of refined.faces) assert.equal(new Set(f).size, f.length);
  for (const v of refined.vertices) assert.ok(v.every(Number.isFinite));
});
