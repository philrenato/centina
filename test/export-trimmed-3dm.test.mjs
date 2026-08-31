// EXPORTING A TRIMMED FACE AS A REAL TRIMMED BREP — the delivery end of the
// write path. Everything upstream of this (fitted curves, pcurves, authoring
// bindings) was reachable only from tests; this is what an Export .3dm
// actually produces.
//
// Until the authoring bindings existed the app REFUSED a trimmed face by name,
// because writing the uncut base surface would have been a plausible-looking
// wrong answer — and the reason to open a boolean in Rhino is to check whether
// it is right. That refusal was correct then and is the wrong answer now.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { exportDocument, trimmedSurfaceToRhinoBrep } from '../io3dm.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const VENDOR = path.join(HERE, '..', 'vendor', 'rhino3dm');
const rhino = await require(path.join(VENDOR, 'rhino3dm.cjs'))({ locateFile: (p) => path.join(VENDOR, p) });

// A bilinear plane over [-10,10]^2 in this kernel's own representation, so
// (u,v) IS (x,y) and a trim loop's parameters are readable by eye.
const PLANE = {
  degU: 1, degV: 1,
  knotsU: [-10, -10, 10, 10], knotsV: [-10, -10, 10, 10],
  ctrlNet: [[[-10, -10, 0, 1], [-10, 10, 0, 1]], [[10, -10, 0, 1], [10, 10, 0, 1]]],
};
const ring = (r, n, cx = 0, cy = 0) => Array.from({ length: n }, (_, i) => {
  const t = (i / n) * Math.PI * 2;
  return [cx + r * Math.cos(t), cy + r * Math.sin(t)];
});

test('THE FIXTURE IS SANE: the plane maps (u,v) straight onto (x,y)', () => {
  // Everything below reads trim parameters as if they were coordinates, which
  // is only true if this holds. Asserted against the SURFACE's own domain and
  // an evaluated point — not against ON_Brep's bounding box, whose looseness
  // is its own business and was measured at [-6, 6] for a radius-5 trim on a
  // 20x20 plane, matching neither the trim nor the surface.
  const built = trimmedSurfaceToRhinoBrep(rhino, { ...PLANE, trimLoop: ring(5, 64), trimHoles: [] }, 0.001);
  assert.ok(built && built.brep, `refused: ${built && built.log}`);
  const srf = built.brep.faces().get(0).underlyingSurface();
  assert.deepEqual(srf.domain(0), [-10, 10]);
  assert.deepEqual(srf.domain(1), [-10, 10]);
  const p = srf.pointAt(3, -4);
  assert.ok(Math.hypot(p[0] - 3, p[1] + 4, p[2]) < 1e-9,
    `(u,v)=(3,-4) must evaluate to (3,-4,0), got ${JSON.stringify(p)}`);
});

test('A TRIMMED FACE BECOMES A VALID BREP, judged by OpenNURBS', () => {
  const built = trimmedSurfaceToRhinoBrep(rhino, { ...PLANE, trimLoop: ring(5, 64), trimHoles: [] }, 0.001);
  assert.ok(built.brep, `refused: ${built.log}`);
  assert.equal(built.log, '', 'valid with an empty log');
  assert.equal(built.brep.faces().count, 1);
  assert.equal(built.brep.faces().get(0).loops.count, 1);
});

test('AND A FACE WITH A HOLE KEEPS BOTH LOOPS', () => {
  const built = trimmedSurfaceToRhinoBrep(rhino, {
    ...PLANE, trimLoop: ring(8, 64), trimHoles: [ring(3, 48).reverse()],
  }, 0.001);
  assert.ok(built.brep, `refused: ${built.log}`);
  assert.equal(built.brep.faces().get(0).loops.count, 2, 'outer plus hole');
});

test('THE WHOLE DOCUMENT EXPORT CARRIES IT — through exportDocument, not the helper', () => {
  // The helper working proves the conversion; this proves the PAYLOAD KIND is
  // wired, which is the part an Export button depends on.
  const { bytes, skipped } = exportDocument(rhino, {
    tolerance: 0.001,
    layers: [{ id: 0, name: 'Default', color: { r: 0, g: 0, b: 0 }, parentId: null }],
    objects: [{ kind: 'trimmedsurface', layerId: 0, name: 'disc', ...PLANE, trimLoop: ring(5, 64), trimHoles: [] }],
  });
  assert.equal(skipped.length, 0, `nothing may be skipped: ${JSON.stringify(skipped)}`);
  const back = rhino.File3dm.fromByteArray(bytes);
  assert.equal(back.objects().count, 1);
  const geo = back.objects().get(0).geometry();
  assert.equal(geo.constructor.name, 'Brep', 'it must arrive as a Brep, not a Surface');
  assert.equal(geo.faces().get(0).loops.count, 1);
  const [valid] = geo.isValidWithLog;
  assert.equal(valid, true, 'and still valid after the round trip');
});

test('AN UNBUILDABLE TRIM IS NAMED, NOT SILENTLY DROPPED', () => {
  // Two points cannot bound a region. The export must say so by name — a
  // silently missing face is the failure this whole path exists to avoid.
  const { skipped } = exportDocument(rhino, {
    tolerance: 0.001, layers: [],
    objects: [{ kind: 'trimmedsurface', layerId: null, name: 'degenerate', ...PLANE, trimLoop: [[0, 0], [1, 1]], trimHoles: [] }],
  });
  assert.equal(skipped.length, 1);
  assert.match(skipped[0].kind, /trimmed surface/);
});
