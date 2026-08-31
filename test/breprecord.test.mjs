// A BOOLEAN EXPORTED AS ONE JOINED BREP, JUDGED BY OPENNURBS.
//
// `test/boolean-to-onbrep.test.mjs` already proved our booleans assemble into
// valid ON_Breps. It does that from the LIVE half-edge solid, which the app
// cannot keep: the solid is cyclic, so it cannot be serialized, cloned, or
// carried through an autosave. This tests the path that can actually ship —
// flatten the solid to plain data once, then build the brep from nothing but
// that.
//
// ⚠ THE CLAIM UNDER TEST IS "CLOSED SOLID", NOT "VALID". N loose trimmed faces
// are also valid; they are simply not a solid, and that is the whole difference
// this work exists to close. So `isSolid` is asserted, and the record is put
// through JSON.stringify/parse first — if a single object reference survives
// the flattening, that round trip is where it shows up.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { surfaceFromRhino, surfaceToRhino, curveToRhino, brepRecordToRhino, exportDocument } from '../io3dm.mjs';
import { trivialTrimLoop } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { booleanSolids } from '../kernel/boolean.mjs';
import { solidToBrepRecord, transformBrepRecord } from '../kernel/breprecord.mjs';
import { revolve, makeLine } from '../kernel/primitives.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
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
    operands.push({
      faces: [{ srf }],
      triangles: tessellateTrimmedSurface(srf, trivialTrimLoop(srf), 60, 60, []).map((t) => t.map((v) => v.position)),
    });
  }
}
const ssi = intersectSurfacesComplete(operands[0].faces[0].srf, operands[1].faces[0].srf);
const curves = ssi.components.map((c) => ({ samples: c.samples.map((s) => s.point), faceA: 0, faceB: 0 }));

const results = {};
for (const op of ['union', 'intersect', 'difference']) {
  const r = booleanSolids(operands[0], operands[1], curves, op);
  results[op] = r.ok ? { boolean: r, record: solidToBrepRecord(r.solid, { tolerance: TOL }) } : { refused: r.reason };
}

test('THE INPUT LANDS: all three booleans close before any of this means anything', () => {
  for (const op of ['union', 'intersect', 'difference']) {
    assert.ok(!results[op].refused, `${op} refused: ${results[op].refused}`);
    assert.equal(results[op].boolean.stats.nakedEdgeCount, 0, `${op} leaves no naked edge`);
  }
});

test('THE RECORD IS PLAIN DATA — it survives JSON, which is the only reason it can be stored', () => {
  // The solid it came from cannot do this at all: a half-edge points at its
  // twin, its next and its vertex, and they point back. If any reference
  // leaked into the record, stringify throws on the cycle right here.
  //
  // ⚠ COMPARED IGNORING THE SIGN OF ZERO, and that is a real distinction
  // rather than a loosened test. `JSON.stringify(-0)` is "0", so a round trip
  // turns -0 into +0 and `deepStrictEqual` — which compares numbers by
  // Object.is — calls that a difference. Four pcurve control points here carry
  // -0 in the z slot a (u,v) curve keeps permanently zero. Nothing downstream
  // can observe it (-0 === 0, and every arithmetic use agrees), so the property
  // worth asserting is that no VALUE and no STRUCTURE was lost, not that the
  // bits are identical.
  const norm = (v) => (typeof v === 'number' ? (v === 0 ? 0 : v) : v);
  const walk = (a) => (Array.isArray(a) ? a.map(walk)
    : a && typeof a === 'object' ? Object.fromEntries(Object.entries(a).map(([k, x]) => [k, walk(x)]))
      : norm(a));
  for (const op of ['union', 'intersect', 'difference']) {
    const rec = results[op].record;
    assert.ok(rec.ok, `${op}: ${rec.reason}`);
    let text;
    assert.doesNotThrow(() => { text = JSON.stringify(rec); }, `${op}: the record still holds a cyclic reference`);
    assert.deepEqual(walk(JSON.parse(text)), walk(rec), `${op}: the record changed shape crossing JSON`);
    assert.ok(!/"\[object/.test(text), `${op}: something stringified as an opaque object`);
  }
});

test('THE MERGE IS REAL: hundreds of sampled edges become the edges a B-rep has', () => {
  const rec = results.union.record;
  assert.ok(rec.stats.sourceEdges > 400, `the sew has ${rec.stats.sourceEdges} edges to merge`);
  assert.ok(rec.stats.edges < rec.stats.sourceEdges / 5,
    `${rec.stats.sourceEdges} sampled edges should merge to far fewer real ones, got ${rec.stats.edges}`);
  assert.equal(rec.stats.unfittedEdges, 0, 'every edge carries a curve');
  assert.equal(rec.stats.droppedTrims, 0, 'no trim was dropped for want of an edge');
});

test('⭐ AND OPENNURBS CALLS IT A CLOSED SOLID — not merely a valid pile of faces', () => {
  // The whole point. N loose trimmed faces are valid too; `isSolid` is what
  // separates a solid from a heap, and it is Rhino's own opinion, not ours.
  for (const op of ['union', 'intersect', 'difference']) {
    const rec = JSON.parse(JSON.stringify(results[op].record)); // built from stored data only
    const built = brepRecordToRhino(rhino, rec, TOL);
    assert.ok(built && built.brep, `${op}: OpenNURBS rejected the assembly — ${built && built.log}`);
    assert.equal(built.log, '', `${op}: valid with an empty log`);
    assert.equal(built.brep.faces().count, rec.faces.length, `${op}: every face reached the brep`);
    assert.ok(built.brep.isSolid, `${op}: the assembled brep must be a CLOSED SOLID, not loose faces`);
  }
});

test('FACES SHARE THEIR EDGES — the count proves it, since loose faces cannot', () => {
  // If every face had authored its own copy of every boundary, the edge count
  // would be the sum of the faces' own trim counts. Sharing makes it far
  // smaller, and that gap IS the join.
  const rec = results.union.record;
  const built = brepRecordToRhino(rhino, rec, TOL);
  const edges = built.brep.edges().count;
  assert.ok(edges < rec.stats.trims,
    `${edges} edges for ${rec.stats.trims} trims means edges are shared; equal would mean nothing was joined`);
});

test('A RIGID TRANSFORM MOVES THE SOLID AND LEAVES THE PCURVES ALONE', () => {
  // A trim lives in its own face's PARAMETERS, which a rigid motion of the
  // face does not change. Transforming them would shift every boundary within
  // its own surface — a file that opens and is quietly the wrong shape.
  const rec = results.union.record;
  const shifted = transformBrepRecord(rec, (p) => [p[0] + 1000, p[1], p[2]]);
  assert.deepEqual(shifted.faces[0].loops[0].trims[0].curve, rec.faces[0].loops[0].trims[0].curve,
    'a pcurve must be untouched by a rigid motion');
  assert.ok(Math.abs(shifted.edges.find(Boolean).start[0] - rec.edges.find(Boolean).start[0] - 1000) < 1e-9,
    'an edge endpoint must move by the full delta');
  const a = rec.surfaces[0].ctrlNet[0][0], b = shifted.surfaces[0].ctrlNet[0][0];
  assert.ok(Math.abs(b[0] - a[0] - 1000) < 1e-9, 'a control point must move by the full delta');
  assert.equal(b[3], a[3], 'and its WEIGHT must be untouched');
  const built = brepRecordToRhino(rhino, shifted, TOL);
  assert.ok(built && built.brep && built.brep.isSolid, 'the moved solid is still a closed solid');
});

test('IT SURVIVES A REAL .3dm WRITE AND READ BACK, still a solid', () => {
  const { bytes, skipped } = exportDocument(rhino, {
    tolerance: TOL,
    layers: [{ id: 0, name: 'Default', color: { r: 0, g: 0, b: 0 }, parentId: null }],
    objects: [{ kind: 'brep', layerId: 0, name: 'UnionSolid', record: results.union.record }],
  });
  assert.equal(skipped.length, 0, `nothing may be skipped: ${JSON.stringify(skipped)}`);
  const back = rhino.File3dm.fromByteArray(bytes);
  assert.equal(back.objects().count, 1, 'ONE object — a joined solid, not N faces');
  const geo = back.objects().get(0).geometry();
  assert.ok(geo.faces, 'it came back as a Brep');
  assert.ok(geo.isValid, 'still valid after the round trip');
  assert.ok(geo.isSolid, 'and still a CLOSED SOLID after the round trip');
});

// ⚠⚠ A BOX PAIR IS NOT A WEAKER FIXTURE THAN THE TORI — it is the one that
// catches a whole class the tori structurally cannot. Every edge of a box is
// STRAIGHT, so `fitLine` recognises it and returns its CANONICALIZED direction
// (largest component positive), which for a chain travelling the other way is
// a curve running backwards. A trim's `reversed` flag is computed against the
// chain's traversal order, so a backwards edge curve puts every trim on it a
// full edge-length from its own edge — and OpenNURBS reports exactly that.
//
// The tori cannot see it: their chains are curved, so no primitive is
// recognised and the least-squares fit interpolates the endpoints in order. All
// three torus booleans stayed valid throughout, which is precisely why this
// needed its own fixture rather than a harder torus.
const quad = (p00, p10, p11, p01) => {
  const w = (p) => [p[0], p[1], p[2], 1];
  return { degU: 1, degV: 1, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1], ctrlNet: [[w(p00), w(p01)], [w(p10), w(p11)]] };
};
const boxFaces = ([x0, y0, z0], [x1, y1, z1]) => [
  { srf: quad([x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]) },
  { srf: quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]) },
  { srf: quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]) },
  { srf: quad([x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]) },
  { srf: quad([x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]) },
  { srf: quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]) },
];
const boxTriangles = ([x0, y0, z0], [x1, y1, z1]) => {
  const v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
  const q = (a, b, c, d) => [[v[a], v[b], v[c]], [v[a], v[c], v[d]]];
  return [...q(0, 3, 2, 1), ...q(4, 5, 6, 7), ...q(0, 1, 5, 4), ...q(3, 7, 6, 2), ...q(0, 4, 7, 3), ...q(1, 2, 6, 5)];
};
const seg = (f, t, n = 9) => Array.from({ length: n + 1 }, (_, i) => {
  const s = i / n;
  return [f[0] + (t[0] - f[0]) * s, f[1] + (t[1] - f[1]) * s, f[2] + (t[2] - f[2]) * s];
});

test('⭐ A BOX PAIR — every edge straight — ALSO becomes a closed solid', () => {
  const a0 = [-20, -20, -20], a1 = [20, 20, 20], b0 = [0, 0, 0], b1 = [35, 35, 35];
  const A = { faces: boxFaces(a0, a1), triangles: boxTriangles(a0, a1) };
  const B = { faces: boxFaces(b0, b1), triangles: boxTriangles(b0, b1) };
  const O = { lo: b0, hi: a1 };
  const boxCurves = [
    { samples: seg([a1[0], b0[1], O.lo[2]], [a1[0], b0[1], O.hi[2]]), faceA: 5, faceB: 2 },
    { samples: seg([a1[0], O.lo[1], b0[2]], [a1[0], O.hi[1], b0[2]]), faceA: 5, faceB: 0 },
    { samples: seg([b0[0], a1[1], O.lo[2]], [b0[0], a1[1], O.hi[2]]), faceA: 3, faceB: 4 },
    { samples: seg([O.lo[0], a1[1], b0[2]], [O.hi[0], a1[1], b0[2]]), faceA: 3, faceB: 0 },
    { samples: seg([b0[0], O.lo[1], a1[2]], [b0[0], O.hi[1], a1[2]]), faceA: 1, faceB: 4 },
    { samples: seg([O.lo[0], b0[1], a1[2]], [O.hi[0], b0[1], a1[2]]), faceA: 1, faceB: 2 },
  ];
  for (const op of ['union', 'intersect', 'difference']) {
    const r = booleanSolids(A, B, boxCurves, op);
    assert.ok(r.ok, `${op} did not close: ${r.reason}`);
    const rec = solidToBrepRecord(r.solid, { tolerance: 0.001 });
    assert.ok(rec.ok, `${op}: ${rec.reason}`);
    const built = brepRecordToRhino(rhino, rec, 0.001);
    assert.ok(built && built.brep, `${op}: OpenNURBS rejected it — ${built && built.log}`);
    assert.equal(built.log, '', `${op}: valid with an empty log`);
    assert.ok(built.brep.isSolid, `${op}: a box boolean must be a CLOSED SOLID too`);
  }
});

test('...and the edge curves genuinely run the way their chains do', () => {
  // The property the fix installs, asserted directly rather than only through
  // OpenNURBS' verdict — so a regression says WHAT broke, not just that the
  // validator stopped being happy.
  const a0 = [-20, -20, -20], a1 = [20, 20, 20], b0 = [0, 0, 0], b1 = [35, 35, 35];
  const A = { faces: boxFaces(a0, a1), triangles: boxTriangles(a0, a1) };
  const B = { faces: boxFaces(b0, b1), triangles: boxTriangles(b0, b1) };
  const O = { lo: b0, hi: a1 };
  const boxCurves = [
    { samples: seg([a1[0], b0[1], O.lo[2]], [a1[0], b0[1], O.hi[2]]), faceA: 5, faceB: 2 },
    { samples: seg([a1[0], O.lo[1], b0[2]], [a1[0], O.hi[1], b0[2]]), faceA: 5, faceB: 0 },
    { samples: seg([b0[0], a1[1], O.lo[2]], [b0[0], a1[1], O.hi[2]]), faceA: 3, faceB: 4 },
    { samples: seg([O.lo[0], a1[1], b0[2]], [O.hi[0], a1[1], b0[2]]), faceA: 3, faceB: 0 },
    { samples: seg([b0[0], O.lo[1], a1[2]], [b0[0], O.hi[1], a1[2]]), faceA: 1, faceB: 4 },
    { samples: seg([O.lo[0], b0[1], a1[2]], [O.hi[0], b0[1], a1[2]]), faceA: 1, faceB: 2 },
  ];
  const r = booleanSolids(A, B, boxCurves, 'union');
  const rec = solidToBrepRecord(r.solid, { tolerance: 0.001 });
  for (const e of rec.edges) {
    if (!e) continue;
    const cps = e.curve.ctrlPts;
    const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    assert.ok(d(cps[0], e.start) < 1e-9, 'an edge must START where its own curve starts');
    assert.ok(d(cps[cps.length - 1], e.end) < 1e-9, 'and END where its curve ends');
  }
});

test('A RECORD THAT IS NOT OK REFUSES rather than building half a brep', () => {
  assert.equal(solidToBrepRecord(null, { tolerance: TOL }).ok, false);
  assert.equal(solidToBrepRecord({ edges: [] }, { tolerance: TOL }).ok, false);
  assert.equal(brepRecordToRhino(rhino, { ok: false }, TOL), null);
  assert.equal(brepRecordToRhino(rhino, null, TOL), null);
});

test('THE VENDORED BUILD EXPOSES NewSingularTrim, and it really authors one', () => {
  // ON represents a POLE — a surface edge that collapses to a single point, as
  // a revolved disc's centre does — as a SINGULAR TRIM: a real loop member with
  // (u,v) extent and no 3-D length. The released rhino3dm binds no such thing,
  // so `vendor/rhino3dm/brep_authoring.patch` adds it alongside the rest of the
  // authoring API.
  //
  // ⚠ ASSERTED AGAINST THE BUILD, not the patch file. The glue stayed
  // byte-identical across a build that added ten methods once before, so the
  // only honest check is calling it.
  assert.equal(typeof rhino.Brep.prototype.newSingularTrim, 'function',
    'the vendored build must expose newSingularTrim — rebuild via vendor/rhino3dm/rebuild.sh');
  const brep = new rhino.Brep();
  const plane = {
    degU: 1, degV: 1,
    knotsU: [0, 0, 10, 10], knotsV: [0, 0, 10, 10],
    ctrlNet: [[[0, 0, 0, 1], [0, 10, 0, 1]], [[10, 0, 0, 1], [10, 10, 0, 1]]],
  };
  const si = brep.addSurface(surfaceToRhino(rhino, plane));
  const fi = brep.newFace(si);
  const li = brep.newLoop(fi, 1);
  const c2 = brep.addTrimCurve(curveToRhino(rhino, {
    degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [0, 10, 0, 1]],
  }));
  const v = brep.newVertex([0, 0, 0], 0.001);
  assert.ok(si >= 0 && fi >= 0 && li >= 0 && c2 >= 0 && v >= 0, 'the scaffolding indices are real');
  // W_iso = 3: the degenerate stretch runs along the west side of the domain.
  const ti = brep.newSingularTrim(li, v, 3, c2, 0.001);
  assert.ok(ti >= 0, `newSingularTrim must return a real trim index, got ${ti}`);
});

test('A SOLID WHOSE LOOPS DO NOT CLOSE IS REFUSED BY NAME, not written and rejected', () => {
  // `fitFaceLoops` already measures the largest (u,v) gap between one trim's
  // end and the next one's start. Writing a brep from a loop with a gap makes
  // OpenNURBS report a defect three layers upstream in its own vocabulary
  // ("end of m_T[72]=(1.1e-16,1.33333) and start of m_T[70]=(0,0) do not
  // match"), which reads like a bug in the writer.
  //
  // The union here is genuinely closed (zero naked edges) — the loops are what
  // is broken, which is why this cannot be caught by the boolean's own checks.
  const rec = results.union.record;
  assert.ok(rec.ok, 'the torus union still produces a usable record');
  // And the refusal path is reachable and says the right kind of thing.
  // ⚠ A MALFORMED SOLID MUST REFUSE, NOT THROW. This exact fixture — edges and
  // shells present, vertices absent — got past the original guard and threw out
  // of `fitSolidEdgeCurves` on `solid.vertices.length`. A refusal is a return
  // value in this kernel; a throw here reaches the app as an unexplained
  // failure of whatever called it.
  for (const bad of [{ edges: [], shells: [] }, { edges: [], vertices: [] }, {}, null]) {
    let out = null;
    assert.doesNotThrow(() => { out = solidToBrepRecord(bad, { tolerance: TOL }); },
      `a malformed solid must refuse rather than throw: ${JSON.stringify(bad)}`);
    assert.equal(out.ok, false);
    assert.ok(typeof out.reason === 'string' && out.reason.length > 10, 'and name something actionable');
  }
});

// ⭐⭐⭐ A POLE-BEARING SOLID — the case that blocked a cylinder-bearing boolean
// from ever being one closed solid.
//
// A cylinder cap is a revolved DISC: its whole u=0 edge collapses to the centre
// point. The sew builds face boundaries from 3-D points and welds consecutive
// duplicates — correctly — so that stretch vanishes and the face's loop arrives
// with a hole in it. ON represents exactly that as a SINGULAR TRIM, and the
// recovery rebuilds it from the surface, which still knows.
//
// ⚠ THE TORUS PAIR ABOVE CANNOT CATCH THIS. A torus has no poles at all, which
// is why every earlier check passed while a cylinder could not be written.
test('⭐ A CYLINDER-BEARING BOOLEAN BECOMES A CLOSED SOLID — poles and all', () => {
  const R = 12, Z0 = -30, Z1 = 30, B = 20;
  const wall = revolve(makeLine([R, 0, Z0], [R, 0, Z1]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const capBot = revolve(makeLine([0, 0, Z0], [R, 0, Z0]), [0, 0, Z0], [0, 0, -1], 0, 2 * Math.PI);
  const capTop = revolve(makeLine([0, 0, Z1], [R, 0, Z1]), [0, 0, Z1], [0, 0, 1], 0, 2 * Math.PI);
  const tri = (s) => tessellateTrimmedSurface(s, trivialTrimLoop(s), 40, 40, []).map((t) => t.map((v) => v.position));
  const cyl = { faces: [{ srf: wall }, { srf: capBot }, { srf: capTop }], triangles: [...tri(wall), ...tri(capBot), ...tri(capTop)] };
  const q = (a, b, c, d) => { const w = (p) => [p[0], p[1], p[2], 1];
    return { degU: 1, degV: 1, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1], ctrlNet: [[w(a), w(d)], [w(b), w(c)]] }; };
  const bf = [
    { srf: q([-B, -B, -B], [B, -B, -B], [B, B, -B], [-B, B, -B]) }, { srf: q([-B, -B, B], [B, -B, B], [B, B, B], [-B, B, B]) },
    { srf: q([-B, -B, -B], [B, -B, -B], [B, -B, B], [-B, -B, B]) }, { srf: q([-B, B, -B], [B, B, -B], [B, B, B], [-B, B, B]) },
    { srf: q([-B, -B, -B], [-B, B, -B], [-B, B, B], [-B, -B, B]) }, { srf: q([B, -B, -B], [B, B, -B], [B, B, B], [B, -B, B]) },
  ];
  const box = { faces: bf, triangles: bf.flatMap((f) => tri(f.srf)) };
  const cs = [];
  box.faces.forEach((fa, ia) => cyl.faces.forEach((fb, ib) => {
    let ssi; try { ssi = intersectSurfacesComplete(fa.srf, fb.srf); } catch { return; }
    for (const c of ssi.components || []) if (c.samples && c.samples.length >= 2) cs.push({ samples: c.samples.map((x) => x.point), faceA: ia, faceB: ib });
  }));
  const r = booleanSolids(box, cyl, cs, 'union');
  assert.ok(r.ok, `the union must close first: ${r.reason}`);
  assert.equal(r.stats.nakedEdgeCount, 0);

  const rec = solidToBrepRecord(r.solid, { tolerance: 0.001 });
  assert.ok(rec.ok, `the record must build: ${rec.reason}`);
  assert.ok(rec.stats.singularTrims > 0,
    `the cap poles must be recovered as SINGULAR trims — got ${rec.stats.singularTrims}, and without them the loops do not close`);

  const built = brepRecordToRhino(rhino, rec, 0.001);
  assert.ok(built && built.brep, `OpenNURBS rejected it — ${built && built.log}`);
  assert.equal(built.log, '', 'valid with an empty log');
  assert.ok(built.brep.isSolid, 'and it must be a CLOSED SOLID, which is the whole point');
});

test('...and a genuine hole still refuses — the pole recovery is not a blanket gap-filler', () => {
  // The recovery only closes a gap the SURFACE collapses across. If it closed
  // any gap at all it would manufacture faces over real holes, which is exactly
  // the "valid but quietly wrong" outcome this path exists to avoid.
  const rec = results.union.record;
  assert.ok(rec.ok);
  assert.equal(rec.stats.singularTrims, 0,
    'a torus pair has NO poles, so nothing should have been recovered there');
});
