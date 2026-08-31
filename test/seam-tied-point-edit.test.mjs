// A CONTROL-POINT EDIT MUST NOT DAMAGE A CLOSED SURFACE'S SEAM.
//
// Reported against a degree-3 torus: dragging control points behaves
// differently from Rhino, which stays smooth. A closed direction is stored
// CLAMPED, its first and last control column holding the same point with
// nothing structural tying the two ends, so writing one column alone both
// TEARS the seam open and CREASES it.
//
// surfaceSeamBandGroup names the points a drag has to carry with it. The app
// adds them to the drag's own target list, beside the pole rows and joint
// vertices already expanded there, so each arrives at applyPointEdits as its
// own edit carrying the same delta — which is why the checks below build
// their edit lists that way rather than expecting applyPointEdits to spread
// one edit itself. Spreading it there TOO would move the band once per member
// and lift the seam by several times what was dragged.
//
// Every seam check is stated against the SAME surface's own untouched seam
// rather than an absolute threshold, the way the seam-crease work already
// measures: an edit is correct when the seam is no worse than it was before.
import test from 'node:test';
import assert from 'node:assert/strict';
import { applyPointEdits } from '../kernel/pointedit.mjs';
import { surfaceSeamBandGroup, surfaceStructuralGroup, surfaceClosure, surfacePoint, surfacePointAndPartials } from '../kernel/surface.mjs';
import { refitSurfaceUV } from '../kernel/loft.mjs';
import { revolve } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';

// A real torus, not a primitive stand-in: closed in BOTH directions, which is
// the case with its own failure mode (one seam fixed at the other's cost).
function torusSurface(R = 50, r = 15, n = 24) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    pts.push([R + r * Math.cos(a), 0, r * Math.sin(a)]);
  }
  pts.push([...pts[0]]);
  return revolve(globalCurveInterp(pts, 3), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
}
// The editable form the app actually hands a drag — a uniform degree-3 net,
// which is what "degree 3 torus" in the report means.
const editableTorus = () => refitSurfaceUV(torusSurface(), 34, 18, 3, 3);

const angleBetween = (a, b) => {
  const na = Math.hypot(...a), nb = Math.hypot(...b);
  const d = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
  return (Math.acos(Math.max(-1, Math.min(1, d))) * 180) / Math.PI;
};
const domain = (s) => ({
  u0: s.knotsU[0], u1: s.knotsU[s.knotsU.length - 1],
  v0: s.knotsV[0], v1: s.knotsV[s.knotsV.length - 1],
});
// Tangent turn ACROSS the seam, sampled just inside each side. Sampling the
// seam parameters themselves would compare a boundary derivative with itself.
function seamTurnV(s, u) {
  const { v0, v1 } = domain(s), eps = (v1 - v0) * 1e-5;
  return angleBetween(surfacePointAndPartials(s, u, v0 + eps).sv, surfacePointAndPartials(s, u, v1 - eps).sv);
}
function seamTurnU(s, v) {
  const { u0, u1 } = domain(s), eps = (u1 - u0) * 1e-5;
  return angleBetween(surfacePointAndPartials(s, u0 + eps, v).su, surfacePointAndPartials(s, u1 - eps, v).su);
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const seamGapV = (s, u) => { const { v0, v1 } = domain(s); return dist(surfacePoint(s, u, v0), surfacePoint(s, u, v1)); };
const seamGapU = (s, v) => { const { u0, u1 } = domain(s); return dist(surfacePoint(s, u0, v), surfacePoint(s, u1, v)); };
// How far the surface actually moved — without this, a rule that simply
// declined to apply the edit would pass every seam check above.
function peakDisplacement(a, b) {
  const { u0, u1, v0, v1 } = domain(a);
  let m = 0;
  for (let x = 0; x <= 30; x++) for (let y = 0; y <= 30; y++) {
    const u = u0 + ((u1 - u0) * x) / 30, v = v0 + ((v1 - v0) * y) / 30;
    m = Math.max(m, dist(surfacePoint(a, u, v), surfacePoint(b, u, v)));
  }
  return m;
}
// EXACTLY WHAT THE APP'S EXPANSION PRODUCES: the grabbed control point plus
// every point that has to travel with it, each as its own edit carrying the
// identical delta. Duplicates are dropped the same way expandWithSeamSiblings
// drops them, so a point named by BOTH groups is still edited once.
function draggedEdits(srf, i0, j0, delta) {
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  const seen = new Set();
  const edits = [];
  const add = (i, j) => {
    const k = `${i}|${j}`;
    if (seen.has(k)) return;
    seen.add(k);
    edits.push({ rowFrac: nu > 1 ? i / (nu - 1) : 0, colFrac: nv > 1 ? j / (nv - 1) : 0, delta: [...delta] });
  };
  add(i0, j0);
  for (const s of surfaceStructuralGroup(srf, i0, j0)) add(s.i, s.j);
  for (const s of surfaceSeamBandGroup(srf, i0, j0)) add(s.i, s.j);
  return edits;
}

test('the editable torus is closed in both directions, and its untouched seams are smooth', () => {
  const s = editableTorus();
  const c = surfaceClosure(s);
  assert.equal(c.closedU, true);
  assert.equal(c.closedV, true);
  const { u0, u1, v0, v1 } = domain(s);
  assert.ok(seamGapV(s, (u0 + u1) / 2) < 1e-9);
  assert.ok(seamGapU(s, (v0 + v1) / 2) < 1e-9);
  assert.ok(seamTurnV(s, (u0 + u1) / 2) < 0.05, 'v seam starts smooth');
  assert.ok(seamTurnU(s, (v0 + v1) / 2) < 0.05, 'u seam starts smooth');
});

test('a drag at EVERY column leaves both seams no worse than untouched, and still moves the surface', () => {
  const s = editableTorus();
  const nu = s.ctrlNet.length, nv = s.ctrlNet[0].length;
  const { u0, u1, v0, v1 } = domain(s);
  const u = (u0 + u1) / 2, v = (v0 + v1) / 2;
  const baseTurnV = seamTurnV(s, u), baseTurnU = seamTurnU(s, v);
  for (let j = 0; j < nv; j++) {
    const out = applyPointEdits(s, draggedEdits(s, Math.floor(nu / 2), j, [0, 0, 12]));
    assert.ok(seamGapV(out, u) < 1e-9, `column ${j} tore the v seam: ${seamGapV(out, u)}`);
    assert.ok(seamGapU(out, v) < 1e-9, `column ${j} tore the u seam: ${seamGapU(out, v)}`);
    assert.ok(seamTurnV(out, u) <= baseTurnV + 1e-6, `column ${j} creased the v seam: ${seamTurnV(out, u)} vs ${baseTurnV}`);
    assert.ok(seamTurnU(out, v) <= baseTurnU + 1e-6, `column ${j} creased the u seam: ${seamTurnU(out, v)} vs ${baseTurnU}`);
    assert.ok(peakDisplacement(s, out) > 1, `column ${j} did not actually move the surface`);
  }
});

test('a drag at every ROW does the same — the rule is not lopsided by direction', () => {
  const s = editableTorus();
  const nu = s.ctrlNet.length, nv = s.ctrlNet[0].length;
  const { u0, u1, v0, v1 } = domain(s);
  const u = (u0 + u1) / 2, v = (v0 + v1) / 2;
  const baseTurnV = seamTurnV(s, u), baseTurnU = seamTurnU(s, v);
  for (let i = 0; i < nu; i++) {
    const out = applyPointEdits(s, draggedEdits(s, i, Math.floor(nv / 2), [0, 0, 12]));
    assert.ok(seamGapV(out, u) < 1e-9, `row ${i} tore the v seam`);
    assert.ok(seamGapU(out, v) < 1e-9, `row ${i} tore the u seam`);
    assert.ok(seamTurnV(out, u) <= baseTurnV + 1e-6, `row ${i} creased the v seam`);
    assert.ok(seamTurnU(out, v) <= baseTurnU + 1e-6, `row ${i} creased the u seam`);
    assert.ok(peakDisplacement(s, out) > 1, `row ${i} did not actually move the surface`);
  }
});

// THE DRAG MUST STAY THE SIZE IT WAS ASKED FOR. This is the check that fails
// if the band is ever expanded in two places at once: every member would carry
// the delta into the whole band again and the seam would rise by a multiple of
// what was dragged. The grabbed point lands exactly one delta from where it
// started, and nothing on the surface moves further than that.
test('the band moves by exactly ONE delta, never a multiple of it', () => {
  const s = editableTorus();
  const nu = s.ctrlNet.length, nv = s.ctrlNet[0].length;
  const i = Math.floor(nu / 2);
  for (const j of [0, 1, 2, nv - 2, nv - 1]) {
    const out = applyPointEdits(s, draggedEdits(s, i, j, [0, 0, 12]));
    assert.ok(Math.abs(out.ctrlNet[i][j][2] - s.ctrlNet[i][j][2] - 12) < 1e-9,
      `the grabbed point at column ${j} moved by ${out.ctrlNet[i][j][2] - s.ctrlNet[i][j][2]}, not 12`);
    let worst = 0;
    for (let a = 0; a < nu; a++) for (let b = 0; b < nv; b++) worst = Math.max(worst, dist(s.ctrlNet[a][b], out.ctrlNet[a][b]));
    assert.ok(worst <= 12 + 1e-9, `some control point moved ${worst}, more than the 12 that was dragged`);
  }
});

// THE NEGATIVE CONTROL. Without it the checks above prove only that some rule
// fired, not that it was needed: this is the same net edited the old way, one
// slot alone, and it must genuinely fail the very assertions the fix passes.
test('NEGATIVE CONTROL: writing the seam column alone really does tear and crease it', () => {
  const s = editableTorus();
  const nu = s.ctrlNet.length, nv = s.ctrlNet[0].length;
  const { u0, u1 } = domain(s);
  const u = (u0 + u1) / 2;
  const i = Math.floor(nu / 2);
  const lone = (j) => applyPointEdits(s, [{ rowFrac: i / (nu - 1), colFrac: j / (nv - 1), delta: [0, 0, 12] }]);
  const seam = lone(0);
  assert.ok(seamGapV(seam, u) > 1, `the untied seam column must open a real gap, got ${seamGapV(seam, u)}`);
  assert.ok(seamTurnV(seam, u) > 10, `the untied seam column must break the tangent, got ${seamTurnV(seam, u)}`);
  // And the column NEXT to the seam tears nothing yet still creases — which is
  // why carrying only the coincident pair would not have been enough.
  const next = lone(1);
  assert.ok(seamGapV(next, u) < 1e-9, 'the column next to the seam does not tear');
  assert.ok(seamTurnV(next, u) > 10, `it must still crease, got ${seamTurnV(next, u)}`);
});

test('an OPEN surface carries nothing — every control point stays independent', () => {
  const open = refitSurfaceUV(revolve(globalCurveInterp([[10, 0, 0], [14, 0, 8], [9, 0, 18], [16, 0, 30]], 3),
    [0, 0, 0], [0, 0, 1], 0, Math.PI / 2), 8, 8, 3, 3);
  const c = surfaceClosure(open);
  assert.equal(c.closedU, false);
  assert.equal(c.closedV, false);
  for (const [i, j] of [[0, 0], [0, 3], [4, 4], [7, 7]]) {
    assert.deepEqual(surfaceSeamBandGroup(open, i, j), []);
  }
  const out = applyPointEdits(open, [{ rowFrac: 0, colFrac: 0, delta: [1, 2, 3] }]);
  let moved = 0;
  for (let i = 0; i < open.ctrlNet.length; i++) for (let j = 0; j < open.ctrlNet[0].length; j++) {
    if (dist(open.ctrlNet[i][j], out.ctrlNet[i][j]) > 1e-12) moved++;
  }
  assert.equal(moved, 1);
});

test('surfaceSeamBandGroup names the degree-wide band at each end, and only there', () => {
  const s = editableTorus();
  const nu = s.ctrlNet.length, nv = s.ctrlNet[0].length;
  const mid = Math.floor(nv / 2), midRow = Math.floor(nu / 2);
  // an interior point in both directions carries nothing
  assert.deepEqual(surfaceSeamBandGroup(s, midRow, mid), []);
  // a point in the v band carries that band across, at its own row
  const cols = surfaceSeamBandGroup(s, midRow, 0).map((p) => p.j).sort((a, b) => a - b);
  assert.deepEqual(cols, [1, 2, nv - 3, nv - 2, nv - 1]);
  assert.ok(surfaceSeamBandGroup(s, midRow, 0).every((p) => p.i === midRow), 'an interior row stays one row');
  // a point in BOTH bands carries the full cross-band — a torus corner
  assert.equal(surfaceSeamBandGroup(s, 0, 0).length, 6 * 6 - 1);
  // and it never names the point it was asked about
  for (const [i, j] of [[0, 0], [1, 5], [nu - 1, nv - 1], [4, nv - 2]]) {
    assert.ok(!surfaceSeamBandGroup(s, i, j).some((p) => p.i === i && p.j === j));
  }
});

// The band and the structural siblings OVERLAP — the coincident seam pair is
// in both. The app drops the duplicate; this pins that there is one to drop,
// so the overlap is a known fact rather than something discovered later.
test('the coincident seam pair is named by BOTH groups, so the app must dedupe', () => {
  const s = editableTorus();
  const nv = s.ctrlNet[0].length, i = Math.floor(s.ctrlNet.length / 2);
  assert.ok(surfaceStructuralGroup(s, i, 0).some((p) => p.i === i && p.j === nv - 1));
  assert.ok(surfaceSeamBandGroup(s, i, 0).some((p) => p.i === i && p.j === nv - 1));
});

test('a net too small to hold two disjoint bands moves as a whole rather than overlapping', () => {
  // 4 columns at degree 3: the band would otherwise be wider than the net.
  const tiny = { degU: 1, degV: 3, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 0, 0, 1, 1, 1, 1],
    ctrlNet: [[[0, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 1]],
              [[0, 0, 5, 1], [1, 0, 5, 1], [1, 1, 5, 1], [0, 0, 5, 1]]] };
  assert.equal(surfaceClosure(tiny).closedV, true);
  const cols = surfaceSeamBandGroup(tiny, 0, 0).map((p) => p.j).sort((a, b) => a - b);
  assert.deepEqual(cols, [1, 2, 3], 'every other column, each listed once');
});
