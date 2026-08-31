import test from 'node:test';
import assert from 'node:assert';
import { sewFragments, fragmentBoundaries3D, keepFragments } from '../kernel/booleansew.mjs';
import { revolve, makeLine } from '../kernel/primitives.mjs';
import { trivialTrimLoop } from '../kernel/trim.mjs';
import { surfacePoint } from '../kernel/surface.mjs';

// A flat bilinear patch through four given corners, parametrized so the unit
// (u,v) square maps onto them in order — degree 1 in both directions, so
// every expected coordinate below is exact and hand-derivable rather than
// sampled off the thing under test.
function quadSurface(p00, p10, p11, p01) {
  const w = (p) => [p[0], p[1], p[2], 1];
  return {
    degU: 1, degV: 1,
    knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[w(p00), w(p01)], [w(p10), w(p11)]],
  };
}

// The full untrimmed domain, walked in the order the corners were given.
const FULL = { outer: [[0, 0], [1, 0], [1, 1], [0, 1]] };

// A 10mm cube as six independent bilinear faces, exactly the shape a boolean
// produces once its fragments are kept: separate surfaces that have never
// met, sharing edges only by coincident position.
function cubeFaces(lo = 0, hi = 10) {
  const c = (x, y, z) => [x, y, z];
  return [
    { srf: quadSurface(c(lo, lo, lo), c(hi, lo, lo), c(hi, hi, lo), c(lo, hi, lo)), ...FULL }, // z = lo
    { srf: quadSurface(c(lo, lo, hi), c(hi, lo, hi), c(hi, hi, hi), c(lo, hi, hi)), ...FULL }, // z = hi
    { srf: quadSurface(c(lo, lo, lo), c(hi, lo, lo), c(hi, lo, hi), c(lo, lo, hi)), ...FULL }, // y = lo
    { srf: quadSurface(c(lo, hi, lo), c(hi, hi, lo), c(hi, hi, hi), c(lo, hi, hi)), ...FULL }, // y = hi
    { srf: quadSurface(c(lo, lo, lo), c(lo, hi, lo), c(lo, hi, hi), c(lo, lo, hi)), ...FULL }, // x = lo
    { srf: quadSurface(c(hi, lo, lo), c(hi, hi, lo), c(hi, hi, hi), c(hi, lo, hi)), ...FULL }, // x = hi
  ];
}

test('six independent trimmed faces sew into one genuinely closed solid', () => {
  const res = sewFragments(cubeFaces());
  assert.ok(res.ok, res.verdict);
  // Euler characteristic of a sphere, from the welder's own count — the
  // check that actually distinguishes a closed solid from a bag of faces.
  assert.equal(res.stats.chi, 2);
  assert.equal(res.stats.V, 8, 'the 24 corner points weld down to the cube\'s real 8');
  assert.equal(res.stats.E, 12);
  assert.equal(res.stats.F, 6);
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.nonManifoldEdgeCount, 0);
  assert.equal(res.stats.shellCount, 1);
  assert.match(res.verdict, /CLOSED SOLID/);
});

test('a missing fragment comes back as an OPEN SHELL by name, never a silent success', () => {
  const faces = cubeFaces();
  faces.pop(); // drop x = hi
  const res = sewFragments(faces);
  assert.equal(res.ok, false);
  assert.equal(res.reason, 'open-shell');
  assert.equal(res.stats.nakedEdgeCount, 4, 'exactly the removed face\'s own four edges');
  assert.match(res.verdict, /OPEN SHELL/);
});

test('THE RESIDUAL CASE: faces that only ALMOST meet still weld, and the gap is reported', () => {
  // Two surfaces agreeing along a shared edge only to within a marcher's own
  // residual is the normal state of affairs across an intersection curve —
  // this is the mechanism that closes the shell there, so it is measured
  // rather than assumed. The offset is well inside the weld tolerance.
  const eps = 2e-6;
  const faces = cubeFaces();
  const nudged = faces[5]; // x = hi
  nudged.srf = quadSurface(
    [10 + eps, 0, 0], [10 + eps, 10, 0], [10 + eps, 10, 10], [10 + eps, 0, 10],
  );
  const res = sewFragments(faces);
  assert.ok(res.ok, res.verdict);
  assert.equal(res.stats.chi, 2, 'still a closed solid despite the mismatch');
  assert.ok(res.worstSharedGap > 0, 'the weld genuinely had a gap to close');
  assert.ok(Math.abs(res.worstSharedGap - eps) < 1e-9, `reports the real gap, got ${res.worstSharedGap}`);
  assert.ok(res.worstSharedGap < res.tolerance, 'and it sat inside the tolerance');
});

test('a residual LARGER than the tolerance leaves a crack, honestly, instead of a wrong solid', () => {
  // The same fixture with the mismatch pushed past the tolerance: the right
  // answer is a refusal naming naked edges, not a shell quietly stitched
  // across a gap it should not have crossed.
  const faces = cubeFaces();
  faces[5].srf = quadSurface(
    [10.01, 0, 0], [10.01, 10, 0], [10.01, 10, 10], [10.01, 0, 10],
  );
  const res = sewFragments(faces, { tolerance: 1e-4 });
  assert.equal(res.ok, false);
  assert.equal(res.reason, 'open-shell');
  assert.ok(res.stats.nakedEdgeCount > 0);
});

// THIS TEST PREVIOUSLY ASSERTED A KEYHOLE MERGE, and that assertion is
// deliberately reversed rather than deleted. A keyhole bridge is correct
// for tessellation and wrong for a B-rep face: it visits its own corners
// twice, and `buildBrepSolid` rejects a repeated vertex in a face loop —
// correctly. So a holed fragment is now SPLIT into two simple faces
// sharing both bridge edges. The invariant worth pinning is the same one
// the old test was reaching for (a hole is handled, not refused), stated
// against what the sew actually needs.
test('a fragment WITH a hole splits into two simple faces rather than one slit loop', () => {
  const srf = quadSurface([0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]);
  const withHole = {
    outer: [[0, 0], [1, 0], [1, 1], [0, 1]],
    holes: [[[0.4, 0.4], [0.4, 0.6], [0.6, 0.6], [0.6, 0.4]]],
  };
  const res = fragmentBoundaries3D(srf, withHole);
  assert.ok(res.ok, res.reason);
  assert.equal(res.loops.length, 2, 'an annulus becomes two faces, not one');

  for (const loop of res.loops) {
    for (const p of loop) assert.ok(Math.abs(p[2]) < 1e-12, 'every point lands on the real surface');
    // The whole point: no face loop visits the same corner twice.
    const keys = loop.map((p) => p.map((c) => Math.round(c / 1e-9)).join('|'));
    assert.equal(new Set(keys).size, keys.length, 'a face loop traverses each of its own corners once');
  }

  // Nothing is lost in the split: every original corner of both loops
  // still appears somewhere across the two faces.
  const all = res.loops.flat();
  const has = (x, y) => all.some((p) => Math.abs(p[0] - x) < 1e-9 && Math.abs(p[1] - y) < 1e-9);
  for (const [u, v] of [[0, 0], [1, 0], [1, 1], [0, 1]]) assert.ok(has(u * 10, v * 10), `outer corner ${u},${v}`);
  for (const [u, v] of [[0.4, 0.4], [0.4, 0.6], [0.6, 0.6], [0.6, 0.4]]) assert.ok(has(u * 10, v * 10), `hole corner ${u},${v}`);
});

test('a fragment whose outer loop collapses in 3D uses its hole as the real boundary', () => {
  // The real pole/seam case, on a real cone rather than a surface that is
  // degenerate everywhere: u=0 is the apex (a whole row collapsed to one
  // point) and the two v-ends are the same seam, so the four corners of the
  // untrimmed domain evaluate to only two distinct places. The cut curve is
  // then the face's only real edge — while the hole itself sits in genuinely
  // non-degenerate interior, which a fixture collapsed everywhere could not
  // have told apart.
  const cone = revolve(makeLine([0, 0, 0], [10, 0, 20]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const corners = trivialTrimLoop(cone).map(([u, v]) => surfacePoint(cone, u, v));
  const key = (p) => p.map((c) => Math.round(c / 1e-6)).join('|');
  assert.equal(new Set(corners.map(key)).size, 2, 'the fixture genuinely collapses');

  const [vMin, vMax] = [cone.knotsV[0], cone.knotsV[cone.knotsV.length - 1]];
  const hole = [[0.4, vMin + 0.25 * (vMax - vMin)], [0.6, vMin + 0.25 * (vMax - vMin)],
    [0.6, vMin + 0.5 * (vMax - vMin)], [0.4, vMin + 0.5 * (vMax - vMin)]];
  const res = fragmentBoundaries3D(cone, { outer: trivialTrimLoop(cone), holes: [hole] });
  assert.ok(res.ok, res.reason);
  assert.equal(res.loops.length, 1, 'the hole alone is the boundary');
  assert.equal(res.loops[0].length, 4, 'and it is the hole, unsplit');
  assert.equal(new Set(res.loops[0].map(key)).size, 4, 'the hole itself is not degenerate');
});

test('a fragment with two holes refuses by name rather than guessing', () => {
  const srf = quadSurface([0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]);
  const res = fragmentBoundaries3D(srf, {
    outer: [[0, 0], [1, 0], [1, 1], [0, 1]],
    holes: [
      [[0.1, 0.1], [0.1, 0.3], [0.3, 0.3], [0.3, 0.1]],
      [[0.6, 0.6], [0.6, 0.8], [0.8, 0.8], [0.8, 0.6]],
    ],
  });
  assert.equal(res.ok, false);
  assert.match(res.reason, /2 holes/);
});

test('closed and unclosed trim-loop forms sew to exactly the same solid', () => {
  // Both forms exist in this kernel, so the sew must not care which it is
  // handed. Stated honestly: the guarantee here is `weldPoints`' own — it
  // pops a repeated first point itself — so this pins the invariant a caller
  // depends on, not a mechanism this module provides. Asserting the two are
  // IDENTICAL rather than each merely valid is what makes it worth having.
  const open = sewFragments(cubeFaces());
  const closed = sewFragments(cubeFaces().map((f) => ({ ...f, outer: [...f.outer, [0, 0]] })));
  assert.ok(open.ok && closed.ok, `${open.verdict} / ${closed.verdict}`);
  assert.deepEqual(closed.stats, open.stats, 'the two forms are indistinguishable downstream');
  assert.equal(closed.stats.chi, 2);
});

test('an empty keep-set and a degenerate fragment both refuse by name', () => {
  const empty = sewFragments([]);
  assert.equal(empty.ok, false);
  assert.match(empty.reason, /no kept fragments/i);

  const srf = quadSurface([0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]);
  const degenerate = sewFragments([{ srf, outer: [[0, 0], [1, 0]] }]);
  assert.equal(degenerate.ok, false);
  assert.match(degenerate.reason, /no usable boundary/i);
});

// ---------------------------------------------------------------------------
// THE KEEP-RULES, applied to real fragments — the step that makes "all three
// operators are the same machine" concrete rather than asserted.
// ---------------------------------------------------------------------------

// A cube as a triangle soup, to classify against.
function cubeTris(lo = 0, hi = 10) {
  const v = [
    [lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo],
    [lo, lo, hi], [hi, lo, hi], [hi, hi, hi], [lo, hi, hi],
  ];
  const quad = (a, b, c, d) => [[v[a], v[b], v[c]], [v[a], v[c], v[d]]];
  return [
    ...quad(0, 3, 2, 1), ...quad(4, 5, 6, 7), ...quad(0, 1, 5, 4),
    ...quad(3, 7, 6, 2), ...quad(0, 4, 7, 3), ...quad(1, 2, 6, 5),
  ];
}

// A plane at z=5 spanning x,y in [-20,20]; the cube above cuts it exactly at
// u,v in [0.5,0.75], derived from the plane's own linear mapping.
function planeAndFragments() {
  const srf = quadSurface([-20, -20, 5], [20, -20, 5], [20, 20, 5], [-20, 20, 5]);
  const insideFrag = { outer: [[0.5, 0.5], [0.75, 0.5], [0.75, 0.75], [0.5, 0.75]] };
  const outsideFrag = { outer: [[0, 0], [0.4, 0], [0.4, 0.4], [0, 0.4]] };
  return { srf, frags: [insideFrag, outsideFrag] };
}

test('the SAME fragments survive differently under each operator — only the rule changes', () => {
  const { srf, frags } = planeAndFragments();
  const tris = cubeTris();

  const union = keepFragments(srf, frags, tris, 'union');
  assert.ok(union.ok, union.reason);
  assert.equal(union.kept.length, 1);
  assert.deepEqual(union.classifications.map((c) => c.region), ['inside', 'outside']);
  assert.deepEqual(union.kept[0].outer, frags[1].outer, 'union keeps what is OUTSIDE the other solid');

  const intersect = keepFragments(srf, frags, tris, 'intersect');
  assert.ok(intersect.ok, intersect.reason);
  assert.equal(intersect.kept.length, 1);
  assert.deepEqual(intersect.kept[0].outer, frags[0].outer, 'intersect keeps what is INSIDE it');
});

test('DIFFERENCE flips the rule for the second operand, not the first', () => {
  const { srf, frags } = planeAndFragments();
  const tris = cubeTris();
  // A's own faces keep what is outside B, exactly like union.
  const fromA = keepFragments(srf, frags, tris, 'difference', { operand: 'a' });
  assert.ok(fromA.ok, fromA.reason);
  assert.deepEqual(fromA.kept[0].outer, frags[1].outer);
  // B's faces keep what is INSIDE A — the reversed operand, which is what
  // makes difference the same machine as intersect rather than a third one.
  const fromB = keepFragments(srf, frags, tris, 'difference', { operand: 'b' });
  assert.ok(fromB.ok, fromB.reason);
  assert.deepEqual(fromB.kept[0].outer, frags[0].outer);
  assert.notDeepEqual(fromA.kept[0].outer, fromB.kept[0].outer, 'the two operands genuinely differ');
});

test('a fragment lying ON the other solid refuses by name — the coincident-face case', () => {
  // A plane sitting exactly on the cube's own top face: every probe on it is
  // a boundary point, which is precisely what the pipeline puts out of scope.
  const srf = quadSurface([0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10]);
  const res = keepFragments(srf, [{ outer: [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]] }], cubeTris(), 'union');
  assert.equal(res.ok, false);
  assert.match(res.reason, /coincident|tangential|ON the other solid/i);
});

test('an unknown operator refuses rather than defaulting to one of the three', () => {
  const { srf, frags } = planeAndFragments();
  const res = keepFragments(srf, frags, cubeTris(), 'subtract-ish');
  assert.equal(res.ok, false);
  assert.match(res.reason, /unknown boolean operation/i);
});
