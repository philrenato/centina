// kernel/brepbuild.mjs — welding a plain set of face boundary loops into a
// real half-edge B-rep, then handing it to kernel/brep.mjs's OWN validator.
//
// Nothing here trusts the builder's own word: every structural claim is
// re-checked by validateBrep (a genuinely separate implementation with its
// own error codes), and every count is checked against the textbook value
// for the polyhedron in question, not against whatever the builder said.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildBrepSolid, boundaryLoops, orientLoops, weldPoints, brepVerdict } from '../kernel/brepbuild.mjs';
import { validateBrep, eulerCharacteristic, hasErrorCode, allFaces } from '../kernel/brep.mjs';

// A unit cube as six quad loops. Faces 0/2/4 are wound one way and 1/3/5 the
// other, exactly the way a corner-by-corner primitive authors them — i.e.
// deliberately NOT consistently oriented, so the builder's own orientation
// pass is exercised rather than bypassed.
function cubeFaces(s = 1) {
  const c = (x, y, z) => [x * s, y * s, z * s];
  return [
    [c(0, 0, 0), c(1, 0, 0), c(1, 1, 0), c(0, 1, 0)], // z = 0
    [c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)], // z = 1
    [c(0, 0, 0), c(1, 0, 0), c(1, 0, 1), c(0, 0, 1)], // y = 0
    [c(0, 1, 0), c(1, 1, 0), c(1, 1, 1), c(0, 1, 1)], // y = 1
    [c(0, 0, 0), c(0, 1, 0), c(0, 1, 1), c(0, 0, 1)], // x = 0
    [c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), c(1, 0, 1)], // x = 1
  ];
}

test('a cube welds to the textbook V=8 E=12 F=6, chi=2, and passes validateBrep', () => {
  const res = buildBrepSolid(cubeFaces());
  assert.equal(res.ok, true, res.reason ?? '');
  assert.equal(res.stats.V, 8);
  assert.equal(res.stats.E, 12);
  assert.equal(res.stats.F, 6);
  assert.equal(res.stats.chi, 2);
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.nonManifoldEdgeCount, 0);
  assert.equal(res.stats.shellCount, 1);
  assert.equal(res.stats.genus, 0);
  // The independent check: kernel/brep.mjs's own validator, not the builder.
  const v = validateBrep(res.solid);
  assert.equal(v.ok, true, JSON.stringify(v.errors));
  const e = eulerCharacteristic(res.solid);
  assert.equal(e.V - e.E + e.F, 2);
  assert.equal(e.ok, true);
  assert.equal(allFaces(res.solid).length, 6);
  // The input faces really were inconsistently wound, so the orientation
  // pass genuinely had work to do — proving the pass, not just its absence.
  assert.ok(res.stats.flippedFaces > 0, 'the fixture is deliberately mis-wound; some faces must have been flipped');
  assert.match(brepVerdict(res), /CLOSED SOLID/);
});

test('a cube with one face removed is an OPEN SHELL with exactly 4 naked edges, not a solid', () => {
  const faces = cubeFaces();
  faces.splice(1, 1); // drop z = 1
  const res = buildBrepSolid(faces);
  assert.equal(res.ok, false);
  assert.equal(res.reason, 'open-shell');
  assert.equal(res.stats.F, 5);
  assert.equal(res.stats.nakedEdgeCount, 4);
  assert.equal(res.stats.V, 8);
  assert.equal(res.stats.E, 12);
  assert.equal(res.stats.chi, 1); // a disk: chi = 1, not 2 — the actual discriminator
  assert.match(brepVerdict(res), /OPEN SHELL/);
  assert.match(brepVerdict(res), /4 naked edges/);
});

test('an edge shared by three faces is refused BY NAME as non-manifold', () => {
  // Three quads hinged on one shared spine edge.
  const spineA = [0, 0, 0], spineB = [1, 0, 0];
  const res = buildBrepSolid([
    [spineA, spineB, [1, 1, 0], [0, 1, 0]],
    [spineA, spineB, [1, 0, 1], [0, 0, 1]],
    [spineA, spineB, [1, -1, -1], [0, -1, -1]],
  ]);
  assert.equal(res.ok, false);
  assert.equal(res.reason, 'non-manifold-edge');
  assert.equal(res.stats.nonManifoldEdgeCount, 1);
  assert.match(brepVerdict(res), /NON-MANIFOLD/);
});

test('two disjoint cubes build as TWO shells with chi = 4, genus 0, and still validate', () => {
  const a = cubeFaces();
  const b = cubeFaces().map((f) => f.map((p) => [p[0] + 10, p[1], p[2]]));
  const res = buildBrepSolid([...a, ...b]);
  assert.equal(res.ok, true, res.reason ?? '');
  assert.equal(res.stats.shellCount, 2);
  assert.equal(res.stats.V, 16);
  assert.equal(res.stats.E, 24);
  assert.equal(res.stats.F, 12);
  assert.equal(res.stats.chi, 4);
  assert.equal(res.stats.genus, 0);
  const v = validateBrep(res.solid);
  assert.equal(v.ok, true, JSON.stringify(v.errors));
  assert.equal(eulerCharacteristic(res.solid).ok, true);
});

test('corners perturbed below tolerance still weld to ONE vertex (and above it do not)', () => {
  const jitter = (p, i) => [p[0] + (i % 3 === 0 ? 4e-7 : 0), p[1], p[2]];
  const faces = cubeFaces().map((f, fi) => f.map((p) => jitter(p, fi)));
  const welded = weldPoints(faces.flatMap((f) => [f]), 1e-6);
  assert.equal(welded.points.length, 8, 'sub-tolerance jitter must not split a corner');
  const tight = weldPoints(faces.flatMap((f) => [f]), 1e-9);
  assert.ok(tight.points.length > 8, 'below the jitter, the same corners must NOT weld — proving the tolerance is real');
});

test('boundaryLoops finds the single closed rim of an open box, and none of a closed one', () => {
  const faces = cubeFaces();
  const openFaces = faces.slice(1); // drop z = 0
  const oriented = orientLoops(weldPoints(openFaces).loops);
  const b = boundaryLoops(oriented.loops);
  assert.equal(b.nakedEdgeCount, 4);
  assert.equal(b.loops.length, 1);
  assert.equal(b.loops[0].closed, true);
  assert.equal(b.loops[0].vertices.length, 4);
  const closed = orientLoops(weldPoints(faces).loops);
  assert.equal(boundaryLoops(closed.loops).nakedEdgeCount, 0);
});

test('a genuinely non-orientable face set is refused BY NAME, not silently solidified', () => {
  // A three-quad Möbius band: the last strip is joined back to the first
  // with a half twist, so no consistent winding exists.
  const p = [
    [0, 0, 0], [0, 1, 0], // 0,1
    [1, 0, 0], [1, 1, 0], // 2,3
    [2, 0, 0], [2, 1, 0], // 4,5
  ];
  const res = buildBrepSolid([
    [p[0], p[2], p[3], p[1]],
    [p[2], p[4], p[5], p[3]],
    [p[4], p[5], p[0], p[1]], // the twist: rail 4 joins to 1 and rail 5 to 0, so the band closes with a half turn
  ]);
  assert.equal(res.ok, false);
  assert.equal(res.reason, 'non-orientable');
  assert.match(brepVerdict(res), /NOT ORIENTABLE/);
});

// --- SHELL ORIENTATION -----------------------------------------------
// Consistency and outwardness are different facts. Every test above is
// satisfied by a shell that is consistently wound INSIDE-OUT, which is
// why these measure SIGNED volume: it is the one quantity that reads the
// direction rather than merely the agreement.

// A cube wound CONSISTENTLY, in a direction the caller chooses. This has
// to be a separate fixture from `cubeFaces` above: that one is
// deliberately mis-wound, so which way `orientLoops` settles it is
// decided by whichever face seeded the traversal — precisely the
// arbitrariness these tests exist to remove, and useless as a starting
// direction to assert against.
function consistentCube(s = 1, outward = true, off = [0, 0, 0]) {
  const c = (x, y, z) => [x * s + off[0], y * s + off[1], z * s + off[2]];
  const faces = [
    [c(0, 0, 0), c(0, 1, 0), c(1, 1, 0), c(1, 0, 0)], // z = 0, normal -Z
    [c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)], // z = 1, normal +Z
    [c(0, 0, 0), c(1, 0, 0), c(1, 0, 1), c(0, 0, 1)], // y = 0, normal -Y
    [c(0, 1, 0), c(0, 1, 1), c(1, 1, 1), c(1, 1, 0)], // y = 1, normal +Y
    [c(0, 0, 0), c(0, 0, 1), c(0, 1, 1), c(0, 1, 0)], // x = 0, normal -X
    [c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), c(1, 0, 1)], // x = 1, normal +X
  ];
  return outward ? faces : faces.map((l) => l.slice().reverse());
}

// Signed volume straight from the BUILT solid's own half-edge loops, by
// the divergence theorem over origin-fanned tetrahedra. Deliberately
// re-derived here from the result rather than imported from the module
// under test, so this measures what was built rather than restating the
// claim of the code that built it.
function brepSignedVolume(solid) {
  let v = 0;
  for (const shell of solid.shells) {
    for (const face of shell.faces) {
      for (const loop of face.loops) {
        const pts = [];
        let he = loop.halfEdge;
        const start = he;
        do { pts.push(he.vertex.point); he = he.next; } while (he && he !== start);
        const a = pts[0];
        for (let i = 1; i + 1 < pts.length; i++) {
          const b = pts[i], c = pts[i + 1];
          v += (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
        }
      }
    }
  }
  return v;
}

test('a consistently INSIDE-OUT cube is turned outward, and reports that it was', () => {
  // Every neighbor agrees with every neighbor here, so `orientLoops`
  // has nothing to fix — this is exactly the case that used to come back
  // inside-out with a clean bill of health.
  const res = buildBrepSolid(consistentCube(3, false));
  assert.equal(res.ok, true, res.reason ?? '');
  assert.equal(res.stats.flippedFaces, 0, 'the fixture is already consistent; orientLoops must find nothing to do');
  assert.equal(res.stats.shellCount, 1);
  assert.equal(res.solid.shells[0].shellKind, 'outer');
  assert.equal(res.stats.shellOrientationUncertain, 0);
  assert.equal(res.stats.flippedShells, 1, 'the whole shell faced inward and must have been reversed');
  const vol = brepSignedVolume(res.solid);
  assert.ok(Math.abs(vol - 27) < 1e-9, `a 3x3x3 cube must measure +27, got ${vol}`);
});

test('an already-outward cube is left completely alone', () => {
  const res = buildBrepSolid(consistentCube(3, true));
  assert.equal(res.ok, true, res.reason ?? '');
  assert.equal(res.stats.flippedShells, 0, 'nothing was wrong; nothing should have moved');
  assert.equal(res.solid.shells[0].shellKind, 'outer');
  const vol = brepSignedVolume(res.solid);
  assert.ok(Math.abs(vol - 27) < 1e-9, `expected +27, got ${vol}`);
});

test('the mis-wound fixture lands outward too, whichever way its traversal happened to seed', () => {
  // `cubeFaces` is inconsistently wound, so orientLoops picks a direction
  // on its own; the ALL-REVERSED twin of the same fixture generally picks
  // the other one. Both must still end up measuring +27 — the real claim
  // is that the OUTCOME no longer depends on that accident.
  for (const faces of [cubeFaces(3), cubeFaces(3).map((l) => l.slice().reverse())]) {
    const res = buildBrepSolid(faces);
    assert.equal(res.ok, true, res.reason ?? '');
    const vol = brepSignedVolume(res.solid);
    assert.ok(Math.abs(vol - 27) < 1e-9, `expected +27 regardless of arrival winding, got ${vol}`);
  }
});

test('two DISJOINT solids are both outer — the case a largest-volume rule would get wrong', () => {
  // Deliberately different sizes, both arriving inside-out. A rule that
  // called the bigger one the solid and the smaller one its void would
  // pass every structural check above and total 27 - 1 instead of 27 + 1.
  const res = buildBrepSolid([
    ...consistentCube(3, false),
    ...consistentCube(1, false, [10, 0, 0]),
  ]);
  assert.equal(res.ok, true, res.reason ?? '');
  assert.equal(res.stats.shellCount, 2);
  assert.deepEqual(res.solid.shells.map((s) => s.shellKind), ['outer', 'outer']);
  assert.equal(res.stats.flippedShells, 2);
  const vol = brepSignedVolume(res.solid);
  assert.ok(Math.abs(vol - 28) < 1e-9, `two disjoint solids must SUM (27 + 1 = 28), got ${vol}`);
});

test('a VOID shell is classified as one and faces INTO the cavity, whichever way it arrives', () => {
  // A hollow cube: a 6-unit outer with a 2-unit cavity centered inside it,
  // fed in with BOTH shells wound outward — so the inner one is genuinely
  // wrong on arrival and must be reversed.
  const res = buildBrepSolid([
    ...consistentCube(6, true),
    ...consistentCube(2, true, [2, 2, 2]),
  ]);
  assert.equal(res.ok, true, res.reason ?? '');
  assert.equal(res.stats.shellCount, 2);
  assert.equal(res.stats.shellOrientationUncertain, 0);
  const kinds = res.solid.shells.map((s) => s.shellKind);
  assert.equal(kinds.filter((k) => k === 'outer').length, 1);
  assert.equal(kinds.filter((k) => k === 'void').length, 1);
  assert.equal(res.stats.flippedShells, 1, 'exactly the inner shell was wrong');
  // 6^3 - 2^3 = 208: the void SUBTRACTS, which is the whole point of the
  // convention — a hollow solid's volume is the plain sum of its shells.
  const vol = brepSignedVolume(res.solid);
  assert.ok(Math.abs(vol - 208) < 1e-9, `a hollow cube must measure 6^3 - 2^3 = 208, got ${vol}`);
});

test('a void arriving ALREADY correct is not flipped back out again', () => {
  const res = buildBrepSolid([
    ...consistentCube(6, true),
    ...consistentCube(2, false, [2, 2, 2]),
  ]);
  assert.equal(res.ok, true, res.reason ?? '');
  assert.equal(res.stats.flippedShells, 0, 'both shells arrived correct; nothing should have moved');
  assert.deepEqual(
    res.solid.shells.map((s) => s.shellKind).sort(),
    ['outer', 'void'],
  );
  const vol = brepSignedVolume(res.solid);
  assert.ok(Math.abs(vol - 208) < 1e-9, `expected 208, got ${vol}`);
});
