// PHASE 1 B-REP TOPOLOGY — tests.
//
// The load-bearing checks here, in order of how much they actually prove:
//   1. Euler-Poincaré holds after EVERY operator and every sequence, not
//      just at the end of a construction. `stepper()` wraps every operator
//      call in a real assertion so a bookkeeping mistake is caught at the
//      step that made it, not 20 steps later.
//   2. Two non-trivial closed solids are built THROUGH THE OPERATORS ALONE
//      (no hand-written pointer surgery anywhere in this file's fixtures):
//      a box, and a rectangular block with a rectangular through-hole
//      (genus 1, two rings — deliberately non-cubic and with an off-centre
//      hole, so a symmetry accident cannot make a wrong answer look right).
//   3. Every operator's inverse restores the prior STRUCTURE, compared by a
//      canonical id-free fingerprint; for the three pairs where object
//      identity is also preservable, exact id-set equality is asserted too.
//   4. validateBrep genuinely fires on each named failure mode, against a
//      deliberately corrupted structure per mode.

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  mvfs, kvfs, mev, kev, mef, kef, kemr, mekr, kfmrh, mfkrh,
  mergeSolidShells,
  eulerCharacteristic, assertEulerPoincare,
  validateBrep, hasErrorCode, brepFingerprint,
  loopHalfEdges, halfEdgesInLoopFrom, destinationOf, allFaces,
  sweepLoop, cutRing,
  attachSurface, assignPlanarPcurves, faceTrimmedSurface, faceTrimValidity,
  pcurvesFromSSISamples,
} from '../kernel/brep.mjs';
import { trimLoopsValid, signedArea2D } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface, tessellationArea } from '../kernel/trimtess.mjs';
import { intersectSurfaces } from '../kernel/ssi.mjs';

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

// Wraps a construction so every single operator call is followed by a real
// Euler-Poincaré assertion. Returns the list of step labels so a test can
// also assert HOW MANY operators ran (a construction that silently took a
// different path would change that count).
function stepper(labels = [], journal = []) {
  return (label, solid, result) => {
    labels.push(label);
    journal.push({ op: label, res: result });
    assertEulerPoincare(solid, `${labels.length}: ${label}`);
  };
}

// Drive a recorded construction backwards, one inverse per forward operator,
// asserting the invariant AND structural validity after every single step.
//
// The `rebind` map is not incidental bookkeeping — it is the honest
// consequence of a real property of this operator set, found by running this
// teardown rather than by reading the code: KEMR destroys an edge that an
// EARLIER MEV created, and its inverse MEKR re-creates that edge as a new
// object. Every handle recorded when the spur was first made is therefore
// dead from the moment KEMR runs, and a backwards replay must re-bind it.
// KEMR hands back exactly what is needed to do so.
function teardown(solid, journal) {
  const rebind = new Map();
  const live = (he) => (rebind.has(he) ? rebind.get(he) : he);
  for (let i = journal.length - 1; i >= 0; i--) {
    const j = journal[i];
    if (j.op === 'mev') kev(live(j.res.halfEdge));
    else if (j.op === 'mef') kef(live(j.res.halfEdge));
    else if (j.op === 'kemr') {
      const remade = mekr(j.res.keepHalfEdge, j.res.ringHalfEdge);
      rebind.set(j.res.killedHalfEdge, remade.halfEdge);
      rebind.set(j.res.killedTwin, remade.twin);
    } else if (j.op === 'kfmrh') mfkrh(j.res.face, j.res.ring);
    else if (j.op === 'mvfs') continue;
    else assert.fail(`teardown does not know how to invert ${j.op}`);
    assertEulerPoincare(solid, `teardown step ${i} (${j.op})`);
    const res = validateBrep(solid, { allowIntermediate: true });
    assert.deepEqual(res.errors, [], `teardown step ${i} (${j.op}) left an invalid structure`);
  }
}

function heFrom(loop, v, dest) {
  const found = halfEdgesInLoopFrom(loop, v).find((h) => destinationOf(h) === dest);
  assert.ok(found, 'expected a half-edge between the two given vertices');
  return found;
}

function idSet(solid) {
  const ids = new Set();
  for (const v of solid.vertices) ids.add(`v${v.id}`);
  for (const e of solid.edges) { ids.add(`e${e.id}`); for (const h of e.halfEdges) ids.add(`h${h.id}`); }
  for (const sh of solid.shells) {
    ids.add(`s${sh.id}`);
    for (const f of sh.faces) { ids.add(`f${f.id}`); for (const l of f.loops) ids.add(`l${l.id}`); }
  }
  return [...ids].sort().join(' ');
}

// A rectangular block, built through the operators only. `journal` records
// every forward operator with the handle its inverse will need, so a full
// teardown can be driven in exact reverse.
function buildBlock(w, d, h, onStep) {
  const A = [0, 0, 0], B = [w, 0, 0], C = [w, d, 0], D = [0, d, 0];
  const { solid, face: fLower, loop: lLower, vertex: vA } = mvfs(A);
  onStep('mvfs', solid, null);

  const rB = mev(lLower, vA, B); onStep('mev', solid, rB);
  const rC = mev(lLower, rB.vertex, C); onStep('mev', solid, rC);
  const rD = mev(lLower, rC.vertex, D); onStep('mev', solid, rD);

  const made = mef(heFrom(lLower, vA, rB.vertex), heFrom(lLower, rD.vertex, rC.vertex));
  onStep('mef', solid, made);

  // `made.loop` is A->B->C->D (counter-clockwise seen from +Z); sweeping THAT
  // one upward is what makes it the top face with an outward-pointing
  // orientation, and leaves fLower (D->C->B->A) as the correctly-oriented
  // bottom. Orientation is a real choice here, not an accident.
  const topVerts = sweepLoop(made.loop, (v) => [v.point[0], v.point[1], h], { onStep });
  return { solid, bottomFace: fLower, bottomLoop: lLower, topFace: made.face, topLoop: made.loop, topVerts };
}

// ---------------------------------------------------------------------
// 1. the invariant itself
// ---------------------------------------------------------------------

test('eulerCharacteristic on a bare MVFS seed: V=1 E=0 F=1 L=1 R=0 S=1 G=0, and the invariant holds', () => {
  const { solid } = mvfs([0, 0, 0]);
  const e = eulerCharacteristic(solid);
  assert.deepEqual(
    { V: e.V, E: e.E, F: e.F, L: e.L, R: e.R, S: e.S, G: e.G },
    { V: 1, E: 0, F: 1, L: 1, R: 0, S: 1, G: 0 }
  );
  assert.equal(e.lhs, 2);
  assert.equal(e.rhs, 2);
  assert.ok(e.ok);
});

test('assertEulerPoincare fires on a deliberately mis-counted solid — the invariant check is not vacuous', () => {
  const { solid } = mvfs([0, 0, 0]);
  assertEulerPoincare(solid, 'seed'); // holds
  solid.genus = 1;                    // a lie no operator would ever tell
  assert.throws(() => assertEulerPoincare(solid, 'tampered'), /Euler-Poincar/);
});

// ---------------------------------------------------------------------
// 2. real solids, built through the operators alone
// ---------------------------------------------------------------------

test('a rectangular block built through the Euler operators alone: the invariant holds after EVERY step, and the finished solid validates', () => {
  const labels = [];
  const { solid } = buildBlock(60, 40, 25, stepper(labels));

  // 1 mvfs + 3 mev + 1 mef + (4 mev + 4 mef) = 13 operator applications.
  assert.equal(labels.length, 13);
  assert.deepEqual(labels.filter((l) => l === 'mev').length, 7);
  assert.deepEqual(labels.filter((l) => l === 'mef').length, 5);

  const e = eulerCharacteristic(solid);
  assert.deepEqual(
    { V: e.V, E: e.E, F: e.F, L: e.L, R: e.R, S: e.S, G: e.G },
    { V: 8, E: 12, F: 6, L: 6, R: 0, S: 1, G: 0 }
  );
  const res = validateBrep(solid);
  assert.deepEqual(res.errors, [], 'a finished block should have zero validity errors');
  assert.ok(res.ok);

  // Every face is a genuine 4-sided closed cycle of distinct vertices.
  for (const f of allFaces(solid)) {
    assert.equal(f.loops.length, 1);
    const hes = loopHalfEdges(f.loops[0]);
    assert.equal(hes.length, 4);
    assert.equal(new Set(hes.map((x) => x.vertex)).size, 4);
  }
  // Every edge is used by exactly two half-edges, in opposite directions.
  for (const edge of solid.edges) {
    assert.equal(edge.halfEdges.length, 2);
    const [a, b] = edge.halfEdges;
    assert.equal(a.twin, b);
    assert.equal(destinationOf(a), b.vertex);
    assert.equal(destinationOf(b), a.vertex);
    assert.notEqual(a.loop, b.loop, 'a finished solid has no dangling edge');
  }
});

// The non-trivial fixture: a 60 x 40 x 25 block with a 30 x 20 rectangular
// through-hole, deliberately OFF-CENTRE (x 15..45 in a 0..60 span, y 10..30
// in a 0..40 span) so nothing here can pass by symmetry.
function buildWasher(onStep) {
  const block = buildBlock(60, 40, 25, onStep);
  const { solid, topLoop, bottomFace } = block;

  // Traced counter-clockwise so the RING left behind on the top face (which
  // is the return path, i.e. the reverse of the traced order) comes out
  // clockwise — opposite the counter-clockwise outer loop, which is exactly
  // the outer/hole winding convention kernel/trim.mjs already enforces.
  const ringTrace = [[15, 10, 25], [45, 10, 25], [45, 30, 25], [15, 30, 25]];
  const cut = cutRing(topLoop, block.topVerts[0], ringTrace, { onStep });

  const tunnel = sweepLoop(cut.capLoop, (v) => [v.point[0], v.point[1], 0], { onStep });
  const punched = kfmrh(bottomFace, cut.capFace);
  onStep('kfmrh', solid, { face: bottomFace, ring: punched.ring });

  return { ...block, ring: cut.ring, tunnelVerts: tunnel, holeRing: punched.ring };
}

test('a block with a rectangular THROUGH-HOLE, built through the Euler operators alone: genus 1, two rings, invariant holds at every step, and it validates clean', () => {
  const labels = [];
  const { solid, topFace, bottomFace, ring, holeRing } = buildWasher(stepper(labels));

  // 13 (block) + 5 (cutRing: 4 mev, 1 mef) + 1 (kemr) + 8 (tunnel sweep) + 1 (kfmrh)
  assert.equal(labels.length, 28);
  assert.equal(labels.filter((l) => l === 'kemr').length, 1, 'exactly one KEMR made the top ring');
  assert.equal(labels.filter((l) => l === 'kfmrh').length, 1, 'exactly one KFMRH punched the tunnel through');

  const e = eulerCharacteristic(solid);
  assert.deepEqual(
    { V: e.V, E: e.E, F: e.F, L: e.L, R: e.R, S: e.S, G: e.G },
    { V: 16, E: 24, F: 10, L: 12, R: 2, S: 1, G: 1 }
  );
  // The invariant, spelled out on this fixture rather than only asserted by
  // the helper: 16 - 24 + 10 - 2 = 0 = 2 * (1 - 1).
  assert.equal(e.lhs, 0);
  assert.equal(e.rhs, 0);

  const res = validateBrep(solid);
  assert.deepEqual(res.errors, [], 'a finished genus-1 solid should have zero validity errors');

  // The two rings really are rings, on the two faces we expect.
  assert.equal(topFace.loops.length, 2);
  assert.equal(topFace.loops[1], ring);
  assert.equal(bottomFace.loops.length, 2);
  assert.equal(bottomFace.loops[1], holeRing);
  assert.equal(loopHalfEdges(ring).length, 4);
  assert.equal(loopHalfEdges(holeRing).length, 4);

  // 4 tunnel walls exist and each is a real quad face distinct from the 6
  // outer-block faces — i.e. the hole is a genuine tunnel, not bookkeeping.
  const quadFaces = allFaces(solid).filter((f) => f.loops.length === 1 && loopHalfEdges(f.loops[0]).length === 4);
  assert.equal(quadFaces.length, 8, '4 outer side walls + 4 tunnel walls');

  // Both rings wind OPPOSITE their own face's outer loop, in the XY plane
  // both those faces live in — the real orientation property, measured, not
  // assumed.
  const xy = (loop) => loopHalfEdges(loop).map((he) => [he.vertex.point[0], he.vertex.point[1]]);
  assert.ok(signedArea2D(xy(topFace.loops[0])) > 0 && signedArea2D(xy(ring)) < 0);
  assert.ok(signedArea2D(xy(bottomFace.loops[0])) < 0 && signedArea2D(xy(holeRing)) > 0);
});

// ---------------------------------------------------------------------
// 3. inverses
// ---------------------------------------------------------------------

test('MEV/KEV and MEF/KEF round-trip EXACTLY — same structure and the same entity ids, not merely an isomorphic rebuild', () => {
  const { solid, topLoop, topVerts } = buildBlock(60, 40, 25, () => {});
  const before = brepFingerprint(solid);
  const beforeIds = idSet(solid);

  const grown = mev(topLoop, topVerts[0], [-5, -5, 40]);
  assert.notEqual(brepFingerprint(solid), before, 'mev must actually change the structure');
  kev(grown.halfEdge);
  assert.equal(brepFingerprint(solid), before);
  assert.equal(idSet(solid), beforeIds, 'kev must leave every surviving entity id untouched');
  assertEulerPoincare(solid, 'mev/kev round trip');

  const a = topVerts[0], b = topVerts[2];
  const made = mef(heFrom(topLoop, a, topVerts[1]), heFrom(topLoop, b, topVerts[3]));
  assert.notEqual(brepFingerprint(solid), before);
  assert.equal(eulerCharacteristic(solid).F, 7);
  kef(made.halfEdge);
  assert.equal(brepFingerprint(solid), before);
  assert.equal(idSet(solid), beforeIds, 'kef must leave every surviving entity id untouched');
  assert.deepEqual(validateBrep(solid).errors, []);
});

test('KEMR/MEKR and KFMRH/MFKRH round-trip to the exact prior STRUCTURE (identity is not preserved — an inverse that re-creates an entity gets a fresh id, stated rather than glossed)', () => {
  const journal = [];
  const { solid, topFace, bottomFace, ring, holeRing } = buildWasher(() => {}, journal);
  const washer = brepFingerprint(solid);

  // KFMRH -> MFKRH
  const restoredCap = mfkrh(bottomFace, holeRing);
  assert.equal(eulerCharacteristic(solid).G, 0);
  assert.equal(bottomFace.loops.length, 1);
  const rePunched = kfmrh(bottomFace, restoredCap.face);
  assert.equal(eulerCharacteristic(solid).G, 1);
  assert.equal(brepFingerprint(solid), washer);
  assert.equal(rePunched.ring, holeRing, 'the same loop object came back as the ring');

  // KEMR -> MEKR. Rebuilding the spur must restore the top face to one loop,
  // then killing it again must reproduce the identical ring.
  const bridged = mekr(topFace.loops[0].halfEdge, ring.halfEdge);
  assert.equal(topFace.loops.length, 1, 'the ring was absorbed back into the outer loop');
  assert.equal(eulerCharacteristic(solid).R, 1);
  assertEulerPoincare(solid, 'mekr');
  const reRing = kemr(bridged.halfEdge);
  assert.equal(topFace.loops.length, 2);
  assert.equal(brepFingerprint(solid), washer, 'structure restored exactly');
  assert.notEqual(reRing.ring, ring, 'the restored ring is a NEW loop object — structure is restored, identity is not');
  assert.deepEqual(validateBrep(solid).errors, []);
});

const SEED_FINGERPRINT = 'V1E0F1L1S1G0;shells(1);F[EMPTY]';

test('build a whole block, tear it down operator-by-operator in exact reverse, and land back on the bare MVFS seed', () => {
  const journal = [];
  const { solid } = buildBlock(60, 40, 25, stepper([], journal));
  assert.deepEqual(validateBrep(solid).errors, []);
  assert.notEqual(brepFingerprint(solid), SEED_FINGERPRINT);
  assert.equal(journal.length, 13);

  teardown(solid, journal);

  const e = eulerCharacteristic(solid);
  assert.deepEqual({ V: e.V, E: e.E, F: e.F, L: e.L, S: e.S, G: e.G }, { V: 1, E: 0, F: 1, L: 1, S: 1, G: 0 });
  assert.equal(brepFingerprint(solid), SEED_FINGERPRINT, 'back exactly where we started');
  kvfs(solid);
  assert.equal(solid.vertices.length, 0);
  assert.equal(solid.shells.length, 0);
});

test('the genus-1 through-hole solid tears all the way down too — every one of the five inverses, including KEMR/MEKR and KFMRH/MFKRH, driven in exact reverse back to the seed', () => {
  const journal = [];
  const { solid } = buildWasher(stepper([], journal));
  assert.equal(journal.length, 28);
  assert.equal(eulerCharacteristic(solid).G, 1);

  teardown(solid, journal);

  const e = eulerCharacteristic(solid);
  assert.deepEqual({ V: e.V, E: e.E, F: e.F, L: e.L, S: e.S, G: e.G }, { V: 1, E: 0, F: 1, L: 1, S: 1, G: 0 });
  assert.equal(brepFingerprint(solid), SEED_FINGERPRINT);
  kvfs(solid);
});

test('KVFS refuses anything that is not a bare seed — it will not silently discard a real solid', () => {
  const { solid } = buildBlock(10, 10, 10, () => {});
  assert.throws(() => kvfs(solid), /only a bare MVFS seed/);
});

// ---------------------------------------------------------------------
// 4. the shell term
// ---------------------------------------------------------------------

test('a two-shell solid (a block with a block-shaped cavity): S=2, the invariant still holds, and it validates', () => {
  const outer = buildBlock(60, 40, 25, () => {});
  const inner = buildBlock(20, 15, 10, () => {});
  // Shift the cavity so no two vertices coincide (also what makes the
  // canonical fingerprint well-defined).
  for (const v of inner.solid.vertices) { v.point[0] += 20; v.point[1] += 12; v.point[2] += 7; }

  mergeSolidShells(outer.solid, inner.solid);
  const e = eulerCharacteristic(outer.solid);
  assert.deepEqual(
    { V: e.V, E: e.E, F: e.F, L: e.L, R: e.R, S: e.S, G: e.G },
    { V: 16, E: 24, F: 12, L: 12, R: 0, S: 2, G: 0 }
  );
  assert.equal(e.lhs, 4);
  assert.equal(e.rhs, 4); // 2 * (2 - 0)
  assert.deepEqual(validateBrep(outer.solid).errors, []);
});

// ---------------------------------------------------------------------
// 5. validateBrep genuinely fires, one deliberately broken structure per mode
// ---------------------------------------------------------------------

test('validateBrep catches a NON-MANIFOLD edge (an edge used by more than two faces)', () => {
  const { solid } = buildBlock(60, 40, 25, () => {});
  assert.ok(validateBrep(solid).ok);
  const edge = solid.edges[0];
  const donor = solid.edges[5].halfEdges[0];
  edge.halfEdges.push({ id: 9999, kind: 'halfedge', solid, vertex: donor.vertex, edge, twin: null, next: null, prev: null, loop: null, pcurve: null });
  const res = validateBrep(solid);
  assert.equal(res.ok, false);
  assert.ok(hasErrorCode(res, 'non-manifold-edge'));
  assert.match(res.errors.find((x) => x.code === 'non-manifold-edge').message, /used by 3 half-edges/);
});

test('validateBrep catches a REPEATED VERTEX in a face loop — and correctly does NOT flag the legitimate mid-construction case when told the build is unfinished', () => {
  // A genuine intermediate state, not a hand-corrupted one: a spur path
  // A->B->C traverses B twice by construction.
  const { solid, loop, vertex: vA } = mvfs([0, 0, 0]);
  const rB = mev(loop, vA, [10, 0, 0]);
  mev(loop, rB.vertex, [10, 10, 0]);
  assertEulerPoincare(solid, 'spur path');

  const strict = validateBrep(solid);
  assert.equal(strict.ok, false);
  assert.ok(hasErrorCode(strict, 'repeated-vertex-in-loop'));
  assert.ok(hasErrorCode(strict, 'dangling-edge'));

  const lenient = validateBrep(solid, { allowIntermediate: true });
  assert.deepEqual(lenient.errors, [], 'mid-construction these are legitimate, and the gate says so explicitly');
});

test('validateBrep catches DANGLING and ORPHANED elements', () => {
  // dangling: a spur grown off a finished block
  const block = buildBlock(60, 40, 25, () => {});
  mev(block.topLoop, block.topVerts[0], [-5, -5, 40]);
  const dangling = validateBrep(block.solid);
  assert.ok(hasErrorCode(dangling, 'dangling-edge'));

  // orphan vertex: a vertex in the solid that no half-edge touches
  const other = buildBlock(60, 40, 25, () => {});
  other.solid.vertices.push({ id: 9001, kind: 'vertex', solid: other.solid, point: [999, 999, 999], halfEdge: null });
  const orphanV = validateBrep(other.solid);
  assert.ok(hasErrorCode(orphanV, 'orphan-vertex'));

  // orphan half-edge: an edge registered on the solid whose half-edges sit
  // in no loop at all
  const third = buildBlock(60, 40, 25, () => {});
  const loose = { id: 9100, kind: 'edge', solid: third.solid, halfEdges: [], curve3d: null, tolerance: null };
  const lv = third.solid.vertices[0];
  loose.halfEdges = [
    { id: 9101, kind: 'halfedge', solid: third.solid, vertex: lv, edge: loose, twin: null, next: null, prev: null, loop: null, pcurve: null },
    { id: 9102, kind: 'halfedge', solid: third.solid, vertex: lv, edge: loose, twin: null, next: null, prev: null, loop: null, pcurve: null },
  ];
  loose.halfEdges[0].twin = loose.halfEdges[1];
  loose.halfEdges[1].twin = loose.halfEdges[0];
  third.solid.edges.push(loose);
  const orphanH = validateBrep(third.solid);
  assert.ok(hasErrorCode(orphanH, 'orphan-halfedge'));
  assert.ok(hasErrorCode(orphanH, 'orphan-edge'));
});

test('validateBrep catches INCONSISTENT WINDING between adjacent faces', () => {
  const { solid } = buildBlock(60, 40, 25, () => {});
  assert.ok(validateBrep(solid).ok);
  const [a, b] = solid.edges[0].halfEdges;
  b.vertex = a.vertex; // both sides now traverse the shared edge the same way
  const res = validateBrep(solid);
  assert.equal(res.ok, false);
  assert.ok(hasErrorCode(res, 'inconsistent-winding'));
  assert.match(res.errors.find((x) => x.code === 'inconsistent-winding').message, /traverse it the same way/);
});

test('validateBrep catches an UNCLOSED loop, in both of the two ways a cycle can break', () => {
  const broken1 = buildBlock(60, 40, 25, () => {});
  loopHalfEdges(broken1.topLoop)[1].next = null;
  const r1 = validateBrep(broken1.solid);
  assert.ok(hasErrorCode(r1, 'unclosed-loop'));
  assert.match(r1.errors.find((x) => x.code === 'unclosed-loop').message, /has no next/);

  const broken2 = buildBlock(60, 40, 25, () => {});
  const hes = loopHalfEdges(broken2.topLoop);
  hes[1].next = hes[3]; // next/prev now disagree
  const r2 = validateBrep(broken2.solid);
  assert.ok(hasErrorCode(r2, 'unclosed-loop'));
  assert.match(r2.errors.find((x) => x.code === 'unclosed-loop').message, /next\/prev disagree/);
});

test('validateBrep catches an EMPTY loop, a STALE vertex half-edge, BROKEN ownership, and an EULER-POINCARÉ violation', () => {
  const seed = mvfs([0, 0, 0]);
  assert.ok(hasErrorCode(validateBrep(seed.solid), 'empty-loop'));
  assert.ok(validateBrep(seed.solid, { allowIntermediate: true }).ok);

  const stale = buildBlock(60, 40, 25, () => {});
  stale.solid.vertices[0].halfEdge = null;
  assert.ok(hasErrorCode(validateBrep(stale.solid), 'stale-vertex-halfedge'));

  const owned = buildBlock(60, 40, 25, () => {});
  const foreign = buildBlock(1, 1, 1, () => {});
  allFaces(owned.solid)[0].shell = foreign.solid.shells[0];
  assert.ok(hasErrorCode(validateBrep(owned.solid), 'broken-ownership'));

  const lying = buildBlock(60, 40, 25, () => {});
  lying.solid.genus = 2;
  const res = validateBrep(lying.solid);
  assert.ok(hasErrorCode(res, 'euler-poincare'));
});

test('validateBrep on an intact solid reports zero errors — it is not a validator that always fires', () => {
  assert.deepEqual(validateBrep(buildBlock(60, 40, 25, () => {}).solid).errors, []);
  assert.deepEqual(validateBrep(buildWasher(() => {}).solid).errors, []);
});

// ---------------------------------------------------------------------
// 6. compatibility with the existing trimmed-surface + SSI models
// ---------------------------------------------------------------------

// A bilinear planar patch whose parameter domain IS its own (x, y) extent,
// so surfacePoint(u, v) = (u, v, z) exactly and the UV mapping below is not
// an approximation of anything.
function planarPatch(w, d, z) {
  return {
    degU: 1, degV: 1,
    knotsU: [0, 0, w, w],
    knotsV: [0, 0, d, d],
    ctrlNet: [
      [[0, 0, z, 1], [0, d, z, 1]],
      [[w, 0, z, 1], [w, d, z, 1]],
    ],
  };
}

test('a B-rep face hands its trim boundary to the EXISTING trimmed-surface model: derived loops pass trimLoopsValid, and tessellating them yields the true annular area', () => {
  const { topFace } = buildWasher(() => {});
  attachSurface(topFace, planarPatch(60, 40, 25));
  assignPlanarPcurves(topFace, (p) => [p[0], p[1]]);

  const ts = faceTrimmedSurface(topFace);
  assert.equal(ts.trimLoop.length, 4);
  assert.equal(ts.trimHoles.length, 1);
  assert.equal(ts.trimHoles[0].length, 4);

  // The face's own half-edge cycles, not a second stored copy, are what
  // produced these — the outer loop's corners really are the block's.
  const corners = new Set(ts.trimLoop.map((p) => p.join(',')));
  assert.deepEqual([...corners].sort(), ['0,0', '0,40', '60,0', '60,40']);

  const validity = trimLoopsValid([ts.trimLoop, ...ts.trimHoles]);
  assert.ok(validity.ok, `derived loops should satisfy the existing trim gate: ${validity.reason ?? ''}`);
  assert.ok(faceTrimValidity(topFace).ok);

  const tris = tessellateTrimmedSurface(ts.srf, ts.trimLoop, 24, 16, ts.trimHoles);
  const area = tessellationArea(tris);
  const expected = 60 * 40 - 30 * 20; // 1800
  assert.ok(Math.abs(area - expected) / expected < 0.005, `trimmed area ${area} should be within 0.5% of ${expected}`);
});

test('loopUVPolyline refuses honestly when a half-edge has no pcurve, rather than inventing one', () => {
  const { topFace } = buildWasher(() => {});
  attachSurface(topFace, planarPatch(60, 40, 25));
  assert.throws(() => faceTrimmedSurface(topFace), /has no pcurve/);
});

test('an edge and its two half-edges take their geometry directly from a real kernel/ssi.mjs intersection — the shapes line up with no adapter beyond field names', () => {
  // Two genuinely crossing patches: a horizontal plate and a vertical one.
  const plate = planarPatch(60, 40, 0);
  const wall = {
    degU: 1, degV: 1,
    knotsU: [0, 0, 60, 60],
    knotsV: [0, 0, 20, 20],
    ctrlNet: [
      [[0, 20, -10, 1], [0, 20, 10, 1]],
      [[60, 20, -10, 1], [60, 20, 10, 1]],
    ],
  };
  const hit = intersectSurfaces(plate, wall, { seedGrid: 12 });
  assert.ok(hit.ok, `expected a real intersection: ${hit.reason ?? ''}`);
  assert.ok(hit.samples.length >= 2);

  const geom = pcurvesFromSSISamples(hit.samples);
  assert.equal(geom.curve3d.length, hit.samples.length);
  assert.equal(geom.uv1.length, hit.samples.length);
  assert.equal(geom.uv2.length, hit.samples.length);

  // Every sampled 3D point really is on the shared line y = 20, z = 0, and
  // each pcurve really is in its OWN surface's parameter space (uv1 in the
  // plate's [0,60]x[0,40], uv2 in the wall's [0,60]x[0,20]).
  for (let i = 0; i < geom.curve3d.length; i++) {
    const [x, y, z] = geom.curve3d[i];
    assert.ok(Math.abs(y - 20) < 1e-3 && Math.abs(z) < 1e-3, `sample ${i} off the true intersection line: ${x},${y},${z}`);
    assert.ok(geom.uv1[i][0] >= -1e-6 && geom.uv1[i][0] <= 60 + 1e-6 && Math.abs(geom.uv1[i][1] - 20) < 1e-3);
    assert.ok(geom.uv2[i][0] >= -1e-6 && geom.uv2[i][0] <= 60 + 1e-6 && Math.abs(geom.uv2[i][1] - 10) < 1e-3);
  }

  // The slots those three arrays are destined for exist on a real edge, one
  // pcurve per side — Weiler's coedge, and the direct counterpart of
  // kernel/trim.mjs's own reserved edge3d/tolerance pair.
  const { solid, topLoop, topVerts } = buildBlock(60, 40, 25, () => {});
  const edge = solid.edges.find((e) => e.halfEdges[0].loop === topLoop || e.halfEdges[1].loop === topLoop);
  edge.curve3d = geom.curve3d;
  edge.tolerance = 1e-3;
  edge.halfEdges[0].pcurve = geom.uv1;
  edge.halfEdges[1].pcurve = geom.uv2;
  assert.equal(edge.halfEdges[0].pcurve.length, edge.halfEdges[1].pcurve.length);
  assert.notEqual(edge.halfEdges[0].pcurve, edge.halfEdges[1].pcurve, 'the two sides carry independent pcurves');
  assert.ok(topVerts.length === 4);
  assert.deepEqual(validateBrep(solid).errors, [], 'attaching geometry changes nothing topological');
});

test('pcurvesFromSSISamples refuses a degenerate sample list rather than returning a one-point "curve"', () => {
  assert.throws(() => pcurvesFromSSISamples([]), /at least 2 SSI samples/);
  assert.throws(() => pcurvesFromSSISamples([{ u1: 0, v1: 0, u2: 0, v2: 0, point: [0, 0, 0] }]), /at least 2 SSI samples/);
});

// ---------------------------------------------------------------------
// 7. operators refuse honestly rather than corrupting
// ---------------------------------------------------------------------

test('every operator refuses an inapplicable case by name instead of silently producing a broken solid', () => {
  const { solid, topLoop, topVerts, topFace, bottomFace } = buildBlock(60, 40, 25, () => {});

  // MEV into an ambiguous vertex sector
  const spur = mev(topLoop, topVerts[0], [-5, -5, 40]);
  assert.throws(() => mev(topLoop, topVerts[0], [-9, -9, 40]), /pass opts.before/);
  // KEV on an edge that is not a valence-1 spur
  assert.throws(() => kev(loopHalfEdges(topFace.loops[0]).find((h) => h.twin.loop !== h.loop)), /not a spur/);
  kev(spur.halfEdge);

  // MEF across two different loops
  assert.throws(() => mef(topLoop.halfEdge, bottomFace.loops[0].halfEdge), /same loop/);
  // KEF on a spur
  const spur2 = mev(topLoop, topVerts[1], [70, -5, 40]);
  assert.throws(() => kef(spur2.halfEdge), /both half-edges in one loop/);
  // KEMR on a valence-1 spur
  assert.throws(() => kemr(spur2.halfEdge), /valence-1 spur/);
  kev(spur2.halfEdge);

  // KFMRH on a face with rings, and on itself
  assert.throws(() => kfmrh(topFace, topFace), /must be different/);
  // MFKRH when there is no ring / no genus to give back
  assert.throws(() => mfkrh(topFace, topFace.loops[0]), /outer loop, not a ring/);

  assert.deepEqual(validateBrep(solid).errors, [], 'every refusal left the solid untouched');
});
