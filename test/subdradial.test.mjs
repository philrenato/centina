// RADIAL SYMMETRY — the kernel primitive. What is worth proving here is not "it returned an array" but
// four things: the GEOMETRIC search agrees with an INDEPENDENT closed-form
// derivation on every vertex; the order check is genuinely an order check
// (not an involution check wearing a different name); the on-axis case is
// handled explicitly rather than silently collapsing the measured order to
// 1; and every refusal is a real refusal with the reason in it.
//
// THE STRONGEST ASSERTION IN THIS FILE is the index cross-check. A SuperB
// torus cage is an nU x nV grid addressed by idx(i,j) = i*nV + j with
// wrapping, so a rotation by k ring steps has a closed form that owes
// NOTHING to the geometric search under test. Two independent derivations
// agreeing on all 48 vertices proves both the search and its tolerance.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  findCageRotationPartners, verifyRotationOrder, orbitFromSeed,
  rotateDelta, rotationFaceIndex, gcd, orbitOrder, divisorsOf, strideForCount,
} from '../kernel/subdradial.mjs';
import { superbTorusCage, superbConeCage, superbBoxCage } from '../kernel/subdprimitives.mjs';

const Z = [0, 0, 1];
const O = [0, 0, 0];
const TAU = Math.PI * 2;

test('gcd / orbitOrder / divisorsOf — the arithmetic behind the one genuinely surprising behavior', () => {
  assert.equal(orbitOrder(12, 3), 4, 'every 3rd of 12 is a 4-member orbit');
  assert.equal(orbitOrder(12, 4), 3);
  assert.equal(orbitOrder(12, 5), 12, 'every 5th of 12 wraps and hits ALL twelve — gcd 1');
  assert.equal(orbitOrder(10, 3), 10, 'every 3rd of 10 hits ALL ten, which is the surprise worth stating out loud');
  assert.equal(orbitOrder(12, 6), 2);
  assert.deepEqual(divisorsOf(12), [1, 2, 3, 4, 6, 12], 'a refusal can name the orbit sizes this ring CAN support');
  assert.equal(gcd(12, 8), 4);
});

test('THE INDEX CROSS-CHECK: the geometric search reproduces the closed-form torus rotation on EVERY vertex', () => {
  const nU = 12, nV = 4, k = 3;
  const cage = superbTorusCage(O, 30, 10, nU, nV);
  assert.equal(cage.vertices.length, nU * nV);
  const { next, onAxisCount, tolerance } = findCageRotationPartners(cage, O, Z, TAU * k / nU);
  assert.equal(onAxisCount, 0, 'a torus has no vertex on its own ring axis');
  assert.ok(tolerance > 0);
  // Derived independently from superbTorusCage's own documented addressing
  // (idx(i,j) = i*nV + j, wrapping in both directions) — this formula never
  // consults the search under test.
  for (let v = 0; v < nU * nV; v++) {
    const expected = ((Math.floor(v / nV) + k) % nU) * nV + (v % nV);
    assert.equal(next[v], expected, `vertex ${v} must map to ${expected}, the closed-form answer`);
  }
});

test('the order check is a REAL order check — closes at exactly n, and at no smaller number of steps', () => {
  const cage = superbTorusCage(O, 30, 10, 12, 4);
  const { next } = findCageRotationPartners(cage, O, Z, TAU * 3 / 12);
  const { order, onAxisCount } = verifyRotationOrder(next, 4);
  assert.equal(order, 4);
  assert.equal(onAxisCount, 0);
  // And it genuinely REFUSES a wrong order rather than shrugging — naming
  // the real order it found, which is what makes the message useful.
  assert.throws(() => verifyRotationOrder(next, 3), /returns to itself after|does not return/);
  assert.throws(() => verifyRotationOrder(next, 6), /returns to itself after 4 steps, not 6/);
});

test('the gcd surprise is real geometry, not just arithmetic: stride 5 on a 12-ring closes at 12, not 2', () => {
  const cage = superbTorusCage(O, 30, 10, 12, 4);
  const { next } = findCageRotationPartners(cage, O, Z, TAU * 5 / 12);
  assert.equal(orbitOrder(12, 5), 12);
  assert.doesNotThrow(() => verifyRotationOrder(next, 12));
  assert.throws(() => verifyRotationOrder(next, 2), /not 2|does not return/);
  // A stride-5 orbit visits every one of the twelve ring positions.
  const slots = orbitFromSeed(next, 0, 12);
  assert.equal(new Set(slots).size, 12, 'twelve genuinely distinct members');
});

test('orbitFromSeed returns an ORDERED ring, and refuses a chain that does not close', () => {
  const nU = 12, nV = 4, k = 3;
  const cage = superbTorusCage(O, 30, 10, nU, nV);
  const { next } = findCageRotationPartners(cage, O, Z, TAU * k / nU);
  const slots = orbitFromSeed(next, 0, 4);
  assert.equal(slots.length, 4);
  assert.equal(slots[0], 0, 'the seed is slot 0 — position in the array IS the group element');
  // Slot j must sit exactly j rotation steps around, by the closed form.
  for (let j = 0; j < 4; j++) {
    assert.equal(slots[j], ((0 + j * k) % nU) * nV, `slot ${j} sits ${j} steps around`);
  }
  assert.throws(() => orbitFromSeed(next, 0, 5), /did not close|closed after/);
});

test('ON-AXIS vertices are excluded EXPLICITLY — a cone apex must not collapse the measured order to 1', () => {
  // A cone's apex sits exactly on its own axis, so it is fixed by every
  // rotation. Without the explicit exclusion, the order check would see it
  // "return to itself after 1 step" and wrongly conclude order 1.
  const cage = superbConeCage(O, 25, 50, 8);
  const { next, onAxisCount } = findCageRotationPartners(cage, O, Z, TAU / 8);
  assert.ok(onAxisCount >= 1, 'the apex is genuinely reported as on-axis');
  const selfMapped = next.filter((p, i) => p === i).length;
  assert.equal(selfMapped, onAxisCount);
  // The real order is still 8 despite the fixed apex — this is the assertion
  // that would fail if on-axis vertices were not excluded.
  assert.doesNotThrow(() => verifyRotationOrder(next, 8), 'an on-axis vertex must not drag the order down to 1');
  // And an on-axis element honestly refuses to have an orbit of its own.
  const apex = next.findIndex((p, i) => p === i);
  assert.throws(() => orbitFromSeed(next, apex, 8), /lies ON the rotation axis/);
});

test('honest refusals: a non-symmetric cage, a non-dividing angle, and a zero angle', () => {
  const box = superbBoxCage(O, [20, 20, 20], 1);
  // A box IS symmetric under 90 deg about Z, so that must NOT refuse —
  // proving the refusals below are about real asymmetry, not a blanket no.
  assert.doesNotThrow(() => findCageRotationPartners(box, O, Z, Math.PI / 2));
  // 45 deg is genuinely not a symmetry of a box.
  assert.throws(() => findCageRotationPartners(box, O, Z, Math.PI / 4), /not rotationally symmetric/);
  assert.throws(() => findCageRotationPartners(box, O, Z, 0), /nonzero finite/);
  // An angle that does not divide the torus's own ring count.
  const cage = superbTorusCage(O, 30, 10, 12, 4);
  assert.throws(() => findCageRotationPartners(cage, O, Z, TAU / 5), /not rotationally symmetric/);
});

test('the refusal NAMES the offending vertex and its real position, so it is actionable', () => {
  const box = superbBoxCage(O, [20, 20, 20], 1);
  try {
    findCageRotationPartners(box, O, Z, Math.PI / 4);
    assert.fail('should have refused');
  } catch (e) {
    assert.match(e.message, /vertex \d+ at \[/, 'names the vertex and its coordinates');
    assert.match(e.message, /tolerance/, 'and the tolerance it was judged against');
  }
});

test('rotateDelta is the conjugation rule — a free vector, rotated by k steps, independent of position', () => {
  const d = [5, 0, 0];
  const quarter = TAU / 4;
  const one = rotateDelta(d, Z, quarter, 1);
  assert.ok(Math.abs(one[0] - 0) < 1e-9 && Math.abs(one[1] - 5) < 1e-9, `a +X delta rotated 90 deg about Z is +Y, got ${JSON.stringify(one)}`);
  const two = rotateDelta(d, Z, quarter, 2);
  assert.ok(Math.abs(two[0] + 5) < 1e-9 && Math.abs(two[1]) < 1e-9, 'two steps is 180 deg');
  // Four steps returns it exactly — the property that makes one-undo-step
  // for a whole orbit consistent.
  const four = rotateDelta(d, Z, quarter, 4);
  for (let c = 0; c < 3; c++) assert.ok(Math.abs(four[c] - d[c]) < 1e-9, 'four quarter steps is the identity');
});

test('rotationFaceIndex maps a face to its counterpart, and returns null rather than guessing', () => {
  const nU = 12, nV = 4, k = 3;
  const cage = superbTorusCage(O, 30, 10, nU, nV);
  const { next } = findCageRotationPartners(cage, O, Z, TAU * k / nU);
  // superbTorusCage pushes faces in the same i-outer/j-inner order as its
  // vertices, so the face counterpart has the SAME closed form — an
  // independent check again, not a restatement of the function.
  for (const f of [0, 1, 7, 23, 47]) {
    const expected = ((Math.floor(f / nV) + k) % nU) * nV + (f % nV);
    assert.equal(rotationFaceIndex(cage, next, f), expected, `face ${f} maps to ${expected}`);
  }
});

test('the correspondence is a genuine PERMUTATION — two vertices can never collapse onto one', () => {
  const cage = superbTorusCage(O, 30, 10, 12, 4);
  const { next } = findCageRotationPartners(cage, O, Z, TAU * 3 / 12);
  assert.equal(new Set(next).size, next.length, 'every vertex is the image of exactly one vertex');
});

test('an asymmetric U/V torus still works — the search does not assume a square grid', () => {
  const nU = 9, nV = 5, k = 3;
  const cage = superbTorusCage(O, 40, 12, nU, nV);
  const { next } = findCageRotationPartners(cage, O, Z, TAU * k / nU);
  for (let v = 0; v < nU * nV; v++) {
    assert.equal(next[v], ((Math.floor(v / nV) + k) % nU) * nV + (v % nV));
  }
  assert.doesNotThrow(() => verifyRotationOrder(next, orbitOrder(nU, k)));
});

// COUNT <-> STRIDE, the two-way pair the Properties/UnWire rows are built
// on. The interesting assertion is the LAST one: the round trip is not the
// identity, and the tests prove that is harmless rather than asserting the
// identity and quietly weakening the check.
test('strideForCount is the other direction, and refuses a count the ring cannot make', () => {
  assert.equal(strideForCount(12, 4), 3, 'a 4-member orbit on a 12-ring is every 3rd');
  assert.equal(strideForCount(12, 6), 2);
  assert.equal(strideForCount(12, 12), 1);
  assert.throws(() => strideForCount(12, 5), /cannot make a 5-member orbit/);
  assert.throws(() => strideForCount(12, 5), /2, 3, 4, 6, 12/, 'and names the sizes that DO work');
  assert.throws(() => strideForCount(12, 1), /at least 2 members/);
});

test('the count<->stride round trip is NOT the identity — and that is provably harmless', () => {
  // Stride 5 on a 12-ring gives 12 members; asking for 12 members gives back
  // stride 1, not 5. If slot labelling mattered, this would be a real bug.
  assert.equal(orbitOrder(12, 5), 12);
  assert.equal(strideForCount(12, 12), 1);
  // It does not matter, because both strides select the SAME member set at
  // the SAME true angular positions — proven here against the real cage,
  // not argued.
  const cage = superbTorusCage(O, 30, 10, 12, 4);
  const setFor = (s) => {
    const { next } = findCageRotationPartners(cage, O, Z, TAU * s / 12);
    return orbitFromSeed(next, 0, orbitOrder(12, s)).slice().sort((a, b) => a - b).join(',');
  };
  assert.equal(setFor(5), setFor(1), 'stride 5 and stride 1 select the identical members');
  // And every round trip that CAN be exact, is.
  for (const c of [2, 3, 4, 6, 12]) {
    assert.equal(orbitOrder(12, strideForCount(12, c)), c, `count ${c} round-trips exactly`);
  }
});
