// OFFSET / THICKEN A SUBD CAGE. What is worth testing is not "the vertices
// moved" but the three things that make the result usable: a sheet ends up
// a slab that is genuinely closed and consistently wound, an already-closed
// cage is REFUSED rather than split into two disconnected nested shells, and
// the offset really is a normal-direction offset (not, say, a scale about
// the origin, which looks identical on a symmetric fixture and is wrong on
// every other one).
import test from 'node:test';
import assert from 'node:assert/strict';
import { offsetCage, thickenCage } from '../kernel/subdedit.mjs';
import { buildTopology, subdivideCatmullClark, edgeKey } from '../kernel/subd.mjs';
import { superbPlaneCage, superbBoxCage, superbTorusCage } from '../kernel/subdprimitives.mjs';

function report(cage) {
  const ctx = buildTopology(cage);
  let boundary = 0, nonManifold = 0;
  for (const e of ctx.edgeMap.values()) {
    if (e.faces.length === 1) boundary++;
    else if (e.faces.length > 2) nonManifold++;
  }
  return { boundary, nonManifold };
}
function windingConsistent(cage) {
  const seen = new Set();
  for (const f of cage.faces) for (let i = 0; i < f.length; i++) {
    const k = `${f[i]}>${f[(i + 1) % f.length]}`;
    if (seen.has(k)) return false;
    seen.add(k);
  }
  return true;
}

test('offsetting a flat sheet moves every vertex along its own normal, by exactly the distance', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const out = offsetCage(cage, 5);
  assert.equal(out.vertices.length, cage.vertices.length);
  assert.equal(out.faces.length, cage.faces.length);
  for (let i = 0; i < cage.vertices.length; i++) {
    const a = cage.vertices[i], b = out.vertices[i];
    assert.ok(Math.abs(b[0] - a[0]) < 1e-9 && Math.abs(b[1] - a[1]) < 1e-9, 'a flat sheet must not move sideways');
    assert.ok(Math.abs(Math.abs(b[2] - a[2]) - 5) < 1e-9, 'and must move exactly the offset distance');
  }
});

test('THE DISCRIMINATING ONE: offsetting a BOX grows it by the distance in every direction — a scale about the origin would grow it proportionally instead, which is a different (and wrong) answer', () => {
  // Deliberately OFF-ORIGIN and non-cubic, so "offset along normals" and
  // "scale about the origin" cannot coincide.
  const cage = superbBoxCage([100, 0, 0], [10, 20, 30], 1);
  const out = offsetCage(cage, 3);
  const extent = (c) => {
    const xs = c.vertices.map((v) => v[0]), ys = c.vertices.map((v) => v[1]), zs = c.vertices.map((v) => v[2]);
    return [Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), Math.max(...zs) - Math.min(...zs)];
  };
  const [ex0, ey0, ez0] = extent(cage);
  const [ex1, ey1, ez1] = extent(out);
  // A box corner's own vertex normal is the diagonal, so each extent grows
  // by 2 * distance * (1/sqrt(3)) — the same absolute amount on all three
  // axes, which is the offset signature. A scale would grow the 60mm axis
  // three times as much as the 20mm one.
  const gx = ex1 - ex0, gy = ey1 - ey0, gz = ez1 - ez0;
  assert.ok(Math.abs(gx - gy) < 1e-6 && Math.abs(gy - gz) < 1e-6, `every axis must grow by the same absolute amount, got ${gx}/${gy}/${gz}`);
  assert.ok(gx > 0, 'and it must grow outward, not shrink');
  const centre = (c) => [0, 1, 2].map((k) => (Math.max(...c.vertices.map((v) => v[k])) + Math.min(...c.vertices.map((v) => v[k]))) / 2);
  const c0 = centre(cage), c1 = centre(out);
  assert.ok(Math.hypot(c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2]) < 1e-9, 'and the box must not drift toward or away from the world origin');
});

test('a negative distance offsets INWARD', () => {
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const out = offsetCage(cage, -2);
  const r = (c) => Math.hypot(...c.vertices[0]);
  assert.ok(r(out) < r(cage), 'an inward offset must bring the corner closer to the center');
});

test('THICKEN turns a sheet into a genuinely closed, consistently wound slab', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const { cage: out } = thickenCage(cage, 4);
  assert.equal(out.vertices.length, cage.vertices.length * 2);
  assert.equal(out.faces.length, cage.faces.length * 2 + 8); // both sheets + one rim quad per naked edge
  assert.deepEqual(report(out), { boundary: 0, nonManifold: 0 });
  assert.equal(windingConsistent(out), true);
});

test('and the slab really subdivides — the downstream risk a wrongly-wound rim would only show up in', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 2);
  const { cage: out } = thickenCage(cage, 4);
  const refined = subdivideCatmullClark(out);
  for (const f of refined.faces) assert.equal(new Set(f).size, f.length);
  for (const v of refined.vertices) assert.ok(v.every(Number.isFinite));
  assert.deepEqual(report(refined), { boundary: 0, nonManifold: 0 });
  assert.equal(windingConsistent(refined), true);
});

// REWRITTEN, deliberately, not weakened. This test previously ASSERTED the
// two-nested-shells result as intended behavior. It is not: with no naked
// edge there is no rim, so one cage becomes two disconnected components —
// proven below by counting them, which the original never did. That is the
// thing reported live as "a superb inside a superb; just have to undo or
// delete", and it fails this app's own rule that an operation which cannot
// keep a SuperB one watertight object must refuse and say why.
function connectedComponentCount(cage) {
  const adj = new Map();
  const link = (a, b) => { if (!adj.has(a)) adj.set(a, new Set()); adj.get(a).add(b); };
  for (const f of cage.faces) for (let i = 0; i < f.length; i++) { const a = f[i], b = f[(i + 1) % f.length]; link(a, b); link(b, a); }
  const seen = new Set(); let n = 0;
  for (const v of adj.keys()) {
    if (seen.has(v)) continue;
    n++; const stack = [v]; seen.add(v);
    while (stack.length) { const x = stack.pop(); for (const y of adj.get(x) || []) if (!seen.has(y)) { seen.add(y); stack.push(y); } }
  }
  return n;
}

test('thickening an ALREADY-CLOSED cage is REFUSED by name — there is no open edge for a wall to grow from', () => {
  for (const cage of [superbBoxCage([0, 0, 0], [10, 10, 10], 1), superbTorusCage([0, 0, 0], 30, 10, 8)]) {
    assert.equal(connectedComponentCount(cage), 1, 'the cage starts as ONE connected object');
    assert.throws(() => thickenCage(cage, -2), /already closed/, 'a closed cage must refuse, naming the reason');
    assert.throws(() => thickenCage(cage, 2), /Delete a face first/, 'and must name the real escape hatch, not just say no');
  }
});

test('the refusal is exactly right: producing it anyway WOULD split one object into two disconnected shells', () => {
  // Proves the refusal is protecting against something real rather than
  // being conservative — reconstructs precisely what thickenCage's own body
  // would have built for a closed cage (original + reversed offset copy, no
  // rim, since no edge has a single owner face) and counts the components.
  const cage = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const offset = offsetCage(cage, -2);
  const n = cage.vertices.length;
  const wouldBe = {
    vertices: [...cage.vertices, ...offset.vertices],
    faces: [...cage.faces.map((f) => f.slice()), ...cage.faces.map((f) => f.slice().reverse().map((vi) => vi + n))],
    creases: {},
  };
  assert.equal(connectedComponentCount(wouldBe), 2, 'one object would have become two disconnected nested shells — the reported "superb inside a superb"');
  assert.deepEqual(report(wouldBe), { boundary: 0, nonManifold: 0 }, 'and each shell is individually watertight, which is exactly why this went unnoticed');
});

test('an OPEN cage still thickens into a real single-component wall — the refusal is scoped to the closed case only', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 1);
  const { cage: out, rimFaceIndices } = thickenCage(cage, 4);
  assert.ok(rimFaceIndices.length > 0, 'a real rim was built');
  assert.equal(connectedComponentCount(out), 1, 'and the result is still ONE connected object, not two');
  assert.deepEqual(report(out), { boundary: 0, nonManifold: 0 });
  assert.equal(windingConsistent(out), true);
});

test('creases ride along onto the offset copy — a hard edge stays hard on both sheets', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 1);
  const k = edgeKey(cage.faces[0][0], cage.faces[0][1]);
  const creased = { ...cage, creases: { [k]: 2 } };
  const { cage: out } = thickenCage(creased, 3);
  const n = cage.vertices.length;
  const [a, b] = k.split('_').map(Number);
  assert.equal(out.creases[k], 2);
  assert.equal(out.creases[edgeKey(a + n, b + n)], 2);
});

test('honest refusals, and the input cage is never mutated', () => {
  const cage = superbPlaneCage([0, 0, 0], 40, 40, 1);
  const before = JSON.stringify(cage);
  assert.throws(() => offsetCage(cage, 0), /nonzero finite/);
  assert.throws(() => thickenCage(cage, Number.NaN), /nonzero finite/);
  offsetCage(cage, 5);
  thickenCage(cage, 5);
  assert.equal(JSON.stringify(cage), before);
});
