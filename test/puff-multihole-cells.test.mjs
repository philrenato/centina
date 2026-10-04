// Two holes that each puff alone puff together.
//
// The multi-hole puff parts the region into one cell per hole and rings each cell with spokes from
// a point inside its hole. Two drawings, traced: an outline with a deep notch whose two walls both
// fall in the right hole's cell (no point of that hole sees the whole cell), and a kidney-shaped
// hole whose own notch hides part of its rim from its centroid. Each hole alone builds; together
// they must build too, cleanly, and keep building when the holes are nudged and turned a little
// (puff-multihole-nudges-notch, -kidney).
// What still refuses names its hole, and beside it the unmoved drawing builds.
//
// And they come out as one fluid form: no line where the two holes' cells meet (the crease score
// of puff-limit-quality), and rims on the drawn curves, both ways, closer than a single-hole puff
// typically lands.
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { buildMultiHolePuff, buildAnnularPuff, classifyPuffLoops, puffInvariants, puffLimitRimMiss } from '../kernel/puff.mjs';
import { subdivideCatmullClark } from '../kernel/subd.mjs';
import { selfCrossings } from './fixtures/mesh-crossings.mjs';
import { NOTCH, KIDNEY, loopsOf, jitterHole, spanOf } from './fixtures/puff_two_hole_cells.mjs';
import { creaseScore, rimMissBothWays, singleHoleMiss, circle } from './fixtures/puff-limit-quality.mjs';

const SINGLE_HOLE_MISS = singleHoleMiss();

const toMesh = (r) => {
  const V = []; for (let i = 0; i < r.nv; i++) V.push([r.positions[3 * i], r.positions[3 * i + 1], r.positions[3 * i + 2]]);
  const F = []; for (let i = 0; i < r.quads.length; i += 4) F.push([r.quads[i], r.quads[i + 1], r.quads[i + 2], r.quads[i + 3]]);
  return { V, F };
};
const nudged = (fx, jit) => {
  const { outer, holes } = loopsOf(fx, fx.view.sy, fx.view.persp);
  const span = spanOf(outer);
  return { outer, holes: holes.map((h, k) => (jit[k] ? jitterHole(h, span, ...jit[k]) : h)) };
};
function assertClean(name, outer, holes, opts = {}) {
  const r = buildMultiHolePuff(outer, holes, opts);
  assert.ok(r.ok, `${name}: builds (${r.reason}: ${r.why})`);
  const inv = puffInvariants(r);
  assert.ok(inv.ok, `${name}: invariants ${JSON.stringify(inv)}`);
  assert.equal(inv.euler, 2 - 2 * holes.length, `${name}: genus`);
  const { V, F } = toMesh(r);
  const s = subdivideCatmullClark(subdivideCatmullClark({ vertices: V, faces: F }));
  assert.equal(selfCrossings(s.vertices, s.faces), 0, `${name}: the limit crosses itself`);
  const miss = puffLimitRimMiss(r, [outer, ...holes]);
  assert.ok(miss.worst < SINGLE_HOLE_MISS, `${name}: limit rim misses its curves by ${miss.worst.toFixed(4)} of the span, a single-hole puff ${SINGLE_HOLE_MISS.toFixed(4)}`);
  return r;
}


// The drawing read at three unprojections of the perspective it was traced from, and the kidney
// at three nudges that the centroid-spoked ring could not mesh.
const FAMILY = [
  ['notch', NOTCH, null, null, null],
  ['notch, flat read', NOTCH, 1, 0, null],
  ['notch, deep read', NOTCH, 1.84, 0.4, null],
  ['kidney', KIDNEY, null, null, null],
  ['kidney nudge a', KIDNEY, null, null, [[-0.0422, -0.0093, -0.2079], [-0.0422, 0.0496, -0.168]]],
  ['kidney nudge b', KIDNEY, null, null, [[-0.0103, -0.0084, -0.2633], [-0.027, 0.0308, 0.2846]]],
  ['kidney nudge c', KIDNEY, null, null, [[-0.0087, -0.0409, -0.1595], [0.0424, 0.0277, 0.1584]]],
];

for (const [name, fx, sy, persp, jit] of FAMILY) {
  test(`${name}: each hole alone and both together`, () => {
    let { outer, holes } = sy == null ? loopsOf(fx, fx.view.sy, fx.view.persp) : loopsOf(fx, sy, persp);
    if (jit) ({ outer, holes } = nudged(fx, jit));
    assert.ok(classifyPuffLoops([outer, ...holes]).ok, `${name}: an outline with two holes`);
    holes.forEach((h, k) => assert.ok(buildAnnularPuff(outer, [h]).ok, `${name}: hole ${k + 1} alone builds`));
    assertClean(name, outer, holes);
  });
}

test('notch at densities 3 and 5', () => {
  const { outer, holes } = loopsOf(NOTCH, NOTCH.view.sy, NOTCH.view.persp);
  for (const density of [3, 5]) assertClean(`notch density ${density}`, outer, holes, { density });
});

test('flat back on the notch', () => {
  const { outer, holes } = loopsOf(NOTCH, NOTCH.view.sy, NOTCH.view.persp);
  for (const bottomScale of [0, -1]) {
    const r = buildMultiHolePuff(outer, holes, { bottomScale });
    assert.ok(r.ok, `Underside ${bottomScale} builds`);
    assert.ok(puffInvariants(r).ok, `Underside ${bottomScale}: invariants`);
  }
});

// A hole pushed in under the notch, close to the other hole: no straight cut both parts the two
// holes and splits the notch's walls between them. It refuses by that hole, with the reason; the
// drawing it was moved from builds (above).
test('a refusal names its hole', () => {
  const { outer, holes } = nudged(NOTCH, [[0.088, -0.021, 0.25], [-0.05, 0.017, -0.179]]);
  holes.forEach((h, k) => assert.ok(buildAnnularPuff(outer, [h]).ok, `hole ${k + 1} alone builds`));
  const r = buildMultiHolePuff(outer, holes);
  assert.equal(r.ok, false);
  assert.equal(r.hole, 1, 'the right hole');
  assert.equal(r.reason, 'cell-not-star');
  assert.match(r.tail, /notch/);
  assert.equal(r.why, `hole 2 ${r.tail}`);
  assert.ok(('Puff — Curve12 ' + r.tail).length <= 71, 'the status line fits');
});

// The cut between two cells is a construction line; on the limit it must not read as one. The
// crease score is the sharpest turn of the highlight anywhere on the crown, per unit length and
// scaled by the half-width of the crown or bridge it is on: a round single-hole ring reads 1.2.
const SEAM_CASES = [
  ['notch', () => loopsOf(NOTCH, NOTCH.view.sy, NOTCH.view.persp)],
  ['kidney', () => loopsOf(KIDNEY, KIDNEY.view.sy, KIDNEY.view.persp)],
  ['two round holes', () => ({ outer: circle(40, 128), holes: [circle(8, 64, -15, 0), circle(8, 64, 15, 0)] })],
];
for (const [name, make] of SEAM_CASES) {
  test(`${name}: no line where the cells meet, rims on the curves`, () => {
    const { outer, holes } = make();
    const r = buildMultiHolePuff(outer, holes);
    assert.ok(r.ok, `${name} builds`);
    const c = creaseScore(r, [outer, ...holes]);
    assert.ok(c.max < 4.5, `${name}: sharpest crown bend ${c.max.toFixed(2)} at (${c.at.map((v) => v.toFixed(0))}) < 4.5`);
    assert.ok(c.p99 < 2.5, `${name}: 99th-percentile bend ${c.p99.toFixed(2)} < 2.5`);
    const miss = rimMissBothWays(r, [outer, ...holes]);
    assert.ok(miss < 0.0075 && miss < SINGLE_HOLE_MISS, `${name}: rims off their curves by ${miss.toFixed(4)} of the span, both ways (< 0.0075; a single-hole puff ${SINGLE_HOLE_MISS.toFixed(4)})`);
  });
}
