import test from 'node:test';
import assert from 'node:assert/strict';
import { refineToCount, harmonizeDirections, countIn } from '../kernel/surfaceknots.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { extrude, makeCircle, makeSquircle2D } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';

// ADDING CONTROL POINTS IS FREE, and this file is the reason that sentence is
// allowed to be said. Knot insertion rewrites a surface's DESCRIPTION and moves
// no point of the surface — so bringing a coarse side up to a fine one to make
// two edges meet costs a denser net and nothing else. If that is ever untrue,
// forcing a match silently deforms the student's model, which is far worse than
// the refusal it replaces.
const profile = (n) => {
  const pts = [];
  for (let i = 0; i < n; i++) pts.push([i * 10, Math.sin(i) * 4, 0]);
  return globalCurveInterp(pts, 3);
};

function maxDeviation(a, b, n = 24) {
  let worst = 0;
  for (let i = 0; i <= n; i++) {
    for (let j = 0; j <= n; j++) {
      const p = surfacePoint(a, i / n, j / n);
      const q = surfacePoint(b, i / n, j / n);
      worst = Math.max(worst, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
    }
  }
  return worst;
}

test('refineToCount reaches the asked-for count', () => {
  const srf = extrude(profile(5), [0, 0, 1], 20);
  for (const target of [7, 9, 12, 20]) {
    const out = refineToCount(srf, 'u', target);
    assert.equal(countIn(out, 'u'), target, `asked for ${target}`);
  }
});

test('...and moves no point of the surface while doing it', () => {
  for (const [name, srf, dir] of [
    ['extruded interp curve', extrude(profile(5), [0, 0, 1], 20), 'u'],
    ['extruded circle', extrude(makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 20), [0, 0, 1], 30), 'u'],
    ['extruded squircle', extrude(makeSquircle2D([0, 0, 0], [1, 0, 0], [0, 1, 0], 20, 20, 0.5), [0, 0, 1], 30), 'u'],
    ['across the extrusion', extrude(profile(6), [0, 0, 1], 20), 'v'],
  ]) {
    const dense = refineToCount(srf, dir, countIn(srf, dir) + 6);
    assert.ok(countIn(dense, dir) === countIn(srf, dir) + 6, `${name}: count grew`);
    const dev = maxDeviation(srf, dense);
    assert.ok(dev < 1e-9, `${name}: refining moved the surface by ${dev}`);
  }
});

test('the new points are spread, not piled into one corner', () => {
  // Widest-span-first is what keeps a forced net draggable; splitting
  // arbitrarily would put every new row in one place.
  const srf = extrude(profile(4), [0, 0, 1], 20);
  const dense = refineToCount(srf, 'u', 10);
  const k = dense.knotsU.slice(dense.degU, dense.knotsU.length - dense.degU);
  const spans = [];
  for (let i = 0; i + 1 < k.length; i++) if (k[i + 1] - k[i] > 1e-9) spans.push(k[i + 1] - k[i]);
  const ratio = Math.max(...spans) / Math.min(...spans);
  assert.ok(ratio < 4, `widest span is ${ratio.toFixed(2)}x the narrowest — the refinement bunched up`);
});

test('harmonizeDirections refuses by default and says the refusal can be forced', () => {
  // A deliberately hopeless pair for the union: identical knots are impossible
  // to reach when one side is asked to keep a count the other cannot match.
  const a = extrude(profile(5), [0, 0, 1], 20);
  const b = extrude(profile(7), [0, 0, 1], 20);
  const ok = harmonizeDirections(a, 'u', b, 'u');
  assert.equal(ok.ok, true, 'an ordinary pair still harmonises without forcing');
  assert.equal(countIn(ok.a, 'u'), countIn(ok.b, 'u'));
  assert.ok(!ok.forced, 'and it did not need to be forced');
});

test('forcing brings two counts together and leaves both surfaces where they were', () => {
  const a = extrude(profile(5), [0, 0, 1], 20);
  const b = refineToCount(extrude(profile(7), [0, 0, 1], 20), 'u', 14);
  const forced = harmonizeDirections(a, 'u', b, 'u', { force: true });
  assert.equal(forced.ok, true);
  assert.equal(countIn(forced.a, 'u'), countIn(forced.b, 'u'), 'counts agree after forcing');
  assert.ok(maxDeviation(a, forced.a) < 1e-9, 'the first surface did not move');
  assert.ok(maxDeviation(b, forced.b) < 1e-9, 'the second surface did not move');
});
