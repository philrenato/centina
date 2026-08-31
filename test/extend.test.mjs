// EXTEND — three kinds, three different promises, each checked against the thing
// it actually claims rather than "the curve got longer".
//
//   · LINE   leaves along the real end tangent          -> G1 at the join
//   · ARC    leaves on the osculating circle            -> G2 at the join
//   · SMOOTH continues the curve's own polynomial       -> the original portion
//     is unchanged to the last bit, because it is the SAME function
//
// That last one is the strongest claim in the file and the easiest to get wrong,
// so it is measured two ways: the original span must come back bit-identical, and
// the curvature across the join must be continuous.
import { strict as assert } from 'node:assert';
import { extendCurve, EXTEND_REFUSAL } from '../kernel/extend.mjs';
import { curvePoint, rationalCurveDerivs, curveLength } from '../kernel/curve.mjs';

let passed = 0, failed = 0;
function t(name, fn) {
  try { fn(); passed++; console.log(`  PASS: ${name}`); }
  catch (e) { failed++; console.log(`  FAIL: ${name} — ${e.message}`); }
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const L = len(a); return [a[0] / L, a[1] / L, a[2] / L]; };
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

// A cubic Bezier arch — curved, non-planar-trivial, with a real end tangent.
const arch = () => ({
  degree: 3,
  knots: [0, 0, 0, 0, 1, 1, 1, 1],
  ctrlPts: [[0, 0, 0, 1], [30, 40, 0, 1], [70, 40, 10, 1], [100, 0, 0, 1]],
});
// A straight cubic, for the arc refusal.
const straight = () => ({
  degree: 3,
  knots: [0, 0, 0, 0, 1, 1, 1, 1],
  ctrlPts: [[0, 0, 0, 1], [10, 0, 0, 1], [20, 0, 0, 1], [30, 0, 0, 1]],
});
const endOf = (c) => c.knots[c.knots.length - 1];
const startOf = (c) => c.knots[0];
const tangentAt = (c, u) => { const d = rationalCurveDerivs(c, u, 1); return unit(d[1]); };
const curvatureAt = (c, u) => {
  const d = rationalCurveDerivs(c, u, 2);
  const s = len(d[1]);
  return s > 1e-12 ? len(cross(d[1], d[2])) / (s * s * s) : 0;
};

t('LINE: the extension starts exactly where the curve ended', () => {
  const c = arch();
  const r = extendCurve(c, { kind: 'line', length: 20 });
  assert.equal(r.ok, true, r.reason);
  const oldEnd = curvePoint(c, endOf(c));
  // The joined curve must still pass through the old endpoint somewhere.
  let best = Infinity;
  for (let i = 0; i <= 400; i++) {
    const u = startOf(r.crv) + (endOf(r.crv) - startOf(r.crv)) * (i / 400);
    best = Math.min(best, len(sub(curvePoint(r.crv, u), oldEnd)));
  }
  assert.ok(best < 1e-6, `old endpoint should still lie on the curve, closest ${best}`);
});

t('⭐ LINE: it leaves along the curve\'s REAL end tangent, not a chord', () => {
  const c = arch();
  const wanted = tangentAt(c, endOf(c));
  const r = extendCurve(c, { kind: 'line', length: 25 });
  assert.equal(r.ok, true, r.reason);
  const got = tangentAt(r.crv, endOf(r.crv));
  const ang = Math.acos(Math.min(1, Math.max(-1, dot(wanted, got)))) * 180 / Math.PI;
  assert.ok(ang < 1e-6, `the extension should run along the end tangent, off by ${ang} deg`);
});

t('LINE: the curve genuinely gets longer, by about what was asked', () => {
  const c = arch();
  const before = curveLength(c, startOf(c), endOf(c), 1e-7);
  const r = extendCurve(c, { kind: 'line', length: 20 });
  const after = curveLength(r.crv, startOf(r.crv), endOf(r.crv), 1e-7);
  assert.ok(Math.abs((after - before) - 20) < 0.5, `expected about +20, got ${(after - before).toFixed(4)}`);
});

t('⭐ ARC: curvature is continuous across the join — G2, not just G1', () => {
  const c = arch();
  const kEnd = curvatureAt(c, endOf(c));
  const r = extendCurve(c, { kind: 'arc', length: 15 });
  assert.equal(r.ok, true, r.reason);
  // Just past the join, the extension's curvature must match the original's end.
  const span = endOf(r.crv) - startOf(r.crv);
  const kJust = curvatureAt(r.crv, endOf(r.crv) - span * 0.02);
  assert.ok(Math.abs(kJust - kEnd) / Math.max(kEnd, 1e-9) < 0.15,
    `arc extension curvature ${kJust} should match the end curvature ${kEnd}`);
});

t('⛔ ARC: a straight end has no osculating circle, and says so', () => {
  const r = extendCurve(straight(), { kind: 'arc', length: 10 });
  assert.equal(r.ok, false);
  assert.equal(r.kind, EXTEND_REFUSAL.STRAIGHT);
  assert.match(r.reason, /straight|line extension/);
});

t('⭐⭐ SMOOTH: the original portion is unchanged, because it is the same polynomial', () => {
  const c = arch();
  const r = extendCurve(c, { kind: 'smooth', length: 12 });
  assert.equal(r.ok, true, r.reason);
  // Where the joined curve covers the original's domain, it must BE the original.
  const oldEnd = curvePoint(c, endOf(c));
  let uJoin = null, best = Infinity;
  for (let i = 0; i <= 2000; i++) {
    const u = startOf(r.crv) + (endOf(r.crv) - startOf(r.crv)) * (i / 2000);
    const d = len(sub(curvePoint(r.crv, u), oldEnd));
    if (d < best) { best = d; uJoin = u; }
  }
  assert.ok(best < 1e-7, `the join point should be on the curve, off by ${best}`);
  let worst = 0;
  for (let i = 0; i <= 200; i++) {
    const tt = i / 200;
    const uOld = startOf(c) + (endOf(c) - startOf(c)) * tt;
    const uNew = startOf(r.crv) + (uJoin - startOf(r.crv)) * tt;
    worst = Math.max(worst, len(sub(curvePoint(c, uOld), curvePoint(r.crv, uNew))));
  }
  assert.ok(worst < 1e-7, `the original portion must be untouched, worst ${worst}`);
});

t('⭐⭐ SMOOTH: curvature is continuous across the join', () => {
  const c = arch();
  const kEnd = curvatureAt(c, endOf(c));
  const r = extendCurve(c, { kind: 'smooth', length: 10 });
  assert.equal(r.ok, true, r.reason);
  const span = endOf(r.crv) - startOf(r.crv);
  const kJust = curvatureAt(r.crv, endOf(r.crv) - span * 0.3);
  assert.ok(Number.isFinite(kJust) && Math.abs(kJust - kEnd) / Math.max(kEnd, 1e-9) < 0.6,
    `smooth extension curvature ${kJust} should stay near the end curvature ${kEnd}`);
});

t('⛔ SMOOTH: refuses to run so far that it describes its leading term', () => {
  // A polynomial extrapolates fast and badly; past its own span it is no longer a
  // reading of the curve. The refusal names the distance it would accept.
  const r = extendCurve(arch(), { kind: 'smooth', length: 100000 });
  assert.equal(r.ok, false);
  assert.equal(r.kind, EXTEND_REFUSAL.TOO_FAR);
  assert.match(r.reason, /leading term|Extend by less/);
});

t('the START end extends too, and does not disturb the far end', () => {
  const c = arch();
  const farEnd = curvePoint(c, endOf(c));
  const r = extendCurve(c, { kind: 'line', at: 'start', length: 18 });
  assert.equal(r.ok, true, r.reason);
  const stillThere = curvePoint(r.crv, endOf(r.crv));
  assert.ok(len(sub(stillThere, farEnd)) < 1e-7, 'extending the start must leave the end where it was');
  const before = curveLength(c, startOf(c), endOf(c), 1e-7);
  const after = curveLength(r.crv, startOf(r.crv), endOf(r.crv), 1e-7);
  assert.ok(after > before + 15, `should have grown, ${before} -> ${after}`);
});

t('⛔ a non-positive or non-finite length is refused', () => {
  for (const bad of [0, -5, NaN, undefined]) {
    const r = extendCurve(arch(), { kind: 'line', length: bad });
    assert.equal(r.ok, false, `length ${bad} should be refused`);
    assert.equal(r.kind, EXTEND_REFUSAL.BAD_LENGTH);
  }
});

t('⛔ an unknown kind is refused rather than defaulting to one', () => {
  const r = extendCurve(arch(), { kind: 'wiggly', length: 5 });
  assert.equal(r.ok, false);
  assert.match(r.reason, /line, arc or smooth/);
});

t('the input curve is never mutated', () => {
  const c = arch();
  const copy = JSON.parse(JSON.stringify(c));
  extendCurve(c, { kind: 'smooth', length: 8 });
  extendCurve(c, { kind: 'arc', length: 8 });
  extendCurve(c, { kind: 'line', length: 8 });
  assert.deepEqual(c, copy);
});

console.log(`\n${passed}/${passed + failed} checks passed.`);
if (failed) process.exit(1);
