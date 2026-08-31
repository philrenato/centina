import test from 'node:test';
import assert from 'node:assert/strict';
import { curvesCoincident, curveCoincidenceGap, adaptiveArcLengthSamples } from '../kernel/curve.mjs';
import { makeLine, makeArc } from '../kernel/primitives.mjs';

const TOL = 0.001; // mm — matches the app's own JOIN_TOLERANCE, reused for this exact class of "spatially coincident" check

// Dense point chain for a NurbsCrv, matching how the app's own
// sampleCurveAdaptive/adaptiveArcLengthSamples resample a curve for display
// and comparison — a plain [x,y,z] array, not {u,pt} pairs.
function samplesOf(crv, uStart, uEnd) {
  return adaptiveArcLengthSamples(crv, uStart ?? crv.knots[0], uEnd ?? crv.knots[crv.knots.length - 1], 0.01).map((s) => s.pt);
}

test('curvesCoincident: two identical lines (same points, same direction) ARE coincident', () => {
  const a = makeLine([0, 0, 0], [10, 0, 0]);
  const b = makeLine([0, 0, 0], [10, 0, 0]);
  assert.ok(curvesCoincident(samplesOf(a), samplesOf(b), TOL));
});

test('curvesCoincident: two identical lines, one REVERSED, are still coincident (direction-agnostic)', () => {
  const a = makeLine([0, 0, 0], [10, 0, 0]);
  const b = makeLine([10, 0, 0], [0, 0, 0]);
  assert.ok(curvesCoincident(samplesOf(a), samplesOf(b), TOL));
});

test('curvesCoincident: two lines offset by less than tolerance ARE coincident (near-duplicate)', () => {
  const a = makeLine([0, 0, 0], [10, 0, 0]);
  const b = makeLine([0, 0.0002, 0], [10, 0.0002, 0]); // 0.0002mm offset, well inside 0.001mm TOL
  assert.ok(curvesCoincident(samplesOf(a), samplesOf(b), TOL));
});

test('curvesCoincident: two lines offset by MORE than tolerance are NOT coincident', () => {
  const a = makeLine([0, 0, 0], [10, 0, 0]);
  const b = makeLine([0, 1, 0], [10, 1, 0]); // 1mm offset, well outside 0.001mm TOL
  assert.equal(curvesCoincident(samplesOf(a), samplesOf(b), TOL), false);
});

test('curvesCoincident: a curve that is a real SUB-SEGMENT of a much longer curve is NOT flagged as a duplicate', () => {
  const full = makeLine([0, 0, 0], [10, 0, 0]);
  const sub = makeLine([2, 0, 0], [4, 0, 0]); // lies exactly on `full`'s own path, but spans only 2 of 10 units
  assert.equal(curvesCoincident(samplesOf(full), samplesOf(sub), TOL), false);
});

test('curvesCoincident: a curve that is a sub-segment starting AT the same origin point is still NOT flagged (extent, not just endpoint, must match)', () => {
  const full = makeLine([0, 0, 0], [10, 0, 0]);
  const sub = makeLine([0, 0, 0], [3, 0, 0]); // shares an endpoint with `full`, but is only 30% of its length
  assert.equal(curvesCoincident(samplesOf(full), samplesOf(sub), TOL), false);
});

test('curvesCoincident: two genuinely different curves (a line and a perpendicular arc through a shared point) are NOT flagged', () => {
  const line = makeLine([0, 0, 0], [10, 0, 0]);
  const arc = makeArc([0, 0, 0], [0, 0, 1], [0, 1, 0], 5, 0, Math.PI); // shares the origin, otherwise nothing like the line
  assert.equal(curvesCoincident(samplesOf(line), samplesOf(arc), TOL), false);
});

test('curvesCoincident: two identical arcs ARE coincident', () => {
  const a = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI);
  const b = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI);
  assert.ok(curvesCoincident(samplesOf(a), samplesOf(b), TOL));
});

test('curvesCoincident: an arc and a shorter sub-arc of the SAME circle are NOT flagged as a duplicate', () => {
  const full = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI);
  const partial = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI / 3);
  assert.equal(curvesCoincident(samplesOf(full), samplesOf(partial), TOL), false);
});

test('curvesCoincident: empty sample chains are never coincident (defensive, not a crash)', () => {
  assert.equal(curvesCoincident([], [[0, 0, 0]], TOL), false);
  assert.equal(curvesCoincident([], [], TOL), false);
});

// ---------------------------------------------------------------------------
// curveCoincidenceGap — the same test read backwards. These assert the
// CONTRACT that makes a refusal able to name a value that would work: the
// returned gap is the exact boundary of curvesCoincident's own answer, so a
// tolerance just above it passes and a tolerance just below it fails. A gap
// that were merely "some proximity measure" would satisfy neither.
// ---------------------------------------------------------------------------

// Asserts the boundary property directly against curvesCoincident, which is
// the only claim callers rely on. Multiplicative nudges (not additive) so the
// same assertion holds at every scale from a machine-epsilon gap to a
// modeling-scale one.
function assertGapIsTheBoundary(a, b) {
  const gap = curveCoincidenceGap(a, b);
  assert.ok(Number.isFinite(gap) && gap > 0, `gap should be a finite positive number, got ${gap}`);
  assert.equal(curvesCoincident(a, b, gap * 1.001), true, `should be coincident just ABOVE the gap (${gap})`);
  assert.equal(curvesCoincident(a, b, gap * 0.999), false, `should NOT be coincident just BELOW the gap (${gap})`);
  return gap;
}

test('curveCoincidenceGap: two identical arcs need a gap at machine precision, not zero-by-luck', () => {
  const a = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI);
  const b = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI);
  const gap = curveCoincidenceGap(samplesOf(a), samplesOf(b));
  assert.ok(gap < 1e-9, `identical arcs should need essentially no tolerance, got ${gap}`);
});

test('curveCoincidenceGap: a rigidly offset arc needs a gap equal to the offset', () => {
  const a = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI);
  const b = makeArc([0, 0.05, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI); // 0.05mm apart everywhere
  const gap = curveCoincidenceGap(samplesOf(a), samplesOf(b));
  assert.ok(Math.abs(gap - 0.05) < 1e-6, `expected a 0.05mm gap, got ${gap}`);
});

test('curveCoincidenceGap: the gap is exactly the boundary of curvesCoincident (offset arcs)', () => {
  const a = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI);
  const b = makeArc([0, 0.05, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI);
  assertGapIsTheBoundary(samplesOf(a), samplesOf(b));
});

test('curveCoincidenceGap: the gap is exactly the boundary for a REVERSED near-duplicate too (direction-agnostic)', () => {
  const a = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI);
  const b = makeArc([0, 0.05, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI);
  const rev = samplesOf(b).slice().reverse();
  assertGapIsTheBoundary(samplesOf(a), rev);
});

test('curveCoincidenceGap: the gap is exactly the boundary for the sub-segment case, where the EXTENT term is what binds', () => {
  const full = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI);
  const partial = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 5, 0, Math.PI / 3);
  assertGapIsTheBoundary(samplesOf(full), samplesOf(partial));
});

test('curveCoincidenceGap: two genuinely unrelated curves report a large finite gap, never Infinity', () => {
  const line = makeLine([0, 0, 0], [10, 0, 0]);
  const arc = makeArc([0, 0, 0], [0, 0, 1], [0, 1, 0], 5, 0, Math.PI);
  const gap = curveCoincidenceGap(samplesOf(line), samplesOf(arc));
  assert.ok(Number.isFinite(gap) && gap > 1, `unrelated curves should report a real distance, got ${gap}`);
});

test('curveCoincidenceGap: an empty sample chain reports Infinity, so a caller can tell "nothing to compare" from "far apart"', () => {
  assert.equal(curveCoincidenceGap([], [[0, 0, 0]]), Infinity);
  assert.equal(curveCoincidenceGap([], []), Infinity);
});
