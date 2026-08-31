import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  makeField, isField, fieldPeak, fieldMean, sampleFieldFraction, fieldNodeParam,
  fieldDimsForNet, FIELD_MIN_DIM, FIELD_MAX_DIM, fieldFromDistances, cloneField,
} from '../kernel/field.mjs';
import { smoothstep } from '../kernel/cage.mjs';

// A small field whose value at every node is DISTINCT and hand-derivable
// (node (i,j) holds i*10 + j), so a sample landing on the wrong node is
// visible as a wrong number rather than a plausible one.
function rampField(uCount, vCount) {
  const f = makeField(uCount, vCount);
  for (let i = 0; i < uCount; i++) for (let j = 0; j < vCount; j++) f.values[i * vCount + j] = i * 10 + j;
  return f;
}

test('a field is a plain Array — a typed array would not survive the JSON round trip persistence puts it through', () => {
  const f = makeField(4, 3);
  assert.ok(Array.isArray(f.values));
  assert.equal(f.values.length, 12);
  const round = JSON.parse(JSON.stringify(f));
  assert.deepEqual(round, f);
  assert.ok(Array.isArray(round.values), 'values must still be an array after a JSON round trip');
});

test('sampling at an open direction\'s own node fractions returns that node EXACTLY', () => {
  const f = rampField(4, 3);
  // Open in both directions: node k sits at fraction k/(N-1).
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 3; j++) {
      const got = sampleFieldFraction(f, i / 3, j / 2, false, false);
      assert.equal(got, i * 10 + j, `node (${i},${j})`);
    }
  }
});

test('bilinear between four nodes is their exact weighted blend, not an approximation', () => {
  const f = rampField(4, 3);
  // Dead center of the cell spanning nodes (0,0)(1,0)(0,1)(1,1):
  // fractions half a cell in, i.e. 0.5/3 and 0.5/2.
  const got = sampleFieldFraction(f, 0.5 / 3, 0.5 / 2, false, false);
  const expected = (0 + 10 + 1 + 11) / 4;
  assert.ok(Math.abs(got - expected) < 1e-12, `got ${got}, expected ${expected}`);
});

test('an OPEN direction clamps at both ends — fraction 1 is the last node, not a wrap to the first', () => {
  const f = rampField(4, 3);
  assert.equal(sampleFieldFraction(f, 1, 0, false, false), 30);
  assert.equal(sampleFieldFraction(f, 0, 1, false, false), 2);
  // Past the end clamps rather than extrapolating.
  assert.equal(sampleFieldFraction(f, 5, -3, false, false), 30);
});

test('a CLOSED direction genuinely bridges the seam: the last cell blends node N-1 back to node 0', () => {
  const f = rampField(4, 3);
  // Wrapped in U: node k sits at k/N, so the final cell spans [3/4, 1)
  // and blends node 3 (value 30) back to node 0 (value 0).
  assert.equal(sampleFieldFraction(f, 3 / 4, 0, true, false), 30, 'node 3 exactly');
  const mid = sampleFieldFraction(f, 3 / 4 + 1 / 8, 0, true, false);
  assert.ok(Math.abs(mid - 15) < 1e-12, `half way across the seam cell should be 15, got ${mid}`);
  // And fraction 1 lands back on node 0 — the whole point of wrapping.
  assert.equal(sampleFieldFraction(f, 1, 0, true, false), 0);
});

test('wrap is per-direction: closing U must not change what V does', () => {
  const f = rampField(4, 3);
  // j contributes exactly +1 per node in this fixture, so the difference
  // between two samples that vary ONLY in V isolates V's own behavior
  // regardless of however U happened to resolve. Differencing rather than
  // reading a value directly is what makes this a real independence
  // check: at an arbitrary fu the U blend is fractional, so any test that
  // tried to read j back out of a single sample would be testing its own
  // arithmetic, not the sampler's.
  for (const wrapU of [false, true]) {
    const atV0 = sampleFieldFraction(f, 0.37, 0, wrapU, false);
    const atV1 = sampleFieldFraction(f, 0.37, 1, wrapU, false);
    assert.ok(Math.abs((atV1 - atV0) - 2) < 1e-12,
      `V must still span its own 2 nodes with wrapU=${wrapU}, got ${atV1 - atV0}`);
  }
  // ...while U genuinely resolved differently, so the two are not the same sample.
  assert.notEqual(sampleFieldFraction(f, 0.37, 1, false, false), sampleFieldFraction(f, 0.37, 1, true, false));
});

test('a uniform field samples to that same value everywhere, wrapped or not', () => {
  const f = makeField(5, 5);
  f.values.fill(0.37);
  for (const [wu, wv] of [[false, false], [true, false], [false, true], [true, true]]) {
    for (const [fu, fv] of [[0, 0], [0.5, 0.5], [1, 1], [0.13, 0.87]]) {
      assert.ok(Math.abs(sampleFieldFraction(f, fu, fv, wu, wv) - 0.37) < 1e-12);
    }
  }
});

test('fieldNodeParam maps a closed direction to [min,max) and an open one to [min,max] inclusive', () => {
  // Open: first and last node hit the domain ends exactly.
  assert.equal(fieldNodeParam(0, 4, false, 2, 10), 2);
  assert.equal(fieldNodeParam(3, 4, false, 2, 10), 10);
  // Closed: the last node stops one cell short, leaving room for the wrap cell.
  assert.equal(fieldNodeParam(0, 4, true, 2, 10), 2);
  assert.equal(fieldNodeParam(4, 4, true, 2, 10), 10);
  assert.equal(fieldNodeParam(3, 4, true, 2, 10), 8);
  // Fractional k is legal — a contour vertex between nodes lands between params.
  assert.equal(fieldNodeParam(0.5, 4, false, 0, 3), 0.5);
});

test('fieldDimsForNet is denser than the control net and clamped at both ends', () => {
  assert.deepEqual(fieldDimsForNet(5, 5), { uCount: 20, vCount: 20 });
  // A coarse net still gets a usable field.
  assert.deepEqual(fieldDimsForNet(2, 2), { uCount: FIELD_MIN_DIM, vCount: FIELD_MIN_DIM });
  // A dense net is capped so painting stays responsive.
  assert.deepEqual(fieldDimsForNet(200, 200), { uCount: FIELD_MAX_DIM, vCount: FIELD_MAX_DIM });
  // Per-direction, not one shared number.
  assert.deepEqual(fieldDimsForNet(2, 200), { uCount: FIELD_MIN_DIM, vCount: FIELD_MAX_DIM });
});

test('fieldPeak reports the real maximum, and zero for an empty or absent field', () => {
  const f = makeField(3, 3);
  assert.equal(fieldPeak(f), 0);
  f.values[4] = 0.62;
  assert.equal(fieldPeak(f), 0.62);
  assert.equal(fieldPeak(null), 0);
});

test('fieldMean averages every node, and answers zero for an empty or absent field', () => {
  const f = makeField(3, 4); // 12 nodes
  assert.equal(fieldMean(f), 0);
  f.values[0] = 1;
  assert.equal(fieldMean(f), 1 / 12);
  for (let i = 0; i < 12; i++) f.values[i] = 1;
  assert.equal(fieldMean(f), 1);
  assert.equal(fieldMean(null), 0);
  assert.equal(fieldMean({ uCount: 2, vCount: 2, values: [] }), 0);
});

test('fieldMean is the reduction that MOVES while the peak does not — the whole reason a field drives a number by its mean', () => {
  // Two fields a student could plausibly paint: a small mark and a large
  // one, both at full brush strength. Their peaks are identical; only the
  // mean can tell them apart, which is what a driven number needs.
  const small = makeField(8, 8), large = makeField(8, 8);
  for (let i = 0; i < 4; i++) small.values[i] = 1;
  for (let i = 0; i < 32; i++) large.values[i] = 1;
  assert.equal(fieldPeak(small), fieldPeak(large));
  assert.equal(fieldMean(small), 4 / 64);
  assert.equal(fieldMean(large), 32 / 64);
  assert.ok(fieldMean(large) > fieldMean(small));
});

test('fieldMean of a [0,1] field stays inside [0,1] — the drive maps it through a parameter range and a mean outside the unit interval would leave that range', () => {
  const f = makeField(5, 5);
  for (let i = 0; i < 25; i++) f.values[i] = (i % 5) / 4; // 0, 0.25, 0.5, 0.75, 1 repeating
  const m = fieldMean(f);
  assert.ok(m >= 0 && m <= 1);
  assert.equal(m, 0.5);
});

test('isField refuses a malformed field rather than sampling one', () => {
  assert.ok(isField(makeField(4, 4)));
  assert.ok(!isField(null));
  assert.ok(!isField({ uCount: 4, vCount: 4 }), 'no values');
  assert.ok(!isField({ uCount: 4, vCount: 4, values: new Array(15).fill(0) }), 'wrong length');
  assert.ok(!isField({ uCount: 1, vCount: 4, values: new Array(4).fill(0) }), 'a single node cannot be interpolated across');
  const nan = makeField(3, 3); nan.values[2] = NaN;
  assert.ok(!isField(nan), 'a non-finite value');
  const round = JSON.parse(JSON.stringify(makeField(3, 3)));
  assert.ok(isField(round), 'and a field that has been through persistence is still valid');
});


// ---------------------------------------------------------------
// fieldFromDistances — the COMPUTED producer
// ---------------------------------------------------------------

// A 5x4 grid whose node (i,j) sits at a hand-derivable distance: the
// distance IS i, so a whole ROW shares one value and a wrong index
// shows up as a wrong row rather than a plausible number.
const distByRow = (i) => i;

test('inside innerRadius is exactly full strength and outside outerRadius is exactly none', () => {
  // inner 1, outer 3: rows 0 and 1 are inside, rows 3 and 4 are outside.
  const f = fieldFromDistances(5, 4, distByRow, 1, 3, null);
  assert.ok(isField(f));
  for (let j = 0; j < 4; j++) {
    assert.equal(f.values[0 * 4 + j], 1, 'row 0 (d=0, inside inner)');
    assert.equal(f.values[1 * 4 + j], 1, 'row 1 (d=1, exactly at inner)');
    assert.equal(f.values[3 * 4 + j], 0, 'row 3 (d=3, exactly at outer)');
    assert.equal(f.values[4 * 4 + j], 0, 'row 4 (d=4, outside outer)');
  }
});

test('the default ramp reproduces 1 - smoothstep(normalized distance) BIT-IDENTICALLY', () => {
  // The same claim R2a proves for the ramp evaluator itself, re-proven
  // for THIS consumer: a producer that re-derived the normalization
  // slightly differently would drift from the deform that shares its
  // ramp, and only ever by a last-bit amount nobody would notice.
  const inner = 2, outer = 7;
  const dist = (i) => i;           // rows 0..9 sit at distance 0..9
  const f = fieldFromDistances(10, 2, dist, inner, outer, null);
  for (let i = 0; i < 10; i++) {
    const t = (i - inner) / (outer - inner);
    const expected = 1 - smoothstep(t < 0 ? 0 : t > 1 ? 1 : t);
    assert.ok(Object.is(f.values[i * 2 + 0], expected), `row ${i} bit-identical`);
  }
});

test('value falls monotonically as distance grows', () => {
  const f = fieldFromDistances(12, 1, (i) => i * 0.5, 0.5, 4.5, null);
  for (let i = 1; i < 12; i++) {
    assert.ok(f.values[i] <= f.values[i - 1], `row ${i} not above row ${i - 1}`);
  }
  // and it genuinely varies — a constant field would satisfy the above
  assert.ok(f.values[0] > f.values[11] + 0.5, 'the field genuinely varies');
});

test('a zero span authors a hard edge at innerRadius, with no division by zero', () => {
  const f = fieldFromDistances(5, 2, distByRow, 2, 2, null);
  assert.ok(isField(f), 'no NaN anywhere');
  assert.equal(f.values[1 * 2 + 0], 1, 'd=1, inside the edge');
  assert.equal(f.values[2 * 2 + 0], 1, 'd=2, exactly at the edge, still inside');
  assert.equal(f.values[3 * 2 + 0], 0, 'd=3, past the edge');
});

test('one degenerate node reads as outside instead of poisoning the whole field', () => {
  // A pole or collapsed row can genuinely answer NaN. The cost must be
  // that ONE node, not an isField-refusing grid.
  const f = fieldFromDistances(4, 3, (i, j) => (i === 2 && j === 1 ? NaN : i), 0, 3, null);
  assert.ok(isField(f), 'the field is still well-formed');
  assert.equal(f.values[2 * 3 + 1], 0, 'the degenerate node reads as fully outside');
  assert.ok(f.values[2 * 3 + 0] > 0, 'its neighbor is unaffected');
});

test("an edited ramp's own endpoints are honoured OUTSIDE the band, not overridden", () => {
  // The R2a structural point, re-proven for this consumer: a ramp that
  // does not reach 0 must not be clipped to 0 past outerRadius. A
  // consumer that kept hardcoded end branches would fail exactly here.
  const ramp = { interp: 'linear', stops: [[0, 0.8], [1, 0.3]] };
  const f = fieldFromDistances(5, 1, distByRow, 1, 3, ramp);
  assert.equal(f.values[0], 0.8, 'inside inner reads the ramp start, not 1');
  assert.equal(f.values[4], 0.3, 'outside outer reads the ramp end, not 0');
  assert.ok(f.values[2] > 0.3 && f.values[2] < 0.8, 'and it genuinely interpolates between');
});

test('the returned grid is u-major with the requested dimensions', () => {
  // distance varies with j only, so a u-major/v-major mix-up is visible.
  const f = fieldFromDistances(3, 5, (i, j) => j, 0, 4, null);
  assert.equal(f.uCount, 3);
  assert.equal(f.vCount, 5);
  assert.equal(f.values.length, 15);
  for (let i = 0; i < 3; i++) {
    assert.equal(f.values[i * 5 + 0], 1, 'j=0 is full strength in every row');
    assert.equal(f.values[i * 5 + 4], 0, 'j=4 is none in every row');
  }
});

// ---------------------------------------------------------------
// cloneField — the copy a field needs the moment it stops belonging to
// the one surface that produced it.
// ---------------------------------------------------------------

test('a cloned field is equal to its original', () => {
  const f = rampField(4, 3);
  const c = cloneField(f);
  assert.equal(c.uCount, 4);
  assert.equal(c.vCount, 3);
  assert.deepEqual(c.values, f.values);
});

test('a cloned field does not ALIAS its original — writing one leaves the other alone', () => {
  // The whole reason this function exists: two records that look
  // independent must not share one values array.
  const f = rampField(4, 3);
  const c = cloneField(f);
  assert.notEqual(c, f);
  assert.notEqual(c.values, f.values);
  c.values[5] = 999;
  assert.equal(f.values[5], rampField(4, 3).values[5]);
  f.values[0] = -1;
  assert.equal(c.values[0], 0);
});

test('a clone is a PLAIN Array, so it survives the JSON round trip persistence puts it through', () => {
  const f = makeField(3, 3);
  f.values[4] = 0.5;
  const c = cloneField(f);
  assert.ok(Array.isArray(c.values));
  assert.deepEqual(JSON.parse(JSON.stringify(c)), c);
});

test('a clone of a field built from a typed array comes back PLAIN', () => {
  // isField only requires Array.isArray, so this is refused rather than
  // silently copied — the refusal is the point: a Float64Array grid would
  // JSON-round-trip into an object with numeric keys, not an array.
  const bad = { uCount: 2, vCount: 2, values: new Float64Array([0, 0, 0, 0]) };
  assert.equal(cloneField(bad), null);
});

test('cloneField REFUSES a malformed field instead of copying it', () => {
  assert.equal(cloneField(null), null);
  assert.equal(cloneField(undefined), null);
  assert.equal(cloneField({}), null);
  assert.equal(cloneField({ uCount: 3, vCount: 3, values: [0, 0, 0] }), null, 'wrong length');
  assert.equal(cloneField({ uCount: 2, vCount: 2, values: [0, NaN, 0, 0] }), null, 'a non-finite value');
  assert.equal(cloneField({ uCount: 1, vCount: 4, values: [0, 0, 0, 0] }), null, 'a degenerate direction');
});

test('a cloned field samples identically to its original, seam included', () => {
  const f = rampField(6, 4);
  const c = cloneField(f);
  for (const wrapU of [false, true]) for (const wrapV of [false, true]) {
    for (const fu of [0, 0.13, 0.5, 0.97, 1]) for (const fv of [0, 0.29, 0.75, 1]) {
      assert.ok(Object.is(sampleFieldFraction(c, fu, fv, wrapU, wrapV), sampleFieldFraction(f, fu, fv, wrapU, wrapV)));
    }
  }
});
