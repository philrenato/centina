import test from 'node:test';
import assert from 'node:assert/strict';
import {
  estimateParamSensitivity,
  solveParamForDrag,
  sampleGeometry,
  SENSITIVITY_DEFAULTS,
} from '../kernel/sensitivity.mjs';
import { makeEllipse, makeCircle } from '../kernel/primitives.mjs';
import { curvePoint } from '../kernel/curve.mjs';

// ---------------------------------------------------------------------------
// FIXTURES
//
// Deliberately NOT symmetric primitives. A symmetric fixture (a circle, a
// box) makes every sample equally sensitive to the param, so a bug in
// sample-to-sample correspondence, in the least-squares mask, or in an index
// somewhere produces the SAME right-looking answer as correct code. Every
// fixture here has per-sample responses that genuinely differ, so a
// misindexed Jacobian cannot pass.
// ---------------------------------------------------------------------------

// A "cam profile": 9 irregular anchor points, each carried along its OWN
// irregular direction by the param `throw`. Exact by construction:
//   P_i(t) = base_i + t * dir_i        =>  dP_i/dt = dir_i
// Nothing here is proportional to anything else, so the analytic Jacobian is
// a genuinely discriminating target.
const CAM_BASE = [
  [12.4, -3.1, 0.7], [18.9, 4.2, -2.3], [7.05, 11.8, 5.4],
  [-6.2, 13.35, 1.1], [-15.7, 5.9, -4.8], [-11.3, -8.4, 3.2],
  [-2.6, -16.1, -0.9], [9.8, -12.7, 6.6], [16.15, -6.05, -3.75],
];
const CAM_DIR = [
  [0.83, -0.21, 0.04], [1.42, 0.37, -0.19], [0.55, 0.91, 0.42],
  [-0.48, 1.03, 0.08], [-1.21, 0.46, -0.37], [-0.87, -0.65, 0.25],
  [-0.2, -1.24, -0.07], [0.76, -0.98, 0.51], [1.24, -0.47, -0.29],
];
function camEvaluate(inputs, p) {
  return {
    points: CAM_BASE.map((b, i) => [
      b[0] + p.throw * CAM_DIR[i][0],
      b[1] + p.throw * CAM_DIR[i][1],
      b[2] + p.throw * CAM_DIR[i][2],
    ]),
  };
}
const CAM_SPEC = { name: 'throw', type: 'number', default: 1, min: -20, max: 20, step: 0.1 };

// An asymmetric truss whose `spread` param enters QUADRATICALLY (a real
// nonlinear case — an area-like or squared-falloff param):
//   P_i(s) = base_i + s^2 * dir_i      =>  dP_i/ds = 2*s*dir_i
function trussEvaluate(inputs, p) {
  return {
    points: CAM_BASE.map((b, i) => [
      b[0] + p.spread * p.spread * CAM_DIR[i][0],
      b[1] + p.spread * p.spread * CAM_DIR[i][1],
      b[2] + p.spread * p.spread * CAM_DIR[i][2],
    ]),
  };
}

// A real kernel fixture: an off-origin ELLIPSE with two genuinely different
// radii. Its `radiusX` derivative is independently known WITHOUT
// differentiating anything, from the affine-invariance identity makeEllipse
// itself is built on: a NURBS point is an affine combination of its control
// points, so
//     P(u; rx, ry) = center + rx*cx(u)*X + ry*cy(u)*Y
// where (cx(u), cy(u)) is the UNIT circle's own point at the same u. Hence
//     dP/drx = cx(u) * X
// and cx(u) comes from a separately constructed unit circle — ground truth
// that never touches this module.
const EL_CENTER = [5, -3, 2];
const EL_X = [1, 0, 0];
const EL_Y = [0, 1, 0];
const EL_SEGMENTS = 4;
function ellipseEvaluate(inputs, p) {
  return { crv: makeEllipse(EL_CENTER, EL_X, EL_Y, p.radiusX, p.radiusY, EL_SEGMENTS) };
}
const EL_SPEC = { name: 'radiusX', type: 'number', default: 20, min: 1e-6, step: 1 };

// The u values sampleGeometry() visits for this curve, so the analytic
// ground truth can be evaluated at exactly the same parameters.
function ellipseSampleParams(count = 24) {
  const crv = makeEllipse(EL_CENTER, EL_X, EL_Y, 37, 13, EL_SEGMENTS);
  const p = crv.degree;
  const u0 = crv.knots[p], u1 = crv.knots[crv.knots.length - 1 - p];
  return Array.from({ length: count }, (_, i) => u0 + (u1 - u0) * (i / (count - 1)));
}

const CAM_SAMPLER = (r) => r.points;

// ===========================================================================
// ANALYTIC CROSS-CHECKS — the reason this module can be trusted at all.
// ===========================================================================

test('sensitivity: the estimated Jacobian matches the EXACT analytic derivative on an irregular linear fixture', () => {
  const params = { throw: 3.5 };
  const res = estimateParamSensitivity({
    evaluate: camEvaluate, inputs: {}, params, key: 'throw',
    spec: CAM_SPEC, sample: CAM_SAMPLER,
  });
  assert.equal(res.ok, true, res.message);
  assert.equal(res.sampleCount, 9);
  assert.equal(res.stepMode, 'central');

  // dP_i/dt = dir_i, exactly.
  let worst = 0;
  for (let i = 0; i < CAM_DIR.length; i++) {
    for (let c = 0; c < 3; c++) {
      worst = Math.max(worst, Math.abs(res.jacobian[i * 3 + c] - CAM_DIR[i][c]));
    }
  }
  assert.ok(worst < 1e-9, `estimated derivative should match the analytic one, worst absolute error ${worst.toExponential(3)}`);

  // The reported scalar sensitivity is RMS |dir_i| per unit param.
  const rms = Math.sqrt(CAM_DIR.reduce((s, d) => s + d[0] * d[0] + d[1] * d[1] + d[2] * d[2], 0) / CAM_DIR.length);
  assert.ok(Math.abs(res.sensitivity - rms) < 1e-9, `sensitivity ${res.sensitivity} vs analytic RMS ${rms}`);
});

test('sensitivity: matches the analytic derivative of a REAL kernel ellipse (radiusX), cross-checked against an independently built unit circle', () => {
  const params = { radiusX: 37, radiusY: 13 };
  const res = estimateParamSensitivity({
    evaluate: ellipseEvaluate, inputs: {}, params, key: 'radiusX', spec: EL_SPEC,
  });
  assert.equal(res.ok, true, res.message);
  assert.equal(res.sampleCount, 24);

  const unit = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 1, EL_SEGMENTS);
  const us = ellipseSampleParams(24);
  let worst = 0, biggest = 0;
  for (let i = 0; i < us.length; i++) {
    const cx = curvePoint(unit, us[i])[0];          // the analytic dP/drx along X
    const truth = [cx * EL_X[0], cx * EL_X[1], cx * EL_X[2]];
    for (let c = 0; c < 3; c++) {
      worst = Math.max(worst, Math.abs(res.jacobian[i * 3 + c] - truth[c]));
      biggest = Math.max(biggest, Math.abs(truth[c]));
    }
  }
  // Not a symmetry-proof case: cx(u) genuinely varies from -1 to +1 across
  // the samples, so a misindexed Jacobian could not pass this.
  assert.ok(biggest > 0.99, 'the analytic derivative must genuinely vary across samples for this to discriminate');
  assert.ok(worst < 1e-6, `real-kernel ellipse derivative, worst absolute error ${worst.toExponential(3)}`);
});

test('solve: dragging one point of the ellipse outward recovers the exact radiusX (linear param, exact least-squares)', () => {
  const params = { radiusX: 37, radiusY: 13 };
  const base = sampleGeometry(ellipseEvaluate({}, params));
  // Sample 0 sits at u=0 => angle 0 => center + radiusX*X. Drag it +5mm in X.
  const target = [base[0][0] + 5, base[0][1], base[0][2]];
  const res = solveParamForDrag({
    evaluate: ellipseEvaluate, inputs: {}, params, key: 'radiusX', spec: EL_SPEC,
    targets: [{ index: 0, point: target }],
  });
  assert.equal(res.ok, true, res.message);
  assert.ok(Math.abs(res.value - 42) < 1e-6, `expected radiusX 42, got ${res.value}`);
  assert.ok(Math.abs(res.alignment - 1) < 1e-9, `a pure radial drag is perfectly aligned, got ${res.alignment}`);
  assert.ok(res.verified === true && res.residual < 1e-9, `measured residual should be ~0, got ${res.residual}`);
  assert.equal(res.clamped, false);
});

test('solve: a multi-point least-squares drag on the irregular cam recovers the exact param (over-determined, no single point decides it)', () => {
  const params = { throw: 3.5 };
  const wanted = 5.25;
  const truth = camEvaluate({}, { throw: wanted }).points;
  // Constrain FIVE of the nine points — genuinely over-determined, and the
  // solution is only reachable if each sample's own direction is used
  // against its own target.
  const targets = [0, 2, 4, 6, 8].map((index) => ({ index, point: truth[index] }));
  const res = solveParamForDrag({
    evaluate: camEvaluate, inputs: {}, params, key: 'throw',
    spec: CAM_SPEC, sample: CAM_SAMPLER, targets,
  });
  assert.equal(res.ok, true, res.message);
  assert.ok(Math.abs(res.value - wanted) < 1e-9, `expected throw ${wanted}, got ${res.value}`);
  assert.ok(res.residual < 1e-9, `an exactly-reachable target should leave ~0 residual, got ${res.residual}`);
  assert.ok(Math.abs(res.alignment - 1) < 1e-9);
});

test('solve: unconstrained samples take NO part in the fit (a one-point drag is not diluted by the other 23 samples)', () => {
  // The regression guard for a real bug: if unconstrained samples were
  // folded in as "must not move", this same drag would return roughly
  // 37 + 5/12 instead of 42 — plausible, undramatic, and wrong.
  const params = { radiusX: 37, radiusY: 13 };
  const base = sampleGeometry(ellipseEvaluate({}, params));
  const res = solveParamForDrag({
    evaluate: ellipseEvaluate, inputs: {}, params, key: 'radiusX', spec: EL_SPEC,
    targets: [{ index: 0, point: [base[0][0] + 5, base[0][1], base[0][2]] }],
  });
  assert.equal(res.ok, true);
  assert.ok(res.value > 41.9, `a one-point drag must deliver the full change, got ${res.value}`);
  // Same drag expressed in the parallel-array form (nulls = unconstrained)
  // must give the identical answer.
  const parallel = base.map(() => null);
  parallel[0] = [base[0][0] + 5, base[0][1], base[0][2]];
  const res2 = solveParamForDrag({
    evaluate: ellipseEvaluate, inputs: {}, params, key: 'radiusX', spec: EL_SPEC,
    targets: parallel,
  });
  assert.equal(res2.ok, true);
  assert.ok(Math.abs(res2.value - res.value) < 1e-12, 'both target forms must agree exactly');
});

// ===========================================================================
// STEP SIZE — the same estimator across params whose scales differ by
// orders of magnitude. A fixed absolute epsilon fails at least one end.
// ===========================================================================

test('step size: a relative step handles a 1e-4-scale param and a 1e3-scale param with the SAME accuracy', () => {
  // One evaluate, two params, seven orders of magnitude apart in scale:
  //   P_i = base_i + (tiny * 1e4) * dir_i + (huge * 1e-3) * dir2_i
  // so both have well-conditioned, exactly-known derivatives.
  const dir2 = CAM_DIR.map((d) => [d[1], -d[2], d[0]]);
  const evaluate = (inputs, p) => ({
    points: CAM_BASE.map((b, i) => [
      b[0] + p.tiny * 1e4 * CAM_DIR[i][0] + p.huge * 1e-3 * dir2[i][0],
      b[1] + p.tiny * 1e4 * CAM_DIR[i][1] + p.huge * 1e-3 * dir2[i][1],
      b[2] + p.tiny * 1e4 * CAM_DIR[i][2] + p.huge * 1e-3 * dir2[i][2],
    ]),
  });
  const params = { tiny: 2.5e-4, huge: 1750 };

  const rTiny = estimateParamSensitivity({
    evaluate, inputs: {}, params, key: 'tiny', sample: CAM_SAMPLER,
    spec: { name: 'tiny', type: 'number', min: 0, max: 1e-3 },
  });
  assert.equal(rTiny.ok, true, rTiny.message);
  assert.ok(rTiny.step < 1e-6 * 100 && rTiny.step > 0, `step should scale down with the param, got ${rTiny.step}`);
  let worstTiny = 0;
  for (let i = 0; i < CAM_DIR.length; i++) for (let c = 0; c < 3; c++) {
    worstTiny = Math.max(worstTiny, Math.abs(rTiny.jacobian[i * 3 + c] - CAM_DIR[i][c] * 1e4));
  }
  assert.ok(worstTiny < 1e-4, `tiny-scale param derivative, worst error ${worstTiny.toExponential(3)}`);

  const rHuge = estimateParamSensitivity({
    evaluate, inputs: {}, params, key: 'huge', sample: CAM_SAMPLER,
    spec: { name: 'huge', type: 'number', min: 0 },
  });
  assert.equal(rHuge.ok, true, rHuge.message);
  assert.ok(Math.abs(rHuge.step - SENSITIVITY_DEFAULTS.relStep * 1750) < 1e-9, `step should scale up with the param, got ${rHuge.step}`);
  let worstHuge = 0;
  for (let i = 0; i < CAM_DIR.length; i++) for (let c = 0; c < 3; c++) {
    worstHuge = Math.max(worstHuge, Math.abs(rHuge.jacobian[i * 3 + c] - dir2[i][c] * 1e-3));
  }
  assert.ok(worstHuge < 1e-9, `huge-scale param derivative, worst error ${worstHuge.toExponential(3)}`);
});

test('step size: a param sitting exactly at its declared MAX is still estimated, by probing backward', () => {
  const params = { throw: 20 }; // CAM_SPEC.max
  const res = estimateParamSensitivity({
    evaluate: camEvaluate, inputs: {}, params, key: 'throw',
    spec: CAM_SPEC, sample: CAM_SAMPLER,
  });
  assert.equal(res.ok, true, res.message);
  assert.equal(res.stepMode, 'backward');
  let worst = 0;
  for (let i = 0; i < CAM_DIR.length; i++) for (let c = 0; c < 3; c++) {
    worst = Math.max(worst, Math.abs(res.jacobian[i * 3 + c] - CAM_DIR[i][c]));
  }
  assert.ok(worst < 1e-9, `backward difference must be just as exact here, worst ${worst.toExponential(3)}`);
});

// ===========================================================================
// NONLINEARITY — where an estimator must be honest rather than accurate.
// ===========================================================================

test('nonlinear param: one linear step OVERSHOOTS and says so in the measured residual; iterating converges', () => {
  const params = { spread: 2 };
  const wanted = 3.5; // s^2: 4 -> 12.25, a big move for a single linearization
  const truth = trussEvaluate({}, { spread: wanted }).points;
  const targets = [1, 3, 5, 7].map((index) => ({ index, point: truth[index] }));
  const spec = { name: 'spread', type: 'number', min: 0, max: 10 };

  const once = solveParamForDrag({
    evaluate: trussEvaluate, inputs: {}, params, key: 'spread', spec,
    sample: CAM_SAMPLER, targets,
  });
  assert.equal(once.ok, true, once.message);
  assert.equal(once.iterations, 1);
  // s^2 is convex, so its tangent at s=2 lies BELOW it: solving on that
  // tangent for a target of 12.25 asks for s=4.0625, past the true 3.5.
  // Overshoot, not shortfall — the honest outcome of one linearization, and
  // the exact reason a predicted residual would be worthless here. What
  // matters is that the MEASURED residual reports the miss.
  assert.ok(once.value > wanted, `one linear step on a convex param should overshoot, got ${once.value}`);
  assert.ok(Math.abs(once.value - 4.0625) < 1e-6, `the overshoot is the exact tangent-line solution, got ${once.value}`);
  assert.ok(once.residual > 0.05, `the residual must expose the miss, got ${once.residual}`);
  assert.equal(once.verified, true);

  const iterated = solveParamForDrag({
    evaluate: trussEvaluate, inputs: {}, params, key: 'spread', spec,
    sample: CAM_SAMPLER, targets, options: { iterations: 6 },
  });
  assert.equal(iterated.ok, true, iterated.message);
  assert.ok(Math.abs(iterated.value - wanted) < 1e-6, `Gauss-Newton should converge, got ${iterated.value}`);
  assert.ok(iterated.residual < 1e-6, `converged residual should be ~0, got ${iterated.residual}`);
});

// ===========================================================================
// HONEST REFUSALS — the point of the module. Bad conditioning is exactly
// when a drag feels broken, so these are not edge cases, they are the job.
// ===========================================================================

test('refuses a DECLARED discrete param (integer / enum / boolean / vec3) without spending an evaluate', () => {
  let calls = 0;
  const counting = (inputs, p) => { calls++; return camEvaluate(inputs, p); };
  for (const spec of [
    { name: 'sides', type: 'number', integer: true },
    { name: 'style', type: 'number', enumValues: ['a', 'b'] },
    { name: 'closed', type: 'boolean' },
    { name: 'center', type: 'vec3' },
    { name: 'pts', type: 'vec3[]' },
  ]) {
    const res = solveParamForDrag({
      evaluate: counting, inputs: {}, params: { throw: 3.5, [spec.name]: 4 }, key: spec.name,
      spec, sample: CAM_SAMPLER, targets: [{ index: 0, point: [0, 0, 0] }],
    });
    assert.equal(res.ok, false, `${spec.name} must be refused`);
    assert.equal(res.reason, 'discrete-param');
    assert.ok(res.message.includes(spec.name));
  }
  assert.equal(calls, 0, 'a structurally undraggable param must be refused before any geometry is evaluated');
});

test('refuses an UNDECLARED discrete param — a real segment count on a real kernel circle is caught as non-smooth', () => {
  // The hard case, and the one that motivated the smoothness gate: sampling
  // a curve at fixed parametric fractions returns the SAME sample count
  // whatever the segment count is, so the count check alone sees nothing
  // wrong and a large, confident, WRONG derivative comes back.
  const evaluate = (inputs, p) => ({ crv: makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], p.radius, p.segments) });
  const spec = { name: 'segments', type: 'number', default: 4, min: 4, step: 1 };
  for (const segments of [4, 6, 9]) {
    const res = estimateParamSensitivity({
      evaluate, inputs: {}, params: { radius: 10, segments }, key: 'segments', spec,
    });
    assert.equal(res.ok, false, `segments=${segments} must be refused, not silently estimated`);
    assert.equal(res.reason, 'non-smooth');
    assert.ok(res.asymmetry > 0.25, `asymmetry ${res.asymmetry} should be well past the gate`);
  }
  // ...while the genuinely continuous param on the SAME evaluate is fine.
  const ok = estimateParamSensitivity({
    evaluate, inputs: {}, params: { radius: 10, segments: 4 }, key: 'radius',
    spec: { name: 'radius', type: 'number', min: 1e-6 },
  });
  assert.equal(ok.ok, true, ok.message);
  assert.ok(ok.asymmetry < 1e-6, 'a smooth param must sail through the same gate');
});

test('refuses a QUANTIZED param (rounded internally, sample count stable) and distinguishes it from an inert one', () => {
  // Rounds to whole units internally: geometry does not budge at a fine
  // nudge but jumps at a coarse one. Count never changes, so only the
  // coarse-probe fallback can tell this apart from a dead param.
  const evaluate = (inputs, p) => {
    const n = Math.round(p.notches);
    return { points: CAM_BASE.map((b, i) => [b[0] + n * CAM_DIR[i][0], b[1] + n * CAM_DIR[i][1], b[2] + n * CAM_DIR[i][2]]) };
  };
  const res = estimateParamSensitivity({
    evaluate, inputs: {}, params: { notches: 3 }, key: 'notches',
    spec: { name: 'notches', type: 'number', min: 0, max: 20 }, sample: CAM_SAMPLER,
  });
  assert.equal(res.ok, false);
  assert.equal(res.reason, 'quantized');
  assert.ok(res.coarse && res.coarse.motion > 0, 'the coarse probe must be reported as evidence');
  assert.notEqual(res.reason, 'insensitive', 'a quantized param must not be conflated with a dead one');
});

test('refuses a ZERO-SENSITIVITY param — geometry does not move at any probe scale', () => {
  const evaluate = (inputs, p) => camEvaluate(inputs, { throw: p.throw }); // `unused` is ignored
  const res = solveParamForDrag({
    evaluate, inputs: {}, params: { throw: 3.5, unused: 12 }, key: 'unused',
    spec: { name: 'unused', type: 'number', min: 0, max: 100 }, sample: CAM_SAMPLER,
    targets: [{ index: 0, point: [99, 99, 99] }],
  });
  assert.equal(res.ok, false);
  assert.equal(res.reason, 'insensitive');
  assert.ok(res.message.includes('unused'));
  // Crucially: it does NOT return a huge value from dividing by a ~zero
  // sensitivity. There is no value at all.
  assert.equal(res.value, undefined);
});

test('refuses an ORTHOGONAL drag — the param is genuinely sensitive, just not in the direction asked', () => {
  // Moves every point purely along +X.
  const evaluate = (inputs, p) => ({ points: CAM_BASE.map((b, i) => [b[0] + p.slide * (1 + 0.1 * i), b[1], b[2]]) });
  const params = { slide: 4 };
  const base = evaluate({}, params).points;
  const res = solveParamForDrag({
    evaluate, inputs: {}, params, key: 'slide',
    spec: { name: 'slide', type: 'number' }, sample: CAM_SAMPLER,
    targets: [{ index: 3, point: [base[3][0], base[3][1] + 8, base[3][2]] }], // pure +Y drag
  });
  assert.equal(res.ok, false);
  assert.equal(res.reason, 'orthogonal');
  // The refusal must carry a REAL, nonzero sensitivity — proving this is
  // distinguished from 'insensitive' rather than lumped in with it.
  assert.ok(res.sensitivity > 1, `sensitivity should be real and nonzero, got ${res.sensitivity}`);
  assert.ok(Math.abs(res.alignment) < 1e-9, `a perpendicular drag has ~zero alignment, got ${res.alignment}`);

  // The same param with a drag it CAN explain still works — proof the
  // refusal is about the drag, not the param.
  const okRes = solveParamForDrag({
    evaluate, inputs: {}, params, key: 'slide',
    spec: { name: 'slide', type: 'number' }, sample: CAM_SAMPLER,
    targets: [{ index: 3, point: [base[3][0] + 1.3, base[3][1], base[3][2]] }],
  });
  assert.equal(okRes.ok, true, okRes.message);
  assert.ok(Math.abs(okRes.residual) < 1e-9);
});

test('refuses when the param moves the geometry elsewhere but NOT the dragged sample', () => {
  // `tipOnly` moves sample 8 alone. Dragging sample 0 with it is hopeless,
  // and must be said so rather than answered with a division by ~zero.
  const evaluate = (inputs, p) => ({
    points: CAM_BASE.map((b, i) => (i === 8 ? [b[0] + p.tipOnly * 2, b[1], b[2]] : [...b])),
  });
  const params = { tipOnly: 1 };
  const res = solveParamForDrag({
    evaluate, inputs: {}, params, key: 'tipOnly',
    spec: { name: 'tipOnly', type: 'number' }, sample: CAM_SAMPLER,
    targets: [{ index: 0, point: [CAM_BASE[0][0] + 5, CAM_BASE[0][1], CAM_BASE[0][2]] }],
  });
  assert.equal(res.ok, false);
  assert.equal(res.reason, 'insensitive-at-target');
  assert.ok(res.sensitivity > 0, 'the param IS sensitive overall — that is the point of the distinct reason code');
});

test('refuses a topology-changing param that genuinely changes the SAMPLE COUNT', () => {
  const evaluate = (inputs, p) => ({
    points: Array.from({ length: Math.max(2, Math.round(p.count)) }, (_, i) => [i * 3.7, i * -1.4, i * 0.9]),
  });
  const res = estimateParamSensitivity({
    evaluate, inputs: {}, params: { count: 6 }, key: 'count',
    spec: { name: 'count', type: 'number', min: 2, max: 40 }, sample: CAM_SAMPLER,
  });
  assert.equal(res.ok, false);
  // Note the honest chain here: at a fine nudge this param rounds straight
  // back to itself and looks inert; it is the COARSE fallback probe that
  // exposes the correspondence break, and that finding is what gets
  // reported rather than the weaker "insensitive".
  assert.equal(res.reason, 'unstable-sampling');
});

test('refuses honestly when evaluate() throws, or returns non-finite geometry — never propagates either', () => {
  const thrower = (inputs, p) => {
    if (p.t !== 1) throw new Error('degenerate recipe');
    return camEvaluate(inputs, { throw: p.t });
  };
  const a = estimateParamSensitivity({
    evaluate: thrower, inputs: {}, params: { t: 1 }, key: 't',
    spec: { name: 't', type: 'number' }, sample: CAM_SAMPLER,
  });
  assert.equal(a.ok, false);
  assert.equal(a.reason, 'evaluate-failed');
  assert.ok(a.message.includes('degenerate recipe'));

  const nanner = (inputs, p) => ({ points: CAM_BASE.map((b) => [b[0] + p.t, b[1] * (p.t === 1 ? 1 : NaN), b[2]]) });
  const b = estimateParamSensitivity({
    evaluate: nanner, inputs: {}, params: { t: 1 }, key: 't',
    spec: { name: 't', type: 'number' }, sample: CAM_SAMPLER,
  });
  assert.equal(b.ok, false);
  assert.equal(b.reason, 'non-finite-geometry');
});

test('refuses a param with no finite value, and a zero-length requested drag', () => {
  const noValue = estimateParamSensitivity({
    evaluate: camEvaluate, inputs: {}, params: { throw: undefined }, key: 'throw',
    sample: CAM_SAMPLER,
  });
  assert.equal(noValue.ok, false);
  assert.equal(noValue.reason, 'non-numeric-param');

  const base = camEvaluate({}, { throw: 3.5 }).points;
  const noDrag = solveParamForDrag({
    evaluate: camEvaluate, inputs: {}, params: { throw: 3.5 }, key: 'throw',
    spec: CAM_SPEC, sample: CAM_SAMPLER,
    targets: [{ index: 2, point: [...base[2]] }], // target IS the current position
  });
  assert.equal(noDrag.ok, false);
  assert.equal(noDrag.reason, 'no-target');
});

// ===========================================================================
// CLAMPING + FINITENESS
// ===========================================================================

test('clamps to the declared range, flags it, stays finite, and reports the shortfall honestly', () => {
  const params = { throw: 3.5 };
  // Ask for a displacement that would need throw far past max=20.
  const truth = camEvaluate({}, { throw: 400 }).points;
  const res = solveParamForDrag({
    evaluate: camEvaluate, inputs: {}, params, key: 'throw',
    spec: CAM_SPEC, sample: CAM_SAMPLER,
    targets: [{ index: 4, point: truth[4] }],
  });
  assert.equal(res.ok, true, res.message);
  assert.equal(res.value, 20, 'must land exactly on the declared max');
  assert.equal(res.clamped, true, 'and must SAY it clamped');
  assert.ok(Number.isFinite(res.value));
  // The measured residual must expose that most of the drag was not achieved.
  assert.ok(res.residual > 0.5, `a hard clamp should leave a large honest residual, got ${res.residual}`);
});

test('every returned number is finite across a sweep of awkward-but-legal param values', () => {
  const spec = { name: 'throw', type: 'number', min: -20, max: 20 };
  for (const v of [-20, -1e-9, 0, 1e-12, 0.5, 19.999999, 20]) {
    const res = estimateParamSensitivity({
      evaluate: camEvaluate, inputs: {}, params: { throw: v }, key: 'throw',
      spec, sample: CAM_SAMPLER,
    });
    assert.equal(res.ok, true, `throw=${v}: ${res.message}`);
    assert.ok(Number.isFinite(res.sensitivity) && res.sensitivity > 0, `throw=${v} sensitivity ${res.sensitivity}`);
    assert.ok(Number.isFinite(res.step) && res.step > 0, `throw=${v} step ${res.step}`);
    for (const j of res.jacobian) assert.ok(Number.isFinite(j), `throw=${v} produced a non-finite Jacobian entry`);
  }
});

test('never mutates the caller\'s params or inputs', () => {
  const params = Object.freeze({ radiusX: 37, radiusY: 13 });
  const inputs = Object.freeze({ profile: 'untouched' });
  const base = sampleGeometry(ellipseEvaluate(inputs, params));
  const res = solveParamForDrag({
    evaluate: ellipseEvaluate, inputs, params, key: 'radiusX', spec: EL_SPEC,
    targets: [{ index: 0, point: [base[0][0] + 5, base[0][1], base[0][2]] }],
  });
  assert.equal(res.ok, true, res.message);
  assert.equal(params.radiusX, 37, 'the caller\'s params object must be left alone');
  assert.equal(res.baseValue, 37);
});

// ===========================================================================
// THE DEFAULT SAMPLER
// ===========================================================================

test('sampleGeometry: stable count and order for curve / points / line / point results, and an honest throw otherwise', () => {
  const crv = sampleGeometry({ crv: makeEllipse(EL_CENTER, EL_X, EL_Y, 37, 13, EL_SEGMENTS) });
  assert.equal(crv.length, 24);
  assert.equal(sampleGeometry({ crv: makeEllipse(EL_CENTER, EL_X, EL_Y, 99, 4, EL_SEGMENTS) }).length, 24,
    'the count must not depend on the param being probed');
  assert.equal(sampleGeometry({ points: CAM_BASE }).length, 9);
  assert.equal(sampleGeometry({ start: [0, 0, 0], end: [1, 2, 3] }).length, 2);
  assert.equal(sampleGeometry({ pos: [4, 5, 6] }).length, 1);
  assert.throws(() => sampleGeometry({ mystery: true }), /unrecognized/);
  assert.throws(() => sampleGeometry(null), /no object/);
});
