// Tests for kernel/transform.mjs — the
// AFFINE-vs-SPACE-MORPH classification, proved numerically, not asserted.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  rotatePoint, scalePoint, shearPoint, taperPoint, twistPoint, bendPoint, charybdisPoint,
  transformPoint, TRANSFORM_CLASS, TRANSFORM_TYPES,
} from '../kernel/transform.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { curvePoint } from '../kernel/curve.mjs';
import { dot, sub, length, normalize } from '../kernel/vec3.mjs';
import { smoothstep } from '../kernel/cage.mjs';
import { DEFAULT_FALLOFF_RAMP } from '../kernel/falloff.mjs';

// A deliberately NON-symmetric, non-planar point set (this project's own
// standing rule against trivial/symmetric test fixtures hiding real bugs).
const RAW_PTS = [
  [-13, 4, 2], [-6, 11, 7], [1, 3, 15], [9, -5, 8], [17, 6, -3], [23, 14, 5],
];

function buildTestCurve(pts = RAW_PTS) {
  return globalCurveInterp(pts.map((p) => [...p]), 3);
}

// Applies an affine per-point map to every control point of a curve (never
// touching the weight component) and returns the resulting curve.
function mapCurveControlPoints(crv, mapFn) {
  return { degree: crv.degree, knots: crv.knots, ctrlPts: crv.ctrlPts.map((cp) => [...mapFn([cp[0], cp[1], cp[2]]), cp[3]]) };
}

test('TRANSFORM_TYPES/TRANSFORM_CLASS: all 7 real types are classified, exactly 3 affine + 4 morph', () => {
  assert.equal(TRANSFORM_TYPES.length, 7);
  const affine = TRANSFORM_TYPES.filter((t) => TRANSFORM_CLASS[t] === 'affine');
  const morph = TRANSFORM_TYPES.filter((t) => TRANSFORM_CLASS[t] === 'morph');
  assert.deepEqual(affine.sort(), ['rotate', 'scale', 'shear']);
  assert.deepEqual(morph.sort(), ['bend', 'charybdis', 'taper', 'twist']);
});

// ============================================================
// AFFINE — EXACT. The real proof: T(sum(R_i P_i)) === sum(R_i T(P_i)).
// Apply the map to every control point, then confirm the MAPPED CURVE's own
// value at an arbitrary interior parameter equals the map applied directly
// to the ORIGINAL curve's value there — never a self-consistency tautology
// (both sides are computed via genuinely different paths: one evaluates a
// freshly-rebuilt curve, the other transforms an independently-evaluated
// point).
// ============================================================
test('ROTATE is exact affine: transforming control points reproduces the true rotated curve at every sampled parameter', () => {
  const crv = buildTestCurve();
  const center = [3, -2, 5];
  const axis = normalize([1, 2, 3]); // a genuinely oblique axis, not a world axis
  const angle = 1.13; // radians, an arbitrary non-multiple-of-90deg value
  const rotated = mapCurveControlPoints(crv, (p) => rotatePoint(p, center, axis, angle));
  for (const u of [crv.knots[3], 0.13, 0.37, 0.5, 0.71, 0.93 * (crv.knots[crv.knots.length - 1] - crv.knots[0]) + crv.knots[0]]) {
    const truePoint = rotatePoint(curvePoint(crv, u), center, axis, angle);
    const mappedPoint = curvePoint(rotated, u);
    const err = length(sub(truePoint, mappedPoint));
    assert.ok(err < 1e-9, `u=${u} error=${err}`);
  }
});

test('SCALE is exact affine: transforming control points reproduces the true scaled curve at every sampled parameter', () => {
  const crv = buildTestCurve();
  const center = [-4, 1, 2];
  const axis = normalize([2, -1, 1]);
  const factor = 2.7;
  const scaled = mapCurveControlPoints(crv, (p) => scalePoint(p, center, axis, factor));
  for (const u of [0.1, 0.4, 0.66, 0.9]) {
    const dom = crv.knots[0] + u * (crv.knots[crv.knots.length - 1] - crv.knots[0]);
    const truePoint = scalePoint(curvePoint(crv, dom), center, axis, factor);
    const mappedPoint = curvePoint(scaled, dom);
    const err = length(sub(truePoint, mappedPoint));
    assert.ok(err < 1e-9, `u=${dom} error=${err}`);
  }
});

test('SHEAR is exact affine: transforming control points reproduces the true sheared curve at every sampled parameter', () => {
  const crv = buildTestCurve();
  const center = [1, 1, 1];
  const measureAxis = normalize([1, 0, 1]);
  const shearDir = normalize([0, 1, 0.3]);
  const factor = 0.85;
  const sheared = mapCurveControlPoints(crv, (p) => shearPoint(p, center, measureAxis, shearDir, factor));
  for (const u of [0.05, 0.3, 0.62, 0.99]) {
    const dom = crv.knots[0] + u * (crv.knots[crv.knots.length - 1] - crv.knots[0]);
    const truePoint = shearPoint(curvePoint(crv, dom), center, measureAxis, shearDir, factor);
    const mappedPoint = curvePoint(sheared, dom);
    const err = length(sub(truePoint, mappedPoint));
    assert.ok(err < 1e-9, `u=${dom} error=${err}`);
  }
});

test('AFFINE identity: rotate/scale/shear by a no-op value reproduce the exact original point', () => {
  const p = [7, -3, 11];
  const center = [0, 0, 0];
  const axis = normalize([1, 1, 1]);
  assert.ok(length(sub(rotatePoint(p, center, axis, 0), p)) < 1e-12);
  assert.ok(length(sub(scalePoint(p, center, axis, 1), p)) < 1e-12);
  assert.ok(length(sub(shearPoint(p, center, axis, normalize([1, -1, 0]), 0), p)) < 1e-12);
});

// ============================================================
// SPACE MORPH — APPROXIMATE, proved to be genuinely NOT affine (a real
// contrast, not just an absence of an exactness test): the SAME per-control-
// point map applied to a mapped curve's own control net does NOT reproduce
// the true value at an interior parameter the way rotate/scale/shear do —
// confirming these really are a different mathematical class, not merely
// unproven.
// ============================================================
test('TWIST is genuinely NOT exact (contrast case): the control-point nudge diverges from a hypothetical "true value" proxy', () => {
  const crv = buildTestCurve();
  const center = [0, 0, -10];
  const axis = normalize([0, 0, 1]);
  const span = 30;
  const angle = Math.PI / 2;
  const twisted = mapCurveControlPoints(crv, (p) => twistPoint(p, center, axis, span, angle));
  // The curve's own raw control points sit at DIFFERENT axial positions
  // than the point the interpolated curve actually passes through at a
  // given u (true only in the degenerate degree-1 case) — so the nudged
  // curve's sampled point differs measurably from directly twisting the
  // ORIGINAL curve's own sampled point. A nonzero gap here is the actual
  // proof this is an approximation, not a missing exactness guarantee.
  let worstGap = 0;
  for (const u of [0.2, 0.5, 0.8]) {
    const dom = crv.knots[0] + u * (crv.knots[crv.knots.length - 1] - crv.knots[0]);
    const trueTwisted = twistPoint(curvePoint(crv, dom), center, axis, span, angle);
    const approxTwisted = curvePoint(twisted, dom);
    worstGap = Math.max(worstGap, length(sub(trueTwisted, approxTwisted)));
  }
  assert.ok(worstGap > 1e-3, `expected a real, measurable gap proving TWIST is not affine, got ${worstGap}`);
});

test('TWIST/TAPER: identity at the base point (axisComp=0) and at total value 0', () => {
  const p = [5, 5, 0];
  const center = [0, 0, 0];
  const axis = normalize([0, 0, 1]);
  assert.ok(length(sub(twistPoint(p, center, axis, 20, 0), p)) < 1e-9);
  assert.ok(length(sub(taperPoint(p, center, axis, 20, 1), p)) < 1e-9);
  // A point exactly AT the center (axisComp=0) is untouched by twist for
  // ANY angle (rotating about an axis through itself changing nothing here
  // since p-center's axial component is 0, so angle(0)=0 regardless of the
  // total requested angle).
  const onAxisAtBase = [0, 0, 0];
  assert.ok(length(sub(twistPoint(onAxisAtBase, center, axis, 20, 1.7), onAxisAtBase)) < 1e-9);
});

test('BEND: reduces to the true identity at angleRad=0, and to the spine\'s own point exactly at the base (h=0,w=0,axisComp=0)', () => {
  const p1 = [2, -1, 0];
  const axis = normalize([1, 0, 0]);
  const planeNormal = normalize([0, 1, 0]);
  const span = 40;
  // angle=0 must be the literal identity (R would be infinite otherwise —
  // this is the guarded special case, not a coincidence).
  const p = [30, 8, -4];
  assert.ok(length(sub(bendPoint(p, p1, axis, span, planeNormal, 0), p)) < 1e-12);
  // A point exactly on the spine (h=0, w=0) at axisComp=0 must map to p1
  // exactly, for any real bend angle.
  assert.ok(length(sub(bendPoint(p1, p1, axis, span, planeNormal, 0.9), p1)) < 1e-9);
});

test('BEND: the spine itself traces a true circular arc of radius span/angle (a real geometric proof, not just an endpoint check)', () => {
  const p1 = [0, 0, 0];
  const axis = normalize([1, 0, 0]);
  const planeNormal = normalize([0, 1, 0]);
  const span = 50;
  const angle = 1.2; // radians
  const R = span / angle;
  const center = [p1[0], p1[1] + R * planeNormal[1], p1[2] + R * planeNormal[2]];
  // Sample several points ALONG the spine (h=0, w=0) and confirm every one
  // sits exactly distance R from the arc's own center — the defining
  // property of a circular arc.
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const onSpine = [p1[0] + t * span * axis[0], p1[1] + t * span * axis[1], p1[2] + t * span * axis[2]];
    const bent = bendPoint(onSpine, p1, axis, span, planeNormal, angle);
    const distFromCenter = length(sub(bent, center));
    assert.ok(Math.abs(distFromCenter - R) < 1e-6, `t=${t} dist=${distFromCenter} expected ${R}`);
  }
});

test('BEND: is genuinely NOT exact affine (contrast case) — the control-point nudge on a curve diverges from directly bending the true curve value', () => {
  const crv = buildTestCurve([[-10, 0, 0], [-4, 2, 1], [3, -1, 4], [11, 3, -2], [18, 0, 5], [25, 4, 0]]);
  const p1 = [0, 0, 0];
  const axis = normalize([1, 0, 0]);
  const planeNormal = normalize([0, 1, 0]);
  const span = 30;
  const angle = 0.8;
  const bent = mapCurveControlPoints(crv, (p) => bendPoint(p, p1, axis, span, planeNormal, angle));
  let worstGap = 0;
  for (const u of [0.2, 0.5, 0.8]) {
    const dom = crv.knots[0] + u * (crv.knots[crv.knots.length - 1] - crv.knots[0]);
    const trueBent = bendPoint(curvePoint(crv, dom), p1, axis, span, planeNormal, angle);
    const approxBent = curvePoint(bent, dom);
    worstGap = Math.max(worstGap, length(sub(trueBent, approxBent)));
  }
  assert.ok(worstGap > 1e-3, `expected a real, measurable gap proving BEND is not affine, got ${worstGap}`);
});

// ============================================================
// CHARYBDIS (Rhino: Maelstrom) — the falloff MUST reuse cageSmoothstep,
// verified directly (not just by import) by cross-checking the exact
// falloff weight this module computes against a direct smoothstep() call.
// ============================================================
test('CHARYBDIS: full rotation strictly inside innerRadius, zero strictly outside outerRadius', () => {
  const center = [0, 0, 0];
  const axis = normalize([0, 0, 1]);
  const angle = Math.PI / 2;
  const inner = 5, outer = 20;
  // A point well inside the inner radius rotates by the FULL angle.
  const pIn = [2, 0, 3];
  const gotIn = charybdisPoint(pIn, center, axis, inner, outer, angle);
  const expectedIn = rotatePoint(pIn, center, axis, angle);
  assert.ok(length(sub(gotIn, expectedIn)) < 1e-9);
  // A point well outside the outer radius does not move at all.
  const pOut = [30, 0, -4];
  const gotOut = charybdisPoint(pOut, center, axis, inner, outer, angle);
  assert.ok(length(sub(gotOut, pOut)) < 1e-9);
});

test('CHARYBDIS: the falloff between the two radii is EXACTLY cageSmoothstep, reused not re-derived', () => {
  const center = [0, 0, 0];
  const axis = normalize([0, 0, 1]);
  const angle = Math.PI / 2;
  const inner = 5, outer = 15;
  for (const r of [7, 9, 11, 13]) {
    const p = [r, 0, 0]; // planar, radial distance r
    const got = charybdisPoint(p, center, axis, inner, outer, angle);
    const expectedWeight = 1 - smoothstep((r - inner) / (outer - inner));
    const expected = rotatePoint(p, center, axis, angle * expectedWeight);
    assert.ok(length(sub(got, expected)) < 1e-9, `r=${r}`);
  }
});

test('CHARYBDIS: falloff is monotonically decreasing from inner to outer radius (a real, working knob, not decorative)', () => {
  const center = [0, 0, 0];
  const axis = normalize([0, 0, 1]);
  const angle = 1.0;
  const inner = 4, outer = 16;
  const original = [0, 1, 0]; // used as the "how far did this angle sweep" reference direction indirectly via distance from a fixed off-axis probe
  let prevSweep = Infinity;
  for (const r of [4, 6, 8, 10, 12, 14, 16, 18]) {
    const p = [r, 0, 2];
    const rotated = charybdisPoint(p, center, axis, inner, outer, angle);
    // The actual angular sweep this point underwent, measured directly.
    const planarBefore = [p[0], p[1], 0];
    const planarAfter = [rotated[0], rotated[1], 0];
    const cosSweep = dot(planarBefore, planarAfter) / (length(planarBefore) * length(planarAfter));
    const sweep = Math.acos(Math.max(-1, Math.min(1, cosSweep)));
    if (r > inner && r < outer) assert.ok(sweep <= prevSweep + 1e-9, `sweep should not increase past r=${r}`);
    prevSweep = sweep;
  }
});

test('CHARYBDIS: axial component (height along the rotation axis) is always preserved exactly — a real radial twist, not a 3D scramble', () => {
  const center = [1, 2, 3];
  const axis = normalize([1, 1, 0]);
  const angle = 0.6;
  const p = [5, -2, 9];
  const before = dot(sub(p, center), axis);
  const after = dot(sub(charybdisPoint(p, center, axis, 3, 10, angle), center), axis);
  assert.ok(Math.abs(before - after) < 1e-9);
});

// ============================================================
// CHARYBDIS FALLOFF RAMP (kernel/falloff.mjs) — the falloff is data now,
// and the default is BIT-IDENTICAL to the expression it replaced. Proven
// against real deformed POINTS here, not just against the ramp evaluator
// in isolation: a scalar test in falloff.test.mjs cannot catch a wrong
// argument order or a dropped end-branch inside this function.
// ============================================================
test('CHARYBDIS: an omitted / null / explicit-default ramp all give BIT-IDENTICAL points', () => {
  const center = [1, -2, 4];
  const axis = normalize([0.3, 1, 0.2]);
  const angle = 1.15;
  const inner = 4, outer = 17;
  const probes = [
    [0, 0, 0], [1, -2, 4], [3, 1, 5], [9, -6, 2], [17, 0, 0],
    [22, 3, -8], [-14, 5, 1], [4.0001, -2, 4], [0.5, 0.5, 0.5],
  ];
  for (const p of probes) {
    const omitted = charybdisPoint(p, center, axis, inner, outer, angle);
    const explicitNull = charybdisPoint(p, center, axis, inner, outer, angle, null);
    const explicitDefault = charybdisPoint(p, center, axis, inner, outer, angle, DEFAULT_FALLOFF_RAMP);
    for (let i = 0; i < 3; i++) {
      assert.ok(Object.is(omitted[i], explicitNull[i]), `null ramp differs at ${p} comp ${i}`);
      assert.ok(Object.is(omitted[i], explicitDefault[i]), `explicit default ramp differs at ${p} comp ${i}`);
    }
  }
});

test('CHARYBDIS: a REVERSED ramp genuinely inverts the falloff — the knob does real work', () => {
  const center = [0, 0, 0];
  const axis = normalize([0, 0, 1]);
  const angle = Math.PI / 2;
  const inner = 5, outer = 15;
  // Nothing at the center, everything at the rim: the exact opposite of
  // the default. A point just inside innerRadius must now barely move,
  // and a point just inside outerRadius must sweep the full angle.
  const reversed = { interp: 'smooth', stops: [[0, 0], [1, 1]] };
  const nearInner = [5.001, 0, 0];
  const nearOuter = [14.999, 0, 0];
  const stillInner = charybdisPoint(nearInner, center, axis, inner, outer, angle, reversed);
  assert.ok(length(sub(stillInner, nearInner)) < 1e-3, 'reversed ramp should leave the inner rim essentially put');
  const sweptOuter = charybdisPoint(nearOuter, center, axis, inner, outer, angle, reversed);
  const fullSweep = rotatePoint(nearOuter, center, axis, angle);
  assert.ok(length(sub(sweptOuter, fullSweep)) < 1e-2, 'reversed ramp should sweep the outer rim fully');
  // ...and the DEFAULT ramp does the opposite on the same two points,
  // so this measures the ramp, not the fixture.
  assert.ok(length(sub(charybdisPoint(nearInner, center, axis, inner, outer, angle), nearInner)) > 1);
  assert.ok(length(sub(charybdisPoint(nearOuter, center, axis, inner, outer, angle), nearOuter)) < 1e-2);
});

test('CHARYBDIS: an edited ramp is honoured at BOTH end radii, not overridden by a leftover hardcoded 1/0', () => {
  // The specific defect the three-branch form would have hidden: with the
  // old `if (r <= inner) weight = 1; else if (r >= outer) weight = 0;`
  // structure, a ramp that says otherwise at its own ends would be
  // silently ignored outside the band. Here the ramp holds a real 0.5
  // everywhere, so EVERY point — deep inside, between, far outside —
  // must sweep exactly half the angle.
  const center = [0, 0, 0];
  const axis = normalize([0, 0, 1]);
  const angle = 1.0;
  const inner = 5, outer = 15;
  const half = { interp: 'linear', stops: [[0, 0.5], [1, 0.5]] };
  for (const p of [[1, 0, 0], [5, 0, 0], [10, 0, 2], [15, 0, 0], [40, 0, -3]]) {
    const got = charybdisPoint(p, center, axis, inner, outer, angle, half);
    const expected = rotatePoint(p, center, axis, angle * 0.5);
    assert.ok(length(sub(got, expected)) < 1e-9, `half-strength ramp not honoured at ${p}`);
  }
});

test('CHARYBDIS: a degenerate band (outer <= inner) stays finite and keeps the original hard split', () => {
  const center = [0, 0, 0];
  const axis = normalize([0, 0, 1]);
  const angle = 0.8;
  for (const [inner, outer] of [[5, 5], [10, 4]]) {
    const insideish = charybdisPoint([2, 0, 1], center, axis, inner, outer, angle);
    const outsideish = charybdisPoint([40, 0, 1], center, axis, inner, outer, angle);
    for (const v of [...insideish, ...outsideish]) assert.ok(Number.isFinite(v), `non-finite at inner=${inner} outer=${outer}`);
    // At or below innerRadius still means full strength; beyond it, none.
    assert.ok(length(sub(insideish, rotatePoint([2, 0, 1], center, axis, angle))) < 1e-9);
    assert.ok(length(sub(outsideish, [40, 0, 1])) < 1e-9);
  }
});

test('transformPoint dispatcher carries a frame.falloff ramp through to charybdis', () => {
  const frame = {
    center: [1, 2, 3], axisDir: normalize([1, 0, 1]), planeNormal: normalize([0, 1, 0]),
    span: 25, innerRadius: 3, outerRadius: 12,
    falloff: { interp: 'linear', stops: [[0, 1], [1, 0]] },
  };
  const p = [8, -4, 6];
  assert.deepEqual(
    transformPoint('charybdis', frame, 0.9, p),
    charybdisPoint(p, frame.center, frame.axisDir, frame.innerRadius, frame.outerRadius, 0.9, frame.falloff),
  );
  // And it is genuinely a DIFFERENT result from the default ramp, so the
  // dispatcher is provably passing it rather than dropping it silently.
  assert.notDeepEqual(
    transformPoint('charybdis', frame, 0.9, p),
    charybdisPoint(p, frame.center, frame.axisDir, frame.innerRadius, frame.outerRadius, 0.9),
  );
});

test('transformPoint dispatcher matches each dedicated function exactly, for all 7 types', () => {
  const frame = { center: [1, 2, 3], axisDir: normalize([1, 0, 1]), planeNormal: normalize([0, 1, 0]), span: 25, innerRadius: 3, outerRadius: 12 };
  const p = [8, -4, 6];
  assert.deepEqual(transformPoint('rotate', frame, 0.5, p), rotatePoint(p, frame.center, frame.axisDir, 0.5));
  assert.deepEqual(transformPoint('scale', frame, 1.8, p), scalePoint(p, frame.center, frame.axisDir, 1.8));
  assert.deepEqual(transformPoint('shear', frame, 0.4, p), shearPoint(p, frame.center, frame.axisDir, frame.planeNormal, 0.4));
  assert.deepEqual(transformPoint('taper', frame, 0.3, p), taperPoint(p, frame.center, frame.axisDir, frame.span, 0.3));
  assert.deepEqual(transformPoint('twist', frame, 1.1, p), twistPoint(p, frame.center, frame.axisDir, frame.span, 1.1));
  assert.deepEqual(transformPoint('bend', frame, 0.7, p), bendPoint(p, frame.center, frame.axisDir, frame.span, frame.planeNormal, 0.7));
  assert.deepEqual(transformPoint('charybdis', frame, 0.9, p), charybdisPoint(p, frame.center, frame.axisDir, frame.innerRadius, frame.outerRadius, 0.9));
  assert.equal(transformPoint('nonsense', frame, 1, p), null);
});

test('every type produces a finite result across a battery of points, no NaN/Infinity anywhere', () => {
  const frame = { center: [0, 0, 0], axisDir: normalize([0, 0, 1]), planeNormal: normalize([1, 0, 0]), span: 10, innerRadius: 2, outerRadius: 8 };
  const points = [[0, 0, 0], [3, 4, 5], [-7, 2, -1], [15, -15, 3], [0.001, 0, 0.001]];
  for (const type of TRANSFORM_TYPES) {
    for (const p of points) {
      const r = transformPoint(type, frame, 0.42, p);
      for (const c of r) assert.ok(Number.isFinite(c), `${type} produced a non-finite component for ${p}`);
    }
  }
});
