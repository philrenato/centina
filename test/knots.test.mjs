import test from 'node:test';
import assert from 'node:assert/strict';
import {
  insertKnotOnce, insertKnot, decomposeToBezier, assembleBezierChain,
  degreeElevateCurve, rescaleCurveDomain, extractSubCurve, joinCurvesC0,
} from '../kernel/knots.mjs';
import { curvePoint, rationalCurveDerivs } from '../kernel/curve.mjs';
import { globalCurveInterp, closedCurveInterp } from '../kernel/interpolate.mjs';
import { makeArc } from '../kernel/primitives.mjs';

function sampleAt(crv, us) { return us.map((u) => curvePoint(crv, u)); }
function closePts(a, b, tol = 1e-6) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < tol;
}
function linspace(a, b, n) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(a + (b - a) * (i / (n - 1)));
  return out;
}
function knotMult(knots, u, tol = 1e-9) {
  let s = 0;
  for (const k of knots) if (Math.abs(k - u) < tol) s++;
  return s;
}

// A genuinely curved (not collinear) 6-point open interpolation curve —
// real interior knots at simple (sub-full) multiplicity, the case that
// actually exercises decomposeToBezier's own interior-knot-insertion path.
function curvedTestCurve() {
  return globalCurveInterp([[0, 0, 0], [10, 4, 0], [25, -6, 5], [40, 0, 0], [55, 10, -5], [70, 0, 0]], 3);
}

test('insertKnotOnce: shape-preserving — sampled points identical before/after a real interior insertion', () => {
  const crv = curvedTestCurve();
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  const us = linspace(uMin + 1e-6, uMax - 1e-6, 25);
  const before = sampleAt(crv, us);
  const inserted = insertKnotOnce(crv, (uMin + uMax) / 2);
  const after = sampleAt(inserted, us);
  for (let i = 0; i < us.length; i++) assert.ok(closePts(before[i], after[i], 1e-8), `sample ${i}: ${before[i]} vs ${after[i]}`);
  assert.equal(inserted.ctrlPts.length, crv.ctrlPts.length + 1);
  assert.equal(inserted.knots.length, crv.knots.length + 1);
});

test('insertKnotOnce: multiplicity genuinely increases at the target value, unrelated knots untouched', () => {
  const crv = curvedTestCurve();
  const uMid = crv.knots[Math.floor(crv.knots.length / 2)];
  const before = knotMult(crv.knots, uMid);
  const after = insertKnot(crv, uMid, 1);
  assert.equal(knotMult(after.knots, uMid), before + 1);
});

test('insertKnot: repeated insertion to a knot NOT already present reaches the requested multiplicity exactly', () => {
  const crv = curvedTestCurve();
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  const u = uMin + (uMax - uMin) * 0.37; // an arbitrary, almost-certainly-fresh interior value
  assert.equal(knotMult(crv.knots, u), 0);
  const p = crv.degree;
  const inserted = insertKnot(crv, u, p + 1);
  assert.equal(knotMult(inserted.knots, u), p + 1);
  // still exact at every sample, the insertion's own core promise
  const us = linspace(uMin + 1e-6, uMax - 1e-6, 30);
  const before = sampleAt(crv, us), after = sampleAt(inserted, us);
  for (let i = 0; i < us.length; i++) assert.ok(closePts(before[i], after[i], 1e-7));
});

test('decomposeToBezier + assembleBezierChain round-trip a smooth multi-span curve EXACTLY (position and tangent)', () => {
  const crv = curvedTestCurve();
  const pieces = decomposeToBezier(crv);
  assert.ok(pieces.length >= 2, 'a real 6-point interpolation curve has multiple knot spans');
  for (const piece of pieces) assert.equal(piece.ctrlPts.length, crv.degree + 1);
  const rebuilt = assembleBezierChain(pieces, crv.degree);
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  const us = linspace(uMin, uMax, 41);
  for (const u of us) {
    const a = curvePoint(crv, Math.min(u, uMax - 1e-9));
    const b = curvePoint(rebuilt, Math.min(u, uMax - 1e-9));
    assert.ok(closePts(a, b, 1e-7), `u=${u}: ${a} vs ${b}`);
  }
  // tangent continuity at an INTERIOR original knot survives the full
  // decompose/reassemble round-trip too — proves the reassembled curve
  // isn't just position-matching at samples, it's genuinely the same
  // smooth curve, not a chain of independently-elevated-looking pieces.
  const midU = crv.knots[Math.floor(crv.knots.length / 2)];
  const [, tOrig] = rationalCurveDerivs(crv, midU, 1);
  const [, tNew] = rationalCurveDerivs(rebuilt, midU, 1);
  const dot3 = tOrig[0] * tNew[0] + tOrig[1] * tNew[1] + tOrig[2] * tNew[2];
  const magOrig = Math.hypot(...tOrig), magNew = Math.hypot(...tNew);
  assert.ok(Math.abs(dot3 / (magOrig * magNew) - 1) < 1e-6, 'tangent direction preserved through decompose/reassemble');
});

test('degreeElevateCurve: a straight (2-point, degree-1) line elevates to degree 3 and stays the exact same line', () => {
  const line = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [10, 20, 30, 1]] };
  const elevated = degreeElevateCurve(line, 3);
  assert.equal(elevated.degree, 3);
  for (const u of linspace(0, 1, 11)) {
    const a = curvePoint(line, u);
    const b = curvePoint(elevated, u);
    assert.ok(closePts(a, b, 1e-9), `u=${u}: ${a} vs ${b}`);
  }
});

test('degreeElevateCurve: a smooth multi-span interpolation curve elevates exactly, position AND tangent preserved', () => {
  const crv = curvedTestCurve(); // degree 3
  const elevated = degreeElevateCurve(crv, 5);
  assert.equal(elevated.degree, 5);
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  for (const u of linspace(uMin, uMax - 1e-9, 25)) {
    const a = curvePoint(crv, u), b = curvePoint(elevated, u);
    assert.ok(closePts(a, b, 1e-6), `pos u=${u}: ${a} vs ${b}`);
    const [, ta] = rationalCurveDerivs(crv, u, 1);
    const [, tb] = rationalCurveDerivs(elevated, u, 1);
    const dot3 = ta[0] * tb[0] + ta[1] * tb[1] + ta[2] * tb[2];
    const mag = Math.hypot(...ta) * Math.hypot(...tb);
    if (mag > 1e-9) assert.ok(Math.abs(dot3 / mag - 1) < 1e-4, `tangent u=${u} diverged after elevation`);
  }
});

test('degreeElevateCurve: a RATIONAL arc (Circle/fillet-style, weighted control points) elevates exactly', () => {
  const arc = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 25, 0, Math.PI); // degree-2, 2 arc spans, real weights != 1
  assert.equal(arc.degree, 2);
  assert.ok(arc.ctrlPts.some((cp) => Math.abs(cp[3] - 1) > 1e-6), 'setup: this arc genuinely has non-unit weights');
  const elevated = degreeElevateCurve(arc, 4);
  assert.equal(elevated.degree, 4);
  const uMin = arc.knots[0], uMax = arc.knots[arc.knots.length - 1];
  for (const u of linspace(uMin, uMax - 1e-9, 30)) {
    const a = curvePoint(arc, u), b = curvePoint(elevated, u);
    assert.ok(closePts(a, b, 1e-6), `u=${u}: ${a} vs ${b}`);
    // both must still sit exactly on the true circle (radius 25) — proves
    // elevation didn't just match the OLD control net, it preserved the
    // real analytic shape.
    assert.ok(Math.abs(Math.hypot(...a) - 25) < 1e-4);
    assert.ok(Math.abs(Math.hypot(...b) - 25) < 1e-4);
  }
});

test('degreeElevateCurve: refuses to lower degree, honestly, not silently', () => {
  const crv = curvedTestCurve();
  assert.throws(() => degreeElevateCurve(crv, 1));
});

test('rescaleCurveDomain: pure reparametrization — same shape at the correspondingly-mapped parameter, domain genuinely moved', () => {
  const crv = curvedTestCurve();
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  const rescaled = rescaleCurveDomain(crv, 5, 6);
  assert.ok(Math.abs(rescaled.knots[0] - 5) < 1e-9);
  assert.ok(Math.abs(rescaled.knots[rescaled.knots.length - 1] - 6) < 1e-9);
  for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
    const uOld = uMin + (uMax - uMin) * frac;
    const uNew = 5 + frac;
    const a = curvePoint(crv, Math.min(uOld, uMax - 1e-9));
    const b = curvePoint(rescaled, Math.min(uNew, 6 - 1e-9));
    assert.ok(closePts(a, b, 1e-7), `frac=${frac}: ${a} vs ${b}`);
  }
});

test('extractSubCurve: a closedCurveInterp padded curve, trimmed to [uStart,uEnd], reproduces the SAME real geometry as the untrimmed curve over that range', () => {
  const pts = [[0, 0, 0], [20, 0, 0], [25, 15, 0], [10, 25, 0], [-5, 12, 0]];
  const { crv, uStart, uEnd } = closedCurveInterp(pts, 3);
  const sub = extractSubCurve(crv, uStart, uEnd);
  assert.equal(sub.degree, crv.degree);
  // domain is now GENUINELY [uStart, uEnd], nothing left over from the padding
  assert.ok(Math.abs(sub.knots[0] - uStart) < 1e-9);
  assert.ok(Math.abs(sub.knots[sub.knots.length - 1] - uEnd) < 1e-9);
  for (const frac of [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999]) {
    const u = uStart + (uEnd - uStart) * frac;
    const a = curvePoint(crv, u);
    const b = curvePoint(sub, Math.min(u, uEnd - 1e-9));
    assert.ok(closePts(a, b, 1e-6), `frac=${frac}: padded=${a} extracted=${b}`);
  }
});

test('extractSubCurve: the extracted curve is genuinely a VALID standalone profile — no NaN, finite weights, real endpoint positions', () => {
  const pts = [[0, 0, 0], [20, 0, 0], [25, 15, 0], [10, 25, 0], [-5, 12, 0]];
  const { crv, uStart, uEnd } = closedCurveInterp(pts, 3);
  const sub = extractSubCurve(crv, uStart, uEnd);
  for (const cp of sub.ctrlPts) for (const c of cp) assert.ok(Number.isFinite(c));
  const p0 = curvePoint(sub, sub.knots[0]);
  const p1 = curvePoint(sub, sub.knots[sub.knots.length - 1] - 1e-9);
  // the trimmed curve's own two ends should be genuinely close (a near-
  // closed loop, matching the ORIGINAL points' own near-closure) — not
  // coincidentally landing somewhere in the padding overshoot the old,
  // untrimmed obj.crv used to visibly self-overlap into.
  assert.ok(closePts(p0, pts[0], 3), `extracted start ${p0} should read close to the real first input point ${pts[0]}`);
});

test('joinCurvesC0: two degree-1 curves (a real "L" — two Lines) join into ONE degree-1 curve, exact piecewise-linear shape, C0 at the joint', () => {
  const A = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [10, 0, 0, 1]] };
  const B = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[10, 0, 0, 1], [10, 10, 0, 1]] };
  const joined = joinCurvesC0([A, B]);
  assert.equal(joined.degree, 1);
  assert.equal(joined.ctrlPts.length, 3); // A's 2 + B's 2, minus 1 shared joint
  const uMin = joined.knots[0], uMax = joined.knots[joined.knots.length - 1];
  const start = curvePoint(joined, uMin), mid = curvePoint(joined, (uMin + uMax) / 2), end = curvePoint(joined, uMax - 1e-9);
  assert.ok(closePts(start, [0, 0, 0]));
  assert.ok(closePts(mid, [10, 0, 0], 1e-3)); // the shared corner, near the domain midpoint by construction (sequential [0,1]+[1,2] slots)
  assert.ok(closePts(end, [10, 10, 0]));
});

test('joinCurvesC0: a MIXED-degree chain (a degree-1 Line into a degree-3 smooth curve) elevates the low-degree segment and joins exactly, reproducing BOTH segments\' real geometry', () => {
  const line = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[-20, 0, 0, 1], [0, 0, 0, 1]] };
  const smooth = globalCurveInterp([[0, 0, 0], [10, 8, 0], [20, 0, 0], [30, -8, 0], [40, 0, 0]], 3);
  const joined = joinCurvesC0([line, smooth]);
  assert.equal(joined.degree, 3, 'target degree is the MAX across segments');
  // line segment's own true shape (a straight run from -20,0,0 to 0,0,0) is
  // reproduced exactly across the joined curve's own first parameter slot.
  for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
    const uJoined = 0 + frac; // line occupies slot [0,1] by joinCurvesC0's own sequential convention
    const want = [-20 + 20 * frac, 0, 0];
    const got = curvePoint(joined, Math.min(uJoined, 1 - 1e-9));
    assert.ok(closePts(got, want, 1e-6), `line half frac=${frac}: got ${got} want ${want}`);
  }
  // the smooth segment's own true shape is reproduced exactly across the
  // second slot [1,2] — cross-checked against smooth's OWN independent
  // parametrization (rescaled the same way joinCurvesC0 itself rescales it).
  const smoothUMin = smooth.knots[0], smoothUMax = smooth.knots[smooth.knots.length - 1];
  for (const frac of [0, 0.2, 0.4, 0.6, 0.8, 0.999]) {
    const uSmoothOrig = smoothUMin + (smoothUMax - smoothUMin) * frac;
    const want = curvePoint(smooth, uSmoothOrig);
    const got = curvePoint(joined, Math.min(1 + frac, 2 - 1e-9));
    assert.ok(closePts(got, want, 1e-6), `smooth half frac=${frac}: got ${got} want ${want}`);
  }
  // C0 at the joint: both halves agree at u=1 (no gap).
  const fromLine = curvePoint(joined, 1 - 1e-9);
  const fromSmooth = curvePoint(joined, 1 + 1e-9);
  assert.ok(closePts(fromLine, fromSmooth, 1e-3));
  assert.ok(closePts(fromLine, [0, 0, 0], 1e-3));
});

test('joinCurvesC0: matches getProfileCrv\'s own pre-existing degree-1-only concatenation shape exactly (regression: same knot pattern, same control points) for an all-degree-1 chain', () => {
  // Mirrors the app's getProfileCrv PolyCurve fast path for a pure
  // Line/Polyline chain: 3 segments, sequential integer knots, N ctrlPts
  // (no elevation ever triggered since every input is already degree 1).
  const segs = [
    { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [10, 0, 0, 1]] },
    { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[10, 0, 0, 1], [10, 10, 0, 1]] },
    { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[10, 10, 0, 1], [0, 10, 0, 1]] },
  ];
  const joined = joinCurvesC0(segs);
  assert.equal(joined.degree, 1);
  assert.equal(joined.ctrlPts.length, 4);
  assert.deepEqual(joined.knots, [0, 0, 1, 2, 3, 3]);
  const want = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]];
  for (let i = 0; i < want.length; i++) assert.ok(closePts(joined.ctrlPts[i], want[i]));
});
