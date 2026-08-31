import test from 'node:test';
import assert from 'node:assert/strict';
import { curvePoint } from '../kernel/curve.mjs';
import { surfacePoint, isFiniteNet, surfaceClosure, nakedEdgeCount } from '../kernel/surface.mjs';
import { makeArc, makeCircle, revolve, extrude } from '../kernel/primitives.mjs';

test('makeCircle at an arbitrary center/radius/plane stays exactly on that circle', () => {
  const center = [5, -3, 2];
  // an arbitrary orthonormal in-plane basis (not axis-aligned, on purpose)
  const xAxis = [Math.SQRT1_2, Math.SQRT1_2, 0];
  const yAxis = [0, 0, 1]; // perpendicular to xAxis, unit length
  const radius = 7;
  const circle = makeCircle(center, xAxis, yAxis, radius);
  for (let u = 0; u <= 4; u += 0.07) {
    const p = curvePoint(circle, u);
    const d = Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
    assert.ok(Math.abs(d - radius) < 1e-9, `u=${u} d=${d}`);
  }
});

test('makeArc for a partial sweep (e.g. 270deg) stays on the circle and matches endpoints', () => {
  const center = [0, 0, 0];
  const xAxis = [1, 0, 0], yAxis = [0, 1, 0];
  const sweep = 1.5 * Math.PI; // 270 degrees, needs 3 spans of 90deg
  const arc = makeArc(center, xAxis, yAxis, 2, 0, sweep);
  for (let u = 0; u <= 3; u += 0.05) {
    const p = curvePoint(arc, u);
    const d = Math.hypot(p[0], p[1], p[2]);
    assert.ok(Math.abs(d - 2) < 1e-9, `u=${u} d=${d}`);
  }
  const start = curvePoint(arc, 0);
  const end = curvePoint(arc, 3);
  assert.ok(Math.abs(start[0] - 2) < 1e-9 && Math.abs(start[1]) < 1e-9);
  assert.ok(Math.abs(end[0]) < 1e-9 && Math.abs(end[1] - (-2)) < 1e-9); // 270deg from (2,0) -> (0,-2)
});

// "Rebuild" up/down leaves the circle alone and only changes how many
// control points describe it — more/fewer via segments,
// staying EXACTLY circular at every level, never an approximating refit.
test('makeCircle with more segments (a "rebuild") stays exactly circular, with proportionally more control points', () => {
  const center = [1, 2, 3], xAxis = [1, 0, 0], yAxis = [0, 1, 0], radius = 6;
  const base = makeCircle(center, xAxis, yAxis, radius); // default segments=4
  const rebuiltUp = makeCircle(center, xAxis, yAxis, radius, 8);
  const rebuiltDown = makeCircle(center, xAxis, yAxis, radius, 1); // clamped back up to the natural minimum (4)
  assert.equal(base.ctrlPts.length, 2 * 4 + 1);
  assert.equal(rebuiltUp.ctrlPts.length, 2 * 8 + 1);
  assert.equal(rebuiltDown.ctrlPts.length, 2 * 4 + 1, 'requesting FEWER than the natural minimum (4 spans for a full sweep) clamps back up to it, not below');
  for (const circle of [base, rebuiltUp, rebuiltDown]) {
    for (let u = 0; u <= circle.knots[circle.knots.length - 1]; u += 0.1) {
      const p = curvePoint(circle, u);
      const d = Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
      assert.ok(Math.abs(d - radius) < 1e-9, `segments=${(circle.ctrlPts.length - 1) / 2}, u=${u}: d=${d}, expected exactly ${radius}`);
    }
  }
});

test('revolve of a straight vertical profile reproduces a cylinder (constant radius)', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[3, 0, 0, 1], [3, 0, 10, 1]] };
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  for (let u = 0; u <= 1; u += 0.25) {
    for (let v = 0; v <= 4; v += 0.1) {
      const p = surfacePoint(srf, u, v);
      const r = Math.hypot(p[0], p[1]);
      assert.ok(Math.abs(r - 3) < 1e-9, `u=${u} v=${v} r=${r}`);
      assert.ok(p[2] >= -1e-9 && p[2] <= 10 + 1e-9);
    }
  }
});

test('revolve handles a profile point ON the axis (pole row, e.g. a dome apex)', () => {
  // Profile: base out at r=4, apex on the axis (r=0) — like a simple dome/lid.
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[4, 0, 0, 1], [0, 0, 3, 1]] };
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  // apex row (u=1) should evaluate to (0,0,3) regardless of v
  for (let v = 0; v <= 4; v += 0.3) {
    const p = surfacePoint(srf, 1, v);
    assert.ok(Math.abs(p[0]) < 1e-9 && Math.abs(p[1]) < 1e-9 && Math.abs(p[2] - 3) < 1e-9);
  }
});

test('extrude produces a ruled surface at constant offset along the direction', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 2);
  const srf = extrude(profile, [0, 0, 1], 5);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  for (let u = 0; u <= 4; u += 0.2) {
    const base = surfacePoint(srf, u, 0);
    const top = surfacePoint(srf, u, 1);
    assert.ok(Math.abs(base[2] - 0) < 1e-9);
    assert.ok(Math.abs(top[2] - 5) < 1e-9);
    assert.ok(Math.abs(Math.hypot(base[0], base[1]) - 2) < 1e-9);
    assert.ok(Math.abs(Math.hypot(top[0], top[1]) - 2) < 1e-9);
  }
});

test('extrude with draftAngleDeg=0 is byte-identical to the un-tapered extrude (regression safety, the default is truly unchanged)', () => {
  const profile = makeCircle([1, -2, 0], [1, 0, 0], [0, 1, 0], 3);
  const plain = extrude(profile, [0, 0, 1], 8);
  const tapered0 = extrude(profile, [0, 0, 1], 8, 0);
  assert.deepEqual(tapered0.ctrlNet, plain.ctrlNet);
});

test('extrude with a nonzero draftAngleDeg grows the top row radially from the profile\'s own centroid by exactly distance*tan(angle) — exact for a regular polygon, where every control point is genuinely equidistant from center', () => {
  // A square built as a degree-1 4-point closed profile (all weights 1) —
  // every corner is exactly equidistant from the centroid, so the
  // "radial grow from centroid" simplification IS a true, exact offset
  // here, not an approximation — a clean case to prove the formula.
  const square = {
    degree: 1, knots: [0, 0, 1, 2, 3, 4, 4],
    ctrlPts: [[-5, -5, 0, 1], [5, -5, 0, 1], [5, 5, 0, 1], [-5, 5, 0, 1]],
  };
  const distance = 10, angleDeg = 30;
  const srf = extrude(square, [0, 0, 1], distance, angleDeg);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  const delta = distance * Math.tan(angleDeg * Math.PI / 180);
  const baseRadius = Math.hypot(5, 5); // every corner's distance from the origin centroid
  for (const [bottom, top] of srf.ctrlNet) {
    const bottomR = Math.hypot(bottom[0], bottom[1]);
    const topR = Math.hypot(top[0], top[1]);
    assert.ok(Math.abs(bottomR - baseRadius) < 1e-9, 'bottom row is untouched by the taper');
    assert.ok(Math.abs(topR - (baseRadius + delta)) < 1e-9, `top row grew radially by exactly distance*tan(angle): got ${topR}, expected ${baseRadius + delta}`);
    assert.ok(Math.abs(top[2] - distance) < 1e-9, 'the taper only changes X/Y, not the extrude distance along Z');
  }
});

test('extrude with a NEGATIVE draftAngleDeg shrinks the top row (an inward taper, not just an outward one)', () => {
  const square = {
    degree: 1, knots: [0, 0, 1, 2, 3, 4, 4],
    ctrlPts: [[-5, -5, 0, 1], [5, -5, 0, 1], [5, 5, 0, 1], [-5, 5, 0, 1]],
  };
  const srf = extrude(square, [0, 0, 1], 10, -15);
  const baseRadius = Math.hypot(5, 5);
  for (const [, top] of srf.ctrlNet) {
    const topR = Math.hypot(top[0], top[1]);
    assert.ok(topR < baseRadius, `a negative draft angle should shrink the top row inward, got ${topR} vs base ${baseRadius}`);
  }
});

// EXTRUDE vDegree — an extrusion may carry a higher degree in the
// direction of extrusion than the profile's own. Optional,
// defaults to 1 (proven above, byte-identical when omitted). A higher
// vDegree degree-elevates each row's own straight bottom-to-top ruling
// line via kernel/knots.mjs's degreeElevateCurve — a real, exact,
// shape-preserving operation, so the resulting surface must still be
// PERFECTLY straight/flat along V at every (u,v), not just at the two
// original endpoints, despite now having real intermediate control
// points to drag.
test('extrude with vDegree omitted (or 1) is byte-identical to before — the new param changes nothing unless asked for', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 4);
  const withoutParam = extrude(profile, [0, 0, 1], 6);
  const explicit1 = extrude(profile, [0, 0, 1], 6, 0, 1);
  assert.deepEqual(explicit1.ctrlNet, withoutParam.ctrlNet);
  assert.equal(explicit1.degV, 1);
});
test('extrude with vDegree=3 gives 4 control points per row (not 2), and every (u,v) point still lies EXACTLY on the original straight ruling line', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 5);
  const direction = [0, 0, 1], distance = 12;
  const srf = extrude(profile, direction, distance, 0, 3);
  assert.equal(srf.degV, 3);
  assert.equal(srf.ctrlNet[0].length, 4, 'degree-3 (a single Bezier span) needs exactly 4 control points per row, not the original 2');
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  for (let u = 0; u <= 4; u += 0.31) {
    const base = surfacePoint(srf, u, 0);
    for (let v = 0; v <= 1; v += 0.1) {
      const p = surfacePoint(srf, u, v);
      // The true straight ruling line at this u: base + v*distance along direction.
      const expected = [base[0], base[1], base[2] + v * distance];
      const d = Math.hypot(p[0] - expected[0], p[1] - expected[1], p[2] - expected[2]);
      assert.ok(d < 1e-9, `u=${u.toFixed(2)} v=${v.toFixed(2)}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(p)} (d=${d})`);
    }
  }
});
test('extrude with vDegree=3 AND a nonzero draft: every row still traces its own real (radially-grown) straight ruling line exactly, at any v, not just row endpoints', () => {
  const square = {
    degree: 1, knots: [0, 0, 1, 2, 3, 4, 4],
    ctrlPts: [[-5, -5, 0, 1], [5, -5, 0, 1], [5, 5, 0, 1], [-5, 5, 0, 1]],
  };
  const distance = 10, angleDeg = 20;
  const srf = extrude(square, [0, 0, 1], distance, angleDeg, 3);
  assert.equal(srf.degV, 3);
  assert.equal(isFiniteNet(srf.ctrlNet), true);
  for (let u = 0; u <= 4; u += 0.37) {
    const bottom = surfacePoint(srf, u, 0);
    const top = surfacePoint(srf, u, 1);
    for (let v = 0; v <= 1; v += 0.25) {
      const p = surfacePoint(srf, u, v);
      const expected = [bottom[0] + v * (top[0] - bottom[0]), bottom[1] + v * (top[1] - bottom[1]), bottom[2] + v * (top[2] - bottom[2])];
      const d = Math.hypot(p[0] - expected[0], p[1] - expected[1], p[2] - expected[2]);
      assert.ok(d < 1e-9, `u=${u.toFixed(2)} v=${v}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(p)}`);
    }
  }
});

// INSPECT/WHAT's own "naked edge count" — genuinely
// computable from the control net's own U/V wrap-around, no Brep needed,
// as far as a single untrimmed-face surface can honestly go.
test('surfaceClosure/nakedEdgeCount: a FULL revolve of an OPEN profile is a cylinder — closed in the sweep (V), open in the profile (U), 2 naked edges (top+bottom rims)', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[3, 0, 0, 1], [3, 0, 10, 1]] };
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, false, 'the profile itself never wraps — its two ends are the two rims');
  assert.equal(closedV, true, 'a full 2*PI revolve wraps the sweep direction into a seam, not a boundary');
  assert.equal(nakedEdgeCount(srf), 2);
});

test('surfaceClosure/nakedEdgeCount: a PARTIAL revolve (270deg) of an open profile has all 4 sides free', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[3, 0, 0, 1], [3, 0, 10, 1]] };
  const srf = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 1.5 * Math.PI);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, false);
  assert.equal(closedV, false, 'a partial sweep never reconnects to itself');
  assert.equal(nakedEdgeCount(srf), 4);
});

test('surfaceClosure/nakedEdgeCount: extrude of a CLOSED profile (a circle) is a tube — closed in the profile (U), open in the extrude direction (V), 2 naked edges (the two rim openings)', () => {
  const profile = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 2);
  const srf = extrude(profile, [0, 0, 1], 5);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, true, 'the circle profile itself already wraps (its own first/last control points coincide)');
  assert.equal(closedV, false, 'a straight extrude never wraps — degV is always 1, two distinct ends');
  assert.equal(nakedEdgeCount(srf), 2);
});

test('surfaceClosure/nakedEdgeCount: extrude of an OPEN profile (a straight line) is a flat ruled patch — all 4 sides free', () => {
  const profile = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[0, 0, 0, 1], [10, 0, 0, 1]] };
  const srf = extrude(profile, [0, 1, 0], 5);
  const { closedU, closedV } = surfaceClosure(srf);
  assert.equal(closedU, false);
  assert.equal(closedV, false);
  assert.equal(nakedEdgeCount(srf), 4);
});
