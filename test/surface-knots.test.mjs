// KNOT SURGERY ON A SURFACE, and the harmonisation Match Edge needs before a
// row edit means anything.
//
// The claim that carries everything: these operations add control points
// WITHOUT MOVING THE SURFACE. So every test measures the surface by evaluation,
// before and after, rather than inspecting the net — a net that gained the
// right number of points and moved the shape is the exact failure this must
// catch, and it is invisible to a count.
import test from 'node:test';
import assert from 'node:assert/strict';
import { surfacePoint } from '../kernel/surface.mjs';
import { makeArc, revolve } from '../kernel/primitives.mjs';
import {
  surfaceInsertKnot, surfaceElevateDegree, surfaceRescaleDomain,
  harmonizeDirections, interiorKnotMultiplicities, seamDirectionFor,
  degreeIn, knotsIn, countIn,
} from '../kernel/surfaceknots.mjs';
import { matchEdge, tangentDeviationAcross, edgeGap } from '../kernel/matchedge.mjs';
import { insertKnot } from '../kernel/knots.mjs';

const dom = (knots, degree) => [knots[degree], knots[knots.length - 1 - degree]];
// Sample a surface over its own domain, so two DIFFERENT parametrisations of
// the same shape can be compared fairly.
function grid(srf, n = 11) {
  const [uMin, uMax] = dom(srf.knotsU, srf.degU);
  const [vMin, vMax] = dom(srf.knotsV, srf.degV);
  const out = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    out.push(surfacePoint(srf, uMin + ((uMax - uMin) * i) / (n - 1), vMin + ((vMax - vMin) * j) / (n - 1)));
  }
  return out;
}
const worstBetween = (a, b) => {
  let w = 0;
  for (let i = 0; i < a.length; i++) w = Math.max(w, Math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1], a[i][2] - b[i][2]));
  return w;
};

function clampedKnots(n, degree) {
  const k = [];
  for (let i = 0; i <= degree; i++) k.push(0);
  const inner = n - degree - 1;
  for (let i = 1; i <= inner; i++) k.push(i / (inner + 1));
  for (let i = 0; i <= degree; i++) k.push(1);
  return k;
}
function patch(rows, degU = 3, degV = 3) {
  return {
    degU, degV,
    knotsU: clampedKnots(rows.length, degU),
    knotsV: clampedKnots(rows[0].length, degV),
    ctrlNet: rows.map((r) => r.map((p) => [p[0], p[1], p[2], 1])),
  };
}
function curvedPatch(x0, x1, nu = 4, nv = 4, degU = 3, degV = 3) {
  const rows = [];
  for (let i = 0; i < nu; i++) {
    const row = [];
    for (let j = 0; j < nv; j++) {
      const x = x0 + ((x1 - x0) * i) / (nu - 1), y = (30 * j) / (nv - 1);
      row.push([x, y, 0.004 * x * x + 0.002 * y * y]);
    }
    rows.push(row);
  }
  return patch(rows, degU, degV);
}

test('inserting a knot adds a control point and moves nothing', () => {
  const s = curvedPatch(0, 30);
  const before = grid(s);
  const out = surfaceInsertKnot(s, 'v', 0.37, 1);
  assert.equal(countIn(out, 'v'), countIn(s, 'v') + 1, 'one control point per insertion');
  assert.ok(worstBetween(before, grid(out)) < 1e-9, 'the surface is unmoved');
});

test('elevating the degree adds points and moves nothing', () => {
  const s = curvedPatch(0, 30, 4, 4, 3, 2);
  const before = grid(s);
  const out = surfaceElevateDegree(s, 'v', 4);
  assert.equal(degreeIn(out, 'v'), 4);
  assert.ok(countIn(out, 'v') > countIn(s, 'v'), 'elevation adds control points');
  assert.ok(worstBetween(before, grid(out)) < 1e-9, 'the surface is unmoved');
});

test('rescaling a direction\'s domain moves nothing', () => {
  const s = curvedPatch(0, 30);
  const before = grid(s);
  const out = surfaceRescaleDomain(s, 'v', -4, 11);
  assert.equal(knotsIn(out, 'v')[0], -4);
  assert.ok(worstBetween(before, grid(out)) < 1e-9, 'an affine reparametrisation moves no point');
});

test('interior multiplicities ignore the clamped ends and count repeats', () => {
  const knots = [0, 0, 0, 0, 0.25, 0.5, 0.5, 0.75, 1, 1, 1, 1];
  const m = interiorKnotMultiplicities(knots, 3);
  assert.deepEqual(m.map((e) => [e.value, e.count]), [[0.25, 1], [0.5, 2], [0.75, 1]]);
});

test('seamDirectionFor names the direction that RUNS ALONG the edge', () => {
  assert.equal(seamDirectionFor('u0'), 'v');
  assert.equal(seamDirectionFor('u1'), 'v');
  assert.equal(seamDirectionFor('v0'), 'u');
  assert.equal(seamDirectionFor('v1'), 'u');
});

test('⭐ harmonising two mismatched surfaces leaves both shapes untouched', () => {
  // Genuinely different along the seam: 4 vs 6 control points, degree 3 vs 2,
  // and different domains.
  const a = curvedPatch(-30, 0, 4, 4, 3, 3);
  const b = surfaceRescaleDomain(curvedPatch(0, 30, 4, 6, 3, 2), 'v', 0, 4);
  assert.notEqual(countIn(a, 'v'), countIn(b, 'v'), 'the fixture really is mismatched');
  assert.notEqual(degreeIn(a, 'v'), degreeIn(b, 'v'), 'and mismatched in degree too');

  const beforeA = grid(a), beforeB = grid(b);
  const res = harmonizeDirections(a, 'v', b, 'v');
  assert.equal(res.ok, true, res.reason);
  assert.equal(countIn(res.a, 'v'), countIn(res.b, 'v'), 'both now carry the same control count along the seam');
  assert.equal(degreeIn(res.a, 'v'), degreeIn(res.b, 'v'), 'and the same degree');
  assert.deepEqual(knotsIn(res.a, 'v'), knotsIn(res.b, 'v'), 'and the same knot vector');

  assert.ok(worstBetween(beforeA, grid(res.a)) < 1e-9, 'surface A is unmoved by harmonisation');
  assert.ok(worstBetween(beforeB, grid(res.b)) < 1e-9, 'surface B is unmoved by harmonisation');
});

test('⭐ a pair Match Edge REFUSED before now matches, and the break closes', () => {
  // The base is creased against the target AND described differently — the
  // ordinary case: two surfaces that genuinely meet but were not built as a
  // pair.
  //
  // ⚠ THE SHARED EDGE IS BUILT BY KNOT INSERTION, not by evaluating the same
  // formula into two different control nets. Two B-spline nets whose points
  // follow the same rule are NOT the same curve — control points are not
  // interpolated — so a fixture built that way has edges that miss each other
  // by millimetres while looking identical on paper, and every match measured
  // on it would be measuring the gap instead.
  const rows = [];
  for (let i = 0; i < 4; i++) {
    const row = [];
    for (let j = 0; j < 4; j++) {
      const x = -30 + (30 * i) / 3, y = (30 * j) / 3;
      row.push([x, y, 0.002 * y * y + (i === 2 ? 14 : 0)]);
    }
    rows.push(row);
  }
  const base = patch(rows);

  // The base's own u1 edge, refined to 6 control points. Same curve exactly.
  const edge = { degree: base.degV, knots: base.knotsV.slice(), ctrlPts: base.ctrlNet[3].map((p) => p.slice()) };
  let refined = insertKnot(edge, 0.3, 1);
  refined = insertKnot(refined, 0.7, 1);
  assert.equal(refined.ctrlPts.length, 6, 'the refined edge really does carry more control points');

  // A target that OWNS that edge as its u0 and curves away in +x.
  const tRows = [refined.ctrlPts.map((p) => p.slice())];
  for (let i = 1; i < 4; i++) {
    const x = (30 * i) / 3;
    tRows.push(refined.ctrlPts.map((p) => [p[0] + x * (p[3] || 1), p[1], p[2] + 0.004 * x * x * (p[3] || 1), p[3] || 1]));
  }
  const target = {
    degU: 3, degV: refined.degree,
    knotsU: clampedKnots(4, 3),
    knotsV: refined.knots.slice(),
    ctrlNet: tRows,
  };
  // The premise, asserted rather than assumed.
  assert.ok(edgeGap(base.ctrlNet, 'u1', target.ctrlNet, 'u0', 'aligned') === Infinity,
    'the counts differ, so a direct gap cannot even be computed — which is exactly why harmonisation is needed');

  const refused = matchEdge(base, 'u1', target, 'u0', { order: 1 });
  assert.equal(refused.ok, false, 'the fixture must genuinely refuse before harmonisation');
  assert.match(refused.reason, /shared knot vector|control counts/i);

  const h = harmonizeDirections(base, seamDirectionFor('u1'), target, seamDirectionFor('u0'));
  assert.equal(h.ok, true, h.reason);

  // The edges meet after harmonisation — it changed description, not position.
  assert.ok(edgeGap(h.a.ctrlNet, 'u1', h.b.ctrlNet, 'u0', 'aligned') < 1e-9, 'the edges coincide once both are described the same way');

  const before = tangentDeviationAcross(h.a, 'u1', h.b, 'u0');
  assert.ok(before > 20, `the harmonised pair is still creased (${before.toFixed(2)} degrees)`);

  const res = matchEdge(h.a, 'u1', h.b, 'u0', { order: 1 });
  assert.equal(res.ok, true, res.reason);
  const after = tangentDeviationAcross({ ...h.a, ctrlNet: res.net }, 'u1', h.b, 'u0');
  assert.ok(after < 0.05, `and the break closes (${before.toFixed(2)} -> ${after.toFixed(4)} degrees)`);
});

test('a revolve\'s own sweep direction harmonises against a hand-built patch', () => {
  // The real domain mismatch this exists for: a revolve runs 0..4 in its sweep
  // direction while a hand-built patch runs 0..1. Taking a union of two knot
  // vectors on different intervals produces a superset of neither.
  const profile = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 50, -Math.PI / 2, Math.PI);
  const sphere = revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const flat = curvedPatch(0, 30, 4, 4, 3, 3);
  assert.notEqual(knotsIn(sphere, 'v')[knotsIn(sphere, 'v').length - 1], knotsIn(flat, 'v')[knotsIn(flat, 'v').length - 1],
    'the fixture really does have different domains');

  const beforeS = grid(sphere), beforeF = grid(flat);
  const res = harmonizeDirections(sphere, 'v', flat, 'v');
  assert.equal(res.ok, true, res.reason);
  assert.equal(countIn(res.a, 'v'), countIn(res.b, 'v'));
  assert.ok(worstBetween(beforeS, grid(res.a)) < 1e-8, 'the rational sphere is unmoved — weights survive the surgery');
  assert.ok(worstBetween(beforeF, grid(res.b)) < 1e-9, 'and so is the patch');
});
