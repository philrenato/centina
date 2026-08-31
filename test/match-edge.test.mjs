// MATCH EDGE — the maths, before any handle exists. The spec asks for exactly
// this order ("prove the hard math before shipping UI"), and continuity is a
// claim that is very easy to make and very easy to make wrongly: a fold back on
// itself reports a perfect tangent ANGLE, and a control net that looks matched
// says nothing about the surface it actually generates.
//
// So every claim here is measured ON THE SURFACES, through
// surfacePointAndPartials, not read off the control net that was just edited.
import test from 'node:test';
import assert from 'node:assert/strict';
import { surfacePoint, surfacePointAndPartials } from '../kernel/surface.mjs';
import { matchEdge, tangentDeviationAcross, edgeRows, edgeOrientation, edgeGap } from '../kernel/matchedge.mjs';

// Two bicubic patches sharing the plane x = 0. The BASE is deliberately creased
// against the target: its second row kicks up in +z, so an unmatched seam has a
// large tangent break and a matched one has none. Both are non-planar, because
// a plane matches a plane by accident.
function clampedKnots(n, degree) {
  const k = [];
  for (let i = 0; i <= degree; i++) k.push(0);
  const inner = n - degree - 1;
  for (let i = 1; i <= inner; i++) k.push(i / (inner + 1));
  for (let i = 0; i <= degree; i++) k.push(1);
  return k;
}
function patch(rows) {
  return {
    degU: 3, degV: 3,
    knotsU: clampedKnots(rows.length, 3),
    knotsV: clampedKnots(rows[0].length, 3),
    ctrlNet: rows.map((r) => r.map((p) => [p[0], p[1], p[2], 1])),
  };
}
// TARGET occupies x in [0, 30]; its edge at x = 0 is the shared seam (its 'u0').
function makeTarget() {
  const rows = [];
  for (let i = 0; i < 4; i++) {
    const row = [];
    for (let j = 0; j < 4; j++) {
      const x = (30 * i) / 3, y = (30 * j) / 3;
      row.push([x, y, 0.004 * x * x + 0.002 * y * y]); // genuinely curved, so tangents are not all parallel
    }
    rows.push(row);
  }
  return patch(rows);
}
// BASE occupies x in [-30, 0]; its edge at x = 0 is its 'u1'. Its second row is
// pulled far out of the target's tangent plane, so the seam starts creased.
function makeBase() {
  const rows = [];
  for (let i = 0; i < 4; i++) {
    const row = [];
    for (let j = 0; j < 4; j++) {
      const x = -30 + (30 * i) / 3, y = (30 * j) / 3;
      const kick = i === 2 ? 14 : 0; // the crease
      row.push([x, y, 0.002 * y * y + kick]);
    }
    rows.push(row);
  }
  return patch(rows);
}
const withNet = (srf, net) => ({ ...srf, ctrlNet: net });
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

test('the fixture is genuinely creased, or nothing below is measuring a fix', () => {
  const base = makeBase(), target = makeTarget();
  const gap = edgeGap(base.ctrlNet, 'u1', target.ctrlNet, 'u0', 'aligned');
  assert.ok(gap < 1e-9, `the edges must already meet for a match to be legitimate, gap ${gap}`);
  const dev = tangentDeviationAcross(base, 'u1', target, 'u0');
  assert.ok(dev > 20, `the unmatched seam must have a real tangent break, got ${dev.toFixed(2)} degrees`);
});

test('G0 puts the boundary rows exactly together', () => {
  const base = makeBase(), target = makeTarget();
  // Pull the base's edge off the seam first, so G0 has something to close.
  const moved = base.ctrlNet.map((r) => r.map((p) => [...p]));
  for (let j = 0; j < 4; j++) moved[3][j] = [moved[3][j][0], moved[3][j][1], moved[3][j][2] + 9, 1];
  const shifted = withNet(base, moved);
  assert.ok(edgeGap(shifted.ctrlNet, 'u1', target.ctrlNet, 'u0', 'aligned') > 8, 'the fixture is genuinely apart before G0');

  const res = matchEdge(shifted, 'u1', target, 'u0', { order: 0 });
  assert.equal(res.ok, true, res.reason);
  assert.ok(edgeGap(res.net, 'u1', target.ctrlNet, 'u0', 'aligned') < 1e-9, 'G0 closes the gap exactly');

  // And it is a SUBSTITUTION, not a fit: the surfaces agree along the whole
  // shared edge, not merely at the control points.
  const matched = withNet(shifted, res.net);
  let worst = 0;
  for (let i = 0; i <= 8; i++) {
    const s = i / 8;
    worst = Math.max(worst, dist(surfacePoint(matched, 1, s), surfacePoint(target, 0, s)));
  }
  assert.ok(worst < 1e-9, `the edge CURVES coincide, worst ${worst}`);
});

test('⭐ G1 closes the tangent break, measured on the surfaces', () => {
  const base = makeBase(), target = makeTarget();
  const before = tangentDeviationAcross(base, 'u1', target, 'u0');
  const res = matchEdge(base, 'u1', target, 'u0', { order: 1 });
  assert.equal(res.ok, true, res.reason);
  const after = tangentDeviationAcross(withNet(base, res.net), 'u1', target, 'u0');
  assert.ok(after < 0.05, `G1 should leave essentially no tangent break, got ${after.toFixed(4)} degrees (was ${before.toFixed(2)})`);
});

test('⭐ G1 does NOT fold the base back over the target', () => {
  // A fold reports a PERFECT tangent angle while putting the base on the wrong
  // side of the seam — the one failure the angle alone cannot see, so it is
  // checked as its own claim.
  const base = makeBase(), target = makeTarget();
  const res = matchEdge(base, 'u1', target, 'u0', { order: 1 });
  const matched = withNet(base, res.net);
  // Just inside each surface, away from the seam: they must sit on opposite
  // sides in x, since the base owns x < 0 and the target x > 0.
  for (let i = 0; i <= 4; i++) {
    const s = i / 4;
    const b = surfacePoint(matched, 0.85, s);
    const t = surfacePoint(target, 0.15, s);
    assert.ok(b[0] < 0, `the matched base must stay on its own side of the seam, got x = ${b[0].toFixed(3)}`);
    assert.ok(t[0] > 0, `and the target on its own, got x = ${t[0].toFixed(3)}`);
  }
});

test('G1 preserves the base\'s own row spacing — redirected, not cloned', () => {
  const base = makeBase(), target = makeTarget();
  const res = matchEdge(base, 'u1', target, 'u0', { order: 1 });
  const rows = edgeRows(base.ctrlNet, 'u1');
  for (let k = 0; k < rows.second.length; k++) {
    const [bi, bj] = rows.second[k];
    const [b0i, b0j] = rows.boundary[k];
    const wasE = base.ctrlNet[bi][bj], was0 = base.ctrlNet[b0i][b0j];
    const isE = res.net[bi][bj], is0 = res.net[b0i][b0j];
    const wasLen = dist([wasE[0], wasE[1], wasE[2]], [was0[0], was0[1], was0[2]]);
    const isLen = dist([isE[0], isE[1], isE[2]], [is0[0], is0[1], is0[2]]);
    assert.ok(Math.abs(wasLen - isLen) < 1e-9,
      `column ${k}: the second row keeps its distance from the edge (${wasLen.toFixed(4)} -> ${isLen.toFixed(4)}) — cloning the target's magnitude would balloon or flatten the surface`);
  }
});

test('the blend is continuous and monotone from untouched to matched', () => {
  const base = makeBase(), target = makeTarget();
  const at = (blend) => tangentDeviationAcross(withNet(base, matchEdge(base, 'u1', target, 'u0', { order: 1, blend }).net), 'u1', target, 'u0');
  const none = at(0), half = at(0.5), full = at(1);
  const raw = tangentDeviationAcross(base, 'u1', target, 'u0');
  assert.ok(Math.abs(none - raw) < 1e-9, `blend 0 must leave the surface untouched (${none.toFixed(4)} vs ${raw.toFixed(4)})`);
  assert.ok(half < none && full < half, `the break must fall as the blend rises (${none.toFixed(2)} -> ${half.toFixed(2)} -> ${full.toFixed(4)})`);
});

test('a reversed neighbour is matched without twisting', () => {
  const base = makeBase(), target = makeTarget();
  // Same target, its shared edge built in the opposite direction.
  const flipped = { ...target, ctrlNet: target.ctrlNet.map((row) => [...row].reverse()) };
  const orientation = edgeOrientation(base.ctrlNet, 'u1', flipped.ctrlNet, 'u0');
  assert.equal(orientation, 'reversed', 'the fixture really is reversed, or this test proves nothing');
  const res = matchEdge(base, 'u1', flipped, 'u0', { order: 1 });
  assert.equal(res.ok, true, res.reason);
  const after = tangentDeviationAcross(withNet(base, res.net), 'u1', flipped, 'u0');
  assert.ok(after < 0.05, `a reversed edge still matches, got ${after.toFixed(4)} degrees`);
});

test('G2 is REFUSED by name, not silently delivered as G1', () => {
  const base = makeBase(), target = makeTarget();
  const res = matchEdge(base, 'u1', target, 'u0', { order: 2 });
  assert.equal(res.ok, false);
  assert.equal(res.net, null);
  assert.match(res.reason, /curvature|not built/i);
});

test('mismatched control counts are refused, naming the shared knot vector', () => {
  const base = makeBase();
  const wide = makeTarget();
  wide.ctrlNet = wide.ctrlNet.map((r) => [...r, [r[3][0], r[3][1] + 10, r[3][2], 1]]);
  wide.knotsV = clampedKnots(5, 3);
  const res = matchEdge(base, 'u1', wide, 'u0', { order: 1 });
  assert.equal(res.ok, false);
  assert.match(res.reason, /shared knot vector|control counts/i);
});

// MERGE — the claim is that the merged surface IS both originals, not something
// close to them. So it is measured by evaluating the merged surface over each
// half and comparing against the patch that half came from.
test('⭐ merging two patches reproduces BOTH of them exactly', async () => {
  const { harmonizeDirections, seamDirectionFor } = await import('../kernel/surfaceknots.mjs');
  const { mergeAcrossSeam } = await import('../kernel/matchedge.mjs');
  const base = makeBase(), target = makeTarget();
  const h = harmonizeDirections(base, seamDirectionFor('u1'), target, seamDirectionFor('u0'));
  assert.equal(h.ok, true, h.reason);

  const merged = mergeAcrossSeam(h.a, 'u1', h.b, 'u0', { orientation: 'aligned' });
  assert.equal(merged.ok, true, merged.reason);
  const m = merged.srf;
  assert.equal(m.ctrlNet.length, h.a.ctrlNet.length + h.b.ctrlNet.length - 1,
    'the shared row appears once, not twice');

  const domOf = (knots, deg) => [knots[deg], knots[knots.length - 1 - deg]];
  const [mvMin, mvMax] = domOf(m.knotsV, m.degV);
  // The merged surface's u runs 0..1 over the base and 1..2 over the target.
  let worstBase = 0, worstTarget = 0;
  for (let i = 0; i <= 8; i++) {
    for (let j = 0; j <= 8; j++) {
      const s = i / 8, t = j / 8;
      const v = mvMin + (mvMax - mvMin) * t;
      const [buMin, buMax] = domOf(h.a.knotsU, h.a.degU);
      const [bvMin, bvMax] = domOf(h.a.knotsV, h.a.degV);
      const pb = surfacePoint(h.a, buMin + (buMax - buMin) * s, bvMin + (bvMax - bvMin) * t);
      const pm = surfacePoint(m, s, v);
      worstBase = Math.max(worstBase, Math.hypot(pm[0] - pb[0], pm[1] - pb[1], pm[2] - pb[2]));

      const [tuMin, tuMax] = domOf(h.b.knotsU, h.b.degU);
      const [tvMin, tvMax] = domOf(h.b.knotsV, h.b.degV);
      const pt = surfacePoint(h.b, tuMin + (tuMax - tuMin) * s, tvMin + (tvMax - tvMin) * t);
      const pm2 = surfacePoint(m, 1 + s, v);
      worstTarget = Math.max(worstTarget, Math.hypot(pm2[0] - pt[0], pm2[1] - pt[1], pm2[2] - pt[2]));
    }
  }
  assert.ok(worstBase < 1e-9, `the merged surface reproduces the BASE half exactly, worst ${worstBase}`);
  assert.ok(worstTarget < 1e-9, `and the TARGET half exactly, worst ${worstTarget}`);
});

test('merge refuses an unharmonised pair by name rather than approximating', async () => {
  const { mergeAcrossSeam } = await import('../kernel/matchedge.mjs');
  const base = makeBase();
  const wide = makeTarget();
  wide.ctrlNet = wide.ctrlNet.map((r) => [...r, [r[3][0], r[3][1] + 10, r[3][2], 1]]);
  const res = mergeAcrossSeam(base, 'u1', wide, 'u0', {});
  assert.equal(res.ok, false);
  assert.match(res.reason, /different control counts|harmonise/i);
});

// ⚠ A RATIONAL PAIR — the case every fixture above misses, because a hand-built
// net has w = 1 everywhere and the weight handling is then invisible. A quarter
// cylinder is genuinely rational, and matching two of them exercises the
// weights in G0's substitution and G1's redirection at once.
test('⭐ matching two RATIONAL surfaces keeps them on their own geometry', async () => {
  const { makeLine: mkLine, revolve: rev } = await import('../kernel/primitives.mjs');
  const { harmonizeDirections, seamDirectionFor } = await import('../kernel/surfaceknots.mjs');
  const quarter = (a0, a1) => rev(mkLine([25, 0, 0], [25, 0, 40]), [0, 0, 0], [0, 0, 1], a0, a1 - a0);
  const A = quarter(0, Math.PI / 2);
  const B = quarter(Math.PI / 2, Math.PI);
  const weights = A.ctrlNet.flat().map((cp) => cp[3]);
  assert.ok(weights.some((w) => Math.abs(w - 1) > 1e-6),
    'the fixture is genuinely RATIONAL — otherwise this test is the same as every one above');

  const h = harmonizeDirections(A, seamDirectionFor('v1'), B, seamDirectionFor('v0'));
  assert.equal(h.ok, true, h.reason);
  const res = matchEdge(h.a, 'v1', h.b, 'v0', { order: 1 });
  assert.equal(res.ok, true, res.reason);

  // The two quarters already meet tangentially, so a correct match must leave
  // the surface ON the cylinder of radius 25. A weight mistake pulls the
  // control points off it and the radius drifts.
  const matched = withNet(h.a, res.net);
  const domOf = (k, d) => [k[d], k[k.length - 1 - d]];
  const [uMin, uMax] = domOf(matched.knotsU, matched.degU);
  const [vMin, vMax] = domOf(matched.knotsV, matched.degV);
  let worst = 0;
  for (let i = 0; i <= 6; i++) {
    for (let j = 0; j <= 6; j++) {
      const p = surfacePoint(matched, uMin + ((uMax - uMin) * i) / 6, vMin + ((vMax - vMin) * j) / 6);
      worst = Math.max(worst, Math.abs(Math.hypot(p[0], p[1]) - 25));
    }
  }
  assert.ok(worst < 1e-6, `every point stays on the 25mm cylinder, worst off by ${worst}`);
});

// G2 — CURVATURE CONTINUITY. The claim is measured on the SURFACES, through
// their own curvature, not read off the control net that was just solved.
test('⭐ G2 closes the CURVATURE jump that G1 leaves behind', async () => {
  const { applyG2, curvatureDeviationAcross } = await import('../kernel/matchedge.mjs');
  const { surfaceDerivs2 } = await import('../kernel/curvature.mjs');
  // Deeper patches: G2 solves the THIRD row, so a 4-deep net is the minimum
  // that can carry one and still have shape left over.
  const mk = (x0, x1, kick) => {
    const rows = [];
    for (let i = 0; i < 5; i++) {
      const row = [];
      for (let j = 0; j < 4; j++) {
        const x = x0 + ((x1 - x0) * i) / 4, y = (30 * j) / 3;
        row.push([x, y, 0.002 * y * y + (i === 2 ? kick : 0) + 0.003 * x * x, 1]);
      }
      rows.push(row);
    }
    const cl = (n, d) => {
      const k = [];
      for (let q = 0; q <= d; q++) k.push(0);
      for (let q = 1; q <= n - d - 1; q++) k.push(q / (n - d));
      for (let q = 0; q <= d; q++) k.push(1);
      return k;
    };
    return { degU: 3, degV: 3, knotsU: cl(5, 3), knotsV: cl(4, 3), ctrlNet: rows };
  };
  const base = mk(-30, 0, 6), target = mk(0, 30, 0);

  const g1 = matchEdge(base, 'u1', target, 'u0', { order: 1 });
  assert.equal(g1.ok, true, g1.reason);
  const afterG1 = withNet(base, g1.net);
  const tangentG1 = tangentDeviationAcross(afterG1, 'u1', target, 'u0');
  assert.ok(tangentG1 < 0.05, `G1 closed the tangent break first (${tangentG1.toFixed(4)} degrees)`);

  const curvBefore = curvatureDeviationAcross(afterG1, 'u1', target, 'u0', surfaceDerivs2);
  assert.ok(curvBefore > 0.05,
    `⚠ the fixture must have a REAL curvature jump left after G1, or G2 has nothing to close — got ${curvBefore.toFixed(4)}`);

  const g2 = applyG2(base, 'u1', target, 'u0', g1.net, { blend: 1, derivs2: surfaceDerivs2 });
  assert.equal(g2.ok, true, g2.reason);
  const afterG2 = withNet(base, g2.net);
  const curvAfter = curvatureDeviationAcross(afterG2, 'u1', target, 'u0', surfaceDerivs2);
  assert.ok(curvAfter < curvBefore * 0.2,
    `⭐ G2 closes the curvature jump (${curvBefore.toFixed(4)} -> ${curvAfter.toFixed(4)})`);

  // AND IT MUST NOT UNDO G1. Solving the third row moves the surface, so the
  // tangent it was given has to survive the solve.
  const tangentG2 = tangentDeviationAcross(afterG2, 'u1', target, 'u0');
  assert.ok(tangentG2 < 0.05,
    `⭐ and the TANGENT continuity survives it (${tangentG2.toFixed(4)} degrees) — a G2 that broke G1 would be worse than none`);
});

test('G2 is refused by name on a surface too shallow to carry it', async () => {
  const { applyG2 } = await import('../kernel/matchedge.mjs');
  const base = makeBase(), target = makeTarget();
  const shallow = { ...base, ctrlNet: base.ctrlNet.slice(0, 2), knotsU: [0, 0, 1, 1], degU: 1 };
  const g1 = matchEdge(shallow, 'u1', target, 'u0', { order: 1 });
  assert.equal(g1.ok, true, g1.reason);
  const res = applyG2(shallow, 'u1', target, 'u0', g1.net, { derivs2: (await import('../kernel/curvature.mjs')).surfaceDerivs2 });
  assert.equal(res.ok, false);
  assert.match(res.reason, /third row|two control points/i);
});

// THE CURVATURE MEASURE AGAINST A PLANAR TARGET. A plane's normal curvature
// across the seam is exactly zero, and a successful G2 drives the base's to
// zero with it — so a purely relative ratio divides a vanishing residual by a
// vanishing scale and returns 1.0, the worst reading it can produce, precisely
// when the join is perfect. The measure has to stay meaningful as both sides
// approach flat, which means the scale needs a floor the MODEL can recognise
// rather than a numerical epsilon.
test('curvature deviation against a plane grows with the residual instead of pinning at 1', async () => {
  const { curvatureDeviationAcross } = await import('../kernel/matchedge.mjs');
  const { surfaceDerivs2 } = await import('../kernel/curvature.mjs');
  const patch = (dz, dx) => ({
    degU: 3, degV: 3, knotsU: clampedKnots(4, 3), knotsV: clampedKnots(4, 3),
    ctrlNet: [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => [i * 30 + dx, j * 30, i === 2 ? dz : 0, 1])),
  });
  const flat = patch(0, 0);
  const read = (dz) => curvatureDeviationAcross(patch(dz, 90), 'u0', flat, 'u1', surfaceDerivs2);

  // Coplanar is zero, and a match that landed within a micron of flat must not
  // read as a total break.
  assert.equal(read(0), 0);
  assert.ok(read(1e-7) < 1e-3, `a 1e-7 residual read ${read(1e-7)}`);

  // And it must still be a MEASURE: monotone in the residual, over the range
  // where the old relative form was saturated at exactly 1.
  const ladder = [1e-4, 1e-2, 1].map(read);
  for (let i = 1; i < ladder.length; i++) {
    assert.ok(ladder[i] > ladder[i - 1] * 5,
      `deviation must grow with the residual, got ${ladder.join(' -> ')}`);
  }
  assert.ok(ladder[2] > 0.05, `a 1mm break on a 90mm patch should read clearly, got ${ladder[2]}`);
});

// TOO SHALLOW IS NOT A REFUSAL ANY MORE. A surface two control points deep
// across the seam has no third row to solve for, and one that is DEGREE 1
// across has no second derivative to move — three rows would still carry
// exactly zero curvature, so depth alone is not enough. Both repairs are exact,
// which is the whole reason they can be applied without asking.
test('a shallow surface can be deepened exactly, and then reaches G2', async () => {
  const { applyG2 } = await import('../kernel/matchedge.mjs');
  const { surfaceElevateDegree, refineToCount, countIn, degreeIn } = await import('../kernel/surfaceknots.mjs');
  const { surfaceDerivs2 } = await import('../kernel/curvature.mjs');
  const base = makeBase(), target = makeTarget();

  // Two points deep across u, degree 1 there: a ruled strip, and the shape the
  // old refusal named.
  const shallow = { ...base, ctrlNet: base.ctrlNet.slice(0, 2), knotsU: [0, 0, 1, 1], degU: 1 };
  const before = matchEdge(shallow, 'u1', target, 'u0', { order: 1 });
  assert.equal(before.ok, true, before.reason);
  const refused = applyG2(shallow, 'u1', target, 'u0', before.net, { derivs2: surfaceDerivs2 });
  assert.equal(refused.ok, false, 'the shallow surface must still be beyond G2 on its own');

  // Deepen: degree first, then count — elevating raises the count too.
  let deep = surfaceElevateDegree(shallow, 'u', 2);
  if (countIn(deep, 'u') < 3) deep = refineToCount(deep, 'u', 3);
  assert.ok(degreeIn(deep, 'u') >= 2 && countIn(deep, 'u') >= 3, 'deepening must deliver both degree and rows');

  // EXACT — the surface is described differently and sits in the same place.
  for (const [u, v] of [[0, 0], [0.25, 0.4], [0.5, 0.5], [1, 0.75], [0.9, 1]]) {
    const a = surfacePoint(shallow, u, v), b = surfacePoint(deep, u, v);
    assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-9,
      `deepening moved the surface at (${u}, ${v})`);
  }

  const g1 = matchEdge(deep, 'u1', target, 'u0', { order: 1 });
  assert.equal(g1.ok, true, g1.reason);
  const g2 = applyG2(deep, 'u1', target, 'u0', g1.net, { derivs2: surfaceDerivs2 });
  assert.equal(g2.ok, true, `after deepening, G2 must go through — got ${g2.reason}`);
});

// NEVER NaN. A refined or force-harmonised seam can carry a zero-width knot
// span, and a derivative taken there is not a number. That value propagated to
// the Properties panel and printed "curvature NaN" beside a match the app was
// calling achieved — arithmetic wearing the clothes of a measurement.
test('the curvature measure returns null rather than NaN when it cannot evaluate', async () => {
  const { curvatureDeviationAcross } = await import('../kernel/matchedge.mjs');
  const base = makeBase(), target = makeTarget();

  // A well-formed pair still answers with a number.
  const ok = curvatureDeviationAcross(base, 'u1', target, 'u0', (await import('../kernel/curvature.mjs')).surfaceDerivs2);
  assert.ok(ok === null || Number.isFinite(ok), `a well-formed pair must give a number or null, got ${ok}`);

  // An evaluator that cannot produce a derivative stands in for the degenerate
  // span: the answer is the absence of a measurement, not a number-shaped one.
  const nanDerivs = () => ({ Su: [NaN, NaN, NaN], Sv: [NaN, NaN, NaN], Suu: [NaN, NaN, NaN], Svv: [NaN, NaN, NaN] });
  const out = curvatureDeviationAcross(base, 'u1', target, 'u0', nanDerivs);
  assert.equal(out, null, `an unevaluable seam must report null, got ${out}`);
  assert.ok(!Number.isNaN(out), 'and must never be NaN');
});
