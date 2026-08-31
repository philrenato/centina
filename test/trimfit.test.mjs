// FITTING A TRIMMED FACE'S BOUNDARY — the (u,v) polyline a trim loop is stored
// as, turned into the runs of real curves a B-rep trim loop is made of.
//
// The property under test throughout is that a REFUSAL IS A REAL OUTCOME: this
// module never returns a boundary that missed its bound, so every "ok" here is
// also a claim about accuracy, and the corner cases are asserted to refuse or
// to split rather than to quietly return something plausible.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fitTrimLoop, splitLoopAtCorners } from '../kernel/trimfit.mjs';
import { fitCurveToPoints } from '../kernel/fitcurve.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { curvePoint } from '../kernel/curve.mjs';

// A bilinear plane over [-10,10]^2, so (u,v) IS (x,y) and a trim loop's
// parameters are readable by eye.
const PLANE = {
  degU: 1, degV: 1,
  knotsU: [-10, -10, 10, 10], knotsV: [-10, -10, 10, 10],
  ctrlNet: [[[-10, -10, 0, 1], [-10, 10, 0, 1]], [[10, -10, 0, 1], [10, 10, 0, 1]]],
};

// A genuinely CURVED surface, so the pcurve/edge consistency measurement has
// something to measure: a bicubic saddle over the same domain. A flat plane
// would make the two agree trivially and prove nothing about the pairing.
const SADDLE = (() => {
  const ctrlNet = [];
  for (let i = 0; i < 4; i++) {
    const row = [];
    for (let j = 0; j < 4; j++) {
      const x = -10 + (20 * i) / 3, y = -10 + (20 * j) / 3;
      row.push([x, y, (x * x - y * y) / 12, 1]);
    }
    ctrlNet.push(row);
  }
  return {
    degU: 3, degV: 3,
    knotsU: [-10, -10, -10, -10, 10, 10, 10, 10],
    knotsV: [-10, -10, -10, -10, 10, 10, 10, 10],
    ctrlNet,
  };
})();

const ring = (r, n, cx = 0, cy = 0) => Array.from({ length: n }, (_, i) => {
  const t = (i / n) * Math.PI * 2;
  return [cx + r * Math.cos(t), cy + r * Math.sin(t)];
});

function square(h, perSide) {
  const corners = [[-h, -h], [h, -h], [h, h], [-h, h]];
  const out = [];
  for (let s = 0; s < 4; s++) {
    const a = corners[s], b = corners[(s + 1) % 4];
    for (let i = 0; i < perSide; i++) {
      const t = i / perSide;
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
  }
  return out;
}

// Two straight sides closed by two semicircular ends — tangent-continuous
// everywhere, so it has no corner despite obviously "having ends".
function stadium(r, len, perEnd) {
  const out = [];
  for (let i = 0; i <= perEnd; i++) {
    const t = -Math.PI / 2 + (Math.PI * i) / perEnd;
    out.push([len / 2 + r * Math.cos(t), r * Math.sin(t)]);
  }
  for (let i = 0; i <= perEnd; i++) {
    const t = Math.PI / 2 + (Math.PI * i) / perEnd;
    out.push([-len / 2 + r * Math.cos(t), r * Math.sin(t)]);
  }
  return out;
}

test('THE PREMISE: one smooth curve genuinely CANNOT fit a loop with corners', () => {
  // This is why splitLoopAtCorners exists at all, so it is asserted rather
  // than assumed. If this ever starts passing, the splitter is not load-bearing
  // any more and this whole module should be reconsidered — it would not be a
  // test to delete quietly.
  const loop = square(5, 10);
  const whole = fitCurveToPoints([...loop, loop[0]].map(([u, v]) => [u, v, 0]), {
    tolerance: 1e-3, closed: false, exactEndpoints: true,
  });
  assert.equal(whole.ok, false, 'a square must not fit as a single smooth curve');
  assert.ok(whole.bestDeviation > 0.1,
    `and it must miss by a LOT, not marginally — got ${whole.bestDeviation}`);
});

test('A SQUARE SPLITS INTO ITS FOUR SIDES, and every one is an exact line', () => {
  const res = fitTrimLoop(PLANE, square(5, 10), { tolerance: 1e-3 });
  assert.ok(res.ok, res.reason);
  assert.equal(res.runs.length, 4);
  assert.equal(res.stats.corners, 4);
  for (const run of res.runs) {
    assert.equal(run.kind, 'line', 'a straight side must be recognised, not splined');
    assert.equal(run.pcurve.ctrlPts.length, 2, 'a line has two control points');
  }
});

test('...INCLUDING THE SIDES THAT RUN BACKWARDS — the direction must not decide the answer', () => {
  // fitLine canonicalizes its direction so near-identical input cannot flicker
  // between opposite ones, which means a run travelling in -x comes back with
  // its endpoints swapped. Rejecting that would discard an exact answer over a
  // convention, and it silently did: two of these four sides fitted as
  // six-control-point splines. Both traversal directions are asserted so the
  // recognition cannot become orientation-dependent again.
  const forward = fitTrimLoop(PLANE, square(5, 10), { tolerance: 1e-3 });
  const reversed = fitTrimLoop(PLANE, square(5, 10).slice().reverse(), { tolerance: 1e-3 });
  assert.ok(forward.ok && reversed.ok);
  assert.equal(forward.stats.exactRuns, 4);
  assert.equal(reversed.stats.exactRuns, 4, 'reversing the loop must not cost a single exact line');
});

test('A CIRCULAR BOUNDARY IS ONE RUN, and far fewer control points than samples', () => {
  const loop = ring(5, 64);
  const res = fitTrimLoop(PLANE, loop, { tolerance: 1e-3 });
  assert.ok(res.ok, res.reason);
  assert.equal(res.runs.length, 1, 'a smooth closed loop has no corner and so no vertex');
  assert.ok(res.stats.controlPoints < loop.length / 3,
    `${res.stats.controlPoints} control points for ${loop.length} samples is not a description`);
  assert.ok(res.stats.worstPcurveDeviation <= 1e-3);
});

test('A TANGENT-CONTINUOUS JOIN IS NOT A CORNER — a stadium stays one run', () => {
  // The straight-to-arc joins here are smooth, so there is no vertex for a
  // B-rep to have and inventing one would be a topology claim this module has
  // no basis for.
  const res = fitTrimLoop(PLANE, stadium(4, 12, 24), { tolerance: 1e-3 });
  assert.ok(res.ok, res.reason);
  assert.equal(res.stats.corners, 0);
  assert.equal(res.runs.length, 1);
});

test('ADJACENT RUNS MEET EXACTLY — the loop closes, it does not nearly close', () => {
  // A gap at a shared corner is a naked edge, so "within tolerance" is not the
  // bar here; the runs share their meeting point and the fitted curves
  // interpolate their own endpoints, so the join must be EXACT.
  const res = fitTrimLoop(PLANE, square(5, 7), { tolerance: 1e-3 });
  assert.ok(res.ok, res.reason);
  for (let i = 0; i < res.runs.length; i++) {
    const a = res.runs[i].pcurve.ctrlPts;
    const b = res.runs[(i + 1) % res.runs.length].pcurve.ctrlPts;
    const end = a[a.length - 1], start = b[0];
    assert.equal(end[0], start[0], `run ${i} ends where run ${i + 1} starts (u)`);
    assert.equal(end[1], start[1], `run ${i} ends where run ${i + 1} starts (v)`);
  }
});

test('ON A CURVED SURFACE THE PCURVE AND THE EDGE DISAGREE, and the disagreement is measured', () => {
  // The pair cannot agree exactly — the surface image of a NURBS curve is not
  // a NURBS curve — so the honest deliverable is the number, not a claim of
  // equality. What must hold is that it is REAL (a curved surface makes it
  // nonzero) and that the tolerance handed on is never tighter than it.
  const res = fitTrimLoop(SADDLE, ring(5, 48), { tolerance: 1e-3 });
  assert.ok(res.ok, res.reason);
  assert.ok(res.stats.worstConsistency > 0, 'a curved surface must produce a real disagreement');
  for (const run of res.runs) {
    assert.ok(run.tolerance >= run.consistency,
      `an edge tolerance (${run.tolerance}) must not claim to be tighter than the measured disagreement (${run.consistency})`);
    assert.ok(run.tolerance >= 1e-3, 'nor tighter than the bound the curves were fitted to');
  }
});

test('THE FITTED PCURVE REALLY DOES TRACE THE BOUNDARY ON THE SURFACE', () => {
  // The end-to-end property a trimmed export depends on, asserted in 3-D
  // rather than in parameters: walk the fitted pcurve, push each point through
  // the surface, and it must land on the boundary the loop described.
  const loop = ring(5, 48);
  const res = fitTrimLoop(SADDLE, loop, { tolerance: 1e-3 });
  assert.ok(res.ok, res.reason);
  const truth = loop.map(([u, v]) => surfacePoint(SADDLE, u, v));
  const pcurve = res.runs[0].pcurve;
  const U = pcurve.knots, p = pcurve.degree;
  const t0 = U[p], t1 = U[U.length - 1 - p];
  let worst = 0;
  for (const q of truth) {
    let best = Infinity;
    for (let i = 0; i <= 400; i++) {
      const [u, v] = curvePoint(pcurve, t0 + (t1 - t0) * (i / 400));
      const s = surfacePoint(SADDLE, u, v);
      best = Math.min(best, Math.hypot(s[0] - q[0], s[1] - q[1], s[2] - q[2]));
    }
    worst = Math.max(worst, best);
  }
  assert.ok(worst < 0.05, `the imaged pcurve strayed ${worst} from the boundary it describes`);
});

test('A DEGENERATE LOOP REFUSES BY NAME rather than returning something plausible', () => {
  assert.equal(fitTrimLoop(PLANE, [[0, 0], [1, 1]], { tolerance: 1e-3 }).ok, false);
  assert.equal(fitTrimLoop(null, ring(5, 16), { tolerance: 1e-3 }).ok, false);
  const badTol = fitTrimLoop(PLANE, ring(5, 16), { tolerance: 0 });
  assert.equal(badTol.ok, false);
  assert.match(badTol.reason, /positive tolerance/);
});

test('splitLoopAtCorners: a single corner is not a split, but it does set the seam', () => {
  // One corner cannot divide a loop into two runs, and pretending otherwise
  // would emit a second trim with no vertex to hang it on. It IS the right
  // place to start the single run, which is where a closed run wants its seam.
  const loop = [...ring(5, 24)];
  loop.splice(6, 0, [0, 0]); // a spike: one unambiguous corner
  const { runs, cornerIndices } = splitLoopAtCorners(loop);
  assert.ok(cornerIndices.length >= 1);
  if (cornerIndices.length < 2) {
    assert.equal(runs.length, 1);
    assert.deepEqual(runs[0][0], runs[0][runs[0].length - 1], 'the single run must close');
  }
});
