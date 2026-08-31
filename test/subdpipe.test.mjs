import test from 'node:test';
import assert from 'node:assert/strict';
import { subdPipeCage } from '../kernel/subdpipe.mjs';
import { buildTopology, subdivideCatmullClark } from '../kernel/subd.mjs';
import { makeLine } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { curvePoint } from '../kernel/curve.mjs';
import { joinCurvesC0 } from '../kernel/knots.mjs';

// A deliberately CURVED rail — a straight line would let a whole class of
// frame/transport bug pass unnoticed, and this project's own record says
// so more than once.
function curvedRail() {
  return globalCurveInterp([[0, 0, 0], [40, 0, 20], [80, 0, 0], [120, 0, 20]], 3);
}

// Every edge of a closed cage is shared by exactly 2 faces; every edge of
// an open one by 1 or 2. Recomputed from raw {vertices, faces} rather
// than asked of buildTopology, so the check cannot agree with the code
// under test about a shared mistake.
function edgeFaceCounts(cage) {
  const counts = new Map();
  for (const f of cage.faces) {
    for (let i = 0; i < f.length; i++) {
      const a = f[i], b = f[(i + 1) % f.length];
      const k = a < b ? `${a}_${b}` : `${b}_${a}`;
      counts.set(k, (counts.get(k) || 0) + 1);
    }
  }
  return counts;
}

// No DIRECTED edge traversed twice — the real orientability property, the
// same one subdedit's own bridge winding search verifies rather than
// derives.
function directedEdgeReuse(cage) {
  const seen = new Set();
  let reused = 0;
  for (const f of cage.faces) {
    for (let i = 0; i < f.length; i++) {
      const k = `${f[i]}>${f[(i + 1) % f.length]}`;
      if (seen.has(k)) reused++;
      seen.add(k);
    }
  }
  return reused;
}

test('subdPipeCage: an open tube has the exact ring/quad counts its density asks for', () => {
  const rail = makeLine([0, 0, 0], [0, 0, 60]);
  const facets = 8, segments = 5;
  const cage = subdPipeCage(rail, { radius: 5, facets, segments });
  assert.equal(cage.vertices.length, facets * (segments + 1));
  assert.equal(cage.faces.length, facets * segments); // no caps: side quads only
  assert.ok(cage.faces.every((f) => f.length === 4));
  assert.equal(cage.startRim.length, facets);
  assert.equal(cage.endRim.length, facets);
});

test('subdPipeCage: every ring vertex sits exactly `radius` from the rail, on a genuinely curved rail', () => {
  const rail = curvedRail();
  const cage = subdPipeCage(rail, { radius: 7, facets: 10, segments: 6 });
  // A parallel-transport frame's origin IS a point of the rail, so the
  // distance from a ring vertex to its own station is the tube radius
  // exactly — checked against the rail's own evaluated point, not against
  // the frame the cage was built from.
  const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];
  const SCAN = 20000;
  const scanRes = (uMax - uMin) / SCAN;
  let worstRadius = 0, worstOnRail = 0;
  for (let s = 0; s <= 6; s++) {
    const ring = cage.vertices.slice(s * 10, s * 10 + 10);
    // A regular polygon's centroid IS its circumcentre, so the ring's own
    // centroid is exactly the frame origin — measure the radius against
    // that, which is exact, rather than against a brute-force nearest
    // rail sample, whose error is just the scan's own resolution.
    const c = ring.reduce((a, p) => [a[0] + p[0] / 10, a[1] + p[1] / 10, a[2] + p[2] / 10], [0, 0, 0]);
    for (const v of ring) {
      worstRadius = Math.max(worstRadius, Math.abs(Math.hypot(v[0] - c[0], v[1] - c[1], v[2] - c[2]) - 7));
    }
    // Separately: that centre genuinely lies ON the rail. Tolerance is
    // the scan's own step, not a claim about the geometry.
    let best = Infinity;
    for (let i = 0; i <= SCAN; i++) {
      const p = curvePoint(rail, uMin + (uMax - uMin) * (i / SCAN));
      best = Math.min(best, Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]));
    }
    worstOnRail = Math.max(worstOnRail, best);
  }
  assert.ok(worstRadius < 1e-9, `worst radius deviation ${worstRadius}`);
  assert.ok(worstOnRail < scanRes * 60, `a ring centre drifted off the rail by ${worstOnRail}`);
});

test('subdPipeCage: stations are ARC-LENGTH even, not parameter even', () => {
  // A rail whose parametrization is deliberately uneven: interpolated
  // through points with very different spacing, so parameter-even and
  // arc-length-even genuinely disagree.
  const rail = globalCurveInterp([[0, 0, 0], [5, 0, 0], [10, 0, 0], [120, 0, 0]], 3);
  const segments = 6;
  const cage = subdPipeCage(rail, { radius: 2, facets: 4, segments });
  const centres = [];
  for (let s = 0; s <= segments; s++) {
    const ring = cage.vertices.slice(s * 4, s * 4 + 4);
    centres.push(ring.reduce((a, p) => [a[0] + p[0] / 4, a[1] + p[1] / 4, a[2] + p[2] / 4], [0, 0, 0]));
  }
  const gaps = [];
  for (let i = 0; i < centres.length - 1; i++) {
    gaps.push(Math.hypot(centres[i + 1][0] - centres[i][0], centres[i + 1][1] - centres[i][1], centres[i + 1][2] - centres[i][2]));
  }
  const min = Math.min(...gaps), max = Math.max(...gaps);
  // Chord vs arc means these are not identical, but they must be CLOSE.
  // Parameter-even stationing on this rail is wildly uneven, so this is a
  // genuinely discriminating bound, not a loose one.
  assert.ok(max / min < 1.15, `stations are not arc-length even: gaps ${gaps.map((g) => g.toFixed(2)).join(', ')}`);
});

test('subdPipeCage: flat caps close the tube into a valid, consistently wound 2-manifold', () => {
  const rail = curvedRail();
  const cage = subdPipeCage(rail, { radius: 6, facets: 8, segments: 4, capStart: 'flat', capEnd: 'flat' });
  const counts = edgeFaceCounts(cage);
  assert.ok([...counts.values()].every((c) => c === 2), 'a flat-capped tube should have no boundary edge');
  assert.equal(directedEdgeReuse(cage), 0, 'inconsistent winding: a directed edge was traversed twice');
  const topo = buildTopology(cage);
  const edges = [...topo.edgeMap.values()];
  assert.equal(edges.filter((e) => e.faces.length !== 2).length, 0);
  // Euler characteristic of a closed genus-0 surface is 2 — a real check
  // that the caps closed the tube rather than merely adding two faces.
  assert.equal(cage.vertices.length - edges.length + cage.faces.length, 2);
});

test('subdPipeCage: an uncapped tube has exactly its two rims as boundary', () => {
  const rail = makeLine([0, 0, 0], [0, 0, 40]);
  const facets = 9;
  const cage = subdPipeCage(rail, { radius: 4, facets, segments: 3 });
  const counts = edgeFaceCounts(cage);
  const naked = [...counts.values()].filter((c) => c === 1).length;
  assert.equal(naked, facets * 2, 'an open tube should be bounded by exactly its two rims');
  assert.equal(directedEdgeReuse(cage), 0);
});

test('subdPipeCage: a round cap is a real dome — it grows PAST the rail end and reports no rim', () => {
  const rail = makeLine([0, 0, 0], [0, 0, 50]);
  const radius = 6;
  const cage = subdPipeCage(rail, { radius, facets: 12, segments: 3, capStart: 'round', capEnd: 'round' });
  const zs = cage.vertices.map((v) => v[2]);
  // The apexes sit exactly one radius beyond each rail end. A flat cap
  // would leave the extremes at 0 and 50 — this is what distinguishes a
  // dome from a lid.
  assert.ok(Math.abs(Math.min(...zs) - -radius) < 1e-9, `start apex at ${Math.min(...zs)}, expected ${-radius}`);
  assert.ok(Math.abs(Math.max(...zs) - (50 + radius)) < 1e-9, `end apex at ${Math.max(...zs)}, expected ${50 + radius}`);
  // No open rim is reported for a domed end: that ring is interior now,
  // and handing it to a bridge would tear the surface.
  assert.deepEqual(cage.startRim, []);
  assert.deepEqual(cage.endRim, []);
  const counts = edgeFaceCounts(cage);
  assert.ok([...counts.values()].every((c) => c === 2), 'a round-capped tube should be closed');
  assert.equal(directedEdgeReuse(cage), 0);
});

test('subdPipeCage: crease 0 writes NO key at all, so soft is provably the same cage as no-crease-feature', () => {
  const rail = curvedRail();
  const a = subdPipeCage(rail, { radius: 5, facets: 8, segments: 3, capStart: 'flat', crease: 0 });
  assert.deepEqual(a.creases, {});
  const b = subdPipeCage(rail, { radius: 5, facets: 8, segments: 3, capStart: 'flat', crease: 1 });
  assert.equal(Object.keys(b.creases).length, 8, 'a creased flat rim should carry exactly one key per rim edge');
  assert.ok(Object.values(b.creases).every((w) => w > 0));
  // Positions are identical either way — a crease changes the LIMIT
  // surface, never the cage.
  assert.deepEqual(a.vertices, b.vertices);
  assert.deepEqual(a.faces, b.faces);
});

test('subdPipeCage: a creased rim genuinely holds its edge under real subdivision', () => {
  const rail = makeLine([0, 0, 0], [0, 0, 40]);
  const soft = subdivideCatmullClark(subdPipeCage(rail, { radius: 5, facets: 8, segments: 3, capStart: 'flat', capEnd: 'flat', crease: 0 }));
  const crisp = subdivideCatmullClark(subdPipeCage(rail, { radius: 5, facets: 8, segments: 3, capStart: 'flat', capEnd: 'flat', crease: 1 }));
  // A soft rim rounds off, pulling the end ring INWARD toward the axis;
  // a creased one holds it out at the cage radius. Measured as the widest
  // real radius at the very end of the tube — a shape difference, not a
  // stored-weight one. (The Z extent is the wrong measure: a flat cap's
  // own face point stays at Z=0 either way, so both read identically.)
  const endRadius = (c) => Math.max(...c.vertices.filter((v) => v[2] < 0.5).map((v) => Math.hypot(v[0], v[1])));
  const softR = endRadius(soft), crispR = endRadius(crisp);
  assert.ok(crispR > softR + 1e-6, `a creased rim should hold its end ring out wider (soft ${softR}, crisp ${crispR})`);
  assert.ok(crispR <= 5 + 1e-9, 'and never wider than the cage itself');
});

test('subdPipeCage: density genuinely changes the limit surface, not just the drawing', () => {
  const rail = makeLine([0, 0, 0], [0, 0, 40]);
  const radius = 10;
  // A cage ring is a POLYGON inscribed in the tube, so its limit circle
  // sits inside the true radius — and gets closer as facets rise. That is
  // the difference between a density dial on a cage and one on a
  // tessellation: here it moves the actual surface.
  const measure = (facets) => {
    const c = subdivideCatmullClark(subdivideCatmullClark(subdPipeCage(rail, { radius, facets, segments: 4 })));
    const mid = c.vertices.filter((v) => Math.abs(v[2] - 20) < 3);
    return Math.max(...mid.map((v) => Math.hypot(v[0], v[1])));
  };
  const coarse = measure(5), fine = measure(24);
  assert.ok(fine > coarse, `more facets should approach the true radius (5 -> ${coarse}, 24 -> ${fine})`);
  assert.ok(fine < radius + 1e-9, 'a Catmull-Clark limit ring never exceeds its own cage polygon');
  // Two subdivision levels of a 24-gon is close, not exact — a limit ring
  // is inscribed in its cage polygon and approaches it from inside. The
  // bound is set from what convergence actually delivers here, not from
  // an assumption about it.
  assert.ok(radius - fine < 0.15, `a dense cage should be close to the true radius (off by ${radius - fine})`);
  assert.ok(radius - coarse > radius - fine, 'and a coarse cage should be measurably further off');
});

test('subdPipeCage: the cage survives real Catmull-Clark subdivision with no NaN', () => {
  const rail = curvedRail();
  for (const caps of [['none', 'none'], ['flat', 'flat'], ['round', 'round'], ['flat', 'round']]) {
    let cage = subdPipeCage(rail, { radius: 5, facets: 7, segments: 5, capStart: caps[0], capEnd: caps[1], crease: 0.5 });
    for (let i = 0; i < 2; i++) cage = subdivideCatmullClark(cage);
    assert.ok(cage.vertices.every((v) => v.every(Number.isFinite)), `NaN after subdividing caps ${caps.join('/')}`);
    assert.equal(directedEdgeReuse(cage), 0, `winding broke after subdividing caps ${caps.join('/')}`);
  }
});

test('subdPipeCage: refuses honestly rather than building a degenerate cage', () => {
  const rail = makeLine([0, 0, 0], [0, 0, 40]);
  assert.throws(() => subdPipeCage(rail, { radius: 0 }), /radius must be positive/);
  assert.throws(() => subdPipeCage(null, {}), /needs a real rail/);
  // Below-minimum density is CLAMPED, not refused — a cage needs at least
  // 3 facets and 1 segment to exist at all, and clamping matches this
  // app's own clamp-not-refuse precedent for a value with a real floor.
  const c = subdPipeCage(rail, { radius: 3, facets: 1, segments: 0 });
  assert.equal(c.vertices.length, 3 * 2);
});

// A rail that turns exactly into the direction the transport picked as its
// own seed normal. `anyPerpendicular([1,0,0])` is [0,0,1], so a rail running
// along X and then straight up Z hands the station just past the corner a
// prevNormal exactly parallel to its new tangent — and a degree<=1 rail's
// tangent changes DISCONTINUOUSLY there, so no amount of station density
// smooths it out. This threw outright before buildParallelTransportFrames'
// own projection guard was applied at every station rather than only at a
// control-point corner. A pipe running along the floor and then up a wall is
// an ordinary thing to draw, so this is a real reachable case, not a
// contrived one.
test('subdPipeCage: a rail turning into its own seed normal still builds', () => {
  const rail = joinCurvesC0([makeLine([0, 0, 0], [60, 0, 0]), makeLine([60, 0, 0], [60, 0, 40])]);
  for (const segments of [1, 2, 3, 5, 8]) {
    const cage = subdPipeCage(rail, { radius: 5, facets: 8, segments });
    assert.equal(cage.vertices.length, 8 * (segments + 1), `segments=${segments}`);
    assert.ok(cage.vertices.every((v) => v.every(Number.isFinite)), `segments=${segments}: finite`);
    assert.equal(directedEdgeReuse(cage), 0, `segments=${segments}: consistently wound`);
  }
  // and the same turn in the other two axis pairs, so this is not one lucky
  // orientation
  for (const [a, b] of [[[0, 60, 0], [0, 60, 40]], [[0, 0, 60], [40, 0, 60]]]) {
    const r = joinCurvesC0([makeLine([0, 0, 0], a), makeLine(a, b)]);
    const cage = subdPipeCage(r, { radius: 5, facets: 8, segments: 4 });
    assert.ok(cage.vertices.every((v) => v.every(Number.isFinite)));
    assert.equal(directedEdgeReuse(cage), 0);
  }
});
