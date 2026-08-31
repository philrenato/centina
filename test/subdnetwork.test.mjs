import test from 'node:test';
import assert from 'node:assert/strict';
import { detectPipeJunctions, subdPipeNetwork } from '../kernel/subdnetwork.mjs';
import { subdPipeCage } from '../kernel/subdpipe.mjs';
import { subdivideCatmullClark } from '../kernel/subd.mjs';
import { makeLine, makeCircle } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { curvePoint } from '../kernel/curve.mjs';

// ── INVARIANTS, recomputed from raw {vertices, faces}. Never asked of
// buildTopology: a check that shares the code under test's own idea of what
// an edge is cannot catch the two of them being wrong together.
function undirectedEdgeCounts(cage) {
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
function facesWithRepeatedVertex(cage) {
  return cage.faces.filter((f) => new Set(f).size !== f.length).length;
}
function orphanVertices(cage) {
  const used = new Set();
  for (const f of cage.faces) for (const v of f) used.add(v);
  let n = 0;
  for (let i = 0; i < cage.vertices.length; i++) if (!used.has(i)) n++;
  return n;
}
function eulerCharacteristic(cage) {
  return cage.vertices.length - undirectedEdgeCounts(cage).size + cage.faces.length;
}
function nakedEdgeCount(cage) {
  let n = 0;
  for (const c of undirectedEdgeCounts(cage).values()) if (c === 1) n++;
  return n;
}
function nonManifoldEdgeCount(cage) {
  let n = 0;
  for (const c of undirectedEdgeCounts(cage).values()) if (c > 2) n++;
  return n;
}
// The full structural gate every cage in this file passes, plus one real
// Catmull-Clark pass — a cage can satisfy every count above and still be
// one the subdivider destroys.
function assertWellFormed(cage, label) {
  assert.ok(cage.vertices.every((v) => v.length === 3 && v.every(Number.isFinite)), `${label}: every vertex finite`);
  assert.ok(cage.faces.every((f) => f.every((i) => Number.isInteger(i) && i >= 0 && i < cage.vertices.length)), `${label}: every face index in range`);
  assert.equal(facesWithRepeatedVertex(cage), 0, `${label}: no face repeats a vertex`);
  assert.equal(directedEdgeReuse(cage), 0, `${label}: no directed edge traversed twice`);
  assert.equal(nonManifoldEdgeCount(cage), 0, `${label}: no edge shared by 3+ faces`);
  assert.equal(orphanVertices(cage), 0, `${label}: no orphan vertices`);
  const refined = subdivideCatmullClark(cage);
  assert.ok(refined.vertices.every((v) => v.every(Number.isFinite)), `${label}: subdivides without NaN`);
  assert.equal(directedEdgeReuse(refined), 0, `${label}: still consistently wound after subdivision`);
  assert.equal(nonManifoldEdgeCount(refined), 0, `${label}: still manifold after subdivision`);
  return refined;
}

// ── FIXTURES. Rails are real curves throughout — makeLine builds a genuine
// clamped degree-1 NurbsCrv, and at least one fixture below is a real
// degree-2 interpolation so a whole class of parametrization bug cannot
// pass unnoticed on straight lines alone.
const R = 5;
function yJunction(len = 60) {
  const A = (2 * Math.PI) / 3;
  return [0, 1, 2].map((k) => makeLine([0, 0, 0], [len * Math.cos(k * A), len * Math.sin(k * A), 0]));
}
function crossJunction(len = 60) {
  return [[1, 0], [0, 1], [-1, 0], [0, -1]].map(([x, y]) => makeLine([0, 0, 0], [len * x, len * y, 0]));
}
// A branch whose end lands NEAR (0.8mm off — well inside a 2.5mm weld
// tolerance, well outside JOIN_TOLERANCE) the host's own interior. This is
// the case a student can actually produce: this app has endpoint snapping
// but no on-curve snap, so a T is never exact.
function tJunction() {
  return [makeLine([-60, 0, 0], [60, 0, 0]), makeLine([5, 0.8, 0], [5, 60, 0])];
}
function awkwardJunction() {
  return [
    makeLine([0, 0, 0], [70, 10, 5]),
    makeLine([0, 0, 0], [-30, 55, -12]),
    globalCurveInterp([[0, 0, 0], [-10, -20, 25], [-40, -35, 60]], 2),
  ];
}

// ─────────────────────────────────────────────────────────────────────────
// DETECTION

test('detectPipeJunctions: three rails sharing an endpoint are one 3-way junction in one component', () => {
  const rails = yJunction();
  const det = detectPipeJunctions(rails, { radius: R });
  assert.equal(det.ok, true);
  assert.equal(det.rails.length, 3, 'nothing was split');
  assert.equal(det.splits.length, 0);
  assert.equal(det.junctions.length, 1);
  assert.equal(det.junctions[0].arity, 3);
  assert.equal(det.components.length, 1);
  assert.deepEqual(det.components[0].sourceRails, [0, 1, 2]);
  // D4: the tolerance is radius-relative, not JOIN_TOLERANCE.
  assert.equal(det.tolerance, R * 0.5);
});

test('detectPipeJunctions: a T SPLITS the host, and the split lands at the real projection', () => {
  const [host, branch] = tJunction();
  const det = detectPipeJunctions([host, branch], { radius: R });
  assert.equal(det.ok, true);
  // The host genuinely became two rails; the branch is untouched.
  assert.equal(det.rails.length, 3);
  assert.equal(det.splits.length, 1);
  assert.equal(det.splits[0].rail, 0);
  assert.equal(det.splits[0].params.length, 1);
  assert.deepEqual(det.railSources, [0, 0, 1]);
  // The touch point is the branch endpoint's own foot on the host: x = 5.
  const cut = curvePoint(host, det.splits[0].params[0]);
  assert.ok(Math.abs(cut[0] - 5) < 1e-6, `split at x=${cut[0]}, expected 5`);
  // And the branch point is now a genuine 3-way meeting, not a 2-way plus a
  // dangling end.
  assert.equal(det.junctions.length, 1);
  assert.equal(det.junctions[0].arity, 3);
  assert.equal(det.components.length, 1);
  // The two host pieces really do cover the original, end to end.
  const ends = det.rails.slice(0, 2).map((c) => [curvePoint(c, c.knots[0]), curvePoint(c, c.knots[c.knots.length - 1])]);
  assert.ok(Math.abs(ends[0][0][0] - -60) < 1e-9);
  assert.ok(Math.abs(ends[0][1][0] - 5) < 1e-6);
  assert.ok(Math.abs(ends[1][0][0] - 5) < 1e-6);
  assert.ok(Math.abs(ends[1][1][0] - 60) < 1e-9);
});

test('detectPipeJunctions: D1 — splitting never touches the caller\'s own rails', () => {
  const rails = tJunction();
  const before = JSON.stringify(rails);
  const det = detectPipeJunctions(rails, { radius: R });
  assert.equal(JSON.stringify(rails), before, 'input rails byte-identical after detection');
  // and the returned rails are genuinely different objects, not aliases
  assert.ok(det.rails.every((c) => c !== rails[0] && c !== rails[1]));
  subdPipeNetwork(rails, { radius: R, facets: 8, segments: 3 });
  assert.equal(JSON.stringify(rails), before, 'input rails byte-identical after assembly too');
});

test('detectPipeJunctions: two runs that never meet are two components', () => {
  const rails = [
    makeLine([0, 0, 0], [50, 0, 0]),
    makeLine([50, 0, 0], [50, 50, 0]),
    makeLine([200, 0, 0], [250, 0, 0]),
  ];
  const det = detectPipeJunctions(rails, { radius: R });
  assert.equal(det.components.length, 2);
  assert.deepEqual(det.components[0].sourceRails, [0, 1]);
  assert.deepEqual(det.components[1].sourceRails, [2]);
  assert.equal(det.junctions.length, 1);
  assert.equal(det.junctions[0].arity, 2);
});

// ─────────────────────────────────────────────────────────────────────────
// ASSEMBLY — structure

test('subdPipeNetwork: a 3-way endpoint junction welds into ONE cage of the right shape', () => {
  const net = subdPipeNetwork(yJunction(), { radius: R, facets: 8, segments: 3 });
  assert.equal(net.ok, true);
  assert.equal(net.cages.length, 1);
  assert.deepEqual(net.junctionCounts, { 3: 1 });
  assert.equal(net.junctionTotal, 1);
  const cage = net.cages[0];
  assertWellFormed(cage, 'Y junction');
  // A sphere with 3 holes: chi = 2 - 3. Three open rims of `facets` edges
  // each, and nothing else naked.
  assert.equal(eulerCharacteristic(cage), -1);
  assert.equal(nakedEdgeCount(cage), 3 * 8);
  assert.equal(cage.freeEndCount, 3);
  assert.deepEqual(cage.sourceRails, [0, 1, 2]);
  assert.equal(cage.junctions.length, 1);
  assert.equal(cage.junctions[0].arity, 3);
});

test('subdPipeNetwork: a T-junction welds into one cage, with the host genuinely split into two arms', () => {
  const net = subdPipeNetwork(tJunction(), { radius: R, facets: 8, segments: 3 });
  assert.equal(net.ok, true);
  assert.equal(net.cages.length, 1);
  assert.equal(net.splitCount, 1);
  assert.deepEqual(net.junctionCounts, { 3: 1 });
  const cage = net.cages[0];
  assert.equal(cage.railCount, 3, 'two host pieces plus the branch');
  assertWellFormed(cage, 'T junction');
  assert.equal(eulerCharacteristic(cage), -1);
  assert.equal(nakedEdgeCount(cage), 3 * 8);
});

test('subdPipeNetwork: a 4-way coplanar cross welds into one cage with chi = 2 - 4', () => {
  const net = subdPipeNetwork(crossJunction(), { radius: R, facets: 8, segments: 3 });
  assert.equal(net.ok, true);
  assert.deepEqual(net.junctionCounts, { 4: 1 });
  const cage = net.cages[0];
  assertWellFormed(cage, '4-way cross');
  assert.equal(eulerCharacteristic(cage), -2);
  assert.equal(nakedEdgeCount(cage), 4 * 8);
  assert.equal(cage.junctions[0].arity, 4);
});

test('subdPipeNetwork: a deliberately awkward junction — uneven angles, out-of-plane tilts, different lengths, one genuinely curved rail', () => {
  const net = subdPipeNetwork(awkwardJunction(), { radius: R, facets: 10, segments: 4 });
  assert.equal(net.ok, true);
  assert.equal(net.cages.length, 1);
  const cage = net.cages[0];
  assertWellFormed(cage, 'awkward junction');
  assert.equal(eulerCharacteristic(cage), -1);
  assert.equal(nakedEdgeCount(cage), 3 * 10);
});

test('subdPipeNetwork: capping every free end gives a genuinely CLOSED solid', () => {
  for (const cap of ['flat', 'round']) {
    const net = subdPipeNetwork(yJunction(), { radius: R, facets: 8, segments: 3, capStart: cap, capEnd: cap });
    assert.equal(net.ok, true, `${cap} cap`);
    const cage = net.cages[0];
    assertWellFormed(cage, `Y junction, ${cap} caps`);
    assert.equal(nakedEdgeCount(cage), 0, `${cap}: no naked edges left`);
    assert.equal(eulerCharacteristic(cage), 2, `${cap}: closed genus-0`);
  }
});

test('subdPipeNetwork: a junction-facing end is built with cap NONE whatever the network cap style is', () => {
  // A 'round' cap returns an EMPTY rim (correctly — the ring under a dome is
  // interior), so a junction end that honoured the network's own style would
  // have nothing to weld. This is the exact case that proves the per-END
  // override is real: the SAME network builds and welds with round caps,
  // and the round caps genuinely landed on the three FREE ends.
  const net = subdPipeNetwork(yJunction(), { radius: R, facets: 8, segments: 3, capEnd: 'round', capStart: 'round' });
  assert.equal(net.ok, true);
  const cage = net.cages[0];
  // Three dome apexes: a valence-8 vertex belonging only to triangles.
  const tri = cage.faces.filter((f) => f.length === 3);
  const apexes = new Set();
  for (let v = 0; v < cage.vertices.length; v++) {
    const inTri = tri.filter((f) => f.includes(v)).length;
    const inAny = cage.faces.filter((f) => f.includes(v)).length;
    if (inAny === 8 && inTri === 8) apexes.add(v);
  }
  assert.equal(apexes.size, 3, 'exactly one dome apex per free end');
});

// ─────────────────────────────────────────────────────────────────────────
// ORDER INDEPENDENCE — a symmetric junction is where an exact tie lives.

test('subdPipeNetwork: a SYMMETRIC junction is order-independent — every input permutation gives the same cage', () => {
  const rails = yJunction();
  const sig = (n) => {
    const c = n.cages[0];
    const verts = c.vertices.map((v) => v.map((x) => Number(x.toFixed(9))));
    verts.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    const sizes = c.faces.map((f) => f.length).sort((a, b) => a - b);
    return JSON.stringify({ verts, sizes, chi: eulerCharacteristic(c), naked: nakedEdgeCount(c) });
  };
  const base = sig(subdPipeNetwork(rails, { radius: R, facets: 8, segments: 3 }));
  for (const p of [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]) {
    const net = subdPipeNetwork(p.map((i) => rails[i]), { radius: R, facets: 8, segments: 3 });
    assert.equal(net.ok, true);
    assert.equal(sig(net), base, `permutation ${p.join('')} matches`);
  }
});

test('subdPipeNetwork: a symmetric 4-way cross is order-independent too', () => {
  const rails = crossJunction();
  const sig = (n) => {
    const c = n.cages[0];
    const verts = c.vertices.map((v) => v.map((x) => Number(x.toFixed(9))));
    verts.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    return JSON.stringify(verts) + '|' + c.faces.length;
  };
  const base = sig(subdPipeNetwork(rails, { radius: R, facets: 8, segments: 3 }));
  for (const p of [[3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]]) {
    assert.equal(sig(subdPipeNetwork(p.map((i) => rails[i]), { radius: R, facets: 8, segments: 3 })), base, `permutation ${p.join('')}`);
  }
});

// ─────────────────────────────────────────────────────────────────────────
// D2 / D3

test('subdPipeNetwork: D2 — two disconnected runs are TWO cages, not one', () => {
  const rails = [
    makeLine([0, 0, 0], [50, 0, 0]),
    makeLine([50, 0, 0], [50, 50, 0]),
    makeLine([200, 0, 0], [250, 0, 0]),
  ];
  const net = subdPipeNetwork(rails, { radius: R, facets: 8, segments: 3, capStart: 'flat', capEnd: 'flat' });
  assert.equal(net.ok, true);
  assert.equal(net.cages.length, 2);
  assert.deepEqual(net.cages[0].sourceRails, [0, 1]);
  assert.deepEqual(net.cages[1].sourceRails, [2]);
  // The 2-way junction was CONCATENATED, not bridged: the first cage is ONE
  // continuous tube along one longer rail, so it has no hub at all.
  assert.equal(net.cages[0].railCount, 1);
  assert.equal(net.cages[0].junctions.length, 0);
  assert.deepEqual(net.junctionCounts, { 2: 1 }, 'the 2-way junction is still reported');
  for (const [i, cage] of net.cages.entries()) {
    assertWellFormed(cage, `run ${i}`);
    assert.equal(nakedEdgeCount(cage), 0);
    assert.equal(eulerCharacteristic(cage), 2);
  }
});

test('subdPipeNetwork: D3 — welding OFF gives N independent tubes and no junctions', () => {
  const rails = crossJunction();
  const net = subdPipeNetwork(rails, { radius: R, facets: 8, segments: 3, weld: false });
  assert.equal(net.ok, true);
  assert.equal(net.welded, false);
  assert.equal(net.cages.length, 4);
  assert.equal(net.junctions.length, 0);
  assert.deepEqual(net.junctionCounts, {});
  assert.equal(net.splitCount, 0);
  for (const [i, cage] of net.cages.entries()) {
    assert.deepEqual(cage.sourceRails, [i]);
    assertWellFormed(cage, `unwelded tube ${i}`);
    // An open cylinder: two boundary circles, chi = 0.
    assert.equal(eulerCharacteristic(cage), 0);
    // Byte-identical to a bare subdPipeCage of that same rail — welding off
    // really does mean "nothing else happened".
    const bare = subdPipeCage(rails[i], { radius: R, facets: 8, segments: 3, capStart: 'none', capEnd: 'none' });
    assert.equal(JSON.stringify(cage.vertices), JSON.stringify(bare.vertices));
    assert.equal(JSON.stringify(cage.faces), JSON.stringify(bare.faces));
  }
});

test('subdPipeNetwork: the arity report names every junction, of every kind', () => {
  // One 3-way (a T onto a host) and one 2-way (an elbow further along).
  const rails = [
    makeLine([-60, 0, 0], [60, 0, 0]),
    makeLine([0, 0.5, 0], [0, 60, 0]),
    makeLine([60, 0, 0], [60, 0, 40]),
  ];
  const net = subdPipeNetwork(rails, { radius: R, facets: 8, segments: 3 });
  assert.equal(net.ok, true);
  assert.deepEqual(net.junctionCounts, { 2: 1, 3: 1 });
  assert.equal(net.junctionTotal, 2);
  assert.equal(net.cages.length, 1);
  assert.equal(net.cages[0].junctions.length, 1, 'only the 3-way needs a hub');
  assertWellFormed(net.cages[0], 'T plus elbow');
});

// ─────────────────────────────────────────────────────────────────────────
// THE SHALLOW-ANGLE FINDING — measured, not assumed.

test('subdPipeNetwork: the inset is angle-aware, so no arm\'s rim ends up buried inside a neighbouring tube', () => {
  // Two arms 60 degrees apart plus a third out of their way. At 60 degrees
  // two radius-5 cylinders interpenetrate out to 5/tan(30) = 8.66mm, so a
  // fixed 0.62*radius = 3.10mm pull-back leaves each rim deep inside the
  // other tube. Measured directly: how far inside the OTHER arm's solid does
  // this arm's rim sit?
  const a = (60 * Math.PI) / 180;
  const dirs = [[1, 0, 0], [Math.cos(a), Math.sin(a), 0], [0, 0, -1]];
  const rails = dirs.map((d) => makeLine([0, 0, 0], d.map((v) => v * 300)));
  const net = subdPipeNetwork(rails, { radius: R, facets: 12, segments: 3 });
  assert.equal(net.ok, true);
  const cage = net.cages[0];
  const inset = cage.junctions[0].inset;
  assert.ok(Math.abs(inset - R / Math.tan(a / 2)) < 1e-9, `inset ${inset} is exactly r/tan(theta/2)`);

  const along = (v, d) => v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
  const offAxis = (v, d) => { const t = along(v, d); return Math.hypot(v[0] - d[0] * t, v[1] - d[1] * t, v[2] - d[2] * t); };
  let worst = 0;
  for (const v of cage.vertices) {
    for (let i = 0; i < 3; i++) {
      // this vertex is on arm i's own rim?
      if (Math.abs(along(v, dirs[i]) - inset) > 1e-6 || Math.abs(offAxis(v, dirs[i]) - R) > 1e-6) continue;
      for (let j = 0; j < 3; j++) {
        if (j === i) continue;
        if (along(v, dirs[j]) < inset) continue; // only where arm j's tube genuinely exists
        worst = Math.max(worst, R - offAxis(v, dirs[j]));
      }
    }
  }
  assert.ok(worst <= 1e-9, `no rim vertex sits inside a neighbouring tube (worst ${worst.toFixed(4)}mm)`);
  assertWellFormed(cage, '60-degree junction');
});

// ─────────────────────────────────────────────────────────────────────────
// REFUSALS — each by name, none producing a wrong result quietly.

test('subdPipeNetwork: a genuinely three-dimensional junction is REFUSED by name, not mis-ordered', () => {
  // The +X/-X/+Y/-Y/+Z frame corner: no single plane orders these arms, and
  // this hub orders around one plane.
  const rails = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]].map((d) => makeLine([0, 0, 0], d.map((v) => v * 60)));
  const net = subdPipeNetwork(rails, { radius: R, facets: 8, segments: 3 });
  assert.equal(net.ok, false);
  assert.match(net.reason, /three-dimensional/);
  assert.ok(net.planarityResidual > 0.25);
});

test('subdPipeNetwork: two arms leaving a junction at a very shallow angle are REFUSED by name', () => {
  const rails = [
    makeLine([0, 0, 0], [60, 0, 0]),
    makeLine([0, 0, 0], [60, 6, 0]), // ~5.7 degrees away
    makeLine([0, 0, 0], [-60, 0, 0]),
  ];
  const net = subdPipeNetwork(rails, { radius: R, facets: 8, segments: 3 });
  assert.equal(net.ok, false);
  assert.match(net.reason, /degrees apart/);
  assert.ok(net.armAngle < (15 * Math.PI) / 180);
});

test('subdPipeNetwork: a rail shorter than the insets its own two junctions demand is REFUSED by name', () => {
  const rails = [
    makeLine([0, 0, 0], [8, 0, 0]), // the short middle
    makeLine([0, 0, 0], [0, 40, 0]), makeLine([0, 0, 0], [0, -40, 0]),
    makeLine([8, 0, 0], [8, 40, 0]), makeLine([8, 0, 0], [8, -40, 0]),
  ];
  const net = subdPipeNetwork(rails, { radius: R, facets: 8, segments: 3 });
  assert.equal(net.ok, false);
  assert.match(net.reason, /leaving no tube in the middle/);
});

test('subdPipeNetwork: a single rail that closes back on itself is REFUSED by name', () => {
  // A closed rail's own two rings land on top of each other, and nothing
  // here welds a tube to itself.
  const net = subdPipeNetwork([makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 40)], { radius: R, facets: 8, segments: 6 });
  assert.equal(net.ok, false);
  assert.match(net.reason, /closes back on itself/);
});

test('subdPipeNetwork: several rails closing into a loop are REFUSED by name too', () => {
  const loop = [
    makeLine([0, 0, 0], [60, 0, 0]),
    makeLine([60, 0, 0], [60, 60, 0]),
    makeLine([60, 60, 0], [0, 0, 0]),
  ];
  const net = subdPipeNetwork(loop, { radius: R, facets: 8, segments: 3 });
  assert.equal(net.ok, false);
  assert.match(net.reason, /closing a loop|closed loop|closes back on itself/);
});

test('detectPipeJunctions: a branch landing on a rail in two places at once is REFUSED, not guessed', () => {
  // A host that doubles back on itself, and a branch stopping between its
  // two legs: the closest point is genuinely two different places at once,
  // and neither is near an end of the host, so there is no single parameter
  // to split it at. Split arbitrarily and the network would weld to
  // whichever leg the search happened to settle on.
  const host = globalCurveInterp([[40, 0, 0], [0, 0, 0], [-3, 2, 0], [0, 4, 0], [40, 4, 0]], 2);
  const branch = makeLine([20, 2, 0], [20, 2, 60]);
  const det = detectPipeJunctions([host, branch], { radius: R, tolerance: 8 });
  assert.equal(det.ok, false);
  assert.match(det.reason, /two genuinely different places/);
});

// ─────────────────────────────────────────────────────────────────────────
// The bend-radius clamp, reused rather than rebuilt.

test('subdPipeNetwork: a rail tighter than the requested tube radius clamps it, and says so', () => {
  // A tight S: its own minimum bend radius is far below the requested tube
  // radius, so a tube that size would swallow itself.
  const tight = globalCurveInterp([[0, 0, 0], [6, 8, 0], [12, 0, 0], [18, 8, 0]], 3);
  const net = subdPipeNetwork([tight], { radius: 20, facets: 8, segments: 4 });
  assert.equal(net.ok, true);
  assert.equal(net.radius.clamped, true);
  assert.ok(net.radius.radius < 20);
  assert.ok(Math.abs(net.radius.radius - net.radius.safeMax) < 1e-12);
  assertWellFormed(net.cages[0], 'clamped tight rail');
  // and an ordinary rail is left alone
  const easy = subdPipeNetwork([makeLine([0, 0, 0], [100, 0, 0])], { radius: 5, facets: 8, segments: 3 });
  assert.equal(easy.radius.clamped, false);
  assert.equal(easy.radius.radius, 5);
});

test('subdPipeNetwork: the weld tolerance is a real dial — a gap welds or does not, by fraction', () => {
  // A branch stopping 3mm short of a host's end. At the default 0.5 fraction
  // of a radius-5 tube that is inside the 2.5mm tolerance... it is not, so it
  // stays two components; raising the fraction welds it.
  const rails = [makeLine([0, 0, 0], [50, 0, 0]), makeLine([53, 0, 0], [53, 50, 0])];
  const apart = subdPipeNetwork(rails, { radius: R, facets: 8, segments: 3 });
  assert.equal(apart.ok, true);
  assert.equal(apart.cages.length, 2, 'a 3mm gap is outside a 2.5mm tolerance');
  const welded = subdPipeNetwork(rails, { radius: R, facets: 8, segments: 3, weldFraction: 0.8 });
  assert.equal(welded.ok, true);
  assert.equal(welded.cages.length, 1, 'a 3mm gap is inside a 4mm tolerance');
  assert.deepEqual(welded.junctionCounts, { 2: 1 });
});
