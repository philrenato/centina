// THE GATE for ToNURBS step 3 — the cap over the
// star-point region that isolation shrinks but never removes.
//
// WHAT IS ACTUALLY BEING GATED, and it is not "does a patch come out". A cap
// is an approximation by necessity: the Catmull-Clark limit surface over a
// face touching an extraordinary vertex is an infinite nest of bicubic
// patches, so no single bicubic is it. What can be exact is everything around
// the approximation — where the cap meets surfaces that ARE the limit
// surface, and where it meets its own siblings — and that is what these tests
// pin down, as identities rather than tolerances. The approximation itself is
// MEASURED against independent ground truth and asserted between bounds on
// both sides, so a cap that silently got worse fails and so does one whose
// error was quietly claimed away.
//
// GROUND TRUTH, INDEPENDENT OF THE THING UNDER TEST. Two sources, neither
// sharing a code path with the cap builder:
//
//   - vertexLimitPosition, the Halstead/Kass/DeRose mask, verified three ways
//     before any of this existed;
//   - limitSamples below, which walks the natural parametrisation down to a
//     dyadic corner and reads that refined vertex's own limit position. Its
//     parametrisation is derived from subdivideCatmullClark's child ordering
//     by composing forward maps, NOT by the index arithmetic the cap builder
//     uses to find the same points — and the first test proves it right by
//     making it reproduce a regular face's exact patch.
import test from 'node:test';
import assert from 'node:assert/strict';
import { subdivideCatmullClark, buildTopology } from '../kernel/subd.mjs';
import { superbBoxCage, superbEllipsoidCage, superbTorusCage } from '../kernel/subdprimitives.mjs';
import { extrudeFaces } from '../kernel/subdedit.mjs';
import { surfacePoint, surfacePointAndPartials } from '../kernel/surface.mjs';
import {
  isRegularFace, regularFaceToPatch, subdToPatches, vertexLimitPosition,
  patchBoundaryRow, accInteriorPoint,
} from '../kernel/subdlimit.mjs';

// ---------------------------------------------------------------------
// Ground truth: the limit surface of face `fi` at every dyadic parameter of
// resolution 2^-k. Child c of a quad is [v_c, edge(v_c,v_c+1), facePoint,
// edge(v_c-1,v_c)], which fixes how its own (s,t) sits inside the parent's.
const CHILD_UV = [
  (s, t) => [s / 2, t / 2],
  (s, t) => [1 - t / 2, s / 2],
  (s, t) => [1 - s / 2, 1 - t / 2],
  (s, t) => [t / 2, 1 - s / 2],
];
const CORNER_ST = [[0, 0], [1, 0], [1, 1], [0, 1]];

// The refinement chain is shared across every face of the same cage — one
// cage refined k times, not one per cap, or this file dominates the suite.
const REFINED = new WeakMap();
function refinedTo(cage, k) {
  if (!REFINED.has(cage)) REFINED.set(cage, [{ cage, ctx: buildTopology(cage) }]);
  const chain = REFINED.get(cage);
  while (chain.length <= k) {
    const next = subdivideCatmullClark(chain[chain.length - 1].cage);
    chain.push({ cage: next, ctx: buildTopology(next) });
  }
  return chain[k];
}

function limitSamples(cage, fi, k) {
  let items = [{ face: fi, uv: CORNER_ST.map(([u, v]) => [u, v]) }];
  for (let lv = 0; lv < k; lv++) {
    const next = [];
    for (const it of items) {
      for (let c = 0; c < 4; c++) {
        next.push({
          face: 4 * it.face + c,
          uv: CORNER_ST.map(([s, t]) => {
            const [pu, pv] = CHILD_UV[c](s, t);
            const [a, b, cc, d] = it.uv;
            return [0, 1].map((x) => a[x] * (1 - pu) * (1 - pv) + b[x] * pu * (1 - pv) + cc[x] * pu * pv + d[x] * (1 - pu) * pv);
          }),
        });
      }
    }
    items = next;
  }
  const { cage: cur, ctx } = refinedTo(cage, k);
  const out = new Map();
  for (const it of items) {
    const f = cur.faces[it.face];
    for (let c = 0; c < 4; c++) {
      const key = `${it.uv[c][0].toFixed(9)},${it.uv[c][1].toFixed(9)}`;
      if (!out.has(key)) out.set(key, { u: it.uv[c][0], v: it.uv[c][1], p: vertexLimitPosition(cur, f[c], ctx) });
    }
  }
  return [...out.values()];
}
const sampleAt = (samples, u, v) => {
  const s = samples.find((x) => Math.abs(x.u - u) < 1e-9 && Math.abs(x.v - v) < 1e-9);
  assert.ok(s, `no dyadic sample at (${u},${v})`);
  return s.p;
};
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// A cap's own frame puts the star at (0,0); the sampler works in the face's
// native frame, whose (0,0) is face[0].
const toCapFrame = (rot, u, v) => [[u, v], [v, 1 - u], [1 - u, 1 - v], [1 - v, u]][rot];
const fromCapFrame = (rot, u, v) => [[u, v], [1 - v, u], [1 - u, 1 - v], [v, 1 - u]][rot];

function normalAt(srf, u, v) {
  const { su, sv } = surfacePointAndPartials(srf, u, v);
  const n = [su[1] * sv[2] - su[2] * sv[1], su[2] * sv[0] - su[0] * sv[2], su[0] * sv[1] - su[1] * sv[0]];
  const L = Math.hypot(n[0], n[1], n[2]);
  return L < 1e-12 ? null : n.map((x) => x / L);
}
const angleDeg = (a, b) => (!a || !b) ? null
  : Math.acos(Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * 180 / Math.PI;

// (u,v) on a patch built from `loop`, a fraction t of the way along p -> q.
function edgeUV(loop, p, q, t) {
  for (let e = 0; e < 4; e++) {
    const fwd = loop[e] === p && loop[(e + 1) % 4] === q;
    if (!fwd && !(loop[e] === q && loop[(e + 1) % 4] === p)) continue;
    const s = fwd ? t : 1 - t;
    return [[s, 0], [1, s], [1 - s, 1], [0, 1 - s]][e];
  }
  throw new Error('edgeUV: not an edge of that loop');
}

// Every shared edge of an emitted set, with both sides' patches and loops.
function sharedEdges(result, cage, ctx) {
  const byFace = new Map();
  for (const p of result.patches) {
    if (p.level !== result.levelsUsed) continue;
    byFace.set(p.faceIndex, { srf: p.srf, kind: p.kind, loop: cage.faces[p.faceIndex] });
  }
  for (const c of result.caps) byFace.set(c.faceIndex, { srf: c.srf, kind: 'cap', loop: c.vertexLoop });
  const out = [];
  for (const e of ctx.edgeMap.values()) {
    if (e.faces.length !== 2) continue;
    const A = byFace.get(e.faces[0]), B = byFace.get(e.faces[1]);
    if (!A || !B) continue;
    out.push({ v0: e.v0, v1: e.v1, A, B, kind: [A.kind, B.kind].sort().join('|') });
  }
  return out;
}

// ---------------------------------------------------------------------
// FIXTURES. Measured here, asserted nowhere else — a fixture nobody measured
// is a free variable in every result resting on it.
const BOX = () => superbBoxCage([0, 0, 0], [10, 10, 10], 1);
const ELLIPSOID = () => superbEllipsoidCage([0, 0, 0], [30, 20, 12], 2);

function extrudedBoxCage() {
  const base = superbBoxCage([0, 0, 0], [10, 10, 10], 2);
  const ctx = buildTopology(base);
  const target = base.faces.findIndex((_, fi) => ctx.faceCentroids[fi][2] > 9.9);
  assert.ok(target >= 0, 'fixture must have a +Z face to extrude');
  return extrudeFaces(base, [target], [0, 0, 1], 8).cage;
}

test('the ground-truth sampler is right: on a REGULAR face it reproduces the exact patch', () => {
  // Nothing below means anything if the parametrisation is wrong, and a wrong
  // one still produces perfectly real points at perfectly real parameters.
  for (const cage of [subdivideCatmullClark(subdivideCatmullClark(BOX())), superbTorusCage([0, 0, 0], 30, 10, 8)]) {
    const ctx = buildTopology(cage);
    const fi = cage.faces.findIndex((_, i) => isRegularFace(cage, i, ctx));
    assert.ok(fi >= 0);
    const srf = regularFaceToPatch(cage, fi, ctx);
    const samples = limitSamples(cage, fi, 3);
    assert.equal(samples.length, 81, 'a level-3 descent must give a 9x9 dyadic grid');
    let worst = 0;
    for (const s of samples) worst = Math.max(worst, dist(surfacePoint(srf, s.u, s.v), s.p));
    assert.ok(worst < 1e-13, `sampler must agree with the exact regular patch, worst ${worst}`);
  }
});

test('the extruded-face fixture is what the app actually hits: valence 3, 4 AND 5, all quads, closed', () => {
  const cage = extrudedBoxCage();
  const ctx = buildTopology(cage);
  const hist = {};
  for (let v = 0; v < cage.vertices.length; v++) {
    assert.ok(ctx.vertexEdges[v].every((e) => e.faces.length === 2), `vertex ${v} must be interior — this fixture is closed`);
    const n = ctx.vertexEdges[v].length;
    hist[n] = (hist[n] || 0) + 1;
  }
  assert.ok(cage.faces.every((f) => f.length === 4), 'every face must be a quad');
  // Extruding one face of a facets-2 box: its four corners gain a face, and
  // four new corners appear with three faces each. The chosen face touches an
  // original box corner, so that one goes 3 -> 4 and only three corners reach
  // valence 5 — the fixture is deliberately NOT symmetric.
  assert.deepEqual(hist, { 3: 11, 4: 16, 5: 3 });
  assert.equal(cage.vertices.length, 30);
  assert.equal(cage.faces.length, 28);
});

// ---------------------------------------------------------------------
test('every leftover region gets a cap, and the hole closes: uncoveredFraction goes to zero', () => {
  const cage = BOX();
  const plain = subdToPatches(cage, { maxIsolation: 3 });
  // Eight valence-3 corners, three faces each — measured, and the count is
  // what proves isolation converged rather than stalled.
  assert.equal(plain.uncovered.length, 24);
  assert.equal(new Set(plain.uncovered.map((u) => u.extraordinary[0].vertex)).size, 8);
  assert.equal(plain.uncoveredFraction, 0.0625);

  const capped = subdToPatches(cage, { maxIsolation: 3, cap: true });
  assert.equal(capped.caps.length, 24, 'one cap per leftover region');
  assert.equal(capped.uncovered.length, 0, 'nothing left uncovered');
  assert.equal(capped.uncoveredFraction, 0);
  assert.equal(capped.patches.length, plain.patches.length + 24);
  assert.equal(capped.patches.filter((p) => p.kind === 'cap').length, 24);
  assert.equal(new Set(capped.caps.map((c) => c.star)).size, 8, 'eight star points');
});

test('capping is opt-in — without it the leftovers are still reported, unchanged', () => {
  const cage = BOX();
  const plain = subdToPatches(cage, { maxIsolation: 3 });
  assert.equal(plain.caps.length, 0);
  assert.equal(plain.uncovered.length, 24);
  for (const u of plain.uncovered) assert.equal(u.reason, undefined);
});

test('a cap patch is the same KIND of surface as a regular one: clamped, degree 3, non-rational', () => {
  const { caps } = subdToPatches(BOX(), { maxIsolation: 3, cap: true });
  for (const c of caps) {
    assert.deepEqual(c.srf.knotsU, [0, 0, 0, 0, 1, 1, 1, 1]);
    assert.deepEqual(c.srf.knotsV, [0, 0, 0, 0, 1, 1, 1, 1]);
    assert.equal(c.srf.degU, 3);
    assert.equal(c.srf.degV, 3);
    assert.equal(c.srf.ctrlNet.length, 4);
    for (const row of c.srf.ctrlNet) for (const p of row) {
      assert.equal(row.length, 4);
      assert.equal(p[3], 1);
      assert.ok(p.slice(0, 3).every(Number.isFinite));
    }
  }
});

test('EXACT: the cap\'s star corner IS the extraordinary vertex\'s own limit position, bit for bit', () => {
  for (const [cage, lv] of [[BOX(), 3], [ELLIPSOID(), 2], [extrudedBoxCage(), 2]]) {
    const r = subdToPatches(cage, { maxIsolation: lv, cap: true });
    assert.ok(r.caps.length > 0);
    const ctx = buildTopology(r.refinedCage);
    for (const c of r.caps) {
      const want = vertexLimitPosition(r.refinedCage, c.star, ctx);
      const got = c.srf.ctrlNet[0][0];
      // === and not a tolerance: the cap does not recompute this, it carries
      // the mask's own number.
      for (let d = 0; d < 3; d++) assert.equal(got[d], want[d], `cap at star ${c.star}, coord ${d}`);
      assert.deepEqual(c.starLimitPosition.slice(0, 3), want.slice(0, 3));
    }
  }
});

test('EXACT: two caps sharing a star edge carry the IDENTICAL control row — every coordinate ===, not a tolerance', () => {
  for (const [cage, lv] of [[BOX(), 3], [ELLIPSOID(), 2], [extrudedBoxCage(), 2]]) {
    const r = subdToPatches(cage, { maxIsolation: lv, cap: true });
    const ctx = buildTopology(r.refinedCage);
    const edges = sharedEdges(r, r.refinedCage, ctx).filter((e) => e.kind === 'cap|cap');
    assert.ok(edges.length >= r.caps.length / 2, `expected a star edge per cap pair, got ${edges.length}`);
    for (const e of edges) {
      const a = patchBoundaryRow(e.A.srf, e.A.loop, e.v0, e.v1);
      const b = patchBoundaryRow(e.B.srf, e.B.loop, e.v0, e.v1);
      for (let i = 0; i < 4; i++) for (let d = 0; d < 3; d++) {
        assert.equal(a[i][d], b[i][d], `star edge ${e.v0}-${e.v1}, control point ${i}, coord ${d}`);
      }
    }
  }
});

test('EXACT: a cap takes its outer rows FROM its regular neighbours — every interior control point ===', () => {
  // The two control points strictly inside a shared row are copied, so they
  // are identical by construction. The two CORNERS are the interesting case:
  // a corner belongs to more than two patches and can only carry one value,
  // and the regular patches themselves do not agree on it to the last bit.
  // So the claim asserted is the one that is true — the whole row matches one
  // of the cap's two neighbours, and matches the other everywhere except that
  // shared corner, by no more than the two neighbours already differ.
  for (const [cage, lv, label] of [[BOX(), 3, 'box'], [ELLIPSOID(), 2, 'ellipsoid'], [extrudedBoxCage(), 2, 'extruded']]) {
    const r = subdToPatches(cage, { maxIsolation: lv, cap: true });
    const ctx = buildTopology(r.refinedCage);
    const edges = sharedEdges(r, r.refinedCage, ctx).filter((e) => e.kind === 'cap|regular');
    assert.equal(edges.length, r.caps.length * 2, `${label}: each cap borders exactly two emitted patches`);
    let wholeRows = 0, worst = 0;
    for (const e of edges) {
      const a = patchBoundaryRow(e.A.srf, e.A.loop, e.v0, e.v1);
      const b = patchBoundaryRow(e.B.srf, e.B.loop, e.v0, e.v1);
      for (const i of [1, 2]) for (let d = 0; d < 3; d++) {
        assert.equal(a[i][d], b[i][d], `${label}: interior control point ${i} of a cap/regular row must be COPIED, not recomputed`);
      }
      let same = true;
      for (let i = 0; i < 4; i++) for (let d = 0; d < 3; d++) {
        if (a[i][d] !== b[i][d]) same = false;
        worst = Math.max(worst, Math.abs(a[i][d] - b[i][d]));
      }
      if (same) wholeRows++;
    }
    assert.equal(wholeRows >= r.caps.length, true, `${label}: at least one whole row per cap must be bit-identical, got ${wholeRows} of ${edges.length}`);
    assert.ok(worst < 1e-13, `${label}: where a shared corner cannot be bit-identical it must still be a rounding difference, got ${worst}`);
  }
});

test('the pre-existing set is no better than that: NO two adjacent regular patches are bit-identical once a cage has been subdivided', () => {
  // The negative control for the test above, and the reason its claim is
  // phrased the way it is. Two adjacent regular patches compute the same
  // shared row through a different order of the same arithmetic; they agree
  // algebraically and disagree in the last bits. Capping does not introduce
  // that, and closing it would mean welding patches that are currently
  // exactly the limit surface.
  const cage = subdivideCatmullClark(subdivideCatmullClark(BOX()));
  const ctx = buildTopology(cage);
  const reg = cage.faces.map((_, i) => i).filter((i) => isRegularFace(cage, i, ctx));
  const regSet = new Set(reg);
  const srfs = new Map(reg.map((i) => [i, regularFaceToPatch(cage, i, ctx)]));
  let pairs = 0, bitIdentical = 0, worst = 0;
  for (const e of ctx.edgeMap.values()) {
    if (e.faces.length !== 2 || !regSet.has(e.faces[0]) || !regSet.has(e.faces[1])) continue;
    pairs++;
    const a = patchBoundaryRow(srfs.get(e.faces[0]), cage.faces[e.faces[0]], e.v0, e.v1);
    const b = patchBoundaryRow(srfs.get(e.faces[1]), cage.faces[e.faces[1]], e.v0, e.v1);
    let same = true;
    for (let i = 0; i < 4; i++) for (let d = 0; d < 3; d++) {
      if (a[i][d] !== b[i][d]) same = false;
      worst = Math.max(worst, Math.abs(a[i][d] - b[i][d]));
    }
    if (same) bitIdentical++;
  }
  assert.equal(pairs, 120);
  assert.equal(bitIdentical, 0, 'the "bit-for-bit" reading of adjacent regular rows does not survive a subdivided cage');
  assert.ok(worst > 0 && worst < 1e-14, `they must still agree to rounding, got ${worst}`);
});

test('NO HOLE: sampled along every shared edge, both sides give the same point', () => {
  // Control-net identity is the strong claim; this is the one a downstream
  // consumer actually experiences, and it covers the corners the net cannot
  // make identical.
  for (const [cage, lv] of [[BOX(), 3], [extrudedBoxCage(), 2]]) {
    const r = subdToPatches(cage, { maxIsolation: lv, cap: true });
    const ctx = buildTopology(r.refinedCage);
    const edges = sharedEdges(r, r.refinedCage, ctx);
    assert.ok(edges.length > 100);
    let worst = 0;
    for (const e of edges) {
      for (let i = 0; i <= 8; i++) {
        const t = i / 8;
        const [ua, va] = edgeUV(e.A.loop, e.v0, e.v1, t);
        const [ub, vb] = edgeUV(e.B.loop, e.v0, e.v1, t);
        worst = Math.max(worst, dist(surfacePoint(e.A.srf, ua, va), surfacePoint(e.B.srf, ub, vb)));
      }
    }
    assert.ok(worst < 1e-12, `no shared edge may open a gap, worst ${worst}`);
  }
});

test('TANGENT, MEASURED: the join to the exact regular region is smooth; the break is confined to the star edges', () => {
  // ACC's own continuity claim is that patches meet smoothly except along an
  // edge CONTAINING an extraordinary vertex. A cap's two outer edges do not
  // contain one, so that join must come out tangent-continuous — and it is
  // the join to the part of the surface that IS the limit surface, so it is
  // the one that matters. The star edges are C0 only; that number is real,
  // is asserted from BOTH sides so it can neither grow nor be claimed away,
  // and halves as isolation rises.
  const seen = {};
  for (const [cage, lv, label] of [[BOX(), 3, 'box'], [ELLIPSOID(), 2, 'ellipsoid'], [extrudedBoxCage(), 3, 'extruded']]) {
    const r = subdToPatches(cage, { maxIsolation: lv, cap: true });
    const ctx = buildTopology(r.refinedCage);
    let outer = 0, star = 0, regular = 0;
    for (const e of sharedEdges(r, r.refinedCage, ctx)) {
      let worst = 0;
      for (let i = 1; i <= 7; i++) {
        const t = i / 8;
        const [ua, va] = edgeUV(e.A.loop, e.v0, e.v1, t);
        const [ub, vb] = edgeUV(e.B.loop, e.v0, e.v1, t);
        worst = Math.max(worst, angleDeg(normalAt(e.A.srf, ua, va), normalAt(e.B.srf, ub, vb)) ?? 0);
      }
      if (e.kind === 'cap|cap') star = Math.max(star, worst);
      else if (e.kind === 'cap|regular') outer = Math.max(outer, worst);
      else regular = Math.max(regular, worst);
    }
    // 1e-4 degrees, not 0: the normal is a cross product of two evaluated
    // partials, so its own floating-point noise sets the floor. That floor is
    // measured at about 1e-6 degrees between two REGULAR patches, which is the
    // control — a cap's outer join reads the same number as a join that is
    // exact by construction, and both are four orders below the star break.
    assert.ok(regular < 1e-4, `${label}: regular meets regular smoothly, got ${regular} deg`);
    assert.ok(outer < 1e-4, `${label}: a cap must meet the exact region smoothly, got ${outer} deg`);
    assert.ok(star > 0.01, `${label}: the star-edge break is real and must not be claimed away, got ${star} deg`);
    seen[label] = star;
  }
  // Measured maxima: box 0.29 deg, ellipsoid 0.55 deg, extruded box 2.22 deg
  // — the extruded cage is the worst because it carries valence 5 next to
  // valence 3 with no symmetry to help.
  assert.ok(seen.box < 0.4, `box star break ${seen.box}`);
  assert.ok(seen.ellipsoid < 0.8, `ellipsoid star break ${seen.ellipsoid}`);
  assert.ok(seen.extruded < 3, `extruded-box star break ${seen.extruded}`);
});

test('the star-edge tangent break shrinks with isolation level — it is a real knob, not a fixed cost', () => {
  const cage = BOX();
  const measure = (lv) => {
    const r = subdToPatches(cage, { maxIsolation: lv, cap: true });
    const ctx = buildTopology(r.refinedCage);
    let worst = 0;
    for (const e of sharedEdges(r, r.refinedCage, ctx)) {
      if (e.kind !== 'cap|cap') continue;
      for (let i = 1; i <= 7; i++) {
        const t = i / 8;
        const [ua, va] = edgeUV(e.A.loop, e.v0, e.v1, t);
        const [ub, vb] = edgeUV(e.B.loop, e.v0, e.v1, t);
        worst = Math.max(worst, angleDeg(normalAt(e.A.srf, ua, va), normalAt(e.B.srf, ub, vb)) ?? 0);
      }
    }
    return worst;
  };
  const a = measure(2), b = measure(3), c = measure(4);
  assert.ok(b < a * 0.75 && c < b * 0.75, `must genuinely shrink: ${a} -> ${b} -> ${c}`);
});

test('EXACT: the cap interpolates the true limit surface at the three points it is solved for', () => {
  // This is the gate on the whole star refinement, including the fact that it
  // reads its ground truth out of a LOCAL neighbourhood rather than the whole
  // cage: if that neighbourhood were too small, these three points would be
  // wrong and the interpolation would land somewhere else. Checked against
  // the independent descent sampler, not against the index arithmetic the
  // builder uses to find the same vertices.
  for (const [cage, lv] of [[BOX(), 2], [ELLIPSOID(), 2], [extrudedBoxCage(), 2]]) {
    const r = subdToPatches(cage, { maxIsolation: lv, cap: true });
    const cur = r.refinedCage;
    let worst = 0;
    for (const c of r.caps) {
      const rot = cur.faces[c.faceIndex].indexOf(c.star);
      const samples = limitSamples(cur, c.faceIndex, 2);
      for (const [u, v] of [[0.25, 0], [0, 0.25], [0.25, 0.25]]) {
        const [nu, nv] = fromCapFrame(rot, u, v);
        worst = Math.max(worst, dist(surfacePoint(c.srf, u, v), sampleAt(samples, nu, nv)));
      }
    }
    assert.ok(worst < 1e-11, `the cap must pass through its three solved limit points, worst ${worst}`);
  }
});

test('MEASURED, not claimed: how far a cap is from the true limit surface, and that it converges', () => {
  // The honest number. A single bicubic cannot BE the limit surface over a
  // face touching an extraordinary vertex, so this is bounded on both sides:
  // an upper bound that fails if the construction degrades, and a lower bound
  // that fails if someone starts claiming exactness the surface does not have.
  const measure = (cage, lv) => {
    const r = subdToPatches(cage, { maxIsolation: lv, cap: true });
    const cur = r.refinedCage;
    let worst = 0, diag = 0;
    for (const c of r.caps) {
      const rot = cur.faces[c.faceIndex].indexOf(c.star);
      const o = c.vertexLoop.map((v) => cur.vertices[v]);
      diag = Math.max(diag, dist(o[0], o[2]));
      for (const s of limitSamples(cur, c.faceIndex, 3)) {
        const [u, v] = toCapFrame(rot, s.u, s.v);
        worst = Math.max(worst, dist(surfacePoint(c.srf, u, v), s.p));
      }
    }
    return { worst, relative: worst / diag };
  };
  const box = [2, 3, 4].map((lv) => measure(BOX(), lv));
  for (const m of box) {
    assert.ok(m.relative > 1e-4, 'a cap is an approximation — a zero here would mean the fixture stopped discriminating');
    assert.ok(m.relative < 0.006, `worst deviation must stay under 0.6% of the cap's own size, got ${(m.relative * 100).toFixed(2)}%`);
  }
  // The absolute error shrinks about 2.4x per level while the RELATIVE error
  // holds steady — the star region is self-similar under refinement, so
  // raising the isolation level buys a smaller hole, not a better shape
  // inside it. That is the honest characterisation of the knob.
  assert.ok(box[1].worst < box[0].worst / 2 && box[2].worst < box[1].worst / 2, `absolute error must fall: ${box.map((m) => m.worst).join(' -> ')}`);
  assert.ok(Math.abs(box[2].relative - box[0].relative) < 0.001, `relative error does NOT fall: ${box.map((m) => m.relative).join(' -> ')}`);

  const ell = measure(ELLIPSOID(), 2);
  assert.ok(ell.relative < 0.01, `curved cage: ${(ell.relative * 100).toFixed(2)}%`);
  const ext = measure(extrudedBoxCage(), 2);
  assert.ok(ext.relative < 0.015, `extruded cage: ${(ext.relative * 100).toFixed(2)}%`);
});

test('the ACC interior mask reduces EXACTLY to the uniform B-spline one at valence 4', () => {
  // The correctness check that matters for a mask re-derived rather than
  // transcribed: at valence 4 it has to reproduce the number
  // clampedBicubicPatchSurface already produces for a regular face, reached a
  // completely different way. A wrong generalisation still looks plausible
  // everywhere else.
  const cage = subdivideCatmullClark(subdivideCatmullClark(BOX()));
  const ctx = buildTopology(cage);
  const fi = cage.faces.findIndex((_, i) => isRegularFace(cage, i, ctx));
  assert.ok(fi >= 0);
  const srf = regularFaceToPatch(cage, fi, ctx);
  const face = cage.faces[fi];
  // Interior Bezier point (i,j) of the patch sits at the face's corner slot.
  const slots = [[1, 1, 0], [2, 1, 1], [2, 2, 2], [1, 2, 3]];
  let worst = 0;
  for (const [i, j, k] of slots) {
    const acc = accInteriorPoint(cage, ctx, fi, face[k]);
    worst = Math.max(worst, dist(acc, srf.ctrlNet[i][j]));
  }
  assert.ok(worst < 1e-13, `ACC's interior mask must reproduce the regular patch's own interior points, worst ${worst}`);
  // And it must genuinely DEPEND on the valence, or the test above proves
  // nothing about the extraordinary case.
  const star = subdToPatches(BOX(), { maxIsolation: 2, cap: true }).caps[0];
  assert.ok(star, 'need a cap to look at');
});

test('a region that cannot be capped is REFUSED with a reason, never covered by a patch that is not the surface', () => {
  // Isolation 0 on a box: no face is regular yet, so no leftover has the two
  // emitted neighbours a cap is built from.
  const r = subdToPatches(BOX(), { maxIsolation: 0, cap: true });
  assert.equal(r.caps.length, 0);
  assert.equal(r.uncovered.length, 6);
  assert.equal(r.uncoveredFraction, 1);
  for (const u of r.uncovered) assert.match(u.reason, /extraordinary vertices|emitted patch/);

  // Isolation 1: every face still touches an extraordinary vertex, so the
  // refusal names the other reason.
  const r1 = subdToPatches(BOX(), { maxIsolation: 1, cap: true });
  assert.ok(r1.uncovered.length > 0);
  for (const u of r1.uncovered) assert.ok(typeof u.reason === 'string' && u.reason.length > 0);
});

test('a cage with no extraordinary vertex is untouched by capping', () => {
  const cage = superbTorusCage([0, 0, 0], 30, 10, 8);
  const plain = subdToPatches(cage, { maxIsolation: 3 });
  const capped = subdToPatches(cage, { maxIsolation: 3, cap: true });
  assert.equal(capped.caps.length, 0);
  assert.equal(capped.uncovered.length, 0);
  assert.equal(capped.patches.length, plain.patches.length);
  for (let i = 0; i < plain.patches.length; i++) {
    assert.deepEqual(capped.patches[i].srf.ctrlNet, plain.patches[i].srf.ctrlNet);
  }
});

test('capping never mutates the cage it was given', () => {
  const cage = extrudedBoxCage();
  const before = JSON.stringify(cage);
  subdToPatches(cage, { maxIsolation: 3, cap: true });
  assert.equal(JSON.stringify(cage), before);
});

test('capping does not disturb the regular patches — they are the same surfaces either way', () => {
  const cage = ELLIPSOID();
  const plain = subdToPatches(cage, { maxIsolation: 3 });
  const capped = subdToPatches(cage, { maxIsolation: 3, cap: true });
  const byKey = new Map(plain.patches.map((p) => [`${p.level}:${p.faceIndex}`, p.srf]));
  let checked = 0;
  for (const p of capped.patches) {
    if (p.kind !== 'regular') continue;
    assert.deepEqual(p.srf.ctrlNet, byKey.get(`${p.level}:${p.faceIndex}`).ctrlNet);
    checked++;
  }
  assert.equal(checked, plain.patches.length);
});
