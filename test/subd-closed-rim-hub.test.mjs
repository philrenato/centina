// N-WAY CLOSED-RIM HUB — N >= 3 tube ends welded into one junction.
//
// The fixtures are REAL subdPipeCage tubes, not hand-typed loops: a rim built
// by the thing that will actually call this carries whatever ring orientation,
// winding and vertex ordering that construction genuinely produces, and a
// toy loop typed to be convenient carries whatever the author assumed.
//
// Every structural claim is recomputed here from the raw { vertices, faces }
// rather than asked of buildTopology, so a shared mistake between this file
// and the code under test cannot agree its way to a pass. The one place
// buildTopology IS used is to read the INPUT cage's naked edges, which is a
// statement about the fixture, not about the result.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bridgeClosedRimsHub, bridgeEdgeRunsHub, bridgeEdgeRuns } from '../kernel/subdedit.mjs';
import { subdPipeCage } from '../kernel/subdpipe.mjs';
import { subdivideCatmullClark, buildTopology, edgeKey } from '../kernel/subd.mjs';
import { makeLine } from '../kernel/primitives.mjs';

// ── structural invariants, recomputed from raw arrays ────────────────────
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
function directedEdgeReuse(cage) {
  const seen = new Set();
  let reused = 0;
  for (const f of cage.faces) for (let i = 0; i < f.length; i++) {
    const k = `${f[i]}>${f[(i + 1) % f.length]}`;
    if (seen.has(k)) reused++;
    seen.add(k);
  }
  return reused;
}
function eulerCharacteristic(cage) {
  return cage.vertices.length - edgeFaceCounts(cage).size + cage.faces.length;
}
// Every claim a valid cage has to satisfy, in one place, so no test can
// accidentally check fewer of them than another.
function assertValidCage(cage, label) {
  assert.ok(cage.vertices.every((p) => p.length === 3 && p.every(Number.isFinite)), `${label}: every vertex is a finite point`);
  for (const f of cage.faces) {
    assert.ok(f.length >= 3, `${label}: no face with fewer than 3 corners`);
    assert.equal(new Set(f).size, f.length, `${label}: no face repeats a vertex`);
    assert.ok(f.every((i) => Number.isInteger(i) && i >= 0 && i < cage.vertices.length), `${label}: every face index is in range`);
  }
  for (const [k, n] of edgeFaceCounts(cage)) assert.ok(n <= 2, `${label}: edge ${k} is used by ${n} faces — a manifold edge has at most 2`);
  assert.equal(directedEdgeReuse(cage), 0, `${label}: no directed edge is traversed twice — consistently wound`);
  assert.equal(new Set(cage.faces.flat()).size, cage.vertices.length, `${label}: every vertex is used by a real face — nothing orphaned`);
}
function nakedEdges(cage) {
  return [...edgeFaceCounts(cage)].filter(([, n]) => n === 1).map(([k]) => k);
}

// ── fixtures: real tubes ─────────────────────────────────────────────────
function rotZ(p, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]];
}
// One arm: a genuine subdPipeCage tube running OUTWARD from `inner` to
// `outer` along `dir`, capped flat at the far end so the ONLY naked edges in
// the assembled fixture are the hub rims — which makes "is the junction
// closed" a question the raw edge counts can answer.
function arm(dir, { radius = 5, facets = 8, segments = 2, inner = 12, outer = 40 } = {}) {
  const d = (() => { const L = Math.hypot(...dir); return dir.map((x) => x / L); })();
  const rail = makeLine(d.map((x) => x * inner), d.map((x) => x * outer));
  const c = subdPipeCage(rail, { radius, facets, segments, capStart: 'none', capEnd: 'flat' });
  return { vertices: c.vertices, faces: c.faces, rim: c.startRim };
}
function transform(part, fn) {
  return { vertices: part.vertices.map(fn), faces: part.faces.map((f) => [...f]), rim: [...part.rim] };
}
function mergeParts(parts) {
  const vertices = [], faces = [], rims = [];
  for (const p of parts) {
    const off = vertices.length;
    for (const v of p.vertices) vertices.push([...v]);
    for (const f of p.faces) faces.push(f.map((i) => i + off));
    rims.push(p.rim.map((i) => i + off));
  }
  return { cage: { vertices, faces, creases: {} }, rims };
}
// SYMMETRIC — every arm is the SAME tube rigidly rotated, so the arms are
// exact images of one another and every "which one is nearer" decision inside
// the hub is a genuine tie rather than merely a close call. This is the
// fixture class that exposed two real order-dependence bugs in the run hub;
// a tilted, unevenly spaced one is robust before and after such a bug and so
// proves nothing about it.
function symmetricJunction(n, opts = {}) {
  const base = arm([1, 0, 0], opts);
  return mergeParts(Array.from({ length: n }, (_, k) => transform(base, (p) => rotZ(p, (k / n) * Math.PI * 2))));
}
// AWKWARD — uneven angles, every arm tilted out of the junction plane by a
// different amount, different radii and different lengths. Nothing here is
// symmetric, so nothing can pass by cancelling.
function awkwardJunction() {
  const specs = [
    { dir: [1, 0, 0.22], radius: 5, inner: 13, outer: 44, segments: 2 },
    { dir: [-0.34, 0.94, -0.30], radius: 7, inner: 16, outer: 38, segments: 3 },
    { dir: [-0.62, -0.78, 0.12], radius: 4, inner: 11, outer: 50, segments: 1 },
  ];
  return mergeParts(specs.map((s) => arm(s.dir, { ...s, facets: 8 })));
}

// ── the core claim ───────────────────────────────────────────────────────
test('bridgeClosedRimsHub: three tube rims weld into one CLOSED solid — every edge used by exactly 2 faces', () => {
  const { cage, rims } = symmetricJunction(3);
  assert.equal(nakedEdges(cage).length, 3 * 8, 'the fixture opens exactly the three hub rims and nothing else');
  const out = bridgeClosedRimsHub(cage, rims);
  assertValidCage(out.cage, 'three-way junction');
  assert.equal(nakedEdges(out.cage).length, 0, 'the junction closes every remaining opening');
  // A Y of three capped tubes is a sphere, so its Euler characteristic is 2 —
  // a genuine topological identity, not a count that happens to match.
  assert.equal(eulerCharacteristic(out.cage), 2, 'the welded result is a genus-0 closed surface');
  assert.equal(out.armCount, 3);
});

test('bridgeClosedRimsHub: the hub adds exactly two poles and 4N faces, for every N and every rim size', () => {
  for (const n of [3, 4, 5, 6]) {
    for (const facets of [3, 4, 5, 6, 7, 8]) {
      const { cage, rims } = symmetricJunction(n, { facets, segments: 2, inner: 14, radius: 4 });
      const out = bridgeClosedRimsHub(cage, rims);
      const label = `N=${n} m=${facets}`;
      assertValidCage(out.cage, label);
      assert.equal(out.cage.vertices.length, cage.vertices.length + 2, `${label}: exactly two new vertices, the two poles`);
      assert.equal(out.cage.faces.length, cage.faces.length + 4 * n, `${label}: 2N arc faces + 2N crotch triangles`);
      assert.equal(out.hubFaceIndices.length, 4 * n, `${label}: every added face is reported`);
      assert.equal(out.poleIndices.length, 2);
      assert.equal(nakedEdges(out.cage).length, 0, `${label}: closed`);
      assert.equal(eulerCharacteristic(out.cage), 2, `${label}: still a sphere`);
    }
  }
});

test('bridgeClosedRimsHub: the two poles are the only new vertices, and each caps a genuinely different side', () => {
  const { cage, rims } = symmetricJunction(4, { facets: 8, inner: 14 });
  const out = bridgeClosedRimsHub(cage, rims);
  const [xi, yi] = out.poleIndices;
  const px = out.cage.vertices[xi], py = out.cage.vertices[yi];
  // The junction plane here is Z=0 by construction, so the two poles have to
  // land on opposite sides of it — a real geometric separation, not merely
  // two distinct indices.
  assert.ok(px[2] * py[2] < 0, `the poles sit on opposite sides of the junction plane (${px[2]} vs ${py[2]})`);
  assert.ok(Math.abs(px[2]) > 1e-6 && Math.abs(py[2]) > 1e-6, 'neither pole collapsed onto the junction plane');
  // Each pole is used by exactly 2N faces: one arc face and one crotch
  // triangle per arm.
  for (const pi of out.poleIndices) {
    const uses = out.cage.faces.filter((f) => f.includes(pi)).length;
    assert.equal(uses, 2 * 4, 'each pole is the apex of one arc face and one triangle per arm');
  }
});

test('bridgeClosedRimsHub: every rim edge stops being naked and becomes a real interior edge', () => {
  const { cage, rims } = symmetricJunction(5, { facets: 6, inner: 15 });
  const out = bridgeClosedRimsHub(cage, rims);
  const counts = edgeFaceCounts(out.cage);
  const m = 6;
  for (const rim of rims) {
    for (let k = 0; k < m; k++) {
      const a = rim[k], b = rim[(k + 1) % m];
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      assert.equal(counts.get(key), 2, `rim edge ${key} — including the CLOSING one — is now shared by two faces`);
    }
  }
});

// ── the crotch pairing: a check winding cannot make ──────────────────────
test('bridgeClosedRimsHub: each crotch edge is a genuine near-contact between neighbouring arms, which no winding check could tell', () => {
  // BOTH neighbour pairings close a valid, consistently wound ring, so
  // directedEdgeReuse cannot discriminate them at all: pairing each arm's
  // next-facing vertex with the WRONG side of its neighbour passes every
  // structural check in this file while flinging each crotch edge right
  // across an arm's own body. Confirmed the hard way — an earlier version of
  // this test only asserted that the two arms joined are ANGULARLY ADJACENT,
  // and the deliberately-inverted pairing passed it 16/16, because it does
  // still join adjacent arms, just at their far sides.
  //
  // What actually settles it is LENGTH: a crotch edge is the near contact
  // between two arms, so it must be shorter than the gap between those two
  // arms' own centres. The inverted pairing measures ~23 against a 16mm
  // centre gap on this fixture; the correct one measures ~9.
  const n = 6, facets = 6, inner = 16;
  const { cage, rims } = symmetricJunction(n, { facets, inner, radius: 4 });
  const out = bridgeClosedRimsHub(cage, rims);
  const rimOf = new Map();
  const centres = rims.map((rim) => rim.reduce((a, vi) => [a[0] + cage.vertices[vi][0] / facets, a[1] + cage.vertices[vi][1] / facets, a[2] + cage.vertices[vi][2] / facets], [0, 0, 0]));
  rims.forEach((rim, i) => rim.forEach((vi) => rimOf.set(vi, i)));
  const poles = new Set(out.poleIndices);
  const triangles = out.hubFaceIndices.map((i) => out.cage.faces[i]).filter((f) => f.length === 3 && f.some((vi) => poles.has(vi)));
  assert.equal(triangles.length, 2 * n, 'one crotch triangle per arm per pole');
  let checked = 0, worst = 0;
  for (const f of triangles) {
    const ends = f.filter((vi) => !poles.has(vi));
    assert.equal(ends.length, 2, 'a crotch triangle is two rim vertices and one pole');
    const a = rimOf.get(ends[0]), b = rimOf.get(ends[1]);
    assert.ok(a !== undefined && b !== undefined, 'both ends are real rim vertices');
    assert.notEqual(a, b, 'a crotch spans two DIFFERENT arms');
    assert.equal(Math.min((a - b + n) % n, (b - a + n) % n), 1, `arms ${a} and ${b} are neighbours in the fixture's own angular order`);
    const gap = Math.hypot(...centres[a].map((c, k) => c - centres[b][k]));
    const span = Math.hypot(...out.cage.vertices[ends[0]].map((c, k) => c - out.cage.vertices[ends[1]][k]));
    assert.ok(span < gap, `crotch edge spans ${span.toFixed(2)}, which has to be under the ${gap.toFixed(2)} between the two arms' own centres`);
    worst = Math.max(worst, span / gap);
    checked++;
  }
  assert.equal(checked, 2 * n);
  assert.ok(worst < 0.75, `every crotch edge is a genuine near contact, not merely under the bound (worst ratio ${worst.toFixed(3)})`);
});

// ── it has to subdivide ──────────────────────────────────────────────────
test('bridgeClosedRimsHub: the result refines through the REAL Catmull-Clark kernel, twice', () => {
  const { cage, rims } = symmetricJunction(3, { facets: 6, inner: 14 });
  const out = bridgeClosedRimsHub(cage, rims);
  let refined = out.cage;
  let corners = refined.faces.reduce((n, f) => n + f.length, 0);
  for (const level of [1, 2]) {
    const before = corners;
    refined = subdivideCatmullClark(refined);
    assert.ok(refined.vertices.every((p) => p.every(Number.isFinite)), `level ${level}: no NaN or Infinity anywhere`);
    // Catmull-Clark emits exactly one quad per corner of the input cage —
    // n-gon hub faces and the flat caps included.
    assert.equal(refined.faces.length, before, `level ${level}: one refined quad per cage corner`);
    assert.ok(refined.faces.every((f) => f.length === 4), `level ${level}: everything is a quad after refinement`);
    assertValidCage(refined, `refined level ${level}`);
    assert.equal(eulerCharacteristic(refined), 2, `level ${level}: refinement preserves the topology`);
    corners = refined.faces.reduce((n, f) => n + f.length, 0);
  }
});

test('bridgeClosedRimsHub: an awkward junction — uneven angles, tilts, radii and lengths — is as valid as a tidy one', () => {
  const { cage, rims } = awkwardJunction();
  const out = bridgeClosedRimsHub(cage, rims);
  assertValidCage(out.cage, 'awkward junction');
  assert.equal(nakedEdges(out.cage).length, 0);
  assert.equal(eulerCharacteristic(out.cage), 2);
  const refined = subdivideCatmullClark(out.cage);
  assert.ok(refined.vertices.every((p) => p.every(Number.isFinite)), 'the awkward junction refines without NaN');
  assertValidCage(refined, 'awkward junction refined');
});

// ── ORDER INDEPENDENCE ───────────────────────────────────────────────────
// The rims arrive as an unordered SET, each one a CYCLE with no privileged
// start and no privileged direction. Three separate accidents are therefore
// available to depend on: which rim is listed first, which way round each rim
// runs, and where each rim's list happens to begin. All three are exercised
// here, on the symmetric fixture, because a symmetric junction is where every
// "which is nearer" comparison becomes an exact tie — the condition that
// produced two real bugs in this file's own sibling hub, and the one a tilted
// fixture never reaches.
function permutations(a) {
  return a.length <= 1 ? [a] : a.flatMap((x, i) => permutations([...a.slice(0, i), ...a.slice(i + 1)]).map((p) => [x, ...p]));
}
function rotated(arr, k) { return arr.slice(k).concat(arr.slice(0, k)); }
function factorial(n) { let f = 1; for (let i = 2; i <= n; i++) f *= i; return f; }

for (const n of [3, 4]) {
  for (const facets of [6, 7]) {
    test(`bridgeClosedRimsHub: a symmetric ${n}-arm junction of ${facets}-sided rims survives EVERY rim order and direction`, () => {
      const { cage, rims } = symmetricJunction(n, { facets, inner: 15, radius: 4 });
      const reference = bridgeClosedRimsHub(cage, rims);
      let tried = 0;
      for (const perm of permutations(rims.map((_, i) => i))) {
        for (let bits = 0; bits < (1 << n); bits++) {
          const variant = perm.map((i, k) => {
            const r = rims[i];
            // Rotate each rim's own starting vertex too, by a different
            // amount per slot, so "where the list begins" is varied
            // independently of order and direction.
            const rot = rotated(r, (k * 3 + bits) % facets);
            return ((bits >> k) & 1) ? [...rot].reverse() : rot;
          });
          tried++;
          const out = bridgeClosedRimsHub(cage, variant);
          const label = `perm ${perm.join('')} bits ${bits}`;
          // Not merely "it did not throw": every variant has to produce the
          // same junction, or the property would be trading a refusal for a
          // silently different result.
          assertValidCage(out.cage, label);
          assert.equal(nakedEdges(out.cage).length, 0, `${label}: closed`);
          assert.equal(eulerCharacteristic(out.cage), 2, `${label}: still a sphere`);
          assert.equal(out.cage.vertices.length, reference.cage.vertices.length, `${label}: same vertex count`);
          assert.equal(out.cage.faces.length, reference.cage.faces.length, `${label}: same face count`);
          assert.equal(out.armCount, n);
          assert.deepEqual(
            out.hubFaceIndices.map((i) => out.cage.faces[i].length).sort((a, b) => a - b),
            reference.hubFaceIndices.map((i) => reference.cage.faces[i].length).sort((a, b) => a - b),
            `${label}: the hub is built from the same shape of faces, not merely the same number`,
          );
        }
      }
      assert.equal(tried, factorial(n) * (1 << n), `all ${tried} order/direction variants were genuinely exercised`);
    });
  }
}

// ── creases ──────────────────────────────────────────────────────────────
test('bridgeClosedRimsHub: creaseWeight creases exactly the rims, closing edge included, and 0 writes no key at all', () => {
  const { cage, rims } = symmetricJunction(3, { facets: 6, inner: 14 });
  const plain = bridgeClosedRimsHub(cage, rims);
  assert.equal(Object.keys(plain.cage.creases).length, 0, 'weight 0 leaves the crease map byte-identical to a cage that never had the feature');
  const creased = bridgeClosedRimsHub(cage, rims, { creaseWeight: 3 });
  const keys = new Set(Object.keys(creased.cage.creases));
  assert.equal(keys.size, 3 * 6, 'one key per rim edge, closing edge included — 3 rims of 6 edges');
  for (const rim of rims) for (let k = 0; k < 6; k++) {
    assert.ok(keys.has(edgeKey(rim[k], rim[(k + 1) % 6])), 'every rim edge is creased');
  }
  // A creased rim has to change the limit surface, or the weight is inert.
  const a = subdivideCatmullClark(subdivideCatmullClark(plain.cage));
  const b = subdivideCatmullClark(subdivideCatmullClark(creased.cage));
  const moved = a.vertices.reduce((mx, p, i) => Math.max(mx, Math.hypot(p[0] - b.vertices[i][0], p[1] - b.vertices[i][1], p[2] - b.vertices[i][2])), 0);
  assert.ok(moved > 1e-6, `the crease genuinely changes the refined surface (max move ${moved})`);
});

// ── refusals ─────────────────────────────────────────────────────────────
test('bridgeClosedRimsHub: refuses honestly, by name, for every case it cannot do', () => {
  const { cage, rims } = symmetricJunction(3, { facets: 6, inner: 14 });
  assert.throws(() => bridgeClosedRimsHub(cage, rims.slice(0, 2)), /at least 3 closed rims/);
  assert.throws(() => bridgeClosedRimsHub(cage, 'not an array'), /must be an array/);
  assert.throws(() => bridgeClosedRimsHub(cage, [rims[0], rims[1], rims[2].slice(0, 4)]), /different vertex counts/);
  assert.throws(() => bridgeClosedRimsHub(cage, [rims[0], rims[1], [...rims[2].slice(0, 5), rims[2][0]]]), /repeats a vertex/);
  assert.throws(() => bridgeClosedRimsHub(cage, [rims[0], rims[1], rims[0]]), /share vertex/);
  // A rim that is not actually naked: cap one end flat first, then offer it.
  const capped = symmetricJunction(3, { facets: 6, inner: 14 });
  capped.cage.faces.push([...capped.rims[0]]);
  assert.throws(() => bridgeClosedRimsHub(capped.cage, capped.rims), /not a naked \(open\) edge/);
  // An OPEN run handed over as a rim: its closing edge simply is not there.
  const open = rims[0].slice(0, 5);
  assert.throws(() => bridgeClosedRimsHub(cage, [open, rims[1].slice(0, 5), rims[2].slice(0, 5)]), /not a closed ring/);
});

test('bridgeClosedRimsHub: refuses three coincident rims — no junction plane to order them around', () => {
  // Every arm starting at the SAME point puts all three rim centroids on top
  // of one another, so there is no spread to take a plane from.
  const { cage, rims } = mergeParts([[1, 0, 0], [-0.5, 0.866, 0], [-0.5, -0.866, 0]].map((d) => arm(d, { facets: 6, inner: 0, outer: 40, radius: 5 })));
  assert.throws(() => bridgeClosedRimsHub(cage, rims), /collinear about their own centre|sits on the junction axis/);
});

test('bridgeClosedRimsHub: refuses a genuinely three-dimensional junction by name rather than guessing at one', () => {
  // Six arms along +/-X, +/-Y, +/-Z. No single plane orders these, and the
  // pair the plane is taken from leaves one opposed pair lying flat IN it —
  // those two arms have no side above or below to divide between the poles.
  // A different construction is needed, and saying so is the honest answer.
  const dirs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  const { cage, rims } = mergeParts(dirs.map((d) => arm(d, { facets: 6, inner: 14, outer: 40, radius: 4 })));
  assert.throws(() => bridgeClosedRimsHub(cage, rims), /lies flat IN the junction plane/);
});

test('bridgeClosedRimsHub: an arm merely LEANING out of the junction plane still builds a valid solid — the refusal above is for the flat case only', () => {
  // Four coplanar arms plus a riser along +Z. The best-fit plane tilts enough
  // that the riser's own rim keeps a real side above and below it, so this
  // does NOT hit the refusal — and what it produces is a genuine closed
  // genus-0 manifold. Only the GEOMETRY is a best effort here (the riser's
  // rim is divided between two poles that are not really its own), which is
  // why nothing beyond structure is claimed.
  const dirs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]];
  const { cage, rims } = mergeParts(dirs.map((d) => arm(d, { facets: 6, inner: 14, outer: 40, radius: 4 })));
  const out = bridgeClosedRimsHub(cage, rims);
  assertValidCage(out.cage, 'riser junction');
  assert.equal(nakedEdges(out.cage).length, 0);
  assert.equal(eulerCharacteristic(out.cage), 2);
  assert.ok(subdivideCatmullClark(out.cage).vertices.every((p) => p.every(Number.isFinite)));
});

// ── the gap this closes: the existing bridges used to take a rim silently ─
test('the existing open-run bridges now REFUSE a closed rim instead of silently slitting it', () => {
  const { cage, rims } = symmetricJunction(3, { facets: 6, inner: 14 });
  // Before this guard: accepted, consistently wound, no repeated-vertex face,
  // and each rim's own closing edge left unattached — 18 naked edges before,
  // 9 after, where a genuine closed-rim junction leaves none.
  assert.throws(() => bridgeEdgeRunsHub(cage, rims, 1), /is a CLOSED rim, not an open run/);
  assert.throws(() => bridgeEdgeRuns(cage, rims[0], rims[1], 1), /is a CLOSED rim, not an open run/);
  // The wrapped spelling of the same mistake, refused for its own reason.
  assert.throws(() => bridgeEdgeRunsHub(cage, rims.map((r) => [...r, r[0]]), 1), /repeats a vertex/);
  // And the case that is genuinely open is untouched: dropping ONE edge of a
  // rim makes a real run again, and the run hub takes it exactly as before.
  const runs = rims.map((r) => r.slice(0, 5));
  const out = bridgeEdgeRunsHub(cage, runs, 1);
  assert.equal(out.armCount, 3);
  assert.equal(directedEdgeReuse(out.cage), 0);
});
