// Fuse members carry an operation (Add, Cut, Common) and an edge (Round,
// Chamfer, Sharp), folded in list order with the first member as the base.
//
// The union-only fold is pinned by fingerprint: three fixtures' field samples,
// gradients, occupancy and wrapped cages hash to the values the union-only
// kernel produced, with the member fields absent and with every member set
// explicitly to Add / Round / follow. The new operations are measured against
// closed forms: a sampled volume for Cut and Common, the field's profile
// across a crease for the three edges.
//
//   SOLIDWRAP_PRINT_FINGERPRINTS=1 node test/solidwrap-members.test.mjs
// prints the current fingerprints instead of asserting them.
import { strict as assert } from 'node:assert';
import { createHash } from 'node:crypto';
import {
  makeSolidsField, smoothMinPoly, solidsFieldComponents, wrapSolidsToSuperbCage, wrapSolidsRefusal,
  fuseMemberFold, SOLID_WRAP_REFUSAL,
} from '../kernel/solidwrap.mjs';

const PRINT = !!process.env.SOLIDWRAP_PRINT_FINGERPRINTS;

function boxSoup(lo, hi) {
  const V = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    .map((c) => c.map((t, d) => (t ? hi[d] : lo[d])));
  const F = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
    [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]];
  const p = new Float64Array(F.length * 9);
  F.forEach((f, i) => f.forEach((vi, k) => {
    p[i * 9 + k * 3] = V[vi][0]; p[i * 9 + k * 3 + 1] = V[vi][1]; p[i * 9 + k * 3 + 2] = V[vi][2];
  }));
  return p;
}

function sphereSoup(center, radius, nu = 48, nv = 24) {
  const at = (i, j) => {
    const phi = (2 * Math.PI * i) / nu, theta = (Math.PI * j) / nv;
    return [
      center[0] + radius * Math.sin(theta) * Math.cos(phi),
      center[1] + radius * Math.sin(theta) * Math.sin(phi),
      center[2] + radius * Math.cos(theta),
    ];
  };
  const t = [];
  const push = (a, b, c) => t.push(...a, ...b, ...c);
  for (let j = 0; j < nv; j += 1) {
    for (let i = 0; i < nu; i += 1) {
      const i1 = (i + 1) % nu;
      const a = at(i, j), b = at(i1, j), c = at(i1, j + 1), d = at(i, j + 1);
      if (j === 0) push(a, d, c);
      else if (j === nv - 1) push(a, c, b);
      else { push(a, c, b); push(a, d, c); }
    }
  }
  return new Float64Array(t);
}

// Fixtures. The first two are the solidwrap twin's: two overlapping boxes,
// and a box with one face removed (signed by ray parity) beside a closed box.
const twinBoxes = () => [
  { positions: boxSoup([0, 0, 0], [100, 100, 100]) },
  { positions: boxSoup([60, 0, 0], [160, 100, 100]) },
];
const twinHoled = () => {
  const full = boxSoup([0, 0, 0], [100, 100, 100]);
  return [
    { positions: full.slice(0, full.length - 18), solid: true },
    { positions: boxSoup([60, 0, 0], [160, 100, 100]) },
  ];
};
const sphereBoxSphere = () => [
  { positions: sphereSoup([0, 0, 0], 30) },
  { positions: boxSoup([10, -15, -15], [70, 15, 15]) },
  { positions: sphereSoup([75, 0, 10], 22) },
];
const FIXTURES = [
  { name: 'two boxes', members: twinBoxes, opts: { fuse: 20, density: 4 } },
  { name: 'holed box + box', members: twinHoled, opts: { fuse: 20, density: 4 } },
  { name: 'sphere + box + sphere', members: sphereBoxSphere, opts: { fuse: 6, density: 6 } },
];

/* Deterministic probe points over the padded bounds: a low-discrepancy (R3)
   sequence, so the samples cover the blend regions without a random source. */
function probes(field, n) {
  const lo = field.bounds.lo.map((v) => v - field.padding);
  const hi = field.bounds.hi.map((v) => v + field.padding);
  const g = 1.2207440846057596;
  const a = [1 / g, 1 / (g * g), 1 / (g * g * g)];
  const out = [];
  for (let i = 1; i <= n; i += 1) {
    out.push([0, 1, 2].map((d) => lo[d] + ((0.5 + a[d] * i) % 1) * (hi[d] - lo[d])));
  }
  return out;
}

function fingerprint(members, opts) {
  const h = createHash('sha256');
  const field = makeSolidsField(members, opts);
  const pts = probes(field, 1000);
  h.update(new Uint8Array(Float64Array.from(pts.map((p) => field.distanceAt(p))).buffer));
  h.update(new Uint8Array(Float64Array.from(pts.slice(0, 60).flatMap((p) => field.gradientAt(p))).buffer));
  h.update(JSON.stringify({ bounds: field.bounds, padding: field.padding, diagonal: field.diagonal }));
  const comp = solidsFieldComponents(field);
  h.update(JSON.stringify([comp.components, comp.occupied, comp.resolution, comp.cell, comp.dims, comp.handles ?? null]));
  const r = wrapSolidsToSuperbCage(members, opts);
  h.update(JSON.stringify([r.ok, r.topology ?? null, r.facets ?? null, r.facetsH ?? null]));
  if (r.ok) {
    h.update(new Uint8Array(Float64Array.from(r.cage.vertices.flat()).buffer));
    h.update(JSON.stringify(r.cage.faces));
    h.update(new Uint8Array(Float64Array.from(r.targets.flat()).buffer));
    h.update(JSON.stringify(r.frame));
  }
  return h.digest('hex');
}

const UNION_ONLY = {
  'two boxes': 'e3e1cea6f4f0778423edd3d6d3b3d290b24f4a176cd52a9e43f3c51a90e08a18',
  'holed box + box': 'b3b0d6fd7be6742b1a2dfe8225c2a1b14df10b4e2a3e76fea926c6b18fc39b8b',
  'sphere + box + sphere': '4dcbceebc819dadcd86448993340b6babc405c5f1d2833bd5838b6ee897c2dd0',
};

// Test 1 — every member at Add / Round / follow is the union-only fold, byte for byte.
{
  for (const fx of FIXTURES) {
    const plain = fingerprint(fx.members(), fx.opts);
    if (PRINT) { process.stdout.write(`  '${fx.name}': '${plain}',\n`); continue; }
    assert.equal(plain, UNION_ONLY[fx.name], `${fx.name}: members without op/edge moved off the union-only fold`);
    assert.equal(wrapSolidsToSuperbCage(fx.members(), fx.opts).ok, true, `${fx.name}: the fixture must wrap`);
    const explicit = fx.members().map((m) => ({ ...m, op: 'add', edge: 'round', blend: null }));
    assert.equal(fingerprint(explicit, fx.opts), UNION_ONLY[fx.name], `${fx.name}: explicit Add / Round / follow moved off the union-only fold`);
  }
}
if (PRINT) process.exit(0);

/* Midpoint-grid volume of the field's inside over [lo, hi]. */
function sampledVolume(field, lo, hi, cell) {
  let n = 0;
  const p = [0, 0, 0];
  for (let x = lo[0] + cell / 2; x < hi[0]; x += cell) {
    p[0] = x;
    for (let y = lo[1] + cell / 2; y < hi[1]; y += cell) {
      p[1] = y;
      for (let z = lo[2] + cell / 2; z < hi[2]; z += cell) { p[2] = z; if (field.distanceAt(p) <= 0) n += 1; }
    }
  }
  return n * cell ** 3;
}
/* Volume of a quad cage's control polyhedron, fanned from each face's first vertex. */
function cageVolume(cage) {
  let v = 0;
  for (const f of cage.faces) {
    const a = cage.vertices[f[0]];
    for (let i = 1; i + 1 < f.length; i += 1) {
      const b = cage.vertices[f[i]], c = cage.vertices[f[i + 1]];
      v += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
    }
  }
  return Math.abs(v);
}
const zeroAlong = (f, lo, hi) => {
  let a = lo, b = hi, fa = f(a);
  assert.ok(fa * f(b) <= 0, `no zero crossing in [${lo}, ${hi}]`);
  for (let i = 0; i < 80; i += 1) {
    const m = (a + b) / 2, fm = f(m);
    if (fa * fm <= 0) b = m; else { a = m; fa = fm; }
  }
  return (a + b) / 2;
};

// Test 2 — the fold step's own algebra.
{
  for (const [a, d, k] of [[3, 7, 2], [-1, 4, 5], [0, 0, 3], [6, 6.2, 1], [-2, -3, 4]]) {
    assert.equal(fuseMemberFold(a, d, 'add', 'round', k), smoothMinPoly(a, d, k), 'Add / Round is the smooth min');
    assert.equal(fuseMemberFold(a, d, undefined, undefined, k), smoothMinPoly(a, d, k), 'an absent op and edge read as Add / Round');
    assert.equal(fuseMemberFold(a, d, 'cut', 'sharp', k), Math.max(a, -d), 'Sharp Cut is max(acc, -d)');
    assert.equal(fuseMemberFold(a, d, 'common', 'sharp', k), Math.max(a, d), 'Sharp Common is max(acc, d)');
    assert.equal(fuseMemberFold(a, d, 'add', 'sharp', k), Math.min(a, d), 'Sharp Add is min');
    for (const op of ['add', 'cut', 'common']) {
      for (const edge of ['round', 'chamfer']) {
        const sharp = fuseMemberFold(a, d, op, 'sharp', k);
        const v = fuseMemberFold(a, d, op, edge, k);
        if (op === 'add') assert.ok(v <= sharp + 1e-12, `${op}/${edge} rose above the exact union`);
        else assert.ok(v >= sharp - 1e-12, `${op}/${edge} fell below the exact ${op}`);
        assert.equal(fuseMemberFold(a, d, op, edge, 0), sharp, `${op}/${edge} at k = 0 is the exact form`);
      }
    }
  }
}

// Test 3 — a box Cut by a sphere loses the sphere's share of it.
{
  const R = 15;
  const box = { positions: boxSoup([0, 0, 0], [60, 60, 60]) };
  const ball = (op, edge, center = [30, 30, 60]) => ({ positions: sphereSoup(center, R, 96, 48), op, edge });
  const lo = [13, 13, 43], hi = [47, 47, 61];
  const cell = 0.5;
  const base = makeSolidsField([box], { fuse: 3 });
  const vBase = sampledVolume(base, lo, hi, cell);
  const vBall = sampledVolume(makeSolidsField([ball()], { fuse: 3 }), lo, [47, 47, 60], cell);
  const removed = {};
  for (const edge of ['sharp', 'round', 'chamfer']) {
    removed[edge] = vBase - sampledVolume(makeSolidsField([box, ball('cut', edge)], { fuse: 3 }), lo, hi, cell);
  }
  /* Sharp removes exactly the grid cells the ball and the box share, and that
     is the hemisphere. */
  assert.equal(removed.sharp, vBall, `Sharp Cut removed ${removed.sharp}, the ball's cells inside the box are ${vBall}`);
  const hemi = (2 / 3) * Math.PI * R ** 3;
  assert.ok(Math.abs(removed.sharp - hemi) / hemi < 0.01, `Sharp Cut removed ${removed.sharp.toFixed(0)}, the hemisphere is ${hemi.toFixed(0)}`);
  /* A rounded or beveled rim removes more than the hemisphere, and only near the rim. */
  for (const edge of ['round', 'chamfer']) {
    assert.ok(removed[edge] > removed.sharp * 1.01 && removed[edge] < removed.sharp * 1.6,
      `${edge} Cut removed ${removed[edge].toFixed(0)} against Sharp's ${removed.sharp.toFixed(0)}`);
  }

  /* The trivial case: a Cut member farther than the blend from the box changes
     nothing, down to the wrapped cage. */
  const far = ball('cut', 'round', [30, 30, 60 + R + 25]);
  assert.equal(fingerprint([box, far], { fuse: 3, density: 6 }), fingerprint([box], { fuse: 3, density: 6 }),
    'a Cut that misses the box moved its cage');

  /* Through the wrap: the dent survives into the cage and its volume falls. */
  const wBase = wrapSolidsToSuperbCage([box], { fuse: 3, density: 10 });
  const wCut = wrapSolidsToSuperbCage([box, ball('cut', 'round')], { fuse: 3, density: 10 });
  assert.equal(wBase.ok, true); assert.equal(wCut.ok, true, `the Cut box must wrap (${wCut.reason})`);
  const fall = cageVolume(wBase.cage) - cageVolume(wCut.cage);
  assert.ok(fall > hemi * 0.4 && fall < hemi * 2, `the wrapped Cut lost ${fall.toFixed(0)} of cage volume, the hemisphere is ${hemi.toFixed(0)}`);
}

// Test 4 — Common of two overlapping boxes is their intersection box.
{
  const A = { positions: boxSoup([0, 0, 0], [40, 40, 40]) };
  const B = (edge) => ({ positions: boxSoup([20, 10, -10], [60, 50, 30]), op: 'common', edge });
  const sharp = makeSolidsField([A, B('sharp')], { fuse: 3 });
  const v = sampledVolume(sharp, [-5, -5, -15], [65, 55, 45], 1);
  assert.equal(v, 20 * 30 * 30, `Sharp Common holds ${v}, the intersection box is ${20 * 30 * 30}`);
  assert.deepEqual(sharp.bounds, { lo: [0, 0, 0], hi: [40, 40, 40] }, 'a Common member set the bounds');
  for (const [p, inside] of [[[30, 25, 15], true], [[10, 25, 15], false], [[30, 5, 15], false], [[30, 25, 35], false], [[50, 25, 15], false]]) {
    assert.equal(sharp.distanceAt(p) <= 0, inside, `Common at ${p}`);
  }
  const round = sampledVolume(makeSolidsField([A, B('round')], { fuse: 3 }), [-5, -5, -15], [65, 55, 45], 1);
  assert.ok(round < v && round > v * 0.9, `Round Common holds ${round}, the box ${v}`);
  const w = wrapSolidsToSuperbCage([A, B('round')], { fuse: 3, density: 10 });
  assert.equal(w.ok, true, `Common must wrap (${w.reason})`);
  const blo = [Infinity, Infinity, Infinity], bhi = [-Infinity, -Infinity, -Infinity];
  for (const q of w.targets) for (let d = 0; d < 3; d += 1) { blo[d] = Math.min(blo[d], q[d]); bhi[d] = Math.max(bhi[d], q[d]); }
  const want = [[20, 40], [10, 40], [0, 30]];
  for (let d = 0; d < 3; d += 1) {
    assert.ok(Math.abs(blo[d] - want[d][0]) < 1.5 && Math.abs(bhi[d] - want[d][1]) < 1.5,
      `the wrapped Common spans ${blo[d].toFixed(2)}..${bhi[d].toFixed(2)} on axis ${d}, the box ${want[d]}`);
  }

  /* Two spheres of radius r, centers c apart, keep the lens π(4r + c)(2r − c)² / 12. */
  const r = 20, c = 20;
  const lens = makeSolidsField([
    { positions: sphereSoup([0, 0, 0], r, 128, 64) },
    { positions: sphereSoup([c, 0, 0], r, 128, 64), op: 'common', edge: 'sharp' },
  ], { fuse: 3 });
  const vLens = sampledVolume(lens, [0, -18, -18], [20, 18, 18], 0.5);
  const wantLens = Math.PI * (4 * r + c) * (2 * r - c) ** 2 / 12;
  assert.ok(Math.abs(vLens - wantLens) / wantLens < 0.01, `the lens holds ${vLens.toFixed(0)}, the closed form ${wantLens.toFixed(0)}`);

  /* Disjoint inputs keep nothing, and the refusal names the change. */
  const apart = makeSolidsField([A, { positions: boxSoup([80, 0, 0], [120, 40, 40]), op: 'common' }], { fuse: 3 });
  const refusal = wrapSolidsRefusal(apart);
  assert.equal(refusal && refusal.reason, SOLID_WRAP_REFUSAL.NO_VOLUME);
  assert.ok(/Add/.test(refusal.message), `the refusal must name the way out: ${refusal.message}`);
}

// Test 5 — Round, Chamfer and Sharp differ across the seam, by their own closed forms.
{
  const fuse = 5, k = 10;
  /* Two slabs meeting in an inner corner along z at x = y = 20: outside both,
     the member distances are exactly a = y - 20 and b = x - 20. */
  const slabA = { positions: boxSoup([0, 0, 0], [60, 20, 40]) };
  const slabB = (edge) => ({ positions: boxSoup([0, 0, 0], [20, 60, 40]), op: 'add', edge });
  const fields = Object.fromEntries(['sharp', 'round', 'chamfer'].map((e) => [e, makeSolidsField([slabA, slabB(e)], { fuse })]));
  const diag = (f) => (t) => f.distanceAt([20 + t, 20 + t, 20]);
  const cross = Object.fromEntries(Object.entries(fields).map(([e, f]) => [e, zeroAlong(diag(f), -3, 15)]));
  assert.ok(Math.abs(cross.sharp) < 1e-6, `Sharp's crease sits on the corner, got ${cross.sharp}`);
  assert.ok(Math.abs(cross.round - k / 4) < 1e-6, `Round's fill reaches k/4 along the bisector, got ${cross.round}`);
  assert.ok(Math.abs(cross.chamfer - k / 2) < 1e-6, `Chamfer's plane crosses the bisector at k/2, got ${cross.chamfer}`);

  /* Round and Chamfer leave the face at the same foot, k from the corner: on
     the face y = 20 the field is below zero short of the foot and zero past it. */
  for (const e of ['round', 'chamfer']) {
    const f = fields[e];
    assert.ok(f.distanceAt([20 + k - 0.05, 20, 20]) < -1e-5, `${e} has not left the face just inside its foot`);
    assert.ok(Math.abs(f.distanceAt([20 + k + 0.05, 20, 20])) < 1e-12, `${e} is back on the face just past its foot`);
  }
  assert.ok(Math.abs(fields.sharp.distanceAt([20 + k - 0.05, 20, 20])) < 1e-12, 'Sharp stays on the face up to the corner');

  /* Chamfer's face is a plane: its zero set across the band is x + y = 40 + k. */
  let worstChamfer = 0, worstRound = 0;
  for (let s = 1; s <= 9; s += 1) {
    const yc = zeroAlong((y) => fields.chamfer.distanceAt([20 + s, y, 20]), 20.000001, 35);
    worstChamfer = Math.max(worstChamfer, Math.abs((20 + s) + yc - (40 + k)));
    const yr = zeroAlong((y) => fields.round.distanceAt([20 + s, y, 20]), 20.000001, 35);
    worstRound = Math.max(worstRound, Math.abs((20 + s) + yr - (40 + k)));
  }
  assert.ok(worstChamfer < 1e-6, `Chamfer's section strays ${worstChamfer} from its plane`);
  assert.ok(worstRound > 0.5, `Round's section must be curved, it strays only ${worstRound} from the chamfer plane`);

  /* Sharp keeps a gradient discontinuity across the bisector; Round does not. */
  const jump = (f, t) => {
    const g1 = f.gradientAt([20 + t + 0.5, 20 + t - 0.5, 20]);
    const g2 = f.gradientAt([20 + t - 0.5, 20 + t + 0.5, 20]);
    return Math.hypot(g1[0] - g2[0], g1[1] - g2[1], g1[2] - g2[2]);
  };
  assert.ok(jump(fields.sharp, 3) > 1.3, `Sharp's gradient must jump across the crease, got ${jump(fields.sharp, 3)}`);
  assert.ok(jump(fields.round, 1) < 0.3, `Round's gradient must turn smoothly, got ${jump(fields.round, 1)}`);

  /* The same three on a Cut: a box with its +x half cut away has a convex
     edge at x = 30, z = 60, and the edges take material off it by the
     mirrored amounts. */
  const box = { positions: boxSoup([0, 0, 0], [60, 60, 60]) };
  const cutter = (edge) => ({ positions: boxSoup([30, -10, -10], [90, 70, 70]), op: 'cut', edge });
  for (const [edge, want] of [['sharp', 0], ['round', k / 4], ['chamfer', k / 2]]) {
    const f = makeSolidsField([box, cutter(edge)], { fuse });
    const t = zeroAlong((u) => f.distanceAt([30 - u, 30, 60 - u]), -3, 15);
    assert.ok(Math.abs(t - want) < 1e-6, `${edge} Cut's edge sits ${t} in along the bisector, expected ${want}`);
  }
}

// Test 6 — order is the fold: A − B is not B − A.
{
  const sphere = { positions: sphereSoup([0, 0, 0], 20) };
  const bar = { positions: boxSoup([0, -10, -10], [40, 10, 10]) };
  const ab = makeSolidsField([sphere, { ...bar, op: 'cut' }], { fuse: 2 });
  const ba = makeSolidsField([bar, { ...sphere, op: 'cut' }], { fuse: 2 });
  assert.ok(ab.distanceAt([-10, 0, 0]) < 0 && ba.distanceAt([-10, 0, 0]) > 0, 'sphere − bar keeps the far side of the sphere, bar − sphere does not');
  assert.ok(ab.distanceAt([30, 0, 0]) > 0 && ba.distanceAt([30, 0, 0]) < 0, 'bar − sphere keeps the bar beyond the sphere, sphere − bar does not');
  const vAB = sampledVolume(ab, [-25, -25, -25], [45, 25, 25], 1.25);
  const vBA = sampledVolume(ba, [-25, -25, -25], [45, 25, 25], 1.25);
  assert.ok(Math.abs(vAB - vBA) > 1000, `A − B and B − A hold ${vAB} and ${vBA}`);
  /* The base's op is never read: a Cut on the first member is still the base. */
  const tagged = makeSolidsField([{ ...sphere, op: 'cut' }, { ...bar, op: 'cut' }], { fuse: 2 });
  assert.equal(tagged.distanceAt([-10, 0, 0]), ab.distanceAt([-10, 0, 0]), 'the first member is the base whatever its op');
  assert.deepEqual(tagged.bounds, ab.bounds, 'the base sets the bounds whatever its op');
  assert.equal(tagged.outerParts.length, 1, 'the base is an outer part whatever its op');
}

// Test 7 — a member's own blend replaces Fuse's for that member only.
{
  const slabA = { positions: boxSoup([0, 0, 0], [60, 20, 40]) };
  const slabB = { positions: boxSoup([0, 0, 0], [20, 60, 40]) };
  const own = makeSolidsField([slabA, { ...slabB, blend: 2 }], { fuse: 5 });
  const t = zeroAlong((u) => own.distanceAt([20 + u, 20 + u, 20]), -3, 15);
  assert.ok(Math.abs(t - 1) < 1e-6, `blend 2 fills to k/4 = 1 along the bisector, got ${t}`);
  const follow = makeSolidsField([slabA, { ...slabB, blend: null }], { fuse: 5 });
  assert.equal(follow.distanceAt([23, 23, 20]), makeSolidsField([slabA, slabB], { fuse: 5 }).distanceAt([23, 23, 20]), 'a null blend follows Fuse');
}
