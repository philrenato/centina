// N-way face bridge — three or more face groups joined through one junction.
//
// The claim is topological and is checked topologically: the result is a
// closed, consistently wound 2-manifold with the Euler characteristic of a
// sphere, for every arm count, segment count and straightness, with a twist
// and with rims of unequal counts; and it is geometric where the pole
// junction was not: the worst dihedral after two Catmull-Clark steps stays
// under the box's own right angle.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bridgeFacesHub, bridgeFaces, bridgeOpenings } from '../kernel/subdedit.mjs';
import { buildTopology, subdivideCatmullClark } from '../kernel/subd.mjs';

// A closed box, half-size h, its local +x along `ax`, so the face toward the
// junction is always face 4 of the six.
function box(c, h, ax) {
  const up = Math.abs(ax[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const cr = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const nz = (a) => { const L = Math.hypot(...a); return a.map((x) => x / L); };
  const ey = nz(cr(up, ax)), ez = cr(ax, ey);
  const v = [];
  for (const sz of [-1, 1]) for (const sy of [-1, 1]) for (const sx of [-1, 1]) v.push([0, 1, 2].map((k) => c[k] + h * (sx * ax[k] + sy * ey[k] + sz * ez[k])));
  return { v, f: [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]] };
}
// One box per direction, each at radius R facing the origin.
function scene(dirs, R = 60, h = 20) {
  const vertices = [], faces = [], groups = [];
  for (const d of dirs) {
    const o = vertices.length, b = box(d.map((x) => x * R), h, d);
    vertices.push(...b.v); faces.push(...b.f.map((fc) => fc.map((i) => i + o)));
    groups.push([(o / 8) * 6 + 4]);
  }
  return { cage: { vertices, faces, creases: {} }, groups };
}
const star = (n) => Array.from({ length: n }, (_, i) => [Math.cos((i / n) * Math.PI * 2), Math.sin((i / n) * Math.PI * 2), 0]);
function stats(cage) {
  const t = buildTopology(cage);
  let naked = 0, over = 0;
  for (const e of t.edgeMap.values()) { if (e.faces.length === 1) naked++; if (e.faces.length > 2) over++; }
  const seen = new Set(); let reuse = 0;
  for (const f of cage.faces) for (let i = 0; i < f.length; i++) { const k = `${f[i]}>${f[(i + 1) % f.length]}`; if (seen.has(k)) reuse++; seen.add(k); }
  return { chi: cage.vertices.length - t.edgeMap.size + cage.faces.length, naked, over, reuse };
}
const faceNormal = (v, f) => { const n = [0, 0, 0]; for (let i = 0; i < f.length; i++) { const a = v[f[i]], b = v[f[(i + 1) % f.length]]; n[0] += (a[1] - b[1]) * (a[2] + b[2]); n[1] += (a[2] - b[2]) * (a[0] + b[0]); n[2] += (a[0] - b[0]) * (a[1] + b[1]); } const L = Math.hypot(...n); return L ? n.map((x) => x / L) : null; };
function worstDihedral(cage) {
  const t = buildTopology(cage); let worst = 0;
  for (const e of t.edgeMap.values()) {
    if (e.faces.length !== 2) continue;
    const a = faceNormal(cage.vertices, cage.faces[e.faces[0]]), b = faceNormal(cage.vertices, cage.faces[e.faces[1]]);
    if (!a || !b) continue;
    worst = Math.max(worst, (Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * 180) / Math.PI);
  }
  return worst;
}
const closedSphere = (cage, what) => {
  const s = stats(cage);
  assert.deepEqual(s, { chi: 2, naked: 0, over: 0, reuse: 0 }, `${what}: a closed, consistently wound sphere (${JSON.stringify(s)})`);
};

test('bridgeFacesHub: a Y of three boxes closes into one sphere, for every segment count and straightness', () => {
  const { cage, groups } = scene(star(3));
  for (const segments of [1, 2, 3]) for (const straightness of [1, 0.5, 0]) {
    const out = bridgeFacesHub(cage, groups, { segments, straightness });
    closedSphere(out.cage, `segments ${segments} straightness ${straightness}`);
    assert.equal(out.armCount, 3);
    assert.equal(out.ringLength, 4);
    assert.equal(out.bridgeFaceIndices.length, 3 * 4 * segments, 'one band of four quads per segment per arm');
    assert.equal(out.hubFaceIndices.length, 3 + 2, 'three crotch quads and two caps');
    assert.equal(out.cage.vertices.length, cage.vertices.length + 3 * 4 * segments, 'the junction adds no vertex of its own');
    assert.doesNotThrow(() => subdivideCatmullClark(out.cage));
  }
});

test('bridgeFacesHub: an X of four and a star of five close the same way', () => {
  for (const n of [4, 5]) {
    const { cage, groups } = scene(star(n));
    const out = bridgeFacesHub(cage, groups, { segments: 2 });
    closedSphere(out.cage, `${n} arms`);
    assert.equal(out.hubFaceIndices.length, n + 2);
  }
});

test('bridgeFacesHub: the junction is smooth — worst dihedral after two subdivisions under 45° on a Y, a T and an X', () => {
  for (const [name, dirs] of [['Y', star(3)], ['T', [[1, 0, 0], [-1, 0, 0], [0, 1, 0]]], ['X', star(4)]]) {
    const { cage, groups } = scene(dirs);
    const out = bridgeFacesHub(cage, groups, { segments: 2 });
    const twice = subdivideCatmullClark(subdivideCatmullClark(out.cage));
    const worst = worstDihedral(twice);
    assert.ok(worst < 45, `${name}: worst dihedral ${worst.toFixed(1)}° after two subdivisions`);
  }
});

test('bridgeFacesHub: a spin twists one arm alone and the result is still closed; a whole-turn spin is the untwisted result', () => {
  const { cage, groups } = scene(star(3));
  const plain = bridgeFacesHub(cage, groups, { segments: 2 });
  const spun = bridgeFacesHub(cage, groups, { segments: 2, spins: [1, 0, 0] });
  closedSphere(spun.cage, 'spun');
  assert.notDeepEqual(spun.cage.faces, plain.cage.faces, 'a spin changes the arm bands');
  const full = bridgeFacesHub(cage, groups, { segments: 2, spins: [4, -4, 8] });
  assert.deepEqual(full.cage.faces, plain.cage.faces, 'a spin of a whole turn is no spin');
  assert.throws(() => bridgeFacesHub(cage, groups, { spins: [0.5, 0, 0] }), /whole number/);
});

test('bridgeFacesHub: rims of unequal counts reconcile in one band — the fine rim keeps its own vertices, the coarse rims each get M - m triangles', () => {
  const { cage, groups } = scene(star(3));
  // Group 0 is two faces of its box: the inward face and its neighbor — a
  // six-edge rim against two four-edge rims.
  const wide = [[groups[0][0], groups[0][0] - 4], groups[1], groups[2]];
  const out = bridgeFacesHub(cage, wide, { segments: 2 });
  closedSphere(out.cage, 'unequal');
  assert.equal(out.ringLength, 6);
  const tris = out.bridgeFaceIndices.filter((fi) => out.cage.faces[fi].length === 3).length;
  assert.equal(tris, 2 * (6 - 4), 'two triangles on each four-edge rim, none anywhere else');
  assert.doesNotThrow(() => subdivideCatmullClark(subdivideCatmullClark(out.cage)));
});

test('bridgeFacesHub: refusals name the case — two groups, a shared face, touching rims, centers in a line', () => {
  const { cage, groups } = scene(star(3));
  assert.throws(() => bridgeFacesHub(cage, groups.slice(0, 2)), /at least 3 face groups/);
  assert.throws(() => bridgeFacesHub(cage, [groups[0], groups[0], groups[2]]), /in openings 1 and 2/);
  const one = scene([[1, 0, 0]]);
  assert.throws(() => bridgeFacesHub(one.cage, [[0], [2], [4]]), /share vertex/);
  // Three boxes along one axis, the same face of each: the group centers are in a line.
  const vertices = [], faces = [];
  for (const cx of [-100, 0, 100]) { const o = vertices.length, b = box([cx, 0, 0], 20, [1, 0, 0]); vertices.push(...b.v); faces.push(...b.f.map((f) => f.map((v) => v + o))); }
  assert.throws(() => bridgeFacesHub({ vertices, faces, creases: {} }, [[4], [10], [16]]), /in a line/);
});

test('bridgeFacesHub: arms out of one plane take the spatial junction — the corner, a tetrahedral four, a six-way cross close and stay smooth; an X raised 45° keeps the ring', () => {
  const nz = (a) => { const L = Math.hypot(...a); return a.map((x) => x / L); };
  const cases = [
    ['corner', [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]]],
    ['tetra', [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]].map(nz)],
    ['six', [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]],
  ];
  for (const [name, dirs] of cases) {
    const { cage, groups } = scene(dirs);
    const out = bridgeFacesHub(cage, groups, { segments: 2 });
    assert.equal(out.spatial, true, `${name} is spatial`);
    closedSphere(out.cage, name);
    assert.ok(out.hubFaceIndices.every((fi) => out.cage.faces[fi].length === 3), `${name}: the hull junction is triangles`);
    const worst = worstDihedral(subdivideCatmullClark(subdivideCatmullClark(out.cage)));
    assert.ok(worst < 55, `${name}: worst dihedral ${worst.toFixed(1)}° after two subdivisions`);
  }
  const th = (45 * Math.PI) / 180;
  const raised = scene([[Math.cos(th), 0, Math.sin(th)], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]);
  const r = bridgeFacesHub(raised.cage, raised.groups, { segments: 2 });
  assert.equal(r.spatial, false, 'an X raised 45° keeps the ring junction');
  closedSphere(r.cage, 'raised 45°');
  const th2 = (60 * Math.PI) / 180;
  const raised2 = scene([[Math.cos(th2), 0, Math.sin(th2)], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]);
  const r2 = bridgeFacesHub(raised2.cage, raised2.groups, { segments: 2 });
  assert.equal(r2.spatial, true, 'an X raised 60° takes the hull');
  assert.ok(worstDihedral(subdivideCatmullClark(subdivideCatmullClark(r2.cage))) < 55, 'and does not fold');
});

test('bridgeFaces: spin turns the two-way tunnel by whole rim steps; a whole turn is the searched answer; a spin the cage already joins is refused', () => {
  const { cage, groups } = scene([[1, 0, 0], [-1, 0, 0]]);
  const plain = bridgeFaces(cage, groups[0], groups[1], 2);
  const spun = bridgeFaces(cage, groups[0], groups[1], 2, 1, 0, 1);
  assert.notDeepEqual(spun.cage.faces, plain.cage.faces);
  assert.deepEqual(bridgeFaces(cage, groups[0], groups[1], 2, 1, 0, 4).cage.faces, plain.cage.faces);
  assert.deepEqual(stats(spun.cage), { chi: 2, naked: 0, over: 0, reuse: 0 });
  assert.throws(() => bridgeFaces(cage, groups[0], groups[1], 2, 1, 0, 0.5), /whole number/);
});

test('bridgeFacesHub: reach places the ports — a small reach is a wide junction near the rims, a large one a small core; out of range refused', () => {
  const { cage, groups } = scene(star(3));
  const portSpan = (reach) => {
    const out = bridgeFacesHub(cage, groups, { segments: 1, reach });
    // The ports are the twelve vertices the arms added, in order; their spread from the origin.
    const ports = out.cage.vertices.slice(cage.vertices.length);
    return Math.max(...ports.map((v) => Math.hypot(v[0], v[1], v[2])));
  };
  const near = portSpan(0.2), far = portSpan(0.8);
  assert.ok(near > far * 2, `ports at reach 0.2 sit much further out than at 0.8 (${near.toFixed(1)} vs ${far.toFixed(1)})`);
  closedSphere(bridgeFacesHub(cage, groups, { segments: 2, reach: 0.2 }).cage, 'reach 0.2');
  closedSphere(bridgeFacesHub(cage, groups, { segments: 2, reach: 0.9 }).cage, 'reach 0.9');
  assert.deepEqual(bridgeFacesHub(cage, groups, { reach: 0.25 }).cage.vertices, bridgeFacesHub(cage, groups).cage.vertices, 'the default reach is 0.25');
  assert.throws(() => bridgeFacesHub(cage, groups, { reach: 1.2 }), /reach must be/);
});

test('bridgeOpenings: open rims and face groups mix — two rims tunnel, a rim and two face groups make a junction, on one body or across bodies', () => {
  // Three boxes in a Y; the first two have their inward faces already deleted (open rims), the third keeps its face.
  const { cage, groups } = scene(star(3));
  const open = { vertices: cage.vertices.map((v) => v.slice()), faces: cage.faces.filter((_, fi) => fi !== groups[0][0] && fi !== groups[1][0]), creases: {} };
  const t = buildTopology(open);
  const nakedOn = (box) => [...t.edgeMap.entries()].find(([, e]) => e.faces.length === 1 && e.v0 >= box * 8 && e.v0 < box * 8 + 8)[0];
  const rimA = nakedOn(0), rimB = nakedOn(1);
  // Face index of the third box's inward face after two faces were removed before it.
  const faceC = groups[2][0] - 2;
  const two = bridgeOpenings(open, [{ rim: rimA }, { rim: rimB }], { segments: 2 });
  assert.equal(two.armCount, 2);
  const st = stats(two.cage);
  assert.ok(st.naked === 0 && st.over === 0 && st.reuse === 0, `two open rims tunnel into one closed body beside an untouched third box (${JSON.stringify(st)})`);
  const three = bridgeOpenings(open, [{ rim: rimA }, { rim: rimB }, { faces: [faceC] }], { segments: 2 });
  closedSphere(three.cage, 'two rims and a face group');
  assert.equal(three.armCount, 3);
  assert.throws(() => bridgeOpenings(open, [{ rim: rimA }, { rim: rimA }]), /the same rim/);
});
