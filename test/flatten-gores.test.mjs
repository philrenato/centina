// cutAndFlattenGores: Cuts N splits a surface into N gores from one point and
// flattens each on its own (LSCM, then arapRefine, then balancedLayoutScale).
// Fixtures are whole revolves, tessellated at 48 so the hemisphere's rim has 48
// vertices and 4, 8, 16 and 24 gores are even (64 for 32 gores).
import test from 'node:test';
import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { makeLine, makeArc, makeCircle, revolve } from '../kernel/primitives.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { principalStrains, materialVerdict, arapRefine, balancedLayoutScale, flattenLSCM } from '../kernel/flatten.mjs';
import { cutAndFlattenGores, cutAndFlattenNurbsSurface, goreCutPaths, describeMeshTopology } from '../kernel/seam.mjs';

const DEG = Math.PI / 180;
const TABLE = JSON.parse(readFileSync(new URL('./fixtures/formability.json', import.meta.url), 'utf8'));
const CARDSTOCK = TABLE.entries.find((e) => e.id === 'mat_paper_cardstock_index').formability[0];
const RES = { uRes: 48, vRes: 48 };
const hemisphere = () => revolve(makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 50, 0, 90 * DEG, 2), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
const cylinder = () => revolve(makeLine([30, 0, 0], [30, 0, 60]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
const torus = () => revolve(makeCircle([40, 0, 0], [1, 0, 0], [0, 0, 1], 10), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);

function worst(r) {
  const s = principalStrains(r.positions, r.faces, r.uv);
  let w = 0;
  for (let f = 0; f < r.faces.length; f++) if (s.s1[f] === s.s1[f]) w = Math.max(w, s.s1[f] - 1, 1 - s.s2[f]);
  return { w, s };
}
const pieceFaces = (r, p) => r.faces.slice(p.start, p.start + p.count);

const hemi = new Map([0, 4, 8, 16, 24].map((n) => [n, cutAndFlattenGores(hemisphere(), { ...RES, cuts: n })]));

test('gores — Cuts 0 is cutAndFlattenNurbsSurface, deep-equal', () => {
  assert.deepStrictEqual(hemi.get(0), cutAndFlattenNurbsSurface(hemisphere(), RES));
});

test('gores — the hemisphere\'s worst strain falls with every count over 0, 4, 8, 16, 24, one piece per cut', () => {
  let last = Infinity;
  for (const [n, r] of hemi) {
    const { w } = worst(r);
    assert.ok(w < last, `${n} cuts: worst strain ${(100 * w).toFixed(3)}% is not below ${(100 * last).toFixed(3)}%`);
    last = w;
    if (n) {
      assert.strictEqual(r.gores.made, n);
      assert.strictEqual(r.pieces.length, n);
      assert.strictEqual(r.distortion.flippedTriangles, 0, `${n} cuts: no triangle reversed`);
    }
  }
});

test('gores — 4 cuts on the hemisphere fail cardstock', () => {
  const r = hemi.get(4);
  const v = materialVerdict(principalStrains(r.positions, r.faces, r.uv), CARDSTOCK, TABLE);
  assert.strictEqual(v.verdict, 'fails', `needs ${(100 * v.needStretch).toFixed(3)}%`);
});

test('gores — every piece is a disk, the pieces keep every triangle, and their layouts do not overlap', () => {
  for (const n of [4, 24]) {
    const r = hemi.get(n);
    assert.strictEqual(r.pieces.reduce((s, p) => s + p.count, 0), hemi.get(0).faces.length);
    const boxes = r.pieces.map((p) => {
      const fs = pieceFaces(r, p);
      assert.ok(describeMeshTopology(r.positions, fs).isDisk, `${n} cuts: a piece is not a disk`);
      const b = [Infinity, Infinity, -Infinity, -Infinity];
      for (const f of fs) for (const v of f) {
        b[0] = Math.min(b[0], r.uv[v][0]); b[1] = Math.min(b[1], r.uv[v][1]);
        b[2] = Math.max(b[2], r.uv[v][0]); b[3] = Math.max(b[3], r.uv[v][1]);
      }
      return b;
    });
    for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j];
      assert.ok(a[2] < b[0] || b[2] < a[0] || a[3] < b[1] || b[3] < a[1], `${n} cuts: pieces ${i} and ${j} overlap`);
    }
  }
});

test('gores — the cut paths form one tree from the meeting point to the rim, so no cut crosses another', () => {
  const disk = cutAndFlattenNurbsSurface(hemisphere(), RES);
  const topo = describeMeshTopology(disk.positions, disk.faces).topo;
  const { center, paths } = goreCutPaths(disk.positions, disk.faces, 24);
  assert.strictEqual(paths.length, 24);
  const verts = new Set(), edges = new Set();
  for (const p of paths) {
    assert.strictEqual(p.vertices[0], center);
    const end = p.vertices[p.vertices.length - 1];
    assert.ok(topo.boundaryVertices.has(end), 'a path ends on the rim');
    for (const v of p.vertices.slice(0, -1)) assert.ok(!topo.boundaryVertices.has(v), 'a path never runs along the rim');
    for (const v of p.vertices) verts.add(v);
    for (const k of p.edgeKeys) edges.add(k);
  }
  assert.strictEqual(edges.size, verts.size - 1, 'the union of the paths is a tree');
  assert.ok(Math.abs(disk.positions[center][2] - 50) < 1e-9, 'the gores meet at the pole');
});

test('gores — a cylinder needs none: Cuts 4 leaves its layout unchanged', () => {
  const zero = cutAndFlattenNurbsSurface(cylinder(), { uRes: 32, vRes: 32 });
  const four = cutAndFlattenGores(cylinder(), { uRes: 32, vRes: 32, cuts: 4 });
  assert.strictEqual(four.gores.made, 0);
  assert.strictEqual(four.pieces.length, 1);
  assert.deepStrictEqual(four.uv, zero.uv);
  assert.deepStrictEqual(principalStrains(four.positions, four.faces, four.uv).summary, principalStrains(zero.positions, zero.faces, zero.uv).summary);
});

test('gores — a torus (one boundary after cutToDisk) still cuts into disks', () => {
  const r = cutAndFlattenGores(torus(), { uRes: 32, vRes: 32, cuts: 4 });
  assert.strictEqual(r.gores.made, 4);
  assert.ok(r.pieces.length >= 2, `${r.pieces.length} pieces`);
  for (const p of r.pieces) assert.ok(describeMeshTopology(r.positions, pieceFaces(r, p)).isDisk);
  assert.strictEqual(r.distortion.flippedTriangles, 0);
  assert.ok(Number.isFinite(worst(r).w));
});

const cardstock = (r) => materialVerdict(principalStrains(r.positions, r.faces, r.uv), CARDSTOCK, TABLE);

test('gores — cardstock: 24 gores sit inside its 0.5% band (marginal) and 32 hold', () => {
  const v24 = cardstock(hemi.get(24));
  assert.strictEqual(v24.verdict, 'marginal', `24: ${(100 * v24.needStretch).toFixed(3)}%`);
  assert.ok(v24.needStretch < 0.005);
  const v32 = cardstock(cutAndFlattenGores(hemisphere(), { uRes: 64, vRes: 64, cuts: 32 }));
  assert.strictEqual(v32.verdict, 'ok', `32: ${(100 * v32.needStretch).toFixed(3)}%`);
});

test('gores — the refinement lowers the worst strain of every count against LSCM alone', () => {
  for (const n of [4, 8, 16, 24]) {
    const lscm = cutAndFlattenGores(hemisphere(), { ...RES, cuts: n, arapIterations: 0 });
    assert.ok(worst(hemi.get(n)).w < worst(lscm).w, `${n} cuts`);
  }
});

test('arapRefine — the energy never rises, and a developable piece stays an isometry', () => {
  const r = cutAndFlattenGores(hemisphere(), { ...RES, cuts: 8 });
  const p = r.pieces[0];
  const local = new Map(), pos = [];
  const fac = r.faces.slice(p.start, p.start + p.count).map((f) => f.map((v) => {
    if (!local.has(v)) { local.set(v, pos.length); pos.push(r.positions[v]); }
    return local.get(v);
  }));
  const lscm = flattenLSCM({ positions: pos, faces: fac });
  let uv = lscm.uv, last = Infinity;
  for (let k = 0; k < 5; k++) {
    const step = arapRefine(pos, fac, uv, { iterations: 1 });
    assert.ok(step.energyEnd <= step.energyStart * (1 + 1e-9) && step.energyEnd <= last * (1 + 1e-9), `round ${k}: ${step.energyStart} -> ${step.energyEnd}`);
    last = step.energyEnd;
    uv = step.uv;
  }
  const cyl = cutAndFlattenNurbsSurface(cylinder(), { uRes: 32, vRes: 32 });
  const refined = arapRefine(cyl.positions, cyl.faces, cyl.uv);
  assert.ok(refined.energyEnd < 1e-12 * cyl.distortion.area.total3d, `cylinder energy ${refined.energyEnd}`);
  assert.ok(Math.abs(balancedLayoutScale(cyl.positions, cyl.faces, refined.uv) - 1) < 1e-9);
});

test('gores — the grid rises to a multiple of the count, so 24 gores on a 50-step rim are even', () => {
  const r = cutAndFlattenGores(hemisphere(), { uRes: 50, vRes: 50, cuts: 24 });
  assert.strictEqual(r.tessellation.uRes, 72);
  assert.strictEqual(r.tessellation.vRes, 72);
  assert.strictEqual(r.pieces.length, 24);
  const counts = new Set(r.pieces.map((p) => p.count));
  assert.strictEqual(counts.size, 1, `every gore has the same triangles (${[...counts]})`);
  const v = cardstock(r);
  assert.ok(v.verdict !== 'fails' && v.needStretch < 0.005, `${v.verdict} at ${(100 * v.needStretch).toFixed(3)}%`);
});

test('gores — one cut is a slit, and the refinement is kept only where it lowers the worst strain', () => {
  const one = cutAndFlattenGores(hemisphere(), { ...RES, cuts: 1 });
  const lscm = cutAndFlattenGores(hemisphere(), { ...RES, cuts: 1, arapIterations: 0 });
  assert.strictEqual(one.pieces.length, 1);
  assert.ok(worst(one).w <= worst(lscm).w, `${worst(one).w} against ${worst(lscm).w}`);
});

test('gores — each cut path carries its strokes in (u, v), on the path from the pole to the rim', () => {
  const r = hemi.get(24);
  assert.strictEqual(r.gores.uvPaths.length, 24);
  const srf = hemisphere();
  for (let i = 0; i < 24; i++) {
    const strokes = r.gores.uvPaths[i], pts = r.gores.paths[i];
    assert.strictEqual(strokes.length, 1, `path ${i}: one stroke`);
    const st = strokes[0];
    assert.strictEqual(st.length, pts.length);
    st.forEach((uv, k) => {
      const q = surfacePoint(srf, uv[0], uv[1]);
      assert.ok(Math.hypot(q[0] - pts[k][0], q[1] - pts[k][1], q[2] - pts[k][2]) < 1e-9, `path ${i} station ${k}`);
    });
    assert.ok(Math.abs(pts[0][2] - 50) < 1e-9 && Math.abs(pts[pts.length - 1][2]) < 1e-9, 'pole to rim');
  }
});
