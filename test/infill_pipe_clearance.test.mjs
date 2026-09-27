import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildInfill } from '../kernel/infill.mjs';
import { subdPipeNetwork } from '../kernel/subdnetwork.mjs';
import { uvSphere, torus, slide } from './fixtures/infill_bodies.mjs';

// Counted here with its own edge-through-triangle test, brute force within a
// sphere-bounded neighborhood, not with the kernel's detector.
function crossingPairs(cage) {
  const V = cage.vertices, tris = [];
  for (const f of cage.faces) for (let k = 1; k + 1 < f.length; k++) {
    const p = [V[f[0]], V[f[k]], V[f[k + 1]]];
    const c = [0, 1, 2].map((d) => (p[0][d] + p[1][d] + p[2][d]) / 3);
    tris.push({ v: [f[0], f[k], f[k + 1]], p, c, r: Math.max(...p.map((q) => Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]))) });
  }
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const crs = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dt = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const hit = (p0, p1, a, b, c) => {
    const d = sub(p1, p0), e1 = sub(b, a), e2 = sub(c, a), h = crs(d, e2), det = dt(e1, h);
    if (Math.abs(det) < 1e-12) return false;
    const s = sub(p0, a), u = dt(s, h) / det; if (u <= 1e-9 || u >= 1 - 1e-9) return false;
    const q = crs(s, e1), v = dt(d, q) / det; if (v <= 1e-9 || u + v >= 1 - 1e-9) return false;
    const t = dt(e2, q) / det; return t > 1e-9 && t < 1 - 1e-9;
  };
  const order = tris.map((_, i) => i).sort((i, j) => tris[i].c[0] - tris[j].c[0]);
  const rmax = Math.max(...tris.map((t) => t.r));
  let pairs = 0;
  for (let x = 0; x < order.length; x++) {
    const A = tris[order[x]];
    for (let y = x + 1; y < order.length; y++) {
      const B = tris[order[y]];
      if (B.c[0] - A.c[0] > A.r + rmax) break;
      if (Math.hypot(A.c[0] - B.c[0], A.c[1] - B.c[1], A.c[2] - B.c[2]) > A.r + B.r) continue;
      if (A.v.some((v) => B.v.includes(v))) continue;
      if ([[0, 1], [1, 2], [2, 0]].some(([i, j]) => hit(A.p[i], A.p[j], B.p[0], B.p[1], B.p[2]) || hit(B.p[i], B.p[j], A.p[0], A.p[1], A.p[2]))) pairs++;
    }
  }
  return pairs;
}
function piped(mesh) {
  const res = buildInfill(mesh, {});
  const rails = res.curves.map((c) => ({ degree: 3, knots: c.curve.knots, ctrlPts: c.curve.ctrlPts.map((p) => [...p, 1]) }));
  const radius = Math.round(res.report.strutLength * 0.12 * 100) / 100;
  return subdPipeNetwork(rails, { radius, facets: 6, segments: 2, crease: 0, junction: 'hull', fitRadius: true, weld: true, weldFraction: 0.5, capStart: 'round', capEnd: 'round' });
}

for (const [name, mesh] of [['sphere', uvSphere()], ['torus', torus()], ['slide', slide()]]) {
  test(`an Infill piped at its defaults on the ${name} is one cage no two faces of which pass through each other`, () => {
    const net = piped(mesh);
    assert.ok(net.ok, net.reason);
    assert.equal(net.cages.length, 1);
    const cage = net.cages[0];
    assert.equal(crossingPairs(cage), 0);
    assert.equal(cage.crossings, 0);
    // A joint pulls each arm back by its own neighbors, and a short strut
    // thins only the arms that hold it: most struts keep the asked radius.
    const kept = cage.railCount - cage.radii.thinned;
    assert.ok(kept >= 0.45 * cage.railCount, `${kept} of ${cage.railCount} rails at the asked radius`);
    assert.ok(cage.meanRadius >= 0.8 * cage.radii.asked, `mean radius ${(cage.meanRadius / cage.radii.asked).toFixed(2)} of asked`);
  });
}
