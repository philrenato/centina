// Relax keeps the minimum arm angle, and a skin strut stays in the body.
//
// Relax pairs a node's struts most-opposite first, so dropping a strut for its angle re-pairs the
// rest: a strut that kept its chord met one now running through toward it, 0 degrees apart. And a
// skin strut whose control points sit on two faces of a concave edge cut the corner through the
// air, 2.2 out on an L.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildInfill, windingNumber, infillBodyCheck, INFILL_DEFAULTS } from '../kernel/infill.mjs';
import { torus } from './fixtures/infill_bodies.mjs';
import { box, lshape } from './fixtures/infill_pipe.mjs';

const unit = (d) => { const l = Math.hypot(...d); return d.map((v) => v / l); };
test('Relax keeps every pair of strut ends at least the minimum arm angle apart', () => {
  for (const [name, mesh, cell, relax] of [['cube', box, 'diamond', 1], ['cube', box, 'diamond', 0.5], ['cube', box, 'kelvin', 1], ['torus', torus, 'diamond', 1]]) {
    const r = buildInfill(mesh(), { cell, relax });
    const ends = r.graph.nodes.map(() => []);
    for (const c of r.curves) {
      const P = c.curve.ctrlPts;
      ends[c.a].push(unit(P[1].map((v, i) => v - P[0][i])));
      ends[c.b].push(unit(P.at(-2).map((v, i) => v - P.at(-1)[i])));
    }
    let worst = 180;
    for (const ds of ends) for (let x = 0; x < ds.length; x++) for (let y = x + 1; y < ds.length; y++) {
      worst = Math.min(worst, Math.acos(Math.max(-1, Math.min(1, ds[x].reduce((s, v, k) => s + v * ds[y][k], 0)))) * 180 / Math.PI);
    }
    assert.ok(worst >= INFILL_DEFAULTS.minArmAngle - 1e-6, `${name} ${cell} relax ${relax}: ${worst.toFixed(2)} degrees`);
  }
});

test('a skin strut across a concave edge stays inside the body', () => {
  const mesh = lshape(), chk = infillBodyCheck(mesh);
  const res = buildInfill(mesh, {});
  const at = (c, u) => {
    const p = c.degree, U = c.knots, P = c.ctrlPts, n = P.length - 1;
    let k = p; while (k < n && U[k + 1] <= u) k++;
    const d = []; for (let j = 0; j <= p; j++) d.push(P[k - p + j].slice(0, 3));
    for (let r = 1; r <= p; r++) for (let j = p; j >= r; j--) { const i = k - p + j, a = (u - U[i]) / (U[i + p - r + 1] - U[i]); d[j] = d[j].map((v, q) => (1 - a) * d[j - 1][q] + a * v); }
    return d[p];
  };
  // The air of this L is the quarter past its inner corner: x and z both over 35.
  let out = 0, skin = 0;
  for (const c of res.curves) {
    if (c.kind !== 'skin') continue;
    skin++;
    for (let s = 1; s < 32; s++) { const p = at(c.curve, s / 32); if (p[0] > 35.02 && p[2] > 35.02 && windingNumber(chk, p) < 0.5) out++; }
  }
  assert.ok(skin > 0, 'the L carries a skin');
  assert.equal(out, 0, `${out} skin samples in the air past the inner corner`);
});
