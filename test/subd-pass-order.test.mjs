// One Catmull-Clark pass places its edge points in buildTopology's edge order
// and hands each creased edge's two children max(0, w - 1), keyed as edgeKey
// keys them: the numerically keyed pass topology changes nothing a caller sees.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { subdivideCatmullClark, buildTopology, computeEdgePoint, creaseWeight, edgeKey } from '../kernel/subd.mjs';

let seed = 11;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
function grid(nu, nv, wrap, creased) {
  const W = nu + (wrap ? 0 : 1);
  const vertices = [];
  for (let j = 0; j <= nv; j++) for (let i = 0; i < W; i++) vertices.push([i + rnd() * 0.3, j + rnd() * 0.3, rnd()]);
  const faces = [];
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) faces.push([j * W + i, j * W + (i + 1) % W, (j + 1) * W + (i + 1) % W, (j + 1) * W + i]);
  const creases = {};
  if (creased) for (const f of faces) if (rnd() < 0.4) creases[edgeKey(f[1], f[2])] = [0.5, 1, 2.5, 150][Math.floor(rnd() * 4)];
  return { vertices, faces, creases };
}

test('edge points follow buildTopology order; crease children follow edgeKey', () => {
  for (let t = 0; t < 12; t++) {
    const cage = grid(2 + (t % 5), 2 + (t % 3), t % 2 === 0, t % 3 !== 0);
    const out = subdivideCatmullClark(cage);
    const ctx = buildTopology(cage);
    const edges = [...ctx.edgeMap.values()];
    const base = cage.vertices.length;
    edges.forEach((e, i) => assert.deepEqual(out.vertices[base + i], computeEdgePoint(cage, ctx, e)));
    const expected = {};
    edges.forEach((e, i) => {
      const nw = Math.max(0, creaseWeight(cage, e.v0, e.v1) - 1);
      if (nw <= 0) return;
      expected[edgeKey(e.v0, base + i)] = nw;
      expected[edgeKey(e.v1, base + i)] = nw;
    });
    assert.deepEqual(out.creases, expected);
  }
});
