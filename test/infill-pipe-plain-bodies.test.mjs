// Pipe on an Infill builds on the plain bodies at their defaults.
//
// Each of these was refused: a joint's rims that are facets read as not (coplanar hull points on
// a cube face, or a neighbor clearing a rim's plane by less than its peak's slope), and rails
// left with no tube where the pair caps ran the radii down without settling. A body scaled to
// 0.003 rounded its radius to zero and threw. Each must build, with no face through another.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildInfill } from '../kernel/infill.mjs';
import { cageCrossingFacePairs } from '../kernel/subdnetwork.mjs';
import { torus, slide, scaleMesh } from './fixtures/infill_bodies.mjs';
import { box, pipeInfill } from './fixtures/infill_pipe.mjs';

const cases = [
  ['cube', box, 'voronoi'], ['cube', box, 'gyroid'], ['cube', box, 'octet'],
  ['torus', torus, 'diamond'], ['torus', torus, 'gyroid'],
  ['slide', slide, 'gyroid'], ['slide', slide, 'octet'],
];
for (const [name, mesh, cell] of cases) {
  test(`${cell} in the ${name} pipes at its defaults`, () => {
    const net = pipeInfill(buildInfill(mesh(), { cell }));
    assert.ok(net.ok, net.reason);
    for (const c of net.cages) {
      assert.equal(c.crossings, 0, 'reported crossings');
      assert.equal(cageCrossingFacePairs(c).length, 0, 'cage crossings');
      assert.ok(c.radii.min > 0, 'every rail keeps a tube');
    }
  });
}

test('a body scaled to 0.003 pipes at a radius of its own scale', () => {
  const res = buildInfill(scaleMesh(torus(), 0.003), {});
  assert.ok(res.ok, res.reason);
  const net = pipeInfill(res);
  assert.ok(net.ok, net.reason);
  assert.ok(net.cages[0].radii.asked > 0, 'a radius above zero');
  assert.equal(cageCrossingFacePairs(net.cages[0]).length, 0);
});
