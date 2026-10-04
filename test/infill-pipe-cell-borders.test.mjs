// Voronoi cell borders piped: no face passes through another, in the cage or its limit.
//
// A border run is two struts joined at a node into one rail with a corner. A tube station on that
// corner stood square to one leg (its faces on the inside of the bend crossed), and the transport
// across the corner could turn the next ring half round against it (a band wound round itself).
// The kernel reported those crossings and the build still said ok; here the count must be zero,
// on the cage and on the twice-subdivided limit, on the plain bodies.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildInfill } from '../kernel/infill.mjs';
import { cageCrossingFacePairs } from '../kernel/subdnetwork.mjs';
import { uvSphere, torus, slide } from './fixtures/infill_bodies.mjs';
import { pipeInfill, limitOf } from './fixtures/infill_pipe.mjs';

for (const [name, mesh] of [['sphere', uvSphere], ['torus', torus], ['slide', slide]]) {
  test(`voronoi cell borders piped on the ${name}: no face passes through another`, () => {
    const net = pipeInfill(buildInfill(mesh(), { cell: 'voronoi', boundary: 'cells' }));
    assert.ok(net.ok, net.reason);
    for (const c of net.cages) {
      assert.equal(cageCrossingFacePairs(c).length, 0, 'cage crossings');
      assert.equal(c.crossings, 0, 'reported crossings');
      assert.equal(cageCrossingFacePairs(limitOf(c)).length, 0, 'limit crossings');
    }
  });
}
