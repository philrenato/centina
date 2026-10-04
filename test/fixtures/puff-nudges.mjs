// Nearby drawings build alike. Each hole of a traced two-hole drawing (puff_two_hole_cells) moved
// up to 3% of the outline's span and turned up to 0.15 rad about its own centroid, fifty times:
// every one builds, free of crossings, its rims nearer their curves than a single-hole puff
// typically lands. Each drawing has its own test file, so the two run side by side.
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { buildMultiHolePuff, classifyPuffLoops, puffInvariants, puffLimitRimMiss } from '../../kernel/puff.mjs';
import { loopsOf, jitterHole, spanOf } from './puff_two_hole_cells.mjs';
import { singleHoleMiss } from './puff-limit-quality.mjs';

export function nudgeFamily(name, fx) {
  test(`${name}: fifty nudges all build`, () => {
    const ref = singleHoleMiss();
    const { outer, holes } = loopsOf(fx, fx.view.sy, fx.view.persp);
    const span = spanOf(outer);
    let seed = 11; const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 * 2 - 1; };
    const failed = [];
    for (let s = 0; s < 50; s++) {
      const hs = holes.map((h) => jitterHole(h, span, rnd() * 0.03, rnd() * 0.03, rnd() * 0.15));
      assert.ok(classifyPuffLoops([outer, ...hs]).ok, `${name} nudge ${s}: still an outline with two holes`);
      const r = buildMultiHolePuff(outer, hs);
      if (!r.ok || !puffInvariants(r).ok) { failed.push(`${s}:${r.reason}`); continue; }
      const miss = puffLimitRimMiss(r, [outer, ...hs]).worst;
      if (!(miss < ref)) failed.push(`${s}: rim miss ${miss.toFixed(4)}`);
    }
    assert.deepEqual(failed, [], `${name}: nudges that refused, crossed, or left their curves (single-hole ${ref.toFixed(4)})`);
  });
}

