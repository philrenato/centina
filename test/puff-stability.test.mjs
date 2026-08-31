// A SLIDER IS DRAGGED, SO A PUFF IS REBUILT SIXTY TIMES A SECOND — AND THE ONLY
// THING WORSE THAN A SLOW REBUILD IS ONE THAT MOVES SOMETHING ELSE.
//
// Two claims, and they are easy to break and invisible when broken:
//   · DENSITY MUST NOT CHANGE HOW FAT THE FORM IS. Left inherited from the cage's
//     own construction, peak height moves 9.5% between density 2 and 3 for no
//     reason a reader could name — an odd-n ball simply has no vertex at its own
//     pole — and the density slider is secretly a shape slider.
//   · PUFFINESS MUST NOT CHANGE THE SILHOUETTE. The plan view is the drawn line,
//     and a height control that nudges it means the shape stops being where it
//     was put.
// Both hold as IDENTITIES here rather than within a tolerance, which is what
// makes them worth asserting: `puffCage` sets the peak explicitly instead of
// reading it off the ball, and the cage's (x, y) never reads a height parameter
// at all — including inside the growth solve, which runs at a fixed reference
// height for exactly this reason.
import { strict as assert } from 'node:assert';
import { puffCage, PUFF_PARAMS } from '../kernel/puff.mjs';
import { subdivideCatmullClark } from '../kernel/subd.mjs';

const opts = { subdivide: subdivideCatmullClark };
const ring = (n, f) => { const p = []; for (let i = 0; i < n; i += 1) { const a = (i / n) * Math.PI * 2; const r = f(a); p.push(Math.cos(a) * r, Math.sin(a) * r); } return p; };
const circle = ring(64, () => 1);
const bean = ring(64, (a) => 1 + 0.45 * Math.cos(a * 2) - 0.2 * Math.sin(a * 3));
const ellipse3 = (() => { const p = []; for (let i = 0; i < 64; i += 1) { const a = (i / 64) * Math.PI * 2; p.push(Math.cos(a) * 3, Math.sin(a)); } return p; })();
// A fat body with a long thin tail: the shape whose reach and whose width say
// opposite things, and the one both invariants are least likely to survive.
const tailed = (() => {
  const p = [], th0 = 0.28;
  for (let i = 0; i <= 72; i += 1) { const a = th0 + (i / 72) * (Math.PI * 2 - 2 * th0); p.push(Math.cos(a), Math.sin(a)); }
  p.push(0.96, -0.28, 1.6, -0.20, 2.2, -0.13, 2.6, -0.08, 2.6, 0.08, 2.2, 0.13, 1.6, 0.20, 0.96, 0.28);
  return p;
})();
const SHAPES = [['circle', circle], ['bean', bean], ['ellipse 3:1', ellipse3], ['tailed', tailed]];
const DENSITIES = [];
for (let d = PUFF_PARAMS.density.min; d <= PUFF_PARAMS.density.max; d += 1) DENSITIES.push(d);
const peak = (cage) => { let m = 0; for (const v of cage.vertices) m = Math.max(m, v[2]); return m; };

// ── DENSITY DOES NOT CHANGE HOW FAT THE FORM IS ──────────────────────────────
for (const [name, poly] of SHAPES) {
  const heights = DENSITIES.map((d) => peak(puffCage(poly, { ...opts, density: d }).cage));
  const spread = (Math.max(...heights) - Math.min(...heights)) / heights[0];
  assert.ok(spread < 1e-12, `${name}: peak height moves ${(100 * spread).toFixed(2)}% across the density ladder — a density control that is secretly a shape control`);
  // And the number is the one the result promises, in the drawing's own units,
  // so a readout cannot quote a height the cage was not built to.
  for (const d of DENSITIES) {
    const r = puffCage(poly, { ...opts, density: d });
    assert.ok(Math.abs(peak(r.cage) - r.height) < 1e-12 * r.height, `${name} at density ${d}: the reported height is not the cage's`);
  }
  console.log(`  ${name.padEnd(12)} peak height ${heights[0].toFixed(9)} at every density ${DENSITIES[0]}..${DENSITIES[DENSITIES.length - 1]}`);
}

// ── THE HEIGHT CONTROLS DO NOT CHANGE THE SILHOUETTE ─────────────────────────
/* ⚠ ASSERTED AS BIT EQUALITY, NOT A TOLERANCE. Catmull-Clark is an affine
   combination taken per coordinate, so a plan view that does not read z at the
   cage cannot read it at the limit either — the claim is exact or it is false,
   and a tolerance here would hide the day it stops being exact. */
const HEIGHT_CONTROLS = ['puffiness', 'follow', 'bottomScale'].filter((k) => PUFF_PARAMS[k]);
for (const [name, poly] of SHAPES) {
  for (const d of DENSITIES) {
    const base = puffCage(poly, { ...opts, density: d });
    for (const key of HEIGHT_CONTROLS) {
      const s = PUFF_PARAMS[key];
      for (const v of [s.min, s.default, (s.min + s.max) / 2, s.max]) {
        const r = puffCage(poly, { ...opts, density: d, [key]: v });
        assert.equal(r.grow, base.grow, `${name}: ${key}=${v} moved the growth solve`);
        assert.equal(r.worstDeviation, base.worstDeviation, `${name}: ${key}=${v} moved the silhouette's own measurement`);
        for (let i = 0; i < base.cage.vertices.length; i += 1) {
          assert.equal(r.cage.vertices[i][0], base.cage.vertices[i][0], `${name}: ${key}=${v} moved the plan view`);
          assert.equal(r.cage.vertices[i][1], base.cage.vertices[i][1], `${name}: ${key}=${v} moved the plan view`);
        }
      }
    }
  }
}
console.log(`  ${HEIGHT_CONTROLS.join(', ')} leave the plan view bit-identical across their whole travel, at every density`);

// ── AND THEY DO CHANGE THE HEIGHT, so the two assertions above are not both
//    passing because nothing happened at all.
for (const [name, poly] of SHAPES) {
  const a = puffCage(poly, { ...opts, puffiness: PUFF_PARAMS.puffiness.min });
  const b = puffCage(poly, { ...opts, puffiness: PUFF_PARAMS.puffiness.max });
  assert.ok(peak(b.cage) > peak(a.cage) * 3, `${name}: puffiness barely moved the height`);
}

// ── REBUILD COST ─────────────────────────────────────────────────────────────
/* Everything expensive is a property of the OUTLINE and nothing else — cleaning
   the stroke, the reach table, the width table, and per density rung a subdivided
   rim and the growth that centers it. A caller keeping one plain object between
   frames pays that once and a fraction of a millisecond per frame after. */
const timeit = (fn, n) => { const t = process.hrtime.bigint(); for (let i = 0; i < n; i += 1) fn(i); return Number(process.hrtime.bigint() - t) / 1e6 / n; };
for (const [name, poly] of [['circle', circle], ['bean', bean], ['tailed', tailed]]) {
  const cold = timeit(() => puffCage(poly, opts), 10);
  const cache = {};
  puffCage(poly, { ...opts, cache });
  const warm = timeit((i) => puffCage(poly, { ...opts, cache, puffiness: 0.2 + 0.02 * (i % 40) }), 400);
  const drag = timeit((i) => puffCage(poly, { ...opts, cache, density: DENSITIES[i % DENSITIES.length] }), 400);
  console.log(`  ${name.padEnd(8)} first build ${cold.toFixed(1)} ms | cached puffiness drag ${warm.toFixed(3)} ms/frame | cached density drag ${drag.toFixed(3)} ms/frame`);
  assert.ok(warm < 3, `${name}: a cached puffiness drag costs ${warm.toFixed(2)} ms a frame`);
  assert.ok(drag < 3, `${name}: a cached density drag costs ${drag.toFixed(2)} ms a frame`);
  assert.ok(cold < 120, `${name}: the first build costs ${cold.toFixed(0)} ms`);
}

// ── A CACHE MUST NEVER ANSWER FOR THE WRONG STROKE ───────────────────────────
/* ⚠⚠ THE ONE FAILURE A CACHE CAN ADD IS THE WORST ONE THIS MODULE HAS: returning
   the previous shape's geometry, silently, for a stroke that was never drawn.
   So a cache handed to a different outline is asserted to give exactly what no
   cache at all gives — not "something reasonable", the same numbers. */
{
  const cache = {};
  puffCage(circle, { ...opts, cache });
  for (const [name, poly] of SHAPES) {
    const fresh = puffCage(poly, opts);
    const reused = puffCage(poly, { ...opts, cache });
    assert.equal(reused.quads, fresh.quads, `${name}: a reused cache changed the cage`);
    assert.equal(reused.grow, fresh.grow, `${name}: a reused cache changed the growth`);
    assert.equal(reused.height, fresh.height, `${name}: a reused cache changed the height`);
    for (let i = 0; i < fresh.cage.vertices.length; i += 1) {
      for (let c = 0; c < 3; c += 1) {
        assert.equal(reused.cage.vertices[i][c], fresh.cage.vertices[i][c], `${name}: a reused cache returned another stroke's geometry`);
      }
    }
  }
  // A stroke of the same length that differs only in one coordinate is the case
  // a length check would miss.
  const nudged = circle.slice(); nudged[7] += 0.3;
  const a = puffCage(nudged, { ...opts, cache }), b = puffCage(nudged, opts);
  assert.equal(a.grow, b.grow);
  // A refusal caches too, and stays a refusal with the same reason.
  const bad = [0, 0, 1, 0, 0, 1, 1, 1];
  const c2 = {};
  const r1 = puffCage(bad, { ...opts, cache: c2 }), r2 = puffCage(bad, { ...opts, cache: c2 });
  assert.equal(r1.ok, false); assert.equal(r2.ok, false); assert.equal(r1.reason, r2.reason);
}
console.log('  a cache handed a different stroke rebuilds rather than answering for the old one');
console.log('puff-stability: ok');
