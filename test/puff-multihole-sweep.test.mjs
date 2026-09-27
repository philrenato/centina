// Every holey puff the partition hands out is closed, on layouts nobody drew by hand: seeded
// random holes (2..8, r 5..60) in an r=200 disc at three densities. An open shell shipped as
// ok:true is the defect this exists for (a cut sampled with different counts by its two cells);
// the worst 3D quad aspect is a ratchet — it may fall, never rise.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildMultiHolePuff, puffInvariants } from '../kernel/puff.mjs';

const circ = (cx, cy, r, n = 40) => Array.from({ length: n }, (_, i) => [cx + r * Math.cos(i / n * 2 * Math.PI), cy + r * Math.sin(i / n * 2 * Math.PI)]);
const mulberry = (a) => () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const outer = circ(0, 0, 200, 96);
const layout = (seed, n) => {
  const rnd = mulberry(seed); const hs = []; let tries = 0;
  while (hs.length < n && tries++ < 4000) {
    const r = 5 + rnd() * 55, a = rnd() * Math.PI * 2, d = rnd() * (200 - r - 6);
    const c = [Math.cos(a) * d, Math.sin(a) * d];
    if (hs.every((h) => Math.hypot(h.c[0] - c[0], h.c[1] - c[1]) > h.r + r + 4)) hs.push({ c, r });
  }
  return hs.length === n ? hs.map((h) => circ(h.c[0], h.c[1], h.r)) : null;
};
const worstAspect = (r) => {
  const P = r.positions, Q = r.quads; let worst = 0;
  for (let q = 0; q < Q.length; q += 4) {
    let mx = 0, mn = Infinity;
    for (let k = 0; k < 4; k++) { const a = Q[q + k], b = Q[q + (k + 1) % 4]; const e = Math.hypot(P[3 * a] - P[3 * b], P[3 * a + 1] - P[3 * b + 1], P[3 * a + 2] - P[3 * b + 2]); mx = Math.max(mx, e); mn = Math.min(mn, e); }
    worst = Math.max(worst, mx / Math.max(mn, 1e-9));
  }
  return worst;
};

test('seeded layouts: every build the partition returns is a closed genus-N shell', () => {
  const aspects = []; let built = 0, ok = 0;
  for (let seed = 1; seed <= 8; seed++) for (let n = 2; n <= 8; n++) {
    const holes = layout(seed * 7919 + n, n); if (!holes) continue;
    for (const d of [2, 5]) {
      built++;
      const r = buildMultiHolePuff(outer, holes, { density: d });
      if (!r.ok) { assert.ok(/move|apart|inward|hole/.test(r.why), `a refusal names the way out (seed ${seed} n${n} d${d}: ${r.why})`); continue; }
      ok++;
      const inv = puffInvariants(r);
      assert.ok(inv.closed && inv.nonManifold === 0 && inv.misoriented === 0 && inv.euler === 2 - 2 * n, `seed ${seed} n${n} d${d}: ${JSON.stringify(inv)}`);
      aspects.push(worstAspect(r));
    }
  }
  assert.ok(built > 90 && ok / built > 0.95, `most layouts build (${ok} of ${built})`);
  aspects.sort((a, b) => a - b);
  const p95 = aspects[Math.floor(aspects.length * 0.95)], max = aspects[aspects.length - 1];
  // Ratchet: measured 141 / 669; a change may lower these, never raise them.
  assert.ok(p95 <= 200, `worst-quad aspect p95 ${p95.toFixed(0)} <= 200`);
  assert.ok(max <= 800, `worst-quad aspect max ${max.toFixed(0)} <= 800`);
});
