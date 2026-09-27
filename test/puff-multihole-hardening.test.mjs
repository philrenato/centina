// The multi-hole puff: scale-invariant feasibility, a refusal that names the tight pair,
// and every topology invariant held across the whole density range.
//
// The "too close" refusal is a ratio (gap vs span), so a uniform scale of every loop must not
// change whether the ring builds — a shape that puffs at size 40 must puff at size 20. And the
// refusal advises by which pair is tight: two holes can be moved apart or merged, but a hole
// against the outer edge cannot be merged with the boundary — it has to move inward. Finally, the
// `density` control (2..6) refines the grid without ever breaking closure, manifoldness,
// orientation, the rim landing on the drawn loop, or valence-4 rims.
import assert from 'node:assert/strict';
import { buildMultiHolePuff, puffInvariants, puffLimitRimMiss } from '../kernel/puff.mjs';

const ok = (c, m) => assert.ok(c, m);
const circle = (r, n, cx = 0, cy = 0) => { const p = []; for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; p.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]); } return p; };
const annDist = (P, x, y) => { let b = Infinity; for (let i = 0; i < P.length; i++) { const a = P[i], c = P[(i + 1) % P.length]; const ex = c[0] - a[0], ey = c[1] - a[1], L2 = ex * ex + ey * ey; let t = L2 > 0 ? ((x - a[0]) * ex + (y - a[1]) * ey) / L2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t; b = Math.min(b, Math.hypot(a[0] + t * ex - x, a[1] + t * ey - y)); } return b; };
const centroidOfAll = (loops) => { let cx = 0, cy = 0, n = 0; for (const L of loops) for (const p of L) { cx += p[0]; cy += p[1]; n++; } return [cx / n, cy / n]; };
const scaleAbout = (L, c, s) => L.map((p) => [c[0] + (p[0] - c[0]) * s, c[1] + (p[1] - c[1]) * s]);

// Scale invariance
{
  const O = circle(40, 120), H = [circle(6, 40, -15, 0), circle(6, 40, 15, 0)];
  const full = buildMultiHolePuff(O, H);
  ok(full.ok, `scale: the r40 two-hole case builds (${full.why || ''})`);
  const c = centroidOfAll([O, ...H]);
  for (const s of [0.5, 0.25, 2.0]) {
    const Os = scaleAbout(O, c, s), Hs = H.map((h) => scaleAbout(h, c, s));
    const scaled = buildMultiHolePuff(Os, Hs);
    ok(scaled.ok === full.ok, `scale ${s}x: builds iff the original builds (ok ${scaled.ok} == ${full.ok}${scaled.ok ? '' : ` — "${(scaled.why || '').slice(0, 50)}"`})`);
    if (scaled.ok) { const inv = puffInvariants(scaled); ok(inv.euler === -2 && inv.closed && inv.nonManifold === 0 && inv.misoriented === 0, `scale ${s}x: still a closed genus-2 tube (euler ${inv.euler})`); }
  }
  // A tight 2-hole case (gap 4), which the grid mesher refuses, builds as a radial cage at any
  // scale: the radial cage needs no clearance at all.
  const crowd = [circle(6, 40, -8, 0), circle(6, 40, 8, 0)];
  const cr = buildMultiHolePuff(O, crowd);
  ok(cr.ok === true && cr.radial === true, 'scale: the crowded r40 case (gap 4) builds a radial cage the grid would refuse');
  const crS = buildMultiHolePuff(scaleAbout(O, [0, 0], 0.5), crowd.map((h) => scaleAbout(h, [0, 0], 0.5)));
  ok(crS.ok === cr.ok, 'scale 0.5x: builds iff the full-size crowded case builds (scale-invariant)');
}

// The grid fallback names the tight pair (N ≥ 4)
// With the radial primary handling N=2/3 tight and near-edge holes, the "too close" refusal is
// reached only on the grid fallback (N ≥ 4, where the radial mesher has no cell decomposition). It
// advises differently by which pair holds the smallest gap.
{
  const O = circle(40, 120);
  // Hole-to-hole: four holes crammed at the center — the tight pair is hole-to-hole.
  const h2h = buildMultiHolePuff(O, [circle(9, 40, -6, 0), circle(9, 40, 6, 0), circle(9, 40, 0, 10), circle(9, 40, 0, -10)]);
  ok(h2h.ok === false && /apart|merge/i.test(h2h.why || '') && !/inward/i.test(h2h.why || ''),
    `refusal (hole-to-hole, N=4 grid): advise moving apart or merging, not inward — "${(h2h.why || '').slice(0, 70)}"`);

  // Hole-to-outer-edge: four holes, three comfortably placed and one pressed to the rim (gap 1).
  const h2r = buildMultiHolePuff(O, [circle(6, 40, -15, -15), circle(6, 40, 15, -15), circle(6, 40, -15, 15), circle(6, 40, 33, 15)]);
  ok(h2r.ok === false && /inward|outer edge/i.test(h2r.why || '') && !/merge/i.test(h2r.why || ''),
    `refusal (hole-to-rim, N=4 grid): advise moving the hole inward, never merging with the boundary — "${(h2r.why || '').slice(0, 70)}"`);
}

// Density: invariants hold at every density, coarse at 2 and finer above
{
  const battery = [
    ['2 holes', circle(40, 120), [circle(6, 40, -15, 0), circle(6, 40, 15, 0)]],
    ['3 holes', circle(40, 120), [circle(5, 32, -18, -8), circle(5, 32, 18, -8), circle(5, 32, 0, 16)]],
  ];
  for (const [name, O, H] of battery) {
    let prevFaces = 0, monotone = true, allInv = true, why = '';
    for (let d = 2; d <= 6; d++) {
      const r = buildMultiHolePuff(O, H, { density: d });
      if (!r.ok) { allInv = false; why = `d${d} refused: ${r.why}`; break; }
      const inv = puffInvariants(r);
      // rim on loop + valence-4
      const eset = new Set(), deg = new Array(r.nv).fill(0);
      for (let q = 0; q < r.quads.length; q += 4) { const f = [r.quads[q], r.quads[q + 1], r.quads[q + 2], r.quads[q + 3]]; for (let k = 0; k < 4; k++) { const a = f[k], b = f[(k + 1) % 4]; const u = a < b ? a + '_' + b : b + '_' + a; if (!eset.has(u)) { eset.add(u); deg[a]++; deg[b]++; } } }
      let rimMax = 0, rimIrr = 0; const loops = [O, ...H];
      for (let i = 0; i < r.nv; i++) { if (Math.abs(r.positions[i * 3 + 2]) < 1e-6) { let dd = Infinity; for (const L of loops) dd = Math.min(dd, annDist(L, r.positions[i * 3], r.positions[i * 3 + 1])); rimMax = Math.max(rimMax, dd); if (deg[i] !== 4) rimIrr++; } }
      // radial (N=2/3) carries H.length extraordinary rim vertices at the medial-cut endpoints; grid keeps 0.
      const expIrr = r.radial ? H.length : 0;
      const lm = puffLimitRimMiss(r, loops, 2);
      const good = inv.euler === 2 - 2 * H.length && inv.closed && inv.nonManifold === 0 && inv.misoriented === 0 && inv.orphans === 0 && inv.nan === 0 && rimMax < 4 && lm.worst < 0.01 && rimIrr === expIrr;
      if (!good) { allInv = false; why = `d${d}: euler ${inv.euler} closed ${inv.closed} nm ${inv.nonManifold} mo ${inv.misoriented} rimMax ${rimMax.toFixed(2)} limitMiss ${(lm.worst * 100).toFixed(2)}% rimIrr ${rimIrr}/${expIrr}`; }
      const faces = r.quads.length / 4;
      if (d > 2 && !(faces >= prevFaces)) monotone = false;
      prevFaces = faces;
    }
    ok(allInv, `${name}: every invariant (euler/closed/manifold/oriented/rim-on-loop/valence-4) holds at density 2..6 ${why}`);
    ok(monotone, `${name}: face count does not decrease as density rises (detail is added, never removed)`);
  }
}
