// A puff with two or more holes is one closed genus-N SuperB, and the rim is exact.
// buildMultiHolePuff is radial/medial for any number of holes (a power-diagram cell per hole, a coarse cage that
// welds one radial annulus per hole along shared medial cuts, and handles tight spacing and
// near-edge holes, rows and surrounded holes), and falls back to the grid builder when a cell cannot be meshed. Either way
// the silhouette is the line drawn and the holes go clean through. Euler = 2 - 2·holes says closed
// and genus-N in one number. Per-rim profiles let each hole be more tangent or more bulged than the
// exterior. The rim is valence-4 except, on the radial path, one vertex per medial-cut endpoint on
// the outer loop. It refuses a single hole (that is the ordinary annular puff) and degenerate input.
import { strict as assert } from 'node:assert';
import { buildMultiHolePuff, puffInvariants, puffLimitRimMiss } from '../kernel/puff.mjs';

const failures = [];
const ok = (c, m) => { if (!c) failures.push(m); };
const circle = (r, n, cx = 0, cy = 0) => { const p = []; for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; p.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]); } return p; };
const annDist = (P, x, y) => { let b = Infinity; for (let i = 0; i < P.length; i++) { const a = P[i], c = P[(i + 1) % P.length]; const ex = c[0] - a[0], ey = c[1] - a[1], L2 = ex * ex + ey * ey; let t = L2 > 0 ? ((x - a[0]) * ex + (y - a[1]) * ey) / L2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t; b = Math.min(b, Math.hypot(a[0] + t * ex - x, a[1] + t * ey - y)); } return b; };
const O = circle(40, 120);

for (const [name, holes, genus] of [
  ['two holes', [circle(6, 40, -15, 0), circle(6, 40, 15, 0)], 2],
  ['three holes', [circle(5, 32, -18, -8), circle(5, 32, 18, -8), circle(5, 32, 0, 16)], 3],
  ['three holes reversed CW', [circle(5, 32, -18, -8), circle(5, 32, 0, 16), circle(5, 32, 18, -8)], 3],
  ['three holes L-shape', [circle(4, 32, 0, 0), circle(4, 32, 16, 0), circle(4, 32, 0, 16)], 3],
  ['three holes collinear', [circle(4, 32, -16, 0), circle(4, 32, 0, 0), circle(4, 32, 16, 0)], 3],
  ['four holes square', [circle(4, 32, -15, -15), circle(4, 32, 15, -15), circle(4, 32, 15, 15), circle(4, 32, -15, 15)], 4],
  ['four holes collinear', [circle(3, 32, -24, 0), circle(3, 32, -8, 0), circle(3, 32, 8, 0), circle(3, 32, 24, 0)], 4],
  ['five holes', [circle(4, 28, -20, 0), circle(4, 28, 20, 0), circle(4, 28, 0, 20), circle(4, 28, 0, -20), circle(4, 28, 0, 0)], 5],
  ['six holes ring', (() => { const h = []; for (let i = 0; i < 6; i++) { const a = (i / 6) * Math.PI * 2; h.push(circle(4, 28, Math.cos(a) * 22, Math.sin(a) * 22)); } return h; })(), 6],
]) {
  const r = buildMultiHolePuff(O, holes);
  ok(r.ok, `${name}: builds (${r.why || ''})`);
  if (!r.ok) continue;
  const inv = puffInvariants(r);
  ok(inv.euler === 2 - 2 * genus, `${name}: Euler ${inv.euler} == ${2 - 2 * genus} (genus ${genus})`);
  ok(inv.closed && inv.nonManifold === 0 && inv.misoriented === 0 && inv.orphans === 0 && inv.nan === 0, `${name}: closed, manifold, oriented, no orphans/NaN (b${inv.boundary} nm${inv.nonManifold} mo${inv.misoriented} or${inv.orphans})`);
  ok(r.holes === holes.length && r.annular === true && r.multiHole === true, `${name}: reports holes=${r.holes}, annular, multiHole`);
  // every rim (z ~ 0) vertex lies on a drawn loop
  const loops = [O, ...holes]; let rimMax = 0, rimN = 0;
  for (let i = 0; i < r.nv; i++) { const z = r.positions[i * 3 + 2]; if (Math.abs(z) < 1e-6) { let d = Infinity; for (const L of loops) d = Math.min(d, annDist(L, r.positions[i * 3], r.positions[i * 3 + 1])); rimMax = Math.max(rimMax, d); rimN++; } }
  // The cage rim is solved past the loops so that the limit rim lands on them (puffRimLimitSolve): the cage stays within a few percent, the limit within 1%.
  ok(rimMax < 4, `${name}: every rim (z=0) vertex stays near a drawn loop (max ${rimMax.toFixed(2)}, n=${rimN})`);
  const lm = puffLimitRimMiss(r, loops, 2);
  ok(lm.worst < 0.01, `${name}: the LIMIT rim lands on the drawn loops within 1% of span (worst ${(lm.worst * 100).toFixed(2)}%, mean ${(lm.mean * 100).toFixed(2)}%)`);
  // rim vertices are valence-4
  const eset = new Set(), deg = new Array(r.nv).fill(0);
  for (let q = 0; q < r.quads.length; q += 4) { const f = [r.quads[q], r.quads[q + 1], r.quads[q + 2], r.quads[q + 3]]; for (let k = 0; k < 4; k++) { const a = f[k], b = f[(k + 1) % 4]; const u = a < b ? a + '_' + b : b + '_' + a; if (!eset.has(u)) { eset.add(u); deg[a]++; deg[b]++; } } }
  let rimIrr = 0, rimTot = 0; for (let i = 0; i < r.nv; i++) if (Math.abs(r.positions[i * 3 + 2]) < 1e-6) { rimTot++; if (deg[i] !== 4) rimIrr++; }
  // The radial path (N=2/3) carries one extraordinary rim vertex per medial-cut endpoint on the
  // outer loop (holes.length of them); the grid fallback keeps every rim vertex valence-4.
  const expIrr = r.radial ? r.feet : 0; // one irregular rim vertex per cut foot; a surrounded hole has none, a row has two per cut
  ok(rimIrr === expIrr, `${name}: rim vertices valence-4 except ${expIrr} arc/cut junction(s) (${r.radial ? 'radial' : 'grid'}, irregular ${rimIrr} of ${rimTot})`);
}

// Per-rim: a hole with a flat tangent-lens profile rises slower near its rim than a round one
{
  const holes = [circle(6, 40, -15, 0), circle(6, 40, 15, 0)];
  const round = buildMultiHolePuff(O, holes, { rimProfiles: [{ family: 'bulge', profile: 1 }, { family: 'bulge', profile: 1 }, { family: 'bulge', profile: 1 }] });
  const lens = buildMultiHolePuff(O, holes, { rimProfiles: [{ family: 'bulge', profile: 1 }, { family: 'tangent', profile: 0 }, { family: 'tangent', profile: 0 }] });
  ok(round.ok && lens.ok, 'per-rim: both build');
  if (round.ok && lens.ok) {
    const nearZ = (r, tx, ty) => { let bz = 0, bd = Infinity; for (let i = 0; i < r.nv; i++) { const z = r.positions[i * 3 + 2]; if (z <= 0) continue; const d = Math.hypot(r.positions[i * 3] - tx, r.positions[i * 3 + 1] - ty); if (d < bd) { bd = d; bz = z; } } return bz; };
    const zr = nearZ(round, -8.5, 0), zl = nearZ(lens, -8.5, 0);
    ok(zl < zr, `per-rim: tangent-lens hole rises slower near its rim than round (lens ${zl.toFixed(2)} < round ${zr.toFixed(2)})`);
  }
}

// Refusals
ok(buildMultiHolePuff(O, [circle(10, 48, 0, 0)]).ok === false, 'refuse: a single hole is the ordinary annular puff, not this');
// Tightly spaced holes (gap 4 in an r40 outer), which the grid builder refuses, build a coarse
// genus-2 radial cage — the case the radial/medial mesher exists for.
{ const r = buildMultiHolePuff(O, [circle(6, 40, -8, 0), circle(6, 40, 8, 0)]); const inv = r.ok ? puffInvariants(r) : null; ok(r.ok && r.radial === true && inv.euler === -2 && inv.closed && inv.nonManifold === 0 && inv.misoriented === 0, `build: crowded holes (gap 4) now build a coarse genus-2 radial cage (ok ${r.ok}, radial ${r.ok && r.radial}, euler ${inv && inv.euler})`); }
// A degenerate layout refuses: a hole grown until it overlaps its neighbor and pokes the rim.
{ const r = buildMultiHolePuff(O, [circle(6, 40, -15, 0), circle(26, 40, 15, 0)]); ok(r.ok === false, `refuse: an overlapping / rim-crossing hole is still refused ("${(r.why || '').slice(0, 48)}")`); }

// Underside: bottomScale negative and flip support on both radial and grid paths
{
  const testUnderside = (name, holes) => {
    for (const [mode, opts, expTopZero, expBottomZero] of [
      ['flat bottom (bs=0)', { bottomScale: 0 }, false, true],
      ['flat top (bs=-1)', { bottomScale: -1 }, true, false],
      ['flipped flat bottom (bs=0, flip)', { bottomScale: 0, flip: true }, true, false],
      ['symmetric (bs=1)', { bottomScale: 1 }, false, false],
    ]) {
      const r = buildMultiHolePuff(O, holes, opts);
      ok(r.ok, `${name} ${mode}: builds`);
      if (!r.ok) continue;
      const inv = puffInvariants(r);
      ok(inv.euler === 2 - 2 * holes.length && inv.closed && inv.nonManifold === 0 && inv.misoriented === 0,
        `${name} ${mode}: topology invariants hold (euler ${inv.euler})`);
      let maxZ = -Infinity, minZ = Infinity;
      for (let i = 0; i < r.nv; i++) {
        const z = r.positions[i * 3 + 2];
        if (z > maxZ) maxZ = z;
        if (z < minZ) minZ = z;
      }
      if (expTopZero) ok(Math.abs(maxZ) < 1e-6 && minZ < -0.1, `${name} ${mode}: flat top (maxZ ${maxZ.toFixed(4)}, minZ ${minZ.toFixed(2)})`);
      if (expBottomZero) ok(maxZ > 0.1 && Math.abs(minZ) < 1e-6, `${name} ${mode}: flat bottom (maxZ ${maxZ.toFixed(2)}, minZ ${minZ.toFixed(4)})`);
    }
  };
  testUnderside('radial 2h', [circle(6, 40, -15, 0), circle(6, 40, 15, 0)]);
  testUnderside('grid 5h', [circle(4, 28, -20, 0), circle(4, 28, 20, 0), circle(4, 28, 0, 20), circle(4, 28, 0, -20), circle(4, 28, 0, 0)]);
}

assert.equal(failures.length, 0, `${failures.length} failed:\n${failures.join('\n')}`);
