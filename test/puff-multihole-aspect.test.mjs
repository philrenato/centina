// The multi-hole puff's worst quad, bounded on both meshers.
// Grid path (N≥4): the worst quad is a rim-collar quad — the band bridging the staircase ring (at
// interior height) to the rim ring (on the drawn loop, z=0). Left as one band its rails span the
// whole rise while its ring edges stay short (tall thin quads, 14:1–30:1). A graded intermediate
// ring (collarFH) splits the rise and drops the worst below 10:1. `collarFH: []` builds the
// single-band geometry, which is the comparison measured here.
// Radial path (N=2/3): no collar — the worst quad is a structural extraordinary region at the
// arc/cut seam. An even angular sampling (a cut sampled coarser than the arc, and a uniform-angle
// hole ring) tames it to ≈ 16:1 at the coarse default (from ~40-70:1 for the naive weld), median ≈2.
// Either way topology is untouched: Euler, closure, manifoldness, orientation, the rim on the drawn
// loop, and valence-4 rims (except the radial junctions) all hold.
import assert from 'node:assert/strict';
import { buildMultiHolePuff, puffInvariants, puffLimitRimMiss } from '../kernel/puff.mjs';

const ok = (c, m) => assert.ok(c, m);

const circle = (r, n, cx = 0, cy = 0) => { const p = []; for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; p.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]); } return p; };
const ellipse = (rx, ry, n, cx = 0, cy = 0, rot = 0) => { const p = []; for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; const x = Math.cos(a) * rx, y = Math.sin(a) * ry; p.push([cx + x * Math.cos(rot) - y * Math.sin(rot), cy + x * Math.sin(rot) + y * Math.cos(rot)]); } return p; };
const wavy = (rB, amp, lb, n, cx = 0, cy = 0) => { const p = []; for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; const r = rB + amp * Math.cos(lb * a); p.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]); } return p; };
const annDist = (P, x, y) => { let b = Infinity; for (let i = 0; i < P.length; i++) { const a = P[i], c = P[(i + 1) % P.length]; const ex = c[0] - a[0], ey = c[1] - a[1], L2 = ex * ex + ey * ey; let t = L2 > 0 ? ((x - a[0]) * ex + (y - a[1]) * ey) / L2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t; b = Math.min(b, Math.hypot(a[0] + t * ex - x, a[1] + t * ey - y)); } return b; };

// worst quad aspect (longest / shortest edge) in 3D, and the 95th-percentile aspect.
function aspects(r) {
  const P = (i) => [r.positions[i * 3], r.positions[i * 3 + 1], r.positions[i * 3 + 2]];
  const list = [];
  for (let q = 0; q < r.quads.length; q += 4) {
    const f = [r.quads[q], r.quads[q + 1], r.quads[q + 2], r.quads[q + 3]].map(P);
    const L = [0, 1, 2, 3].map((k) => { const a = f[k], b = f[(k + 1) % 4]; return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); });
    const mn = Math.min(...L), mx = Math.max(...L);
    if (mn > 1e-12) list.push(mx / mn);
  }
  list.sort((a, b) => a - b);
  return { worst: list[list.length - 1], p95: list[Math.min(list.length - 1, Math.floor(0.95 * list.length))], med: list[Math.floor(0.5 * list.length)] };
}

// The full invariant battery the aspect work must not regress. Returns a per-check breakdown.
function invBattery(r, outer, holes) {
  const inv = puffInvariants(r);
  const loops = [outer, ...holes];
  // every z ~ 0 vertex sits on a drawn loop, and is valence-4
  const eset = new Set(), deg = new Array(r.nv).fill(0);
  for (let q = 0; q < r.quads.length; q += 4) { const f = [r.quads[q], r.quads[q + 1], r.quads[q + 2], r.quads[q + 3]]; for (let k = 0; k < 4; k++) { const a = f[k], b = f[(k + 1) % 4]; const u = a < b ? a + '_' + b : b + '_' + a; if (!eset.has(u)) { eset.add(u); deg[a]++; deg[b]++; } } }
  let rimMax = 0, rimIrr = 0, rimN = 0;
  for (let i = 0; i < r.nv; i++) { if (Math.abs(r.positions[i * 3 + 2]) < 1e-6) { rimN++; let d = Infinity; for (const L of loops) d = Math.min(d, annDist(L, r.positions[i * 3], r.positions[i * 3 + 1])); rimMax = Math.max(rimMax, d); if (deg[i] !== 4) rimIrr++; } }
  return {
    euler: inv.euler, expEuler: 2 - 2 * holes.length,
    closed: inv.closed, nonManifold: inv.nonManifold, misoriented: inv.misoriented,
    orphans: inv.orphans, nan: inv.nan, repeated: inv.repeated,
    rimMax, rimIrr, rimN,
  };
}

// The battery: 2/3/5 holes, concave outer, elongated body, elliptical holes, and the densest case
// the cage will build (eight holes near the spacing floor), where the rim ring is finest.
const H8 = []; for (let i = 0; i < 8; i++) { const a = (i / 8) * Math.PI * 2; H8.push(circle(7, 40, Math.cos(a) * 70, Math.sin(a) * 70)); }
const fixtures = [
  ['2 holes',            circle(40, 120), [circle(6, 40, -15, 0), circle(6, 40, 15, 0)]],
  ['3 holes',            circle(40, 120), [circle(5, 32, -18, -8), circle(5, 32, 18, -8), circle(5, 32, 0, 16)]],
  ['5 holes',            circle(40, 120), [circle(4, 28, -20, 0), circle(4, 28, 20, 0), circle(4, 28, 0, 20), circle(4, 28, 0, -20), circle(4, 28, 0, 0)]],
  ['concave outer 3h',   wavy(44, 7, 6, 160), [circle(5, 40, -16, -6), circle(5, 40, 16, -6), circle(5, 40, 0, 14)]],
  ['elongated 140x36',   ellipse(70, 18, 160), [circle(6, 40, -28, 0), circle(6, 40, 28, 0)]],
  ['elliptical holes',   circle(40, 120), [ellipse(12, 5, 60, -16, 0, 0.6), ellipse(9, 6, 60, 16, 0, 0.35)]],
  ['thin 160x28',        ellipse(80, 14, 200), [circle(3.5, 36, -30, 0), circle(3.5, 36, 30, 0)]],
  ['eight holes dense',  circle(110, 240), H8],
];

// Worst-quad bounds differ by path. The grid path keeps the < 10:1 collar guarantee, and a graded
// intermediate ring (collarFH) beats the single-band collar (collarFH: []). The radial path
// (N=2/3) has no collar; its worst quad is a structural extraordinary region at the arc/cut seam,
// tamed to ≈ 16:1 at the coarse default by an even angular sampling (a coarser cut than the arc, and
// a uniform-angle hole ring) — well down from the ~40-70:1 of the naive weld, with a median near 2.
const RADIAL_ASPECT_CEIL = 16;
for (const [name, outer, holes] of fixtures) {
  const after = buildMultiHolePuff(outer, holes);
  ok(after.ok, `${name}: builds ok (${after.why || ''})`);
  if (!after.ok) continue;
  const aa = aspects(after), ai = invBattery(after, outer, holes);
  const path = after.radial ? 'rad' : 'grid';
  const expIrr = after.radial ? after.feet : 0; // one irregular rim vertex per cut foot; a surrounded hole has none, a row has two per cut

  if (after.radial) {
    ok(aa.worst < RADIAL_ASPECT_CEIL, `${name} (radial): worst aspect ${aa.worst.toFixed(2)}:1 < ${RADIAL_ASPECT_CEIL}:1`);
    // Median stays low; the bound allows for extreme-anisotropy bodies (a thin/elongated outer makes
    // every cell elongated regardless of mesher — the median tracks the body, not a seam defect).
    ok(aa.med < 8, `${name} (radial): median aspect ${aa.med.toFixed(2)}:1 stays low`);
  } else {
    // Grid path: the collar guarantee holds, and the graded ring beats the single band.
    const before = buildMultiHolePuff(outer, holes, { collarFH: [] });
    const ab = aspects(before);
    ok(aa.worst < 10, `${name} (grid): worst aspect ${aa.worst.toFixed(2)}:1 < 10:1`);
    ok(aa.worst < ab.worst - 1e-6, `${name} (grid): worst aspect improved by the graded ring (${ab.worst.toFixed(2)} -> ${aa.worst.toFixed(2)})`);
    ok(aa.p95 <= ab.p95 + 1e-6, `${name} (grid): 95th-percentile aspect did not regress (${ab.p95.toFixed(2)} -> ${aa.p95.toFixed(2)})`);
  }

  // Topology on either path: the invariants hold, with the radial junction allowance.
  ok(ai.euler === ai.expEuler, `${name}: Euler ${ai.euler} == ${ai.expEuler} (genus ${holes.length})`);
  ok(ai.closed && ai.nonManifold === 0 && ai.misoriented === 0 && ai.orphans === 0 && ai.nan === 0 && ai.repeated === 0, `${name}: closed, manifold, oriented, no orphans/NaN (b${ai.closed ? 0 : 1} nm${ai.nonManifold} mo${ai.misoriented} or${ai.orphans} nan${ai.nan})`);
  { const lm = puffLimitRimMiss(after, [outer, ...holes], 2);
    ok(ai.rimMax < 0.06 * lm.span, `${name}: every z=0 rim vertex stays near a drawn loop — the cage is solved past it (max ${ai.rimMax.toFixed(2)} of span ${lm.span.toFixed(0)}, n=${ai.rimN})`);
    // Measured at density 2 after the rim solve: 0.3–1.9% (elongated 140×36 the worst); without the solve, 5–7%.
    ok(lm.worst < 0.025, `${name}: the limit rim lands on the drawn loops within 2.5% of span (worst ${(lm.worst * 100).toFixed(2)}%)`); }
  ok(ai.rimIrr === expIrr, `${name}: rim vertices valence-4 except ${expIrr} junction(s) (${path}, irregular ${ai.rimIrr} of ${ai.rimN})`);
  ok(after.holes === holes.length && after.annular === true && after.multiHole === true, `${name}: reports holes=${after.holes}, annular, multiHole`);
}

// Crowded holes (gap 4), which the grid mesher refuses, build on the radial path.
// A single hole refuses (that is the annular puff), and well-spaced holes build.
{
  const O = circle(40, 120);
  const r = buildMultiHolePuff(O, [circle(6, 40, -8, 0), circle(6, 40, 8, 0)]);
  ok(r.ok === true && r.radial === true, `crowded holes (gap 4) build a coarse radial cage (ok=${r.ok}, radial=${r.ok && r.radial})`);
  ok(buildMultiHolePuff(O, [circle(10, 48, 0, 0)]).ok === false, 'single hole refuses (not-multi)');
  ok(buildMultiHolePuff(O, [circle(6, 40, -15, 0), circle(6, 40, 15, 0)]).ok === true, 'well-spaced holes build');
}
