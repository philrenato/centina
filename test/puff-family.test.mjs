// Face quality across a family of shapes, at every density the control offers.
//
// Counting faces is not measuring a cage: a ring construction can hit every
// face-count target and still reach 29:1 within a single face, which no
// count-based check sees. So the number that governs here is the worst
// edge ratio within a face, swept over a family chosen because its members fail
// differently — round, elongated, lobed, cornered, and two whose reach and whose
// width say opposite things.
//
// And a silhouette claim is separate from a face claim. A cage can carry the
// drawn line out of slivers, or carry square faces nowhere near it.
// Both are asserted, and both are reported with their worst case named.
import { strict as assert } from 'node:assert';
import { puffCage, PUFF_PARAMS } from '../kernel/puff.mjs';
import { subdivideCatmullClark } from '../kernel/subd.mjs';
import { subdToPatches } from '../kernel/subdlimit.mjs';

const opts = { subdivide: subdivideCatmullClark };
const ring = (n, f) => { const p = []; for (let i = 0; i < n; i += 1) { const a = (i / n) * Math.PI * 2; const r = f(a); p.push(Math.cos(a) * r, Math.sin(a) * r); } return p; };
const ellipse = (k) => { const p = []; for (let i = 0; i < 64; i += 1) { const a = (i / 64) * Math.PI * 2; p.push(Math.cos(a) * k, Math.sin(a)); } return p; };
/* A fat body with a long thin tail. Built explicitly rather than from r(theta)
   because that is the only way to get a tail that is narrow: every
   smooth r(theta) spike wide enough to survive a resample is also broad. Its
   reach along the tail is 2.6x the body's and its width there is a sixth of it,
   which is the pair of facts a height law has to tell apart. */
const tailed = (() => {
  const p = [], th0 = 0.28;
  for (let i = 0; i <= 72; i += 1) { const a = th0 + (i / 72) * (Math.PI * 2 - 2 * th0); p.push(Math.cos(a), Math.sin(a)); }
  p.push(0.96, -0.28, 1.6, -0.20, 2.2, -0.13, 2.6, -0.08, 2.6, 0.08, 2.2, 0.13, 1.6, 0.20, 0.96, 0.28);
  return p;
})();
const FAMILY = [
  ['round', ring(64, () => 1)],
  ['elongated 2:1', ellipse(2)],
  ['elongated 3:1', ellipse(3)],
  ['lobed', ring(96, (a) => 1 + 0.35 * Math.cos(a * 3))],
  ['near-triangular', ring(96, (a) => 1 + 0.30 * Math.cos(a * 3))],
  ['bean', ring(64, (a) => 1 + 0.45 * Math.cos(a * 2) - 0.2 * Math.sin(a * 3))],
  ['with a stalk', ring(160, (a) => 1 + 1.6 * Math.pow(Math.max(0, Math.cos(a)), 8))],
  ['fat body, thin tail', tailed],
];
const DENSITIES = [];
for (let d = PUFF_PARAMS.density.min; d <= PUFF_PARAMS.density.max; d += 1) DENSITIES.push(d);

/* The bar, 8:1, is a bar on the cage a reader is handed at the defaults — not on every extreme the controls
   can reach, which are quoted in `PUFF_PARAMS` and are the reader's own request
   for a flatter or a thinner form. */
const ASPECT_BAR = 8;
const DEVIATION_BAR = 0.25;

const euler = (cage) => {
  const E = new Set();
  for (const f of cage.faces) for (let k = 0; k < f.length; k += 1) {
    const a = f[k], b = f[(k + 1) % f.length];
    E.add(a < b ? `${a}_${b}` : `${b}_${a}`);
  }
  return cage.vertices.length - E.size + cage.faces.length;
};
const valences = (cage) => {
  const seen = new Set(), deg = new Map();
  for (const f of cage.faces) for (let k = 0; k < f.length; k += 1) {
    const a = f[k], b = f[(k + 1) % f.length];
    const key = a < b ? `${a}_${b}` : `${b}_${a}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deg.set(a, (deg.get(a) || 0) + 1); deg.set(b, (deg.get(b) || 0) + 1);
  }
  return deg;
};
const edgeUse = (cage) => {
  const use = new Map();
  for (const f of cage.faces) for (let k = 0; k < f.length; k += 1) {
    const a = f[k], b = f[(k + 1) % f.length];
    const key = a < b ? `${a}_${b}` : `${b}_${a}`;
    use.set(key, (use.get(key) || 0) + 1);
  }
  return use;
};

for (const [name, poly] of FAMILY) {
  for (const d of DENSITIES) {
    const r = puffCage(poly, { ...opts, density: d });
    const at = `${name} at density ${d}`;
    assert.equal(r.ok, true, `${at}: ${r.why || r.reason}`);

    // Topology. Every one of these is something the NURBS conversion or the
    //    subdivider refuses, and none of them is visible in a render.
    assert.ok(r.cage.faces.every((f) => f.length === 4), `${at}: not all quads`);
    assert.equal(r.quads, 6 * d * d, `${at}: ${r.quads} quads, not 6d^2`);
    assert.equal(euler(r.cage), 2, `${at}: Euler ${euler(r.cage)} — not a closed genus-0 cage`);
    for (const [v, deg] of valences(r.cage)) assert.ok(deg >= 3, `${at}: vertex ${v} has valence ${deg}`);
    for (const [key, n] of edgeUse(r.cage)) assert.equal(n, 2, `${at}: edge ${key} is used by ${n} faces`);
    assert.equal(new Set(r.cage.vertices.map((_, i) => i)).size, r.cage.vertices.length);
    for (const f of r.cage.faces) assert.equal(new Set(f).size, 4, `${at}: a face repeats a corner`);
    assert.ok(r.cage.vertices.every((v) => v.every(Number.isFinite)), `${at}: a NaN in the cage`);
    assert.equal(Object.keys(r.cage.creases || {}).length, 0, `${at}: the rim was creased — that is a flange, not a form`);
    // And it still converts: a cage that cannot become patches cannot become a
    // surface, and an irregular rim would leave holes along the whole outline.
    const out = subdToPatches(r.cage, {});
    const list = Array.isArray(out) ? out : (out.patches || []);
    assert.equal(list.filter((p) => p.kind !== 'regular').length, 0,
      `${at}: NURBS coverage has caps — holes at the silhouette`);

    // Quality.
    assert.ok(r.aspect <= ASPECT_BAR, `${at}: worst face edge ratio ${r.aspect.toFixed(1)}:1 — a sliver, not a face you can grab`);
    assert.ok(r.worstDeviation <= DEVIATION_BAR, `${at}: silhouette locally off by ${(100 * r.worstDeviation).toFixed(1)}%`);
  }
}

/* The deviation falls with density and the face ratio does not rise with it,
   which is the pair that says the density control is worth having. A density
   ladder whose top rung is a finer cage that fits the drawing worse is of no
   use, and only a comparison across rungs can see it. */
for (const [name, poly] of FAMILY) {
  const coarse = puffCage(poly, { ...opts, density: DENSITIES[0] });
  const fine = puffCage(poly, { ...opts, density: DENSITIES[DENSITIES.length - 1] });
  assert.ok(fine.worstDeviation <= coarse.worstDeviation + 1e-9,
    `${name}: the finest cage fits the drawing worse (${(100 * fine.worstDeviation).toFixed(1)}%) than the coarsest (${(100 * coarse.worstDeviation).toFixed(1)}%)`);
}

/* The thing the width-follow control exists for, asserted on the shape that
   motivates it. A fat body with a long thin tail reaches furthest along the tail
   and is narrowest there. Scaled by reach, the tail comes out as tall as the
   body — the control is inert on the difference that matters, as if it
   ignored width entirely. Scaled by local width it comes out proportionate. */
{
  const halfWidth = (x, y) => {
    let b = Infinity;
    for (let i = 0; i < tailed.length; i += 2) {
      const j = (i + 2) % tailed.length;
      const ax = tailed[i], ay = tailed[i + 1], bx = tailed[j], by = tailed[j + 1];
      const dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
      let t = L2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / L2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      b = Math.min(b, Math.hypot(ax + dx * t - x, ay + dy * t - y));
    }
    return b;
  };
  const heightAbove = (cage, px, py) => {
    let c = cage;
    for (let k = 0; k < 3; k += 1) c = subdivideCatmullClark(c);
    let h = 0, n = 0;
    for (const v of c.vertices) if (Math.hypot(v[0] - px, v[1] - py) < 0.30) { h = Math.max(h, v[2]); n += 1; }
    assert.ok(n > 0, 'the probe found no surface above the point it asks about');
    return h;
  };
  const TAIL = [1.9, 0], BODY = [-0.45, 0];
  const wTail = halfWidth(...TAIL), wBody = halfWidth(...BODY);
  assert.ok(wTail < wBody * 0.4, `the fixture is wrong: its tail is ${wTail.toFixed(2)} wide against a body of ${wBody.toFixed(2)}`);

  const at = (f) => {
    const cage = puffCage(tailed, { ...opts, follow: f }).cage;
    const t = heightAbove(cage, ...TAIL), b = heightAbove(cage, ...BODY);
    return { t, b, ratio: t / b, tailOverWidth: t / wTail };
  };
  const off = at(0), on = at(1), mid = at(PUFF_PARAMS.follow.default);
  // The assertion is on the ratio, not on the tail's height alone. Every value
  // of `follow` that simply scaled the whole form down would pass a bare
  // "the tail got shorter" test while changing nothing about the shape.
  assert.ok(on.ratio < off.ratio * 0.7,
    `full width-follow only takes the tail from ${off.ratio.toFixed(2)} of the body's height to ${on.ratio.toFixed(2)} — the control is not acting on the difference it names`);
  assert.ok(mid.ratio < off.ratio,
    `the default does not follow width at all (tail/body height ${mid.ratio.toFixed(3)} at the default, ${off.ratio.toFixed(3)} at follow 0)`);
  // And it is a redistribution, not a shrink: the peak of the form is untouched.
  const peak = (f) => Math.max(...puffCage(tailed, { ...opts, follow: f }).cage.vertices.map((v) => v[2]));
  assert.ok(Math.abs(peak(0) - peak(1)) < 1e-12, 'width-follow moved the overall height instead of redistributing it');
  // The tail ends up proportionate to its own width rather than to its reach.
  assert.ok(on.tailOverWidth < off.tailOverWidth * 0.7,
    `at full follow the tail is still as tall relative to its width as it was (${on.tailOverWidth.toFixed(2)} half-widths against ${off.tailOverWidth.toFixed(2)})`);
}
