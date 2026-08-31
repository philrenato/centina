// WHAT A PUFF DOES WITH A STROKE THAT IS NOT A SHAPE.
//
// Two rules, and the second is the one that costs: NOTHING THROWS, and every
// refusal NAMES WHAT A READER WOULD CHANGE. A thrown exception reaches a person
// as a dead tool; a refusal that says only "invalid" reaches them as a dead tool
// with a sentence attached. Both are the same failure with different manners.
import { strict as assert } from 'node:assert';
import { puffCage } from '../kernel/puff.mjs';
import { subdivideCatmullClark } from '../kernel/subd.mjs';

const opts = { subdivide: subdivideCatmullClark };
const ring = (n, f) => { const p = []; for (let i = 0; i < n; i += 1) { const a = (i / n) * Math.PI * 2; const r = f(a); p.push(Math.cos(a) * r, Math.sin(a) * r); } return p; };
const circle = ring(64, () => 1);
const bean = ring(64, (a) => 1 + 0.45 * Math.cos(a * 2) - 0.2 * Math.sin(a * 3));

/* An ACTION, not a diagnosis. Every refusal has to end somewhere a hand can go:
   draw it differently, draw it wider, pass the missing argument. */
const NAMES_A_CHANGE = /\b(draw|pass|widen|wider)\b/i;
const call = (label, outline, o = opts) => {
  let r;
  try { r = puffCage(outline, o); } catch (e) { assert.fail(`${label}: threw ${e && e.message}`); }
  assert.equal(typeof r, 'object', `${label}: returned no result`);
  assert.equal(typeof r.ok, 'boolean', `${label}: no ok flag`);
  if (!r.ok) {
    assert.ok(typeof r.reason === 'string' && r.reason.length > 0, `${label}: refused without a reason code`);
    assert.ok(typeof r.why === 'string' && r.why.length > 10, `${label}: refused without saying why`);
    assert.ok(NAMES_A_CHANGE.test(r.why), `${label}: "${r.why}" diagnoses without naming a change`);
    assert.equal(r.cage, undefined, `${label}: a refusal carried geometry anyway`);
  }
  return r;
};

// ── STROKES THAT ARE NOT SHAPES ──────────────────────────────────────────────
const figure8 = (() => { const p = []; for (let i = 0; i < 80; i += 1) { const t = (i / 80) * Math.PI * 2; p.push(Math.sin(2 * t), Math.sin(t)); } return p; })();
const outAndBack = (() => { const p = []; for (let i = 0; i <= 20; i += 1) p.push(i * 0.1, 0); for (let i = 19; i > 0; i -= 1) p.push(i * 0.1, 0); return p; })();
const nearlyStraight = (() => {
  const p = [];
  for (let i = 0; i <= 40; i += 1) p.push(i * 0.05, 0.0004 * Math.sin(i));
  for (let i = 39; i > 0; i -= 1) p.push(i * 0.05, -0.0004);
  return p;
})();
const openArc = (() => { const p = []; for (let i = 0; i <= 40; i += 1) { const a = (i / 40) * Math.PI * 0.9; p.push(Math.cos(a), Math.sin(a)); } return p; })();

const refusals = [
  ['self-crossing (figure eight)', figure8],
  ['encloses no area (out and back)', outAndBack],
  ['a nearly straight stroke', nearlyStraight],
  ['an open arc', openArc],
  ['two points', [0, 0, 1, 1]],
  ['every point identical', [1, 1, 1, 1, 1, 1, 1, 1]],
  ['a NaN coordinate', [0, 0, 1, 0, NaN, 1]],
  ['an Infinity coordinate', [0, 0, 1, 0, Infinity, 1]],
  ['nothing at all', []],
  ['null', null],
  ['undefined', undefined],
];
for (const [label, outline] of refusals) {
  const r = call(label, outline);
  assert.equal(r.ok, false, `${label}: accepted — a stroke that is not a shape came back as geometry`);
  console.log(`  ${label.padEnd(32)} [${r.reason}] ${r.why}`);
}
// A missing subdivider is the caller's mistake and is named as the caller's own change.
{
  const r = call('no subdivider', circle, {});
  assert.equal(r.ok, false);
  assert.equal(r.reason, 'nosubdiv');
  assert.ok(/opts\.subdivide/.test(r.why), 'the refusal must name the argument to pass');
  call('no options at all', circle, undefined);
}

// ── STROKES THAT ARE SHAPES AND LOOK LIKE MISTAKES ───────────────────────────
/* ⚠ A POINTER THAT REPORTS THE SAME POSITION TWICE IS ORDINARY, AND IT USED TO
   BE REFUSED BY NAME: a zero-length segment lies on top of its neighbor, and the
   self-intersection test reads that as the outline crossing itself. The reader
   was told their circle crossed itself. */
{
  const doubled = [];
  for (let i = 0; i < 40; i += 1) {
    const a = (i / 40) * Math.PI * 2;
    doubled.push(Math.cos(a), Math.sin(a), Math.cos(a), Math.sin(a));
  }
  const r = call('duplicate consecutive points', doubled);
  assert.equal(r.ok, true, `a stroke with repeated samples must build: ${r.why}`);
  const plain = puffCage(circle, opts);
  assert.ok(Math.abs(r.worstDeviation - plain.worstDeviation) < 0.02,
    'a repeated sample changed the shape rather than being ignored');
  // Repeated FIRST and LAST point too — a stroke explicitly closed by the caller.
  const closed = circle.concat([circle[0], circle[1]]);
  assert.equal(call('explicitly closed stroke', closed).ok, true);
}

// ── SCALE INVARIANCE ─────────────────────────────────────────────────────────
/* ⚠⚠ THE SAME DRAWING AT TWO ZOOMS MUST BE THE SAME CAGE. Every threshold on the
   path from stroke to cage is relative — the closure gap against the stroke's own
   path length, the crossing test against which SIDE of a ray a vertex falls on
   rather than against a distance, the rim selection against the height it just
   set. An absolute tolerance anywhere in there passes at size 1 and fails at 0.001
   or 1000, which is a defect nobody sees until someone works in millimeters. */
for (const [name, poly] of [['circle', circle], ['bean', bean]]) {
  const base = puffCage(poly, opts);
  for (const k of [1e-3, 1e3, 1e6]) {
    const r = call(`${name} scaled by ${k}`, poly.map((v) => v * k));
    assert.equal(r.ok, true, `${name} at scale ${k}: ${r.why}`);
    assert.equal(r.grow, base.grow, `${name} at scale ${k}: a different growth was solved`);
    assert.equal(r.quads, base.quads);
    let worst = 0;
    for (let i = 0; i < base.cage.vertices.length; i += 1) {
      for (let c = 0; c < 3; c += 1) worst = Math.max(worst, Math.abs(r.cage.vertices[i][c] - base.cage.vertices[i][c] * k));
    }
    assert.ok(worst / k < 1e-12, `${name} at scale ${k}: the cage is not the same cage scaled (off by ${(worst / k).toExponential(1)})`);
  }
  // And it does not depend on where the drawing sits.
  const moved = call(`${name} moved`, poly.map((v, i) => (i % 2 ? v - 500 : v + 1000)));
  let worst = 0;
  for (let i = 0; i < base.cage.vertices.length; i += 1) {
    const want = [base.cage.vertices[i][0] + 1000, base.cage.vertices[i][1] - 500, base.cage.vertices[i][2]];
    for (let c = 0; c < 3; c += 1) worst = Math.max(worst, Math.abs(moved.cage.vertices[i][c] - want[c]));
  }
  assert.ok(worst < 1e-11, `${name}: the cage depends on where the drawing sits (off by ${worst.toExponential(1)})`);
}
console.log('  the same drawing at 0.001, 1, 1000 and 1000000, and moved a thousand units, gives one cage up to scale');

// ── AND IT DOES NOT DEPEND ON WHERE THE HAND STARTED ─────────────────────────
/* The cage's directions are spaced by arc length along the drawn line, anchored
   at angle zero rather than at the stroke's first point — so rotating the seam
   must not rotate the vertices. Anchored at the first point this is the failure
   that hides: the same shape comes back with its cage in different places. */
{
  const base = puffCage(bean, opts);
  const rotated = bean.slice(20).concat(bean.slice(0, 20));
  const r = call('the same loop started elsewhere', rotated);
  let worst = 0;
  for (let i = 0; i < base.cage.vertices.length; i += 1) {
    for (let c = 0; c < 3; c += 1) worst = Math.max(worst, Math.abs(r.cage.vertices[i][c] - base.cage.vertices[i][c]));
  }
  assert.ok(worst < 0.02, `starting the same stroke elsewhere moved the cage by ${worst.toFixed(4)}`);
}

// ── A NON-STAR-SHAPED OUTLINE IS APPROXIMATED, NOT REFUSED, AND SAYS SO ──────
/* ⚠⚠⚠ THIS IS A DECIDED CALL AND THE REASONING IS IN THE MODULE HEADER. The
   radial warp takes the OUTERMOST crossing, so a horseshoe is inflated across its
   own mouth. Refusing would be the cheaper code and the worse tool — a comma, a
   bean and a boomerang are ordinary things to draw, and at two dozen faces no
   construction represents a deep concavity anyway, so the refusal would spend the
   tool and buy nothing. What the result owes instead is an honest account of what
   it did: `starShaped`, the fraction of directions that meet the line more than
   once, the measured distance from the drawn line, and a sentence a UI can show. */
const horseshoe = (gap) => {
  const p = [], a0 = gap, a1 = Math.PI * 2 - gap;
  for (let i = 0; i <= 80; i += 1) { const a = a0 + ((a1 - a0) * i) / 80; p.push(Math.cos(a), Math.sin(a)); }
  for (let i = 80; i >= 0; i -= 1) { const a = a0 + ((a1 - a0) * i) / 80; p.push(Math.cos(a) * 0.5, Math.sin(a) * 0.5); }
  return p;
};
for (const gap of [0.35, 0.8, 1.3]) {
  const r = call(`horseshoe, mouth ${gap} rad`, horseshoe(gap));
  assert.equal(r.ok, true, 'a horseshoe must still produce geometry');
  assert.equal(r.starShaped, false, 'a horseshoe is not star-shaped and the result must say so');
  assert.ok(r.reentrant > 0.3, `only ${(100 * r.reentrant).toFixed(0)}% of directions read as re-entrant on a horseshoe`);
  assert.ok(r.note.length > 40 && NAMES_A_CHANGE.test(r.note), 'the approximation must say what it did and what to do instead');
  // BOUNDED: the approximation never leaves the drawing's own reach.
  let far = 0, ext = 0;
  const poly = horseshoe(gap);
  for (let i = 0; i < poly.length; i += 2) ext = Math.max(ext, Math.hypot(poly[i], poly[i + 1]));
  for (const v of r.cage.vertices) far = Math.max(far, Math.hypot(v[0], v[1]));
  assert.ok(far <= ext * 1.35, `the bridged cage reaches ${(far / ext).toFixed(2)} of the drawing's own extent`);
  console.log(`  horseshoe mouth ${gap} rad: ${(100 * r.reentrant).toFixed(0)}% of directions re-entrant, `
    + `rim ${(100 * r.worstDeviation).toFixed(0)}% of the reach off the line, worst face ${r.aspect.toFixed(1)}:1`);
}
// A star-shaped outline says so, and carries no note to show.
{
  const r = puffCage(bean, opts);
  assert.equal(r.starShaped, true);
  assert.equal(r.reentrant, 0);
  assert.equal(r.note, '');
}

// ── AND NOTHING IN A LONG TAIL OF JUNK THROWS ────────────────────────────────
/* A refusal is a decision; a throw is the absence of one. The inputs below are
   not shapes and are not expected to be — what is asserted is only that each one
   comes back as a decision. */
let refusedCount = 0, builtCount = 0;
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
for (let trial = 0; trial < 400; trial += 1) {
  const n = 1 + Math.floor(rnd() * 40);
  const p = [];
  for (let i = 0; i < n * 2; i += 1) {
    const r = rnd();
    p.push(r < 0.02 ? 0 : r < 0.04 ? 1e-18 : r < 0.06 ? 1e18 : (rnd() - 0.5) * 4);
  }
  const r = call(`fuzz ${trial}`, p);
  if (r.ok) {
    builtCount += 1;
    assert.ok(r.cage.faces.every((f) => f.length === 4), `fuzz ${trial}: not all quads`);
    assert.ok(r.cage.vertices.every((v) => v.every(Number.isFinite)), `fuzz ${trial}: a NaN in the cage`);
  } else refusedCount += 1;
}
console.log(`  400 random strokes: ${refusedCount} refused with a reason, ${builtCount} built a finite all-quad cage, 0 threw`);

/* ⚠ AND THE SAME FUZZ ON THE ACCEPTING SIDE. Random points are almost never a
   simple closed loop, so the sweep above exercises the refusals and barely
   touches the build. A hand-drawn circle carrying tremor, repeated samples and
   the odd spike is the input this tool actually receives, and it has to come
   back as a cage every time. */
let jitterBuilt = 0;
for (let trial = 0; trial < 200; trial += 1) {
  const p = [], n = 24 + Math.floor(rnd() * 90);
  for (let i = 0; i < n; i += 1) {
    const a = (i / n) * Math.PI * 2;
    const rad = 1 + 0.35 * Math.sin(a * (1 + Math.floor(rnd() * 4))) + (rnd() - 0.5) * 0.06;
    p.push(Math.cos(a) * rad, Math.sin(a) * rad);
    if (rnd() < 0.08) p.push(p[p.length - 2], p[p.length - 1]);       // a repeated sample
  }
  const r = call(`jitter ${trial}`, p);
  if (!r.ok) continue;
  jitterBuilt += 1;
  assert.ok(r.cage.faces.every((f) => f.length === 4), `jitter ${trial}: not all quads`);
  assert.ok(r.cage.vertices.every((v) => v.every(Number.isFinite)), `jitter ${trial}: a NaN in the cage`);
  assert.ok(r.aspect > 1 && Number.isFinite(r.aspect), `jitter ${trial}: aspect ${r.aspect}`);
  assert.ok(Math.abs(Math.max(...r.cage.vertices.map((v) => v[2])) - r.height) < 1e-9 * r.height,
    `jitter ${trial}: the cage is not the height it reports`);
}
assert.ok(jitterBuilt > 180, `only ${jitterBuilt} of 200 wobbly hand-drawn loops built — the accepting path is barely exercised`);
console.log(`  200 wobbly loops with repeated samples: ${jitterBuilt} built, every one all-quad, finite, and the height it reports`);
console.log('puff-robust: ok');
