// A parameter that changes nothing is a parameter nobody can read.
//
// `PUFF_PARAMS` is what a UI builds rows from, so everything a row needs has to
// be in it and everything in it has to be true of the code. Two failures this
// catches and nothing else does: a bound written in the declaration while the
// code clamps to a different one, and a control that is wired up, labeled,
// draggable and inert.
import { strict as assert } from 'node:assert';
import { puffCage, puffFaceAspect, PUFF_PARAMS, puffResolveParams, puffDensityForFaces } from '../kernel/puff.mjs';
import { subdivideCatmullClark } from '../kernel/subd.mjs';

const opts = { subdivide: subdivideCatmullClark };
const ring = (n, f) => { const p = []; for (let i = 0; i < n; i += 1) { const a = (i / n) * Math.PI * 2; const r = f(a); p.push(Math.cos(a) * r, Math.sin(a) * r); } return p; };
const circle = ring(64, () => 1);
const ellipse3 = (() => { const p = []; for (let i = 0; i < 64; i += 1) { const a = (i / 64) * Math.PI * 2; p.push(Math.cos(a) * 3, Math.sin(a)); } return p; })();
const bean = ring(64, (a) => 1 + 0.45 * Math.cos(a * 2) - 0.2 * Math.sin(a * 3));
const SHAPES = [['circle', circle], ['ellipse 3:1', ellipse3], ['bean', bean]];
const extent = (poly) => { let e = 0; for (let i = 0; i < poly.length; i += 2) e = Math.max(e, Math.hypot(poly[i], poly[i + 1])); return e; };

// The declaration is complete and internally consistent
const keys = Object.keys(PUFF_PARAMS);
assert.ok(keys.length >= 3, 'a puff needs at least density, puffiness and width-follow');
for (const key of ['density', 'puffiness', 'follow']) {
  assert.ok(keys.includes(key), `PUFF_PARAMS is missing ${key}`);
}
const orders = new Set();
for (const key of keys) {
  const s = PUFF_PARAMS[key];
  for (const field of ['label', 'help', 'min', 'max', 'step', 'default', 'order']) {
    assert.ok(s[field] !== undefined, `${key}: no ${field} — a row cannot be built from it`);
  }
  assert.equal(typeof s.label, 'string');
  // A help string that only restates the label tells a reader nothing they did
  // not already have from the label.
  assert.ok(s.help.length > 40 && !s.help.startsWith(s.label), `${key}: help says nothing the label did not`);
  assert.ok(s.min < s.max, `${key}: empty range`);
  assert.ok(s.step > 0 && s.step <= s.max - s.min, `${key}: step ${s.step} cannot travel its own range`);
  assert.ok(s.default >= s.min && s.default <= s.max, `${key}: default ${s.default} is outside its own range`);
  if (s.integer) assert.equal(s.default, Math.round(s.default), `${key}: integer control with a fractional default`);
  assert.ok(!orders.has(s.order), `${key}: two controls claim row ${s.order}`);
  orders.add(s.order);
  // Frozen, because a UI holding a reference to this could otherwise write a
  // user's dragged value back into the declaration and move everyone's default.
  assert.ok(Object.isFrozen(s), `${key}: the declaration is writable`);
}
assert.ok(Object.isFrozen(PUFF_PARAMS));

// The code clamps to the declaration, not to literals of its own
/* A bound stated in one place and enforced in another drifts silently: the UI
   builds a slider from the declaration, the kernel obeys its own number, and the
   end of the slider does nothing. So the clamp is asserted against the same
   object the row is built from, and every declared control is walked — a
   hand-written list of three would pass while a fourth went unenforced. */
for (const key of keys) {
  const s = PUFF_PARAMS[key];
  assert.equal(puffResolveParams({ [key]: s.max * 10 + 100 })[key], s.max, `${key}: not clamped above`);
  assert.equal(puffResolveParams({ [key]: s.min - 100 })[key], s.min, `${key}: not clamped below`);
  assert.equal(puffResolveParams({ [key]: 'nonsense' })[key], s.default, `${key}: garbage did not fall back to the default`);
  assert.equal(puffResolveParams({ [key]: NaN })[key], s.default, `${key}: NaN did not fall back to the default`);
  assert.equal(puffResolveParams({})[key], s.default, `${key}: no default applied`);
}
// And the clamp is what the caller is told it got, so a readout cannot show a
// value the geometry was not built from.
{
  const r = puffCage(circle, { ...opts, puffiness: 99, follow: -3, density: 99 });
  assert.equal(r.ok, true);
  assert.equal(r.puffiness, PUFF_PARAMS.puffiness.max);
  assert.equal(r.follow, PUFF_PARAMS.follow.min);
  assert.equal(r.density, PUFF_PARAMS.density.max);
  assert.equal(r.quads, 6 * PUFF_PARAMS.density.max ** 2, 'density does not mean what its help says');
}
// `faces` is an alias of density: it lands on the coarsest rung that reaches it.
assert.equal(puffDensityForFaces(24), 2);
assert.equal(puffDensityForFaces(25), 3);
assert.equal(puffDensityForFaces(96), 4);
assert.equal(puffCage(circle, { ...opts, faces: 96 }).quads, 96);
assert.equal(puffCage(circle, { ...opts, faces: 96, density: 2 }).quads, 24, 'density must win over the faces alias');
for (let d = PUFF_PARAMS.density.min; d <= PUFF_PARAMS.density.max; d += 1) {
  assert.equal(puffCage(circle, { ...opts, density: d }).quads, 6 * d * d, `density ${d} does not give 6d^2 faces`);
}

// Every control moves the geometry, and by how much is measured
/* Checked on several shapes, because a control can be alive on one and dead on
   another: width-follow has nothing to say about a circle's own uniform width
   and everything to say about a bean's, and a single round fixture would have
   passed it either way. */
const MOVE = 0.05;                       // of the outline's own reach
for (const key of keys) {
  const s = PUFF_PARAMS[key];
  /* A control with a precondition is swept with that precondition met, and the
     precondition is read from the declaration rather than listed here. Swept at
     the bare defaults, `flipFlat` moves the cage 0% — correctly, because there
     is no flat side to flip — and this check would have to be either weakened or
     given an exception by name, both of which turn "every control moves the
     geometry" into "every control except the ones we excused". Satisfying the
     requirement instead keeps the claim total: a declared control moves the
     cage wherever it is offered at all. */
  const base = s.requires ? { ...opts, [s.requires]: PUFF_PARAMS[s.requires].max } : opts;
  for (const [name, poly] of SHAPES) {
    const lo = puffCage(poly, { ...base, [key]: s.min });
    const hi = puffCage(poly, { ...base, [key]: s.max });
    assert.equal(lo.ok && hi.ok, true, `${key} on ${name}: refused at an end of its own range`);
    if (lo.cage.vertices.length !== hi.cage.vertices.length) {
      assert.notEqual(lo.quads, hi.quads, `${key} on ${name}: changed the cage without changing it`);
      continue;
    }
    let worst = 0;
    for (let i = 0; i < lo.cage.vertices.length; i += 1) {
      for (let c = 0; c < 3; c += 1) worst = Math.max(worst, Math.abs(lo.cage.vertices[i][c] - hi.cage.vertices[i][c]));
    }
    const moved = worst / extent(poly);
    assert.ok(moved > MOVE, `${key} on ${name}: end to end it moves the cage ${(100 * moved).toFixed(1)}% of the drawing — a control nobody can read`);
  }
}

// Puffiness at its extremes, and why zero is not one of them
/* At puffiness 0 the cage folds onto a plane, and nothing in the result
   measures it. The quad ball carries a vertex at (x, y, +z) for every
   (x, y, -z); flattened, those become the same point, the two sheets lie on each
   other, and the enclosed volume is exactly zero. The worst face ratio of that
   object is 4.2:1 — a clean number, well inside any bar this module asserts.
   Face aspect sees a sliver that face counts cannot, and it cannot see this
   either. So 0 is outside the declared range, and this is the
   demonstration that it has to be rather than an assertion that it is. */
{
  const flat = puffCage(circle, opts).cage;
  const collapsed = { vertices: flat.vertices.map((v) => [v[0], v[1], 0]), faces: flat.faces, creases: {} };
  const key = (v) => `${v[0].toFixed(9)},${v[1].toFixed(9)},${v[2].toFixed(9)}`;
  const seen = new Set(collapsed.vertices.map(key));
  const coincident = collapsed.vertices.length - seen.size;
  assert.ok(coincident > 0, 'the demonstration is wrong: flattening did not make any two vertices coincide');

  const signedVolume = (cage) => {
    const tri = (a, b, c) => (a[0] * (b[1] * c[2] - b[2] * c[1])
      - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
    let v = 0;
    for (const f of cage.faces) {
      const p = f.map((i) => cage.vertices[i]);
      v += tri(p[0], p[1], p[2]) + tri(p[0], p[2], p[3]);
    }
    return v;
  };
  assert.equal(signedVolume(collapsed), 0, 'a flattened cage should enclose nothing');
  assert.ok(signedVolume(flat) > 0, 'the control is wrong: the inflated cage encloses nothing either');
  assert.ok(puffFaceAspect(collapsed) < 8,
    'face aspect reports the flattened cage — the argument for the declared puffiness floor must be restated');
  assert.ok(PUFF_PARAMS.puffiness.min > 0, 'puffiness must not be allowed to reach 0');
  assert.equal(puffResolveParams({ puffiness: 0 }).puffiness, PUFF_PARAMS.puffiness.min);
  assert.ok(PUFF_PARAMS.bottomScale === undefined || PUFF_PARAMS.bottomScale.min > 0,
    'the underside must not be allowed to reach 0 either — it folds the same way');
}
// At the declared floor and ceiling the cage is still a cage: no coincident
// vertices, no zero-length edge, and the subdivider takes it without a NaN.
for (const p of [PUFF_PARAMS.puffiness.min, PUFF_PARAMS.puffiness.max]) {
  for (const [name, poly] of SHAPES) {
    for (const d of [PUFF_PARAMS.density.min, PUFF_PARAMS.density.max]) {
      const r = puffCage(poly, { ...opts, puffiness: p, density: d });
      assert.equal(r.ok, true, `${name} at puffiness ${p}: ${r.why}`);
      let shortest = Infinity, ext = 0;
      for (const v of r.cage.vertices) ext = Math.max(ext, Math.hypot(v[0], v[1]));
      for (const f of r.cage.faces) {
        for (let k = 0; k < 4; k += 1) {
          const a = r.cage.vertices[f[k]], b = r.cage.vertices[f[(k + 1) % 4]];
          shortest = Math.min(shortest, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
        }
      }
      assert.ok(shortest / ext > 1e-3, `${name} at puffiness ${p} density ${d}: shortest edge is ${(shortest / ext).toExponential(1)} of the extent`);
      let c = r.cage;
      for (let k = 0; k < 2; k += 1) c = subdivideCatmullClark(c);
      assert.ok(c.vertices.every((v) => v.every(Number.isFinite)), `${name} at puffiness ${p}: the subdivider produced a NaN`);
    }
  }
}
