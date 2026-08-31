// A FLAT BACK IS WHAT LETS A DRAWN FORM SIT ON A SURFACE.
//
// The asked-for case is a computer mouse: a domed top over a planar base. The
// interesting part is not the projection, which is one line, but the CREASE it
// needs — left smooth, Catmull-Clark rolls the base under at the rim, the
// underside stops being planar exactly where it meets the ground, and the form
// rocks. That failure is invisible from above and obvious on a table.
import { strict as assert } from 'node:assert';
import { puffCage } from '../kernel/puff.mjs';
import { subdivideCatmullClark } from '../kernel/subd.mjs';
import { subdToPatches } from '../kernel/subdlimit.mjs';

const ring = (n, f) => { const p = []; for (let i = 0; i < n; i += 1) { const a = (i / n) * Math.PI * 2; const r = f(a); p.push(Math.cos(a) * r, Math.sin(a) * r); } return p; };
const shapes = [
  ['round', ring(96, () => 1)],
  ['mouse', ring(96, (a) => 1 + 0.28 * Math.cos(a))],
  ['lobed', ring(96, (a) => 1 + 0.4 * Math.cos(a * 2))],
];

for (const [name, poly] of shapes) {
  const flat = puffCage(poly, { subdivide: subdivideCatmullClark, flatBack: 1 });
  const dome = puffCage(poly, { subdivide: subdivideCatmullClark, flatBack: 0 });
  assert.equal(flat.ok, true, `${name}: ${flat.why || flat.reason}`);

  let c = flat.cage;
  for (let k = 0; k < 3; k += 1) c = subdivideCatmullClark(c);
  let zmin = Infinity, zmax = -Infinity, below = 0, onPlane = 0;
  for (const v of c.vertices) {
    zmin = Math.min(zmin, v[2]); zmax = Math.max(zmax, v[2]);
    if (v[2] < -1e-4) below += 1;
    if (Math.abs(v[2]) < 1e-6) onPlane += 1;
  }

  /* ⚠ THE LIMIT SURFACE, NOT THE CAGE. Projecting cage vertices onto the plane
     is trivial; keeping the SMOOTH surface they define on it is the claim, and
     it is the one a crease is needed for. Without the crease the limit rolls
     under and this is negative. */
  assert.ok(below === 0, `${name}: ${below} points of the limit surface sit BELOW the plane it is meant to rest on (min ${zmin.toFixed(4)})`);
  assert.ok(onPlane > 20, `${name}: only ${onPlane} points actually lie on the plane — that is a rounded base, not a flat one`);

  // It must still be a form, not a pancake: the top has to dome.
  assert.ok(zmax > 0.05, `${name}: nothing rises above the base (zmax ${zmax.toFixed(3)})`);

  // The crease belongs on the silhouette and nowhere else.
  const creased = Object.keys(flat.cage.creases || {}).length;
  assert.ok(creased > 0, `${name}: a flat back with no crease will roll under at its rim`);
  assert.equal(Object.keys(dome.cage.creases || {}).length, 0, `${name}: a domed puff must carry no crease`);

  // Topology survives the projection: still all quads, still closed.
  assert.ok(flat.cage.faces.every((f) => f.length === 4), `${name}: not all quads`);
  const E = new Set();
  for (const f of flat.cage.faces) for (let k = 0; k < 4; k += 1) {
    const a = f[k], b = f[(k + 1) % 4];
    E.add(a < b ? `${a}_${b}` : `${b}_${a}`);
  }
  assert.equal(flat.cage.vertices.length - E.size + flat.cage.faces.length, 2, `${name}: no longer closed`);

  // And it still converts — a crease makes the faces beside it irregular.
  const out = subdToPatches(flat.cage, {});
  assert.ok(out.uncoveredFraction < 0.05, `${name}: NURBS coverage ${(out.uncoveredFraction * 100).toFixed(2)}% uncovered`);

  console.log(`  ${name.padEnd(6)} ${String(flat.quads).padStart(3)} quads  ${creased} creased  z ${zmin.toFixed(3)}..${zmax.toFixed(3)}  ${onPlane} points on the plane  ${(out.uncoveredFraction * 100).toFixed(2)}% uncovered`);
}

/* THE CONTROL. A domed puff must have points below the plane — otherwise the
   flat-back assertion above would pass on both and prove nothing. */
{
  const dome = puffCage(ring(96, () => 1), { subdivide: subdivideCatmullClark, flatBack: 0 });
  let c = dome.cage;
  for (let k = 0; k < 3; k += 1) c = subdivideCatmullClark(c);
  const below = c.vertices.filter((v) => v[2] < -1e-4).length;
  assert.ok(below > 50, `the control failed: a domed puff should hang below the plane, ${below} points do`);
  console.log(`  control: a domed puff puts ${below} points below the plane, so the flat-back check can fail`);
}
console.log('puff-flatback: ok');
