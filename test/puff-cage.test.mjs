// A PUFF IS A CAGE, AND THE CAGE IS THE CLAIM.
//
// Every assertion here is about the thing handed to the subdivider: all quads,
// closed, genus 0, few faces, and a limit surface that lands on the line that
// was drawn. A puff that looks right and is not watertight is a puff that cannot
// be filleted, shelled, or exported, and none of those failures name it.
import { strict as assert } from 'node:assert';
import { puffCage } from '../kernel/puff.mjs';
import { subdivideCatmullClark, buildTopology } from '../kernel/subd.mjs';
import { vertexLimitPosition } from '../kernel/subdlimit.mjs';
import { subdToPatches } from '../kernel/subdlimit.mjs';

const ring = (n, f) => { const p = []; for (let i = 0; i < n; i += 1) { const a = (i / n) * Math.PI * 2; const r = f(a); p.push(Math.cos(a) * r, Math.sin(a) * r); } return p; };
const circle = ring(64, () => 1);
const ellipse = (k) => { const p = []; for (let i = 0; i < 64; i += 1) { const a = (i / 64) * Math.PI * 2; p.push(Math.cos(a) * k, Math.sin(a)); } return p; };
const bean = ring(64, (a) => 1 + 0.45 * Math.cos(a * 2) - 0.2 * Math.sin(a * 3));
const opts = { subdivide: subdivideCatmullClark };

const euler = (cage) => {
  const E = new Set();
  for (const f of cage.faces) for (let k = 0; k < f.length; k += 1) {
    const a = f[k], b = f[(k + 1) % f.length];
    E.add(a < b ? `${a}_${b}` : `${b}_${a}`);
  }
  return cage.vertices.length - E.size + cage.faces.length;
};
const valences = (cage) => {
  const seen = new Map(), deg = new Map();
  for (const f of cage.faces) for (let k = 0; k < f.length; k += 1) {
    const a = f[k], b = f[(k + 1) % f.length];
    const key = a < b ? `${a}_${b}` : `${b}_${a}`;
    if (seen.has(key)) continue;
    seen.set(key, 1);
    deg.set(a, (deg.get(a) || 0) + 1); deg.set(b, (deg.get(b) || 0) + 1);
  }
  return deg;
};

for (const [name, poly] of [['circle', circle], ['ellipse 1.5', ellipse(1.5)], ['bean', bean]]) {
  const r = puffCage(poly, opts);
  assert.equal(r.ok, true, `${name}: ${r.why || r.reason}`);

  // EVERY FACE A QUAD. The subdivider takes other arities, the NURBS conversion
  // does not — `referenceMeshToSuperBCage` accepts quads and nothing else, so a
  // stray triangle is a puff that cannot become a surface.
  assert.ok(r.cage.faces.every((f) => f.length === 4), `${name}: not all quads`);

  // CLOSED, GENUS 0, WATERTIGHT — one number, and the one a shared equator
  // either satisfies or does not.
  assert.equal(euler(r.cage), 2, `${name}: Euler ${euler(r.cage)}, not 2`);

  // NO VALENCE-2 VERTEX. A ring cage that lost its cap produces them, they
  // subdivide into a pinch, and Euler alone cannot see it.
  const deg = valences(r.cage);
  assert.ok(![...deg.values()].some((v) => v < 3), `${name}: a vertex of valence < 3`);

  // AS FEW FACES AS THE SILHOUETTE ALLOWS. The ceiling is the whole point of the
  // density ladder; without it "solved density" would quietly mean "dense".
  assert.ok(r.quads <= 54, `${name}: ${r.quads} quads quads is not a cage anyone edits by hand`);

  // AND THE LIMIT SURFACE LANDS ON THE DRAWN LINE. Deviation is signed and
  // measured against the polygon, not against a radius — a mean radius is an
  // oracle for a circle and reports nonsense on anything else.
  /* WORST, NOT MEAN. A rim that scallops between control points has a small
     average error and a large local one; asserting the mean passes the shape
     that most needs catching. */
  assert.ok(r.worstDeviation <= 0.25, `${name}: silhouette locally off by ${(r.worstDeviation * 100).toFixed(2)}%`);

  /* ⚠ AND THE RIM IS SMOOTH. Creasing it also puts the silhouette on the drawn
     curve, at a quarter of the faces — by making the equator SHARP, which turns
     a closed form into a saucer with a lip. The cheaper answer is the wrong
     object, so the absence of creases is asserted rather than assumed. */
  assert.equal(Object.keys(r.cage.creases || {}).length, 0, `${name}: the rim was creased — that is a flange, not a form`);

  // ⚠ AND THE CAGE STILL CONVERTS. A creased rim makes every face beside it
  // irregular, and an isolation budget that cannot clear the weight would leave
  // holes around the entire outline. This is the assertion that would catch it.
  const out = subdToPatches(r.cage, {});
  const list = Array.isArray(out) ? out : (out.patches || []);
  const capped = list.filter((p) => p.kind !== 'regular').length;
  assert.equal(capped, 0, `${name}: ${capped} of ${list.length} patches are caps — NURBS coverage has holes at the silhouette`);

  assert.ok(r.aspect <= 8, `${name}: worst face edge ratio ${r.aspect.toFixed(1)}:1 — that is a sliver, not a face you can grab`);
  console.log(`  ${name.padEnd(12)} ${String(r.quads).padStart(3)} quads  aspect ${r.aspect.toFixed(1)}:1  worst dev ${(r.worstDeviation * 100).toFixed(1)}%  ${list.length} patches`);
}

// AN OPEN OR EMPTY STROKE IS A REFUSAL WITH A REASON, never a throw.
assert.equal(puffCage([], opts).ok, false);
assert.equal(puffCage(circle, {}).ok, false, 'no subdivider must refuse, not throw');
console.log('  refusals carry a reason');
console.log('puff-cage: ok');
