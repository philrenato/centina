// A holey puff with a flat side keeps that side flat and unfolded.
//
// With Underside 0 (or -1) every vertex of one sheet lies on the drawing plane, the same plane as
// the rims. A rim found by |z| = 0 then takes the whole flat sheet with it, and the rim solve drags
// those vertices onto the drawn loops: the flat side folds over itself while the topology stays
// closed and consistently wound. The flat sheet is checked here by the winding of each quad's
// projection (one sheet faces up, the other down, none folded) and by the cage and its limit
// staying free of self-crossings once the flat sheet is lifted off the plane by a hair. The
// invariants must also see the fold on their own.
import { strict as assert } from 'node:assert';
import { buildMultiHolePuff, buildAnnularPuff, puffInvariants, puffLimitRimMiss } from '../kernel/puff.mjs';
import { subdivideCatmullClark } from '../kernel/subd.mjs';
import { selfCrossings } from './fixtures/mesh-crossings.mjs';

const circle = (r, n, cx = 0, cy = 0) => { const p = []; for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; p.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]); } return p; };
const O = circle(40, 120);
const fixtures = [
  ['two holes', O, [circle(6, 40, -15, 0), circle(6, 40, 15, 0)]],
  ['three holes', O, [circle(5, 32, -18, -8), circle(5, 32, 18, -8), circle(5, 32, 0, 16)]],
];
const toMesh = (r) => {
  const V = []; for (let i = 0; i < r.nv; i++) V.push([r.positions[3 * i], r.positions[3 * i + 1], r.positions[3 * i + 2]]);
  const F = []; for (let i = 0; i < r.quads.length; i += 4) F.push([r.quads[i], r.quads[i + 1], r.quads[i + 2], r.quads[i + 3]]);
  return { V, F };
};
// The flat sheet lifted off the plane by a small, vertex-varying amount, so a fold inside it shows
// as a crossing rather than as coplanar overlap.
const lifted = (V, sign) => V.map((p, i) => (Math.abs(p[2]) <= 1e-9 ? [p[0], p[1], p[2] + sign * 1e-3 * (1 + ((i * 0.618034) % 1))] : p));

for (const [name, outer, holes] of fixtures) {
  for (const bs of [0, -1]) {
    const r = buildMultiHolePuff(outer, holes, { bottomScale: bs });
    assert.ok(r.ok, `${name} Underside ${bs} builds`);
    const { V, F } = toMesh(r);
    // Which sheet is flat: Underside 0 flattens the back (z <= 0), Underside -1 the front.
    const flatSign = bs === 0 ? -1 : 1;
    // Rims stay on the plane; the flat sheet's quads all project with one winding.
    let folded = 0, flat = 0;
    for (const f of F) {
      if (!f.every((v) => Math.abs(V[v][2]) <= 1e-6)) continue;
      flat++;
      let a = 0; for (let k = 0; k < 4; k++) { const p = V[f[k]], q = V[f[(k + 1) % 4]]; a += p[0] * q[1] - q[0] * p[1]; }
      // Outward normal of the flat sheet is flatSign·z, so its projected winding has that sign.
      if (Math.sign(a) !== flatSign) folded++;
    }
    assert.ok(flat > 20, `${name} Underside ${bs}: a flat sheet is present (${flat} quads)`);
    assert.equal(folded, 0, `${name} Underside ${bs}: ${folded} of ${flat} flat quads are folded over`);
    const VL = lifted(V, flatSign);
    assert.equal(selfCrossings(VL, F), 0, `${name} Underside ${bs}: the cage crosses itself`);
    let cage = { vertices: VL, faces: F };
    cage = subdivideCatmullClark(subdivideCatmullClark(cage));
    assert.equal(selfCrossings(cage.vertices, cage.faces), 0, `${name} Underside ${bs}: the limit crosses itself`);
    const inv = puffInvariants(r);
    assert.ok(inv.ok, `${name} Underside ${bs}: invariants ok (${JSON.stringify(inv)})`);
    // The limit rim still lands on the drawn loops.
    const miss = puffLimitRimMiss(r, [outer, ...holes]);
    assert.ok(miss.worst < 0.02, `${name} Underside ${bs}: limit rim miss ${miss.worst.toFixed(4)} of the span`);
  }
}

// One hole, flat back: the single-ring builder shares the rim solve.
{
  const r = buildAnnularPuff(O, [circle(10, 48)], { bottomScale: 0 });
  assert.ok(r.ok, 'annular flat back builds');
  const { V, F } = toMesh(r);
  let folded = 0;
  for (const f of F) {
    if (!f.every((v) => Math.abs(V[v][2]) <= 1e-6)) continue;
    let a = 0; for (let k = 0; k < 4; k++) { const p = V[f[k]], q = V[f[(k + 1) % 4]]; a += p[0] * q[1] - q[0] * p[1]; }
    if (a > 0) folded++;
  }
  assert.equal(folded, 0, 'annular flat back: no folded quad');
  assert.ok(puffInvariants(r).ok, 'annular flat back: invariants ok');
}

// The invariants see a folded sheet and a crossing cage on their own: fold one flat-back quad of a
// good build over its neighbor and the check must fail.
{
  const r = buildMultiHolePuff(O, [circle(6, 40, -15, 0), circle(6, 40, 15, 0)], { bottomScale: 0.5 });
  assert.ok(puffInvariants(r).ok, 'reference build passes');
  // Push one back-sheet vertex far up through the front sheet: the cage now crosses itself.
  const bad = { ...r, positions: r.positions.slice() };
  let vi = -1, lo = 0;
  for (let i = 0; i < r.nv; i++) if (r.positions[3 * i + 2] < lo) { lo = r.positions[3 * i + 2]; vi = i; }
  bad.positions[3 * vi + 2] = -lo * 3;
  const inv = puffInvariants(bad);
  assert.equal(inv.ok, false, 'a back vertex pushed through the front is not ok');
  assert.ok(inv.crossings > 0, 'and the invariants count the crossing');
}
// A fold inside the flat sheet, in its own plane, crosses nothing; `reversed` must see it.
{
  const r = buildMultiHolePuff(O, [circle(6, 40, -15, 0), circle(6, 40, 15, 0)], { bottomScale: 0 });
  assert.ok(puffInvariants(r).ok, 'flat-back reference build passes');
  const bad = { ...r, positions: r.positions.slice() };
  // The flat vertex nearest the middle, carried across its whole one-ring in the plane.
  let vi = -1, best = Infinity;
  for (let i = 0; i < r.nv; i++) if (!r.rim[i] && r.positions[3 * i + 2] === 0) { const d = Math.hypot(r.positions[3 * i], r.positions[3 * i + 1] - 20); if (d < best) { best = d; vi = i; } }
  let reach = 0;
  for (let f = 0; f < r.quads.length; f += 4) { const q = Array.from(r.quads.slice(f, f + 4)); if (q.includes(vi)) for (const v of q) reach = Math.max(reach, Math.hypot(r.positions[3 * v] - r.positions[3 * vi], r.positions[3 * v + 1] - r.positions[3 * vi + 1])); }
  bad.positions[3 * vi] += 2.5 * reach;
  const inv = puffInvariants(bad);
  assert.equal(inv.ok, false, 'a flat sheet folded in its plane is not ok');
  assert.ok(inv.reversed > 0, `and the invariants count the reversed faces (${inv.reversed}, crossings ${inv.crossings})`);
}
