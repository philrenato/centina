// SPLITS AS AN ORDERED FEATURE LIST ON THE SURFACE.
//
// A surface carries a list of `{ direction, frac }` entries instead of being
// wrapped in a container per split. The properties worth testing are not "it
// produces pieces" — it obviously does — but the two that made the list the
// right shape:
//   · every fraction is measured on the ORIGINAL, so entries are INDEPENDENT
//     (moving one does not move another) and each piece is exactly ONE refit
//     deep however many entries there are;
//   · the pieces still lie ON the original surface, which is the whole claim.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { applySplitFeatures } from '../kernel/splitfeatures.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { revolve, makeLine } from '../kernel/primitives.mjs';

// ⚠ A CURVED FIXTURE, not a plane. A flat surface is reproduced exactly by any
// refit, so every accuracy claim below would pass on a bug — the approximation
// this design exists to bound would be invisible.
const SADDLE = (() => {
  const ctrlNet = [];
  for (let i = 0; i < 5; i++) {
    const row = [];
    for (let j = 0; j < 5; j++) {
      const x = -10 + (20 * i) / 4, y = -10 + (20 * j) / 4;
      row.push([x, y, (x * x - y * y) / 14, 1]);
    }
    ctrlNet.push(row);
  }
  return {
    degU: 3, degV: 3,
    knotsU: [-10, -10, -10, -10, 0, 10, 10, 10, 10],
    knotsV: [-10, -10, -10, -10, 0, 10, 10, 10, 10],
    ctrlNet,
  };
})();

const near = (a, b, tol) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) <= tol;
// How far a piece strays from the surface it was cut from, sampled over the
// piece's own domain and compared against the ORIGINAL at the same parameters.
function worstDeviation(piece, srf, n = 9) {
  const u0 = piece.knotsU[0], u1 = piece.knotsU[piece.knotsU.length - 1];
  const v0 = piece.knotsV[0], v1 = piece.knotsV[piece.knotsV.length - 1];
  let worst = 0;
  for (let i = 0; i <= n; i++) {
    for (let j = 0; j <= n; j++) {
      const u = u0 + (u1 - u0) * (i / n), v = v0 + (v1 - v0) * (j / n);
      const a = surfacePoint(piece, u, v), b = surfacePoint(srf, u, v);
      worst = Math.max(worst, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
    }
  }
  return worst;
}

test('NO ENTRIES IS THE SURFACE ITSELF, not a refusal and not a rebuild', () => {
  const r = applySplitFeatures(SADDLE, []);
  assert.ok(r.ok, r.reason);
  assert.equal(r.pieces.length, 1);
  assert.deepEqual(r.pieces[0].ctrlNet, SADDLE.ctrlNet, 'an unsplit surface is handed back untouched');
});

test('ONE ENTRY CUTS IN TWO, and both halves lie on the original', () => {
  const r = applySplitFeatures(SADDLE, [{ direction: 'u', frac: 0.4 }]);
  assert.ok(r.ok, r.reason);
  assert.equal(r.pieces.length, 2);
  for (const p of r.pieces) assert.ok(worstDeviation(p, SADDLE) < 0.05, `a piece strayed ${worstDeviation(p, SADDLE)}`);
});

test('TWO ENTRIES IN ONE DIRECTION GIVE THREE STRIPS; CROSSED GIVE A GRID', () => {
  const strips = applySplitFeatures(SADDLE, [{ direction: 'u', frac: 0.3 }, { direction: 'u', frac: 0.7 }]);
  assert.ok(strips.ok, strips.reason);
  assert.equal(strips.pieces.length, 3);
  const grid = applySplitFeatures(SADDLE, [{ direction: 'u', frac: 0.5 }, { direction: 'v', frac: 0.5 }]);
  assert.ok(grid.ok, grid.reason);
  assert.equal(grid.pieces.length, 4);
  const wide = applySplitFeatures(SADDLE, [
    { direction: 'u', frac: 0.25 }, { direction: 'u', frac: 0.75 }, { direction: 'v', frac: 0.5 },
  ]);
  assert.equal(wide.pieces.length, 6, 'three cuts, 3x2');
});

test('⭐ ENTRIES ARE INDEPENDENT — adding one does not move the pieces the others make', () => {
  // The property the list exists for: "can the split location be dragged
  // differently later" only means anything if the entries do not define each
  // other. Cut at u=0.3; then cut at u=0.3 AND v=0.6 and check the first cut
  // still lands in exactly the same place.
  const one = applySplitFeatures(SADDLE, [{ direction: 'u', frac: 0.3 }]);
  const two = applySplitFeatures(SADDLE, [{ direction: 'u', frac: 0.3 }, { direction: 'v', frac: 0.6 }]);
  assert.ok(one.ok && two.ok);
  assert.equal(one.cuts.u[0], two.cuts.u[0], 'the u cut is at the same parameter in both');
  // And the first piece's own u-extent is unchanged by the unrelated v entry.
  assert.equal(one.pieces[0].knotsU[0], two.pieces[0].knotsU[0]);
  assert.equal(one.pieces[0].knotsU[one.pieces[0].knotsU.length - 1],
    two.pieces[0].knotsU[two.pieces[0].knotsU.length - 1]);
});

test('⭐⭐ EVERY PIECE IS ONE REFIT DEEP, however many entries there are', () => {
  // The reason fractions are measured on the ORIGINAL. `splitSurface` resamples
  // and refits, so cutting an already-cut piece would compound. If this were
  // implemented by repeated splitting, accuracy would decay with entry count;
  // measured against the ORIGINAL it must not.
  const one = applySplitFeatures(SADDLE, [{ direction: 'u', frac: 0.5 }]);
  const many = applySplitFeatures(SADDLE, [
    { direction: 'u', frac: 0.2 }, { direction: 'u', frac: 0.5 }, { direction: 'u', frac: 0.8 },
    { direction: 'v', frac: 0.3 }, { direction: 'v', frac: 0.65 },
  ]);
  assert.ok(one.ok && many.ok, one.reason || many.reason);
  assert.equal(many.pieces.length, 12);
  const worstOne = Math.max(...one.pieces.map((p) => worstDeviation(p, SADDLE)));
  const worstMany = Math.max(...many.pieces.map((p) => worstDeviation(p, SADDLE)));
  assert.ok(worstMany <= worstOne + 1e-9,
    `five entries must not be LESS accurate than one (${worstMany} vs ${worstOne}) — that would mean pieces are being re-split`);
});

test('COINCIDENT AND OUT-OF-RANGE ENTRIES COLLAPSE rather than making zero-width pieces', () => {
  const r = applySplitFeatures(SADDLE, [
    { direction: 'u', frac: 0.5 }, { direction: 'u', frac: 0.5 },   // the same cut twice
    { direction: 'u', frac: 0 }, { direction: 'u', frac: 1 },        // on the boundary: not cuts
  ]);
  assert.ok(r.ok, r.reason);
  assert.equal(r.cuts.u.length, 1, 'four entries, one real cut');
  assert.equal(r.pieces.length, 2);
  assert.equal(r.stats.dropped, 3);
});

test('A CLOSED DIRECTION REFUSES BY NAME rather than offering a dead cut', () => {
  // A full revolve is closed in its swept direction: cutting a closed loop at
  // ONE parameter unrolls it into a single open piece, not two. The same
  // refusal `splitSurface` already makes, at the list level.
  const cylinder = revolve(makeLine([10, 0, -20], [10, 0, 20]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const closedDir = cylinder.knotsU.length > cylinder.knotsV.length ? 'u' : 'v';
  const refused = applySplitFeatures(cylinder, [{ direction: closedDir, frac: 0.5 }]);
  assert.equal(refused.ok, false, 'a closed direction must refuse');
  assert.match(refused.reason, /CLOSED/);
  // ...and the OTHER direction still works on the same surface, so the refusal
  // is about the direction and not about the object.
  const open = applySplitFeatures(cylinder, [{ direction: closedDir === 'u' ? 'v' : 'u', frac: 0.5 }]);
  assert.ok(open.ok, `the open direction must still split: ${open.reason}`);
  assert.equal(open.pieces.length, 2);
});

test('A MALFORMED CALL REFUSES rather than throwing', () => {
  for (const bad of [null, {}, { ctrlNet: [] }]) {
    let out = null;
    assert.doesNotThrow(() => { out = applySplitFeatures(bad, [{ direction: 'u', frac: 0.5 }]); });
    assert.equal(out.ok, false);
  }
  // Entries that are not entries are ignored, not fatal.
  const r = applySplitFeatures(SADDLE, [null, { direction: 'x', frac: 0.5 }, 'nonsense']);
  assert.ok(r.ok);
  assert.equal(r.pieces.length, 1);
});
