// ARC-LENGTH GRID RESOLUTION — see kernel/surface.mjs's own
// `tessellationGridResolution` header for the full derivation. Short version:
// a declared `uRes x vRes` is a count and says nothing about the shape it
// counts across, so the same grid that is well proportioned on a sphere puts
// 192 divisions along the 7mm straight wall of a disc extrusion whose
// circumference is 283mm. The function re-solves both counts for EQUAL chord
// deviation, against the declared grid's own worst deviation as the tolerance,
// and can never return a count higher than the one it was handed.
//
// The four properties asserted here are the ones the rest of the app relies on:
// a flat patch is left exactly alone, a ruled direction collapses, a bilinear
// SADDLE is NOT treated as flat (this is the case a planarity test on the
// control net gets wrong — every isocurve of that patch is a straight line),
// and a global density multiplier still scales linear density linearly.

import test from 'node:test';
import assert from 'node:assert/strict';
import { makeCircle, makeLine, makeArc, extrude, revolve } from '../kernel/primitives.mjs';
import { tessellationGridResolution, tessellationVSamples as tessellationVSamplesFor } from '../kernel/surface.mjs';
import * as surfaceMod from '../kernel/surface.mjs';

const discWall = extrude(makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 45, 4), [0, 0, 1], 7);
const flatPlane = extrude(makeLine([0, 0, 0], [100, 0, 0]), [0, 1, 0], 100);
const cone = revolve(makeLine([20, 0, 0], [0, 0, 40]), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
const sphere = revolve(makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 30, -Math.PI / 2, Math.PI),
  [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);

test('a count is never returned higher than the one handed in', () => {
  for (const srf of [discWall, flatPlane, cone, sphere]) {
    for (const [u, v] of [[96, 48], [192, 96], [442, 221], [64, 64]]) {
      const r = tessellationGridResolution(srf, u, v);
      assert.ok(r.uRes <= u && r.vRes <= v, `${r.uRes}x${r.vRes} exceeds ${u}x${v}`);
      assert.ok(r.uRes >= 1 && r.vRes >= 1);
    }
  }
});

test('a straight ruling collapses and the curved direction is untouched', () => {
  // The disc extrusion: 283mm of circumference across u, 7mm of straight
  // ruling across v. The curved direction sets the tolerance, so it solves
  // back to exactly what it was declared at.
  const r = tessellationGridResolution(discWall, 192, 96);
  assert.equal(r.uRes, 192);
  assert.ok(r.vRes <= 4, `expected a handful of divisions across a straight ruling, got ${r.vRes}`);
  // A cone's slant is straight too, so only the aspect guard divides it.
  const c = tessellationGridResolution(cone, 192, 96);
  assert.equal(c.vRes, 96);
  assert.ok(c.uRes <= 16, `expected the straight slant to collapse, got ${c.uRes}`);
});

test('a flat patch keeps the resolution it was declared with', () => {
  // A plane in a modelling application is resolved for the shape it is about
  // to be deformed into, not the flat one it starts as, so a function that
  // can only see the current shape must decline rather than collapse it.
  const r = tessellationGridResolution(flatPlane, 40, 40);
  assert.equal(r.engaged, false);
  assert.equal(r.uRes, 40);
  assert.equal(r.vRes, 40);
});

test('a bilinear saddle is not mistaken for a flat patch', () => {
  const saddle = JSON.parse(JSON.stringify(flatPlane));
  const R = saddle.ctrlNet.length - 1, C = saddle.ctrlNet[0].length - 1;
  saddle.ctrlNet[R][C][2] = 40 * (saddle.ctrlNet[R][C][3] ?? 1);
  const r = tessellationGridResolution(saddle, 40, 40);
  assert.equal(r.engaged, true);
  // Every isocurve of this patch is still a straight line, so the per-direction
  // sagittas are zero and only the TWIST term can see the shape at all.
  assert.ok(r.devU < 1e-9 && r.devV < 1e-9, `isocurves should read straight, got ${r.devU}/${r.devV}`);
  assert.ok(r.devT > 1e-3, `the twist term should see the lifted corner, got ${r.devT}`);
  assert.ok(r.uRes > 8 && r.vRes > 8, `a saddle needs real divisions both ways, got ${r.uRes}x${r.vRes}`);
});

test('a small declared grid is left alone rather than probed', () => {
  const r = tessellationGridResolution(discWall, 24, 8);
  assert.equal(r.engaged, false);
  assert.equal(r.uRes, 24);
  assert.equal(r.vRes, 8);
});

test('a global density multiplier still scales linear density linearly', () => {
  // Doubling both declared counts quarters the tolerance and doubles both
  // solved counts, so the mechanism composes with the Mesh Quality slider
  // instead of replacing it.
  const a = tessellationGridResolution(sphere, 96, 192);
  const b = tessellationGridResolution(sphere, 192, 384);
  assert.ok(Math.abs(b.uRes / a.uRes - 2) < 0.06, `${a.uRes} -> ${b.uRes}`);
  assert.ok(Math.abs(b.vRes / a.vRes - 2) < 0.06, `${a.vRes} -> ${b.vRes}`);
});

test('the solved grid meets the declared grid\'s own worst chord deviation', () => {
  for (const srf of [discWall, cone, sphere]) {
    const r = tessellationGridResolution(srf, 192, 96);
    if (!r.engaged) continue;
    const at = (nu, nv) => Math.max(r.devU * (r.probeU / nu) ** 2, r.devV * (r.probeV / nv) ** 2,
      r.devT * (r.probeU / nu) * (r.probeV / nv));
    // The solved grid is no worse than the declared one it replaces.
    assert.ok(at(r.uRes, r.vRes) <= at(192, 96) * 1.0001,
      `solved deviation ${at(r.uRes, r.vRes)} exceeds declared ${at(192, 96)}`);
  }
});

// THE V-SAMPLE FLOOR MUST NOT MANUFACTURE SLIVERS. `tessellationVSamples`
// merges a per-span density floor into the uniform list through a `Set`, which
// dedupes on exact equality only — so before the separation guard, a span one
// sample short of the floor got 47 uniform samples at k/48 and 48 forced ones
// at k/49, and the pairs at the span ends survived a millionth of a span apart.
// That is the widest-aspect triangle in the whole mesh on every closed revolve.
test('the per-span density floor never lands two samples on top of each other', () => {
  const circle = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 30, 4); // four full-multiplicity arc spans
  const revolved = revolve(makeLine([30, 0, 0], [30, 0, 60]), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
  assert.ok(circle.knots.length > 0);
  for (const vRes of [96, 120, 144, 168, 190, 191, 192, 193, 240]) {
    const vs = tessellationVSamplesFor(revolved, vRes);
    const span = vs[vs.length - 1] - vs[0];
    let worst = Infinity;
    for (let i = 0; i < vs.length - 1; i++) worst = Math.min(worst, vs[i + 1] - vs[i]);
    const uniform = span / vRes;
    assert.ok(worst > uniform * 0.02,
      `vRes ${vRes}: smallest gap ${worst} is ${(worst / uniform).toFixed(4)} of a uniform step (${vs.length} samples)`);
  }
});

test('the density floor still guarantees its per-span sample count', () => {
  const revolved = revolve(makeLine([30, 0, 0], [30, 0, 60]), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
  for (const vRes of [24, 96, 190, 192]) {
    const vs = tessellationVSamplesFor(revolved, vRes);
    const knots = revolved.knotsV, degV = revolved.degV;
    const vMin = knots[degV], vMax = knots[knots.length - 1 - degV];
    const mult = new Map();
    for (const k of knots) mult.set(k, (mult.get(k) || 0) + 1);
    const bounds = [vMin];
    for (const [k, m] of mult) if (k > vMin + 1e-9 && k < vMax - 1e-9 && m >= degV) bounds.push(k);
    bounds.push(vMax); bounds.sort((a, b) => a - b);
    if (bounds.length <= 2) continue;
    for (let s = 0; s < bounds.length - 1; s++) {
      let inside = 0;
      for (const v of vs) if (v > bounds[s] + 1e-9 && v < bounds[s + 1] - 1e-9) inside++;
      assert.ok(inside >= 48, `vRes ${vRes}, span ${s}: only ${inside} samples inside`);
    }
  }
});

// ---------------------------------------------------------------------------
// RELATIVE TOLERANCE — the mode the trimmed path needs.
//
// The default target is the DECLARED grid's own worst chord deviation, which
// makes this function a pure redistribution: it can never call a grid finer
// than it needs to be. That is right when the declared grid carries intent, and
// useless when it is a constant. The trimmed path hands every piece a flat
// 64x64, so a 5mm fillet band on a 90mm box edge solved straight back to 64x64
// and spent 8192 triangles where its arc needed about fifteen.
//
// `relTolerance` sets the target as a fraction of arc length PER DIRECTION,
// which is the part that matters: a band's bounding box is dominated by the
// 90mm it is straight along, so any single tolerance drawn from the diagonal is
// set by the direction that needs no samples at all.

// A box-edge fillet band is a rational quarter arc swept along a straight edge.
const filletBand = (r, L) => {
  const w = Math.SQRT1_2;
  const arc = [[r, 0, 0, 1], [r, r, 0, w], [0, r, 0, 1]];
  return {
    degU: 2, knotsU: [0, 0, 0, 1, 1, 1],
    degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: arc.map(([x, y, z, ww]) => [[x, y, 0, ww], [x, y, L, ww]]),
  };
};

test('without relTolerance the solved grid is exactly what it always was', () => {
  // The pin that keeps this an opt-in: the untrimmed path passes no options and
  // must not move. A default that silently changed would be a re-tessellation
  // of every surface in the app disguised as a new feature.
  for (const srf of [discWall, flatPlane, cone, sphere, filletBand(5, 90)]) {
    for (const [u, v] of [[96, 48], [64, 64], [192, 96]]) {
      const bare = tessellationGridResolution(srf, u, v);
      const withNull = tessellationGridResolution(srf, u, v, { relTolerance: null });
      assert.equal(withNull.uRes, bare.uRes);
      assert.equal(withNull.vRes, bare.vRes);
      assert.equal(withNull.engaged, bare.engaged);
    }
  }
});

test('a fillet band collapses under a relative tolerance where it would not without one', () => {
  const srf = filletBand(5, 90);
  const bare = tessellationGridResolution(srf, 64, 64);
  assert.equal(bare.uRes, 64, 'the self-referential target agrees with whatever it is handed');
  assert.equal(bare.vRes, 64);
  const rel = tessellationGridResolution(srf, 64, 64, { relTolerance: 1e-3 });
  assert.ok(rel.engaged);
  assert.ok(rel.uRes * rel.vRes < bare.uRes * bare.vRes / 4,
    `expected a band to cost under a quarter of the declared grid, got ${rel.uRes}x${rel.vRes}`);
});

test('the arc direction is solved scale-invariantly', () => {
  // The same quarter arc needs the same segment count for the same RELATIVE
  // accuracy whatever its radius; a count that moved with radius would mean the
  // tolerance was being taken from something other than that direction's arc.
  // Declared well above what the arc needs, so equality can only be reached by
  // solving. Against a 64x64 the counts agree at 64 whether the solver ran or
  // not, and the test passes with the whole mode switched off.
  const counts = [2, 5, 20, 50].map((r) => tessellationGridResolution(filletBand(r, 90), 256, 256, { relTolerance: 1e-3 }).uRes);
  assert.equal(new Set(counts).size, 1, `radius changed the arc count: ${counts.join(', ')}`);
  assert.ok(counts[0] < 64, `expected the arc to be solved well under the declared grid, got ${counts[0]}`);
});

test('a relative tolerance never returns a count higher than the declared grid', () => {
  for (const srf of [discWall, cone, sphere, filletBand(5, 90), filletBand(2, 200)]) {
    for (const [u, v] of [[64, 64], [16, 16], [8, 200]]) {
      const r = tessellationGridResolution(srf, u, v, { relTolerance: 1e-4 });
      assert.ok(r.uRes <= u, `uRes ${r.uRes} exceeded declared ${u}`);
      assert.ok(r.vRes <= v, `vRes ${r.vRes} exceeded declared ${v}`);
    }
  }
});

test('a tighter relative tolerance never returns a coarser grid', () => {
  const srf = filletBand(5, 90);
  let prev = 0;
  for (const rel of [4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4]) {
    const r = tessellationGridResolution(srf, 256, 256, { relTolerance: rel });
    assert.ok(r.uRes >= prev, `tolerance ${rel} went backwards: ${r.uRes} after ${prev}`);
    prev = r.uRes;
  }
});

test('the solved grid actually meets the relative tolerance it was asked for', () => {
  // The claim the function makes, checked against the surface rather than
  // against its own arithmetic: sample the arc at the solved count and measure
  // the real sagitta of each chord.
  const { surfacePoint } = surfaceMod;
  for (const r of [2, 5, 20]) {
    for (const rel of [2e-3, 1e-3, 5e-4]) {
      const srf = filletBand(r, 90);
      const a = tessellationGridResolution(srf, 256, 256, { relTolerance: rel });
      const arcLen = r * Math.PI / 2;
      let worst = 0;
      for (let i = 0; i < a.uRes; i++) {
        const u0 = i / a.uRes, u1 = (i + 1) / a.uRes;
        const p0 = surfacePoint(srf, u0, 0), p1 = surfacePoint(srf, u1, 0);
        const mid = surfacePoint(srf, (u0 + u1) / 2, 0);
        const chord = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        const to = [mid[0] - p0[0], mid[1] - p0[1], mid[2] - p0[2]];
        const cr = [chord[1] * to[2] - chord[2] * to[1], chord[2] * to[0] - chord[0] * to[2], chord[0] * to[1] - chord[1] * to[0]];
        const sag = Math.hypot(cr[0], cr[1], cr[2]) / Math.hypot(chord[0], chord[1], chord[2]);
        if (sag > worst) worst = sag;
      }
      assert.ok(worst <= arcLen * rel * 1.5,
        `r=${r} relTol=${rel}: measured sagitta ${worst.toExponential(3)} exceeded ${(arcLen * rel).toExponential(3)}`);
    }
  }
});
