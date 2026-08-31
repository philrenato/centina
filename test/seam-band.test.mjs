// SEAM-CROSSING (WRAPPED) INTERSECTION LOOP -> TWO BAND LOOPS.
//
// The wall the trimmed-surface spec's own "SEAM/POLE PROBLEM"
// named, and the one that stopped IntersectSplit on that doc's OWN
// recommended reference fixture. The proof that matters here is not "two
// loops came back" — it is that the two bands genuinely PARTITION the
// parametric rectangle (their areas sum to the whole domain, and every
// one of them is a valid simple loop), because that is what makes them a
// real split rather than two plausible-looking polygons.
import test from 'node:test';
import assert from 'node:assert/strict';
import { seamBandLoops, signedArea2D, trimLoopsValid, trivialTrimLoop, pointInUVPolygon } from '../kernel/trim.mjs';
import { intersectSurfaces } from '../kernel/ssi.mjs';
import { surfaceClosure } from '../kernel/surface.mjs';
import { makeCircle, extrude } from '../kernel/primitives.mjs';

// The SAME two-cylinder fixture test/ssi.test.mjs already proves the
// marching against — reused deliberately rather than rebuilt, so this
// tests the real reference case and not a friendlier stand-in.
function makeCylinderZ(R, halfHeight) {
  const profile = makeCircle([0, 0, -halfHeight], [1, 0, 0], [0, 1, 0], R, 4);
  return extrude(profile, [0, 0, 1], 2 * halfHeight);
}
function makeCylinderXOffset(R, halfLength, zOffset) {
  const profile = makeCircle([-halfLength, 0, zOffset], [0, 1, 0], [0, 0, 1], R, 4);
  return extrude(profile, [1, 0, 0], 2 * halfLength);
}
function domainOf(srf) {
  return {
    uMin: srf.knotsU[0], uMax: srf.knotsU[srf.knotsU.length - 1],
    vMin: srf.knotsV[0], vMax: srf.knotsV[srf.knotsV.length - 1],
  };
}
// A synthetic once-around wrap: v varies smoothly with u across the FULL
// closed u range, sampled as a cyclic list exactly the way marchDirection
// hands one back (so the seam shows up as one big jump, not as an ordered
// open chain that would make the unwrap trivially easy).
function syntheticWrap(srf, samples = 40, amp = 0.25) {
  const { uMin, uMax, vMin, vMax } = domainOf(srf);
  const uSpan = uMax - uMin, vMid = (vMin + vMax) / 2, vAmp = (vMax - vMin) * amp;
  const pts = [];
  for (let i = 0; i < samples; i++) {
    const t = i / samples;
    // Deliberately start PART WAY around, so the seam crossing lands in the
    // middle of the sample list rather than conveniently at index 0.
    const u = uMin + ((t + 0.37) % 1) * uSpan;
    pts.push([u, vMid + vAmp * Math.sin(2 * Math.PI * t)]);
  }
  return pts;
}

test('a synthetic once-around wrap splits into two bands that PARTITION the domain', () => {
  const cyl = makeCylinderZ(5, 9);
  const { closedU, closedV } = surfaceClosure(cyl);
  assert.ok(closedU || closedV, 'fixture must actually be closed in some direction');
  const loop = syntheticWrap(cyl);
  const r = seamBandLoops(loop, cyl);
  assert.ok(r.ok, `expected a band split, got: ${r.reason}`);
  assert.equal(r.loops.length, 2);

  const { uMin, uMax, vMin, vMax } = domainOf(cyl);
  const domainArea = Math.abs((uMax - uMin) * (vMax - vMin));
  const a0 = Math.abs(signedArea2D(r.loops[0]));
  const a1 = Math.abs(signedArea2D(r.loops[1]));
  // THE REAL INVARIANT: a split loses nothing and double-counts nothing.
  assert.ok(Math.abs((a0 + a1) - domainArea) < 1e-6 * domainArea,
    `bands must partition the domain: ${a0} + ${a1} != ${domainArea}`);
  assert.ok(a0 > 1e-9 && a1 > 1e-9, 'neither band may be degenerate');
});

test('each band is a valid simple loop, wound like trivialTrimLoop', () => {
  const cyl = makeCylinderZ(5, 9);
  const r = seamBandLoops(syntheticWrap(cyl), cyl);
  assert.ok(r.ok);
  const outerSign = Math.sign(signedArea2D(trivialTrimLoop(cyl)));
  for (const band of r.loops) {
    const v = trimLoopsValid([band]);
    assert.ok(v.ok, `band invalid: ${v.reason}`);
    assert.equal(Math.sign(signedArea2D(band)), outerSign, 'band winding must match the domain rect');
  }
});

test('the two bands are genuinely opposite sides of the curve, not the same region twice', () => {
  const cyl = makeCylinderZ(5, 9);
  const { uMin, uMax, vMin, vMax } = domainOf(cyl);
  const r = seamBandLoops(syntheticWrap(cyl, 40, 0.2), cyl);
  assert.ok(r.ok);
  const [low, high] = r.loops;
  // A point hard against each v extreme must be in exactly ONE band, and
  // not the same one — the cheapest honest proof they are complementary
  // rather than two copies of a plausible-looking polygon.
  const uProbe = uMin + (uMax - uMin) * 0.31;
  const vLo = vMin + (vMax - vMin) * 0.01, vHi = vMax - (vMax - vMin) * 0.01;
  // pointInUVPolygon reports 'inside' | 'outside' | 'boundary' — a third
  // honest state, not a boolean, so compare against the real vocabulary.
  assert.notEqual(pointInUVPolygon(low, uProbe, vLo), 'outside', 'the low band must contain a point near vMin');
  assert.notEqual(pointInUVPolygon(high, uProbe, vHi), 'outside', 'the high band must contain a point near vMax');
  assert.equal(pointInUVPolygon(high, uProbe, vLo), 'outside', 'the high band must NOT contain a point near vMin');
  assert.equal(pointInUVPolygon(low, uProbe, vHi), 'outside', 'the low band must NOT contain a point near vMax');
});

test('THE REFERENCE CASE: the smaller cylinder of a real pipe-through-pipe now splits', () => {
  // the recommended fixture for it, and
  // the exact configuration that made IntersectSplit refuse: R2 < R1, so
  // the intersection necessarily wraps the SMALLER cylinder completely.
  const R1 = 5, R2 = 3, z0 = 2;
  const big = makeCylinderZ(R1, 9);
  const small = makeCylinderXOffset(R2, 9, z0);
  const res = intersectSurfaces(big, small, {});
  assert.ok(res.ok, `SSI must find the intersection: ${res.reason}`);
  assert.ok(res.closed, 'this configuration closes into a loop');
  const samples = res.samples.slice(0, -1);
  const loopSmall = samples.map((s) => [s.u2, s.v2]);

  const r = seamBandLoops(loopSmall, small);
  assert.ok(r.ok, `the smaller cylinder must now split rather than refuse: ${r.reason}`);
  assert.equal(r.loops.length, 2);

  const { uMin, uMax, vMin, vMax } = domainOf(small);
  const domainArea = Math.abs((uMax - uMin) * (vMax - vMin));
  const total = Math.abs(signedArea2D(r.loops[0])) + Math.abs(signedArea2D(r.loops[1]));
  assert.ok(Math.abs(total - domainArea) < 1e-6 * domainArea,
    `bands must partition the smaller cylinder's domain: ${total} vs ${domainArea}`);
  for (const band of r.loops) assert.ok(trimLoopsValid([band]).ok);
});

test('an ordinary NON-wrapping loop is refused by name, not silently banded', () => {
  const cyl = makeCylinderZ(5, 9);
  const { uMin, uMax, vMin, vMax } = domainOf(cyl);
  const cu = uMin + (uMax - uMin) * 0.5, cv = (vMin + vMax) / 2;
  const ru = (uMax - uMin) * 0.15, rv = (vMax - vMin) * 0.15;
  const loop = [];
  for (let i = 0; i < 24; i++) {
    const t = (i / 24) * 2 * Math.PI;
    loop.push([cu + ru * Math.cos(t), cv + rv * Math.sin(t)]);
  }
  const r = seamBandLoops(loop, cyl);
  assert.equal(r.ok, false);
  assert.match(r.reason, /does not cross/i);
});

test('a loop that straddles the seam and returns is refused as its OWN case', () => {
  // Crosses the seam TWICE — an ordinary hole sitting on the seam, which
  // genuinely does need a periodic clip. Must not be mistaken for a wrap.
  const cyl = makeCylinderZ(5, 9);
  const { uMin, uMax, vMin, vMax } = domainOf(cyl);
  const cv = (vMin + vMax) / 2, rv = (vMax - vMin) * 0.15;
  const ru = (uMax - uMin) * 0.12;
  const loop = [];
  for (let i = 0; i < 24; i++) {
    const t = (i / 24) * 2 * Math.PI;
    // centered exactly ON the seam (u = uMin), so half the loop lands at the
    // top of the u range and half at the bottom
    let u = uMin + ru * Math.cos(t);
    if (u < uMin) u += (uMax - uMin);
    loop.push([u, cv + rv * Math.sin(t)]);
  }
  const r = seamBandLoops(loop, cyl);
  assert.equal(r.ok, false);
  assert.match(r.reason, /returns|periodic/i);
});
