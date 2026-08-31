// TOUCHING BOTH SEAMS IS NOT THE SAME AS WRAPPING BOTH, and the difference is
// net winding.
//
// A torus is closed in u AND v, so its two seams meet at the corners of the
// domain rectangle. A small region sitting over one of those corners crosses
// each seam twice and winds around neither: contractible, ordinary geometry
// that happens to land in an awkward place in the parameters. It used to be
// refused as a "torus-like double wrap" — a false statement about the
// topology — because the test asked only whether both directions carried
// jumps, which a straddle does too.
//
// THE INVARIANT THAT PROVES THE SPLIT, rather than a shape check a wrong split
// would also satisfy: each returned chain runs between two ADJACENT edges of
// the rectangle, so closing it through the corner between them gives one
// polygon per piece — and the four polygons' areas must sum to the area of the
// same region laid out with no seam in the way. That is ground truth the
// function never sees.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeCircle, revolve } from '../kernel/primitives.mjs';
import { surfaceClosure } from '../kernel/surface.mjs';
import { seamCrossingSpine, seamStraddleChains, seamDoubleStraddleChains, signedArea2D } from '../kernel/trim.mjs';

// A REAL torus, revolved rather than declared: closed in both directions, and
// curved in both, so nothing here rests on a trivially flat fixture.
function torus(R = 30, r = 10) {
  return revolve(makeCircle([R, 0, 0], [1, 0, 0], [0, 0, 1], r, 4), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
}

function domainOf(srf) {
  const c = surfaceClosure(srf);
  assert.ok(c.closedU && c.closedV, 'the fixture is closed in BOTH directions — the whole point of it');
  return {
    uMin: srf.knotsU[0], uMax: srf.knotsU[srf.knotsU.length - 1],
    vMin: srf.knotsV[0], vMax: srf.knotsV[srf.knotsV.length - 1],
  };
}

// A circle in the surface's own parameters with every point brought back into
// the domain, which is what a real projection returns and what makes the chain
// discontinuous at both seams.
function cornerCircle(dom, ru, rv, n) {
  const uSpan = dom.uMax - dom.uMin, vSpan = dom.vMax - dom.vMin;
  const wrap = (x, min, span) => min + ((((x - min) % span) + span) % span);
  const pts = [];
  for (let i = 0; i < n; i++) {
    const th = (2 * Math.PI * i) / n;
    pts.push([wrap(dom.uMin + ru * Math.cos(th), dom.uMin, uSpan), wrap(dom.vMin + rv * Math.sin(th), dom.vMin, vSpan)]);
  }
  return pts;
}

// The same circle with no wrapping: the region's true shape. Ground truth.
function unwrappedCircle(dom, ru, rv, n) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const th = (2 * Math.PI * i) / n;
    pts.push([dom.uMin + ru * Math.cos(th), dom.vMin + rv * Math.sin(th)]);
  }
  return pts;
}

const onU = (p, dom) => Math.abs(p[0] - dom.uMin) < 1e-12 || Math.abs(p[0] - dom.uMax) < 1e-12;
const onV = (p, dom) => Math.abs(p[1] - dom.vMin) < 1e-12 || Math.abs(p[1] - dom.vMax) < 1e-12;

test('DOUBLE STRADDLE: a region over the seams\' corner is refused BY THE RIGHT NAME, with its winding measured', () => {
  const srf = torus();
  const dom = domainOf(srf);
  const loop = cornerCircle(dom, (dom.uMax - dom.uMin) * 0.1, (dom.vMax - dom.vMin) * 0.1, 64);

  const spine = seamCrossingSpine(loop, srf);
  assert.equal(spine.ok, false);
  assert.equal(spine.code, 'double-straddle', 'not a wrap — it crosses both seams and returns along each');
  assert.deepEqual(spine.net, [0, 0], 'net winding is zero in both directions, which is what makes it contractible');

  // The single-seam routines correctly decline it: choosing between two
  // genuinely different topologies by accident is the failure this prevents.
  const single = seamStraddleChains(loop, srf);
  assert.equal(single.ok, false);
  assert.equal(single.code, 'double-straddle');
});

test('DOUBLE STRADDLE: a genuine wrap in both directions is still refused, and still called a double wrap', () => {
  const srf = torus();
  const dom = domainOf(srf);
  const uSpan = dom.uMax - dom.uMin, vSpan = dom.vMax - dom.vMin;
  // A (1,1) curve on the torus: once around u and once around v together, so
  // both nets are non-zero. This is the topology the refusal is really about.
  const n = 96;
  const loop = [];
  for (let i = 0; i < n; i++) {
    loop.push([dom.uMin + (i / n) * uSpan, dom.vMin + (i / n) * vSpan]);
  }
  const r = seamCrossingSpine(loop, srf);
  assert.equal(r.ok, false);
  assert.equal(r.code, 'double-wrap');
  assert.ok(r.net[0] !== 0 && r.net[1] !== 0, `both directions genuinely wind (got ${r.net})`);
  // And the handler declines it rather than producing chains for it.
  assert.equal(seamDoubleStraddleChains(loop, srf).code, 'not-a-double-straddle');
});

test('DOUBLE STRADDLE: the split returns one chain per piece, each running between two adjacent domain edges', () => {
  const srf = torus();
  const dom = domainOf(srf);
  const loop = cornerCircle(dom, (dom.uMax - dom.uMin) * 0.1, (dom.vMax - dom.vMin) * 0.1, 64);

  const r = seamDoubleStraddleChains(loop, srf);
  assert.ok(r.ok, r.reason);
  assert.equal(r.chains.length, 4, 'a circle over the corner is four pieces in UV — one per quadrant');

  for (const chain of r.chains) {
    assert.ok(chain.length >= 3, 'each piece keeps its own samples');
    const first = chain[0], last = chain[chain.length - 1];
    for (const end of [first, last]) {
      assert.ok(onU(end, dom) || onV(end, dom), `chain end sits on the domain boundary (got ${end})`);
    }
    // One end on each seam: a corner piece leaves through the u seam and
    // through the v seam, never twice through the same one.
    const ends = [first, last];
    assert.ok(ends.some((p) => onU(p, dom)) && ends.some((p) => onV(p, dom)),
      `a corner piece reaches both seams (got ${first} and ${last})`);
    // No interior sample jumps either seam — each piece is one connected run.
    for (let i = 1; i < chain.length; i++) {
      assert.ok(Math.abs(chain[i][0] - chain[i - 1][0]) < (dom.uMax - dom.uMin) / 2, 'no piece jumps the u seam internally');
      assert.ok(Math.abs(chain[i][1] - chain[i - 1][1]) < (dom.vMax - dom.vMin) / 2, 'no piece jumps the v seam internally');
    }
  }
});

test('DOUBLE STRADDLE: the pieces account for the whole region — closed areas sum to the unwrapped area', () => {
  const srf = torus();
  const dom = domainOf(srf);
  const ru = (dom.uMax - dom.uMin) * 0.1, rv = (dom.vMax - dom.vMin) * 0.1;
  const N = 128;

  const r = seamDoubleStraddleChains(cornerCircle(dom, ru, rv, N), srf);
  assert.ok(r.ok, r.reason);

  // A piece is closed THROUGH THE CORNER its two ends straddle, because that
  // corner is where the two seams meet and is genuinely part of its boundary.
  // Closing with the bare chord instead would cut each corner off and under-
  // report every area by the same triangle.
  const total = r.chains.reduce((sum, chain) => {
    const a = chain[0], b = chain[chain.length - 1];
    const cornerU = Math.abs(a[0] - dom.uMin) < 1e-12 || Math.abs(b[0] - dom.uMin) < 1e-12 ? dom.uMin : dom.uMax;
    const cornerV = Math.abs(a[1] - dom.vMin) < 1e-12 || Math.abs(b[1] - dom.vMin) < 1e-12 ? dom.vMin : dom.vMax;
    return sum + Math.abs(signedArea2D([...chain, [cornerU, cornerV]]));
  }, 0);
  const truth = Math.abs(signedArea2D(unwrappedCircle(dom, ru, rv, N)));
  assert.ok(Math.abs(total - truth) < truth * 1e-9,
    `the four pieces account for the whole region (${total} vs ${truth})`);
});
