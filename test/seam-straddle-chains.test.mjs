// A LOOP THAT STRADDLES A SEAM IS SEVERAL PIECES IN UV, AND THE POINT OF
// seamStraddleChains IS TO SAY SO EXACTLY.
//
// A closed region sitting across a closed direction's seam crosses it an EVEN
// number of times: out and back. It is not a wrap, and there is no single
// chain that describes it in the surface's own parameters — the domain
// rectangle genuinely cuts it in two. seamCrossingSpine refuses it by name
// for that reason; this function returns the pieces instead.
//
// THE INVARIANT THAT ACTUALLY PROVES IT, and the reason this file exists
// rather than a shape check: closing each returned chain along its own edge
// of the domain gives one polygon per piece, and their areas must sum to the
// area of the whole region measured with the seam taken out of the way. That
// is checkable against ground truth the function never sees, unlike "two
// chains came back", which a wrong split would also satisfy.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeLine, revolve } from '../kernel/primitives.mjs';
import { surfaceClosure } from '../kernel/surface.mjs';
import { seamStraddleChains, seamCrossingSpine, signedArea2D } from '../kernel/trim.mjs';

// A cylinder wall: closed in whichever direction the revolve sweeps, open in
// the other. Which one that is falls out of surfaceClosure rather than being
// assumed, so this fixture cannot quietly test the wrong axis.
function wall() {
  return revolve(makeLine([12, 0, -30], [12, 0, 30]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
}

function axisOf(srf) {
  const c = surfaceClosure(srf);
  assert.ok(c.closedU || c.closedV, 'the fixture wall wraps a closed direction');
  const ai = c.closedU ? 0 : 1;
  const knots = ai === 0 ? srf.knotsU : srf.knotsV;
  const oKnots = ai === 0 ? srf.knotsV : srf.knotsU;
  return {
    ai, oi: 1 - ai,
    aMin: knots[0], aMax: knots[knots.length - 1],
    oMin: oKnots[0], oMax: oKnots[oKnots.length - 1],
  };
}

// A circle in the surface's own parameters, centered at `aCentre` on the wrap
// axis, with every point brought back into the domain — which is what a real
// projection returns, and what makes the chain discontinuous.
function uvCircle(dom, aCentre, oCentre, ra, ro, n) {
  const span = dom.aMax - dom.aMin;
  const wrap = (x) => {
    let t = (x - dom.aMin) % span;
    if (t < 0) t += span;
    return dom.aMin + t;
  };
  const pts = [];
  for (let i = 0; i < n; i++) {
    const th = (2 * Math.PI * i) / n;
    const p = [0, 0];
    p[dom.ai] = wrap(aCentre + ra * Math.cos(th));
    p[dom.oi] = oCentre + ro * Math.sin(th);
    pts.push(p);
  }
  return pts;
}

// The same circle with no wrapping applied: the region's true shape, laid out
// as if the seam were not in the way. Ground truth for the area invariant.
function uvCircleUnwrapped(dom, aCentre, oCentre, ra, ro, n) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const th = (2 * Math.PI * i) / n;
    const p = [0, 0];
    p[dom.ai] = aCentre + ra * Math.cos(th);
    p[dom.oi] = oCentre + ro * Math.sin(th);
    pts.push(p);
  }
  return pts;
}

test('SEAM STRADDLE: a loop across the seam returns one chain per piece, each ending on the domain boundary', () => {
  const srf = wall();
  const dom = axisOf(srf);
  const oMid = (dom.oMin + dom.oMax) / 2;
  const ra = (dom.aMax - dom.aMin) * 0.12;
  const ro = (dom.oMax - dom.oMin) * 0.12;
  const loop = uvCircle(dom, dom.aMin, oMid, ra, ro, 64);

  // The fixture really is the case under test, not a wrap: seamCrossingSpine
  // must refuse it, by that name.
  const wrapAttempt = seamCrossingSpine(loop, srf);
  assert.equal(wrapAttempt.ok, false);
  assert.equal(wrapAttempt.code, 'seam-straddle');

  const r = seamStraddleChains(loop, srf);
  assert.ok(r.ok, r.reason);
  assert.equal(r.chains.length, 2, 'a circle on the seam is two pieces in UV');
  assert.equal(r.axisIndex, dom.ai);

  for (const chain of r.chains) {
    assert.ok(chain.length >= 3, 'each piece keeps its own samples');
    const first = chain[0], last = chain[chain.length - 1];
    // Both ends land exactly ON the domain boundary — the property the
    // arrangement needs, and the one a raw wrapped chain lacks.
    for (const end of [first, last]) {
      const onEdge = Math.abs(end[dom.ai] - dom.aMin) < 1e-12 || Math.abs(end[dom.ai] - dom.aMax) < 1e-12;
      assert.ok(onEdge, `chain end sits on the wrap axis boundary (got ${end[dom.ai]})`);
    }
    // ...and on the SAME edge as each other, because this piece never crosses
    // the seam: it is one side of it.
    assert.ok(Math.abs(first[dom.ai] - last[dom.ai]) < 1e-12, 'both ends of a piece are on one edge');
    // Every interior sample stays on that side too.
    for (let i = 1; i < chain.length - 1; i++) {
      const d = Math.abs(chain[i][dom.ai] - first[dom.ai]);
      assert.ok(d < (dom.aMax - dom.aMin) / 2, 'no piece jumps the seam internally');
    }
  }

  // The two pieces sit on OPPOSITE edges — together they are the whole region.
  const edges = r.chains.map((c) => c[0][dom.ai]).sort((a, b) => a - b);
  assert.ok(Math.abs(edges[0] - dom.aMin) < 1e-12 && Math.abs(edges[1] - dom.aMax) < 1e-12,
    'one piece on each edge of the wrap axis');
});

test('SEAM STRADDLE: the pieces account for the whole region — closed areas sum to the unwrapped area', () => {
  const srf = wall();
  const dom = axisOf(srf);
  const oMid = (dom.oMin + dom.oMax) / 2;
  const ra = (dom.aMax - dom.aMin) * 0.12;
  const ro = (dom.oMax - dom.oMin) * 0.12;
  const N = 96;

  const r = seamStraddleChains(uvCircle(dom, dom.aMin, oMid, ra, ro, N), srf);
  assert.ok(r.ok, r.reason);

  // Each chain closes along its own edge, so the chain read as a polygon IS
  // the piece: the closing segment shoelace supplies runs straight down that
  // edge, which is exactly where the seam is.
  const total = r.chains.reduce((s, c) => s + Math.abs(signedArea2D(c)), 0);
  const truth = Math.abs(signedArea2D(uvCircleUnwrapped(dom, dom.aMin, oMid, ra, ro, N)));
  assert.ok(Math.abs(total - truth) < truth * 1e-9,
    `pieces account for the whole region (${total} vs ${truth})`);
});

test('SEAM STRADDLE: an off-center straddle still splits into exactly the two real pieces', () => {
  const srf = wall();
  const dom = axisOf(srf);
  const oMid = (dom.oMin + dom.oMax) / 2;
  const ra = (dom.aMax - dom.aMin) * 0.1;
  const ro = (dom.oMax - dom.oMin) * 0.1;
  const N = 96;
  // Center pushed off the seam so the two pieces are genuinely unequal — an
  // even split would pass a symmetric fixture for the wrong reason.
  const centre = dom.aMin + ra * 0.4;

  const r = seamStraddleChains(uvCircle(dom, centre, oMid, ra, ro, N), srf);
  assert.ok(r.ok, r.reason);
  assert.equal(r.chains.length, 2);
  const areas = r.chains.map((c) => Math.abs(signedArea2D(c))).sort((a, b) => a - b);
  assert.ok(areas[0] > 0 && areas[1] / areas[0] > 1.5, 'the fixture is genuinely lopsided');
  const total = areas[0] + areas[1];
  const truth = Math.abs(signedArea2D(uvCircleUnwrapped(dom, centre, oMid, ra, ro, N)));
  assert.ok(Math.abs(total - truth) < truth * 1e-9, `pieces still account for the whole (${total} vs ${truth})`);
});

test('SEAM STRADDLE: a once-around wrap is refused here by name, not silently mishandled', () => {
  const srf = wall();
  const dom = axisOf(srf);
  const oMid = (dom.oMin + dom.oMax) / 2;
  // A loop that genuinely goes all the way around: the wrap case, which
  // belongs to seamCrossingSpine.
  const n = 64;
  const loop = [];
  for (let i = 0; i < n; i++) {
    const t = i / n;
    const p = [0, 0];
    p[dom.ai] = dom.aMin + t * (dom.aMax - dom.aMin);
    p[dom.oi] = oMid + (dom.oMax - dom.oMin) * 0.05 * Math.sin(2 * Math.PI * t);
    loop.push(p);
  }
  const r = seamStraddleChains(loop, srf);
  assert.equal(r.ok, false);
  assert.equal(r.code, 'not-a-straddle');
  // ...and the wrap path accepts the same loop, so the two are a genuine
  // partition of the cases rather than both refusing.
  assert.equal(seamCrossingSpine(loop, srf).ok, true);
});

test('SEAM STRADDLE: a loop nowhere near a seam is reported as not entangled at all', () => {
  const srf = wall();
  const dom = axisOf(srf);
  const oMid = (dom.oMin + dom.oMax) / 2;
  const aMid = (dom.aMin + dom.aMax) / 2;
  const loop = uvCircle(dom, aMid, oMid, (dom.aMax - dom.aMin) * 0.1, (dom.oMax - dom.oMin) * 0.1, 48);
  const r = seamStraddleChains(loop, srf);
  assert.equal(r.ok, false);
  assert.equal(r.code, 'no-seam-crossing');
});
