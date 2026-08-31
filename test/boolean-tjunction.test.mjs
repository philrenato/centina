import test from 'node:test';
import assert from 'node:assert';
import { booleanSolids } from '../kernel/boolean.mjs';
import { fragmentBoundaries3D, insertTJunctionVertices, keepFragments } from '../kernel/booleansew.mjs';
import { splitFaceByCurves } from '../kernel/facesplit.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { projectPointsToSurfaceUV, trivialTrimLoop } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';

// ---------------------------------------------------------------------------
// THE FIXTURE: a 5-pointed STAR PRISM cut by a horizontal slab that passes
// clean through it. Chosen deliberately over another box-on-box pair — a star
// is genuinely non-convex, its ten side faces meet the slab's caps along ten
// separate cut curves, and each cap is split by all ten at once. That is what
// produces the T-junction this file exists for: a face is paved at ITS OWN
// crossings, so the cap can carry a vertex partway along a cut edge that the
// side face has no reason to place, and the two sides then disagree about how
// that shared edge is subdivided even though every vertex they DO share
// matches to machine precision.
// ---------------------------------------------------------------------------

function quadSurface(p00, p10, p11, p01) {
  const w = (p) => [p[0], p[1], p[2], 1];
  return {
    degU: 1, degV: 1,
    knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[w(p00), w(p01)], [w(p10), w(p11)]],
  };
}

function starPoints(points, outerR, innerR) {
  const out = [];
  for (let i = 0; i < points * 2; i++) {
    const a = (i / (points * 2)) * Math.PI * 2 + Math.PI / 2;
    const r = i % 2 ? innerR : outerR;
    out.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  return out;
}

const CAP_HALF = 25;
const capPlane = (z) => quadSurface(
  [-CAP_HALF, -CAP_HALF, z], [CAP_HALF, -CAP_HALF, z],
  [CAP_HALF, CAP_HALF, z], [-CAP_HALF, CAP_HALF, z],
);
const capTrimLoop = (pts) => pts.map(([x, y]) => [(x + CAP_HALF) / (2 * CAP_HALF), (y + CAP_HALF) / (2 * CAP_HALF)]);

function boxFaces(lo, hi) {
  const c = (x, y, z) => [x, y, z];
  return [
    { srf: quadSurface(c(lo[0], lo[1], lo[2]), c(hi[0], lo[1], lo[2]), c(hi[0], hi[1], lo[2]), c(lo[0], hi[1], lo[2])) },
    { srf: quadSurface(c(lo[0], lo[1], hi[2]), c(hi[0], lo[1], hi[2]), c(hi[0], hi[1], hi[2]), c(lo[0], hi[1], hi[2])) },
    { srf: quadSurface(c(lo[0], lo[1], lo[2]), c(hi[0], lo[1], lo[2]), c(hi[0], lo[1], hi[2]), c(lo[0], lo[1], hi[2])) },
    { srf: quadSurface(c(lo[0], hi[1], lo[2]), c(hi[0], hi[1], lo[2]), c(hi[0], hi[1], hi[2]), c(lo[0], hi[1], hi[2])) },
    { srf: quadSurface(c(lo[0], lo[1], lo[2]), c(lo[0], hi[1], lo[2]), c(lo[0], hi[1], hi[2]), c(lo[0], lo[1], hi[2])) },
    { srf: quadSurface(c(hi[0], lo[1], lo[2]), c(hi[0], hi[1], lo[2]), c(hi[0], hi[1], hi[2]), c(hi[0], lo[1], hi[2])) },
  ];
}

function tessellate(faces, res) {
  const tris = [];
  for (const f of faces) {
    for (const tri of tessellateTrimmedSurface(f.srf, f.trimLoop ?? trivialTrimLoop(f.srf), res, res, f.trimHoles ?? [])) {
      tris.push(tri.map((v) => v.position));
    }
  }
  return tris;
}

// The curves are found by a REAL face-pair SSI sweep, not hand-derived — this
// fixture exists to exercise what the marcher's own discretization does to the
// sew, so hand-writing the curves would test the wrong thing.
function buildFixture() {
  const pts = starPoints(5, 20, 8);
  const faces = [];
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    faces.push({ srf: quadSurface([a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], 30], [a[0], a[1], 30]) });
  }
  faces.push({ srf: capPlane(0), trimLoop: capTrimLoop(pts) });
  faces.push({ srf: capPlane(30), trimLoop: capTrimLoop(pts) });
  const star = { faces, triangles: tessellate(faces, 8) };

  const slabFaces = boxFaces([-40, -40, 10], [40, 40, 20]);
  const slab = { faces: slabFaces, triangles: tessellate(slabFaces, 2) };

  const curves = [];
  for (let i = 0; i < star.faces.length; i++) {
    for (let j = 0; j < slab.faces.length; j++) {
      let r;
      try { r = intersectSurfacesComplete(star.faces[i].srf, slab.faces[j].srf); } catch { continue; }
      if (!r.ok) continue;
      for (const comp of r.components) curves.push({ samples: comp.samples.map((s) => s.point), faceA: i, faceB: j });
    }
  }
  return { star, slab, curves };
}

test('a T-junction is inserted from the far side and nowhere else', () => {
  // Two loops sharing an edge from (0,0,0) to (10,0,0). The second subdivides
  // it at (4,0,0); the first does not. That is the whole phenomenon.
  const loops = [
    [[0, 0, 0], [10, 0, 0], [10, 5, 0]],
    [[0, 0, 0], [4, 0, 0], [10, 0, 0], [10, -5, 0]],
  ];
  const out = insertTJunctionVertices(loops, 1e-4);

  assert.equal(out[0].length, 4, 'the undivided loop gains exactly the one missing vertex');
  assert.deepEqual(out[0][1], [4, 0, 0], 'inserted at the right place, in the right order');
  assert.deepEqual(out[1], loops[1], 'the loop that already had it is untouched');

  // Nothing is invented: every point out is a point that was already in.
  const wasIn = new Set(loops.flat().map((p) => p.join(',')));
  for (const p of out.flat()) assert.ok(wasIn.has(p.join(',')), `${p} was not an existing vertex`);
});

test('a vertex at an edge endpoint is not an interior split', () => {
  const loops = [
    [[0, 0, 0], [10, 0, 0], [10, 5, 0]],
    [[0, 0, 0], [10, 0, 0], [10, -5, 0]],
  ];
  const out = insertTJunctionVertices(loops, 1e-4);
  assert.deepEqual(out, loops, 'shared corners are shared corners, not T-junctions');
});

test('a vertex off the edge is left alone', () => {
  // Every point of the second loop is deliberately clear of ALL THREE edges of
  // the first, closing edge included — an earlier draft put (8,4,0) exactly on
  // the closing edge's own line (y = x/2) by accident and read as a bug in the
  // pass rather than in the fixture.
  const loops = [
    [[0, 0, 0], [10, 0, 0], [10, 5, 0]],
    [[4, 1, 0], [8, 1, 0], [8, 2, 0]],
  ];
  const out = insertTJunctionVertices(loops, 1e-4);
  assert.deepEqual(out, loops, 'a nearby vertex is not on the edge and must not be pulled onto it');
});

test('a vertex from an unrelated face MERELY PASSING NEAR an edge is not spliced into it', () => {
  // The off-edge test is not the weld test, and this is the case that
  // separates them. The stray vertex sits half a weld tolerance off the
  // edge — close enough that reusing the weld tolerance here would splice
  // it in, bending a straight boundary toward a face it shares nothing
  // with and welding the two into a vertex-level pinch that the edge-based
  // manifold checks downstream cannot see.
  const TOL = 1e-4;
  const loops = [
    [[0, 0, 0], [10, 0, 0], [10, 5, 0]],
    [[4, TOL / 2, 0], [8, 3, 0], [4, 3, 0]],
  ];
  const out = insertTJunctionVertices(loops, TOL);
  assert.deepEqual(out[0], loops[0], 'the near-miss vertex is left where it belongs');
  assert.deepEqual(out[1], loops[1]);
});

test('a genuine T-junction is still inserted at real marched precision', () => {
  // The other side of the same threshold: an offset at the scale a real
  // marcher actually leaves (~1e-14) must still be recognized, or the
  // tightened tolerance would have bought correctness by breaking the
  // feature it guards.
  const loops = [
    [[0, 0, 0], [10, 0, 0], [10, 5, 0]],
    [[0, 0, 0], [4, 1.4e-14, 0], [10, 0, 0], [10, -5, 0]],
  ];
  const out = insertTJunctionVertices(loops, 1e-4);
  assert.equal(out[0].length, 4, 'the real T-junction is still inserted');
  assert.deepEqual(out[0][1], [4, 1.4e-14, 0], 'and it is the far side\'s own coordinate');
});

test('a subdivision point off the chord is spliced in when the far side corroborates it', () => {
  // The curved-shared-edge case. The far side subdivides the edge at a point
  // that is genuinely on the shared curve and therefore OFF this straight
  // chord — here by 0.4% of its length, the same order as the sagitta a real
  // marched circle leaves between samples, and six orders past the absolute
  // floor. It is accepted because the far side already carries BOTH (a,p) and
  // (p,b), not because it is close.
  const loops = [
    [[0, 0, 0], [10, 0, 0], [10, 5, 0]],
    [[0, 0, 0], [4, 0.04, 0], [10, 0, 0], [10, -5, 0]],
  ];
  const out = insertTJunctionVertices(loops, 1e-4);
  assert.equal(out[0].length, 4, 'the undivided loop gains the far side\'s own subdivision');
  assert.deepEqual(out[0][1], [4, 0.04, 0], 'and it is the far side\'s own coordinate, not a projection');
  assert.deepEqual(out[1], loops[1], 'the loop that already had it is untouched');
});

test('the same off-chord point is REFUSED without corroboration', () => {
  // The negative control for the route above, and the one that makes it a
  // real test: identical geometry, identical distances, but the point now
  // belongs to a face that shares neither endpoint of the edge. Nothing about
  // proximity changed, so if this were still inserted the corroboration would
  // be decorative.
  const loops = [
    [[0, 0, 0], [10, 0, 0], [10, 5, 0]],
    [[4, 0.04, 0], [8, 3, 0], [4, 3, 0]],
  ];
  const out = insertTJunctionVertices(loops, 1e-4);
  assert.deepEqual(out, loops, 'an unrelated face does not get to subdivide this edge');
});

test('a corroborated point far off the chord is still refused', () => {
  // A sliver face sharing both endpoints is the one shape that can corroborate
  // falsely, so the relative bound stays a real gate: at 10% of the edge's own
  // length this is a genuine third corner, not a subdivision of the edge.
  const loops = [
    [[0, 0, 0], [10, 0, 0], [10, 5, 0]],
    [[0, 0, 0], [5, 1, 0], [10, 0, 0], [10, -5, 0]],
  ];
  const out = insertTJunctionVertices(loops, 1e-4);
  assert.deepEqual(out[0], loops[0], 'a face that merely shares both endpoints is not a subdivision');
});

test('a chain of two corroborated points is followed the whole way', () => {
  // Two cuts landing inside one sample gap subdivide the edge as
  // a -> p1 -> p2 -> b, and then neither (a,p2) nor (p1,b) exists anywhere —
  // a single hop finds nothing and would leave the shorter edges naked.
  const loops = [
    [[0, 0, 0], [10, 0, 0], [10, 5, 0]],
    [[0, 0, 0], [3, 0.03, 0], [7, 0.02, 0], [10, 0, 0], [10, -5, 0]],
  ];
  const out = insertTJunctionVertices(loops, 1e-4);
  assert.equal(out[0].length, 5, 'both subdivision points are inserted');
  assert.deepEqual(out[0][1], [3, 0.03, 0]);
  assert.deepEqual(out[0][2], [7, 0.02, 0], 'and in order along the edge');
});

test('a star prism cut by a slab INTERSECTS into one closed solid', () => {
  const { star, slab, curves } = buildFixture();
  assert.equal(curves.length, 20, 'ten side faces, each crossing both slab caps');

  const res = booleanSolids(star, slab, curves, 'intersect');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.stats.nakedEdgeCount, 0, 'every shared edge welded');
  assert.equal(res.stats.chi, 2, 'a sphere, topologically');
  assert.equal(res.stats.shellCount, 1);
  assert.equal(res.stats.F, 12, 'ten side fragments plus the two cap fragments');
  // The weld had real work to do and did not need to strain: the surviving
  // gap is the marcher's own residual, orders below the tolerance.
  assert.ok(res.worstSharedGap < 1e-9, `worst weld gap ${res.worstSharedGap}`);
});

test('the same pair UNIONS into one closed solid — the holed-cap case', () => {
  // The operation that needs the annulus split. Union keeps the slab's own
  // CAP fragments, and a cap is a square with a star-shaped hole where the
  // prism passes through it — a face with a hole, which a keyhole bridge
  // cannot express as a B-rep face without repeating a corner.
  const { star, slab, curves } = buildFixture();
  const res = booleanSolids(star, slab, curves, 'union');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.stats.nakedEdgeCount, 0, 'every shared edge welded');
  assert.equal(res.stats.chi, 2, 'one closed solid');
  assert.equal(res.stats.shellCount, 1);
  // Both caps are holed, and each becomes two faces rather than one.
  assert.ok(res.stats.F > 12, `the split adds faces, got F=${res.stats.F}`);
});

test('the same pair DIFFERENCES into two genuinely separate shells', () => {
  const { star, slab, curves } = buildFixture();
  const res = booleanSolids(star, slab, curves, 'difference');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.stats.nakedEdgeCount, 0);
  // The slab passes clean through, so the prism above and the prism below are
  // two closed solids: chi = 2 per shell.
  assert.equal(res.stats.shellCount, 2);
  assert.equal(res.stats.chi, 4);
});

test('without the T-junction pass the same fixture leaves 30 naked edges', () => {
  // The negative control, and it runs on the REAL fragment loops rather than a
  // stand-in: count the edges a plain weld would leave with exactly one
  // incident face, before and after the pass. Without this the test above
  // could be passing for some other reason entirely.
  const { star, slab, curves } = buildFixture();

  const loops = [];
  for (const side of ['a', 'b']) {
    const mine = side === 'a' ? star : slab;
    const other = side === 'a' ? slab : star;
    const port = side === 'a' ? 'faceA' : 'faceB';
    for (let fi = 0; fi < mine.faces.length; fi++) {
      const face = mine.faces[fi];
      const outer = face.trimLoop ?? trivialTrimLoop(face.srf);
      const holes = face.trimHoles ?? [];
      const pcurves = curves.filter((c) => c[port] === fi)
        .map((c) => projectPointsToSurfaceUV(c.samples, face.srf).uv);
      const fragments = pcurves.length
        ? splitFaceByCurves({ outer, holes }, pcurves).fragments
        : [{ outer, holes }];
      const decision = keepFragments(face.srf, fragments, other.triangles, 'intersect', { operand: side });
      for (const f of decision.kept) loops.push(...fragmentBoundaries3D(f.srf, f).loops);
    }
  }

  const TOL = 1e-4;
  const nakedCount = (ls) => {
    const key = (p) => p.map((x) => Math.round(x / TOL)).join('|');
    const seen = new Map();
    for (const loop of ls) {
      for (let i = 0; i < loop.length; i++) {
        const a = key(loop[i]), b = key(loop[(i + 1) % loop.length]);
        if (a === b) continue;
        const k = a < b ? `${a}#${b}` : `${b}#${a}`;
        seen.set(k, (seen.get(k) ?? 0) + 1);
      }
    }
    return [...seen.values()].filter((n) => n === 1).length;
  };

  assert.equal(nakedCount(loops), 30, 'the raw fragment loops genuinely do not close');
  assert.equal(nakedCount(insertTJunctionVertices(loops, TOL)), 0, 'and the pass is what closes them');
});
