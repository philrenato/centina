import test from 'node:test';
import assert from 'node:assert';
import { splitFaceByCurves, representativeInteriorPoint, SRC_TRIM, SRC_INTERSECTION } from '../kernel/facesplit.mjs';
import { signedArea2D, pointInUVPolygon } from '../kernel/trim.mjs';

// Every expected answer below is written down independently — a rectangle's
// area is width times height, an L's area is the big rectangle minus the
// notch — never read back from what the function returned last time.

const SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]];
const area = (loop) => Math.abs(signedArea2D(loop));
const fragArea = (f) => area(f.outer) - f.holes.reduce((s, h) => s + area(h), 0);
const totalArea = (frs) => frs.reduce((s, f) => s + fragArea(f), 0);
const sortedAreas = (frs) => frs.map(fragArea).sort((a, b) => a - b);
const allTags = (f) => [...f.sources.outer, ...f.sources.holes.flat()];

test('an open curve crossing a face end to end splits it into exactly two fragments, of the two areas the cut position implies', () => {
  const r = splitFaceByCurves({ outer: SQUARE }, [[[4, 0], [4, 10]]]);
  assert.ok(r.ok, r.reason);
  assert.equal(r.fragments.length, 2);
  // x=4 through a 10x10 square: 4 wide and 6 wide, both 10 tall.
  const areas = sortedAreas(r.fragments);
  assert.ok(Math.abs(areas[0] - 40) < 1e-6, `left fragment area ${areas[0]}, expected 40`);
  assert.ok(Math.abs(areas[1] - 60) < 1e-6, `right fragment area ${areas[1]}, expected 60`);
  assert.ok(Math.abs(totalArea(r.fragments) - 100) < 1e-6, 'area is conserved by the split');
});

test('a CLOSED interior loop splits the face into the disk it encloses and the annulus around it — the loop becomes a real hole, not a second overlapping region', () => {
  const loop = [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]];
  const r = splitFaceByCurves({ outer: SQUARE }, [loop]);
  assert.ok(r.ok, r.reason);
  assert.equal(r.fragments.length, 2);
  const areas = sortedAreas(r.fragments);
  assert.ok(Math.abs(areas[0] - 4) < 1e-6, `disk area ${areas[0]}, expected 4`);
  assert.ok(Math.abs(areas[1] - 96) < 1e-6, `annulus area ${areas[1]}, expected 96`);
  // The annulus must carry the loop as a genuine HOLE. Without that it would
  // be a full square overlapping the disk, and the areas would not add up.
  const annulus = r.fragments.find((f) => fragArea(f) > 50);
  assert.equal(annulus.holes.length, 1, 'the annulus has exactly one hole');
  assert.ok(Math.abs(area(annulus.holes[0]) - 4) < 1e-6);
  const disk = r.fragments.find((f) => fragArea(f) < 50);
  assert.equal(disk.holes.length, 0);
});

test('EDGE PROVENANCE: the cut edge is tagged as intersection and the original boundary as trim — the distinction Phase 7 and 8 both need', () => {
  const r = splitFaceByCurves({ outer: SQUARE }, [[[4, 0], [4, 10]]]);
  assert.ok(r.ok, r.reason);
  for (const f of r.fragments) {
    const tags = allTags(f);
    assert.ok(tags.some((t) => t.includes(SRC_INTERSECTION)), 'every fragment has at least one edge from the intersection curve');
    assert.ok(tags.some((t) => t.includes(SRC_TRIM)), 'every fragment retains part of the original trim boundary');
    // A fragment of a square cut once is a rectangle: three trim sides and
    // one cut side, so exactly one edge may claim the intersection.
    const cut = tags.filter((t) => t.includes(SRC_INTERSECTION));
    assert.equal(cut.length, 1, `expected exactly one intersection edge per fragment, got ${cut.length}`);
    assert.ok(!cut[0].includes(SRC_TRIM), 'the cut edge is not also a trim edge — it is genuinely interior');
  }
});

test('a face WITH A HOLE splits correctly: the hole is respected, and its own interior never comes back as a fragment', () => {
  const face = { outer: SQUARE, holes: [[[4, 4], [6, 4], [6, 6], [4, 6]]] };
  // The cut runs straight through the hole, so the hole is bisected too —
  // and both halves of it must be dropped, not returned as regions.
  const r = splitFaceByCurves(face, [[[5, 0], [5, 10]]]);
  assert.ok(r.ok, r.reason);
  assert.equal(r.fragments.length, 2);
  // 100 minus the 2x2 hole is 96, cut symmetrically: 48 and 48.
  const areas = sortedAreas(r.fragments);
  assert.ok(Math.abs(areas[0] - 48) < 1e-6, `area ${areas[0]}, expected 48`);
  assert.ok(Math.abs(areas[1] - 48) < 1e-6, `area ${areas[1]}, expected 48`);
  assert.ok(Math.abs(totalArea(r.fragments) - 96) < 1e-6, 'the hole stays subtracted from the total');
});

test('a NON-CONVEX face splits correctly — the case a centroid-based interior test would get wrong', () => {
  // An L: a 10x10 square with a 6x6 notch taken out of the top right.
  const L = [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]];
  assert.ok(Math.abs(area(L) - 64) < 1e-9, 'the L fixture is 100 minus a 36 notch');
  const r = splitFaceByCurves({ outer: L }, [[[0, 2], [10, 2]]]);
  assert.ok(r.ok, r.reason);
  assert.equal(r.fragments.length, 2);
  const areas = sortedAreas(r.fragments);
  assert.ok(Math.abs(areas[0] - 20) < 1e-6, `bottom strip ${areas[0]}, expected 10x2=20`);
  assert.ok(Math.abs(areas[1] - 44) < 1e-6, `upper L ${areas[1]}, expected 64-20=44`);
});

test('a curve that DANGLES inside the face splits nothing, and says so rather than silently returning an unchanged face', () => {
  const r = splitFaceByCurves({ outer: SQUARE }, [[[3, 3], [7, 7]]]);
  assert.ok(r.ok, r.reason);
  assert.equal(r.fragments.length, 1, 'a curve reaching no boundary cannot divide the face');
  assert.ok(Math.abs(fragArea(r.fragments[0]) - 100) < 1e-6);
  assert.equal(r.danglingCurves, 1, 'the curve that contributed nothing is reported');
  assert.ok(allTags(r.fragments[0]).every((t) => !t.includes(SRC_INTERSECTION)));
});

test('a curve lying exactly ALONG a trim boundary is reported as coincident, and does not split the face', () => {
  const r = splitFaceByCurves({ outer: SQUARE }, [[[0, 0], [10, 0]]]);
  assert.ok(r.ok, r.reason);
  assert.equal(r.fragments.length, 1, 'running along an existing edge adds no new region');
  assert.ok(r.alongBoundary > 0, 'the coincidence is reported rather than passing as an ordinary split');
  const both = allTags(r.fragments[0]).filter((t) => t.includes(SRC_TRIM) && t.includes(SRC_INTERSECTION));
  assert.ok(both.length > 0, 'the shared edge carries BOTH tags rather than one silently winning');
});

test('TWO curves crossing each other inside the face produce four fragments, and area is still conserved', () => {
  const r = splitFaceByCurves({ outer: SQUARE }, [[[5, 0], [5, 10]], [[0, 5], [10, 5]]]);
  assert.ok(r.ok, r.reason);
  assert.equal(r.fragments.length, 4);
  for (const a of sortedAreas(r.fragments)) {
    assert.ok(Math.abs(a - 25) < 1e-6, `each quarter should be 25, got ${a}`);
  }
  assert.ok(Math.abs(totalArea(r.fragments) - 100) < 1e-6);
});

test('the fragments are a genuine PARTITION — no two of them overlap, which conserved area alone would not prove', () => {
  const r = splitFaceByCurves({ outer: SQUARE }, [[[5, 0], [5, 10]], [[0, 5], [10, 5]]]);
  assert.ok(r.ok, r.reason);
  const probes = r.fragments.map((f) => representativeInteriorPoint(f.outer, f.holes));
  assert.ok(probes.every(Boolean));
  for (let i = 0; i < r.fragments.length; i++) {
    for (let j = 0; j < r.fragments.length; j++) {
      if (i === j) continue;
      const p = probes[i];
      assert.notEqual(
        pointInUVPolygon(r.fragments[j].outer, p[0], p[1]), 'inside',
        `fragment ${i}'s interior point falls inside fragment ${j} — the fragments overlap`,
      );
    }
  }
});

test('a curve running PAST the face boundary still splits it correctly — the overshoot is dropped, not treated as geometry', () => {
  // Extends well beyond the square at both ends, as a marched intersection
  // curve clipped only approximately to a face would.
  const r = splitFaceByCurves({ outer: SQUARE }, [[[4, -5], [4, 15]]]);
  assert.ok(r.ok, r.reason);
  assert.equal(r.fragments.length, 2);
  const areas = sortedAreas(r.fragments);
  assert.ok(Math.abs(areas[0] - 40) < 1e-6, `left fragment ${areas[0]}, expected 40`);
  assert.ok(Math.abs(areas[1] - 60) < 1e-6);
  assert.ok(Math.abs(totalArea(r.fragments) - 100) < 1e-6, 'nothing outside the face leaked into the total');
});

test('SCALE INDEPENDENCE: the same split at 1000x behaves identically, so no tolerance here is secretly absolute', () => {
  const big = SQUARE.map(([x, y]) => [x * 1000, y * 1000]);
  const r = splitFaceByCurves({ outer: big }, [[[4000, 0], [4000, 10000]]]);
  assert.ok(r.ok, r.reason);
  assert.equal(r.fragments.length, 2);
  const areas = sortedAreas(r.fragments);
  assert.ok(Math.abs(areas[0] / 1e6 - 40) < 1e-3, `scaled left fragment ${areas[0] / 1e6}, expected 40`);
  assert.ok(Math.abs(areas[1] / 1e6 - 60) < 1e-3);
});

test('CURVE ORDER does not change the result — the cheapest order-dependence detector there is', () => {
  const a = splitFaceByCurves({ outer: SQUARE }, [[[5, 0], [5, 10]], [[0, 5], [10, 5]]]);
  const b = splitFaceByCurves({ outer: SQUARE }, [[[0, 5], [10, 5]], [[5, 0], [5, 10]]]);
  assert.ok(a.ok && b.ok);
  assert.deepEqual(sortedAreas(a.fragments), sortedAreas(b.fragments));
});

test('a face with no curves at all comes back as itself, unsplit and area-exact', () => {
  const r = splitFaceByCurves({ outer: SQUARE }, []);
  assert.ok(r.ok, r.reason);
  assert.equal(r.fragments.length, 1);
  assert.ok(Math.abs(fragArea(r.fragments[0]) - 100) < 1e-6);
});

test('a degenerate face refuses by name, and the two degenerate cases refuse DIFFERENTLY rather than sharing one vague message', () => {
  // Too few points to close a loop at all.
  const noLoop = splitFaceByCurves({ outer: [[0, 0], [10, 0]] }, [[[4, 0], [4, 10]]]);
  assert.equal(noLoop.ok, false);
  assert.match(noLoop.reason, /outer trim loop/i);
  // A real closed loop, but collinear — enough points, zero area.
  const flat = splitFaceByCurves({ outer: [[0, 0], [10, 0], [5, 0], [0, 0]] }, [[[4, 0], [4, 10]]]);
  assert.equal(flat.ok, false);
  assert.match(flat.reason, /area/i);
  // A hole exactly as large as the face leaves nothing to split — caught by
  // the same area check rather than producing zero fragments silently.
  const eaten = splitFaceByCurves({ outer: SQUARE, holes: [SQUARE] }, [[[4, 0], [4, 10]]]);
  assert.equal(eaten.ok, false);
  assert.match(eaten.reason, /area/i);
});

test('a SELF-INTERSECTING trim loop refuses by naming that, not by reporting a lost region — the cause, not the symptom', () => {
  // A bowtie. Its signed area partly cancels (-20) while its two real lobes
  // total 42.5, so the area backstop WOULD catch this; the point of the
  // explicit check is that the message names what is actually wrong.
  const bowtie = [[0, 0], [10, 10], [10, 0], [0, 6]];
  const r = splitFaceByCurves({ outer: bowtie }, [[[4, -5], [4, 15]]]);
  assert.equal(r.ok, false);
  assert.match(r.reason, /intersects itself/i);
  // A self-intersecting HOLE is caught the same way and says which loop.
  const h = splitFaceByCurves({ outer: SQUARE, holes: [bowtie] }, []);
  assert.equal(h.ok, false);
  assert.match(h.reason, /hole 1/i);
});

test('representativeInteriorPoint finds a point inside a thin non-convex region, and refuses on a degenerate one', () => {
  // A thin U — the case where stepping inward from a short edge can overshoot
  // straight out the far side, which is why every edge is tried, not one.
  const U = [[0, 0], [10, 0], [10, 10], [9, 10], [9, 1], [1, 1], [1, 10], [0, 10]];
  const p = representativeInteriorPoint(U, []);
  assert.ok(p, 'a point was found in the thin region');
  assert.equal(pointInUVPolygon(U, p[0], p[1]), 'inside');
  assert.equal(representativeInteriorPoint([[0, 0], [1, 0], [2, 0]], []), null, 'a zero-area loop has no interior to find');
});
