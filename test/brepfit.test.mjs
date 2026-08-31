// kernel/brepfit.mjs — turning a sewn boolean's over-decomposed edge set into
// the edges a B-rep actually has, each with a fitted curve.
//
// Driven by the SAME banked fixture test/boolean-torus-pair-sew.test.mjs uses,
// through the real boolean, because the thing under test is a property of what
// sewing actually produces and not of any shape invented here.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import rhino3dmFactory from 'rhino3dm';
import { surfaceFromRhino } from '../io3dm.mjs';
import { trivialTrimLoop } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { booleanSolids } from '../kernel/boolean.mjs';
import { fitSolidEdgeCurves, fitHalfEdgePcurves, fitFaceLoops } from '../kernel/brepfit.mjs';
import { maxDeviationFromCurve } from '../kernel/fitcurve.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const rhino = await rhino3dmFactory();
const doc = rhino.File3dm.fromByteArray(new Uint8Array(fs.readFileSync(
  path.join(HERE, 'fixtures', 'two_tori_seam_straddle_sew_open.3dm'))));
const solids = [];
{
  const objs = doc.objects();
  for (let i = 0; i < objs.count; i++) {
    const g = objs.get(i).geometry();
    const ns = g instanceof rhino.NurbsSurface ? g : (g.toNurbsSurface ? g.toNurbsSurface() : null);
    if (!ns) continue;
    const srf = surfaceFromRhino(ns);
    solids.push({ faces: [{ srf }], triangles: tessellateTrimmedSurface(srf, trivialTrimLoop(srf), 60, 60, []).map((t) => t.map((v) => v.position)) });
  }
}
const [A, B] = solids;
const ssi = intersectSurfacesComplete(A.faces[0].srf, B.faces[0].srf);
const union = booleanSolids(A, B, ssi.components.map((c) => ({ samples: c.samples.map((s) => s.point), faceA: 0, faceB: 0 })), 'union');

const TOL = 0.01; // the fixture document's own model tolerance
const fitted = fitSolidEdgeCurves(union.solid, { tolerance: TOL });

test('THE INPUT LANDS: the union still sews closed, with the over-decomposed edge set this exists to fix', () => {
  assert.ok(union.ok, `union refused: ${union.reason}`);
  assert.equal(union.stats.nakedEdgeCount, 0);
  assert.ok(union.solid.edges.length > 400, `sewing produces hundreds of edges (${union.solid.edges.length})`);
});

test('EVERY ORIGINAL EDGE IS CLAIMED EXACTLY ONCE — no hole, no duplicate', () => {
  // A missed edge is a gap in the exported boundary and a doubled one is a
  // duplicate edge; neither shows up later as anything but a wrong picture.
  assert.equal(fitted.stats.doubledEdges, 0, 'no edge lands in two chains');
  assert.equal(fitted.stats.coveredEdges, union.solid.edges.length, 'every edge lands in one');
});

test('THE MERGE IS THE WHOLE POINT: hundreds of segment-edges become a handful of real ones', () => {
  assert.ok(fitted.ok, `refused: ${JSON.stringify(fitted.stats)}`);
  assert.ok(fitted.stats.fittedEdges < fitted.stats.originalEdges / 5,
    `${fitted.stats.originalEdges} sewn edges -> ${fitted.stats.fittedEdges} fitted (stats: ${JSON.stringify(fitted.stats)})`);
  // And the merged count should be near the corner count, since a chain runs
  // between corners: a merge that stopped early would leave far more.
  assert.ok(fitted.stats.fittedEdges <= fitted.stats.cornerVertices * 3,
    `${fitted.stats.fittedEdges} edges against ${fitted.stats.cornerVertices} corners`);
});

test('EVERY FITTED CURVE HOLDS ITS BOUND, re-measured here rather than taken from the fitter', () => {
  // The fitter reports its own deviation. This re-measures each curve against
  // the points it replaced, so a fitter that mis-reported would still be
  // caught — the report and the check are not the same computation.
  for (const e of fitted.edges) {
    assert.ok(e.curve, `a chain of ${e.points.length} points was refused: ${e.reason}`);
    const dev = maxDeviationFromCurve(e.points, e.curve, { closed: e.closed });
    assert.ok(dev <= TOL, `a fitted edge sits ${dev} off its own points, above ${TOL}`);
  }
  assert.ok(fitted.stats.worstDeviation <= TOL);
});

test('CHAIN ENDPOINTS ARE EXACT, so neighboring edges still meet where the topology says', () => {
  // The fitter interpolates its endpoints rather than least-squaring them,
  // precisely so a shared corner cannot drift. A gap here is a naked edge in
  // anything built from these curves.
  for (const e of fitted.edges) {
    const first = e.curve.ctrlPts[0], last = e.curve.ctrlPts[e.curve.ctrlPts.length - 1];
    const p0 = e.points[0], pN = e.closed ? e.points[0] : e.points[e.points.length - 1];
    for (let k = 0; k < 3; k++) {
      assert.ok(Math.abs(first[k] - p0[k]) < 1e-9, `chain start drifted (${first[k]} vs ${p0[k]})`);
      if (!e.closed) assert.ok(Math.abs(last[k] - pN[k]) < 1e-9, `chain end drifted (${last[k]} vs ${pN[k]})`);
    }
  }
});

test('THE SOLID IS NOT TOUCHED — the naked-edge count cannot move, by construction', () => {
  // This is a derived record, not a mutation. Asserting it directly because
  // "it does not mutate" is exactly the kind of claim that quietly stops being
  // true when someone later reaches for the convenient in-place write.
  assert.equal(union.solid.edges.length, fitted.stats.originalEdges);
  assert.equal(union.stats.nakedEdgeCount, 0, 'still closed after fitting');
  for (const e of union.solid.edges) {
    assert.equal(e.curve3d, null, 'an over-decomposed edge keeps its honest null rather than borrowing a chain curve');
  }
});

// ---------------------------------------------------------------------------
// THE (u,v) HALF. A B-rep trim is a curve in its own face's parameters, so one
// 3-D chain yields TWO pcurves — one per face it separates — and neither is
// derivable from the other.
// ---------------------------------------------------------------------------
const pcurves = fitHalfEdgePcurves(union.solid, fitted, { tolerance: TOL });

test('EVERY FACE KNOWS WHAT SURFACE IT IS — the slot sewing used to leave null', () => {
  // buildBrepSolid works in 3-D loops alone, so `face.surface` came back null
  // on every face and nothing downstream could ask a question in a face's own
  // parameters. sewBoundaries3D already knew the answer and threw it away.
  const faces = union.solid.shells.flatMap((sh) => sh.faces || []);
  assert.ok(faces.length > 0);
  assert.equal(faces.filter((f) => f.surface).length, faces.length,
    `${faces.filter((f) => f.surface).length} of ${faces.length} faces carry their surface`);
});

test('EVERY HALF-EDGE GETS A PCURVE, both sides of every fitted edge', () => {
  assert.ok(pcurves.ok, `refused: ${JSON.stringify(pcurves.stats)}`);
  assert.equal(pcurves.stats.sides, fitted.edges.length * 2, 'two sides per fitted edge');
  assert.equal(pcurves.stats.refusedSides, 0);
  assert.equal(pcurves.stats.withPcurves, pcurves.stats.sides);
});

test("AND THAT IS THE SURFACE ATTRIBUTION'S OWN PROOF", () => {
  // The projection refuses when a chain does not lie on the surface it is
  // given, so 100 sides projecting within tolerance is a direct check that
  // each face was handed the RIGHT surface. A positional mapping that had
  // slipped by one would fail here loudly rather than produce wrong geometry
  // quietly.
  assert.ok(pcurves.stats.worstDeviation <= TOL,
    `worst pcurve deviation ${pcurves.stats.worstDeviation} must clear ${TOL}`);
});

test('A CHAIN CROSSING A SEAM BECOMES SEVERAL PCURVES, not one with a phantom chord', () => {
  // On a closed direction the chain leaves the parametric rectangle and
  // re-enters at the far edge. A single curve through those samples would run
  // straight across the face. seamOpenChains splits exactly where the curve
  // genuinely leaves the domain, so more pieces than sides is the correct
  // outcome and the count is asserted rather than assumed to be one-to-one.
  assert.ok(pcurves.stats.seamSplitSides > 0, 'the torus pair does cross seams');
  assert.ok(pcurves.stats.totalPieces > pcurves.stats.sides,
    `${pcurves.stats.totalPieces} pieces from ${pcurves.stats.sides} sides`);
});

test('EVERY PCURVE LIES INSIDE ITS FACE\'S OWN PARAMETRIC RECTANGLE', () => {
  // A pcurve outside the domain is not a trim, whatever its shape. Checked
  // against each face's real knot range, with a tolerance-scaled slack for the
  // seam copies that sit exactly ON an edge of the rectangle.
  for (const p of pcurves.pcurves) {
    const face = union.solid.shells.flatMap((sh) => sh.faces || []).find((f) => f.id === p.faceId);
    const s = face.surface;
    const uMin = s.knotsU[0], uMax = s.knotsU[s.knotsU.length - 1];
    const vMin = s.knotsV[0], vMax = s.knotsV[s.knotsV.length - 1];
    const eu = (uMax - uMin) * 1e-6, ev = (vMax - vMin) * 1e-6;
    for (const piece of p.pieces) {
      for (const [u, v] of piece.uv) {
        assert.ok(u >= uMin - eu && u <= uMax + eu, `u=${u} outside [${uMin}, ${uMax}]`);
        assert.ok(v >= vMin - ev && v <= vMax + ev, `v=${v} outside [${vMin}, ${vMax}]`);
      }
    }
  }
});

// ---------------------------------------------------------------------------
// LOOP-TRAVERSAL ORDER. A B-rep loop is an ordered, oriented cycle, and
// assembling from fitted-edge order produced exactly the complaint OpenNURBS
// makes: consecutive trims that do not join end-to-start in (u,v).
// ---------------------------------------------------------------------------
const faceLoops = fitFaceLoops(union.solid, fitted, { tolerance: TOL });

test('EVERY FACE YIELDS ORDERED LOOPS, none refused', () => {
  assert.ok(faceLoops.ok, `refused: ${JSON.stringify(faceLoops.stats)}`);
  assert.equal(faceLoops.stats.refusedFaces, 0);
  assert.ok(faceLoops.stats.trims > 0);
});

test('HALF THE TRIMS RUN AGAINST THEIR EDGE — orientation is computed, not assumed', () => {
  // A closed shell traverses every shared edge once in each direction, so a
  // roughly even split is the expected shape. All-forward would mean the
  // orientation was never worked out at all, which is what the first assembly
  // did and what made every second trim run backwards.
  const { trims, reversedTrims } = faceLoops.stats;
  assert.ok(reversedTrims > trims * 0.3 && reversedTrims < trims * 0.7,
    `${reversedTrims} of ${trims} trims reversed`);
});

test('THE WALK CAN START MID-CHAIN, and the wrap-around group is merged', () => {
  // `loop.halfEdge` is whichever half-edge the builder stored, so one chain can
  // appear as both the first and last group of a walk. Left alone that emits
  // the same edge twice and neither piece joins its neighbor. It genuinely
  // happens here, so the merge is exercised rather than theoretical.
  assert.ok(faceLoops.stats.mergedWrapGroups > 0,
    `${faceLoops.stats.mergedWrapGroups} loops started mid-chain`);
});

test('AND THE LOOPS NOW CLOSE IN (u,v) — no break at all, seams included', () => {
  // ⚠ THIS TEST USED TO ASSERT THE OPPOSITE, and it firing is what announced
  // the fix. The surviving breaks were exactly 1.0 and sqrt(2) — one full
  // parametric period, or one in each direction — which looked like faces
  // wrapping the closed direction and wanting explicit seam trims.
  //
  // They were not. The wraps were MANUFACTURED BY THE MEASUREMENT: each trim
  // was projected independently, the projector wraps parameters into the
  // domain, and a point near a seam therefore came back at whichever end its
  // own walk reached it from. Measured across all three booleans — 36, 12 and
  // 24 loops — every single one fits inside ONE period once unwrapped as a
  // whole, and not one genuinely wraps. So the cure was a consistent walk, not
  // the seam trims the symptom was asking for.
  const { worstLoopJoin, loopJoinBreaks } = faceLoops.stats;
  assert.equal(loopJoinBreaks, 0, `every trim must join the next (worst gap ${worstLoopJoin})`);
  assert.ok(worstLoopJoin < 1e-9,
    `and join exactly, not nearly — worst gap ${worstLoopJoin}`);
});
