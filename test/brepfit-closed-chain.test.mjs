// A BOOLEAN'S CLOSED EDGE, MEASURED AS A CLOSED CURVE.
//
// `fitSolidEdgeCurves` is the second caller of `fitCurveToPoints`' closed
// branch — the first is Text. An intersection loop with no branch point
// anywhere on it walks as one chain marked `closed`, and every property that
// branch gets wrong lands on a solid's own boundary rather than on a letter.
//
// ⚠ THE OBVIOUS FIXTURE CANNOT SEE IT. `two_tori_seam_straddle_sew_open.3dm`,
// which the rest of the brepfit suite runs on, sews into 50 fitted edges and
// NOT ONE of them is closed: every chain there runs corner to corner, so the
// closed branch is never entered and the whole file stays green whatever that
// branch does. The offset pair below yields exactly one edge and it is the
// closed one, which is what makes this a control rather than a repetition.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import rhino3dmFactory from 'rhino3dm';
import { surfaceFromRhino } from '../io3dm.mjs';
import { trivialTrimLoop } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { booleanSolids } from '../kernel/boolean.mjs';
import { fitSolidEdgeCurves } from '../kernel/brepfit.mjs';
import { curvePoint, isCurveClosed } from '../kernel/curve.mjs';
import { maxDeviationFromCurve } from '../kernel/fitcurve.mjs';

const HERE = import.meta.dirname;
const TOL = 0.01; // the fixture document's own model tolerance
const rhino = await rhino3dmFactory();
const doc = rhino.File3dm.fromByteArray(new Uint8Array(fs.readFileSync(
  path.join(HERE, 'fixtures', 'two_tori_offset_union_works.3dm'))));
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
const fitted = fitSolidEdgeCurves(union.solid, { tolerance: TOL });
const closedEdges = fitted.edges.filter((e) => e.closed && e.curve);

function arcLength(crv, n = 4000) {
  const p = crv.degree, U = crv.knots;
  const t0 = U[p], t1 = U[U.length - 1 - p];
  let L = 0, prev = curvePoint(crv, t0);
  for (let i = 1; i <= n; i++) {
    const c = curvePoint(crv, t0 + ((t1 - t0) * i) / n);
    L += Math.hypot(c[0] - prev[0], c[1] - prev[1], c[2] - prev[2]);
    prev = c;
  }
  return L;
}

function chainLength(points) {
  let L = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i], b = points[(i + 1) % points.length];
    L += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  }
  return L;
}

test('THE FIXTURE REACHES THE CLOSED BRANCH AT ALL — which the rest of the suite does not', () => {
  assert.ok(union.ok, `union refused: ${union.reason}`);
  assert.equal(closedEdges.length, 1, `exactly one fitted edge, and it is the closed intersection loop (got ${closedEdges.length} of ${fitted.edges.length})`);
  assert.equal(closedEdges[0].kind, 'nurbs',
    'and it is a least-squares fit, not a recognised circle or ellipse — the primitive path would sidestep the branch entirely');
});

test('A CLOSED EDGE IS NOT LONGER THAN THE CHAIN IT FITS', () => {
  for (const e of closedEdges) {
    const ratio = arcLength(e.curve) / chainLength(e.points);
    // The marched chain is inscribed in the true intersection, so the curve is
    // legitimately a hair longer. 1.001 is two orders of magnitude under the
    // 1.0227 an untrimmed wrap-padded fit measured on this same edge.
    assert.ok(ratio < 1.001, `the fitted edge is ${ratio.toFixed(4)}x the length of its own ${e.points.length}-point chain`);
  }
});

test('A CLOSED EDGE ACTUALLY CLOSES — an open loop is a hole in the exported boundary', () => {
  for (const e of closedEdges) {
    const cp = e.curve.ctrlPts;
    const gap = Math.hypot(cp[0][0] - cp[cp.length - 1][0], cp[0][1] - cp[cp.length - 1][1], cp[0][2] - cp[cp.length - 1][2]);
    assert.equal(gap, 0, `first and last control point must coincide exactly (gap ${gap.toExponential(3)})`);
    assert.ok(isCurveClosed(e.curve, 1e-9), 'and the kernel\'s own predicate must agree');
  }
});

test('AND IT STILL HOLDS ITS BOUND — the closure is not bought with accuracy', () => {
  for (const e of closedEdges) {
    const dev = maxDeviationFromCurve(e.points, e.curve, { closed: true });
    assert.ok(dev <= TOL, `re-measured deviation ${dev.toExponential(3)} exceeds the ${TOL} it was fitted to`);
    assert.ok(Math.abs(dev - e.maxDeviation) < 1e-12, 'and the fitter reported the curve it actually returned');
  }
});
