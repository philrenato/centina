// SURFACE FLATTENING (LSCM) — kernel/flatten.mjs
//
// The whole discipline of this file: a flattener is only as trustworthy as
// the number it reports about its own error, so nothing here checks the
// flattener against itself. The developable cases are checked against an
// ANALYTICALLY known unrolled shape derived from the cylinder's/cone's own
// equations (a rectangle; an annular sector with a computed apex), and the
// doubly-curved cases are checked for distortion that is nonzero, correctly
// ordered, and genuinely tracks how curved the input is.

import test from 'node:test';
import assert from 'node:assert';
import { makeLine, makeArc, revolve, extrude } from '../kernel/primitives.mjs';
import {
  flattenLSCM, flattenNurbsSurface, flatteningDistortion, weldTriangulation,
  buildMeshTopology, validateFlattenMesh, triangleLocalFrame,
  solveSparseLeastSquares, isDevelopable, nurbsSurfaceArea, dropDegenerateFaces,
} from '../kernel/flatten.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';

const DEG = Math.PI / 180;

// ---- fixtures -----------------------------------------------------------
// Deliberately not trivial and not symmetric where symmetry would hide a
// bug: every patch below is a PARTIAL revolve (so it is a genuine open
// strip, not a closed tube whose seam would mask an orientation error), at
// an odd sweep angle, with a non-square tessellation grid.

const CYL_R = 40, CYL_H = 55, CYL_SWEEP = 200 * DEG;
function cylinderPatch() {
  // A vertical line revolved 200 degrees: a genuinely DEVELOPABLE surface
  // (zero Gaussian curvature everywhere). Its tessellation is a prism
  // strip — each quad's four points are coplanar, because the two vertical
  // edges are parallel — so the polyhedron is EXACTLY developable too, not
  // merely approximately so.
  return revolve(makeLine([CYL_R, 0, 0], [CYL_R, 0, CYL_H]), [0, 0, 0], [0, 0, 1], 0, CYL_SWEEP);
}

const CONE_R1 = 40, CONE_R2 = 16, CONE_H = 50, CONE_SWEEP = 170 * DEG;
function conePatch() {
  // A slanted line revolved 170 degrees: a cone frustum, also developable.
  // Its tessellation is exactly developable for the same reason — every
  // quad lies in the plane through the cone's apex and its two rim points.
  return revolve(makeLine([CONE_R1, 0, 0], [CONE_R2, 0, CONE_H]), [0, 0, 0], [0, 0, 1], 0, CONE_SWEEP);
}

const SPHERE_R = 60;
function spherePatch(phi0Deg, phi1Deg, sweepDeg) {
  // A meridian arc revolved: a genuine sphere patch, nonzero Gaussian
  // curvature everywhere (K = 1/R^2). Poles are deliberately avoided so
  // the tessellation has no degenerate triangles of its own.
  const prof = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], SPHERE_R, phi0Deg * DEG, (phi1Deg - phi0Deg) * DEG, 2);
  return revolve(prof, [0, 0, 0], [0, 0, 1], 0, sweepDeg * DEG);
}

// Sort a ring of vertices by their true (unwrapped) revolve angle.
function ringAt(result, z) {
  const ang = (p) => { let a = Math.atan2(p[1], p[0]); if (a < -1e-9) a += 2 * Math.PI; return a; };
  return result.positions
    .map((p, i) => ({ i, p }))
    .filter((o) => Math.abs(o.p[2] - z) < 1e-9)
    .sort((a, b) => ang(a.p) - ang(b.p));
}
function uvDist(result, a, b) {
  return Math.hypot(result.uv[a][0] - result.uv[b][0], result.uv[a][1] - result.uv[b][1]);
}

// ---- developable exactness anchor: CYLINDER ----------------------------

test('DEVELOPABLE EXACTNESS ANCHOR — a cylinder patch flattens with distortion at numerical-noise level', () => {
  const r = flattenNurbsSurface(cylinderPatch(), { uRes: 10, vRes: 14 });
  const d = r.distortion;

  // The input's own discrete Gaussian curvature is zero, so by Theorema
  // Egregium an exact unrolling EXISTS — this is what makes a tight
  // assertion legitimate here and illegitimate on a curved patch.
  assert.ok(d.intrinsic.maxAbsInteriorAngleDefect < 1e-12,
    `cylinder should be developable; max interior angle defect was ${d.intrinsic.maxAbsInteriorAngleDefect}`);
  assert.strictEqual(r.developable, true);

  // Every 3D edge length must survive the flattening. Tolerance stated
  // explicitly: 1e-8 relative and 1e-6 mm absolute, on a patch whose edges
  // run 5-20mm. Measured at ~2e-11 relative, so this has three orders of
  // headroom and is still tight enough to fail on any real defect.
  assert.ok(d.edge.maxRelErr < 1e-8, `edge maxRelErr ${d.edge.maxRelErr}`);
  assert.ok(d.edge.maxAbsErr < 1e-6, `edge maxAbsErr ${d.edge.maxAbsErr} mm`);
  assert.ok(d.angle.maxDeg < 1e-6, `angle maxDeg ${d.angle.maxDeg}`);
  assert.ok(Math.abs(d.area.totalRatio - 1) < 1e-9, `area ratio ${d.area.totalRatio}`);
  assert.strictEqual(d.flippedTriangles, 0);
});

test('DEVELOPABLE GROUND TRUTH — the flattened cylinder patch IS the analytically predicted rectangle', () => {
  const r = flattenNurbsSurface(cylinderPatch(), { uRes: 10, vRes: 14 });
  const bottom = ringAt(r, 0), top = ringAt(r, CYL_H);
  assert.ok(bottom.length >= 8 && bottom.length === top.length, 'fixture should give two matching rings');

  // Confirm the fixture really is a cylinder before using cylinder maths
  // on it (never trust a fixture's label).
  for (const p of r.positions) {
    assert.ok(Math.abs(Math.hypot(p[0], p[1]) - CYL_R) < 1e-9, 'every vertex must lie on the cylinder');
  }

  // GROUND TRUTH, from the cylinder's own equation, not from the flattener:
  // unrolling a prism strip lays its polygonal cross-section out straight,
  // so the unrolled width is the sum of the cross-section's chords,
  // 2R*sin(dTheta/2) per step; the unrolled height is exactly the cylinder's
  // own height.
  const ang = (p) => { let a = Math.atan2(p[1], p[0]); if (a < -1e-9) a += 2 * Math.PI; return a; };
  let analyticWidth = 0;
  for (let i = 1; i < bottom.length; i++) {
    analyticWidth += 2 * CYL_R * Math.sin((ang(bottom[i].p) - ang(bottom[i - 1].p)) / 2);
  }
  const measuredWidth = uvDist(r, bottom[0].i, bottom[bottom.length - 1].i);
  assert.ok(Math.abs(measuredWidth - analyticWidth) < 1e-7,
    `unrolled width: analytic ${analyticWidth}, measured ${measuredWidth}`);

  assert.ok(Math.abs(uvDist(r, bottom[0].i, top[0].i) - CYL_H) < 1e-7, 'unrolled height at one end');
  assert.ok(Math.abs(uvDist(r, bottom[bottom.length - 1].i, top[top.length - 1].i) - CYL_H) < 1e-7,
    'unrolled height at the other end');

  // ...and it is genuinely a RECTANGLE, not merely the right size: the
  // corner between the unrolled cross-section and the unrolled ruling is a
  // right angle.
  const o = r.uv[bottom[0].i];
  const v1 = [r.uv[bottom[bottom.length - 1].i][0] - o[0], r.uv[bottom[bottom.length - 1].i][1] - o[1]];
  const v2 = [r.uv[top[0].i][0] - o[0], r.uv[top[0].i][1] - o[1]];
  const cosA = (v1[0] * v2[0] + v1[1] * v2[1]) / (Math.hypot(...v1) * Math.hypot(...v2));
  assert.ok(Math.abs(Math.acos(cosA) / DEG - 90) < 1e-6, `corner angle ${Math.acos(cosA) / DEG} deg`);
});

// ---- developable exactness anchor: CONE --------------------------------

test('DEVELOPABLE GROUND TRUTH — a cone frustum flattens to the analytically predicted annular sector', () => {
  const r = flattenNurbsSurface(conePatch(), { uRes: 8, vRes: 12 });
  assert.strictEqual(r.developable, true);
  assert.ok(r.distortion.edge.maxRelErr < 1e-8, `cone edge maxRelErr ${r.distortion.edge.maxRelErr}`);
  assert.strictEqual(r.distortion.flippedTriangles, 0);

  // GROUND TRUTH from the cone's own geometry: the apex sits where the
  // radius reaches zero, and unrolling preserves the straight-line ruling
  // from apex to rim, so in the flattened plane EVERY rim point must lie
  // at exactly that analytic slant distance from one common apex.
  const zApex = CONE_H * CONE_R1 / (CONE_R1 - CONE_R2);
  const slant1 = Math.hypot(CONE_R1, zApex);
  const slant2 = Math.hypot(CONE_R2, zApex - CONE_H);

  const bot = ringAt(r, 0), top = ringAt(r, CONE_H);
  assert.ok(bot.length >= 8 && bot.length === top.length, 'fixture should give two matching rings');

  // Extend each ruling to where the apex must be, and check they all agree.
  const apexes = bot.map((o, k) => {
    const b = r.uv[o.i], t = r.uv[top[k].i];
    const dx = t[0] - b[0], dy = t[1] - b[1];
    const L = Math.hypot(dx, dy);
    return [b[0] + dx / L * slant1, b[1] + dy / L * slant1];
  });
  let spread = 0;
  for (let i = 1; i < apexes.length; i++) {
    spread = Math.max(spread, Math.hypot(apexes[i][0] - apexes[0][0], apexes[i][1] - apexes[0][1]));
  }
  assert.ok(spread < 1e-6, `unrolled rulings must converge to one apex; spread ${spread} mm`);

  const A = apexes[0];
  for (const o of bot) {
    const d = Math.hypot(r.uv[o.i][0] - A[0], r.uv[o.i][1] - A[1]);
    assert.ok(Math.abs(d - slant1) < 1e-6, `outer rim radius ${d}, analytic ${slant1}`);
  }
  for (const o of top) {
    const d = Math.hypot(r.uv[o.i][0] - A[0], r.uv[o.i][1] - A[1]);
    assert.ok(Math.abs(d - slant2) < 1e-6, `inner rim radius ${d}, analytic ${slant2}`);
  }
});

test('a flat plane is exact (smoke test, not a fixture)', () => {
  const plane = extrude(makeLine([0, 0, 0], [37, 0, 0]), [0, 1, 0], 23);
  const r = flattenNurbsSurface(plane, { uRes: 5, vRes: 7 });
  assert.strictEqual(r.developable, true);
  assert.ok(r.distortion.edge.maxAbsErr < 1e-9, `plane edge maxAbsErr ${r.distortion.edge.maxAbsErr}`);
  assert.ok(r.distortion.angle.maxDeg < 1e-8, `plane angle maxDeg ${r.distortion.angle.maxDeg}`);
});

// ---- doubly curved: distortion must be real, and must track reality ----

test('THEOREMA EGREGIUM — a doubly-curved sphere patch reports genuinely nonzero distortion, and says why', () => {
  const r = flattenNurbsSurface(spherePatch(10, 80, 140), { uRes: 10, vRes: 10 });
  const d = r.distortion;

  assert.strictEqual(r.developable, false);
  assert.ok(d.intrinsic.maxAbsInteriorAngleDefect > 1e-3,
    `a sphere patch must have real Gaussian curvature; defect ${d.intrinsic.maxAbsInteriorAngleDefect}`);
  assert.match(r.note, /Theorema Egregium/);

  // A distortion metric that is always tiny is decorative. On this patch
  // the flattening genuinely costs real material: assert real magnitudes,
  // not just "greater than zero".
  assert.ok(d.area.maxRelErr > 0.05, `area maxRelErr ${d.area.maxRelErr} should be a real, visible cost`);
  assert.ok(d.edge.maxRelErr > 0.02, `edge maxRelErr ${d.edge.maxRelErr}`);
  assert.ok(d.angle.maxDeg > 0.5, `angle maxDeg ${d.angle.maxDeg}`);
  assert.ok(d.edge.maxAbsErr > 0.1, `edge maxAbsErr ${d.edge.maxAbsErr} mm — a real, cuttable error`);
});

test('the reported distortion TRACKS REALITY — a larger patch of the same sphere distorts more, on every metric', () => {
  const small = flattenNurbsSurface(spherePatch(35, 50, 40), { uRes: 10, vRes: 10 });
  const large = flattenNurbsSurface(spherePatch(10, 80, 140), { uRes: 10, vRes: 10 });

  const s = small.distortion, l = large.distortion;
  assert.ok(l.intrinsic.maxAbsInteriorAngleDefect > 2 * s.intrinsic.maxAbsInteriorAngleDefect,
    `curvature: ${s.intrinsic.maxAbsInteriorAngleDefect} -> ${l.intrinsic.maxAbsInteriorAngleDefect}`);
  assert.ok(l.angle.rmsDeg > 2 * s.angle.rmsDeg, `angle rmsDeg: ${s.angle.rmsDeg} -> ${l.angle.rmsDeg}`);
  assert.ok(l.area.rmsRelErr > 2 * s.area.rmsRelErr, `area rmsRelErr: ${s.area.rmsRelErr} -> ${l.area.rmsRelErr}`);
  assert.ok(l.edge.rmsRelErr > 2 * s.edge.rmsRelErr, `edge rmsRelErr: ${s.edge.rmsRelErr} -> ${l.edge.rmsRelErr}`);

  // ...and the small patch, while genuinely curved, is not reported as
  // developable either. No silent rounding of a real cost down to zero.
  assert.strictEqual(small.developable, false);
  assert.ok(s.area.rmsRelErr > 1e-4, `small patch area rmsRelErr ${s.area.rmsRelErr}`);
});

test('CONFORMALITY — LSCM preserves ANGLES better than it preserves AREAS or LENGTHS', () => {
  // This is the property that proves the implementation is LSCM and not
  // something else: it minimizes the conformal (angle) energy, so on a
  // genuinely curved patch the angle error must come out materially
  // smaller than the area and length errors it never optimized.
  for (const patch of [spherePatch(35, 50, 40), spherePatch(10, 80, 140)]) {
    const d = flattenNurbsSurface(patch, { uRes: 10, vRes: 10 }).distortion;
    assert.ok(d.angle.rmsRel < 0.5 * d.area.rmsRelErr,
      `angle rmsRel ${d.angle.rmsRel} should be well under area rmsRelErr ${d.area.rmsRelErr}`);
    assert.ok(d.angle.rmsRel < 0.7 * d.edge.rmsRelErr,
      `angle rmsRel ${d.angle.rmsRel} should be under edge rmsRelErr ${d.edge.rmsRelErr}`);
    assert.ok(d.angle.rmsRel > 0, 'angle distortion is still real, not zero');
  }
});

test('every returned coordinate and every reported number is finite', () => {
  for (const r of [
    flattenNurbsSurface(cylinderPatch(), { uRes: 10, vRes: 14 }),
    flattenNurbsSurface(conePatch(), { uRes: 8, vRes: 12 }),
    flattenNurbsSurface(spherePatch(10, 80, 140), { uRes: 10, vRes: 10 }),
  ]) {
    for (const [u, v] of r.uv) {
      assert.ok(Number.isFinite(u) && Number.isFinite(v), 'uv must be finite');
    }
    const walk = (o, path) => {
      for (const [k, v] of Object.entries(o)) {
        if (typeof v === 'number') assert.ok(Number.isFinite(v), `${path}.${k} = ${v}`);
        else if (v && typeof v === 'object' && !Array.isArray(v)) walk(v, `${path}.${k}`);
      }
    };
    walk(r.distortion, 'distortion');
    assert.ok(Number.isFinite(r.scaleApplied) && r.scaleApplied > 0);
    assert.ok(Number.isFinite(r.solver.residual));
  }
});

// ---- honest refusals ---------------------------------------------------

test('refuses a DEGENERATE TRIANGLE by name (zero area)', () => {
  assert.throws(
    () => flattenLSCM({ positions: [[0, 0, 0], [1, 0, 0], [2, 0, 0]], faces: [[0, 1, 2]] }),
    /DEGENERATE TRIANGLE \(zero area/,
  );
});

test('refuses a DEGENERATE TRIANGLE by name (a repeated vertex)', () => {
  assert.throws(
    () => flattenLSCM({ positions: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces: [[0, 1, 1]] }),
    /DEGENERATE TRIANGLE \(it uses the same vertex twice/,
  );
});

test('refuses a NON-MANIFOLD mesh by name', () => {
  assert.throws(
    () => flattenLSCM({
      positions: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
      faces: [[0, 1, 2], [0, 1, 3], [0, 1, 4]],
    }),
    /NON-MANIFOLD/,
  );
});

test('refuses a DISCONNECTED mesh by name', () => {
  assert.throws(
    () => flattenLSCM({
      positions: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]],
      faces: [[0, 1, 2], [3, 4, 5]],
    }),
    /DISCONNECTED/,
  );
});

test('refuses a mesh that is not a TOPOLOGICAL DISK — a full revolve welded shut at its seam is an annulus', () => {
  const tube = revolve(makeLine([40, 0, 0], [40, 0, 55]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  assert.throws(() => flattenNurbsSurface(tube, { uRes: 6, vRes: 16 }), /TOPOLOGICAL DISK/);
});

test('refuses a CLOSED mesh (no boundary at all) by name', () => {
  // A tetrahedron: manifold, connected, but with nowhere for the surface
  // to open out into the plane.
  const positions = [[0, 0, 0], [10, 0, 0], [0, 11, 0], [0, 0, 12]];
  const faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
  assert.throws(() => flattenLSCM({ positions, faces }), /CLOSED/);
});

test('refuses TOO FEW FACES / vertices to constrain the solve, by name', () => {
  assert.throws(() => flattenLSCM({ positions: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces: [] }),
    /no triangles at all/);
  assert.throws(() => flattenLSCM({ positions: [[0, 0, 0], [1, 0, 0]], faces: [[0, 1, 0]] }),
    /at least 3/);
});

test('a single triangle is legal and flattens exactly (the smallest well-posed case)', () => {
  const r = flattenLSCM({ positions: [[0, 0, 0], [10, 0, 0], [0, 7, 0]], faces: [[0, 1, 2]] });
  assert.ok(r.distortion.edge.maxAbsErr < 1e-12, `edge maxAbsErr ${r.distortion.edge.maxAbsErr}`);
  assert.strictEqual(r.distortion.flippedTriangles, 0);
});

// ---- supporting machinery ----------------------------------------------

test('weldTriangulation turns a non-indexed tessellation into a genuinely shared-vertex mesh', () => {
  const tris = tessellateTrimmedSurface(cylinderPatch(), null, 10, 14);
  const mesh = weldTriangulation(tris);
  assert.strictEqual(mesh.faces.length, tris.length);
  // 280 triangles carry 840 raw corners; a welded 11x15 grid has 165.
  assert.strictEqual(mesh.positions.length, 11 * 15);
  const topo = buildMeshTopology(mesh.positions, mesh.faces);
  assert.strictEqual(topo.componentCount, 1);
  assert.strictEqual(topo.nonManifoldEdges.length, 0);
  assert.ok(topo.boundaryEdges.length > 0, 'an open patch must have a boundary');
  // Euler characteristic of a disk.
  assert.strictEqual(topo.usedVertices.size - topo.edges.length + topo.faceCount, 1);
});

test('triangleLocalFrame lays one triangle out isometrically', () => {
  const p1 = [3, -2, 7], p2 = [3 + 5, -2, 7], p3 = [3, -2 + 4, 7 + 3];
  const fr = triangleLocalFrame(p1, p2, p3);
  const d = (ax, ay, bx, by) => Math.hypot(bx - ax, by - ay);
  assert.ok(Math.abs(d(fr.x1, fr.y1, fr.x2, fr.y2) - 5) < 1e-12);
  assert.ok(Math.abs(d(fr.x1, fr.y1, fr.x3, fr.y3) - 5) < 1e-12); // |p3-p1| = hypot(4,3)
  assert.ok(fr.twoArea > 0 && Number.isFinite(fr.twoArea));
});

test('solveSparseLeastSquares solves a small overdetermined system to its known least-squares answer', () => {
  // A = [[1,0],[0,1],[1,1]], b = [1,2,4]  ->  x = [4/3, 7/3] (hand-derived
  // from the normal equations [[2,1],[1,2]] x = [5,6]).
  const triplets = [
    { row: 0, col: 0, val: 1 },
    { row: 1, col: 1, val: 1 },
    { row: 2, col: 0, val: 1 }, { row: 2, col: 1, val: 1 },
  ];
  const r = solveSparseLeastSquares(triplets, [1, 2, 4], 2);
  assert.ok(r.converged);
  assert.ok(Math.abs(r.x[0] - 4 / 3) < 1e-9, `x0 ${r.x[0]}`);
  assert.ok(Math.abs(r.x[1] - 7 / 3) < 1e-9, `x1 ${r.x[1]}`);
});

test('isDevelopable separates a cylinder from a sphere patch before any solve is paid for', () => {
  const cyl = weldTriangulation(tessellateTrimmedSurface(cylinderPatch(), null, 8, 12));
  const sph = weldTriangulation(tessellateTrimmedSurface(spherePatch(10, 80, 140), null, 8, 12));
  const a = isDevelopable(cyl.positions, cyl.faces);
  const b = isDevelopable(sph.positions, sph.faces);
  assert.strictEqual(a.developable, true);
  assert.strictEqual(b.developable, false);
  assert.ok(b.maxAbsInteriorAngleDefect > 1e-3);
});

test('nurbsSurfaceArea matches the cylinder patch\'s analytic area, and the tessellation\'s own loss is reported honestly', () => {
  const cyl = cylinderPatch();
  const analytic = CYL_R * CYL_SWEEP * CYL_H;
  const measured = nurbsSurfaceArea(cyl);
  assert.ok(Math.abs(measured - analytic) / analytic < 1e-9,
    `analytic ${analytic}, integrated ${measured}`);

  const r = flattenNurbsSurface(cyl, { uRes: 10, vRes: 14 });
  const t = r.tessellation;
  assert.ok(Number.isFinite(t.trueSurfaceArea) && Number.isFinite(t.areaLostToTessellation));
  // A chord always undercuts its arc, so a triangulated curved patch is
  // always SMALLER than the true surface — and that loss happens before
  // the flattening does anything at all. Reporting it separately is the
  // point: this patch loses ~0.27% to tessellation alone.
  assert.ok(t.tessellatedArea < t.trueSurfaceArea, 'tessellation must undercut the true surface');
  assert.ok(t.areaLostToTessellation > 0 && t.areaRatioVsTrue < 1 && t.areaRatioVsTrue > 0.99);
});

test('flatteningDistortion is a real measurement, not a report from the solver — a deliberately wrong layout scores badly', () => {
  const r = flattenNurbsSurface(cylinderPatch(), { uRes: 8, vRes: 10 });
  const good = flatteningDistortion(r.positions, r.faces, r.uv);
  // Squash the layout in one direction: the map is still a valid function,
  // just no longer conformal or isometric. The metrics must notice.
  const squashed = r.uv.map(([u, v]) => [u, v * 0.5]);
  const bad = flatteningDistortion(r.positions, r.faces, squashed);
  assert.ok(good.angle.maxDeg < 1e-6 && bad.angle.maxDeg > 10,
    `good ${good.angle.maxDeg} deg vs squashed ${bad.angle.maxDeg} deg`);
  assert.ok(bad.area.totalRatio < 0.6, `squashed area ratio ${bad.area.totalRatio}`);
  // Mirroring the layout must be caught as a flip, not silently accepted.
  const mirrored = r.uv.map(([u, v]) => [-u, v]);
  const flip = flatteningDistortion(r.positions, r.faces, mirrored);
  assert.strictEqual(flip.flippedTriangles, r.faces.length);
});

test('validateFlattenMesh is reusable on its own and returns the topology it checked', () => {
  const mesh = weldTriangulation(tessellateTrimmedSurface(conePatch(), null, 6, 9));
  const topo = validateFlattenMesh(mesh.positions, mesh.faces);
  assert.strictEqual(topo.componentCount, 1);
  assert.strictEqual(topo.faceCount, mesh.faces.length);
  assert.ok(topo.boundaryVertices.size > 0);
});

// ================================================================
// POLE / CONE-APEX REGRESSION
// ================================================================
// A revolve whose profile TOUCHES the axis (a cone with its apex, a
// sphere, an ellipsoid) has a POLE: the whole pole row of the parameter
// grid collapses to one point. `tessellateTrimmedSurface` emits two
// triangles per grid cell unconditionally, so after welding, the pole
// cell's second triangle has a repeated vertex — a face the NURBS entry
// point manufactured ITSELF and `validateFlattenMesh` then correctly
// refused, making every pole-bearing surface unflattenable, INCLUDING a
// partial cone sector that is exactly developable.
//
// A repeated-vertex face carries zero area and zero topology (the "quad"
// at a pole is genuinely a triangle), so dropping it is lossless, not a
// tolerance. It is done ONLY in the NURBS entry points, where the weld
// that created the problem happened — `validateFlattenMesh` keeps
// refusing a degenerate face in a mesh a CALLER supplied, which is a
// different situation and a real thing to catch.
test('dropDegenerateFaces removes exactly the repeated-vertex faces welding created, and nothing else', () => {
  const cone = revolve(makeLine([30, 0, 0], [0, 0, 50]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const raw = weldTriangulation(tessellateTrimmedSurface(cone, null, 8, 12));
  const bad = raw.faces.filter(([a, b, c]) => a === b || b === c || c === a);
  assert.ok(bad.length > 0, 'the pole must actually produce repeated-vertex faces, or this test proves nothing');
  const cleaned = dropDegenerateFaces(raw);
  assert.strictEqual(cleaned.faces.length, raw.faces.length - bad.length);
  assert.strictEqual(cleaned.droppedFaces, bad.length);
  assert.ok(cleaned.faces.every(([a, b, c]) => a !== b && b !== c && c !== a));
  assert.strictEqual(cleaned.positions, raw.positions, 'positions are untouched — not one coordinate moves');
  // Zero area dropped: the summed triangle area is bit-for-bit unchanged.
  const area = (fs) => fs.reduce((s, [a, b, c]) => {
    const p = raw.positions, ux = p[b][0] - p[a][0], uy = p[b][1] - p[a][1], uz = p[b][2] - p[a][2];
    const vx = p[c][0] - p[a][0], vy = p[c][1] - p[a][1], vz = p[c][2] - p[a][2];
    return s + 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
  }, 0);
  assert.strictEqual(area(cleaned.faces), area(raw.faces));
});

test('a CONE SECTOR (a pole plus a partial sweep) flattens EXACTLY — the pole no longer blocks it', () => {
  // 90-degree sweep, so the apex sits on the domain corner and is a
  // BOUNDARY vertex: all of the cone's Gaussian curvature is concentrated
  // at that one point, and putting it on the boundary is precisely what
  // makes the sector isometric to the plane.
  const R = 30, H = 50, sweep = Math.PI / 2;
  const sector = revolve(makeLine([R, 0, 0], [0, 0, H]), [0, 0, 0], [0, 0, 1], 0, sweep);
  const r = flattenNurbsSurface(sector, { uRes: 10, vRes: 24 });
  assert.strictEqual(r.developable, true, `expected developable, defect ${r.distortion.intrinsic.maxAbsInteriorAngleDefect}`);
  assert.ok(r.distortion.area.maxRelErr < 1e-6, `area err ${r.distortion.area.maxRelErr}`);
  assert.ok(r.distortion.edge.maxRelErr < 1e-6, `edge err ${r.distortion.edge.maxRelErr}`);
  // The unrolled sector's own radial edge is the cone's SLANT LENGTH — an
  // exact analytic number this construction never had a chance to fit to.
  const slant = Math.hypot(R, H);
  let maxR = 0;
  for (const [u, v] of r.uv) maxR = Math.max(maxR, Math.hypot(u, v));
  // uv is translated to a (0,0)-based bbox, so measure the layout's own
  // longest edge instead: the apex-to-rim ruling, present as a real mesh edge.
  const poleIdx = r.positions.findIndex((p) => Math.hypot(p[0], p[1]) < 1e-9);
  assert.ok(poleIdx >= 0, 'the welded pole vertex must exist');
  let maxSpan = 0;
  for (let i = 0; i < r.uv.length; i++) {
    maxSpan = Math.max(maxSpan, Math.hypot(r.uv[i][0] - r.uv[poleIdx][0], r.uv[i][1] - r.uv[poleIdx][1]));
  }
  assert.ok(Math.abs(maxSpan - slant) < 1e-6 * slant,
    `unrolled apex-to-rim distance ${maxSpan} should be the exact slant length ${slant}`);
});

test('a FULL cone with its apex is honestly reported as NOT developable — the apex is a real concentrated-curvature point', () => {
  // Mathematically load-bearing, and the opposite of a bug: a cone's whole
  // Gaussian curvature lives at its apex (total 2*pi - 2*pi*R/slant). With
  // the apex INTERIOR (a full 360-degree sweep is a topological disk whose
  // only boundary is the rim), no isometric flattening exists — unrolling a
  // cone genuinely requires a cut out to the boundary first. So the honest
  // answer here is "doubly curved", with the real number, not "exact".
  const R = 30, H = 50;
  const cone = revolve(makeLine([R, 0, 0], [0, 0, H]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const expected = 2 * Math.PI * (1 - R / Math.hypot(R, H)); // the exact cone-point curvature
  const r = flattenNurbsSurface(cone, { uRes: 10, vRes: 64 });
  assert.strictEqual(r.developable, false);
  const defect = r.distortion.intrinsic.maxAbsInteriorAngleDefect;
  assert.ok(Math.abs(defect - expected) < 2e-3,
    `apex angle defect ${defect} should be the analytic cone-point curvature ${expected}`);
  // And it is a genuine discretization limit, not a fudged tolerance: the
  // measured defect CONVERGES onto the analytic value as the sweep is
  // sampled more finely.
  const coarse = flattenNurbsSurface(cone, { uRes: 10, vRes: 16 }).distortion.intrinsic.maxAbsInteriorAngleDefect;
  const fine = flattenNurbsSurface(cone, { uRes: 10, vRes: 128 }).distortion.intrinsic.maxAbsInteriorAngleDefect;
  assert.ok(Math.abs(fine - expected) < Math.abs(defect - expected)
         && Math.abs(defect - expected) < Math.abs(coarse - expected),
    `defect must converge: coarse ${coarse}, mid ${defect}, fine ${fine}, analytic ${expected}`);
  // The cost of pretending otherwise is enormous and honestly measured.
  assert.ok(r.distortion.area.maxRelErr > 0.5, `apex area error ${r.distortion.area.maxRelErr}`);
});
