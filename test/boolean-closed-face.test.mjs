import test from 'node:test';
import assert from 'node:assert';
import { booleanSolids } from '../kernel/boolean.mjs';
import { fragmentBoundaries3D, insertTJunctionVertices } from '../kernel/booleansew.mjs';
import { intersectSurfacesComplete } from '../kernel/ssi.mjs';
import { makeLine, revolve } from '../kernel/primitives.mjs';
import { projectPointsToSurfaceUV, trivialTrimLoop } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import { surfaceClosure } from '../kernel/surface.mjs';

// ---------------------------------------------------------------------------
// THE FIXTURE: a CYLINDER passing clean through a BOX. Its own hard case is
// neither the pole (boolean-organic.test.mjs owns that) nor the T-junction on
// a straight edge (boolean-tjunction.test.mjs owns that) — it is a FACE THAT
// WRAPS A CLOSED DIRECTION AND IS CUT ALL THE WAY AROUND.
//
// Three things only this shape reaches:
//
//   1. The cut curve is a full circle. It is CLOSED in 3D, but in the wall's
//      own (u,v) it runs off one edge of the domain rectangle and resumes at
//      the other, because u=uMin and u=uMax are the same physical place. Fed
//      to the arrangement as sampled, it reads as an out-and-back that stops
//      short of both edges and is pruned as a dangling spur, and the face
//      comes back UNSPLIT.
//
//   2. What survives the cut is a TUBE, not a disk. Its corner polygon
//      revisits the seam, so it has no boundary a solid builder can accept.
//
//   3. Cutting that tube open puts a new vertex partway along the shared
//      intersection circle — a point on the true curve, and therefore off the
//      neighbouring cap's straight chord by that chord's own sagitta. Both
//      sides then disagree about how the shared edge is subdivided, at a
//      distance far too large for an on-edge test to accept on proximity.
//
// The cylinder passes entirely through the box in Z and sits well inside it in
// X and Y, so the wall meets only the box's two caps and every intersection is
// one of these full circles. That is deliberate: a partial overlap would give
// open arcs, which are the case that already worked.
// ---------------------------------------------------------------------------

function quadSurface(p00, p10, p11, p01) {
  const w = (p) => [p[0], p[1], p[2], 1];
  return {
    degU: 1, degV: 1,
    knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[w(p00), w(p01)], [w(p10), w(p11)]],
  };
}

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

const R = 12, Z0 = -30, Z1 = 30, BOX = 20;

// Real face-pair SSI, not hand-derived circles: the whole point of the fixture
// is what the marcher's own discretization does to a wrapped cut.
function buildFixture() {
  const wall = revolve(makeLine([R, 0, Z0], [R, 0, Z1]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
  const capBot = revolve(makeLine([0, 0, Z0], [R, 0, Z0]), [0, 0, Z0], [0, 0, -1], 0, 2 * Math.PI);
  const capTop = revolve(makeLine([0, 0, Z1], [R, 0, Z1]), [0, 0, Z1], [0, 0, 1], 0, 2 * Math.PI);
  const cyl = { faces: [{ srf: wall }, { srf: capBot }, { srf: capTop }] };
  cyl.triangles = tessellate(cyl.faces, 24);

  const bx = { faces: boxFaces([-BOX, -BOX, -BOX], [BOX, BOX, BOX]) };
  bx.triangles = tessellate(bx.faces, 2);

  const curves = [];
  for (let i = 0; i < cyl.faces.length; i++) {
    for (let j = 0; j < bx.faces.length; j++) {
      let r;
      try { r = intersectSurfacesComplete(cyl.faces[i].srf, bx.faces[j].srf); } catch { continue; }
      if (!r || !r.ok) continue;
      for (const comp of r.components) {
        if (comp.samples.length < 2) continue;
        curves.push({ samples: comp.samples.map((s) => s.point), faceA: i, faceB: j });
      }
    }
  }
  return { cyl, bx, curves };
}

test('the fixture really is the hard case: a closed face cut by a closed curve', () => {
  const { cyl, curves } = buildFixture();
  const wall = cyl.faces[0].srf;

  const closure = surfaceClosure(wall);
  assert.ok(closure.closedU || closure.closedV, 'the wall wraps a closed direction');

  const wallCurves = curves.filter((c) => c.faceA === 0);
  assert.equal(wallCurves.length, 2, 'the wall crosses both box caps');

  for (const c of wallCurves) {
    const first = c.samples[0], last = c.samples[c.samples.length - 1];
    const gap = Math.hypot(first[0] - last[0], first[1] - last[1], first[2] - last[2]);
    let step = 0;
    for (let i = 1; i < c.samples.length; i++) {
      const p = c.samples[i], q = c.samples[i - 1];
      step = Math.max(step, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
    }
    assert.ok(gap <= step, `the cut is closed in 3D (gap ${gap} vs step ${step})`);

    // ...and yet discontinuous in the wall's own parameters: the projected
    // chain jumps more than half the closed direction's span somewhere.
    //
    // THAT ALONE DOES NOT DISCRIMINATE, which is the point of what follows. An
    // ordinary hole sitting ON the seam also jumps — it crosses the seam going
    // out and crosses back coming home. What separates the two is the NET
    // WINDING: a straddling hole's crossings cancel to zero, while a genuine
    // wrap crosses an odd number of times and nets +/-1, because it goes all
    // the way around the closed direction and never comes back. The closing
    // segment is included, so this is the winding number of the whole closed
    // loop, not of an arbitrary cut of it.
    const uv = projectPointsToSurfaceUV(c.samples, wall).uv;
    const axis = closure.closedU ? 0 : 1;
    const knots = closure.closedU ? wall.knotsU : wall.knotsV;
    const span = knots[knots.length - 1] - knots[0];
    let biggest = 0, crossings = 0, winding = 0;
    for (let i = 0; i < uv.length; i++) {
      const d = uv[(i + 1) % uv.length][axis] - uv[i][axis];
      biggest = Math.max(biggest, Math.abs(d));
      if (Math.abs(d) > span / 2) { crossings++; winding += d > 0 ? -1 : 1; }
    }
    assert.ok(biggest > span / 2, `the projection jumps (biggest step ${biggest} of span ${span})`);
    assert.equal(crossings % 2, 1, `an odd number of seam crossings (${crossings}) — a straddle would be even`);
    assert.equal(Math.abs(winding), 1, `net winding +/-1 (got ${winding}) — a straddle would net 0`);
  }
});

test('the wrapped face is genuinely cut, into an inside and two outsides', () => {
  const { cyl, bx, curves } = buildFixture();
  const res = booleanSolids(cyl, bx, curves, 'union');
  const wallReport = res.faceReports.find((r) => r.label === 'A face 0');
  assert.equal(wallReport.curves, 2, 'both cut curves reached the wall');
  assert.equal(wallReport.fragments, 3, 'and split it in three');
  assert.deepEqual(wallReport.regions, ['outside', 'inside', 'outside'],
    'the middle band is the part inside the box');
});

test('a cylinder through a box UNIONS into one closed solid', () => {
  const { cyl, bx, curves } = buildFixture();
  const res = booleanSolids(cyl, bx, curves, 'union');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.stats.nakedEdgeCount, 0, 'every shared edge welded');
  assert.equal(res.stats.chi, 2, 'one closed solid');
  assert.equal(res.stats.shellCount, 1);
  assert.ok(res.worstSharedGap < 1e-6, `worst weld gap ${res.worstSharedGap}`);
});

test('the same pair INTERSECTS into the cylinder segment inside the box', () => {
  const { cyl, bx, curves } = buildFixture();
  const res = booleanSolids(cyl, bx, curves, 'intersect');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.chi, 2, 'a single closed solid');
  assert.equal(res.stats.shellCount, 1);
});

test('the same pair DIFFERENCES into the two ends left outside the box', () => {
  // A minus B is the cylinder minus the box, and the box cuts a band out of
  // the middle of it — so what is left is genuinely two separate closed
  // solids, one at each end.
  const { cyl, bx, curves } = buildFixture();
  const res = booleanSolids(cyl, bx, curves, 'difference');
  assert.ok(res.ok, res.reason || res.verdict);
  assert.equal(res.stats.nakedEdgeCount, 0);
  assert.equal(res.stats.shellCount, 2);
  assert.equal(res.stats.chi, 4, 'chi 2 per shell');
});

test('the cut-open seam is what leaves the naked edges, and the T-junction pass is what closes them', () => {
  // The negative control, on the REAL fragment loops. Cutting the wrapped face
  // open introduces a vertex partway along the shared intersection circle that
  // the box cap has no reason to place; without the pass those edges stay
  // one-sided. Counting before and after is what proves the union above closes
  // for this reason and not some other.
  const { cyl, bx, curves } = buildFixture();
  const res = booleanSolids(cyl, bx, curves, 'union');
  assert.ok(res.fragments, 'the kept set is available to count against');

  const loops = [];
  for (const f of res.fragments) {
    const b = fragmentBoundaries3D(f.srf, f);
    assert.ok(b.ok, b.reason);
    loops.push(...b.loops);
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

  assert.equal(nakedCount(loops), 12, 'the raw fragment loops genuinely do not close');
  assert.equal(nakedCount(insertTJunctionVertices(loops, TOL)), 0, 'and the pass is what closes them');
});
