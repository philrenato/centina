import test from 'node:test';
import assert from 'node:assert';
import { classifyPointInSolid, classifyFragment, spiralDirections, keepRuleFor, DEGENERATE_TRIANGLE_AREA_RATIO } from '../kernel/classify.mjs';
import { splitFaceByCurves } from '../kernel/facesplit.mjs';
import { revolve } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { trivialTrimLoop } from '../kernel/trim.mjs';
import { tessellateTrimmedSurface } from '../kernel/trimtess.mjs';
import { sub, cross, length } from '../kernel/vec3.mjs';

// A cube from (0,0,0) to (10,10,10), as 12 triangles, wound consistently
// outward. Every expected answer below follows from the cube's own bounds
// written down directly — never from what the classifier said last time.
function cube(lo = 0, hi = 10) {
  const v = [
    [lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo],
    [lo, lo, hi], [hi, lo, hi], [hi, hi, hi], [lo, hi, hi],
  ];
  const quad = (a, b, c, d) => [[v[a], v[b], v[c]], [v[a], v[c], v[d]]];
  return [
    ...quad(0, 3, 2, 1), // z = lo
    ...quad(4, 5, 6, 7), // z = hi
    ...quad(0, 1, 5, 4), // y = lo
    ...quad(3, 7, 6, 2), // y = hi
    ...quad(0, 4, 7, 3), // x = lo
    ...quad(1, 2, 6, 5), // x = hi
  ];
}

test('a point at the centre is inside, and a point well outside is outside', () => {
  const tris = cube();
  assert.equal(classifyPointInSolid(tris, [5, 5, 5]).region, 'inside');
  assert.equal(classifyPointInSolid(tris, [50, 5, 5]).region, 'outside');
  assert.equal(classifyPointInSolid(tris, [-1, -1, -1]).region, 'outside');
});

test('a point ON a face is BOUNDARY, not silently forced to one side — the case a two-state test has to guess', () => {
  const tris = cube();
  const r = classifyPointInSolid(tris, [5, 5, 0]);
  assert.equal(r.region, 'boundary');
  assert.equal(r.raysUsed, 0, 'the boundary answer is reached without casting anything');
  // On an edge and on a corner too — both are boundary, not inside/outside.
  assert.equal(classifyPointInSolid(tris, [0, 5, 0]).region, 'boundary');
  assert.equal(classifyPointInSolid(tris, [0, 0, 0]).region, 'boundary');
});

test('a point DELIBERATELY NEAR a face still classifies correctly — the named hard case', () => {
  const tris = cube();
  // Far closer to the face than anything a tessellation would blur, but
  // genuinely off it: these must be inside and outside respectively, not
  // collapsed to 'boundary' by an over-wide tolerance.
  assert.equal(classifyPointInSolid(tris, [5, 5, 0.001]).region, 'inside');
  assert.equal(classifyPointInSolid(tris, [5, 5, -0.001]).region, 'outside');
});

test('a ray aimed exactly along an EDGE is rejected as degenerate rather than counted — and the point still classifies', () => {
  const tris = cube();
  // From the centre toward the x=10 / y=10 edge: the hit lands exactly on the
  // seam shared by two triangles, where parity counting is ill-defined.
  const edgeDir = [1, 1, 0];
  const forced = classifyPointInSolid(tris, [5, 5, 5], { directions: [edgeDir] });
  assert.equal(forced.region, null, 'a single grazing ray yields no answer rather than a guess');
  assert.match(forced.reason, /degenerate|graze/i);
  // With the ordinary direction set it recovers and gets the right answer.
  assert.equal(classifyPointInSolid(tris, [5, 5, 5]).region, 'inside');
});

test('a ray lying IN a face plane is rejected too, not counted as zero crossings', () => {
  const tris = cube();
  // The point must be genuinely OFF the boundary or the boundary check
  // (correctly) answers first: sit well outside the cube but exactly coplanar
  // with its z=0 face, then fire straight at it. The ray travels through the
  // plane of two triangles without ever crossing them transversally.
  const inPlane = classifyPointInSolid(tris, [30, 5, 0], { directions: [[-1, 0, 0]] });
  assert.equal(inPlane.region, null, 'a ray in a triangle\'s own plane gives no countable crossing');
  assert.match(inPlane.reason, /plane|degenerate/i);
  // The same point with the ordinary direction set recovers and is outside.
  assert.equal(classifyPointInSolid(tris, [30, 5, 0]).region, 'outside');
});

test('a mesh that is NOT closed refuses by name instead of returning the more popular answer', () => {
  // A cube with one face removed: rays through the hole and rays through the
  // remaining walls genuinely disagree, and that disagreement is the signal.
  const open = cube().slice(0, 10); // drop the two x=hi triangles
  const r = classifyPointInSolid(open, [5, 5, 5], { rayCount: 12 });
  assert.equal(r.region, null);
  assert.match(r.reason, /disagree|not the closed manifold/i);
});

test('the classifier is DETERMINISTIC — the same point and solid answer identically every time', () => {
  const tris = cube();
  const first = classifyPointInSolid(tris, [3, 7, 4]);
  for (let i = 0; i < 20; i++) {
    const again = classifyPointInSolid(tris, [3, 7, 4]);
    assert.equal(again.region, first.region);
    assert.equal(again.raysUsed, first.raysUsed);
  }
});

test('SCALE INDEPENDENCE: a cube 1000x larger classifies the same points the same way', () => {
  const big = cube(0, 10000);
  assert.equal(classifyPointInSolid(big, [5000, 5000, 5000]).region, 'inside');
  assert.equal(classifyPointInSolid(big, [50000, 5000, 5000]).region, 'outside');
  assert.equal(classifyPointInSolid(big, [5000, 5000, 0]).region, 'boundary');
});

test('a NON-CONVEX solid classifies correctly in its concavity — where a convex-only shortcut would be wrong', () => {
  // An L-shaped prism: the 10x10x10 cube with the (5..10, 5..10) column
  // removed for its full height. A point in the removed notch is OUTSIDE
  // even though it sits well within the cube's bounding box.
  const L2D = [[0, 0], [10, 0], [10, 5], [5, 5], [5, 10], [0, 10]];
  const tris = [];
  const zLo = 0, zHi = 10;
  for (let i = 0; i < L2D.length; i++) {
    const a = L2D[i], b = L2D[(i + 1) % L2D.length];
    tris.push([[a[0], a[1], zLo], [b[0], b[1], zLo], [b[0], b[1], zHi]]);
    tris.push([[a[0], a[1], zLo], [b[0], b[1], zHi], [a[0], a[1], zHi]]);
  }
  // Caps, fanned from vertex 0 — the L is star-shaped about it, so a fan is
  // a valid triangulation here.
  for (let i = 1; i < L2D.length - 1; i++) {
    tris.push([[L2D[0][0], L2D[0][1], zLo], [L2D[i + 1][0], L2D[i + 1][1], zLo], [L2D[i][0], L2D[i][1], zLo]]);
    tris.push([[L2D[0][0], L2D[0][1], zHi], [L2D[i][0], L2D[i][1], zHi], [L2D[i + 1][0], L2D[i + 1][1], zHi]]);
  }
  assert.equal(classifyPointInSolid(tris, [2, 2, 5]).region, 'inside', 'in the solid arm');
  assert.equal(classifyPointInSolid(tris, [8, 8, 5]).region, 'outside', 'in the notch, inside the bounding box but outside the solid');
  assert.equal(classifyPointInSolid(tris, [8, 2, 5]).region, 'inside', 'in the other arm');
});

test('an empty solid refuses rather than reporting everything outside', () => {
  const r = classifyPointInSolid([], [1, 2, 3]);
  assert.equal(r.region, null);
  assert.match(r.reason, /no boundary geometry/i);
});

test('spiralDirections are unit-length, distinct, and well spread rather than clustered', () => {
  const dirs = spiralDirections(16);
  assert.equal(dirs.length, 16);
  for (const d of dirs) {
    assert.ok(Math.abs(Math.hypot(d[0], d[1], d[2]) - 1) < 1e-9, 'unit length');
  }
  // No two directions nearly coincide — a clustered set would defeat the
  // whole point of casting several independent rays.
  for (let i = 0; i < dirs.length; i++) {
    for (let j = i + 1; j < dirs.length; j++) {
      const d = dirs[i], e = dirs[j];
      assert.ok(d[0] * e[0] + d[1] * e[1] + d[2] * e[2] < 0.995, `directions ${i} and ${j} are nearly identical`);
    }
  }
});

test('the keep-rules are the ONLY difference between the three operators', () => {
  assert.equal(keepRuleFor('union')('outside'), true);
  assert.equal(keepRuleFor('union')('inside'), false);
  assert.equal(keepRuleFor('intersect')('inside'), true);
  assert.equal(keepRuleFor('intersect')('outside'), false);
  // Difference is Intersect against a reversed operand, so its own rule reads
  // as Union's — stated rather than left implied.
  assert.equal(keepRuleFor('difference')('outside'), true);
  assert.equal(keepRuleFor('nonsense'), null);
});

// ---------------------------------------------------------------------------
// PHASE 6 -> PHASE 7 COMPOSITION. The fragments a real face split produces,
// classified against a real solid — the shape Phase 8 actually consumes.
// Deliberately end-to-end through splitFaceByCurves rather than hand-built
// fragments: the whole point is that the two phases genuinely compose.
// ---------------------------------------------------------------------------

// A flat bilinear plane at z=5, spanning x,y in [-20,20], parametrized so
// (u,v) in [0,1]^2 maps linearly onto it. Degree 1 in both directions, so the
// mapping is exact and every expected coordinate below is hand-derivable.
function planeSurface() {
  const at = (u, v) => [-20 + 40 * u, -20 + 40 * v, 5, 1];
  return {
    degU: 1, degV: 1,
    knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[at(0, 0), at(0, 1)], [at(1, 0), at(1, 1)]],
  };
}

test('a split face\'s own fragments classify inside vs outside the other solid', () => {
  const srf = planeSurface();
  // The cube spans 0..10 in x and y, so at z=5 it cuts the plane exactly at
  // u,v in [0.5, 0.75] — derived from the plane's own linear mapping, not
  // read off the splitter.
  const cut = [[0.5, 0.5], [0.75, 0.5], [0.75, 0.75], [0.5, 0.75], [0.5, 0.5]];
  const face = { outer: [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]] };
  const split = splitFaceByCurves(face, [cut]);
  assert.ok(split.ok, split.reason);
  assert.equal(split.fragments.length, 2, 'a closed interior cut gives the square and the rest');

  const tris = cube();
  const regions = split.fragments.map((f) => classifyFragment(srf, f, tris));
  for (const r of regions) assert.ok(r.region, r.reason);
  const sorted = regions.map((r) => r.region).sort();
  assert.deepEqual(sorted, ['inside', 'outside'], 'exactly one fragment lies within the cube');

  // The probe point is genuinely ON the surface, not an approximation of it —
  // z is exactly the plane's own height for both fragments.
  for (const r of regions) assert.ok(Math.abs(r.point[2] - 5) < 1e-9);

  // And the INSIDE one's own probe really does sit within the cube's bounds,
  // checked independently of the classifier's own answer.
  const insideProbe = regions.find((r) => r.region === 'inside').point;
  assert.ok(insideProbe[0] > 0 && insideProbe[0] < 10, 'probe x within the cube');
  assert.ok(insideProbe[1] > 0 && insideProbe[1] < 10, 'probe y within the cube');
});

test('a fragment with no findable interior point refuses rather than guessing', () => {
  const srf = planeSurface();
  const degenerate = { outer: [[0, 0], [1, 0]], holes: [] };
  const r = classifyFragment(srf, degenerate, cube());
  assert.equal(r.region, null);
  assert.match(r.reason, /interior point/i);
});

// ---------------------------------------------------------------------------
// COLLAPSED TRIANGLES — a mesh degeneracy that used to poison EVERY ray.
//
// A revolve collapses its whole pole row, so the tessellator emits triangles
// with two coincident corners. Their cross product is not exactly zero (the
// two "identical" corners are separate float evaluations of the same pole),
// so it is a direction made entirely of roundoff — and any ray is parallel to
// noise. One such triangle anywhere in a mesh made every cast report itself
// untrustworthy, leaving NO point classifiable against ANY solid of
// revolution: a point 100 units clear of the object refused just as flatly as
// a genuinely ambiguous one.
//
// THE FIXTURE IS A REAL REVOLVE, DELIBERATELY, and that is the whole lesson
// of how this test was written. A first version hand-built slivers with
// near-coincident corners, and a negative control proved it VACUOUS — it
// passed just as happily with the guard switched off. The real failure needs
// the exact roundoff a genuine pole evaluation produces: only there does the
// noise normal land near-perpendicular to the ray, which is what drives the
// in-plane distance to zero and trips the degeneracy report. Reproduce it
// with real geometry or not at all.
// ---------------------------------------------------------------------------

// A genuinely irregular silhouette — not a sphere or a cone. The profile
// touches the axis at both ends, so the revolve is a closed solid with a
// collapsed pole row at each end: the shape the bug actually lives in.
function wavyBlob() {
  const profile = globalCurveInterp(
    [[0, 0, 0], [12, 0, 6], [8, 0, 14], [15, 0, 24], [6, 0, 34], [0, 0, 40]], 3);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
}
function blobTriangles(res) {
  const srf = wavyBlob();
  return tessellateTrimmedSurface(srf, trivialTrimLoop(srf), res, res)
    .map((tri) => tri.map((v) => v.position));
}

test('a real solid of revolution classifies at all — the collapsed pole row must not poison every ray', () => {
  // Every resolution: the bug is not a sampling artifact, one collapsed
  // triangle anywhere is enough. res=8 (128 triangles) already reproduces it.
  for (const res of [8, 16, 40]) {
    const tris = blobTriangles(res);
    // Answers taken from the profile's own numbers, never from the classifier.
    // The profile never reaches x=100, and its widest bulge is x=15 at z=24.
    assert.equal(classifyPointInSolid(tris, [100, 100, 100]).region, 'outside',
      `res=${res}: a point plainly clear of the solid`);
    assert.equal(classifyPointInSolid(tris, [40, 0, 20]).region, 'outside',
      `res=${res}: well outside the widest bulge`);
    // The axis is the sharpest place to stand: every spiral direction from it
    // passes near a collapsed pole corner at once.
    assert.equal(classifyPointInSolid(tris, [0, 0, 20]).region, 'inside',
      `res=${res}: on the axis, between the two poles`);
    assert.equal(classifyPointInSolid(tris, [0, 0, 39]).region, 'inside',
      `res=${res}: on the axis, close under the top pole`);
  }
});

test('the degeneracy threshold sits in a real GAP — it is not tuned against this fixture', () => {
  // 2*area / longestEdge^2, the exact quantity castRay thresholds on, measured
  // across a real tessellation. If collapsed and healthy triangles were close
  // together, any threshold would be a guess trading one failure for another.
  const ratios = blobTriangles(16).map(([a, b, c]) => {
    const e1 = sub(b, a), e2 = sub(c, a);
    const maxEdge = Math.max(length(e1), length(e2), length(sub(c, b)));
    return maxEdge > 0 ? length(cross(e1, e2)) / (maxEdge * maxEdge) : 0;
  });
  // Read from the kernel rather than restated, so moving the constant
  // re-checks it against the measured gap instead of silently drifting away
  // from a number a test happens to still hold.
  const collapsed = ratios.filter((r) => r < DEGENERATE_TRIANGLE_AREA_RATIO);
  const healthy = ratios.filter((r) => r >= DEGENERATE_TRIANGLE_AREA_RATIO);
  assert.ok(collapsed.length > 0, 'the fixture genuinely contains collapsed pole triangles');
  assert.ok(healthy.length > collapsed.length, 'and is mostly real geometry');
  // The two populations are separated by many orders of magnitude, with the
  // threshold sitting in empty space between them — asserted as real clearance
  // ON EITHER SIDE of wherever the constant currently is, so a change that
  // moved it toward either population would fail here rather than pass.
  assert.ok(Math.max(...collapsed) < DEGENERATE_TRIANGLE_AREA_RATIO * 1e-3,
    `worst collapsed ratio ${Math.max(...collapsed)} should sit orders below the threshold`);
  assert.ok(Math.min(...healthy) > DEGENERATE_TRIANGLE_AREA_RATIO * 1e6,
    `flattest real triangle ${Math.min(...healthy)} should sit orders above it`);
});
