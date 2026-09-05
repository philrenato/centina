// TRIANGLE SHAPE ON A FLAT CAP.
//
// `triangulatePolygon2D` is an ear-clipper, and on a convex loop every ear it
// can find shares ONE apex vertex: an n-gon comes back as n-2 needles radiating
// from a single boundary point. Area is exact — the failure is entirely one of
// SHAPE — and shape is invisible to a rasteriser and expensive to a path
// tracer, where a needle is a poor BVH leaf and ray-intersection precision
// degrades with the triangle.
//
// `triangulateConvexFanFromCentroid` adds ONE interior vertex and fans from it.
// These tests pin the three things that must hold together: the shape floor
// actually improves, the area is still exactly covered, and the fan REFUSES
// (returns null, leaving the ear-clipper the general answer) on anything
// non-convex or degenerate rather than covering ground the polygon excludes.

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  triangulatePolygon2D,
  triangulateConvexFanFromCentroid,
  tessellateTrimmedSurface,
  tessellationArea,
} from '../kernel/trimtess.mjs';
import { makeLine, extrude } from '../kernel/primitives.mjs';

const NS = [12, 48, 96, 240];

function ngon(n, r, cx = 0, cy = 0) {
  const p = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * 2 * Math.PI;
    p.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return p;
}

// The shape a filleted box's flat top actually is: a rectangle with rounded
// corners, sampled the way a fillet samples it — densely round the arcs and
// barely at all along the straights. That non-uniformity is what makes it the
// worse fixture of the two, and measuring it is the point of having it here.
function roundedRect(halfX, halfY, r, perArc, perSide) {
  const pts = [];
  const corners = [[halfX - r, halfY - r, 0], [-halfX + r, halfY - r, Math.PI / 2],
                   [-halfX + r, -halfY + r, Math.PI], [halfX - r, -halfY + r, 3 * Math.PI / 2]];
  for (let c = 0; c < 4; c++) {
    const [cx, cy, a0] = corners[c];
    for (let i = 0; i <= perArc; i++) {
      const a = a0 + (i / perArc) * (Math.PI / 2);
      pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
    }
    const [nx, ny] = corners[(c + 1) % 4];
    const from = pts[pts.length - 1];
    const to = [nx + r * Math.cos(a0 + Math.PI / 2), ny + r * Math.sin(a0 + Math.PI / 2)];
    for (let i = 1; i < perSide; i++) {
      const s = i / perSide;
      pts.push([from[0] + (to[0] - from[0]) * s, from[1] + (to[1] - from[1]) * s]);
    }
  }
  return pts;
}

// THE INVARIANT THAT MATCHES THE PICTURE. The visible defect is streaks
// converging on ONE point of the face's boundary, and that point is a vertex
// shared by every triangle in the face. So the property to hold is a valence
// bound on the BOUNDARY: a single-apex fan puts n-2 triangles on one boundary
// vertex; a centroid fan puts exactly 2 on every boundary vertex and its only
// high-valence vertex is interior, where all the normals agree anyway.
function maxBoundaryValence(pointCount, tris) {
  const uses = new Map();
  for (const t of tris) for (const i of t) if (i < pointCount) uses.set(i, (uses.get(i) || 0) + 1);
  let worst = 0;
  for (const [, k] of uses) if (k > worst) worst = k;
  return worst;
}

function shoelace(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [u0, v0] = poly[i], [u1, v1] = poly[(i + 1) % poly.length];
    a += u0 * v1 - u1 * v0;
  }
  return Math.abs(a / 2);
}

// Minimum interior angle (degrees) and radius ratio (circumradius / 2*inradius;
// 1.0 is equilateral) over a triangulation, plus the absolute area it covers.
function quality(pts, tris) {
  let area = 0, minAngle = Infinity, worstRatio = 0;
  for (const [i, j, k] of tris) {
    const a = pts[i], b = pts[j], c = pts[k];
    const ab = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const bc = Math.hypot(c[0] - b[0], c[1] - b[1]);
    const ca = Math.hypot(a[0] - c[0], a[1] - c[1]);
    const ar = Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2;
    area += ar;
    if (ar <= 0) continue;
    const s = [ab, bc, ca].sort((x, y) => x - y);
    const cosMin = (s[1] * s[1] + s[2] * s[2] - s[0] * s[0]) / (2 * s[1] * s[2]);
    const ang = Math.acos(Math.max(-1, Math.min(1, cosMin))) * 180 / Math.PI;
    if (ang < minAngle) minAngle = ang;
    const ratio = ((ab * bc * ca) / (4 * ar)) / (2 * (2 * ar / (ab + bc + ca)));
    if (ratio > worstRatio) worstRatio = ratio;
  }
  return { area, minAngle, worstRatio };
}

test('the centroid fan raises the minimum-angle floor on a convex loop, at every sampling density', () => {
  for (const n of NS) {
    const poly = ngon(n, 45);
    const ear = quality(poly, triangulatePolygon2D(poly));
    const fan = triangulateConvexFanFromCentroid(poly);
    assert.ok(fan, `n=${n}: a regular ${n}-gon is convex, the fan must apply`);
    const q = quality(fan.points, fan.tris);

    // The ear-clipper's fan shares one apex, so every triangle carries an angle
    // of exactly half the arc one edge subtends: 180/n degrees.
    assert.ok(Math.abs(ear.minAngle - 180 / n) < 1e-6,
      `n=${n}: ear-clip min angle ${ear.minAngle} should be the single-apex signature ${180 / n}`);
    /* Concentric rings, not one fan, so the floor no longer falls with n: a
       single fan gives every triangle the arc it subtends, 360/n, which is
       1.9 degrees by n=192 and is what a tracer draws as radial streaks. The
       rings hold an ABSOLUTE floor instead, which is the whole point of them
       and the claim worth pinning. */
    assert.ok(q.minAngle >= 15,
      `n=${n}: the ringed fan must hold an absolute angle floor, got ${q.minAngle}`);
    assert.ok(q.minAngle > ear.minAngle * 1.999,
      `n=${n}: the floor must at least double (${ear.minAngle} -> ${q.minAngle})`);
    assert.ok(q.minAngle > 360 / n - 1e-9 || n <= 6,
      `n=${n}: and must never be WORSE than the single fan it replaced (${360 / n} -> ${q.minAngle})`);

    // The shape gain that matters to a tracer is the radius ratio, and it is
    // far larger than the angle gain: the needles span the whole disc, the fan
    // triangles span one radius. Order n/2 better, so at least 20x by n=96.
    assert.ok(q.worstRatio < ear.worstRatio / (n >= 96 ? 20 : 2),
      `n=${n}: radius ratio must drop hard (${ear.worstRatio} -> ${q.worstRatio})`);
    assert.ok(q.worstRatio < 5, `n=${n}: no triangle should be elongated (${q.worstRatio})`);
  }
});

test('the centroid fan covers exactly the polygon it was given — no area lost, none added', () => {
  for (const n of NS) {
    const poly = ngon(n, 45, 3, -7);
    const want = shoelace(poly);
    const fan = triangulateConvexFanFromCentroid(poly);
    const q = quality(fan.points, fan.tris);
    assert.ok(Math.abs(q.area - want) <= want * 1e-12,
      `n=${n}: fan covered ${q.area}, polygon encloses ${want}`);
    /* The ringed tiling adds interior vertices, so neither count is `n` any
       more. What must still hold is that the cost stays proportional to the
       loop rather than exploding, and that a loop dense enough to need rings
       actually got them -- a silent fallback to the single fan would restore
       the streaks while every area check above stayed green. */
    assert.ok(fan.tris.length <= 24 * n,
      `n=${n}: ${fan.tris.length} triangles is out of proportion to the loop`);
    assert.ok(fan.points.length >= n + 1, `n=${n}: the centroid at least is added`);
    if (n >= 48) {
      assert.ok(fan.points.length > n + 1,
        `n=${n}: a loop this dense must gain interior ring points, got ${fan.points.length}`);
      assert.ok(fan.tris.length > n,
        `n=${n}: a loop this dense must be ringed, not single-fanned (${fan.tris.length})`);
    }
    // Every triangle must be wound the same way — a flipped one would still sum
    // to the right SIGNED area while covering ground twice.
    for (const [i, j, k] of fan.tris) {
      const a = fan.points[i], b = fan.points[j], c = fan.points[k];
      const cr = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
      assert.ok(cr > 0, `n=${n}: every fan triangle must be CCW, got cross ${cr}`);
    }
  }
});

test('NO BOUNDARY VERTEX CARRIES THE WHOLE FACE — the valence bound, across every convex fixture', () => {
  const fixtures = {
    'disc, 96 points': ngon(96, 45),
    'disc, 240 points': ngon(240, 45),
    'square': [[-30, -30], [30, -30], [30, 30], [-30, 30]],
    'rounded rect, evenly sampled': roundedRect(30, 30, 5, 12, 12),
    'rounded rect, fillet-like sampling': roundedRect(30, 30, 5, 40, 3),
    'very flat ellipse': Array.from({ length: 120 }, (_, i) => {
      const a = (i / 120) * 2 * Math.PI; return [80 * Math.cos(a), 4 * Math.sin(a)];
    }),
  };
  for (const [name, poly] of Object.entries(fixtures)) {
    const ear = triangulatePolygon2D(poly);
    const fan = triangulateConvexFanFromCentroid(poly);
    assert.ok(fan, `${name}: convex, so the fan must apply`);
    const earValence = maxBoundaryValence(poly.length, ear);
    const fanValence = maxBoundaryValence(poly.length, fan.tris);
    // The ear-clipper piles most or all of the face onto one boundary vertex —
    // not always literally every triangle (a rounded rectangle's corners let it
    // clip a few ears elsewhere first), but the great majority, which is what
    // makes one point of the rim the place the streaks converge.
    assert.ok(earValence >= (poly.length - 2) * 0.5,
      `${name}: the ear-clip is expected to fan from one apex (valence ${earValence} of ${poly.length - 2} triangles)`);
    /* A quad strip gives each boundary vertex three triangles -- its own two
       plus its neighbour's -- so the bound is small and constant rather than 2.
       ⚠ AND THE BOUNDARY ALONE NO LONGER SETTLES IT. A single fan also puts
       only two on each RIM vertex; what it does is pile all n onto the
       CENTROID, so a regression back to it would sail past a rim-only check.
       The claim is that no vertex anywhere carries the face. */
    assert.ok(fanValence <= 4, `${name}: max boundary valence must stay small, got ${fanValence}`);
    const counts = new Map();
    for (const t of fan.tris) for (const v of t) counts.set(v, (counts.get(v) || 0) + 1);
    const worstValence = Math.max(...counts.values());
    assert.ok(worstValence <= Math.max(12, poly.length / 4),
      `${name}: no vertex may carry the face — worst valence ${worstValence} of ${fan.tris.length} triangles`);

    const want = shoelace(poly);
    const q = quality(fan.points, fan.tris);
    assert.ok(Math.abs(q.area - want) <= want * 1e-12,
      `${name}: fan covered ${q.area}, polygon encloses ${want}`);
    // And the shape floor never moves the wrong way. It cannot IMPROVE on a
    // quadrilateral, where the ear-clip's two triangles are already as good as
    // the fan's four — the fan's gain is over a fan of many needles.
    const e = quality(poly, ear);
    assert.ok(q.minAngle >= e.minAngle - 1e-9, `${name}: min angle ${e.minAngle} -> ${q.minAngle}`);
    assert.ok(q.worstRatio <= e.worstRatio + 1e-9, `${name}: radius ratio ${e.worstRatio} -> ${q.worstRatio}`);
    if (poly.length > 5) {
      assert.ok(q.minAngle > e.minAngle, `${name}: min angle must improve (${e.minAngle} -> ${q.minAngle})`);
      assert.ok(q.worstRatio < e.worstRatio, `${name}: radius ratio must improve (${e.worstRatio} -> ${q.worstRatio})`);
    }

    /* ⭐ THE FLOOR IS THE LOOP'S OWN, AND THE FAN REACHES IT. An absolute
       minimum-angle number is not a property of the triangulator: the fan's
       triangle over one boundary edge is isoceles with its apex at the centroid,
       so its smallest angle IS the angle that edge subtends from there, and no
       triangulation using only the loop's vertices plus one interior point can
       beat that. Beating it needs Steiner points — Delaunay refinement — which
       is a separate piece of work. So the checkable claim is that the fan
       ATTAINS the loop's own bound, on any convex loop however unevenly it is
       sampled. */
    let bound = Infinity;
    const cx = fan.points[fan.points.length - 1][0], cy = fan.points[fan.points.length - 1][1];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      const au = a[0] - cx, av = a[1] - cy, bu = b[0] - cx, bv = b[1] - cy;
      const la = Math.hypot(au, av), lb = Math.hypot(bu, bv);
      if (la === 0 || lb === 0) continue;
      const ang = Math.acos(Math.max(-1, Math.min(1, (au * bu + av * bv) / (la * lb)))) * 180 / Math.PI;
      if (ang < bound) bound = ang;
    }
    assert.ok(q.minAngle >= Math.min(bound, 45) * 0.98,
      `${name}: the fan must attain the loop's own bound — got ${q.minAngle}, bound ${bound}`);
  }
});

test('a fillet-sampled rounded rectangle — the filleted box top — stops emitting degenerate triangles', () => {
  // Measured on the app's own render mesh for a 60mm cube filleted at r=5: the
  // ear-clip fan gave a worst minimum angle of 0 degrees and a worst radius
  // ratio of 9.4e13 (genuinely degenerate triangles), the centroid fan 1.17
  // degrees and 34.8. The loop below reproduces that shape's character — dense
  // corner arcs, sparse straights — so the floor cannot silently return.
  const poly = roundedRect(30, 30, 5, 40, 3);
  const ear = quality(poly, triangulatePolygon2D(poly));
  const fan = triangulateConvexFanFromCentroid(poly);
  const q = quality(fan.points, fan.tris);
  // The ear-clip fan's worst triangle is genuinely degenerate — zero area to
  // float precision, which is what an unbounded radius ratio means.
  assert.ok(ear.worstRatio > 1e6, `the ear-clip fixture must actually be degenerate, got ${ear.worstRatio}`);
  assert.ok(q.worstRatio < 200, `worst radius ratio ${q.worstRatio} must become bounded`);
  assert.ok(q.worstRatio < ear.worstRatio / 1e4,
    `radius ratio must improve by orders of magnitude (${ear.worstRatio} -> ${q.worstRatio})`);
  assert.ok(q.minAngle > ear.minAngle, `min angle ${ear.minAngle} -> ${q.minAngle}`);
  assert.ok(maxBoundaryValence(poly.length, fan.tris) <= 4, 'no boundary vertex carries the face');
  const want = shoelace(poly);
  assert.ok(Math.abs(q.area - want) <= want * 1e-12, `area ${q.area} vs ${want}`);
});

test('a non-convex loop is REFUSED by the fan and still triangulates correctly under the ear-clipper', () => {
  // An L, and a five-pointed star — the star's centroid is inside the polygon
  // but not visible from every edge, which is the case a bare "is the centroid
  // inside?" test would wave through.
  const ell = [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]];
  const star = [];
  for (let i = 0; i < 10; i++) {
    const a = (i / 10) * 2 * Math.PI;
    const r = i % 2 === 0 ? 10 : 4;
    star.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  for (const [name, poly] of [['L', ell], ['star', star]]) {
    assert.equal(triangulateConvexFanFromCentroid(poly), null,
      `${name}: a non-convex loop must be refused, not fanned`);
    const tris = triangulatePolygon2D(poly);
    const q = quality(poly, tris);
    const want = shoelace(poly);
    assert.ok(Math.abs(q.area - want) <= want * 1e-9,
      `${name}: ear-clip covered ${q.area}, polygon encloses ${want}`);
    assert.equal(tris.length, poly.length - 2, `${name}: n-2 triangles`);
  }
});

test('degenerate input is refused rather than producing NaN', () => {
  const cases = {
    'three collinear points': [[0, 0], [1, 0], [2, 0]],
    'six collinear points': [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
    'all points identical': [[2, 2], [2, 2], [2, 2], [2, 2]],
    'a zero-area slit': [[0, 0], [5, 0], [10, 0], [5, 0]],
    'empty': [],
    'a single point': [[1, 1]],
    'a bare triangle': [[0, 0], [1, 0], [0, 1]],
    'a NaN coordinate': [[0, 0], [1, 0], [NaN, 1], [0, 1]],
    'an infinite coordinate': [[0, 0], [1, 0], [Infinity, 1], [0, 1]],
  };
  for (const [name, poly] of Object.entries(cases)) {
    const fan = triangulateConvexFanFromCentroid(poly);
    assert.equal(fan, null, `${name}: must be refused`);
  }
  // And a loop that is convex but microscopically thin still either refuses or
  // produces finite numbers — never a NaN vertex.
  const sliver = [[0, 0], [10, 0], [10, 1e-11], [0, 1e-11]];
  const fan = triangulateConvexFanFromCentroid(sliver);
  if (fan) for (const p of fan.points) assert.ok(Number.isFinite(p[0]) && Number.isFinite(p[1]), 'no NaN vertex');
});

test('a real trimmed planar face comes out of tessellateTrimmedSurface with the improved shape floor and the same area', () => {
  // extrude() of a degree-1 line is an exactly bilinear (affine) 10x10 plane,
  // so a UV area maps to a world area with a constant Jacobian of 100 and the
  // area comparison below is exact rather than convergent.
  const srf = extrude(makeLine([0, 0, 0], [10, 0, 0]), [0, 1, 0], 10);
  const n = 96;
  const loop = ngon(n, 0.4, 0.5, 0.5); // a circular cap, well inside the domain
  const tris = tessellateTrimmedSurface(srf, loop, 8, 8);
  const area = tessellationArea(tris);
  const want = shoelace(loop) * 100;
  assert.ok(Math.abs(area - want) <= want * 1e-9,
    `traced area ${area} should equal the trim loop's own ${want}`);
  // Ringed, so no longer one per edge — but still proportional to the loop, and
  // strictly more than a single fan would have emitted.
  assert.ok(tris.length > n && tris.length <= 24 * n,
    `the ringed cap should be proportional to its loop, got ${tris.length} for ${n} points`);

  // The mesh's own worst angle, measured on the 3D positions the renderer gets.
  let minAngle = Infinity;
  for (const t of tris) {
    const [a, b, c] = t.map((v) => v.position);
    const ab = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const bc = Math.hypot(c[0] - b[0], c[1] - b[1], c[2] - b[2]);
    const ca = Math.hypot(a[0] - c[0], a[1] - c[1], a[2] - c[2]);
    const s = [ab, bc, ca].sort((x, y) => x - y);
    const cosMin = (s[1] * s[1] + s[2] * s[2] - s[0] * s[0]) / (2 * s[1] * s[2]);
    minAngle = Math.min(minAngle, Math.acos(Math.max(-1, Math.min(1, cosMin))) * 180 / Math.PI);
  }
  assert.ok(minAngle > 180 / n * 1.9,
    `the rendered cap's worst angle ${minAngle} should be past the single-apex fan's ${180 / n}`);

  // No vertex may be shared by every triangle on the BOUNDARY of the face —
  // that shared rim apex is the defect's own signature. The centroid is shared
  // by all n, but it is interior, so it carries the face's own normal and
  // nothing else's.
  const uses = new Map();
  for (const t of tris) {
    for (const v of t) {
      const key = `${Math.round(v.position[0] * 1e6)},${Math.round(v.position[1] * 1e6)}`;
      uses.set(key, (uses.get(key) || 0) + 1);
    }
  }
  /* The single fan left exactly one vertex carrying half the face — its
     centroid. The rings leave NONE: the apex is now a six-to-sixteen triangle
     fan on an inner ring, so nothing is shared that heavily. Asserting zero is
     the stronger claim and it is the one the streaks turned on. */
  let shared = 0;
  for (const [, c] of uses) if (c > n / 2) shared++;
  assert.equal(shared, 0, `no vertex should carry half the face any more, found ${shared}`);
});
