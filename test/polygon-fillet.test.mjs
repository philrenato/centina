import test from 'node:test';
import assert from 'node:assert/strict';
import { curvePoint } from '../kernel/curve.mjs';
import { filletCornerArc, filletPolygon } from '../kernel/primitives.mjs';

const Z = [0, 0, 1]; // plane normal for every fixture below — all built flat in XY

function square(half) {
  return [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]];
}
function regularPolygon(n, radius) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const theta = (2 * Math.PI * i) / n;
    pts.push([radius * Math.cos(theta), radius * Math.sin(theta), 0]);
  }
  return pts;
}
function arcCrv(c) {
  return { degree: 2, knots: [0, 0, 0, 1, 1, 1], ctrlPts: [[...c.p0, 1], [...c.apex, c.weight], [...c.p2, 1]] };
}
// Independent cross-check: the fillet construction never computes a
// geometric CENTER at all (apex+trim+weight only, the tangent-line-
// intersection formulation) — this derives a circumcenter from 3 sampled
// arc points via the plain, generic, sign-agnostic circumcenter formula
// (works for ANY 3 non-collinear points, convex or reflex, no assumption
// about which "side" the fillet bulges toward) and then confirms OTHER,
// independently-sampled points on the same arc sit at exactly that
// circumradius from it — a real proof the whole arc is one genuine circle,
// not just that 3 arbitrary points happen to fit some circle.
function circumcenter(a, b, c) {
  const ax = a[0], ay = a[1], bx = b[0], by = b[1], cx = c[0], cy = c[1];
  const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  const ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
  const uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;
  return [ux, uy, 0];
}
function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function dist(a, b) { return Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]); }
function norm(a) { const l = Math.hypot(a[0],a[1],a[2]); return [a[0]/l,a[1]/l,a[2]/l]; }

test('filletCornerArc on a square (90deg convex corner): trim=radius, weight=cos(45deg), tangent to both edges', () => {
  const pts = square(10);
  const c = filletCornerArc(pts[1], pts[0], pts[2], 3, Z);
  assert.equal(c.ok, true);
  assert.ok(Math.abs(c.trim - 3) < 1e-9, `trim should be exactly the radius for a 90deg corner, got ${c.trim}`);
  assert.ok(Math.abs(c.weight - Math.cos(Math.PI / 4)) < 1e-9);
  // Tangency: apex - p0 is parallel to the incoming edge direction; p2 - apex parallel to outgoing.
  const dIn = norm(sub(pts[1], pts[0]));
  const dOut = norm(sub(pts[2], pts[1]));
  const apexMinusP0 = norm(sub(c.apex, c.p0));
  const p2MinusApex = norm(sub(c.p2, c.apex));
  assert.ok(dist(apexMinusP0, dIn) < 1e-9, 'p0->apex direction matches the incoming edge exactly');
  assert.ok(dist(p2MinusApex, dOut) < 1e-9, 'apex->p2 direction matches the outgoing edge exactly');
});

test('filletCornerArc: the constructed arc is a genuine circle of the requested radius, independently cross-checked via a circumcenter derived from 3 sampled points, then verified against OTHER points', () => {
  const pts = square(10);
  const radius = 2.5;
  const c = filletCornerArc(pts[1], pts[0], pts[2], radius, Z);
  const crv = arcCrv(c);
  const center = circumcenter(curvePoint(crv, 0), curvePoint(crv, 0.5), curvePoint(crv, 1));
  const r0 = dist(curvePoint(crv, 0), center);
  assert.ok(Math.abs(r0 - radius) < 1e-9, `circumradius from 3 sampled points should equal the requested radius, got ${r0}`);
  for (const u of [0.1, 0.25, 0.4, 0.6, 0.75, 0.9]) {
    const d = dist(curvePoint(crv, u), center);
    assert.ok(Math.abs(d - radius) < 1e-9, `u=${u}: distance from the independently-derived circumcenter = ${d}, expected ${radius}`);
  }
});

test('filletCornerArc handles a REFLEX corner (turn angle < 0, e.g. a star polygon\'s inner vertex) with the same formula, no sign-flip special case needed by the caller', () => {
  // A simple reflex vertex: path goes A(0,0) -> V(1,0.3) -> B(2,0), a shallow
  // "dent" — the turn at V is a right turn (reflex from the interior of the
  // upper region), i.e. phi < 0.
  const A = [0, 0, 0], V = [1, 0.3, 0], B = [2, 0, 0];
  const c = filletCornerArc(V, A, B, 0.05, Z);
  assert.equal(c.ok, true);
  assert.ok(c.turnAngle < 0, 'this corner is a right/reflex turn, turnAngle should be negative');
  const crv = arcCrv(c);
  const center = circumcenter(curvePoint(crv, 0), curvePoint(crv, 0.5), curvePoint(crv, 1));
  for (const u of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
    const d = dist(curvePoint(crv, u), center);
    assert.ok(Math.abs(d - 0.05) < 1e-9, `reflex corner arc point at u=${u} should stay exactly on the circle, d=${d}`);
  }
});

test('filletCornerArc refuses a straight (non-corner) point honestly', () => {
  const c = filletCornerArc([1, 0, 0], [0, 0, 0], [2, 0, 0], 1, Z);
  assert.equal(c.ok, false);
});

test('filletPolygon on a square: every corner becomes an exact quarter-circle arc, straight runs remain exactly straight and tangent', () => {
  const half = 10;
  const pts = square(half);
  const radius = 3;
  const res = filletPolygon(pts, radius, Z);
  assert.equal(res.ok, true);
  assert.equal(res.segments.length, 8, '4 corners -> 4 arcs + 4 lines');
  const arcs = res.segments.filter((s) => s.type === 'arc');
  const lines = res.segments.filter((s) => s.type === 'line');
  assert.equal(arcs.length, 4);
  assert.equal(lines.length, 4);
  for (const line of lines) {
    // Every remaining straight run on a square is axis-aligned (X or Y constant).
    const dx = Math.abs(line.b[0] - line.a[0]), dy = Math.abs(line.b[1] - line.a[1]);
    assert.ok(dx < 1e-9 || dy < 1e-9, 'a square\'s trimmed edges stay axis-aligned');
    const len = dist(line.a, line.b);
    assert.ok(len > 0 && len < 2 * half, `trimmed edge length ${len} should be shorter than the full ${2 * half}mm edge`);
  }
  for (const arc of arcs) {
    assert.ok(Math.abs(arc.weight - Math.cos(Math.PI / 4)) < 1e-9, 'every square corner is the identical 90deg fillet');
  }
});

test('filletPolygon refuses honestly when the radius is too large for the polygon\'s own edge lengths (adjacent fillets would overlap), and reports a usable maxSafeRadius', () => {
  const half = 10; // 20mm square edges
  const pts = square(half);
  const tooLarge = filletPolygon(pts, 11, Z); // trim would be 11mm > 10mm half-edge -> overlaps well past the midpoint
  assert.equal(tooLarge.ok, false);
  assert.ok(tooLarge.maxSafeRadius > 0 && tooLarge.maxSafeRadius < 11, 'a real, smaller safe radius should be reported');
  // The reported maxSafeRadius should ACTUALLY be safe when re-tried.
  const retried = filletPolygon(pts, tooLarge.maxSafeRadius, Z);
  assert.equal(retried.ok, true, 'the reported maxSafeRadius should itself succeed when retried');
  // And just barely above it should still fail.
  const justOver = filletPolygon(pts, tooLarge.maxSafeRadius * 1.02, Z);
  assert.equal(justOver.ok, false);
});

test('filletPolygon on a regular hexagon: a modest radius succeeds and every arc is the same exterior-angle sweep (2*PI/6)', () => {
  const pts = regularPolygon(6, 20);
  const res = filletPolygon(pts, 2, Z);
  assert.equal(res.ok, true);
  const arcs = res.segments.filter((s) => s.type === 'arc');
  assert.equal(arcs.length, 6);
  const expectedWeight = Math.cos(Math.PI / 6); // half of 2*PI/6
  for (const arc of arcs) assert.ok(Math.abs(arc.weight - expectedWeight) < 1e-9);
});

test('filletPolygon on a low-side-count triangle: even a small radius can be geometrically infeasible if requested too large, and a genuinely small one still succeeds', () => {
  const pts = regularPolygon(3, 10); // a small equilateral triangle - short edges, sharp 60deg corners
  const tiny = filletPolygon(pts, 0.5, Z);
  assert.equal(tiny.ok, true, 'a small enough radius on a triangle should still succeed');
  const huge = filletPolygon(pts, 20, Z);
  assert.equal(huge.ok, false, 'a radius much larger than the triangle itself must refuse, not produce garbage geometry');
});
