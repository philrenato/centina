// ANALYTIC CURVE KINDS ACROSS A REAL .3dm ROUND TRIP.
//
// A Circle, an Arc, a Line and a Polyline are four different objects in this
// app and were one thing in the file: every curve went out as a plain
// NurbsCurve, so a circle came back as a freeform SketchCurve whose center and
// radius no longer existed anywhere. The geometry was never wrong — the KIND
// was gone, in our file and in Rhino's reading of it too.
//
// ⚠ EVERY FIXTURE HERE IS DELIBERATELY AWKWARD. A circle at the origin on the
// world XY plane is symmetric under exactly the transforms a plane/basis bug
// gets wrong, so the circle and the arc sit at an oblique center on a tilted
// orthonormal frame, and the polylines are irregular and non-planar. A
// straight-edged, axis-aligned fixture would pass with the plane discarded.
//
// ⚠ AN ARC IS A CIRCLE WITH A SWEEP. The app stores both as one `Circle`
// object and lets circleStart/circleEnd be the only difference, so the two
// directions of that mistake — a partial sweep written as a closed circle, a
// full turn written as an arc — are asserted explicitly and from the FILE, not
// only from the payload.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import rhino3dmFactory from 'rhino3dm';
import { makeArc, makeCircle } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { curvePoint } from '../kernel/curve.mjs';
import { exportDocument, importDocument } from '../io3dm.mjs';

const rhino = await rhino3dmFactory();

// An oblique center and a tilted orthonormal in-plane frame: |X| = |Y| = 1,
// X·Y = 0, and neither lies in a world plane.
const CENTER = [3.7, -2.1, 5.9];
const XAXIS = [0.6, 0.8, 0];
const YAXIS = [-0.48, 0.36, 0.8];
const RADIUS = 2.75;
const ARC_START = 0.4;
const ARC_SWEEP = 1.8;

const LINE_A = [1.3, -4.2, 0.75];
const LINE_B = [6.1, 2.4, -3.05];

// Irregular spacing, no two segments parallel, and genuinely non-planar.
const POLY_PTS = [
  [0.3, -1.2, 0.7],
  [2.9, 0.4, -1.1],
  [5.2, -3.3, 2.8],
  [1.1, 4.4, 3.9],
  [-2.6, 1.7, -0.45],
];

function polylineCrv(points, closed) {
  const ctrlPts = points.map((p) => [...p, 1]);
  if (closed) ctrlPts.push(ctrlPts[0]);
  const m = ctrlPts.length;
  const knots = [0, 0];
  for (let i = 1; i <= m - 2; i++) knots.push(i);
  knots.push(m - 1, m - 1);
  return { degree: 1, knots, ctrlPts };
}

function roundTrip(objects) {
  const payload = {
    tolerance: 0.001,
    layers: [{ id: 0, name: 'Default', color: { r: 140, g: 141, b: 144 }, parentId: null }],
    objects,
  };
  const { bytes, skipped } = exportDocument(rhino, payload);
  assert.deepEqual(skipped, [], 'nothing may be skipped on export');
  const result = importDocument(rhino, bytes);
  assert.deepEqual(result.skipped, [], 'nothing may be skipped on import');
  return { bytes, result };
}

// What a SECOND reader sees in the bytes — the claim "the kind is in the file"
// is about OpenNURBS's own object class, not about our payload.
function fileGeometryClasses(bytes) {
  const doc = rhino.File3dm.fromByteArray(bytes);
  const out = new Map();
  for (let i = 0; i < doc.objects().count; i++) {
    const o = doc.objects().get(i);
    out.set(o.attributes().name, o.geometry());
  }
  return out;
}

function near(a, b, tol, what) {
  assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (tolerance ${tol})`);
}

function nearPoint(a, b, tol, what) {
  const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  assert.ok(d <= tol, `${what}: [${a}] vs [${b}], off by ${d} (tolerance ${tol})`);
}

test('a circle round-trips as a circle: the analytic kind survives, and the NURBS form is unchanged', () => {
  const circle = makeCircle(CENTER, XAXIS, YAXIS, RADIUS);
  const { bytes, result } = roundTrip([{ kind: 'curve', layerId: 0, name: 'Circle01', ...circle }]);
  const crv = result.objects[0];

  // Nothing is taken away: the plain NURBS fields still arrive exactly as sent.
  assert.equal(crv.degree, circle.degree);
  assert.deepEqual(crv.knots.map((k) => Math.round(k * 1e9) / 1e9), circle.knots);
  assert.equal(crv.ctrlPts.length, circle.ctrlPts.length);

  assert.equal(crv.curveKind, 'circle');
  nearPoint(crv.circleCenter, CENTER, 1e-9, 'circle center');
  near(crv.circleRadius, RADIUS, 1e-9, 'circle radius');
  nearPoint(crv.circleAxes.xAxis, XAXIS, 1e-9, 'circle xAxis');
  nearPoint(crv.circleAxes.yAxis, YAXIS, 1e-9, 'circle yAxis');
  near(crv.circleEnd - crv.circleStart, Math.PI * 2, 1e-9, 'circle sweep is a full turn');

  // And it is a circle IN THE FILE, not only in our reading of it.
  const geo = fileGeometryClasses(bytes).get('Circle01');
  assert.equal(geo.constructor.name, 'ArcCurve');
  assert.equal(geo.isCompleteCircle, true, 'a full turn must be written closed');
});

test('an arc round-trips as an arc, never as a closed circle', () => {
  const arc = makeArc(CENTER, XAXIS, YAXIS, RADIUS, ARC_START, ARC_SWEEP);
  const { bytes, result } = roundTrip([{ kind: 'curve', layerId: 0, name: 'Arc01', ...arc }]);
  const crv = result.objects[0];

  assert.equal(crv.degree, arc.degree);
  assert.deepEqual(crv.knots.map((k) => Math.round(k * 1e9) / 1e9), arc.knots);
  assert.equal(crv.ctrlPts.length, arc.ctrlPts.length);

  assert.equal(crv.curveKind, 'arc');
  nearPoint(crv.circleCenter, CENTER, 1e-9, 'arc center');
  near(crv.circleRadius, RADIUS, 1e-9, 'arc radius');
  near(crv.circleEnd - crv.circleStart, ARC_SWEEP, 1e-9, 'arc sweep');
  assert.ok(Math.abs((crv.circleEnd - crv.circleStart) - Math.PI * 2) > 1e-6,
    'an arc must NOT come back reporting a full turn');

  /* The recovered frame is OpenNURBS's own canonical one (see the module: the
     start angle is absorbed into a rotated in-plane basis), so the oracle is
     the GEOMETRY, not the angle numbers. Rebuild the arc from what came back
     and it must be the arc that was sent, end for end. */
  const rebuilt = makeArc(crv.circleCenter, crv.circleAxes.xAxis, crv.circleAxes.yAxis,
    crv.circleRadius, crv.circleStart, crv.circleEnd - crv.circleStart);
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const uA = arc.knots[arc.degree] + t * (arc.knots[arc.knots.length - 1 - arc.degree] - arc.knots[arc.degree]);
    const uB = rebuilt.knots[rebuilt.degree] + t * (rebuilt.knots[rebuilt.knots.length - 1 - rebuilt.degree] - rebuilt.knots[rebuilt.degree]);
    nearPoint(curvePoint(rebuilt, uB), curvePoint(arc, uA), 1e-9, `arc point at ${t}`);
  }

  const geo = fileGeometryClasses(bytes).get('Arc01');
  assert.equal(geo.constructor.name, 'ArcCurve');
  assert.equal(geo.isCompleteCircle, false, 'a partial sweep must NOT be written as a closed circle');
  assert.equal(geo.isClosed, false);
});

test('a line round-trips as a line', () => {
  const line = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[...LINE_A, 1], [...LINE_B, 1]] };
  const { bytes, result } = roundTrip([{ kind: 'curve', layerId: 0, name: 'Line01', ...line }]);
  const crv = result.objects[0];

  assert.equal(crv.degree, 1);
  assert.deepEqual(crv.knots.map((k) => Math.round(k * 1e9) / 1e9), [0, 0, 1, 1]);

  assert.equal(crv.curveKind, 'line');
  nearPoint(crv.lineStart, LINE_A, 1e-9, 'line start');
  nearPoint(crv.lineEnd, LINE_B, 1e-9, 'line end');

  const geo = fileGeometryClasses(bytes).get('Line01');
  assert.equal(geo.constructor.name, 'LineCurve');
});

test('an open polyline round-trips as a polyline, with every irregular point in place', () => {
  const poly = polylineCrv(POLY_PTS, false);
  const { bytes, result } = roundTrip([{ kind: 'curve', layerId: 0, name: 'Poly01', ...poly }]);
  const crv = result.objects[0];

  assert.equal(crv.degree, 1);
  assert.deepEqual(crv.knots.map((k) => Math.round(k * 1e9) / 1e9), poly.knots);

  assert.equal(crv.curveKind, 'polyline');
  assert.equal(crv.polylineClosed, false);
  assert.equal(crv.polylinePoints.length, POLY_PTS.length);
  POLY_PTS.forEach((p, i) => nearPoint(crv.polylinePoints[i], p, 1e-9, `polyline point ${i}`));

  const geo = fileGeometryClasses(bytes).get('Poly01');
  assert.equal(geo.constructor.name, 'PolylineCurve');
  assert.equal(geo.isClosed, false);
});

test('a closed polyline round-trips closed, and its points do not grow a duplicate', () => {
  const poly = polylineCrv(POLY_PTS, true);
  const { bytes, result } = roundTrip([{ kind: 'curve', layerId: 0, name: 'PolyC01', ...poly }]);
  const crv = result.objects[0];

  assert.equal(crv.curveKind, 'polyline');
  assert.equal(crv.polylineClosed, true);
  /* The closing repeat is a fact about the CONTROL POLYGON, not about the
     point list — the app stores a closed Polyline's points once and appends
     the repeat when it builds the curve. Handing back the repeat would grow
     one duplicate vertex per save/open cycle. */
  assert.equal(crv.polylinePoints.length, POLY_PTS.length);
  POLY_PTS.forEach((p, i) => nearPoint(crv.polylinePoints[i], p, 1e-9, `closed polyline point ${i}`));

  const geo = fileGeometryClasses(bytes).get('PolyC01');
  assert.equal(geo.constructor.name, 'PolylineCurve');
  assert.equal(geo.isClosed, true);
});

test('a freeform curve is reported as having NO analytic kind, rather than the field simply being absent', () => {
  const free = globalCurveInterp([[0, 0, 0], [1.3, 2.7, -0.4], [3.9, 1.1, 2.2], [5.5, -2.4, 0.8], [7.1, 3.3, -1.9]], 3);
  const { bytes, result } = roundTrip([{ kind: 'curve', layerId: 0, name: 'Sketch01', ...free }]);
  const crv = result.objects[0];

  /* The field is always present. A caller that switches on it must be able to
     tell "this is a plain NURBS curve" from "this build does not report kinds
     at all" — an absent field reads as the second and behaves as the first. */
  assert.ok('curveKind' in crv, 'curveKind must be reported for every curve');
  assert.equal(crv.curveKind, null);
  assert.equal(crv.circleCenter, undefined);
  assert.equal(crv.polylinePoints, undefined);
  assert.equal(crv.lineStart, undefined);

  assert.equal(crv.degree, free.degree);
  assert.equal(crv.ctrlPts.length, free.ctrlPts.length);
  free.ctrlPts.forEach((p, i) => nearPoint(crv.ctrlPts[i], p, 1e-9, `freeform control point ${i}`));

  const geo = fileGeometryClasses(bytes).get('Sketch01');
  assert.equal(geo.constructor.name, 'NurbsCurve');
});

test('four kinds in one document keep their four kinds', () => {
  const { bytes, result } = roundTrip([
    { kind: 'curve', layerId: 0, name: 'Circle01', ...makeCircle(CENTER, XAXIS, YAXIS, RADIUS) },
    { kind: 'curve', layerId: 0, name: 'Arc01', ...makeArc(CENTER, XAXIS, YAXIS, RADIUS, ARC_START, ARC_SWEEP) },
    { kind: 'curve', layerId: 0, name: 'Line01', ...{ degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[...LINE_A, 1], [...LINE_B, 1]] } },
    { kind: 'curve', layerId: 0, name: 'Poly01', ...polylineCrv(POLY_PTS, false) },
  ]);
  const byName = new Map(result.objects.map((o) => [o.name, o]));
  assert.deepEqual(
    ['Circle01', 'Arc01', 'Line01', 'Poly01'].map((n) => byName.get(n).curveKind),
    ['circle', 'arc', 'line', 'polyline'],
  );
  const geo = fileGeometryClasses(bytes);
  assert.deepEqual(
    ['Circle01', 'Arc01', 'Line01', 'Poly01'].map((n) => geo.get(n).constructor.name),
    ['ArcCurve', 'ArcCurve', 'LineCurve', 'PolylineCurve'],
  );
});

test('a curve that is only NEARLY a circle is not promoted to one', () => {
  /* The detector's tolerance is nanometres, not the document tolerance. This
     dent is well INSIDE the 0.001 mm the document declares — so a detector
     handed the model tolerance calls it a circle, rewrites it as one, and
     hands back a center and a radius nobody drew. It must stay freeform. */
  const circle = makeCircle(CENTER, XAXIS, YAXIS, RADIUS);
  const dented = { ...circle, ctrlPts: circle.ctrlPts.map((p, i) => (i === 3 ? [p[0] + 2e-4, p[1], p[2], p[3]] : p)) };
  /* Both arms in one test, because "refused" is only meaningful next to
     "accepted": a detector that answered null for everything would satisfy the
     refusal on its own. */
  const { bytes, result } = roundTrip([
    { kind: 'curve', layerId: 0, name: 'Dent01', ...dented },
    { kind: 'curve', layerId: 0, name: 'True01', ...circle },
  ]);
  const byName = new Map(result.objects.map((o) => [o.name, o]));
  assert.equal(byName.get('Dent01').curveKind, null);
  assert.equal(byName.get('True01').curveKind, 'circle');
  const geo = fileGeometryClasses(bytes);
  assert.equal(geo.get('Dent01').constructor.name, 'NurbsCurve');
  assert.equal(geo.get('True01').constructor.name, 'ArcCurve');
  // And the dent is still where it was put — a refusal must not round the curve.
  nearPoint(byName.get('Dent01').ctrlPts[3], dented.ctrlPts[3], 1e-9, 'the dented control point');
});
