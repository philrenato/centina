// KNOT SURGERY IN ONE DIRECTION OF A SURFACE, and the harmonisation Match Edge
// needs before a control-row edit means anything.
//
// A control point exists at a specific PARAMETER. Matching one surface's
// boundary row to another's is only meaningful if both edges agree on where
// those parameters fall — otherwise row k of one and row k of the other are
// simply different places, and an edit that looks tidy in the net produces a
// surface nobody asked for. So before any match: bring both to a common domain,
// a common degree, and the UNION of their knots along the seam.
//
// EVERY OPERATION HERE IS SHAPE-PRESERVING BY CONSTRUCTION. Knot insertion
// (Boehm) and degree elevation both add control points without moving the
// surface, so harmonising is free in geometry and only costs description size.
// That is what the tests assert — not that the numbers look reasonable, but
// that the surface evaluates identically afterwards.
//
// A surface direction is just a family of curves: the v direction is the rows
// of the control net, the u direction its columns. So this file is mostly the
// bookkeeping of turning a net into curves and back, and the real work stays in
// knots.mjs where it is already tested.
import { insertKnot, degreeElevateCurve, rescaleCurveDomain } from './knots.mjs';

const KNOT_TOL = 1e-9;

// The rows (dir 'v') or columns (dir 'u') of a surface, as curves.
function asCurves(srf, dir) {
  if (dir === 'v') {
    return srf.ctrlNet.map((row) => ({ degree: srf.degV, knots: srf.knotsV.slice(), ctrlPts: row.map((p) => p.slice()) }));
  }
  const nv = srf.ctrlNet[0].length;
  const out = [];
  for (let j = 0; j < nv; j++) {
    out.push({ degree: srf.degU, knots: srf.knotsU.slice(), ctrlPts: srf.ctrlNet.map((r) => r[j].slice()) });
  }
  return out;
}

function fromCurves(srf, dir, curves) {
  if (dir === 'v') {
    return { ...srf, degV: curves[0].degree, knotsV: curves[0].knots.slice(), ctrlNet: curves.map((c) => c.ctrlPts.map((p) => p.slice())) };
  }
  const nu = curves[0].ctrlPts.length;
  const net = [];
  for (let i = 0; i < nu; i++) net.push(curves.map((c) => c.ctrlPts[i].slice()));
  return { ...srf, degU: curves[0].degree, knotsU: curves[0].knots.slice(), ctrlNet: net };
}

export function surfaceInsertKnot(srf, dir, u, r = 1) {
  return fromCurves(srf, dir, asCurves(srf, dir).map((c) => insertKnot(c, u, r)));
}

export function surfaceElevateDegree(srf, dir, targetDegree) {
  return fromCurves(srf, dir, asCurves(srf, dir).map((c) => degreeElevateCurve(c, targetDegree)));
}

export function surfaceRescaleDomain(srf, dir, newMin, newMax) {
  return fromCurves(srf, dir, asCurves(srf, dir).map((c) => rescaleCurveDomain(c, newMin, newMax)));
}

export function degreeIn(srf, dir) { return dir === 'v' ? srf.degV : srf.degU; }
export function knotsIn(srf, dir) { return dir === 'v' ? srf.knotsV : srf.knotsU; }
export function countIn(srf, dir) { return dir === 'v' ? srf.ctrlNet[0].length : srf.ctrlNet.length; }

// The interior knots and how many times each appears — the clamped ends are
// structural and never part of a union.
export function interiorKnotMultiplicities(knots, degree) {
  const out = [];
  for (let i = degree + 1; i < knots.length - degree - 1; i++) {
    const k = knots[i];
    const last = out[out.length - 1];
    if (last && Math.abs(last.value - k) <= KNOT_TOL) last.count++;
    else out.push({ value: k, count: 1 });
  }
  return out;
}

// Bring two surfaces to a common description ALONG ONE DIRECTION each: same
// domain, same degree, same knots — and therefore the same control count, which
// is the thing Match Edge actually needs.
//
// ⚠ THE DOMAINS ARE RECONCILED FIRST, and that is not a formality. A revolve's
// sweep direction runs 0..4 while a hand-built patch runs 0..1; taking the
// union of two knot vectors that do not live on the same interval produces a
// vector that is not a superset of either, and every insertion after it lands
// somewhere arbitrary. Rescaling is an affine reparametrization and moves no
// point on the surface.
export function harmonizeDirections(a, dirA, b, dirB, opts = {}) {
  const degA = degreeIn(a, dirA), degB = degreeIn(b, dirB);
  const kA = knotsIn(a, dirA), kB = knotsIn(b, dirB);
  const domain = [kA[0], kA[kA.length - 1]];
  if (!(domain[1] > domain[0])) return { ok: false, reason: 'the seam direction has an empty parameter domain', a: null, b: null };

  let A = a;
  let B = surfaceRescaleDomain(b, dirB, domain[0], domain[1]);

  const deg = Math.max(degA, degB);
  if (degA < deg) A = surfaceElevateDegree(A, dirA, deg);
  if (degB < deg) B = surfaceElevateDegree(B, dirB, deg);

  // Union of interior knots, by multiplicity. Elevation rewrites them, so this
  // has to be read AFTER it rather than before.
  const mA = interiorKnotMultiplicities(knotsIn(A, dirA), deg);
  const mB = interiorKnotMultiplicities(knotsIn(B, dirB), deg);
  const values = [];
  for (const { value } of [...mA, ...mB]) {
    if (!values.some((v) => Math.abs(v - value) <= KNOT_TOL)) values.push(value);
  }
  values.sort((x, y) => x - y);
  const countAt = (list, value) => (list.find((e) => Math.abs(e.value - value) <= KNOT_TOL) || { count: 0 }).count;
  for (const value of values) {
    const want = Math.max(countAt(mA, value), countAt(mB, value));
    const needA = want - countAt(mA, value);
    const needB = want - countAt(mB, value);
    if (needA > 0) A = surfaceInsertKnot(A, dirA, value, needA);
    if (needB > 0) B = surfaceInsertKnot(B, dirB, value, needB);
  }

  if (countIn(A, dirA) !== countIn(B, dirB)) {
    // THE UNION HAS DONE WHAT IT CAN AND THE TWO STILL DISAGREE. Refusing here
    // is honest but it is also the end of the road for the student, who is
    // looking at two edges that plainly ought to meet and being told about
    // control counts.
    //
    // ADDING CONTROL POINTS IS FREE, in the only sense that matters: knot
    // insertion is EXACT. It rewrites a surface's description and moves no
    // point of the surface at all, so bringing the coarser side up to the
    // finer one costs nothing but a denser net — which a match was going to
    // rewrite anyway. That is what `force` does, and it is offered rather
    // than assumed, because a denser net IS a real change to what the student
    // will be dragging afterwards.
    if (!opts.force) {
      return {
        ok: false,
        reason: `harmonisation did not converge (${countIn(A, dirA)} vs ${countIn(B, dirB)} control points along the seam)`,
        a: null, b: null, forceable: true,
        counts: [countIn(A, dirA), countIn(B, dirB)],
      };
    }
    if (countIn(A, dirA) < countIn(B, dirB)) A = refineToCount(A, dirA, countIn(B, dirB));
    else B = refineToCount(B, dirB, countIn(A, dirA));
    if (countIn(A, dirA) !== countIn(B, dirB)) {
      return { ok: false, reason: `harmonisation could not be forced (${countIn(A, dirA)} vs ${countIn(B, dirB)} control points along the seam)`, a: null, b: null };
    }
  }
  return { ok: true, a: A, b: B, degree: deg, knots: knotsIn(A, dirA).slice(), forced: !!opts.force };
}

// Insert knots into one direction until it carries `target` control points,
// always splitting the WIDEST remaining span. Widest-first keeps the resulting
// net as even as it can be, and an even net is what a student expects to drag;
// splitting arbitrarily would pile new points into one corner of the seam.
//
// Every insertion is exact — this changes the surface's description and not the
// surface — so the only thing at stake is where the new points sit, not where
// the geometry does.
export function refineToCount(srf, dir, target) {
  let out = srf;
  let guard = 0;
  while (countIn(out, dir) < target && guard++ < 1000) {
    const knots = knotsIn(out, dir);
    const deg = degreeIn(out, dir);
    let bestAt = null, bestWidth = 0;
    for (let i = deg; i < knots.length - deg - 1; i++) {
      const width = knots[i + 1] - knots[i];
      if (width > bestWidth + KNOT_TOL) { bestWidth = width; bestAt = (knots[i] + knots[i + 1]) / 2; }
    }
    if (bestAt === null || !(bestWidth > KNOT_TOL)) break; // nothing left to split
    out = surfaceInsertKnot(out, dir, bestAt, 1);
  }
  return out;
}

// Which direction runs ALONG a given edge — the one that has to be harmonised.
// An edge at constant u is traversed by varying v, and vice versa; getting this
// backwards harmonises the direction that crosses the seam instead of the one
// that follows it, which changes the wrong control count and still looks busy.
export function seamDirectionFor(edge) {
  if (edge === 'u0' || edge === 'u1') return 'v';
  if (edge === 'v0' || edge === 'v1') return 'u';
  throw new Error(`seamDirectionFor: unknown edge ${edge}`);
}
