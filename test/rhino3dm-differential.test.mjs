// DIFFERENTIAL VALIDATION AGAINST A SECOND, INDEPENDENT NURBS IMPLEMENTATION.
//
// Every other test in this suite was written by the same author as the code
// under test, so a misread basis-function recursion would produce a test that
// expects the wrong answer and passes forever. This file is the only check in
// the repository that can catch that class of error: it evaluates the SAME
// geometry through our kernel and through rhino3dm (openNURBS, McNeel) and
// compares the numbers.
//
// BLACK-BOX ONLY. rhino3dm is used strictly as a behavioural oracle: same
// inputs in, outputs compared. No openNURBS source is read, and none of the
// kernel's own algorithms were derived from it. See the BEHAVIORAL ORACLE —
// NOT A SOURCE distinction: it is an oracle, never a source of code.
//
// WHAT THE BINDING CAN AND CANNOT ORACLE, enumerated directly rather than
// assumed. The vendored wasm binding exposes evaluation on the BASE classes
// Surface and Curve, which NurbsSurface/NurbsCurve inherit:
//
//   surface point at (u,v)      Surface.pointAt        COVERED BELOW
//   surface normal              Surface.normalAt       COVERED BELOW
//   curve point                 Curve.pointAt          COVERED BELOW
//   curve 1st/2nd derivative    Curve.derivativeAt     COVERED BELOW
//   surface 1st/2nd derivative  (no direct method)     covered INDIRECTLY:
//       normalAt is normalize(Su x Sv), so it constrains the DIRECTION of
//       both partials jointly, but not their magnitudes.
//   knot insertion / refinement (absent from the binding)  NOT COVERABLE
//   closest-point / projection  (absent from the binding)  NOT COVERABLE
//
// The last two are genuinely unavailable, not skipped for convenience — the
// evaluation for them lives in openNURBS and is not in the wasm build. Our
// own closestPointOnSurface/closestPointOnCurve therefore still have no
// second implementation to check against, and that gap is real.
//
// CONVENTIONS, each verified empirically before any assertion was written
// rather than assumed from documentation:
//   - normalAt agrees with normalize(su x sv) in SIGN as well as direction
//     (measured: dot = 1.000000000 across a 3x3 sample grid, no flip).
//   - derivativeAt(t, d) returns [C, C', C'' ...] in the same order as
//     rationalCurveDerivs(crv, t, d).
//   - our ctrlPts are EUCLIDEAN + weight; rhino3dm wants PRE-MULTIPLIED
//     homogeneous, and its knot vector drops our first and last entry. Both
//     conversions are handled by io3dm.mjs's own curveToRhino/surfaceToRhino,
//     reused here rather than re-derived, and independently proven correct by
//     test/io3dm.test.mjs.
//
// A DISAGREEMENT IS A FINDING, NOT AUTOMATICALLY OUR BUG. Tolerances here are
// tight enough that anything failing is a real divergence worth reading, not
// float noise: these are two independent evaluations of the same polynomial,
// so agreement should be near machine precision, not merely "close".

import { test } from 'node:test';
import assert from 'node:assert/strict';
import rhino3dmFactory from 'rhino3dm';
import { makeArc, makeCircle, makeEllipsoidProfile, revolve, extrude } from '../kernel/primitives.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { curvePoint, rationalCurveDerivs } from '../kernel/curve.mjs';
import { surfacePoint, surfacePointAndPartials } from '../kernel/surface.mjs';
import { curveToRhino, surfaceToRhino } from '../io3dm.mjs';

const rhino = await rhino3dmFactory();

// Two independent evaluations of the same polynomial should agree to roughly
// machine precision scaled by the geometry's own magnitude. These fixtures sit
// in the tens of millimetres, so 1e-9 leaves six orders of headroom over the
// ~1e-15 actually observed — tight enough that a real algorithmic divergence
// cannot hide under it.
const TOL_POINT = 1e-9;
const TOL_DERIV = 1e-6;   // derivatives amplify roundoff by the span scale
const TOL_NORMAL = 1e-9;  // unit vectors, so this is an absolute angle bound

const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const unit = (v) => {
  const L = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / L, v[1] / L, v[2] / L];
};
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

const curveDomain = (c) => [c.knots[c.degree], c.knots[c.knots.length - 1 - c.degree]];
const surfDomain = (s) => [
  s.knotsU[s.degU], s.knotsU[s.knotsU.length - 1 - s.degU],
  s.knotsV[s.degV], s.knotsV[s.knotsV.length - 1 - s.degV],
];

// Samples strictly INSIDE the domain plus both exact ends. `inset` pulls the
// endpoints in when a fixture has a singularity there (a pole row), where the
// quantity under test is genuinely undefined rather than merely hard.
function paramsAcross(lo, hi, n, inset = 0) {
  const a = lo + (hi - lo) * inset;
  const b = hi - (hi - lo) * inset;
  return Array.from({ length: n + 1 }, (_, i) => a + (b - a) * i / n);
}

// ---------------------------------------------------------------------------
// FIXTURES — deliberately the hard cases, not the easy ones.
// ---------------------------------------------------------------------------

// Rational, non-unit weights, CLOSED (periodic) in the sweep direction.
const cylinder = revolve(
  { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[5, 0, 0, 1], [5, 0, 20, 1]] },
  [0, 0, 0], [0, 0, 1], 0, Math.PI * 2,
);

// DEGENERATE POLE ROW at both ends: the meridian touches the axis, so an
// entire control row collapses to a single point. Rational in both directions.
const sphere = revolve(
  makeEllipsoidProfile([0, 0, 0], [1, 0, 0], [0, 0, 1], 12, 12, 2),
  [0, 0, 0], [0, 0, 1], 0, Math.PI * 2,
);

// Closed in BOTH directions, rational in both.
const torus = revolve(
  makeCircle([30, 0, 0], [1, 0, 0], [0, 0, 1], 8, 4),
  [0, 0, 0], [0, 0, 1], 0, Math.PI * 2,
);

// TRIPLE-MULTIPLICITY INTERIOR KNOT on a degree-3 curve: multiplicity equals
// the degree, so the curve is only C0 there — a genuine kink, and the exact
// place a mis-set knot span would show up.
const kinkCurve = {
  degree: 3,
  knots: [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
  ctrlPts: [
    [0, 0, 0, 1], [4, 8, 0, 1], [10, 10, 0, 1],
    [16, 4, 0, 1],
    [22, 10, 0, 1], [28, 12, 0, 1], [34, 2, 0, 1],
  ],
};

// The same triple-multiplicity knot carried into a SURFACE's U direction.
const kinkSurface = extrude(kinkCurve, [0, 0, 1], 15);

// Fully-pinned clamped ends, non-rational, multi-span interior.
const interpCurve = globalCurveInterp(
  [[0, 0, 0], [10, 4, 2], [16, 12, 6], [8, 20, 3], [-4, 16, 9]], 3,
);

// A rational arc with genuinely non-unit interior weights.
const arc = makeArc([2, -3, 1], [1, 0, 0], [0, 1, 0], 14, 0, Math.PI / 2, 1);

const CURVES = [
  ['rational quarter arc (non-unit weights)', arc],
  ['degree-3 clamped interpolation (multi-span)', interpCurve],
  ['degree-3 with triple-multiplicity interior knot (C0 kink)', kinkCurve],
];

const SURFACES = [
  ['cylinder (rational, closed in V)', cylinder, 0],
  ['torus (rational, closed in BOTH directions)', torus, 0],
  ['extruded C0-kink curve (triple-multiplicity interior knot in U)', kinkSurface, 0],
  // A pole row makes the surface NORMAL undefined at u=uMin/uMax, so normals
  // are sampled away from it. The POINT is still exactly defined there and is
  // checked at the exact ends by its own test below.
  ['sphere (degenerate pole row at both ends)', sphere, 0.04],
];

// ---------------------------------------------------------------------------
// SURFACE POINT
// ---------------------------------------------------------------------------

for (const [label, srf] of SURFACES) {
  test(`surface point matches rhino3dm: ${label}`, () => {
    const ns = surfaceToRhino(rhino, srf);
    const [uLo, uHi, vLo, vHi] = surfDomain(srf);
    let worst = 0, at = null;
    for (const u of paramsAcross(uLo, uHi, 9)) {
      for (const v of paramsAcross(vLo, vHi, 9)) {
        const d = dist(surfacePoint(srf, u, v), ns.pointAt(u, v));
        if (d > worst) { worst = d; at = [u, v]; }
      }
    }
    assert.ok(
      worst < TOL_POINT,
      `worst surface-point deviation ${worst.toExponential(3)} at (u,v)=${JSON.stringify(at)}`,
    );
  });
}

test('surface point matches rhino3dm AT a degenerate pole row', () => {
  // The pole is where a collapsed control row makes the parametrisation
  // singular. The point itself is still well defined and every sample along
  // the pole row must land on the SAME physical point in both kernels.
  const ns = surfaceToRhino(rhino, sphere);
  const [uLo, uHi, vLo, vHi] = surfDomain(sphere);
  for (const u of [uLo, uHi]) {
    let worst = 0;
    for (const v of paramsAcross(vLo, vHi, 12)) {
      worst = Math.max(worst, dist(surfacePoint(sphere, u, v), ns.pointAt(u, v)));
    }
    assert.ok(worst < TOL_POINT, `pole row u=${u}: worst deviation ${worst.toExponential(3)}`);
  }
});

// ---------------------------------------------------------------------------
// SURFACE NORMAL — the only available check on our own surface partials.
// ---------------------------------------------------------------------------

for (const [label, srf, inset] of SURFACES) {
  test(`surface normal matches rhino3dm: ${label}`, () => {
    const ns = surfaceToRhino(rhino, srf);
    const [uLo, uHi, vLo, vHi] = surfDomain(srf);
    let worst = 0, at = null;
    for (const u of paramsAcross(uLo, uHi, 7, inset)) {
      for (const v of paramsAcross(vLo, vHi, 7, inset)) {
        const { su, sv } = surfacePointAndPartials(srf, u, v);
        const ours = unit(cross(su, sv));
        const theirs = ns.normalAt(u, v);
        // Both are unit vectors, so the chord distance IS the angle error to
        // first order. Compared componentwise rather than by |dot| so a SIGN
        // flip would fail loudly instead of being absorbed.
        const d = dist(ours, theirs);
        if (d > worst) { worst = d; at = [u, v]; }
      }
    }
    assert.ok(
      worst < TOL_NORMAL,
      `worst surface-normal deviation ${worst.toExponential(3)} at (u,v)=${JSON.stringify(at)}`,
    );
  });
}

// ---------------------------------------------------------------------------
// CURVE POINT AND DERIVATIVES
// ---------------------------------------------------------------------------

for (const [label, crv] of CURVES) {
  test(`curve point matches rhino3dm: ${label}`, () => {
    const nc = curveToRhino(rhino, crv);
    const [lo, hi] = curveDomain(crv);
    let worst = 0, at = null;
    for (const u of paramsAcross(lo, hi, 40)) {
      const d = dist(curvePoint(crv, u), nc.pointAt(u));
      if (d > worst) { worst = d; at = u; }
    }
    assert.ok(worst < TOL_POINT, `worst curve-point deviation ${worst.toExponential(3)} at u=${at}`);
  });

  test(`curve 1st and 2nd derivative match rhino3dm: ${label}`, () => {
    const nc = curveToRhino(rhino, crv);
    const [lo, hi] = curveDomain(crv);
    let worst1 = 0, worst2 = 0, at1 = null, at2 = null;
    // Sampled strictly inside: at a C0 kink the two one-sided derivatives
    // genuinely differ, so which one a given implementation reports there is a
    // convention, not a correctness question.
    for (const u of paramsAcross(lo, hi, 40, 0.01)) {
      const ours = rationalCurveDerivs(crv, u, 2);
      const theirs = nc.derivativeAt(u, 2);
      const d1 = dist(ours[1], theirs[1]);
      const d2 = dist(ours[2], theirs[2]);
      if (d1 > worst1) { worst1 = d1; at1 = u; }
      if (d2 > worst2) { worst2 = d2; at2 = u; }
    }
    assert.ok(worst1 < TOL_DERIV, `worst 1st-derivative deviation ${worst1.toExponential(3)} at u=${at1}`);
    assert.ok(worst2 < TOL_DERIV, `worst 2nd-derivative deviation ${worst2.toExponential(3)} at u=${at2}`);
  });
}

// ---------------------------------------------------------------------------
// THE FIRST GATE, kept as a permanent regression: if THIS ever fails, the
// harness is wrong (a conversion or a sampling-domain mistake), not the
// kernel. It is asked for explicitly and it is cheap to keep.
// ---------------------------------------------------------------------------

test('harness sanity: a plain clamped degree-3 surface agrees to machine precision', () => {
  const flat = extrude(interpCurve, [0, 0, 1], 10);
  const ns = surfaceToRhino(rhino, flat);
  const [uLo, uHi, vLo, vHi] = surfDomain(flat);
  let worst = 0;
  for (const u of paramsAcross(uLo, uHi, 6)) {
    for (const v of paramsAcross(vLo, vHi, 6)) {
      worst = Math.max(worst, dist(surfacePoint(flat, u, v), ns.pointAt(u, v)));
    }
  }
  assert.ok(worst < 1e-12, `first gate: worst deviation ${worst.toExponential(3)} exceeds machine precision`);
});
