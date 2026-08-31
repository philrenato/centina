import test from 'node:test';
import assert from 'node:assert/strict';
import { surfacePoint } from '../kernel/surface.mjs';
import { envelopeSection, envelopeSectionArc, blendSurfaceFromSections, blendSurfaceToTolerance, variableRadiusFeasible } from '../kernel/fillet.mjs';
import {
  radiusProfile, thickThinThickProfile, variableRadiusSection, variableRadiusSectionArc,
  spineFrame, profileFeasibility, canalDeviation, variableRadiusBlend,
} from '../kernel/varradius.mjs';

const near = (a, b, tol = 1e-12) => Math.abs(a - b) < tol;

/* ═══════════════════════════════════════════════════════════════════════
   THE RADIUS PROFILE
   ═══════════════════════════════════════════════════════════════════════ */

test('a profile refuses everything that would make r(t) meaningless, by name', () => {
  assert.equal(radiusProfile([[0, 3]]).ok, false, 'one stop is a constant-radius fillet, not a profile');
  assert.match(radiusProfile([[0, 3], [1, 0]]).reason, /positive/);
  assert.match(radiusProfile([[0, 3], [1, -2]]).reason, /positive/);
  // Equal parameters are the real trap: they divide by a zero segment and would
  // emit Infinity slopes rather than refusing.
  assert.match(radiusProfile([[0, 3], [0.5, 2], [0.5, 4], [1, 3]]).reason, /strictly ordered/);
  assert.match(radiusProfile([[0, 3], [0.7, 2], [0.4, 4], [1, 3]]).reason, /strictly ordered/);
  assert.match(radiusProfile([[0.2, 3], [1, 2]]).reason, /span the whole edge/);
  assert.match(radiusProfile([[0, 3], [0.8, 2]]).reason, /span the whole edge/);
  assert.match(radiusProfile([[0, 3], [NaN, 2], [1, 3]]).reason, /finite/);
  assert.equal(radiusProfile([[0, 3], [1, 5]]).ok, true);
  assert.equal(radiusProfile([{ t: 0, radius: 3 }, { t: 1, radius: 5 }]).ok, true, 'records read the same as pairs');
});

test('the profile passes through every stop exactly, and two stops are a straight taper in closed form', () => {
  const stops = [[0, 8], [0.15, 0.4], [0.9, 0.5], [1, 6]];
  const p = radiusProfile(stops);
  assert.equal(p.ok, true, p.reason);
  for (const [t, r] of stops) assert.ok(near(p.radiusAt(t), r, 1e-14), `stop at ${t} must evaluate to ${r}, got ${p.radiusAt(t)}`);

  // TWO STOPS ARE THE STRAIGHT LINE, exactly — the Hermite cubic through two
  // points whose end slopes are both the segment slope IS the line, so this is a
  // closed form and not an approximation of one.
  const taper = radiusProfile([[0, 2], [1, 7]]);
  for (let i = 0; i <= 200; i++) {
    const t = i / 200;
    assert.ok(near(taper.radiusAt(t), 2 + 5 * t, 1e-14), `a two-stop profile must be exactly linear; at ${t} it is ${taper.radiusAt(t)} not ${2 + 5 * t}`);
    assert.ok(near(taper.slopeAt(t), 5, 1e-12), `and its slope must be exactly the taper (${taper.slopeAt(t)})`);
  }
});

/* A NATURAL CUBIC SPLINE THROUGH THE SAME STOPS, as the control. Without it,
   "no overshoot" is a claim about a function nobody compared against anything —
   every interpolant looks well behaved on a plot of itself. */
function naturalCubic(ts, rs) {
  const n = ts.length, h = [], al = new Array(n).fill(0);
  for (let i = 0; i + 1 < n; i++) h.push(ts[i + 1] - ts[i]);
  for (let i = 1; i + 1 < n; i++) al[i] = 3 * ((rs[i + 1] - rs[i]) / h[i] - (rs[i] - rs[i - 1]) / h[i - 1]);
  const l = new Array(n).fill(1), mu = new Array(n).fill(0), z = new Array(n).fill(0);
  for (let i = 1; i + 1 < n; i++) {
    l[i] = 2 * (ts[i + 1] - ts[i - 1]) - h[i - 1] * mu[i - 1];
    mu[i] = h[i] / l[i];
    z[i] = (al[i] - h[i - 1] * z[i - 1]) / l[i];
  }
  const c = new Array(n).fill(0), b = new Array(n).fill(0), d = new Array(n).fill(0);
  for (let j = n - 2; j >= 0; j--) {
    c[j] = z[j] - mu[j] * c[j + 1];
    b[j] = (rs[j + 1] - rs[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
    d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
  }
  return (x) => {
    let i = 0;
    while (i + 2 < n && x >= ts[i + 1]) i++;
    const dx = x - ts[i];
    return rs[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
  };
}

test('the interpolation is shape-preserving, and the smoother alternative goes NEGATIVE on the same stops', () => {
  /* ⚠ THE STOP SETS ARE CHOSEN TO REACH EVERY CLAMP, because a profile that
     merely looks reasonable exercises none of them. Fritsch & Carlson's rule has
     three separate guards and each needs its own shape to fire:
       · an interior extremum (zero the slope),
       · an END whose three-point extrapolation points the WRONG WAY — a gentle
         first segment followed by a steep one, where the unclamped slope is
         -8.8 and drags a profile whose smallest stop is 1 down to 0.354,
       · an END at a turning point where the extrapolation is more than three
         times the segment slope, which overshoots the largest stop.
     Range-checking one well-behaved profile passes with all three removed. */
  const sets = [
    [[0, 8], [0.15, 0.4], [0.9, 0.5], [1, 6]],   // interior extremum, uneven spacing
    [[0, 1], [0.5, 1.05], [1, 10]],              // gentle then steep: the start clamp
    [[0, 10], [0.5, 1.05], [1, 1]],              // and its mirror at the far end
    [[0, 5], [0.5, 6], [1, 1]],                  // turning point at the start: the 3x clamp
    [[0, 1], [0.5, 6], [1, 5]],                  // and its mirror
  ];
  for (const stops of sets) {
    const p = radiusProfile(stops);
    assert.equal(p.ok, true, p.reason);
    const rs = stops.map((s) => s[1]);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i <= 8000; i++) {
      const r = p.radiusAt(i / 8000);
      lo = Math.min(lo, r); hi = Math.max(hi, r);
    }
    // The whole point: r(t) never leaves the range of the stops, so positive
    // stops are a positive radius everywhere with nothing left to hope for.
    assert.ok(near(lo, Math.min(...rs), 1e-12), `${JSON.stringify(stops)}: the minimum must be the smallest stop, exactly (${lo})`);
    assert.ok(near(hi, Math.max(...rs), 1e-12), `${JSON.stringify(stops)}: and the maximum the largest (${hi})`);
    assert.ok(near(p.minRadius, lo, 1e-12) && near(p.maxRadius, hi, 1e-12), 'and the reported range must be the measured one');
  }
  // THE CONTROL. A natural cubic through the very same stops asks for a ball of
  // radius -8.5 — a fillet inverted through its own spine.
  const stops = sets[0];
  const nat = naturalCubic(stops.map((s) => s[0]), stops.map((s) => s[1]));
  let natLo = Infinity, natHi = -Infinity;
  for (let i = 0; i <= 4000; i++) { const q = nat(i / 4000); natLo = Math.min(natLo, q); natHi = Math.max(natHi, q); }
  assert.ok(natLo < 0, `the control must actually overshoot or this test proves nothing (natural cubic minimum ${natLo})`);
  assert.ok(natLo < -8, `and it goes far negative (${natLo.toFixed(4)})`);
  assert.ok(natHi >= 8, `${natHi}`);
});

test('the profile is C1, and the waist of a three-stop profile has exactly zero slope', () => {
  const p = radiusProfile([[0, 5], [0.37, 1], [1, 4]]);
  // slopeAt is the analytic derivative and is what sets every section's tilt, so
  // it is checked against a difference of radiusAt rather than trusted — across
  // the stop as well as within the segments, which is where a C0 interpolation
  // would show up.
  for (let i = 1; i < 400; i++) {
    const t = i / 400;
    const h = 1e-6;
    const fd = (p.radiusAt(Math.min(1, t + h)) - p.radiusAt(Math.max(0, t - h))) / (Math.min(1, t + h) - Math.max(0, t - h));
    assert.ok(Math.abs(fd - p.slopeAt(t)) < 2e-4, `slopeAt disagrees with a difference of radiusAt at t=${t}: ${p.slopeAt(t)} vs ${fd}`);
  }
  assert.ok(near(p.slopeAt(0.37), 0, 1e-14), `an interior minimum stop gets a zero slope, so the waist lands ON it (${p.slopeAt(0.37)})`);
  // And approaching from both sides agrees — the C1 claim, not just C0.
  assert.ok(Math.abs(p.slopeAt(0.37 - 1e-7) - p.slopeAt(0.37 + 1e-7)) < 1e-5, 'the slope must not jump at a stop');
});

test('thick -> thin -> thick is the named default, symmetric and bounded by its ends', () => {
  const p = thickThinThickProfile({ ends: 4, waist: 1.5 });
  assert.equal(p.ok, true, p.reason);
  assert.equal(p.shape, 'thick-thin-thick');
  assert.ok(near(p.radiusAt(0), 4, 1e-14) && near(p.radiusAt(1), 4, 1e-14), 'the ends are equal and exact');
  assert.ok(near(p.radiusAt(0.5), 1.5, 1e-14), 'the waist is exactly the waist, at exactly the middle');
  assert.ok(near(p.slopeAt(0.5), 0, 1e-14), 'and it is a genuine minimum there, not a point the curve passes through on its way down');
  for (let i = 0; i <= 1000; i++) {
    const t = i / 1000;
    assert.ok(near(p.radiusAt(t), p.radiusAt(1 - t), 1e-14), `symmetric stops must give a symmetric profile (t=${t})`);
    assert.ok(p.radiusAt(t) >= 1.5 - 1e-14 && p.radiusAt(t) <= 4 + 1e-14, `and nothing outside [waist, ends] (${p.radiusAt(t)} at ${t})`);
  }
  // The one-knob form the UI wants, and the same profile it would have built.
  const frac = thickThinThickProfile({ ends: 4, waistFraction: 0.375 });
  assert.equal(frac.ok, true);
  assert.ok(near(frac.radiusAt(0.5), 1.5, 1e-14), `waistFraction must resolve to the same waist (${frac.radiusAt(0.5)})`);
  // An off-center waist stays where it was put.
  const off = thickThinThickProfile({ ends: 4, waist: 1.5, at: 0.25 });
  assert.ok(near(off.radiusAt(0.25), 1.5, 1e-14));
  assert.ok(near(off.slopeAt(0.25), 0, 1e-14));
  // And the refusals, which are what keep the name honest.
  assert.match(thickThinThickProfile({ ends: 4, waist: 4 }).reason, /not thinner|constant/);
  assert.match(thickThinThickProfile({ ends: 4, waist: 6 }).reason, /not thinner|constant/);
  assert.match(thickThinThickProfile({ ends: 4, waist: 1, at: 0 }).reason, /strictly inside/);
  assert.match(thickThinThickProfile({ ends: 0, waist: 1 }).reason, /positive/);
});

/* ═══════════════════════════════════════════════════════════════════════
   THE SECTION — checked against closed forms, not against itself
   ═══════════════════════════════════════════════════════════════════════ */

/* THE CONE. A straight spine with a LINEAR radius law has an exact envelope:
   the balls' boundary is a right circular cone. For centers on the z axis and
   r = a + b*z (so b IS dr/ds), the contact point of the ball at height z sits at
   radius r*sqrt(1-b^2) from the axis and at height z - r*b, which eliminates to

       sqrt(x^2 + y^2) * sqrt(1 - b^2)  =  a + b*z.

   Every point of a correct variable-radius blend on this spine satisfies that
   to machine precision. It is the closed form this whole module is checked
   against, and it exists only because the radius varies — a constant radius
   would make it a cylinder and hide every tilt error in the construction. */
const L = 40, aCone = 2, dCone = 6;
const bCone = dCone / L;
const kCone = Math.sqrt(1 - bCone * bCone);
const coneProfile = radiusProfile([[0, aCone], [1, aCone + dCone]]);
function coneBall() {
  return { centre: null, toTouchA: [kCone, 0, -bCone], toTouchB: [0, kCone, -bCone] };
}
function coneBallAt(t) {
  return { ...coneBall(), centre: [0, 0, L * t], dCentreDt: [0, 0, L], dCentreDr: [0, 0, 0] };
}
const coneError = (p) => Math.hypot(p[0], p[1]) * kCone - (aCone + bCone * p[2]);

test('a variable-radius blend on a straight spine IS the exact cone, to machine precision', () => {
  const built = variableRadiusBlend({ ballAt: (t) => coneBallAt(t), profile: coneProfile, tolerance: 1e-9, evalSrf: surfacePoint });
  assert.equal(built.ok, true, built.reason);
  assert.equal(built.metTolerance, true, `${built.deviation}`);
  let worst = 0;
  for (let i = 0; i <= 16; i++) {
    for (let j = 0; j <= 16; j++) {
      const p = surfacePoint(built.srf, i / 16, j / 16);
      worst = Math.max(worst, Math.abs(coneError(p)));
    }
  }
  assert.ok(worst < 1e-12, `every point must lie ON the cone, not near it (worst ${worst.toExponential(3)})`);
  /* TWO RULERS, ONE NUMBER. The cone equation and the signed canal measure share
     no algebra — one is a closed form in the surface's own coordinates, the
     other a minimization over the sphere family — so their agreement is what
     licenses the canal measure to be believed on the fixtures below, where no
     closed form exists. */
  assert.ok(built.deviation < 1e-12, `and the canal measure must agree it is exact (${built.deviation.toExponential(3)})`);
  assert.ok(Math.abs(built.deviation - worst) < 1e-12, `the two rulers must agree (${built.deviation.toExponential(3)} vs ${worst.toExponential(3)})`);
});

test('the section TILT is load-bearing: great-circle sections miss the same cone by 1.5e-2 and never converge', () => {
  /* A constant-radius section builder puts the arc on a GREAT circle of the
     ball. Every point of that arc is still exactly `radius` from the ball
     center, so a radius-only instrument reports a perfect blend — this is the
     error the unsigned measure cannot see, and the reason `canalDeviation` is
     signed. */
  const measure = (n) => {
    const arcs = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1), r = coneProfile.radiusAt(t), b = coneBallAt(t);
      arcs.push(envelopeSectionArc(envelopeSection({ centre: b.centre, radius: r, toTouchA: b.toTouchA, toTouchB: b.toTouchB })));
    }
    const built = blendSurfaceFromSections(arcs);
    assert.equal(built.ok, true, built.reason);
    let cone = 0, ballRadius = 0;
    for (let i = 0; i <= 12; i++) {
      for (let j = 0; j <= 12; j++) {
        const p = surfacePoint(built.srf, i / 12, j / 12);
        cone = Math.max(cone, Math.abs(coneError(p)));
      }
    }
    // What the radius-only question would have answered, at the sections
    // themselves: exactly zero, for a surface that is 1.5e-2 off the cone.
    for (let i = 0; i <= 12; i++) {
      const u = i / 12, t = 0.5;
      const p = surfacePoint(built.srf, u, t);
      const idx = Math.round(t * (n - 1)) / (n - 1);
      const c = coneBallAt(idx).centre, r = coneProfile.radiusAt(idx);
      ballRadius = Math.max(ballRadius, Math.abs(Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) - r));
    }
    const at = (t) => ({ centre: coneBallAt(t).centre, radius: coneProfile.radiusAt(t) });
    const canal = canalDeviation(built.srf, at, surfacePoint, { spineSamples: 257 });
    return { cone, ballRadius, canal: canal.worst, signed: canal.worstSigned };
  };
  const m9 = measure(9), m33 = measure(33), m129 = measure(129);
  assert.ok(m9.cone > 1e-2, `the great-circle build must genuinely miss the cone (${m9.cone.toExponential(3)})`);
  // STRUCTURAL, NOT A SAMPLING CHOICE. Every section is individually in the
  // wrong plane, so adding sections cannot help — and this is exactly the shape
  // of error that gets mistaken for "needs more sections".
  assert.ok(m129.cone > 0.9 * m9.cone,
    `and it must NOT fall with section count — 9: ${m9.cone.toExponential(3)}, 33: ${m33.cone.toExponential(3)}, 129: ${m129.cone.toExponential(3)}`);
  assert.ok(m9.canal > 1e-2 && m9.signed < 0, `the signed canal measure sees it, and sees that it cuts too deep (${m9.signed.toExponential(3)})`);
  console.log(`      great-circle sections on a tapered spine: ${m9.cone.toExponential(2)}mm off the cone at 9 sections, ${m129.cone.toExponential(2)}mm at 129 — structural`);
});

/* THE REAL FILLET FIXTURE: a cylinder of radius 20 about +z cut by a plane
   inclined at 60 degrees, so the dihedral genuinely varies around the
   intersection ellipse — AND the radius varies along it. A ball of radius r
   tangent to both sits at distance (Rc - r) from the axis and r from the plane,
   both closed form, and the center MOVES when r changes, which is what makes
   this a real rolling-ball spine rather than a prescribed one. */
const Rc = 20, alpha = 60 * Math.PI / 180, S0 = -1.4, SPAN = 2.8;
function filletBall(t, r) {
  const s = S0 + SPAN * t, rho = Rc - r;
  const x = rho * Math.cos(s), y = rho * Math.sin(s);
  return {
    centre: [x, y, (x * Math.sin(alpha) - r) / Math.cos(alpha)],
    toTouchA: [Math.cos(s), Math.sin(s), 0],
    toTouchB: [-Math.sin(alpha), 0, Math.cos(alpha)],
  };
}
function filletBallAnalytic(t, r) {
  const s = S0 + SPAN * t, rho = Rc - r;
  const dxdt = -rho * Math.sin(s) * SPAN, dydt = rho * Math.cos(s) * SPAN;
  const dxdr = -Math.cos(s), dydr = -Math.sin(s);
  return {
    ...filletBall(t, r),
    dCentreDt: [dxdt, dydt, dxdt * Math.sin(alpha) / Math.cos(alpha)],
    dCentreDr: [dxdr, dydr, (dxdr * Math.sin(alpha) - 1) / Math.cos(alpha)],
  };
}
const waisted = thickThinThickProfile({ ends: 3, waist: 1.2 });
const planeDist = (p) => -p[0] * Math.sin(alpha) + p[2] * Math.cos(alpha);
const cylDist = (p) => Math.abs(Math.hypot(p[0], p[1]) - Rc);

test('the fixture is a real rolling ball before anything is blamed on the code', () => {
  // Contact, not radius: each touch point must LIE ON its surface.
  for (let i = 0; i <= 12; i++) {
    const t = i / 12, r = waisted.radiusAt(t), b = filletBall(t, r);
    const pA = b.centre.map((v, k) => v + r * b.toTouchA[k]);
    const pB = b.centre.map((v, k) => v + r * b.toTouchB[k]);
    assert.ok(cylDist(pA) < 1e-12, `touch A must lie on the cylinder wall (${cylDist(pA)})`);
    assert.ok(Math.abs(planeDist(pB)) < 1e-12, `touch B must lie on the plane (${planeDist(pB)})`);
  }
  /* AND THE ROLLING INVARIANT, which is what makes the tilt computable at all:
     for any ball kept tangent to a fixed surface, n . m' = -r' identically. Both
     touch directions must therefore have the SAME component along the spine
     tangent, equal to -dr/ds. If this fails the fixture is not a rolling ball
     and every measurement below would be judging the wrong object. */
  let worstMismatch = 0, minRate = Infinity, maxRate = -Infinity;
  for (let i = 0; i <= 12; i++) {
    const t = i / 12;
    const f = spineFrame(filletBall, waisted, t);
    assert.equal(f.ok, true, f.reason);
    const dA = f.toTouchA.reduce((s, v, k) => s + v * f.spineTangent[k], 0);
    const dB = f.toTouchB.reduce((s, v, k) => s + v * f.spineTangent[k], 0);
    worstMismatch = Math.max(worstMismatch, Math.abs(dA + f.radiusRate), Math.abs(dB + f.radiusRate));
    minRate = Math.min(minRate, f.radiusRate); maxRate = Math.max(maxRate, f.radiusRate);
  }
  assert.ok(worstMismatch < 1e-9, `n . T must equal -dr/ds on both faces (worst ${worstMismatch.toExponential(3)})`);
  assert.ok(maxRate - minRate > 0.1, `and the radius must genuinely be changing, or this is a constant-radius test wearing a profile (${minRate.toFixed(4)} .. ${maxRate.toFixed(4)})`);
  // The dihedral varies too, so the fixture exercises both things at once.
  let minAng = Infinity, maxAng = -Infinity;
  for (let i = 0; i <= 12; i++) {
    const b = filletBall(i / 12, 2);
    const c = b.toTouchA.reduce((s, v, k) => s + v * b.toTouchB[k], 0);
    const ang = Math.acos(Math.max(-1, Math.min(1, c)));
    minAng = Math.min(minAng, ang); maxAng = Math.max(maxAng, ang);
  }
  assert.ok((maxAng - minAng) * 180 / Math.PI > 15, `the dihedral must vary as well (${((maxAng - minAng) * 180 / Math.PI).toFixed(1)} degrees)`);
});

test('a CONSTANT profile reproduces the constant-radius blend BIT FOR BIT', () => {
  /* The strongest closed form available: with dr/ds = 0 the tilt is zero, the
     contact circle's center is the ball center and its radius is the ball's, so
     every control point must come out IDENTICAL — not close, identical. A
     construction that merely converged to the constant-radius one would be a
     second implementation of the same thing, and the two would drift. */
  const flat = radiusProfile([[0, 3], [0.5, 3], [1, 3]]);
  let compared = 0;
  for (let i = 0; i < 17; i++) {
    const t = i / 16;
    const f = spineFrame(filletBall, flat, t);
    assert.equal(f.ok, true, f.reason);
    assert.equal(f.radiusRate, 0, `a constant profile must give exactly zero rate, not nearly zero (${f.radiusRate})`);
    const varArc = variableRadiusSectionArc(variableRadiusSection(f));
    const r = flat.radiusAt(t), b = filletBall(t, r);
    const constArc = envelopeSectionArc(envelopeSection({ centre: b.centre, radius: r, toTouchA: b.toTouchA, toTouchB: b.toTouchB }));
    assert.ok(varArc && constArc);
    for (let q = 0; q < 3; q++) {
      for (let c = 0; c < 4; c++) {
        assert.equal(varArc.ctrlPts[q][c], constArc.ctrlPts[q][c],
          `control point ${q}[${c}] at t=${t} must be identical: ${varArc.ctrlPts[q][c]} vs ${constArc.ctrlPts[q][c]}`);
        compared++;
      }
    }
  }
  assert.equal(compared, 17 * 12, 'and every one of them must actually have been compared');
});

test('a section refuses a radius outrunning the spine, and a spine that is not the one the ball is rolling on', () => {
  const base = { centre: [0, 0, 0], radius: 2, toTouchA: [1, 0, 0], toTouchB: [0, 1, 0], spineTangent: [0, 0, 1] };
  // |dr/ds| >= 1 is the Peternell & Pottmann condition, pointwise.
  assert.match(variableRadiusSection({ ...base, radiusRate: 1.2 }).reason, /ceiling|collapsed/);
  assert.match(variableRadiusSection({ ...base, radiusRate: 1 }).reason, /ceiling|collapsed/);
  assert.match(variableRadiusSection({ ...base, radiusRate: -1.0001 }).reason, /ceiling|collapsed/);
  // A rate this section's own touch directions contradict. Both directions here
  // are perpendicular to the tangent, so they assert dr/ds = 0; claiming 0.3
  // describes a ball that is not in contact with anything.
  const bad = variableRadiusSection({ ...base, radiusRate: 0.3 });
  assert.equal(bad.ok, false, 'an inconsistent tilt must refuse, not be averaged away');
  assert.match(bad.reason, /not rolling in contact/);
  assert.ok(near(bad.tiltMismatch, 0.3, 1e-12), `and it must say by how much (${bad.tiltMismatch})`);
  assert.equal(variableRadiusSection({ ...base, radiusRate: NaN }).ok, false, 'a missing rate must refuse rather than default to a constant radius');
  assert.match(variableRadiusSection({ ...base, radiusRate: undefined }).reason, /dr\/ds/);
  // The consistent version of the same section builds, and its contact circle is
  // offset along the spine and shrunk by exactly the closed-form amounts.
  const rate = 0.3, k = Math.sqrt(1 - rate * rate);
  const good = variableRadiusSection({
    centre: [0, 0, 0], radius: 2, spineTangent: [0, 0, 1], radiusRate: rate,
    toTouchA: [k, 0, -rate], toTouchB: [0, k, -rate],
  });
  assert.equal(good.ok, true, good.reason);
  assert.ok(near(good.contactCentre[2], -2 * rate, 1e-14), `the contact circle sits r*tilt along the spine (${good.contactCentre[2]})`);
  assert.ok(near(good.contactRadius, 2 * k, 1e-14), `and has radius r*sqrt(1-tilt^2) (${good.contactRadius})`);
  assert.ok(near(Math.hypot(...good.tangencyA.map((v, i) => v - good.contactCentre[i])), good.contactRadius, 1e-14),
    'and both tangency points must lie exactly on it');
  assert.ok(near(Math.hypot(...good.tangencyB.map((v, i) => v - good.contactCentre[i])), good.contactRadius, 1e-14));

  /* ⚠ AND THE CASE THE "OPPOSED FACES" GUARD CANNOT SEE. Tilting both touch
     directions towards the spine pulls them together: two directions that are
     diametrically opposite IN THE CONTACT PLANE have nA . nB = +0.28 once they
     share a tilt of 0.8, which is nowhere near the -1 that guard tests for. The
     contact arc still sweeps a full 180 degrees, which no single rational
     quadratic can carry. Refused by the sweep, not by the dot product. */
  const tilt = 0.8, kt = Math.sqrt(1 - tilt * tilt);
  const flat = variableRadiusSection({
    centre: [0, 0, 0], radius: 2, spineTangent: [0, 0, 1], radiusRate: -tilt,
    toTouchA: [kt, 0, tilt], toTouchB: [-kt, 0, tilt],
  });
  assert.equal(flat.ok, false, 'a 180-degree contact arc must refuse');
  assert.match(flat.reason, /sweeps/);
  assert.ok(flat.sweep > Math.PI - 1e-9, `${flat.sweep}`);
});

/* ═══════════════════════════════════════════════════════════════════════
   MEASURING — proving the ruler before quoting it
   ═══════════════════════════════════════════════════════════════════════ */

function buildFillet(n, profile = waisted, ballAt = filletBall) {
  const arcs = [];
  for (let i = 0; i < n; i++) {
    const f = spineFrame(ballAt, profile, i / (n - 1));
    assert.equal(f.ok, true, f.reason);
    const s = variableRadiusSection(f);
    assert.equal(s.ok, true, s.reason);
    arcs.push(variableRadiusSectionArc(s));
  }
  const built = blendSurfaceFromSections(arcs);
  assert.equal(built.ok, true, built.reason);
  return built;
}
const filletAt = (t) => {
  const r = waisted.radiusAt(t);
  return { centre: filletBall(t, r).centre, radius: r };
};

test('the ruler is refined, not merely dense — and the number it reports stops moving', () => {
  const built = buildFillet(49);
  const readings = [33, 65, 129, 257, 1025].map((s) => ({
    s,
    refined: canalDeviation(built.srf, filletAt, surfacePoint, { spineSamples: s }).worst,
    raw: canalDeviation(built.srf, filletAt, surfacePoint, { spineSamples: s, refine: false }).worst,
  }));
  const ref = readings[0].refined;
  for (const r of readings) {
    assert.ok(Math.abs(r.refined - ref) / ref < 1e-6,
      `a refined reading must not depend on the sampling: ${r.s} samples gives ${r.refined.toExponential(6)} against ${ref.toExponential(6)}`);
  }
  /* AND THE UNREFINED READING IS NOT A CRUDER VERSION OF THE SAME NUMBER — it is
     a different number by three orders. A sampled minimum of a function with a
     stationary minimum always OVERSTATES, so density alone reports a better
     surface than exists, and it does so at every density anyone would pay for. */
  assert.ok(readings[3].raw > 100 * ref,
    `the sampled ruler at 257 must be wildly too kind, or refinement is decorative (${readings[3].raw.toExponential(2)} vs ${ref.toExponential(2)})`);
  assert.ok(readings[4].raw < readings[0].raw / 100, 'and the sampled one must at least fall with density, confirming it is the ruler moving');
  console.log(`      ruler: refined ${ref.toExponential(6)}mm at every density 33..1025; sampled reads ${readings[3].raw.toExponential(2)}mm at 257 and ${readings[4].raw.toExponential(2)}mm at 1025`);
  // The measure reports the ruler's own contribution rather than leaving it to
  // be guessed at.
  const d = canalDeviation(built.srf, filletAt, surfacePoint, { spineSamples: 257 });
  assert.equal(d.instrumentBound, false, 'a converged search is not instrument-bound');
  assert.ok(d.refinedBracket < 1e-9, `${d.refinedBracket}`);
  assert.ok(d.sampleFloor > d.worst, 'and it must still say how much the refinement was worth');
});

test('the blend converges with section count, at better than third order', () => {
  const at = (n) => canalDeviation(buildFillet(n).srf, filletAt, surfacePoint, { spineSamples: 257, vSteps: 41 });
  const a1 = at(13), a2 = at(25), a3 = at(49);
  for (const a of [a1, a2, a3]) assert.equal(a.instrumentBound, false, 'the order is only asserted where the ruler is not the bound');
  const o1 = Math.log2(a1.worst / a2.worst), o2 = Math.log2(a2.worst / a3.worst);
  assert.ok(o1 > 3 && o2 > 3, `cubic interpolation in V should give roughly fourth order (${o1.toFixed(2)}, ${o2.toFixed(2)})`);
  assert.ok(a3.worst < 1e-5, `and by 49 sections it is under 1e-5mm (${a3.worst.toExponential(2)})`);
  console.log(`      variable radius on a varying dihedral: 13 sections ${a1.worst.toExponential(2)}mm -> 25 ${a2.worst.toExponential(2)}mm -> 49 ${a3.worst.toExponential(2)}mm (orders ${o1.toFixed(2)}, ${o2.toFixed(2)})`);
});

test('closed-form derivatives and differenced ones build the same surface', () => {
  // The frame can come from the caller in closed form or be differenced here.
  // They are independent routes to the same tilt, so a drift in either shows up
  // as a disagreement rather than as a slightly worse surface nobody noticed.
  let worst = 0;
  for (let i = 0; i <= 20; i++) {
    const t = i / 20;
    const fd = spineFrame(filletBall, waisted, t);
    const an = spineFrame(filletBallAnalytic, waisted, t);
    assert.equal(an.analytic, true);
    assert.equal(fd.analytic, false);
    worst = Math.max(worst, Math.abs(fd.radiusRate - an.radiusRate),
      ...fd.spineTangent.map((v, k) => Math.abs(v - an.spineTangent[k])));
  }
  assert.ok(worst < 1e-9, `differenced and closed-form frames must agree (${worst.toExponential(3)})`);
  const bFd = buildFillet(25, waisted, filletBall), bAn = buildFillet(25, waisted, filletBallAnalytic);
  let net = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < bFd.srf.ctrlNet[i].length; j++) {
      for (let c = 0; c < 4; c++) net = Math.max(net, Math.abs(bFd.srf.ctrlNet[i][j][c] - bAn.srf.ctrlNet[i][j][c]));
    }
  }
  assert.ok(net < 1e-9, `and the surfaces they build must agree (${net.toExponential(3)})`);
});

test('the tangency curves still lie on both supporting surfaces when the radius varies', () => {
  /* The check a radius-only instrument cannot make: the blend has to TOUCH both
     faces, and where it touches is where they must be trimmed back to.

     ⚠ MEASURED ON THE BORDER CURVES, NOT ON THEIR CONTROL POINTS. The tangency
     curves are interpolants through the section tangency points, so their
     control polygons sit off a curved supporting surface by design — 0.125mm
     here, which is a control polygon standing away from a wall and not a blend
     missing it.

     ⚠ AND THIS IS NOT THE QUANTITY THE TOLERANCE LOOP CERTIFIES. That loop
     measures the ENVELOPE deviation — how far the surface strays from the balls.
     How far the tangency curve strays from the face it is supposed to lie in is
     a different number of the same order, and it is the one a TRIM depends on.
     Asserted separately for that reason. */
  const border = (n) => {
    const arcs = [];
    for (let i = 0; i < n; i++) arcs.push(variableRadiusSectionArc(variableRadiusSection(spineFrame(filletBall, waisted, i / (n - 1)))));
    const b = blendSurfaceFromSections(arcs);
    assert.equal(b.ok, true, b.reason);
    let cyl = 0, plane = 0;
    for (let j = 0; j <= 400; j++) {
      const v = j / 400;
      cyl = Math.max(cyl, cylDist(surfacePoint(b.srf, 0, v)));
      plane = Math.max(plane, Math.abs(planeDist(surfacePoint(b.srf, 1, v))));
    }
    return { cyl, plane };
  };
  const b13 = border(13), b25 = border(25), b49 = border(49);
  // ON A PLANAR FACE IT IS EXACT AND STAYS EXACT: an interpolant through
  // coplanar points cannot leave their plane, at any section count.
  for (const b of [b13, b25, b49]) assert.ok(b.plane < 1e-13, `the tangency curve on a PLANAR face must be exact, not merely close (${b.plane.toExponential(2)})`);
  // On the CURVED face it is an interpolation error and must fall like one.
  const o1 = Math.log2(b13.cyl / b25.cyl), o2 = Math.log2(b25.cyl / b49.cyl);
  assert.ok(o1 > 3 && o2 > 3, `and on the curved face it must converge at better than third order (${o1.toFixed(2)}, ${o2.toFixed(2)})`);
  assert.ok(b49.cyl < 1e-4, `${b49.cyl.toExponential(2)}`);
  // The curves handed to a trim are those same two borders of the built surface.
  const built = variableRadiusBlend({ ballAt: filletBall, profile: waisted, tolerance: 0.001, evalSrf: surfacePoint });
  assert.equal(built.ok, true, built.reason);
  assert.equal(built.tangencyCurveA.ctrlPts.length, built.srf.ctrlNet[0].length);
  assert.equal(built.tangencyCurveB.ctrlPts[0][0], built.srf.ctrlNet[2][0][0]);
  console.log(`      tangency curve off the CURVED face: ${b13.cyl.toExponential(2)}mm at 13 sections -> ${b49.cyl.toExponential(2)}mm at 49 (orders ${o1.toFixed(2)}, ${o2.toFixed(2)}); exact on the planar face at every count`);
});

test('the tolerance loop still chooses the section count by measuring, with a radius that varies', () => {
  const loose = variableRadiusBlend({ ballAt: filletBall, profile: waisted, tolerance: 0.01, evalSrf: surfacePoint });
  assert.equal(loose.ok, true, loose.reason);
  assert.equal(loose.metTolerance, true, `${loose.deviation}`);
  assert.ok(loose.deviation <= 0.01);
  const tight = variableRadiusBlend({ ballAt: filletBall, profile: waisted, tolerance: 1e-5, evalSrf: surfacePoint });
  assert.equal(tight.ok, true, tight.reason);
  assert.ok(tight.sections > loose.sections, `a tighter ask must cost sections (${tight.sections} vs ${loose.sections})`);
  assert.ok(tight.deviation < loose.deviation, `and achieve less deviation (${tight.deviation} vs ${loose.deviation})`);
  // An impossible ask returns a real surface and SAYS it fell short.
  const impossible = variableRadiusBlend({ ballAt: filletBall, profile: waisted, tolerance: 1e-15, evalSrf: surfacePoint, maxSections: 33 });
  assert.equal(impossible.ok, true, 'a surface that missed the target is still a surface');
  assert.equal(impossible.metTolerance, false, 'and it must say it missed');
  assert.ok(impossible.sections <= 33, 'and honour the ceiling it was given');
  console.log(`      to 0.01mm: ${loose.sections} sections, achieved ${loose.deviation.toExponential(2)}mm; to 1e-5mm: ${tight.sections} sections, ${tight.deviation.toExponential(2)}mm`);
});

test('the constant-radius builder still refuses a varying radius when no measure replaces its own', () => {
  // The hook that lets a variable-radius blend through must not have opened the
  // door for a caller who did not mean to vary anything.
  const r = blendSurfaceToTolerance((t) => {
    const b = filletBall(t, 2 + t);
    return { centre: b.centre, radius: 2 + t, toTouchA: b.toTouchA, toTouchB: b.toTouchB };
  }, 0.01, { evalSrf: surfacePoint });
  assert.equal(r.ok, false);
  assert.match(r.reason, /ONE radius/);
});

/* ═══════════════════════════════════════════════════════════════════════
   FEASIBILITY — ENFORCED
   ═══════════════════════════════════════════════════════════════════════ */

/* A PRESCRIBED SPINE, which is the only way the canal condition is reachable.
   For a ball genuinely rolling on fixed faces, n . m' = -r' makes |dr/ds| <= 1
   an identity — see the right-angle test below. What violates the condition is a
   center path that does NOT move when the radius does: a variable-radius pipe,
   and equally an app that takes its spine from an offset of the edge curve and
   then applies a profile to it. */
function pipeBall(profile) {
  return (t) => {
    const rate = profile.slopeAt(t) / L;
    const k = Math.sqrt(Math.max(0, 1 - rate * rate));
    return { centre: [0, 0, L * t], toTouchA: [k, 0, -rate], toTouchB: [0, k, -rate], dCentreDt: [0, 0, L], dCentreDr: [0, 0, 0] };
  };
}

test('the canal condition is enforced at exactly |dr/ds| = 1, and the closed form says where that is', () => {
  // A straight spine of length L with a linear taper has dr/ds = D/L exactly, so
  // the threshold is a number and not an estimate.
  for (const rate of [0.5, 0.9, 0.999]) {
    const p = radiusProfile([[0, 1], [1, 1 + rate * L]]);
    const f = profileFeasibility({ ballAt: pipeBall(p), profile: p });
    assert.equal(f.ok, true, `dr/ds = ${rate} must be buildable: ${f.reason}`);
    assert.ok(Math.abs(f.worstRate - rate) < 1e-9, `and the rate it reports must be the closed form (${f.worstRate} vs ${rate})`);
    assert.ok(Math.abs(f.headroom - (1 - rate)) < 1e-9, `${f.headroom}`);
  }
  for (const rate of [1.001, 1.5, 4]) {
    const p = radiusProfile([[0, 1], [1, 1 + rate * L]]);
    const ballAt = pipeBall(p);
    const f = profileFeasibility({ ballAt, profile: p });
    assert.equal(f.ok, false, `dr/ds = ${rate} has no envelope and must be refused`);
    assert.ok(Math.abs(f.worstRate - rate) < 1e-9, `${f.worstRate}`);
    assert.match(f.reason, /outruns/);
    assert.ok(f.excess > 0, `and it must say by how much (${f.excess})`);
    // ENFORCED, NOT REPORTED. The builder must refuse — not return a surface
    // with a residual attached to it.
    const blend = variableRadiusBlend({ ballAt, profile: p, tolerance: 0.01, evalSrf: surfacePoint });
    assert.equal(blend.ok, false, 'an infeasible profile must not come back with a surface');
    assert.equal(blend.srf, undefined, 'and must not come back with geometry at all');
    assert.match(blend.reason, /outruns/);
    assert.equal(blend.feasibility.ok, false);
  }
});

test('the refusal names WHERE along the edge it fails, not just that it does', () => {
  // Steep only between t = 0.2 and t = 0.25; everywhere else this profile is
  // gentle. A refusal that named the whole edge would be useless to a user.
  const p = radiusProfile([[0, 1], [0.2, 1.2], [0.25, 30], [1, 32]]);
  const f = profileFeasibility({ ballAt: pipeBall(p), profile: p });
  assert.equal(f.ok, false);
  assert.ok(f.worstAt.at > 0.2 && f.worstAt.at < 0.25,
    `the named span must be the steep one, not somewhere else on the edge (t = ${f.worstAt.at})`);
  assert.match(f.reason, /t = 0\.2[0-4]/);
  assert.ok(f.worstRate > 20, `and the rate must be reported (${f.worstRate})`);
  // The two independent readings of the same condition — fillet.mjs's margin
  // ds^2 - dr^2 and the scale-free rate — must never disagree about the verdict.
  assert.ok(f.worstMargin < 0, `the margin must agree it is infeasible (${f.worstMargin})`);
  // And the gentle profile on the same spine builds.
  const gentle = radiusProfile([[0, 2], [0.5, 8], [1, 2]]);
  const built = variableRadiusBlend({ ballAt: pipeBall(gentle), profile: gentle, tolerance: 1e-4, evalSrf: surfacePoint });
  assert.equal(built.ok, true, built.reason);
  assert.equal(built.metTolerance, true, `${built.deviation}`);
  assert.ok(built.feasibility.worstRate < 1, `${built.feasibility.worstRate}`);
});

test('the margin and the rate cross zero at the same taper — two readings of one condition', () => {
  // Swept across the threshold, the sign of fillet.mjs's ds^2 - dr^2 and the
  // side of 1 the rate falls on must agree at every step. They are algebraically
  // the same inequality, and a disagreement would mean one of them is not
  // reading the geometry it claims to.
  for (let i = 0; i <= 40; i++) {
    const rate = 0.8 + i * 0.01;
    const p = radiusProfile([[0, 1], [1, 1 + rate * L]]);
    const ballAt = pipeBall(p);
    const spine = [], radii = [];
    for (let j = 0; j < 65; j++) {
      const t = j / 64, r = p.radiusAt(t);
      spine.push(ballAt(t).centre); radii.push(r);
    }
    const direct = variableRadiusFeasible(spine, radii);
    const f = profileFeasibility({ ballAt, profile: p, samples: 65 });
    assert.equal(f.ok, direct.ok, `at rate ${rate.toFixed(3)} the two readings disagree`);
    assert.equal(f.ok, f.worstRate <= 1, `and the rate must decide the same way (${f.worstRate})`);
  }
});

test('a right-angled edge cannot violate the condition however steep the taper — and that is a closed form', () => {
  /* Two planes meeting at a right angle, material in the first quadrant, edge up
     +z. A ball of radius r sits at (r, r, z), so the center moves in BOTH faces'
     normal directions as the radius grows and the spine outruns the radius by
     sqrt(2) even in the limit:

         dr/ds = D / sqrt(2 D^2 + Lz^2)   ->   1/sqrt(2) as D -> infinity.

     This is why a real fillet never trips the canal condition, and why the check
     above had to be given a prescribed spine to have anything to refuse. It is
     asserted rather than left as an argument, because "the refusal is
     unreachable on real input" is exactly the belief that lets a check rot. */
  const Lz = 10;
  const cornerBall = (t, r) => ({ centre: [r, r, Lz * t], toTouchA: [-1, 0, 0], toTouchB: [0, -1, 0] });
  let worst = 0;
  for (const D of [1, 5, 20, 100, 1000, 1e5]) {
    const p = radiusProfile([[0, 1], [1, 1 + D]]);
    const f = profileFeasibility({ ballAt: cornerBall, profile: p });
    const closed = D / Math.hypot(D * Math.SQRT2, Lz);
    assert.equal(f.ok, true, `a taper of ${D} over ${Lz} must still be buildable: ${f.reason}`);
    assert.ok(Math.abs(f.worstRate - closed) < 1e-9, `rate must be D/sqrt(2D^2+Lz^2) (${f.worstRate} vs ${closed})`);
    assert.ok(f.worstRate < Math.SQRT1_2, `and stay under 1/sqrt(2) (${f.worstRate})`);
    worst = Math.max(worst, f.worstRate);
  }
  assert.ok(worst > 0.7, `the sweep must actually approach the bound or it proves nothing (${worst})`);
});

test('feasibility refuses a profile it cannot even sample, rather than throwing through its own contract', () => {
  const p = radiusProfile([[0, 1], [1, 2]]);
  assert.match(profileFeasibility({ ballAt: () => null, profile: p }).reason, /no ball center/);
  assert.match(profileFeasibility({ profile: p }).reason, /ball placement/);
  assert.match(profileFeasibility({ ballAt: pipeBall(p), profile: radiusProfile([[0, 1]]) }).reason, /not usable/);
  assert.match(variableRadiusBlend({ ballAt: pipeBall(p), profile: { ok: false, reason: 'x' }, evalSrf: surfacePoint }).reason, /not usable/);
  assert.match(variableRadiusBlend({ ballAt: pipeBall(p), profile: p }).reason, /evaluator/);
  assert.match(variableRadiusBlend({ profile: p, evalSrf: surfacePoint }).reason, /ball placement/);
});
