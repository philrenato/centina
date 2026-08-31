import test from 'node:test';
import assert from 'node:assert/strict';
import { makeCircle, makeArc } from '../kernel/primitives.mjs';
import { curvePoint, curvePointAndTangent, closestPointOnCurve } from '../kernel/curve.mjs';
import { dot, sub, normalize, length } from '../kernel/vec3.mjs';
import { surfacePoint, isFiniteNet } from '../kernel/surface.mjs';
import { buildParallelTransportFrames, sweepNProfiles } from '../kernel/sweep.mjs';

const line = (p0, p1) => ({ degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[...p0, 1], [...p1, 1]] });

// ---------------------------------------------------------------------
// closestPointOnCurve — the Q1 "profile-to-rail correspondence" machinery
// ---------------------------------------------------------------------

test('closestPointOnCurve on a straight rail finds the exact perpendicular-foot projection', () => {
  const rail = line([0, 0, 0], [0, 0, 20]);
  const hit = closestPointOnCurve(rail, [5, 0, 7]);
  assert.ok(Math.abs(hit.u - 0.35) < 1e-9, `u=${hit.u}`);
  assert.ok(Math.hypot(hit.point[0] - 0, hit.point[1] - 0, hit.point[2] - 7) < 1e-9);
  assert.ok(Math.abs(hit.distance - 5) < 1e-9);
  assert.equal(hit.ambiguous, false);
});

test('closestPointOnCurve on a curved rail satisfies the real closest-point condition ((C(u)-P) perpendicular to C\'(u)), an independent geometric check not a hardcoded number', () => {
  const rail = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 0, Math.PI / 2);
  const target = [15, 8, 3];
  const hit = closestPointOnCurve(rail, target);
  const { tangent } = curvePointAndTangent(rail, hit.u);
  const toTarget = sub(target, hit.point);
  // Interior optimum: (C(u)-P) . C'(u) = 0 (up to numerical tolerance).
  // At a domain boundary this can legitimately fail (a clamped endpoint),
  // so only assert it when the found u is genuinely interior.
  if (hit.u > rail.knots[0] + 1e-6 && hit.u < rail.knots[rail.knots.length - 1] - 1e-6) {
    assert.ok(Math.abs(dot(toTarget, tangent)) < 1e-6, `perpendicularity residual=${dot(toTarget, tangent)}`);
  }
  assert.equal(hit.ambiguous, false);
});

test('closestPointOnCurve is honestly AMBIGUOUS on a rail that loops back near itself (a hairpin/U-shaped rail), not a silent arbitrary pick', () => {
  // A degree-1 "U" rail: left leg down, across the bottom, right leg back
  // up — symmetric about x=0. A target point ON that symmetry plane, above
  // both leg-tops, is exactly equidistant from the two separate legs.
  const rail = {
    degree: 1,
    knots: [0, 0, 1, 2, 3, 3],
    ctrlPts: [[-5, 0, 10, 1], [-5, 0, 0, 1], [5, 0, 0, 1], [5, 0, 10, 1]],
  };
  const hit = closestPointOnCurve(rail, [0, 0, 12]);
  assert.equal(hit.ambiguous, true);
  assert.ok(hit.ambiguousWith !== null);
  // The two candidate stations should be genuinely far apart in parameter
  // (opposite legs), not two samples of the SAME dip.
  assert.ok(Math.abs(hit.u - hit.ambiguousWith) > 1, `u=${hit.u} ambiguousWith=${hit.ambiguousWith}`);
});

test('closestPointOnCurve is NOT ambiguous for an ordinary, non-self-proximate rail (no false positives)', () => {
  const rail = line([0, 0, 0], [0, 0, 20]);
  const hit = closestPointOnCurve(rail, [3, 0, 4]);
  assert.equal(hit.ambiguous, false);
});

// ---------------------------------------------------------------------
// buildParallelTransportFrames(rail, extraParams) — the Q2 extension
// ---------------------------------------------------------------------

test('buildParallelTransportFrames: extraParams=[] is byte-identical to the original single-argument call (regression safety for sweep1Rigid and Sweep1)', () => {
  const rail = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 0, 1.3 * Math.PI);
  const a = buildParallelTransportFrames(rail);
  const b = buildParallelTransportFrames(rail, []);
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.deepEqual(a[i], b[i]);
  }
  assert.deepEqual(b.extra, []);
});

test('buildParallelTransportFrames: an extra parameter that coincides with a rail control point\'s own Greville station reuses that EXACT frame, not a numerically-separate twin', () => {
  const rail = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 0, 1.3 * Math.PI);
  const gParams = (() => {
    // recompute independently via the frames themselves (frames[i].u is
    // exactly the Greville parameter used)
    const frames = buildParallelTransportFrames(rail);
    return frames.map((f) => f.u);
  })();
  const frames = buildParallelTransportFrames(rail, [gParams[2]]);
  assert.deepEqual(frames.extra[0], frames[2], 'extra frame at a coincident station must be the SAME frame as the control-point one');
});

test('buildParallelTransportFrames: a frame at a genuine INTERIOR (non-control-point) rail parameter is orthonormal and its tangent matches an independent curvePointAndTangent call', () => {
  const rail = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 10, 0, 1.3 * Math.PI);
  const uMid = (rail.knots[0] + rail.knots[rail.knots.length - 1]) / 2 + 0.017; // deliberately off any knot/Greville value
  const frames = buildParallelTransportFrames(rail, [uMid]);
  const f = frames.extra[0];
  assert.ok(Math.abs(length(f.xAxis) - 1) < 1e-9);
  assert.ok(Math.abs(length(f.yAxis) - 1) < 1e-9);
  assert.ok(Math.abs(length(f.zAxis) - 1) < 1e-9);
  assert.ok(Math.abs(dot(f.xAxis, f.yAxis)) < 1e-9);
  assert.ok(Math.abs(dot(f.xAxis, f.zAxis)) < 1e-9);
  assert.ok(Math.abs(dot(f.yAxis, f.zAxis)) < 1e-9);
  const { tangent } = curvePointAndTangent(rail, uMid);
  assert.ok(Math.hypot(f.zAxis[0] - tangent[0], f.zAxis[1] - tangent[1], f.zAxis[2] - tangent[2]) < 1e-9);
});

// ---------------------------------------------------------------------
// sweepNProfiles — Q3, the exactness proof
// ---------------------------------------------------------------------

test('sweepNProfiles: N=3 different-radius circles along a STRAIGHT rail — the cross-section AT each profile\'s own rail station matches that profile\'s own real circle exactly (ground-truth radius+plane check, not a tautological self-comparison)', () => {
  const rail = line([0, 0, 0], [0, 0, 20]);
  const p0 = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 3);
  const p1 = makeCircle([0, 0, 10], [1, 0, 0], [0, 1, 0], 5);
  const p2 = makeCircle([0, 0, 20], [1, 0, 0], [0, 1, 0], 8);
  const srf = sweepNProfiles(rail, [p0, p1, p2], 24);
  assert.equal(isFiniteNet(srf.ctrlNet), true);

  // Stations must have been inferred correctly: profile 0 at u=0 (z=0),
  // profile 1 at u=0.5 (z=10), profile 2 at u=1 (z=20) — and, since the
  // rail is straight, the TRUE arc-length fraction is exactly the same
  // 0/0.5/1 (this function's own real fix over loft()'s chord-of-grid
  // approximation).
  const vOf = (idx) => srf.stations.find((s) => s.profileIndex === idx).v;
  assert.ok(Math.abs(vOf(0) - 0) < 1e-9);
  assert.ok(Math.abs(vOf(1) - 0.5) < 1e-9);
  assert.ok(Math.abs(vOf(2) - 1) < 1e-9);

  // The exactness guarantee (global interpolation, applied twice) is that
  // surfacePoint(srf, ubar[i], v_k) reproduces grid[i][k] EXACTLY — grid[i]
  // being each profile's own resample at relative fraction i/(N-1) of ITS
  // OWN domain. This is checked AT the actual ubar[i] samples (not an
  // arbitrary continuous parameter in between, where a discrete-sample
  // reconstruction is only an approximation of the true circle — see
  // loft.test.mjs's own identical corner-vs-continuous distinction).
  const checks = [[p0, 3, 0, vOf(0)], [p1, 5, 10, vOf(1)], [p2, 8, 20, vOf(2)]];
  for (const [profile, radius, z, v] of checks) {
    for (const u of srf.ubar) {
      const pt = surfacePoint(srf, u, v);
      const r = Math.hypot(pt[0], pt[1]);
      assert.ok(Math.abs(r - radius) < 1e-7, `radius u=${u} v=${v}: got ${r} want ${radius}`);
      assert.ok(Math.abs(pt[2] - z) < 1e-7, `z u=${u} v=${v}: got ${pt[2]} want ${z}`);
    }
  }
});

test('sweepNProfiles: exactness also holds along a CURVED rail (not just the trivial straight case) — each station\'s cross-section matches an INDEPENDENT resample of that profile\'s own curve', () => {
  const rail = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 20, 0, Math.PI); // a semicircular rail, radius 20
  // Profiles positioned at three different stations along the rail, each
  // roughly perpendicular to the rail's own tangent there.
  const stationU = [rail.knots[0], (rail.knots[0] + rail.knots[rail.knots.length - 1]) / 2, rail.knots[rail.knots.length - 1]];
  const profiles = stationU.map((u, i) => {
    const { point, tangent } = curvePointAndTangent(rail, u);
    // build a small circle centered AT this rail point, roughly facing
    // the tangent (exact axis choice doesn't matter for this test — only
    // that the profile SITS at the right world location, since stationing
    // is by centroid-proximity)
    const arbitraryUp = Math.abs(tangent[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    let xAxis = normalize([
      arbitraryUp[1] * tangent[2] - arbitraryUp[2] * tangent[1],
      arbitraryUp[2] * tangent[0] - arbitraryUp[0] * tangent[2],
      arbitraryUp[0] * tangent[1] - arbitraryUp[1] * tangent[0],
    ]);
    const yAxis = normalize([
      tangent[1] * xAxis[2] - tangent[2] * xAxis[1],
      tangent[2] * xAxis[0] - tangent[0] * xAxis[2],
      tangent[0] * xAxis[1] - tangent[1] * xAxis[0],
    ]);
    return makeCircle(point, xAxis, yAxis, 2 + i);
  });
  const srf = sweepNProfiles(rail, profiles, 20);
  assert.equal(isFiniteNet(srf.ctrlNet), true);

  const uSampleCount = srf.ubar.length;
  for (let i = 0; i < 3; i++) {
    const v = srf.stations.find((s) => s.profileIndex === i).v;
    const profile = profiles[i];
    const p0u = profile.knots[0], p1u = profile.knots[profile.knots.length - 1];
    for (let k = 0; k < uSampleCount; k++) {
      const t = k / (uSampleCount - 1);
      const want = curvePoint(profile, p0u + t * (p1u - p0u)); // independent resample of the ORIGINAL profile
      const got = surfacePoint(srf, srf.ubar[k], v);
      assert.ok(Math.hypot(got[0] - want[0], got[1] - want[1], got[2] - want[2]) < 1e-7, `profile ${i} t=${t}: got ${got} want ${want}`);
    }
  }
});

test('sweepNProfiles: localized profiles (per-station frame decomposition) round-trip back to the ORIGINAL world control points exactly, at an arbitrary (non-control-point) rail station', () => {
  const rail = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 20, 0, Math.PI);
  const p0 = makeCircle(curvePoint(rail, rail.knots[0]), [1, 0, 0], [0, 0, 1], 2);
  const mid = (rail.knots[0] + rail.knots[rail.knots.length - 1]) / 2;
  const p1 = makeCircle(curvePoint(rail, mid), [1, 0, 0], [0, 0, 1], 3);
  const p2 = makeCircle(curvePoint(rail, rail.knots[rail.knots.length - 1]), [1, 0, 0], [0, 0, 1], 4);
  const srf = sweepNProfiles(rail, [p0, p1, p2], 16);
  for (let k = 0; k < 3; k++) {
    const frame = srf.frames[k];
    const local = srf.localizedProfiles[k];
    const original = [p0, p1, p2][k];
    for (let cp = 0; cp < local.ctrlPts.length; cp++) {
      const [lx, ly, lz, lw] = local.ctrlPts[cp];
      const world = [
        frame.origin[0] + lx * frame.xAxis[0] + ly * frame.yAxis[0] + lz * frame.zAxis[0],
        frame.origin[1] + lx * frame.xAxis[1] + ly * frame.yAxis[1] + lz * frame.zAxis[1],
        frame.origin[2] + lx * frame.xAxis[2] + ly * frame.yAxis[2] + lz * frame.zAxis[2],
      ];
      const orig = original.ctrlPts[cp];
      assert.ok(Math.hypot(world[0] - orig[0], world[1] - orig[1], world[2] - orig[2]) < 1e-9, `station ${k} ctrlPt ${cp}`);
      assert.ok(Math.abs(lw - orig[3]) < 1e-9);
    }
  }
});

test('sweepNProfiles throws honestly on fewer than 2 profiles', () => {
  const rail = line([0, 0, 0], [0, 0, 20]);
  const p0 = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 3);
  assert.throws(() => sweepNProfiles(rail, [p0]));
});

test('sweepNProfiles throws honestly when two profiles station at (nearly) the same rail parameter, rather than silently building a degenerate surface', () => {
  const rail = line([0, 0, 0], [0, 0, 20]);
  const p0 = makeCircle([0, 0, 10], [1, 0, 0], [0, 1, 0], 3);
  const p1 = makeCircle([0, 0, 10.0000001], [1, 0, 0], [0, 1, 0], 5);
  assert.throws(() => sweepNProfiles(rail, [p0, p1]), /same rail parameter|same rail station/i);
});

test('sweepNProfiles propagates closestPointOnCurve\'s own ambiguity as an honest, named error (not a silently-wrong surface) on a self-proximate rail', () => {
  const rail = {
    degree: 1,
    knots: [0, 0, 1, 2, 3, 3],
    ctrlPts: [[-5, 0, 10, 1], [-5, 0, 0, 1], [5, 0, 0, 1], [5, 0, 10, 1]],
  };
  // A 2-point line profile's centroid is an EXACT unweighted midpoint (no
  // rational-weight offset the way a circle's own control polygon has —
  // see this file's own "closestPointOnCurve on a straight rail" test and
  // sweep.mjs's own honest caveat that a profile's centroid is only a
  // representative anchor, not its true geometric centroid), so this
  // places p0's station EXACTLY on the hairpin's own symmetry plane.
  const p0 = { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[-1, 0, 12, 1], [1, 0, 12, 1]] };
  const p1 = makeCircle([-5, 0, 5], [1, 0, 0], [0, 1, 0], 1); // an ordinary, unambiguous station
  assert.throws(() => sweepNProfiles(rail, [p0, p1]), /AMBIGUOUS/);
});

test('sweepNProfiles (the curved-rail fix): a REAL curved rail\'s own true arc is what the surface follows between stations — NOT the straight chord between the two profiles. The regression: an N-profile Sweep1 "acted like a Loft" on a curved rail: a 90-degree, radius-100 arc rail with a small circle profile at each end should have its v=0.5 cross-section sit near the arc\'s own true midpoint (70.71, 70.71), NOT the straight chord\'s midpoint (50, 50).', () => {
  const rail = makeArc([0, 0, 0], [1, 0, 0], [0, 1, 0], 100, 0, Math.PI / 2);
  const startPt = [100, 0, 0];
  const endPt = [0, 100, 0];
  const p1 = makeCircle(startPt, [1, 0, 0], [0, 0, 1], 10);
  const p2 = makeCircle(endPt, [0, 1, 0], [0, 0, 1], 30);
  const srf = sweepNProfiles(rail, [p1, p2]);
  assert.equal(isFiniteNet(srf.ctrlNet), true);

  const trueArcMid = [70.710678, 70.710678, 0]; // radius 100 at 45 degrees
  const straightChordMid = [50, 50, 0]; // midpoint of the two profile centers — the OLD (buggy) behavior

  // "Spine" at v=0.5: average of the cross-section around the tube, the
  // same cheap, dependency-free proxy the interactive repro
  // uses (a true circle's own centroid, discretely sampled — not exact,
  // but more than precise enough to tell "hugging the arc" from "cutting
  // the chord" apart, which differ by ~28 units here).
  let sx = 0, sy = 0, sz = 0, k = 0;
  for (const u of [0, 0.25, 0.5, 0.75, 1]) {
    const p = surfacePoint(srf, u, 0.5);
    sx += p[0]; sy += p[1]; sz += p[2]; k++;
  }
  const spine = [sx / k, sy / k, sz / k];

  const distTo = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const distToArc = distTo(spine, trueArcMid);
  const distToChord = distTo(spine, straightChordMid);

  assert.ok(distToArc < 10, `surface v=0.5 spine ${spine} should sit within ~10 units of the rail's TRUE arc midpoint ${trueArcMid}, got distance ${distToArc}`);
  assert.ok(distToChord > 20, `surface v=0.5 spine ${spine} should be CLEARLY far from the straight-chord midpoint ${straightChordMid} (the old, buggy loft-like behavior), got distance ${distToChord}`);
  assert.ok(distToArc < distToChord / 3, `spine should track the rail's arc far more closely than the chord: distToArc=${distToArc} distToChord=${distToChord}`);
});

test('sweepNProfiles: profiles picked OUT OF RAIL ORDER are still correctly re-ordered by inferred station, not by pick order', () => {
  const rail = line([0, 0, 0], [0, 0, 20]);
  const pEnd = makeCircle([0, 0, 20], [1, 0, 0], [0, 1, 0], 8);
  const pStart = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 3);
  const pMid = makeCircle([0, 0, 10], [1, 0, 0], [0, 1, 0], 5);
  // picked in the order END, START, MID — deliberately scrambled
  const srf = sweepNProfiles(rail, [pEnd, pStart, pMid], 12);
  const stationsByV = [...srf.stations].sort((a, b) => a.v - b.v);
  assert.equal(stationsByV[0].profileIndex, 1, 'pStart (pick index 1) should be the FIRST station (smallest v)');
  assert.equal(stationsByV[1].profileIndex, 2, 'pMid (pick index 2) should be the MIDDLE station');
  assert.equal(stationsByV[2].profileIndex, 0, 'pEnd (pick index 0) should be the LAST station (largest v)');
});
