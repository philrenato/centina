// A MEASUREMENT THAT COULD NOT BE MADE MUST NOT BE REPORTED AS A MEASURED VALUE.
//
// Every case below is one bisection or one sampling march that can fail at every
// station. Each of them used to fall out of its loop with its accumulator still
// sitting at the value it was initialised to — 0 for a running maximum, 0 for a
// bisection's lower bound — and hand that back as a number somebody took. The
// caller then cannot tell "there is no room" from "I could not measure the room",
// and in the worst case builds degenerate geometry out of the difference.
//
// What each fix owes, and what these tests hold it to:
//   · an UNMEASURABLE case yields null, or a refusal by name — never 0;
//   · a MEASURABLE case is bit-for-bit what it was;
//   · and for Shell specifically, a solid whose safe thickness cannot be
//     measured REFUSES rather than emitting a zero-thickness shell.
import { test } from 'node:test';
import assert from 'node:assert';
import { shellSolid } from '../kernel/offset.mjs';
import { safeInwardOffset, shellRevolvedSolid, makeCylinderProfile } from '../kernel/offsetrevolve.mjs';
import { chamferFlatnessDeviation, smoothProfileDeviation, blendSurfaceToTolerance, rollingBallSection } from '../kernel/fillet.mjs';
import { filletPolygon, filletOpenPolyline } from '../kernel/primitives.mjs';
import { surfacePoint } from '../kernel/surface.mjs';

// ---------------------------------------------------------------------------
// fixtures — a box and a prism as flat bilinear panels, the exact shape the Box
// primitive builds, constructed HERE so no test grades the kernel with its own
// pencil.
// ---------------------------------------------------------------------------
function bilinearPanel(p00, p10, p01, p11) {
  return { srf: {
    degU: 1, knotsU: [0, 0, 1, 1], degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: [[[...p00, 1], [...p01, 1]], [[...p10, 1], [...p11, 1]]],
  } };
}
function boxPanels(hx, hy, hz) {
  const c = (sx, sy, sz) => [sx * hx, sy * hy, sz * hz];
  return [
    bilinearPanel(c(-1, -1, 1), c(1, -1, 1), c(-1, 1, 1), c(1, 1, 1)),
    bilinearPanel(c(-1, -1, -1), c(-1, 1, -1), c(1, -1, -1), c(1, 1, -1)),
    bilinearPanel(c(1, -1, -1), c(1, 1, -1), c(1, -1, 1), c(1, 1, 1)),
    bilinearPanel(c(-1, -1, -1), c(-1, -1, 1), c(-1, 1, -1), c(-1, 1, 1)),
    bilinearPanel(c(-1, 1, -1), c(-1, 1, 1), c(1, 1, -1), c(1, 1, 1)),
    bilinearPanel(c(-1, -1, -1), c(1, -1, -1), c(-1, -1, 1), c(1, -1, 1)),
  ];
}

// ---------------------------------------------------------------------------
// 1. shellSolid — THE ONE THAT DID NOT REFUSE.
//
// `shellWallFits` succeeds TRIVIALLY at t = 0 (the inner surface is the outer
// surface; nothing moved), so a bisection that failed everywhere left safeLo at
// its starting bound of 0, the clamp took the wall to 0, the confirming call got
// that trivial success back, and a zero-thickness shell was committed.
//
// The reachable version of that: a solid flat enough that its opposite faces WELD
// into one corner set. Two anti-parallel planes then meet at each corner, which
// `shellSolveInner` refuses as contradictory once their residual passes its 1e-6
// tolerance — so the bisection converges on 5e-7mm, an absolute constant that says
// nothing about the solid, and a 0.0000005mm "shell" was built and announced as
// "auto-clamped to 0.00mm".
// ---------------------------------------------------------------------------
test('shellSolid: a solid whose safe thickness cannot be measured REFUSES BY NAME instead of building a zero-thickness shell', () => {
  const flat = boxPanels(20, 20, 1e-9);   // 40 x 40 x 0.000000002 mm
  assert.throws(() => shellSolid(flat, [], 2), (e) => {
    assert.match(e.message, /^shellSolid: /, 'the refusal names the function');
    assert.match(e.message, /could not be measured/, 'and says the measurement failed, not that the answer is zero');
    assert.doesNotMatch(e.message, /computed as 0\.0000mm/, 'and never quotes a fabricated zero as the safe maximum');
    return true;
  });
});

test('shellSolid: an ordinary solid is completely unchanged — same wall, same panel count, same safe maximum', () => {
  const r = shellSolid(boxPanels(20, 20, 20), [], 2);
  assert.equal(r.appliedDistance, 2, 'a 2mm wall in a 40mm box is passed through untouched');
  assert.equal(r.clamped, false);
  assert.equal(r.panels.length, 12, '6 outer + 6 inner, no rim (nothing was opened)');
  assert.ok(Math.abs(r.safeMaxDistance - 20) < 1e-5,
    `the safe maximum on a half-extent-20 box is still 20mm (got ${r.safeMaxDistance})`);
});

test('shellSolid: a genuinely thin but genuinely measurable solid still shells — the refusal is not a blanket ban on thin walls', () => {
  const r = shellSolid(boxPanels(20, 20, 1e-3), [], 2);
  assert.ok(r.clamped, 'a 2mm wall does not fit a 0.002mm-thick plate, so it clamps');
  assert.ok(r.appliedDistance > 9e-4 && r.appliedDistance < 1e-3,
    `and clamps to the measured ceiling, ~0.001mm (got ${r.appliedDistance})`);
  assert.ok(r.appliedDistance > 0, 'and the wall it builds has real thickness');
});

test('shellSolid: no accepted result can carry a zero wall — the class of defect, stated as an invariant', () => {
  for (const hz of [20, 1, 1e-2, 1e-3]) {
    const r = shellSolid(boxPanels(20, 20, hz), [], 2);
    assert.ok(r.appliedDistance > 0, `hz=${hz}: a committed shell always has positive thickness`);
    assert.ok(r.safeMaxDistance > 0, `hz=${hz}: and a reported safe maximum is a real positive measurement`);
  }
});

// ---------------------------------------------------------------------------
// 2. safeInwardOffset / shellRevolvedSolid — the same bisection, surfaced as
//    "safe maximum computes as 0.0000mm".
// ---------------------------------------------------------------------------
const cylinder = (radius, height) =>
  makeCylinderProfile({ center: [0, 0, 0], axis: [0, 0, 1], refDir: [1, 0, 0], radius, height });

test('safeInwardOffset: a measurable profile is unchanged (a cylinder of radius R and height H still reports min(R, H/2))', () => {
  const safe = safeInwardOffset(cylinder(10, 40));
  assert.notEqual(safe, null, 'a real cylinder is measurable');
  assert.ok(Math.abs(safe - 10) < 1e-6, `a 10mm-radius, 40mm-tall cylinder carries a 10mm wall (got ${safe})`);
  assert.ok(Math.abs(safeInwardOffset(cylinder(30, 10)) - 5) < 1e-6, 'and a squat one is limited by its height, not its radius');
});

test('safeInwardOffset: an unmeasurable profile returns null, never 0 — "no thickness is known to fit", not "a wall of exactly zero fits"', () => {
  // A cylinder whose radius and height are both below the module's own 1e-6mm
  // collapsed-edge length: no offset it can take leaves a cavity distinguishable
  // from the profile itself.
  const safe = safeInwardOffset(cylinder(1e-7, 2e-7));
  assert.equal(safe, null, `an unmeasurable profile reports null (got ${safe})`);
  assert.notEqual(safe, 0, 'and specifically not 0, which a caller would clamp to and build');
});

test('shellRevolvedSolid: refuses by name on a profile whose safe maximum cannot be measured, rather than quoting 0.0000mm', () => {
  assert.throws(() => shellRevolvedSolid(cylinder(1e-7, 2e-7), 5e-8), (e) => {
    assert.match(e.message, /^shellRevolvedSolid: /);
    assert.match(e.message, /could not be measured/);
    assert.doesNotMatch(e.message, /computes as 0\.0000mm/, 'the fabricated zero is gone from the message too');
    return true;
  });
});

test('shellRevolvedSolid: an ordinary cylinder still shells, with the wall it was asked for', () => {
  const r = shellRevolvedSolid(cylinder(10, 40), 2);
  assert.equal(r.clamped, false, '2mm fits inside a 10mm-radius cylinder');
  assert.equal(r.appliedDistance, 2);
  assert.ok(r.safeMaxDistance > 9.9, `and its safe maximum is still the measured ~10mm (got ${r.safeMaxDistance})`);
});

// ---------------------------------------------------------------------------
// 3. chamferFlatnessDeviation / smoothProfileDeviation — THE INVERTED CASE.
//    An unmeasured surface used to certify as PERFECTLY FLAT (ok: true,
//    worst: 0) and `blendSurfaceToTolerance` stopped refining on it. Here the
//    honest answer is "unknown", and unknown must not read as perfect.
// ---------------------------------------------------------------------------
// A flat degree-1 patch, and an evaluator that refuses every station — the shape
// of a measure that cannot be taken, whatever the reason for it.
const FLAT = {
  degU: 1, knotsU: [0, 0, 1, 1], degV: 1, knotsV: [0, 0, 1, 1],
  ctrlNet: [[[0, 0, 0, 1], [0, 10, 0, 1]], [[10, 0, 0, 1], [10, 10, 0, 1]]],
};
const blind = () => null;

test('chamferFlatnessDeviation: a surface no station could be sampled on reports NOT ok — it does not certify as flat', () => {
  const d = chamferFlatnessDeviation(FLAT, blind);
  assert.equal(d.ok, false, 'unknown is not ok');
  assert.equal(d.worst, undefined, 'and it reports no deviation figure at all, rather than 0');
  assert.match(d.reason, /UNKNOWN rather than perfect/);
});

test('chamferFlatnessDeviation: a surface that CAN be sampled is unchanged — a flat patch measures flat, from real stations', () => {
  const d = chamferFlatnessDeviation(FLAT, surfacePoint);
  assert.equal(d.ok, true, d.reason);
  assert.ok(d.worst < 1e-12, `a genuinely flat patch is genuinely flat (got ${d.worst})`);
  assert.ok(d.stations > 0, 'and it says how many stations that verdict rests on');
});

test('smoothProfileDeviation: same rule — no measurable station means NOT ok, not a perfect score', () => {
  const d = smoothProfileDeviation(FLAT, blind);
  assert.equal(d.ok, false);
  assert.equal(d.worst, undefined);
  assert.match(d.reason, /UNKNOWN rather than zero/);
});

test('blendSurfaceToTolerance: an unmeasurable surface stops the refinement WITH THE MEASURE\'S OWN REASON, instead of being certified at the first attempt', () => {
  const radius = 4;
  const sectionAt = (t) => {
    const s = rollingBallSection({ point: [0, 0, t * 48], coNormalA: [0, 1, 0], coNormalB: [1, 0, 0], theta: Math.PI / 2, radius });
    return { centre: s.centre, radius, normalA: [-1, 0, 0], normalB: [0, -1, 0] };
  };
  const r = blendSurfaceToTolerance(sectionAt, 0.01, {
    evalSrf: surfacePoint,
    measure: (srf) => chamferFlatnessDeviation(srf, blind),
  });
  assert.equal(r.ok, false, 'a blend whose flatness is unknown is not a blend that met tolerance');
  assert.match(r.reason, /no station on this surface could be measured/,
    `the measure's own reason survives to the caller (got: ${r.reason})`);
});

// ---------------------------------------------------------------------------
// 4. filletPolygon / filletOpenPolyline — the user-facing MESSAGE lied. Both
//    other consumers were already guarded (`> 1e-6`, `> 0`), so the cost was a
//    status line offering "the largest radius these corners allow is 0.0000mm"
//    for a corner spacing that was never measurable.
// ---------------------------------------------------------------------------
const Z = [0, 0, 1];

test('filletPolygon: a too-large radius on a real polygon still reports a real, retriable maxSafeRadius', () => {
  const pts = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]];
  const tooLarge = filletPolygon(pts, 9, Z);
  assert.equal(tooLarge.ok, false);
  assert.ok(tooLarge.maxSafeRadius > 0 && tooLarge.maxSafeRadius < 9, 'a measurable ceiling is still a number');
  assert.equal(filletPolygon(pts, tooLarge.maxSafeRadius, Z).ok, true, 'and it succeeds when retried');
});

test('filletPolygon / filletOpenPolyline: the Infinity branch that used to answer 0 is UNREACHABLE — normalize refuses a sub-1e-12 edge first, so the fabricated 0 was dead code, and it is now null in any case', () => {
  // Both refusals compute `needed / edgeLen`, so only an edge of length 0 (or
  // under 1e-12, which is `normalize`'s own floor) sends worstRatio to Infinity.
  // But every such edge is also one whose direction each adjacent corner has to
  // normalize BEFORE any ratio is taken, and that throws. Recorded as a test so
  // the claim is checked rather than asserted in a comment: if a future change
  // makes the branch live, the `null` is already the right answer and this test
  // is the thing that notices the reachability changed.
  const dup = [[0, 0, 0], [10, 0, 0], [10, 0, 0], [10, 10, 0]];
  assert.throws(() => filletPolygon(dup, 3, Z), /zero-length vector/);
  assert.throws(() => filletOpenPolyline(dup, 3, { closed: false }), /zero-length vector/);
});

test('no fillet refusal ever offers 0 as a retriable radius — the invariant the message depended on', () => {
  const rail = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [20, 10, 0]];
  for (const r of [50, 20, 9, 6]) {
    const open = filletOpenPolyline(rail, r, { closed: false });
    if (!open.ok && 'maxSafeRadius' in open) {
      assert.notEqual(open.maxSafeRadius, 0, `r=${r}: an open rail never offers 0mm as the radius to retry with`);
    }
    const poly = filletPolygon([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], r, Z);
    if (!poly.ok && 'maxSafeRadius' in poly) {
      assert.notEqual(poly.maxSafeRadius, 0, `r=${r}: neither does a closed polygon`);
    }
  }
});
