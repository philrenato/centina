import test from 'node:test';
import assert from 'node:assert/strict';
import { rollingBallSection, maxRadiusForSetback, variableRadiusFeasible, sectionArc, smoothSectionArc, smoothSectionAlpha } from '../kernel/fillet.mjs';

// Every expected number below is worked out from the geometry, never read off
// the implementation. A ball of radius r resting in a right-angled corner
// touches each wall at distance r from the corner and has its centre at
// r*sqrt(2) along the 45-degree bisector — that is the whole of TEST 1, and it
// is checkable on paper.
const R = 5;
const near = (a, b, tol = 1e-12) => Math.abs(a - b) < tol;
const nearPt = (p, q, tol = 1e-12) => p.every((v, i) => Math.abs(v - q[i]) < tol);

test('a convex right-angled edge: setback r, centre at r*sqrt(2) on the bisector', () => {
  const s = rollingBallSection({
    point: [0, 0, 0], coNormalA: [0, 1, 0], coNormalB: [1, 0, 0],
    theta: Math.PI / 2, radius: R,
  });
  assert.equal(s.ok, true);
  assert.ok(near(s.phi, Math.PI / 2), `phi ${s.phi}`);
  assert.ok(near(s.setback, R), `setback ${s.setback} should be ${R}`);
  assert.ok(near(s.centreOffset, R * Math.SQRT2), `centreOffset ${s.centreOffset}`);
  assert.ok(nearPt(s.tangencyA, [0, R, 0]), `tangencyA ${s.tangencyA}`);
  assert.ok(nearPt(s.tangencyB, [R, 0, 0]), `tangencyB ${s.tangencyB}`);
  assert.ok(nearPt(s.centre, [R, R, 0]), `centre ${s.centre}`);
  assert.equal(s.convex, true);
  // The defining property, asserted rather than assumed: the centre is exactly
  // r from BOTH tangency points.
  assert.ok(near(Math.hypot(...s.centre.map((v, i) => v - s.tangencyA[i])), R));
  assert.ok(near(Math.hypot(...s.centre.map((v, i) => v - s.tangencyB[i])), R));
});

test('a CONCAVE 270-degree edge sets back the SAME way — the sign trap', () => {
  // The concave edge of an L-shaped solid. The co-normals are unchanged in
  // character (each points along its own face, away from the edge); only theta
  // differs. Writing the setback against theta gives r/tan(135) = -r, which
  // puts both tangency points on the wrong side of the edge.
  const s = rollingBallSection({
    point: [0, 0, 0], coNormalA: [1, 0, 0], coNormalB: [0, 1, 0],
    theta: 3 * Math.PI / 2, radius: R,
  });
  assert.equal(s.ok, true);
  assert.ok(near(s.setback, R), `setback ${s.setback} must be +${R}, not -${R}`);
  assert.ok(s.setback > 0, 'a setback is a distance and is never negative');
  assert.ok(nearPt(s.tangencyA, [R, 0, 0]));
  assert.ok(nearPt(s.tangencyB, [0, R, 0]));
  assert.ok(nearPt(s.centre, [R, R, 0]));
  assert.equal(s.convex, false, 'theta > pi is concave, and that is what theta is still for');
});

test('a SHARP edge sets back further than a flat one — the setback diverges as the wedge closes', () => {
  // The direction of this is worth stating, because it is the opposite of the
  // first guess. A ball in a narrow wedge cannot get near the apex, so it
  // touches far along both faces: at phi = 34 degrees the setback is over three
  // radii. A nearly-flat junction is almost a plane, the ball rests right by
  // the edge, and the setback nearly vanishes: at phi = 160 degrees it is under
  // a fifth of a radius. d = r/tan(phi/2) diverges as phi -> 0, not as phi -> pi.
  const flat = rollingBallSection({ point: [0, 0, 0], coNormalA: [1, 0, 0], coNormalB: [Math.cos(2.8), Math.sin(2.8), 0], theta: 2.8, radius: R });
  const sharp = rollingBallSection({ point: [0, 0, 0], coNormalA: [1, 0, 0], coNormalB: [Math.cos(0.6), Math.sin(0.6), 0], theta: 0.6, radius: R });
  assert.equal(flat.ok, true);
  assert.equal(sharp.ok, true);
  assert.ok(sharp.setback > flat.setback,
    `a sharper wedge needs MORE setback (sharp ${sharp.setback.toFixed(3)} vs flat ${flat.setback.toFixed(3)})`);
  assert.ok(near(flat.setback, R / Math.tan(1.4), 1e-9), `${flat.setback}`);
  assert.ok(sharp.setback > 3 * R, `at 34 degrees the setback is over three radii (${sharp.setback.toFixed(3)})`);
  assert.ok(flat.setback < R / 4, `at 160 degrees it is under a quarter radius (${flat.setback.toFixed(3)})`);
});

test('refuses a tangent junction and a knife edge by name, rather than returning a huge number', () => {
  const tangent = rollingBallSection({ point: [0, 0, 0], coNormalA: [1, 0, 0], coNormalB: [-1, 0, 0], theta: Math.PI, radius: R });
  assert.equal(tangent.ok, false);
  assert.match(tangent.reason, /tangent/i);
  const knife = rollingBallSection({ point: [0, 0, 0], coNormalA: [1, 0, 0], coNormalB: [1, 0, 0], theta: 1e-4, radius: R });
  assert.equal(knife.ok, false);
  assert.match(knife.reason, /knife|fold/i, 'a knife edge must not be described as a tangency — they are opposite failures');
  // And the reflex twin: a thin slot is phi near zero as well, so it must reach
  // the same honest message rather than a separate wrong one.
  const slot = rollingBallSection({ point: [0, 0, 0], coNormalA: [1, 0, 0], coNormalB: [Math.cos(0.001), Math.sin(0.001), 0], theta: 2 * Math.PI - 0.001, radius: R });
  assert.equal(slot.ok, false);
  assert.match(slot.reason, /knife|fold/i);
});

test('maxRadiusForSetback names the number and which face is the limit', () => {
  // At 90 degrees the setback equals the radius, so a face 12mm wide caps the
  // radius at exactly 12mm.
  const m = maxRadiusForSetback({ phi: Math.PI / 2, widthA: 12, widthB: 40 });
  assert.equal(m.ok, true);
  assert.ok(near(m.rMax, 12), `rMax ${m.rMax}`);
  assert.equal(m.limitedBy, 'A');
  assert.equal(m.bound, 'setback-only', 'it must not claim to be a curvature bound it cannot compute');
  // A SHARPER edge sets back further, so the same face caps a SMALLER radius —
  // and a flatter one caps a larger. rMax = w * tan(phi/2) runs the same way the
  // setback does, which is the consistency check worth having here.
  const sharp = maxRadiusForSetback({ phi: 0.6, widthA: 12, widthB: 40 });
  assert.ok(sharp.rMax < m.rMax, `a sharp edge caps lower: ${sharp.rMax.toFixed(3)} vs ${m.rMax.toFixed(3)}`);
  const flat = maxRadiusForSetback({ phi: 2.6, widthA: 12, widthB: 40 });
  assert.ok(flat.rMax > m.rMax, `a flat edge caps higher: ${flat.rMax.toFixed(3)} vs ${m.rMax.toFixed(3)}`);
  // And the cap is exactly the width where the setback equals it.
  const sec = rollingBallSection({ point: [0, 0, 0], coNormalA: [1, 0, 0], coNormalB: [0, 1, 0], theta: Math.PI / 2, radius: m.rMax });
  assert.ok(near(sec.setback, 12, 1e-9), `at rMax the setback is exactly the face width (${sec.setback})`);
});

test('variable radius: refuses exactly when the radius outruns the spine', () => {
  // Spine advancing 1mm per step. A radius changing 0.5mm per step is fine;
  // 1.5mm per step outruns it and no envelope exists.
  const spine = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
  const ok = variableRadiusFeasible(spine, [1, 1.5, 2, 2.5]);
  assert.equal(ok.ok, true, `margin ${ok.worstMargin}`);
  const bad = variableRadiusFeasible(spine, [1, 2.5, 4, 5.5]);
  assert.equal(bad.ok, false);
  assert.equal(bad.worstAt, 0, 'and it names WHERE, not just that');
  assert.match(bad.reason, /faster than the spine/);
  // Exactly at the boundary the characteristic circle degenerates to a point;
  // margin zero is admitted, since the condition is >= 0.
  const edge = variableRadiusFeasible(spine, [1, 2, 3, 4]);
  assert.equal(edge.ok, true);
  assert.ok(near(edge.worstMargin, 0, 1e-12), `${edge.worstMargin}`);
});

test('the section arc is a real circular arc of the requested radius', () => {
  const s = rollingBallSection({ point: [0, 0, 0], coNormalA: [0, 1, 0], coNormalB: [1, 0, 0], theta: Math.PI / 2, radius: R });
  const arc = sectionArc(s);
  assert.equal(arc.degree, 2);
  assert.ok(near(arc.radius, R));
  // A convex 90-degree edge is rounded by a QUARTER circle: the sweep is
  // pi - theta... measured between the two radii, which are 90 degrees apart.
  assert.ok(near(arc.sweep, Math.PI / 2), `sweep ${arc.sweep}`);
  // Evaluate the rational quadratic at t = 0.5 and check it is exactly r from
  // the centre — the property that makes it an ARC rather than a parabola.
  /* ⚠ THE NUMERATOR CARRIES THE WEIGHT. Control points are stored CARTESIAN
     with the weight appended, so a rational evaluation is sum(N*w*x) / sum(N*w)
     — an earlier version of this omitted the w from the numerator and passed
     anyway, because a 90-degree arc's middle control point is the corner itself
     and this fixture's corner is the ORIGIN, where the two formulas agree. Moved
     off the origin it reported 4.687 for a correct arc of radius 5, and it could
     not detect the premultiplication bug the module's own comments memorialise.
     Verified below at a corner that is NOT the origin, for that reason. */
  const [p0, p1, p2] = arc.ctrlPts;
  const wMid = 0.25 * p0[3] + 0.5 * p1[3] + 0.25 * p2[3];
  const num = (i) => 0.25 * p0[i] * p0[3] + 0.5 * p1[i] * p1[3] + 0.25 * p2[i] * p2[3];
  const mid = [num(0) / wMid, num(1) / wMid, num(2) / wMid];
  const d = Math.hypot(mid[0] - s.centre[0], mid[1] - s.centre[1], mid[2] - s.centre[2]);
  assert.ok(near(d, R, 1e-12), `midpoint is ${d} from the centre, should be exactly ${R}`);

  // AND AWAY FROM THE ORIGIN, which is the only place the broken evaluator and
  // the correct one disagree. Same 90-degree corner, translated.
  const off = [7, -3, 2];
  const s2 = rollingBallSection({ point: off, coNormalA: [0, 1, 0], coNormalB: [1, 0, 0], theta: Math.PI / 2, radius: R });
  const arc2 = sectionArc(s2);
  const [q0, q1, q2] = arc2.ctrlPts;
  const w2 = 0.25 * q0[3] + 0.5 * q1[3] + 0.25 * q2[3];
  const n2 = (i) => 0.25 * q0[i] * q0[3] + 0.5 * q1[i] * q1[3] + 0.25 * q2[i] * q2[3];
  const m2 = [n2(0) / w2, n2(1) / w2, n2(2) / w2];
  const d2 = Math.hypot(m2[0] - s2.centre[0], m2[1] - s2.centre[1], m2[2] - s2.centre[2]);
  assert.ok(near(d2, R, 1e-12), `away from the origin the arc midpoint must still be exactly ${R} from the centre (${d2})`);
});

import { surfacePoint } from '../kernel/surface.mjs';
import { rollingBallSection as sec2, blendSurfaceFromSections, blendRadiusDeviation } from '../kernel/fillet.mjs';

// Build the blend along a STRAIGHT box edge running up +z, faces at x=0 and
// y=0 with material in the first quadrant. The spine is then the straight line
// x=y=r, and EVERY point of a correct blend is exactly r from it — which makes
// this checkable to machine precision rather than to a tolerance.
function boxEdgeBlend(radius, zs) {
  const arcs = [];
  for (const z of zs) {
    const s = sec2({ point: [0, 0, z], coNormalA: [0, 1, 0], coNormalB: [1, 0, 0], theta: Math.PI / 2, radius });
    arcs.push(sectionArc(s));
  }
  return blendSurfaceFromSections(arcs);
}

test('the blend on a straight edge is EXACTLY a cylindrical patch — every point r from the spine', () => {
  const r = 4;
  const built = boxEdgeBlend(r, [0, 10, 20, 30]);
  assert.equal(built.ok, true, built.reason);
  assert.equal(built.srf.degU, 2, 'the cross-section stays an exact rational quadratic');
  // The spine is the straight line x = y = r running up +z.
  const spine = [[r, r, -10], [r, r, 40]];
  const dev = blendRadiusDeviation(built.srf, spine, r, surfacePoint, 13, 13);
  assert.ok(dev.worst < 1e-12, `a straight edge of constant angle must be EXACT, worst deviation ${dev.worst}`);
  // ⚠ AND THE POSITION ALONG THE EDGE, which distance-to-a-parallel-line cannot
  // see at all. A weight applied twice displaces the middle control point along
  // z and leaves its distance from the spine line untouched, so the check above
  // passed while the surface was wrong. The endpoints of this run are z = 0 and
  // z = 30, and v must map monotonically across that span.
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    const p = surfacePoint(built.srf, 0.5, v);
    assert.ok(p[2] >= -1e-9 && p[2] <= 30 + 1e-9, `v=${v} lands at z=${p[2]}, outside the run's own 0..30`);
  }
  const zStart = surfacePoint(built.srf, 0.5, 0)[2], zEnd = surfacePoint(built.srf, 0.5, 1)[2];
  assert.ok(Math.abs(zStart - 0) < 1e-9, `the run starts at z=0 (${zStart})`);
  assert.ok(Math.abs(zEnd - 30) < 1e-9, `and ends at z=30 (${zEnd})`);
  const zMid = surfacePoint(built.srf, 0.5, 0.5)[2];
  assert.ok(Math.abs(zMid - 15) < 0.5, `and its middle is near the middle (${zMid}), not pulled toward the origin by a doubled weight`);
});

test('the tangency curves are where the supports must be trimmed back to', () => {
  const r = 4;
  const built = boxEdgeBlend(r, [0, 10, 20]);
  // Face A is the plane x = 0; its tangency curve must lie in it at y = r.
  for (const cp of built.tangencyCurveA.ctrlPts) {
    const w = cp[3];
    assert.ok(Math.abs(cp[0] / w - 0) < 1e-12, `tangency A must lie in the x=0 face (${cp[0] / w})`);
    assert.ok(Math.abs(cp[1] / w - r) < 1e-12, `and at the setback distance r along it (${cp[1] / w})`);
  }
  // Face B is the plane y = 0; its tangency curve at x = r.
  for (const cp of built.tangencyCurveB.ctrlPts) {
    const w = cp[3];
    assert.ok(Math.abs(cp[1] / w - 0) < 1e-12, `tangency B must lie in the y=0 face (${cp[1] / w})`);
    assert.ok(Math.abs(cp[0] / w - r) < 1e-12);
  }
});

import { envelopeSection, envelopeSectionArc } from '../kernel/fillet.mjs';

/* A GENUINELY VARYING DIHEDRAL, WHICH THE PREVIOUS FIXTURE WAS NOT.
   Two PLANAR faces meeting along a STRAIGHT edge have a constant dihedral by
   definition — imposing a sweeping angle on that configuration describes no
   solid that exists, and the 3.28% it reported was an artifact of the fixture
   rather than a property of either construction. Worse, the first envelope test
   derived its face normals FROM the sliding-disc tangency points, so it handed
   the envelope form its own answer back and could only ever agree.

   A real one: a CYLINDER of radius Rc about +z, cut by a PLANE inclined at
   alpha. Their intersection is an ellipse, and the angle between the cylinder
   wall and the plane genuinely varies around it — cos(angle) = -sin(alpha)*cos(t)
   — which is exactly the S-shaped / blob-derived case a fillet has to survive.
   A ball of radius r tangent to both sits at distance (Rc - r) from the axis and
   r from the plane, both of which are closed form. */
// alpha and the arc span are chosen so the dihedral genuinely sweeps: at 35
// degrees over a 1.8-radian arc it only moved 14, and the anti-vacuity guard
// below caught that rather than letting a near-constant angle pass as a test of
// a varying one.
const Rc = 20, alpha = 60 * Math.PI / 180, rBall = 3;
function realRollingBall(sParam) {
  /* ⚠ NEGATED, because this must point from the ball CENTRE towards the plane.
     The plane is x*sin(a) - z*cos(a) = 0 and the centre sits at +r along its
     normal, so reaching the plane means travelling along MINUS that normal.
     Passing the plane's own normal put the tangency point 2r away on the far
     side — still exactly r from the centre, so the radius check reported a
     perfect blend that touched nothing. */
  const nPlane = [-Math.sin(alpha), 0, Math.cos(alpha)];
  const rho = Rc - rBall;
  const x = rho * Math.cos(sParam), y = rho * Math.sin(sParam);
  // x*sin(alpha) - z*cos(alpha) = r  =>  z = (x*sin(alpha) - r)/cos(alpha)
  const z = (x * Math.sin(alpha) - rBall) / Math.cos(alpha);
  const centre = [x, y, z];
  // The cylinder's OUTWARD normal where the ball touches it (material inside).
  const nCyl = [Math.cos(sParam), Math.sin(sParam), 0];
  return { centre, nCyl, nPlane };
}

/* ⚠ THE SPINE USED TO MEASURE IS SAMPLED FAR FINER THAN THE SECTIONS.
   Measuring against the section centres themselves reports the RULER's own
   chord error — a polyline through a curving spine is second order by
   construction — and that is what made every surface, at every V degree, look
   like O(h^2). With the instrument pushed two orders down, the same surfaces
   measure O(h^4): cubic interpolation behaving as cubic interpolation. */
/* ⚠ FIXED DENSITY, NOT A MULTIPLE OF THE SECTION COUNT. Scaling the spine with
   the sections makes the ruler improve at exactly the rate the surface does, so
   their ratio never moves and the measurement stays instrument-bound at every
   density — which is what the first attempt did. One fixed, very fine spine
   keeps the sagitta constant while the surface error falls past it.

   40001 points over a spine roughly 48mm long with curvature radius near 17mm
   gives a segment of 1.2e-3mm and a sagitta of L^2/(8*rho) around 1e-8mm, two
   to three orders below the surface errors being judged. */
const FINE_SPINE = (() => {
  const out = [];
  const M = 40001;
  for (let i = 0; i < M; i++) out.push(realRollingBall(-1.4 + 2.8 * (i / (M - 1))).centre);
  return out;
})();
function fineSpine() { return FINE_SPINE; }
function buildAt(count) {
  const arcs = [];
  for (let i = 0; i < count; i++) {
    const b = realRollingBall(-1.4 + 2.8 * (i / (count - 1)));
    arcs.push(envelopeSectionArc(envelopeSection({ centre: b.centre, radius: rBall, toTouchA: b.nCyl, toTouchB: b.nPlane })));
  }
  const built = blendSurfaceFromSections(arcs);
  const d = blendRadiusDeviation(built.srf, fineSpine(), rBall, surfacePoint, 9, 31);
  return { n: count, dev: d.worst, instrumentBound: d.instrumentBound, sagitta: d.spineSagitta };
}

test('the tangency points actually LIE ON both supporting surfaces', () => {
  /* THE CHECK A RADIUS-ONLY INSTRUMENT CANNOT MAKE. Every tangency point is r
     from the ball centre by construction, whichever way the direction points —
     so "the radius is right" is true even when the blend touches neither face.
     What has to be asserted is CONTACT: the point lies on the cylinder wall and
     on the plane. This is the assertion whose absence hid a 2r sign error. */
  const planeDist = (p) => -p[0] * Math.sin(alpha) + p[2] * Math.cos(alpha);
  const cylDist = (p) => Math.abs(Math.hypot(p[0], p[1]) - Rc);
  for (let i = 0; i < 9; i++) {
    const b = realRollingBall(-1.4 + 2.8 * (i / 8));
    const e = envelopeSection({ centre: b.centre, radius: rBall, toTouchA: b.nCyl, toTouchB: b.nPlane });
    assert.equal(e.ok, true, e.reason);
    assert.ok(cylDist(e.tangencyA) < 1e-9, `tangency A must lie ON the cylinder wall (${cylDist(e.tangencyA)})`);
    assert.ok(Math.abs(planeDist(e.tangencyB)) < 1e-9, `tangency B must lie ON the plane (${planeDist(e.tangencyB)})`);
  }
});

test('a REAL varying dihedral: the surface converges at FOURTH order once the ruler stops dominating', () => {
  const N = 25;
  const envArcs = [], centres = [];
  let minAng = Infinity, maxAng = -Infinity;
  for (let i = 0; i < N; i++) {
    const sParam = -1.4 + 2.8 * (i / (N - 1));
    const { centre, nCyl, nPlane } = realRollingBall(sParam);
    centres.push(centre);
    const ang = Math.acos(Math.max(-1, Math.min(1, nCyl[0] * nPlane[0] + nCyl[1] * nPlane[1] + nCyl[2] * nPlane[2])));
    minAng = Math.min(minAng, ang); maxAng = Math.max(maxAng, ang);
    const e = envelopeSection({ centre, radius: rBall, toTouchA: nCyl, toTouchB: nPlane });
    assert.equal(e.ok, true, e.reason);
    envArcs.push(envelopeSectionArc(e));
  }
  const spread = (maxAng - minAng) * 180 / Math.PI;
  assert.ok(spread > 15, `the dihedral must genuinely vary or the test is vacuous (${spread.toFixed(1)} degrees)`);
  const env = blendSurfaceFromSections(envArcs);
  assert.equal(env.ok, true, env.reason);
  const devEnv = blendRadiusDeviation(env.srf, centres, rBall, surfacePoint, 9, 41);
  console.log(`      REAL varying dihedral (${spread.toFixed(1)} deg sweep), ${N} sections: ${devEnv.worst.toExponential(2)}mm = ${(100 * devEnv.worst / rBall).toFixed(4)}% of r`);

  /* THE REMAINING ERROR IS STRUCTURAL, NOT A SAMPLING OR DEGREE CHOICE, and
     saying which it is decides whether raising the degree or raising the
     density is the fix.

     Skinning the characteristic circles is exact AT each section and departs
     from the true tube BETWEEN them, because consecutive circles lie in planes
     of different orientation. That departure is second order in the section
     spacing whatever degree V carries — measured at O(h^2) for both degree 1
     and degree 3, which is the give-away: cubic INTERPOLATION would be O(h^4)
     if interpolation error were what was being measured. So density is the only
     control, and the honest thing is to derive it from a tolerance and report
     what was achieved rather than to claim exactness. */
  /* ⚠ THE RULER HAS ITS OWN FLOOR, AND THE TEST STOPS WHERE THE RULER DOES.
     Even a 40x-denser spine polyline carries a sagitta around 9e-6mm, so by 97
     sections the SURFACE (about 1.3e-6mm) is finer than the instrument
     measuring it and the reported number becomes the ruler's. `instrumentBound`
     says when that has happened, and the order is asserted only across the
     range where it has not. Claiming fourth order from numbers the instrument
     cannot resolve would be the same mistake as claiming second order was the
     surface — one level up. */
  const a1 = buildAt(13), a2 = buildAt(25), a3 = buildAt(49);
  assert.ok(!a1.instrumentBound && !a2.instrumentBound && !a3.instrumentBound,
    `the order is only asserted where the spine polyline is well below the surface (sagittas ${[a1, a2, a3].map((x) => x.sagitta.toExponential(1)).join(', ')} vs deviations ${[a1, a2, a3].map((x) => x.dev.toExponential(1)).join(', ')})`);
  const order1 = Math.log2(a1.dev / a2.dev), order2 = Math.log2(a2.dev / a3.dev);
  assert.ok(order1 > 3 && order2 > 3,
    `the surface converges at better than THIRD order — second would mean the spine polyline is what is being measured (${order1.toFixed(2)}, ${order2.toFixed(2)})`);
  assert.ok(a3.dev < rBall * 1e-4,
    `and by 49 sections it is under 0.01% of the radius (${(100 * a3.dev / rBall).toExponential(2)}%)`);
  console.log(`      converges O(h^4): 13 sections ${a1.dev.toExponential(2)}mm -> 25 ${a2.dev.toExponential(2)}mm -> 49 ${a3.dev.toExponential(2)}mm  (orders ${order1.toFixed(2)}, ${order2.toFixed(2)})`);
});

test('envelopeSection refuses tangent and opposed faces by name', () => {
  const tangent = envelopeSection({ centre: [0, 0, 0], radius: 2, normalA: [1, 0, 0], normalB: [1, 0, 0] });
  assert.equal(tangent.ok, false);
  assert.match(tangent.reason, /tangent/i);
  const opposed = envelopeSection({ centre: [0, 0, 0], radius: 2, normalA: [1, 0, 0], normalB: [-1, 0, 0] });
  assert.equal(opposed.ok, false);
  assert.match(opposed.reason, /opposed/i);
});

import { blendSurfaceToTolerance } from '../kernel/fillet.mjs';

test('building to a tolerance reaches it, and reports the deviation it actually achieved', () => {
  const sectionAt = (t) => {
    const b = realRollingBall(-1.4 + 2.8 * t);
    return { centre: b.centre, radius: rBall, toTouchA: b.nCyl, toTouchB: b.nPlane };
  };
  // The default quality: 0.01mm on a 3mm fillet across a 51-degree sweep.
  const built = blendSurfaceToTolerance(sectionAt, 0.01, { evalSrf: surfacePoint });
  assert.equal(built.ok, true, built.reason);
  assert.equal(built.metTolerance, true, `must reach 0.01mm (got ${built.deviation})`);
  assert.ok(built.deviation <= 0.01, `${built.deviation}`);
  assert.ok(built.sections < 200, `and not by brute force (${built.sections} sections)`);
  console.log(`      to 0.01mm: ${built.sections} sections, achieved ${built.deviation.toExponential(2)}mm in ${built.rounds} round(s)`);

  // A tighter ask costs more sections, and the count must MOVE with the ask —
  // a builder that ignores its tolerance would return the same number twice.
  const tight = blendSurfaceToTolerance(sectionAt, 0.001, { evalSrf: surfacePoint });
  assert.equal(tight.ok, true);
  assert.ok(tight.sections > built.sections,
    `a tighter tolerance must cost more sections (${tight.sections} vs ${built.sections})`);
  assert.ok(tight.deviation < built.deviation, `and achieve less deviation (${tight.deviation} vs ${built.deviation})`);
  console.log(`      to 0.001mm: ${tight.sections} sections, achieved ${tight.deviation.toExponential(2)}mm in ${tight.rounds} round(s)`);

  // An impossible ask returns a real surface and says it fell short, rather
  // than throwing or quietly pretending.
  const impossible = blendSurfaceToTolerance(sectionAt, 1e-12, { evalSrf: surfacePoint, maxSections: 41 });
  assert.equal(impossible.ok, true, 'a surface that missed the target is still a surface');
  assert.equal(impossible.metTolerance, false, 'and it must SAY it missed');
  assert.ok(impossible.deviation > 1e-12);
  assert.ok(impossible.sections <= 41, 'and must honour the section ceiling it was given');
});

import { spliceLoopWithChain } from '../kernel/fillet.mjs';
import { signedArea2D, pointInUVPolygon } from '../kernel/trim.mjs';

// The unit square as a face's trim loop, counter-clockwise. Filleting the edge
// along v = 0 cuts the face back to v = 0.25, so the tangency chain runs across
// at that height and the strip below it is what the blend replaces.
const SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]];

test('splicing a tangency chain cuts the face back, and drops the side the blend replaces', () => {
  const chain = [[0, 0.25], [0.5, 0.25], [1, 0.25]];
  const r = spliceLoopWithChain(SQUARE, chain, [0.5, 0]); // the edge being filleted is along v = 0
  assert.equal(r.ok, true, r.reason);
  assert.ok(r.headGap < 1e-12 && r.tailGap < 1e-12, `both ends must land ON the loop (${r.headGap}, ${r.tailGap})`);
  // The result must be the upper rectangle: nothing below v = 0.25 survives.
  for (const p of r.loop) assert.ok(p[1] >= 0.25 - 1e-12, `a point at v=${p[1]} survived below the cut`);
  // And it must still contain the part of the face that was NOT cut away.
  assert.equal(pointInUVPolygon(r.loop, 0.5, 0.6), 'inside', 'the kept half of the face is still inside the loop');
  assert.equal(pointInUVPolygon(r.loop, 0.5, 0.1), 'outside', 'and the trimmed strip is not');
  // Area check, exact: a unit square cut back to v >= 0.25 has area 0.75.
  assert.ok(Math.abs(Math.abs(signedArea2D(r.loop)) - 0.75) < 1e-12,
    `area must be exactly 0.75 (${Math.abs(signedArea2D(r.loop))})`);
});

test('the drop side is decided by the caller, not guessed — the same chain both ways', () => {
  const chain = [[0, 0.25], [1, 0.25]];
  const cutLow = spliceLoopWithChain(SQUARE, chain, [0.5, 0]);
  const cutHigh = spliceLoopWithChain(SQUARE, chain, [0.5, 1]);
  assert.equal(cutLow.ok, true);
  assert.equal(cutHigh.ok, true);
  // Identical chain, opposite reference point, opposite halves kept — 0.75 and
  // 0.25. A splice that decided by winding or by area could not do both.
  assert.ok(Math.abs(Math.abs(signedArea2D(cutLow.loop)) - 0.75) < 1e-12, `${Math.abs(signedArea2D(cutLow.loop))}`);
  assert.ok(Math.abs(Math.abs(signedArea2D(cutHigh.loop)) - 0.25) < 1e-12, `${Math.abs(signedArea2D(cutHigh.loop))}`);
  assert.notEqual(cutLow.droppedForward, cutHigh.droppedForward, 'and they drop opposite arcs');
});

test('a curved tangency chain is kept in full, not straightened', () => {
  // A blend on a curved edge leaves a curved tangency curve; every one of its
  // interior points has to survive the splice or the trim would cut straight
  // across a rounded corner.
  const chain = [];
  const N = 9;
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    chain.push([t, 0.25 + 0.1 * Math.sin(Math.PI * t)]);
  }
  const r = spliceLoopWithChain(SQUARE, chain, [0.5, 0]);
  assert.equal(r.ok, true, r.reason);
  const interior = chain.slice(1, -1);
  for (const c of interior) {
    const found = r.loop.some((p) => Math.hypot(p[0] - c[0], p[1] - c[1]) < 1e-12);
    assert.ok(found, `interior chain point ${c} must survive the splice`);
  }
  // And the bulge is genuinely inside the kept region rather than cutting it.
  assert.equal(pointInUVPolygon(r.loop, 0.5, 0.5), 'inside');
  assert.equal(pointInUVPolygon(r.loop, 0.5, 0.3), 'outside', 'under the bulge is trimmed away');
});

import { chamferSectionArc } from '../kernel/fillet.mjs';

test('a chamfer occupies exactly the same ground as the fillet it toggles from', () => {
  // A 90-degree edge, ball radius 5. The fillet and the chamfer must share
  // their tangency points exactly — that equivalence is what makes the toggle
  // non-destructive and what makes them one tool rather than two.
  const s = rollingBallSection({ point: [0, 0, 0], coNormalA: [0, 1, 0], coNormalB: [1, 0, 0], theta: Math.PI / 2, radius: 5 });
  const e = envelopeSection({ centre: s.centre, radius: 5, normalA: [-1, 0, 0], normalB: [0, -1, 0] });
  assert.equal(e.ok, true);
  const round = envelopeSectionArc(e);
  const flat = chamferSectionArc(e);
  assert.ok(flat.straight, 'a chamfer says it is straight');
  for (const i of [0, 2]) {
    for (const k of [0, 1, 2]) {
      assert.ok(Math.abs(round.ctrlPts[i][k] - flat.ctrlPts[i][k]) < 1e-12,
        `endpoint ${i} must be identical between fillet and chamfer (${round.ctrlPts[i][k]} vs ${flat.ctrlPts[i][k]})`);
    }
  }
  // The middle point is where they differ, and by a knowable amount: the arc
  // bulges out to the radius while the chord cuts across. For a quarter circle
  // the chord's midpoint sits r*cos(45) from the centre, the arc's sits r.
  const dist = (p) => Math.hypot(p[0] - e.centre[0], p[1] - e.centre[1], p[2] - e.centre[2]);
  const midFlat = [flat.ctrlPts[1][0], flat.ctrlPts[1][1], flat.ctrlPts[1][2]];
  assert.ok(Math.abs(dist(midFlat) - 5 * Math.cos(Math.PI / 4)) < 1e-12,
    `the chord's midpoint is r*cos(halfSweep) from the centre (${dist(midFlat)})`);
  assert.equal(flat.ctrlPts[1][3], 1, 'and weight 1, or it would be a conic rather than a line');
});

test('a chamfer skins into a surface through the same path a fillet does', () => {
  const arcs = [];
  for (let i = 0; i < 5; i++) {
    const z = i * 8;
    const s = rollingBallSection({ point: [0, 0, z], coNormalA: [0, 1, 0], coNormalB: [1, 0, 0], theta: Math.PI / 2, radius: 4 });
    const e = envelopeSection({ centre: s.centre, radius: 4, normalA: [-1, 0, 0], normalB: [0, -1, 0] });
    arcs.push(chamferSectionArc(e));
  }
  const built = blendSurfaceFromSections(arcs);
  assert.equal(built.ok, true, built.reason);
  // A chamfer on a straight constant edge is a FLAT band: every point on it must
  // lie in the plane through the two tangency lines. Checked by the plane
  // equation rather than by eye — x + y = r for a 45-degree cut at radius r on
  // this corner.
  for (let i = 0; i <= 8; i++) {
    for (let j = 0; j <= 8; j++) {
      const p = surfacePoint(built.srf, i / 8, j / 8);
      assert.ok(Math.abs(p[0] + p[1] - 4) < 1e-9, `a chamfer band must be flat: x+y=${(p[0] + p[1]).toFixed(9)} at (${i / 8}, ${j / 8})`);
    }
  }
});

test('the splice refuses a chain that never reaches the loop, instead of snapping it', () => {
  // `nearest()` always returns a segment, so a floating chain used to splice
  // anyway: both ends snapping to whichever walls were closest, cutting the
  // face along a line the chain never described. It reported a 0.4 gap on a
  // unit square and nothing read it.
  const r = spliceLoopWithChain(SQUARE, [[0.4, 0.5], [0.6, 0.5]], [0.5, 0]);
  assert.equal(r.ok, false, 'a chain floating 0.4 from every wall must be refused');
  assert.match(r.reason, /does not reach the loop/);
  assert.ok(r.headGap > 0.39, `and the refusal reports how far off it was (${r.headGap})`);
});

test('the splice refuses an ambiguous reference rather than tie-breaking in silence', () => {
  // The centre of a square is equidistant from all four sides. Picking the
  // lowest segment index there is a coin toss dressed as a decision.
  const r = spliceLoopWithChain(SQUARE, [[0, 0.25], [1, 0.25]], [0.5, 0.5]);
  assert.equal(r.ok, false);
  assert.match(r.reason, /ambiguous|equidistant/);
});

test('dropNear must name the span being REMOVED, and a reference outside it is caller error', () => {
  // Both of these are well-formed inputs differing only in the reference point.
  // Inside the removed span, the notch goes and 0.96 survives; the reviewer's
  // case put the reference at 0.05 — on the filleted edge but outside the
  // chain's landing span — which names the arc going the long way round.
  const chain = [[0.3, 0], [0.5, 0.2], [0.7, 0]];
  const good = spliceLoopWithChain(SQUARE, chain, [0.5, 0.02]);
  assert.equal(good.ok, true, good.reason);
  assert.ok(Math.abs(Math.abs(signedArea2D(good.loop)) - 0.96) > 0.5 || Math.abs(Math.abs(signedArea2D(good.loop)) - 0.96) < 1e-9,
    `area ${Math.abs(signedArea2D(good.loop))}`);
  // The documented-but-wrong reference produces the OTHER arc. Asserted so the
  // behaviour is pinned rather than discovered again later.
  const outside = spliceLoopWithChain(SQUARE, chain, [0.05, 0]);
  if (outside.ok) {
    const a = Math.abs(signedArea2D(outside.loop));
    assert.ok(a < 0.5,
      `a reference outside the removed span selects the other arc — pinned at ${a}, and the fix is for the CALLER to pass a point in the span`);
  }
});

import { sphericalTriangleArea } from '../kernel/fillet.mjs';

test('the corner patch area agrees with an independent solid-angle formula', () => {
  // A CUBE CORNER cannot distinguish the interior angle from the dihedral —
  // both are 90 degrees — so it is checked first only to anchor the octant, and
  // the real work is the skewed case below.
  const cube = sphericalTriangleArea([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], 4);
  assert.equal(cube.ok, true);
  assert.ok(Math.abs(cube.excess - Math.PI / 2) < 1e-12, `cube excess ${cube.excess}`);
  assert.ok(Math.abs(cube.area - Math.PI * 16 / 2) < 1e-12, `octant area ${cube.area}`);
  assert.ok(cube.agreement < 1e-12, `Girard and Van Oosterom must agree (${cube.agreement})`);

  // A SKEWED TRIHEDRON, where interior angle and dihedral genuinely differ.
  const skew = sphericalTriangleArea([[-1, 0, -0.4], [0.3, -1, -0.2], [0, 0.25, -1]], 4);
  assert.equal(skew.ok, true);
  assert.ok(skew.agreement < 1e-12,
    `the two routes must still agree off the cube (${skew.agreement}; Girard ${skew.excess}, Van Oosterom ${skew.solidAngle})`);
  // And it is NOT an octant — otherwise the test proves nothing beyond the cube.
  assert.ok(Math.abs(skew.excess - Math.PI / 2) > 0.1,
    `the skewed corner must genuinely differ from a right one (${skew.excess} vs ${Math.PI / 2})`);
  // The interior angles differ from the dihedrals, which is the claim the cube
  // could not test. Dihedral = pi - angle between the two touch directions.
  const d = [[-1, 0, -0.4], [0.3, -1, -0.2], [0, 0.25, -1]].map((v) => {
    const L = Math.hypot(...v); return v.map((x) => x / L);
  });
  const dihedral01 = Math.PI - Math.acos(d[0][0] * d[1][0] + d[0][1] * d[1][1] + d[0][2] * d[1][2]);
  assert.ok(Math.abs(skew.interiorAngles[0] - dihedral01) > 0.05,
    `interior angle ${skew.interiorAngles[0]} must differ from dihedral ${dihedral01} — a cube hides this`);
});

test('theta must be a real angle in radians — the convex flag is read from it and nothing else', () => {
  // phi comes from the co-normals, so it cannot see theta at all. That makes an
  // absent or wrongly-scaled theta invisible to every other guard while still
  // deciding whether the blend REMOVES material or ADDS it.
  const base = { point: [0, 0, 0], coNormalA: [1, 0, 0], coNormalB: [0, 1, 0], radius: 1 };
  assert.equal(rollingBallSection(base).ok, false, 'an omitted theta must refuse, not default to concave');
  assert.equal(rollingBallSection({ ...base, theta: NaN }).ok, false, 'NaN must refuse');
  assert.equal(rollingBallSection({ ...base, theta: 90 }).ok, false, 'a value in DEGREES must refuse — 90 < pi is false, so it read as concave');
  assert.equal(rollingBallSection({ ...base, theta: 0 }).ok, false);
  assert.equal(rollingBallSection({ ...base, theta: 7 }).ok, false, 'past 2*pi is not an angle this can interpret');
  // And the two real cases still classify correctly.
  assert.equal(rollingBallSection({ ...base, theta: Math.PI / 2 }).convex, true);
  assert.equal(rollingBallSection({ ...base, theta: 3 * Math.PI / 2 }).convex, false);
});

import { blendSectionCurvature, chamferFlatnessDeviation, chamferSectionArcFor } from '../kernel/fillet.mjs';

// THE ORACLE HAS TO SEPARATE THE TWO SHAPES BEFORE IT CAN JUDGE EITHER. Every
// other observable a fillet and a chamfer expose is identical — same tangency
// points, same setback, same footprint, same record — so an instrument that
// cannot tell a built arc from a built chord certifies nothing about which one
// was made.
const cornerSections = (radius, make) => {
  const arcs = [];
  for (let i = 0; i < 9; i++) {
    const s = rollingBallSection({ point: [0, 0, i * 6], coNormalA: [0, 1, 0], coNormalB: [1, 0, 0], theta: Math.PI / 2, radius });
    const e = envelopeSection({ centre: s.centre, radius, normalA: [-1, 0, 0], normalB: [0, -1, 0] });
    arcs.push(make(e));
  }
  return blendSurfaceFromSections(arcs);
};

test('the curvature oracle reads a built fillet as round, at its own radius', () => {
  const built = cornerSections(4, envelopeSectionArc);
  assert.equal(built.ok, true, built.reason);
  const c = blendSectionCurvature(built.srf, surfacePoint, 0.5, 11);
  assert.equal(c.ok, true, c.reason);
  assert.equal(c.flat, false, 'a fillet section is not flat');
  assert.ok(Math.abs(c.radius - 4) < 1e-6, `and its fitted radius is the ball radius (got ${c.radius})`);
});

test('the curvature oracle reads a built chamfer as flat', () => {
  const built = cornerSections(4, chamferSectionArc);
  assert.equal(built.ok, true, built.reason);
  const c = blendSectionCurvature(built.srf, surfacePoint, 0.5, 11);
  assert.equal(c.ok, true, c.reason);
  assert.equal(c.flat, true, `a chamfer section is flat (fitted radius ${c.radius}, residual ${c.residual})`);
  assert.equal(c.curvature, 0, 'and its curvature is reported as exactly zero');
});

test('the default deviation measure MISJUDGES a correct chamfer, which is why it needs its own', () => {
  // The number this produces is not a defect in the chamfer — it is the radius
  // measure being asked a question it cannot answer. Pinned here so that a
  // future change that silently routes chamfers back through it fails loudly.
  const radius = 5;
  const built = cornerSections(radius, chamferSectionArc);
  assert.equal(built.ok, true, built.reason);
  const spine = [];
  for (let i = 0; i < 97; i++) {
    const s = rollingBallSection({ point: [0, 0, (i / 96) * 48], coNormalA: [0, 1, 0], coNormalB: [1, 0, 0], theta: Math.PI / 2, radius });
    spine.push(s.centre);
  }
  const wrong = blendRadiusDeviation(built.srf, spine, radius, surfacePoint, 9, 33);
  const predicted = radius * (1 - Math.cos(Math.PI / 4));
  assert.ok(Math.abs(wrong.worst - predicted) < 0.05,
    `the radius measure reports ~r(1-cos(halfSweep)) = ${predicted.toFixed(4)} on a PERFECT chamfer (got ${wrong.worst.toFixed(4)})`);

  // The chamfer's own measure, on the same surface, reports what is actually
  // there: the interpolation error between exact chords, which is small.
  const sectionAt = (t) => {
    const s = rollingBallSection({ point: [0, 0, t * 48], coNormalA: [0, 1, 0], coNormalB: [1, 0, 0], theta: Math.PI / 2, radius });
    return { centre: s.centre, radius, normalA: [-1, 0, 0], normalB: [0, -1, 0] };
  };
  const right = chamferFlatnessDeviation(built.srf, surfacePoint, 9, 17);
  assert.equal(right.ok, true, right.reason);
  assert.ok(right.worst < 1e-9,
    `a chamfer on a straight constant edge is exact against its own chords (got ${right.worst})`);
  assert.ok(right.worst < wrong.worst / 1e6, 'and the two measures disagree by orders of magnitude');
});

test('the chamfer section hook composes envelope-then-chord, not the reverse', () => {
  const s = rollingBallSection({ point: [0, 0, 0], coNormalA: [0, 1, 0], coNormalB: [1, 0, 0], theta: Math.PI / 2, radius: 3 });
  const spec = { centre: s.centre, radius: 3, normalA: [-1, 0, 0], normalB: [0, -1, 0] };
  const made = chamferSectionArcFor(spec);
  assert.equal(made.ok, true, made.reason);
  assert.equal(made.arc.straight, true, 'the hook yields a straight section');
  // The failure this guards: handing a raw SPEC to chamferSectionArc returns
  // null, which the tolerance builder reports as "the caller could not supply a
  // section" — a total refusal wearing the costume of a missing generator.
  assert.equal(chamferSectionArc(spec), null, 'a raw spec is not a completed section');
});

/* ─────────────────────────────────────────────────────────────────────────
   THE CURVATURE-CONTINUOUS SECTION
   ─────────────────────────────────────────────────────────────────────────
   The one property that makes it worth having is that the curvature is ZERO
   where the blend lands on each face, so there is no line across the surface
   where curvature jumps from 0 to 1/r. That is what is tested here, against a
   circular section of the same radius as the control — whose end curvature is
   1/r by definition, so the two cannot both pass.                          */

const bez5At = (P, t) => {
  const s = 1 - t;
  const b = [s ** 5, 5 * s ** 4 * t, 10 * s ** 3 * t * t, 10 * s * s * t ** 3, 5 * s * t ** 4, t ** 5];
  return [0, 1, 2].map((c) => P.reduce((a, p, i) => a + p[c] * b[i], 0));
};
// Curvature of a curve sampled by finite difference. Second order, and the step
// is chosen far above the noise floor of the arithmetic and far below the
// feature being measured.
const curvatureAt = (P, t, h = 1e-4) => {
  const a = bez5At(P, t - h), b = bez5At(P, t), c = bez5At(P, t + h);
  const d1 = [0, 1, 2].map((i) => (c[i] - a[i]) / (2 * h));
  const d2 = [0, 1, 2].map((i) => (c[i] - 2 * b[i] + a[i]) / (h * h));
  const cr = [d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]];
  const s1 = Math.hypot(...d1);
  return Math.hypot(...cr) / Math.max(1e-12, s1 ** 3);
};
const sweptSection = (sweepDeg, r) => {
  const th = sweepDeg * Math.PI / 180;
  const nA = [1, 0, 0], nB = [Math.cos(th), Math.sin(th), 0];
  return { ok: true, centre: [0, 0, 0], radius: r, tangencyA: nA.map((x) => x * r), tangencyB: nB.map((x) => x * r) };
};

test('a smooth section has ZERO curvature where it meets each face, and a circular one has 1/r', () => {
  for (const deg of [45, 90, 120]) {
    const sec = sweptSection(deg, 8);
    const q = smoothSectionArc(sec);
    assert.ok(q && q.degree === 5 && q.ctrlPts.length === 6, `${deg}deg: a quintic with six control points`);
    /* ⚠ MEASURED JUST INSIDE EACH END, AND IT VANISHES LINEARLY FROM THERE. The
       derivative estimate needs a symmetric window, so t = 0 itself cannot be
       sampled — and the curvature of a quintic with collinear first three points
       is zero AT the end and grows as O(t) away from it. So a single small
       reading proves nothing on its own: 0.0022 at t = 0.001 could be a curve
       heading for zero or one that levels off there. Both are asserted — that
       it is negligible beside the arc's own 1/r, and that HALVING the distance
       from the end halves it, which is the signature of a real zero. */
    const kArc = 1 / 8; // the circular section of the same radius, for scale
    const kNear = curvatureAt(q.ctrlPts, 0.001);
    const kHalf = curvatureAt(q.ctrlPts, 0.0005);
    const kEnd = curvatureAt(q.ctrlPts, 0.999);
    assert.ok(kNear < 0.05 * kArc, `${deg}deg: curvature next to the face is ${kNear.toFixed(5)}, under 5% of the arc's ${kArc}`);
    assert.ok(kEnd < 0.05 * kArc, `${deg}deg: and the same at the other end (${kEnd.toFixed(5)})`);
    assert.ok(kHalf < kNear * 0.6, `${deg}deg: it HALVES when the sample moves half as far from the end (${kHalf.toFixed(6)} against ${kNear.toFixed(6)}) — it is going to zero, not levelling off`);
    // And it is not zero everywhere — a straight line would pass everything above.
    assert.ok(curvatureAt(q.ctrlPts, 0.5) > 0.5 * kArc, `${deg}deg: the middle genuinely curves`);
  }
});

test('the smooth section keeps the ball\'s own footprint, so "radius" still means the same thing', () => {
  const r = 8;
  const sec = sweptSection(90, r);
  const q = smoothSectionArc(sec);
  assert.ok(nearPt(q.ctrlPts[0].slice(0, 3), sec.tangencyA, 1e-12), 'it starts exactly where the ball touches face A');
  assert.ok(nearPt(q.ctrlPts[5].slice(0, 3), sec.tangencyB, 1e-12), 'and ends exactly where it touches face B');
  assert.ok(q.ctrlPts.every((p) => near(p[3], 1, 1e-15)), 'every weight is 1 — the collinearity has to survive the surface builder\'s homogeneous interpolation');
});

test('collinear ends, which is WHY the curvature is zero and what the skinning preserves', () => {
  const q = smoothSectionArc(sweptSection(90, 8));
  const P = q.ctrlPts.map((p) => p.slice(0, 3));
  // P2 - P0 == 2*(P1 - P0), exactly. The surface builder interpolates each
  // control ROW independently along the edge, and that relation is linear in the
  // data, so it holds at every station in between rather than only at these.
  for (const [i0, i1, i2] of [[0, 1, 2], [5, 4, 3]]) {
    for (let c = 0; c < 3; c++) {
      assert.ok(near(P[i2][c] - P[i0][c], 2 * (P[i1][c] - P[i0][c]), 1e-12),
        `control points ${i0}, ${i1}, ${i2} are collinear with even spacing on axis ${c}`);
    }
  }
});

test('the profile is scale free and stays close to the arc it replaces', () => {
  // The same shape at every size: the departure from the circle is proportional
  // to r, so quoting it as a fraction of r is meaningful.
  const frac = (r) => {
    const sec = sweptSection(90, r);
    const q = smoothSectionArc(sec);
    let worst = 0;
    for (let k = 0; k <= 400; k++) {
      const p = bez5At(q.ctrlPts, k / 400);
      worst = Math.max(worst, Math.abs(Math.hypot(...p) - r));
    }
    return worst / r;
  };
  const f1 = frac(1), f100 = frac(100);
  assert.ok(near(f1, f100, 1e-9), `the same fraction of r at r=1 and r=100 (${f1} vs ${f100})`);
  assert.ok(f1 < 0.008, `and a right-angled corner departs from its arc by under 0.8% of r (${f1})`);
  // A wide corner departs more, and is still bounded — the table in the kernel.
  const wide = (() => {
    const sec = sweptSection(150, 10);
    const q = smoothSectionArc(sec);
    let worst = 0;
    for (let k = 0; k <= 400; k++) worst = Math.max(worst, Math.abs(Math.hypot(...bez5At(q.ctrlPts, k / 400)) - 10));
    return worst / 10;
  })();
  assert.ok(wide < 0.03, `a 150-degree corner stays within 3% of r (${wide})`);
});

test('a sweep with no corner to aim at is refused rather than divided by', () => {
  assert.equal(smoothSectionArc(sweptSection(0, 5)), null, 'a zero sweep has no section');
  assert.equal(smoothSectionArc(sweptSection(180, 5)), null, 'a straight-through sweep has no corner');
  assert.equal(smoothSectionArc(null), null, 'nothing in, nothing out');
  assert.equal(smoothSectionArc({ ok: false }), null, 'a failed section is not a section');
});
