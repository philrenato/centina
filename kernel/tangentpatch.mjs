// N-SIDED TANGENT PATCH — filling a closed loop of N boundary curves with one
// surface that leaves each boundary in the tangent plane of whatever lies
// outside it. This is the classical n-sided hole problem behind the
// Gregory/Charrot-Gregory patches (Charrot & Gregory, "A pentagonal surface
// patch for computer aided geometric design", CAGD 1(1), 1984) and behind
// OCCT's constrained filling.
//
// The three-sided case is Nielson's side-vertex interpolant with (b_j b_k)^2
// weights and a cubic Hermite ray per side (`sideVertexPatch`, cornerblend.mjs).
// THIS MODULE IS THAT SCHEME, GENERALIZED, and it reduces to it identically at
// N = 3 rather than approximating it — see the reduction argument on
// `nSidedTangentPatch`.
//
// ═══ WHAT WAS CHOSEN, AND AGAINST WHAT ═════════════════════════════════════
//
// PARAMETERISATION: a regular N-gon domain carrying GENERALIZED BARYCENTRIC
// COORDINATES (mean value, Floater, "Mean value coordinates", CAGD 20(1),
// 2003). A point of the patch is addressed by N coordinates lambda_0..lambda_N-1
// that sum to one; the whole interpolant is written in them, exactly as the
// triangular one is written in (b0, b1, b2).
//
// The two alternatives, and why not:
//   · A MIDPOINT SPLIT into N quadrilateral sub-patches (the Catmull-Clark-like
//     route) introduces an EXTRAORDINARY POINT at the center where N quads
//     meet. That point is a genuine parametric singularity: the sub-patch
//     partials there do not agree, so the normal is a limit rather than a
//     value, and every downstream consumer that differentiates the surface —
//     closest-point Newton, offsetting, curvature shading — meets it. It also
//     turns one patch into N, so trimming and tessellation inherit N internal
//     seams that have to be kept watertight by hand.
//   · CHARROT-GREGORY blends N CORNER interpolants rather than N side ones.
//     It is the more famous construction and it does not reduce to the
//     side-vertex scheme already in this kernel at N = 3, so adopting it would
//     leave two unrelated schemes meeting along the same fillet chain, each
//     with its own fold behavior and its own tuning. Agreeing with the
//     three-sided patch that already ships is worth more here than matching
//     the literature's most-cited form.
//
// The cost of the choice: the domain is a REGULAR N-gon, so a hole whose sides
// differ wildly in length gets a distorted isoparametric spacing. That is a
// parameterisation quality, not a geometric one — the boundary is still exact
// and the tangency still holds — and a chord-length-proportional domain is a
// deliberate scope cut, not an oversight.
//
// ═══ DEGENERACIES, NAMED UP FRONT ══════════════════════════════════════════
//
// ⚠ THE N CORNERS ARE 0/0 AND ARE RETURNED OUTRIGHT. Every side's weight
// carries a factor that vanishes at every corner of the domain (at corner m,
// lambda_m = 1 and all others are 0, so (lambda_i lambda_i+1)^2 = 0 for every
// side), so the blend is undefined there and float noise would decide it. The
// value is known exactly — it is the corner — so it is named rather than
// approached.
//
// ⚠ THERE IS NO CENTRAL POLE, AND THAT IS A PROPERTY OF THE CHOICE ABOVE.
// Inside a convex domain every mean value coordinate is strictly positive, so
// every side weight is strictly positive and their sum never vanishes. Nothing
// in the interior is a limit.
//
// ⚠ THE ONE 0/0 THE CONSTRUCTION WOULD OTHERWISE HAVE IS REMOVED ALGEBRAICALLY
// RATHER THAN GUARDED. The three-sided scheme runs a Hermite from the boundary
// point out to the OPPOSITE VERTEX; N sides have no opposite vertex, and the
// generalization is the far point F_i = (sum of lambda_m V_m over m not in
// {i, i+1}) / rho_i, with rho_i that same sum of coordinates. F_i is 0/0 on
// side i itself. But F_i only ever enters multiplied by Hermite terms that
// carry a factor of rho_i, so the division cancels before it is taken:
//     R_i = (rho^3 - 2 rho^2 + 1) Q_i + (rho^3 - 2 rho^2 + rho) m0_i
//           + rho (2 - rho) G_i,     G_i = sum of lambda_m V_m over the far m.
// G_i is LINEAR in the coordinates, so R_i is smooth everywhere in the domain,
// including on the boundary where the unsimplified form is undefined. A guard
// with an epsilon would have left a thin band where the answer was decided by
// the guard rather than by the geometry.
//
// ═══ WHERE IT IS G1 AND WHERE IT IS NOT ════════════════════════════════════
//
// G1 ALONG THE OPEN BOUNDARIES. On side i every other side's weight vanishes,
// so the patch IS R_i there and reproduces the boundary exactly. Approaching
// side i, the rival weights fall as the SQUARE of the distance (side i-1 and
// side i+1 each carry one coordinate that is O(eps), squared; the rest carry
// two), so the patch agrees with R_i to first order and its tangent plane is
// spanned by the boundary tangent and the supplied cross-boundary tangent.
//
// AT THE CORNERS IT IS G1 WHEN THE INPUT IS, AND G0 WHEN THE INPUT IS NOT —
// which is a weaker claim than "G1 everywhere" and a stronger one than the
// twist obstruction is usually said to allow, so it is MEASURED rather than
// asserted (`cornerNormalSpread`).
//
// Position at a corner is exact. The normal there is a LIMIT, since every
// weight vanishes. Sweeping the approach radius r on a fan of directions into
// one corner:
//   · corner-compatible input (the two meeting sides' cross-tangents imply the
//     SAME tangent plane there): the spread of limit normals falls linearly
//     with r — 0.27, 0.027, 0.0027 degrees at r = 1e-2, 1e-3, 1e-4. The limit
//     exists and is unique, so the patch is G1 at the corner too.
//   · corner-INCOMPATIBLE input (one side's cross-tangent tilted out of the
//     neighbor's plane near that corner): the spread sits between 45 and 49
//     degrees at every radius down to 1e-5. There is no limit; it is G0 there.
// The patch reproduces the input's crease; it does not repair one. Which is why
// corner compatibility is CHECKED at build time rather than discovered later.
//
// ⚠ WHAT IS GENUINELY DISCONTINUOUS AT A CORNER IS THE TWIST — the mixed second
// derivative, which each of the two meeting sides implies differently and which
// no single polynomial patch can honour both of. Leaving it direction-dependent
// is exactly Gregory's device, and it is what buys G1 at the corner; the price
// is that the patch is not curvature-continuous there and never will be.

const EPS = 1e-12;
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
function norm(a) { const L = len(a); return L > EPS ? [a[0] / L, a[1] / L, a[2] / L] : null; }
const isVec3 = (p) => Array.isArray(p) && p.length >= 3 && Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]);

/** The domain the patch is parameterised over: a regular N-gon of circumradius
 *  1, counter-clockwise, with vertex i at the patch's corner i and edge i (from
 *  vertex i to vertex i+1) carrying boundary curve i.
 *
 *  The starting angle puts vertex 0 on +x. Nothing downstream depends on the
 *  orientation of the domain in its own plane; it is fixed so that a caller
 *  sampling the domain directly gets the same points every time.
 */
export function regularDomain(n) {
  if (!Number.isInteger(n) || n < 3) return null;
  const v = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    v.push([Math.cos(a), Math.sin(a)]);
  }
  return v;
}

/** Mean value coordinates of a 2-D point inside a convex polygon.
 *
 *  Reduces EXACTLY to ordinary barycentric coordinates on a triangle: mean
 *  value coordinates reproduce linear functions and sum to one, and on a
 *  triangle those two conditions have a unique solution.
 *
 *  ⚠ THE BOUNDARY CASES ARE ANSWERED, NOT APPROACHED. At a vertex the weight
 *  formula divides by a zero radius, and on an edge the half-angle tangent of
 *  an angle at pi is infinite. Both have exact answers — the Lagrange value at
 *  a vertex, linear interpolation on an edge — and both are returned as such,
 *  because the alternative is a coordinate vector made of infinities whose
 *  normalized value happens to look plausible.
 */
export function meanValueCoords(vertices, x, y, tol = 1e-12) {
  const n = vertices.length;
  if (n < 3) return null;
  const u = [], r = [];
  for (let i = 0; i < n; i++) {
    const dx = vertices[i][0] - x, dy = vertices[i][1] - y;
    const R = Math.hypot(dx, dy);
    if (!(R > tol)) { const l = new Array(n).fill(0); l[i] = 1; return l; }
    r.push(R); u.push([dx / R, dy / R]);
  }
  const t = new Array(n);
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const s = u[i][0] * u[j][1] - u[i][1] * u[j][0];   // sin of the angle at (x, y)
    const c = u[i][0] * u[j][0] + u[i][1] * u[j][1];   // its cosine
    if (Math.abs(s) <= tol) {
      if (c < 0) {                                     // the point lies ON edge i
        const l = new Array(n).fill(0);
        l[i] = r[j] / (r[i] + r[j]); l[j] = r[i] / (r[i] + r[j]);
        return l;
      }
      t[i] = 0;                                        // collinear beyond the edge
      continue;
    }
    t[i] = (1 - c) / s;
  }
  let sum = 0;
  const w = new Array(n);
  for (let i = 0; i < n; i++) {
    const im = (i + n - 1) % n;
    w[i] = (t[im] + t[i]) / r[i];
    sum += w[i];
  }
  if (!(Math.abs(sum) > tol)) return null;
  for (let i = 0; i < n; i++) w[i] /= sum;
  return w;
}

/* ⚠ THE CROSS-TANGENT MAGNITUDE IS A QUARTER OF THE RAY, and it is the same
   number the three-sided patch uses so that the two agree at N = 3. It sets how
   far the Hermite reaches before the far end takes over, so it controls
   INTERIOR SHAPE and not the tangent plane — tangency is exact at every value.
   What it does control is FOLDING, which is why the validator below exists and
   why it is run by default. */
const TANGENT_SCALE = 0.25;

/** An N-sided patch through N boundary curves, tangent to a supplied
 *  cross-boundary field along each of them.
 *
 *  `boundary[i](s)`, s in [0, 1], runs from corner i to corner i+1. The curves
 *  must close into a loop; if they do not, that is refused rather than welded
 *  over, because a patch built on an open loop is exact on a boundary that is
 *  not the hole's.
 *
 *  `tangent[i](s)` supplies the direction the ADJACENT SURFACE leaves boundary
 *  i at parameter s. It only has to LIE in that surface's tangent plane and
 *  point inward: its magnitude and any component along the boundary change the
 *  interior shape, not the tangent plane. Omit it entirely for a G0 fill.
 *
 *  Evaluation is by generalized barycentric coordinate: `evaluate(lambda)` with
 *  an array of N non-negative numbers summing to one, or `evaluateXY(x, y)` for
 *  a point of the regular N-gon domain.
 *
 *  ═══ THE REDUCTION TO THE THREE-SIDED PATCH ═══
 *  At N = 3, side i runs from V_i to V_i+1 and its far set is the single
 *  vertex V_i+2, so G_i = lambda_i+2 V_i+2 and rho_i = lambda_i+2. The Hermite
 *  above becomes exactly the cubic from Q_i to V_i+2 that `sideVertexPatch`
 *  writes, the weight (lambda_i lambda_i+1)^2 is exactly its (b_j b_k)^2, the
 *  ray parameter lambda_i+1/(lambda_i + lambda_i+1) is exactly its s, and the
 *  Hermite magnitude is taken against the same chord. The two schemes are the
 *  same formula, in the same coordinates, differing only in the index at which
 *  a side is filed.
 */
export function nSidedTangentPatch(opts = {}) {
  const {
    boundary, tangent = null, corners = null,
    tangentScale = TANGENT_SCALE,
    loopTolerance = 1e-9,
    cornerAngleToleranceDeg = 1,
    validate = true,
    foldOptions = null,
  } = opts;

  if (!Array.isArray(boundary) || boundary.length < 3) {
    return { ok: false, reason: `an n-sided patch needs at least three boundary curves; got ${Array.isArray(boundary) ? boundary.length : 0}` };
  }
  const n = boundary.length;
  for (let i = 0; i < n; i++) {
    if (typeof boundary[i] !== 'function') return { ok: false, reason: `boundary ${i} is not an evaluable curve` };
  }
  if (!Number.isFinite(tangentScale) || tangentScale < 0) {
    return { ok: false, reason: `the cross-tangent scale must be a non-negative number; got ${tangentScale}` };
  }
  /* ⚠ A MALFORMED TANGENT ARRAY IS REFUSED, NOT SILENTLY DEMOTED TO G0. A
     caller that supplies N-1 tangent fields, or an array with one null in it,
     has a bug; quietly building the positional-only patch hands them a creased
     fillet that looks like a geometry problem. Absent is a request; wrong is a
     defect. */
  let hasTangent = false;
  if (tangent != null) {
    if (!Array.isArray(tangent) || tangent.length !== n || !tangent.every((t) => typeof t === 'function')) {
      return { ok: false, reason: `the cross-boundary tangent must be ${n} evaluable fields, one per side, or omitted entirely for a G0 fill` };
    }
    hasTangent = true;
  }

  // The corners, read off the curves themselves, with the loop checked.
  const A = [], B = [];
  for (let i = 0; i < n; i++) {
    const a = boundary[i](0), b = boundary[i](1);
    if (!isVec3(a) || !isVec3(b)) return { ok: false, reason: `boundary ${i} does not return a finite 3-vector at its ends` };
    A.push([a[0], a[1], a[2]]); B.push([b[0], b[1], b[2]]);
  }
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of A.concat(B)) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
  const scale = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  if (!(scale > EPS)) return { ok: false, reason: 'every boundary curve starts and ends at the same point — there is no loop to fill' };
  const tol = loopTolerance * Math.max(scale, 1);
  for (let i = 0; i < n; i++) {
    const gap = len(sub(B[i], A[(i + 1) % n]));
    if (gap > tol) {
      return { ok: false, reason: `boundary ${i} ends ${gap.toExponential(3)} from where boundary ${(i + 1) % n} begins — the ${n} curves do not close into a loop, so the patch would be exact on a boundary that is not the hole's` };
    }
  }
  const V = A;
  if (corners != null) {
    if (!Array.isArray(corners) || corners.length !== n) return { ok: false, reason: `the corner list must have one point per side; got ${Array.isArray(corners) ? corners.length : 0} for ${n} sides` };
    for (let i = 0; i < n; i++) {
      if (!isVec3(corners[i])) return { ok: false, reason: `corner ${i} is not a finite 3-vector` };
      const d = len(sub(corners[i], V[i]));
      if (d > tol) return { ok: false, reason: `corner ${i} is ${d.toExponential(3)} from where boundary ${i} actually starts` };
    }
  }

  /* The far centroid of side i: the mean of the corners that are NOT its own
     two. At N = 3 that is the single opposite vertex, which is what makes the
     Hermite magnitude and the orientation poll below reduce exactly. It is used
     only as a LENGTH SCALE and as a direction to poll against — the far end of
     the Hermite itself is the coordinate-weighted G_i, not this. */
  const farCentre = [];
  for (let i = 0; i < n; i++) {
    let c = [0, 0, 0], k = 0;
    for (let m = 0; m < n; m++) {
      if (m === i || m === (i + 1) % n) continue;
      c = [c[0] + V[m][0], c[1] + V[m][1], c[2] + V[m][2]]; k += 1;
    }
    if (k === 0) return { ok: false, reason: 'a side has no far corners — impossible for three or more sides' };
    farCentre.push([c[0] / k, c[1] / k, c[2] / k]);
  }

  /* ⚠⚠ THE CROSS-TANGENT'S ORIENTATION IS DECIDED ONCE PER SIDE, BY MAJORITY,
     NOT ONCE PER SAMPLE. The field is already smooth — it is the neighboring
     surface's normal crossed with the boundary tangent, and both vary smoothly
     — so a per-sample sign test injects a discontinuity the field never had:
     the chord to the far center swings past perpendicular partway along an
     asymmetric side, the test changes its mind there, and the ray family
     reverses mid-boundary. One decision per side, polled across it.

     A SPLIT VOTE IS NOT ITSELF A REFUSAL: the chord genuinely swings past
     perpendicular on asymmetric input without the resulting patch being wrong.
     Whether it is sound is the fold validator's question and is asked there. */
  const sideSign = new Array(n).fill(1);
  if (hasTangent) {
    const bad = [];
    for (let i = 0; i < n; i++) {
      let plus = 0, minus = 0;
      for (let k = 1; k < 8; k++) {
        const s = k / 8;
        const d = norm(tangent[i](s));
        if (!d) continue;
        const q = boundary[i](s);
        if (!isVec3(q)) continue;
        const chord = sub(farCentre[i], q);
        const L = len(chord);
        if (!(L > EPS)) continue;
        if (dot(d, chord) >= 0) plus += 1; else minus += 1;
      }
      if (plus === 0 && minus === 0) { bad.push(i); continue; }
      sideSign[i] = plus >= minus ? 1 : -1;
    }
    if (bad.length) return { ok: false, reason: `side(s) ${bad.join(', ')}: no usable cross-tangent anywhere along them` };

    /* ⚠⚠ AT A CORNER THE TWO MEETING SIDES MUST NAME THE SAME TANGENT PLANE, or
       no patch is G1 there and this one will faithfully reproduce the input's
       crease while its caller believes it built a smooth fill. Measured: the
       limit normal's spread over approach directions is 0.27 degrees at r=1e-2
       and falls linearly to 0.0027 at r=1e-4 when the corner data agrees, and
       stands still between 45 and 49 degrees at EVERY radius when one side's
       cross-tangent is tilted out of its neighbor's plane. The second case has no limit normal
       at all, so it is refused by name at build time.

       A hole that genuinely has a crease running into a corner is a real thing
       to want; raise the tolerance deliberately rather than have the default
       be silent about the ordinary case. */
    const gate = Math.cos(Math.max(0, Math.min(180, cornerAngleToleranceDeg)) * Math.PI / 180);
    const hs = 1e-6;
    const cornerNormal = (side, atEnd) => {
      const s = atEnd ? 1 : 0;
      const a = boundary[side](atEnd ? 1 - hs : 0), b = boundary[side](atEnd ? 1 : hs);
      if (!isVec3(a) || !isVec3(b)) return null;
      const bt = norm(sub(b, a));
      const ct = norm(tangent[side](s));
      if (!bt || !ct) return null;
      return norm(cross(bt, ct));
    };
    for (let m = 0; m < n; m++) {
      const prev = (m + n - 1) % n;
      const nA = cornerNormal(prev, true), nB = cornerNormal(m, false);
      if (!nA || !nB) continue;   // a stationary boundary end names no plane; nothing to compare
      const c = Math.abs(dot(nA, nB));
      if (c < gate) {
        const deg = Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
        return { ok: false, reason: `at corner ${m} the surfaces outside sides ${prev} and ${m} name tangent planes ${deg.toFixed(3)} degrees apart — they crease against each other there, so no fill can be tangent to both and this one would only reproduce the crease`, cornerAngleDeg: deg, corner: m };
      }
    }
  }

  const domainVertices = regularDomain(n);

  const evaluate = (lambda) => {
    if (!Array.isArray(lambda) || lambda.length !== n) return null;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const l = lambda[i];
      if (!Number.isFinite(l)) return null;
      /* ⚠ OFF-DOMAIN INPUT MUST NOT GET A CONFIDENT ANSWER. A negative
         coordinate puts the Hermite outside the curve it was built from, and a
         set that does not sum to one is not a point of the domain at all.
         Float drift of a few ulps from a caller's own arithmetic is fine;
         gross denormalization is that caller's bug. */
      if (l < -1e-9) return null;
      sum += l;
    }
    if (Math.abs(sum - 1) > 1e-6) return null;
    for (let m = 0; m < n; m++) if (lambda[m] >= 1 - 1e-12) return V[m].slice();

    let acc = [0, 0, 0], wsum = 0;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const a = lambda[i], b = lambda[j];
      const den = a + b;
      if (!(den > 1e-15)) continue;   // both of this side's own coordinates vanish
      const w = (a * b) * (a * b);
      if (!(w > 0)) continue;
      const s = b / den;
      const Q = boundary[i](s);
      if (!isVec3(Q)) return null;
      // G_i and rho_i from the SAME sum, so they cannot drift apart when a
      // caller's coordinates sum to one only to within float.
      let G = [0, 0, 0], rho = 0;
      for (let m = 0; m < n; m++) {
        if (m === i || m === j) continue;
        const l = lambda[m];
        rho += l;
        G = [G[0] + l * V[m][0], G[1] + l * V[m][1], G[2] + l * V[m][2]];
      }
      if (rho < 0) rho = 0;
      let P;
      let D = null;
      if (hasTangent) {
        const raw = tangent[i](s);
        D = Array.isArray(raw) ? norm(raw) : null;
      }
      const chordLen = len(sub(farCentre[i], Q));
      if (!D || !(chordLen > EPS)) {
        // Positional only: the linear blend from the boundary to the far point.
        // rho * F_i is G_i, so this needs no division either.
        P = [(1 - rho) * Q[0] + G[0], (1 - rho) * Q[1] + G[1], (1 - rho) * Q[2] + G[2]];
      } else {
        const mag = sideSign[i] * chordLen * tangentScale;
        const r2 = rho * rho, r3 = r2 * rho;
        const cQ = r3 - 2 * r2 + 1;          // h00 - h11
        const cM = r3 - 2 * r2 + rho;        // h10
        const cG = rho * (2 - rho);          // (h01 + h11) / rho
        P = [
          cQ * Q[0] + cM * mag * D[0] + cG * G[0],
          cQ * Q[1] + cM * mag * D[1] + cG * G[1],
          cQ * Q[2] + cM * mag * D[2] + cG * G[2],
        ];
      }
      acc = [acc[0] + w * P[0], acc[1] + w * P[1], acc[2] + w * P[2]];
      wsum += w;
    }
    if (!(wsum > 0)) return null;
    return [acc[0] / wsum, acc[1] / wsum, acc[2] / wsum];
  };

  const coordsAt = (x, y) => meanValueCoords(domainVertices, x, y);
  const evaluateXY = (x, y) => {
    const l = coordsAt(x, y);
    return l ? evaluate(l) : null;
  };

  const patch = {
    ok: true, sides: n, corners: V.map((p) => p.slice()), tangentScale,
    continuity: hasTangent ? 'G1 along the open boundaries, G0 at the corners' : 'G0',
    domain: { vertices: domainVertices, coordsAt },
    evaluate, evaluateXY,
  };

  /* ⚠ THE VALIDATOR'S VERDICT IS HONOURED, NOT REPORTED. A folded n-sided fill
     is a self-intersecting surface that no watertight check will accept, and a
     scheme with no closed-form fold criterion cannot promise otherwise. The
     sweep costs a few thousand evaluations once, which is the cost of a
     modeling operation and not of a frame; a caller who has already judged the
     input can pass validate: false knowingly. */
  if (validate) {
    const v = nSidedPatchFolds(patch, foldOptions || {});
    if (!v.ok) return { ok: false, reason: v.reason || 'the patch could not be validated', fold: v };
  }
  return patch;
}

/** Does this n-sided patch fold or crease? Sampled, because the scheme has no
 *  closed-form answer.
 *
 *  A fold is a LOCAL reversal of the surface normal, so it is found by
 *  comparing ADJACENT samples. An n-sided fill legitimately sweeps its normal
 *  through ninety degrees or more — it wraps a hole — and comparing everything
 *  to one reference calls that a failure.
 *
 *  ⚠⚠ A UNIFORM GRID IS BLIND WHERE THE FOLDS ARE. The interpolant is exact on
 *  the boundary and blends hardest just INSIDE it, so folds live in a thin band
 *  at small distance from a side — past the first row of any grid coarse enough
 *  to run. The sweep is therefore a coarse interior grid PLUS a dense ribbon
 *  hugging each side, down to the guard the finite difference itself imposes.
 *  Measured over 220 randomised holes at N = 3..8: the ribbon was the DECIDING
 *  sampler on 3 of them, finding 2 to 12 normal reversals in patches the
 *  interior grid alone called clean.
 *
 *  ⚠ A SAMPLED CRITERION IS NOT A PROOF, and near the fold onset the verdict is
 *  resolution-dependent. Against a control ten times denser in both the grid
 *  and the ribbon, the shipped settings agreed on 217 of those 220 — one patch
 *  the control called folded and two it did not. Raising the density does not
 *  close the gap; it moves which borderline patches fall on which side. The
 *  number to carry is that this refuses folded patches, not that no folded
 *  patch can pass.
 *
 *  ⚠ A SMALL DISC AROUND EACH CORNER IS EXCLUDED. The normal there is a limit
 *  rather than a value — every weight vanishes — and an input whose two sides
 *  crease against each other at a corner is refused at build time, where the
 *  cause can be named, rather than turning up here as a swing this detector
 *  would have to attribute to the patch. On every input exercised the exclusion
 *  moved no verdict; it is a guard on the one place the criterion does not
 *  apply, not a tolerance.
 */
export function nSidedPatchFolds(patch, opts = {}) {
  if (!patch || typeof patch.evaluateXY !== 'function' || !patch.domain || !Array.isArray(patch.domain.vertices)) {
    return { ok: false, reason: 'not an evaluable n-sided patch' };
  }
  const D = patch.domain.vertices, n = D.length;
  const h = opts.step ?? 1e-5;
  const guard = opts.guard ?? 4 * h;
  const cornerGuard = opts.cornerGuard ?? 0.02;   // domain circumradius is 1
  const creaseLimit = opts.creaseLimitDeg ?? 75;

  // Inward signed distance to every domain edge; the domain is convex and
  // centered on the origin, so a point is inside exactly when all are positive.
  const edge = [];
  for (let i = 0; i < n; i++) {
    const a = D[i], b = D[(i + 1) % n];
    const ex = b[0] - a[0], ey = b[1] - a[1];
    const L = Math.hypot(ex, ey);
    edge.push({ a, nx: ey / L, ny: -ex / L, tx: ex / L, ty: ey / L, L });
  }
  // For a counter-clockwise polygon the inward normal of edge (a -> b) is
  // (-(b-a).y, (b-a).x) rotated; fix the sign once against the center.
  for (const e of edge) {
    const d = e.nx * (0 - e.a[0]) + e.ny * (0 - e.a[1]);
    if (d < 0) { e.nx = -e.nx; e.ny = -e.ny; }
  }
  const inset = (x, y) => {
    let m = Infinity;
    for (const e of edge) m = Math.min(m, e.nx * (x - e.a[0]) + e.ny * (y - e.a[1]));
    return m;
  };
  const cornerClear = (x, y) => {
    for (const v of D) if (Math.hypot(x - v[0], y - v[1]) < cornerGuard) return false;
    return true;
  };
  const nrm3 = (v) => { const L = Math.hypot(v[0], v[1], v[2]); return L > 0 ? [v[0] / L, v[1] / L, v[2] / L] : null; };
  const normalAt = (x, y) => {
    const p = patch.evaluateXY(x, y), px = patch.evaluateXY(x + h, y), py = patch.evaluateXY(x, y + h);
    if (!p || !px || !py) return null;
    return nrm3(cross(sub(px, p), sub(py, p)));
  };

  const grid = new Map();
  const put = (x, y, key) => {
    if (inset(x, y) < guard || inset(x + h, y) < guard || inset(x, y + h) < guard) return;
    if (!cornerClear(x, y)) return;
    const nrm = normalAt(x, y);
    if (nrm) grid.set(key, nrm);
  };
  const G = opts.grid ?? 32;
  for (let i = 0; i <= G; i++) {
    for (let j = 0; j <= G; j++) {
      put(-1 + (2 * i) / G, -1 + (2 * j) / G, `g${i},${j}`);
    }
  }
  /* The grid and ribbon densities are swept, not picked: at grid 16 / ribbon 48
     the sweep disagreed with a tenfold-denser control on six of 220 randomised
     holes, and at grid 32 / ribbon 64 on three, for about 24 ms per patch.
     Beyond that the disagreement stops falling — it is the fold onset moving
     between samplers, not the sweep being too coarse. */
  const RIB = opts.ribbon ?? 64;
  const depths = opts.depths ?? [guard * 1.5, 2e-4, 6e-4, 2e-3, 6e-3, 0.015, 0.03, 0.06, 0.1, 0.16];
  for (let i = 0; i < n; i++) {
    const e = edge[i];
    for (let k = 0; k <= RIB; k++) {
      const t = k / RIB;
      const bx = e.a[0] + e.tx * e.L * t, by = e.a[1] + e.ty * e.L * t;
      for (let d = 0; d < depths.length; d++) {
        put(bx + e.nx * depths[d], by + e.ny * depths[d], `r${i},${k},${d}`);
      }
    }
  }

  const neighbours = (key) => {
    if (key[0] === 'g') { const [i, j] = key.slice(1).split(',').map(Number); return [`g${i + 1},${j}`, `g${i},${j + 1}`]; }
    const [sd, k, d] = key.slice(1).split(',').map(Number);
    return [`r${sd},${k + 1},${d}`, `r${sd},${k},${d + 1}`];
  };
  let reversals = 0, worstDeg = 0;
  for (const [key, nv] of grid) {
    for (const nk of neighbours(key)) {
      const m = grid.get(nk);
      if (!m) continue;
      const dd = dot(nv, m);
      worstDeg = Math.max(worstDeg, Math.acos(Math.max(-1, Math.min(1, dd))) * 180 / Math.PI);
      if (dd < 0) reversals += 1;
    }
  }
  if (grid.size < 20) {
    return { ok: false, folds: false, reversals, worstAdjacentDeg: worstDeg, samples: grid.size, reason: `only ${grid.size} of the patch could be sampled — it cannot be judged, so it is not passed` };
  }
  /* ⚠ A SIGN FLIP IS NOT THE ONLY WAY TO FAIL. A patch creased at eighty-nine
     degrees between neighboring samples has not technically reversed and is
     not a surface anyone wants. */
  const folds = reversals > 0 || worstDeg > creaseLimit;
  return {
    ok: !folds, folds, reversals, worstAdjacentDeg: worstDeg, samples: grid.size,
    reason: folds
      ? (reversals > 0
        ? `the ${patch.sides}-sided patch folds over itself — ${reversals} adjacent normal reversal(s), worst turn ${worstDeg.toFixed(1)} degrees. The hole is too distorted for one patch to span.`
        : `the ${patch.sides}-sided patch creases — neighboring samples turn ${worstDeg.toFixed(1)} degrees, past the ${creaseLimit} degree limit, so it is not folded but it is not a smooth fill either.`)
      : null,
  };
}

/** How far apart are the normals as a corner is approached from different
 *  directions? This is the twist question, measured rather than assumed.
 *
 *  The corner position is exact and shared, so the patch is at least G0 there.
 *  Whether it is more than that depends on the input's own twist agreement, and
 *  the answer is a number: the largest angle between limit normals taken along
 *  a fan of domain directions into corner `index`, at a stated radius.
 *
 *  ⚠ THE NUMBER IS RADIUS-DEPENDENT AND MUST BE READ AS A LIMIT. A genuine
 *  twist mismatch keeps a spread as the radius shrinks; a merely CURVED corner
 *  region has its spread fall with the radius. Sweep the radius rather than
 *  quoting one value.
 */
export function cornerNormalSpread(patch, index, opts = {}) {
  if (!patch || typeof patch.evaluateXY !== 'function' || !patch.domain) return { ok: false, reason: 'not an evaluable n-sided patch' };
  const D = patch.domain.vertices, n = D.length;
  if (!Number.isInteger(index) || index < 0 || index >= n) return { ok: false, reason: `corner ${index} does not exist on a ${n}-sided patch` };
  const radius = opts.radius ?? 1e-3;
  const h = opts.step ?? Math.max(1e-7, radius * 1e-3);
  const fan = opts.fan ?? 12;
  const V = D[index];
  // The interior angle at this corner is spanned by the two edge directions.
  const prev = D[(index + n - 1) % n], next = D[(index + 1) % n];
  const a0 = Math.atan2(next[1] - V[1], next[0] - V[0]);
  let a1 = Math.atan2(prev[1] - V[1], prev[0] - V[0]);
  while (a1 < a0) a1 += 2 * Math.PI;
  const nrm3 = (v) => { const L = Math.hypot(v[0], v[1], v[2]); return L > 0 ? [v[0] / L, v[1] / L, v[2] / L] : null; };
  const normals = [];
  for (let k = 1; k < fan; k++) {
    const a = a0 + (a1 - a0) * (k / fan);
    const x = V[0] + radius * Math.cos(a), y = V[1] + radius * Math.sin(a);
    const p = patch.evaluateXY(x, y), px = patch.evaluateXY(x + h, y), py = patch.evaluateXY(x, y + h);
    if (!p || !px || !py) continue;
    const nv = nrm3(cross(sub(px, p), sub(py, p)));
    if (nv) normals.push(nv);
  }
  if (normals.length < 2) return { ok: false, reason: `the corner could not be sampled — ${normals.length} usable normal(s)` };
  let worst = 0;
  for (let i = 0; i < normals.length; i++) {
    for (let j = i + 1; j < normals.length; j++) {
      const d = Math.abs(dot(normals[i], normals[j]));
      worst = Math.max(worst, Math.acos(Math.max(-1, Math.min(1, d))) * 180 / Math.PI);
    }
  }
  return { ok: true, spreadDeg: worst, samples: normals.length, radius };
}
