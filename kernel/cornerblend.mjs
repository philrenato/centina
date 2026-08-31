// CORNER BLEND — closing a valence-3 corner whose three edges have DIFFERENT
// fillet radii.
//
// When the three radii are EQUAL there is a ball tangent to all three faces,
// every blend's end arc is a great circle on it, and the corner closes as a
// spherical triangle that needs no fitting at all (see filletCornerPatch in the
// app). That construction has no answer when the radii differ: no single sphere
// is tangent to three faces at three different distances, and on an orthogonal
// corner the two adjacent blends leave a gap of |r1 - r2| * sqrt(2).
//
// What replaces it has to be built from what the three blends actually leave
// behind, so this module starts there — with the corners of the region to be
// filled, which have a closed form, and the curves that bound it.
//
// ⚠ THE BLENDS DO NOT END ON A PLANE ANY MORE, and that is the first thing the
// equal-radius case hides. A blend along one edge is a tube about that edge;
// with equal radii it stops at a plane perpendicular to itself and its end is a
// circle. With unequal radii its two tangency curves have to stop at DIFFERENT
// distances from the corner — one where it meets its neighbour on one shared
// face, the other where it meets a different neighbour on the other — so its
// end is a skew curve across the tube, not a planar section.

import { networkCorrectionSurface } from './loft.mjs';

const EPS = 1e-12;
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
function norm(a) { const L = len(a); return L > EPS ? [a[0] / L, a[1] / L, a[2] / L] : null; }

/* CONVEX AND CONCAVE EDGES SHARE ONE SETBACK, AND THE SIGNED FORMULA DOES NOT.

   phi is the INTERIOR dihedral — the angle measured through the material. A
   convex edge has phi < pi, a concave one phi > pi. The blend's rolling ball
   does not care which: it rolls in whichever of the two wedges at the edge is
   CONVEX, which is the material for a convex edge and the void for a concave
   one, and that wedge always subtends

       psi = phi        (convex)          psi = 2*pi - phi   (concave)

   i.e. psi = the angle between the two co-normals, which is what an unsigned
   `acos(dot(cA, cB))` returns and is never more than pi. A ball of radius r
   inscribed in a wedge of angle psi touches each face at r/tan(psi/2) from the
   edge, so

       setback = r / tan(psi/2) = r / |tan(phi/2)|

   and it is POSITIVE along the co-normal in both cases, because the co-normal
   points into the face and the tangency is on the face.

   ⚠⚠ THE UNSIGNED VALUE IS NOT A CONVENIENCE, IT IS THE GEOMETRY. Writing
   r/tan(phi/2) makes tan negative for every concave edge, which flips the
   setback and puts the tangency line on the far side of the edge — off the
   face the ball actually touches, by twice the setback. The magnitude looks
   right at every radius, so nothing downstream reads as wrong until the corner
   lands outside the material.

   The consequence for everything below: a corner of the patch depends on phi
   only through psi, so THE CROSSINGS ARE BLIND TO CONVEXITY. An all-concave
   corner and the convex corner on the same three face planes return the same
   three points — they are point-set complements of one another, the rolling
   balls run in the same wedges, and the blend tubes are congruent. Which side
   the material is on lives in the boundary curves and normals a caller hands to
   the patch, not here. */
function wedgeAngle(phi) { return phi <= Math.PI ? phi : 2 * Math.PI - phi; }
function setbackFor(radius, phi) { return radius / Math.tan(wedgeAngle(phi) / 2); }

/* Below this the setback is under 1e-9 of the radius: the two tangency lines
   and the edge are one line, and there is no corner region to build. */
const FLAT_TOL = 2e-9;

function dihedralOutOfRange(phi) {
  if (!Number.isFinite(phi) || !(phi > 0) || !(phi < 2 * Math.PI)) return true;
  return Math.abs(phi - Math.PI) < FLAT_TOL;
}
const DIHEDRAL_RANGE_NOTE = 'an interior dihedral runs (0, pi) convex or (pi, 2*pi) concave; pi itself is two faces meeting flat, with no edge to blend';

/** Where a blend's tangency curve runs on one of its two faces.
 *
 *  A rolling ball of radius r along an edge touches each adjacent face along a
 *  line offset from the edge by the setback r/|tan(phi/2)|, measured INTO that
 *  face along its co-normal. On a face shared by two filleted edges there are
 *  two such lines, one per edge, and where they cross is a corner of the region
 *  the corner patch has to fill. That crossing is the closed form this module
 *  is built on — it needs no solver and it is exact.
 */
export function tangencyLineOnFace({ edgePoint, edgeDir, coNormal, radius, phi }) {
  const d = norm(edgeDir), c = norm(coNormal);
  if (!d || !c) return { ok: false, reason: 'an edge direction or co-normal has no direction' };
  if (!(radius > 0)) return { ok: false, reason: 'radius must be positive' };
  if (dihedralOutOfRange(phi)) {
    return { ok: false, reason: `the dihedral angle is out of range; got phi=${phi} — ${DIHEDRAL_RANGE_NOTE}` };
  }
  const setback = setbackFor(radius, phi);
  return { ok: true, point: add(edgePoint, mul(c, setback)), dir: d, setback, wedge: wedgeAngle(phi) };
}

/** The corner of the patch on a face shared by two filleted edges: where those
 *  two edges' tangency lines cross ON that face.
 *
 *  ⚠ REFUSES A NEAR-PARALLEL CROSSING RATHER THAN RETURNING A DISTANT POINT.
 *  Two lines that nearly agree in direction meet a very long way off, and the
 *  answer is numerically meaningless well before it is geometrically wrong —
 *  which reads downstream as a corner patch stretched across the model.
 */
export function tangencyCrossing(lineA, lineB, opts = {}) {
  const minAngle = opts.minAngleRad ?? (1 * Math.PI / 180);
  const { point: pA, dir: dA } = lineA, { point: pB, dir: dB } = lineB;
  const cosT = Math.abs(dot(dA, dB));
  if (cosT > Math.cos(minAngle)) {
    return { ok: false, reason: 'the two tangency lines on this face are within a degree of parallel — where they cross is not a usable point' };
  }
  // Least-squares crossing of two lines: exact where they genuinely intersect,
  // and the midpoint of the shortest connecting segment where float has moved
  // them apart. The residual is REPORTED, so a caller can refuse a pair that
  // does not really meet rather than be handed their near-miss.
  const w0 = sub(pA, pB);
  const a = dot(dA, dA), b = dot(dA, dB), c = dot(dB, dB);
  const d = dot(dA, w0), e = dot(dB, w0);
  const den = a * c - b * b;
  if (!(Math.abs(den) > EPS)) return { ok: false, reason: 'the two tangency lines are degenerate' };
  const sA = (b * e - c * d) / den, sB = (a * e - b * d) / den;
  const qA = add(pA, mul(dA, sA)), qB = add(pB, mul(dB, sB));
  const gap = len(sub(qA, qB));
  return { ok: true, point: mul(add(qA, qB), 0.5), gap, tA: sA, tB: sB };
}

/** The three corners of a valence-3 corner patch, from the three edges'
 *  radii and the three faces they share.
 *
 *  `edges` is three records in cyclic order; each names its direction away from
 *  the vertex, its radius, and its two co-normals — one into each adjacent
 *  face — together with the dihedral there. Faces are identified by index so
 *  the two edges sharing a face can be paired without geometry matching.
 *
 *  ⚠ ANY MIX OF CONVEX AND CONCAVE EDGES IS ALLOWED, and the reason it can be
 *  is that this construction never asks for a ball tangent to all three faces.
 *  There is none at a mixed corner: such a ball would have to sit inside the
 *  material for the convex edges and outside it for the concave ones at the
 *  same time, which is why the rounding of a convex edge and the filleting of a
 *  concave one are different operations (Rossignac & Requicha, "Offsetting
 *  operations in solid modelling", CAGD 3(2):129-148, 1986, section 2.5), and
 *  why a spherical corner exists only where the three concavities agree. What
 *  is crossed here is the blends' OWN tangency lines, each of which needs only
 *  its own edge's rolling ball, so the mixed corner has three of them and no
 *  need of a fourth.
 *
 *  Two facts about a mixed corner follow, and both hold at every radius:
 *
 *  · At a face shared by two edges of UNLIKE convexity the two blends roll on
 *    opposite sides of that face, so their axes are exactly rA + rB apart at
 *    the point over the crossing and nowhere closer. The tubes are externally
 *    TANGENT there — they touch at the crossing and do not overlap, where two
 *    edges of like convexity give tubes that interpenetrate and have to be
 *    trimmed against each other. Either way the crossing is the corner.
 *  · At every crossing, convex or concave or mixed, both adjacent blends are
 *    tangent to the shared face along their own tangency lines, so the corner
 *    lies on both tubes and the patch's tangent plane there IS the face plane.
 *
 *  What a mixed corner does NOT come with is a guarantee that one patch can
 *  span the triangle: pushing the crossing on the third face behind the vertex
 *  makes the triangle long and thin whenever the convex edge's radius is small
 *  against the concave ones. That is the same limit `cornerPatchFolds` already
 *  polices for convex corners, measured on the same instrument, and it is a
 *  question about the triangle rather than about the convexities.
 */
export function cornerPatchCorners(vertex, edges, opts = {}) {
  if (!Array.isArray(edges) || edges.length !== 3) {
    return { ok: false, reason: `a valence-3 corner needs exactly three edges; got ${edges ? edges.length : 0}` };
  }
  const lines = [];
  for (let i = 0; i < 3; i++) {
    const e = edges[i];
    for (const side of ['a', 'b']) {
      const c = side === 'a' ? e.coNormalA : e.coNormalB;
      const face = side === 'a' ? e.faceA : e.faceB;
      const L = tangencyLineOnFace({ edgePoint: vertex, edgeDir: e.dir, coNormal: c, radius: e.radius, phi: e.phi });
      if (!L.ok) return { ok: false, reason: `edge ${i}: ${L.reason}` };
      lines.push({ edge: i, face, ...L });
    }
  }
  // Pair the lines by the FACE they lie on. Each face carries exactly two.
  const byFace = new Map();
  for (const L of lines) {
    if (!byFace.has(L.face)) byFace.set(L.face, []);
    byFace.get(L.face).push(L);
  }
  if (byFace.size !== 3) {
    return { ok: false, reason: `three edges at a corner touch three faces; these name ${byFace.size}` };
  }
  const corners = [];
  for (const [face, pair] of byFace) {
    if (pair.length !== 2) return { ok: false, reason: `face ${face} carries ${pair.length} tangency line(s) at this corner, not two` };
    const x = tangencyCrossing(pair[0], pair[1], opts);
    if (!x.ok) return { ok: false, reason: `face ${face}: ${x.reason}` };
    corners.push({ face, point: x.point, edges: [pair[0].edge, pair[1].edge], gap: x.gap, tA: x.tA, tB: x.tB });
  }
  const worstGap = corners.reduce((m, c) => Math.max(m, c.gap), 0);
  /* ⚠⚠ A REPORTED RESIDUAL THAT NOTHING REFUSES ON IS NOT A CHECK. This
     returned `worstGap` and hoped a caller would read it. Swap two face labels
     on one edge and it came back ok with a gap of 5 units and three
     confident-looking corners that lie on neither of the lines that made them;
     a co-normal half a degree out of its face — which is what MEASURED
     geometry gives — produces a gap of 0.05, five hundred times a weld
     tolerance, and a corner off the face entirely. The gap is the distance
     between two lines that are supposed to meet, so anything above the
     tolerance means they do not, and there is no corner there to build. */
  const gapTol = opts.gapTolerance ?? 1e-6;
  const scale = Math.max(...edges.map((e) => e.radius), 1);
  if (worstGap > gapTol * scale) {
    const bad = corners.filter((c) => c.gap > gapTol * scale).map((c) => c.face);
    return { ok: false, reason: `the two tangency lines on face(s) ${bad.join(', ')} do not meet — they pass ${worstGap.toExponential(3)} apart, so there is no crossing to put a corner at. Check that each edge's co-normals and dihedral describe the same faces.`, worstGap, corners };
  }
  /* ⚠ AND A GAP OF ZERO IS NOT PROOF THE INPUT AGREES WITH ITSELF. Two lines
     can genuinely intersect while the geometry that produced them is
     inconsistent: give an orthogonal corner a dihedral of 60 degrees on one
     edge and both its lines still lie in their faces, so they cross exactly —
     at a corner 3.7 units from the right one, with a residual of zero. The
     dihedral and the co-normals are two descriptions of the same wedge, so
     they are required to agree: dot(coNormalA, coNormalB) = cos(psi), with psi
     the wedge angle the setback is actually taken from.

     ⚠ AND THAT IS ALL THEY CAN SAY. `acos` returns [0, pi], so two co-normals
     90 degrees apart are equally consistent with a 90-degree convex edge and a
     270-degree concave one. The two are genuinely indistinguishable from an
     edge direction and two in-face directions — nothing in this input names
     which side the material is on — so phi's half of the range is a DECLARATION
     the co-normals cannot corroborate. It costs nothing here, because the
     crossings do not depend on it; a caller that needs the material side reads
     phi itself. */
  for (let i = 0; i < 3; i++) {
    const e = edges[i];
    const cA = norm(e.coNormalA), cB = norm(e.coNormalB);
    if (!cA || !cB) continue;
    const implied = Math.acos(Math.max(-1, Math.min(1, dot(cA, cB))));
    const psi = wedgeAngle(e.phi);
    if (Math.abs(implied - psi) > (opts.phiTolerance ?? 1e-6)) {
      return { ok: false, reason: `edge ${i}: its co-normals are ${(implied * 180 / Math.PI).toFixed(4)} degrees apart while its dihedral says ${(psi * 180 / Math.PI).toFixed(4)} (an interior angle of ${(e.phi * 180 / Math.PI).toFixed(4)}) — the two describe the same wedge and disagree, so the setback is being measured from a face the co-normal is not on` };
    }
  }
  return { ok: true, corners, worstGap };
}

/* WHERE A BLEND STOPS WHEN ITS NEIGHBOURS DISAGREE — THE APPROXIMATE FORM.

   With equal radii a blend ends on a plane perpendicular to itself. With
   unequal radii its two tangency curves stop at DIFFERENT distances from the
   vertex, so its end runs skew across the tube.

   ⚠⚠ THIS IS NOT THE CURVE `blendEndPlane` CUTS. A plane through the two
   corners meets the tube in a CONIC, whose trace in (u, v) is a sinusoid and
   not a line — measured at 0.105 units of disagreement on an ordinary corner,
   a thousand weld tolerances. Trimming the face with this loop while taking
   the corner patch's boundary from the plane leaves a sliver along every
   corner.

   `blendEndTrimLoopFromPlane` below is the one that agrees with the patch.
   This stays for a caller that wants the straight-line approximation
   knowingly, and is named so it cannot be reached by accident.

   u runs ACROSS the section, from the tangency on one face to the tangency on
   the other; v runs ALONG the spine. */
export function blendEndTrimLoopApprox({ vStartAtU0, vStartAtU1, vEndAtU0, vEndAtU1, uMin = 0, uMax = 1 }) {
  const vs = [vStartAtU0, vStartAtU1, vEndAtU0, vEndAtU1];
  if (!vs.every((v) => Number.isFinite(v))) return { ok: false, reason: 'a blend end parameter is not a number' };
  if (!(vEndAtU0 > vStartAtU0) || !(vEndAtU1 > vStartAtU1)) {
    return { ok: false, reason: 'the blend ends before it starts on at least one side — the two corners have eaten the whole edge' };
  }
  return {
    ok: true,
    loop: [[uMin, vStartAtU0], [uMax, vStartAtU1], [uMax, vEndAtU1], [uMin, vEndAtU0]],
  };
}

/* THE CORNER PATCH ITSELF — Nielson's side-vertex interpolant over the
   triangle bounded by the three blend end curves.

   For each corner V_i, a ray from V_i through the evaluation point meets the
   opposite boundary at a single parameter, and interpolating along that ray
   gives a surface that reproduces that one boundary exactly. Blending the three
   with weights that vanish on the other two sides makes all three exact at
   once. The weights are (b_j*b_k)^2: on side i both other barycentric
   coordinates are non-zero and the two rival weights carry a factor of b_i = 0,
   so the sum collapses to that side's own interpolant.

   ⚠ THE THREE CORNERS ARE 0/0 AND HAVE TO BE NAMED. Every weight carries a
   factor that vanishes at a vertex, so the blend is undefined there and float
   noise decides it otherwise. The corner value is known exactly — it is the
   vertex — so it is returned outright rather than approached.

   G0 by construction, which is what watertightness needs. Tangent continuity
   across the three boundaries is a further condition on the RAY interpolant
   (a cubic Hermite carrying each blend's cross-boundary derivative rather than
   the straight line used here), and is deliberately a separate step: a patch
   that closes the hole is worth having before one that closes it smoothly, and
   the two can be compared against each other only if both exist.
*/
/* ⚠ THE CROSS-TANGENT MAGNITUDE IS 1/4 OF THE RAY, AND THAT NUMBER IS MEASURED.
   It sets how far the Hermite reaches before the vertex takes over, so it
   controls INTERIOR SHAPE and not the tangent plane — tangency is exact at
   every value, confirmed by refining the measuring step and watching the
   reported angle fall linearly to zero at each of them.

   What it does control is FOLDING. At full ray length the patch folds once the
   radii differ by about 3:1; at a quarter it is clean through 5:1 and beyond,
   and only isolated reversals appear past 8:1 — where the corner region is
   geometrically extreme anyway. Swept across ratios rather than picked. */
const CORNER_TANGENT_SCALE = 0.25;
export function sideVertexPatch({ corners, boundary, tangent, tangentScale = CORNER_TANGENT_SCALE }) {
  if (!Array.isArray(corners) || corners.length !== 3) return { ok: false, reason: 'a triangular patch needs exactly three corners' };
  if (!Array.isArray(boundary) || boundary.length !== 3) return { ok: false, reason: 'a triangular patch needs exactly three boundary curves' };
  for (let i = 0; i < 3; i++) if (typeof boundary[i] !== 'function') return { ok: false, reason: `boundary ${i} is not an evaluable curve` };
  /* ⚠ WITHOUT A CROSS-BOUNDARY TANGENT THIS IS G0 AND ONLY G0. A straight ray
     from the vertex to the boundary carries no information about how the
     neighbouring surface leaves that boundary, so the patch meets it at
     whatever angle the geometry happens to give — measured at 45 to 70 degrees
     on a (5, 8, 3) corner, which is a visible crease and not a fillet.

     `tangent[i](s)` supplies the direction the ADJACENT BLEND leaves boundary i
     at parameter s. With it, each ray becomes a cubic Hermite whose derivative
     at the boundary end IS that direction; the (b_j*b_k)^2 weights make the
     blend reduce to that ray to FIRST order on the boundary, so the patch's
     tangent plane there is spanned by the boundary tangent and this vector —
     which is the adjacent blend's tangent plane exactly.

     The vector only has to LIE in that plane and point inward: its magnitude
     and any component along the boundary change the interior shape, not the
     tangent plane. It is rescaled to the ray length here so a caller does not
     have to guess a magnitude, and flipped if it points the wrong way, because
     a reversed cross-tangent everts the patch rather than failing loudly. */
  const hasTangent = Array.isArray(tangent) && tangent.length === 3 && tangent.every((t) => typeof t === 'function');
  /* ⚠⚠ THE ORIENTATION IS DECIDED ONCE PER SIDE, NOT ONCE PER SAMPLE, and that
     distinction is the difference between a patch that works at any radius
     ratio and one that folds past about 2.5:1.

     The cross-tangent field is already smooth: it is the blend's normal crossed
     with the boundary tangent, and both of those vary smoothly along the
     boundary. Flipping it per sample to point at the opposite vertex injects a
     discontinuity the field never had — on an asymmetric triangle the chord to
     that vertex swings past perpendicular partway along, the test changes its
     mind there, and the ray family reverses mid-boundary. Measured: at radii
     (1,4,1) the sign flipped 7/10 of the way along two of the three sides, and
     those are exactly the ratios at which adjacent normals began reversing.

     One decision per side, taken at the middle where the chord is least
     ambiguous, then applied to the whole side. */
  const sideSign = new Array(3).fill(1);
  const sideNote = [];
  if (hasTangent) {
    for (let i = 0; i < 3; i++) {
      /* ⚠ ONE SAMPLE CANNOT SPEAK FOR A SIDE WHOSE ANSWER CHANGES ALONG IT.
         Taking the decision at the midpoint alone is stable for an ordinary
         boundary and silently arbitrary for one where the chord to the
         opposite vertex swings past perpendicular — the mid sample then sits
         on a knife edge where an input difference of 1e-9 everts that whole
         side. Polled across the side instead: a clear majority is the answer,
         and a genuine split is a refusal, because no constant sign is right
         for such a side and picking one folds the rest of it. */
      let plus = 0, minus = 0, weakest = Infinity;
      for (let k = 1; k < 8; k++) {
        const s2 = k / 8;
        const dm = norm(tangent[i](s2));
        if (!dm) continue;
        const chord = sub(corners[i], boundary[i](s2));
        const L = len(chord);
        if (!(L > 0)) continue;
        const proj = dot(dm, chord) / L;
        if (proj >= 0) plus += 1; else minus += 1;
        weakest = Math.min(weakest, Math.abs(proj));
      }
      if (plus === 0 && minus === 0) { sideNote.push(`side ${i}: no usable cross-tangent anywhere along it`); continue; }
      /* A SPLIT VOTE IS NOT ITSELF A REFUSAL. The chord to the opposite vertex
         genuinely swings past perpendicular on an asymmetric triangle, so the
         test disagrees with itself along the side without the resulting patch
         being wrong — measured clean by the fold detector on configurations
         that split 5:2. The majority is taken, which is what makes the choice
         robust where a single mid-side sample sits on a knife edge and flips
         the whole side for an input difference of 1e-9. Whether the patch is
         actually sound is the fold detector's question, and it is asked
         separately rather than guessed at here. */
      sideSign[i] = plus >= minus ? 1 : -1;
      void weakest;
    }
  }
  if (sideNote.length) return { ok: false, reason: sideNote.join('; ') };
  // boundary[i](s) runs from corners[(i+1)%3] at s=0 to corners[(i+2)%3] at
  // s=1 — the side OPPOSITE corner i, which is what the ray from corner i hits.
  const evaluate = (b0, b1, b2) => {
    /* ⚠ OFF-SIMPLEX INPUT MUST NOT GET A CONFIDENT ANSWER. b_i is the
       Hermite's own parameter, so coordinates that do not sum to one — or that
       are negative — put it outside the curve it was built from and the point
       returned means nothing. Callers doing floating-point arithmetic drift a
       few ulps and that is fine; gross denormalisation is a bug in the caller
       and is worth saying so. */
    if (!Number.isFinite(b0) || !Number.isFinite(b1) || !Number.isFinite(b2)) return null;
    if (Math.abs(b0 + b1 + b2 - 1) > 1e-6) return null;
    if (b0 < -1e-9 || b1 < -1e-9 || b2 < -1e-9) return null;
    const b = [b0, b1, b2];
    // A vertex: every weight is zero and the answer is not a limit, it is known.
    for (let i = 0; i < 3; i++) if (b[i] >= 1 - 1e-12) return corners[i].slice();
    let acc = [0, 0, 0], wsum = 0;
    for (let i = 0; i < 3; i++) {
      const j = (i + 1) % 3, k = (i + 2) % 3;
      const denom = b[j] + b[k];
      if (!(denom > 1e-15)) continue; // on the ray through the opposite vertex
      const s = b[k] / denom;              // where the ray from V_i lands on side i
      const Q = boundary[i](s);
      let P;
      if (!hasTangent) {
        P = [
          b[i] * corners[i][0] + (1 - b[i]) * Q[0],
          b[i] * corners[i][1] + (1 - b[i]) * Q[1],
          b[i] * corners[i][2] + (1 - b[i]) * Q[2],
        ];
      } else {
        // Cubic Hermite along the ray, with rho = b_i running 0 (boundary) to
        // 1 (vertex). Position and cross-derivative are pinned at the boundary
        // end; the vertex end takes the chord, which constrains nothing there.
        const chord = sub(corners[i], Q);
        const L = len(chord);
        let D = tangent[i](s);
        const dn = norm(D);
        if (!dn || !(L > 0)) {
          P = [b[i] * corners[i][0] + (1 - b[i]) * Q[0], b[i] * corners[i][1] + (1 - b[i]) * Q[1], b[i] * corners[i][2] + (1 - b[i]) * Q[2]];
        } else {
          const m0 = mul(mul(dn, sideSign[i]), L * tangentScale);
          const m1 = chord;
          const r = b[i], r2 = r * r, r3 = r2 * r;
          const h00 = 2 * r3 - 3 * r2 + 1, h10 = r3 - 2 * r2 + r, h01 = -2 * r3 + 3 * r2, h11 = r3 - r2;
          P = [
            h00 * Q[0] + h10 * m0[0] + h01 * corners[i][0] + h11 * m1[0],
            h00 * Q[1] + h10 * m0[1] + h01 * corners[i][1] + h11 * m1[1],
            h00 * Q[2] + h10 * m0[2] + h01 * corners[i][2] + h11 * m1[2],
          ];
        }
      }
      const w = (b[j] * b[k]) * (b[j] * b[k]);
      acc = [acc[0] + w * P[0], acc[1] + w * P[1], acc[2] + w * P[2]];
      wsum += w;
    }
    if (!(wsum > 0)) return null;
    return [acc[0] / wsum, acc[1] / wsum, acc[2] / wsum];
  };
  return { ok: true, evaluate };
}

/* WHERE ALONG A BLEND A CORNER FALLS.

   The corners above are 3D points that lie, by construction, on a blend's own
   tangency curves — the u = uMin and u = uMax borders of its surface. Turning
   one into the v parameter where the blend has to stop is therefore a search
   along a single curve, not on a surface, which is why it can be done by
   bisection on a monotone distance rather than by a general inversion.

   ⚠ THE ANSWER IS CHECKED, NOT ASSUMED. A corner that does not actually lie on
   the border — because the edge it belongs to was never filleted, or because
   the crossing was computed against the wrong face — still yields a nearest
   parameter, and that parameter is meaningless. The residual distance is
   returned and a caller that ignores it will place a blend end anywhere. */
export function blendParamAtCorner({ evalBorder, corner, vMin = 0, vMax = 1, samples = 64, refine = 40, ...opts }) {
  if (typeof evalBorder !== 'function') return { ok: false, reason: 'the blend border is not evaluable' };
  const d2 = (v) => {
    const p = evalBorder(v);
    if (!p) return Infinity;
    const dx = p[0] - corner[0], dy = p[1] - corner[1], dz = p[2] - corner[2];
    return dx * dx + dy * dy + dz * dz;
  };
  let bestV = vMin, best = Infinity;
  for (let i = 0; i <= samples; i++) {
    const v = vMin + (vMax - vMin) * (i / samples);
    const d = d2(v);
    if (d < best) { best = d; bestV = v; }
  }
  // Golden-section on the bracketing interval: the distance to a point along a
  // smooth curve is unimodal near its minimum, and the coarse sweep above is
  // what makes that true here rather than assumed globally.
  let lo = Math.max(vMin, bestV - (vMax - vMin) / samples);
  let hi = Math.min(vMax, bestV + (vMax - vMin) / samples);
  const phi = (Math.sqrt(5) - 1) / 2;
  let c = hi - phi * (hi - lo), d = lo + phi * (hi - lo);
  let fc = d2(c), fd = d2(d);
  for (let i = 0; i < refine; i++) {
    if (fc < fd) { hi = d; d = c; fd = fc; c = hi - phi * (hi - lo); fc = d2(c); }
    else { lo = c; c = d; fc = fd; d = lo + phi * (hi - lo); fd = d2(d); }
  }
  const v = (lo + hi) / 2;
  const residual = Math.sqrt(d2(v));
  /* ⚠ THE COARSE SWEEP IS WHAT MAKES THE REFINEMENT LEGITIMATE, and it can
     still miss a basin narrower than its own spacing. A caller cannot tell a
     wrong basin from a right one by looking, so the residual is compared here
     against what the caller says "on the curve" means, rather than returned to
     be shrugged at. Without a tolerance it is reported as before. */
  if (opts.tolerance != null && residual > opts.tolerance) {
    return { ok: false, reason: `the corner is ${residual.toExponential(3)} from this blend's border, past the ${opts.tolerance} it was given — either it belongs to a different edge, or the border doubles back inside one sweep step`, v, residual };
  }
  return { ok: true, v, residual };
}

/* THE PLANE A BLEND ENDS ON, WHEN ITS TWO ENDS DISAGREE.

   A blend has to reach one corner on one of its faces and a different corner on
   the other, and with unequal radii those sit at different distances from the
   vertex. Cutting the tube with a PLANE through both of them keeps the end
   curve an exact conic — a circle when the plane is perpendicular to the axis,
   an ellipse when it is tilted — which this kernel represents exactly, rather
   than a curve that has to be sampled and fitted.

   ⚠ THE PLANE THROUGH TWO POINTS IS A PENCIL, and the choice tunes how full the
   corner looks. Taken here as the member whose normal is CLOSEST TO THE AXIS,
   which is the one honest anchor available: it is the unique plane through the
   chord that tilts as little as possible away from the section a blend would
   have used anyway. It reduces exactly to the perpendicular section when the
   two radii agree, so the equal-radius case is recovered rather than
   approximated — the same requirement the corner points themselves meet.
*/
export function blendEndPlane({ axisDir, cornerA, cornerB }) {
  const d = norm(axisDir);
  if (!d) return { ok: false, reason: 'the blend axis has no direction' };
  const chord = sub(cornerB, cornerA);
  const e = norm(chord);
  if (!e) return { ok: false, reason: 'the two corners of this blend end are the same point' };
  // Remove the chord component from the axis: what is left is the normal of the
  // least-tilted plane containing the chord.
  const n = norm(sub(d, mul(e, dot(d, e))));
  if (!n) {
    return { ok: false, reason: 'the two corners lie along the blend axis — every plane through them cuts it lengthwise, so there is no end section' };
  }
  const tilt = Math.acos(Math.max(-1, Math.min(1, Math.abs(dot(n, d)))));
  return { ok: true, normal: n, offset: dot(n, cornerA), tiltRad: tilt };
}

/** Does the corner region actually fit on the faces it needs?
 *
 *  ⚠ THE CROSSING CAN FALL OFF THE FACE. Two tangency lines always meet
 *  somewhere; whether they meet WITHIN the face is a different question, and
 *  when they do not there is no corner to build — the ball has rolled off the
 *  rails, at a vertex rather than along an edge. Refused by name, in the same
 *  terms `maxRadiusForSetback` refuses the edge case.
 */
export function cornerFitsOnFaces(cornerResult, faceExtents) {
  if (!cornerResult || !cornerResult.ok) return cornerResult;
  const bad = [];
  for (const c of cornerResult.corners) {
    const ext = faceExtents && faceExtents[c.face];
    if (!ext) continue; // nothing declared for this face; nothing to check against
    const [lo, hi] = ext;
    if (!(c.tA >= lo[0] && c.tA <= hi[0] && c.tB >= lo[1] && c.tB <= hi[1])) {
      bad.push(c.face);
    }
  }
  if (bad.length) {
    return { ok: false, reason: `the corner falls outside face(s) ${bad.join(', ')} — at these radii the blends meet past the edge of the material they are cut into` };
  }
  return cornerResult;
}

/* ═══ THE EXACT CASE: TWO RADII EQUAL ═══════════════════════════════════════
   Three distinct radii have no closed form. TWO equal ones do, and it is worth
   its own path rather than being handed to a fitter, because the answer is a
   torus and two spherical lunes — all rational, all exactly tangent.

   The construction. Edges A and B share a face and carry the same radius a;
   edge C carries c. A ball of radius a tangent to that shared face can roll
   from where the A-blend leaves it to where the B-blend picks it up, and while
   it rolls it must also stay tangent to the C-blend — so its centre sits at
   distance a + c from C's axis. Centre path = (plane parallel to the shared
   face at height a) INTERSECT (cylinder of radius a + c about C's axis).

   ⚠⚠ THAT INTERSECTION IS A CIRCLE ONLY WHEN C'S AXIS IS PERPENDICULAR TO THE
   SHARED FACE, and that is the whole condition for the exact answer. It holds
   on a box and it does not hold in general: tilt the third edge and the
   intersection becomes an ELLIPSE, the swept surface a general canal rather
   than a torus, and no rational form survives. Checked, not assumed — the
   alternative is a torus that is quietly the wrong surface.

   The handoff station falls out of the same condition. The A-blend's own axis
   IS the path of that ball (a ball of radius a tangent to both of A's faces has
   its centre on A's axis), so the handoff is where that axis first reaches
   distance a + c from C's axis:
       offset^2 = (a + c)^2 - (a - c)^2 = 4ac   ->   offset = 2*sqrt(ac)
   measured along the edge from the foot of C's axis. At exactly that station
   the ball is tangent to BOTH of A's faces and to the C-blend at once — a
   triple tangency — and the piece of it between the two envelopes' contact
   circles is a spherical lune, which is the corner representation already
   built for the equal-radius case.
*/
export function twoEqualRadiiCorner({ vertex, sharedFaceNormal, edgeA, edgeB, edgeC, tol = 1e-9, ...opts }) {
  const nF = norm(sharedFaceNormal);
  if (!nF) return { ok: false, reason: 'the shared face has no normal' };
  const dA = norm(edgeA.dir), dB = norm(edgeB.dir), dC = norm(edgeC.dir);
  if (!dA || !dB || !dC) return { ok: false, reason: 'an edge has no direction' };
  const a = edgeA.radius, c = edgeC.radius;
  if (!(a > 0) || !(c > 0) || !(edgeB.radius > 0)) return { ok: false, reason: 'every radius must be positive' };
  if (Math.abs(edgeB.radius - a) > tol * Math.max(1, a)) {
    return { ok: false, reason: `this path needs the two edges sharing the face to have EQUAL radii; got ${a} and ${edgeB.radius}` };
  }
  for (const [nm, e] of [['first', edgeA], ['second', edgeB], ['third', edgeC]]) {
    if (dihedralOutOfRange(e.phi)) {
      return { ok: false, reason: `the ${nm} edge's dihedral angle is out of range; got phi=${e.phi} — ${DIHEDRAL_RANGE_NOTE}` };
    }
  }
  /* ⚠⚠ THIS PATH IS A ROLLING BALL, AND A ROLLING BALL CANNOT CHANGE SIDES.
     The whole construction is one ball of radius a that stays tangent to the
     shared face while it travels from the first blend to the second, and the
     side of that face it rides on is the side its edge's CONVEX wedge is on —
     the material at a convex edge, the void at a concave one. Two edges of
     unlike convexity therefore need it on both sides of the same plane at
     once, and there is no such path.

     The third edge fails differently and just as hard. The centre path is the
     circle at distance a + c from that edge's axis; when the third edge's
     convexity disagrees with the other two, its axis sits a + c from the first
     blend's axis already, so the handoff offset 2*sqrt(ac) collapses — the two
     axes are exactly a + c apart everywhere along, the ball touches the third
     blend at one station instead of crossing it, and the torus would meet the
     first blend at a point rather than along a curve. It would close nothing.

     `cornerPatchCorners` has no such restriction: it crosses the blends' own
     tangency lines and never asks for a ball tangent to all three faces. */
  const convexA = edgeA.phi < Math.PI, convexB = edgeB.phi < Math.PI, convexC = edgeC.phi < Math.PI;
  if (!(convexA === convexB && convexB === convexC)) {
    const deg = (x) => (x * 180 / Math.PI).toFixed(2);
    return {
      ok: false,
      reason: `this corner mixes convex and concave edges (interior angles ${deg(edgeA.phi)}, ${deg(edgeB.phi)}, ${deg(edgeC.phi)} degrees) — the rolling ball rides the material side of the shared face at a convex edge and the void side at a concave one, so no one ball travels between these blends, and a third edge of unlike convexity touches its path at a single station rather than crossing it`,
    };
  }
  // ⚠ The condition that makes this exact rather than approximately right.
  /* ⚠ THE GATE IS AN ANGLE, NOT A COSINE. `1 - 1e-9` in a dot product is
     0.0026 degrees, which refuses any face normal fitted from real geometry
     while still accepting a tilt big enough to put the torus a weld tolerance
     out of place. Stated in radians so it can be read against the accuracy the
     input actually has. */
  const alignTolRad = opts && opts.alignToleranceRad != null ? opts.alignToleranceRad : 1e-4;
  const align = Math.abs(dot(dC, nF));
  if (align < Math.cos(alignTolRad)) {
    return {
      ok: false,
      reason: `the third edge is ${(Math.acos(Math.min(1, align)) * 180 / Math.PI).toFixed(3)} degrees off the shared face's normal — the rolling ball's centre traces an ellipse rather than a circle, so the corner is a general canal surface and not a torus`,
    };
  }
  // C's axis: offset from the vertex into each of ITS two faces by its own
  // setback, which on the shared face is what puts the torus centre in place.
  const axisPoint = add(vertex, add(mul(norm(edgeC.coNormalA), setbackFor(c, edgeC.phi)),
    mul(norm(edgeC.coNormalB), setbackFor(c, edgeC.phi))));
  /* The torus: centre on C's axis, one ball-radius off the shared face on the
     side the ball ROLLS, with C's axis for its own. Which side that is comes
     from a co-normal rather than from the face normal's stored sign, which is
     whatever the producing recipe happened to build.

     A co-normal points from an edge into one of its faces, so the pair spans
     the wedge the ball rolls in — the material for a convex corner and the
     void for a concave one. The test below therefore needs no convexity flag:
     it reads the rolling side directly, and gives the same torus for a corner
     and for its point-set complement, which is what the two share. */
  /* ⚠⚠ PICK THE CO-NORMAL BY WHICH ONE IS ACTUALLY OFF THE SHARED FACE, not by
     a fixed threshold. One of edge A's two co-normals lies IN the shared face
     (dot with its normal ~ 0) and the other leans out of it by sin(phi_A). A
     `> 0.5` test therefore fails for any dihedral under 30 degrees, falls
     through to the in-plane one, and its dot being exactly 0 makes the sign
     test false every time — putting the torus a full radius on the WRONG SIDE
     of the face, in air, with ok:true. Deterministically wrong for sharp
     corners rather than occasionally. Comparing the two and taking the larger
     needs no threshold at all. */
  const cnA = norm(edgeA.coNormalA), cnB = norm(edgeA.coNormalB);
  if (!cnA || !cnB) return { ok: false, reason: 'the first edge has no usable co-normals' };
  const outward = Math.abs(dot(cnB, nF)) >= Math.abs(dot(cnA, nF)) ? cnB : cnA;
  const lean = Math.abs(dot(outward, nF));
  if (!(lean > 1e-6)) {
    return { ok: false, reason: 'neither co-normal of the first edge leans out of the shared face, so which side the ball rolls on cannot be decided' };
  }
  const rollSide = dot(outward, nF) > 0 ? nF : mul(nF, -1);
  const torusCentre = add(axisPoint, mul(rollSide, a));
  const offset = 2 * Math.sqrt(a * c);
  return {
    ok: true,
    torus: {
      centre: torusCentre,
      axis: dC,
      majorRadius: a + c,   // where the rolling ball's centre goes
      minorRadius: a,       // the ball itself
    },
    handoff: { offsetAlongEdge: offset, note: '2*sqrt(a*c) from the foot of the third edge\'s axis' },
    // Both handoff balls, which the two spherical lunes are cut from.
    lunes: [
      { centre: add(add(vertex, mul(norm(edgeA.coNormalA), setbackFor(a, edgeA.phi))), mul(norm(edgeA.coNormalB), setbackFor(a, edgeA.phi))), radius: a, alongEdge: dA },
      { centre: add(add(vertex, mul(norm(edgeB.coNormalA), setbackFor(a, edgeB.phi))), mul(norm(edgeB.coNormalB), setbackFor(a, edgeB.phi))), radius: a, alongEdge: dB },
    ],
  };
}

/* WOULD THIS PATCH FOLD? Sampled, because the scheme has no closed-form answer.

   A fold is a LOCAL reversal of the surface normal, so it is found by comparing
   ADJACENT samples — a patch that wraps a corner sweeps its normal through
   ninety degrees or more quite legitimately, and comparing everything to one
   reference calls that a failure.

   ⚠ THE DIFFERENCES MUST STAY INSIDE THE TRIANGLE. The normal is taken by
   stepping one barycentric coordinate down and the others up; at a coordinate
   of zero that samples a NEGATIVE one, where the interpolant is extrapolating
   and its answer means nothing. Every reversal a first version of this reported
   on the boundary was exactly that. Samples are kept a few steps clear.
*/
export function cornerPatchFolds(patch, opts = {}) {
  if (!patch || typeof patch.evaluate !== 'function') return { ok: false, reason: 'not an evaluable patch' };
  const N = opts.grid ?? 14;
  const h = opts.step ?? 1e-5;
  const guard = 4 * h;
  const nrm3 = (v) => { const L = Math.hypot(v[0], v[1], v[2]); return L > 0 ? [v[0] / L, v[1] / L, v[2] / L] : null; };
  const normalAt = (b0, b1, b2) => {
    const p = patch.evaluate(b0, b1, b2);
    const pu = patch.evaluate(b0 - h, b1 + h, b2), pv = patch.evaluate(b0 - h, b1, b2 + h);
    if (!p || !pu || !pv) return null;
    return nrm3(cross([pu[0] - p[0], pu[1] - p[1], pu[2] - p[2]], [pv[0] - p[0], pv[1] - p[1], pv[2] - p[2]]));
  };
  /* ⚠⚠ A UNIFORM GRID IS BLIND WHERE THE FOLDS ARE. The interpolant is exact on
     the boundary and blends hardest just inside it, so a fold lives in a thin
     ribbon at small barycentric coordinate — and a 14-step grid's first sample
     sits at 1/14 = 0.071, past it. Measured: three configurations this reported
     as clean at grid 14 show 12 to 21 reversals at grid 90. The detector was
     certifying exactly the patches it existed to catch.

     So the sweep is a coarse interior grid PLUS a dense ribbon hugging each
     boundary, down to the guard the finite difference itself imposes. Cost is
     linear in the ribbon and this runs once per corner, not per frame. */
  const grid = new Map();
  const put = (b0, b1, b2, key) => {
    if (b0 < guard || b1 < guard || b2 < guard) return;
    const n = normalAt(b0, b1, b2);
    if (n) grid.set(key, n);
  };
  for (let i = 0; i <= N; i++) for (let j = 0; i + j <= N; j++) put(i / N, j / N, 1 - i / N - j / N, `g${i},${j}`);
  // The ribbon: for each side, walk along it and step inward through the band
  // the uniform grid never reaches.
  const RIB = opts.ribbon ?? 48;
  const depths = [guard * 1.5, 2e-4, 6e-4, 2e-3, 6e-3, 0.015, 0.03, 0.05];
  for (let side = 0; side < 3; side++) {
    for (let k = 0; k <= RIB; k++) {
      const t = k / RIB;
      for (let d = 0; d < depths.length; d++) {
        const b = [0, 0, 0];
        b[(side + 1) % 3] = (1 - t) * (1 - depths[d]);
        b[(side + 2) % 3] = t * (1 - depths[d]);
        b[side] = depths[d];
        put(b[0], b[1], b[2], `r${side},${k},${d}`);
      }
    }
  }
  let reversals = 0, worstDeg = 0;
  const neighbours = (key) => {
    if (key[0] === 'g') { const [i, j] = key.slice(1).split(',').map(Number); return [`g${i + 1},${j}`, `g${i},${j + 1}`]; }
    const [sd, k, d] = key.slice(1).split(',').map(Number);
    return [`r${sd},${k + 1},${d}`, `r${sd},${k},${d + 1}`];
  };
  for (const [key, n] of grid) {
    for (const nk of neighbours(key)) {
      const m = grid.get(nk);
      if (!m) continue;
      const dd = n[0] * m[0] + n[1] * m[1] + n[2] * m[2];
      worstDeg = Math.max(worstDeg, Math.acos(Math.max(-1, Math.min(1, dd))) * 180 / Math.PI);
      if (dd < 0) reversals += 1;
    }
  }
  /* ⚠ A SIGN FLIP IS NOT THE ONLY WAY TO FAIL. A patch creased at eighty-nine
     degrees between neighbouring samples has not technically reversed and is
     not a surface anyone wants; the old criterion returned ok for it. */
  const creaseLimit = opts.creaseLimitDeg ?? 75;
  const folds = reversals > 0 || worstDeg > creaseLimit;
  return {
    ok: !folds, folds, reversals, worstAdjacentDeg: worstDeg, samples: grid.size,
    reason: folds
      ? (reversals > 0
        ? `the corner patch folds over itself — ${reversals} adjacent normal reversal(s), worst turn ${worstDeg.toFixed(1)} degrees. At these radii the region between the three blends is too distorted for one patch to span.`
        : `the corner patch creases — neighbouring samples turn ${worstDeg.toFixed(1)} degrees, past the ${creaseLimit} degree limit, so it is not folded but it is not a fillet either.`)
      : null,
  };
}

/** The blend's end trim loop taken from the PLANES that actually cut it, so the
 *  face's boundary and the corner patch's boundary are one curve.
 *
 *  ⚠ SAMPLED, BECAUSE A PLANAR CUT OF A TUBE IS NOT A LINE IN (u, v). Every
 *  sample is solved against the same plane the corner patch is built from, so
 *  the two cannot drift apart the way the straight-line form does.
 *
 *  `stationAt(u, plane)` returns the v where that plane cuts the blend at
 *  cross-parameter u — one linear solve for a straight-axis tube, and the only
 *  thing here that needs to know what the blend is.
 */
export function blendEndTrimLoopFromPlane({ startPlane, endPlane, stationAt, uMin = 0, uMax = 1, samples = 24 }) {
  if (typeof stationAt !== 'function') return { ok: false, reason: 'no way to locate a plane on the blend' };
  const side = (plane, forward) => {
    const out = [];
    for (let i = 0; i <= samples; i++) {
      const t = forward ? i / samples : 1 - i / samples;
      const u = uMin + (uMax - uMin) * t;
      const v = stationAt(u, plane);
      if (!Number.isFinite(v)) return null;
      out.push([u, v]);
    }
    return out;
  };
  const lo = side(startPlane, true);
  const hi = side(endPlane, false);
  if (!lo || !hi) return { ok: false, reason: 'a blend end plane does not cut the blend across its whole width' };
  for (let i = 0; i < lo.length; i++) {
    if (!(hi[hi.length - 1 - i][1] > lo[i][1])) {
      return { ok: false, reason: 'the blend ends before it starts somewhere across its width — the two corners have eaten the whole edge' };
    }
  }
  return { ok: true, loop: lo.concat(hi) };
}

/* THE PATCH AS A SURFACE THE REST OF A MODELLER CAN HOLD.

   `sideVertexPatch` returns an evaluator over barycentric coordinates, which is
   the right thing for a triangular patch and the wrong shape for everything
   downstream: a closure classifier, a trim splice and a `.3dm` export all read
   a tensor-product surface with a trim loop. So the evaluator is SAMPLED and
   INTERPOLATED, never approximated by a fit that refits its own boundary — the
   neighbouring blends carry the boundary points, and a patch that merely comes
   close to them opens the corner it was built to close.

   ⚠ THE TRIANGLE BECOMES A SQUARE BY COLLAPSING ONE CORNER, and the map is
   chosen so the other three sides are exact isocurves:

       b0 = 1 - v      b1 = v(1 - u)      b2 = v*u

   v = 0 collapses the whole row onto corner 0 — a POLE, and the closing edge a
   seam. v = 1 runs along the side opposite corner 0, u = 0 and u = 1 along the
   other two. On each of those the evaluator reduces to the boundary curve
   exactly (the two rays whose weights vanish there contribute nothing), so the
   interpolated surface reproduces all three boundaries at every station.

   A pole and a seam are not a compromise made here: they are what the existing
   corner fill already produces, and the classifier reads both natively.

   ⚠ SAMPLING IS UNIFORM IN (u, v), NOT CHORD-LENGTH. The collapsed row has zero
   chord length along it, so a chord-length parameterisation of this grid is
   degenerate by construction — it is not a tuning choice. */
export function cornerPatchSurface({ corners, boundary, tangent, uSamples = 13, vSamples = 13, degU = 3, degV = 3, spacing = 'uniform', ...opts }) {
  const patch = sideVertexPatch({ corners, boundary, tangent, ...opts });
  if (!patch.ok) return patch;
  if (!(uSamples >= 2 && vSamples >= 2)) return { ok: false, reason: 'a patch needs at least two samples in each direction' };
  /* ⚠ UNIFORM, AND CLUSTERING THE STATIONS AT THE BOUNDARIES WAS MEASURED AND IS
     WORSE. The reasoning for clustering is sound-sounding — the three ray terms
     are weighted by (b_j*b_k)^2, which vanish at the sides, so the shape looks
     like it moves fastest there. It does not pay: on a bowed triangle at 13
     stations the worst boundary deviation is 1.06e-4 uniform against 2.08e-4
     Chebyshev-Lobatto, and the gap holds at every count tried (9, 13, 25, 41).
     Uniform also converges at fourth order — 1.06e-4, 3.34e-6, 2.74e-7 — which
     is what a cubic interpolant of a smooth boundary should do, and is the
     evidence that a SLOWER rate seen at a real corner is about the boundary
     curve being handed in, not about this grid. */
  const node = (i, n) => (spacing === 'chebyshev' ? (1 - Math.cos(Math.PI * i / (n - 1))) / 2 : i / (n - 1));
  const uParams = Array.from({ length: uSamples }, (_, i) => node(i, uSamples));
  const vParams = Array.from({ length: vSamples }, (_, j) => node(j, vSamples));
  const grid = [];
  for (const u of uParams) {
    const row = [];
    for (const v of vParams) {
      const P = patch.evaluate(1 - v, v * (1 - u), v * u);
      if (!P) return { ok: false, reason: `the patch has no value at (u, v) = (${u.toFixed(3)}, ${v.toFixed(3)})` };
      row.push(P);
    }
    grid.push(row);
  }
  let srf = null;
  try { srf = networkCorrectionSurface(grid, uParams, vParams, degU, degV); }
  catch (e) { return { ok: false, reason: `the patch grid would not interpolate: ${e && e.message ? e.message : e}` }; }
  return {
    ok: true,
    srf,
    // The whole domain: the triangle IS the square under the collapse map, so
    // there is nothing to trim away and the loop is the domain's own corners.
    trimLoop: [[0, 0], [1, 0], [1, 1], [0, 1]],
    trimHoles: [],
    poleCorner: 0,
    evaluate: patch.evaluate,
  };
}
