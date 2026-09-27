// Cage / free-form deformation (FFD): the kernel math for a cage as a
// component. No app-layer wiring here.
//
// The math is classic Sederberg-Parry Free-Form Deformation (SIGGRAPH
// 1986): an axis-aligned box lattice of control points; deforming a point
// means (1) finding its (s,t,u) local coordinates within the box's
// undeformed rest frame (a linear inversion, since the rest frame is an
// axis-aligned box — no iterative point inversion, unlike a general NURBS
// surface), then (2) evaluating a trivariate Bernstein tensor volume at
// (s,t,u) using the control points' current (possibly moved) positions.
// The module relies on this identity: a Bernstein/Bezier interpolation of a
// linear (unmoved) control net reproduces the exact linear map, regardless of
// degree (Bernstein polynomials partition unity and the control points lie
// on a single affine hyperplane). So evaluating an unmoved lattice at a
// point's own local coordinates returns that same point exactly, with no
// separate identity code path; a dedicated test checks it.
//
// Additive bands, each with an independent rest lattice (rather than
// Rhino's CageEdit approach of baking density once): each density level
// added becomes its own lattice at its own rest frame; the total
// deformation at a point is the original point plus the sum of every band's
// displacement (that band's deformed evaluation minus its rest evaluation,
// which by the identity above is "deformed evaluation minus the original
// point"). This avoids having to refine an existing lattice to a finer
// density without disturbing prior edits (which would need a
// shape-preserving knot-insertion-style refinement, not implemented here):
// adding a finer band for more local detail never touches any earlier
// band's data, by construction.
//
// Scope: axis-aligned rest boxes only (no oriented/rotated lattice). A
// point outside a band's [0,1]^3 local-coordinate range, past the falloff
// margin below, gets zero displacement from that band: bands are local zones
// of influence, not globally extrapolating fields, which avoids the
// Bernstein-extrapolation blow-up for points far outside a box at higher
// lattice densities.

import { add, sub, scale, dot, cross, normalize } from './vec3.mjs';

// n choose k, small values only (lattice density stays small; this is not
// a generic combinatorics utility).
function binomial(n, k) {
  if (k < 0 || k > n) return 0;
  let r = 1;
  for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
  return Math.round(r);
}

// Bernstein basis polynomial B_i^n(s) = C(n,i) * s^i * (1-s)^(n-i).
// Evaluated directly rather than by a recursive/De Casteljau scheme: a
// lattice's degree is always small (density-1 per axis, single digits in
// practice), so the closed-form binomial coefficient is exact and there is
// no numerical-stability concern at this scale.
export function bernstein(n, i, s) {
  return binomial(n, i) * s ** i * (1 - s) ** (n - i);
}

// Builds a rest lattice — an axis-aligned box, control points spaced
// evenly along each axis at their undeformed (identity) positions.
// densityU/V/W are control point counts along each axis (>=2 — a density
// of 2 is the coarsest lattice, degree-1/trilinear in that axis).
export function makeRestLattice(min, max, densityU, densityV, densityW) {
  if (densityU < 2 || densityV < 2 || densityW < 2) {
    throw new Error(`makeRestLattice: every axis needs at least 2 control points (got ${densityU}x${densityV}x${densityW})`);
  }
  const ctrlPts = [];
  for (let i = 0; i < densityU; i++) {
    const su = i / (densityU - 1);
    const plane = [];
    for (let j = 0; j < densityV; j++) {
      const sv = j / (densityV - 1);
      const row = [];
      for (let k = 0; k < densityW; k++) {
        const sw = k / (densityW - 1);
        row.push([
          min[0] + su * (max[0] - min[0]),
          min[1] + sv * (max[1] - min[1]),
          min[2] + sw * (max[2] - min[2]),
        ]);
      }
      plane.push(row);
    }
    ctrlPts.push(plane);
  }
  return { min: [...min], max: [...max], densityU, densityV, densityW, ctrlPts };
}

// Inverts the lattice's axis-aligned rest box to get a point's local
// (s,t,u) coordinates — a linear map, not an iterative point inversion (the
// rest frame is always a box, so this is exact and closed-form). Not
// clamped to [0,1]: a point outside the box gets an out-of-range local
// coordinate, and callers decide whether that means "no influence from
// this band" (see deformWithBands below).
export function latticeLocalCoords(lattice, point) {
  const { min, max } = lattice;
  const span = (a, b) => (b - a > 1e-12 ? b - a : 1e-12); // degenerate (zero-thickness) axis guard
  return [
    (point[0] - min[0]) / span(min[0], max[0]),
    (point[1] - min[1]) / span(min[1], max[1]),
    (point[2] - min[2]) / span(min[2], max[2]),
  ];
}

// The trivariate Bernstein tensor evaluation, against the lattice's current
// control points (which may have been moved from their rest positions —
// this is the deformation). Evaluating an unmoved lattice here at a point's
// own local coords returns that same point exactly (see the module header);
// a dedicated test checks it.
export function evaluateLattice(lattice, s, t, u) {
  const { densityU, densityV, densityW, ctrlPts } = lattice;
  const nu = densityU - 1, nv = densityV - 1, nw = densityW - 1;
  let result = [0, 0, 0];
  for (let i = 0; i <= nu; i++) {
    const bu = bernstein(nu, i, s);
    if (bu === 0) continue;
    for (let j = 0; j <= nv; j++) {
      const bv = bernstein(nv, j, t);
      if (bv === 0) continue;
      for (let k = 0; k <= nw; k++) {
        const bw = bernstein(nw, k, u);
        if (bw === 0) continue;
        const w = bu * bv * bw;
        const cp = ctrlPts[i][j][k];
        result = [result[0] + w * cp[0], result[1] + w * cp[1], result[2] + w * cp[2]];
      }
    }
  }
  return result;
}

// Deforms a single point through one lattice (its current, possibly moved
// control points) — localCoords + evaluate, composed.
export function deformPoint(lattice, point) {
  const [s, t, u] = latticeLocalCoords(lattice, point);
  return evaluateLattice(lattice, s, t, u);
}

// A point is "inside" a band's zone of influence if its local coords all
// fall within [0,1], with a small epsilon of slack at the box's faces so a
// point exactly on the lattice's boundary — the common case for a lattice
// built to just enclose a surface — is not excluded by floating-point noise.
const BAND_INFLUENCE_EPS = 1e-9;
export function pointInsideLatticeBounds(lattice, point) {
  const [s, t, u] = latticeLocalCoords(lattice, point);
  return s >= -BAND_INFLUENCE_EPS && s <= 1 + BAND_INFLUENCE_EPS
    && t >= -BAND_INFLUENCE_EPS && t <= 1 + BAND_INFLUENCE_EPS
    && u >= -BAND_INFLUENCE_EPS && u <= 1 + BAND_INFLUENCE_EPS;
}

// Boundary falloff. A cage band's rest-box bounds are fixed when the band is
// created, but Rebuild/Fair/Point Edits all run before Cage in the modifier
// chain (applySrfModifiers) and a later edit can push some of the surface's
// control points outside that box (a denser Rebuild refit, or a large Point
// Edit, both routinely overshoot their own sample grid). With a hard inside
// test, two points a fraction of a millimeter apart straddling the box's
// face could differ by the full displacement magnitude — a visible crease
// (two points 0.002mm apart differing by ~5mm once one crosses the
// boundary).
//
// So each band's influence extends a small margin (10% of its local [0,1]
// extent per axis) past its boundary, ramping smoothly from full weight (1)
// at the boundary to zero at the margin's outer edge. The margin is outside
// [0,1], not inside: every point strictly inside the box (including all 8
// corners, where FFD's "exact at a moved control point" property holds)
// keeps its full-strength, unweighted deformation; only the transition out
// of the box is smoothed. A point past the margin still contributes exactly
// zero, preserving the local-zone-of-influence choice in the module header.
export const CAGE_BOUNDARY_FALLOFF_MARGIN = 0.1;
// Smoothstep (3t^2-2t^3): C1-continuous, zero derivative at both ends, so
// neighboring bands/axes combine without a new kink at the margin's outer
// edge. Exported so any other kernel module needing a
// zero-derivative-at-the-boundary falloff uses this same curve —
// kernel/falloff.mjs, behind kernel/transform.mjs's Charybdis (Rhino:
// Maelstrom) falloff, imports it.
export function smoothstep(t) { const c = Math.max(0, Math.min(1, t)); return c * c * (3 - 2 * c); }
function axisFalloffWeight(c, margin) {
  if (c >= 0 && c <= 1) return 1; // strictly inside (or exactly on the face) — full strength
  const d = (c < 0 ? -c : c - 1); // distance past whichever face was crossed
  return d >= margin ? 0 : 1 - smoothstep(d / margin);
}
// Per-band influence weight — 1 strictly inside, smoothly ramping to 0
// across CAGE_BOUNDARY_FALLOFF_MARGIN just outside, 0 beyond that margin.
// A function of local coordinates only (no Bernstein evaluation), so
// deformWithBands can skip the tensor evaluation entirely once the weight
// is exactly 0.
export function bandInfluenceWeight(lattice, point, margin = CAGE_BOUNDARY_FALLOFF_MARGIN) {
  const [s, t, u] = latticeLocalCoords(lattice, point);
  return axisFalloffWeight(s, margin) * axisFalloffWeight(t, margin) * axisFalloffWeight(u, margin);
}

// The multi-band entry point — the additive-bands design. `bands` is an
// array of lattice objects (each carrying its current, possibly edited
// control points). No separate rest lattice is threaded through, since a
// band's displacement contribution is exactly (deformPoint(band, point) -
// point), by the linear-reproduction identity. A point outside a band's box
// past the falloff margin contributes zero from that band (see the module
// header); a point just past the box's face contributes a smoothly decaying
// partial amount instead of an abrupt cliff (see bandInfluenceWeight).
export function deformWithBands(bands, point) {
  let total = [0, 0, 0];
  for (const band of bands) {
    const weight = bandInfluenceWeight(band, point);
    if (weight <= 0) continue;
    const deformed = deformPoint(band, point);
    total = add(total, scale(sub(deformed, point), weight));
  }
  return add(point, total);
}

// Ring-based cage arrangement — rings are the shared shape across
// Twist/Taper/Shear. A rectangular rest lattice is already a stack of rings
// along whichever axis is chosen (every control point sharing that axis's
// index is one ring); these two functions pre-arrange a lattice's control
// points into a twisted/tapered/sheared shape before deformWithBands runs.
// Both return a fresh lattice object (never mutate the input), as every
// other builder in this module does, and both keep `min`/`max` verbatim
// from the input: those two arrays are the rest frame latticeLocalCoords
// inverts every point against, not a property of the moved control points,
// so recomputing them from the transformed points would corrupt every
// later deformation through this lattice.
//
// Returns, for each ring index 0..N-1 along the chosen axis, the full list
// of [i,j,k] index triples (not partial pairs) making up that ring, so a
// caller cannot transpose the other two axes.
export function ringIndicesAlongAxis(lattice, axis) {
  const { densityU, densityV, densityW } = lattice;
  const ringCount = axis === 'U' ? densityU : axis === 'V' ? densityV : axis === 'W' ? densityW : null;
  if (ringCount == null) throw new Error(`ringIndicesAlongAxis: axis must be 'U', 'V', or 'W' (got ${axis})`);
  const rings = [];
  for (let ring = 0; ring < ringCount; ring++) {
    const triples = [];
    if (axis === 'U') {
      for (let j = 0; j < densityV; j++) for (let k = 0; k < densityW; k++) triples.push([ring, j, k]);
    } else if (axis === 'V') {
      for (let i = 0; i < densityU; i++) for (let k = 0; k < densityW; k++) triples.push([i, ring, k]);
    } else {
      for (let i = 0; i < densityU; i++) for (let j = 0; j < densityV; j++) triples.push([i, j, ring]);
    }
    rings.push(triples);
  }
  return rings;
}

// For each ring along `axis`, computes its fractional position
// t = ringIndex/(N-1) (0 at the first ring, 1 at the last; 0 for the
// degenerate N=1 case, which makeRestLattice's density>=2 floor prevents),
// calls `ringTransformFn(t)` to get a per-point transform for that ring,
// and applies it to every control point in the ring.
export function applyRingTransform(lattice, axis, ringTransformFn) {
  const rings = ringIndicesAlongAxis(lattice, axis);
  const N = rings.length;
  const ctrlPts = lattice.ctrlPts.map((plane) => plane.map((row) => row.map((p) => [...p])));
  for (let ringIdx = 0; ringIdx < N; ringIdx++) {
    const t = N > 1 ? ringIdx / (N - 1) : 0;
    const transformPoint = ringTransformFn(t);
    for (const [i, j, k] of rings[ringIdx]) ctrlPts[i][j][k] = transformPoint(ctrlPts[i][j][k]);
  }
  return {
    min: [...lattice.min], max: [...lattice.max],
    densityU: lattice.densityU, densityV: lattice.densityV, densityW: lattice.densityW,
    ctrlPts,
  };
}

function lerp(a, b, t) { return a + (b - a) * t; }

// The standard Rodrigues rotation formula, rotating `p` about the axis
// through `center` with direction `axisDir`, by `angleRad`. Implemented here
// rather than shared with the app's `gimbalRotatePoint` (which calls
// THREE.Vector3's `applyAxisAngle`) because kernel/cage.mjs is a
// standalone, dependency-free ES module; the two compute the same formula.
export function rotatePointAboutAxis(p, center, axisDir, angleRad) {
  const rel = sub(p, center);
  const k = normalize(axisDir);
  const cosA = Math.cos(angleRad), sinA = Math.sin(angleRad);
  const rotated = add(add(scale(rel, cosA), scale(cross(k, rel), sinA)), scale(k, dot(k, rel) * (1 - cosA)));
  return add(center, rotated);
}

// Stretches the component of `p - center` perpendicular to `axisDir` by
// `factor`; the axis-parallel component is left untouched. The kernel-side
// counterpart of the app's `taperPoint` formula, for the same
// module-boundary reason as rotatePointAboutAxis above.
export function scaleRadially(p, center, axisDir, factor) {
  const rel = sub(p, center);
  const k = normalize(axisDir);
  const axisComp = dot(rel, k);
  const perp = sub(rel, scale(k, axisComp));
  return add(add(center, scale(k, axisComp)), scale(perp, factor));
}

// The three ring transforms — Twist, Taper and Shear. Each is a factory:
// given the deformer's fixed parameters, returns `(t) => (p) => p'` — the
// shape `applyRingTransform` expects, one closure per ring's fractional
// position t. `falloff` defaults to linear (t itself); a caller passing a
// different curve (e.g. this module's `smoothstep`) reshapes how the
// deformer eases along the axis without touching the three formulas below.
//
// Known limit of this substrate: Twist's interior rings do not reach their
// assigned angle anywhere in the final geometry, and the perpendicular
// radius contracts between rings ("candy-wrapper thinning") — a Bernstein
// blend of points on a circle at different angles does not pass through the
// circle at the blended angle. This is an expected, ring-density-capped
// property of using the Bernstein-tensor FFD as the evaluation substrate,
// measured in this module's test suite. Taper and Shear do not share it (a
// Bernstein blend of a value that varies linearly with ring index
// reproduces the linearly varying scalar/vector exactly, by the same
// partition-of-unity/linear-precision property `deformPoint` relies on), but
// both have an analogous, smaller artifact once a non-linear falloff is
// chosen, for the same reason (a Bernstein blend of a nonlinear function's
// discrete ring samples does not reproduce that function evaluated
// continuously); the tests measure that too.
export function twistRingTransform({ center, axisDir, startAngle, endAngle, falloff = (t) => t }) {
  return (t) => {
    const angle = lerp(startAngle, endAngle, falloff(t));
    return (p) => rotatePointAboutAxis(p, center, axisDir, angle);
  };
}

export function taperRingTransform({ center, axisDir, startFactor, endFactor, falloff = (t) => t }) {
  return (t) => {
    const factor = lerp(startFactor, endFactor, falloff(t));
    return (p) => scaleRadially(p, center, axisDir, factor);
  };
}

// A pure per-ring translation — no center/axis needed for the transform
// itself (kept out of the signature rather than accepted and ignored),
// which is why Shear, unlike Twist/Taper, reproduces an exact translation
// at any query point under the default linear falloff: the blended
// displacement is a Bernstein blend of a vector that varies linearly with
// ring index, which the same linear-precision property reproduces exactly,
// continuously, not just at the rings.
export function shearRingTransform({ shearDir, startAmount, endAmount, falloff = (t) => t }) {
  return (t) => {
    const amount = lerp(startAmount, endAmount, falloff(t));
    return (p) => add(p, scale(shearDir, amount));
  };
}
