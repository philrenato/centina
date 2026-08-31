// CAGE / FREE-FORM DEFORMATION (FFD) — the "CAGE-AS-COMPONENT" design,
// honestly named as the "genuinely
// NEW kernel + app-layer work... its own real, multi-session build, not a
// quick wiring exercise." Here: the kernel math only, real and
// node-tested — no app-layer wiring yet (matching this project's own
// established "prove the hard math first, defer UI to a later round"
// pattern: SSI, Sweep1 N-profiles, and the trim/tessellation work
// all shipped kernel-first the identical way).
//
// THE MATH — classic Sederberg-Parry Free-Form Deformation (SIGGRAPH
// 1986): an axis-aligned box lattice of control points; deforming a point
// means (1) finding its (s,t,u) LOCAL coordinates within the box's own
// UNDEFORMED rest frame (a cheap linear inversion, since the rest frame is
// a plain axis-aligned box — no iterative point inversion needed, unlike
// a general NURBS surface), then (2) evaluating a trivariate BERNSTEIN
// tensor volume at (s,t,u) using the control points' CURRENT (possibly
// moved) positions. A real, load-bearing identity this module leans on
// directly rather than re-deriving by accident: a Bernstein/Bezier
// interpolation of a perfectly linear (unmoved) control net reproduces the
// exact linear map, regardless of degree (Bernstein polynomials partition
// unity and the control points lie on a single affine hyperplane) — so
// evaluating an UNMOVED lattice at a point's own local coordinates always
// returns that same point exactly, with no separate "identity" code path
// needed; this is proven directly by a dedicated test below, not assumed.
//
// ADDITIVE BANDS (the "additive-bands-with-independent-rest-lattices"
// design, NOT Rhino's CageEdit precedent of baking density once):
// each density level the student adds becomes its OWN independent lattice
// at its own rest frame; the total deformation at a point is the ORIGINAL
// point plus the SUM of every band's own individual displacement
// contribution (that band's deformed evaluation minus its own rest
// evaluation, which per the identity above is just "deformed evaluation
// minus the original point"). This sidesteps the much harder "must refine
// an existing lattice to a finer density WITHOUT disturbing prior edits"
// problem entirely (which would need a true shape-preserving knot-
// insertion-style refinement, not attempted here) — adding a new, finer
// band for more local detail never touches any earlier band's own data at
// all, by construction.
//
// V1 SCOPE, stated honestly: axis-aligned rest boxes only (no oriented/
// rotated lattice yet — a real, later generalization, not attempted this
// round); a point outside a band's own [0,1]^3 local-coordinate range
// contributes ZERO displacement from that band (bands are treated as
// LOCAL zones of influence, not globally-extrapolating fields — a
// deliberate choice avoiding the well-known Bernstein-extrapolation
// blow-up for points far outside a box at higher lattice densities, not
// an oversight).

import { add, sub, scale, dot, cross, normalize } from './vec3.mjs';

// n choose k, small values only (lattice density stays small — a real,
// reasonable degree ceiling, not a generic combinatorics utility).
function binomial(n, k) {
  if (k < 0 || k > n) return 0;
  let r = 1;
  for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
  return Math.round(r);
}

// Bernstein basis polynomial B_i^n(s) = C(n,i) * s^i * (1-s)^(n-i).
// Deliberately evaluated directly (not via a recursive/De Casteljau
// scheme) — a lattice's own degree is always small (density-1 per axis,
// realistically single digits), so the closed-form binomial coefficient
// is exact and simple; no numerical-stability concern at this scale.
export function bernstein(n, i, s) {
  return binomial(n, i) * s ** i * (1 - s) ** (n - i);
}

// Builds a REST lattice — a plain axis-aligned box, control points spaced
// evenly along each axis at their own undeformed (identity) positions.
// densityU/V/W are CONTROL POINT COUNTS along each axis (>=2 — a density
// of 2 is the coarsest possible lattice, degree-1/trilinear in that axis).
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

// Inverts the lattice's own AXIS-ALIGNED rest box to get a point's local
// (s,t,u) coordinates — a plain linear map, not an iterative point
// inversion (the rest frame is always a box, never a general curved
// volume, so this is exact and closed-form). NOT clamped to [0,1] — a
// point outside the box legitimately gets an out-of-range local
// coordinate; callers decide whether that means "no influence from this
// band" (see deformWithBands below), not this function's own concern.
export function latticeLocalCoords(lattice, point) {
  const { min, max } = lattice;
  const span = (a, b) => (b - a > 1e-12 ? b - a : 1e-12); // degenerate (zero-thickness) axis guard — divide-by-near-zero, not by exactly zero
  return [
    (point[0] - min[0]) / span(min[0], max[0]),
    (point[1] - min[1]) / span(min[1], max[1]),
    (point[2] - min[2]) / span(min[2], max[2]),
  ];
}

// The core trivariate Bernstein tensor evaluation, against the lattice's
// CURRENT control points (which may have been moved from their own rest
// positions — this is the actual deformation). Evaluating an UNMOVED
// lattice here at a point's own local coords is mathematically guaranteed
// to return that same point exactly (see this module's own header
// comment) — proven by a dedicated test, not assumed or special-cased.
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

// Deforms a single point through ONE lattice (its own current, possibly-
// moved control points) — localCoords + evaluate, composed.
export function deformPoint(lattice, point) {
  const [s, t, u] = latticeLocalCoords(lattice, point);
  return evaluateLattice(lattice, s, t, u);
}

// A point is "inside" a band's own zone of influence if its local coords
// all fall within [0,1] (a small epsilon of slack at the box's own faces,
// so a point exactly ON the lattice's own boundary — the common,
// realistic case for a lattice built to just enclose a surface — isn't
// spuriously excluded by floating-point noise).
const BAND_INFLUENCE_EPS = 1e-9;
export function pointInsideLatticeBounds(lattice, point) {
  const [s, t, u] = latticeLocalCoords(lattice, point);
  return s >= -BAND_INFLUENCE_EPS && s <= 1 + BAND_INFLUENCE_EPS
    && t >= -BAND_INFLUENCE_EPS && t <= 1 + BAND_INFLUENCE_EPS
    && u >= -BAND_INFLUENCE_EPS && u <= 1 + BAND_INFLUENCE_EPS;
}

// FROZEN-BOUNDS C0 CLIFF FIX (found by a robustness audit) — a
// real, confirmed gap: a cage band's own rest-box bounds are frozen at
// CREATION time, but Rebuild/Fair/Point Edits all run BEFORE Cage in the
// modifier chain (applySrfModifiers) and can, on a LATER edit, push some
// of the surface's own control points outside that frozen box (a denser
// Rebuild refit, or a large Point Edit, both routinely overshoot their
// own sample grid). The OLD hard boolean gate above meant two points a
// fraction of a millimeter apart, straddling the box's own face, could
// differ by the FULL displacement magnitude — a real, visible crease,
// confirmed numerically during the audit (two points 0.002mm apart
// differing by ~5mm once one crossed the boundary).
//
// Fixed by EXTENDING each band's own influence a small margin (10% of
// its own local [0,1] extent per axis) PAST its true boundary, ramping
// smoothly from full weight (1) AT the boundary down to zero at the
// margin's outer edge — deliberately NOT a margin INSIDE [0,1]: every
// point strictly inside the box (including all 8 corners, where FFD's
// own defining "exact at a moved control point" property already lives,
// proven above) must keep its EXISTING full-strength, unweighted
// deformation, unchanged — the fix only smooths what happens crossing
// OUT of the box, not what happens near a face from the inside. A point
// already past the margin still contributes exactly zero (the "local
// zone of influence" design choice stated in this file's own header,
// avoiding the Bernstein-extrapolation blowup a truly unbounded band
// would risk, is preserved — only the sharp CLIFF at the exact boundary
// is smoothed into a continuous transition).
export const CAGE_BOUNDARY_FALLOFF_MARGIN = 0.1;
// Smoothstep (3t^2-2t^3): C1-continuous, zero derivative at both ends,
// so neighboring bands/axes combine without introducing a NEW kink of
// their own at the margin's own outer edge. Exported (not just a local
// helper) so any other kernel module needing a zero-derivative-at-the-
// boundary falloff reuses this SAME curve rather than re-deriving one —
// kernel/transform.mjs's own Charybdis (Rhino: Maelstrom) falloff does
// exactly that, per this project's own standing rule against a second
// falloff curve existing anywhere in the codebase.
export function smoothstep(t) { const c = Math.max(0, Math.min(1, t)); return c * c * (3 - 2 * c); }
function axisFalloffWeight(c, margin) {
  if (c >= 0 && c <= 1) return 1; // strictly inside (or exactly on the true face) — full strength, unchanged from the original hard-cutoff behavior
  const d = (c < 0 ? -c : c - 1); // distance PAST whichever face was crossed
  return d >= margin ? 0 : 1 - smoothstep(d / margin);
}
// Per-band influence weight — 1 strictly inside, smoothly ramping to 0
// across CAGE_BOUNDARY_FALLOFF_MARGIN just outside, 0 beyond that margin.
// A cheap, real function of local coordinates only (no Bernstein
// evaluation), so deformWithBands can skip the expensive tensor
// evaluation entirely once weight is exactly 0, same as the old boolean
// gate did.
export function bandInfluenceWeight(lattice, point, margin = CAGE_BOUNDARY_FALLOFF_MARGIN) {
  const [s, t, u] = latticeLocalCoords(lattice, point);
  return axisFalloffWeight(s, margin) * axisFalloffWeight(t, margin) * axisFalloffWeight(u, margin);
}

// THE REAL, MULTI-BAND ENTRY POINT — the additive-bands design.
// `bands` is a plain array of lattice objects (each already carrying its
// own current, possibly-edited control points) — no separate "rest
// lattice" needs to be threaded through, since a band's own displacement
// contribution is exactly (deformPoint(band, point) - point), per this
// module's own linear-reproduction identity. A point outside a given
// band's own box (PAST the falloff margin) contributes ZERO from that
// band (stated in this file's own header as a deliberate v1 scope
// choice, not an oversight) — a point just past the box's own true face
// contributes a smoothly-decaying PARTIAL amount instead of an abrupt
// cliff (see bandInfluenceWeight's own comment, the robustness-audit fix).
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

// RING-BASED CAGE ARRANGEMENT — rings are the shared shape across
// Twist/Taper/Shear. A rectangular
// rest lattice is already, structurally, a stack of rings along whichever
// axis is chosen (every control point sharing that axis's own index is
// one ring) — nothing above needed to change; these two functions are the
// only new machinery required to PRE-ARRANGE a lattice's control points
// into a twisted/tapered/sheared shape before deformWithBands ever runs.
// Both return a FRESH lattice object (never mutate the input) — the same
// convention makeRestLattice/every other builder in this module already
// follows — and both preserve `min`/`max` VERBATIM from the input: those
// two arrays are the REST FRAME latticeLocalCoords inverts every point
// against, not a property of the (now-moved) control points, so
// recomputing them from the transformed points would silently corrupt
// every future deformation through this lattice.
//
// Returns, for each ring index 0..N-1 along the chosen axis, the full
// list of [i,j,k] index TRIPLES (not partial pairs) making up that ring
// — deliberately unambiguous about which of the other two axes is which,
// so a caller can never accidentally transpose them.
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

// For each ring along `axis`, computes its own fractional position
// t = ringIndex/(N-1) (0 at the first ring, 1 at the last; 0 for the
// degenerate N=1 case, which ringIndicesAlongAxis's own axis-length
// check already prevents via makeRestLattice's own density>=2 floor),
// calls `ringTransformFn(t)` to get a per-point transform for THAT ring
// specifically, and applies it to every control point in the ring.
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
// through `center` with direction `axisDir`, by `angleRad`. Re-implemented
// here (rather than shared with the app's own `gimbalRotatePoint`,
// which calls THREE.Vector3's `applyAxisAngle`) because kernel/cage.mjs is
// a standalone, dependency-free ES module — the two are meant to compute
// the identical formula, not share code across the module boundary.
export function rotatePointAboutAxis(p, center, axisDir, angleRad) {
  const rel = sub(p, center);
  const k = normalize(axisDir);
  const cosA = Math.cos(angleRad), sinA = Math.sin(angleRad);
  const rotated = add(add(scale(rel, cosA), scale(cross(k, rel), sinA)), scale(k, dot(k, rel) * (1 - cosA)));
  return add(center, rotated);
}

// Stretches the component of `p - center` PERPENDICULAR to `axisDir` by
// `factor`; the axis-parallel component is left untouched. The kernel-
// side twin of the app's own `taperPoint` formula, for the same
// module-boundary reason as rotatePointAboutAxis above.
export function scaleRadially(p, center, axisDir, factor) {
  const rel = sub(p, center);
  const k = normalize(axisDir);
  const axisComp = dot(rel, k);
  const perp = sub(rel, scale(k, axisComp));
  return add(add(center, scale(k, axisComp)), scale(perp, factor));
}

// THE THREE RING TRANSFORMS — Twist, Taper and Shear.
// Each is a factory: given the deformer's own fixed
// parameters, returns `(t) => (p) => p'` — the shape `applyRingTransform`
// expects, one closure per ring's own fractional position t. `falloff`
// defaults to linear (t itself); a caller passing a different curve
// (e.g. this module's own `smoothstep`) reshapes how the deformer eases
// along the axis without touching any of the three formulas below.
//
// A REAL, NAMED LIMIT OF THIS SUBSTRATE, stated honestly, not glossed
// over: TWIST's own interior
// rings do NOT achieve their assigned angle anywhere in the final
// geometry, and the perpendicular radius measurably CONTRACTS between
// rings ("candy-wrapper thinning") — Bernstein-blending points that sit
// on a circle at different angles does not pass through the circle at
// the blended angle. This is a real, expected, ring-density-capped
// property of reusing this app's already-proven Bernstein-tensor FFD as
// the evaluation substrate, not a bug — proven directly, with real
// measured numbers, in this module's own test suite, not just asserted.
// TAPER and SHEAR do not share this specific defect (a Bernstein blend
// of a value that itself varies LINEARLY with ring index reproduces the
// true linearly-varying scalar/vector exactly, by the same partition-of-
// unity/linear-precision property `deformPoint`'s own header comment
// already leans on) — but BOTH still inherit an analogous, smaller
// artifact the moment a NON-linear falloff is chosen, for the identical
// underlying reason (a Bernstein blend of a nonlinear function's own
// discrete ring samples does not reproduce that function evaluated
// continuously) — also proven directly, not assumed either way.
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
// itself (kept out of the signature rather than accepted-and-ignored),
// which is exactly why Shear, unlike Twist/Taper, reproduces an EXACT
// translation at any query point under the default linear falloff: the
// blended displacement is a Bernstein blend of a vector that itself
// varies linearly with ring index, which the same linear-precision
// property reproduces exactly, continuously, not just at the rings
// themselves.
export function shearRingTransform({ shearDir, startAmount, endAmount, falloff = (t) => t }) {
  return (t) => {
    const amount = lerp(startAmount, endAmount, falloff(t));
    return (p) => add(p, scale(shearDir, amount));
  };
}
