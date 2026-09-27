// Puff — a closed outline, inflated into closed quad geometry, with controls for thickness and
// profile.
//
// The construction takes one part from each of two prior implementations:
//   The Swift version gives the topology — a shared equator ring. Front and back reference the
//     same ring, so there is nothing to stitch. The Grasshopper plugin lacks this and pays a
//     four-level join fallback cascade (three JoinBreps at widening tolerances, three CreateSolid,
//     a mesh-weld, and finally "return the halves unjoined") for it. There is no join code in this
//     file, because there is nothing to join.
//   The Grasshopper plugin gives the mathematics — elevation from a distance field. The Swift
//     version sets height from ring index alone: it never looks at the shape, so a thin ribbon and
//     a fat blob of the same bounding box get the same height.
//
// Distance is exact on the polygon, not sampled on a voxel grid. A round blob cannot show a
// sub-voxel failure, because its inradius is the brush radius by construction; a crescent of the
// same drawn extent has inradius 0.0438 against the disc's 0.1599 — 0.70 of a voxel against 2.56.
// This module computes distance exactly on the polygon at every lattice vertex, where a crescent's
// 0.0162 resolves as precisely as a disc's 0.1599 and the sub-voxel class does not arise. The
// crescent, not the disc, is the fixture that tests it.
//
// One center in this file: `inradius.at`. The Swift version carries three in one function —
// bounding-box mid-point for normalization, centroid for the rings, and the world origin for the
// cap — so an asymmetric outline gets a skewed cap. The incenter is the only point at which the
// normalized distance reaches 1, which makes it the apex.
//
// Nothing here is parameterized by angle. Swift's `sampleAngle` is a nearest-angle pick with no
// interpolation, so any non-star-shaped outline — a crescent, a C, an S — silently collapses into
// a plausible blob that is not what was drawn. Ring membership here is by index into an arc-length-resampled
// outline, and elevation is from `distanceToBoundary`, which is exact for any simple polygon
// whether or not it is star-shaped.

import { buildSpine, attachToSpine } from './puffspine.mjs';
import { buildGridCage, sigmaAlong } from './puffgridcap.mjs';
import { subdivideCatmullClark as ccSubdivide } from './subd.mjs';
/* One subdivision, through this kernel's own Catmull-Clark. The refusal test
   below asks whether the subdivided surface bulges outside the drawn line, and
   that answer must come from the same subdivider the shape will be
   drawn with — a second implementation agreeing to within a tolerance is not
   the same claim. The two differ only in packing: flat arrays here, a cage
   there. */
function buildTopology() { return null; }
function subdivideCatmullClark(positions, quads) {
  const vertices = [];
  for (let i = 0; i < positions.length; i += 3) vertices.push([positions[i], positions[i + 1], positions[i + 2]]);
  const faces = [];
  for (let i = 0; i < quads.length; i += 4) faces.push([quads[i], quads[i + 1], quads[i + 2], quads[i + 3]]);
  const out = ccSubdivide({ vertices, faces, creases: {} });
  const P = new Float64Array(out.vertices.length * 3);
  for (let i = 0; i < out.vertices.length; i += 1) {
    P[i * 3] = out.vertices[i][0]; P[i * 3 + 1] = out.vertices[i][1]; P[i * 3 + 2] = out.vertices[i][2];
  }
  return { positions: P };
}
import {
  prepareOutline, distanceToBoundary, pointInPolygon, taubinSmoothClosed,
} from './puffoutline.mjs';

// The two profile families, ported verbatim
/**
 * Perpendicular walls (Lamé).   f(t) = (1 - (1-t)^n)^(1/n),   n = 2 * 2.5^(1-2p),  n >= 1.1
 *
 * Measured max‖∇F‖ over the family: 28.92 at p=0, 9.24 at
 * p=0.5, 1.25 at p=1, against a `sdCapsule` control that reads 1.000 exactly. A smooth-minimum
 * blend acts where |a-b| < k, so a field L times steeper produces a fillet k/L wide — a
 * Melt control would mean something different at every Profile setting of this family. A disc at
 * p=0.5 is an exact hemisphere of height H, which is the only closed-form oracle this module has.
 *
 * p = 0.5 is the hemisphere here and a paraboloid in the other family. The Grasshopper README
 * calls p=0.5 "an exact hemisphere" without saying which family; it is true of this one (n=2) and
 * false of the tangent foot.
 */
export function lameProfile(t, p) {
  const n = Math.max(1.1, 2 * Math.pow(2.5, 1 - 2 * p));
  const u = Math.min(Math.max(t, 0), 1);
  return Math.pow(1 - Math.pow(1 - u, n), 1 / n);
}

/**
 * Tangent foot.   f(t) = 1 - (1 - t^2)^a,   a = 2.5^(1-2p)
 *
 * buildPuff's default family. max‖∇F‖ is 1.19 / 1.26 / 1.80 at p = 0 / 0.5 / 1 — the same order as the
 * capsule control, so a blend radius means the same thing across the whole range.
 *
 * The Grasshopper README describes this family backwards: its "mathematical heart" says it "rises
 * perpendicular from the outline, every time". Differentiate it: f'(t) = 2a·t·(1-t²)^(a-1), so
 * f'(0) = 0 — it leaves the outline tangent, which is what the source comment (not the README)
 * says and what the name says. A perpendicular wall is the other family.
 *
 * No minimum clamp on `a`, matching the sources: of the two families only
 * `n` carries one. a runs 2.5 (p=0) to 0.4 (p=1) and is well-behaved throughout.
 */
export function tangentProfile(t, p) {
  const a = Math.pow(2.5, 1 - 2 * p);
  const u = Math.min(Math.max(t, 0), 1);
  return 1 - Math.pow(1 - u * u, a);
}

/**
 * Bulge — one slider from a tangent lens to a round torus, for a ring rim.
 * b = 0 is the tangent foot (rises tangent to the base, a flat-topped lens);
 * b = 1 is the Lamé hemisphere (rises perpendicular, a round tube). A convex
 * blend of the two, so f(0)=0 and f(1)=1 hold for every b (the rim stays level
 * and the peak is shared) and the morph is monotone throughout. This is the
 * hole-rim control, from more tangent to more bulged. The two
 * mixed feet are taken at their own p=0.5 canonical case; the blend `b` is the
 * only knob so a single slider spans the whole range.
 */
export function bulgeProfile(t, b) {
  const w = Math.min(Math.max(b, 0), 1);
  return (1 - w) * tangentProfile(t, 0.5) + w * lameProfile(t, 0.5);
}

export const PROFILES = Object.freeze({ tangent: tangentProfile, lame: lameProfile, bulge: bulgeProfile });

// Refusals — a puff is one closed loop with no hole
/**
 * Refusals are named, never silent, and never repaired. The Swift version's self-intersection
 * guard `isContourClean` exists, is off by default, and guards only the randomizers — so a
 * hand-drawn self-intersecting outline goes straight in and comes out as something else. A
 * repair would be a guess about which of two readings the hand meant.
 *
 * Accepts either one loop (a flat [x,y,...] array, or an array of [x,y] pairs) or an array of
 * loops. More than one loop is refused, and the two ways of having more than one are told apart,
 * because they call for different actions: a hole (one shape inside another) or two disjoint
 * shapes, puffed separately.
 */
export function classifyLoops(loops) {
  if (loops.length <= 1) return { ok: true };
  // A loop whose every vertex lies inside another is a hole; anything else is a separate region.
  for (let i = 0; i < loops.length; i++) {
    for (let j = 0; j < loops.length; j++) {
      if (i === j) continue;
      const inner = loops[i], outer = loops[j];
      let allIn = true;
      for (const p of inner) if (!pointInPolygon(outer, p[0], p[1])) { allIn = false; break; }
      if (allIn) {
        return { ok: false, reason: 'hole', why:
          'A puff is one closed loop with no hole — this outline has one shape inside another. '
          + 'Puff the outer shape, or draw the two separately.' };
      }
    }
  }
  return { ok: false, reason: 'multi-region', why: 'two separate shapes — puff them one at a time.' };
}

// Normalize whatever the caller drew into an array of loops of [x,y] pairs.
function toLoops(raw) {
  if (!Array.isArray(raw) || raw.length === 0) return [];
  if (typeof raw[0] === 'number') {                                  // flat [x,y,x,y,...]
    const l = []; for (let i = 0; i + 1 < raw.length; i += 2) l.push([raw[i], raw[i + 1]]);
    return [l];
  }
  if (Array.isArray(raw[0]) && typeof raw[0][0] === 'number') return [raw];   // one loop of pairs
  return raw.map(toLoops).map(l => l[0]).filter(Boolean);            // array of loops
}

// The build
/**
 * @param raw      one closed outline, or an array of loops (which is refused — see classifyLoops)
 * @param opts
 *   thickness   0..1, relative. `H = thickness * 4 * dmax`.
 *               Relative because absolute does not travel. The usable height range is 7-12x
 *               smaller on a crescent than on a disc of the same drawn size, so a slider in world
 *               units is live over a different fraction of its travel on every shape drawn. The
 *               absolute value is a readout, not the control.
 *   profile     0..1, into the tangent family
 *   family      'tangent' (default) | 'lame' (the analytic oracle)
 *   smoothing   passes of the damped Laplacian over the field. 0 is legal and is the identity.
 *   bands       rings per side, M
 *   count       points per ring, N
 *   bottomScale 1.0 = symmetric
 * @returns { ok, reason, why, positions, quads, tris, ... } — a refusal carries no geometry
 */
// Measured on the subdivided surface, which is the one on screen; a test on the cage passes
// shapes whose displayed surface fills the hollow. The cage of a C
// spills 0.00000 outside the drawn line; subdivided twice it spills 0.227, which is 81% of its own
// inradius, and a comma spills 99.5%. A 1.3-turn spiral passes a cage test and spills 269%.
// Catmull-Clark pulls vertices toward neighborhood averages, so a fan whose straight cage edges
// stay inside a concave outline bulges straight across the hollow once it is smoothed.
//
// One level, not two. It is a refusal test, not a rendering: one level already separates the two
// populations by more than an order of magnitude and costs a quarter as much.
//
// The threshold is not tuned. Measured, as a fraction of the inradius: everything that looks
// right lands under 10% and everything that looks wrong lands over 200%. 0.25 sits in that gap.
const SPILL_LIMIT = 0.25;
function hollowSpill(pts, positions, quads, dmax) {
  const topo = buildTopology(quads, positions.length / 3);
  const s1 = subdivideCatmullClark(positions, quads, topo);
  const P = s1.positions;
  let worst = 0, n = 0;
  for (let i = 0; i < P.length; i += 3) {
    if (pointInPolygon(pts, P[i], P[i + 1])) continue;
    n++;
    let best = Infinity;
    for (let k = 0; k < pts.length; k++) {
      const a = pts[k], c = pts[(k + 1) % pts.length];
      const ex = c[0] - a[0], ey = c[1] - a[1], L2 = ex * ex + ey * ey;
      let t = L2 > 0 ? ((P[i] - a[0]) * ex + (P[i + 1] - a[1]) * ey) / L2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const d = Math.hypot(a[0] + t * ex - P[i], a[1] + t * ey - P[i + 1]);
      if (d < best) best = d;
    }
    if (best > worst) worst = best;
  }
  return { worst, n, frac: dmax > 0 ? worst / dmax : 0 };
}
function spilled(spill, escaping, dmax) {
  return { ok: false, reason: 'not-star-shaped', escaping, spill: spill.worst, spillFrac: spill.frac, why:
    `This shape curves back on itself more than Puff can follow — inflating it bulges `
    + `${(100 * spill.frac).toFixed(0)}% of its own width across the hollow you drew. `
    + `Draw it as two overlapping shapes.` };
}

export function buildPuff(raw, opts = {}) {
  const {
    thickness = 0.35, profile = 0.5, family = 'tangent',
    smoothing = 2, bands = null, count = 128, bottomScale = 1.0,
    // cap: 'apex' is the ring-to-apex lattice; 'grid' is the pole-free cage — arc-spaced rings
    // down to a hole plus a Coons grid patch, see puffgridcap.mjs; 'auto' picks by profile
    // family (S3c). Excursion outside the drawn line, measured on the subdivided surface: on a
    // star-shaped outline the apex path gives 0.00000 (disc and hand) and the grid slightly
    // worsens the hand (0 -> 1.3%); on a spined outline the apex fan sails across the drawn
    // hollow — a C spills 81% of its inradius and a comma 99.5% — and the grid gives 0.7% and 0.6%.
    cap = 'auto', capFrac = 0.5, capBands = null, capRows = null,
    // Iterations of Taubin fairing applied to the drawn curve before any geometry is derived.
    curveSmooth = 20,
    // Verification only. The spill test refuses the apex cage on a C or a comma (it bulges
    // 81-99% of its own width across the drawn hollow); this builds it anyway, so a comparison
    // can measure it beside the grid cage.
    allowSpill = false,
  } = opts;

  const loops = toLoops(raw);
  if (loops.length === 0) return { ok: false, reason: 'empty', why: 'Nothing was drawn.' };
  const cls = classifyLoops(loops);
  if (!cls.ok) return { ok: false, reason: cls.reason, why: cls.why };

  // S1: hygiene. Every refusal here comes from prepareOutline
  const prep = prepareOutline(loops[0], { count });
  if (!prep.ok) return { ok: false, reason: 'outline', why: prep.why, at: prep.at };

  // The drawn curve is faired before anything is measured from it. Every quantity below — the
  // distance field, the inradius, the medial spine — is derived from `pts`, so smoothing it here
  // means the tremor never enters the geometry rather than being smoothed out of the surface
  // afterward. Smoothing the surface instead does not work: it flattens the dome into a
  // plateau and leaves the lobes, because by then the lobes are the shape.
  // Set `curveSmooth: 0` to build from the raw stroke.
  let { pts, area, inradius: dmax, inradiusAt: centre, inradiusSpacing: spacing, diag } = prep;
  if (curveSmooth > 0) {
    const faired = taubinSmoothClosed(pts, { iters: curveSmooth });
    const re = prepareOutline(faired, { count });
    // If fairing makes a curve the preparer will not accept, keep the original rather than
    // refusing a shape that was drawable. Fairing is an improvement, not a requirement.
    if (re.ok) { pts = re.pts; area = re.area; dmax = re.inradius; centre = re.inradiusAt; spacing = re.inradiusSpacing; diag = re.diag; }
  }

  // S2: is there an interior at all, and is it measured?
  // `inradius` returns its own `spacing` so a caller can say how sure it is rather than
  // implying exactness. A region whose deepest point is within one sample of the boundary has not
  // been measured, it has been missed — and a silent zero-height mesh would be a plausible result
  // that is not what was drawn.
  if (!(dmax > 0) || !centre) {
    return { ok: false, reason: 'sliver', why:
      'This outline has no measurable inside — it is a sliver rather than a shape. Draw it wider.' };
  }
  if (dmax <= spacing) {
    return { ok: false, reason: 'sliver', why:
      `This outline is too thin to measure — its deepest point is ${dmax.toFixed(4)} from the edge, `
      + `within the ${spacing.toFixed(4)} the measurement can resolve. Draw it wider.` };
  }

  // S3: resolution from the inradius, never from the drawn extent
  // The bounding box is not a proxy for size here, measured: a crescent and a disc of
  // the same extent differ by 21% in half-bounding-diagonal and by 0.137 against 1.000 in inradius.
  // Anything sized from the extent is sized by the wrong number on exactly the shapes that matter.
  // N must be even: each pole is
  // closed with paired quads — one per two ring vertices — because the app this feeds is strictly
  // quads-only: its buildTopology strides by four and has no concept of face arity, and its
  // validateQuadMesh rejects a triangle written as [a,b,c,c] by name, as "a degenerate quad". An
  // odd ring cannot be paired, so it is refused here rather than discovered at install time.
  const N = pts.length;
  const M = bands != null ? Math.max(1, bands | 0)
    : Math.max(2, Math.min(24, Math.round(N / 8)));
  if (N % 2 !== 0) {
    return { ok: false, reason: 'odd-ring', why:
      `A puff closes each end with paired quads, so it needs an even number of points around the `
      + `edge and was given ${N}. Ask for an even count.` };
  }

  // S3b: is this shape star-shaped about its own deepest point?
  // The ring lattice carries a star-shape assumption that the field does not: "ring k vertex i =
  // lerp(pts[i], centre, k/(M+1))" is a straight segment from each boundary point to one interior
  // point. On a crescent that segment leaves the shape: measured, 23 of 96 boundary points, against
  // 0 of 96 on a disc. The ring vertices then land in the bite, and the puff bulges across a
  // concavity that was drawn empty.
  const inside = (x, y) => pointInPolygon(pts, x, y);
  let escaping = 0;
  for (const q of pts) {
    for (let sIdx = 1; sIdx <= 12; sIdx++) {
      const fr = sIdx / 13;
      if (!inside(q[0] + (centre[0] - q[0]) * fr, q[1] + (centre[1] - q[1]) * fr)) { escaping++; break; }
    }
  }
  // Escaping points change the construction rather than refuse the shape. A crescent, a C, an S,
  // a comma, a bean and a boomerang are ordinary things to draw. The escape is a fact about one
  // construction: every boundary point lerped toward a single interior point, which on a shape
  // that curves back on itself leaves the polygon and fills in the drawn hollow. With the medial
  // axis each point aims at the spine sample whose inscribed ball reaches it, so the spoke is
  // provably inside.
  let spineTarget = null, spineSigma = null;
  if (escaping > 0) {
    // `buildSpine` takes the inscribed ball as `{ at, r }`, not a bare point — it refines the
    // root outward from a radius, so handing it only a position throws inside refineBall.
    const sp = buildSpine(pts, { seed: { at: centre, r: dmax } });
    if (sp && sp.ok) {
      const att = attachToSpine(pts, sp);
      // The residual is the safety statement: how far each point lies outside the ball assigned to
      // it. A form the spine does not describe (a branch, a third limb) shows up here and nowhere
      // else, so it is checked rather than assumed.
      // The residual is a proxy and it over-predicts. It bounds how far a spoke could leave the
      // polygon, but a bound is not the defect. At `0.25 * dmax` it would refuse a zigzag whose
      // worst residual is only 34% over the line, on 3 of its 96 points, and which builds with
      // Euler 2, quads only, and a measured excursion outside the drawn outline of 0.00000.
      // So the residual is only a loose sanity bound (1.0 * dmax), and the real test is the
      // consequence itself — every lattice point is checked for containment below, which is
      // the condition the refusal names ("inflating it would fill in the hollow you drew").
      let worst = 0;
      for (let i = 0; i < att.residual.length; i++) if (att.residual[i] > worst) worst = att.residual[i];
      if (worst <= dmax * 1.0) { spineTarget = att.target; spineSigma = sigmaAlong(att.target, sp.pts); }
    }
    if (!spineTarget) {
      return { ok: false, reason: 'not-star-shaped', escaping, why:
        `This shape curves back on itself more than Puff can follow — ${escaping} of ${pts.length} points `
        + `around the edge cannot see its deepest point, and its medial spine does not describe the whole `
        + `form either. Draw it as two overlapping shapes.` };
    }
  }

  // S3c: the pole-free cage
  // Same refusals, same spine decision, same elevation law (distance field through the profile),
  // same shared equator — a different lattice. Everything below this block is the apex path.
  // Known limitation: this path returns before the lattice containment check in S4, which is what
  // refuses a 1.9-turn spiral on the apex path; here only the spill test applies.
  // 'auto' picks the cap from the profile family, measured. A grid patch closes a dome well and a
  // cone tip badly: with the tangent family, whose profile has slope 2 at the peak, the apex path
  // beats the grid on a disc 7.1 degrees against 84.9 — a cone tip wants a pole, because a patch
  // has to flatten across the point the profile is making. With Lamé n=2 (a hemisphere) the grid
  // wins everywhere, and on an irregular stroke the apex fan leaves a star of creases radiating
  // from the pole that a dihedral scan does not catch. `buildPuff`'s default family is 'tangent',
  // so a caller that passes no family keeps the pole. Pass `cap` explicitly to override either way.
  const wantsGrid = cap === 'grid' || (cap === 'auto' && family === 'lame');
  if (wantsGrid) {
    const H0 = thickness * 4 * dmax;
    const eH = Math.min(H0, dmax * 4.0);
    const pf = PROFILES[family] || tangentProfile;
    const targets = spineTarget || pts.map(() => [centre[0], centre[1]]);
    const g = buildGridCage(pts, targets, spineSigma, {
      dmax, effH: eH, prof: (t) => pf(t, profile),
      smoothing: Math.max(0, smoothing | 0), bottomScale, capFrac, capBands, capRows,
    });
    const spill = hollowSpill(pts, g.positions, g.quads, dmax);
    if (!allowSpill && spill.frac > SPILL_LIMIT) return spilled(spill, escaping, dmax);
    return {
      ok: true, reason: null, why: '',
      positions: g.positions, quads: g.quads, tris: new Uint32Array(0),
      outline: pts,
      nv: g.nv, N, M: g.M,
      dmax, centre, area, spacing, diag,
      height: eH, requestedHeight: H0, capped: H0 > dmax * 4.0,
      profile, family, smoothing: Math.max(0, smoothing | 0), bottomScale,
      spined: !!spineTarget, escaping,
      equatorCount: N,
      field: g.field,
      cap: g.cap,
    };
  }

  // S4: the ring lattice. Constant N per ring — no taper
  // The taper is not load-bearing for the shape, and dropping it is what makes the output quads.
  // Measured on the Swift output at its defaults: 3,064 quad pairs against 256 lone triangles,
  // 96.0% pairable, and the lone count is exactly N — the equator sample count — independent of
  // band count. So the triangles come from the emission, not the geometry.
  let latticeEscaped = 0;
  const ringXY = [];                                   // ringXY[k][i] = [x, y]
  for (let k = 0; k <= M; k++) {
    const f = k / (M + 1);
    const ring = new Array(N);
    for (let i = 0; i < N; i++) {
      // Per-point target when the shape curves back on itself, the single center otherwise; for a
      // star-shaped outline `spineTarget` is null and this is the single-center construction.
      const tx = spineTarget ? spineTarget[i][0] : centre[0];
      const ty = spineTarget ? spineTarget[i][1] : centre[1];
      ring[i] = [pts[i][0] + (tx - pts[i][0]) * f,
                 pts[i][1] + (ty - pts[i][1]) * f];
      // The containment test. Ring 0 is the outline, where a
      // point-in-polygon test is undefined, so it is exempt; every interior ring point must be
      // inside the shape or the lattice has landed in the hollow. Only the spine path can fail
      // this — with a single center, `escaping === 0` already guarantees it.
      if (spineTarget && k > 0 && !inside(ring[i][0], ring[i][1])) latticeEscaped++;
    }
    ringXY.push(ring);
  }

  // Refused on the measured condition, not on a bound. If the lattice lands outside the
  // drawn shape then inflating it would fill in the hollow, and that is refused by name.
  // If it does not, the shape resolves.
  if (latticeEscaped > 0) {
    return { ok: false, reason: 'not-star-shaped', escaping, latticeEscaped, why:
      `This shape curves back on itself more than Puff can follow — ${latticeEscaped} points of the `
      + `inflated lattice land outside the line you drew, so it would fill in the hollow. `
      + `Draw it as two overlapping shapes.` };
  }

  // S5: the field. Exact distance to the drawn boundary, per lattice vertex
  const d = [];
  for (let k = 0; k <= M; k++) {
    const row = new Float64Array(N);
    for (let i = 0; i < N; i++) row[i] = k === 0 ? 0 : distanceToBoundary(pts, ringXY[k][i][0], ringXY[k][i][1]);
    d.push(row);
  }

  // S6: smooth the field, damped and boundary-locked
  // λ = 0.5 and ring 0 never moves. Swift's Laplacian is undamped (λ=1) and unlocked, which drags
  // the equator off the drawn outline, so the silhouette leaves the drawn line. Ring 0 is
  // byte-identical before and after this loop.
  // 0 passes is legal and is the identity. Swift's `max(2, …)` makes smoothing:0 unreachable, so
  // the unsmoothed field cannot be seen even to compare against.
  const passes = Math.max(0, smoothing | 0);
  for (let it = 0; it < passes; it++) {
    const next = d.map(r => Float64Array.from(r));
    for (let k = 1; k <= M; k++) {                     // k = 0 locked
      for (let i = 0; i < N; i++) {
        const im = (i - 1 + N) % N, ip = (i + 1) % N;
        const km = k - 1, kp = Math.min(M, k + 1);
        const avg = (d[k][im] + d[k][ip] + d[km][i] + d[kp][i]) / 4;
        next[k][i] = d[k][i] + 0.5 * (avg - d[k][i]);
      }
    }
    for (let k = 0; k <= M; k++) d[k].set(next[k]);
  }

  // S7/S8/S9: normalize, cap, lift
  // Normalized by the pre-smoothing dmax. Renormalizing to the smoothed maximum would cancel the
  // smoother's own effect and make "peak height drops with pass count" unmeasurable.
  const H = thickness * 4 * dmax;
  const effectiveH = Math.min(H, dmax * 4.0);          // the ported cap, and its ported constant
  const f = PROFILES[family] || tangentProfile;

  const zTop = [];
  for (let k = 0; k <= M; k++) {
    const row = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const t = Math.min(Math.max(d[k][i] / dmax, 0), 1);
      row[i] = effectiveH * f(t, profile);
    }
    zTop.push(row);
  }
  const apexZ = effectiveH * f(1, profile);

  // S10/S11: vertices and quads, about one shared equator
  //   equator(i)   = i                       z = 0   shared by both halves
  //   front(k,i)   = N + (k-1)*N + i         k = 1..M
  //   frontApex    = N + M*N
  //   back(k,i)    = N + M*N + 1 + (k-1)*N + i
  //   backApex     = N + 2*M*N + 1
  const V = N * (2 * M + 1) + 2;
  const positions = new Float32Array(V * 3);
  const put = (idx, x, y, z) => { positions[idx*3] = x; positions[idx*3+1] = y; positions[idx*3+2] = z; };
  const front = (k, i) => (k === 0 ? i : N + (k - 1) * N + i);
  const frontApex = N + M * N;
  const back  = (k, i) => (k === 0 ? i : N + M * N + 1 + (k - 1) * N + i);
  const backApex = N + 2 * M * N + 1;

  for (let i = 0; i < N; i++) put(i, ringXY[0][i][0], ringXY[0][i][1], 0);   // the shared ring
  for (let k = 1; k <= M; k++) for (let i = 0; i < N; i++) {
    put(front(k, i), ringXY[k][i][0], ringXY[k][i][1],  zTop[k][i]);
    put(back(k, i),  ringXY[k][i][0], ringXY[k][i][1], -zTop[k][i] * bottomScale);
  }
  put(frontApex, centre[0], centre[1],  apexZ);
  put(backApex,  centre[0], centre[1], -apexZ * bottomScale);

  // The winding is correct by construction and there is no normal-flip heuristic in this file.
  // `prepareOutline` guarantees counter-clockwise, so the interior lies to the left of the tangent
  // and the front order (outer_i, outer_j, inner_j, inner_i) faces +z on every band of every shape,
  // convex or not. Swift cannot get this by construction and uses a centroid-based
  // majority-vote flip, implemented twice, in two layers, at two different sample rates — a
  // heuristic that is structurally wrong for a non-convex blob, because on a crescent a
  // legitimately outward normal points toward the centroid.
  // Quads only, including at the poles. A triangle fan, the textbook pole closure, is what the app
  // cannot take: buildTopology is hard-wired to `quads.length / 4` with no
  // face-arity concept, and validateQuadMesh refuses a triangle written as [a,b,c,c]
  // because a repeated corner is its definition of a degenerate quad. The repo's own primitives
  // follow the same constraint — the cylinder is a deformed cube rather than a capped tube, and the
  // cone keeps a tiny disc instead of a point.
  //
  // Paired pole quads: quad(ring_i, ring_i+1, ring_i+2, apex) for every even i. Each ring edge is
  // used exactly once, each even spoke is shared by exactly two of them, the odd ring vertices grow
  // no spoke at all, and all four corners are distinct — so nothing reads as degenerate. The apex
  // ends at valence N/2 rather than N, which is also a better cage: a valence-128 pole subdivides
  // into a visible star.
  const POLE = N / 2;
  const quads = new Uint32Array((2 * M * N + 2 * POLE) * 4);
  let q = 0;
  for (let k = 0; k < M; k++) for (let i = 0; i < N; i++) {
    const j = (i + 1) % N;
    quads[q++] = front(k, i);     quads[q++] = front(k, j);
    quads[q++] = front(k + 1, j); quads[q++] = front(k + 1, i);
  }
  for (let i = 0; i < N; i += 2) {
    quads[q++] = front(M, i); quads[q++] = front(M, (i + 1) % N);
    quads[q++] = front(M, (i + 2) % N); quads[q++] = frontApex;
  }
  for (let k = 0; k < M; k++) for (let i = 0; i < N; i++) {
    const j = (i + 1) % N;
    quads[q++] = back(k, i);      quads[q++] = back(k + 1, i);
    quads[q++] = back(k + 1, j);  quads[q++] = back(k, j);
  }
  for (let i = 0; i < N; i += 2) {
    quads[q++] = back(M, i); quads[q++] = backApex;
    quads[q++] = back(M, (i + 2) % N); quads[q++] = back(M, (i + 1) % N);
  }
  // An empty array: `tris` is part of this module's returned shape and
  // a consumer destructuring it should get something iterable, not undefined.
  const tris = new Uint32Array(0);

  // The same spill test on the apex path. A star-shaped outline spills nothing here, so this costs one
  // subdivision and changes no accepted shape — but a spined outline that somehow reaches this path
  // must not escape a check the grid path applies.
  const spillA = hollowSpill(pts, positions, quads, dmax);
  if (!allowSpill && spillA.frac > SPILL_LIMIT) return spilled(spillA, escaping, dmax);
  return {
    ok: true, reason: null, why: '',
    positions, quads, tris,
    // The prepared outline, returned rather than re-derived by the caller. `repinRim` needs the
    // exact polygon this cage was built against — resampled, closed, wound and deduplicated by
    // `prepareOutline`. Preparing it a second time from the same raw stroke is not guaranteed to
    // agree, and a disagreement lands a pin one vertex out.
    outline: pts,
    nv: V, N, M,
    dmax, centre, area, spacing, diag,
    height: effectiveH, requestedHeight: H, capped: H > dmax * 4.0,
    profile, family, smoothing: passes, bottomScale,
    spined: !!spineTarget, escaping,
    equatorCount: N,
    // The smoothed field itself, returned so it can be asserted on. The heights are a monotone
    // but non-linear function of it, so a midpoint in the field is not a midpoint in z, and a
    // check reading only geometry cannot tell a damped step from an undamped one.
    field: d,
  };
}

// Annular puff — a puff with a hole. The apex becomes the inner rim.
// The disc puff (above) rises from one shared equator to an apex; an annular puff has two shared
// equators — the outer rim and the inner (hole) rim, both at z = 0 — and no apex. Front bands run
// from the outer equator inward to the inner equator at z = +h, back bands mirror at z = -h, and
// front and back meet at both rims, so the tube closes around the hole. Closed genus 1: V-E+F = 0.
// The height is the same distance-field law, but d is the distance to the nearest rim (outer or
// hole), so h → 0 at both and the surface meets its mirror there with nothing to boolean. This
// builder takes one hole and requires both loops star-shaped about the hole's center, so an
// angular correspondence exists with no twist; two or more holes take buildMultiHolePuff, and a
// single non-star ring is a named refusal.

/** Which loop is the outer and which are holes — the app-facing classifier the annular path reads.
 *  Distinct from `classifyLoops` above, which is `buildPuff`'s plain refusal.
 *  Returns { ok, kind:'single'|'annular'|'multi', outer, holes[], why }. A hole nested in a hole
 *  (an island), overlapping holes, or no single containing outer are refused by name. */
export function classifyPuffLoops(rawLoops) {
  const loops = rawLoops.map((L) => (typeof L[0] === 'number'
    ? (() => { const o = []; for (let i = 0; i + 1 < L.length; i += 2) o.push([L[i], L[i + 1]]); return o; })()
    : L.map((p) => p.slice())));
  if (loops.length === 0) return { ok: false, kind: 'empty', why: 'Nothing was drawn.' };
  if (loops.length === 1) return { ok: true, kind: 'single', outer: loops[0], holes: [] };
  // containment matrix: contains[i][j] = every vertex of loop j lies inside loop i
  const contains = loops.map((Li) => loops.map((Lj) => Lj === Li ? false : Lj.every((p) => pointInPolygon(Li, p[0], p[1]))));
  // the outer is the one nobody contains
  const outers = loops.map((_, i) => i).filter((i) => !contains.some((row, k) => k !== i && row[i]));
  if (outers.length !== 1) {
    // Two loops that cross each other are drawn wrong, not two shapes.
    for (let a = 0; a < loops.length; a++) for (let b = a + 1; b < loops.length; b++) {
      for (let i = 0; i < loops[a].length; i++) for (let j = 0; j < loops[b].length; j++) {
        if (annSegsCross(loops[a][i], loops[a][(i + 1) % loops[a].length], loops[b][j], loops[b][(j + 1) % loops[b].length])) return { ok: false, kind: 'overlap', why: 'these loops cross — draw them clear of each other.' };
      }
    }
    return { ok: false, kind: 'multi', why: 'two separate shapes — puff them one at a time.' };
  }
  const oi = outers[0];
  const holeIdx = loops.map((_, i) => i).filter((i) => i !== oi);
  // every hole must be inside the outer, none inside another hole, none overlapping another hole
  for (const h of holeIdx) {
    if (!contains[oi][h]) return { ok: false, kind: 'multi', why: 'these shapes sit side by side — puff them one at a time.' };
    for (const g of holeIdx) {
      if (g === h) continue;
      if (contains[h][g]) return { ok: false, kind: 'island', why: 'One hole sits inside another — that is a shape inside a hole, which a puff cannot be. Switch its Form to Solid.' };
      // crude overlap: any vertex of g inside h without full containment == a crossing
      const anyIn = loops[g].some((p) => pointInPolygon(loops[h], p[0], p[1]));
      let segCross = false;
      for (let i = 0; i < loops[g].length && !segCross; i++) {
        const a1 = loops[g][i], a2 = loops[g][(i + 1) % loops[g].length];
        for (let j = 0; j < loops[h].length && !segCross; j++) {
          const b1 = loops[h][j], b2 = loops[h][(j + 1) % loops[h].length];
          if (annSegsCross(a1, a2, b1, b2)) segCross = true;
        }
      }
      if ((anyIn && !contains[h][g]) || segCross) return { ok: false, kind: 'overlap', why: 'two holes overlap — draw them clear of each other.' };
    }
  }
  return { ok: true, kind: 'annular', outer: loops[oi], holes: holeIdx.map((i) => loops[i]), outerIndex: oi };
}

// Signed area (CCW positive), centroid, ray-to-loop radius, and point-to-boundary distance — the
// small planar helpers the annular build needs.
function annPolyCentroid(P) {
  let a = 0, cx = 0, cy = 0;
  for (let i = 0; i < P.length; i++) { const p = P[i], q = P[(i + 1) % P.length]; const cr = p[0] * q[1] - q[0] * p[1]; a += cr; cx += (p[0] + q[0]) * cr; cy += (p[1] + q[1]) * cr; }
  a *= 0.5;
  if (Math.abs(a) < 1e-12) { let sx = 0, sy = 0; for (const p of P) { sx += p[0]; sy += p[1]; } return [sx / P.length, sy / P.length]; }
  return [cx / (6 * a), cy / (6 * a)];
}
function annRayRadius(P, c, dx, dy) { // nearest positive hit of the ray c+t·d against the loop
  let best = null;
  for (let i = 0; i < P.length; i++) {
    const a = P[i], b = P[(i + 1) % P.length];
    const ex = b[0] - a[0], ey = b[1] - a[1];
    const den = dx * ey - dy * ex; if (Math.abs(den) < 1e-12) continue;
    const ax = a[0] - c[0], ay = a[1] - c[1];
    const t = (ax * ey - ay * ex) / den, s = (ax * dy - ay * dx) / den;
    if (t >= 0 && s >= -1e-9 && s <= 1 + 1e-9 && (best === null || t < best)) best = t;
  }
  return best;
}
function annDistToPoly(P, x, y) {
  let best = Infinity;
  for (let i = 0; i < P.length; i++) {
    const a = P[i], b = P[(i + 1) % P.length];
    const ex = b[0] - a[0], ey = b[1] - a[1], L2 = ex * ex + ey * ey;
    let t = L2 > 0 ? ((x - a[0]) * ex + (y - a[1]) * ey) / L2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t;
    const d = Math.hypot(a[0] + t * ex - x, a[1] + t * ey - y); if (d < best) best = d;
  }
  return best;
}
function annSignedArea(P) { let a = 0; for (let i = 0; i < P.length; i++) { const p = P[i], q = P[(i + 1) % P.length]; a += p[0] * q[1] - q[0] * p[1]; } return a / 2; }
function annSegsCross(a, b, c, d) {
  const o = (p, q, r) => Math.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]));
  return o(a, b, c) !== o(a, b, d) && o(c, d, a) !== o(c, d, b);
}
function annQuadIsBowtie(p0, p1, p2, p3) {
  return annSegsCross(p0, p1, p2, p3) || annSegsCross(p1, p2, p3, p0);
}

/**
 * Annular offset — synthesize a hole (or a bigger rim) by offsetting a loop. `dist > 0` moves the
 * wall inward (the offset is a smaller loop, used as the hole); `dist < 0` moves it outward (the
 * offset becomes the outer rim and the drawn loop is the hole). This is the backend for Puff's
 * Annular toggle + wall-width slider. Per-edge offset with the vertices recomputed as the offset
 * lines' intersections — clean on a convex or gently-concave loop, and refused, never silently
 * broken, when a sharp concavity or a thin neck makes the offset self-intersect, collapse, or
 * invert (the same refusal contract the rest of the puff kernel keeps).
 * @returns { ok, loop } or { ok:false, reason, why }
 */
export function offsetLoop(rawLoop, dist) {
  const P0 = (typeof rawLoop[0] === 'number' ? (() => { const o = []; for (let i = 0; i + 1 < rawLoop.length; i += 2) o.push([rawLoop[i], rawLoop[i + 1]]); return o; })() : rawLoop.map((p) => p.slice()));
  const P = P0.filter((p, i) => { const q = P0[(i - 1 + P0.length) % P0.length]; return Math.hypot(p[0] - q[0], p[1] - q[1]) > 1e-9; });
  const n = P.length;
  if (n < 3) return { ok: false, reason: 'degenerate', why: 'That loop has too few distinct points to offset.' };
  if (dist === 0) return { ok: true, loop: P.map((p) => p.slice()) };
  const ccw = annSignedArea(P) > 0;
  // inward (dist>0) is the left normal for a CCW loop; flip for CW or for an outward offset.
  const sign = (ccw ? 1 : -1) * (dist > 0 ? 1 : -1);
  const off = Math.abs(dist);
  // each edge shifted along its inward normal; a vertex is where consecutive shifted edges meet.
  const lines = []; // {px,py,dx,dy}
  for (let i = 0; i < n; i++) {
    const a = P[i], b = P[(i + 1) % n];
    let dx = b[0] - a[0], dy = b[1] - a[1]; const L = Math.hypot(dx, dy); dx /= L; dy /= L;
    const nx = -dy * sign, ny = dx * sign; // left normal, oriented inward
    lines.push({ px: a[0] + nx * off, py: a[1] + ny * off, dx, dy });
  }
  const out = [];
  for (let i = 0; i < n; i++) {
    const L1 = lines[(i - 1 + n) % n], L2 = lines[i];
    const den = L1.dx * L2.dy - L1.dy * L2.dx;
    if (Math.abs(den) < 1e-9) { out.push([L2.px, L2.py]); continue; } // parallel edges: take the seam point
    const t = ((L2.px - L1.px) * L2.dy - (L2.py - L1.py) * L2.dx) / den;
    out.push([L1.px + L1.dx * t, L1.py + L1.dy * t]);
  }
  // Validity: same winding sign (not inverted), a real area left, and no self-crossing.
  const a1 = annSignedArea(out);
  if (Math.sign(a1) !== Math.sign(annSignedArea(P)) || Math.abs(a1) < Math.abs(annSignedArea(P)) * 1e-3) {
    return { ok: false, reason: 'collapsed', why: `Offsetting by ${off} pinches this shape off — it has a concavity or a neck narrower than the wall. Use a smaller offset, or draw the inner shape.` };
  }
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    if (i === j || (i + 1) % n === j || (j + 1) % n === i) continue;
    if (annSegsCross(out[i], out[(i + 1) % n], out[j], out[(j + 1) % n])) {
      return { ok: false, reason: 'self-intersect', why: `Offsetting by ${off} makes the shape cross itself. Use a smaller offset, or draw the inner shape.` };
    }
  }
  return { ok: true, loop: out };
}

/**
 * @param outerLoop  one closed outline, [x,y] pairs
 * @param holeLoops  array of closed inner loops (exactly one)
 * @param opts       thickness, profile, family, bands (K), bottomScale, count (N, forced even)
 * @returns { ok, positions, quads, tris, ... } — flat arrays in `buildPuff`'s own shape, or a
 *          named refusal carrying no geometry.
 */
export function buildAnnularPuff(outerLoop, holeLoops, opts = {}) {
  const { thickness = 0.35, profile = 0.5, family = 'tangent', bands = null, bottomScale = 1.0, count = null } = opts;
  if (!Array.isArray(holeLoops) || holeLoops.length === 0) return { ok: false, reason: 'no-hole', why: 'An annular puff needs at least one hole loop.' };
  if (holeLoops.length > 1) return { ok: false, reason: 'multi-hole-todo', why: 'More than one hole builds through the multi-hole path; this single-ring builder takes one.' };
  const outer = outerLoop.map((p) => p.slice());
  const hole = holeLoops[0].map((p) => p.slice());
  const c = annPolyCentroid(hole); // the hole's own center is the ring's center
  // Density sets how many angular samples the ring carries, so it lands as the
  // coarsest editable cage and Subdivide adds detail — parity with the disc
  // puff, where `density` 2..6 sets 6·d·d faces. 4·density even samples give a
  // coarse ring of a few tens of faces at d=2 (2·K·N with K≈N/8). An explicit
  // `count` still overrides it, so a caller can pin a resolution.
  const density = Math.max(2, Math.min(6, Math.round(Number.isFinite(+opts.density) ? +opts.density : 2)));
  let N = count != null ? Math.max(8, count | 0) : 4 * density;
  if (N % 2 !== 0) N += 1; // even: the rims are paired-quad rings

  // Sample both rims by angle about c, so ring i on the outer and ring i on the hole share a
  // radial spoke — no twist. A missed ray means that rim is not star-shaped about c.
  const outerR = [], innerR = [];
  for (let i = 0; i < N; i++) {
    const th = (2 * Math.PI * i) / N, dx = Math.cos(th), dy = Math.sin(th);
    const ro = annRayRadius(outer, c, dx, dy), ri = annRayRadius(hole, c, dx, dy);
    if (ro === null || ri === null) return { ok: false, reason: 'not-star', why: 'The outer and the hole must each be star-shaped about the hole\'s centre for a clean ring, and this pair is not. Round the hole.' };
    if (ri >= ro) return { ok: false, reason: 'hole-escapes', why: 'The hole reaches outside the outer shape.' };
    outerR.push([c[0] + ro * dx, c[1] + ro * dy]); innerR.push([c[0] + ri * dx, c[1] + ri * dy]);
  }
  const K = bands != null ? Math.max(1, bands | 0) : Math.max(2, Math.min(24, Math.round(N / 8)));

  const dAt = (x, y) => Math.min(annDistToPoly(outer, x, y), annDistToPoly(hole, x, y));
  let dmax = 0;
  for (let i = 0; i < N; i++) for (let k = 1; k < K; k++) {
    const t = k / K, x = outerR[i][0] + (innerR[i][0] - outerR[i][0]) * t, y = outerR[i][1] + (innerR[i][1] - outerR[i][1]) * t;
    const d = dAt(x, y); if (d > dmax) dmax = d;
  }
  if (!(dmax > 0)) return { ok: false, reason: 'degenerate', why: 'The ring has no width to puff.' };
  const H0 = thickness * 4 * dmax;
  const H = Math.min(H0, 2 * dmax); // a hemispherical tube is H = dmax; 2·dmax leaves headroom before the walls meet

  /* Per-rim falloff. The outer rim carries the exterior character and each hole
     its own, so a ring can be a gently-sloped exterior with a bulging hole (or
     the reverse). rimProfiles[0] is the outer, [1..] the holes; every family has
     f(1)=1, so all rims reach the same peak H at the medial and the height stays
     continuous where one rim's territory hands off to another (only the slope
     differs, along that ridge). Height at a point uses the nearest rim's foot. */
  const rimProfiles = (Array.isArray(opts.rimProfiles) && opts.rimProfiles.length >= 2)
    ? opts.rimProfiles : [{ family, profile }, { family, profile }];
  const fOuter = PROFILES[rimProfiles[0].family] || tangentProfile, pOuter = rimProfiles[0].profile;
  const fHole = PROFILES[rimProfiles[1].family] || tangentProfile, pHole = rimProfiles[1].profile;
  const heightAtXY = (x, y) => {
    const dOut = annDistToPoly(outer, x, y), dHole = annDistToPoly(hole, x, y);
    const t = Math.min(dOut, dHole) / dmax;
    return H * (dOut <= dHole ? fOuter(t, pOuter) : fHole(t, pHole));
  };

  const nInterm = K - 1;
  const V = 2 * N + 2 * nInterm * N;
  const positions = new Float32Array(V * 3);
  const put = (idx, x, y, z) => { positions[idx * 3] = x; positions[idx * 3 + 1] = y; positions[idx * 3 + 2] = z; };
  const outerEq = (i) => i;                                             // 0..N-1, z = 0
  const innerEq = (i) => N + i;                                         // N..2N-1, z = 0
  const F = (k, i) => (k === 0 ? outerEq(i) : k === K ? innerEq(i) : 2 * N + (k - 1) * N + i);            // front ring k
  const B = (k, i) => (k === 0 ? outerEq(i) : k === K ? innerEq(i) : 2 * N + nInterm * N + (k - 1) * N + i); // back ring k
  for (let i = 0; i < N; i++) { put(outerEq(i), outerR[i][0], outerR[i][1], 0); put(innerEq(i), innerR[i][0], innerR[i][1], 0); }
  const bs = bottomScale != null ? bottomScale : 1.0;
  let topFactor = bs >= 0 ? 1.0 : Math.max(0, 1.0 + bs);
  let bottomFactor = bs >= 0 ? bs : 1.0;
  if (opts.flip) {
    const tmp = topFactor;
    topFactor = bottomFactor;
    bottomFactor = tmp;
  }
  for (let k = 1; k < K; k++) for (let i = 0; i < N; i++) {
    const t = k / K, x = outerR[i][0] + (innerR[i][0] - outerR[i][0]) * t, y = outerR[i][1] + (innerR[i][1] - outerR[i][1]) * t;
    const h = heightAtXY(x, y);
    put(F(k, i), x, y, h * topFactor); put(B(k, i), x, y, -h * bottomFactor);
  }
  const quads = new Uint32Array(2 * K * N * 4);
  let q = 0;
  for (let k = 0; k < K; k++) for (let i = 0; i < N; i++) { const j = (i + 1) % N; quads[q++] = F(k, i); quads[q++] = F(k, j); quads[q++] = F(k + 1, j); quads[q++] = F(k + 1, i); }
  for (let k = 0; k < K; k++) for (let i = 0; i < N; i++) { const j = (i + 1) % N; quads[q++] = B(k, i); quads[q++] = B(k + 1, i); quads[q++] = B(k + 1, j); quads[q++] = B(k, j); }
  const tris = new Uint32Array(0);
  for (let qi = 0; qi < quads.length; qi += 4) {
    const i0 = quads[qi], i1 = quads[qi + 1], i2 = quads[qi + 2], i3 = quads[qi + 3];
    const p0 = [positions[i0 * 3], positions[i0 * 3 + 1]], p1 = [positions[i1 * 3], positions[i1 * 3 + 1]];
    const p2 = [positions[i2 * 3], positions[i2 * 3 + 1]], p3 = [positions[i3 * 3], positions[i3 * 3 + 1]];
    if (annQuadIsBowtie(p0, p1, p2, p3)) return { ok: false, reason: 'bowtie-quad', why: 'that annular ring produces self-intersecting quads' };
  }
  const solved = puffRimLimitSolve({
    ok: true, reason: null, why: '', positions, quads, tris,
    nv: V, N, M: K, K, dmax, centre: c, area: null, spacing: null,
    height: H, requestedHeight: H0, capped: H0 > 2 * dmax,
    profile, family, bottomScale, holes: 1, annular: true,
    outerEquatorCount: N, innerEquatorCount: N,
  }, [outer, hole]);
  return opts.lifts ? puffLiftRims(solved, [outer, hole], opts.lifts) : solved;
}

// Multiple holes — a puff with two or more holes, one closed genus-N cage
/* The rim is a collar, not a projected grid. A grid clipped to a region with holes puts an
   extraordinary vertex on the drawn outline and lets its rim wander a cell-width off the curve
   (measured 3–6 mm on a 40 mm blob), and the silhouette must be the
   line that was drawn. So the interior is a grid set back from every loop; each staircase
   boundary ring is resampled evenly along its owning loop (same count, same winding, so the
   correspondence is 1:1 with no zip); and the collar band between staircase and loop carries the
   rim — the rim ring lands exactly on the curve, is valence-4 like the single-hole equator, and
   every extraordinary vertex stays interior where subdivision hides it. Front and back mirror
   across the shared rim rings, giving a closed genus-N cage (Euler = 2 − 2·holes). Orientation is
   propagated by a flood fill from one seed face, then flipped whole if the signed volume points
   inward — a per-face winding rule is not globally consistent across the front/back/collar joins.

   Measured spacing limit. Holes puff as a coarse editable cage only when every gap (hole-to-hole
   and hole-to-rim) is at least ~10% of the object's size. Below that a clean grid channel forces a
   dense mesh — a gap of 0.10·span needs ~2300 faces, 0.05·span ~9400 — so it refuses rather than
   return a mesh too fine to edit. A single hole takes the cleaner radial buildAnnularPuff. */
function mhInPoly(P, x, y) { let c = false; for (let i = 0, j = P.length - 1; i < P.length; j = i++) { const xi = P[i][0], yi = P[i][1], xj = P[j][0], yj = P[j][1]; if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) c = !c; } return c; }
function mhProj(P, x, y) { let best = Infinity, seg = 0, tt = 0, px = 0, py = 0; for (let i = 0; i < P.length; i++) { const a = P[i], b = P[(i + 1) % P.length]; const ex = b[0] - a[0], ey = b[1] - a[1], L2 = ex * ex + ey * ey; let t = L2 > 0 ? ((x - a[0]) * ex + (y - a[1]) * ey) / L2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t; const qx = a[0] + t * ex, qy = a[1] + t * ey; const d = Math.hypot(qx - x, qy - y); if (d < best) { best = d; seg = i; tt = t; px = qx; py = qy; } } return { d: best, seg, t: tt, px, py }; }
function mhDmin(loops, x, y) { let b = Infinity; for (const L of loops) { const d = annDistToPoly(L, x, y); if (d < b) b = d; } return b; }
function mhNearest(loops, x, y) { let bi = 0, bd = Infinity; for (let i = 0; i < loops.length; i++) { const d = annDistToPoly(loops[i], x, y); if (d < bd) { bd = d; bi = i; } } return bi; }
// The smallest gap between any two loops, and which pair holds it. loops[0] is
// the outer boundary and loops[1..] the holes, so a min at index 0 is a
// hole-to-outer-edge gap (advise moving the hole inward) while a min between two
// holes is hole-to-hole (advise moving apart or merging) — the two refusals give
// different advice and this is what tells them apart.
function mhMinGap(loops) {
  let g = Infinity, ga = 0, gb = 1;
  const upd = (d, a, b) => { if (d < g) { g = d; ga = a; gb = b; } };
  for (let a = 0; a < loops.length; a++) for (let b = a + 1; b < loops.length; b++) {
    for (const p of loops[a]) upd(annDistToPoly(loops[b], p[0], p[1]), a, b);
    for (const p of loops[b]) upd(annDistToPoly(loops[a], p[0], p[1]), a, b);
  }
  return { gap: g, a: ga, b: gb, toRim: ga === 0 };
}
function mhArclen(L) { const cum = [0]; let s = 0; for (let i = 0; i < L.length; i++) { const a = L[i], b = L[(i + 1) % L.length]; s += Math.hypot(b[0] - a[0], b[1] - a[1]); cum.push(s); } return { cum, perim: s }; }
function mhArcPos(L, al, pt) { const r = mhProj(L, pt[0], pt[1]); const a = L[r.seg], b = L[(r.seg + 1) % L.length]; return al.cum[r.seg] + r.t * Math.hypot(b[0] - a[0], b[1] - a[1]); }
function mhPointAt(L, al, s) { s = ((s % al.perim) + al.perim) % al.perim; let i = 0; while (i < L.length && al.cum[i + 1] < s) i++; const a = L[i % L.length], b = L[(i + 1) % L.length]; const seg = al.cum[i + 1] - al.cum[i] || 1, t = (s - al.cum[i]) / seg; return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]; }
function mhMeshTop(outer, holes, res, back) {
  const loops = [outer, ...holes];
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const p of outer) { lo[0] = Math.min(lo[0], p[0]); lo[1] = Math.min(lo[1], p[1]); hi[0] = Math.max(hi[0], p[0]); hi[1] = Math.max(hi[1], p[1]); }
  const cell = Math.max(hi[0] - lo[0], hi[1] - lo[1]) / res, setback = back * cell;
  const nx = Math.ceil((hi[0] - lo[0]) / cell) + 2, ny = Math.ceil((hi[1] - lo[1]) / cell) + 2;
  const ox = lo[0] - cell, oy = lo[1] - cell, gx = (i) => ox + i * cell, gy = (j) => oy + j * cell;
  const keep = []; for (let i = 0; i <= nx; i++) { keep[i] = []; for (let j = 0; j <= ny; j++) { const x = gx(i), y = gy(j); keep[i][j] = mhInPoly(outer, x, y) && !holes.some((h) => mhInPoly(h, x, y)) && mhDmin(loops, x, y) >= setback; } }
  const idx = [], V = []; let n = 0;
  for (let i = 0; i <= nx; i++) { idx[i] = []; for (let j = 0; j <= ny; j++) { if (keep[i][j]) { idx[i][j] = n++; V.push([gx(i), gy(j)]); } else idx[i][j] = -1; } }
  const quads = []; for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) { const a = idx[i][j], b = idx[i + 1][j], c = idx[i + 1][j + 1], d = idx[i][j + 1]; if (a >= 0 && b >= 0 && c >= 0 && d >= 0) quads.push([a, b, c, d]); }
  const he = new Map(), ek = (a, b) => a + '_' + b;
  for (const f of quads) for (let k = 0; k < 4; k++) he.set(ek(f[k], f[(k + 1) % 4]), [f[k], f[(k + 1) % 4]]);
  const bd = new Map(); for (const [, [a, b]] of he) if (!he.has(ek(b, a))) bd.set(a, b);
  const gbLoops = [], seen = new Set();
  for (const start of bd.keys()) { if (seen.has(start)) continue; const L = []; let cur = start, g = 0; while (cur !== undefined && !seen.has(cur) && g++ < 1e5) { seen.add(cur); L.push(cur); cur = bd.get(cur); } if (L.length >= 3) gbLoops.push(L); }
  return { V, quads, gbLoops };
}
function mhRelax(V, quads, gbLoops, iters) {
  const adj = new Map(), add = (a, b) => { if (!adj.has(a)) adj.set(a, new Set()); adj.get(a).add(b); };
  for (const f of quads) for (let k = 0; k < 4; k++) { add(f[k], f[(k + 1) % 4]); add(f[(k + 1) % 4], f[k]); }
  const onBd = new Set(); for (const L of gbLoops) for (const v of L) onBd.add(v);
  for (let it = 0; it < iters; it++) { const nV = V.map((p) => p.slice()); for (let i = 0; i < V.length; i++) { if (onBd.has(i)) continue; const nb = adj.get(i); if (!nb) continue; let sx = 0, sy = 0, c = 0; for (const j of nb) { sx += V[j][0]; sy += V[j][1]; c++; } nV[i] = [sx / c, sy / c]; } for (let i = 0; i < V.length; i++) V[i] = nV[i]; }
}
// Collar grading: one intermediate ring at 0.5 of the band's width carrying 0.3 of the staircase
// height. Placing it below the geometric midpoint biases the rise toward the coarser staircase side
// of the band, where the ring edges are longer, so both split quads land near the same aspect. The
// height fraction is tuned to minimize the worst aspect over the densest packing the cage builds
// (eight holes at the spacing floor), where the rim ring is finest and the near-rim quad governs.
const MH_COLLAR_FH = [0.5];
const RAD_COLLAR_FZ = 0.3;
const MH_COLLAR_FZ = [0.3];
/* The collar band bridges the staircase boundary ring (at interior height) to the rim ring (on the
   drawn loop, z = 0). Left as a single band its rail edges span the whole vertical rise while its
   ring edges stay short, so the 3D quads are tall and thin. fH places one or more intermediate
   rings across the band horizontally (fraction from rim toward the staircase); each splits the rise
   into shorter quads. The matching height fractions fZ are applied at lift, where the staircase
   height is known. */
function mhCollars(V, quads, gbLoops, loops, fH) {
  const als = loops.map(mhArclen); const rimRings = [];
  const levels = Array.isArray(fH) ? fH.filter((f) => f > 0 && f < 1).slice().sort((a, b) => a - b) : [];
  for (const gb of gbLoops) {
    const votes = new Map(); for (const vi of gb) { const o = mhNearest(loops, V[vi][0], V[vi][1]); votes.set(o, (votes.get(o) || 0) + 1); }
    let owner = 0, mx = -1; for (const [o, c] of votes) if (c > mx) { mx = c; owner = o; }
    const L = loops[owner], al = als[owner], m = gb.length;
    const s0 = mhArcPos(L, al, V[gb[0]]); const s1 = mhArcPos(L, al, V[gb[1 % m]]); const dS = ((s1 - s0 + al.perim * 1.5) % al.perim) - al.perim * 0.5; const dir = dS >= 0 ? 1 : -1;
    const rim = []; for (let i = 0; i < m; i++) { const p = mhPointAt(L, al, s0 + dir * (i / m) * al.perim); const ri = V.length; V.push([p[0], p[1]]); rim.push(ri); }
    // Intermediate rings, ordered rim -> staircase. Each ring i sits at fraction f of the way from
    // the rim vertex toward its staircase vertex; the staircase ring itself is the last band's top.
    const mids = []; for (const f of levels) { const ring = []; for (let i = 0; i < m; i++) { const a = V[rim[i]], b = V[gb[i]]; const mi = V.length; V.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]); ring.push(mi); } mids.push(ring); }
    rimRings.push({ ring: rim, owner, mids, gbRef: gb.slice() });
    const stack = [rim, ...mids, gb]; // bottom (z=0) to top (staircase height)
    for (let b = 0; b + 1 < stack.length; b++) { const lo = stack[b], up = stack[b + 1]; for (let i = 0; i < m; i++) { const j = (i + 1) % m; quads.push([up[i], up[j], lo[j], lo[i]]); } }
  }
  return rimRings;
}
function mhOrient(P, quads) {
  const eface = new Map(), uk = (a, b) => (a < b ? a + '_' + b : b + '_' + a);
  for (let fi = 0; fi < quads.length; fi++) { const f = quads[fi]; for (let k = 0; k < 4; k++) { const k2 = uk(f[k], f[(k + 1) % 4]); if (!eface.has(k2)) eface.set(k2, []); eface.get(k2).push(fi); } }
  const seen = new Array(quads.length).fill(false);
  const has = (f, a, b) => { for (let k = 0; k < 4; k++) if (f[k] === a && f[(k + 1) % 4] === b) return true; return false; };
  for (let s0 = 0; s0 < quads.length; s0++) { if (seen[s0]) continue; seen[s0] = true; const st = [s0];
    while (st.length) { const fi = st.pop(), f = quads[fi]; for (let k = 0; k < 4; k++) { const a = f[k], b = f[(k + 1) % 4]; for (const gj of (eface.get(uk(a, b)) || [])) { if (gj === fi || seen[gj]) continue; if (has(quads[gj], a, b)) quads[gj].reverse(); seen[gj] = true; st.push(gj); } } } }
  let vol = 0; const tri = (p, q, r) => p[0] * (q[1] * r[2] - q[2] * r[1]) - p[1] * (q[0] * r[2] - q[2] * r[0]) + p[2] * (q[0] * r[1] - q[1] * r[0]);
  for (const f of quads) { const A = P[f[0]], B = P[f[1]], C = P[f[2]], D = P[f[3]]; vol += tri(A, B, C) + tri(A, C, D); }
  if (vol < 0) for (const f of quads) f.reverse();
}
function mhCheck(P, quads, target, nRims, nLoops) {
  const eu = new Map(), dir = new Map();
  for (const f of quads) for (let k = 0; k < 4; k++) { const a = f[k], b = f[(k + 1) % 4]; const u = a < b ? a + '_' + b : b + '_' + a; eu.set(u, (eu.get(u) || 0) + 1); const dk = a + '>' + b; dir.set(dk, (dir.get(dk) || 0) + 1); }
  let boundary = 0, nonmanifold = 0, misor = 0; for (const c of eu.values()) { if (c === 1) boundary++; else if (c > 2) nonmanifold++; } for (const c of dir.values()) if (c !== 1) misor++;
  const used = new Set(); for (const f of quads) for (const v of f) used.add(v);
  const euler = P.length - eu.size + quads.length;
  return { F: quads.length, ok: boundary === 0 && nonmanifold === 0 && misor === 0 && (P.length - used.size) === 0 && euler === target && nRims === nLoops };
}

/**
 * Grid multi-hole puff — the fallback path. A grid set back from every loop, each staircase
 * boundary carried onto its drawn curve by a valence-4 collar. Every rim vertex is valence-4, but
 * it needs ~10% clearance (a clean grid channel) between loops or it refuses rather than mesh
 * dense. `buildMultiHolePuff` reaches here only when the radial/medial primary cannot do the job
 * (N ≥ 4, or a cell that is not star-shaped about its hole, or an invalid medial junction).
 * opts: thickness (default 0.35), bottomScale (1.0), setback (1.0 cell), relax (2 Laplacian passes),
 * faceCap (2000 — the coarse-cage ceiling), and rimProfiles — one { family, profile } per loop,
 * [0] the outer rim and [1..] the holes, defaulting to a round torus (bulge = 1) everywhere.
 * @returns the buildAnnularPuff shape { ok, positions, quads, tris, nv, dmax, height, holes, ... }
 *          or { ok:false, reason, why }.
 */
function buildGridMultiHolePuff(outerLoop, holeLoops, opts = {}) {
  const outer = outerLoop.map((p) => p.slice());
  const holes = holeLoops.map((h) => h.map((p) => p.slice()));
  if (holes.length < 2) return { ok: false, reason: 'not-multi', why: 'Two or more holes are needed here; a single hole takes the ordinary annular puff.' };
  const loops = [outer, ...holes];
  const back = opts.setback != null ? opts.setback : 1.0;
  const faceCap = opts.faceCap != null ? opts.faceCap : Math.min(4800, Math.max(2500, 600 * (1 + holes.length)));
  const iters = opts.relax != null ? opts.relax : 2;
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const p of outer) { lo[0] = Math.min(lo[0], p[0]); lo[1] = Math.min(lo[1], p[1]); hi[0] = Math.max(hi[0], p[0]); hi[1] = Math.max(hi[1], p[1]); }
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
  const gapInfo = mhMinGap(loops);
  const gap = gapInfo.gap;
  if (!(gap > 0)) return { ok: false, reason: 'overlap', why: 'these loops cross or touch — move them apart.' };
  // The refusal names the tight pair (`gapInfo.toRim`: a hole against the outer edge moves inward;
  // two close holes move apart) and the gap measured, nothing it cannot measure: the room a grid
  // needs depends on every loop at once. House shape: lowercase, em-dash, lowercase second sentence
  // — matching the overlap string above.
  const g = gap.toFixed(1);
  const tooCloseWhy = () => gapInfo.toRim
    ? 'a hole sits too close to the edge for a puff — ' + g + ' out. move it inward.'
    : 'holes too close for a puff — smallest gap ' + g + '. move them apart.';
  // Density sets grid detail relative to the channel floor, so the multi-hole
  // cage matches the disc puff's semantics: the lowest density is the coarsest
  // cage the geometry allows and higher densities add detail (Subdivide takes it
  // further, nothing takes it away). The floor is `gap/(2·back+2)` — the coarsest
  // cell that still threads a setback grid through the narrowest gap — and it is
  // never crossed: `densFine` is 1 at density 2 (exactly the floor) and shrinks
  // the cell from there. It cannot make the grid coarser than the channel, and
  // it never touches the RES_CAP feasibility refusal below. A coarser start than
  // the floor is not reachable: the retry loop only shrinks, and starting past
  // the floor false-refuses buildable cases it cannot climb back to.
  const density = Math.max(2, Math.min(6, Math.round(Number.isFinite(+opts.density) ? +opts.density : 2)));
  const densFine = 2 / density; // 1 at d2 (channel floor), 1/3 at d6
  const target = 2 - 2 * holes.length;
  const bs = opts.bottomScale != null ? opts.bottomScale : 1.0;
  let topFactor = bs >= 0 ? 1.0 : Math.max(0, 1.0 + bs);
  let bottomFactor = bs >= 0 ? bs : 1.0;
  if (opts.flip) {
    const tmp = topFactor;
    topFactor = bottomFactor;
    bottomFactor = tmp;
  }
  const rimProfiles = (Array.isArray(opts.rimProfiles) && opts.rimProfiles.length >= loops.length) ? opts.rimProfiles : loops.map(() => ({ family: 'bulge', profile: 1 }));
  // One graded intermediate ring in the collar band: horizontal placement collarFH (fraction from
  // rim toward the staircase) and height collarFZ (fraction of the staircase-ring height). This
  // splits the tall rim-collar quads into two shorter ones and roughly halves their worst aspect.
  const collarFH = Array.isArray(opts.collarFH) ? opts.collarFH : MH_COLLAR_FH;
  const collarFZ = Array.isArray(opts.collarFZ) ? opts.collarFZ : MH_COLLAR_FZ;
  // A vanishing gap would drive res -> infinity and the grid is O(res^2), so a coarse cage that
  // cannot thread the channel must be refused before any allocation, never built and rejected
  // after. RES_CAP bounds the grid; if the coarsest channel-threading cell already needs more,
  // the holes are too close for a coarse cage. This also catches overlapping loops (gap -> 0).
  const RES_CAP = 120;
  const resNeeded = Math.ceil(span / Math.min(gap / (2 * back + 2.0), span / 10));
  if (!(resNeeded <= RES_CAP)) return { ok: false, reason: 'too-close', why: tooCloseWhy() };
  let cell = Math.min(gap / (2 * back + 2.0), span / 10) * densFine;
  let built = null;
  for (let attempt = 0; attempt < 6; attempt++) {
    const res = Math.min(RES_CAP, Math.max(8, Math.ceil(span / cell)));
    const top = mhMeshTop(outer, holes, res, back);
    mhRelax(top.V, top.quads, top.gbLoops, iters);
    const rimRings = mhCollars(top.V, top.quads, top.gbLoops, loops, collarFH);
    const rimSet = new Set(); for (const r of rimRings) for (const v of r.ring) rimSet.add(v);
    const nV = top.V.length;
    let dmax = 0; for (const p of top.V) { const d = mhDmin(loops, p[0], p[1]); if (d > dmax) dmax = d; }
    const H0 = (opts.thickness != null ? opts.thickness : 0.35) * 4 * dmax, H = Math.min(H0, 2 * dmax);
    const heightAt = (x, y) => { const o = mhNearest(loops, x, y); const t = Math.min(mhDmin(loops, x, y) / (dmax || 1), 1); const rp = rimProfiles[o] || { family: 'bulge', profile: 1 }; return H * (PROFILES[rp.family] || tangentProfile)(t, rp.profile); };
    // Height per top vertex. The collar's intermediate rings are graded: each carries a fraction of
    // its staircase vertex's height, so the vertical rise is shared across the collar's shorter
    // quads rather than falling on a single tall band. Front and back read the same z so the mirror
    // stays exact.
    const zTop = new Float64Array(nV);
    for (let i = 0; i < nV; i++) { const p = top.V[i]; zTop[i] = heightAt(p[0], p[1]); }
    for (const r of rimRings) { if (!r.mids) continue; for (let l = 0; l < r.mids.length; l++) { const fz = collarFZ[l] != null ? collarFZ[l] : collarFH[l]; for (let i = 0; i < r.mids[l].length; i++) { const g = top.V[r.gbRef[i]]; zTop[r.mids[l][i]] = heightAt(g[0], g[1]) * fz; } } }
    for (const r of rimRings) for (const v of r.ring) zTop[v] = 0;
    const P = [], backOf = new Array(nV).fill(-1);
    for (let i = 0; i < nV; i++) { const p = top.V[i]; P.push([p[0], p[1], zTop[i] * topFactor]); }
    for (let i = 0; i < nV; i++) { if (rimSet.has(i)) { backOf[i] = i; } else { const p = top.V[i]; backOf[i] = P.length; P.push([p[0], p[1], -zTop[i] * bottomFactor]); } }
    const quads = []; for (const f of top.quads) quads.push(f.slice()); for (const f of top.quads) quads.push(f.map((v) => backOf[v]));
    mhOrient(P, quads);
    const chk = mhCheck(P, quads, target, rimRings.length, loops.length);
    if (chk.ok) {
      // The collar refinement adds a fixed number of quads (both mirror sides) that has nothing to
      // do with how tightly the holes are packed, so the coarse-cage ceiling is measured against
      // the un-refined face count, which the collar does not change.
      // It is judged at density 2, not at the requested density. Density refines the grid (finer
      // cell = ~1/densFine× the resolution = ~1/densFine² the faces), which is a choice about detail,
      // not a statement that the holes are too close. Scaling the count back by densFine² recovers
      // the density-2 face count, so a shape that is too close for a coarse cage refuses at every
      // density and one that builds coarse builds at every density (finer, bounded by RES_CAP).
      let collarExtra = 0; for (const r of rimRings) if (r.mids) collarExtra += r.mids.length * r.ring.length * 2;
      if ((chk.F - collarExtra) * densFine * densFine > faceCap) return { ok: false, reason: 'too-close', why: tooCloseWhy() };
      built = { P, quads, dmax, H, H0 }; break;
    }
    cell *= 0.8;
  }
  if (!built) return { ok: false, reason: 'no-cage', why: tooCloseWhy() };
  const nv = built.P.length;
  const positions = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) { positions[i * 3] = built.P[i][0]; positions[i * 3 + 1] = built.P[i][1]; positions[i * 3 + 2] = built.P[i][2]; }
  const quads = new Uint32Array(built.quads.length * 4);
  for (let i = 0; i < built.quads.length; i++) for (let k = 0; k < 4; k++) quads[i * 4 + k] = built.quads[i][k];
  return {
    ok: true, reason: null, why: '', positions, quads, tris: new Uint32Array(0),
    nv, dmax: built.dmax, height: built.H, requestedHeight: built.H0, capped: built.H0 > 2 * built.dmax,
    bottomScale: bs, holes: holes.length, annular: true, multiHole: true, rimProfiles,
  };
}

// Radial / medial multi-hole — the primary path for any number of holes
/* The medial cut is the ridge, not a rim. A disc with N holes is cut into N cells, one per hole,
   each an annulus meshed radially like buildAnnularPuff — angular spokes from the hole centroid, K
   radial bands, front(+z)/back(−z) sheets. Neighboring cells are welded by sharing the medial
   cut's vertices exactly. The cut is the interior ridge of the puff (front high, back low, they do
   not meet there); front meets back only on the true rims — the outer loop and the hole loops — so
   the result is one closed genus-N shell (Euler = 2 − 2·holes). This carries the silhouette exactly
   (rims land on the drawn curves) as a coarse cage, and it does not need the grid's ~10% clearance:
   the two cases the grid refuses — tightly-spaced holes and a hole near the outer edge — build here.

   The cut feet and junctions are extraordinary. Where an outer-arc rim meets a cut (a foot), and
   where three cells meet (a junction), a vertex is valence 3/5, not 4 — reported, and hidden
   interior under subdivision. Everywhere else the rims stay valence-4 like the single-hole equator.

   A refusal falls back to the grid. A cell that is not star-shaped about its hole centroid
   (`hole-not-star` / `hole-escapes-cell`), a cut that cannot be clipped, or an invalid junction
   returns { ok:false }, and buildMultiHolePuff falls through to buildGridMultiHolePuff. */
function radAddV(top, x, y, isRim) { top.V.push([x, y]); top.rim.push(!!isRim); return top.V.length - 1; }

// One radial annulus cell. ringIdx = ordered cell-boundary vertex indices (rim arc points + shared
// cut points). Builds the hole ring (a ray from c along each ring point's own direction, so every
// spoke is a true radial and no chord can cross the hole) + K interior bands. K adapts to the
// longest spoke so radial edges stay near the ring-segment length. Returns { ok } or a named refusal.
function radMeshCell(top, ringIdx, holeLoop, c, segLen) {
  const M = ringIdx.length;
  // Angles are unwrapped monotonically so the sweep is continuous; a boundary that does not wind
  // once about c, or turns back, is not a cell this can mesh.
  const TAU = Math.PI * 2;
  const ang = ringIdx.map((vi) => { const P = top.V[vi]; return Math.atan2(P[1] - c[1], P[0] - c[0]); });
  const un = [ang[0]];
  for (let j = 1; j < M; j++) { let a = ang[j]; while (a - un[j - 1] > Math.PI) a -= TAU; while (a - un[j - 1] < -Math.PI) a += TAU; un.push(a); }
  const total = un[M - 1] - un[0];
  if (Math.abs(total) < Math.PI * 1.2 || Math.abs(total) > Math.PI * 2.8) return { ok: false, reason: 'cell-not-enclosing', why: 'cell boundary does not enclose hole centroid' };
  for (let j = 1; j < M; j++) {
    if ((un[j] - un[j - 1]) * total <= 1e-9) return { ok: false, reason: 'cell-not-star', why: 'a cell boundary is not star-shaped about its hole' };
  }
  const holeRing = []; let maxSpoke = 0;
  for (let j = 0; j < M; j++) {
    const P = top.V[ringIdx[j]];
    const Lo = Math.hypot(P[0] - c[0], P[1] - c[1]);
    if (Lo < 1e-9) return { ok: false, reason: 'centroid-on-ring', why: 'a cell boundary point coincides with its hole centroid' };
    const th = un[j];
    const dx = Math.cos(th), dy = Math.sin(th);
    const ri = annRayRadius(holeLoop, c, dx, dy);
    if (ri === null) return { ok: false, reason: 'hole-not-star', why: 'a hole is not star-shaped about its centroid along a spoke' };
    if (ri >= Lo) return { ok: false, reason: 'hole-escapes-cell', why: 'a hole rim reaches past its cell boundary along a spoke' };
    if (Lo - ri > maxSpoke) maxSpoke = Lo - ri;
    holeRing.push(radAddV(top, c[0] + ri * dx, c[1] + ri * dy, true));
  }
  /* Rows at equal lifted length, not equal distance. The bulge profile has a vertical tangent at
     the rim, so rows spaced evenly across the floor put the whole climb into the first one: at
     density 4 the rim band carries 74% of the wall's rise and a 96-unit quad stands beside 33-unit
     ring edges, and a higher density adds rows across the flat top where nothing happens — none
     at the wall however high it goes. Each spoke is measured along the
     lifted profile and its rows placed at equal fractions of that; K follows the longest lifted
     spoke at 1.5 ring-segments a row, so the rim quad shrinks with density like every other edge
     (rim aspect ≤ 2 at every density, +16-25% faces over even spacing). */
  let K = Math.max(2, Math.round(maxSpoke / segLen)), spokes = null;
  if (top.heightAt) {
    spokes = []; let longest = 0;
    for (let j = 0; j < M; j++) { const run = radLiftedRun(top, top.V[ringIdx[j]], top.V[holeRing[j]], top.rim[ringIdx[j]], true, 96); spokes.push(run); if (run.L > longest) longest = run.L; }
    /* At least six rows: with fewer the cubic limit cannot stand up inside the
       rim band and density 2 reads as a low-pass of the density-5 wall (23% of
       H apart on the row fixture; 10% with the floor, at 420 → 588 faces). */
    K = Math.max(6, Math.round(longest / (1.5 * segLen)));
  }
  const rings = [ringIdx];
  for (let k = 1; k < K; k++) { const r = []; for (let j = 0; j < M; j++) { const A = top.V[ringIdx[j]], B = top.V[holeRing[j]], t = spokes ? spokes[j].at(spokes[j].L * k / K) : k / K; r.push(radAddV(top, A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, false)); } rings.push(r); }
  // A graded collar at the hole ring — the radial analogue of the grid's rim collar. The height
  // field rises with a vertical tangent off the rim, so the last band climbs nearly the whole dome
  // in one step while its hole-ring edge (short, on a small ring) stays tiny: the junction sliver.
  // One extra ring in the last band, set halfway in horizontally but low in z (a fraction of its
  // interior neighbor's height), splits that climb into two shorter quads. z is assigned in
  // radCloseShell, so tag the collar with the neighbor to scale from.
  const inner = rings[rings.length - 1]; // the interior band adjacent to the hole
  const collar = [];
  for (let j = 0; j < M; j++) {
    const A = top.V[inner[j]], B = top.V[holeRing[j]];
    const mx = (A[0] + B[0]) / 2, my = (A[1] + B[1]) / 2;
    if (mhInPoly(holeLoop, mx, my)) return { ok: false, reason: 'collar-in-hole', why: 'collar falls inside hole' };
    const vi = radAddV(top, mx, my, false);
    collar.push(vi);
    top.collar.push({ vi, ref: inner[j], fz: RAD_COLLAR_FZ });
  }
  rings.push(collar);
  rings.push(holeRing);
  for (let k = 0; k < rings.length - 1; k++) {
    for (let j = 0; j < M; j++) {
      const jn = (j + 1) % M;
      const q = [rings[k][j], rings[k][jn], rings[k + 1][jn], rings[k + 1][j]];
      const p0 = top.V[q[0]], p1 = top.V[q[1]], p2 = top.V[q[2]], p3 = top.V[q[3]];
      if (annQuadIsBowtie(p0, p1, p2, p3)) return { ok: false, reason: 'bowtie-quad', why: 'cell quads self-intersect' };
      top.quads.push(q);
    }
  }
  return { ok: true };
}

// The partition is the power diagram of the holes, clipped to the outer loop. Each hole's cell
// is the set of points nearer its disc (centroid, radius of its containing circle) than any other's,
// inside the outer loop: convex about the disc it contains, so star-shaped about the centroid by
// construction, and the disc of every other hole lies in its own cell. The piece of a radical line
// between two cells is a cut (the ridge); where three cells meet is a junction. A straight cut for
// two holes, a wheel for a ring of holes, a chain of cuts for a row, a cell with no rim for a hole
// among holes — all one construction.
// A vertex is named by what it lies on, never found by coordinate: a junction by its three cells, a
// foot by its two cells and the outer segment, a cut sample by its two cells and its index. Both
// neighbors ask for the same name, so the cage welds with no seam. Junctions closer than a quarter
// segment (four holes on a square meet in one point, up to rounding) are merged once, globally.
function radPartitionCells(outer, holes, opts, rimProfiles) {
  const N = holes.length;
  const cents = holes.map(annPolyCentroid);
  const radii = holes.map((h, i) => h.reduce((m, p) => Math.max(m, Math.hypot(p[0] - cents[i][0], p[1] - cents[i][1])), 0));
  const { segLen } = radSpanSeg(outer, opts);
  const cutStep = segLen * 2.5; // a cut runs nearly radially off a hole; sampled at the arc's step it pins a sliver quad at the seam
  const al = mhArclen(outer);
  // The radical line between discs i and j: perpendicular to the centroid line, crossing it t from ci.
  const plane = (i, j) => {
    let dx = cents[j][0] - cents[i][0], dy = cents[j][1] - cents[i][1]; const D = Math.hypot(dx, dy);
    if (D < 1e-9) return null;
    const t = (D * D + radii[i] * radii[i] - radii[j] * radii[j]) / (2 * D);
    return { m: [cents[i][0] + dx * t / D, cents[i][1] + dy * t / D], dx: dx / D, dy: dy / D };
  };
  const sideOf = (pl, p) => (p[0] - pl.m[0]) * pl.dx + (p[1] - pl.m[1]) * pl.dy;
  const keyJ = (a, b, c) => 'J' + [a, b, c].sort((x, y) => x - y).join(',');
  // A foot is named by its arc position, not its segment: a cut through an outer vertex lands on
  // segment k in one cell and k+1 in the other.
  const keyF = (a, b, p) => 'F' + Math.min(a, b) + ',' + Math.max(a, b) + ':' + Math.round(mhArcPos(outer, al, p) / al.perim * 1e6);
  // Sutherland–Hodgman, carrying on each vertex the label of the edge that leaves it: an outer
  // segment (a rim edge) or a cut against cell j. A new vertex on an outer edge is a foot; on a
  // cut edge it is a junction.
  const cells = [];
  for (let i = 0; i < N; i++) {
    let poly = outer.map((p, k) => ({ p: [p[0], p[1]], key: 'O' + k, rim: true, label: { kind: 'outer', seg: k } }));
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      const pl = plane(i, j); if (!pl) return { ok: false, reason: 'coincident-holes' };
      const out = [];
      const nameFor = (label, p) => label.kind === 'outer' ? keyF(i, j, p) : keyJ(i, j, label.j);
      for (let e = 0; e < poly.length; e++) {
        const a = poly[e], b = poly[(e + 1) % poly.length];
        const sa = sideOf(pl, a.p), sb = sideOf(pl, b.p), ain = sa <= 0, bin = sb <= 0;
        const cross = () => { const t = sa / (sa - sb); return [a.p[0] + (b.p[0] - a.p[0]) * t, a.p[1] + (b.p[1] - a.p[1]) * t]; };
        if (ain && bin) out.push(b);
        else if (ain) { const p = cross(); out.push({ p, key: nameFor(a.label, p), rim: a.label.kind === 'outer', label: { kind: 'cut', j } }); }
        else if (bin) { const p = cross(); out.push({ p, key: nameFor(a.label, p), rim: a.label.kind === 'outer', label: a.label }); out.push(b); }
      }
      poly = out;
      if (poly.length < 3) return { ok: false, reason: 'cell-empty' };
    }
    cells.push(poly);
  }
  // Junctions that coincide up to rounding become one vertex, for every cell that names either;
  // a junction within a quarter segment of a foot merges into the foot (the foot wins, so the
  // vertex stays on the rim) — otherwise three ring vertices sit at one angle from the hole and
  // every row of that spoke inherits a hair-thin quad.
  const rep = new Map();
  const find = (k) => { while (rep.has(k) && rep.get(k) !== k) k = rep.get(k); return k; };
  const union = (a, b) => { a = find(a); b = find(b); if (a === b) return; if (a[0] === 'F') rep.set(b, a); else rep.set(a, b); };
  const short = segLen * 0.25;
  const named0 = [];
  for (const poly of cells) for (const v of poly) if (v.key[0] === 'J' || v.key[0] === 'F') named0.push(v);
  for (let a = 0; a < named0.length; a++) for (let b = a + 1; b < named0.length; b++) {
    const u = named0[a], v = named0[b];
    if (u.key === v.key || (u.key[0] === 'F' && v.key[0] === 'F')) continue;
    if (Math.hypot(u.p[0] - v.p[0], u.p[1] - v.p[1]) < short) union(u.key, v.key);
  }
  const merged = new Map(); // rep key → the position every member shares: a foot's own, else the mean
  for (const poly of cells) for (const v of poly) {
    const r = find(v.key); if (r === v.key && !merged.has(r)) continue;
    const m = merged.get(r) || { x: 0, y: 0, n: 0 };
    if (r[0] === 'F') { if (v.key === r) { m.x = v.p[0]; m.y = v.p[1]; m.n = 1; } else if (m.n === 0) { m.x = v.p[0]; m.y = v.p[1]; m.n = 1; } }
    else { m.x += v.p[0]; m.y += v.p[1]; m.n++; }
    merged.set(r, m);
  }
  // The cage. Every named vertex is made once; arc samples belong to one cell and are not named.
  const top = { V: [], rim: [], quads: [], collar: [] };
  radDomainHeight(top, outer, holes, opts, rimProfiles);
  const named = new Map();
  const vertexFor = (key, p, rim) => {
    if (named.has(key)) return named.get(key);
    const m = merged.get(key);
    const idx = radAddV(top, m ? m.x / m.n : p[0], m ? m.y / m.n : p[1], rim);
    named.set(key, idx); return idx;
  };
  let feet = 0;
  const cutRuns = new Map();
  for (let i = 0; i < N; i++) {
    const poly = [];
    for (const v of cells[i]) { const k = find(v.key); if (poly.length && poly[poly.length - 1].key === k) { poly[poly.length - 1].label = v.label; continue; } poly.push({ ...v, key: k, rim: v.rim || k[0] === 'F' }); }
    if (poly.length > 1 && poly[0].key === poly[poly.length - 1].key) poly.pop(); // the run wraps: the edge leaving it is poly[0]'s own
    if (poly.length < 3) return { ok: false, reason: 'cell-empty' };
    const ring = [];
    for (let e = 0; e < poly.length; e++) {
      const u = poly[e], w = poly[(e + 1) % poly.length];
      if (u.label.kind === 'outer') {
        // A run of outer edges is one arc from foot to foot, resampled by arc length so the rim is
        // as even here as on a single-hole puff; the polygon's own outer vertices are dropped.
        if (u.key[0] !== 'F') continue;
        let f = e + 1; while (poly[f % poly.length].label.kind === 'outer') f++;
        const end = poly[f % poly.length];
        const s1 = mhArcPos(outer, al, u.p), s2 = mhArcPos(outer, al, end.p);
        let len = (s2 - s1 + al.perim) % al.perim;
        if (f === e + 1 && len > al.perim / 2) len -= al.perim; // one segment, walked against the loop's direction
        ring.push(vertexFor(u.key, u.p, true));
        const n = Math.max(1, Math.round(Math.abs(len) / segLen));
        for (let q = 1; q < n; q++) { const pt = mhPointAt(outer, al, s1 + len * q / n); ring.push(radAddV(top, pt[0], pt[1], true)); }
        e = f - 1;
        continue;
      }
      const ui = vertexFor(u.key, u.p, u.rim), wi = vertexFor(w.key, w.p, w.rim);
      ring.push(ui);
      const j = u.label.j, a = u.key < w.key ? u : w, b = a === u ? w : u; // canonical direction, so both cells sample the same points
      // The cut is sampled along its lifted length too (the ridge falls from H to 0 within one
      // step at a foot, the sharpest crease of the whole cage otherwise). One run per cut, from
      // the named endpoints, read by both cells: each cell's own copy of an endpoint differs by
      // rounding after a merge, and a length that sits on a rounding boundary gives the two
      // cells different sample counts — an unwelded cut, an open shell. A foot-to-foot cut keeps
      // at least one lifted vertex (with none it is a rim edge both sheets share, non-manifold);
      // any other cut may be one edge.
      const cutKey = 'C' + Math.min(i, j) + ',' + Math.max(i, j) + ':' + a.key + '>' + b.key;
      let cut = cutRuns.get(cutKey);
      if (!cut) {
        const A = top.V[a === u ? ui : wi], B = top.V[a === u ? wi : ui];
        const run = top.heightAt ? radLiftedRun(top, A, B, !!a.rim, !!b.rim, 128) : null;
        const L = run ? run.L : Math.hypot(B[0] - A[0], B[1] - A[1]);
        const n = Math.max(a.rim && b.rim ? 2 : 1, Math.round(L / cutStep));
        cut = { A, B, run, L, n }; cutRuns.set(cutKey, cut);
      }
      const samples = [];
      for (let q = 1; q < cut.n; q++) { const t = cut.run ? cut.run.at(cut.L * q / cut.n) : q / cut.n; samples.push(vertexFor(cutKey + ':' + q, [cut.A[0] + (cut.B[0] - cut.A[0]) * t, cut.A[1] + (cut.B[1] - cut.A[1]) * t], false)); }
      if (a !== u) samples.reverse();
      ring.push(...samples);
    }
    for (const v of poly) if (v.key[0] === 'F') feet++;
    const r = radMeshCell(top, ring, holes[i], cents[i], segLen); if (!r.ok) return r;
  }
  const built = radCloseShell(top, [outer, ...holes], opts, rimProfiles); if (!built) return { ok: false, reason: 'no-width' };
  const packed = radPack(built, N, rimProfiles);
  packed.feet = feet / 2; // each foot is named by two cells
  return packed;
}

// The height field belongs to the domain, not to the cage. Read off the cage's own vertices, the
// dome's peak dmax (the deepest point from every rim) moves with density — a three-hole fixture
// stands 79 tall at density 2 and 119 at 3 — and no ring can be placed by the height before the
// vertices exist. So it is found on a grid over the region and climbed locally, once; the rings,
// the cut samples and the shell all read this one field.
function radDomainHeight(top, outer, holes, opts, rimProfiles) {
  const loops = [outer, ...holes];
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const p of outer) { lo[0] = Math.min(lo[0], p[0]); lo[1] = Math.min(lo[1], p[1]); hi[0] = Math.max(hi[0], p[0]); hi[1] = Math.max(hi[1], p[1]); }
  const G = 80; let best = 0, bx = 0, by = 0;
  const inside = (x, y) => mhInPoly(outer, x, y) && !holes.some((h) => mhInPoly(h, x, y));
  for (let a = 0; a <= G; a++) for (let b = 0; b <= G; b++) {
    const x = lo[0] + (hi[0] - lo[0]) * a / G, y = lo[1] + (hi[1] - lo[1]) * b / G;
    if (!inside(x, y)) continue;
    const d = mhDmin(loops, x, y); if (d > best) { best = d; bx = x; by = y; }
  }
  let step = Math.max(hi[0] - lo[0], hi[1] - lo[1]) / G;
  for (let it = 0; it < 30; it++) {
    let moved = false;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]) {
      const x = bx + dx * step, y = by + dy * step;
      if (!inside(x, y)) continue;
      const d = mhDmin(loops, x, y); if (d > best) { best = d; bx = x; by = y; moved = true; }
    }
    if (!moved) step /= 2;
  }
  if (!(best > 0)) return;
  const thickness = opts.thickness != null ? opts.thickness : 0.35;
  const H0 = thickness * 4 * best, H = Math.min(H0, 2 * best);
  top.dmax = best; top.H = H; top.H0 = H0;
  top.heightAt = (x, y) => { const o = mhNearest(loops, x, y); const t = Math.min(mhDmin(loops, x, y) / best, 1); const rp = rimProfiles[o] || { family: 'bulge', profile: 1 }; return H * (PROFILES[rp.family] || tangentProfile)(t, rp.profile); };
}
// The lifted length along a straight run from A to B, sampled; `cum` lets a point be placed at a
// fraction of that length rather than of the run. A rim endpoint sits at z = 0.
function radLiftedRun(top, A, B, aRim, bRim, NS) {
  let L = 0, px = A[0], py = A[1], pz = aRim ? 0 : top.heightAt(A[0], A[1]);
  const cum = [0];
  for (let s = 1; s <= NS; s++) {
    const t = s / NS, x = A[0] + (B[0] - A[0]) * t, y = A[1] + (B[1] - A[1]) * t;
    const z = (s === NS && bRim) ? 0 : top.heightAt(x, y);
    L += Math.hypot(x - px, y - py, z - pz); cum.push(L); px = x; py = y; pz = z;
  }
  const at = (want) => { let i = 0; while (i < NS && cum[i + 1] < want) i++; return (i + (want - cum[i]) / ((cum[i + 1] - cum[i]) || 1)) / NS; };
  return { L, at };
}

// lift the 2D front sheet (top) to a closed shell: rim vertices at z=0, interior lifted by the
// per-rim height profile of the nearest rim (matching the grid path's height field), non-rim
// vertices mirrored to −z, front + back quads, orientation flooded and volume-corrected.
function radCloseShell(top, loops, opts, rimProfiles) {
  const thickness = opts.thickness != null ? opts.thickness : 0.35;
  const bs = opts.bottomScale != null ? opts.bottomScale : 1.0;
  let topFactor = bs >= 0 ? 1.0 : Math.max(0, 1.0 + bs);
  let bottomFactor = bs >= 0 ? bs : 1.0;
  if (opts.flip) {
    const tmp = topFactor;
    topFactor = bottomFactor;
    bottomFactor = tmp;
  }
  const nV = top.V.length;
  let dmax = top.dmax || 0;
  if (!dmax) for (let i = 0; i < nV; i++) if (!top.rim[i]) { const d = mhDmin(loops, top.V[i][0], top.V[i][1]); if (d > dmax) dmax = d; }
  if (!(dmax > 0)) return null;
  const H0 = top.H0 || thickness * 4 * dmax, H = top.H || Math.min(H0, 2 * dmax);
  const heightAt = top.heightAt || ((x, y) => { const o = mhNearest(loops, x, y); const t = Math.min(mhDmin(loops, x, y) / (dmax || 1), 1); const rp = rimProfiles[o] || { family: 'bulge', profile: 1 }; return H * (PROFILES[rp.family] || tangentProfile)(t, rp.profile); });
  const P = [], backOf = new Array(nV).fill(-1);
  for (let i = 0; i < nV; i++) { const p = top.V[i]; P.push([p[0], p[1], top.rim[i] ? 0 : heightAt(p[0], p[1])]); }
  // Graded collar rings carry a fraction of their interior neighbor's height (front and back read
  // the same z, so the mirror stays exact). Applied before the mirror pass below.
  for (const c of top.collar) P[c.vi][2] = P[c.ref][2] * c.fz;
  const zBase = new Float64Array(nV);
  for (let i = 0; i < nV; i++) {
    zBase[i] = P[i][2];
    P[i][2] = zBase[i] * topFactor;
  }
  for (let i = 0; i < nV; i++) { if (top.rim[i]) { backOf[i] = i; } else { backOf[i] = P.length; const p = top.V[i]; P.push([p[0], p[1], -zBase[i] * bottomFactor]); } }
  const quads = []; for (const f of top.quads) quads.push(f.slice()); for (const f of top.quads) quads.push(f.map((v) => backOf[v]));
  mhOrient(P, quads);
  return { P, quads, dmax, H, H0, bs };
}

function radPack(built, nHoles, rimProfiles) {
  const nv = built.P.length;
  const positions = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) { positions[i * 3] = built.P[i][0]; positions[i * 3 + 1] = built.P[i][1]; positions[i * 3 + 2] = built.P[i][2]; }
  const quads = new Uint32Array(built.quads.length * 4);
  for (let i = 0; i < built.quads.length; i++) for (let k = 0; k < 4; k++) quads[i * 4 + k] = built.quads[i][k];
  return {
    ok: true, reason: null, why: '', positions, quads, tris: new Uint32Array(0),
    nv, dmax: built.dmax, height: built.H, requestedHeight: built.H0, capped: built.H0 > 2 * built.dmax,
    bottomScale: built.bs, holes: nHoles, annular: true, multiHole: true, radial: true, rimProfiles,
  };
}

function radSpanSeg(outer, opts) {
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const p of outer) { lo[0] = Math.min(lo[0], p[0]); lo[1] = Math.min(lo[1], p[1]); hi[0] = Math.max(hi[0], p[0]); hi[1] = Math.max(hi[1], p[1]); }
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
  const density = Math.max(2, Math.min(6, Math.round(Number.isFinite(+opts.density) ? +opts.density : 2)));
  return { span, segLen: span / (4 + 2 * density) };
}


function buildRadialMultiHolePuff(outerLoop, holeLoops, opts = {}) {
  const outer = outerLoop.map((p) => p.slice());
  const holes = holeLoops.map((h) => h.map((p) => p.slice()));
  const loops = [outer, ...holes];
  // Degenerate containment is declined here. The uniform-angle hole ring samples discrete
  // directions, so a hole poking outside the outer — or two holes overlapping — can build a
  // topologically closed but self-intersecting cage. A vertex-containment test catches both; decline
  // so the grid path (or the dispatcher) refuses cleanly. (Point-to-loop distance stays positive
  // through an overlap, so mhMinGap cannot see this.)
  for (const h of holes) for (const p of h) if (!mhInPoly(outer, p[0], p[1])) return { ok: false, reason: 'hole-outside' };
  for (let a = 0; a < holes.length; a++) for (let b = a + 1; b < holes.length; b++) if (holes[a].some((p) => mhInPoly(holes[b], p[0], p[1])) || holes[b].some((p) => mhInPoly(holes[a], p[0], p[1]))) return { ok: false, reason: 'holes-overlap' };
  const rimProfiles = (Array.isArray(opts.rimProfiles) && opts.rimProfiles.length >= loops.length) ? opts.rimProfiles : loops.map(() => ({ family: 'bulge', profile: 1 }));
  const built = radPartitionCells(outer, holes, opts, rimProfiles);
  if (!built || !built.ok) return built;
  const inv = puffInvariants(built);
  if (!inv.closed || inv.nonManifold || inv.misoriented || inv.euler !== 2 - 2 * holes.length) return { ok: false, reason: 'not-closed', why: 'the radial cage did not close' };
  const P = built.positions;
  for (let q = 0; q < built.quads.length; q += 4) {
    const i0 = built.quads[q], i1 = built.quads[q + 1], i2 = built.quads[q + 2], i3 = built.quads[q + 3];
    const p0 = [P[i0 * 3], P[i0 * 3 + 1]], p1 = [P[i1 * 3], P[i1 * 3 + 1]];
    const p2 = [P[i2 * 3], P[i2 * 3 + 1]], p3 = [P[i3 * 3], P[i3 * 3 + 1]];
    if (annQuadIsBowtie(p0, p1, p2, p3)) return { ok: false, reason: 'bowtie-quad' };
  }
  for (let i = 0; i < built.nv; i++) {
    if (Math.abs(P[i * 3 + 2]) > 1e-4) {
      const x = P[i * 3], y = P[i * 3 + 1];
      for (const h of holes) {
        if (mhInPoly(h, x, y)) return { ok: false, reason: 'vertex-in-hole' };
      }
    }
  }
  return built;
}

/**
 * A puff with two or more holes — one closed genus-N SuperB. Radial/medial is the primary mesher
 * (coarse cage, tight spacing and near-edge holes both fine) for N=2 and N=3 when each cell is
 * star-shaped about its hole; it falls back to the grid builder otherwise (N ≥ 4, a non-star cell,
 * an invalid junction), and only a degenerate input (overlapping loops, a single hole)
 * refuses. Same return shape as buildAnnularPuff/buildGridMultiHolePuff, plus `radial:true` when the
 * radial path built it. opts: thickness, bottomScale, density (2..6), rimProfiles ([0]=outer,[1..]=holes).
 */
/**
 * The limit rim on the drawn curve. A cage rim placed on the curve subdivides to a
 * limit rim inside it — every rim vertex's limit point averages its neighbors,
 * which all lie in the body — by 12.7 units on an r200 outline at density 2
 * (6.9 at 3, 3.6 at 5), and a hole comes out that much larger than drawn. The disc
 * puff solves this with one growth factor about its center; a ring with holes has
 * a rim per loop, so each rim vertex is solved in two stages. First, ten Jacobi
 * rounds move an approximate limit point (n²·v + 4·Σ edge midpoints + Σ face
 * centroids) / (n(n+5)) onto the nearest point of its loop, displacing the vertex
 * in the plane with that stencil's self-weight inverted. The stencil is not the
 * Catmull-Clark limit mask (its regular center weight is 25/36, the mask's 16/36);
 * it only seeds the second stage, which measures the rim of a real two-level
 * ccSubdivide and corrects against it. Swapping in the true mask moves the final
 * worst miss by under 0.1% of the span either way. z stays 0, so the rim stays in
 * the drawing plane and the sheets meet there. Topology is untouched.
 */
function puffRimLimitSolve(p, loops, rounds = 10) {
  const nv = p.nv | 0, pos = p.positions, q = p.quads;
  if (!nv || !q || !q.length) return p;
  const edges = Array.from({ length: nv }, () => new Set()), faces = Array.from({ length: nv }, () => []);
  for (let f = 0; f + 3 < q.length; f += 4) {
    const a = q[f], b = q[f + 1], c = q[f + 2], d = q[f + 3];
    edges[a].add(b); edges[a].add(d); edges[b].add(a); edges[b].add(c); edges[c].add(b); edges[c].add(d); edges[d].add(c); edges[d].add(a);
    faces[a].push(f); faces[b].push(f); faces[c].push(f); faces[d].push(f);
  }
  const polys = loops.map((L) => L.flat());
  const nearest = (poly, x, y) => {
    const m = poly.length / 2; let bd = Infinity, bx = x, by = y;
    for (let i = 0; i < m; i += 1) {
      const j = (i + 1) % m, ax = poly[i * 2], ay = poly[i * 2 + 1], dx = poly[j * 2] - ax, dy = poly[j * 2 + 1] - ay, L2 = dx * dx + dy * dy;
      let t = L2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / L2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t;
      const px = ax + dx * t, py = ay + dy * t, d = Math.hypot(px - x, py - y);
      if (d < bd) { bd = d; bx = px; by = py; }
    }
    return [bx, by, bd];
  };
  // Rim vertices: on the drawing plane. Each belongs to the loop it lies on.
  const rim = [];
  let span = 0;
  for (let i = 0; i < nv; i++) { span = Math.max(span, Math.abs(pos[3 * i]), Math.abs(pos[3 * i + 1])); }
  const tol = Math.max(1e-9, span * 1e-9);
  for (let i = 0; i < nv; i++) {
    if (Math.abs(pos[3 * i + 2]) > tol || edges[i].size < 3) continue;
    let best = Infinity, li = 0;
    for (let k = 0; k < polys.length; k++) { const d = nearest(polys[k], pos[3 * i], pos[3 * i + 1])[2]; if (d < best) { best = d; li = k; } }
    rim.push([i, li]);
  }
  if (!rim.length) return p;
  const limit = (i) => {
    const n = edges[i].size; let mx = 0, my = 0;
    for (const j of edges[i]) { mx += (pos[3 * i] + pos[3 * j]) / 2; my += (pos[3 * i + 1] + pos[3 * j + 1]) / 2; }
    let cx = 0, cy = 0;
    for (const f of faces[i]) { cx += (pos[3 * q[f]] + pos[3 * q[f + 1]] + pos[3 * q[f + 2]] + pos[3 * q[f + 3]]) / 4; cy += (pos[3 * q[f] + 1] + pos[3 * q[f + 1] + 1] + pos[3 * q[f + 2] + 1] + pos[3 * q[f + 3] + 1]) / 4; }
    const w = n * (n + 5);
    return [(n * n * pos[3 * i] + 4 * mx + cx) / w, (n * n * pos[3 * i + 1] + 4 * my + cy) / w, n];
  };
  for (let r = 0; r < rounds; r++) {
    const moves = [];
    for (const [i, li] of rim) {
      const [lx, ly, n] = limit(i);
      const [tx, ty] = nearest(polys[li], lx, ly);
      moves.push([i, (tx - lx) * (n + 5) / (n + 2.25), (ty - ly) * (n + 5) / (n + 2.25)]);
    }
    for (const [i, dx, dy] of moves) { pos[3 * i] += dx; pos[3 * i + 1] += dy; }
  }
  /* Between vertices the limit rim still bows where a cut meets a rim: the edge
     points there average in the cut's lifted samples. Measured on the rim of
     the twice-subdivided cage, each sample's error charged to its nearest rim
     vertex. Row fixture (r200, three r30 holes), worst limit-rim miss: 12.7 →
     3.0 at density 2, 6.9 → 1.5 at 3, 3.6 → 0.9 at 5; the rest is the bow
     between two solved vertices, which no in-plane move of theirs removes. */
  const faceList = []; for (let f = 0; f + 3 < q.length; f += 4) faceList.push([q[f], q[f + 1], q[f + 2], q[f + 3]]);
  for (let r = 0; r < 4; r++) {
    let cage = { vertices: Array.from({ length: nv }, (_, i) => [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]), faces: faceList };
    cage = ccSubdivide(ccSubdivide(cage));
    const acc = new Map();
    for (const v of cage.vertices) {
      if (Math.abs(v[2]) > tol) continue;
      let best = Infinity, bi = -1, bl = 0;
      for (const [i, li] of rim) { const d = Math.hypot(pos[3 * i] - v[0], pos[3 * i + 1] - v[1]); if (d < best) { best = d; bi = i; bl = li; } }
      if (bi < 0) continue;
      const [tx, ty] = nearest(polys[bl], v[0], v[1]);
      const e = acc.get(bi) || [0, 0, 0]; e[0] += tx - v[0]; e[1] += ty - v[1]; e[2] += 1; acc.set(bi, e);
    }
    for (const [i, e] of acc) { pos[3 * i] += e[0] / e[2]; pos[3 * i + 1] += e[1] / e[2]; }
  }
  return p;
}

/**
 * Rims not drawn flat — an eyelet at its own height and angle.
 *
 * A puff hole is a seam, not a bore: the front and back sheets meet on the
 * drawn curve. A hole drawn tilted or raised out of the outline's plane is
 * therefore a tilted or raised eyelet, and the cage carries it by giving every
 * rim vertex the height of its own drawn loop at that point and extending those
 * heights over the sheets. `loops` are the 2-D loops the cage was built on and
 * `lifts[k]` the height of loop k's drawn vertices above the plane, one per
 * loop vertex (null or all-zero for a flat loop). A rim vertex takes the height
 * interpolated along its loop at its projection — the rim was solved past the
 * loop in the plane (puffRimLimitSolve), so the nearest point is the right one.
 *
 * The interior is harmonic: each non-rim vertex's lift is the mean of its
 * neighbors', the rims held (Gauss-Seidel over the cage graph, 32–2,000
 * vertices, converged to 1e-7 of the lift range). Mean value coordinates, which
 * the disc puff uses, are a one-polygon interpolant with no maximum principle
 * on a concave rim; a domain with holes has several rims, and the harmonic
 * extension is defined on it and never leaves the range of the rim heights. The
 * front and back sheets are mirror graphs sharing the rims, so one solve over
 * the whole cage gives both the same lift and the mirror through each rim
 * stays exact. The puff's own height rides on top: front z = lift + h, back
 * z = lift - h, rim z = lift.
 *
 * The 1/0.924 gain is the disc puff's (liftCageToOutline): the Catmull-Clark
 * limit rim undershoots the cage rim's height by that factor, and the limit is
 * linear in the cage z. Measured on an r20 hole in an r100 disc tilted 15–45°
 * at densities 2–5: the limit rim's height over the drawn height averages 1.00
 * ± 0.01, worst miss 1.0 of a 23.1 range. `p.lift` is returned per vertex so a
 * reader can check the field itself.
 */
export function puffLiftRims(p, loops, lifts, opts = {}) {
  const nv = p.nv | 0, pos = p.positions, q = p.quads;
  if (!nv || !q || !q.length || !Array.isArray(lifts)) return p;
  const gain = opts.gain ?? 1 / 0.924;
  let any = false;
  for (const L of lifts) if (L) for (const h of L) if (Math.abs(h) > 1e-12) { any = true; break; }
  if (!any) return p;
  const edges = Array.from({ length: nv }, () => new Set());
  for (let f = 0; f + 3 < q.length; f += 4) {
    const a = q[f], b = q[f + 1], c = q[f + 2], d = q[f + 3];
    edges[a].add(b); edges[a].add(d); edges[b].add(a); edges[b].add(c); edges[c].add(b); edges[c].add(d); edges[d].add(c); edges[d].add(a);
  }
  let span = 0;
  for (let i = 0; i < nv; i++) span = Math.max(span, Math.abs(pos[3 * i]), Math.abs(pos[3 * i + 1]));
  const tol = Math.max(1e-9, span * 1e-9);
  // A rim vertex's height: its loop, then the height along that loop at its projection.
  const heightOn = (k, x, y) => {
    const L = loops[k], H = lifts[k], m = L.length;
    let bd = Infinity, bh = 0;
    for (let i = 0; i < m; i++) {
      const j = (i + 1) % m, ax = L[i][0], ay = L[i][1], dx = L[j][0] - ax, dy = L[j][1] - ay, L2 = dx * dx + dy * dy;
      let t = L2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / L2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t;
      const d = Math.hypot(ax + dx * t - x, ay + dy * t - y);
      if (d < bd) { bd = d; bh = H ? H[i] * (1 - t) + H[j] * t : 0; }
    }
    return [bh, bd];
  };
  const lift = new Float64Array(nv), fixed = new Uint8Array(nv);
  let lo = Infinity, hi = -Infinity, rims = 0;
  for (let i = 0; i < nv; i++) {
    if (Math.abs(pos[3 * i + 2]) > tol || edges[i].size < 3) continue;
    let best = Infinity, bh = 0;
    for (let k = 0; k < loops.length; k++) { const [h, d] = heightOn(k, pos[3 * i], pos[3 * i + 1]); if (d < best) { best = d; bh = h; } }
    lift[i] = bh; fixed[i] = 1; rims++;
    if (bh < lo) lo = bh; if (bh > hi) hi = bh;
  }
  if (!rims) return p;
  const range = Math.max(hi - lo, Math.abs(hi), Math.abs(lo), 1e-12);
  const mean = (lo + hi) / 2;
  for (let i = 0; i < nv; i++) if (!fixed[i]) lift[i] = mean;
  for (let it = 0; it < 4000; it++) {
    let worst = 0;
    for (let i = 0; i < nv; i++) {
      if (fixed[i]) continue;
      let sum = 0;
      for (const j of edges[i]) sum += lift[j];
      const next = sum / edges[i].size, d = Math.abs(next - lift[i]);
      if (d > worst) worst = d;
      lift[i] = next;
    }
    if (worst < range * 1e-7) break;
  }
  let peak = 0;
  for (let i = 0; i < nv; i++) { pos[3 * i + 2] += gain * lift[i]; if (Math.abs(lift[i]) > peak) peak = Math.abs(lift[i]); }
  return { ...p, lift, liftPeak: peak, lifted: true };
}

/** How far the limit rim of a built cage misses its drawn loops: the worst and mean
 *  distance of the z=0 samples of the cage subdivided `levels` times, as a fraction
 *  of the loops' span. The cage rim itself is solved past the loops so that this
 *  lands on them (puffRimLimitSolve). */
export function puffLimitRimMiss(p, loops, levels = 2) {
  const faces = []; for (let f = 0; f + 3 < p.quads.length; f += 4) faces.push([p.quads[f], p.quads[f + 1], p.quads[f + 2], p.quads[f + 3]]);
  let cage = { vertices: Array.from({ length: p.nv }, (_, i) => [p.positions[3 * i], p.positions[3 * i + 1], p.positions[3 * i + 2]]), faces };
  for (let k = 0; k < levels; k++) cage = ccSubdivide(cage);
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const L of loops) for (const q of L) { lo[0] = Math.min(lo[0], q[0]); lo[1] = Math.min(lo[1], q[1]); hi[0] = Math.max(hi[0], q[0]); hi[1] = Math.max(hi[1], q[1]); }
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1]) || 1;
  const polys = loops.map((L) => L.flat());
  let worst = 0, sum = 0, n = 0;
  for (const v of cage.vertices) {
    if (Math.abs(v[2]) > span * 1e-9) continue;
    let d = Infinity; for (const poly of polys) d = Math.min(d, puffNearestOnPoly(poly, v[0], v[1]));
    worst = Math.max(worst, d); sum += d; n++;
  }
  return { worst: worst / span, mean: n ? sum / n / span : 0, samples: n, span };
}

export function buildMultiHolePuff(outerLoop, holeLoops, opts = {}) {
  const n = Array.isArray(holeLoops) ? holeLoops.length : 0;
  const lift = (built) => (opts.lifts ? puffLiftRims(built, [outerLoop, ...holeLoops], opts.lifts) : built);
  let radial = null;
  if (n >= 2) {
    radial = buildRadialMultiHolePuff(outerLoop, holeLoops, opts);
    if (radial && radial.ok) return lift(puffRimLimitSolve(radial, [outerLoop, ...holeLoops]));
    // radial declined (non-star cell / bad cut or junction) — the grid path may still build it.
  }
  const grid = buildGridMultiHolePuff(outerLoop, holeLoops, opts);
  if (grid.ok) return lift(puffRimLimitSolve(grid, [outerLoop, ...holeLoops]));
  if (!radial) return grid;
  // Both declined. The grid's message names a gap; when the radial declined for the shape of a
  // cell rather than for spacing, the gap is not what stopped the puff and the copy must not say so.
  const shape = ['hole-not-star', 'cell-not-star', 'cell-not-enclosing', 'collar-in-hole', 'bowtie-quad', 'vertex-in-hole', 'not-closed', 'junction-invalid', 'cell-empty', 'centroid-on-ring'];
  if (shape.includes(radial.reason)) return { ...grid, reason: radial.reason, why: 'the puff could not mesh a cell around one hole — move that hole a little.' };
  return grid;
}

// Self-check — a mesh this file would not hand out
/**
 * V - E + F = 2 says closed, genus 0 and watertight in a
 * single number, and it is the assertion a shared equator either satisfies or does not. The four
 * cheaper checks are here because Euler alone cannot tell a correct mesh from two different errors
 * that cancel.
 */
export function puffInvariants(p) {
  const faces = [];
  for (let i = 0; i < p.quads.length; i += 4) faces.push([p.quads[i], p.quads[i+1], p.quads[i+2], p.quads[i+3]]);
  for (let i = 0; i < p.tris.length;  i += 3) faces.push([p.tris[i],  p.tris[i+1],  p.tris[i+2]]);

  const edge = new Map();
  // Directed edges, because an undirected count cannot see a reversed half. Reversing the back
  // half's winding leaves every edge still shared by exactly two faces — the undirected count is
  // unchanged, Euler is unchanged, and the mesh is inside-out. In a correctly oriented closed
  // manifold each directed edge a→b occurs exactly once and its opposite exactly once.
  const dir = new Map();
  let repeated = 0;
  for (const f of faces) {
    if (new Set(f).size !== f.length) repeated++;
    for (let i = 0; i < f.length; i++) {
      const a = f[i], b = f[(i + 1) % f.length];
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      edge.set(key, (edge.get(key) || 0) + 1);
      const dk = `${a}>${b}`;
      dir.set(dk, (dir.get(dk) || 0) + 1);
    }
  }
  let boundary = 0, nonManifold = 0;
  for (const c of edge.values()) { if (c === 1) boundary++; else if (c > 2) nonManifold++; }
  let misoriented = 0;
  for (const c of dir.values()) if (c !== 1) misoriented++;
  const used = new Set(); for (const f of faces) for (const v of f) used.add(v);
  let nan = 0; for (let i = 0; i < p.positions.length; i++) if (!Number.isFinite(p.positions[i])) nan++;
  const V = p.nv, E = edge.size, F = faces.length;
  return {
    V, E, F, euler: V - E + F,
    closed: boundary === 0, boundary, nonManifold, repeated, misoriented,
    orphans: V - used.size, nan,
    ok: boundary === 0 && nonManifold === 0 && repeated === 0 && (V - used.size) === 0
        && nan === 0 && (V - E + F) === ((p && typeof p.holes === 'number') ? (2 - 2 * p.holes) : 2) && misoriented === 0,
  };
}

/** Peak absolute height — what a thickness readout reports, and what the smoother must lower. */
export function peakHeight(p) {
  let m = 0;
  for (let i = 2; i < p.positions.length; i += 3) m = Math.max(m, Math.abs(p.positions[i]));
  return m;
}

// The cage a puff hands to this kernel
/* The cage is a deformed quad ball, not a set of rings, for face quality at a
   count a reader can edit.

   The ring construction above does not coarsen well. Its rings
   run inward to a small cap, so coarsening it does not make bigger quads — it
   makes slivers. Measured on a drawn circle, worst edge ratio within a face:

       rim/bands   faces   worst aspect
        12 / 1       34      29.2 : 1
        10 / 2       48      25.0 : 1
        16 / 3      110       5.0 : 1
        20 / 3      170       3.9 : 1

   Nothing at or under four dozen faces is better than 25:1, and at 34 faces the
   cage has only two distinct ring radii — the rim and a tiny cap — so every face
   spans the whole gap between them. A cage like that is hard to grab and
   reads as a pinched surface however good its silhouette is.

   A cube-derived quad ball is near-square by construction at any size — 1.3:1
   before it is warped at every n from 2 to 6 — and warped onto a drawn circle it
   comes back at 3.1:1 for 24 faces and 2.3:1 for 54. It carries eight valence-3
   extraordinary vertices at the cube corners and nothing else extraordinary,
   none of them on the silhouette.

   The trade is paid in silhouette. A ball warped radially onto the
   outline follows a round shape closely (1%) and a lobed one loosely (13-19%).
   At two dozen faces the ring cage measures 30-49% on the same outlines, with
   slivers. The count is the reader's constraint; the silhouette is what gives.

   The radial warp assumes the outline is star-shaped about its own
   centroid. A shape that folds back has directions with more than one crossing,
   and this takes the outermost, so a C is inflated across its own mouth.

   That is approximated, not refused, and the result says so: a comma, a bean
   and a boomerang are ordinary things to draw. The
   approximation is bounded — the outermost crossing is still the outline's own
   reach, so the cage never leaves the drawn extent — and every result carries
   `starShaped`, `reentrant` (the fraction of sampled directions crossing the
   outline more than once) and `worstDeviation` (how far the limit silhouette
   lands from the drawn line), which says how much of
   the shape was lost. A caller with somewhere to put a warning has one; a caller
   with nowhere to put it still gets geometry.
   A deep concavity cannot be represented at this face count, so the face
   budget, not the star-shape assumption, is what costs the concavity. */
const PUFF_BALL_N = [2, 3, 4, 5, 6];
const PUFF_RADIUS_SAMPLES = 128;

/* The parameter surface, declared once and enforced from the declaration.
   `puffCage` clamps every value against this object rather than against literals
   of its own, so a bound cannot be written here and disagree with the code that
   runs. A UI builds one row per entry and needs to know nothing else about puff.

   Each of these moves the geometry and none of them moves another's
   effect, which is what makes them draggable:
     · `density` changes the cage and not the peak height. The peak is imposed —
       `puffiness` times the outline's own mean reach — rather than inherited from
       the ball's z, so the two are independent by construction. Inherited, a
       density slider would also be a shape slider: the
       peak moves 9.5% between density 2 and 3 purely because an odd-n ball has no
       vertex at its own pole.
     · `puffiness`, `follow` and `bottomScale` change the height field and not
       the silhouette. The cage's (x, y) never reads any of them, and the growth
       solve that centers the silhouette runs on a fixed reference height, so the
       plan view is bit-for-bit the same across the whole of their travel.

   Every lower bound here is a face-quality bound, and each one is measured.
   A cage is only worth editing if a hand can grab a face, so
   the number that governs is the worst edge ratio within a face. Over a family of
   eight outlines at densities 2 through 6 it is 7.4:1 at these defaults, and at
   each parameter's own extreme:

       puffiness 0.15 (min)  13.7:1      puffiness 1.5 (max)   5.7:1
       bottomScale 0.35      13.1:1      follow 1 (max)       16.3:1

   The pattern is one fact: less height over the same plan makes longer, thinner
   faces. `puffiness` 0 and `bottomScale` 0 are excluded because they are not
   merely thin — they fold the cage onto a plane. The quad ball carries a vertex
   at (x, y, +z) for every (x, y, -z); flattened, those become the same point, so
   a 26-vertex cage comes back with 9 of its vertices sitting on another one and
   an enclosed volume of exactly zero: two sheets lying on each other, which no
   weld, export or NURBS conversion can make a solid of.
   Face aspect does not detect it: the flattened cage's worst face ratio is 4.2:1
   on a circle against 3.1:1 for the same cage inflated — a clean number for an
   object with no inside — so the bound is what keeps a slider from producing a
   degenerate solid.
   `follow` keeps its full 0..1 because 1 is the meaning of the control — the
   narrowest part of the drawing comes out as thin as it is narrow, and long thin
   faces there are what was asked for, not a defect. */
export const PUFF_PARAMS = Object.freeze({
  density: Object.freeze({
    order: 0, label: 'Density', min: 2, max: 6, step: 1, default: 2, integer: true,
    help: 'How many quads the cage carries: 6 x density x density, so 24 at 2 and 216 at 6. '
      + 'A cage is a thing a hand grabs a vertex of, and Subdivide adds detail afterwards '
      + 'while nothing takes it away again — so it starts coarse.',
  }),
  puffiness: Object.freeze({
    order: 1, label: 'Puffiness', min: 0.15, max: 1.5, step: 0.01, default: 0.45,
    help: 'Peak height as a fraction of how far the outline reaches. At 1 the form is about as '
      + 'tall as it is wide.',
  }),
  follow: Object.freeze({
    order: 2, label: 'Width follow', min: 0, max: 1, step: 0.01, default: 0.5,
    help: 'How strongly thickness tracks the LOCAL WIDTH of the outline. 0 is one even dome over '
      + 'the whole shape; 1 makes a narrow tail as thin as it is narrow while the body stays full. '
      + 'On a disc with a thin tail the surface over the tail is as tall as the body at 0 and about '
      + 'three fifths of it at 1.',
  }),
  flatBack: Object.freeze({
    order: 4, label: 'Flat back', min: 0, max: 1, step: 1, default: 0, integer: true, boolean: true,
    help: 'Press the underside onto the drawing plane, so the form sits on a surface — a domed top '
      + 'over a flat base, which is how a mouse or a handle is shaped. The silhouette is creased '
      + 'when this is on, because a flat bottom that rolls under at its edge does not sit flat.',
  }),
  /* A modifier, declared as one. Without a flat side there is nothing to flip:
     at the defaults this moves the cage by exactly 0%. `requires`
     names the parameter that has to be on for it to mean anything — declared
     once here so a panel can withdraw the row and a parameter sweep can
     satisfy the precondition before measuring, instead of each deciding for
     itself. */
  flipFlat: Object.freeze({
    order: 5, label: 'Flip flat side', min: 0, max: 1, step: 1, default: 0, integer: true, boolean: true,
    requires: 'flatBack',
    help: 'Which side of the drawing plane is the flat one. A drawn outline carries no up, so which '
      + 'way the form domes follows from how the curve happened to be traced — this flips it without '
      + 'redrawing. Offered only while Flat back is on, since until then there is no flat side.',
  }),
  bottomScale: Object.freeze({
    order: 3, label: 'Underside', min: 0.35, max: 1, step: 0.01, default: 1,
    help: 'How deep the back is against the front. 1 is a symmetric pillow; low values press the '
      + 'back toward flat, for a form that sits on a surface.',
  }),
});

/** Every declared parameter, filled from `opts`, clamped and rounded to its own declaration. */
export function puffResolveParams(opts = {}) {
  const out = {};
  for (const key of Object.keys(PUFF_PARAMS)) {
    const spec = PUFF_PARAMS[key];
    let v = Number(opts[key]);
    if (!Number.isFinite(v)) v = spec.default;
    v = Math.min(spec.max, Math.max(spec.min, v));
    out[key] = spec.integer ? Math.round(v) : v;
  }
  return out;
}

/* `faces` is an alternative spelling of the same control, accepted because a
   caller thinking in faces is thinking about the constraint that governs. It is
   the wanted count; the answer is the coarsest rung that reaches it. */
export function puffDensityForFaces(want) {
  let d = PUFF_BALL_N[0];
  for (const cand of PUFF_BALL_N) { d = cand; if (6 * cand * cand >= want) break; }
  return d;
}

function puffCentroid(poly) {
  let cx = 0, cy = 0;
  const n = poly.length / 2;
  for (let i = 0; i < n; i += 1) { cx += poly[i * 2]; cy += poly[i * 2 + 1]; }
  return [cx / n, cy / n];
}

/* How far the outline reaches in one direction, and how many times it crosses.
   The reach is the outermost crossing, so a shape that folds back is bounded
   rather than pinched to its first fold; the crossing count is what says the
   fold happened, and it is the only thing that can.

   The side test is half-open and there is no epsilon, which is the same
   idiom point-in-polygon uses and for the same reason. A ray that passes exactly
   through a vertex touches two segments; parameterized along the segment and
   accepted on a tolerance, rounding decides whether that is one crossing or two,
   and the decision changes with the drawing's scale — a plain circle reads as
   re-entrant at size 1 and star-shaped at size 0.001 and 1000. Classifying each
   endpoint by which side of the ray line it lies on, with `> 0` and its negation
   partitioning the shared vertex, makes the count exact for every scale and
   removes the tolerance entirely: a crossing puts the two endpoints on opposite
   sides, so the denominator cannot be zero. */
function puffRadiusAt(poly, c, t) {
  const dx = Math.cos(t), dy = Math.sin(t), m = poly.length / 2;
  let best = 0, crossings = 0;
  let ax = poly[(m - 1) * 2] - c[0], ay = poly[(m - 1) * 2 + 1] - c[1];
  let ca = ax * dy - ay * dx;
  for (let i = 0; i < m; i += 1) {
    const bx = poly[i * 2] - c[0], by = poly[i * 2 + 1] - c[1];
    const cb = bx * dy - by * dx;
    if ((ca > 0) !== (cb > 0)) {
      const s = ca / (ca - cb);
      const u = (ax + (bx - ax) * s) * dx + (ay + (by - ay) * s) * dy;
      if (u > 0) { crossings += 1; if (u > best) best = u; }
    }
    ax = bx; ay = by; ca = cb;
  }
  return { r: best, crossings };
}

/* A cube, subdivided n x n per face and pushed onto a sphere. Eight valence-3
   vertices at the cube corners and nothing else extraordinary, and every face
   near-square whatever n is. */
function puffQuadBall(n) {
  const idx = new Map(), vertices = [], faces = [];
  const put = (x, y, z) => {
    const k = `${x.toFixed(6)},${y.toFixed(6)},${z.toFixed(6)}`;
    if (idx.has(k)) return idx.get(k);
    const l = Math.hypot(x, y, z) || 1;
    const i = vertices.push([x / l, y / l, z / l]) - 1;
    idx.set(k, i);
    return i;
  };
  const g = (i) => (i / n) * 2 - 1;
  const side = (fn, flip) => {
    for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) {
      const q = [fn(i, j), fn(i + 1, j), fn(i + 1, j + 1), fn(i, j + 1)];
      faces.push(flip ? q.slice().reverse() : q);
    }
  };
  side((i, j) => put(+1, g(i), g(j)), false); side((i, j) => put(-1, g(i), g(j)), true);
  side((i, j) => put(g(i), +1, g(j)), true);  side((i, j) => put(g(i), -1, g(j)), false);
  side((i, j) => put(g(i), g(j), +1), false); side((i, j) => put(g(i), g(j), -1), true);
  return { vertices, faces };
}

/* Local width is not radial reach, and using the reach is backwards on the
   shapes the control exists for.

   The reach from the centroid at angle theta says how far the outline goes that
   way. Thickness wants to know how wide it is that way, and the two disagree in
   exactly the case a person notices: a fat body with a long thin tail reaches
   furthest along the tail and is narrowest there, so a height scaled by reach
   comes out fattest on the thinnest part of the drawing. Measured on a disc with
   a spike, height over the tail against height over the body: 1.32 with reach,
   0.44 with width.

   Width here is the largest inscribed circle still reachable going outward:
   for a point at fraction s along the ray at angle theta, the maximum of
   `distanceToBoundary` over the rest of that ray. Written as a suffix maximum it
   costs one pass and it is monotone non-increasing in s by construction, so
   thickness can only fall off toward the rim and never rise.

   The plain maximum over the whole ray is not enough. Every ray starts at the
   centroid, which on a disc-with-a-spike sits in the body — so the maximum over the whole ray is the body's width in
   every direction including the tail's, and the tail reads as fat again. Taking
   the maximum only from the point outward is what makes a point out in the tail
   see the tail.

   There is no direction at the center. Row 0 is the shape's own largest
   inscribed radius for every theta, so the apex height is a property of the
   shape rather than of whichever way `atan2(0, 0)` happens to point.

   The lookup stops half way out, or the rim band collapses.
   The distance to the boundary is zero at the boundary, by definition — so read
   at its own fraction, a vertex near the rim scores a width near zero, and at
   full following the band either side of the equator loses its height entirely:
   unclamped, on a bean at density 5, two of its four edges fall to 0.001 against
   0.43 and the worst face ratio reaches 465:1. The shape's own dome already
   carries the fall-off toward the rim; the width table says how thick
   this direction should be, and past the half-way point it stops answering that
   and starts reporting the boundary approaching. Clamped, the same cage measures
   5.7:1. */
const PUFF_WIDTH_REACH = 0.5;
const PUFF_WIDTH_STEPS = 24;
function puffWidthTable(poly, centre, reach) {
  const rows = PUFF_WIDTH_STEPS + 1;
  const W = new Float64Array(PUFF_RADIUS_SAMPLES * rows);
  for (let k = 0; k < PUFF_RADIUS_SAMPLES; k += 1) {
    const t = (k / PUFF_RADIUS_SAMPLES) * Math.PI * 2;
    const dx = Math.cos(t) * reach[k], dy = Math.sin(t) * reach[k];
    const base = k * rows;
    for (let i = 0; i < rows; i += 1) {
      const s = i / PUFF_WIDTH_STEPS;
      const x = centre[0] + dx * s, y = centre[1] + dy * s;
      W[base + i] = puffInsidePoly(poly, x, y) ? puffNearestOnPoly(poly, x, y) : 0;
    }
    for (let i = rows - 2; i >= 0; i -= 1) W[base + i] = Math.max(W[base + i], W[base + i + 1]);
  }
  let wmax = 0;
  for (let k = 0; k < PUFF_RADIUS_SAMPLES; k += 1) wmax = Math.max(wmax, W[k * rows]);
  for (let k = 0; k < PUFF_RADIUS_SAMPLES; k += 1) W[k * rows] = wmax;
  return { W, wmax, rows };
}

function puffWidthAt(wt, t, rho) {
  const f = ((t / (Math.PI * 2)) % 1 + 1) % 1 * PUFF_RADIUS_SAMPLES;
  const a = Math.floor(f), b = (a + 1) % PUFF_RADIUS_SAMPLES, u = f - a;
  const g = Math.min(Math.max(rho, 0), 1) * PUFF_WIDTH_STEPS;
  const c = Math.min(PUFF_WIDTH_STEPS - 1, Math.floor(g)), v = g - c;
  const A = wt.W[a * wt.rows + c] * (1 - v) + wt.W[a * wt.rows + c + 1] * v;
  const B = wt.W[b * wt.rows + c] * (1 - v) + wt.W[b * wt.rows + c + 1] * v;
  return A * (1 - u) + B * u;
}

/* The cage's directions are spaced by arc length along the drawn line, not
   by angle about the centroid; on an anisotropic shape that is the
   difference between a cage and a fan of slivers.

   A ball warped by angle spends its vertices evenly in angle, so a shape whose
   reach varies — an ellipse, anything with a tail — gets the same number of
   vertices covering a long stretch of curve as a short one. The long stretch
   ends up spanned by one enormous quad and the short one by a pinched one.
   Measured as worst edge ratio within a face, at density 2 through 6, follow
   0.5:

       shape                 by angle                by arc length
       ellipse 3:1     6.5  3.7  5.9  5.4  5.8    3.9  3.5  4.1  3.6  4.1
       disc + thin tail 11.1  3.4 13.2  5.0 15.4    5.1  5.1  5.6  5.4  5.6

   Reparameterizing costs nothing at run time — the outline is already resampled
   by arc length, so the map is that polygon's own vertex angles read as a table —
   and it leaves the topology, the height law and the growth solve untouched.

   It is anchored at angle zero, not at the stroke's first point. Anchored at
   the first point, the cage would rotate with wherever the hand happened to
   start, so the same drawn shape would come back with its vertices in different
   places.

   It declines on a re-entrant outline, returning null so the warp
   falls back to plain angle. A shape whose angle about the centroid is not
   monotone has no arc-length-to-angle map at all, and forcing one would fold the
   cage — a much worse failure than the coarse approximation the fallback gives.
   The `> PI` step guard is what catches it: a monotone traverse of a simple loop
   never turns half a revolution between two resampled neighbors. */
function puffDirMap(poly, centre) {
  const m = poly.length / 2;
  const th = new Float64Array(m + 1);
  let prev = Math.atan2(poly[1] - centre[1], poly[0] - centre[0]);
  th[0] = prev;
  for (let i = 1; i <= m; i += 1) {
    const j = i % m;
    let a = Math.atan2(poly[j * 2 + 1] - centre[1], poly[j * 2] - centre[0]);
    while (a < prev) a += Math.PI * 2;
    if (a - prev > Math.PI) return null;
    th[i] = a; prev = a;
  }
  if (Math.abs(th[m] - th[0] - Math.PI * 2) > 1e-6) return null;
  const target = th[0] + ((0 - th[0]) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);
  let j0 = 0;
  for (let i = 0; i < m; i += 1) {
    if (th[i] <= target && th[i + 1] >= target) { j0 = i + (target - th[i]) / (th[i + 1] - th[i]); break; }
  }
  return (t) => {
    const u = ((t / (Math.PI * 2)) % 1 + 1) % 1;
    const f = j0 + u * m, a = Math.floor(f), v = f - a;
    const ia = a % m, ib = (a + 1) % m;
    const A = th[ia];
    let B = th[ib];
    if (B < A) B += Math.PI * 2;
    return A + (B - A) * v;
  };
}

/* Height is set, not inherited. The peak is `puffiness` times the outline's
   mean radial reach, imposed by rescaling the whole height field to it once the
   field is built — so it is the same number at every density, at every `follow`,
   and a density slider cannot change the shape. Read off the ball's own
   z instead, the peak moves 9.5% between density 2 and density 3,
   purely because an odd-n ball has no vertex at its own pole.
   The reach is the base and the width is the modulation, which is the division
   of labor the two controls describe: `puffiness` says how tall the form is
   against its own drawn size, `follow` says how much of that height the narrow
   parts give back. Basing the height on the width instead makes the two controls
   fight — an elongated outline then comes out flatter as a side effect of a
   control the reader did not touch.
   (x, y) never reads a height parameter. That is what makes the silhouette
   independent of `puffiness`, `follow` and `bottomScale` exactly rather than
   nearly: Catmull-Clark is affine and per-coordinate, so a plan view that does
   not depend on z at the cage cannot depend on it at the limit either. */
function puffWarp(centre, n, grow, radial, wt, rbar, P, dirmap) {
  const ball = puffQuadBall(n);
  const H = P.puffiness * rbar;
  const raw = new Float64Array(ball.vertices.length);
  let peak = 0;
  const dirs = new Float64Array(ball.vertices.length);
  for (let i = 0; i < ball.vertices.length; i += 1) {
    const v = ball.vertices[i];
    const rho = Math.hypot(v[0], v[1]);
    const t = Math.atan2(v[1], v[0]);
    dirs[i] = dirmap ? dirmap(t) : t;
    const w = puffWidthAt(wt, dirs[i], Math.min(rho, PUFF_WIDTH_REACH)) / wt.wmax;
    raw[i] = v[2] * (1 - P.follow + P.follow * w);
    const m = Math.abs(raw[i]);
    if (m > peak) peak = m;
  }
  const k = peak > 0 ? H / peak : 0;
  const vertices = ball.vertices.map((v, i) => {
    const rho = Math.hypot(v[0], v[1]);
    const R = radial(dirs[i]) * grow * rho;
    let z = raw[i] * k;
    /* A flat back is a projection, not a small `bottomScale`. Pressing the
       underside toward the plane by a factor never reaches it — the back stays a
       shallow dome — and the form still rocks. Setting the whole lower half to
       the plane makes the base planar, and the ball's lower vertices
       land inside the outline where they tile it. */
    /* Which side is flat is a choice, not a fact about the curve. A drawn
       outline carries no up, so which way the dome faces follows from the
       stroke's winding and the view it was drawn in — neither of which the
       person drawing was thinking about. `flipFlat` mirrors the test rather than
       the geometry: everything downstream (the crease solve, the silhouette fit,
       the frame transform) then works on the flipped form without knowing. */
    const under = P.flipFlat ? z > 0 : z < 0;
    if (P.flatBack && under) z = 0;
    else if (under) z *= P.bottomScale;
    return [centre[0] + Math.cos(dirs[i]) * R, centre[1] + Math.sin(dirs[i]) * R, z];
  });
  const faces = ball.faces.map((f) => f.slice());
  /* A flat bottom needs a crease at its edge, and it is the one place this
     construction wants one. Left smooth, Catmull-Clark rolls the base under at
     the rim: the underside stops being planar exactly where it meets the ground,
     and the form rocks instead of sitting. The silhouette is the boundary
     between the faces that were flattened and the faces that were not — asked of
     the geometry rather than assumed of an index, because whether a ball has a
     ring exactly at its equator depends on whether its density is even. */
  const creases = {};
  if (P.flatBack) {
    const flat = (fi) => faces[fi].every((v) => Math.abs(vertices[v][2]) < 1e-9);
    const edgeOwner = new Map();
    faces.forEach((f, fi) => {
      for (let k = 0; k < 4; k += 1) {
        const a2 = f[k], b2 = f[(k + 1) % 4];
        const key = a2 < b2 ? `${a2}_${b2}` : `${b2}_${a2}`;
        const prev = edgeOwner.get(key);
        if (prev === undefined) edgeOwner.set(key, fi);
        else if (flat(prev) !== flat(fi)) creases[key] = 3;
      }
    });
  }
  return { vertices, faces, creases };
}

function puffNearestOnPoly(poly, x, y) {
  const m = poly.length / 2;
  let bd = Infinity;
  for (let q = 0; q < m; q += 1) {
    const r = (q + 1) % m;
    const ax = poly[q * 2], ay = poly[q * 2 + 1], cx = poly[r * 2], cy = poly[r * 2 + 1];
    const dx = cx - ax, dy = cy - ay, L2 = dx * dx + dy * dy;
    let t = L2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / L2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const d = Math.hypot(ax + dx * t - x, ay + dy * t - y);
    if (d < bd) bd = d;
  }
  return bd;
}
function puffInsidePoly(poly, x, y) {
  const m = poly.length / 2;
  let inside = false;
  for (let i = 0, j = m - 1; i < m; j = i++) {
    const xi = poly[i * 2], yi = poly[i * 2 + 1], xj = poly[j * 2], yj = poly[j * 2 + 1];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/* The silhouette solve is one subdivision, not one per bisection step, and
   that is an identity rather than an approximation.

   Every cage vertex is `centre + (unit ball point) * reach(theta) * grow`, so
   `grow` is a uniform scaling of the whole cage about `centre`. Catmull-Clark is
   an affine combination of vertices, so the subdivided surface scales with it
   the same way. The limit rim at any `grow` is therefore the limit rim at
   `grow = 1` scaled — one subdivision serves every step of the search, and the
   search itself becomes a scan over a fixed list of points.
   The bisection's 24 steps share that one subdivision. With the
   outline's own tables kept between frames (see `opts.cache`), a density drag
   costs 0.24 ms a frame and a puffiness drag 0.04 ms, against a first build of
   14-21 ms.

   The rim is selected by height and the tolerance is relative. An absolute one
   selects nothing on a millimeter outline and everything on a flat one. */
const PUFF_RIM_TOL = 1e-6;
function puffRimRing(cage, subdivide, levels, height) {
  let c = cage;
  for (let k = 0; k < levels; k += 1) c = subdivide(c);
  const tol = height * PUFF_RIM_TOL;
  const ring = [];
  for (const v of c.vertices) if (Math.abs(v[2]) <= tol) ring.push([v[0], v[1]]);
  return ring;
}

function puffRimDev(ring, centre, poly, scale, grow) {
  let worst = 0, signed = 0;
  for (const q of ring) {
    const x = centre[0] + (q[0] - centre[0]) * grow;
    const y = centre[1] + (q[1] - centre[1]) * grow;
    const d = puffNearestOnPoly(poly, x, y);
    signed += puffInsidePoly(poly, x, y) ? -d : d;
    if (d > worst) worst = d;
  }
  return { worst: worst / scale, signed: ring.length ? signed / ring.length / scale : 0 };
}

/* Worst edge ratio within a face. Counting faces is not measuring a cage: a ring
   construction meets every face-count target this one does and reaches 29:1,
   which no count-based check can see. */
export function puffFaceAspect(cage) {
  const V = cage.vertices;
  let worst = 0;
  for (const q of cage.faces) {
    const L = [0, 1, 2, 3].map((k) => {
      const a = V[q[k]], b = V[q[(k + 1) % 4]];
      return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    });
    const mn = Math.min(...L);
    if (mn > 1e-12) worst = Math.max(worst, Math.max(...L) / mn);
  }
  return worst;
}


/* The height parameters are held at a fixed reference for the solve, so `grow`
   is a function of the outline and the density and nothing else. Solved against
   the live values instead, the rim selection would move with `puffiness` — not
   because the silhouette changes but because the test that finds the rim reads
   z — and a puffiness drag would wobble the plan view for no geometric reason. */
const PUFF_SOLVE_REF = Object.freeze({ puffiness: 0.45, follow: 0, bottomScale: 1 });
const PUFF_GROW_HI = 3.0;
const PUFF_GROW_STEPS = 28;

/* What a drag frame costs, and why the cache is part of the design.

   Everything expensive here is a property of the outline, not of the
   parameters: cleaning and resampling the stroke, the radial reach table, the
   local-width table, and — per density rung — the subdivided rim ring and the
   growth that centers it. A slider moves none of those. So a caller that keeps
   one plain object between frames and passes it as `opts.cache` pays the whole
   cost once per stroke and a few tenths of a millisecond per frame after that.

   The cache is checked against the outline it was built from, and a mismatch
   rebuilds rather than refusing. A stale cache handed to a new stroke would
   otherwise return the previous shape's geometry silently — a plausible result
   that is not what was drawn. The
   fingerprint is exact (length, both ends, and the sum of every coordinate), so
   the only way past it is a different stroke with identical
   arithmetic, and the recompute path is the safe direction anyway. */
function puffFingerprint(outline) {
  let sum = 0;
  for (let i = 0; i < outline.length; i += 1) sum += outline[i];
  const n = outline.length;
  return `${n}:${outline[0]}:${outline[1]}:${outline[n - 2]}:${outline[n - 1]}:${sum}`;
}

/* Everything derived from the stroke alone. A refusal is cached too — asking the
   same unanswerable question sixty times a second should cost the same as asking
   it once. */
function puffPrepare(outline) {
  /* A repeated point is dropped here and nowhere else. A pointer that reports
     the same position twice contributes a zero-length segment, and a zero-length
     segment lies on top of its neighbor — which the self-intersection test reads
     as the outline crossing itself, refusing an ordinary stroke from a device
     that repeats samples. The duplicate carries no information about the shape,
     so removing it is not a repair of the drawing. */
  const pairs = [];
  for (let i = 0; i + 1 < outline.length; i += 2) {
    const last = pairs[pairs.length - 1];
    if (last && last[0] === outline[i] && last[1] === outline[i + 1]) continue;
    pairs.push([outline[i], outline[i + 1]]);
  }
  if (pairs.length > 1) {
    const a = pairs[0], b = pairs[pairs.length - 1];
    if (a[0] === b[0] && a[1] === b[1]) pairs.pop();
  }
  if (pairs.length < 3) {
    return { ok: false, reason: 'empty', why:
      'Every point of that stroke is the same point — draw a closed loop that goes somewhere.' };
  }
  /* The stroke is cleaned before anything is measured from it. `prepareOutline`
     closes it, winds it consistently, refuses one that crosses itself — which is
     not a shape and would silently produce a folded cage — and resamples it
     evenly so the radial reach is not dominated by wherever the hand slowed
     down. A self-crossing loop caught here is named; caught later it is a mesh
     nobody can explain. */
  const prep = prepareOutline(pairs, { count: 96 });
  /* The diagnosis comes from `prepareOutline` and the action is added here.
     Its messages are exact about what is wrong and some of them stop there — "the
     outline encloses no area" is true and leaves a reader with nothing to do, so
     each refusal here also names a change. */
  if (!prep.ok) {
    return { ok: false, reason: 'outline', at: prep.at, why: /draw/i.test(prep.why)
      ? prep.why : `${prep.why} — draw one closed loop with room inside it` };
  }
  const poly = [];
  for (const q of prep.pts) poly.push(q[0], q[1]);
  const centre = puffCentroid(poly);
  let scale = 0;
  for (let i = 0; i < poly.length; i += 2) {
    scale = Math.max(scale, Math.hypot(poly[i] - centre[0], poly[i + 1] - centre[1]));
  }
  if (!(scale > 0)) return { ok: false, reason: 'empty', why: 'That stroke has no extent — draw a closed loop.' };

  /* An outline with no measurable inside is refused by the same test the rest
     of this module uses: the deepest interior point against what the sampling can
     resolve. A stroke that goes out and back is a line however much area rounding
     leaves it with, and a zero-height cage for one would be a plausible result
     that is not what was drawn. */
  if (!(prep.inradius > prep.inradiusSpacing)) {
    return { ok: false, reason: 'thin', why:
      `This outline has no measurable inside — its deepest point is ${prep.inradius.toFixed(6)} from `
      + `the edge, within the ${prep.inradiusSpacing.toFixed(6)} the measurement can resolve. Draw it wider.` };
  }

  /* The radial reach and the re-entrancy count, sampled once. Asking the polygon
     per cage vertex is the same answer at many times the cost. */
  const reach = new Float64Array(PUFF_RADIUS_SAMPLES);
  let reentrant = 0, rbar = 0;
  for (let k = 0; k < PUFF_RADIUS_SAMPLES; k += 1) {
    const hit = puffRadiusAt(poly, centre, (k / PUFF_RADIUS_SAMPLES) * Math.PI * 2);
    reach[k] = hit.r;
    rbar += hit.r;
    if (hit.crossings > 1) reentrant += 1;
  }
  rbar /= PUFF_RADIUS_SAMPLES;
  if (!(rbar > 0)) return { ok: false, reason: 'thin', why: 'That outline encloses nothing to puff — draw it wider.' };
  reentrant /= PUFF_RADIUS_SAMPLES;
  const radial = (t) => {
    const f = ((t / (Math.PI * 2)) % 1 + 1) % 1 * PUFF_RADIUS_SAMPLES;
    const a = Math.floor(f), b = (a + 1) % PUFF_RADIUS_SAMPLES, u = f - a;
    return reach[a] * (1 - u) + reach[b] * u;
  };

  const wt = puffWidthTable(poly, centre, reach);
  if (!(wt.wmax > 0)) return { ok: false, reason: 'thin', why: 'That outline encloses nothing to puff — draw it wider.' };
  return { ok: true, poly, centre, scale, reach, radial, rbar, wt, reentrant,
    dirmap: puffDirMap(poly, centre), rungs: new Map() };
}

/* The silhouette solve, once per density rung. `grow` reads no height parameter,
   so a rung solved during a puffiness drag stays solved. */
function puffSolveRung(pre, n, subdivide) {
  const hit = pre.rungs.get(n);
  if (hit) return hit;
  /* Subdivide only as far as the rim needs sampling. The rim carries 8n control
     points, so a level doubles the sample count; the target is ~64 samples around
     the silhouette, which is the resolution the worst-case deviation is quoted at.
     A fixed three levels costs 9.5 ms at density 5 and buys nothing — a
     finer cage is already closer to its own limit surface. */
  const levels = n <= 2 ? 3 : 2;
  const refH = PUFF_SOLVE_REF.puffiness * pre.rbar;
  const ring = puffRimRing(
    puffWarp(pre.centre, n, 1, pre.radial, pre.wt, pre.rbar, PUFF_SOLVE_REF, pre.dirmap),
    subdivide, levels, refH);

  /* The growth that centers the limit silhouette on the drawn line. Bisected
     because the deviation is monotone in it and there is no closed form. */
  let lo = 1, hi = PUFF_GROW_HI;
  const outer = puffRimDev(ring, pre.centre, pre.poly, pre.scale, hi);
  for (let k = 0; k < PUFF_GROW_STEPS; k += 1) {
    const mid = (lo + hi) / 2;
    if (puffRimDev(ring, pre.centre, pre.poly, pre.scale, mid).signed < 0) lo = mid; else hi = mid;
  }
  const grow = (lo + hi) / 2;
  const dev = puffRimDev(ring, pre.centre, pre.poly, pre.scale, grow);
  /* The solve ran out of room rather than converging — reported, because a
     silhouette that never reached the drawn line is a different result from one
     that did, and only this flag tells them apart. */
  const out = { grow, dev, rim: ring.length, growClamped: !(outer.signed > 0) };
  pre.rungs.set(n, out);
  return out;
}

/**
 * A closed drawn outline, as a cage coarse enough to edit by hand.
 *
 * `outline` is a flat [x0,y0,...] closed loop in the drawing plane.
 * `opts.subdivide` is this kernel's own Catmull-Clark, passed rather than
 * imported so the cage is measured against the subdivider it will be drawn with.
 * `opts.cache` is a plain object the caller keeps between frames of a drag; it
 * is optional, and passing none costs the full rebuild every call.
 * Every other option is declared in `PUFF_PARAMS` and clamped to it; `faces` is
 * accepted as an alternative spelling of `density`.
 *
 * Refusals name what a reader would change and none of them throws.
 */
/**
 * Lift a flat cage onto an outline that was not drawn flat.
 *
 * The solve is two-dimensional and stays that way — every silhouette promise
 * this module makes is a statement about the shadow, and the shadow is what the
 * solve owns. This adds the third dimension afterward: an outline drawn with
 * height carries an `N·d` component, and without the lift a saddle drawn 25mm
 * out of plane reports its silhouette "within 1.0%" while sitting 18.1mm off
 * the surface built from it.
 *
 * The height is extended from the rim over the interior by mean value
 * coordinates: positive-weighted on a convex outline, defined on any simple
 * polygon, and exact on the rim by construction. It is linear in the rim
 * heights, which is what makes the gain below a scalar rather than a solve.
 *
 * MVC has no maximum principle on a concave outline. Its weights can go
 * negative there, so an interior point can rise above the highest rim point —
 * on a crescent with one horn lifted, a bulge appears in the middle where
 * nothing was dragged. Measured: nil on a fish, 0.6% for a single point
 * lifted 20mm, 8.1% on a crescent. `overshoot` is returned so a caller can
 * report it.
 *
 * The limit surface is not the cage. Catmull-Clark smooths a coarsely
 * sampled height field, so the rim of the limit surface undershoots the drawn
 * height by a fixed factor — measured at 0.924, flat across amplitude. The gain
 * divides it out, which is exact because the limit rim is linear in the cage z.
 */
export function liftCageToOutline(cage, ring, lift, opts = {}) {
  const gain = opts.gain ?? 1 / 0.924;
  const n = ring.length / 2;
  if (!cage || !cage.vertices || !n || lift.length !== n) {
    return { ok: false, reason: 'shape', why: 'the outline and its height list are different lengths' };
  }
  let flat = true;
  for (let i = 0; i < n; i += 1) if (Math.abs(lift[i]) > 1e-12) { flat = false; break; }
  /* An unlifted outline leaves the cage alone, by identity. A flat outline is
     the common case, and a lift that perturbed it — even at 1e-16 — would
     change a flat puff's silhouette. */
  if (flat) return { ok: true, lifted: 0, overshoot: 0, peak: 0, flat: true };

  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i += 1) { if (lift[i] < lo) lo = lift[i]; if (lift[i] > hi) hi = lift[i]; }
  const span = hi - lo;
  let worst = 0, moved = 0;
  for (const v of cage.vertices) {
    const h = mvcInterpolate(ring, lift, v[0], v[1]);
    if (!Number.isFinite(h)) continue;
    if (h > hi) worst = Math.max(worst, h - hi);
    if (h < lo) worst = Math.max(worst, lo - h);
    v[2] += gain * h;
    moved += 1;
  }
  return { ok: true, lifted: moved, overshoot: span > 0 ? worst / span : 0, peak: gain * hi, flat: false };
}

/* Mean value coordinates at (x, y) over a closed polygon, returning the value
   interpolated from `values`. On a vertex it is that vertex's value; on an edge
   it is the linear blend along it — both handled explicitly, because the
   general formula divides by a distance that is zero there. */
export function mvcInterpolate(ring, values, x, y) {
  const n = ring.length / 2;
  const EPS = 1e-9;
  const dx = new Array(n), dy = new Array(n), r = new Array(n);
  for (let i = 0; i < n; i += 1) {
    dx[i] = ring[i * 2] - x; dy[i] = ring[i * 2 + 1] - y;
    r[i] = Math.hypot(dx[i], dy[i]);
    if (r[i] < EPS) return values[i];
  }
  let wsum = 0, vsum = 0;
  for (let i = 0; i < n; i += 1) {
    const j = (i + 1) % n;
    const cross = dx[i] * dy[j] - dy[i] * dx[j];
    const dot = dx[i] * dx[j] + dy[i] * dy[j];
    if (Math.abs(cross) < EPS && dot < 0) {
      // On the edge between i and j: the general weight is singular, and the
      // answer is the straight blend along it.
      const t = r[i] / (r[i] + r[j]);
      return values[i] * (1 - t) + values[j] * t;
    }
    // tan(a_i / 2) accumulated onto both endpoints of the edge.
    const c = cross / (r[i] * r[j] + dot);
    const wi = c / r[i], wj = c / r[j];
    wsum += wi + wj; vsum += wi * values[i] + wj * values[j];
  }
  return wsum === 0 ? 0 : vsum / wsum;
}

export function puffCage(outline, opts = {}) {
  const { subdivide, cache } = opts;
  if (typeof subdivide !== 'function') {
    return { ok: false, reason: 'nosubdiv', why:
      'puffCage measures its cage against the subdivider the cage will be drawn with — pass this '
      + 'kernel\'s own Catmull-Clark as opts.subdivide.' };
  }
  if (!outline || outline.length < 6) {
    return { ok: false, reason: 'empty', why: 'Nothing was drawn — draw a closed loop of at least three points.' };
  }
  for (let i = 0; i < outline.length; i += 1) {
    if (!Number.isFinite(outline[i])) {
      return { ok: false, reason: 'empty', why:
        'That outline carries a coordinate that is not a finite number — draw it again.' };
    }
  }
  const want = (opts.density == null && opts.faces != null)
    ? { ...opts, density: puffDensityForFaces(opts.faces) } : opts;
  const P = puffResolveParams(want);
  const n = P.density;

  const key = cache ? puffFingerprint(outline) : null;
  let pre = (cache && cache.key === key) ? cache.pre : null;
  if (!pre) {
    pre = puffPrepare(outline);
    if (cache) { cache.key = key; cache.pre = pre; }
  }
  if (!pre.ok) return { ok: false, reason: pre.reason, why: pre.why, at: pre.at };

  const solved = puffSolveRung(pre, n, subdivide);
  const cage = puffWarp(pre.centre, n, solved.grow, pre.radial, pre.wt, pre.rbar, P, pre.dirmap);
  return {
    ok: true, reason: null, why: '',
    cage, quads: cage.faces.length, n, density: n, faces: cage.faces.length,
    grow: solved.grow, growClamped: solved.growClamped,
    puffiness: P.puffiness, follow: P.follow, bottomScale: P.bottomScale,
    /* The peak is a stated number, not a measured one: this is what the height
       field was scaled to, and a readout can quote it in the drawing's units. */
    height: P.puffiness * pre.rbar, reach: pre.rbar, width: pre.wt.wmax,
    aspect: puffFaceAspect(cage),
    deviation: solved.dev.signed, worstDeviation: solved.dev.worst, rimSamples: solved.rim,
    /* How much of the drawing the radial warp could not follow — see the header.
       An approximation that reports its own size is a different object from one
       that does not. */
    starShaped: pre.reentrant === 0, reentrant: pre.reentrant,
    note: pre.reentrant === 0 ? '' :
      `This outline folds back on itself — ${(100 * pre.reentrant).toFixed(0)}% of the directions `
      + `out of its middle meet the line more than once, so the puff bridges the hollow and its edge `
      + `sits up to ${(100 * solved.dev.worst).toFixed(0)}% of the drawing's own reach off the line `
      + `you drew. Draw it as two overlapping shapes to keep the hollow.`,
    rung: PUFF_BALL_N.indexOf(n),
  };
}
