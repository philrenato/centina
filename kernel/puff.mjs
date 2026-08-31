// PUFF — a closed outline, inflated into closed quad geometry.
//
// Draw a closed outline with a pencil, inflate it into quads, with controls for
// thickness and profile, a Bake button. "If it's possible."
//
// ⚠⚠ THIS IS THE UNION OF TWO IMPLEMENTATIONS AND NEITHER ONE ALONE. The research pass measured
// three prior versions and the honest summary is:
//   SWIFT gives the TOPOLOGY — a SHARED EQUATOR RING. Front and back reference the same ring, so
//     there is nothing to stitch. The Grasshopper plugin lost this and pays a four-level join
//     fallback cascade (three JoinBreps at widening tolerances, three CreateSolid, a mesh-weld,
//     and finally "return the halves unjoined") for it. There is no join code in this file at all,
//     because there is nothing to join.
//   GRASSHOPPER gives the MATHEMATICS — elevation from a DISTANCE FIELD. Swift set height from
//     RING INDEX ALONE: it never looks at the shape, so a thin ribbon and a fat blob of the same
//     bounding box get the same height. That is the defect the distance field exists to fix.
//
// ⚠⚠⚠ AND THE FIXTURE IS THE FINDING. The research plan's own §0 records that its central
// measurement — the one that cleared a voxel route — was taken on a ROUND BLOB, the one shape
// structurally incapable of exhibiting the failure being ruled out: a blob's inradius IS the brush
// radius by construction. Measured on a crescent of the same drawn extent, the inradius is 0.0438
// against the disc's 0.1599 — 0.70 of a voxel against 2.56. So this module computes distance
// EXACTLY ON THE POLYGON at every lattice vertex, where a crescent's 0.0162 resolves as precisely
// as a disc's 0.1599 and the whole sub-voxel class does not arise. THE CRESCENT IS THE FIXTURE
// THAT MATTERS, not the disc.
//
// ⚠ ONE CENTRE IN THIS WHOLE FILE, and it is `inradius.at`. The Swift version carried THREE in one
// function — bounding-box mid-point for normalisation, centroid for the rings, and the WORLD ORIGIN
// for the cap — so an asymmetric outline got a skewed cap. The incentre is the only point at which
// the normalised distance reaches 1, which is the only defensible apex.
//
// ⚠ AND NOTHING HERE IS PARAMETERISED BY ANGLE. Swift's `sampleAngle` was a nearest-angle pick with
// no interpolation, so any non-star-shaped outline — a crescent, a C, an S — silently collapsed into
// "a plausible blob that is not what was drawn", which the research pass calls the sharpest
// correctness gap in that file. Ring membership here is by INDEX into an arc-length-resampled
// outline, and elevation is from `distanceToBoundary`, which is exact for any simple polygon
// whether or not it is star-shaped.

import { buildSpine, attachToSpine } from './puffspine.mjs';
import { buildGridCage, sigmaAlong } from './puffgridcap.mjs';
import { subdivideCatmullClark as ccSubdivide } from './subd.mjs';
/* ⚠ ONE SUBDIVISION, THROUGH THIS KERNEL'S OWN CATMULL-CLARK. The refusal test
   below asks whether the SUBDIVIDED surface bulges outside the drawn line, and
   that answer must come from the same subdivider the shape will actually be
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
  prepareOutline, distanceToBoundary, pointInPolygon, signedArea, taubinSmoothClosed,
} from './puffoutline.mjs';

// ================================================================================================
// THE TWO PROFILE FAMILIES — ported verbatim, and only one of them is exposed
// ================================================================================================
/**
 * PERPENDICULAR WALLS (Lamé).   f(t) = (1 - (1-t)^n)^(1/n),   n = 2 * 2.5^(1-2p),  n >= 1.1
 *
 * ⚠ IMPLEMENTED, DELIBERATELY NOT EXPOSED. Measured max‖∇F‖ over the family: 28.92 at p=0, 9.24 at
 * p=0.5, 1.25 at p=1, against a `sdCapsule` control that reads 1.000 exactly. A smooth-minimum
 * blend acts where |a-b| < k, so a field L times steeper produces a fillet k/L wide — which means a
 * Melt control would mean something different at every Profile setting. Profile is not a free
 * slider. It is kept because a disc at p=0.5 is an EXACT HEMISPHERE of height H, which is the only
 * closed-form oracle this module has.
 *
 * ⚠ p = 0.5 IS THE HEMISPHERE HERE AND A PARABOLOID IN THE OTHER FAMILY. The Grasshopper README
 * calls p=0.5 "an exact hemisphere" without saying which family; it is true of this one (n=2) and
 * false of the tangent foot. Do not carry that sentence anywhere near the other function.
 */
export function lameProfile(t, p) {
  const n = Math.max(1.1, 2 * Math.pow(2.5, 1 - 2 * p));
  const u = Math.min(Math.max(t, 0), 1);
  return Math.pow(1 - Math.pow(1 - u, n), 1 / n);
}

/**
 * TANGENT FOOT.   f(t) = 1 - (1 - t^2)^a,   a = 2.5^(1-2p)
 *
 * THE SHIPPED FAMILY. max‖∇F‖ is 1.19 / 1.26 / 1.80 at p = 0 / 0.5 / 1 — the same order as the
 * capsule control, so a blend radius means the same thing across the whole range.
 *
 * ⚠⚠ THE DOCUMENTATION FOR THIS FUNCTION IS EXACTLY BACKWARDS IN ITS ORIGINAL AND THE ERROR IS
 * WORTH KEEPING VISIBLE. The Grasshopper README's "mathematical heart" says this family "rises
 * perpendicular from the outline, every time". Differentiate it: f'(t) = 2a·t·(1-t²)^(a-1), so
 * f'(0) = 0 — it leaves the outline TANGENT, which is what the source comment (not the README)
 * says and what the name says. A perpendicular wall is the other family. Twelve documentation
 * claims were measured against that source and this was one of two that were mathematically wrong.
 *
 * ⚠ NO MINIMUM CLAMP ON `a`, and that asymmetry with the Lamé family is in the sources too: only
 * `n` carries one. a runs 2.5 (p=0) to 0.4 (p=1) and is well-behaved throughout.
 */
export function tangentProfile(t, p) {
  const a = Math.pow(2.5, 1 - 2 * p);
  const u = Math.min(Math.max(t, 0), 1);
  return 1 - Math.pow(1 - u * u, a);
}

export const PROFILES = Object.freeze({ tangent: tangentProfile, lame: lameProfile });

// ================================================================================================
// THE REFUSALS — a puff is ONE closed loop with NO hole
// ================================================================================================
/**
 * ⚠ REFUSALS ARE NAMED, NEVER SILENT, AND NEVER REPAIRED. The Swift version's self-intersection
 * guard `isContourClean` exists, is OFF BY DEFAULT, and guards only the randomisers — so a
 * hand-drawn self-intersecting outline goes straight in and comes out as something else. And a
 * repair would be a guess about which of two readings the hand meant.
 *
 * Accepts either one loop (a flat [x,y,...] array, or an array of [x,y] pairs) or an ARRAY OF
 * LOOPS. More than one loop is refused, and the two ways of having more than one are told apart,
 * because they are different things for a person to do about it: a hole means "this shape needs a
 * feature that does not exist yet", disjoint means "that is two gestures, puff them separately".
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
  return { ok: false, reason: 'multi-region', why:
    'This is more than one separate shape, and a puff is one body. Draw and puff them one at a time.' };
}

// Normalise whatever the caller drew into an array of loops of [x,y] pairs.
function toLoops(raw) {
  if (!Array.isArray(raw) || raw.length === 0) return [];
  if (typeof raw[0] === 'number') {                                  // flat [x,y,x,y,...]
    const l = []; for (let i = 0; i + 1 < raw.length; i += 2) l.push([raw[i], raw[i + 1]]);
    return [l];
  }
  if (Array.isArray(raw[0]) && typeof raw[0][0] === 'number') return [raw];   // one loop of pairs
  return raw.map(toLoops).map(l => l[0]).filter(Boolean);            // array of loops
}

// ================================================================================================
// THE BUILD
// ================================================================================================
/**
 * @param raw      one closed outline, or an array of loops (which is refused — see classifyLoops)
 * @param opts
 *   thickness   0..1, RELATIVE. `H = thickness * 4 * dmax`.
 *               ⚠ RELATIVE BECAUSE ABSOLUTE DOES NOT TRAVEL. The usable height range is 7-12x
 *               smaller on a crescent than on a disc of the same drawn size, so a slider in world
 *               units is live over a different fraction of its travel on every shape drawn. The
 *               absolute value is a READOUT, not the control.
 *   profile     0..1, into the tangent family
 *   family      'tangent' (shipped) | 'lame' (the gate's analytic oracle)
 *   smoothing   passes of the damped Laplacian over the FIELD. 0 is legal and is the identity.
 *   bands       rings per side, M
 *   count       points per ring, N
 *   bottomScale 1.0 = symmetric
 * @returns { ok, reason, why, positions, quads, tris, ... } — a refusal carries no geometry
 */
// ⚠⚠⚠ MEASURED ON THE SUBDIVIDED SURFACE, WHICH IS THE ONE ON SCREEN — and the cage version of this
// test was green while the displayed shape filled the hollow almost completely. The cage of a C
// spills 0.00000 outside the drawn line; subdivided twice it spills 0.227, which is 81% of its own
// inradius, and a comma spills 99.5%. A 1.3-turn spiral passed the cage test and spilled 269%.
// Catmull-Clark pulls vertices toward neighbourhood averages, so a fan whose straight cage edges
// stay inside a concave outline bulges straight across the hollow once it is smoothed. Testing the
// cage was testing the wrong surface.
//
// ⚠ ONE LEVEL, NOT TWO. It is a refusal test, not a rendering: one level already separates the two
// populations by more than an order of magnitude and costs a quarter as much.
//
// ⚠ THE THRESHOLD IS NOT TUNED. Measured, as a fraction of the inradius: everything that looks
// right lands under 10% and everything that looks wrong lands over 200%. 0.25 sits in a gap with
// twenty times the margin on either side, which is the opposite of a fitted constant.
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
    // ⚠ THE POLE-FREE CAGE, DEFAULT OFF. cap:'grid' replaces the ring-to-apex lattice with
    // arc-spaced rings down to a hole plus a Coons grid patch — see puffgridcap.mjs. The
    // default 'apex' path below is character-for-character what shipped.
    // ⚠ 'auto' USES THE GRID EXACTLY WHERE THE APEX FAILS, and that boundary is measured, not
    // guessed. On a STAR-SHAPED outline the apex path webs nothing (disc and hand both 0.00000
    // excursion outside the drawn line, subdivided) and the grid slightly worsens the hand
    // (0 -> 1.3%), so star-shaped keeps the apex path character-for-character. On a SPINED outline
    // the apex fan sails across the drawn hollow: measured on the SUBDIVIDED surface, a C spills
    // 81% of its inradius outside the line it was drawn from and a comma 99.5%. The grid takes
    // those to 0.7% and 0.6%.
    cap = 'auto', capFrac = 0.5, capBands = null, capRows = null,
    // Iterations of Taubin fairing applied to the DRAWN CURVE before any geometry is derived.
    curveSmooth = 20,
    // ⚠ VERIFY-ONLY, AND IT EXISTS BECAUSE A GATE MUST BE ABLE TO MEASURE WHAT IT COMPARES
    // AGAINST. The spill test now refuses the apex cage on a C or a comma — correctly, it bulges
    // 81-99% of its own width across the drawn hollow — but `puff_gridcap_guard`'s control arm has
    // to BUILD that shape to measure it. Without this the control cannot exist and the comparison
    // becomes a quoted number instead of a measurement taken beside the treatment.
    allowSpill = false,
  } = opts;

  const loops = toLoops(raw);
  if (loops.length === 0) return { ok: false, reason: 'empty', why: 'Nothing was drawn.' };
  const cls = classifyLoops(loops);
  if (!cls.ok) return { ok: false, reason: cls.reason, why: cls.why };

  // ---- S1: hygiene. Every one of these refusals already exists and is already gated ------------
  const prep = prepareOutline(loops[0], { count });
  if (!prep.ok) return { ok: false, reason: 'outline', why: prep.why, at: prep.at };

  // ⚠ THE DRAWN CURVE IS FAIRED BEFORE ANYTHING IS MEASURED FROM IT. Every quantity below — the
  // distance field, the inradius, the medial spine — is derived from `pts`, so smoothing it here
  // means the tremor never enters the geometry rather than being smoothed out of the surface
  // afterwards. Smoothing the SURFACE was tried and does not work: it flattens the dome into a
  // plateau and leaves the lobes, because by then the lobes are the shape.
  // Set `curveSmooth: 0` to build from the raw stroke.
  let { pts, area, inradius: dmax, inradiusAt: centre, inradiusSpacing: spacing, diag } = prep;
  if (curveSmooth > 0) {
    const faired = taubinSmoothClosed(pts, { iters: curveSmooth });
    const re = prepareOutline(faired, { count });
    // ⚠ IF FAIRING MAKES A CURVE THE PREPARER WILL NOT ACCEPT, KEEP THE ORIGINAL rather than
    // refusing a shape that was perfectly drawable. Fairing is an improvement, not a gate.
    if (re.ok) { pts = re.pts; area = re.area; dmax = re.inradius; centre = re.inradiusAt; spacing = re.inradiusSpacing; diag = re.diag; }
  }

  // ---- S2: is there an interior at all, and do we KNOW there is? ------------------------------
  // ⚠ `inradius` returns its own `spacing` precisely so a caller can say how sure it is rather than
  // implying exactness. A region whose deepest point is within one sample of the boundary has not
  // been measured, it has been missed — and returning a zero-height mesh silently is the
  // "plausible result that is not what was drawn" failure in its quietest form.
  if (!(dmax > 0) || !centre) {
    return { ok: false, reason: 'sliver', why:
      'This outline has no measurable inside — it is a sliver rather than a shape. Draw it wider.' };
  }
  if (dmax <= spacing) {
    return { ok: false, reason: 'sliver', why:
      `This outline is too thin to measure — its deepest point is ${dmax.toFixed(4)} from the edge, `
      + `within the ${spacing.toFixed(4)} the measurement can resolve. Draw it wider.` };
  }

  // ---- S3: resolution FROM THE INRADIUS, never from the drawn extent ---------------------------
  // ⚠ THE BOUNDING BOX IS NOT A PROXY FOR SIZE HERE AND THAT IS MEASURED: a crescent and a disc of
  // the same extent differ by 21% in half-bounding-diagonal and by 0.137 against 1.000 in inradius.
  // Anything sized from the extent is sized by the wrong number on exactly the shapes that matter.
  // ⚠⚠ N MUST BE EVEN, AND THAT IS THE POLE'S REQUIREMENT RATHER THAN A PREFERENCE. Each pole is
  // closed with PAIRED QUADS — one per two ring vertices — because the app this feeds is strictly
  // quads-only: its buildTopology strides by four and has no concept of face arity, and its
  // validateQuadMesh rejects a triangle written as [a,b,c,c] by name, as "a degenerate quad". An
  // odd ring cannot be paired, so it is rounded up here rather than discovered at install time.
  const N = pts.length;
  const M = bands != null ? Math.max(1, bands | 0)
    : Math.max(2, Math.min(24, Math.round(N / 8)));
  if (N % 2 !== 0) {
    return { ok: false, reason: 'odd-ring', why:
      `A puff closes each end with paired quads, so it needs an even number of points around the `
      + `edge and was given ${N}. Ask for an even count.` };
  }

  // ---- S3b: IS THIS SHAPE STAR-SHAPED ABOUT ITS OWN DEEPEST POINT? -----------------------------
  // ⚠⚠⚠ THE LATTICE CARRIES A STAR-SHAPE ASSUMPTION THAT THE FIELD DOES NOT, AND THE RESEARCH PLAN
  // THIS WAS BUILT FROM DID NOT NOTICE. Its step 4 is "ring k vertex i = lerp(pts[i], centre,
  // k/(M+1))" — a straight segment from each boundary point to one interior point. On a crescent
  // that segment LEAVES THE SHAPE: measured, 23 of 96 boundary points, against 0 of 96 on a disc.
  // So the ring vertices land in the bite, and the puff bulges across a concavity that was drawn
  // empty. That is the SAME failure the plan condemns the ported source for — "returns a plausible
  // blob that is not what was drawn" — arriving by a different route, in the half of the algorithm
  // nobody was watching, because all the attention was on replacing angular sampling in the FIELD.
  //
  // ⚠ SO IT IS REFUSED BY NAME RATHER THAN SILENTLY COLLAPSED, and that is strictly better than the
  // source, which returns the blob. What it is NOT is the crescent working: a puff whose rings
  // follow the distance field's own gradient onto a medial SPINE, rather than converging on a
  // single apex, is a different and larger algorithm. Recorded as unbuilt rather than implied.
  const inside = (x, y) => pointInPolygon(pts, x, y);
  let escaping = 0;
  for (const q of pts) {
    for (let sIdx = 1; sIdx <= 12; sIdx++) {
      const fr = sIdx / 13;
      if (!inside(q[0] + (centre[0] - q[0]) * fr, q[1] + (centre[1] - q[1]) * fr)) { escaping++; break; }
    }
  }
  // ⚠⚠⚠ ESCAPING POINTS ARE A REASON TO CHANGE THE CONSTRUCTION, NOT TO REFUSE THE SHAPE.
  // "Every puff shape drawn needs to resolve" — said twice, and it is right. A crescent, a C, an S,
  // a comma, a bean and a boomerang are ordinary things to draw, and a tool whose promise is "draw
  // a closed loop, get a solid" cannot answer a comma with a paragraph about why it will not.
  //
  // The escape is real and it is a fact about ONE construction: every boundary point is lerped
  // toward a SINGLE interior point, and on a shape that curves back on itself that segment leaves
  // the polygon and fills in the hollow you drew. The medial axis is the fix — each point aims at
  // the spine sample whose inscribed ball actually reaches it, so the spoke is provably inside —
  // and it is now what happens instead of the refusal.
  //
  // ⚠ THE SPINE'S OWN RECORDED BLOCKER WAS MEASURED AGAINST THE WRONG PROFILE. Its header says it
  // could not ship because the cap creased: 172 degrees on a crescent, 178 on a hand-drawn loop.
  // Both numbers were taken with `family: 'tangent'`, whose profile has slope 2 at the peak — a
  // CONE TIP — which is the same defect that turned out to be the flange and the pinched centre on
  // every ordinary puff. The cap was blamed for the profile's crease. Re-measure before believing
  // that paragraph.
  let spineTarget = null, spineSigma = null;
  if (escaping > 0) {
    // ⚠ `buildSpine` takes the inscribed ball as `{ at, r }`, not a bare point — it refines the
    // root outward from a RADIUS, so handing it only a position throws inside refineBall.
    const sp = buildSpine(pts, { seed: { at: centre, r: dmax } });
    if (sp && sp.ok) {
      const att = attachToSpine(pts, sp);
      // The residual IS the safety statement: how far each point lies OUTSIDE the ball assigned to
      // it. A form the spine does not describe (a branch, a third limb) shows up here and nowhere
      // else, so it is checked rather than assumed.
      // ⚠⚠ THE RESIDUAL IS A PROXY AND IT OVER-PREDICTS. It bounds how far a point lies OUTSIDE the
      // ball assigned to it, which bounds how far a spoke could leave the polygon — but a bound is
      // not the thing. At `0.25 * dmax` it refused a ZIGZAG whose worst residual was only 34% over
      // the line, on 3 of its 96 points, and which builds perfectly: Euler 2, quads only, and a
      // measured excursion outside the drawn outline of 0.00000. A shape that works was being
      // refused by a number I picked.
      // So the residual is kept only as a CHEAP SANITY BOUND, and the real test is the consequence
      // itself — every lattice point is checked for containment below, which is exactly the
      // condition the refusal claims to prevent ("inflating it would fill in the hollow you drew").
      // Refusing on the actual defect rather than on a proxy for it is the whole change.
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

  // ---- S3c: the POLE-FREE cage, behind its option ----------------------------------------------
  // Same refusals, same spine decision, same elevation law (distance field through the profile),
  // same shared equator — a different LATTICE. Everything below this block is untouched when the
  // option is off.
  // ⚠ THIS PATH RETURNS BEFORE THE LATTICE CONTAINMENT CHECK IN S4, so a shape whose lattice would
  // land outside the drawn line is NOT refused here. That check is what refuses a 1.9-turn spiral
  // on the default path; `puff_curved_guard` A6 exercises the default path only. Owed before this
  // option can become the default.
  // ⚠ 'auto' NOW MEANS THE GRID FOR EVERY SHAPE, not only for the ones that curve back. It was
  // spined-only because on my CLEAN fixtures the apex path spilled nothing and the grid slightly
  // worsened the hand's silhouette. Those fixtures were the wrong input — "lines drawn into puff
  // will always be irregular, blobby" — and on an irregular stroke the apex fan is the dominant
  // artefact: a star of creases radiating from the pole, plainly visible in a render and invisible
  // to every dihedral scan tried. Pass `cap: 'apex'` to get the old lattice.
  // ⚠⚠⚠ 'auto' PICKS THE CAP FROM THE PROFILE FAMILY, and that is measured rather than chosen.
  // A grid patch closes a DOME well and a CONE TIP badly: with the tangent family, whose profile
  // has slope 2 at the peak, the apex path beats the grid on a disc 7.1 degrees against 84.9 — a
  // cone tip genuinely wants a pole, because a patch has to flatten across the point the profile is
  // trying to make. With Lame n=2 (a hemisphere) the grid wins everywhere.
  // ⚠ THIS WAS BRIEFLY 'grid for every shape' AND IT REGRESSED EVERY CALLER THAT DOES NOT PASS A
  // FAMILY. `buildPuff`'s own default is still 'tangent', so puff_guard's disc went 4.5 -> 51.0
  // degrees. The app states 'lame' and gets the grid; a caller that wants the tangent foot keeps
  // the pole that suits it. Pass `cap` explicitly to override either way.
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

  // ---- S4: the ring lattice. CONSTANT N per ring — no taper ------------------------------------
  // ⚠ THE TAPER IS NOT LOAD-BEARING FOR THE SHAPE, and dropping it is what makes the output quads.
  // Measured on the Swift output at shipped defaults: 3,064 quad pairs against 256 lone triangles,
  // 96.0% pairable, and the lone count is exactly N — the equator sample count — independent of
  // band count. So the triangles were never the geometry, they were the emission.
  let latticeEscaped = 0;
  const ringXY = [];                                   // ringXY[k][i] = [x, y]
  for (let k = 0; k <= M; k++) {
    const f = k / (M + 1);
    const ring = new Array(N);
    for (let i = 0; i < N; i++) {
      // Per-point target when the shape curves back on itself, the single centre otherwise — and
      // for a star-shaped outline `spineTarget` is null, so this is character-for-character the
      // construction that shipped. A shape that worked before cannot change.
      const tx = spineTarget ? spineTarget[i][0] : centre[0];
      const ty = spineTarget ? spineTarget[i][1] : centre[1];
      ring[i] = [pts[i][0] + (tx - pts[i][0]) * f,
                 pts[i][1] + (ty - pts[i][1]) * f];
      // ⚠ THE CONTAINMENT TEST THAT REPLACES THE PROXY. Ring 0 IS the outline, where a
      // point-in-polygon test is undefined, so it is exempt; every interior ring point must be
      // inside the shape or the lattice has landed in the hollow. Only the spine path can fail
      // this — with a single centre, `escaping === 0` already guarantees it.
      if (spineTarget && k > 0 && !inside(ring[i][0], ring[i][1])) latticeEscaped++;
    }
    ringXY.push(ring);
  }

  // ⚠ REFUSED ON THE MEASURED CONDITION, NOT ON A BOUND. If the lattice actually landed outside the
  // drawn shape then inflating it WOULD fill in the hollow, and that is worth refusing by name.
  // If it did not, the shape resolves — which is the whole point of the spine path.
  if (latticeEscaped > 0) {
    return { ok: false, reason: 'not-star-shaped', escaping, latticeEscaped, why:
      `This shape curves back on itself more than Puff can follow — ${latticeEscaped} points of the `
      + `inflated lattice land outside the line you drew, so it would fill in the hollow. `
      + `Draw it as two overlapping shapes.` };
  }

  // ---- S5: the field. Exact distance to the drawn boundary, per lattice vertex ------------------
  const d = [];
  for (let k = 0; k <= M; k++) {
    const row = new Float64Array(N);
    for (let i = 0; i < N; i++) row[i] = k === 0 ? 0 : distanceToBoundary(pts, ringXY[k][i][0], ringXY[k][i][1]);
    d.push(row);
  }

  // ---- S6: smooth the FIELD, damped and boundary-LOCKED -----------------------------------------
  // ⚠ λ = 0.5 AND RING 0 NEVER MOVES. Swift's Laplacian is undamped (λ=1) and unlocked, which drags
  // the equator off the drawn outline — the silhouette stops being where you drew it, which is the
  // one thing a student is entitled to. Ring 0 is byte-identical before and after this loop.
  // ⚠ 0 PASSES IS LEGAL AND IS THE IDENTITY. Swift's `max(2, …)` made smoothing:0 unreachable, so
  // the unsmoothed field could not be seen even to compare against.
  const passes = Math.max(0, smoothing | 0);
  for (let it = 0; it < passes; it++) {
    const next = d.map(r => Float64Array.from(r));
    for (let k = 1; k <= M; k++) {                     // k = 0 LOCKED
      for (let i = 0; i < N; i++) {
        const im = (i - 1 + N) % N, ip = (i + 1) % N;
        const km = k - 1, kp = Math.min(M, k + 1);
        const avg = (d[k][im] + d[k][ip] + d[km][i] + d[kp][i]) / 4;
        next[k][i] = d[k][i] + 0.5 * (avg - d[k][i]);
      }
    }
    for (let k = 0; k <= M; k++) d[k].set(next[k]);
  }

  // ---- S7/S8/S9: normalise, cap, lift -----------------------------------------------------------
  // ⚠ NORMALISED BY THE PRE-SMOOTHING dmax. Renormalising to the smoothed maximum would cancel the
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

  // ---- S10/S11: vertices and quads, about ONE shared equator ------------------------------------
  //   equator(i)   = i                       z = 0   SHARED by both halves
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

  // ⚠ THE WINDING IS CORRECT BY CONSTRUCTION AND THERE IS NO NORMAL-FLIP HEURISTIC IN THIS FILE.
  // `prepareOutline` guarantees counter-clockwise, so the interior lies to the LEFT of the tangent
  // and the front order (outer_i, outer_j, inner_j, inner_i) faces +z on every band of every shape,
  // convex or not. Swift could not get this by construction and paid for it with a centroid-based
  // majority-vote flip implemented TWICE, in two layers, at two different sample rates — and that
  // heuristic is structurally wrong for a non-convex blob, because on a crescent a legitimately
  // outward normal points toward the centroid.
  // ⚠⚠⚠ QUADS ONLY, INCLUDING AT THE POLES, AND THE FAN THIS REPLACED WAS UNINSTALLABLE.
  // The first version closed each pole with a triangle fan, which is the textbook answer and is
  // exactly what this app cannot take: buildTopology is hard-wired to `quads.length / 4` with no
  // face-arity concept at all, and validateQuadMesh refuses a triangle smuggled in as [a,b,c,c]
  // because a repeated corner IS its definition of a degenerate quad. The repo's own primitives
  // are shaped around this — the cylinder is a deformed cube rather than a capped tube, and the
  // cone keeps a tiny disc instead of a point, with the card telling the student why. So a fan was
  // never a topology this module could hand over, however correct it was in the abstract.
  //
  // PAIRED POLE QUADS: quad(ring_i, ring_i+1, ring_i+2, apex) for every EVEN i. Each ring edge is
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
  // Kept as an empty array rather than removed: `tris` is part of this module's returned shape and
  // a consumer destructuring it should get something iterable, not undefined.
  const tris = new Uint32Array(0);

  // The SAME test on the apex path. A star-shaped outline spills nothing here, so this costs one
  // subdivision and changes no accepted shape — but a spined outline that somehow reaches this path
  // must not escape a check the grid path applies.
  const spillA = hollowSpill(pts, positions, quads, dmax);
  if (!allowSpill && spillA.frac > SPILL_LIMIT) return spilled(spillA, escaping, dmax);
  return {
    ok: true, reason: null, why: '',
    positions, quads, tris,
    // ⚠ THE PREPARED OUTLINE, RETURNED RATHER THAN RE-DERIVED BY THE CALLER. `repinRim` needs the
    // exact polygon this cage was built against — resampled, closed, wound and deduplicated by
    // `prepareOutline`. A caller preparing it a second time from the same raw stroke would almost
    // certainly agree, and "almost certainly" is how a pin lands one vertex out.
    outline: pts,
    nv: V, N, M,
    dmax, centre, area, spacing, diag,
    height: effectiveH, requestedHeight: H, capped: H > dmax * 4.0,
    profile, family, smoothing: passes, bottomScale,
    spined: !!spineTarget, escaping,
    equatorCount: N,
    // ⚠ THE SMOOTHED FIELD ITSELF, RETURNED SO IT CAN BE ASSERTED ON. The heights are a monotone
    // but NON-LINEAR function of it, so a midpoint in the field is not a midpoint in z — which
    // means a gate reading only geometry cannot tell a damped step from an undamped one. It
    // measured identical either way until this was exposed.
    field: d,
  };
}

// ================================================================================================
// THE SELF-CHECK — a mesh this file would not hand out
// ================================================================================================
/**
 * ⚠ EULER IS THE ONE THAT CANNOT BE FUDGED. V - E + F = 2 says closed, genus 0 and watertight in a
 * single number, and it is the assertion a shared equator either satisfies or does not. The four
 * cheaper checks are here because Euler alone cannot tell a correct mesh from two different errors
 * that cancel.
 */
export function puffInvariants(p) {
  const faces = [];
  for (let i = 0; i < p.quads.length; i += 4) faces.push([p.quads[i], p.quads[i+1], p.quads[i+2], p.quads[i+3]]);
  for (let i = 0; i < p.tris.length;  i += 3) faces.push([p.tris[i],  p.tris[i+1],  p.tris[i+2]]);

  const edge = new Map();
  // ⚠ DIRECTED EDGES, BECAUSE AN UNDIRECTED COUNT CANNOT SEE A REVERSED HALF. Reversing the back
  // half's winding leaves every edge still shared by exactly two faces — the undirected count is
  // unchanged, Euler is unchanged, and the mesh is inside-out. In a correctly oriented closed
  // manifold each DIRECTED edge a→b occurs exactly once and its opposite exactly once.
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
        && nan === 0 && (V - E + F) === 2 && misoriented === 0,
  };
}

/** Peak absolute height — what a thickness readout reports, and what the smoother must lower. */
export function peakHeight(p) {
  let m = 0;
  for (let i = 2; i < p.positions.length; i += 3) m = Math.max(m, Math.abs(p.positions[i]));
  return m;
}

// ================================================================================================
// THE CAGE A PUFF HANDS TO THIS KERNEL
// ================================================================================================
/* ⭐ THE CAGE IS A DEFORMED QUAD BALL, NOT A SET OF RINGS, AND THE REASON IS
   FACE QUALITY AT A COUNT A READER CAN EDIT.

   The ring construction above is excellent and it is not usable here. Its rings
   run inward to a small cap, so coarsening it does not make bigger quads — it
   makes SLIVERS. Measured on a drawn circle, worst edge ratio within a face:

       rim/bands   faces   worst aspect
        12 / 1       34      29.2 : 1
        10 / 2       48      25.0 : 1
        16 / 3      110       5.0 : 1
        20 / 3      170       3.9 : 1

   Nothing at or under four dozen faces is better than 25:1, and at 34 faces the
   cage has only TWO distinct ring radii — the rim and a tiny cap — so every face
   spans the whole gap between them. A cage like that is unpleasant to grab and
   reads as a pinched surface however good its silhouette is.

   A cube-derived quad ball is near-square BY CONSTRUCTION at any size — 1.3:1
   before it is warped at every n from 2 to 6 — and warped onto a drawn circle it
   comes back at 3.1:1 for 24 faces and 2.3:1 for 54. It carries eight valence-3
   extraordinary vertices at the old cube corners and nothing else extraordinary,
   none of them on the silhouette.

   ⚠ THE TRADE IS PAID IN SILHOUETTE, KNOWINGLY. A ball warped radially onto the
   outline follows a round shape closely (1%) and a lobed one loosely (13-19%).
   At two dozen faces no construction does better — the ring cage measured 30-49%
   on the same outlines, and with slivers. The count is the reader's constraint;
   the silhouette is what gives.

   ⚠⚠ AND THE RADIAL WARP ASSUMES THE OUTLINE IS STAR-SHAPED about its own
   centroid. A shape that folds back has directions with more than one crossing,
   and this takes the OUTERMOST, so a C is inflated across its own mouth.

   THAT IS APPROXIMATED, NOT REFUSED, AND THE RESULT SAYS SO. A refusal would be
   the cheaper code and the worse tool: the promise is "draw a closed loop, get a
   solid", and a comma, a bean and a boomerang are ordinary things to draw. The
   approximation is BOUNDED — the outermost crossing is still the outline's own
   reach, so the cage never leaves the drawn extent — and every result carries
   `starShaped`, `reentrant` (the fraction of sampled directions crossing the
   outline more than once) and `worstDeviation` (how far the limit silhouette
   actually lands from the drawn line), which is the number that says how much of
   the shape was lost. A caller with somewhere to put a warning has one; a caller
   with nowhere to put it still gets geometry.
   ⚠ A DEEP CONCAVITY CANNOT BE REPRESENTED AT THIS FACE COUNT BY ANY
   CONSTRUCTION, so the star-shape assumption is not what is costing the reader
   the concavity — the face budget is. Refusing would spend the tool and buy
   nothing back. */
const PUFF_BALL_N = [2, 3, 4, 5, 6];
const PUFF_RADIUS_SAMPLES = 128;

/* ⭐⭐ THE PARAMETER SURFACE, DECLARED ONCE AND ENFORCED FROM THE DECLARATION.
   `puffCage` clamps every value against THIS OBJECT rather than against literals
   of its own, so a bound cannot be written here and disagree with the code that
   runs. A UI builds one row per entry and needs to know nothing else about puff.

   ⚠ EVERY ONE OF THESE MOVES THE GEOMETRY AND NONE OF THEM MOVES ANOTHER'S
   EFFECT, which is a stronger claim than "they are all wired up" and is the
   claim that makes them draggable:
     · `density` changes the cage and NOT the peak height. The peak is imposed —
       `puffiness` times the outline's own mean reach — rather than inherited from
       the ball's z, so the two are independent by construction rather than by
       luck. Inherited, a density slider would secretly be a shape slider: the
       peak moves 9.5% between density 2 and 3 purely because an odd-n ball has no
       vertex at its own pole.
     · `puffiness`, `follow` and `bottomScale` change the height field and NOT
       the silhouette. The cage's (x, y) never reads any of them, and the growth
       solve that centres the silhouette runs on a fixed reference height, so the
       plan view is bit-for-bit the same across the whole of their travel.

   ⚠⚠ EVERY LOWER BOUND HERE IS A FACE-QUALITY BOUND, AND EACH ONE IS QUOTED
   RATHER THAN ASSUMED. A cage is only worth editing if a hand can grab a face, so
   the number that governs is the worst edge ratio within a face. Over a family of
   eight outlines at densities 2 through 6 it is 7.4:1 at these defaults, and at
   each parameter's own extreme:

       puffiness 0.15 (min)  13.7:1      puffiness 1.5 (max)   5.7:1
       bottomScale 0.35      13.1:1      follow 1 (max)       16.3:1

   The pattern is one fact: less height over the same plan makes longer, thinner
   faces. `puffiness` 0 and `bottomScale` 0 are excluded because they are not
   merely thin — they FOLD THE CAGE ONTO A PLANE. The quad ball carries a vertex
   at (x, y, +z) for every (x, y, -z); flattened, those become the same point, so
   a 26-vertex cage comes back with 9 of its vertices sitting on another one and
   an enclosed volume of exactly zero: two sheets lying on each other, which no
   weld, export or NURBS conversion can make a solid of.
   ⚠⚠ AND NOTHING MEASURES IT. The flattened cage's worst face ratio is 4.2:1 on
   a circle against 3.1:1 for the same cage inflated — a clean number for an
   object with no inside. Face aspect was added here because face COUNTS could not
   see a sliver; it cannot see this either, so the bound is the only thing
   standing between a slider and a degenerate solid.
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
  /* ⚠ A MODIFIER, AND IT SAYS SO. Without a flat side there is nothing to flip:
     at the defaults this moves the cage by exactly 0%, which is the shape of
     control this project treats as a defect rather than a caveat. `requires`
     names the parameter that has to be on for it to mean anything — declared
     once here so the panel can withdraw the row and the parameter sweep can
     satisfy the precondition before measuring, instead of each deciding for
     itself and drifting. */
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

/* `faces` is the older spelling of the same control and still travels, because a
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

/* HOW FAR THE OUTLINE REACHES IN ONE DIRECTION, and HOW MANY TIMES IT CROSSES.
   The reach is the OUTERMOST crossing, so a shape that folds back is bounded
   rather than pinched to its first fold; the crossing count is what says the
   fold happened, and it is the only thing that can.

   ⚠⚠ THE SIDE TEST IS HALF-OPEN AND THERE IS NO EPSILON, which is the same
   idiom point-in-polygon uses and for the same reason. A ray that passes exactly
   through a vertex touches two segments; parameterised along the SEGMENT and
   accepted on a tolerance, rounding decides whether that is one crossing or two,
   and the decision changes with the drawing's scale — a plain circle read as
   re-entrant at size 1 and star-shaped at size 0.001 and 1000. Classifying each
   endpoint by which SIDE of the ray line it lies on, with `> 0` and its negation
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

/* A CUBE, SUBDIVIDED n x n PER FACE AND PUSHED ONTO A SPHERE. Eight valence-3
   vertices at the old corners and nothing else extraordinary, and every face
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

/* ⭐⭐ LOCAL WIDTH IS NOT RADIAL REACH, AND USING THE REACH IS BACKWARDS ON THE
   SHAPES THAT MOTIVATE THE CONTROL AT ALL.

   The reach from the centroid at angle theta says how FAR the outline goes that
   way. Thickness wants to know how WIDE it is that way, and the two disagree in
   exactly the case a person notices: a fat body with a long thin tail reaches
   furthest along the tail and is narrowest there, so a height scaled by reach
   comes out FATTEST on the thinnest part of the drawing. Measured on a disc with
   a spike, height over the tail against height over the body: 1.32 with reach,
   0.44 with width. The control was not weak, it was inverted.

   WIDTH HERE IS THE LARGEST INSCRIBED CIRCLE STILL REACHABLE GOING OUTWARD:
   for a point at fraction s along the ray at angle theta, the maximum of
   `distanceToBoundary` over the rest of that ray. Written as a suffix maximum it
   costs one pass and it is monotone non-increasing in s by construction, so
   thickness can only fall off toward the rim and never rise.

   ⚠ THE PLAIN MAXIMUM OVER THE WHOLE RAY IS NOT ENOUGH, and it is the obvious
   thing to write. Every ray starts at the centroid, which on a disc-with-a-spike
   sits in the BODY — so the maximum over the whole ray is the body's width in
   every direction including the tail's, and the tail reads as fat again. Taking
   the maximum only from the point OUTWARD is what makes a point out in the tail
   see the tail.

   ⚠ AND THERE IS NO DIRECTION AT THE CENTRE. Row 0 is the shape's own largest
   inscribed radius for every theta, so the apex height is a property of the
   shape rather than of whichever way `atan2(0, 0)` happens to point.

   ⚠⚠ THE LOOKUP STOPS HALF WAY OUT, AND WITHOUT THAT THE RIM BAND COLLAPSES.
   The distance to the boundary is ZERO at the boundary, by definition — so read
   at its own fraction, a vertex near the rim scores a width near zero, and at
   full following the band either side of the equator loses its height entirely:
   measured on a bean at density 5, two of its four edges fell to 0.001 against
   0.43 and the worst face ratio reached 465:1. The shape's own dome already
   carries the fall-off toward the rim; the width table is there to say how thick
   this DIRECTION should be, and past the half-way point it stops answering that
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

/* ⭐⭐ THE CAGE'S DIRECTIONS ARE SPACED BY ARC LENGTH ALONG THE DRAWN LINE, NOT
   BY ANGLE ABOUT THE CENTROID, AND ON AN ANISOTROPIC SHAPE THAT IS THE
   DIFFERENCE BETWEEN A CAGE AND A FAN OF SLIVERS.

   A ball warped by angle spends its vertices evenly in angle, so a shape whose
   reach varies — an ellipse, anything with a tail — gets the same number of
   vertices covering a long stretch of curve as a short one. The long stretch
   ends up spanned by one enormous quad and the short one by a pinched one.
   Measured as worst edge ratio within a face, at density 2 through 6, follow
   0.5:

       shape                 by angle                by arc length
       ellipse 3:1     6.5  3.7  5.9  5.4  5.8    3.9  3.5  4.1  3.6  4.1
       disc + thin tail 11.1  3.4 13.2  5.0 15.4    5.1  5.1  5.6  5.4  5.6

   Reparameterising costs nothing at run time — the outline is already resampled
   by arc length, so the map is that polygon's own vertex angles read as a table —
   and it leaves the topology, the height law and the growth solve untouched.

   ⚠ IT IS ANCHORED AT ANGLE ZERO, NOT AT THE STROKE'S FIRST POINT. Anchored at
   the first point, the cage would rotate with wherever the hand happened to
   start, so the same drawn shape would come back with its vertices in different
   places — a seam dependency, and an invisible one.

   ⚠ AND IT REFUSES ITSELF ON A RE-ENTRANT OUTLINE, returning null so the warp
   falls back to plain angle. A shape whose angle about the centroid is not
   monotone has no arc-length-to-angle map at all, and forcing one would fold the
   cage — a much worse failure than the coarse approximation the fallback gives.
   The `> PI` step guard is what catches it: a monotone traverse of a simple loop
   never turns half a revolution between two resampled neighbours. */
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

/* ⭐ HEIGHT IS SET, NOT INHERITED. The peak is `puffiness` times the outline's
   mean radial reach, imposed by RESCALING the whole height field to it once the
   field is built — so it is the same number at every density, at every `follow`,
   and a density slider cannot secretly be a shape slider. Read off the ball's own
   z instead, the peak moves 9.5% between density 2 and density 3 for no reason a
   reader could name, purely because an odd-n ball has no vertex at its own pole.
   ⚠ THE REACH IS THE BASE AND THE WIDTH IS THE MODULATION, which is the division
   of labour the two controls describe: `puffiness` says how tall the form is
   against its own drawn size, `follow` says how much of that height the narrow
   parts give back. Basing the height on the width instead makes the two controls
   fight — an elongated outline then comes out flatter as a side effect of a
   control the reader did not touch.
   ⚠ AND (x, y) NEVER READS A HEIGHT PARAMETER. That is what makes the silhouette
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
    /* ⚠ A FLAT BACK IS A PROJECTION, NOT A SMALL `bottomScale`. Pressing the
       underside toward the plane by a factor never reaches it — the back stays a
       shallow dome — and the form still rocks. Setting the whole lower half to
       the plane makes the base genuinely planar, and the ball's lower vertices
       land inside the outline where they tile it. */
    /* WHICH SIDE IS FLAT IS A CHOICE, NOT A FACT ABOUT THE CURVE. A drawn
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
  /* ⚠⚠ A FLAT BOTTOM NEEDS A CREASE AT ITS EDGE, and it is the one place this
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

/* ⭐⭐ THE SILHOUETTE SOLVE IS ONE SUBDIVISION, NOT ONE PER BISECTION STEP, AND
   THAT IS AN IDENTITY RATHER THAN AN APPROXIMATION.

   Every cage vertex is `centre + (unit ball point) * reach(theta) * grow`, so
   `grow` is a UNIFORM SCALING OF THE WHOLE CAGE ABOUT `centre`. Catmull-Clark is
   an affine combination of vertices, so the subdivided surface scales with it
   the same way. The limit rim at any `grow` is therefore the limit rim at
   `grow = 1` scaled — one subdivision serves every step of the search, and the
   search itself becomes a scan over a fixed list of points.
   That is the whole of the rebuild cost: 24 subdivisions became 1. With the
   outline's own tables kept between frames (see `opts.cache`), a density drag
   costs 0.24 ms a frame and a puffiness drag 0.04 ms, against a first build of
   14-21 ms.

   ⚠ THE RIM IS SELECTED BY HEIGHT AND THE TOLERANCE IS RELATIVE. An absolute one
   selects nothing on a millimetre outline and everything on a flat one. */
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

/* WORST EDGE RATIO WITHIN A FACE. Counting faces is not measuring a cage: a ring
   construction hit every face-count target this one does and reached 29:1, which
   no count-based check could see and a single highlighted face made obvious. */
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


/* THE HEIGHT PARAMETERS ARE HELD AT A FIXED REFERENCE FOR THE SOLVE, so `grow`
   is a function of the outline and the density and NOTHING ELSE. Solved against
   the live values instead, the rim selection would move with `puffiness` — not
   because the silhouette changes but because the test that finds the rim reads
   z — and a puffiness drag would wobble the plan view for no geometric reason. */
const PUFF_SOLVE_REF = Object.freeze({ puffiness: 0.45, follow: 0, bottomScale: 1 });
const PUFF_GROW_HI = 3.0;
const PUFF_GROW_STEPS = 28;

/* ⭐⭐ WHAT A DRAG FRAME ACTUALLY COSTS, AND WHY THE CACHE IS PART OF THE DESIGN
   RATHER THAN AN OPTIMISATION BOLTED ON.

   Everything expensive here is a property of the OUTLINE, not of the
   parameters: cleaning and resampling the stroke, the radial reach table, the
   local-width table, and — per density rung — the subdivided rim ring and the
   growth that centres it. A slider moves none of those. So a caller that keeps
   one plain object between frames and passes it as `opts.cache` pays the whole
   cost once per stroke and a few tenths of a millisecond per frame after that.

   ⚠ THE CACHE IS CHECKED AGAINST THE OUTLINE IT WAS BUILT FROM, and a mismatch
   REBUILDS rather than refusing. A stale cache handed to a new stroke would
   otherwise return the previous shape's geometry silently, which is the worst
   failure this file has — a plausible result that is not what was drawn. The
   fingerprint is exact (length, both ends, and the sum of every coordinate), so
   the only way past it is a genuinely different stroke with identical
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
  /* ⚠⚠ A REPEATED POINT IS DROPPED HERE AND NOWHERE ELSE. A pointer that reports
     the same position twice contributes a ZERO-LENGTH SEGMENT, and a zero-length
     segment lies on top of its neighbour — which the self-intersection test reads
     as the outline crossing itself. So a perfectly ordinary stroke from a device
     that repeats samples was refused by name, with a message about a crossing
     that does not exist. The duplicate carries no information about the shape, so
     removing it is not a repair of the drawing; leaving it in is. */
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
  /* ⚠ THE STROKE IS CLEANED BEFORE ANYTHING IS MEASURED FROM IT. `prepareOutline`
     closes it, winds it consistently, REFUSES one that crosses itself — which is
     not a shape and would silently produce a folded cage — and resamples it
     evenly so the radial reach is not dominated by wherever the hand slowed
     down. The refusals are the valuable half: a self-crossing loop caught here
     is named, and caught later is a mesh nobody can explain. */
  const prep = prepareOutline(pairs, { count: 96 });
  /* ⚠ THE DIAGNOSIS COMES FROM `prepareOutline` AND THE ACTION IS ADDED HERE.
     Its messages are exact about WHAT is wrong and some of them stop there — "the
     outline encloses no area" is true and leaves a reader with nothing to do. A
     refusal that names no change is a dead end wearing a sentence. */
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

  /* ⚠ AN OUTLINE WITH NO MEASURABLE INSIDE IS REFUSED BY THE SAME TEST THE REST
     OF THIS MODULE USES: the deepest interior point against what the sampling can
     resolve. A stroke that goes out and back is a line however much area rounding
     leaves it with, and returning a zero-height cage for one is the quietest
     version of "a plausible result that is not what was drawn". */
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
  /* ⚠ SUBDIVIDE ONLY AS FAR AS THE RIM NEEDS SAMPLING. The rim carries 8n control
     points, so a level doubles the sample count; the target is ~64 samples around
     the silhouette, which is the resolution the worst-case deviation is quoted at.
     A fixed three levels instead costs 9.5 ms at density 5 and buys nothing — a
     finer cage is already closer to its own limit surface. */
  const levels = n <= 2 ? 3 : 2;
  const refH = PUFF_SOLVE_REF.puffiness * pre.rbar;
  const ring = puffRimRing(
    puffWarp(pre.centre, n, 1, pre.radial, pre.wt, pre.rbar, PUFF_SOLVE_REF, pre.dirmap),
    subdivide, levels, refH);

  /* The growth that centres the limit silhouette on the drawn line. Bisected
     because the deviation is monotone in it and there is no closed form. */
  let lo = 1, hi = PUFF_GROW_HI;
  const outer = puffRimDev(ring, pre.centre, pre.poly, pre.scale, hi);
  for (let k = 0; k < PUFF_GROW_STEPS; k += 1) {
    const mid = (lo + hi) / 2;
    if (puffRimDev(ring, pre.centre, pre.poly, pre.scale, mid).signed < 0) lo = mid; else hi = mid;
  }
  const grow = (lo + hi) / 2;
  const dev = puffRimDev(ring, pre.centre, pre.poly, pre.scale, grow);
  /* ⚠ THE SOLVE RAN OUT OF ROOM RATHER THAN CONVERGING — reported, because a
     silhouette that never reached the drawn line is a different result from one
     that did, and only this flag tells them apart. */
  const out = { grow, dev, rim: ring.length, growClamped: !(outer.signed > 0) };
  pre.rungs.set(n, out);
  return out;
}

/**
 * A CLOSED DRAWN OUTLINE, AS A CAGE COARSE ENOUGH TO EDIT BY HAND.
 *
 * `outline` is a flat [x0,y0,...] closed loop in the drawing plane.
 * `opts.subdivide` is this kernel's own Catmull-Clark, passed rather than
 * imported so the cage is measured against the subdivider it will be drawn with.
 * `opts.cache` is a plain object the caller keeps between frames of a drag; it
 * is optional, and passing none costs the full rebuild every call.
 * Every other option is declared in `PUFF_PARAMS` and clamped to it; `faces` is
 * accepted as the older spelling of `density`.
 *
 * Refusals name what a reader would change and none of them throws.
 */
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
    /* HOW MUCH OF THE DRAWING THE RADIAL WARP COULD NOT FOLLOW — see the header.
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
