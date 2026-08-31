// OFFSET / THICKEN — general surface offset (distinct from Pipe's
// own tube-specific Thick/Hollow). Given a single, already-built, untrimmed
// NURBS surface, produce a genuinely offset copy at a chosen distance along
// the surface's own local normal at every control point.
//
// HONEST FRAMING — this is an APPROXIMATE offset, and that is a real,
// well-known CAD fact, not a shortcut. An EXACT offset of a general NURBS
// surface is, in general, NOT itself an exact NURBS surface — real Rhino's
// own OffsetSrf uses an approximating/refitting technique for exactly this
// reason. This module implements the standard, simplest, most defensible
// approximate-offset: move EVERY control point along the surface's own
// local normal at that control point's own Greville (u,v) parametric
// position (the Greville-abscissa-based normal, reusing the already-proven
// surfacePointAndPartials / Su x Sv machinery). The approximation is EXACT
// only for a truly flat/developable-in-the-non-rational-sense surface,
// where the true offset trivially agrees with the control-point-offset
// (verified as the correctness anchor: a flat planar surface offset by d is
// a parallel plane at distance d, to float precision). For a genuinely
// curved surface the quality degrades with local curvature and offset
// distance — the same honest "approximate, stated plainly, not oversold"
// convention Taper/Twist already use.

import { surfacePointAndPartials } from './surface.mjs';

// The Greville abscissa of each control point in one direction (its
// parameter of maximum influence) — a copy of noise.mjs / pointedit.mjs's
// own grevilleFromKnots. A control point's own offset normal is read at the
// parameter that control point actually governs
// (NOT a per-point independently-computed normal).
function grevilleFromKnots(knots, p, count) {
  const g = [];
  for (let i = 0; i < count; i++) {
    if (p <= 0) { g.push(knots[i] ?? 0); continue; }
    let s = 0;
    for (let k = i + 1; k <= i + p; k++) s += knots[k];
    g.push(s / p);
  }
  return g;
}

// The surface's local unit normal at (u,v) = normalize(Su x Sv), with the
// same pole-nudge honest-degrade as noise.mjs's own surfaceNormalAtParam (a
// Greville abscissa can land on a degenerate pole where the cross product
// collapses; nudge slightly toward the domain interior, and return null
// only if even that fails).
function surfaceNormalAtParam(srf, u, v) {
  const uMin = srf.knotsU[srf.degU], uMax = srf.knotsU[srf.knotsU.length - 1 - srf.degU];
  const vMin = srf.knotsV[srf.degV], vMax = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  const tries = [
    [u, v],
    [u + (u < (uMin + uMax) / 2 ? 1 : -1) * (uMax - uMin) * 1e-4, v + (v < (vMin + vMax) / 2 ? 1 : -1) * (vMax - vMin) * 1e-4],
  ];
  for (const [uu, vv] of tries) {
    const cu = Math.max(uMin, Math.min(uMax, uu)), cv = Math.max(vMin, Math.min(vMax, vv));
    const { su, sv } = surfacePointAndPartials(srf, cu, cv);
    const nx = su[1] * sv[2] - su[2] * sv[1];
    const ny = su[2] * sv[0] - su[0] * sv[2];
    const nz = su[0] * sv[1] - su[1] * sv[0];
    const len = Math.hypot(nx, ny, nz);
    if (len > 1e-9) return [nx / len, ny / len, nz / len];
  }
  return null;
}

// The per-control-point outward unit normal grid, at each control point's
// own Greville (u,v). A pole (null normal) contributes a zero vector — that
// control point simply isn't offset, an honest no-op rather than a NaN.
function normalGrid(srf) {
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  const gU = grevilleFromKnots(srf.knotsU, srf.degU, nu);
  const gV = grevilleFromKnots(srf.knotsV, srf.degV, nv);
  const grid = [];
  for (let i = 0; i < nu; i++) {
    const row = [];
    for (let j = 0; j < nv; j++) {
      const n = surfaceNormalAtParam(srf, gU[i], gV[j]);
      row.push(n || [0, 0, 0]);
    }
    grid.push(row);
  }
  return grid;
}

// SELF-INTERSECTION / DEGENERATE-OFFSET DETECTION — required, not optional.
// A large enough offset distance relative to the surface's own local radius
// of curvature (a concave region, or a small-radius Cylinder/Torus) can make
// the offset surface fold back on itself. This is a real, well-known
// geometric fact, not a bug to silently ignore. Detect it with a cheap,
// honest v1 local proxy: for each pair of ADJACENT control points P_a, P_b,
// the offset edge is e' = (P_b + d*n_b) - (P_a + d*n_a) = e + d*(n_b - n_a),
// so exactly e'.e = |e|^2 + d*a with a = (n_b - n_a).e. e'.e starts strictly
// positive at d=0 (=|e|^2) and, being linear in d, first reaches 0 (the edge
// has CROSSED / reversed relative to its pre-offset order) at |d| = |e|^2/|a|
// in whichever direction shrinks it. The largest safe |d| is the minimum of
// that over every adjacent edge — a real, COMPUTED safe maximum, not a
// guessed constant. Matches the shape of Pipe's own honest auto-clamp-and-
// explain (cornerRadius floored to the tube radius, with a status message
// naming why). A full global self-intersection test is genuinely harder,
// separate scope; this local adjacent-neighbour-crossing check is the honest
// cheap v1 signal.
const SELF_INTERSECT_SAFETY = 0.98; // stay just off the exact degenerate boundary (matches Pipe's own small margin), so the clamped result is provably fold-free, never sitting on the zero-area edge

function safeOffsetMagnitude(srf, normals, signDir) {
  // signDir is +1 or -1 (the sign of the requested distance). Returns the
  // largest |distance| in THAT direction for which no adjacent edge crosses,
  // or Infinity if nothing constrains it (e.g. a flat plane: every
  // n_b - n_a is zero, a is zero, no edge can ever flip).
  const net = srf.ctrlNet, nu = net.length, nv = net[0].length;
  let maxMag = Infinity;
  const consider = (ai, aj, bi, bj) => {
    const Pa = net[ai][aj], Pb = net[bi][bj];
    const na = normals[ai][aj], nb = normals[bi][bj];
    const ex = Pb[0] - Pa[0], ey = Pb[1] - Pa[1], ez = Pb[2] - Pa[2];
    const dnx = nb[0] - na[0], dny = nb[1] - na[1], dnz = nb[2] - na[2];
    const e2 = ex * ex + ey * ey + ez * ez;
    if (e2 < 1e-18) return; // coincident control points (a pole edge) — nothing to cross
    const a = dnx * ex + dny * ey + dnz * ez;
    // e'.e = e2 + d*a. With d = signDir*|d|: e2 + signDir*|d|*a. It reaches 0
    // when signDir*a < 0, at |d| = e2/|a|. If signDir*a >= 0 the edge grows
    // (or is unchanged) in this direction and never crosses.
    if (signDir * a < -1e-15) {
      const crit = e2 / Math.abs(a);
      if (crit < maxMag) maxMag = crit;
    }
  };
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    if (i + 1 < nu) consider(i, j, i + 1, j); // adjacent in U
    if (j + 1 < nv) consider(i, j, i, j + 1); // adjacent in V
  }
  return maxMag === Infinity ? Infinity : maxMag * SELF_INTERSECT_SAFETY;
}

// offsetSurface(srf, distance) -> { surface, appliedDistance, clamped,
// safeMaxDistance }. `surface` is a genuine NURBS surface with the SAME
// degree/knots (and unchanged rational weights) as the input, its control
// net moved along the per-control-point Greville normal. If the requested
// distance would self-intersect by the adjacent-crossing check, it is
// clamped down to the largest value that doesn't (appliedDistance), with
// clamped=true and safeMaxDistance naming the computed safe bound.
export function offsetSurface(srf, distance) {
  if (!Number.isFinite(distance)) throw new Error('offsetSurface: distance must be a finite number');
  const normals = normalGrid(srf);
  let applied = distance, clamped = false, safeMax = Infinity;
  if (distance !== 0) {
    const signDir = distance > 0 ? 1 : -1;
    safeMax = safeOffsetMagnitude(srf, normals, signDir);
    if (Math.abs(distance) > safeMax) { applied = signDir * safeMax; clamped = true; }
  }
  const net = srf.ctrlNet.map((row, i) => row.map((cp, j) => {
    const n = normals[i][j];
    return [cp[0] + n[0] * applied, cp[1] + n[1] * applied, cp[2] + n[2] * applied, cp[3]];
  }));
  return { surface: { ...srf, ctrlNet: net }, appliedDistance: applied, clamped, safeMaxDistance: safeMax };
}

// Reverse a surface's U parametric direction: reverse the control ROW order
// and re-base knotsU onto the same domain (U'_j = a + b - U_{m-j}). The
// geometry is bit-for-bit unchanged; only the parametrization direction (and
// therefore the surface's normal orientation, Su x Sv) flips. Used by
// thickenSolid so the offset cap's normal points the OPPOSITE way from the
// original cap — a real, checkable requirement for a valid closed solid.
export function reverseSurfaceU(srf) {
  const a = srf.knotsU[0], b = srf.knotsU[srf.knotsU.length - 1];
  const newKnotsU = srf.knotsU.map((k) => a + b - k).reverse();
  const newNet = srf.ctrlNet.slice().reverse().map((row) => row.map((cp) => cp.slice()));
  return { ...srf, knotsU: newKnotsU, ctrlNet: newNet };
}

// The closed loop of boundary control points, walked once around the four
// parametric sides (each corner visited exactly once) — the rim's own base
// ring. Returns plain [x,y,z] (weight dropped; the rim panels are flat
// bilinear, weight 1, matching ruledLoftPanels' own convention). EXPORTED so
// shellSolid (below) can build a rim ring at a removed face's own opening
// edge, exactly the way thickenSolid already uses it for its own outer
// boundary — the same operation, applied at each opening's own loop.
export function boundaryLoop(net) {
  const nu = net.length, nv = net[0].length;
  const loop = [];
  const xyz = (cp) => [cp[0], cp[1], cp[2]];
  for (let i = 0; i < nu; i++) loop.push(xyz(net[i][0]));            // side j = 0
  for (let j = 1; j < nv; j++) loop.push(xyz(net[nu - 1][j]));       // side i = last
  for (let i = nu - 2; i >= 0; i--) loop.push(xyz(net[i][nv - 1]));  // side j = last (reversed)
  for (let j = nv - 2; j >= 1; j--) loop.push(xyz(net[0][j]));       // side i = 0 (reversed)
  return loop;
}

// THICKEN — cap the offset result into a real closed solid, reusing the
// exact "wall + rim panels" RuledLoft-container pattern this app already
// uses (Cap, Extrude caps, Pipe Thick). Returns a `panels` array of {srf}
// (the RuledLoft container shape): [ORIGINAL surface, OFFSET surface with
// REVERSED winding, ...RIM panels bridging their shared boundary edges].
// The rim reuses ruledLoftPanels (the same proven "ruled panel between two
// corresponding boundary point rings" technique Cap/RuledLoft use) — direct
// reuse, not new math. Same self-intersection clamp as offsetSurface.
export function thickenSolid(srf, distance, ruledLoftPanelsFn) {
  const off = offsetSurface(srf, distance);
  const original = { ...srf, ctrlNet: srf.ctrlNet.map((row) => row.map((cp) => cp.slice())) };
  const offsetReversed = reverseSurfaceU(off.surface); // reversed so its normal points opposite the original's — a valid closed solid
  const origLoop = boundaryLoop(srf.ctrlNet);
  const offLoop = boundaryLoop(off.surface.ctrlNet);
  // Rim ORDER matters for winding consistency: with the original cap keeping
  // its natural normal and the offset cap reversed, both caps point the SAME
  // way relative to the enclosed volume; the rim must match. Verified
  // numerically (test/offset.test.mjs's divergence-theorem volume check) that
  // ruledLoftPanels(origLoop, offLoop, ...) — original ring as the base row,
  // offset ring as the far row — winds the rim consistently with the two
  // caps (a swapped order left the rim outward while the caps were inward,
  // halving the measured enclosed volume — the real bug this order fixes).
  const rim = ruledLoftPanelsFn(origLoop, offLoop, true).map((s) => ({ srf: s }));
  const panels = [{ srf: original }, { srf: offsetReversed }, ...rim];
  return { panels, appliedDistance: off.appliedDistance, clamped: off.clamped, safeMaxDistance: off.safeMaxDistance };
}

// ============================================================================
// EXACT PLANAR SHELL — TWIN BLOCK BEGIN
// (mirrored VERBATIM into the host app, which has no module loader; a gate
//  regenerates this block from this file and requires the app to contain it)
// ============================================================================
//
// SHELL (Rhino: Shell).
// Hollows a solid into a shell of ONE UNIFORM wall thickness, optionally
// OPENING one or more of its own faces so the interior is reachable.
//
// WHY THIS IS A PLANE PROBLEM, NOT A SURFACE-OFFSET PROBLEM. The first
// version of this command offset each remaining face along its OWN local
// normals, independently of its neighbours. Every inner face therefore
// shrank toward its own centre, so at every inner corner the neighbours
// missed each other — gapping in places, crossing in others — and the rim
// at an opening was built by offsetting the REMOVED face inward, which for
// a box puts the rim inside the plane of the wall beside it, buried and
// invisible. The result looked like a shell and was not a solid.
//
// For a PLANAR face there is nothing to approximate. An inner wall is that
// face's own PLANE pushed inward by the wall thickness, and two adjacent
// inner walls meet exactly where their two offset planes intersect. A
// corner where three faces meet is one point: the intersection of three
// offset planes. So the whole construction reduces to
//
//     for every corner of the solid, solve for the ONE point that lies on
//     every incident face's offset plane,
//
// which is a 3x3 linear system with an exact answer — no blending, no
// tolerance, no iteration. The inner surface built from those points is
// watertight and self-intersection-free BY CONSTRUCTION, because every
// inner face shares its corners with its neighbours rather than computing
// them independently.
//
// A corner touching a face that is being OPENED uses that face's ORIGINAL
// plane instead of its offset one. That single rule is what makes the rim
// correct: the inner wall stops exactly in the opening's own plane, so the
// rim is the flat annulus between the outer opening edge and the inner one
// — a real, visible lip — rather than a band buried inside the wall.
//
// A corner may be incident to fewer than three distinct planes (a point in
// the middle of a face, or on an edge between two). The general answer is
// the MINIMUM-NORM displacement satisfying every incident plane
// constraint, computed by Gram-Schmidt below: for three independent planes
// it is exactly the triple intersection; for two it is the nearest point on
// the offset edge line; for one it is a plain perpendicular offset. Same
// code, no special cases.
//
// CURVED FACES ARE STILL REFUSED BY NAME, not approximated. A cylinder wall
// meeting a flat cap has no exact junction in this kernel (the true offset
// of a general NURBS surface is not itself a NURBS surface — see
// offsetSurface's own header), and the junction blend that would make one
// is real, separate work. A shell that is not a valid solid is worse than
// no shell, so a solid with any CURVED (non-degree-1 or rational) face
// refuses and says so. A MIXED solid (some planar faces, some curved)
// refuses for the same reason and names the curved faces — the curved/
// planar junction is precisely the unsolved case, so meeting it halfway
// would ship exactly the invalid result this rewrite exists to remove.
//
// A WARPED BILINEAR FACE IS A DIFFERENT CASE, and this is where this
// module generalizes past its original planar-only scope. Push-pull
// deliberately allows a single vertex/edge drag to leave a box's own
// adjacent quads non-planar (doc: "a bilinear patch represents the warped
// quad exactly ... refusing would block legitimate direct modeling") — the
// face is still degree-1 x degree-1 and non-rational, still a REAL, EXACT
// NURBS surface, it simply no longer has ONE plane. But it still has a
// TANGENT PLANE at every one of its four corners (spanned by that corner's
// own two adjacent edges — for a bilinear patch this is exact, not an
// approximation: Su and Sv at a corner of S(u,v) are literally the two
// edge vectors meeting there). Offsetting each corner's own tangent plane
// by the wall thickness and intersecting them with its neighbours' offset
// tangent planes is the IDENTICAL Gram-Schmidt corner solve below — the
// planar case is the special case where all four corners share one
// tangent plane (the face's own), so that path is untouched byte-for-byte
// (shellFaceCornerNormals is only ever consulted for a face the decompose
// step below actually found non-planar). What is NOT exact: the true
// offset of a warped bilinear surface is not itself bilinear, so a
// bilinear patch through the four offset corners is a real approximation
// IN THE INTERIOR of a warped face — exact at every corner and edge
// (shared with a neighbour, so still watertight), approximate only in
// between. shellSolid measures that residual directly (interiorApprox on
// its return value) rather than asserting it away.
const SHELL_PLANAR_TOL = 1e-6;   // mm — these control points are coplanar BY CONSTRUCTION or not at all, never within a hand-drawn tolerance (matches the app's own PANEL_FLAT_TOL/SOLID_FACE_TOL)
const SHELL_WELD_TOL = 1e-6;     // mm — same reasoning for "is this the same corner"
const SHELL_CLAMP_SAFETY = 0.999; // stay just off the exact degenerate wall thickness, matching offsetSurface's own SELF_INTERSECT_SAFETY margin

function shellDot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function shellCross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }

// Newell's method — correct for a non-convex loop, unlike a single cross
// product of the first three points, and the same technique the app's own
// capProfileNormal/solidLoopPlane already use for exactly that reason.
// Returns the loop's plane {n (unit, right-hand rule on the loop's own
// winding), c = n.p}, plus its real area, or null for a degenerate loop.
function shellLoopPlane(pts) {
  let nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    nx += (a[1] - b[1]) * (a[2] + b[2]);
    ny += (a[2] - b[2]) * (a[0] + b[0]);
    nz += (a[0] - b[0]) * (a[1] + b[1]);
  }
  const m = Math.hypot(nx, ny, nz);
  if (!(m > 1e-12)) return null;
  const n = [nx / m, ny / m, nz / m];
  return { n, c: shellDot(n, pts[0]), area: m / 2 };
}
// One flat bilinear (degree-1 x degree-1) panel from 4 corners given in
// PERIMETER order — the same panel shape bilinearPanelArr/ruledLoftPanels
// already produce everywhere else in this app, built directly here so this
// block has no dependency to keep in sync. A 3-point loop repeats its last
// corner, which is exactly how a fan triangle is already represented.
function shellFacePanel(loopPts) {
  const p = loopPts.length === 3 ? [loopPts[0], loopPts[1], loopPts[2], loopPts[2]] : loopPts;
  const w = (q) => [q[0], q[1], q[2], 1];
  return { srf: {
    degU: 1, knotsU: [0, 0, 1, 1], degV: 1, knotsV: [0, 0, 1, 1],
    ctrlNet: [[w(p[0]), w(p[3])], [w(p[1]), w(p[2])]],
  } };
}

// Every panel cut into flat quads. A degree-1 x degree-1 surface IS exactly
// the union of the bilinear patches spanned by each 2x2 block of its own
// control net (the defining property of a degree-1 tensor B-spline), so
// this decomposition is lossless, not a tessellation — which is what lets
// an extruded polygon (one degree-1 tube panel with N control columns)
// take the exact path alongside a Box (six 2x2 panels).
// HOW FAR A CURVED FACE BOWS OFF FLAT, as an upper bound.
// The curvature test above is structural (degree, rationality), not a
// threshold, so there is no gap to invert the way a tolerance refusal has.
// The number a student actually needs is different: whether this face is a
// real curve or a nearly-flat one they could rebuild as a plane and shell.
// Measured over the CONTROL NET rather than the surface, deliberately — by
// the convex-hull property the net bounds the surface, so this OVERSTATES the
// true sagitta and never understates it, and the refusal says "at most" for
// exactly that reason. Cheap, and it cannot claim a face is flatter than it is.
function shellFaceFlatnessBound(srf) {
  const pts = [];
  for (const row of srf.ctrlNet) for (const cp of row) pts.push([cp[0], cp[1], cp[2]]);
  if (pts.length < 3) return null;
  const c = [0, 0, 0];
  for (const p of pts) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
  c[0] /= pts.length; c[1] /= pts.length; c[2] /= pts.length;
  // Newell over the net's own boundary ring — robust on a non-convex or
  // near-collinear outline, where a single 3-point cross product degrades.
  const net = srf.ctrlNet, nu = net.length, nv = net[0].length;
  const ring = [];
  for (let a = 0; a < nu; a++) ring.push(net[a][0]);
  for (let b = 1; b < nv; b++) ring.push(net[nu - 1][b]);
  for (let a = nu - 2; a >= 0; a--) ring.push(net[a][nv - 1]);
  for (let b = nv - 2; b >= 1; b--) ring.push(net[0][b]);
  const n = [0, 0, 0];
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i], b = ring[(i + 1) % ring.length];
    n[0] += (a[1] - b[1]) * (a[2] + b[2]);
    n[1] += (a[2] - b[2]) * (a[0] + b[0]);
    n[2] += (a[0] - b[0]) * (a[1] + b[1]);
  }
  const len = Math.hypot(n[0], n[1], n[2]);
  if (!(len > 1e-9)) return null; // a degenerate outline has no plane to bow off
  n[0] /= len; n[1] /= len; n[2] /= len;
  let worst = 0;
  for (const p of pts) worst = Math.max(worst, Math.abs((p[0] - c[0]) * n[0] + (p[1] - c[1]) * n[1] + (p[2] - c[2]) * n[2]));
  return worst;
}
function shellDecomposeToQuads(panels) {
  const quads = [], curved = [], curvature = new Map();
  for (let i = 0; i < panels.length; i++) {
    const srf = panels[i] && panels[i].srf;
    if (!srf || !Array.isArray(srf.ctrlNet) || !srf.ctrlNet.length) continue;
    let rational = false;
    for (const row of srf.ctrlNet) for (const cp of row) if (cp.length > 3 && Math.abs(cp[3] - 1) > 1e-9) rational = true;
    // A rational or higher-degree panel is genuinely curved geometry (a
    // revolve wall, a swept tube) — named, not silently flattened.
    if (srf.degU !== 1 || srf.degV !== 1 || rational) { curved.push(i); curvature.set(i, shellFaceFlatnessBound(srf)); continue; }
    const net = srf.ctrlNet, nu = net.length, nv = net[0].length;
    for (let a = 0; a + 1 < nu; a++) for (let b = 0; b + 1 < nv; b++) {
      const pts = [net[a][b], net[a + 1][b], net[a + 1][b + 1], net[a][b + 1]].map((cp) => [cp[0], cp[1], cp[2]]);
      const pl = shellLoopPlane(pts);
      if (!pl) continue; // a zero-area span (a fan's own collapsed apex row) carries no wall — nothing to shell there
      // A non-planar span is KEPT, not refused — see this module's own
      // header for why a warped bilinear quad still has an exact tangent
      // plane at each of its own four corners. `warped` is read only by
      // shellSolid's own per-face corner-normal choice below; every other
      // consumer of a quad treats it identically either way.
      const warped = !pts.every((p) => Math.abs(shellDot(pl.n, p) - pl.c) <= SHELL_PLANAR_TOL);
      quads.push({ pts, source: i, warped, row: a, col: b, nu, nv });
    }
  }
  return { quads, curved, curvature };
}

// Weld coincident corners into one shared vertex list. Spatial-hashed (the
// cell is three orders of magnitude wider than the tolerance, so a match can
// only ever be in this cell or one of its 26 neighbours) so a rebuilt Box
// with a dense control grid stays cheap.
function shellWeld(quads) {
  const pts = [], cells = new Map();
  const CELL = 1e-3;
  const cellKey = (x, y, z) => `${x}|${y}|${z}`;
  const vertexFor = (p) => {
    const cx = Math.floor(p[0] / CELL), cy = Math.floor(p[1] / CELL), cz = Math.floor(p[2] / CELL);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const bucket = cells.get(cellKey(cx + dx, cy + dy, cz + dz));
      if (!bucket) continue;
      for (const vi of bucket) {
        const q = pts[vi];
        if (Math.abs(q[0] - p[0]) <= SHELL_WELD_TOL && Math.abs(q[1] - p[1]) <= SHELL_WELD_TOL && Math.abs(q[2] - p[2]) <= SHELL_WELD_TOL) return vi;
      }
    }
    const vi = pts.length;
    pts.push([p[0], p[1], p[2]]);
    const k = cellKey(cx, cy, cz);
    if (!cells.has(k)) cells.set(k, []);
    cells.get(k).push(vi);
    return vi;
  };
  // panelGrid[sourceIndex] = { nu, nv, grid } — grid[row][col] is the welded
  // vertex id at that ORIGINAL (undecomposed) macro-panel position, filled in
  // as each sub-quad is welded. This is what lets shellSolid later recognize
  // "this welded vertex sits strictly INSIDE a single flat macro panel's own
  // grid" (row/col away from every panel edge) — the exact case a plain
  // single-plane offset gets geometrically wrong on a REBUILT (dense-grid)
  // flat face; see shellCorrectFlatPanelInteriors' own header for why.
  const panelGrid = new Map();
  const faces = [];
  for (const q of quads) {
    const raw = q.pts.map(vertexFor);
    if (!panelGrid.has(q.source)) {
      panelGrid.set(q.source, { nu: q.nu, nv: q.nv, grid: Array.from({ length: q.nu }, () => new Array(q.nv).fill(null)) });
    }
    const pg = panelGrid.get(q.source).grid;
    pg[q.row][q.col] = raw[0];
    pg[q.row + 1][q.col] = raw[1];
    pg[q.row + 1][q.col + 1] = raw[2];
    pg[q.row][q.col + 1] = raw[3];
    const loop = [];
    for (const v of raw) if (!loop.length || loop[loop.length - 1] !== v) loop.push(v);
    while (loop.length > 1 && loop[0] === loop[loop.length - 1]) loop.pop();
    if (loop.length < 3) continue;                       // collapsed to an edge — no wall to build
    if (new Set(loop).size !== loop.length) continue;     // pinched (a repeated non-adjacent corner) — refused rather than guessed at
    faces.push({ loop, source: q.source, warped: !!q.warped });
  }
  return { pts, faces, panelGrid };
}

// Consistent OUTWARD orientation, derived rather than guessed. Two faces
// sharing an edge agree when they traverse that edge in OPPOSITE directions;
// a breadth-first walk propagates that from one seed face, and the sign of
// the component's own divergence-theorem volume then says whether the whole
// component came out inside-in. This replaces the previous centroid-dot
// heuristic, which is only reliable on a convex solid — "which way is out"
// is a topological fact here, not a proximity guess.
function shellOrientOutward(pts, faces) {
  const ekey = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
  const uses = new Map();
  const faceUses = faces.map(() => []);
  faces.forEach((f, fi) => {
    const l = f.loop;
    for (let i = 0; i < l.length; i++) {
      const u = { fi, a: l[i], b: l[(i + 1) % l.length] };
      const k = ekey(u.a, u.b);
      if (!uses.has(k)) uses.set(k, []);
      uses.get(k).push(u);
      faceUses[fi].push({ u, k });
    }
  });
  let naked = 0, nonManifold = 0;
  for (const [, list] of uses) { if (list.length === 1) naked++; else if (list.length > 2) nonManifold++; }
  if (nonManifold) return { ok: false, reason: 'non-manifold', nonManifold };
  if (naked) return { ok: false, reason: 'open', naked };
  const flip = new Array(faces.length).fill(null);
  const comps = [];
  for (let seed = 0; seed < faces.length; seed++) {
    if (flip[seed] !== null) continue;
    flip[seed] = false;
    const comp = [seed], stack = [seed];
    while (stack.length) {
      const fi = stack.pop();
      for (const { u, k } of faceUses[fi]) {
        const pair = uses.get(k);
        const other = pair[0] === u ? pair[1] : pair[0];
        // this face's ACTUAL traversal of the shared edge, after its own flip
        const fa = flip[fi] ? u.b : u.a, fb = flip[fi] ? u.a : u.b;
        // the neighbour must traverse it the other way round
        const wants = (other.a === fb && other.b === fa) ? false : true;
        if (flip[other.fi] === null) { flip[other.fi] = wants; comp.push(other.fi); stack.push(other.fi); }
        else if (flip[other.fi] !== wants) return { ok: false, reason: 'non-orientable' };
      }
    }
    comps.push(comp);
  }
  const loops = faces.map((f, fi) => (flip[fi] ? f.loop.slice().reverse() : f.loop.slice()));
  // per-component divergence-theorem volume; a negative one is inside-out
  for (const comp of comps) {
    let vol = 0;
    for (const fi of comp) {
      const l = loops[fi], p0 = pts[l[0]];
      for (let i = 1; i + 1 < l.length; i++) vol += shellDot(p0, shellCross(pts[l[i]], pts[l[i + 1]])) / 6;
    }
    if (vol < 0) for (const fi of comp) loops[fi].reverse();
  }
  return { ok: true, loops };
}

// The minimum-norm displacement d with n_i . d = b_i for every incident
// plane, by Gram-Schmidt. Each constraint contributes ONLY in the direction
// orthogonal to everything already satisfied, so nothing added earlier is
// disturbed and no motion is added that no constraint asked for — which is
// what makes rank 3 the exact triple-plane corner, rank 2 the nearest point
// on the offset edge, and rank 1 a plain perpendicular offset, with no
// branch. A repeated plane costs nothing (its residual is already zero); a
// CONTRADICTORY one (the same plane wanted at two different offsets) is
// detected and refused rather than silently satisfying only the last.
function shellSolveInner(v, constraints, maxMove) {
  let d = [0, 0, 0];
  const basis = [];
  for (const { n, b } of constraints) {
    const r = b - shellDot(n, d);
    let w = n.slice();
    for (const e of basis) { const p = shellDot(w, e); w = [w[0] - p * e[0], w[1] - p * e[1], w[2] - p * e[2]]; }
    const wl = Math.hypot(w[0], w[1], w[2]);
    if (wl < 1e-7) { if (Math.abs(r) > 1e-6) return null; continue; }
    const e = [w[0] / wl, w[1] / wl, w[2] / wl];
    const alpha = r / wl;
    d = [d[0] + alpha * e[0], d[1] + alpha * e[1], d[2] + alpha * e[2]];
    basis.push(e);
  }
  // A near-tangential pair of faces makes the offset corner shoot off to a
  // physically meaningless distance. Bounded and refused rather than emitted.
  if (Math.hypot(d[0], d[1], d[2]) > maxMove) return null;
  return [v[0] + d[0], v[1] + d[1], v[2] + d[2]];
}

// THE TANGENT-PLANE GENERALIZATION. Every incident face contributes one
// offset-PLANE constraint per corner (see shellInnerCorners just below) —
// for a planar face that plane is the same at all four corners, but a
// warped bilinear quad has a genuinely different tangent plane at each of
// its own four corners, spanned by that corner's own two adjacent edges
// (exact for a bilinear surface: Su/Sv AT a corner of a bilinear patch
// ARE its two adjacent edge vectors there, no derivative approximation
// involved). Returns one unit normal per loop position, oriented to agree
// with `fallback` (the face's own best-fit Newell normal) so a corner
// whose two edges are nearly collinear — or, at extreme warp, briefly
// wound the "wrong" way — never silently flips which side is outward.
function shellFaceCornerNormals(pts, fallback) {
  const n = pts.length;
  return pts.map((curr, k) => {
    const next = pts[(k + 1) % n], prev = pts[(k + n - 1) % n];
    const eNext = [next[0] - curr[0], next[1] - curr[1], next[2] - curr[2]];
    const ePrev = [prev[0] - curr[0], prev[1] - curr[1], prev[2] - curr[2]];
    const c = shellCross(eNext, ePrev);
    const m = Math.hypot(c[0], c[1], c[2]);
    if (!(m > 1e-9)) return fallback;                     // collinear edges at this corner — no defined tangent, fall back to the face's own average
    let nrm = [c[0] / m, c[1] / m, c[2] / m];
    if (fallback && shellDot(nrm, fallback) < 0) nrm = [-nrm[0], -nrm[1], -nrm[2]];
    return nrm;
  });
}

// Every corner's inner twin at wall thickness t. Constraint per incident
// face: n . x = n . v - t for a KEPT face (its own corner tangent plane
// moves inward), and n . x = n . v for a face being OPENED (the inner
// wall stops in the opening's own corner tangent plane, which is exactly
// what makes the rim a flat lip). `cornerNormals[fi][k]` is the SAME
// value at every k for a planar face (byte-identical to the pre-
// generalization single-plane-per-face version) and the real per-corner
// tangent normal for a warped one — one formula, no branch either way.
function shellInnerCorners(pts, loops, cornerNormals, kept, t, maxMove) {
  const cons = pts.map(() => []);
  loops.forEach((l, fi) => {
    const b = kept[fi] ? -t : 0;
    const cn = cornerNormals[fi];
    l.forEach((v, k) => cons[v].push({ n: cn[k], b }));
  });
  return pts.map((v, i) => shellSolveInner(v, cons[i], maxMove));
}
// Is a wall of thickness t geometrically real on this solid? The test is
// per EDGE, and it is the same one safeOffsetMagnitude already uses one
// direction down: an inner edge that runs OPPOSITE to its own outer edge has
// crossed over it, which is precisely a wall that has eaten through the
// solid and come out the far side. It is exact (an edge reverses at one
// value of t and stays reversed), local, and therefore strictly monotone in
// t — which is what makes the bisection below well-posed rather than a
// search over a region that might not be an interval.
//
// A whole-face normal test is NOT enough and was tried first: pushing a box
// wall past half its width mirrors each face in BOTH of its own in-plane
// axes, which is a 180-degree rotation, so the face's normal comes back
// UNCHANGED and a normal test happily accepts an inside-out solid. The edge
// test catches it because each edge individually reverses.
function shellWallFits(pts, loops, cornerNormals, kept, t, maxMove) {
  const inner = shellInnerCorners(pts, loops, cornerNormals, kept, t, maxMove);
  if (inner.some((p) => !p)) return null;
  for (let fi = 0; fi < loops.length; fi++) {
    if (!kept[fi]) continue;
    const l = loops[fi];
    const ip = l.map((v) => inner[v]);
    const pl = shellLoopPlane(ip);
    if (!pl) return null;                                // inner face collapsed to nothing
    for (let i = 0; i < l.length; i++) {
      const a = pts[l[i]], b = pts[l[(i + 1) % l.length]];
      const ia = inner[l[i]], ib = inner[l[(i + 1) % l.length]];
      const eo = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const ei = [ib[0] - ia[0], ib[1] - ia[1], ib[2] - ia[2]];
      if (shellDot(eo, ei) <= 0) return null;            // this edge turned back on itself
    }
  }
  return inner;
}

// A REBUILT (dense-grid) FLAT PANEL'S OWN NON-CORNER VERTICES — the one
// case the per-vertex corner solve above still gets geometrically wrong,
// and the fix for it. Only a panel's own FOUR TRUE CORNERS ever collect
// enough incident planes for the Gram-Schmidt solve above to be fully
// determined: a corner where 3 faces meet solves all 3 axes; a genuine
// 2x2 (un-rebuilt) Box has ONLY corners, which is exactly why that case is
// already correct (item 7's own "standard 2x2 Box panels are unaffected").
// Rebuild a face to a dense N x N grid and two NEW kinds of point appear
// that the un-rebuilt box never had: an EDGE-INTERIOR point (partway along
// a real box edge, touched by exactly the SAME 2 planes as every other
// point on that edge — correct perpendicular to both, but the direction
// ALONG the edge is left completely unconstrained by a 2-plane solve, so
// it silently keeps its ORIGINAL un-shrunk position instead of sliding
// toward the edge's own two now-correctly-inset endpoints), and a
// genuinely 2D-INTERIOR point (touched by only its own ONE face plane,
// correct perpendicular to it but left at its original in-plane position
// entirely). Both are the same underlying gap — an under-constrained
// per-vertex solve leaving a whole extra degree of freedom exactly where
// it needs to shrink toward the panel's own (already-correct) corners —
// and both have the same fix: since the panel is flat and was built as a
// straight-edged bilinear grid over its own four corners
// (boxPanelsWithResolution / bilinearPanelGridArr's own construction), and
// each of those corners' own true offset is already exact (the 3-plane —
// or fewer, for an opened/boundary case — solve above), the true inset
// surface at EVERY other grid position is exactly the bilinear
// interpolation of those same four corners at that position's own
// original (row,col) grid fraction: an edge-interior point's fraction has
// one coordinate at 0 or 1, so the bilinear formula degenerates to plain
// LINEAR interpolation between that edge's own two corners, which is
// exactly the missing "slide along the edge" component; a 2D-interior
// point gets the full bilinear surface. Not an approximation for this
// case — a bilinear map of a straight-edged planar quad by a per-edge
// AFFINE displacement field (true here: a box's own perpendicular offset
// is literally constant along a whole straight edge) is itself bilinear.
// Panels with no true interior (nu<=2 or nv<=2 — every point already IS a
// corner or lies on a 2-point edge) are untouched, and so is every one of
// the four true corners themselves.
//
// GUARDED to only the case this reasoning actually holds for: a genuinely
// ONE-PLANE macro panel. A single "panel" entry is not always literally one
// flat quad — a closed-profile Extrude/Pipe wall is one degree-1 panel with
// N columns wrapping a WHOLE polygon (several distinct facet planes stitched
// into one control net, its own "row 0" and "row N-1" both landing on the
// SAME physical seam point) — there, "the four corners of the grid" are not
// four distinct real corners at all (two of them coincide at the seam), and
// bilinear interpolation between a degenerate pair collapses the interior to
// nonsense. Guarded against directly: every one of the panel's own OUTER
// (pre-shell) points must lie on one common plane (the same SHELL_PLANAR_TOL
// this module already uses to decide "is this quad warped"), and the four
// corners must be four genuinely DISTINCT welded points — exactly true for a
// rebuilt Box's own macro face, and false for a wrapped multi-facet tube.
function shellCorrectFlatPanelInteriors(inner, outerPts, panelGrid, removed) {
  const fixed = inner.slice();
  for (const [source, { nu, nv, grid }] of panelGrid) {
    if (removed.has(source)) continue;             // an opened face's own interior never feeds any output panel
    if (nu <= 2 && nv <= 2) continue;               // every point already IS one of the four corners — nothing to correct
    const c00 = grid[0][0], c10 = grid[nu - 1][0], c11 = grid[nu - 1][nv - 1], c01 = grid[0][nv - 1];
    if (c00 == null || c10 == null || c11 == null || c01 == null) continue; // a degenerate/skipped corner — leave this panel alone
    if (new Set([c00, c10, c11, c01]).size !== 4) continue; // a wrapped/closed panel's own "corners" coincide — not a real single quad
    const p00 = inner[c00], p10 = inner[c10], p11 = inner[c11], p01 = inner[c01];
    if (!p00 || !p10 || !p11 || !p01) continue;
    const outerLoopPlane = shellLoopPlane([outerPts[c00], outerPts[c10], outerPts[c11], outerPts[c01]]);
    if (!outerLoopPlane) continue;
    let onePlane = true;
    outer: for (let a = 0; a < nu && onePlane; a++) for (let b = 0; b < nv; b++) {
      const vi = grid[a][b];
      if (vi == null) continue;
      if (Math.abs(shellDot(outerLoopPlane.n, outerPts[vi]) - outerLoopPlane.c) > SHELL_PLANAR_TOL) { onePlane = false; break outer; }
    }
    if (!onePlane) continue; // several real facets stitched into one panel (e.g. a wrapped tube) — this correction does not apply
    for (let a = 0; a < nu; a++) {
      const u = a / (nu - 1);
      for (let b = 0; b < nv; b++) {
        const isCorner = (a === 0 || a === nu - 1) && (b === 0 || b === nv - 1);
        if (isCorner) continue;                     // already exact — a true corner's own multi-plane solve, never overwritten
        const vi = grid[a][b];
        if (vi == null) continue;
        const v = b / (nv - 1);
        const w00 = (1 - u) * (1 - v), w10 = u * (1 - v), w11 = u * v, w01 = (1 - u) * v;
        fixed[vi] = [
          w00 * p00[0] + w10 * p10[0] + w11 * p11[0] + w01 * p01[0],
          w00 * p00[1] + w10 * p10[1] + w11 * p11[1] + w01 * p01[1],
          w00 * p00[2] + w10 * p10[2] + w11 * p11[2] + w01 * p01[2],
        ];
      }
    }
  }
  return fixed;
}

export function shellSolid(panels, removedIndices, distance) {
  if (!Array.isArray(panels) || panels.length === 0) throw new Error('shellSolid: needs a panel-based solid (at least one face)');
  if (!Number.isFinite(distance) || distance === 0) throw new Error('shellSolid: thickness must be a nonzero finite number');
  const removed = new Set(removedIndices || []);
  const sourceCount = panels.length;
  let remainingSources = 0;
  for (let i = 0; i < sourceCount; i++) if (!removed.has(i)) remainingSources++;
  if (remainingSources === 0) throw new Error('shellSolid: every face was removed — nothing left to shell (leave at least one face)');

  const dec = shellDecomposeToQuads(panels);
  if (dec.curved.length) {
    // NAMED AND MEASURED. Which faces are curved is not enough to act on —
    // a face bowing 0.02mm is one a student would rebuild flat and shell,
    // and one bowing 40mm is not. The flattest of them is reported because
    // that is the one closest to being fixable.
    const bows = dec.curved.map((i) => dec.curvature.get(i)).filter((v) => v != null);
    const flattest = bows.length ? Math.min(...bows) : null;
    const bowNote = flattest != null
      ? ` The flattest of them bows at most ${flattest.toFixed(3)}mm off its own plane; a face that is genuinely near-flat can be rebuilt as a plane and shelled.`
      : '';
    throw new Error(`shellSolid: ${dec.curved.length} of ${sourceCount} face${sourceCount === 1 ? '' : 's'} ${dec.curved.length === 1 ? 'is' : 'are'} CURVED (face${dec.curved.length === 1 ? '' : 's'} ${dec.curved.join(', ')}). An exact shell offsets each face's own PLANE and meets its neighbours at the intersection; a curved wall meeting a flat cap needs a real junction blend this kernel does not have yet. Refused rather than shipping a shell that is not a valid solid.${bowNote}`);
  }
  if (!dec.quads.length) throw new Error('shellSolid: no face of this solid has any area to shell');

  const welded = shellWeld(dec.quads);
  const oriented = shellOrientOutward(welded.pts, welded.faces);
  if (!oriented.ok) {
    if (oriented.reason === 'open') throw new Error(`shellSolid: this is not a closed solid — ${oriented.naked} edge${oriented.naked === 1 ? '' : 's'} border only one face, so there is no inside to hollow.`);
    if (oriented.reason === 'non-manifold') throw new Error(`shellSolid: ${oriented.nonManifold} edge${oriented.nonManifold === 1 ? '' : 's'} border more than two faces — this is not a manifold solid and has no single wall to build.`);
    throw new Error('shellSolid: this solid cannot be consistently oriented, so "inward" is not defined for it.');
  }
  const loops = oriented.loops;
  // planes[fi] is still every face's own best-fit (Newell) normal — the
  // exact plane for a planar face, a sensible "which way is outward"
  // reference for a warped one (shellFaceCornerNormals' own fallback/sign
  // check) and the degenerate-face guard just below, unchanged from
  // before this generalization.
  const planes = loops.map((l) => shellLoopPlane(l.map((v) => welded.pts[v])));
  if (planes.some((p) => !p)) throw new Error('shellSolid: a face lost its plane after welding — refused rather than guessed at');
  // cornerNormals[fi] is ONE normal repeated at every corner for a planar
  // face (byte-identical to the pre-generalization single-plane path) and
  // a real per-corner TANGENT PLANE normal for a warped one. This is the
  // one new piece of math this generalization adds — everything below it
  // (shellInnerCorners' Gram-Schmidt solve, the bisection, the output
  // assembly) is completely unchanged, just fed a per-corner normal
  // instead of a per-face one.
  const warpedFace = welded.faces.map((f) => !!f.warped);
  const cornerNormals = loops.map((l, fi) => {
    const loopPts = l.map((v) => welded.pts[v]);
    if (!warpedFace[fi]) return loopPts.map(() => planes[fi].n);
    return shellFaceCornerNormals(loopPts, planes[fi].n);
  });
  const kept = welded.faces.map((f) => !removed.has(f.source));
  if (!kept.some(Boolean)) throw new Error('shellSolid: every face was removed — nothing left to shell (leave at least one face)');

  // Scale bound for a runaway corner + the bisection ceiling: the solid's own
  // bounding-box diagonal. A wall can never legitimately be thicker than that.
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of welded.pts) for (let k = 0; k < 3; k++) { if (p[k] < lo[k]) lo[k] = p[k]; if (p[k] > hi[k]) hi[k] = p[k]; }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  const maxMove = diag * 4;

  // THE SAFE MAXIMUM, computed not guessed: bisect for the largest wall
  // thickness whose inner surface still exists and still faces outward. For a
  // Box of half-extent h this converges on exactly h — the wall that closes
  // the cavity to a point — and any request under it is passed through
  // untouched, so an ordinary thickness is never silently altered.
  const requested = Math.abs(distance);
  let safeLo = 0, safeHi = diag;
  if (shellWallFits(welded.pts, loops, cornerNormals, kept, safeHi, maxMove)) { safeLo = safeHi; }
  else {
    for (let it = 0; it < 60; it++) {
      const mid = 0.5 * (safeLo + safeHi);
      if (shellWallFits(welded.pts, loops, cornerNormals, kept, mid, maxMove)) safeLo = mid; else safeHi = mid;
    }
  }
  const safeMaxDistance = safeLo;
  let applied = requested, clamped = false;
  if (requested > safeMaxDistance * SHELL_CLAMP_SAFETY) { applied = safeMaxDistance * SHELL_CLAMP_SAFETY; clamped = true; }
  const solvedInner = shellWallFits(welded.pts, loops, cornerNormals, kept, applied, maxMove);
  if (!solvedInner) throw new Error(`shellSolid: no wall thickness fits inside this solid (its own safe maximum computed as ${safeMaxDistance.toFixed(4)}mm)`);
  // Post-process ONLY — the bisection/validity search above is untouched and
  // still runs against the plain per-vertex solve (a conservative, already-
  // proven check); this correction is applied once, to the FINAL accepted
  // inner array, fixing up exactly the genuinely-interior vertices of a
  // rebuilt flat panel that the per-vertex solve leaves un-inset in-plane.
  const inner = shellCorrectFlatPanelInteriors(solvedInner, welded.pts, welded.panelGrid, removed);

  // OUTPUT ORDER is load-bearing and unchanged from the previous version:
  // [every kept face's OUTER panel, then every kept face's INNER panel in the
  // SAME order, then the rim]. out[i] and out[keptCount + i] are one face's
  // outer/inner pair, which is what makes the wall-thickness check a direct
  // index lookup rather than a search.
  const keptIdx = [];
  for (let fi = 0; fi < loops.length; fi++) if (kept[fi]) keptIdx.push(fi);
  const out = [];
  for (const fi of keptIdx) out.push(shellFacePanel(loops[fi].map((v) => welded.pts[v])));
  // The inner twin is the same loop REVERSED, so its normal points back at
  // the outer face across the wall — a consistently wound closed wall.
  for (const fi of keptIdx) out.push(shellFacePanel(loops[fi].slice().reverse().map((v) => inner[v])));
  // THE RIM. Every edge that borders exactly one KEPT face is an opening
  // boundary, and each one gets the quad bridging its outer edge to the same
  // edge's inner twin. Because that inner twin lies in the opening's OWN
  // plane, this rim is the flat lip a real shell has — and because it is
  // built from the shared welded edge, it meets both the outer wall and the
  // inner wall exactly, with nothing left to close.
  const ekey = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
  const keptEdgeCount = new Map();
  for (const fi of keptIdx) { const l = loops[fi]; for (let i = 0; i < l.length; i++) { const k = ekey(l[i], l[(i + 1) % l.length]); keptEdgeCount.set(k, (keptEdgeCount.get(k) || 0) + 1); } }
  let rimCount = 0;
  for (const fi of keptIdx) {
    const l = loops[fi];
    for (let i = 0; i < l.length; i++) {
      const a = l[i], b = l[(i + 1) % l.length];
      if (keptEdgeCount.get(ekey(a, b)) !== 1) continue;
      // wound b -> a so the rim traverses the shared edge opposite to the
      // outer face, the same convention that keeps two ordinary neighbours
      // consistent
      out.push(shellFacePanel([welded.pts[b], welded.pts[a], inner[a], inner[b]]));
      rimCount++;
    }
  }

  // THE HONEST NUMBER. Exact at every corner (and therefore every shared
  // edge — the whole solid is still watertight, this is not a validity
  // concern) is proven by construction above; the INTERIOR of a warped
  // face is where a bilinear patch through four offset corners can only
  // approximate the true offset surface. Measured here, not asserted:
  // sample each warped face's own outer/inner bilinear patches away from
  // the corners (a 3x3 interior grid; the four corners are excluded
  // because those are already exact) and report the worst deviation from
  // the nominal wall thickness, in mm and as a fraction of it.
  const warpedIdx = keptIdx.filter((fi) => warpedFace[fi]);
  let interiorApprox = null;
  if (warpedIdx.length) {
    const lerp3 = (a, b, s) => [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s];
    // loop order is P0(0,0) P1(1,0) P2(1,1) P3(0,1) — see shellDecomposeToQuads.
    const bilinearAt = (p, u, v) => lerp3(lerp3(p[0], p[1], u), lerp3(p[3], p[2], u), v);
    let worstAbs = 0, worstRel = 0;
    for (const fi of warpedIdx) {
      const l = loops[fi];
      const outer = l.map((v) => welded.pts[v]);
      const innerQ = l.map((v) => inner[v]);
      for (const u of [0.25, 0.5, 0.75]) for (const v of [0.25, 0.5, 0.75]) {
        const po = bilinearAt(outer, u, v), pi = bilinearAt(innerQ, u, v);
        const d = Math.hypot(po[0] - pi[0], po[1] - pi[1], po[2] - pi[2]);
        const absErr = Math.abs(d - applied);
        if (absErr > worstAbs) worstAbs = absErr;
        const rel = applied > 0 ? absErr / applied : 0;
        if (rel > worstRel) worstRel = rel;
      }
    }
    interiorApprox = { warpedFaceCount: warpedIdx.length, worstAbsoluteError: worstAbs, worstRelativeError: worstRel };
  }

  return {
    panels: out, appliedDistance: applied, clamped, safeMaxDistance,
    remainingCount: keptIdx.length, removedCount: sourceCount - remainingSources,
    outerCount: keptIdx.length, innerCount: keptIdx.length, rimCount,
    faceCount: loops.length, exact: warpedIdx.length === 0,
    warpedFaceCount: warpedIdx.length, interiorApprox,
  };
}
// ============================================================================
// EXACT PLANAR SHELL — TWIN BLOCK END
// ============================================================================
