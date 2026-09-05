// OFFSETREVOLVE — an EXACT inward offset for a solid whose curved faces are
// SURFACES OF REVOLUTION (a Cylinder wall, a Cone wall), together with the
// planar caps those walls meet.
//
// WHY THIS EXISTS, stated against the thing it replaces. kernel/offset.mjs's
// `offsetSurface` moves every control point along the surface's own Greville
// normal. That is exact for a flat surface and honestly approximate for a
// curved one — but for a RATIONAL surface it is not even exact for a
// mathematically developable cylinder, and the reason is worth writing down
// because it is the entire motivation for this module:
//
//   A full circle in this kernel is the standard rational quadratic — four
//   90-degree spans whose control polygon is a SQUARE circumscribing the
//   circle. Half of its control points sit ON the circle (radius R) and half
//   are the off-curve tangent-line corners at radius R*sqrt(2), carrying
//   weight cos(45 deg). The true parallel circle of radius R-d is again that
//   same construction, so ITS corner control points sit at (R-d)*sqrt(2) —
//   i.e. the corner must move inward by d*sqrt(2), NOT by d. `offsetSurface`
//   moves every control point by exactly d along its own normal (radial,
//   here), so the corner lands at R*sqrt(2) - d, short of where an exact
//   parallel circle needs it by d*(sqrt(2)-1) ~= 0.414*d. The result is not
//   a circle at all: its radius oscillates with v. Compounding that, a flat
//   cap's own normal is purely AXIAL, so the offset cap slides along the axis
//   while the offset wall's ends do not move axially at all — the two offset
//   faces end up meeting nowhere near each other, which is precisely why
//   `shellSolid` refuses a curved face by name rather than shipping a shell
//   that is not a valid solid.
//
// THE FIX, and why it is exact rather than better-approximate. For ANY
// surface of revolution, the surface normal at every point lies in that
// point's own MERIDIAN half-plane (the half-plane through the axis
// containing the point): the normal is perpendicular to the circumferential
// tangent, and the circumferential tangent is exactly the direction
// perpendicular to the meridian half-plane. So offsetting the surface by t
// along its own normal moves every point WITHIN its meridian half-plane, by
// t, perpendicular to the generating profile. That is, by definition, the 2D
// offset of the generating profile — and re-revolving the offset profile
// reproduces the offset surface EXACTLY, not approximately, because the
// re-revolve is the same closed-form rational construction that built the
// original (`revolve`, P&T A8.1). An inward-offset cylinder comes back as a
// genuine cylinder of radius R-t; an inward-offset cone comes back as a
// genuine cone of the SAME half-angle.
//
// THE JUNCTION FALLS OUT FOR FREE, which is the real prize. A wall and the
// cap it meets are two segments of ONE meridian profile. Offsetting that
// profile as a single chain puts an exact MITER at their shared corner — the
// intersection of the two offset lines — so the inner wall and the inner cap
// meet at one shared profile point, and therefore (revolved by the same axis,
// the same reference direction and the same knot vector) at one shared
// circle, control point for control point. Nothing is blended and nothing is
// stitched: the junction is computed, exactly, in 2D.
//
// WHAT IS EXACT AND WHAT IS NOT, per the convention offset.mjs and
// offsetcurve.mjs already set:
//   - A profile made of STRAIGHT segments (Cylinder, Cone, and any straight-
//     walled turned shape) offsets EXACTLY. Each segment translates rigidly
//     by t along its own perpendicular, and each corner is the exact
//     intersection of two lines. Verified numerically in test/, not asserted.
//   - A profile with genuinely CURVED segments (a fillet, a bulged vase) is
//     out of representation here, deliberately rather than by omission: a
//     profile in this module is a POINT CHAIN, so there is no way to hand one
//     in and have its error quietly inherited by something described as
//     exact. `offsetCurve2D`'s own honest approximation would apply per curved
//     segment if that were ever added, and the claim would have to weaken to
//     match.
//   - A DOUBLY-CURVED junction (a curved wall meeting a curved wall that is
//     not a surface of revolution about the SAME axis) is a materially larger
//     problem — a real blend surface — and is not attempted here.
//
// SIGN CONVENTION, stated once. A profile's points run in order, and the
// MATERIAL lies on the left of the direction of travel when viewed with the
// meridian plane normal `refDir x axisDir` pointing at you. A POSITIVE
// distance offsets INTO the material (inward), which is what hollowing wants.
// This is the same right-hand rule offsetcurve.mjs's `offsetPerp` already
// uses; the profile builders below emit their points in the order that makes
// it true, and `normalizeMeridianProfile` checks it rather than assuming it.

import { revolve, makeLine, makeArc } from './primitives.mjs';
import { offsetPolyline } from './offsetcurve.mjs';

// ============================================================================
// EXACT REVOLVED SHELL — TWIN BLOCK BEGIN
// ============================================================================
// Everything from here to the matching END marker is hand-pasted into the
// host app's KERNEL section (that app has no module loader). The paste
// is MECHANICAL — drop the two import lines above, drop the six plain-array
// vector helpers this file shares byte-for-byte with that file, drop the
// `export ` keyword — and a gate
// regenerates it that way and requires the app to contain the result
// VERBATIM, so an unmirrored edit to either side fails a test instead of
// silently desyncing.

const EPS = 1e-9;

function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function add3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function scale3(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function dist3(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function norm3(v) {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len < 1e-12) return null;
  return [v[0] / len, v[1] / len, v[2] / len];
}

// The plane the whole offset happens in. `refDir x axisDir` (NOT the other
// order) is the normal for which `cross(n, direction of travel)` points into
// the material for a profile walked base-outward-then-up — see the sign
// convention note in this file's header, and `normalizeMeridianProfile`,
// which verifies it on a real profile instead of trusting the derivation.
export function meridianPlaneNormal(refDir, axisDir) {
  const n = norm3(cross3(refDir, axisDir));
  if (!n) throw new Error('offsetrevolve: refDir and axisDir are parallel, so they span no meridian plane');
  return n;
}

// A profile's own (r, z) coordinates: signed radius along refDir, height
// along axisDir, both measured from axisPoint. Signed rather than absolute
// deliberately — the clip below needs to know that an offset point has
// crossed the axis, which |r| would hide.
export function profileRZ(profile, p) {
  const rel = sub3(p, profile.axisPoint);
  return [dot3(rel, profile.refDir), dot3(rel, profile.axisDir)];
}

// Lift an (r, z) pair back into 3D. Exact inverse of profileRZ for any point
// that genuinely lies in the meridian plane (refDir/axisDir are orthonormal).
export function profileFromRZ(profile, r, z) {
  return add3(profile.axisPoint, add3(scale3(profile.refDir, r), scale3(profile.axisDir, z)));
}

// The orthonormal (axisPoint, axisDir, refDir) frame a meridian half-plane is
// measured in. Orthonormalizes refDir against the axis rather than demanding
// the caller hand over an already-perfect pair — the app's own primitive
// frames are built from picked points and are orthonormal to float precision,
// not exactly.
export function meridianFrame(axisPoint, axis, refDirIn) {
  const axisDir = norm3(axis);
  if (!axisDir) throw new Error('offsetrevolve: axisDir must be a real, nonzero vector');
  if (!refDirIn) throw new Error('offsetrevolve: a profile needs a refDir (the zero-angle reference direction of the revolve)');
  const raw = refDirIn.slice(0, 3);
  const refDir = norm3(sub3(raw, scale3(axisDir, dot3(raw, axisDir))));
  if (!refDir) throw new Error('offsetrevolve: refDir is parallel to the axis, so it names no radial direction');
  return { axisPoint: axisPoint.slice(0, 3), axisDir, refDir };
}

// Validate and canonicalize a meridian profile. Refuses, by name, anything
// this module cannot honestly claim exactness for.
export function normalizeMeridianProfile(profile) {
  const { axisPoint, axisDir, refDir } = meridianFrame(profile.axisPoint, profile.axisDir, profile.refDir);
  const pts = [];
  for (const p of profile.points) {
    const q = p.slice(0, 3);
    if (!pts.length || dist3(pts[pts.length - 1], q) > 1e-9) pts.push(q);
  }
  if (pts.length < 2) throw new Error('offsetrevolve: a meridian profile needs at least 2 distinct points');
  const base = { axisPoint, axisDir, refDir, points: pts };
  const n = meridianPlaneNormal(refDir, axisDir);
  for (let i = 0; i < pts.length; i++) {
    const off = dot3(sub3(pts[i], axisPoint), n);
    if (Math.abs(off) > 1e-6) throw new Error(`offsetrevolve: profile point ${i} is ${off.toFixed(6)} off the meridian plane — a surface of revolution's generating profile must lie in one half-plane through the axis`);
    const [r] = profileRZ(base, pts[i]);
    if (r < -1e-9) throw new Error(`offsetrevolve: profile point ${i} has a negative radius (${r.toFixed(6)}) — a generating profile must stay on one side of the axis`);
  }
  return base;
}

// A Cylinder's own generating profile, walked base-center -> base-rim ->
// top-rim -> top-center. Three segments: bottom cap, wall, top cap. Both ends
// sit ON the axis, which is what makes the offset caps stay full disks rather
// than annuli (see `clipProfileToAxis`).
export function makeCylinderProfile({ center, axis, refDir, radius, height }) {
  if (!(radius > 0)) throw new Error('offsetrevolve: cylinder radius must be positive');
  if (!(height > 0)) throw new Error('offsetrevolve: cylinder height must be positive');
  const base = meridianFrame(center, axis, refDir);
  return normalizeMeridianProfile({
    ...base,
    points: [
      profileFromRZ(base, 0, 0),
      profileFromRZ(base, radius, 0),
      profileFromRZ(base, radius, height),
      profileFromRZ(base, 0, height),
    ],
  });
}

// A Cone's own generating profile, walked base-center -> base-rim -> apex.
// Two segments: the base cap and the slanted wall. The apex is a genuine pole
// (radius exactly 0), which `revolve` already handles.
export function makeConeProfile({ center, axis, refDir, radius, height }) {
  if (!(radius > 0)) throw new Error('offsetrevolve: cone base radius must be positive');
  if (!(height > 0)) throw new Error('offsetrevolve: cone height must be positive');
  const base = meridianFrame(center, axis, refDir);
  return normalizeMeridianProfile({
    ...base,
    points: [
      profileFromRZ(base, 0, 0),
      profileFromRZ(base, radius, 0),
      profileFromRZ(base, 0, height),
    ],
  });
}

// THE AXIS CLIP, and why it is a real geometric rule rather than tidying.
// A generating profile lives in a HALF-plane (r >= 0); the axis is the
// boundary of its own domain, not a boundary of the solid. Offsetting can
// push a point across it — a cone's apex is the clean example: the apex sits
// on the axis, and the inward offset of the slant line at that end leaves the
// half-plane entirely. The correct inner cavity is a cone whose own apex sits
// back ON the axis, lower down, which is exactly where the offset slant line
// re-crosses r = 0. So the offset chain is clipped to r >= 0 and terminated
// exactly at the crossing — never flat-capped at wherever the perpendicular
// happened to land, which would leave a phantom sliver of surface hanging off
// the axis.
export function clipProfileToAxis(profile, points) {
  const rOf = (p) => profileRZ(profile, p)[0];
  const pieces = [];
  let cur = [];
  const pushCur = () => { if (cur.length >= 2) pieces.push(cur); cur = []; };
  for (let i = 0; i + 1 < points.length; i++) {
    const a = points[i], b = points[i + 1];
    const ra = rOf(a), rb = rOf(b);
    const aIn = ra >= -EPS, bIn = rb >= -EPS;
    if (aIn && bIn) {
      if (!cur.length) cur.push(a);
      cur.push(b);
      continue;
    }
    if (!aIn && !bIn) { pushCur(); continue; }
    const t = ra / (ra - rb);                      // the exact r = 0 crossing
    const x = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    if (aIn) { if (!cur.length) cur.push(a); cur.push(x); pushCur(); }
    else { pushCur(); cur.push(x); cur.push(b); }
  }
  pushCur();
  return pieces;
}

// OFFSET THE PROFILE. Reuses `offsetPolyline` (kernel/offsetcurve.mjs) rather
// than re-deriving a second corner-miter implementation: for a straight-
// segment chain with miter joins and no caps its result is EXACT (each edge a
// rigid translation, each corner the exact intersection of the two offset
// lines) and it is already node-tested. Pruning is left ON so a genuinely
// self-crossing request is caught rather than returned; caps are OFF because
// this is one side of an offset, not a closed outline.
//
// `miterLimit` defaults deliberately HIGH here (not offsetcurve's own SVG
// default of 4): a bevel fallback would cut the corner and put the junction
// somewhere other than the exact intersection, which is the one thing this
// module exists to get right. A corner sharp enough to blow past this limit
// is reported (`miterLimitFallbacks`) rather than silently accepted.
export function offsetMeridianProfile(profileIn, distance, opts = {}) {
  const profile = normalizeMeridianProfile(profileIn);
  if (!Number.isFinite(distance)) throw new Error('offsetrevolve: offset distance must be a finite number');
  if (distance === 0) return { ok: true, profile, points: profile.points.map((p) => p.slice()), clippedAtAxis: false, miterLimitFallbacks: 0 };
  const n = meridianPlaneNormal(profile.refDir, profile.axisDir);
  let res;
  try {
    res = offsetPolyline(profile.points, distance, n, {
      closed: false, capStart: 'none', capEnd: 'none',
      join: 'miter', miterLimit: opts.miterLimit ?? 1e6, prune: true,
    });
  } catch (err) {
    return { ok: false, reason: `the offset profile could not be built (${err.message})` };
  }
  if (res.pruned) return { ok: false, reason: 'the offset profile crosses itself — this wall thickness does not fit inside the shape' };
  const pieces = clipProfileToAxis(profile, res.points);
  if (pieces.length === 0) return { ok: false, reason: 'the whole offset profile fell outside the axis half-plane — this wall thickness is larger than the shape itself' };
  if (pieces.length > 1) return { ok: false, reason: `the offset profile leaves and re-enters the axis in ${pieces.length} places, so the cavity is not one connected region — refused rather than guessed at` };
  const points = pieces[0];
  const clippedAtAxis = points.length !== res.points.length
    || dist3(points[0], res.points[0]) > 1e-12
    || dist3(points[points.length - 1], res.points[res.points.length - 1]) > 1e-12;

  // INDEPENDENT VALIDITY, checked geometrically rather than by trusting the
  // corner bookkeeping: every retained edge must still run the same way as
  // the original edge it came from (an over-offset corner reverses an edge
  // before it self-intersects), and the region between the offset chain and
  // the axis must still enclose real area (that region IS the cavity, and by
  // Pappus its area is the cavity's own volume / 2*pi*centroidRadius).
  for (let i = 0; i + 1 < points.length; i++) {
    const d = norm3(sub3(points[i + 1], points[i]));
    if (!d) return { ok: false, reason: 'an offset edge collapsed to zero length' };
    let best = -Infinity;
    for (let j = 0; j + 1 < profile.points.length; j++) {
      const od = norm3(sub3(profile.points[j + 1], profile.points[j]));
      if (od) best = Math.max(best, dot3(d, od));
    }
    if (best < 1e-6) return { ok: false, reason: 'an offset edge reversed direction — this wall thickness is larger than the shape can carry there' };
  }
  const area = profileEnclosedArea(profile, points);
  if (!(area > 1e-12)) return { ok: false, reason: 'the offset profile encloses no area — the cavity has closed up entirely' };
  return { ok: true, profile, points, clippedAtAxis, miterLimitFallbacks: res.miterLimitFallbacks, enclosedArea: area };
}

// Signed area of the region between a profile chain and the axis, closed by
// running back down the axis. Positive for a chain walked in the profile
// order this module defines (material on the left).
export function profileEnclosedArea(profile, points) {
  const rz = points.map((p) => profileRZ(profile, p));
  const loop = rz.slice();
  loop.push([0, rz[rz.length - 1][1]]);
  loop.push([0, rz[0][1]]);
  let a = 0;
  for (let i = 0; i < loop.length; i++) {
    const p = loop[i], q = loop[(i + 1) % loop.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a / 2;
}

// THE SAFE MAXIMUM, computed rather than guessed — the same posture (and the
// same bisection shape) `shellSolid` already uses for the planar case. Every
// failure mode above (edge reversal, self-crossing, the cavity closing to
// nothing, the profile splitting at the axis) is monotone in the requested
// thickness, so bisection converges on the exact largest thickness that still
// leaves a real cavity. For a Cylinder of radius R and height H this comes out
// at min(R, H/2); for a Cone of base radius R, height H and slant L it comes
// out at R*H/(R+L) — both checked against those closed forms in test/, so the
// bisection is verified against analysis rather than against itself.
//
// ⚠ RETURNS null WHEN NOTHING COULD BE MEASURED, NOT 0. The bisection starts at
// lo = 0, so a profile on which EVERY probed thickness fails leaves lo at its
// starting bound — and 0 read back as a measurement says "this shape can hold a
// wall of exactly zero", which a caller then clamps to and builds. It is not a
// measurement at all; it is the absence of one. null says that, and every caller
// has to decide what to do about it instead of multiplying it by a safety factor.
// The floor is the coarser of an absolute 1e-6mm — the same length below which
// `offsetMeridianProfile` already declares an offset edge collapsed — and a
// billionth of the profile's own extent, which takes over on a very large shape.
// Under it the bisection is reporting its own arithmetic rather than a wall.
export function safeInwardOffset(profileIn, opts = {}) {
  const profile = normalizeMeridianProfile(profileIn);
  let extent = 0;
  for (const p of profile.points) {
    const [r, z] = profileRZ(profile, p);
    extent = Math.max(extent, Math.abs(r), Math.abs(z));
  }
  extent = Math.max(extent, 1);
  let hi = extent * 2;
  if (offsetMeridianProfile(profile, hi, opts).ok) return hi; // nothing constrains it
  let lo = 0;
  for (let it = 0; it < 80; it++) {
    const mid = 0.5 * (lo + hi);
    if (offsetMeridianProfile(profile, mid, opts).ok) lo = mid; else hi = mid;
  }
  return lo > Math.max(1e-6, extent * 1e-9) ? lo : null;
}

// Revolve a meridian chain into panels — one full-circle revolved surface per
// straight segment. A segment with BOTH ends on the axis revolves to nothing
// and is dropped (it is a degenerate artifact of the half-plane
// representation, not a face).
//
// ORIENTATION, derived rather than eyeballed. For a segment whose meridian
// direction is d = (alpha, beta) in (r, z), `revolve` produces Su parallel to
// d and Sv parallel to axis x radial, so Su x Sv = (-beta, alpha) in (r, z) —
// which is exactly `cross(meridianPlaneNormal, d)`, the same perpendicular
// offsetcurve.mjs offsets ALONG. In this module's own convention that
// perpendicular points INTO the material of the ORIGINAL solid. So:
//   - an OUTER skin panel must be revolved with its segment REVERSED
//     (`flip: true`) for its normal to point out of the material;
//   - an INNER (offset) skin panel must NOT be flipped, because at the inner
//     surface "into the original material" and "out of the wall, into the
//     cavity" are opposite sides — the same perpendicular that pointed into
//     the solid at radius R points into the CAVITY at radius R-t.
// Both cases therefore end up with normals pointing out of the wall, which is
// what makes an outer/inner pair a consistently wound closed solid rather than
// two coincidentally nested surfaces. (Noted honestly: the app's own Cylinder/
// Cone primitive builds its wall unflipped and so carries the opposite sign
// today; reconciling that is wiring work, not math.)
//
// ARCS, and why a face may be emitted as several panels rather than one.
// `opts.arcs` (default 1 — a single full-circle panel, byte-identical to
// this function's original behavior) splits the 2*pi sweep into that many
// equal panels of the SAME surface. The geometry is unchanged either way:
// a full circle in this kernel already IS four rational quarter spans, so
// asking for four panels hands back exactly those four spans, each an
// exact restriction of the same surface, meeting at bit-identical control
// points (the same `angleStart + k*dtheta` arithmetic on both sides of
// every join). What it buys is TOPOLOGY: a full revolve is closed in its
// own sweep direction, so it has no corner boundary at all — two circles,
// no corners, nothing a corner-welding B-rep builder can weld. Split into
// arcs, every panel is corner-bounded, adjacent panels share real edges,
// and the assembled solid can be handed to a real half-edge topology
// builder and validity checker instead of being reported as unavailable.
export function revolveProfilePanels(profileIn, points, opts = {}) {
  const profile = normalizeMeridianProfile(profileIn);
  const flip = !!opts.flip;
  const uRes = opts.uRes ?? 1;
  const vRes = opts.vRes;
  const arcs = Math.max(1, Math.round(opts.arcs ?? 1));
  const sweep = 2 * Math.PI / arcs;
  const panels = [];
  for (let i = 0; i + 1 < points.length; i++) {
    let a = points[i], b = points[i + 1];
    const [ra] = profileRZ(profile, a);
    const [rb] = profileRZ(profile, b);
    if (Math.abs(ra) < 1e-9 && Math.abs(rb) < 1e-9) continue; // lies on the axis: revolves to nothing
    if (flip) { const t = a; a = b; b = t; }
    for (let k = 0; k < arcs; k++) {
      const srf = revolve(makeLine(a, b), profile.axisPoint, profile.axisDir, k * sweep, sweep);
      const panel = { srf, uRes };
      if (vRes != null) panel.vRes = vRes;
      panels.push(panel);
    }
  }
  return panels;
}

// Intersect two lines given in a profile's own (r, z) plane, each as a point
// plus a UNIT direction. Returns null for a parallel pair rather than a
// point flung out to wherever a near-zero denominator happens to land.
function intersectRZ(p, d, q, e) {
  const den = d[0] * e[1] - d[1] * e[0];
  if (Math.abs(den) < 1e-9) return null;                  // d and e are unit, so this IS |sin(angle)|
  const s = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den;
  return [p[0] + d[0] * s, p[1] + d[1] * s];
}

// SHELL A REVOLVED SOLID. Outer skin = the kept profile revolved; inner skin
// = the offset profile revolved, facing the other way; plus, at each end
// where a face was OPENED, a RIM lip closing the wall's own thickness.
// Returns all three, plus the exact JUNCTION circles (one per interior corner
// of the inner profile — the shared rings where an inner wall meets an inner
// cap) so a caller, and every test, can check the junction directly rather
// than infer it.
//
// OPENING A FACE IS A CHANGE TO THE PROFILE, not a trim applied afterward.
// A "face" here is one SEGMENT of the generating profile (a Cylinder's own
// three: bottom cap, wall, top cap), and opening one drops it from the chain
// before the offset runs. Two things then fall out for free:
//
//   - `offsetPolyline` translates an open chain's END segment rigidly by t
//     perpendicular to ITSELF and miters only the INTERIOR corners, so the
//     inner wall arrives at the opening with the wall thickness intact and
//     nothing to blend.
//   - The inner wall's own end is then EXTENDED (or trimmed) along its own
//     line to meet the OPENED face's own unmoved line. That is exactly the
//     planar `shellSolid`'s rule for an opened face — its plane does not
//     move, so the inner corner lands in it — arrived at here as an ordinary
//     line/line intersection in the meridian plane rather than as a special
//     case. It matters: for a CONE opened at its base, the naive
//     perpendicular offset would put the inner wall's end BELOW the base
//     plane, poking out of the opening it is supposed to stop at.
//
// The lip is then the revolve of the straight line between the outer end
// point and that inner one. Both lie on the OPENED face's own line in one
// meridian half-plane, so the lip is a sub-segment of the very face that was
// removed, and it sweeps the exact annulus between two exact circles on one
// axis. Nothing is stitched and nothing is blended.
//
// LIP ORIENTATION is not eyeballed: the lip lies on the removed face's own
// line, so ordering its two points to run the SAME way that face ran in the
// profile chain and flipping it exactly like an outer panel gives it the
// removed face's own outward normal, by construction. The sign is COMPUTED
// from the two directions below rather than assumed from an example.
//
// Only a CONTIGUOUS run of faces may be kept. Opening a Cylinder's WALL while
// keeping both caps would leave two disks with nothing joining them; that is
// refused by name rather than emitted as a pair of loose faces.
export function shellRevolvedSolid(profileIn, thickness, opts = {}) {
  const full = normalizeMeridianProfile(profileIn);
  if (!Number.isFinite(thickness) || thickness === 0) throw new Error('shellRevolvedSolid: thickness must be a nonzero finite number');
  const segCount = full.points.length - 1;
  const removed = new Set();
  for (const s of (opts.removedSegments || [])) {
    const i = Math.round(s);
    if (!(i >= 0 && i < segCount)) throw new Error(`shellRevolvedSolid: face ${s} is not one of this profile's own ${segCount} faces`);
    removed.add(i);
  }
  if (removed.size >= segCount) throw new Error('shellRevolvedSolid: every face was opened — leave at least one face for a wall');
  let first = 0; while (removed.has(first)) first++;
  let last = segCount - 1; while (removed.has(last)) last--;
  for (let i = first; i <= last; i++) {
    if (removed.has(i)) throw new Error('shellRevolvedSolid: the opened faces are not next to each other along the profile, so what is left is two disconnected walls rather than one — refused rather than guessed at');
  }
  const openStart = first > 0, openEnd = last < segCount - 1;
  const profile = normalizeMeridianProfile({ ...full, points: full.points.slice(first, last + 2) });

  // The inner chain at a given wall thickness, INCLUDING the open-end rule
  // above. Written as one function because the safe maximum has to be
  // measured against the chain that actually gets built, not against a
  // looser proxy that ignores the extension to the opening.
  const buildInner = (t) => {
    const off = offsetMeridianProfile(profile, t, opts);
    if (!off.ok) return null;
    const pts = off.points.map((p) => p.slice());
    if (pts.length < 2) return null;
    const adjustEnd = (endIdx, neighborIdx, removedSegIdx) => {
      const a = profileRZ(profile, pts[endIdx]), b = profileRZ(profile, pts[neighborIdx]);
      let d = [b[0] - a[0], b[1] - a[1]];
      const dl = Math.hypot(d[0], d[1]);
      if (!(dl > 1e-12)) return false;
      d = [d[0] / dl, d[1] / dl];
      const q0 = profileRZ(full, full.points[removedSegIdx]), q1 = profileRZ(full, full.points[removedSegIdx + 1]);
      let e = [q1[0] - q0[0], q1[1] - q0[1]];
      const el = Math.hypot(e[0], e[1]);
      if (!(el > 1e-12)) return false;
      e = [e[0] / el, e[1] / el];
      const x = intersectRZ(a, d, q0, e);
      if (!x || x[0] < -EPS) return false;                 // crossed the axis: no rim left to build
      if ((b[0] - x[0]) * d[0] + (b[1] - x[1]) * d[1] <= 0) return false; // the end segment turned back on itself
      pts[endIdx] = profileFromRZ(profile, x[0], x[1]);
      return true;
    };
    if (openStart && !adjustEnd(0, 1, first - 1)) return null;
    if (openEnd && !adjustEnd(pts.length - 1, pts.length - 2, last + 1)) return null;
    return { pts, off };
  };

  const requested = Math.abs(thickness);
  const safety = opts.clampSafety ?? 0.98;
  let safeMaxDistance = safeInwardOffset(profile, opts);
  if (safeMaxDistance !== null && (openStart || openEnd) && !buildInner(safeMaxDistance * safety)) {
    /* SAME NORMALIZATION AS `safeInwardOffset`'s OWN, AND FOR THE SAME REASON:
       this second bisection also starts at lo = 0, so a profile where no
       thickness at all reaches the opened face leaves it there, and 0 carried
       forward is a fabricated measurement rather than a refusal. */
    const ceiling = safeMaxDistance;
    let lo = 0, hi = safeMaxDistance;
    for (let it = 0; it < 60; it++) { const mid = 0.5 * (lo + hi); if (buildInner(mid)) lo = mid; else hi = mid; }
    safeMaxDistance = lo > Math.max(1e-6, ceiling * 1e-9) ? lo : null;
  }
  if (safeMaxDistance === null) {
    throw new Error('shellRevolvedSolid: the largest safe wall thickness for this shape could not be measured — nothing thicker than a millionth of a millimetre leaves a real cavity inside it, which is not a wall but a second copy of the outer skin. Refused rather than reporting a safe maximum of zero and building it anyway.');
  }
  let applied = requested, clamped = false;
  if (requested > safeMaxDistance * safety) { applied = safeMaxDistance * safety; clamped = true; }
  const built = buildInner(applied);
  if (!built) {
    const why = offsetMeridianProfile(profile, applied, opts);
    throw new Error(`shellRevolvedSolid: no wall of ${requested.toFixed(4)}mm fits inside this shape — ${why.ok ? 'the inner wall cannot reach the face that was opened' : why.reason} (its own safe maximum computes as ${safeMaxDistance.toFixed(4)}mm)`);
  }
  const inner = built.pts;
  const arcs = Math.max(1, Math.round(opts.arcs ?? 1));

  const outerPanels = revolveProfilePanels(profile, profile.points, { ...opts, arcs, flip: true });
  // The inner skin is NOT flipped — see revolveProfilePanels' own orientation
  // note: unflipped means "normal along the perpendicular that pointed into
  // the original solid", which at the offset surface is the direction of the
  // cavity, i.e. out of the wall. Outer and inner therefore face away from
  // each other across the wall thickness.
  const innerPanels = revolveProfilePanels(profile, inner, { ...opts, arcs, flip: false });
  // The lip, ordered to run the same way the face it replaces ran.
  const rimPointsFor = (pOuter, pInner, removedSegIdx) => {
    const a = profileRZ(full, full.points[removedSegIdx]), b = profileRZ(full, full.points[removedSegIdx + 1]);
    const o = profileRZ(profile, pOuter), i2 = profileRZ(profile, pInner);
    const along = (i2[0] - o[0]) * (b[0] - a[0]) + (i2[1] - o[1]) * (b[1] - a[1]);
    return along >= 0 ? [pOuter, pInner] : [pInner, pOuter];
  };
  const rimPanels = [];
  if (openStart) rimPanels.push(...revolveProfilePanels(profile, rimPointsFor(profile.points[0], inner[0], first - 1), { ...opts, arcs, flip: true }));
  if (openEnd) rimPanels.push(...revolveProfilePanels(profile, rimPointsFor(profile.points[profile.points.length - 1], inner[inner.length - 1], last + 1), { ...opts, arcs, flip: true }));

  const junctions = [];
  for (let i = 1; i + 1 < inner.length; i++) {
    const [r, z] = profileRZ(profile, inner[i]);
    junctions.push({ point: inner[i].slice(), radius: r, height: z });
  }
  return {
    outerPanels, innerPanels, rimPanels, panels: [...outerPanels, ...innerPanels, ...rimPanels],
    innerProfile: inner, junctions,
    appliedDistance: applied, clamped, safeMaxDistance,
    clippedAtAxis: built.off.clippedAtAxis, openStart, openEnd,
    outerCount: outerPanels.length, innerCount: innerPanels.length, rimCount: rimPanels.length,
    remainingCount: last - first + 1, removedCount: removed.size, faceCount: segCount,
    exact: true,
  };
}

// ARC MERIDIAN PROFILES — the exact offset for a CIRCULAR generating
// profile (Torus, Sphere), the direct arc-generalization of the point-chain
// case above. A circular arc's own surface normal at every point already
// lies exactly along its own RADIAL direction (a circle's defining
// property), so a positive-distance offset (into the material, this
// module's own sign convention — see the file header) is exactly a RADIUS
// CHANGE: same local center, same angle range, radius' = radius - distance.
// None of the point-chain machinery above (offsetPolyline, the axis clip,
// the interior-corner miter) applies to, or is needed by, a genuine
// circular arc.
//
// SCOPED HONESTLY to what this app's own primitives actually build
// (buildTorusSrf / buildEllipsoidSrf in the app): a Torus's
// meridian is a FULL circle offset from the axis (closed, never touching
// the axis); a Sphere's meridian is a HALF circle centered ON the axis,
// pole to pole (open, both ends already exactly on the axis by
// construction, at r=0, regardless of radius — the same
// cos(+-PI/2)~0 property makeEllipsoidProfile's own header already
// proves). Neither case needs an axis clip: the torus profile never
// approaches the axis at all, and the sphere profile's own two ends stay
// pinned to it at the SAME two angles for any radius, so there is nothing
// to clip and nothing to re-derive a new pole crossing for.
//
// A general ELLIPSE (unequal radii — a real Ellipsoid) is deliberately NOT
// covered here: an ellipse's exact offset is not itself an ellipse (no
// closed form for it), so there is no honest "just shrink the radius"
// shortcut the way there is for a true circle. This module's exactness
// claim stops at circular arcs; Ellipsoid does not get an exact shell.
//
// A MIXED profile (a straight segment joined to a circular arc) is not
// attempted either: no primitive this app actually builds has one — Torus
// and Sphere are each a single, PURE arc — so there is no real case to
// prove a miter against. If a future primitive genuinely needs one, the
// same "offset each segment on its own terms, then intersect the two
// offset curves at their shared corner" principle the point-chain case
// above already uses applies directly: the arc offsets to a concentric
// arc, the line offsets to a parallel line, and the exact intersection of
// a line and a circle (a real, well-known closed form) is the joint —
// deliberately deferred until there is a real, testable case for it.

// An arc profile lives in the SAME (axisPoint, axisDir, refDir) meridian
// frame every point-chain profile above does, but is described by ONE
// circular arc instead of a polyline: a local center (centerR, centerZ) in
// the profile's own (r, z) coordinates, a radius, a start angle and a
// signed sweep (radians). `closed: true` marks a genuine full circle
// (its own two ends coincide, e.g. Torus); false marks a real open arc
// with two distinct ends (e.g. Sphere) — informational only, nothing in
// this module's own math branches on it (a full circle's two ends
// already produce identical points from the trig alone).
export function makeArcMeridianProfile({ axisPoint, axisDir, refDir, centerR, centerZ, radius, angleStart, sweep, closed }) {
  const base = meridianFrame(axisPoint, axisDir, refDir);
  if (!(radius > 0)) throw new Error('offsetrevolve: an arc meridian profile needs a positive radius');
  if (!Number.isFinite(sweep) || sweep === 0) throw new Error('offsetrevolve: an arc meridian profile needs a nonzero sweep');
  if (!Number.isFinite(centerR) || !Number.isFinite(centerZ) || !Number.isFinite(angleStart)) throw new Error('offsetrevolve: an arc meridian profile needs a finite center and start angle');
  // The arc must stay on the r >= 0 side of the axis. r(theta) =
  // centerR + radius*cos(theta) is a single cosine, so its minimum over any
  // sweep sits at an endpoint OR at theta = angleStart +/- PI when that
  // falls within a sweep wide enough to reach it (checked directly rather
  // than sampled — a closed form, not a search) — covers both real cases
  // this module targets (a full circle's true minimum is diametrically
  // opposite its start; a half-circle's minimum is always one of its own
  // two ends).
  const rAt = (theta) => centerR + radius * Math.cos(theta);
  const candidates = [angleStart, angleStart + sweep];
  if (Math.abs(sweep) >= Math.PI - 1e-9) candidates.push(angleStart + Math.PI, angleStart - Math.PI);
  const worst = Math.min(...candidates.map(rAt));
  if (worst < -1e-6) throw new Error(`offsetrevolve: this arc profile crosses the axis (worst radius ${worst.toFixed(6)}) — an arc meridian profile must stay on one side of the axis`);
  return { ...base, centerR, centerZ, radius, angleStart, sweep, closed: !!closed };
}

// THE SAFE MAXIMUM, closed-form. Offsetting a circular arc inward by d
// changes only its radius (radius - d), so the sole way an offset can fail
// is that radius collapsing to (or past) the arc's own local center — the
// safe maximum is exactly the arc's own radius, no bisection needed (unlike
// the point-chain case, where self-intersection/axis-crossing depends on
// the whole chain's shape, not one closed-form number).
export function safeArcInwardOffset(profileIn) {
  return profileIn.radius;
}

// THE EXACT ARC OFFSET. Same center, same angle range, radius - distance —
// see this section's own header for why that is exact, not approximate.
export function offsetArcMeridianProfile(profileIn, distance) {
  const base = meridianFrame(profileIn.axisPoint, profileIn.axisDir, profileIn.refDir);
  if (!Number.isFinite(distance)) throw new Error('offsetrevolve: offset distance must be a finite number');
  const newRadius = profileIn.radius - distance;
  if (!(newRadius > 1e-9)) {
    return { ok: false, reason: `the offset shrinks this arc's own radius to ${newRadius <= 0 ? 'zero or past' : 'nearly zero'} its own local center — this wall thickness does not fit inside the shape` };
  }
  return {
    ok: true,
    profile: { ...base, centerR: profileIn.centerR, centerZ: profileIn.centerZ, radius: newRadius, angleStart: profileIn.angleStart, sweep: profileIn.sweep, closed: profileIn.closed },
  };
}

// REVOLVE AN ARC MERIDIAN PROFILE into a real 3D NURBS surface — reuses
// primitives.mjs's own makeArc + revolve() directly, the SAME construction
// buildTorusSrf/buildEllipsoidSrf already use for the un-offset profile, so
// an offset arc profile becomes a genuine, exact revolve surface with zero
// new revolve math.
//
// ORIENTATION, derived the same way revolveProfilePanels' own comment
// derives it for a straight segment, and reused verbatim: for a curve
// walked with increasing angle, Su is parallel to the arc's own tangent
// direction d = (-sin(theta), cos(theta)) in this profile's own (r, z)
// coordinates, so Su x Sv = cross(meridianPlaneNormal, d) — worked out
// directly, in the same orthonormal (refDir, axisDir, n) frame
// meridianPlaneNormal already builds, this equals
// -(cos(theta)*refDir + sin(theta)*axisDir), i.e. the NEGATIVE of the
// radially-outward direction from the arc's own local center: it points
// TOWARD that center. For BOTH real cases this module targets, "toward the
// arc's own local center" is toward the solid interior (a torus's tube
// center; a sphere's own center) — exactly the same "un-flipped points
// into the material of the original solid" property the point-chain case
// already established, so the identical rule applies: an OUTER skin panel
// needs flip:true (reversed, so its normal points OUT of the material); an
// INNER (offset) skin panel needs flip:false (unflipped — "into the
// original material" and "out of the wall, into the cavity" are the same
// direction at the offset surface). Proven numerically, not just derived,
// in test/offsetrevolve.test.mjs.
export function revolveArcMeridianPanel(profileIn, opts = {}) {
  const flip = !!opts.flip;
  const minSegments = opts.minSegments ?? 1;
  const angleStart = flip ? profileIn.angleStart + profileIn.sweep : profileIn.angleStart;
  const sweep = flip ? -profileIn.sweep : profileIn.sweep;
  const center3D = profileFromRZ(profileIn, profileIn.centerR, profileIn.centerZ);
  const arcCurve = makeArc(center3D, profileIn.refDir, profileIn.axisDir, profileIn.radius, angleStart, sweep, minSegments);
  const srf = revolve(arcCurve, profileIn.axisPoint, profileIn.axisDir, 0, 2 * Math.PI);
  const panel = { srf };
  if (opts.uRes != null) panel.uRes = opts.uRes;
  if (opts.vRes != null) panel.vRes = opts.vRes;
  return panel;
}

// SHELL A REVOLVED SOLID BUILT FROM A PURE ARC PROFILE (Torus, Sphere).
// Outer skin = the original profile revolved, reversed; inner skin = the
// offset profile revolved, unreversed — see revolveArcMeridianPanel's own
// orientation note. NO RIM PANELS, for a real structural reason rather than
// an omission: both real cases this targets are already fully CLOSED
// solids with no boundary anywhere in the meridian direction (a torus's
// meridian is a closed loop; a sphere's meridian touches the axis at both
// ends, and revolve() collapses an axis-touching end to a single pole
// point, which is not a boundary either) — there is no "opened face" for a
// rim to bridge, and none is offered: this builder has no removedSegments
// option at all, unlike shellRevolvedSolid's point-chain case, because a
// single pure arc has no adjacent segment to remove it FROM.
export function shellArcRevolvedSolid(profileIn, thickness, opts = {}) {
  if (!Number.isFinite(thickness) || thickness === 0) throw new Error('shellArcRevolvedSolid: thickness must be a nonzero finite number');
  const requested = Math.abs(thickness);
  const safety = opts.clampSafety ?? 0.98;
  const safeMaxDistance = safeArcInwardOffset(profileIn);
  let applied = requested, clamped = false;
  if (requested > safeMaxDistance * safety) { applied = safeMaxDistance * safety; clamped = true; }
  const off = offsetArcMeridianProfile(profileIn, applied);
  if (!off.ok) throw new Error(`shellArcRevolvedSolid: ${off.reason} (its own safe maximum computes as ${safeMaxDistance.toFixed(4)}mm)`);
  const panelOpts = { minSegments: opts.minSegments, uRes: opts.uRes, vRes: opts.vRes };
  const outerPanels = [revolveArcMeridianPanel(profileIn, { ...panelOpts, flip: true })];
  const innerPanels = [revolveArcMeridianPanel(off.profile, { ...panelOpts, flip: false })];
  return {
    outerPanels, innerPanels, rimPanels: [], panels: [...outerPanels, ...innerPanels],
    innerProfile: off.profile,
    appliedDistance: applied, clamped, safeMaxDistance,
    outerCount: outerPanels.length, innerCount: innerPanels.length, rimCount: 0,
    remainingCount: 1, removedCount: 0, faceCount: 1,
    junctions: [], exact: true,
  };
}
// ============================================================================
// EXACT REVOLVED SHELL — TWIN BLOCK END
// ============================================================================
