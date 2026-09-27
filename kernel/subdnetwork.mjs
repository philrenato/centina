// SubD pipe network — junction detection and cage assembly for a SuperB
// pipe network, under rules D1-D4 below.
//
// Two functions. `detectPipeJunctions` answers a question purely about
// curves — where do these rails meet, and which of them therefore belong to
// the same object — without building a single vertex. `subdPipeNetwork`
// consumes that answer and builds cages. Keeping them apart lets the app
// show how many junctions it found, and of what arity, before committing to
// any geometry, and makes the detection half testable against curve facts
// alone.
//
// D1 — A T-junction is a junction. A rail whose end lands on another rail's
// interior is not a near miss to be ignored: detection splits the host
// there, so the touch point becomes an endpoint of two host pieces and the
// meeting is an ordinary N-way one from then on. Every split goes through
// `extractSubCurve` (knots.mjs), and the caller's own rails are never
// touched — every returned rail is a fresh curve, split or not.
//
// D2 — One cage per connected group. Welded arms share vertices, so a
// connected network is one cage; two runs that never meet stay two.
//
// D3 — Welding is an option, default on. With it off the result is N
// independent tubes and zero junctions, built from the caller's own rails
// with no splitting and no insetting at all.
//
// D4 — The weld tolerance is radius-relative: `radius * weldFraction`,
// default fraction 0.5. At half the radius a branch's own endpoint is already
// inside the host tube's solid, so the two tubes are visibly touching and a
// user expects them to weld; past a full radius they do not touch at all and
// welding them would bridge a visible gap. Half the radius is the middle of
// the band where "these are touching" is unambiguous, and it scales — a
// 500mm pipe welds within 250mm, a 0.5mm pipe within 0.25mm. It is not
// JOIN_TOLERANCE (0.001mm): the app has endpoint snapping but no on-curve
// snap, so a user can land a T exactly on a rail's endpoint and essentially
// never on its interior, and an exact tolerance would make D1 unreachable in
// practice.
//
// The inset is derived from the junction's own tightest angle, not fixed.
// `bridgeClosedRimsHub` attaches its hub directly to the rims it is given
// and refuses rims that are effectively coincident, so the arms must be
// pulled back along their own rails first — that pull-back is this file's
// job, and the thing subdedit.mjs's HUB_INSET (0.62) generalizes to. The
// number transfers; the mechanism does not. HUB_INSET lerps a rim toward
// the hub center; a pipe network needs a translation along each rail, so
// the tube is shorter rather than squashed sideways. And a single fraction
// cannot be right at every angle: two cylinders of radius r whose axes cross
// at angle theta interpenetrate out to exactly r/tan(theta/2) from the
// crossing point, which is r at 90 degrees and grows without bound as the
// arms close up. So the inset is
// `radius * max(hubInsetFraction, 1/tan(thetaMin/2))` — the exact end of
// the interpenetration region, with 0.62 as a floor for wide angles. A Y of
// three arms at 120 degrees needs 1/tan(60) = 0.577, so 0.62 is close to the
// natural value for the commonest junction.

import { add, sub, scale, dot, cross, length, normalize } from './vec3.mjs';
import {
  curvePoint,
  rationalCurveDerivs,
  closestPointOnCurve,
  buildArcLengthTable,
  paramAtArcLength,
  concatRailsAtJunction,
} from './curve.mjs';
import { extractSubCurve } from './knots.mjs';
import { pipeSafeTubeRadius } from './sweep.mjs';
import { subdPipeCage } from './subdpipe.mjs';
import { bridgeClosedRimsHub, convexHullFaces } from './subdedit.mjs';

// See the header: half the tube radius is the middle of the band in which
// two tubes are unambiguously touching on screen.
export const PIPE_NETWORK_WELD_FRACTION = 0.5;
// HUB_INSET's value, used as the wide-angle floor.
export const PIPE_NETWORK_HUB_INSET_FRACTION = 0.62;
// Below this the required inset runs away (1/tan(7.5deg) is already 7.6
// radii) and the "junction" is two nearly-parallel tubes running
// alongside each other. Refused by name rather than silently demanding a
// pull-back longer than any rail in an ordinary model.
export const PIPE_NETWORK_MIN_ARM_ANGLE = Math.PI / 12; // 15 degrees
// How far out of one plane a junction's arms may sit before the
// single-plane hub cannot be trusted to order them. Measured as the
// worst arm's own out-of-plane offset against the spread of the arms about
// their mean, so it is a shape ratio, not a length.
export const PIPE_NETWORK_PLANARITY_TOLERANCE = 0.25;
// A hull junction needs every rim to be a facet of the hull of all of them.
// A rim at inset a on its arm clears a neighboring rim at angle theta
// exactly when a >= r cot(theta/2) — the same bound the interpenetration
// inset already meets, so at that inset the two rims touch the same
// supporting plane and the hull has a tie. The hull's inset is that bound
// times this margin, so each rim is a facet with room to spare.
export const PIPE_NETWORK_HULL_INSET_MARGIN = 1.3;
// With fitRadius, the share of a rail its two pull-backs may take; the rest
// is tube.
export const PIPE_NETWORK_FIT_HEADROOM = 0.9;
// How many times a hull junction whose rims are not all facets is pulled
// back a quarter further before it is built as it stands (and the hull
// names the rim that still fails).
export const PIPE_NETWORK_HULL_ROUNDS = 4;
// With fitRadius, a built cage is checked for faces passing through each
// other; the tubes involved (a joint's arms, or a tube passing a joint it is
// not part of) are thinned by this factor and the component rebuilt, at most
// PIPE_NETWORK_CLEAR_ROUNDS times. A tube bending back inside a joint's
// pull-back, or a chain kinking beside a joint, is invisible to the per-joint
// inset, which only sees each arm's own end direction.
export const PIPE_NETWORK_CLEAR_THIN = 0.85;
export const PIPE_NETWORK_CLEAR_ROUNDS = 6;

// Small curve helpers

function cloneCurve(c) {
  return { degree: c.degree, knots: c.knots.slice(), ctrlPts: c.ctrlPts.map((p) => p.slice()) };
}
function domainOf(c) {
  return [c.knots[0], c.knots[c.knots.length - 1]];
}
function endParam(c, which) {
  const [a, b] = domainOf(c);
  return which === 'start' ? a : b;
}
function endPointOf(c, which) {
  return curvePoint(c, endParam(c, which));
}
// The unit direction pointing into the rail from one of its own ends —
// the direction an arm leaves its junction along. At `start` that is the
// curve's own travel direction; at `end` it is the reverse, because travel
// there points out of the rail rather than into it.
function inwardDirectionAt(c, which) {
  const u = endParam(c, which);
  const [, C1] = rationalCurveDerivs(c, u, 1);
  const L = length(C1);
  if (!(L > 0)) return null;
  const t = scale(C1, 1 / L);
  return which === 'start' ? t : scale(t, -1);
}
function otherEnd(which) {
  return which === 'start' ? 'end' : 'start';
}

function makeUnionFind(n) {
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i) => {
    while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; }
    return i;
  };
  const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb; };
  return { find, union };
}

// Lexicographic point compare, used only to put results in a deterministic
// order. Junctions are distinct points by construction (two junctions
// within tolerance of each other would have clustered into one), so this
// never has to break a real tie.
function comparePoints(a, b) {
  return a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
}

// Detection
//
// Returns
//   {
//     ok: true,
//     tolerance,
//     rails:        [curve]        — post-split, always fresh objects
//     railSources:  [inputIndex]   — which input rail each came from
//     splits:       [{ rail, params }]
//     junctions:    [{ point, arity, arms: [{ rail, end }] }]
//     components:   [{ rails: [i], sourceRails: [i], junctions: [ji] }]
//   }
// or { ok: false, reason } for an ambiguous input.
export function detectPipeJunctions(rails, opts = {}) {
  if (!Array.isArray(rails) || rails.length === 0) throw new Error('detectPipeJunctions: rails must be a non-empty array of curves');
  rails.forEach((r, i) => {
    if (!r || !Array.isArray(r.ctrlPts) || !Array.isArray(r.knots) || r.ctrlPts.length < 2) {
      throw new Error(`detectPipeJunctions: rail ${i} is not a curve with at least 2 control points`);
    }
  });
  const radius = opts.radius ?? 5;
  if (!(radius > 0)) throw new Error(`detectPipeJunctions: radius must be positive (got ${radius})`);
  const weldFraction = opts.weldFraction ?? PIPE_NETWORK_WELD_FRACTION;
  if (!(weldFraction > 0)) throw new Error(`detectPipeJunctions: weldFraction must be positive (got ${weldFraction})`);
  const tolerance = opts.tolerance ?? radius * weldFraction;
  if (!(tolerance > 0)) throw new Error(`detectPipeJunctions: tolerance must be positive (got ${tolerance})`);

  // T-junctions. Every rail end is tested against every other rail's body.
  // A hit within tolerance of that host's own endpoint is not a T at all —
  // it is an ordinary endpoint meeting, which the clustering below picks up
  // without any split — so only interior hits are recorded here.
  // A curve lies in the box of its control points (positive weights), so a
  // rail whose box, grown by the tolerance, misses an end cannot hold that
  // end: the closest-point search is spent only where it can answer.
  const boxes = rails.map((r) => {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const q of r.ctrlPts) {
      const w = q.length > 3 ? q[3] : 1;
      for (let k = 0; k < 3; k++) { const v = q[k] / w; if (v < lo[k]) lo[k] = v; if (v > hi[k]) hi[k] = v; }
    }
    return { lo: lo.map((v) => v - tolerance), hi: hi.map((v) => v + tolerance) };
  });
  const inBox = (P, b) => P[0] >= b.lo[0] && P[0] <= b.hi[0] && P[1] >= b.lo[1] && P[1] <= b.hi[1] && P[2] >= b.lo[2] && P[2] <= b.hi[2];
  const splitParams = rails.map(() => []);
  for (let bi = 0; bi < rails.length; bi++) {
    for (const which of ['start', 'end']) {
      const P = endPointOf(rails[bi], which);
      for (let hi = 0; hi < rails.length; hi++) {
        if (hi === bi) continue; // a rail touching its own interior is a closed/self-crossing rail, a different problem
        if (!inBox(P, boxes[hi])) continue;
        const host = rails[hi];
        const near = closestPointOnCurve(host, P);
        if (!(near.distance <= tolerance)) continue;
        const hs = endPointOf(host, 'start'), he = endPointOf(host, 'end');
        if (length(sub(near.point, hs)) <= tolerance || length(sub(near.point, he)) <= tolerance) continue;
        if (near.ambiguous) {
          return {
            ok: false,
            reason: `rail ${bi}'s ${which} lands on rail ${hi} in two genuinely different places at once (that rail loops back near itself there), so there is no single point to split it at`,
          };
        }
        splitParams[hi].push(near.u);
      }
    }
  }

  // Split. Several branches can land on one host, and two of them can land
  // in the same place; merging by 3D distance rather than by parameter keeps
  // that correct on an unevenly parametrized rail.
  const outRails = [];
  const railSources = [];
  const splits = [];
  for (let i = 0; i < rails.length; i++) {
    const src = rails[i];
    const [uMin, uMax] = domainOf(src);
    const sorted = splitParams[i].slice().sort((a, b) => a - b);
    const merged = [];
    for (const u of sorted) {
      if (merged.length && length(sub(curvePoint(src, u), curvePoint(src, merged[merged.length - 1]))) <= tolerance) continue;
      merged.push(u);
    }
    if (!merged.length) {
      outRails.push(cloneCurve(src));
      railSources.push(i);
      continue;
    }
    splits.push({ rail: i, params: merged.slice() });
    const bounds = [uMin, ...merged, uMax];
    for (let k = 0; k + 1 < bounds.length; k++) {
      outRails.push(extractSubCurve(src, bounds[k], bounds[k + 1]));
      railSources.push(i);
    }
  }

  // Endpoint clustering. Every endpoint of every (post-split) rail, grouped
  // by proximity. A cluster of 2+ is a junction.
  const eps = [];
  outRails.forEach((c, i) => {
    eps.push({ rail: i, end: 'start', pt: endPointOf(c, 'start') });
    eps.push({ rail: i, end: 'end', pt: endPointOf(c, 'end') });
  });
  // Pairs within tolerance, found through a hash of cells one tolerance
  // wide: only the 27 cells round an endpoint can hold a partner.
  const uf = makeUnionFind(eps.length);
  const cellOf = (p) => p.map((v) => Math.floor(v / tolerance));
  const cells = new Map();
  eps.forEach((e, i) => {
    const c = cellOf(e.pt);
    const key = c.join(',');
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key).push(i);
  });
  for (let i = 0; i < eps.length; i++) {
    const c = cellOf(eps[i].pt);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const list = cells.get(`${c[0] + dx},${c[1] + dy},${c[2] + dz}`);
      if (!list) continue;
      for (const j of list) if (j > i && length(sub(eps[i].pt, eps[j].pt)) <= tolerance) uf.union(i, j);
    }
  }
  const clusters = new Map();
  for (let i = 0; i < eps.length; i++) {
    const r = uf.find(i);
    if (!clusters.has(r)) clusters.set(r, []);
    clusters.get(r).push(i);
  }
  const junctions = [];
  for (const members of clusters.values()) {
    if (members.length < 2) continue;
    const arms = members
      .map((i) => ({ rail: eps[i].rail, end: eps[i].end }))
      .sort((a, b) => a.rail - b.rail || (a.end === b.end ? 0 : a.end === 'start' ? -1 : 1));
    // The mean of the member endpoints, summed in a fixed (sorted) order so
    // the same geometry gives the same number whatever order the rails
    // arrived in.
    const pts = members.slice().sort((a, b) => comparePoints(eps[a].pt, eps[b].pt)).map((i) => eps[i].pt);
    const point = scale(pts.reduce((acc, p) => add(acc, p), [0, 0, 0]), 1 / pts.length);
    junctions.push({ point, arity: arms.length, arms });
  }
  junctions.sort((a, b) => comparePoints(a.point, b.point));

  // Components.
  const cuf = makeUnionFind(outRails.length);
  for (const J of junctions) for (let k = 1; k < J.arms.length; k++) cuf.union(J.arms[0].rail, J.arms[k].rail);
  const byRoot = new Map();
  for (let i = 0; i < outRails.length; i++) {
    const r = cuf.find(i);
    if (!byRoot.has(r)) byRoot.set(r, []);
    byRoot.get(r).push(i);
  }
  const components = [...byRoot.values()].map((railIdxs) => {
    const set = new Set(railIdxs);
    return {
      rails: railIdxs.slice().sort((a, b) => a - b),
      sourceRails: [...new Set(railIdxs.map((i) => railSources[i]))].sort((a, b) => a - b),
      junctions: junctions.map((_, ji) => ji).filter((ji) => set.has(junctions[ji].arms[0].rail)),
    };
  });
  components.sort((a, b) => a.sourceRails[0] - b.sourceRails[0]);

  return { ok: true, tolerance, rails: outRails, railSources, splits, junctions, components };
}

// Assembly

function mergeCage(target, cage) {
  const offset = target.vertices.length;
  for (const v of cage.vertices) target.vertices.push(v.slice());
  for (const f of cage.faces) target.faces.push(f.map((i) => i + offset));
  for (const [k, w] of Object.entries(cage.creases || {})) {
    const [a, b] = k.split('_').map(Number);
    const A = a + offset, B = b + offset;
    target.creases[A < B ? `${A}_${B}` : `${B}_${A}`] = w;
  }
  return offset;
}

// The plane residual of a set of points about their own mean, as a
// fraction of how far they spread. This is the quantity junctionPlaneOrder
// needs to be small: it fits one plane through the rim centers and orders
// them by angle in it, which only means anything if they lie in a plane.
// Three points always do, so this is only asked of N >= 4.
function planarityResidual(pts) {
  const centre = scale(pts.reduce((acc, p) => add(acc, p), [0, 0, 0]), 1 / pts.length);
  let nrm = [0, 0, 0], best = 0;
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const c = cross(sub(pts[i], centre), sub(pts[j], centre));
      const cl = length(c);
      if (cl > best) { best = cl; nrm = c; }
    }
  }
  if (!(best > 0)) return 0; // every point on one line through the center: no plane is picked out, and no arm is out of it
  const n = normalize(nrm);
  const spread = pts.reduce((acc, p) => acc + length(sub(p, centre)), 0) / pts.length;
  if (!(spread > 0)) return 0;
  return Math.max(...pts.map((p) => Math.abs(dot(sub(p, centre), n)))) / spread;
}

// Builds the network.
//
//   radius, facets, segments, crease — passed straight through to
//     subdPipeCage; one shared value across the whole network, which is
//     what makes every rim's vertex count match for free (and
//     bridgeClosedRimsHub's hardest precondition).
//   capStart / capEnd — the network's free ends only. A junction-facing end
//     is always built with cap 'none', because a 'round' cap returns an
//     empty rim (the ring under a dome is interior) and there would be
//     nothing left to weld.
//   weld — D3. Default true.
//   weldFraction / tolerance — D4.
//   hubInsetFraction — the wide-angle floor on the pull-back, see header.
//   junctionCrease — crease weight written onto the welded rims. 0 by
//     default: a soft junction is the purpose of the SuperB expression.
//   clampRadiusToBend — reuse Pipe's own self-intersection clamp so a rail
//     tighter than the tube cannot silently swallow itself. On by default.
//
// Returns { ok: true, cages, junctions, junctionCounts, ... } or
// { ok: false, reason } for a refusal the user can act on.
export function subdPipeNetwork(rails, opts = {}) {
  if (!Array.isArray(rails) || rails.length === 0) throw new Error('subdPipeNetwork: rails must be a non-empty array of curves');
  const requestedRadius = opts.radius ?? 5;
  if (!(requestedRadius > 0)) throw new Error(`subdPipeNetwork: radius must be positive (got ${requestedRadius})`);
  const facets = Math.max(3, Math.round(opts.facets ?? 8));
  const segments = Math.max(1, Math.round(opts.segments ?? 4));
  const capStart = opts.capStart ?? 'none';
  const capEnd = opts.capEnd ?? 'none';
  const crease = opts.crease ?? 0;
  const junctionCrease = opts.junctionCrease ?? 0;
  // 'auto': the ring hub where the arms lie round one plane, the hull where
  // they do not. 'hull': the hull at every junction. 'ring': the ring only,
  // and a junction out of one plane is refused.
  const junction = opts.junction ?? 'auto';
  const weld = opts.weld !== false;
  const hubInsetFraction = opts.hubInsetFraction ?? PIPE_NETWORK_HUB_INSET_FRACTION;

  // Radius clamp. subdPipeCage has no bend-radius guard of its own; the
  // NURBS side's does, and it is a pure curve fact, so it is reused. One
  // shared radius means the tightest rail in the network governs.
  let radius = requestedRadius;
  let radiusClamp = { clamped: false, requested: requestedRadius, radius, safeMax: Infinity, rail: null };
  // With fitRadius each rail is held to its own bend, so one tight rail thins
  // itself and not the network.
  let railSafe = null;
  if (opts.fitRadius && opts.clampRadiusToBend !== false) {
    railSafe = rails.map((r) => pipeSafeTubeRadius(r, requestedRadius).radius);
    const bent = railSafe.filter((x) => x < requestedRadius).length;
    radiusClamp = { clamped: bent > 0, requested: requestedRadius, radius, safeMax: Math.min(...railSafe), rail: null, perRail: true, bendThinned: bent };
  } else if (opts.clampRadiusToBend !== false) {
    let worst = null;
    rails.forEach((r, i) => {
      const s = pipeSafeTubeRadius(r, requestedRadius);
      if (!worst || s.safeMax < worst.safeMax) worst = { ...s, rail: i };
    });
    if (worst && worst.clamped) {
      radius = worst.radius;
      radiusClamp = { clamped: true, requested: requestedRadius, radius, safeMax: worst.safeMax, rail: worst.rail, minBendRadius: worst.minBendRadius };
    } else if (worst) {
      radiusClamp = { clamped: false, requested: requestedRadius, radius, safeMax: worst.safeMax, rail: worst.rail, minBendRadius: worst.minBendRadius };
    }
  }

  const buildTube = (curve, cs, ce, rr = radius) => subdPipeCage(curve, { radius: rr, facets, segments, capStart: cs, capEnd: ce, crease });

  // D3, welding off. N independent tubes, from the caller's own rails,
  // with no detection, no splitting and no insetting at all.
  if (!weld) {
    const cages = rails.map((r, i) => {
      const cage = buildTube(cloneCurve(r), capStart, capEnd);
      return { vertices: cage.vertices, faces: cage.faces, creases: cage.creases, sourceRails: [i], railCount: 1, junctions: [], freeEndCount: 2 };
    });
    return { ok: true, welded: false, cages, junctions: [], junctionCounts: {}, junctionTotal: 0, splitCount: 0, tolerance: null, radius: radiusClamp };
  }

  const det = detectPipeJunctions(rails, { radius, weldFraction: opts.weldFraction, tolerance: opts.tolerance });
  if (!det.ok) return det;
  const tolerance = det.tolerance;

  const junctionCounts = {};
  for (const J of det.junctions) junctionCounts[J.arity] = (junctionCounts[J.arity] || 0) + 1;

  const cages = [];
  for (const comp of det.components) {
    const built = buildComponent(comp, det, {
      radius, facets, segments, capStart, capEnd, crease, junctionCrease, hubInsetFraction, tolerance, buildTube, junction, fitRadius: !!opts.fitRadius, railSafe,
    });
    if (!built.ok) return built;
    cages.push(built.cage);
  }

  return {
    ok: true,
    welded: true,
    cages,
    junctions: det.junctions.map((J) => ({ point: J.point, arity: J.arity })),
    junctionCounts,
    junctionTotal: det.junctions.length,
    splitCount: det.splits.length,
    splits: det.splits,
    tolerance,
    radius: radiusClamp,
  };
}

function buildComponent(comp, det, cfg) {
  const { radius, capStart, capEnd, junctionCrease, hubInsetFraction, tolerance, buildTube, junction } = cfg;

  // Working rails, local to this component. `null` marks one consumed by a
  // concatenation; indices never shift, so every arm reference stays valid.
  const localOf = new Map();
  comp.rails.forEach((ri, k) => localOf.set(ri, k));
  const work = comp.rails.map((ri) => ({ curve: det.rails[ri], sources: [det.railSources[ri]] }));
  let junctions = comp.junctions.map((ji) => ({
    point: det.junctions[ji].point,
    arms: det.junctions[ji].arms.map((a) => ({ rail: localOf.get(a.rail), end: a.end })),
  }));

  // Arity-2 junctions are concatenated, not bridged. Two rails meeting at a
  // point are one longer rail, and one continuous tube gives a better elbow
  // than welding two stubs — the same move pipeRailForSweep makes for a
  // single rail's own corners, and the reason concatRailsAtJunction exists.
  let changed = true;
  while (changed) {
    changed = false;
    for (let ji = 0; ji < junctions.length; ji++) {
      const J = junctions[ji];
      if (J.arms.length !== 2) continue;
      const [a, b] = J.arms;
      if (a.rail === b.rail) {
        return { ok: false, reason: `a rail closes back on itself (its ${a.end} meets its own ${b.end}) — a closed-loop rail's two rings land on each other and there is nothing here to weld them with` };
      }
      const A = work[a.rail], B = work[b.rail];
      const res = concatRailsAtJunction(A.curve, B.curve, { tolerance });
      if (!res.ok) {
        return { ok: false, reason: `two rails meet at a point but cannot be joined into one: ${res.reason}` };
      }
      const nStart = endPointOf(res.curve, 'start');
      const nEnd = endPointOf(res.curve, 'end');
      if (length(sub(nStart, nEnd)) <= tolerance) {
        return { ok: false, reason: 'these rails join into a closed loop — a loop has no free ends and its own seam is a second junction nothing here has looked at' };
      }
      const farA = { rail: a.rail, end: otherEnd(a.end), pt: endPointOf(A.curve, otherEnd(a.end)) };
      const farB = { rail: b.rail, end: otherEnd(b.end), pt: endPointOf(B.curve, otherEnd(b.end)) };
      const startIsFarA = length(sub(nStart, farA.pt)) <= length(sub(nStart, farB.pt));
      const remap = new Map([
        [`${farA.rail}:${farA.end}`, { rail: a.rail, end: startIsFarA ? 'start' : 'end' }],
        [`${farB.rail}:${farB.end}`, { rail: a.rail, end: startIsFarA ? 'end' : 'start' }],
      ]);
      work[a.rail] = { curve: res.curve, sources: [...new Set([...A.sources, ...B.sources])].sort((x, y) => x - y) };
      work[b.rail] = null;
      junctions.splice(ji, 1);
      for (const K of junctions) {
        for (const arm of K.arms) {
          const m = remap.get(`${arm.rail}:${arm.end}`);
          if (m) { arm.rail = m.rail; arm.end = m.end; }
        }
      }
      changed = true;
      break;
    }
  }

  // Hub geometry. Every remaining junction is an N >= 3 meeting.
  const junctionInfo = [];
  for (const J of junctions) {
    const N = J.arms.length;
    if (N < 3) return { ok: false, reason: `a junction of ${N} arms survived the concatenation pass, which should not be possible` };
    const dirs = J.arms.map((arm) => inwardDirectionAt(work[arm.rail].curve, arm.end));
    if (dirs.some((d) => !d)) return { ok: false, reason: 'a rail has no well-defined direction at a junction (a zero-length derivative there)' };

    let thetaMin = Math.PI, tightest = null;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const ang = Math.acos(Math.max(-1, Math.min(1, dot(dirs[i], dirs[j]))));
        if (ang < thetaMin) { thetaMin = ang; tightest = [i, j]; }
      }
    }
    if (thetaMin < PIPE_NETWORK_MIN_ARM_ANGLE) {
      return {
        ok: false,
        reason: `two arms of the junction at (${J.point.map((v) => v.toFixed(2)).join(', ')}) leave it only ${(thetaMin * 180 / Math.PI).toFixed(1)} degrees apart — below ${(PIPE_NETWORK_MIN_ARM_ANGLE * 180 / Math.PI).toFixed(0)} degrees the two tubes run alongside each other and the pull-back a clean hub would need (${(1 / Math.tan(Math.max(thetaMin, 1e-9) / 2)).toFixed(1)} radii) is longer than any ordinary rail`,
        junctionPoint: J.point,
        armAngle: thetaMin,
      };
    }
    // Exactly where two cylinders of this radius crossing at this angle
    // stop interpenetrating, with the wide-angle floor, in radii.
    let factor = Math.max(hubInsetFraction, 1 / Math.tan(thetaMin / 2));
    let spatial = junction === 'hull';

    if (N >= 4 && !spatial) {
      // Three arms always lie in a plane, so the single-plane hub can only
      // be wrong from four up. Measured on the arm directions, which is
      // equivalent to measuring the rim centers: every rim sits the same
      // inset along its own arm, so the two sets differ by one uniform
      // scale and one translation, neither of which changes planarity.
      const residual = planarityResidual(dirs);
      if (residual > PIPE_NETWORK_PLANARITY_TOLERANCE && junction === 'auto') spatial = true;
      else if (residual > PIPE_NETWORK_PLANARITY_TOLERANCE) {
        return {
          ok: false,
          reason: `the ${N}-arm junction at (${J.point.map((v) => v.toFixed(2)).join(', ')}) is genuinely three-dimensional (its arms sit ${(residual * 100).toFixed(0)}% of their own spread out of any one plane) — this hub orders its arms around ONE plane, and two arms on opposite sides of it can project to the same angle and be joined to the wrong neighbor. A truly 3D junction needs a different construction, not a bigger version of this one`,
          junctionPoint: J.point,
          planarityResidual: residual,
        };
      }
    }

    if (spatial) factor *= PIPE_NETWORK_HULL_INSET_MARGIN;
    // A hull joint pulls each arm back by its own pairs: arm a against arm b
    // needs pair[a][b] times the fatter of the two, so an arm with wide
  // neighbors stays long however tight two other arms are.
    const pair = spatial ? dirs.map((da, a) => dirs.map((db, b) => (a === b ? 0
      : PIPE_NETWORK_HULL_INSET_MARGIN * Math.max(hubInsetFraction, 1 / Math.tan(Math.acos(Math.max(-1, Math.min(1, dot(da, db)))) / 2))))) : null;
    junctionInfo.push({ point: J.point, arity: N, arms: J.arms, dirs, factor, baseFactor: factor, armAngle: thetaMin, tightest, spatial, pair });
  }

  // Radii. One radius for every rail, unless `fitRadius`: then each rail
  // keeps the asked radius where its own two joints leave room, and is
  // thinned where they do not. A junction's pull-back is its factor times
  // the fattest arm meeting there (a thin arm has to clear a fat one), so a
  // short rail thins the arms round it, not the whole network, and the
  // thinning only ever goes down, so it settles.
  const lengths = work.map((w) => {
    if (!w) return 0;
    const [uMin, uMax] = domainOf(w.curve);
    return buildArcLengthTable(w.curve, uMin, uMax).total;
  });
  const r = work.map((w) => (w && cfg.railSafe ? Math.min(radius, ...w.sources.map((i) => cfg.railSafe[i])) : radius));
  // A hull joint's per-arm pull-backs, solved together: each clears its
  // pairs, and each rim lies behind every other rim's plane (d_a >= d_b cos +
  // r_b sin, with a twentieth of a radius to spare), so every rim is a facet
  // of the hull. The second condition is a contraction by the cosine of the
  // tightest angle; it is iterated to rest. Unscaled, as the ports want them.
  const armBase = (info) => {
    const N = info.arms.length, rad = info.arms.map((arm) => r[arm.rail]);
    const d = rad.map((ra, a) => { let m = 0; for (let b = 0; b < N; b++) if (b !== a) m = Math.max(m, info.pair[a][b] * Math.max(ra, rad[b])); return m; });
    for (let it = 0; it < 200; it++) {
      let moved = 0;
      for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) {
        if (a === b) continue;
        const c = dot(info.dirs[a], info.dirs[b]);
        if (!(c > 0)) continue;
        const need = d[b] * c + rad[b] * (Math.sqrt(Math.max(0, 1 - c * c)) + 0.05);
        if (need > d[a]) { moved = Math.max(moved, need - d[a]); d[a] = need; }
      }
      if (!(moved > 1e-6 * Math.max(...rad))) break;
    }
    return d;
  };
  const armInset = (info, a) => (info.factor / info.baseFactor) * info.armD[a];
  const insetFor = () => {
    const ins = work.map(() => ({ start: 0, end: 0 }));
    for (const info of junctionInfo) {
      const fat = Math.max(...info.arms.map((arm) => r[arm.rail]));
      info.inset = info.factor * fat;
      if (info.pair) info.armD = armBase(info);
      info.arms.forEach((arm, a) => {
        const d = info.pair ? armInset(info, a) : info.inset;
        ins[arm.rail][arm.end] = Math.max(ins[arm.rail][arm.end], d);
      });
    }
    return ins;
  };
  const endJunctions = work.map(() => []);
  for (const info of junctionInfo) for (const arm of info.arms) endJunctions[arm.rail].push(info);
  const fitRadii = () => {
    for (let pass = 0; pass < 100; pass++) {
      let thinned = false;
      const ins = insetFor();
      work.forEach((w, li) => {
        if (!w) return;
        const need = ins[li].start + ins[li].end;
        const room = PIPE_NETWORK_FIT_HEADROOM * lengths[li];
        if (!(need > room)) return;
        const k = room / need;
        for (const info of endJunctions[li]) {
          if (info.pair) {
            // Only the arms holding this rail's own end back: the pairs whose
            // clearance exceeds k of the present inset are capped, and the
            // rest of the joint keeps its radius.
            const scale = info.factor / info.baseFactor;
            info.arms.forEach((arm, a) => {
              if (arm.rail !== li) return;
              const target = k * armInset(info, a);
              info.arms.forEach((other, b) => {
                if (b === a) return;
                const cap = target / (scale * info.pair[a][b]);
                for (const rail of [arm.rail, other.rail]) if (r[rail] > cap) { r[rail] = cap; thinned = true; }
              });
              // The facet coupling can hold an end back past every pair: then
              // the rail and its arm's neighbors thin together.
              if (!thinned && armInset(info, a) > target) for (const o of info.arms) if (r[o.rail] > k * r[o.rail]) { r[o.rail] *= k; thinned = true; }
            });
            continue;
          }
          const fat = Math.max(...info.arms.map((arm) => r[arm.rail]));
          for (const arm of info.arms) if (r[arm.rail] > k * fat) { r[arm.rail] = k * fat; thinned = true; }
        }
      });
      if (!thinned) return;
    }
  };
  // Trim. A rail shorter than the insets its own two junctions demand
  // has no tube left in the middle — refused by name, with the numbers.
  const trimAll = (insets) => work.map((w, li) => {
    if (!w) return null;
    const [uMin, uMax] = domainOf(w.curve);
    const table = buildArcLengthTable(w.curve, uMin, uMax);
    const total = table.total;
    const need = insets[li].start + insets[li].end;
    if (!(total - need > 0)) {
      return { fail: `a rail ${total.toFixed(2)}mm long sits between junctions that need it pulled back ${insets[li].start.toFixed(2)}mm at one end and ${insets[li].end.toFixed(2)}mm at the other — ${need.toFixed(2)}mm in all, leaving no tube in the middle. Move the junctions apart, or use a smaller radius` };
    }
    const uA = insets[li].start > 0 ? paramAtArcLength(table, insets[li].start) : uMin;
    const uB = insets[li].end > 0 ? paramAtArcLength(table, total - insets[li].end) : uMax;
    const curve = (uA > uMin || uB < uMax) ? extractSubCurve(w.curve, uA, uB) : w.curve;
    return { curve, length: total, remaining: total - need };
  });
  // A hull junction's rims, measured where they are. The inset factor
  // assumes each rim sits square to its arm's end direction; a rail that
  // bends inside the pull-back puts its rim somewhere else. Each rim must
  // lie wholly behind every other rim's plane (so it is a facet of the
  // hull); a junction whose trimmed rims do not is pulled back a quarter
  // further and trimmed again.
  const rimClear = (info, trimmed) => {
    const ends = info.arms.map((arm) => {
      const c = trimmed[arm.rail].curve;
      return { p: endPointOf(c, arm.end), n: inwardDirectionAt(c, arm.end), r: r[arm.rail] };
    });
    if (ends.some((e) => !e.n)) return true;
    const slack = 1e-6 * info.inset;
    for (let i = 0; i < ends.length; i++) for (let j = 0; j < ends.length; j++) {
      if (i === j) continue;
      const d = dot(ends[j].n, ends[i].n);
      const reach = dot(sub(ends[j].p, ends[i].p), ends[i].n) + ends[j].r * Math.sqrt(Math.max(0, 1 - d * d));
      if (reach > -slack) return false;
    }
    return true;
  };
  for (let clearRound = 0; ; clearRound++) {
  for (const info of junctionInfo) { info.factor = info.baseFactor; info.viaPorts = false; }
  let trimmed;
  for (let round = 0; ; round++) {
    if (cfg.fitRadius) fitRadii();
    trimmed = trimAll(insetFor());
    for (const t of trimmed) if (t && t.fail) return { ok: false, reason: t.fail };
    const tight = junctionInfo.filter((info) => info.spatial && !rimClear(info, trimmed));
    if (!tight.length || round >= PIPE_NETWORK_HULL_ROUNDS) {
      // A junction whose own rims still are not all facets is closed through
      // ports: rings square to each arm's end direction at the junction's
      // first pull-back, which are facets by construction, each joined to its
      // real rim by one band of quads (Bridge's reach, at a junction).
      for (const info of tight) info.viaPorts = true;
      break;
    }
    for (const info of tight) info.factor *= 1.25;
  }

  // Tubes. A junction-facing end is always 'none'; only a free end gets
  // the network's own cap style.
  const atJunction = work.map(() => ({ start: false, end: false }));
  for (const J of junctions) for (const arm of J.arms) atJunction[arm.rail][arm.end] = true;

  const cage = { vertices: [], faces: [], creases: {} };
  const rims = work.map(() => null);
  const tubeFaces = work.map(() => null);
  let freeEndCount = 0;
  const sources = new Set();
  let railCount = 0;
  for (let li = 0; li < work.length; li++) {
    if (!work[li]) continue;
    railCount++;
    for (const s of work[li].sources) sources.add(s);
    const cs = atJunction[li].start ? 'none' : capStart;
    const ce = atJunction[li].end ? 'none' : capEnd;
    if (!atJunction[li].start) freeEndCount++;
    if (!atJunction[li].end) freeEndCount++;
    const tube = buildTube(trimmed[li].curve, cs, ce, r[li]);
    const f0 = cage.faces.length;
    const offset = mergeCage(cage, tube);
    tubeFaces[li] = [f0, cage.faces.length];
    // Each rim listed in the direction its own tube's faces walk it, which is
    // what the hull junction winds against.
    const walked = (rim) => {
      if (rim.length < 2) return rim;
      const a = rim[0], b = rim[1];
      const along = tube.faces.some((f) => f.some((v, k) => v === a && f[(k + 1) % f.length] === b));
      return along ? rim : [...rim].reverse();
    };
    rims[li] = {
      start: walked(tube.startRim).map((v) => v + offset),
      end: walked(tube.endRim).map((v) => v + offset),
    };
  }

  // Hubs. bridgeClosedRimsHub only appends vertices and faces, so every rim
  // index recorded above stays valid across all of them.
  let current = cage;
  const hubs = [];
  for (const info of junctionInfo) {
    const armRims = info.arms.map((arm) => rims[arm.rail][arm.end]);
    if (armRims.some((r) => !r || r.length === 0)) {
      return { ok: false, reason: 'a junction-facing tube end produced no open rim to weld — a capped end cannot take a junction' };
    }
    let res;
    try {
      if (info.spatial) {
        const base = current.faces.length;
        let hullRims = armRims;
        if (info.viaPorts) {
          const fat = Math.max(...info.arms.map((arm) => r[arm.rail]));
          hullRims = armRims.map((rim, k) => {
            const t = info.dirs[k], rk = r[info.arms[k].rail];
            const centre = add(info.point, scale(t, info.pair ? info.armD[k] : info.baseFactor * fat));
            const rimC = scale(rim.reduce((acc, vi) => add(acc, current.vertices[vi]), [0, 0, 0]), 1 / rim.length);
            const port = rim.map((vi) => {
              const u = sub(current.vertices[vi], rimC);
              const w = sub(u, scale(t, dot(u, t)));
              current.vertices.push(add(centre, scale(normalize(length(w) > 0 ? w : u), rk)));
              return current.vertices.length - 1;
            });
            for (let i = 0; i < rim.length; i++) {
              const j = (i + 1) % rim.length;
              current.faces.push([rim[j], rim[i], port[i], port[j]]);
            }
            return port;
          });
        }
        const hull = bridgeClosedRimsHull(current.vertices, hullRims);
        for (const f of hull.faces) current.faces.push(f);
        res = { cage: current, hubFaceIndices: current.faces.map((_, i) => i).slice(base), poleIndices: [] };
      } else {
        res = bridgeClosedRimsHub(current, armRims, { creaseWeight: junctionCrease });
      }
    } catch (err) {
      return { ok: false, reason: `the ${info.arity}-arm junction at (${info.point.map((v) => v.toFixed(2)).join(', ')}) could not be welded: ${err.message}` };
    }
    current = res.cage;
    hubs.push({ point: info.point, arity: info.arity, inset: info.inset, armAngle: info.armAngle, spatial: !!info.spatial, faceIndices: res.hubFaceIndices, poleIndices: res.poleIndices });
  }

  // Clearance. Thin only the tubes a crossing touches, then rebuild.
  const crossings = cfg.fitRadius ? cageCrossingFacePairs(current) : [];
  if (crossings.length && clearRound < PIPE_NETWORK_CLEAR_ROUNDS) {
    const hubOf = new Map();
    hubs.forEach((h, k) => { for (const f of h.faceIndices) hubOf.set(f, k); });
    const railOf = (f) => tubeFaces.findIndex((t) => t && f >= t[0] && f < t[1]);
    const thin = new Set();
    for (const pair of crossings) for (const f of pair) {
      const k = hubOf.get(f);
      if (k != null) for (const arm of junctionInfo[k].arms) thin.add(arm.rail);
      else { const li = railOf(f); if (li >= 0) thin.add(li); }
    }
    for (const li of thin) r[li] *= PIPE_NETWORK_CLEAR_THIN;
    continue;
  }

  return {
    ok: true,
    cage: {
      vertices: current.vertices,
      faces: current.faces,
      creases: current.creases,
      sourceRails: [...sources].sort((a, b) => a - b),
      railCount,
      junctions: hubs.map((h) => ({ point: h.point, arity: h.arity, inset: h.inset, armAngle: h.armAngle, spatial: h.spatial })),
      radii: (() => { const live = r.filter((_, li) => work[li]); return { asked: radius, min: Math.min(...live), max: Math.max(...live), thinned: live.filter((x) => x < radius).length }; })(),
      crossings: crossings.length, meanRadius: (() => { const live = r.filter((_, li) => work[li]); return live.reduce((a, b) => a + b, 0) / live.length; })(),
      hubs,
      freeEndCount,
    },
  };
  }
}

// Face pairs of a cage that pass through each other: an edge of one triangle
// (of a fan-split face) piercing the interior of another, faces sharing a
// vertex skipped. Triangles are bucketed on a grid of twice the median
// triangle extent; a pair is tested only in the cell holding the low corner
// of the overlap of their boxes, so each pair is tested once.
export function cageCrossingFacePairs(cage) {
  const V = cage.vertices, tris = [];
  cage.faces.forEach((f, fi) => {
    for (let k = 1; k + 1 < f.length; k++) {
      const a = V[f[0]], b = V[f[k]], c = V[f[k + 1]];
      tris.push({ fi, v: [f[0], f[k], f[k + 1]], p: [a, b, c],
        lo: [0, 1, 2].map((d) => Math.min(a[d], b[d], c[d])), hi: [0, 1, 2].map((d) => Math.max(a[d], b[d], c[d])) });
    }
  });
  if (!tris.length) return [];
  const eps = 1e-10;
  const pierces = (p0, p1, a, b, c) => {
    const d = sub(p1, p0), e1 = sub(b, a), e2 = sub(c, a), h = cross(d, e2), det = dot(e1, h);
    if (Math.abs(det) < eps) return false;
    const inv = 1 / det, sv = sub(p0, a), u = dot(sv, h) * inv;
    if (u < eps || u > 1 - eps) return false;
    const q = cross(sv, e1), v = dot(d, q) * inv;
    if (v < eps || u + v > 1 - eps) return false;
    const t = dot(e2, q) * inv;
    return t > eps && t < 1 - eps;
  };
  const ext = tris.map((T) => Math.max(T.hi[0] - T.lo[0], T.hi[1] - T.lo[1], T.hi[2] - T.lo[2])).sort((x, y) => x - y);
  const pitch = 2 * ext[ext.length >> 1];
  if (!(pitch > 0)) return [];
  const cellOf = (x) => Math.floor(x / pitch);
  const key = (x, y, z) => `${x}_${y}_${z}`;
  const buckets = new Map();
  tris.forEach((T, i) => {
    const a = T.lo.map(cellOf), b = T.hi.map(cellOf);
    for (let x = a[0]; x <= b[0]; x++) for (let y = a[1]; y <= b[1]; y++) for (let z = a[2]; z <= b[2]; z++) {
      const k = key(x, y, z);
      let list = buckets.get(k);
      if (!list) buckets.set(k, list = []);
      list.push(i);
    }
  });
  const out = [];
  for (const [k, list] of buckets) {
    for (let a = 0; a < list.length; a++) for (let b = a + 1; b < list.length; b++) {
      const A = tris[list[a]], B = tris[list[b]];
      if (A.fi === B.fi) continue;
      let apart = false;
      const low = [0, 0, 0];
      for (let d = 0; d < 3; d++) { if (A.hi[d] < B.lo[d] || B.hi[d] < A.lo[d]) { apart = true; break; } low[d] = cellOf(Math.max(A.lo[d], B.lo[d])); }
      if (apart || key(low[0], low[1], low[2]) !== k) continue;
      if (A.v.some((x) => B.v.includes(x))) continue;
      let hit = false;
      for (const [x, y] of [[0, 1], [1, 2], [2, 0]]) if (pierces(A.p[x], A.p[y], B.p[0], B.p[1], B.p[2]) || pierces(B.p[x], B.p[y], A.p[0], A.p[1], A.p[2])) { hit = true; break; }
      if (hit) out.push([A.fi, B.fi]);
    }
  }
  return out;
}

// The hull junction — N open rims joined by the convex hull of all their
// vertices, with each rim's own facet left open: a sphere with N holes,
// in triangles, for arms in any directions at all. Each rim is listed in the
// direction its own tube walks it; the hull walks every rim edge the other
// way, which is what makes it one consistently wound skin with the tubes.
// A rim is a planar ring and four coplanar corners are what an incremental
// hull cannot triangulate cleanly, so each rim also gets a peak, its center
// pushed a hair outward along its own arm: its facet becomes a strictly
// convex pyramid, every triangle touching a peak belongs to that rim and is
// dropped, and the peaks never enter the cage (the construction Bridge's
// spatial junction uses). Only the rims' own vertices are read: the cost is
// the junction's, not the cage's.
// Returns { faces } in the cage's vertex indices. Throws when a rim is not a
// facet of the hull (its arm ends inside the junction the others make).
export function bridgeClosedRimsHull(vertices, rims) {
  const N = rims.length;
  if (N < 2) throw new Error('bridgeClosedRimsHull: at least two rims');
  const pts = rims.flat();
  const P = pts.map((vi) => vertices[vi]);
  const centreOf = (rim) => scale(rim.reduce((acc, vi) => add(acc, vertices[vi]), [0, 0, 0]), 1 / rim.length);
  const centres = rims.map(centreOf);
  const junction = scale(centres.reduce((acc, c) => add(acc, c), [0, 0, 0]), 1 / N);
  let extent = 0;
  for (const q of P) extent = Math.max(extent, length(sub(q, junction)));
  const rimOf = new Map();
  rims.forEach((rim, k) => rim.forEach((vi) => rimOf.set(vi, k)));
  // Each peak rises off its rim's own plane (Newell's normal), on the side
  // away from the junction, so it sits over that facet and no other.
  const peakBase = P.length;
  rims.forEach((rim, k) => {
    let nrm = [0, 0, 0];
    for (let i = 0; i < rim.length; i++) {
      const a = vertices[rim[i]], b = vertices[rim[(i + 1) % rim.length]];
      nrm = add(nrm, [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]);
    }
    if (!(length(nrm) > 0)) nrm = sub(centres[k], junction);
    if (dot(nrm, sub(centres[k], junction)) < 0) nrm = scale(nrm, -1);
    P.push(add(centres[k], scale(normalize(nrm), extent * 1e-4)));
  });
  const hull = convexHullFaces(P);
  const onHull = new Set();
  let faces = [];
  for (const f of hull) {
    const peak = f.find((i) => i >= peakBase);
    if (peak !== undefined) {
      const k = peak - peakBase;
      for (const i of f) if (i < peakBase && rimOf.get(pts[i]) !== k) throw new Error(`bridgeClosedRimsHull: rim ${k + 1} is not a facet of the junction's hull — its arm ends inside the junction the others make`);
      for (const i of f) if (i < peakBase) onHull.add(pts[i]);
      continue;
    }
    const vs = f.map((i) => pts[i]);
    vs.forEach((vi) => onHull.add(vi));
    if (new Set(vs.map((vi) => rimOf.get(vi))).size > 1) faces.push(vs);
  }
  for (const vi of pts) if (!onHull.has(vi)) throw new Error(`bridgeClosedRimsHull: rim ${rimOf.get(vi) + 1} is not a facet of the junction's hull — its arm ends inside the junction the others make`);
  // Wind against the tubes: find a hull face on a rim edge and compare.
  const tubeDir = new Set();
  for (const rim of rims) for (let k = 0; k < rim.length; k++) tubeDir.add(`${rim[k]}>${rim[(k + 1) % rim.length]}`);
  let same = 0, opposite = 0;
  for (const f of faces) for (let k = 0; k < 3; k++) {
    const a = f[k], b = f[(k + 1) % 3];
    if (tubeDir.has(`${a}>${b}`)) same++;
    else if (tubeDir.has(`${b}>${a}`)) opposite++;
  }
  if (same > opposite) faces = faces.map((f) => [f[0], f[2], f[1]]);
  // Every rim edge now carried once, the other way round.
  const used = new Map();
  for (const f of faces) for (let k = 0; k < 3; k++) { const key = `${f[k]}>${f[(k + 1) % 3]}`; used.set(key, (used.get(key) || 0) + 1); }
  for (const rim of rims) for (let k = 0; k < rim.length; k++) {
    const back = `${rim[(k + 1) % rim.length]}>${rim[k]}`;
    if (used.get(back) !== 1) throw new Error('bridgeClosedRimsHull: a rim edge is not closed exactly once by the hull');
  }
  return { faces };
}
