// SUBD PIPE NETWORK — junction detection and cage assembly for a SuperB
// pipe network. Expression 2, step 2, under its own D1-D4.
//
// TWO FUNCTIONS, and the split is not arbitrary. `detectPipeJunctions`
// answers a question purely about CURVES — where do these rails meet, and
// which of them therefore belong to the same object — and answers it
// without building a single vertex. `subdPipeNetwork` consumes that answer
// and builds cages. Keeping them apart is what lets the app show a student
// how many junctions it found, and of what arity, before committing to any
// geometry, and it is what makes the detection half testable against curve
// facts alone.
//
// D1 — A T-JUNCTION IS A JUNCTION. A rail whose END lands on ANOTHER
// RAIL'S INTERIOR is not a near miss to be ignored: detection SPLITS the
// host there, so the touch point becomes a genuine endpoint of two host
// pieces and the meeting is an ordinary N-way one from that moment on.
// Every split goes through `extractSubCurve` (knots.mjs, proven), and the
// caller's own rails are never touched — every returned rail is a fresh
// curve, split or not.
//
// D2 — ONE CAGE PER CONNECTED GROUP. Welded arms share vertices, so a
// connected network is one cage; two runs that never meet stay two.
//
// D3 — WELDING IS AN OPTION, DEFAULT ON. With it off the result is N
// independent tubes and zero junctions, built from the caller's own rails
// with no splitting and no insetting at all.
//
// D4 — THE WELD TOLERANCE IS RADIUS-RELATIVE. `radius * weldFraction`,
// default fraction 0.5. WHY HALF THE RADIUS specifically: at that distance
// a branch's own endpoint is already INSIDE the host tube's solid, so the
// two tubes are visibly touching on screen and a student reasonably expects
// them to weld; past a full radius they do not touch at all and welding
// them would be bridging a visible gap. Half the radius is the middle of
// the band where "these are touching" is unambiguous, and it scales — a
// 500mm pipe welds within 250mm, a 0.5mm pipe within 0.25mm. It is
// deliberately NOT JOIN_TOLERANCE (0.001mm): this app has endpoint
// snapping but no on-curve snap, so a student can land a T exactly on a
// rail's ENDPOINT and essentially never on its INTERIOR, and an exact
// tolerance would make D1 unreachable in practice.
//
// THE INSET IS DERIVED FROM THE JUNCTION'S OWN TIGHTEST ANGLE, not fixed.
// `bridgeClosedRimsHub` attaches its hub directly to the rims it is given
// and refuses rims that are effectively coincident, so the arms must be
// pulled back along their own rails first — that pull-back is this file's
// job, and the thing HUB_INSET (0.62) generalizes to. The
// number transfers; the mechanism does not. HUB_INSET lerps a rim toward
// the hub CENTER; what a pipe network needs is a translation ALONG each
// rail, so the tube is genuinely shorter rather than squashed sideways.
// And a single fraction cannot be right at every angle: two cylinders of
// radius r whose axes cross at angle theta interpenetrate out to exactly
// r/tan(theta/2) from the crossing point, which is r at 90 degrees and
// grows without bound as the arms close up. So the inset is
// `radius * max(hubInsetFraction, 1/tan(thetaMin/2))` — the exact end of
// the interpenetration region, with 0.62 as a floor for wide angles. That
// floor is not a coincidence either: a Y of three arms at 120 degrees needs
// 1/tan(60) = 0.577, so 0.62 is very nearly the natural value for the
// commonest junction there is, which is presumably how it was arrived at.

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
import { bridgeClosedRimsHub } from './subdedit.mjs';

// See the header: half the tube radius is the middle of the band in which
// two tubes are unambiguously touching on screen.
export const PIPE_NETWORK_WELD_FRACTION = 0.5;
// The original HUB_INSET, kept as the wide-angle floor.
export const PIPE_NETWORK_HUB_INSET_FRACTION = 0.62;
// Below this the required inset runs away (1/tan(7.5deg) is already 7.6
// radii) and the "junction" is really two nearly-parallel tubes running
// alongside each other. Refused by name rather than silently demanding a
// pull-back longer than any rail in an ordinary model.
export const PIPE_NETWORK_MIN_ARM_ANGLE = Math.PI / 12; // 15 degrees
// How far out of one plane a junction's arms may sit before the
// single-plane hub can no longer be trusted to order them. Measured as the
// worst arm's own out-of-plane offset against the spread of the arms about
// their mean, so it is a shape ratio, not a length.
export const PIPE_NETWORK_PLANARITY_TOLERANCE = 0.25;

// ─────────────────────────────────────────────────────────────────────────
// small curve helpers

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
// The unit direction pointing INTO the rail from one of its own ends —
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

// Lexicographic point compare, used ONLY to put results in a deterministic
// order. Junctions are distinct points by construction (two junctions
// within tolerance of each other would have clustered into one), so this
// never has to break a real tie.
function comparePoints(a, b) {
  return a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
}

// ─────────────────────────────────────────────────────────────────────────
// DETECTION
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
// or { ok: false, reason } for a genuinely ambiguous input.
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

  // ── T-JUNCTIONS. Every rail END is tested against every OTHER rail's
  // body. A hit within tolerance of that host's own endpoint is not a T at
  // all — it is an ordinary endpoint meeting, which the clustering below
  // picks up without any split — so only genuinely INTERIOR hits are
  // recorded here.
  const splitParams = rails.map(() => []);
  for (let bi = 0; bi < rails.length; bi++) {
    for (const which of ['start', 'end']) {
      const P = endPointOf(rails[bi], which);
      for (let hi = 0; hi < rails.length; hi++) {
        if (hi === bi) continue; // a rail touching its own interior is a closed/self-crossing rail, a different problem
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

  // ── SPLIT. Several branches can land on one host, and two of them can
  // land in the same place; merging by real 3D distance rather than by
  // parameter is what keeps that correct on an unevenly parametrized rail.
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

  // ── ENDPOINT CLUSTERING. Every endpoint of every (post-split) rail,
  // grouped by proximity. A cluster of 2+ is a junction.
  const eps = [];
  outRails.forEach((c, i) => {
    eps.push({ rail: i, end: 'start', pt: endPointOf(c, 'start') });
    eps.push({ rail: i, end: 'end', pt: endPointOf(c, 'end') });
  });
  const uf = makeUnionFind(eps.length);
  for (let i = 0; i < eps.length; i++) {
    for (let j = i + 1; j < eps.length; j++) {
      if (length(sub(eps[i].pt, eps[j].pt)) <= tolerance) uf.union(i, j);
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

  // ── COMPONENTS.
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

// ─────────────────────────────────────────────────────────────────────────
// ASSEMBLY

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
// fraction of how far they spread. Exactly the quantity junctionPlaneOrder
// needs to be small: it fits ONE plane through the rim centers and orders
// them by angle in it, which only means anything if they genuinely lie in
// a plane. Three points always do, so this is only ever asked of N >= 4.
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
//   capStart / capEnd — the network's FREE ends only. A junction-facing end
//     is always built with cap 'none' regardless, because a 'round' cap
//     returns an EMPTY rim (correctly: the ring under a dome is interior)
//     and there would be nothing left to weld.
//   weld — D3. Default true.
//   weldFraction / tolerance — D4.
//   hubInsetFraction — the wide-angle floor on the pull-back, see header.
//   junctionCrease — crease weight written onto the welded rims. 0 by
//     default: a soft junction is the whole point of the SuperB expression.
//   clampRadiusToBend — reuse Pipe's own self-intersection clamp so a rail
//     tighter than the tube cannot silently swallow itself. On by default.
//
// Returns { ok: true, cages, junctions, junctionCounts, ... } or
// { ok: false, reason } for a refusal a student can act on.
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
  const weld = opts.weld !== false;
  const hubInsetFraction = opts.hubInsetFraction ?? PIPE_NETWORK_HUB_INSET_FRACTION;

  // ── RADIUS CLAMP. subdPipeCage has no bend-radius guard of its own; the
  // NURBS side's does, and it is a pure curve fact, so it is reused rather
  // than reasoned about a second time. One shared radius means the tightest
  // rail in the network governs.
  let radius = requestedRadius;
  let radiusClamp = { clamped: false, requested: requestedRadius, radius, safeMax: Infinity, rail: null };
  if (opts.clampRadiusToBend !== false) {
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

  const buildTube = (curve, cs, ce) => subdPipeCage(curve, { radius, facets, segments, capStart: cs, capEnd: ce, crease });

  // ── D3, WELDING OFF. N independent tubes, from the caller's own rails,
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
      radius, facets, segments, capStart, capEnd, crease, junctionCrease, hubInsetFraction, tolerance, buildTube,
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
  const { radius, capStart, capEnd, junctionCrease, hubInsetFraction, tolerance, buildTube } = cfg;

  // Working rails, local to this component. `null` marks one consumed by a
  // concatenation; indices never shift, so every arm reference stays valid.
  const localOf = new Map();
  comp.rails.forEach((ri, k) => localOf.set(ri, k));
  const work = comp.rails.map((ri) => ({ curve: det.rails[ri], sources: [det.railSources[ri]] }));
  let junctions = comp.junctions.map((ji) => ({
    point: det.junctions[ji].point,
    arms: det.junctions[ji].arms.map((a) => ({ rail: localOf.get(a.rail), end: a.end })),
  }));

  // ── ARITY-2 JUNCTIONS ARE NOT BRIDGED, THEY ARE CONCATENATED. Two rails
  // meeting at a point are one longer rail, and one continuous tube gives a
  // better elbow than welding two stubs — the same move pipeRailForSweep
  // already makes for a single rail's own corners, and the reason
  // concatRailsAtJunction exists.
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

  // ── HUB GEOMETRY. Every remaining junction is a genuine N >= 3 meeting.
  const insets = work.map(() => ({ start: 0, end: 0 }));
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
    // stop interpenetrating, with the wide-angle floor.
    const inset = radius * Math.max(hubInsetFraction, 1 / Math.tan(thetaMin / 2));

    if (N >= 4) {
      // Three arms always lie in a plane, so the single-plane hub can only
      // be wrong from four up. Measured on the arm DIRECTIONS, which is
      // equivalent to measuring the rim centers: every rim sits the same
      // inset along its own arm, so the two sets differ by one uniform
      // scale and one translation, neither of which changes planarity.
      const residual = planarityResidual(dirs);
      if (residual > PIPE_NETWORK_PLANARITY_TOLERANCE) {
        return {
          ok: false,
          reason: `the ${N}-arm junction at (${J.point.map((v) => v.toFixed(2)).join(', ')}) is genuinely three-dimensional (its arms sit ${(residual * 100).toFixed(0)}% of their own spread out of any one plane) — this hub orders its arms around ONE plane, and two arms on opposite sides of it can project to the same angle and be joined to the wrong neighbor. A truly 3D junction needs a different construction, not a bigger version of this one`,
          junctionPoint: J.point,
          planarityResidual: residual,
        };
      }
    }

    for (const arm of J.arms) insets[arm.rail][arm.end] = Math.max(insets[arm.rail][arm.end], inset);
    junctionInfo.push({ point: J.point, arity: N, arms: J.arms, inset, armAngle: thetaMin, tightest });
  }

  // ── TRIM. A rail shorter than the insets its own two junctions demand
  // has no tube left in the middle — refused by name, with the numbers.
  const trimmed = work.map((w, li) => {
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
  for (const t of trimmed) if (t && t.fail) return { ok: false, reason: t.fail };

  // ── TUBES. A junction-facing end is always 'none'; only a free end gets
  // the network's own cap style.
  const atJunction = work.map(() => ({ start: false, end: false }));
  for (const J of junctions) for (const arm of J.arms) atJunction[arm.rail][arm.end] = true;

  const cage = { vertices: [], faces: [], creases: {} };
  const rims = work.map(() => null);
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
    const tube = buildTube(trimmed[li].curve, cs, ce);
    const offset = mergeCage(cage, tube);
    rims[li] = {
      start: tube.startRim.map((v) => v + offset),
      end: tube.endRim.map((v) => v + offset),
    };
  }

  // ── HUBS. bridgeClosedRimsHub only ever APPENDS vertices and faces, so
  // every rim index recorded above stays valid across all of them.
  let current = cage;
  const hubs = [];
  for (const info of junctionInfo) {
    const armRims = info.arms.map((arm) => rims[arm.rail][arm.end]);
    if (armRims.some((r) => !r || r.length === 0)) {
      return { ok: false, reason: 'a junction-facing tube end produced no open rim to weld — a capped end cannot take a junction' };
    }
    let res;
    try {
      res = bridgeClosedRimsHub(current, armRims, { creaseWeight: junctionCrease });
    } catch (err) {
      return { ok: false, reason: `the ${info.arity}-arm junction at (${info.point.map((v) => v.toFixed(2)).join(', ')}) could not be welded: ${err.message}` };
    }
    current = res.cage;
    hubs.push({ point: info.point, arity: info.arity, inset: info.inset, armAngle: info.armAngle, faceIndices: res.hubFaceIndices, poleIndices: res.poleIndices });
  }

  return {
    ok: true,
    cage: {
      vertices: current.vertices,
      faces: current.faces,
      creases: current.creases,
      sourceRails: [...sources].sort((a, b) => a - b),
      railCount,
      junctions: hubs.map((h) => ({ point: h.point, arity: h.arity, inset: h.inset, armAngle: h.armAngle })),
      hubs,
      freeEndCount,
    },
  };
}
