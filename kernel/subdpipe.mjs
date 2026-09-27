// SubD pipe — a SuperB (Catmull-Clark) tube cage swept along a rail.
//
// The SuperB half of Pipe's two expressions: a NURBS network with miters,
// and a SuperB with soft corners. This file builds one tube along one rail;
// junctions are built in kernel/subdnetwork.mjs (see the note at the bottom).
//
// Why a cage and not a surface. A swept NURBS tube (sweep1Rigid) is exact
// and has no notion of softness: its corners are as sharp as its rail. A
// Catmull-Clark cage's limit surface rounds every corner by construction,
// which is the point of the SuperB expression, and it makes density a real
// dial rather than a tessellation setting, because a cage's segment counts
// change the shape, not just how finely it is drawn.
//
// Frames are reused, not rederived. The rings ride
// buildParallelTransportFrames' frames — the same anti-flip transport
// sweep1Rigid uses, sampled at arbitrary arc-length stations through its
// `extraParams` path. A pipe with no twist along a straight run is a
// property of that transport, not something this file re-establishes.
//
// Stations are arc-length even, not parameter-even. A rail's own
// parametrization bunches on uneven spans, and a cage whose rings bunch is
// a cage whose limit surface bunches — unlike a tessellation, where uneven
// sampling of an exact surface is merely wasteful. divideByArcLength follows
// the same reasoning.

import { buildArcLengthTable, paramAtArcLength } from './curve.mjs';
import { buildParallelTransportFrames } from './sweep.mjs';
import { add, scale } from './vec3.mjs';
// The subdivider's own key function, imported rather than re-derived — the
// creases written here have to be the exact keys it reads.
import { edgeKey } from './subd.mjs';

// Crease weight for a rim, from a 0..1 dial. 0 leaves no key at all
// (byte-identical to a cage that never had a crease), so "fully soft" is the
// same shape as "no crease feature present."
export const SUBD_PIPE_MAX_RIM_CREASE = 4;
function rimCreaseWeight(crease) {
  const c = Math.max(0, Math.min(1, Number(crease) || 0));
  return c * SUBD_PIPE_MAX_RIM_CREASE;
}

// The rail's own arc-length-even station parameters, ends exact.
function stationParams(rail, segments) {
  const uMin = rail.knots[0], uMax = rail.knots[rail.knots.length - 1];
  const table = buildArcLengthTable(rail, uMin, uMax);
  const total = table.total;
  const out = [uMin];
  for (let i = 1; i < segments; i++) out.push(paramAtArcLength(table, (i / segments) * total));
  out.push(uMax);
  return out;
}

// Builds the tube cage.
//
//   radius    tube radius
//   facets    vertices around the tube (>=3) — density, circumferentially
//   segments  spans along the rail (>=1) — density, longitudinally
//   capStart  'none' | 'flat' | 'round'
//   capEnd    same
//   crease    0..1, rim sharpness. Only meaningful for an open ('none')
//             or flat rim: a round cap has no rim to crease, and creasing
//             one would be creasing the middle of a dome.
//
// Returns `{ vertices, faces, creases, startRim, endRim }`. The two rims
// are the ring vertex indices in ring order, kept because a junction needs
// them, and rederiving a boundary loop afterward is slower and less certain
// than keeping what was just built.
//
// A 'round' cap's rim index list comes back empty rather than naming the
// ring underneath the apex — that ring is interior once a dome sits on it,
// so handing it to a bridge would tear the surface. Empty is the answer to
// "what open rim does this end have," not a missing value.
export function subdPipeCage(rail, opts = {}) {
  const radius = opts.radius ?? 5;
  const facets = Math.max(3, Math.round(opts.facets ?? 8));
  const segments = Math.max(1, Math.round(opts.segments ?? 4));
  const capStart = opts.capStart ?? 'none';
  const capEnd = opts.capEnd ?? 'none';
  const crease = rimCreaseWeight(opts.crease);
  if (!rail || !Array.isArray(rail.ctrlPts) || rail.ctrlPts.length < 2) throw new Error('subdPipeCage: needs a real rail curve');
  if (!(radius > 0)) throw new Error(`subdPipeCage: radius must be positive (got ${radius})`);

  const params = stationParams(rail, segments);
  const frames = buildParallelTransportFrames(rail, params).extra;
  if (!frames || frames.length !== params.length) throw new Error('subdPipeCage: frame count does not match the requested stations');

  const vertices = [];
  const rings = [];
  for (const f of frames) {
    const ring = [];
    for (let i = 0; i < facets; i++) {
      const a = (i / facets) * Math.PI * 2;
      const p = add(f.origin, add(scale(f.xAxis, radius * Math.cos(a)), scale(f.yAxis, radius * Math.sin(a))));
      if (!p.every(Number.isFinite)) throw new Error('subdPipeCage: the rail produced a degenerate frame — a zero-length or self-reversing segment');
      ring.push(vertices.push(p) - 1);
    }
    rings.push(ring);
  }

  const faces = [];
  for (let s = 0; s < rings.length - 1; s++) {
    const a = rings[s], b = rings[s + 1];
    for (let i = 0; i < facets; i++) {
      const j = (i + 1) % facets;
      faces.push([a[i], a[j], b[j], b[i]]);
    }
  }

  const creases = {};
  // The first ring's outward direction is backwards along the rail, the
  // last ring's is forwards — a cap has to grow away from the tube, not back
  // into it.
  // `isStart` decides winding, not a comparison against the frame's own
  // axis: every ring is built counter-clockwise about its frame's +zAxis,
  // so the end cap keeps that order to face outward and the start cap must
  // reverse it. (Comparing the outward vector to `zAxis` cannot work — the
  // start's outward is a freshly scaled array, never the same object, and
  // comparing components would re-derive what the caller already knows.)
  const f0 = frames[0], fN = frames[frames.length - 1];
  const startRim = applyCap(capStart, rings[0], scale(f0.zAxis, -1), f0.origin, true);
  const endRim = applyCap(capEnd, rings[rings.length - 1], fN.zAxis, fN.origin, false);

  function applyCap(style, ring, outward, origin, isStart) {
    if (style === 'flat') {
      // An n-gon is a legal Catmull-Clark face — superbCylinderCage's own
      // caps are exactly this, so nothing here special-cases face size.
      faces.push(isStart ? ring.slice().reverse() : ring.slice());
      if (crease > 0) for (let i = 0; i < ring.length; i++) creases[edgeKey(ring[i], ring[(i + 1) % ring.length])] = crease;
      return ring.slice();
    }
    if (style === 'round') {
      // A single apex, one triangle per facet — as SuperBCone's apex, an
      // extraordinary vertex of valence `facets` that computeVertexPoint
      // handles at any valence. It smooths into a dome rather than the exact
      // hemisphere the NURBS side revolves; a cage rounds by construction, it
      // does not interpolate.
      const apex = vertices.push(add(origin, scale(outward, radius))) - 1;
      for (let i = 0; i < ring.length; i++) {
        const j = (i + 1) % ring.length;
        faces.push(isStart ? [ring[j], ring[i], apex] : [ring[i], ring[j], apex]);
      }
      return [];
    }
    // 'none' — an open rim. A boundary edge is already forced fully sharp by
    // the subdivider (kernel/subd.mjs's edgeSharpness), so a crease key here
    // would be inert and is not written.
    return ring.slice();
  }

  return { vertices, faces, creases, startRim, endRim };
}

// Where a junction lives, and why it is not in this file.
//
// A junction of exactly two rails is not a bridging problem — two rails
// meeting at a point are one longer rail, and sweeping a single continuous
// tube along it gives a better elbow than welding two stubs. That is the
// same move pipeRailForSweep makes for a single rail's own corners.
//
// A junction of three or more has its own function: kernel/subdedit.mjs's
// `bridgeClosedRimsHub`, which takes N closed rims in exactly the shape
// `startRim`/`endRim` return them.
//
// The neighboring functions do not substitute for it. bridgeEdgeRunsHub
// takes open edge runs, and a tube rim is a closed loop — but nothing
// intrinsic to an open run's preconditions catches a rim: a closed loop's
// consecutive pairs are all naked edges, which is all that function
// requires, so it builds a junction that winds consistently, contains no
// malformed face, and leaves each rim's closing edge unattached — a slit
// down every arm (three facets-6 tubes: 18 naked edges before, 9 after,
// where a closed-rim junction leaves none). It carries an explicit
// closed-rim refusal for that reason. bridgeBoundaryLoops handles closed
// loops but only two of them.
