// SUBD PIPE — a SuperB (Catmull-Clark) tube cage swept along a rail.
//
// Expression 2, step 1: the SuperB half of "pipe should have two
// expressions, a NURBS network with miters and a SuperB with soft
// corners." This file builds ONE tube along ONE rail. Junctions are step
// 2 and are deliberately not here — see the note at the bottom for what
// is actually blocking them, which is a real topology gap, not effort.
//
// WHY A CAGE AND NOT A SURFACE. A swept NURBS tube (sweep1Rigid) is exact
// and has no notion of softness: its corners are as sharp as its rail.
// A Catmull-Clark cage's limit surface rounds every corner by
// construction, which is the entire point of the SuperB expression — and
// it makes DENSITY a real dial rather than a tessellation setting,
// because a cage's own segment counts genuinely change the shape, not
// just how finely it is drawn.
//
// FRAMES ARE REUSED, NOT REDERIVED. The rings ride
// buildParallelTransportFrames' own frames — the same anti-flip transport
// sweep1Rigid has used since P0, sampled at arbitrary arc-length stations
// through its `extraParams` path. A pipe with no twist along a straight
// run is a property of that transport, already proven, not something this
// file re-establishes.
//
// STATIONS ARE ARC-LENGTH EVEN, not parameter-even. A rail's own
// parametrization bunches on uneven spans, and a cage whose rings bunch
// is a cage whose LIMIT SURFACE bunches — unlike a tessellation, where
// uneven sampling of an exact surface is merely wasteful. This is the
// same reasoning divideByArcLength already carries.

import { buildArcLengthTable, paramAtArcLength } from './curve.mjs';
import { buildParallelTransportFrames } from './sweep.mjs';
import { add, scale } from './vec3.mjs';
// The subdivider's OWN key function, imported rather than re-derived —
// the creases written here have to be the exact keys it reads, and a
// second local copy is precisely the kind of duplicate that drifts.
import { edgeKey } from './subd.mjs';

// Crease weight for a rim, from a 0..1 dial. 0 leaves no key at all
// (byte-identical to a cage that never had a crease), which is what makes
// "fully soft" provably the same shape as "no crease feature present."
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
//   facets    vertices AROUND the tube (>=3) — density, circumferentially
//   segments  spans ALONG the rail (>=1) — density, longitudinally
//   capStart  'none' | 'flat' | 'round'
//   capEnd    same
//   crease    0..1, rim sharpness. Only meaningful for an OPEN ('none')
//             or FLAT rim: a round cap has no rim to crease, and creasing
//             one would be creasing the middle of a dome.
//
// Returns `{ vertices, faces, creases, startRim, endRim }`. The two rims
// are the ring vertex indices in ring order, kept because a junction (step
// 2) needs to find them and rederiving a boundary loop afterward is both
// slower and less certain than simply not throwing away what was just
// built.
//
// A 'round' cap's rim index list comes back EMPTY rather than naming the
// ring underneath the apex — that ring is interior once a dome sits on it,
// so handing it to a bridge would tear the surface. Empty is the honest
// answer to "what open rim does this end have," not a missing value.
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
  // The FIRST ring's own outward direction is backwards along the rail,
  // the LAST ring's is forwards — a cap has to grow AWAY from the tube,
  // not back into it.
  // `isStart` decides winding, not a comparison against the frame's own
  // axis: every ring is built counter-clockwise about its frame's +zAxis,
  // so the END cap keeps that order to face outward and the START cap
  // must reverse it. (Deriving this by comparing the outward vector to
  // `zAxis` cannot work — the start's outward is a freshly scaled array,
  // never the same object, and comparing components instead would still
  // be re-deriving something the caller already knows.)
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
      // A single apex, one triangle per facet — SuperBCone's own apex
      // precedent, a genuine extraordinary vertex of valence `facets`
      // that computeVertexPoint already handles at any valence. It smooths
      // into a dome rather than the exact hemisphere the NURBS side
      // revolves; a cage rounds by construction, it does not interpolate.
      const apex = vertices.push(add(origin, scale(outward, radius))) - 1;
      for (let i = 0; i < ring.length; i++) {
        const j = (i + 1) % ring.length;
        faces.push(isStart ? [ring[j], ring[i], apex] : [ring[i], ring[j], apex]);
      }
      return [];
    }
    // 'none' — a genuinely open rim. A boundary edge is already forced
    // fully sharp by the subdivider (kernel/subd.mjs's own edgeSharpness),
    // so a crease key here would be inert; not written, rather than
    // written and silently ignored.
    return ring.slice();
  }

  return { vertices, faces, creases, startRim, endRim };
}

// WHERE A JUNCTION LIVES, and why it is not in this file.
//
// A junction of exactly TWO rails is not a bridging problem at all — two
// rails meeting at a point are one longer rail, and sweeping a single
// continuous tube along it gives a better elbow than welding two stubs
// ever would. That is the same move pipeRailForSweep already makes for a
// single rail's own corners.
//
// A junction of THREE OR MORE is the real one, and it has its own
// function: kernel/subdedit.mjs's `bridgeClosedRimsHub`, which takes N
// closed rims in exactly the shape `startRim`/`endRim` return them.
//
// WHAT THE NEIGHBORING MACHINERY DOES AND DOES NOT DO, stated precisely,
// because being wrong about it is worse than not knowing. bridgeEdgeRunsHub
// takes OPEN edge runs, and a tube rim is a CLOSED loop — but handing it a
// rim is not caught by anything intrinsic to an open run's own
// preconditions: a closed loop's consecutive pairs are all naked edges,
// which is the whole of what that function requires, so it builds a
// junction that winds consistently, contains no malformed face, and
// silently leaves each rim's own CLOSING edge unattached — a slit down
// every arm. Measured on three facets-6 tubes: 18 naked edges before, 9
// after, where a genuine closed-rim junction leaves none. It carries an
// explicit closed-rim refusal for that reason. bridgeBoundaryLoops handles
// closed loops but only ever two of them.
