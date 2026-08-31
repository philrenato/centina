// SuperB CAGE SUB-OBJECT TOPOLOGY — EDGE LOOP / EDGE RING
// (milestone 3, "real sub-object selection on a SuperB's own CAGE").
// KERNEL ONLY: pure cage-in-data-out topology walks, no app-layer object/
// selection-state/UI here — matches kernel/subd.mjs's own discipline
// exactly. Reuses subd.mjs's own `buildTopology`/`edgeKey` directly rather
// than re-deriving cage adjacency a second time.
//
// THE LOOP-VS-RING DISTINCTION, derived from first principles before
// writing a line of code here (per the explicit
// instruction: "verify you have this distinction right before coding it,
// they are NOT the same thing") — worked out on a concrete example (a
// UxV quad grid, e.g. a cylindrical band) and cross-checked against this
// milestone's own more careful wording ("a loop runs around the cage's
// own 'waist' through shared vertices; a ring runs through a sequence of
// parallel edges across faces"), which is the definition actually
// implemented below — NOT the milestone's own looser, throwaway first
// mention ("opposite edge of each quad, chain across shared faces" for
// double-click), which turns out on inspection to describe the RING
// algorithm, not the loop:
//
//   EDGE LOOP (edgeLoopFromSeed) — a VERTEX-based chain. At each endpoint
//   vertex of the current edge, continue via whichever OTHER incident
//   edge shares NEITHER of the current edge's own (up to 2) adjacent
//   faces — the unique "straight across" continuation at a regular
//   (valence-4) vertex. On a UxV grid this walks a circumferential edge
//   AROUND the band (through shared vertices, closing into a real loop
//   once the topology is genuinely closed that direction) — the classic
//   "loop around the waist." Terminates (does not continue in that
//   direction) at any vertex that is not valence-4, or where the
//   continuation edge isn't unique — a real, honest stop, not a crash,
//   matching this project's own "refuse rather than guess" standard for
//   an extraordinary/boundary vertex.
//
//   EDGE RING (edgeRingFromSeed) — a FACE-based chain. Within a quad face
//   containing the current edge, continue via the edge "opposite" it in
//   that same face (two positions away in the face's own 4-edge cyclic
//   order — the standard pair {top,bottom}/{left,right} of a quad),
//   crossing into whichever OTHER face shares that opposite edge, and
//   repeating. On the same UxV grid this advances to a DIFFERENT quad
//   ROW/COLUMN each step — a band of PARALLEL edges running lengthwise
//   across faces, never sharing a vertex with its own predecessor.
//   Terminates at a boundary edge (only one face) or a non-quad face
//   (ring is only defined through quads — an n-gon has no well-defined
//   "opposite edge" pairing for n != 4).
//
// Both walk in BOTH directions from the seed edge (via its own up-to-2
// incident faces/its own 2 endpoint vertices) and stop early if they'd
// re-visit an edge already in the result (a genuinely CLOSED loop/ring,
// e.g. all the way around a real closed band) rather than looping
// forever.

import { buildTopology, edgeKey } from './subd.mjs';

// The edge in `face` that is TWO POSITIONS away (in the face's own
// cyclic vertex order) from the edge matching `seedKey` — the standard
// "opposite edge of a quad" pairing ({0,2} and {1,3} of a 4-edge cyclic
// loop). Returns null for a non-quad face (n != 4) or if `seedKey` isn't
// actually one of this face's own edges (defensive; every real caller
// below only ever asks this of a face it already knows contains the
// edge).
export function oppositeEdgeInFace(cage, faceIdx, seedKey) {
  const face = cage.faces[faceIdx];
  const n = face.length;
  if (n !== 4) return null;
  for (let c = 0; c < n; c++) {
    const a = face[c], b = face[(c + 1) % n];
    if (edgeKey(a, b) === seedKey) {
      const oc = (c + 2) % n;
      return edgeKey(face[oc], face[(oc + 1) % n]);
    }
  }
  return null;
}

// The unique edge at vertex `vIdx` sharing NEITHER of `excludeEdge`'s own
// (up to 2) incident faces — the edge-loop's own "continue straight
// through a regular vertex" rule. Returns null (loop terminates here)
// whenever the vertex isn't valence-4, or the exclusion doesn't leave
// exactly one candidate (an extraordinary/boundary vertex, or — for a
// vertex with fewer than 4 real edges — nothing well-defined to
// continue with).
function loopContinuationEdge(topology, vIdx, excludeEdge) {
  const edgesAtV = topology.vertexEdges[vIdx];
  const excludeFaces = new Set(excludeEdge.faces);
  const excludeKey = edgeKey(excludeEdge.v0, excludeEdge.v1);
  // A NAKED (boundary) EDGE CONTINUES ALONG THE BOUNDARY. The valence-4
  // rule below is the right "straight through" continuation for an INTERIOR
  // edge, but a boundary vertex is valence-3 on an ordinary open grid, so
  // that rule stopped a boundary seed dead at its own first vertex — a
  // double-click on an open edge selected one edge and nothing else, while
  // the identical gesture on an interior edge walked a whole loop. A
  // boundary is a closed 1-manifold chain, so a boundary vertex has exactly
  // TWO naked edges: the continuation is the other one, no guessing. A pinch
  // point (3+ naked edges at one vertex) has no unique answer and stops
  // honestly, the same way an extraordinary interior vertex already does.
  if (excludeEdge.faces.length === 1) {
    const naked = edgesAtV.filter((e) => e.faces.length === 1 && edgeKey(e.v0, e.v1) !== excludeKey);
    return naked.length === 1 ? naked[0] : null;
  }
  if (edgesAtV.length !== 4) return null;
  const candidates = edgesAtV.filter((e) => {
    if (edgeKey(e.v0, e.v1) === excludeKey) return false;
    return !e.faces.some((f) => excludeFaces.has(f));
  });
  return candidates.length === 1 ? candidates[0] : null;
}

// EDGE LOOP — see this file's own header for the full derivation. Always
// includes the seed edge itself; walks both directions until each stops
// (an extraordinary/boundary vertex) or closes back onto an edge already
// in the result (a genuine closed loop). Throws honestly if `seedKey`
// isn't a real edge of this cage — the one thing this function can't
// recover from, matching every other kernel function's own "refuse
// rather than guess" convention for malformed input.
export function edgeLoopFromSeed(cage, seedKey) {
  const topology = buildTopology(cage);
  const seedEdge = topology.edgeMap.get(seedKey);
  if (!seedEdge) throw new Error(`edgeLoopFromSeed: "${seedKey}" is not a real edge of this cage`);
  const result = [seedKey];
  const seen = new Set([seedKey]);
  for (const startVertex of [seedEdge.v0, seedEdge.v1]) {
    let curVertex = startVertex;
    let curEdge = seedEdge;
    for (;;) {
      const next = loopContinuationEdge(topology, curVertex, curEdge);
      if (!next) break;
      const nextKey = edgeKey(next.v0, next.v1);
      if (seen.has(nextKey)) break;
      seen.add(nextKey);
      result.push(nextKey);
      curVertex = (next.v0 === curVertex) ? next.v1 : next.v0;
      curEdge = next;
    }
  }
  return result;
}

// EDGE RING — see this file's own header. Always includes the seed edge;
// walks both directions (one per incident face of the seed edge) until
// each stops (a boundary edge, or a non-quad face with no defined
// "opposite edge") or closes back onto an edge already in the result.
export function edgeRingFromSeed(cage, seedKey) {
  const topology = buildTopology(cage);
  const seedEdge = topology.edgeMap.get(seedKey);
  if (!seedEdge) throw new Error(`edgeRingFromSeed: "${seedKey}" is not a real edge of this cage`);
  const result = [seedKey];
  const seen = new Set([seedKey]);
  for (const startFace of seedEdge.faces) {
    let curKey = seedKey;
    let curFace = startFace;
    for (;;) {
      const oppKey = oppositeEdgeInFace(cage, curFace, curKey);
      if (!oppKey || seen.has(oppKey)) break;
      const oppEdge = topology.edgeMap.get(oppKey);
      if (!oppEdge) break;
      seen.add(oppKey);
      result.push(oppKey);
      const nextFaces = oppEdge.faces.filter((f) => f !== curFace);
      if (nextFaces.length !== 1) break; // boundary (0 left) or non-manifold (2+ left) — stop, don't guess
      curKey = oppKey;
      curFace = nextFaces[0];
    }
  }
  return result;
}

// The edge keys of one face, in its own cyclic order.
function faceEdgeKeys(face) {
  const n = face.length;
  const out = [];
  for (let c = 0; c < n; c++) out.push(edgeKey(face[c], face[(c + 1) % n]));
  return out;
}

// FACE LOOP — the strip of faces running through a seed face in ONE of
// its two directions. This is deliberately NOT a new traversal: it is
// edgeRingFromSeed's own walk (cross a face to the edge opposite the one
// you entered by, step into whichever face shares it), collecting the
// FACES it passes through instead of the edges it crosses. The primitive
// that makes both work is the same `oppositeEdgeInFace` already proven
// for the ring, so a face loop needed no new topology reasoning — only a
// different thing recorded per step.
//
// A DIRECTION IS REQUIRED, and that is the honest shape of the problem,
// not an API inconvenience. A quad has TWO face loops through it (one per
// edge-pair) and nothing about the face alone chooses between them, so
// the seed edge IS the direction. A caller holding only a face (a plain
// face selection, a double-click) must therefore pick one — see
// faceLoopDirections below, which hands back exactly the two real
// choices rather than letting the caller invent a third.
//
// Terminates the same three honest ways the ring does: a boundary edge
// (nothing on the far side), a non-quad face (no defined opposite edge),
// or arriving back at a face already collected (a genuinely closed loop
// around a band) rather than circling forever.
export function faceLoopFromSeed(cage, faceIdx, seedKey) {
  const face = cage.faces[faceIdx];
  if (!face) throw new Error(`faceLoopFromSeed: ${faceIdx} is not a real face of this cage`);
  const topology = buildTopology(cage);
  if (!topology.edgeMap.has(seedKey)) throw new Error(`faceLoopFromSeed: "${seedKey}" is not a real edge of this cage`);
  if (!faceEdgeKeys(face).includes(seedKey)) throw new Error(`faceLoopFromSeed: "${seedKey}" is not an edge of face ${faceIdx}`);
  const result = [faceIdx];
  const seen = new Set([faceIdx]);
  // BOTH directions from the seed face: out through the seed edge, and
  // out through the edge opposite it. A non-quad seed face has no
  // opposite, so it walks the one direction it genuinely has.
  const startKeys = [seedKey, oppositeEdgeInFace(cage, faceIdx, seedKey)].filter(Boolean);
  for (const startKey of startKeys) {
    let curFace = faceIdx;
    let curKey = startKey;
    for (;;) {
      const edge = topology.edgeMap.get(curKey);
      if (!edge) break;
      const nextFaces = edge.faces.filter((f) => f !== curFace);
      if (nextFaces.length !== 1) break; // boundary (0 left) or non-manifold (2+ left) — stop, don't guess
      const nextFace = nextFaces[0];
      if (seen.has(nextFace)) break; // closed all the way around
      seen.add(nextFace);
      result.push(nextFace);
      const opp = oppositeEdgeInFace(cage, nextFace, curKey);
      if (!opp) break; // an n-gon in the strip — no defined continuation
      curFace = nextFace;
      curKey = opp;
    }
  }
  return result;
}

// THE TWO REAL DIRECTIONS a face loop can run through a quad — its own
// first two edges, whose opposites complete the pair. Exists so a caller
// that has only a face (no seed edge) picks from the genuine choices
// instead of inventing one, and so an ambiguity-resolving caller has a
// stable, ordered pair to alternate between. A non-quad face has no
// well-defined pairing and honestly returns the empty list.
export function faceLoopDirections(cage, faceIdx) {
  const face = cage.faces[faceIdx];
  if (!face || face.length !== 4) return [];
  return [edgeKey(face[0], face[1]), edgeKey(face[1], face[2])];
}

// Squared distance from `p` to the segment ab, clamped to the segment
// (not the infinite line) — a face's edges are real bounded spans, and an
// unclamped line distance would let a far-off edge's own extension win.
function distSqToSegment(p, a, b) {
  const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
  const apx = p[0] - a[0], apy = p[1] - a[1], apz = p[2] - a[2];
  const denom = abx * abx + aby * aby + abz * abz;
  let t = denom > 0 ? (apx * abx + apy * aby + apz * abz) / denom : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t; // a degenerate (zero-length) edge collapses to its own endpoint
  const dx = apx - abx * t, dy = apy - aby * t, dz = apz - abz * t;
  return dx * dx + dy * dy + dz * dz;
}

// WHICH EDGE OF THIS FACE DID THE POINTER MEAN — the nearest of the
// face's own edges to `point`, as an edge key ready to hand straight to
// faceLoopFromSeed.
//
// THIS IS WHAT STOPS A FACE LOOP BEING A GUESS. A quad has two loops
// through it and the face alone cannot choose, but the CLICK can: press
// near the right-hand edge and you have pointed at the strip that runs
// left-right, because that strip is exactly the one crossing that edge.
// So the direction is read from the gesture rather than assumed, and the
// explicit re-read control is left to handle only the genuinely
// ambiguous press (dead centre) or a misread.
//
// Deliberately geometric and n-gon-safe: it is nearest-of-n edges, so it
// answers for any face, not only the quads a loop can actually walk. A
// dead-centre press on a square is a real tie; ties resolve to the
// earliest edge in the face's own cyclic order, which is stable across
// calls rather than arbitrary per call — a caller re-reading the same
// click must get the same answer, or the re-read control would appear to
// do nothing on its second press.
export function nearestFaceEdgeToPoint(cage, faceIdx, point) {
  const face = cage.faces[faceIdx];
  if (!face) throw new Error(`nearestFaceEdgeToPoint: ${faceIdx} is not a real face of this cage`);
  const n = face.length;
  let bestKey = null, bestD = Infinity;
  for (let c = 0; c < n; c++) {
    const a = cage.vertices[face[c]], b = cage.vertices[face[(c + 1) % n]];
    const d = distSqToSegment(point, a, b);
    if (d < bestD) { bestD = d; bestKey = edgeKey(face[c], face[(c + 1) % n]); }
  }
  return bestKey;
}
