// SuperB cage sub-object topology: edge loop, edge ring, face loop.
// Kernel only: pure cage-in-data-out topology walks, no selection state or
// UI. Reuses subd.mjs's `buildTopology`/`edgeKey` for cage adjacency.
//
// Loop versus ring, on a UxV quad grid (e.g. a cylindrical band): a loop
// runs around the cage's waist through shared vertices; a ring runs
// through a sequence of parallel edges across faces.
//
//   Edge loop (edgeLoopFromSeed) — a vertex-based chain. At each endpoint
//   vertex of the current edge, continue via the other incident edge that
//   shares neither of the current edge's (up to 2) adjacent faces — the
//   unique "straight across" continuation at a regular (valence-4)
//   vertex. On a UxV grid this walks a circumferential edge around the
//   band, closing into a loop when the topology is closed in that
//   direction. Stops at any interior vertex that is not valence-4, or
//   where the continuation edge is not unique; a boundary edge continues
//   along the boundary (see loopContinuationEdge).
//
//   Edge ring (edgeRingFromSeed) — a face-based chain. Within a quad face
//   containing the current edge, continue via the edge opposite it in
//   that face (two positions away in the face's 4-edge cyclic order),
//   crossing into the other face that shares that opposite edge, and
//   repeating. On the same grid this advances to a different quad
//   row/column each step — a band of parallel edges running lengthwise
//   across faces, never sharing a vertex with its predecessor. Stops at a
//   boundary edge (only one face) or a non-quad face (an n-gon has no
//   well-defined opposite edge for n != 4).
//
// Both walk in both directions from the seed edge (via its up-to-2
// incident faces or its 2 endpoint vertices) and stop if they would
// re-visit an edge already in the result (a closed loop/ring).

import { buildTopology, edgeKey } from './subd.mjs';

// The edge in `face` two positions away (in the face's cyclic vertex
// order) from the edge matching `seedKey` — the opposite edge of a quad
// ({0,2} and {1,3} of a 4-edge cyclic loop). Returns null for a non-quad
// face (n != 4) or if `seedKey` is not one of this face's edges.
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

// The unique edge at vertex `vIdx` sharing neither of `excludeEdge`'s
// (up to 2) incident faces — the edge loop's "continue straight through a
// regular vertex" rule. Returns null (the loop stops here) when an
// interior edge meets a vertex that is not valence-4, or the exclusion
// does not leave exactly one candidate.
function loopContinuationEdge(topology, vIdx, excludeEdge) {
  const edgesAtV = topology.vertexEdges[vIdx];
  const excludeFaces = new Set(excludeEdge.faces);
  const excludeKey = edgeKey(excludeEdge.v0, excludeEdge.v1);
  // A naked (boundary) edge continues along the boundary. The valence-4
  // rule below is the "straight through" continuation for an interior
  // edge, but a boundary vertex is valence-3 on an open grid, so that rule
  // would stop a boundary seed at its first vertex. A boundary is a closed
  // 1-manifold chain, so a boundary vertex has exactly two naked edges and
  // the continuation is the other one. A pinch point (3+ naked edges at one
  // vertex) has no unique answer and stops, as an extraordinary interior
  // vertex does.
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

// Edge loop — see the header. Always includes the seed edge; walks both
// directions until each stops (an extraordinary vertex or a boundary
// pinch) or closes back onto an edge already in the result. Throws if
// `seedKey` is not an edge of this cage.
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

// Edge ring — see the header. Always includes the seed edge;
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

// Face loop — the strip of faces running through a seed face in one of
// its two directions. It is edgeRingFromSeed's walk (cross a face to the
// edge opposite the one you entered by, step into the face that shares
// it), collecting the faces it passes through instead of the edges it
// crosses; both rest on `oppositeEdgeInFace`.
//
// A direction is required: a quad has two face loops through it (one per
// edge pair) and the face alone does not choose between them, so the
// seed edge is the direction. A caller holding only a face picks one of
// the two from faceLoopDirections or nearestFaceEdgeToPoint below.
//
// Stops the three ways the ring does: a boundary edge (nothing on the far
// side), a non-quad face (no defined opposite edge), or arriving back at
// a face already collected (a closed loop around a band).
export function faceLoopFromSeed(cage, faceIdx, seedKey) {
  const face = cage.faces[faceIdx];
  if (!face) throw new Error(`faceLoopFromSeed: ${faceIdx} is not a real face of this cage`);
  const topology = buildTopology(cage);
  if (!topology.edgeMap.has(seedKey)) throw new Error(`faceLoopFromSeed: "${seedKey}" is not a real edge of this cage`);
  if (!faceEdgeKeys(face).includes(seedKey)) throw new Error(`faceLoopFromSeed: "${seedKey}" is not an edge of face ${faceIdx}`);
  const result = [faceIdx];
  const seen = new Set([faceIdx]);
  // Both directions from the seed face: out through the seed edge, and
  // out through the edge opposite it. A non-quad seed face has no
  // opposite, so it walks the one direction it has.
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

// The two directions a face loop can run through a quad — its first two
// edges, whose opposites complete the pair — as a stable, ordered pair a
// caller holding only a face can pick from or alternate between. A
// non-quad face has no well-defined pairing and returns the empty list.
export function faceLoopDirections(cage, faceIdx) {
  const face = cage.faces[faceIdx];
  if (!face || face.length !== 4) return [];
  return [edgeKey(face[0], face[1]), edgeKey(face[1], face[2])];
}

// Squared distance from `p` to the segment ab, clamped to the segment
// (not the infinite line) — an unclamped line distance would let a
// far-off edge's extension win.
function distSqToSegment(p, a, b) {
  const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
  const apx = p[0] - a[0], apy = p[1] - a[1], apz = p[2] - a[2];
  const denom = abx * abx + aby * aby + abz * abz;
  let t = denom > 0 ? (apx * abx + apy * aby + apz * abz) / denom : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t; // a degenerate (zero-length) edge collapses to its endpoint
  const dx = apx - abx * t, dy = apy - aby * t, dz = apz - abz * t;
  return dx * dx + dy * dy + dz * dz;
}

// The nearest of the face's edges to `point`, as an edge key for
// faceLoopFromSeed: the face-loop direction read from where the pointer
// pressed. A press near the right-hand edge of a quad picks the strip
// that runs left-right, the one crossing that edge.
//
// Nearest-of-n edges, so it answers for any face, not only quads. A
// dead-center press on a square is a tie; ties resolve to the earliest
// edge in the face's cyclic order, so the same point always gives the
// same answer.
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
