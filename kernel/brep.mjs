// HALF-EDGE B-REP TOPOLOGY + EULER OPERATORS + VALIDITY CHECKING
// ================================================================
// PHASE 1 of the B-rep work: topology layer + Euler operators + validity
// checking, kernel-only, with its own test suite.
// KERNEL ONLY: no THREE.js, no app-layer object, no UI, no
// tessellation policy — pure topology-in, topology-out, exactly like every
// other kernel/*.mjs module here.
//
// WHY HALF-EDGE (a choice the spec left open, resolved here and
// re-confirmed against the actual code before writing any):
//   - kernel/subd.mjs's own buildTopology is NEITHER half-edge NOR
//     winged-edge — it is a plain unordered edge-adjacency map
//     (`edgeKey -> {v0, v1, faces[]}`) with no per-edge-use orientation and
//     no next/twin pointers. There is therefore no in-project convention to
//     inherit either way; the consideration drops out entirely and the
//     choice is made on the merits.
//   - Mäntylä's own GWB kernel — this module's primary citation — is
//     implemented with half-edges, so the citation and the code match.
//   - Winged-edge traversal needs per-step case analysis on edge
//     orientation (the classic critique). Half-edge's next/twin invariants
//     remove that case analysis entirely, which matters most in a
//     hand-rolled, student-legible kernel.
//   - Decisively for THIS kernel: a trimmed face's loop is naturally a
//     CYCLE OF HALF-EDGES, and each half-edge is exactly where a per-side
//     pcurve lives (ACIS calls it a "coedge", Parasolid a "fin"; Weiler
//     1985 is the standard reference for extending an edge-based structure
//     to curved-surface B-reps). kernel/trim.mjs already reserves the dual
//     `{ uv, edge3d, tolerance }` slots per loop "from day one" — those map
//     ONE-TO-ONE onto a half-edge (its own pcurve) and the shared edge (its
//     own 3D curve + tolerance). Winged-edge has no equally clean home for
//     a per-side pcurve.
//
// PRIOR ART (cited before this code landed, per this project's own
// citation-discipline rule — read for TECHNIQUE, never transcribed):
//   - Baumgart, "A Polyhedron Representation for Computer Vision" (1975) —
//     the original winged-edge structure this whole family descends from.
//   - Braid, Hillyard & Stroud, "Stepwise Construction of Polyhedra in
//     Geometric Modeling" (1978) — the original Euler operators.
//   - Mäntylä, "An Introduction to Solid Modeling" (1988) — the canonical
//     half-edge B-rep + Euler-operator reference; MVFS / MEV / MEF / KEMR /
//     KFMRH and their inverses, and the Euler-Poincaré invariant in the
//     form used below.
//   - Requicha, "Representations for Rigid Solids" (ACM Computing Surveys,
//     1980) — B-rep validity: what makes a boundary representation a real
//     solid rather than a pile of faces.
//   - Weiler, "Edge-Based Data Structures for Solid Modeling in
//     Curved-Surface Environments" (IEEE CG&A, 1985) — the coedge concept;
//     why a per-half-edge pcurve is the right home for a trimmed face's own
//     2D boundary.
//
// THE EULER-POINCARÉ INVARIANT, in the exact form this module checks:
//
//     V - E + F - R = 2 * (S - G)
//
//   V vertices, E edges, F faces, R "rings" (inner loops — every loop
//   beyond the one outer loop each face carries, so R = L - F), S shells,
//   G genus (through-holes/handles). Every operator below preserves it, and
//   `eulerCharacteristic()` recomputes both sides from live counts so a
//   test can assert it rather than trust a comment.
//
//   HONEST NOTE ON GENUS, stated rather than glossed: G is not derivable
//   from V/E/F/L/S alone (that would make the check circular). It is
//   TRACKED on the solid, incremented by KFMRH and decremented by MFKRH —
//   the standard GWB approach. The invariant check is therefore a real
//   cross-check that every operator's own V/E/F/L bookkeeping agrees with
//   the tracked genus/shell count: an operator that changed counts without
//   the matching genus update fails it immediately. It is NOT an
//   independent measurement of genus from the mesh.
//
// PHASE 1 SCOPE — what this module deliberately does NOT do, named rather
// than silently missing:
//   - No booleans, no surface-surface intersection, no face splitting
//     (Phase 2/3). This module never computes geometry; it only carries it.
//   - No void shells. Every solid built through these operators has exactly
//     one shell (S = 1). The S term in the invariant is therefore correct
//     but only ever exercised at S = 1 by the operator set itself —
//     `mergeSolidShells()` (below, explicitly NOT an Euler operator) exists
//     so a multi-shell configuration can still be built and checked.
//   - No geometric validity: self-intersection, pcurve-vs-3D-curve
//     agreement, face-normal orientation against real geometry. That was
//     resolved deliberately — validity is TWO layers,
//     and this module is the COMBINATORIAL one (cited math). The geometric
//     layer is engineering on top of this app's own existing tolerance
//     model (JOIN_TOLERANCE and siblings) and is not attempted here.
//   - No non-manifold modeling. An edge with more than two half-edges is
//     REPORTED as an error by validateBrep, not supported.

import { trimLoopsValid } from './trim.mjs';

// ---------------------------------------------------------------------
// ENTITIES
// ---------------------------------------------------------------------
// Every entity carries an integer `id` (unique within its own solid), a
// `kind` tag, and a `solid` back-pointer. The back-pointer is redundant
// with the ownership chain (halfEdge -> loop -> face -> shell -> solid) and
// that redundancy is deliberate: validateBrep cross-checks the two against
// each other, which is exactly how an orphaned or mis-parented entity gets
// caught instead of quietly participating in a traversal.
//
//   Vertex   { id, point:[x,y,z]|null, halfEdge }         one outgoing half-edge
//   Edge     { id, halfEdges:[he,he], curve3d, tolerance } exactly 2 when manifold
//   HalfEdge { id, vertex, edge, twin, next, prev, loop, pcurve }
//              `vertex` is the ORIGIN. The destination is `next.vertex`.
//              `pcurve` is this side's own 2D boundary in the OWNING FACE's
//              surface parameter space — Weiler's coedge slot, and the
//              direct counterpart of kernel/trim.mjs's own per-loop `uv`.
//   Loop     { id, face, halfEdge }                        one half-edge on the cycle
//   Face     { id, shell, loops:[outer, ...rings], surface }
//              loops[0] is ALWAYS the outer loop; every later entry is a
//              ring (an inner boundary — a hole in this face).
//              `surface` is a nullable NurbsSrf slot (see the TRIMMED
//              SURFACE BRIDGE section at the bottom).
//   Shell    { id, solid, faces:[], kind }                 kind reserved, null in Phase 1
//   Solid    { id, vertices, edges, shells, genus, nextId }

function newId(solid) { return solid.nextId++; }

function makeVertex(solid, point) {
  const v = {
    id: newId(solid), kind: 'vertex', solid,
    point: point ? [point[0], point[1], point[2]] : null,
    halfEdge: null,
  };
  solid.vertices.push(v);
  return v;
}

// Both half-edges of an edge are created together, always — an edge with
// one half-edge is not a state this module can be in. A DANGLING edge (a
// spur, produced by MEV) is represented by both of its half-edges sitting
// in the SAME loop, not by a missing twin; that is what keeps the "every
// edge has exactly two half-edges" invariant true at every intermediate
// step of a construction, not just at the end.
function makeEdgePair(solid, v1, v2) {
  const edge = { id: newId(solid), kind: 'edge', solid, halfEdges: [], curve3d: null, tolerance: null };
  const h1 = { id: newId(solid), kind: 'halfedge', solid, vertex: v1, edge, twin: null, next: null, prev: null, loop: null, pcurve: null };
  const h2 = { id: newId(solid), kind: 'halfedge', solid, vertex: v2, edge, twin: h1, next: null, prev: null, loop: null, pcurve: null };
  h1.twin = h2;
  edge.halfEdges = [h1, h2];
  solid.edges.push(edge);
  if (!v1.halfEdge) v1.halfEdge = h1;
  if (!v2.halfEdge) v2.halfEdge = h2;
  return { edge, h1, h2 };
}

function makeLoop(solid, face) {
  const loop = { id: newId(solid), kind: 'loop', solid, face, halfEdge: null };
  return loop;
}

function makeFace(solid, shell) {
  const face = { id: newId(solid), kind: 'face', solid, shell, loops: [], surface: null };
  shell.faces.push(face);
  return face;
}

function makeShell(solid) {
  // `shellKind` ('outer' | 'void') is a reserved slot — Phase 1 never
  // classifies shells, and says so rather than storing a guess.
  const shell = { id: newId(solid), kind: 'shell', solid, faces: [], shellKind: null };
  solid.shells.push(shell);
  return shell;
}

function removeFrom(arr, item) {
  const i = arr.indexOf(item);
  if (i < 0) return false;
  arr.splice(i, 1);
  return true;
}

// ---------------------------------------------------------------------
// TRAVERSAL
// ---------------------------------------------------------------------

// Walk a loop's cycle. Throws on a broken cycle — operators rely on this
// being an invariant, so a failure here is a real internal bug, not user
// input. validateBrep uses walkLoopSafe() instead, which REPORTS the same
// conditions rather than throwing (a validator that throws on the very
// corruption it exists to find is not a validator).
export function loopHalfEdges(loop) {
  const res = walkLoopSafe(loop);
  if (!res.ok) throw new Error(`brep: loop ${loop.id} is not a closed cycle — ${res.reason}`);
  return res.halfEdges;
}

export function walkLoopSafe(loop, maxSteps = 100000) {
  if (!loop.halfEdge) return { ok: true, halfEdges: [], empty: true };
  const out = [];
  let he = loop.halfEdge;
  for (let step = 0; step < maxSteps; step++) {
    out.push(he);
    if (!he.next) return { ok: false, reason: `half-edge ${he.id} has no next`, halfEdges: out };
    if (he.next.prev !== he) return { ok: false, reason: `half-edge ${he.id}.next.prev is not itself (next/prev disagree)`, halfEdges: out };
    he = he.next;
    if (he === loop.halfEdge) return { ok: true, halfEdges: out, empty: false };
  }
  return { ok: false, reason: `did not return to its start within ${maxSteps} steps`, halfEdges: out };
}

export function allHalfEdges(solid) {
  const out = [];
  for (const e of solid.edges) for (const he of e.halfEdges) out.push(he);
  return out;
}

export function allFaces(solid) {
  const out = [];
  for (const sh of solid.shells) for (const f of sh.faces) out.push(f);
  return out;
}

export function allLoops(solid) {
  const out = [];
  for (const f of allFaces(solid)) for (const l of f.loops) out.push(l);
  return out;
}

// The destination vertex of a half-edge. Not stored — derived, so it cannot
// go stale independently of the cycle.
export function destinationOf(he) { return he.next ? he.next.vertex : null; }

// Every half-edge in `loop` whose ORIGIN is `vertex`. More than one is
// entirely legal mid-construction (a vertex that already has a spur hanging
// off it appears twice in the same loop), which is exactly why the
// operators take an explicit disambiguating half-edge when it happens.
export function halfEdgesInLoopFrom(loop, vertex) {
  return loopHalfEdges(loop).filter((he) => he.vertex === vertex);
}

function uniqueHalfEdgeInLoopFrom(loop, vertex, opName) {
  const cands = halfEdgesInLoopFrom(loop, vertex);
  if (cands.length === 0) throw new Error(`brep ${opName}: vertex ${vertex.id} has no half-edge in loop ${loop.id}`);
  if (cands.length > 1) {
    throw new Error(`brep ${opName}: vertex ${vertex.id} appears ${cands.length} times in loop ${loop.id} — pass opts.before to say which sector to split (this is an ambiguity, not a failure)`);
  }
  return cands[0];
}

// ---------------------------------------------------------------------
// EULER-POINCARÉ
// ---------------------------------------------------------------------

export function eulerCharacteristic(solid) {
  const V = solid.vertices.length;
  const E = solid.edges.length;
  const faces = allFaces(solid);
  const F = faces.length;
  let L = 0;
  for (const f of faces) L += f.loops.length;
  const S = solid.shells.length;
  const G = solid.genus;
  const R = L - F;               // rings = every loop beyond each face's own outer loop
  const lhs = V - E + F - R;
  const rhs = 2 * (S - G);
  return { V, E, F, L, R, S, G, lhs, rhs, ok: lhs === rhs };
}

// Throwing form, for use inside a construction sequence so an operator that
// silently broke the invariant is caught at the exact step that broke it
// rather than at the end of a 28-step build.
export function assertEulerPoincare(solid, whereLabel = '') {
  const e = eulerCharacteristic(solid);
  if (!e.ok) {
    throw new Error(
      `brep: Euler-Poincaré violated${whereLabel ? ` after ${whereLabel}` : ''} — ` +
      `V-E+F-R = ${e.lhs} but 2(S-G) = ${e.rhs}  ` +
      `(V=${e.V} E=${e.E} F=${e.F} L=${e.L} R=${e.R} S=${e.S} G=${e.G})`
    );
  }
  return e;
}

// ---------------------------------------------------------------------
// EULER OPERATORS
// ---------------------------------------------------------------------
// The canonical Mäntylä set, five forward operators and their five
// inverses. Each one's own comment states the delta it applies to
// (V, E, F, R, S, G) and shows the invariant arithmetic explicitly — that
// arithmetic is what the tests assert, not a claim.
//
// A NOTE ON THE OPERATOR LIST THIS WAS SPECIFIED FROM, corrected rather
// than silently worked around. That list
// names "mve, mev, mef, kemh, etc." — "mve" and "mev" are the same operator
// under two spellings (MEV, Make Edge and Vertex, is the standard name),
// and "kemh" does not correspond to any operator in the standard set. The
// two operators it is closest to, and the two actually needed to build a
// face with a ring and a solid with a handle, are KEMR (Kill Edge, Make
// Ring) and KFMRH (Kill Face, Make Ring and Hole). Both are implemented
// below under their real names.

// MVFS — Make Vertex, Face, Solid. The seed: a solid consisting of one
// vertex, one face whose single loop is EMPTY, one shell.
//   V+1, F+1, S+1  ->  lhs +2 ; rhs 2*(+1 - 0) = +2   OK
export function mvfs(point, opts = {}) {
  const solid = { id: 0, kind: 'solid', nextId: 0, vertices: [], edges: [], shells: [], genus: 0, name: opts.name ?? null };
  solid.id = newId(solid);
  const shell = makeShell(solid);
  const face = makeFace(solid, shell);
  const loop = makeLoop(solid, face);
  face.loops.push(loop);
  const vertex = makeVertex(solid, point);
  return { solid, shell, face, loop, vertex };
}

// KVFS — the inverse. Only legal on a solid that is still exactly the seed
// state: one vertex, no edges, one shell, one face, one empty loop.
export function kvfs(solid) {
  const e = eulerCharacteristic(solid);
  if (!(e.V === 1 && e.E === 0 && e.F === 1 && e.S === 1 && e.L === 1 && e.G === 0)) {
    throw new Error(`brep kvfs: only a bare MVFS seed can be killed (found V=${e.V} E=${e.E} F=${e.F} L=${e.L} S=${e.S} G=${e.G})`);
  }
  const face = solid.shells[0].faces[0];
  if (face.loops[0].halfEdge) throw new Error('brep kvfs: the seed loop still holds half-edges');
  solid.vertices.length = 0;
  solid.shells.length = 0;
  return solid;
}

// MEV — Make Edge and Vertex. Grows a new vertex at `point`, joined to
// `vertex` by a new edge, both of whose half-edges sit in `loop` (a spur).
//   V+1, E+1  ->  lhs +1 -1 = 0 ; rhs unchanged   OK
//
// Two cases, both real:
//   (a) SEED — `loop` is still empty (straight out of MVFS). The two new
//       half-edges become the whole cycle.
//   (b) INSERT — `loop` already has a cycle. The spur is spliced in
//       immediately BEFORE the half-edge leaving `vertex`. When `vertex`
//       occurs more than once in the loop the caller must say which
//       occurrence via `opts.before`; guessing there would silently pick a
//       different sector and produce a different (still valid) solid.
export function mev(loop, vertex, point, opts = {}) {
  const solid = loop.solid;
  if (vertex.solid !== solid) throw new Error('brep mev: vertex belongs to a different solid');
  const seeding = loop.halfEdge === null;
  let at = null;
  if (seeding) {
    if (vertex.halfEdge) throw new Error(`brep mev: loop ${loop.id} is empty but vertex ${vertex.id} already has edges`);
  } else {
    at = opts.before ?? uniqueHalfEdgeInLoopFrom(loop, vertex, 'mev');
    if (at.loop !== loop) throw new Error('brep mev: opts.before is not in the given loop');
    if (at.vertex !== vertex) throw new Error('brep mev: opts.before does not start at the given vertex');
  }

  const v2 = makeVertex(solid, point);
  const { edge, h1, h2 } = makeEdgePair(solid, vertex, v2); // h1: vertex -> v2, h2: v2 -> vertex
  h1.loop = loop; h2.loop = loop;

  if (seeding) {
    h1.next = h2; h1.prev = h2;
    h2.next = h1; h2.prev = h1;
    loop.halfEdge = h1;
  } else {
    const p = at.prev;
    p.next = h1; h1.prev = p;
    h1.next = h2; h2.prev = h1;
    h2.next = at; at.prev = h2;
  }
  return { edge, vertex: v2, halfEdge: h1, twin: h2 };
}

// KEV — Kill Edge and Vertex, the exact inverse of MEV. `he` must be the
// half-edge running OUT to the doomed vertex, and its own `next` must be
// its twin — i.e. the far end has valence 1 and nothing else hangs off it.
// That single condition is what makes this a true inverse rather than a
// general edge-collapse (which is not an Euler operator and is not offered).
export function kev(he) {
  const solid = he.solid;
  const t = he.twin;
  if (t.loop !== he.loop) throw new Error(`brep kev: edge ${he.edge.id} is not a spur — its half-edges are in different loops (use kef)`);
  if (he.next !== t) throw new Error(`brep kev: half-edge ${he.id} does not run out to a valence-1 vertex (he.next is not its twin)`);
  const loop = he.loop;
  const doomed = t.vertex;
  const p = he.prev, n = t.next;

  if (p === t) {
    // The loop was exactly [he, t] — it becomes empty again (the seed state).
    loop.halfEdge = null;
  } else {
    p.next = n; n.prev = p;
    if (loop.halfEdge === he || loop.halfEdge === t) loop.halfEdge = p;
  }

  removeFrom(solid.edges, he.edge);
  removeFrom(solid.vertices, doomed);
  repointVertexHalfEdges(solid);
  return { solid };
}

// MEF — Make Edge and Face. Splits one loop into two by joining the origins
// of two half-edges already in that loop, and gives the new loop its own
// new face in the same shell.
//   E+1, F+1, L+1 (so R unchanged)  ->  lhs -1 +1 -0 = 0 ; rhs unchanged  OK
//
// The NEW face is the side containing `he1` (documented, not incidental —
// the inverse depends on it).
export function mef(he1, he2) {
  const solid = he1.solid;
  const loop = he1.loop;
  if (he2.loop !== loop) throw new Error('brep mef: both half-edges must be in the same loop');
  if (he1 === he2) throw new Error('brep mef: he1 and he2 must be different half-edges');

  const v1 = he1.vertex, v2 = he2.vertex;
  const p1 = he1.prev, p2 = he2.prev;
  const { edge, h1: heA, h2: heB } = makeEdgePair(solid, v1, v2); // heA: v1 -> v2, heB: v2 -> v1

  // The ORIGINAL loop keeps he2 ... p1, closed by heA (v1 -> v2).
  p1.next = heA; heA.prev = p1;
  heA.next = he2; he2.prev = heA;
  // The NEW loop takes he1 ... p2, closed by heB (v2 -> v1).
  p2.next = heB; heB.prev = p2;
  heB.next = he1; he1.prev = heB;

  const face = makeFace(solid, loop.face.shell);
  const newLoop = makeLoop(solid, face);
  face.loops.push(newLoop);
  newLoop.halfEdge = heB;
  heA.loop = loop;
  loop.halfEdge = heA;
  for (const h of loopHalfEdges(newLoop)) h.loop = newLoop;

  return { edge, face, loop: newLoop, halfEdge: heA, twin: heB };
}

// KEF — Kill Edge and Face, the exact inverse of MEF. Removes the edge and
// merges the two faces it separates. `he`'s own face survives; the twin's
// face is killed.
//
// Refuses honestly (rather than guessing) when the doomed face carries
// rings of its own: re-homing those rings is a real decision, not an
// implementation detail, and no caller in Phase 1 needs it.
export function kef(he) {
  const solid = he.solid;
  const t = he.twin;
  if (t.loop === he.loop) throw new Error(`brep kef: edge ${he.edge.id} has both half-edges in one loop (a spur — use kev or kemr)`);
  const keepLoop = he.loop;
  const deadLoop = t.loop;
  const deadFace = deadLoop.face;
  if (deadFace.loops.length !== 1) throw new Error(`brep kef: face ${deadFace.id} carries ${deadFace.loops.length} loops — its rings would have to be re-homed, which this operator will not guess at`);
  if (deadLoop !== deadFace.loops[0]) throw new Error('brep kef: the doomed half-edge is not on its face\'s outer loop');
  if (keepLoop.face === deadFace) throw new Error('brep kef: both half-edges already belong to the same face');

  const hePrev = he.prev, heNext = he.next, tPrev = t.prev, tNext = t.next;
  if (heNext === he || tNext === t) throw new Error(`brep kef: edge ${he.edge.id} is a whole loop by itself — nothing would survive the merge`);
  hePrev.next = tNext; tNext.prev = hePrev;
  tPrev.next = heNext; heNext.prev = tPrev;
  keepLoop.halfEdge = heNext;
  for (const h of loopHalfEdges(keepLoop)) h.loop = keepLoop;

  removeFrom(deadFace.loops, deadLoop);
  removeFrom(deadFace.shell.faces, deadFace);
  removeFrom(solid.edges, he.edge);
  repointVertexHalfEdges(solid);
  return { solid };
}

// KEMR — Kill Edge, Make Ring. Removes a bridge edge whose two half-edges
// both lie in ONE loop, splitting that loop into two loops on the SAME
// face; the new one is a RING (an inner boundary).
//   E-1, L+1 (R+1)  ->  lhs +1 -1 = 0 ... careful: lhs = V-E+F-R, so
//   -(-1) - (+1) = +1 - 1 = 0 ; rhs unchanged   OK
//
// The cycle reachable by following `he.next` becomes the new RING; the
// face's existing loop keeps the other cycle. That direction is the
// caller's to choose by which half-edge it hands in, and is documented
// because the inverse (MEKR) depends on it.
export function kemr(he) {
  const solid = he.solid;
  const t = he.twin;
  const loop = he.loop;
  if (t.loop !== loop) throw new Error(`brep kemr: edge ${he.edge.id} spans two loops (use kef)`);
  if (he.next === t || t.next === he) throw new Error(`brep kemr: edge ${he.edge.id} is a valence-1 spur — killing it makes no ring (use kev)`);
  const face = loop.face;

  const c1Start = he.next, c1End = t.prev;   // becomes the ring
  const c2Start = t.next,  c2End = he.prev;  // stays with the original loop
  c1End.next = c1Start; c1Start.prev = c1End;
  c2End.next = c2Start; c2Start.prev = c2End;

  const ring = makeLoop(solid, face);
  ring.halfEdge = c1Start;
  face.loops.push(ring);
  for (const h of loopHalfEdges(ring)) h.loop = ring;
  loop.halfEdge = c2Start;
  for (const h of loopHalfEdges(loop)) h.loop = loop;

  removeFrom(solid.edges, he.edge);
  repointVertexHalfEdges(solid);
  // `keepHalfEdge`/`ringHalfEdge` are exactly the two arguments MEKR needs to
  // undo this call (see MEKR's own splice): handing them back is what makes
  // the inverse mechanical rather than a re-derivation by the caller.
  //
  // `killedHalfEdge`/`killedTwin` are handed back for a subtler reason, found
  // the hard way while driving a full reverse teardown: KEMR is the one
  // forward operator that DESTROYS an edge some EARLIER operator created
  // (typically an MEV spur). MEKR re-creates that edge as a NEW object, so
  // any handle a caller recorded when the spur was first made goes stale the
  // moment KEMR runs. A caller replaying a construction backwards has to
  // re-bind those handles, and these two fields are what it re-binds FROM.
  return { solid, ring, loop, keepHalfEdge: c2Start, ringHalfEdge: c1Start, killedHalfEdge: he, killedTwin: t };
}

// MEKR — Make Edge, Kill Ring, the exact inverse of KEMR. Bridges a ring
// back into another loop of the same face with a new edge, dissolving the
// ring. `heA` is in the surviving loop, `heB` is in the ring being absorbed.
export function mekr(heA, heB) {
  const solid = heA.solid;
  const keepLoop = heA.loop, ringLoop = heB.loop;
  if (keepLoop === ringLoop) throw new Error('brep mekr: both half-edges are already in the same loop');
  const face = keepLoop.face;
  if (ringLoop.face !== face) throw new Error('brep mekr: the two loops belong to different faces');
  if (face.loops[0] === ringLoop) throw new Error(`brep mekr: loop ${ringLoop.id} is face ${face.id}'s OUTER loop, not a ring — absorbing it would leave the face without an outer boundary`);

  const v1 = heA.vertex, v2 = heB.vertex;
  const pA = heA.prev, pB = heB.prev;
  const { edge, h1, h2 } = makeEdgePair(solid, v1, v2); // h1: v1 -> v2, h2: v2 -> v1

  pA.next = h1; h1.prev = pA;
  h1.next = heB; heB.prev = h1;
  pB.next = h2; h2.prev = pB;
  h2.next = heA; heA.prev = h2;

  h1.loop = keepLoop; h2.loop = keepLoop;
  keepLoop.halfEdge = heA;
  for (const h of loopHalfEdges(keepLoop)) h.loop = keepLoop;
  removeFrom(face.loops, ringLoop);
  return { edge, halfEdge: h1, twin: h2 };
}

// KFMRH — Kill Face, Make Ring and Hole. The genus-raising operator: the
// doomed face's outer loop becomes a RING of `face`, and the solid gains a
// through-hole. Combinatorially this is what "punches the tunnel through".
//   F-1, L unchanged (so R+1), G+1
//   ->  lhs: -(F 1) - (R +1) = -1 -1 = -2 ; rhs: 2*(0 - 1) = -2   OK
//
// Geometric sanity (the loops actually coinciding in space) is NOT checked
// here — that is the geometric half of validity, deliberately out of Phase
// 1 scope, and stated so rather than implied.
export function kfmrh(face, deadFace) {
  const solid = face.solid;
  if (face === deadFace) throw new Error('brep kfmrh: face and deadFace must be different');
  if (face.shell !== deadFace.shell) throw new Error('brep kfmrh: both faces must be in the same shell — a handle is within one shell');
  if (deadFace.loops.length !== 1) throw new Error(`brep kfmrh: face ${deadFace.id} carries ${deadFace.loops.length} loops — only a ring-free face can become a ring`);

  const ring = deadFace.loops[0];
  removeFrom(deadFace.loops, ring);
  removeFrom(deadFace.shell.faces, deadFace);
  ring.face = face;
  face.loops.push(ring);
  solid.genus += 1;
  return { solid, ring };
}

// MFKRH — Make Face, Kill Ring and Hole, the exact inverse. A ring is
// promoted back out into a face of its own and the genus drops.
export function mfkrh(face, ring) {
  const solid = face.solid;
  if (face.loops[0] === ring) throw new Error('brep mfkrh: that is the face\'s outer loop, not a ring');
  if (!face.loops.includes(ring)) throw new Error('brep mfkrh: ring does not belong to that face');
  if (solid.genus <= 0) throw new Error('brep mfkrh: solid genus is already 0 — nothing to un-punch');
  removeFrom(face.loops, ring);
  const newFace = makeFace(solid, face.shell);
  ring.face = newFace;
  newFace.loops.push(ring);
  solid.genus -= 1;
  return { face: newFace, loop: ring };
}

// A vertex's stored outgoing half-edge can be one that just got deleted.
// Repoint any such vertex at a surviving outgoing half-edge, or null it out
// if it now has none. Derived by scanning the edge list rather than
// maintained incrementally: at these sizes the scan is free, and a derived
// answer cannot drift the way an incrementally-patched one can.
function repointVertexHalfEdges(solid) {
  const live = new Set();
  const byVertex = new Map();
  for (const he of allHalfEdges(solid)) {
    live.add(he);
    if (!byVertex.has(he.vertex)) byVertex.set(he.vertex, he);
  }
  for (const v of solid.vertices) {
    if (v.halfEdge && live.has(v.halfEdge) && v.halfEdge.vertex === v) continue;
    v.halfEdge = byVertex.get(v) ?? null;
  }
}

// NOT AN EULER OPERATOR — plain bookkeeping, labeled as such. Moves every
// shell of `donor` into `host`, so a solid with an internal void (or two
// disjoint lumps) can be represented and checked. The invariant survives by
// plain addition and nothing else:
//   (V1+V2) - (E1+E2) + (F1+F2) - (R1+R2)
//     = [2(S1-G1) + ... ] ... = 2((S1+S2) - (G1+G2))
// i.e. shells and genus both simply add, which is why this needs no special
// handling. Ids are renumbered into the host's own sequence.
export function mergeSolidShells(host, donor) {
  if (host === donor) throw new Error('brep mergeSolidShells: host and donor are the same solid');
  const renumber = (ent) => { ent.id = newId(host); ent.solid = host; };
  for (const v of donor.vertices) { renumber(v); host.vertices.push(v); }
  for (const e of donor.edges) { renumber(e); host.edges.push(e); for (const he of e.halfEdges) renumber(he); }
  for (const sh of donor.shells) {
    renumber(sh);
    for (const f of sh.faces) { renumber(f); for (const l of f.loops) renumber(l); }
    host.shells.push(sh);
  }
  host.genus += donor.genus;
  donor.vertices = []; donor.edges = []; donor.shells = []; donor.genus = 0;
  return host;
}

// ---------------------------------------------------------------------
// VALIDITY
// ---------------------------------------------------------------------
// Every error carries a stable `code` so a test can assert a specific
// failure mode BY NAME rather than by substring-matching prose. The codes
// map onto the failure classes this project has actually hit in its own
// topology work before now (see kernel/subdedit.mjs's own review history:
// repeated-vertex faces and 3+-face edges are both real, shipped-and-caught
// bugs there, not hypotheticals).
//
//   'unclosed-loop'          a loop's next/prev cycle does not close
//   'empty-loop'             a loop with no half-edges (legal only mid-build)
//   'non-manifold-edge'      an edge used by other than exactly two half-edges
//   'twin-mismatch'          he.twin.twin !== he, or the twins disagree on their edge
//   'inconsistent-winding'   he.twin's origin is not he's destination
//   'repeated-vertex-in-loop' a loop visits one vertex more than once
//   'dangling-edge'          both half-edges of an edge in the same loop (a spur)
//   'orphan-halfedge'        a half-edge not reachable from its own loop, or
//                            reachable from a loop but not registered on an edge
//   'orphan-vertex'          a vertex with no incident half-edge, or one
//                            referenced by a half-edge but absent from the solid
//   'orphan-edge'            an edge whose half-edges sit in no loop
//   'stale-vertex-halfedge'  vertex.halfEdge does not start at that vertex
//   'face-without-loops'     a face carrying zero loops
//   'broken-ownership'       an entity's `solid`/parent back-pointer disagrees
//                            with the chain it is actually reachable through
//   'euler-poincare'         the invariant does not hold
//
// `allowIntermediate: true` suppresses exactly the three codes that are
// legitimate DURING a construction and only wrong at the end —
// 'empty-loop', 'dangling-edge', and the 'repeated-vertex-in-loop' that a
// dangling edge necessarily causes. Everything else is checked either way.
export function validateBrep(solid, opts = {}) {
  const allowIntermediate = opts.allowIntermediate === true;
  const errors = [];
  const err = (code, message, extra = {}) => errors.push({ code, message, ...extra });

  if (!solid || !Array.isArray(solid.vertices) || !Array.isArray(solid.edges) || !Array.isArray(solid.shells)) {
    return { ok: false, errors: [{ code: 'broken-ownership', message: 'not a brep solid: missing vertices/edges/shells arrays' }] };
  }

  const faces = allFaces(solid);
  const vertexSet = new Set(solid.vertices);
  const edgeSet = new Set(solid.edges);

  // --- edges + half-edge pairing -------------------------------------
  const registeredHalfEdges = new Set();
  for (const edge of solid.edges) {
    if (edge.solid !== solid) err('broken-ownership', `edge ${edge.id} points at a different solid`, { edgeId: edge.id });
    if (!Array.isArray(edge.halfEdges) || edge.halfEdges.length !== 2) {
      err('non-manifold-edge', `edge ${edge.id} is used by ${edge.halfEdges ? edge.halfEdges.length : 0} half-edges — a manifold edge has exactly two`, { edgeId: edge.id, uses: edge.halfEdges ? edge.halfEdges.length : 0 });
    }
    for (const he of edge.halfEdges || []) {
      registeredHalfEdges.add(he);
      if (he.edge !== edge) err('twin-mismatch', `half-edge ${he.id} is listed on edge ${edge.id} but points at edge ${he.edge ? he.edge.id : 'null'}`, { edgeId: edge.id, halfEdgeId: he.id });
      if (!he.twin) err('twin-mismatch', `half-edge ${he.id} has no twin`, { halfEdgeId: he.id });
      else {
        if (he.twin.twin !== he) err('twin-mismatch', `half-edge ${he.id}.twin.twin is not itself`, { halfEdgeId: he.id });
        if (he.twin.edge !== he.edge) err('twin-mismatch', `half-edge ${he.id} and its twin disagree on their edge`, { halfEdgeId: he.id });
      }
      if (!he.vertex) err('orphan-vertex', `half-edge ${he.id} has no origin vertex`, { halfEdgeId: he.id });
      else if (!vertexSet.has(he.vertex)) err('orphan-vertex', `half-edge ${he.id} originates at vertex ${he.vertex.id}, which is not in the solid`, { halfEdgeId: he.id, vertexId: he.vertex.id });
    }
  }

  // --- loops -----------------------------------------------------------
  const seenInLoops = new Set();
  for (const face of faces) {
    if (face.solid !== solid) err('broken-ownership', `face ${face.id} points at a different solid`, { faceId: face.id });
    if (!face.shell || !solid.shells.includes(face.shell)) err('broken-ownership', `face ${face.id}'s shell is not in the solid`, { faceId: face.id });
    else if (!face.shell.faces.includes(face)) err('broken-ownership', `face ${face.id} is not listed on the shell it points at`, { faceId: face.id });
    if (!Array.isArray(face.loops) || face.loops.length === 0) {
      err('face-without-loops', `face ${face.id} carries no loops`, { faceId: face.id });
      continue;
    }
    for (const loop of face.loops) {
      if (loop.face !== face) err('broken-ownership', `loop ${loop.id} points at face ${loop.face ? loop.face.id : 'null'} but is listed on face ${face.id}`, { loopId: loop.id, faceId: face.id });
      const walk = walkLoopSafe(loop);
      if (!walk.ok) {
        err('unclosed-loop', `loop ${loop.id} (face ${face.id}) is not a closed cycle — ${walk.reason}`, { loopId: loop.id, faceId: face.id });
        for (const he of walk.halfEdges) seenInLoops.add(he);
        continue;
      }
      if (walk.empty) {
        if (!allowIntermediate) err('empty-loop', `loop ${loop.id} (face ${face.id}) holds no half-edges`, { loopId: loop.id, faceId: face.id });
        continue;
      }
      const vertexCounts = new Map();
      for (const he of walk.halfEdges) {
        seenInLoops.add(he);
        if (he.loop !== loop) err('orphan-halfedge', `half-edge ${he.id} is in loop ${loop.id}'s cycle but claims loop ${he.loop ? he.loop.id : 'null'}`, { halfEdgeId: he.id, loopId: loop.id });
        if (!he.edge || !edgeSet.has(he.edge)) err('orphan-halfedge', `half-edge ${he.id} is in a loop but its edge is not registered on the solid`, { halfEdgeId: he.id, loopId: loop.id });
        if (he.vertex) vertexCounts.set(he.vertex, (vertexCounts.get(he.vertex) ?? 0) + 1);
        // Orientation: the twin must start where this half-edge ends. This is
        // exactly the combinatorial statement of "adjacent faces are wound
        // consistently" — each edge traversed once in each direction.
        const dest = destinationOf(he);
        if (he.twin && dest && he.twin.vertex !== dest) {
          err('inconsistent-winding', `half-edge ${he.id} ends at vertex ${dest.id} but its twin starts at vertex ${he.twin.vertex ? he.twin.vertex.id : 'null'} — the two faces sharing edge ${he.edge ? he.edge.id : '?'} traverse it the same way`, { halfEdgeId: he.id, loopId: loop.id, faceId: face.id });
        }
      }
      if (!allowIntermediate) {
        for (const [v, n] of vertexCounts) {
          if (n > 1) err('repeated-vertex-in-loop', `loop ${loop.id} (face ${face.id}) visits vertex ${v.id} ${n} times`, { loopId: loop.id, faceId: face.id, vertexId: v.id, times: n });
        }
      }
    }
  }

  // --- reachability both directions -----------------------------------
  for (const he of registeredHalfEdges) {
    if (!seenInLoops.has(he)) err('orphan-halfedge', `half-edge ${he.id} (edge ${he.edge ? he.edge.id : '?'}) is not reachable from any loop`, { halfEdgeId: he.id });
  }
  for (const he of seenInLoops) {
    if (!registeredHalfEdges.has(he)) err('orphan-halfedge', `half-edge ${he.id} is reachable from a loop but not listed on any registered edge`, { halfEdgeId: he.id });
  }
  for (const edge of solid.edges) {
    const inAnyLoop = (edge.halfEdges || []).some((he) => seenInLoops.has(he));
    if (!inAnyLoop) err('orphan-edge', `edge ${edge.id} has no half-edge in any loop`, { edgeId: edge.id });
  }

  // --- dangling edges (spurs) ------------------------------------------
  if (!allowIntermediate) {
    for (const edge of solid.edges) {
      const [a, b] = edge.halfEdges || [];
      if (a && b && a.loop && a.loop === b.loop) {
        err('dangling-edge', `edge ${edge.id} has both half-edges in loop ${a.loop.id} — a dangling spur, not part of a closed face boundary`, { edgeId: edge.id, loopId: a.loop.id });
      }
    }
  }

  // --- vertices ---------------------------------------------------------
  const originOf = new Map();
  for (const he of registeredHalfEdges) {
    if (!he.vertex) continue;
    if (!originOf.has(he.vertex)) originOf.set(he.vertex, []);
    originOf.get(he.vertex).push(he);
  }
  for (const v of solid.vertices) {
    if (v.solid !== solid) err('broken-ownership', `vertex ${v.id} points at a different solid`, { vertexId: v.id });
    const outgoing = originOf.get(v) ?? [];
    if (solid.edges.length > 0 && outgoing.length === 0) {
      err('orphan-vertex', `vertex ${v.id} has no incident half-edge`, { vertexId: v.id });
    }
    if (v.halfEdge) {
      if (!registeredHalfEdges.has(v.halfEdge)) err('stale-vertex-halfedge', `vertex ${v.id}.halfEdge is not a live half-edge of this solid`, { vertexId: v.id });
      else if (v.halfEdge.vertex !== v) err('stale-vertex-halfedge', `vertex ${v.id}.halfEdge starts at vertex ${v.halfEdge.vertex ? v.halfEdge.vertex.id : 'null'}`, { vertexId: v.id });
    } else if (outgoing.length > 0) {
      err('stale-vertex-halfedge', `vertex ${v.id} has ${outgoing.length} incident half-edges but stores none`, { vertexId: v.id });
    }
  }

  // --- the invariant ----------------------------------------------------
  const e = eulerCharacteristic(solid);
  if (!e.ok) {
    err('euler-poincare', `V-E+F-R = ${e.lhs} but 2(S-G) = ${e.rhs} (V=${e.V} E=${e.E} F=${e.F} L=${e.L} R=${e.R} S=${e.S} G=${e.G})`, { euler: e });
  }

  return { ok: errors.length === 0, errors, euler: e };
}

export function hasErrorCode(result, code) {
  return result.errors.some((x) => x.code === code);
}

// ---------------------------------------------------------------------
// STRUCTURAL FINGERPRINT
// ---------------------------------------------------------------------
// A canonical, ID-FREE description of a solid's topology, used to prove
// that an operator's inverse genuinely restored the prior structure.
//
// WHY ID-FREE, stated rather than quietly assumed: three of the five
// inverses (MEKR, MFKRH, and MEV's own re-creation of an edge) necessarily
// CREATE an entity where the forward operator destroyed one, so the
// restored entity is a new object with a fresh id. Structure is restored;
// object identity is not, and cannot be without an id-recycling scheme this
// module deliberately does not have. Comparing canonical structure is the
// honest strong claim; the tests additionally assert exact id-set equality
// for the pairs where it IS achievable (MEV/KEV, MEF/KEF, MVFS/KVFS).
//
// Canonicalisation keys vertices by their 3D point, so it requires distinct
// vertex positions — true of every fixture here, and asserted rather than
// assumed (throws otherwise instead of returning a silently ambiguous key).
export function brepFingerprint(solid) {
  const key = (v) => (v.point ? v.point.map((c) => (Object.is(c, -0) ? 0 : c).toFixed(9)).join(',') : `#${v.id}`);
  const seen = new Map();
  for (const v of solid.vertices) {
    const k = key(v);
    if (seen.has(k)) throw new Error(`brep fingerprint: vertices ${seen.get(k).id} and ${v.id} share the position ${k} — canonicalisation would be ambiguous`);
    seen.set(k, v);
  }
  const order = [...solid.vertices].map(key).sort();
  const index = new Map(order.map((k, i) => [k, i]));
  const vi = (v) => index.get(key(v));

  const loopWord = (loop) => {
    const hes = walkLoopSafe(loop);
    if (!hes.ok) return `BROKEN(${hes.reason})`;
    if (hes.empty) return 'EMPTY';
    const seq = hes.halfEdges.map((he) => vi(he.vertex));
    // rotate so the cycle starts at its smallest index (a cycle has no
    // inherent start; the direction is meaningful and is preserved)
    let best = 0;
    for (let i = 1; i < seq.length; i++) if (seq[i] < seq[best]) best = i;
    return seq.slice(best).concat(seq.slice(0, best)).join('-');
  };

  const faceWords = allFaces(solid).map((f) => {
    const outer = loopWord(f.loops[0]);
    const rings = f.loops.slice(1).map(loopWord).sort();
    return `F[${outer}]${rings.length ? `R{${rings.join('|')}}` : ''}`;
  }).sort();

  const shellWords = solid.shells.map((sh) => sh.faces.length).sort((a, b) => a - b).join(',');
  const e = eulerCharacteristic(solid);
  return `V${e.V}E${e.E}F${e.F}L${e.L}S${e.S}G${e.G};shells(${shellWords});${faceWords.join(';')}`;
}

// ---------------------------------------------------------------------
// TRIMMED SURFACE BRIDGE
// ---------------------------------------------------------------------
// Faces REFERENCE the surface model this kernel already has; nothing here
// re-invents it. A face carries a plain NurbsSrf in `face.surface` (the
// same shape kernel/surface.mjs evaluates and kernel/trimtess.mjs
// tessellates), and its trim loops are DERIVED from the half-edge cycles'
// own pcurves rather than stored a second time — so the topology and the
// trim boundary cannot drift apart, which is the whole reason Weiler's
// coedge lives on the half-edge in the first place.
//
// The derived shape is exactly `{ srf, trimLoop, trimHoles }` — the shape
// kernel/trimtess.mjs's own tessellateTrimmedSurface(srf, loop, uRes, vRes,
// holes) already takes, and the same field names the app's existing
// Trim/Split containers already use.

export function attachSurface(face, srf) {
  face.surface = srf;
  return face;
}

// Concatenate one loop's pcurves into a single closed UV polyline. Each
// half-edge's pcurve runs from its own origin to its own destination and
// shares its last point with the next half-edge's first, so the joint point
// is dropped once per half-edge (the loop is implicitly closed, matching
// kernel/trim.mjs's own convention exactly).
export function loopUVPolyline(loop) {
  const hes = loopHalfEdges(loop);
  const out = [];
  for (const he of hes) {
    if (!he.pcurve || he.pcurve.length < 2) {
      throw new Error(`brep loopUVPolyline: half-edge ${he.id} has no pcurve (needs at least 2 [u,v] points)`);
    }
    for (let i = 0; i < he.pcurve.length - 1; i++) out.push([he.pcurve[i][0], he.pcurve[i][1]]);
  }
  return out;
}

export function faceTrimmedSurface(face) {
  if (!face.surface) throw new Error(`brep faceTrimmedSurface: face ${face.id} has no surface attached`);
  return {
    srf: face.surface,
    trimLoop: loopUVPolyline(face.loops[0]),
    trimHoles: face.loops.slice(1).map(loopUVPolyline),
  };
}

// Run the EXISTING trim-loop validity gate (kernel/trim.mjs) against a
// face's derived loops — outer/hole winding, self-intersection. This is the
// geometric half of validity for a single face, reusing what is already
// built and tested rather than a second implementation.
export function faceTrimValidity(face) {
  const { trimLoop, trimHoles } = faceTrimmedSurface(face);
  return trimLoopsValid([trimLoop, ...trimHoles]);
}

// Convenience for the common planar case: derive every half-edge's pcurve
// in a face from its endpoints via a caller-supplied 3D->UV map. Exact for
// a face whose surface parametrization is affine in the face's own plane
// (every planar B-rep face this app produces); NOT a general projection,
// and deliberately not pretending to be one — a curved face's pcurve is a
// real curve, not a two-point segment, and belongs to whichever operation
// actually produced that edge (Phase 2's intersection curves, via
// `pcurvesFromSSISamples` below).
export function assignPlanarPcurves(face, pointToUV) {
  for (const loop of face.loops) {
    for (const he of loopHalfEdges(loop)) {
      const a = he.vertex.point, b = destinationOf(he).point;
      he.pcurve = [pointToUV(a), pointToUV(b)];
    }
  }
  return face;
}

// Adapter for kernel/ssi.mjs's own documented return shape. `intersectSurfaces`
// yields samples of `{ u1, v1, u2, v2, point }` — one shared 3D curve plus
// its two pcurves, one per surface. That is EXACTLY the data an Edge and
// its two half-edges need: `curve3d` on the edge, `pcurve` on each side.
// Pure and shape-only — this does not itself run an intersection, and Phase
// 2 owns everything about when and whether an intersection is trustworthy.
export function pcurvesFromSSISamples(samples) {
  if (!Array.isArray(samples) || samples.length < 2) {
    throw new Error('brep pcurvesFromSSISamples: need at least 2 SSI samples');
  }
  return {
    curve3d: samples.map((s) => [s.point[0], s.point[1], s.point[2]]),
    uv1: samples.map((s) => [s.u1, s.v1]),
    uv2: samples.map((s) => [s.u2, s.v2]),
  };
}

// ---------------------------------------------------------------------
// COMPOSITE CONSTRUCTIONS
// ---------------------------------------------------------------------
// Built ENTIRELY out of the Euler operators above — no direct pointer
// surgery — so they double as proof that the operator set is complete
// enough to reach real solids.

// `opts.onStep(label, solid, result)` — called after every operator these
// composites apply, with that operator's own return value, so a caller can
// journal the sequence and drive an exact reverse teardown (which is what
// test/brep.test.mjs does to prove the inverses).
//
// Sweep every vertex of `loop` to a new position and wall the result in:
// MEV per vertex, then MEF per wall. This is the standard Euler-operator
// "extrude a face" and it is what turns a two-face lamina into a box.
// Returns the new vertices in loop order.
export function sweepLoop(loop, offsetOf, opts = {}) {
  const onStep = opts.onStep ?? null;
  const base = loopHalfEdges(loop).map((he) => he.vertex);
  const grown = [];
  for (const v of base) {
    const at = uniqueHalfEdgeInLoopFrom(loop, v, 'sweepLoop');
    const r = mev(loop, v, offsetOf(v), { before: at });
    grown.push(r.vertex);
    if (onStep) onStep('mev', loop.solid, r);
  }
  for (let i = 0; i < grown.length; i++) {
    const a = grown[i], b = grown[(i + 1) % grown.length];
    const he1 = uniqueHalfEdgeInLoopFrom(loop, a, 'sweepLoop');
    const he2 = uniqueHalfEdgeInLoopFrom(loop, b, 'sweepLoop');
    const m = mef(he1, he2);
    if (onStep) onStep('mef', loop.solid, m);
  }
  return grown;
}

// Cut a ring (an inner boundary) into `loop`'s face, tracing `points`:
// a spur out from `fromVertex`, the ring traced along it, MEF to close the
// ring into its own face, then KEMR to dissolve the spur and leave the ring
// behind. Returns { ring, capFace, capLoop, vertices }.
export function cutRing(loop, fromVertex, points, opts = {}) {
  const onStep = opts.onStep ?? null;
  const solid = loop.solid;
  if (points.length < 3) throw new Error('brep cutRing: a ring needs at least 3 points');
  const at = opts.before ?? uniqueHalfEdgeInLoopFrom(loop, fromVertex, 'cutRing');
  const spur = mev(loop, fromVertex, points[0], { before: at });
  if (onStep) onStep('mev', solid, spur);
  const verts = [spur.vertex];
  let cur = spur.vertex;
  for (let i = 1; i < points.length; i++) {
    const r = mev(loop, cur, points[i]);
    if (onStep) onStep('mev', solid, r);
    verts.push(r.vertex);
    cur = r.vertex;
  }
  // Close the ring: he1 leaves the FIRST ring vertex outward along the
  // traced path, he2 leaves the LAST one back along it.
  const he1 = halfEdgesInLoopFrom(loop, verts[0]).find((h) => destinationOf(h) === verts[1]);
  const he2 = halfEdgesInLoopFrom(loop, verts[verts.length - 1]).find((h) => destinationOf(h) === verts[verts.length - 2]);
  if (!he1 || !he2) throw new Error('brep cutRing: could not locate the traced ring path');
  const made = mef(he1, he2);
  if (onStep) onStep('mef', solid, made);
  const spurHe = spur.halfEdge;
  const res = kemr(spurHe);
  if (onStep) onStep('kemr', solid, res);
  return { ring: res.ring, capFace: made.face, capLoop: made.loop, vertices: verts };
}
