// SuperB CAGE TOPOLOGY EDITS — EXTRUDESUBD / INSERTEDGE / BRIDGE / STITCH /
// DELETE FACES / FILLSUBDHOLE (milestones 4 and 6, plus the
// v1-listed DELETE FACES/FILLSUBDHOLE pair, building on the already-shipped
// sub-object selection milestone: kernel/subdselect.mjs's edgeLoopFromSeed/
// edgeRingFromSeed and kernel/subd.mjs's own buildTopology/edgeKey). KERNEL
// ONLY: pure cage-in, cage-out topology surgery, no app-layer object/UI/undo
// here, matching every other kernel/*.mjs module's own discipline exactly.
//
// EXTRUDESUBD/INSERTEDGE/BRIDGE grow the cage (new vertices/edges/faces);
// DELETE FACES shrinks it (removes faces, prunes any now-orphaned vertex);
// STITCH welds two separate boundary vertex chains into one (net vertex
// count shrinks, but no faces are removed — the two chains' own edges become
// shared interior edges instead); FILLSUBDHOLE adds exactly one new face,
// zero new vertices. None of these ever mutate the input cage — every
// function here returns a brand-new cage object, matching
// subdivideCatmullClark's own convention.

function distSq(a, b) { const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2]; return dx * dx + dy * dy + dz * dz; }

import { add, sub, scale, length, cross, dot } from './vec3.mjs';
import { edgeKey, buildTopology, subdivideCatmullClark } from './subd.mjs';
import { oppositeEdgeInFace } from './subdselect.mjs';
import { cubicHermiteSegment } from './interpolate.mjs';

function lerp3(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }

// BRIDGE CREASE — Rhino's own fourth Bridge Option. What it creases is the RIM,
// not the rungs: the edges where the new wall MEETS the existing surface, so the
// bridge reads as a distinct tube joining two forms rather than blending
// smoothly into both. Creasing the rungs instead would crease the wall's own
// length, which is not what the option means.
//
// Takes a WEIGHT rather than a boolean so the kernel stays free of any app-side
// "how hard is a hard crease" constant, and so a partial crease is available for
// free — the same 0..N semi-sharp range this kernel's own crease machinery
// already carries everywhere else. 0 means untouched: no key is written at all,
// so the DEFAULT is byte-identical to a Bridge built before this existed.
function creaseChain(creases, seq, weight, closed) {
  if (!(weight > 0)) return;
  const n = seq.length;
  const last = closed ? n : n - 1;
  for (let i = 0; i < last; i++) creases[edgeKey(seq[i], seq[(i + 1) % n])] = weight;
}

// ── BRIDGE STRAIGHTNESS ──────────────────────────────────────────────────
// Rhino's own Bridge Options carries a STRAIGHTNESS percentage alongside
// Segments, and it is the whole difference between a bridge that reads as a
// faceted sleeve and one that flows out of the surrounding surface: at 100%
// a rung's interior rows are a plain straight lerp between the two rims
// (what this module built before straightness existed); at 0% they follow a
// cubic that leaves each rim along that rim's OWN outgoing surface
// direction, so the tunnel meets the cage tangentially instead of at a
// crease. Anything between is a plain blend of the two.
//
// NO NEW CURVE MATH: the cubic is `cubicHermiteSegment` (kernel/
// interpolate.mjs), already node-tested with exact-endpoint, analytic-tangent
// and finite-difference proofs from the internal-B-handles round — reused
// here, only evaluated at a parameter rather than handed to a renderer. It
// returns a degree-3 Bezier whose single span is [0,1], so evaluating it is
// the plain cubic Bernstein basis, no knot machinery needed.
//
// TWO IDENTITIES THIS GUARANTEES BY CONSTRUCTION, asserted rather than
// assumed in this module's own test file: straightness 1 returns the lerp
// BIT-IDENTICALLY (an early return, not an arithmetic coincidence), and at
// segments=1 there are no interior rows at all, so straightness cannot
// change the result by any value — which is exactly why Rhino defaults
// Segments to 2.
function bridgeSpanPoint(pA, pB, mA, mB, t, straightness) {
  const straight = lerp3(pA, pB, t);
  if (straightness >= 1 || !mA || !mB) return straight;
  const b = cubicHermiteSegment(pA, pB, mA, mB).ctrlPts;
  const u = 1 - t;
  const w0 = u * u * u, w1 = 3 * u * u * t, w2 = 3 * u * t * t, w3 = t * t * t;
  const herm = [0, 1, 2].map((k) => b[0][k] * w0 + b[1][k] * w1 + b[2][k] * w2 + b[3][k] * w3);
  return lerp3(straight, herm, 1 - straightness);
}

// THE OUTGOING SURFACE DIRECTION AT ONE RIM VERTEX — the same "read it off
// the owner face's own loop" trick this module's own winding already relies
// on, rather than a face normal or a guessed axis. A rim vertex sits on
// exactly one naked edge pair of its own chain; its owner face gives it one
// neighbor ALONG the rim (the partner passed in) and one INWARD, into the
// surface the rim bounds. `vertex - inward` therefore points away from that
// surface, along the direction the surface itself is traveling as it
// reaches the rim — which is precisely the tangent a tangent-continuous
// bridge has to leave on.
//
// Returns null for a genuinely degenerate neighborhood (a zero-length
// inward edge). Callers fall back to the straight chord there, so a
// degenerate cage produces a plain straight rung rather than a NaN — an
// honest degrade, matching this module's own posture everywhere else.
function rimOutgoingDirection(cage, topology, vIdx, partnerIdx) {
  const edge = topology.edgeMap.get(edgeKey(vIdx, partnerIdx));
  if (!edge || edge.faces.length !== 1) return null;
  const face = cage.faces[edge.faces[0]];
  const n = face.length;
  const c = face.indexOf(vIdx);
  if (c < 0 || n < 3) return null;
  const prev = face[(c - 1 + n) % n], next = face[(c + 1) % n];
  const inward = prev === partnerIdx ? next : prev;
  const d = sub(cage.vertices[vIdx], cage.vertices[inward]);
  return length(d) < 1e-12 ? null : d;
}

// Per-vertex outgoing tangents for a whole rim, scaled to that rung's own
// chord length (the standard Hermite magnitude choice — a tangent scaled to
// the span it crosses produces a blend proportional to the gap, so a narrow
// bridge bulges less than a wide one for the same straightness value).
// `closed` distinguishes a Bridge rim (a closed loop, every vertex has a
// rim partner on both sides) from an open edge RUN (the last vertex's only
// rim partner is behind it).
function rimTangents(cage, topology, seq, otherPts, closed, sign) {
  const n = seq.length;
  const out = [];
  for (let i = 0; i < n; i++) {
    const partner = closed ? seq[(i + 1) % n] : (i + 1 < n ? seq[i + 1] : seq[i - 1]);
    let d = rimOutgoingDirection(cage, topology, seq[i], partner);
    if (!d) { out.push(null); continue; }
    const L = Math.sqrt(distSq(cage.vertices[seq[i]], otherPts[i]));
    const m = length(d);
    out.push(scale(d, (sign * L) / m));
  }
  return out;
}

// A consistently-wound 2-manifold never traverses the SAME directed edge
// twice — the standard property this module's own test file already checks
// the finished cage against. Used here to DECIDE an orientation rather than
// verify one after the fact: a candidate rung set that would traverse any
// directed edge a second time is simply not the correct winding, so the
// choice needs no hand-derived formula at all.
function directedEdgeReuseCount(faces) {
  const seen = new Set();
  let reused = 0;
  for (const f of faces) {
    for (let c = 0; c < f.length; c++) {
      const k = `${f[c]}>${f[(c + 1) % f.length]}`;
      if (seen.has(k)) reused++;
      seen.add(k);
    }
  }
  return reused;
}

// FACE NORMAL — Newell's method (robust for a non-planar or bowed n-gon, not
// just a plain 3-point cross product), normalized. Returns [0,0,0] for a
// genuinely degenerate (zero-area) face — callers decide how to react; this
// module's own computeAverageNormal below treats an all-degenerate selection
// as a real refusal, not a silent [0,0,0] direction.
export function computeFaceNormal(cage, faceIdx) {
  const face = cage.faces[faceIdx];
  const n = face.length;
  let nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < n; i++) {
    const [x0, y0, z0] = cage.vertices[face[i]];
    const [x1, y1, z1] = cage.vertices[face[(i + 1) % n]];
    nx += (y0 - y1) * (z0 + z1);
    ny += (z0 - z1) * (x0 + x1);
    nz += (x0 - x1) * (y0 + y1);
  }
  const len = Math.hypot(nx, ny, nz);
  if (len < 1e-12) return [0, 0, 0];
  return [nx / len, ny / len, nz / len];
}

// AVERAGE NORMAL of a set of selected faces — EXTRUDESUBD's own DEFAULT
// direction ("extrude selected faces along their own average
// normal, or a picked direction"). A plain per-face-normal average (not
// area-weighted — a real, simple, honestly-stated v1 choice: area-weighting
// would need each face's own real area, extra work the
// "along their own average normal" wording doesn't demand). Throws honestly
// if the selection's own normals cancel out (e.g. two exactly opposite faces
// of a box selected together) rather than silently returning a meaningless
// zero vector — the caller (the app layer's EXTRUDESUBD command) is expected
// to require an explicit picked direction in that case.
export function computeAverageNormal(cage, faceIndices) {
  if (!faceIndices || !faceIndices.length) throw new Error('computeAverageNormal: faceIndices must be a non-empty array');
  let sum = [0, 0, 0];
  for (const fi of faceIndices) sum = add(sum, computeFaceNormal(cage, fi));
  const len = length(sum);
  if (len < 1e-9) throw new Error('computeAverageNormal: the selected faces\' own normals cancel out (e.g. two directly opposite faces) — pick an explicit direction instead');
  return scale(sum, 1 / len);
}

// EXTRUDESUBD — extrude the selected face(s) along `direction` (any nonzero
// vector; the caller normalizes it against computeAverageNormal or its own
// picked direction) by `distance` (signed — a negative distance extrudes
// INTO the surface, matching Rotate/Scale's own "any finite typed number,
// sign included" convention). The real "grow the cage" topology surgery
// the spec calls for, not a point move: boundary edges of the selected
// region (edges touching EXACTLY ONE selected face — whether the other side
// is a non-selected neighbor face or a genuine cage boundary, both count
// identically, see below) gain new connecting SIDE faces; a vertex touching
// any such boundary edge is DUPLICATED (the old copy stays behind, still
// used by whatever non-selected geometry referenced it; a new copy is
// created at `position + direction*distance` for the selected face(s) to
// use instead); a vertex used ONLY by selected faces (no boundary edge
// touches it at all) is simply TRANSLATED in place — no duplicate needed,
// since nothing outside the selection references its old position. This is
// the standard "extrude region" algorithm (the same one Blender/Maya mesh
// tools use), generalized correctly to an ARBITRARY (not necessarily
// singly-connected) set of selected faces: the boundary/interior
// classification is derived per-EDGE, never assumes the whole selection
// forms one simply-connected patch.
export function extrudeFaces(cage, faceIndices, direction, distance) {
  if (!faceIndices || !faceIndices.length) throw new Error('extrudeFaces: faceIndices must be a non-empty array');
  for (const fi of faceIndices) {
    if (!Number.isInteger(fi) || fi < 0 || fi >= cage.faces.length) throw new Error(`extrudeFaces: face index ${fi} is out of range`);
  }
  if (!Number.isFinite(distance) || Math.abs(distance) < 1e-9) throw new Error('extrudeFaces: distance must be a nonzero finite number');
  const dirLen = length(direction);
  if (!Number.isFinite(dirLen) || dirLen < 1e-9) throw new Error('extrudeFaces: direction must be a nonzero vector');
  const offset = scale(direction, distance / dirLen);

  const selectedSet = new Set(faceIndices);
  const topology = buildTopology(cage);

  // A BOUNDARY-REGION edge is one touching EXACTLY ONE selected face — this
  // covers both "shared with a non-selected neighbor face" and "a genuine
  // cage boundary edge of an open cage" identically (both need a new side
  // face + duplicated endpoints; see this function's own header). An
  // INTERIOR-region edge (touching exactly TWO selected faces) needs
  // neither — the two faces move together, the shared edge simply comes
  // along for the ride via its own endpoints' in-place translation.
  const boundaryVertexSet = new Set();
  const boundaryEdgeKeys = new Set();
  for (const [key, edge] of topology.edgeMap) {
    const selCount = edge.faces.filter((f) => selectedSet.has(f)).length;
    if (selCount === 1) {
      boundaryEdgeKeys.add(key);
      boundaryVertexSet.add(edge.v0);
      boundaryVertexSet.add(edge.v1);
    }
  }

  // DEGENERATE-DIRECTION GUARD — a real gap found while testing this
  // function against a genuinely reachable workflow (extruding the SAME
  // face a SECOND time along a direction chosen freehand, e.g. sideways
  // instead of up): if the extrude direction runs exactly PARALLEL to one
  // of the region's own boundary-region edges, that edge's own new SIDE
  // face degenerates to four COLLINEAR points (a pure in-plane slide along
  // that one edge's own line, zero area) — a real, silent violation of
  // this operation's own "no degenerate zero-area faces" requirement that
  // nothing upstream of this point can see, since every individual number
  // involved (the direction, the edge, the resulting vertices) is
  // perfectly finite; only the CROSS PRODUCT reveals the degeneracy.
  // Refused honestly and entirely (not just the one bad side face, which
  // would leave a genuine hole in the boundary instead) — the same "refuse
  // rather than guess" standard this app's own Mirror/Shear degenerate-axis
  // checks already establish.
  for (const [key, edge] of topology.edgeMap) {
    if (!boundaryEdgeKeys.has(key)) continue;
    const edgeDir = sub(cage.vertices[edge.v1], cage.vertices[edge.v0]);
    const edgeLen = length(edgeDir);
    if (edgeLen < 1e-12) continue; // a genuinely zero-length edge is the INPUT cage's own problem, not this guard's to diagnose
    const cr = length(cross(edgeDir, offset));
    if (cr < 1e-9 * edgeLen * length(offset)) {
      throw new Error(`extrudeFaces: the extrude direction runs exactly parallel to boundary edge "${key}" — that edge's own new side face would collapse to four collinear points (zero area); pick a direction with some component out of the selected face's own plane`);
    }
  }

  const newVertices = cage.vertices.map((v) => v.slice());
  const newIndexForOld = new Map();
  for (const vi of boundaryVertexSet) {
    newIndexForOld.set(vi, newVertices.length);
    newVertices.push(add(cage.vertices[vi], offset));
  }
  // Interior vertices — used by >=1 selected face, but never touch a
  // boundary-region edge — translate in place (same index, new position);
  // nothing outside the selection ever referenced the old position, so no
  // duplicate is needed (see this function's own header).
  const interiorSet = new Set();
  for (const fi of selectedSet) for (const vi of cage.faces[fi]) if (!boundaryVertexSet.has(vi)) interiorSet.add(vi);
  for (const vi of interiorSet) newVertices[vi] = add(cage.vertices[vi], offset);

  const newFaces = cage.faces.map((f) => f.slice());
  for (const fi of selectedSet) {
    newFaces[fi] = newFaces[fi].map((vi) => (boundaryVertexSet.has(vi) ? newIndexForOld.get(vi) : vi));
  }
  // One new SIDE face per boundary-region edge, wound using the ORIGINAL
  // (pre-remap) vertex order as it appears in whichever selected face owns
  // that edge — a boundary-region edge is, by construction, owned by
  // EXACTLY one selected face, so this loop visits each exactly once, no
  // dedup needed.
  const sideFaces = [];
  for (const fi of selectedSet) {
    const face = cage.faces[fi]; // original, pre-remap indices
    const n = face.length;
    for (let c = 0; c < n; c++) {
      const va = face[c], vb = face[(c + 1) % n];
      const key = edgeKey(va, vb);
      if (!boundaryEdgeKeys.has(key)) continue;
      sideFaces.push([va, vb, newIndexForOld.get(vb), newIndexForOld.get(va)]);
    }
  }

  // CREASE REMAP — a real gap, not anticipated up
  // front: a boundary-region (selCount===1) edge's crease survives this
  // surgery for free — its own new SIDE face reuses that edge's ORIGINAL,
  // pre-remap vertex pair as its own "near" edge by construction (the
  // loop just above), so the old crease key stays a genuinely valid edge
  // afterward, untouched. But an INTERIOR-region (selCount===2) edge —
  // shared by two SELECTED faces — is a different story the instant BOTH
  // its own endpoints sit on the selection's own boundary loop (and are
  // therefore duplicated above): every selected face touching it gets
  // remapped onto the NEW vertex copies, so the edge's real location in
  // the OUTPUT topology is the REMAPPED key, not the old one. Left
  // unhandled, the crease's own old key becomes a dangling entry pointing
  // at an edge that no longer exists anywhere in the new cage, while the
  // edge's real (moved) location gets no weight at all — a previously-hard
  // seam silently going smooth, with zero warning. `mapVertex` mirrors the
  // exact remap `newFaces` itself already applies a few lines up.
  const mapVertex = (vi) => (boundaryVertexSet.has(vi) ? newIndexForOld.get(vi) : vi);
  const newCreases = { ...(cage.creases || {}) };
  for (const [key, weight] of Object.entries(cage.creases || {})) {
    const edge = topology.edgeMap.get(key);
    if (!edge) continue; // a foreign/stale key already in the input — nothing real to remap
    const selCount = edge.faces.filter((f) => selectedSet.has(f)).length;
    if (selCount !== 2) continue; // boundary/exterior edges keep their own real, unchanged key
    const newKey = edgeKey(mapVertex(edge.v0), mapVertex(edge.v1));
    if (newKey === key) continue; // neither endpoint actually moved
    delete newCreases[key];
    newCreases[newKey] = weight;
  }

  return {
    cage: { vertices: newVertices, faces: [...newFaces, ...sideFaces], creases: newCreases },
    capFaceIndices: [...selectedSet],
    sideFaceIndices: sideFaces.map((_, i) => newFaces.length + i),
  };
}

// EXTRUDESUBD, THE EDGE CASE — extrude selected NAKED (boundary) edges into
// a new strip of faces. Rhino's own SubD extrude works on an edge as well as
// a face, and the two are genuinely different operations rather than one
// generalized to the other: extruding a FACE moves that face and walls in
// the gap behind it (the surface keeps the same boundary), while extruding
// an EDGE leaves every existing vertex exactly where it is and GROWS new
// surface outward from the open edge — the natural way to pull a plane or a
// hole's rim into a longer sheet.
//
// SCOPED TO NAKED EDGES, honestly and on purpose. An INTERIOR edge (two
// faces) has no free side to grow into; extruding one means TEARING the
// surface open along it and deciding which side each existing face follows —
// a real, different operation with a real, different answer, not something
// to guess at silently. Refused by name, the same standard the degenerate-
// direction guard below (and Mirror/Shear's own axis checks) already set.
//
// A CHAIN OF EDGES EXTRUDES AS ONE STRIP, not as detached quads: an endpoint
// shared by two selected edges is duplicated ONCE and reused, so the new
// faces share their rung and the result stays a single manifold surface.
// This is what makes "double-click an open edge, extrude the whole boundary
// loop" produce a collar rather than N loose flaps.
//
// `direction` may be NULL, meaning "each edge grows the way its own surface
// already points" — perpendicular to the edge, in its owner face's plane,
// away from that face. This is the only sensible default for a whole
// boundary LOOP: a single shared direction would push one side of the rim
// out and the opposite side straight through the sheet, and averaging the
// loop's own outward directions cancels to nothing. A vertex shared by two
// selected edges moves by the average of their two outward directions, which
// is what keeps the collar's own corners closed.
export function extrudeEdges(cage, edgeKeys, direction, distance) {
  if (!edgeKeys || !edgeKeys.length) throw new Error('extrudeEdges: edgeKeys must be a non-empty array');
  if (!Number.isFinite(distance) || Math.abs(distance) < 1e-9) throw new Error('extrudeEdges: distance must be a nonzero finite number');
  let offset = null;
  if (direction != null) {
    const dirLen = length(direction);
    if (!Number.isFinite(dirLen) || dirLen < 1e-9) throw new Error('extrudeEdges: direction must be a nonzero vector');
    offset = scale(direction, distance / dirLen);
  }

  const topology = buildTopology(cage);
  const keys = [...new Set(edgeKeys)];
  for (const key of keys) {
    const edge = topology.edgeMap.get(key);
    if (!edge) throw new Error(`extrudeEdges: "${key}" is not a real edge of this cage`);
    if (edge.faces.length !== 1) {
      throw new Error(`extrudeEdges: edge "${key}" is an INTERIOR edge (${edge.faces.length} faces) — only a naked (open) edge can be extruded, since an interior one has no free side to grow into; extruding it would mean tearing the surface open along it, which is a different operation`);
    }
  }
  // Same degeneracy the face case guards: a direction running exactly along
  // a selected edge collapses that edge's own new face to four collinear
  // points. Refused entirely rather than per-edge, so the result can never
  // come back with a hole where one bad quad was skipped.
  if (offset) {
    for (const key of keys) {
      const edge = topology.edgeMap.get(key);
      const edgeDir = sub(cage.vertices[edge.v1], cage.vertices[edge.v0]);
      const edgeLen = length(edgeDir);
      if (edgeLen < 1e-12) continue; // a zero-length edge is the input cage's own problem
      const cr = length(cross(edgeDir, offset));
      if (cr < 1e-9 * edgeLen * length(offset)) {
        throw new Error(`extrudeEdges: the extrude direction runs exactly parallel to edge "${key}" — its new face would collapse to four collinear points (zero area); pick a direction with some component across the edge`);
      }
    }
  }
  // PER-EDGE OUTWARD (the `direction == null` default) — one vector per
  // selected edge, then per-VERTEX by averaging whichever selected edges
  // actually touch it, so a shared corner gets one offset and one copy.
  const offsetByVertex = new Map();
  if (!offset) {
    const acc = new Map();
    for (const key of keys) {
      const edge = topology.edgeMap.get(key);
      const fi = edge.faces[0];
      const n = computeFaceNormal(cage, fi);
      const a = cage.vertices[edge.v0], b = cage.vertices[edge.v1];
      const ed = sub(b, a);
      let out = cross(n, ed);
      const ol = length(out);
      if (ol < 1e-12) continue; // a degenerate face normal or edge — nothing to point with
      out = scale(out, 1 / ol);
      const face = cage.faces[fi];
      const c = face.reduce((s2, vi) => add(s2, scale(cage.vertices[vi], 1 / face.length)), [0, 0, 0]);
      const mid = scale(add(a, b), 0.5);
      const sign = (out[0] * (mid[0] - c[0]) + out[1] * (mid[1] - c[1]) + out[2] * (mid[2] - c[2])) >= 0 ? 1 : -1;
      const dirOut = scale(out, sign);
      for (const vi of [edge.v0, edge.v1]) acc.set(vi, add(acc.get(vi) || [0, 0, 0], dirOut));
    }
    for (const [vi, sum] of acc) {
      const l = length(sum);
      if (l < 1e-9) {
        throw new Error(`extrudeEdges: vertex ${vi}'s own selected edges point in exactly opposite directions, so it has no outward to grow along — pick an explicit direction instead`);
      }
      offsetByVertex.set(vi, scale(sum, distance / l));
    }
  }

  const newVertices = cage.vertices.map((v) => v.slice());
  const newIndexForOld = new Map();
  const dup = (vi) => {
    if (!newIndexForOld.has(vi)) {
      newIndexForOld.set(vi, newVertices.length);
      newVertices.push(add(cage.vertices[vi], offset || offsetByVertex.get(vi)));
    }
    return newIndexForOld.get(vi);
  };

  // WINDING — the new face must traverse the shared edge OPPOSITE to the way
  // its own existing owner face does, which is what keeps the two
  // consistently oriented (the same rule extrudeFaces' own side faces
  // follow). Read the direction straight off the owner's vertex loop rather
  // than assuming the key's own sorted order.
  const newFaces = cage.faces.map((f) => f.slice());
  const addedFaces = [];
  for (const key of keys) {
    const edge = topology.edgeMap.get(key);
    const owner = cage.faces[edge.faces[0]];
    let va = edge.v0, vb = edge.v1;
    for (let c = 0; c < owner.length; c++) {
      const a = owner[c], b = owner[(c + 1) % owner.length];
      if (edgeKey(a, b) === key) { va = a; vb = b; break; }
    }
    addedFaces.push([vb, va, dup(va), dup(vb)]);
  }

  return {
    cage: { vertices: newVertices, faces: [...newFaces, ...addedFaces], creases: { ...(cage.creases || {}) } },
    newFaceIndices: addedFaces.map((_, i) => newFaces.length + i),
    newVertexIndices: [...newIndexForOld.values()],
  };
}

// INSERT POINT — split ONE edge by putting a new vertex on it. The smallest
// real topology tool on 74's list, and genuinely different from INSERT EDGE
// beside it: that one threads a whole new loop across a strip of faces, this
// one touches exactly the edge you picked. Every face using that edge gains
// one vertex in its own loop, so a quad becomes a 5-gon — which this kernel's
// Catmull-Clark accepts natively (n-gon faces are already load-bearing here:
// the cylinder and cone cage caps are n-gons by construction).
//
// NO T-JUNCTION, BY CONSTRUCTION. The new vertex is inserted into the loop of
// EVERY face that used the edge, not just one — the same rule per-face
// subdivide already follows when it widens a neighbor. Skipping the second
// face would leave the new vertex used by one face only: a cage that still
// renders, still counts right, and subdivides into a crack.
export function insertPointOnEdge(cage, edgeKeyStr, t = 0.5) {
  if (!Number.isFinite(t) || t <= 0 || t >= 1) throw new Error('insertPointOnEdge: t must be a number strictly between 0 and 1');
  const topology = buildTopology(cage);
  const edge = topology.edgeMap.get(edgeKeyStr);
  if (!edge) throw new Error(`insertPointOnEdge: "${edgeKeyStr}" is not a real edge of this cage`);

  const newIndex = cage.vertices.length;
  const vertices = [...cage.vertices.map((v) => v.slice()), lerp3(cage.vertices[edge.v0], cage.vertices[edge.v1], t)];
  const touched = new Set(edge.faces);
  const faces = cage.faces.map((f, fi) => {
    if (!touched.has(fi)) return f.slice();
    const out = [];
    for (let c = 0; c < f.length; c++) {
      const a = f[c], b = f[(c + 1) % f.length];
      out.push(a);
      if (edgeKey(a, b) === edgeKeyStr) out.push(newIndex);
    }
    return out;
  });

  // CREASE — the split edge no longer exists, so a weight left on its own key
  // would dangle while the two halves that replaced it went smooth. Both
  // halves inherit it, which is what keeps a hard edge hard through the edit.
  const creases = { ...(cage.creases || {}) };
  if (creases[edgeKeyStr] !== undefined) {
    const w = creases[edgeKeyStr];
    delete creases[edgeKeyStr];
    creases[edgeKey(edge.v0, newIndex)] = w;
    creases[edgeKey(newIndex, edge.v1)] = w;
  }
  return { cage: { vertices, faces, creases }, newVertexIndex: newIndex, widenedFaceIndices: [...touched] };
}

// WELD VERTICES — collapse a selected set of cage vertices into ONE. Real
// topology surgery, and the closest thing this module already had is
// stitchEdgeRuns, which is exactly a vertex remap: the same discipline is
// reused here rather than a second, subtly different merge.
//
// WHERE THE WELD LANDS is the caller's choice, matching Stitch's own
// vocabulary: 'average' (the centroid of the selection — the default, and
// the only one that treats every picked vertex equally), or 'first' (hold
// the first-picked vertex still and pull the rest onto it, which is what you
// want when one of them is already in the right place).
//
// WHAT IT REFUSES, and why refusing beats guessing. A weld that would make a
// face non-manifold (an edge ending up with 3+ faces) produces a cage that
// still renders and still counts right, then subdivides into garbage — the
// same silent-corruption class Bridge's own shared-vertex bug turned out to
// be. Checked on the RESULT, not predicted from the input, so no case has to
// be anticipated in advance to be caught.
export function weldVertices(cage, vertexIndices, position = 'average') {
  const picked = [...new Set(vertexIndices || [])];
  if (picked.length < 2) throw new Error('weldVertices: pick at least 2 vertices to weld (one alone is a no-op)');
  for (const vi of picked) {
    if (!Number.isInteger(vi) || vi < 0 || vi >= cage.vertices.length) throw new Error(`weldVertices: vertex index ${vi} is out of range`);
  }
  if (position !== 'average' && position !== 'first') throw new Error("weldVertices: position must be 'average' or 'first'");

  const keep = picked[0];
  const target = position === 'first'
    ? cage.vertices[keep].slice()
    : picked.reduce((acc, vi) => add(acc, scale(cage.vertices[vi], 1 / picked.length)), [0, 0, 0]);

  // Same shape as stitchEdgeRuns': a remap of the merged-away indices, then
  // one compaction pass that also renumbers everything that survived.
  const remap = new Map();
  for (const vi of picked) if (vi !== keep) remap.set(vi, keep);
  const removedSet = new Set(remap.keys());
  const working = cage.vertices.map((v, vi) => (vi === keep ? target.slice() : v.slice()));
  const oldToNew = new Map();
  const newVertices = [];
  working.forEach((v, vi) => {
    if (removedSet.has(vi)) return;
    oldToNew.set(vi, newVertices.length);
    newVertices.push(v);
  });
  const finalIndexOf = (oldIdx) => oldToNew.get(remap.has(oldIdx) ? remap.get(oldIdx) : oldIdx);

  let collapsedFaceCount = 0;
  const newFaces = [];
  for (const f of cage.faces) {
    const mapped = f.map(finalIndexOf);
    const cleaned = [];
    for (const vi of mapped) if (cleaned.length === 0 || cleaned[cleaned.length - 1] !== vi) cleaned.push(vi);
    if (cleaned.length > 1 && cleaned[0] === cleaned[cleaned.length - 1]) cleaned.pop();
    // A face can also fold onto itself NON-consecutively (welding two
    // opposite corners of a quad) — that is a real bowtie, not a triangle,
    // and it is dropped rather than kept as a repeated-vertex face.
    if (cleaned.length < 3 || new Set(cleaned).size !== cleaned.length) { collapsedFaceCount++; continue; }
    newFaces.push(cleaned);
  }
  if (!newFaces.length) throw new Error('weldVertices: that weld would collapse every face in the cage — nothing would be left');

  // Prune any vertex nothing uses any more (a face that collapsed may have
  // been its only user), compacting a second time.
  const used = new Set(newFaces.flat());
  const keptToFinal = new Map();
  const prunedVertices = [];
  newVertices.forEach((v, vi) => {
    if (!used.has(vi)) return;
    keptToFinal.set(vi, prunedVertices.length);
    prunedVertices.push(v);
  });
  const prunedFaces = newFaces.map((f) => f.map((vi) => keptToFinal.get(vi)));
  const finalOf = (oldIdx) => {
    const mid = finalIndexOf(oldIdx);
    return mid === undefined ? undefined : keptToFinal.get(mid);
  };

  const newCreases = {};
  for (const [key, weight] of Object.entries(cage.creases || {})) {
    const [a, b] = key.split('_').map(Number);
    const na = finalOf(a), nb = finalOf(b);
    if (na === undefined || nb === undefined || na === nb) continue; // merged onto itself, or dropped with a collapsed face
    newCreases[edgeKey(na, nb)] = weight;
  }

  const out = { vertices: prunedVertices, faces: prunedFaces, creases: newCreases };
  // THE SAFETY NET, checked on the real result.
  const ctx = buildTopology(out);
  for (const e of ctx.edgeMap.values()) {
    if (e.faces.length > 2) {
      throw new Error(`weldVertices: that weld would leave edge ${e.v0}-${e.v1} shared by ${e.faces.length} faces — a non-manifold cage that renders fine and subdivides into garbage; pick vertices that do not fold the surface onto itself`);
    }
  }
  return {
    cage: out,
    weldedVertexIndex: finalOf(keep),
    removedVertexCount: cage.vertices.length - prunedVertices.length,
    collapsedFaceCount,
  };
}

// SLIDE EDGE — move a selected edge (or a whole loop of them) ALONG the
// surface, without changing topology at all: every vertex of the selection
// travels down one of its own incident RAIL edges — the edges running across
// the loop rather than along it — by a fraction t. The cheapest tool on 74's
// own list precisely because it is pure geometry: no face is added, removed
// or rewritten, only positions move, so nothing downstream can break in a
// way a position drag could not already cause.
//
// WHICH WAY IS "FORWARD" is the only real problem here. Each loop vertex has
// two rails, one on each side, and choosing per-vertex independently gives a
// zig-zag rather than a slide. The choice is PROPAGATED instead: the first
// vertex picks a rail, and every subsequent vertex takes whichever of its own
// two rails points most nearly the same way as the previous vertex's choice.
// A vertex whose rails are genuinely ambiguous (no incident edge outside the
// selection) simply stays put — an honest local no-op rather than a guess.
//
// t IS BOUNDED, deliberately. At |t| = 1 a sliding vertex lands exactly on
// the neighbor it is sliding toward, producing a zero-length edge and a
// degenerate face — a cage that renders and subdivides into garbage. Refused
// past 0.95 rather than clamped silently.
export function slideEdges(cage, edgeKeys, t) {
  if (!edgeKeys || !edgeKeys.length) throw new Error('slideEdges: edgeKeys must be a non-empty array');
  if (!Number.isFinite(t)) throw new Error('slideEdges: t must be a finite number');
  if (Math.abs(t) > 0.95) throw new Error('slideEdges: |t| must stay under 0.95 — at 1 the sliding vertices land exactly on the ones they are sliding toward, collapsing edges to zero length');
  const topology = buildTopology(cage);
  const selected = new Set();
  for (const key of edgeKeys) {
    if (!topology.edgeMap.has(key)) throw new Error(`slideEdges: "${key}" is not a real edge of this cage`);
    selected.add(key);
  }
  // The vertices being slid, in a stable order so propagation is
  // deterministic: walk the selected edges and take their endpoints.
  const order = [];
  const seen = new Set();
  for (const key of edgeKeys) {
    const e = topology.edgeMap.get(key);
    for (const vi of [e.v0, e.v1]) if (!seen.has(vi)) { seen.add(vi); order.push(vi); }
  }
  const railsFor = (vi) => topology.vertexEdges[vi]
    .filter((e) => !selected.has(edgeKey(e.v0, e.v1)))
    .map((e) => (e.v0 === vi ? e.v1 : e.v0));

  const vertices = cage.vertices.map((v) => v.slice());
  let slidCount = 0, stuckCount = 0;
  // ONE pass. The first vertex with a real choice seeds the direction — rail
  // 0 for a positive t, the rail most OPPOSITE to it for a negative one —
  // and every vertex after it takes whichever of its own rails agrees best
  // with that seed. This is what turns a set of independent per-vertex
  // choices into a single coherent slide.
  let reference = null;
  for (const vi of order) {
    const rails = railsFor(vi);
    const dirs = rails
      .map((r) => ({ r, d: sub(cage.vertices[r], cage.vertices[vi]) }))
      .map(({ r, d }) => ({ r, d, l: length(d) }))
      .filter(({ l }) => l > 1e-12)
      .map(({ r, d, l }) => ({ r, unit: scale(d, 1 / l), vec: d }));
    if (!dirs.length) { stuckCount++; continue; }
    let pick;
    if (!reference) {
      // Seed: rail 0 going forward, or the rail furthest from it going back.
      pick = dirs[0];
      if (t < 0 && dirs.length > 1) {
        let worst = Infinity;
        for (const c of dirs) {
          const d = dot(c.unit, dirs[0].unit);
          if (d < worst) { worst = d; pick = c; }
        }
      }
    } else {
      let best = -Infinity;
      for (const c of dirs) {
        const d = dot(c.unit, reference);
        if (d > best) { best = d; pick = c; }
      }
    }
    reference = pick.unit;
    vertices[vi] = add(cage.vertices[vi], scale(pick.vec, Math.abs(t)));
    slidCount++;
  }
  return {
    cage: { vertices, faces: cage.faces.map((f) => f.slice()), creases: { ...(cage.creases || {}) } },
    slidVertexCount: slidCount,
    stuckVertexCount: stuckCount,
  };
}

// OFFSET / THICKEN A CAGE — the SubD answer to the NURBS Offset/Thicken
// pair. Not the same machinery: a NURBS offset moves a tensor surface's own
// control net along Greville normals, and a cage is not that. Here every
// cage VERTEX moves along its own normal — the normalized sum of the normals
// of the faces meeting there, which is the standard vertex-normal estimate
// and, for a cage, the direction the limit surface itself locally faces.
//
// HONEST ABOUT WHAT THIS IS. Offsetting a control cage does NOT produce a
// mathematically exact offset of its own limit surface — no polynomial
// surface has one in general, which is exactly why this app's own NURBS
// offset says the same thing about itself. It produces a cage whose limit
// surface runs roughly `distance` away from the original's, exactly as
// every SubD modeler's own offset does. Stated here rather than implied.
//
// SELF-INTERSECTION IS NOT GUARDED. A large enough offset on a concave
// region folds the cage through itself. The NURBS side has a real
// safeOffsetMagnitude search for this; the cage side does not yet, and
// pretending otherwise would be worse than saying so.
export function offsetCage(cage, distance) {
  if (!Number.isFinite(distance) || Math.abs(distance) < 1e-9) throw new Error('offsetCage: distance must be a nonzero finite number');
  const normals = vertexNormals(cage);
  const vertices = cage.vertices.map((v, i) => {
    const n = normals[i];
    if (!n) throw new Error(`offsetCage: vertex ${i} has no usable normal (its faces' own normals cancel, or it belongs to no face) — nothing to offset it along`);
    return add(v, scale(n, distance));
  });
  return { vertices, faces: cage.faces.map((f) => f.slice()), creases: { ...(cage.creases || {}) } };
}

// The normalized sum of the normals of every face meeting each vertex.
// Returns null in that slot when they cancel (or there are none) — the
// caller decides whether that is fatal.
function vertexNormals(cage) {
  const sums = cage.vertices.map(() => [0, 0, 0]);
  cage.faces.forEach((face, fi) => {
    const n = computeFaceNormal(cage, fi);
    for (const vi of face) sums[vi] = add(sums[vi], n);
  });
  return sums.map((s) => {
    const l = length(s);
    return l < 1e-9 ? null : scale(s, 1 / l);
  });
}

// THICKEN — give a cage real thickness: the original, an offset copy wound
// the other way, and (for an OPEN cage) a rim of quads closing the gap
// between the two boundaries, so a sheet becomes a slab. A CLOSED cage has
// no boundary to close, so it comes back as two nested shells — a solid
// with a cavity, which is the honest answer there and matches what the
// NURBS thicken already does for the same input.
//
// Winding: the offset copy is REVERSED (its faces read backwards) so its
// own outward side faces away from the original — without that the two
// sheets face the same way and the slab is inside-out on one side. The rim
// quads are wound from the original's own boundary direction, so every
// shared edge is traversed once each way.
// REFUSES AN ALREADY-CLOSED CAGE, and this is the interesting case. The rim
// below is built from NAKED edges — one wall quad per edge with a single
// owner face. A closed cage has none, so the rim comes out empty and the
// result is the original cage plus a reversed offset copy with nothing
// joining them: one object silently becomes TWO disconnected shells, one
// nested inside the other, with no wall anywhere. Verified structurally, not
// reasoned about — a closed box or torus goes from 1 connected component to
// 2, while an open sheet correctly stays at 1 with a real rim.
//
// Two nested closed shells is a perfectly good B-rep hollow solid, which is
// why this went unnoticed and was even written down as intended. It is NOT a
// good SubD cage: the object stops being a single connected manifold, every
// downstream cage command sees two components, and on screen it reads as a
// duplicate of itself rather than as a wall — which is exactly how it was
// reported ("a superb inside a superb; just have to undo or delete").
// Refused rather than produced, per this app's own rule that an operation
// which cannot keep a SuperB one watertight object should say so instead. The
// escape hatch is real and named in the message: delete a face first, and the
// thicken then builds a genuine wall around that opening.
export function thickenCage(cage, distance) {
  if (!Number.isFinite(distance) || Math.abs(distance) < 1e-9) throw new Error('thickenCage: distance must be a nonzero finite number');
  const closedCheck = buildTopology(cage);
  let nakedCount = 0;
  for (const [, e] of closedCheck.edgeMap) if (e.faces.length === 1) nakedCount++;
  if (nakedCount === 0) {
    throw new Error('this cage is already closed, so there is no open edge for a wall to grow from — thickening it would leave an offset copy floating inside the original as a second, disconnected shell rather than one watertight object. Delete a face first to open it, then Thicken builds a real wall around that opening.');
  }
  const offset = offsetCage(cage, distance);
  const n = cage.vertices.length;
  const vertices = [...cage.vertices.map((v) => v.slice()), ...offset.vertices.map((v) => v.slice())];
  const faces = [
    ...cage.faces.map((f) => f.slice()),
    ...cage.faces.map((f) => f.slice().reverse().map((vi) => vi + n)),
  ];
  // RIM — one quad per naked edge, wound against the owner face's own
  // traversal so it agrees with both sheets at once.
  const topology = buildTopology(cage);
  const rimFaces = [];
  for (const [key, edge] of topology.edgeMap) {
    if (edge.faces.length !== 1) continue;
    const owner = cage.faces[edge.faces[0]];
    let va = edge.v0, vb = edge.v1;
    for (let c = 0; c < owner.length; c++) {
      const a = owner[c], b = owner[(c + 1) % owner.length];
      if (edgeKey(a, b) === key) { va = a; vb = b; break; }
    }
    rimFaces.push([vb, va, va + n, vb + n]);
  }
  // Creases ride along on BOTH sheets — a hard edge stays hard on the copy.
  const creases = {};
  for (const [key, w] of Object.entries(cage.creases || {})) {
    creases[key] = w;
    const [a, b] = key.split('_').map(Number);
    creases[edgeKey(a + n, b + n)] = w;
  }
  return {
    cage: { vertices, faces: [...faces, ...rimFaces], creases },
    offsetFaceIndices: cage.faces.map((_, i) => cage.faces.length + i),
    rimFaceIndices: rimFaces.map((_, i) => 2 * cage.faces.length + i),
  };
}

// INSERTEDGE — insert a new edge loop PARALLEL to and offset from a picked
// existing loop ("insert an edge loop beside a picked loop,
// position slider 0-1 along the ring — Rhino's InsertEdge. The fundamental
// tighten move"). `seedKey` is any real cage edge; `side` (0 or 1) picks
// WHICH of its own (up to 2) adjacent faces the new loop is inserted toward
// — v1 always defaults to 0 (a real, stated scope cut: the OTHER neighbor
// loop is reached by picking a seed edge on the opposite side instead, not
// by a second app-layer control). `t` in (0,1) exclusive (0 or 1
// would coincide EXACTLY with an existing loop, producing zero-area faces —
// the "no degenerate zero-area faces" requirement stated
// explicitly) is the position fraction: t->0 sits near the picked loop,
// t->1 sits near its neighbor on the chosen side.
//
// ALGORITHM (derived from first principles on a concrete UxV quad-strip
// example — a cylinder side band — before writing this function; see this
// module's own test file for the hand-derived worked numbers): a directed
// BFS walks the QUAD STRIP between the picked loop and its one chosen
// neighbor loop. Each strip face is entered already knowing its own "near"
// edge (an edge of the PICKED loop, or — deeper into the strip — the
// corresponding edge of whichever loop is currently nearest); within that
// face, `oppositeEdgeInFace` gives the "far" edge (the corresponding edge of
// the NEIGHBOR loop), and the face's other two edges are its two RUNGS
// (connecting a near vertex to its corresponding far vertex). Crossing a
// rung into the next face along the strip requires knowing, at that specific
// vertex, which of the new face's own two OTHER incident edges continues the
// SAME loop-A direction (not the rung itself, and not the far-side
// continuation) — found directly via the new face's own vertex-index
// neighbors, not by re-deriving "opposite" (which needs a KNOWN near edge to
// start from, not yet known for the new face). A rung is shared by exactly
// two strip faces (for an interior rung) — its own new inserted vertex is
// computed once and reused, which is exactly what stitches the per-face new
// edges into one continuous new loop.
//
// LOCAL MODE (`opts.local`) — the same cut, stopped after the seed face.
// Whole-strip propagation is the right answer for "tighten this whole band"
// and the wrong one for "refine THIS face": picking one edge and getting the
// entire chain refined is a global edit answering a local ask. `local` is a
// pure SCOPE restriction of the walk above — identical `t`, identical `side`,
// identical near/far orientation reasoning, identical crease rules — it
// simply never enqueues the neighbors across its own rungs. That is why it
// is an option on this function rather than a separate one: a sibling copy
// would have to restate the whole orientation derivation and would drift
// from it. `recomputeInsertedLoopPositions` keeps working on the result for
// free, since it is driven entirely by the returned rung pairs.
//
// WHAT LOCAL LEAVES BEHIND, deliberately. The seed face becomes two quads
// separated by the new edge. That edge's two endpoints sit in the interiors
// of the seed face's two RUNGS, which are the only edges this operation ever
// splits — the near edge (the seed itself) and the far edge are reused
// VERBATIM by the two replacement quads, so nothing on either side of either
// one changes. Each rung's OTHER face therefore has to receive that same new
// vertex on its own boundary, growing from n sides to n+1: a vertex present
// on one side of an edge and absent on the other is a crack, not a
// T-junction. The resulting n-gon is a legal Catmull-Clark cage face —
// computeFacePoint is a plain centroid for any n>=3 — so the limit surface
// stays well defined; that n-gon plus the extraordinary vertices at the new
// T-vertices is the substrate a T-spline-style local workflow needs.
//
// THE SEED EDGE'S OTHER FACE IS NOT SPLIT, on purpose. `side` already names
// which of the seed edge's faces is being refined, and the seed edge is not
// split by this operation at all, so the other face has no missing vertex to
// repair — it is genuinely untouched, not left inconsistent. Splitting it too
// would not carry one cut straight across; it would start a SECOND,
// independent parallel cut running away from the shared edge in the opposite
// direction, doubling both the new faces and the T-junctions for a move asked
// for once. That second cut is one more local insert with the other `side`.
//
// LOCAL'S OWN SCOPE LIMIT: the SEED face must still be a quad, because "the
// opposite edge" is only well defined for a quad. A face already carrying a
// T-junction vertex (an n-gon, n>=5) therefore cannot itself seed the next
// insert — but it can be, and is, correctly WIDENED AGAIN as the neighbor of
// one, which is the case that actually recurs when refining locally.
export function insertEdgeLoop(cage, seedKey, t = 0.5, side = 0, opts = {}) {
  const local = !!(opts && opts.local);
  if (!Number.isFinite(t) || t <= 0 || t >= 1) throw new Error('insertEdgeLoop: t must be a number strictly between 0 and 1');
  if (side !== 0 && side !== 1) throw new Error('insertEdgeLoop: side must be 0 or 1');
  const topology = buildTopology(cage);
  const seedEdge = topology.edgeMap.get(seedKey);
  if (!seedEdge) throw new Error(`insertEdgeLoop: "${seedKey}" is not a real edge of this cage`);
  const face0 = seedEdge.faces[side];
  if (face0 === undefined) throw new Error(`insertEdgeLoop: no face on side ${side} of edge "${seedKey}" (it has ${seedEdge.faces.length} adjacent face${seedEdge.faces.length === 1 ? '' : 's'})`);

  const faceInfos = new Map(); // faceIdx -> { na, nb, farNa, farNb }
  const visited = new Set();
  const queue = [{ faceIdx: face0, nearKey: seedKey }];

  while (queue.length) {
    const { faceIdx, nearKey } = queue.shift();
    if (visited.has(faceIdx)) continue;
    visited.add(faceIdx);
    const face = cage.faces[faceIdx];
    if (face.length !== 4) throw new Error(`insertEdgeLoop: face ${faceIdx} is not a quad (has ${face.length} sides) — ${local ? 'a local edge insert needs a well-defined opposite edge, which only a quad has' : 'InsertEdge is only defined for a quad-faced strip'}`);
    const nearEdge = topology.edgeMap.get(nearKey);
    const farKey = oppositeEdgeInFace(cage, faceIdx, nearKey);
    if (!farKey) throw new Error(`insertEdgeLoop: face ${faceIdx} has no well-defined opposite edge for "${nearKey}"`);
    // ORIENT THE NEAR PAIR TO THE FACE'S OWN TRAVERSAL, not to the edge's
    // canonical (sorted) one. An edgeKey is order-independent by design —
    // "3_7" says nothing about which way either of its two faces walks it,
    // and the two walk it in OPPOSITE directions precisely because the
    // cage is consistently wound. The replacement quads below are built in
    // traversal order ([na, nb, Nb, Na]), so taking na/nb from the edge
    // record instead of from this face produced a correctly-shaped but
    // BACKWARDS-wound pair for every face whose own traversal happened to
    // run against the canonical order — a cage that still has the right
    // vertex and face counts, still renders, and is silently
    // non-orientable. Found by the generated corpus sweeping `side`, not
    // by any hand-built fixture: side 0 on a box happens to walk with the
    // canonical order throughout and is clean, side 1 starts from the
    // OTHER face of the same seed edge and was wrong at every step.
    const ia0 = face.indexOf(nearEdge.v0);
    if (ia0 === -1) throw new Error(`insertEdgeLoop: face ${faceIdx} does not contain vertex ${nearEdge.v0}`);
    let na, nb, farNa, farNb;
    if (face[(ia0 + 1) % 4] === nearEdge.v1) {
      // this face walks v0 -> v1
      na = nearEdge.v0; nb = nearEdge.v1;
      farNb = face[(ia0 + 2) % 4]; farNa = face[(ia0 + 3) % 4];
    } else if (face[(ia0 + 3) % 4] === nearEdge.v1) {
      // this face walks v1 -> v0: relabel so na is genuinely the corner
      // this face reaches FIRST, carrying each one's own far partner with
      // it (the rung pairing is unchanged — only which is called "a").
      na = nearEdge.v1; nb = nearEdge.v0;
      farNa = face[(ia0 + 2) % 4]; farNb = face[(ia0 + 1) % 4];
    } else throw new Error(`insertEdgeLoop: face ${faceIdx} does not contain edge "${nearKey}" as one of its own sides`);

    faceInfos.set(faceIdx, { na, nb, farNa, farNb });

    if (local) continue; // LOCAL: the cut stops at this face's own rungs, so no neighbor is ever enqueued

    for (const [nearV, farV] of [[na, farNa], [nb, farNb]]) {
      const rKey = edgeKey(nearV, farV);
      const rEdge = topology.edgeMap.get(rKey);
      if (!rEdge) continue; // shouldn't happen for a well-formed quad, but an honest skip beats a crash
      const others = rEdge.faces.filter((f) => f !== faceIdx);
      if (others.length !== 1) continue; // a true cage boundary (0 left) or non-manifold (2+ left) — stop expanding this direction, don't guess
      const nextFaceIdx = others[0];
      if (visited.has(nextFaceIdx)) continue;
      const nextFace = cage.faces[nextFaceIdx];
      const idx2 = nextFace.indexOf(nearV);
      if (idx2 === -1) throw new Error(`insertEdgeLoop: rung "${rKey}" traversal error at face ${nextFaceIdx}`);
      const n2 = nextFace.length;
      const neighbors2 = [nextFace[(idx2 + 1) % n2], nextFace[(idx2 - 1 + n2) % n2]];
      const other2 = neighbors2.find((x) => x !== farV);
      if (other2 === undefined) throw new Error(`insertEdgeLoop: rung "${rKey}" traversal error at face ${nextFaceIdx} (degenerate face)`);
      queue.push({ faceIdx: nextFaceIdx, nearKey: edgeKey(nearV, other2) });
    }
  }

  // Materialize one new vertex per DISTINCT rung actually touched by the
  // strip (a rung shared by 2 strip faces is discovered twice but only
  // built once — same `newVertexOf` map both discoveries look up).
  const newVertices = cage.vertices.map((v) => v.slice());
  const newVertexOf = new Map();
  const insertedVertexIndices = [];
  const rungPairs = [];
  function ensureNewVertex(nearIdx, farIdx) {
    const key = edgeKey(nearIdx, farIdx);
    let idx = newVertexOf.get(key);
    if (idx === undefined) {
      idx = newVertices.length;
      newVertices.push(lerp3(cage.vertices[nearIdx], cage.vertices[farIdx], t));
      newVertexOf.set(key, idx);
      insertedVertexIndices.push(idx);
      rungPairs.push([nearIdx, farIdx]);
    }
    return idx;
  }
  for (const info of faceInfos.values()) {
    ensureNewVertex(info.na, info.farNa);
    ensureNewVertex(info.nb, info.farNb);
  }

  const facesToRemove = new Set(faceInfos.keys());
  const replacementFaces = [];
  for (const info of faceInfos.values()) {
    const NaIdx = newVertexOf.get(edgeKey(info.na, info.farNa));
    const NbIdx = newVertexOf.get(edgeKey(info.nb, info.farNb));
    replacementFaces.push([info.na, info.nb, NbIdx, NaIdx]); // near half (toward the picked loop)
    replacementFaces.push([NaIdx, NbIdx, info.farNb, info.farNa]); // far half (toward the neighbor loop)
  }
  // T-JUNCTION REPAIR — every rung this insert split is, in general, shared
  // with a face that is NOT itself being split, and that face's own boundary
  // still walks the rung as a single side. Left alone it would omit a vertex
  // its neighbor has, which is not a T-junction but a crack: the two faces
  // would agree on the endpoints and disagree about everything between them,
  // and subdivideCatmullClark would refine the mismatch rather than close it.
  // So the new vertex is spliced into that face's loop, in traversal order,
  // between the rung's own two endpoints — turning an n-gon into an (n+1)-gon
  // that is still legal input to every rule in kernel/subd.mjs.
  //
  // A face is rebuilt only if one of its own sides is genuinely a split rung;
  // otherwise it is copied through unchanged. In the whole-loop (non-local)
  // case, every face across every split rung is itself in `facesToRemove`
  // (that is exactly what the walk enqueued), so nothing is rebuilt and the
  // output is unchanged — this repair is reached only by a cut that stops.
  const splitRungMid = new Map(); // rung edgeKey -> the new vertex sitting in its interior
  rungPairs.forEach(([nearIdx, farIdx], i) => splitRungMid.set(edgeKey(nearIdx, farIdx), insertedVertexIndices[i]));

  const keptFaces = [];
  const tJunctionFaceIndices = [];
  cage.faces.forEach((face, fi) => {
    if (facesToRemove.has(fi)) return;
    const n = face.length;
    let splits = false;
    for (let c = 0; c < n && !splits; c++) splits = splitRungMid.has(edgeKey(face[c], face[(c + 1) % n]));
    if (!splits) { keptFaces.push(face.slice()); return; }
    const widened = [];
    for (let c = 0; c < n; c++) {
      const a = face[c], b = face[(c + 1) % n];
      widened.push(a);
      const mid = splitRungMid.get(edgeKey(a, b));
      if (mid !== undefined) widened.push(mid);
    }
    tJunctionFaceIndices.push(keptFaces.length); // index into the OUTPUT faces array, which starts with keptFaces
    keptFaces.push(widened);
  });

  // CREASE REMAP — the same real gap named in extrudeFaces' own header,
  // in this operation's own shape: the near/far edges of every strip face
  // (na-nb, farNa-farNb — the loop-direction edges) are reused VERBATIM
  // by the near-half/far-half replacement faces above, so a crease on
  // either survives for free, unchanged. A RUNG (na-farNa or nb-farNb —
  // the perpendicular edges the loop actually crosses) is a different
  // story: it gets SPLIT into two brand-new edges (near-to-inserted,
  // inserted-to-far) that never existed before, and the original rung
  // itself no longer exists as an edge anywhere in the output topology —
  // its own old crease key would otherwise silently dangle while both of
  // its real replacements got no weight at all. Fixed by transferring the
  // rung's own weight onto BOTH of its new halves, reusing the already-
  // built `rungPairs`/`insertedVertexIndices` (built above for exactly
  // this near/far correspondence, one entry per rung actually crossed).
  const newCreases = { ...(cage.creases || {}) };
  rungPairs.forEach(([nearIdx, farIdx], i) => {
    const oldKey = edgeKey(nearIdx, farIdx);
    const weight = newCreases[oldKey];
    if (weight === undefined) return;
    delete newCreases[oldKey];
    const midIdx = insertedVertexIndices[i];
    newCreases[edgeKey(nearIdx, midIdx)] = weight;
    newCreases[edgeKey(midIdx, farIdx)] = weight;
  });

  return {
    cage: { vertices: newVertices, faces: [...keptFaces, ...replacementFaces], creases: newCreases },
    insertedVertexIndices,
    rungPairs,
    // Output-array indices, so a caller can highlight what it just made
    // without re-deriving it: the two halves of every split face, and the
    // untouched-but-widened neighbors that now carry a T-junction vertex
    // (always empty for a whole-loop insert, which leaves none).
    splitFaceIndices: replacementFaces.map((_, i) => keptFaces.length + i),
    tJunctionFaceIndices,
  };
}

// The LOCAL edge insert, named. Identical arguments to insertEdgeLoop's own
// first four and identical return shape — this exists so an app-layer call
// site reads as the operation the user asked for rather than as a loop insert
// with a flag turned off, and it holds no logic of its own precisely so the
// two can never disagree about anything.
export function insertEdgeLocal(cage, seedKey, t = 0.5, side = 0) {
  return insertEdgeLoop(cage, seedKey, t, side, { local: true });
}

// Re-positions an ALREADY-inserted loop's own vertices to a NEW `t`, given
// the (nearIdx, farIdx) rung pairs insertEdgeLoop returned — pure reposition,
// no topology change at all (same vertex/face/edge counts before and after).
// This is what lets the app layer offer a real, continuously-draggable
// position slider AFTER the loop already exists (Surface Fair's own
// Smoothness-slider precedent): the
// expensive topology surgery above runs exactly ONCE, at insertion time;
// every subsequent slider drag is just this cheap reposition.
export function recomputeInsertedLoopPositions(cage, insertedVertexIndices, rungPairs, t) {
  if (!Number.isFinite(t) || t <= 0 || t >= 1) throw new Error('recomputeInsertedLoopPositions: t must be a number strictly between 0 and 1');
  const vertices = cage.vertices.map((v) => v.slice());
  insertedVertexIndices.forEach((vi, i) => {
    const [nearIdx, farIdx] = rungPairs[i];
    vertices[vi] = lerp3(cage.vertices[nearIdx], cage.vertices[farIdx], t);
  });
  return { vertices, faces: cage.faces, creases: cage.creases };
}

// DELETE FACES — the v1-listed "makes holes" command. Removes
// the given faces from the cage and prunes any vertex left with ZERO
// remaining faces (an orphan — kept around it would break
// subdivideCatmullClark's own vertex-point rule, which assumes every vertex
// has at least one incident face), compacting the surviving vertex indices.
// A crease is kept only when the edge it names still genuinely exists in the
// OUTPUT topology (both endpoints survive AND some remaining face still uses
// that exact edge) — a dangling crease key pointing at a vertex pair that's
// no longer an edge anywhere would be a real, silent correctness gap, not
// caught by a plain "did both endpoints survive" check alone (two vertices
// can each individually survive, used by different faces, without the
// specific edge between them still existing). Returns `vertexRemap` (old
// index -> new index, for every SURVIVING vertex only) so a caller (BRIDGE,
// below) that already computed something in terms of the ORIGINAL indices
// can translate it, without re-deriving the same compaction a second time.
export function deleteFaces(cage, faceIndices) {
  if (!faceIndices || !faceIndices.length) throw new Error('deleteFaces: faceIndices must be a non-empty array');
  for (const fi of faceIndices) {
    if (!Number.isInteger(fi) || fi < 0 || fi >= cage.faces.length) throw new Error(`deleteFaces: face index ${fi} is out of range`);
  }
  const removeSet = new Set(faceIndices);
  if (removeSet.size >= cage.faces.length) throw new Error('deleteFaces: refusing to delete every face of the cage — a SuperB needs at least one face left');

  const keptFaces = cage.faces.filter((_, fi) => !removeSet.has(fi));
  const usedVerts = new Set();
  for (const f of keptFaces) for (const vi of f) usedVerts.add(vi);

  const vertexRemap = new Map();
  const newVertices = [];
  cage.vertices.forEach((v, vi) => {
    if (!usedVerts.has(vi)) return; // orphaned by this deletion — dropped, not kept as dead data
    vertexRemap.set(vi, newVertices.length);
    newVertices.push(v.slice());
  });
  const newFaces = keptFaces.map((f) => f.map((vi) => vertexRemap.get(vi)));

  const newTopology = buildTopology({ vertices: newVertices, faces: newFaces, creases: {} });
  const newCreases = {};
  for (const [key, weight] of Object.entries(cage.creases || {})) {
    const [a, b] = key.split('_').map(Number);
    if (!vertexRemap.has(a) || !vertexRemap.has(b)) continue; // an endpoint was orphaned by this deletion
    const newKey = edgeKey(vertexRemap.get(a), vertexRemap.get(b));
    if (!newTopology.edgeMap.has(newKey)) continue; // both endpoints survive, but this specific edge no longer exists anywhere
    newCreases[newKey] = weight;
  }

  return {
    cage: { vertices: newVertices, faces: newFaces, creases: newCreases },
    vertexRemap,
    removedVertexCount: cage.vertices.length - newVertices.length,
  };
}

// The single boundary edge at `vIdx` OTHER than `excludeKey` — used by
// boundaryLoopFromSeed's own walk below. Returns null when there isn't
// exactly one (a dead end, or a non-manifold boundary vertex where 2+
// boundary edges meet) — an honest stop, matching every other kernel walk
// in this file's own "refuse rather than guess" convention.
function otherBoundaryEdgeAt(topology, vIdx, excludeKey) {
  const candidates = topology.vertexEdges[vIdx].filter((e) => e.faces.length === 1 && edgeKey(e.v0, e.v1) !== excludeKey);
  return candidates.length === 1 ? candidates[0] : null;
}

// BOUNDARY LOOP FROM SEED — walks a real cage BOUNDARY (an edge used by
// exactly one face) all the way around, starting from any one boundary edge,
// returning the ordered vertex-index loop. This is a DIFFERENT walk from
// kernel/subdselect.mjs's own edgeLoopFromSeed (which requires a regular
// valence-4 interior vertex at every step and has no notion of "boundary" at
// all) — a hole's own rim vertices can have any valence, so the only real
// constraint here is "exactly one OTHER boundary edge at this vertex,"
// checked by otherBoundaryEdgeAt above.
//
// ORIENTATION, derived deliberately, not guessed: `seedKey`'s one adjacent
// face F traverses the edge as (v0 -> v1) somewhere in its own cyclic vertex
// order (by definition of how buildTopology records `edge.v0`/`edge.v1` — see
// its own construction, which always records the edge in the SAME direction
// the owning face visits it). For the cage to stay a consistently-oriented
// manifold once this hole is FILLED, the new fill face must traverse this
// same edge in the OPPOSITE direction, i.e. (v1 -> v0). Starting the walk at
// v0 and stepping AWAY from v1 (via v0's own OTHER boundary edge) produces
// exactly the ordered list [v0, x2, ..., xk, v1] — reading this list as a
// face's own cyclic vertex order, its wrap-around edge (last element back to
// first) is (v1 -> v0), the required opposite direction, with zero extra
// reversal step needed. Proven directly in this module's own test file (a
// face built from this loop shares every edge with its one real neighbor in
// the opposite direction — a genuine winding-consistency check, not assumed).
export function boundaryLoopFromSeed(cage, seedKey) {
  const topology = buildTopology(cage);
  const seedEdge = topology.edgeMap.get(seedKey);
  if (!seedEdge) throw new Error(`boundaryLoopFromSeed: "${seedKey}" is not a real edge of this cage`);
  if (seedEdge.faces.length !== 1) throw new Error(`boundaryLoopFromSeed: "${seedKey}" is not a boundary edge (it has ${seedEdge.faces.length} adjacent faces, a boundary edge has exactly 1)`);
  const { v0, v1 } = seedEdge;
  const loop = [v0];
  let curVertex = v0, curKey = seedKey;
  for (;;) {
    const nextEdge = otherBoundaryEdgeAt(topology, curVertex, curKey);
    if (!nextEdge) throw new Error(`boundaryLoopFromSeed: vertex ${curVertex} does not have exactly one other boundary edge — this hole's own boundary is not a single simple loop (non-manifold boundary vertex, or a dead end)`);
    const nextVertex = nextEdge.v0 === curVertex ? nextEdge.v1 : nextEdge.v0;
    if (nextVertex === v1) { loop.push(v1); break; }
    if (loop.includes(nextVertex)) throw new Error(`boundaryLoopFromSeed: the hole boundary revisited vertex ${nextVertex} before closing back at the seed edge's own other end — not a simple loop`);
    loop.push(nextVertex);
    curVertex = nextVertex;
    curKey = edgeKey(nextEdge.v0, nextEdge.v1);
    if (loop.length > cage.vertices.length) throw new Error('boundaryLoopFromSeed: the boundary walk failed to close within a reasonable number of steps (a non-manifold or disconnected boundary)');
  }
  if (loop.length < 3) throw new Error(`boundaryLoopFromSeed: the hole boundary at "${seedKey}" only has ${loop.length} distinct vertices — not a real face-able loop`);
  return loop;
}

// FILLSUBDHOLE — the inverse of DELETE FACES. Fills a real, currently-open
// boundary loop with ONE new planar n-gon face, matching the loop's own
// vertex count exactly (the stated honest v1 scope — no
// attempt at a fancier multi-face/centroid-fan fill). Refuses
// honestly if the given loop isn't a genuine, currently-open boundary
// (each consecutive pair must be a real edge of the cage used by EXACTLY one
// face right now) rather than silently building an invalid/overlapping face.
export function fillHoleWithNGon(cage, loopVertexIndices) {
  if (!loopVertexIndices || loopVertexIndices.length < 3) throw new Error('fillHoleWithNGon: loopVertexIndices needs at least 3 vertices');
  if (new Set(loopVertexIndices).size !== loopVertexIndices.length) throw new Error('fillHoleWithNGon: loopVertexIndices contains a repeated vertex — not a simple loop');
  const topology = buildTopology(cage);
  const n = loopVertexIndices.length;
  for (let i = 0; i < n; i++) {
    const a = loopVertexIndices[i], b = loopVertexIndices[(i + 1) % n];
    const key = edgeKey(a, b);
    const edge = topology.edgeMap.get(key);
    if (!edge) throw new Error(`fillHoleWithNGon: "${key}" is not a real edge of this cage — the loop no longer matches an actual hole boundary`);
    if (edge.faces.length !== 1) throw new Error(`fillHoleWithNGon: "${key}" already has ${edge.faces.length} adjacent face${edge.faces.length === 1 ? '' : 's'} — it isn't an open boundary edge anymore`);
  }
  return {
    cage: { vertices: cage.vertices.map((v) => v.slice()), faces: [...cage.faces, [...loopVertexIndices]], creases: { ...(cage.creases || {}) } },
    faceIndex: cage.faces.length,
  };
}

// The ordered boundary-loop walk of an entire FACE SET (as opposed to
// boundaryLoopFromSeed's single already-open boundary edge) — BRIDGE's own
// prerequisite: before the two selected patches are actually deleted, find
// what their own rim will look like ONCE they are. A "boundary-region" edge
// of the set (exactly one of its up-to-2 adjacent faces is IN the set,
// mirroring extrudeFaces' own identical definition one milestone up) is
// only usable here when its OTHER side is a real, different, un-selected
// face (edge.faces.length === 2) — v1's own honest, stated scope cut:
// Bridge does not attempt to reason about a selection whose own rim touches
// an ALREADY-open cage edge (deleting the selection there wouldn't create a
// new hole edge at all, it would just erase that edge from the topology
// entirely), refusing by name rather than silently producing a malformed
// tunnel.
//
// ORIENTATION — a real bug, found and fixed via this module's own test
// file, not assumed correct on the first try: the selected face's own
// SURVIVING neighbor across a rim edge (va,vb) traverses that SAME edge in
// the OPPOSITE direction (vb,va) — the standard consistent-orientation
// manifold property, true of any two faces sharing an edge. Per
// boundaryLoopFromSeed's own derivation, a NEW face filling this rim must
// traverse the edge opposite the SURVIVING neighbor's own direction, i.e.
// opposite of (vb,va) = (va,vb) — the SAME direction the selected
// (about-to-be-deleted) face itself already used. So `nextOf` must record
// the SELECTED face's own forward direction (va -> vb) verbatim, not
// reversed — proven directly: a hand round-trip (delete a face, refill its
// rim via THIS function) is checked byte-for-byte against the SAME rim
// walked from a seed edge via the independently-derived, already-proven
// boundaryLoopFromSeed, in this module's own test file.
export function orderedBoundaryLoopOfFaceSet(cage, faceIndices) {
  const topology = buildTopology(cage);
  const selSet = new Set(faceIndices);
  const nextOf = new Map(); // va -> vb, the SAME direction the selected face itself traverses this rim edge
  for (const fi of faceIndices) {
    const face = cage.faces[fi];
    const n = face.length;
    for (let c = 0; c < n; c++) {
      const va = face[c], vb = face[(c + 1) % n];
      const key = edgeKey(va, vb);
      const edge = topology.edgeMap.get(key);
      const selCount = edge.faces.filter((f) => selSet.has(f)).length;
      if (selCount !== 1) continue; // interior to the selection (2 selected faces share it) — not part of the rim
      if (edge.faces.length !== 2) throw new Error(`orderedBoundaryLoopOfFaceSet: the selection's own rim touches an already-open cage edge ("${key}") — Bridge v1 needs a fully interior patch, surrounded by other faces on every side`);
      if (nextOf.has(va)) throw new Error(`orderedBoundaryLoopOfFaceSet: vertex ${va} has more than one rim continuation — the selection's own boundary is not a single simple loop`);
      nextOf.set(va, vb);
    }
  }
  if (!nextOf.size) throw new Error('orderedBoundaryLoopOfFaceSet: the selected faces have no rim at all (they cover the whole cage) — nothing to bridge from');
  const startVertex = nextOf.keys().next().value;
  const loop = [startVertex];
  let cur = startVertex;
  for (;;) {
    const next = nextOf.get(cur);
    if (next === undefined) throw new Error('orderedBoundaryLoopOfFaceSet: the selection\'s own rim does not close into a simple loop');
    if (next === startVertex) break;
    if (loop.includes(next)) throw new Error('orderedBoundaryLoopOfFaceSet: the selection\'s own rim revisits a vertex without closing — not a simple loop');
    loop.push(next);
    cur = next;
    if (loop.length > cage.vertices.length) throw new Error('orderedBoundaryLoopOfFaceSet: the rim walk failed to close within a reasonable number of steps');
  }
  return loop;
}

// BRIDGE — milestone 6: "connect two face/edge-loop selections
// with a tunnel of faces, Segments chip." v1 SCOPE, stated honestly: exactly
// TWO FACE selections (each a single simply-connected interior patch, per
// orderedBoundaryLoopOfFaceSet's own refusal above) on the SAME SuperB
// object — feeding it two already-open EDGE-LOOP selections directly (e.g.
// two holes DELETE FACES already made) is a real, separate generalization,
// not wired (a student reaches that case today by re-selecting
// the hole's own rim faces — there are none once deleted — so this is a
// genuine, named gap, not a workaround). Refuses honestly, naming the real
// counts, when the two rims don't have matching vertex counts.
//
// CORRESPONDENCE — which vertex of loop A lines up with which of loop B —
// splits into two genuinely different questions, one deterministic and one
// that genuinely needs a heuristic:
//   DIRECTION is NOT a free choice — both loops come out of
//   orderedBoundaryLoopOfFaceSet in the SAME "correct if independently
//   filled with one face" convention (proven directly, see that function's
//   own header), and a tunnel connecting two such loops always needs
//   exactly ONE of them walked in REVERSE of its own fill-direction (the
//   standard fact that a cylinder wall's two rims run opposite senses
//   relative to how each would independently cap off alone) — this module
//   always reverses loop B, never loop A, a fixed, arbitrary-but-consistent
//   choice (reversing A instead would work identically by symmetry).
//   Proven directly: this module's own test file checks the RESULT is a
//   fully consistent 2-manifold (every shared edge traversed in opposite
//   directions by its two faces) on multiple real fixtures, not merely
//   "looks plausible."
//   ROTATION (which vertex of the now-correctly-reversed loop B lines up
//   with loop A's own start) has no single correct answer for two
//   independently-selected patches — resolved via the standard "bridge two
//   loops" heuristic (the same idea Blender's own Bridge Edge Loops uses):
//   search every rotation, keep whichever minimizes the total squared
//   distance between corresponding vertex pairs. No manual per-vertex/twist
//   override — a real, stated limitation for a pathological
//   pair of loops where the nearest-distance heuristic picks a visually
//   twisted correspondence; the common, intended case (two roughly facing
//   openings) is exact and unambiguous.
export function bridgeFaces(cage, faceIndicesA, faceIndicesB, segments = 1, straightness = 1, creaseWeight = 0) {
  if (!faceIndicesA || !faceIndicesA.length || !faceIndicesB || !faceIndicesB.length) throw new Error('bridgeFaces: faceIndicesA and faceIndicesB must both be non-empty arrays');
  if (!Number.isInteger(segments) || segments < 1) throw new Error('bridgeFaces: segments must be a positive integer');
  const setA = new Set(faceIndicesA), setB = new Set(faceIndicesB);
  for (const fi of faceIndicesA) if (setB.has(fi)) throw new Error(`bridgeFaces: face ${fi} is in both selections — they must be disjoint`);

  const loopA0 = orderedBoundaryLoopOfFaceSet(cage, faceIndicesA);
  const loopB0raw = orderedBoundaryLoopOfFaceSet(cage, faceIndicesB);
  // The two patches are deleted FIRST — that is what turns a face selection
  // into a pair of open rims, which is the only shape the tunnel builder
  // below ever works on. An ALREADY-open pair of rims (two holes a student
  // made earlier) skips this step entirely and calls the same builder
  // directly: see bridgeBoundaryLoops below.
  const del = deleteFaces(cage, [...faceIndicesA, ...faceIndicesB]);
  return bridgeRims(cage, del.cage, del.vertexRemap, loopA0, loopB0raw, segments, 'bridgeFaces', straightness, creaseWeight);
}

// BRIDGE ON TWO ALREADY-OPEN EDGE LOOPS — the generalization bridgeFaces'
// own header named as unwired ("feeding it two already-open EDGE-LOOP
// selections directly is a real, separate generalization"), asked for
// directly: bridge should work on a pair of open edge loops, not only on two
// face selections. Each seed is ONE boundary edge of its own hole; the rest
// of that hole's rim is walked from it by boundaryLoopFromSeed, which is
// exactly the rim-finding half that already existed.
//
// WHY THIS NEEDS NO NEW ORIENTATION REASONING: boundaryLoopFromSeed and
// orderedBoundaryLoopOfFaceSet both return their loop in the SAME
// convention — "the order a single new face filling this rim would use" —
// each derived and proven independently in its own header above. So both
// paths hand the tunnel builder loops of identical meaning, and the builder's
// own reverse-B-then-rotate correspondence applies unchanged.
export function bridgeBoundaryLoops(cage, seedKeyA, seedKeyB, segments = 1, straightness = 1, creaseWeight = 0) {
  if (!Number.isInteger(segments) || segments < 1) throw new Error('bridgeBoundaryLoops: segments must be a positive integer');
  const loopA0 = boundaryLoopFromSeed(cage, seedKeyA);
  const loopB0raw = boundaryLoopFromSeed(cage, seedKeyB);
  // Two seeds on the SAME rim would walk the identical loop and "bridge" a
  // hole to itself. Caught by identity here, by name, rather than left to
  // the shared-vertex refusal below (which would also fire, but with a
  // message about two openings touching, which is not what happened).
  if (loopA0.length === loopB0raw.length && loopA0.every((v) => loopB0raw.includes(v))) {
    throw new Error(`bridgeBoundaryLoops: "${seedKeyA}" and "${seedKeyB}" are both on the SAME open boundary loop — Bridge needs one edge from each of TWO different holes`);
  }
  // Nothing is deleted here: the rims are already open, so the "cage the
  // tunnel is built onto" IS this cage, and every rim vertex keeps its own
  // index (an identity remap, where the face path needs a real one).
  const workCage = { vertices: cage.vertices.map((v) => v.slice()), faces: cage.faces.map((f) => [...f]), creases: { ...(cage.creases || {}) } };
  const identity = new Map(cage.vertices.map((_, i) => [i, i]));
  return bridgeRims(cage, workCage, identity, loopA0, loopB0raw, segments, 'bridgeBoundaryLoops', straightness, creaseWeight);
}

// The tunnel builder both Bridge paths share: given two rims (in fill-face
// order, per the two walkers' own headers), the cage those rims are open on,
// and how each rim's vertex indices map into it, build the ring-by-ring wall
// between them. Factored out when Bridge grew its second entry point rather
// than duplicated — the correspondence search, the duplicate-edge refusal
// and the winding are all one definition, so the two paths cannot drift.
// UNEQUAL RIMS — the monotone correspondence, and the whole of what
// "the bridge resamples" means here.
//
// THE RESAMPLE HAPPENS INSIDE THE BRIDGE, NEVER ON A RIM. Both rims keep
// every vertex they had, in place: nothing is inserted into them, nothing is
// removed, nothing moves. That is what lets this coexist with the rule the
// old refusal was protecting ("matching one rim to another by resampling
// would move vertices that already exist, which a bridge may never do") —
// the reconciliation lives entirely in the NEW faces.
//
// `c[j] = floor(j * nCoarse / nFine)` is monotone and advances by exactly 0
// or 1 per step, so walking the fine rim once emits exactly `nCoarse` quads
// and `nFine - nCoarse` triangles. A 12-rim bridged to an 8-rim gives 8
// quads and 4 triangles.
//
// THE TRIANGLES ARE THE HONEST COST, stated rather than hidden: each one
// becomes a valence-3 extraordinary vertex under Catmull-Clark, so the limit
// surface is slightly less even there than an all-quad bridge between
// matched rims. That is a real difference a modeler can see if they look
// for it, and it is strictly better than the alternative of refusing, or of
// silently editing one of the two objects the student built.
//
// WHERE THE TRIANGLES LAND IS NOT ARBITRARY: they are distributed evenly
// around the ring by the floor() itself, not bunched at the seam, because
// `c` advances on a uniform schedule rather than after a run of steps.
function rimCorrespondence(nFine, nCoarse) {
  const c = [];
  for (let j = 0; j <= nFine; j++) c.push(Math.floor((j * nCoarse) / nFine));
  return c; // c[nFine] === nCoarse, i.e. the wrap back to coarse[0]
}
// One band of the tunnel wall, between two rings of possibly different
// counts. Equal counts take the original quad path unchanged, in the
// original order, so every existing Bridge result is byte-identical.
function bridgeBandFaces(ringNear, ringFar) {
  const p = ringNear.length, q = ringFar.length;
  const out = [];
  if (p === q) {
    for (let i = 0; i < p; i++) {
      out.push([ringNear[i], ringNear[(i + 1) % p], ringFar[(i + 1) % q], ringFar[i]]);
    }
    return out;
  }
  // The winding below is the equal-count face [a, b, bFar, aFar] with the
  // repeated far (or near) vertex dropped — so a triangle is that same quad
  // degenerating, not a separately-derived orientation.
  if (p > q) {
    const c = rimCorrespondence(p, q);
    for (let j = 0; j < p; j++) {
      const a = ringNear[j], b = ringNear[(j + 1) % p];
      const f0 = ringFar[c[j] % q], f1 = ringFar[c[j + 1] % q];
      out.push(f0 === f1 ? [a, b, f0] : [a, b, f1, f0]);
    }
    return out;
  }
  const c = rimCorrespondence(q, p);
  for (let j = 0; j < q; j++) {
    const bFar = ringFar[j], aFar = ringFar[(j + 1) % q];
    const n0 = ringNear[c[j] % p], n1 = ringNear[c[j + 1] % p];
    out.push(n0 === n1 ? [n0, aFar, bFar] : [n0, n1, aFar, bFar]);
  }
  return out;
}
// THE OPEN-RUN SIBLING of rimCorrespondence, and the difference is the whole
// point: a rim WRAPS, so its correspondence is a uniform schedule that lands
// back on coarse[0] after a full turn and has no distinguished starting
// vertex. A RUN has ENDS, and the ends are not free — the first vertex of one
// run must meet the first of the other and the last must meet the last, or the
// wall would run off the end of the shorter run and leave a dangling stub. So
// this map is monotone with c[0] === 0 and c[p - 1] === q - 1 (both ends
// PINNED), and the interior is spread evenly between them by the round()
// itself rather than bunched against either end.
//
// `p >= 2` is guaranteed by the caller's own "at least 2 vertices" refusal, so
// the p - 1 divisor is never zero.
function openRunCorrespondence(p, q) {
  const c = [];
  for (let j = 0; j < p; j++) c.push(Math.round((j * (q - 1)) / (p - 1)));
  return c;
}
// One band of an OPEN bridge wall, between two rows of possibly different
// counts — the open-chain counterpart of bridgeBandFaces, which wraps with
// `% p` because a rim closes. Nothing here wraps: the walk stops one short of
// the end of whichever row is FINE, so the fine row's own edges each get
// exactly one face and the coarse row stalls where the correspondence does.
//
// Equal counts take the original quad path unchanged, in the original order
// and in whichever of the two windings the caller's orientation search
// settled on, so every existing Bridge result is byte-identical.
function bridgeRunBandFaces(rowNear, rowFar, flip) {
  const p = rowNear.length, q = rowFar.length;
  const out = [];
  if (p === q) {
    for (let i = 0; i + 1 < p; i++) {
      out.push(flip ? [rowNear[i + 1], rowNear[i], rowFar[i], rowFar[i + 1]] : [rowNear[i], rowNear[i + 1], rowFar[i + 1], rowFar[i]]);
    }
    return out;
  }
  // A stalled step repeats a vertex, and the face it produces is that same
  // quad with the repeat dropped — so a triangle here is the quad
  // degenerating, not a separately-derived orientation that could disagree
  // with its neighbors' winding.
  const push = (a, b, f0, f1) => {
    if (a === b) out.push(flip ? [a, f0, f1] : [a, f1, f0]);
    else if (f0 === f1) out.push(flip ? [b, a, f0] : [a, b, f0]);
    else out.push(flip ? [b, a, f0, f1] : [a, b, f1, f0]);
  };
  if (p > q) {
    const c = openRunCorrespondence(p, q);
    for (let j = 0; j + 1 < p; j++) push(rowNear[j], rowNear[j + 1], rowFar[c[j]], rowFar[c[j + 1]]);
    return out;
  }
  const c = openRunCorrespondence(q, p);
  for (let j = 0; j + 1 < q; j++) push(rowNear[c[j]], rowNear[c[j + 1]], rowFar[j], rowFar[j + 1]);
  return out;
}
function bridgeRims(sourceCage, workCage, vertexRemap, loopA0, loopB0raw, segments, label, straightness = 1, creaseWeight = 0) {
  const cage = sourceCage;
  // SHARED-VERTEX REFUSAL — a real gap in the
  // FIRST fix (the rungA===rungB check a few lines below only rejects a
  // rotation that pairs the shared vertex with ITSELF; it does nothing for
  // a rotation where the shared vertex survives paired with something
  // else). Two real, separately-found failure modes if this isn't caught
  // up front: (1) the "closest rotation that doesn't duplicate anything"
  // search can still ACCEPT a rotation where the shared vertex simply
  // isn't in either rung pair that round — the tunnel builds successfully,
  // but that shared vertex's own face neighborhood is no longer a topological
  // disk (a genuine vertex-non-manifold pinch — the limit surface
  // collapses to a point there), invisible to every check in this module,
  // which all reason about EDGES, not vertex link-disks; (2) if the two
  // loops share TWO vertices without sharing an edge, some rotation can
  // pair them crosswise (A[i]=u,B[i]=v and A[j]=v,B[j]=u), creating the
  // SAME rung edge (u,v) twice — a real edge with 4 incident faces, only
  // caught by the existing edgeMap check if (u,v) happens to already
  // exist elsewhere, not in general. Both vanish together by refusing
  // outright the instant the two loops' own vertex SETS intersect at
  // all — two openings that already touch have no genuine gap to tunnel
  // through, matching this project's own "refuse rather than guess"
  // standard (the same posture `stitchEdgeRuns`, one function below,
  // already takes for its own runA/runB shared-vertex case).
  const sharedVertex = loopA0.find((vi) => loopB0raw.includes(vi));
  if (sharedVertex !== undefined) {
    throw new Error(`${label}: the two boundary loops share vertex ${sharedVertex} — they already touch, with no real gap to tunnel through (pick two openings separated by real, uninterrupted cage surface)`);
  }
  const nA = loopA0.length, nB = loopB0raw.length;
  // Every ring the tunnel builds internally carries the FINE count, so the
  // reconciliation happens in exactly one band — the one touching the coarse
  // rim — rather than being smeared through every intermediate ring.
  const nF = Math.max(nA, nB);
  const n = nA; // kept: every equal-count expression below reads the same as before
  const loopB0 = [...loopB0raw].reverse(); // deterministic — see this function's own header
  // The rung pairs for one candidate alignment: which A vertex each B vertex
  // reaches across to. With equal counts this is exactly index-to-index, so
  // the scoring, the duplicate-edge check and the rings below are unchanged.
  function rungPairs(seqB) {
    const pairs = [];
    if (nA === nB) { for (let i = 0; i < nA; i++) pairs.push([i, i]); return pairs; }
    if (nA < nB) { const c = rimCorrespondence(nB, nA); for (let j = 0; j < nB; j++) pairs.push([c[j], j]); return pairs; }
    const c = rimCorrespondence(nA, nB);
    for (let i = 0; i < nA; i++) pairs.push([i, c[i]]);
    return pairs;
  }

  // ROTATION VALIDITY — a real, found-via-testing correctness gap, not
  // theoretical: on a cage with real internal symmetry (e.g. a plain box),
  // SOME rotation offsets pair loopA[i] with a loopB[i] that ALREADY sits
  // exactly one edge away via a completely different, already-kept face
  // (the two openings' own rims happen to share a "diagonal" neighbor) —
  // building the tunnel with that offset would create a rung edge
  // DUPLICATING one that already exists elsewhere in the cage, a genuine
  // non-manifold result (an edge used by 4 faces) that the distance
  // heuristic alone can't see (a duplicate can be the SHORTEST rung, not
  // the longest). So every rotation candidate is checked against the
  // cage's OWN real topology (after deletion, so only genuinely surviving
  // faces are considered) before ever being accepted, and candidates are
  // tried in ascending-distance order — the closest ROTATION THAT DOESN'T
  // DUPLICATE ANYTHING wins, not just the closest one, period.
  const del = { cage: workCage, vertexRemap };
  const delTopology = buildTopology(workCage);
  function rotate(arr, k) { return arr.slice(k).concat(arr.slice(0, k)); }
  function totalDistSq(seqB) {
    let s = 0;
    for (const [ai, bi] of rungPairs(seqB)) s += distSq(cage.vertices[loopA0[ai]], cage.vertices[seqB[bi]]);
    return s;
  }
  function wouldDuplicateExistingEdge(seqB0) {
    for (const [ai, bi] of rungPairs(seqB0)) {
      const rungA = del.vertexRemap.get(loopA0[ai]), rungB = del.vertexRemap.get(seqB0[bi]);
      if (rungA === undefined || rungB === undefined) return true; // shouldn't happen for a well-formed selection; treat as invalid defensively
      // A real, found-via-review gap: when the two selections' rims share a
      // cage vertex WITHOUT sharing an edge (e.g. two diagonal faces of the
      // same 2x2 grid, meeting only at the shared center vertex), that
      // shared vertex survives deleteFaces (it's still used by the
      // surviving, non-selected faces) and rungA===rungB for whichever
      // rotation happens to pair them — a DEGENERATE rung (a "tunnel" edge
      // from a vertex to itself), not caught by the edgeMap check above
      // (a real cage never has a self-loop edge to find), and worse: this
      // exact degenerate pairing has distSq===0, so the "closest rotation"
      // heuristic actively PREFERS it over every real, non-degenerate
      // rotation. Reject it explicitly, the same way stitchEdgeRuns (one
      // function below) already refuses runA/runB sharing a vertex.
      if (rungA === rungB) return true;
      if (delTopology.edgeMap.has(edgeKey(rungA, rungB))) return true;
    }
    return false;
  }
  const candidates = [];
  for (let k = 0; k < nB; k++) {
    const candidate = rotate(loopB0, k);
    candidates.push({ candidate, dist: totalDistSq(candidate) });
  }
  candidates.sort((x, y) => x.dist - y.dist);
  const validCandidate = candidates.find((c) => !wouldDuplicateExistingEdge(c.candidate));
  if (!validCandidate) {
    throw new Error(`${label}: every possible correspondence between the two rims would duplicate an edge the cage already has elsewhere (the two openings are already directly connected by existing geometry) — pick two openings with a genuine gap of empty space between them`);
  }
  const alignedLoopB0 = validCandidate.candidate;

  const loopA = loopA0.map((vi) => del.vertexRemap.get(vi));
  const loopB = alignedLoopB0.map((vi) => del.vertexRemap.get(vi));
  if (loopA.some((v) => v === undefined) || loopB.some((v) => v === undefined)) {
    throw new Error(`${label}: internal error — a rim vertex did not survive face deletion (this should not happen for a well-formed selection)`);
  }

  const newVertices = del.cage.vertices.map((v) => v.slice());
  const ptsA = loopA0.map((vi) => cage.vertices[vi]);
  const ptsB = alignedLoopB0.map((vi) => cage.vertices[vi]);
  // Tangents are read off workCage (post-deletion, where the rims are
  // genuinely naked) using the REMAPPED indices, while positions come from
  // the source cage — deleteFaces preserves every surviving vertex position
  // exactly, so the two agree by construction.
  // The corresponded partner point for each rim vertex — with equal counts
  // this is ptsB/ptsA verbatim, so the tangents are unchanged. With unequal
  // counts a coarse vertex has several partners; the FIRST is used, since
  // this only aims the tangent and any of them points the same way.
  const pairs = rungPairs(alignedLoopB0);
  const partnerOfA = new Array(nA), partnerOfB = new Array(nB);
  for (const [ai, bi] of pairs) {
    if (partnerOfA[ai] === undefined) partnerOfA[ai] = ptsB[bi];
    if (partnerOfB[bi] === undefined) partnerOfB[bi] = ptsA[ai];
  }
  const tanA = rimTangents(workCage, delTopology, loopA, partnerOfA, true, 1);
  const tanB = rimTangents(workCage, delTopology, loopB, partnerOfB, true, -1);
  const rings = [loopA];
  for (let k = 1; k < segments; k++) {
    const t = k / segments;
    const ring = [];
    // An intermediate ring carries the FINE count, so the tunnel keeps its
    // full resolution all the way along and only the single band against the
    // coarse rim reconciles. At t < 1 the several fine points sharing one
    // coarse partner are still genuinely distinct, so no face degenerates.
    for (let j = 0; j < nF; j++) {
      const [ai, bi] = pairs[j % pairs.length];
      ring.push(newVertices.length);
      newVertices.push(bridgeSpanPoint(ptsA[ai], ptsB[bi], tanA[ai], tanB[bi], t, straightness));
    }
    rings.push(ring);
  }
  rings.push(loopB);

  // Face winding [a, b, bFar, aFar] (both rings walked FORWARD, in their
  // own already-correct direction — loopB was already reversed once above,
  // deterministically, so no second reversal belongs here) — derived
  // empirically against a real fixture (two SEPARATE boxes with a genuine
  // gap between them, one opened face each — the realistic Bridge scenario,
  // not two already-side-connected faces of the SAME closed box, which has
  // no true "gap" to tunnel through at all and is a poor orientation test
  // for exactly that reason), proven by this module's own test file via a
  // real 2-manifold/consistent-winding check (every edge shared by exactly
  // 2 faces traverses it in opposite directions), not assumed from a
  // plausible-looking formula.
  const newFaces = [...del.cage.faces];
  for (let r = 0; r < rings.length - 1; r++) {
    for (const f of bridgeBandFaces(rings[r], rings[r + 1])) newFaces.push(f);
  }
  // Rim edges only, and on the REMAPPED indices — the crease map is keyed by the
  // indices the returned cage actually uses, which after deleteFaces are not the
  // source cage's own.
  const outCreases = { ...del.cage.creases };
  creaseChain(outCreases, loopA, creaseWeight, true);
  creaseChain(outCreases, loopB, creaseWeight, true);
  return {
    cage: { vertices: newVertices, faces: newFaces, creases: outCreases },
    tunnelFaceIndices: newFaces.map((_, i) => i).slice(del.cage.faces.length),
  };
}

// BRIDGE ON TWO OPEN EDGE RUNS — the full generalization asked for directly:
// "Any 1 edge to any other 1 edge. Or 2 to 2 etc. Different bodies, one or
// two holes etc." This is the SAME-HOLE and N-to-N case, and it is
// genuinely EASIER than the closed-loop Bridge above, not harder — worth
// stating plainly, because the reverse is the natural assumption:
//
//   A CLOSED loop has ROTATIONAL freedom, which is why bridgeRims runs an
//   n-way rotation search scored by total squared distance, and why it needs
//   a duplicate-edge validity check to stop the closest rotation from being
//   a non-manifold one. Two OPEN runs have ENDS. There is no rotation at
//   all — only a direction choice, forward or reversed, exactly 2
//   candidates.
//
// WHAT DECIDES WHAT, and this is the real design point: ORIENTATION is not a
// free choice and is not derived by hand here — a consistently-wound
// 2-manifold never traverses the same DIRECTED edge twice, so a candidate
// rung set that would reuse one is simply the wrong winding, and
// directedEdgeReuseCount decides it by direct verification of that property.
// CORRESPONDENCE (which end of runA meets which end of runB) is the one
// genuinely geometric question, and total squared distance answers it. So
// topology settles orientation, geometry settles correspondence, and neither
// is left to a plausible-looking formula.
//
// ADDITIVE ONLY — unlike bridgeFaces, nothing is deleted (the runs are
// already naked edges), so there is no vertexRemap to thread anywhere and
// every existing vertex index stays valid.
//
// TOPOLOGY, named honestly: bridging two runs of the SAME hole SPLITS that
// one boundary loop into two, where the closed-loop Bridge above MERGES two
// loops into a tunnel. Both are legal on a Catmull-Clark cage; the result
// here is still open (the two ends of the new wall are naked edges, as they
// must be), so a caller should not expect a closed solid from this.
//
// UNEQUAL COUNTS ARE SUPPORTED, by the same means the closed-loop Bridge
// above already uses and for the same reason: nothing is resampled, because a
// bridge may never move a vertex that already exists. The fine run drives the
// walk, the coarse run stalls where openRunCorrespondence says it does, and
// each stalled step's quad degenerates to a triangle — `p - q` of them for a
// p-to-q bridge, spread evenly along the wall rather than bunched at one end.
// The triangles are the honest cost (each becomes a valence-3 extraordinary
// vertex under Catmull-Clark), and the ENDS are pinned first-to-first and
// last-to-last, which is the one freedom a closed rim has and an open run
// does not.
// AN OPEN RUN IS NOT A CLOSED RIM, and nothing here used to say so. A run's
// own precondition — "every consecutive pair is a real naked edge" — is
// satisfied just as well by a CLOSED loop handed over as a plain vertex list,
// because a closed loop's consecutive pairs are all naked edges too. The
// result is not a refusal: it is a junction that builds cleanly, winds
// consistently, has no repeated-vertex face, and leaves every rim's own
// CLOSING edge unattached — a slit down each arm, invisible to every
// structural check this module makes. Measured on three facets-6
// subdPipeCage tubes: accepted, 18 naked edges before and 9 after, where a
// genuine closed-rim junction leaves none.
//
// Two distinct shapes of closed input, both refused here by name:
//   - the bare cycle [v0..v_{m-1}], detected by its own closing edge existing
//     in the cage as a naked edge between the run's two ENDS. A genuinely
//     open run can only have that if it already spans its whole rim bar one
//     edge — which is the same loop under another name, so refusing it is
//     right rather than over-strict.
//   - the wrapped list [v0..v_{m-1}, v0], which is not a simple chain at all.
// Length 2 is exempt from the first check: a 2-vertex run's own single edge
// IS the edge between its ends, and no loop can close in two vertices.
//
// `wording` exists because the CONSEQUENCE of handing a closed rim to an
// open-run function is not the same for every caller, and a refusal that names
// the wrong consequence is worse than a generic one. A bridge BUILDS faces onto
// the run and leaves the rim's own closing edge unattached; a weld builds
// nothing and fails a different way (see stitchEdgeRuns). The checks themselves
// are shared — one guard, not a second near-copy that later drifts — and the
// defaults reproduce the bridge wording exactly, so a caller that passes
// nothing is byte-identical to before this parameter existed.
const OPEN_RUN_WORDING_DEFAULT = {
  verb: 'bridge',
  closedTail: 'Bridging it as a run would leave that one edge unattached, a slit along the arm. Use bridgeClosedRimsHub for a junction of closed rims, or bridgeBoundaryLoops for exactly two of them',
};
function checkOpenRunChain(cage, topology, run, label, who, wording = OPEN_RUN_WORDING_DEFAULT) {
  if (new Set(run).size !== run.length) throw new Error(`${who}: ${label} repeats a vertex — an edge run has to be a simple open chain, and a rim written as a wrapped list (its first vertex again at the end) is a CLOSED loop, which this function cannot ${wording.verb}`);
  for (let i = 0; i + 1 < run.length; i++) {
    const key = edgeKey(run[i], run[i + 1]);
    const edge = topology.edgeMap.get(key);
    if (!edge) throw new Error(`${who}: ${label} is not a connected chain — vertices ${run[i]} and ${run[i + 1]} are not joined by a real cage edge`);
    if (edge.faces.length !== 1) throw new Error(`${who}: ${label}'s edge "${key}" is not a naked (open) edge — it already has ${edge.faces.length} faces, so there is no opening there to ${wording.verb} from`);
  }
  if (run.length >= 3) {
    const closing = topology.edgeMap.get(edgeKey(run[0], run[run.length - 1]));
    if (closing && closing.faces.length === 1) throw new Error(`${who}: ${label} is a CLOSED rim, not an open run — its two ends are already joined by the naked edge "${edgeKey(run[0], run[run.length - 1])}". ${wording.closedTail}`);
  }
}

export function bridgeEdgeRuns(cage, runA, runB, segments = 1, straightness = 1, creaseWeight = 0) {
  if (!Array.isArray(runA) || !Array.isArray(runB)) throw new Error('bridgeEdgeRuns: runA and runB must both be arrays of vertex indices');
  if (runA.length < 2 || runB.length < 2) throw new Error('bridgeEdgeRuns: each run needs at least 2 vertices (at least one edge)');
  if (!Number.isInteger(segments) || segments < 1) throw new Error('bridgeEdgeRuns: segments must be a positive integer');
  if (!(straightness >= 0 && straightness <= 1)) throw new Error('bridgeEdgeRuns: straightness must be between 0 and 1');

  // Two runs that already touch have no genuine gap to bridge, and a shared
  // vertex would pinch the result vertex-non-manifold — the same refusal, for
  // the same reason, that bridgeRims and stitchEdgeRuns both already make.
  const shared = runA.find((vi) => runB.includes(vi));
  if (shared !== undefined) throw new Error(`bridgeEdgeRuns: the two edge runs share vertex ${shared} — they already touch, with no real gap to bridge across`);

  const topology = buildTopology(cage);
  checkOpenRunChain(cage, topology, runA, 'runA', 'bridgeEdgeRuns');
  checkOpenRunChain(cage, topology, runB, 'runB', 'bridgeEdgeRuns');

  const nA = runA.length, nB = runB.length;
  // Which runB vertex each runA vertex reaches across to, and vice versa —
  // the same pairing bridgeRunBandFaces walks, needed here for the distance
  // score and again below for the tangents and the interior rows. With equal
  // counts it is exactly index-to-index, so every expression that reads it is
  // unchanged. An open run has no rotational freedom, so unlike bridgeRims'
  // own rungPairs this depends on the counts alone and is built once.
  const pairs = [];
  if (nA === nB) { for (let i = 0; i < nA; i++) pairs.push([i, i]); }
  else if (nA < nB) { const c = openRunCorrespondence(nB, nA); for (let j = 0; j < nB; j++) pairs.push([c[j], j]); }
  else { const c = openRunCorrespondence(nA, nB); for (let i = 0; i < nA; i++) pairs.push([i, c[i]]); }

  const candidates = [];
  for (const reversed of [false, true]) {
    const seqB = reversed ? [...runB].reverse() : [...runB];
    // Scored over CORRESPONDING pairs, not over a shared index — with unequal
    // counts a plain index walk would score the two runs against vertices that
    // never meet, and would not even visit the longer run's tail.
    let dist = 0;
    for (const [ai, bi] of pairs) dist += distSq(cage.vertices[runA[ai]], cage.vertices[seqB[bi]]);
    for (const flip of [false, true]) {
      // Probed through the band helper itself, so directedEdgeReuseCount
      // judges the topology the bridge will actually build — including the
      // degenerate triangles, which carry the rim edges that decide the flip.
      const faces = bridgeRunBandFaces(runA, seqB, flip);
      candidates.push({ seqB, flip, dist, reuse: directedEdgeReuseCount([...cage.faces, ...faces]) });
    }
  }
  const valid = candidates.filter((c) => c.reuse === 0);
  if (!valid.length) throw new Error('bridgeEdgeRuns: no correspondence between these two edge runs produces a consistently-wound result — the runs may not both bound the same open region, or already meet through existing geometry');
  valid.sort((x, y) => x.dist - y.dist);
  const chosen = valid[0];
  const seqB = chosen.seqB;

  // TWISTED-BRIDGE SIGNAL — a real case found by testing, not anticipated:
  // winding validity is a HARD constraint and distance is only a preference,
  // so when the two runs' own surfaces are oppositely oriented (each face
  // normal pointing the same way rather than continuing one surface), the
  // NEAREST correspondence is not manifold-legal and the only legal one pairs
  // each rim vertex with the FAR end of the other run — a genuinely twisted
  // wall whose rungs cross in the middle. That is still valid cage topology,
  // so refusing outright would block a legitimate join between two
  // independently-built bodies; but it is almost never what a student meant,
  // and saying nothing would be exactly the silent-wrong-result this project
  // refuses everywhere else. So it is built AND reported, for the caller to
  // name in its own status line.
  const nearestDist = Math.min(...candidates.map((c) => c.dist));
  const twisted = chosen.dist > nearestDist * (1 + 1e-9);

  const ptsA = runA.map((vi) => cage.vertices[vi]);
  const ptsB = seqB.map((vi) => cage.vertices[vi]);
  // rimTangents wants one partner POINT per vertex of the run it is aiming.
  // With equal counts that is ptsB/ptsA verbatim, so the tangents are
  // unchanged. With unequal counts a coarse vertex has several partners; the
  // FIRST is used, since this only aims the tangent and any of them points
  // the same way — the same choice bridgeRims makes for its own rims.
  const partnerOfA = new Array(nA), partnerOfB = new Array(nB);
  for (const [ai, bi] of pairs) {
    if (partnerOfA[ai] === undefined) partnerOfA[ai] = ptsB[bi];
    if (partnerOfB[bi] === undefined) partnerOfB[bi] = ptsA[ai];
  }
  const tanA = rimTangents(cage, topology, runA, partnerOfA, false, 1);
  const tanB = rimTangents(cage, topology, seqB, partnerOfB, false, -1);

  const newVertices = cage.vertices.map((v) => v.slice());
  const rows = [runA];
  // Every interior row carries the FINE count (pairs.length is exactly that),
  // so the reconciliation happens in the single band touching the coarse run
  // and every other band is plain quads. At t < 1 the several fine points
  // sharing one coarse partner are still genuinely distinct, so no interior
  // face degenerates.
  for (let k = 1; k < segments; k++) {
    const t = k / segments;
    const row = [];
    for (let j = 0; j < pairs.length; j++) {
      const [ai, bi] = pairs[j];
      row.push(newVertices.length);
      newVertices.push(bridgeSpanPoint(ptsA[ai], ptsB[bi], tanA[ai], tanB[bi], t, straightness));
    }
    rows.push(row);
  }
  rows.push(seqB);

  const newFaces = cage.faces.map((f) => [...f]);
  const added = [];
  for (let r = 0; r < rows.length - 1; r++) {
    for (const f of bridgeRunBandFaces(rows[r], rows[r + 1], chosen.flip)) added.push(f);
  }
  // The orientation above was decided against a single-segment wall. A
  // multi-segment one repeats the same band pattern per row pair — but at
  // unequal counts only ONE of those bands is the reconciling one and the
  // rest are plain quads, so "holds by construction" is a weaker claim there
  // than it looks. Hence the check stays, verified rather than assumed, since
  // a silently non-manifold cage is the one outcome worth refusing outright.
  const finalFaces = [...newFaces, ...added];
  if (directedEdgeReuseCount(finalFaces) !== 0) throw new Error('bridgeEdgeRuns: internal error — the assembled bridge is not consistently wound (this should not happen for two well-formed open runs)');

  // Rim edges only — an OPEN run is not closed, so its last vertex has no edge
  // back to its first (passing closed:true here would invent one).
  const outCreases = { ...(cage.creases || {}) };
  creaseChain(outCreases, runA, creaseWeight, false);
  creaseChain(outCreases, seqB, creaseWeight, false);
  return {
    cage: { vertices: newVertices, faces: finalFaces, creases: outCreases },
    bridgeFaceIndices: added.map((_, i) => newFaces.length + i),
    twisted,
  };
}

// STITCH — milestone 7: "merge two edge runs into one (First/
// Second/Average position chips, tracked from Rhino's Stitch)." `runA`/
// `runB` are each an ORDERED array of vertex indices tracing a real,
// connected boundary chain (need not be a closed loop — an open "run" of
// consecutive boundary edges, exactly as Rhino's own Stitch expects two
// nearby-but-disconnected boundary edges/curves). Requires matching vertex
// counts, refused honestly by name otherwise — and unlike Bridge one command
// up, that is not a scope cut this could lift the same way. Bridge reconciles
// unequal counts by letting a quad DEGENERATE to a triangle, which costs only
// a face; a weld reconciles by IDENTIFYING vertices, so pairing two fine
// vertices onto one coarse one would merge them into each other and move
// geometry the student built.
//
// THE ACTUAL MECHANISM — a real topology-merge, not a face-rebuild: once
// each corresponding pair of vertices is IDENTIFIED as one vertex (moved to
// the position `position` dictates) and every face reference to the
// "losing" vertex is remapped onto the "keeping" one, whatever edge used to
// run ALONG one run (with exactly one adjacent face, e.g. runA[i]-runA[i+1])
// and whatever edge ran along the OTHER run (runB[i]-runB[i+1], also one
// adjacent face) become — after the remap — the exact SAME edgeKey, now used
// by BOTH original faces: a genuine 2-face INTERIOR edge, precisely closing
// the gap Stitch exists to close. No new faces are built or need to be —
// this is the entire operation.
//
// CORRESPONDENCE — an open run has no rotational ambiguity (unlike Bridge's
// closed loops); the only real question is which END matches which, i.e.
// forward vs reversed — resolved the same minimal-total-distance way Bridge
// resolves its own larger (rotation + direction) search.
//
// THAT SENTENCE IS THE WHOLE REASON A CLOSED RIM IS REFUSED HERE, and the
// failure it produces is NOT the same one the bridges have. A weld builds no
// faces, so there is no unattached closing edge and no slit: handed two
// facets-6 tube rims, this welds every edge, including each rim's own closing
// one — naked edge count goes 12 to 0. What it cannot do is find the ROTATION.
// Forward-vs-reversed is the entire search, so vertex i of one rim is paired
// with vertex i of the other wherever that happens to land: on two coaxial
// radius-5 rims the merged ring came back at radii 0, 4.33, 4.33, 0, 4.33,
// 4.33 instead of 5 throughout — pairs of diametrically opposite vertices
// averaged onto the axis — and all six welded edges were traversed twice in
// the SAME direction, so the result was not consistently wound either. Both
// failures are silent. A closed rim's rotation is exactly what bridgeRims'
// own n-way rotation search exists for; refusing here and naming that is the
// honest answer, not a rotation search bolted onto a merge.
//
// The three preconditions this shares with the bridges — a run is a simple
// chain, its consecutive pairs are real cage edges, and those edges are naked
// — were also unchecked, and each is separately reachable: a list of
// non-adjacent vertices welds unrelated points into a vertex pinch, and a run
// of INTERIOR (2-face) vertices produced edges with three faces, a genuinely
// non-manifold cage. checkOpenRunChain covers all of it.
const STITCH_OPEN_RUN_WORDING = {
  verb: 'weld',
  closedTail: 'A closed rim has rotational freedom this function does not search — it pairs the two rims by index, forward or reversed only — so on two ordinary tube rims it welds every edge and still crushes the merged ring toward the axis and leaves the result inconsistently wound. Use bridgeBoundaryLoops to join two closed rims, or bridgeClosedRimsHub for three or more',
};
export function stitchEdgeRuns(cage, runA, runB, position = 'average') {
  if (!Array.isArray(runA) || !Array.isArray(runB) || runA.length < 2 || runB.length < 2) throw new Error('stitchEdgeRuns: both runs need at least 2 vertices (at least one edge each)');
  if (runA.length !== runB.length) throw new Error(`stitchEdgeRuns: the two edge runs have different vertex counts (${runA.length} vs ${runB.length}) — Stitch needs matching counts`);
  if (!['first', 'second', 'average'].includes(position)) throw new Error(`stitchEdgeRuns: position must be "first", "second", or "average" (got "${position}")`);
  const n = runA.length;
  const setA = new Set(runA);
  for (const vi of runB) if (setA.has(vi)) throw new Error(`stitchEdgeRuns: vertex ${vi} appears in both runs — the two runs must be genuinely separate before stitching`);

  const topology = buildTopology(cage);
  checkOpenRunChain(cage, topology, runA, 'runA', 'stitchEdgeRuns', STITCH_OPEN_RUN_WORDING);
  checkOpenRunChain(cage, topology, runB, 'runB', 'stitchEdgeRuns', STITCH_OPEN_RUN_WORDING);

  function totalDistSq(seq) {
    let s = 0;
    for (let i = 0; i < n; i++) s += distSq(cage.vertices[runA[i]], cage.vertices[seq[i]]);
    return s;
  }
  const reversedRunB = [...runB].reverse();
  const alignedRunB = totalDistSq(reversedRunB) < totalDistSq(runB) ? reversedRunB : runB;

  const remap = new Map(); // old alignedRunB[i] index -> the surviving runA[i] index
  const mergedPositions = new Map(); // runA[i] index -> its own final [x,y,z]
  for (let i = 0; i < n; i++) {
    remap.set(alignedRunB[i], runA[i]);
    const pA = cage.vertices[runA[i]], pB = cage.vertices[alignedRunB[i]];
    mergedPositions.set(runA[i], position === 'first' ? pA.slice() : position === 'second' ? pB.slice() : lerp3(pA, pB, 0.5));
  }

  const workingVerts = cage.vertices.map((v, vi) => (mergedPositions.has(vi) ? mergedPositions.get(vi) : v.slice()));
  const removedSet = new Set(remap.keys());
  const oldToNew = new Map();
  const newVertices = [];
  workingVerts.forEach((v, vi) => {
    if (removedSet.has(vi)) return; // this index is being merged away; every reference to it redirects through `remap` below
    oldToNew.set(vi, newVertices.length);
    newVertices.push(v);
  });
  function finalIndexOf(oldIdx) {
    const redirected = remap.has(oldIdx) ? remap.get(oldIdx) : oldIdx;
    return oldToNew.get(redirected);
  }

  const remappedFaces = cage.faces.map((f) => f.map((vi) => finalIndexOf(vi)));
  // Defensive cleanup (not expected to fire for a genuinely well-formed
  // stitch of two separate runs, since a bijective per-index merge can't by
  // itself collapse two DIFFERENT already-distinct face vertices onto one
  // index): drop any now-consecutive-duplicate vertex within a face, and
  // drop any face that collapses below a real 3-sided minimum. `collapsedFaceCount`
  // reports this honestly rather than silently producing a corrupt cage.
  let collapsedFaceCount = 0;
  const newFaces = [];
  for (const f of remappedFaces) {
    const cleaned = [];
    for (const vi of f) if (cleaned.length === 0 || cleaned[cleaned.length - 1] !== vi) cleaned.push(vi);
    if (cleaned.length > 1 && cleaned[0] === cleaned[cleaned.length - 1]) cleaned.pop();
    if (cleaned.length < 3) { collapsedFaceCount++; continue; }
    newFaces.push(cleaned);
  }

  const newCreases = {};
  for (const [key, weight] of Object.entries(cage.creases || {})) {
    const [a, b] = key.split('_').map(Number);
    const na = finalIndexOf(a), nb = finalIndexOf(b);
    if (na === undefined || nb === undefined || na === nb) continue; // a dropped/merged-to-self edge — nothing real left to crease
    newCreases[edgeKey(na, nb)] = weight;
  }

  return {
    cage: { vertices: newVertices, faces: newFaces, creases: newCreases },
    mergedVertexIndices: runA.map((vi) => finalIndexOf(vi)),
    collapsedFaceCount,
  };
}

// SUBDIVIDE (global) — the v1.1-listed "SUBDIVIDE (global or
// per-face)". The GLOBAL case is the cheap, exact, zero-new-topology-risk
// one, and it is genuinely just a refinement: subdivideCatmullClark (the
// already-proven Catmull-Clark refinement step, kernel/subd.mjs) produces a
// FINER control cage whose LIMIT surface is EXACTLY the input cage's own
// limit surface — the defining property of a subdivision surface, not an
// approximation. So "subdivide the whole cage one level" and "commit that
// finer cage as the new live-editable cage" is one call: every quad becomes
// 4, an n-gon becomes n quads, the shape a student sees is unchanged, but
// there are now more editable control vertices to push and pull. Semi-sharp
// crease weights decrement by exactly 1 per level (the standard DeRose
// semi-sharp decay subdivideCatmullClark already does) — which is precisely
// what KEEPS the limit surface identical across the refinement, not a side
// effect to work around. Returns a brand-new cage (the input is never
// mutated), matching every other function in this file.
//
// This is deliberately the WHOLE-cage operation only. A genuine PER-FACE
// subdivide (refine only some selected faces, leaving the rest coarse) is a
// real, separate, HARDER problem, honestly deferred as a v1.1-follow-up:
// refining one face's edges without also splitting the shared edge of every
// un-refined neighbor leaves a T-junction / non-manifold cage, which
// subdivideCatmullClark (and Reflect/Bridge/every other edit here) all assume
// never happens; repairing that cleanly needs real edge-midpoint-insertion-
// and-neighbor-re-topologizing machinery (splitting each neighbor quad into a
// 5-gon, with crease/winding bookkeeping across the split), exactly the class
// of unverified topology surgery this project's own standing rule refuses to
// rush. Not attempted here — named honestly rather than shipped half-built or
// accepting a silently-corrupt non-manifold boundary.
export function subdivideCageGlobal(cage) {
  const refined = subdivideCatmullClark(cage); // validateCage runs inside
  return {
    cage: refined,
    faceCountBefore: cage.faces.length,
    faceCountAfter: refined.faces.length,
    vertexCountBefore: cage.vertices.length,
    vertexCountAfter: refined.vertices.length,
  };
}

// MERGEFACES — the v1.1-listed companion to SUBDIVIDE.
// Dissolves the shared INTERNAL edges between a connected group of 2+ edge-
// adjacent selected faces, replacing the whole group with ONE new N-gon face
// spanning their combined outer boundary (Rhino's own MergeFaces, exactly).
// The dissolved internal edges' own creases (if any) simply disappear along
// with the edges — the correct, expected behavior matching Rhino: there is no
// edge left there to carry a crease. Every SURVIVING (rim) edge keeps its own
// crease untouched.
//
// Reuses two already-proven functions in this file wholesale rather than re-
// deriving anything: orderedBoundaryLoopOfFaceSet gives the new face's own
// ordered vertex loop, wound in the SAME direction the selected faces
// themselves traverse it — which is exactly "opposite the surviving neighbor"
// (the standard manifold property: two faces sharing an edge traverse it in
// opposite directions), so the new n-gon is consistently oriented with its
// surviving neighbors by construction, the exact property that function was
// built and proven for. deleteFaces does the actual removal + orphaned-vertex
// prune/compaction + crease rebuild — it already DROPS any crease whose edge
// no longer exists in the output (which is EXACTLY what makes a dissolved
// internal edge's crease vanish correctly) and KEEPS every surviving edge's
// crease (the rim edges still exist via their un-selected neighbor while
// deleteFaces runs, so their creases survive). The new n-gon is then
// appended, its loop translated through deleteFaces' own vertexRemap.
//
// CONNECTED-GROUP REQUIREMENT, refused honestly (not guessed): the selected
// faces must form a SINGLE edge-connected group. A disconnected selection
// (e.g. two faces picked on opposite sides of the cage, sharing no edge) has
// no single merged n-gon to become — orderedBoundaryLoopOfFaceSet would
// silently walk only ONE component's rim and ignore the rest, a real silent-
// corruption risk, so it is refused up front by name, BEFORE that function is
// ever called. Merging one connected group at a time is the honest v1
// contract (the app layer partitions a multi-group selection and asks for one
// group at a time, matching Bridge's own identical single-patch posture). The
// rim also may not touch an already-open cage boundary edge
// (orderedBoundaryLoopOfFaceSet's own v1 cut, inherited here by name) — the
// same honest, stated limitation Bridge already has.
export function mergeFaces(cage, faceIndices) {
  if (!faceIndices || faceIndices.length < 2) throw new Error('mergeFaces: select at least 2 edge-adjacent faces to merge (merging a single face is a no-op)');
  const seen = new Set();
  for (const fi of faceIndices) {
    if (!Number.isInteger(fi) || fi < 0 || fi >= cage.faces.length) throw new Error(`mergeFaces: face index ${fi} is out of range`);
    if (seen.has(fi)) throw new Error(`mergeFaces: face index ${fi} was selected more than once`);
    seen.add(fi);
  }

  // Single-edge-connected-group check — MUST run before
  // orderedBoundaryLoopOfFaceSet (which would otherwise silently return just
  // one component's rim for a disconnected selection; see this function's own
  // header). Uses the same shared-edge adjacency the app layer's own
  // superbFaceGroupsFromSelection grouping uses.
  const topology = buildTopology(cage);
  const selSet = new Set(faceIndices);
  const visited = new Set([faceIndices[0]]);
  const queue = [faceIndices[0]];
  while (queue.length) {
    const fi = queue.shift();
    const face = cage.faces[fi];
    for (let c = 0; c < face.length; c++) {
      const edge = topology.edgeMap.get(edgeKey(face[c], face[(c + 1) % face.length]));
      for (const nfi of edge.faces) if (selSet.has(nfi) && !visited.has(nfi)) { visited.add(nfi); queue.push(nfi); }
    }
  }
  if (visited.size !== faceIndices.length) throw new Error(`mergeFaces: the selected faces are not all edge-connected into one group (${faceIndices.length} selected, only ${visited.size} reachable through shared edges) — merge one connected group at a time`);

  // Ordered rim of the merged region — throws on an open-boundary rim edge or
  // a non-simple rim (both honest v1 refusals inherited from this shared
  // function, see its own header).
  const loop = orderedBoundaryLoopOfFaceSet(cage, faceIndices);

  // Remove the group, prune the now-orphaned interior vertices, compact, and
  // rebuild creases (dropping the dissolved internal edges' own creases along
  // with their edges) — all in one already-proven call.
  const del = deleteFaces(cage, faceIndices);
  const newLoop = loop.map((vi) => del.vertexRemap.get(vi));
  if (newLoop.some((v) => v === undefined)) {
    // A rim vertex is shared with a surviving neighbor face, so it can never
    // be orphaned by deleteFaces — this is a genuine internal-consistency
    // guard, not an expected path.
    throw new Error('mergeFaces: internal error — a rim vertex did not survive the merge (this should not happen for a well-formed connected selection)');
  }
  const newFaces = [...del.cage.faces, newLoop];
  const newCage = { vertices: del.cage.vertices, faces: newFaces, creases: del.cage.creases };

  // MANIFOLD SAFETY NET — the same last-resort structural gate
  // kernel/subdconvert.mjs already runs on its own output (no face with a
  // repeated vertex; no edge shared by more than 2 faces). Not expected to
  // fire for a well-formed connected selection, but a topology edit has no
  // business ever committing a corrupt cage silently (this app's own "no
  // silent failure" standing rule), and the check is cheap.
  for (const f of newCage.faces) {
    if (new Set(f).size !== f.length) throw new Error('mergeFaces: the merge produced a face with a repeated vertex — refusing to build a corrupt cage');
  }
  const checkTopo = buildTopology(newCage);
  for (const e of checkTopo.edgeMap.values()) {
    if (e.faces.length > 2) throw new Error(`mergeFaces: the merge produced a non-manifold edge (between vertices ${e.v0} and ${e.v1}, shared by ${e.faces.length} faces) — refusing to build a corrupt cage`);
  }

  return {
    cage: newCage,
    faceIndex: del.cage.faces.length, // index of the new merged n-gon in newCage.faces
    mergedFaceCount: faceIndices.length,
    ngonSize: newLoop.length,
  };
}

// SUBDIVIDE SELECTED FACES — refine only what is picked, and leave the
// rest of the cage where it is. Direct ask: "when a face is selected in
// SuperB and subdivide is run it should subdivide ONLY that face/the
// selected faces."
//
// THE PROBLEM THIS HAS TO SOLVE, and the reason it was deferred once
// already: refining one face puts a new vertex at the MIDDLE of an edge
// that an unrefined neighbor still describes as a single side. That is
// a T-JUNCTION — the neighbor's own face loop never mentions the new
// vertex, so the two faces disagree about where their shared boundary
// is, and subdivideCatmullClark/Reflect/Bridge all inherit a cage that is
// no longer manifold. A crack that renders fine and corrupts quietly.
//
// THE FIX, and why it needs no new machinery: this kernel's Catmull-Clark
// accepts ANY face size — that property is already load-bearing elsewhere
// (superbCylinderCage's own n-gon caps rely on it), not a new assumption
// taken on for this. So the neighbor does not need splitting into quads.
// INSERT the new midpoint into the NEIGHBOR'S OWN VERTEX LOOP: a quad
// bordering one refined face becomes a 5-gon, one bordering two becomes a
// 6-gon, and the T-junction is gone BY CONSTRUCTION rather than by
// tolerance. Every edge still has exactly two faces that agree on it.
//
// WHAT THIS DELIBERATELY DOES NOT DO — and it is the honest difference
// from the GLOBAL SubdivideSubD: it does not move any existing vertex,
// and it places new vertices at plain edge midpoints and face centroids
// rather than through the smooth Catmull-Clark rules. Applying those
// rules locally would move the selection's own boundary vertices, which
// belong just as much to faces nobody asked to refine — the surface
// would change where the student did not touch it. The consequence,
// stated rather than glossed: a LOCAL subdivide genuinely does change
// the limit surface near the refined region, unlike the global one,
// which is exactly limit-preserving. Real Rhino's own local subdivide
// behaves the same way, for the same reason.
export function subdivideFaces(cage, faceIndices) {
  const selected = [...new Set(faceIndices)];
  if (!selected.length) throw new Error('subdivideFaces: no faces selected');
  for (const fi of selected) {
    if (!Number.isInteger(fi) || fi < 0 || fi >= cage.faces.length) throw new Error(`subdivideFaces: ${fi} is not a face of this cage`);
    if (cage.faces[fi].length < 3) throw new Error(`subdivideFaces: face ${fi} is degenerate`);
  }
  const selectedSet = new Set(selected);
  const newVertices = cage.vertices.map((v) => v.slice());

  // One edge point per DISTINCT edge of the selection — an edge shared by
  // two selected faces is discovered twice and built once, which is what
  // keeps those two faces agreeing on their shared boundary.
  const edgePointOf = new Map();
  const splitPairs = [];
  function edgePoint(a, b) {
    const key = edgeKey(a, b);
    let idx = edgePointOf.get(key);
    if (idx === undefined) {
      idx = newVertices.length;
      newVertices.push(scale(add(cage.vertices[a], cage.vertices[b]), 0.5));
      edgePointOf.set(key, idx);
      splitPairs.push([a, b, idx]);
    }
    return idx;
  }
  for (const fi of selected) {
    const f = cage.faces[fi];
    for (let c = 0; c < f.length; c++) edgePoint(f[c], f[(c + 1) % f.length]);
  }

  const newFaces = [];
  const insertedVertexIndices = [];
  let widenedNeighbours = 0;
  cage.faces.forEach((f, fi) => {
    if (selectedSet.has(fi)) {
      // n quads, exactly as one Catmull-Clark step would produce for this
      // face alone: corner, next edge point, face point, previous edge point.
      const centroid = f.reduce((acc, v) => add(acc, cage.vertices[v]), [0, 0, 0]);
      const fpIdx = newVertices.length;
      newVertices.push(scale(centroid, 1 / f.length));
      insertedVertexIndices.push(fpIdx);
      for (let c = 0; c < f.length; c++) {
        const vCurr = f[c];
        const eNext = edgePointOf.get(edgeKey(vCurr, f[(c + 1) % f.length]));
        const ePrev = edgePointOf.get(edgeKey(f[(c - 1 + f.length) % f.length], vCurr));
        newFaces.push([vCurr, eNext, fpIdx, ePrev]);
      }
      return;
    }
    // An UNSELECTED face keeps its own shape and gains only the midpoints
    // that now genuinely exist on its own sides — this is the whole
    // T-junction fix, and it is why a neighbor can legally come out a
    // 5-gon or a 6-gon.
    const loop = [];
    let widened = false;
    for (let c = 0; c < f.length; c++) {
      const a = f[c], b = f[(c + 1) % f.length];
      loop.push(a);
      const mid = edgePointOf.get(edgeKey(a, b));
      if (mid !== undefined) { loop.push(mid); widened = true; }
    }
    if (widened) widenedNeighbours++;
    newFaces.push(loop);
  });
  for (const [, , idx] of splitPairs) insertedVertexIndices.push(idx);

  // CREASE REMAP — same rule and same reasoning as insertEdgeLoop's own:
  // a split edge no longer exists in the output topology at all, so its
  // weight would dangle on a dead key while both real halves silently
  // came out smooth. Transfer to BOTH halves. No decay: this is a
  // topology refinement, not a subdivision step, so nothing has been
  // subdivided for a semi-sharp weight to decay against.
  const newCreases = { ...(cage.creases || {}) };
  for (const [a, b, mid] of splitPairs) {
    const oldKey = edgeKey(a, b);
    const weight = newCreases[oldKey];
    if (weight === undefined) continue;
    delete newCreases[oldKey];
    newCreases[edgeKey(a, mid)] = weight;
    newCreases[edgeKey(mid, b)] = weight;
  }

  const result = { vertices: newVertices, faces: newFaces, creases: newCreases };
  // LAST-RESORT MANIFOLD GATE, the same one subdconvert.mjs already runs
  // on its own output: no face may repeat a vertex, and no edge may be
  // shared by more than two faces. The construction above should make
  // both impossible; this is what makes "should" checkable rather than
  // hoped for, on an operation whose whole point is not corrupting a cage.
  const seen = new Map();
  for (const f of result.faces) {
    if (new Set(f).size !== f.length) throw new Error('subdivideFaces: produced a face with a repeated vertex');
    for (let c = 0; c < f.length; c++) {
      const k = edgeKey(f[c], f[(c + 1) % f.length]);
      seen.set(k, (seen.get(k) || 0) + 1);
    }
  }
  for (const [k, n] of seen) if (n > 2) throw new Error(`subdivideFaces: produced a non-manifold edge (${k} shared by ${n} faces)`);

  return {
    cage: result,
    insertedVertexIndices,
    widenedNeighbours,
    faceCountBefore: cage.faces.length,
    faceCountAfter: result.faces.length,
  };
}

// ─────────────────────────────────────────────────────────────────────────
// N-WAY BRIDGE — a Y, a cross, or any N ≥ 3 open edge runs joined through a
// single hub. Rhino's own SubD Bridge does not do this; two runs is all it
// offers, and asking for three is where the construction genuinely changes
// rather than merely getting bigger.
//
// WHY IT IS A DIFFERENT CONSTRUCTION, not more of the same. Bridging TWO
// runs is a quad grid — topologically a strip between two chains, with one
// correspondence to choose and nothing else to decide. Three or more runs
// meet at a BRANCHING JUNCTION, and there is no canonical quad topology for
// one: something has to occupy the middle. Here that something is an
// explicit HUB — each run is bridged inward to its own shrunken copy, those
// copies are welded corner to corner into one closed ring, and the ring is
// capped by a single n-gon.
//
// The n-gon is not a compromise. Catmull-Clark refines a face of any size,
// and this cage already ships n-gon caps on SuperBCylinder/SuperBCone plus
// an n-valence apex on SuperBCone — so a hub face and its extraordinary
// corner vertices are ordinary geometry here, not a special case to be
// engineered around. That is exactly why this is reachable on a SubD cage
// and not on a tensor-product NURBS patch, where a branching junction
// cannot be one surface at all.
//
// WHAT IS DECIDED RATHER THAN DERIVED, and how:
//  - CYCLIC ORDER. The runs arrive as an unordered set; a hub needs to know
//    which run sits next to which. Taken from each run's own angle about the
//    hub center, measured in the junction's own best-fit plane.
//  - RUN DIRECTION. Each run is traversed so that walking the ring is
//    consistent: a run's far end is whichever of its two endpoints lies
//    nearer the NEXT run around the ring.
//  - WINDING. Not derived at all. Both arm windings and both cap directions
//    are built and tested against `directedEdgeReuseCount`, and the one that
//    produces a consistently-wound 2-manifold is kept — the same "verify the
//    property rather than trust a hand-derived rule" discipline the two-run
//    bridge already uses for its own correspondence search.
//
// Straightness applies at the RIM only, and honestly so: a rung leaves each
// run along that run's own outgoing surface direction, and arrives at the
// hub flat, because there is no surface at the hub to be tangent to yet.
const HUB_INSET = 0.62; // how far in from its own run each inner ring sits, as a fraction of the way to the hub center
function norm3(a) { const L = length(a); return L > 1e-12 ? scale(a, 1 / L) : [0, 0, 0]; }

// THE JUNCTION PLANE AND THE CYCLIC ORDER AROUND IT — shared by both hub
// constructions in this file, one definition, so a fix to either can never
// apply to only one of them.
//
// THE PLANE IS TAKEN FROM THE SINGLE MOST SPREAD PAIR, NOT A SUM. Summing the
// signed cross products over array-ordered pairs is ORDER-DEPENDENT — swap two
// arms and terms change sign — and for a symmetric junction (N arms evenly
// spaced, which is the ordinary case, not an exotic one) the terms cancel to
// zero and a perfectly coplanar, perfectly well-spread set gets refused as
// "collinear". Measured before fixing: 256 of 384 permutation/direction
// variants of a 4-arc square rim were refused that way. For coplanar centroids
// EVERY pair's cross product is parallel to the true normal, so taking the
// LARGEST is both exact and independent of the order the arms arrive in; for a
// non-coplanar set it is an honest best available estimate, exactly as the sum
// was. The same technique kernel/selfintersect.mjs's own bestFitPlane already
// uses against the same underlying failure.
//
// ITS SIGN DOES NOT MATTER to either caller: it only fixes which way round the
// ring is numbered, and each construction absorbs that (the run hub searches
// both windings; the closed-rim hub is provably invariant, its two poles
// simply swapping roles).
//
// THE CYCLIC ORDER IS A RING IN ONE PLANE, and that is a real limitation
// rather than a universal truth. Three arms are always coplanar, so N = 3 is
// safe by construction. A genuinely three-dimensional N >= 4 junction — the
// ±X/±Y/+Z frame corner — has arms that no single plane orders sensibly, and
// two of them can project to nearly the same angle here and be mis-ordered.
// The closed-rim hub below refuses that case by name via its own equator test;
// the run hub does not, and this is where it would have to.
function junctionPlaneOrder(mids, centre, who, noun) {
  let nrm = [0, 0, 0], nrmLen = 0;
  for (let i = 0; i < mids.length; i++) for (let j = i + 1; j < mids.length; j++) {
    const c = cross(sub(mids[i], centre), sub(mids[j], centre));
    const cl = length(c);
    if (cl > nrmLen) { nrm = c; nrmLen = cl; }
  }
  nrm = norm3(nrm);
  if (!length(nrm)) throw new Error(`${who}: the ${noun}s are collinear about their own center, so there is no junction plane to order them around — move one ${noun} off the line the others share`);
  // An in-plane reference direction: the first arm's own offset from the
  // center, with any out-of-plane part removed. WHICH arm is first only
  // rotates every measured angle by a constant, so the cyclic sequence the
  // sort produces is the same one cut at a different place — a rotation of the
  // ring, which changes nothing about the junction it describes.
  const d0 = sub(mids[0], centre);
  const u = norm3(sub(d0, scale(nrm, dot(d0, nrm))));
  if (!length(u)) throw new Error(`${who}: the first ${noun} sits on the junction axis itself, leaving no in-plane direction to measure the others against`);
  const v = cross(nrm, u);
  const order = mids.map((_, i) => i).sort((a, b) => {
    const da = sub(mids[a], centre), db = sub(mids[b], centre);
    return Math.atan2(dot(da, v), dot(da, u)) - Math.atan2(dot(db, v), dot(db, u));
  });
  return { nrm, u, v, order };
}
export function bridgeEdgeRunsHub(cage, runs, segments = 1, straightness = 1, creaseWeight = 0) {
  if (!Array.isArray(runs) || runs.some((r) => !Array.isArray(r))) throw new Error('bridgeEdgeRunsHub: runs must be an array of vertex-index arrays');
  if (runs.length < 3) throw new Error(`bridgeEdgeRunsHub: a hub junction needs at least 3 edge runs (got ${runs.length}) — two runs is an ordinary bridge, use bridgeEdgeRuns`);
  const m = runs[0].length;
  if (m < 2) throw new Error('bridgeEdgeRunsHub: each run needs at least 2 vertices (at least one edge)');
  if (runs.some((r) => r.length !== m)) throw new Error(`bridgeEdgeRunsHub: the runs have different vertex counts (${runs.map((r) => r.length).join(', ')}) — a hub needs matching counts, since every arm meets the same ring (an unequal pairing would need a run resampled, a real separate generalization)`);
  if (!Number.isInteger(segments) || segments < 1) throw new Error('bridgeEdgeRunsHub: segments must be a positive integer');
  if (!(straightness >= 0 && straightness <= 1)) throw new Error('bridgeEdgeRunsHub: straightness must be between 0 and 1');
  for (let i = 0; i < runs.length; i++) for (let j = i + 1; j < runs.length; j++) {
    const shared = runs[i].find((vi) => runs[j].includes(vi));
    if (shared !== undefined) throw new Error(`bridgeEdgeRunsHub: runs ${i} and ${j} share vertex ${shared} — they already touch, with no real gap to bridge across`);
  }
  const topology = buildTopology(cage);
  runs.forEach((run, i) => checkOpenRunChain(cage, topology, run, `run ${i}`, 'bridgeEdgeRunsHub'));

  // The hub's own center and plane — see junctionPlaneOrder below, which both
  // hub constructions share so the two cannot drift apart.
  const mids = runs.map((run) => scale(run.reduce((acc, vi) => add(acc, cage.vertices[vi]), [0, 0, 0]), 1 / run.length));
  const centre = scale(mids.reduce((acc, p) => add(acc, p), [0, 0, 0]), 1 / mids.length);
  const { order } = junctionPlaneOrder(mids, centre, 'bridgeEdgeRunsHub', 'run');

  // Each run is walked so its FAR end is the one nearer the next run around
  // the ring — that is what makes "run i's end meets run i+1's start" true
  // for every i, which is the whole reason the ring closes.
  //
  // THIS RULE IS RIGHT EXCEPT UNDER AN EXACT TIE, and a tie is not exotic: on a
  // symmetric junction (three arcs evenly spaced around one rim) a run's two
  // endpoints are EQUIDISTANT from the next run to the bit, so `<` falls through
  // to "keep" and the direction is decided by the order the caller happened to
  // hand the run in rather than by geometry. Half of those orders produce a ring
  // that cannot be wound consistently — measured at 24 of 48 permutation and
  // direction variants of a 3-arc square rim before this was searched.
  //
  // Under a tie every direction is geometrically equally good, so the honest
  // resolution is to SEARCH the assignment rather than trust a rule the tie
  // defeats — the same "verify the property rather than derive it" discipline
  // the winding below already uses. The derived assignment is tried FIRST and is
  // what every non-degenerate junction uses, so the search costs nothing in the
  // ordinary case.
  const N = order.length;
  const derivedFlips = order.map((ri, k) => {
    const run = runs[ri];
    const nextMid = mids[order[(k + 1) % N]];
    return distSq(cage.vertices[run[0]], nextMid) < distSq(cage.vertices[run[run.length - 1]], nextMid);
  });
  const seqFor = (flips) => order.map((ri, k) => (flips[k] ? [...runs[ri]].reverse() : [...runs[ri]]));
  // The derived assignment first, then every other combination. Flipping ALL of
  // them at once only reverses the ring's own orientation, which the winding
  // search already absorbs, so this converges early in practice. Capped so a
  // pathological arm count cannot turn a refusal into a hang; beyond the cap the
  // derived assignment is the only one tried, exactly as before.
  const MAX_SEARCHED_ARMS = 8;
  const directionCandidates = [derivedFlips];
  if (N <= MAX_SEARCHED_ARMS) {
    for (let bits = 0; bits < (1 << N); bits++) {
      const flips = Array.from({ length: N }, (_, k) => !!((bits >> k) & 1));
      if (flips.every((f, k) => f === derivedFlips[k])) continue;
      directionCandidates.push(flips);
    }
  }
  const attempt = (seq) => buildHubForSeq(cage, topology, seq, m, N, centre, segments, straightness);
  let built = null;
  for (const flips of directionCandidates) {
    built = attempt(seqFor(flips));
    if (built) break;
  }
  if (!built) throw new Error('bridgeEdgeRunsHub: no winding of these runs produces a consistently-wound junction — they may not all bound the same open region, or two of them already meet through existing geometry');
  const outCreases = { ...(cage.creases || {}) };
  for (const run of built.seq) creaseChain(outCreases, run, creaseWeight, false);
  return {
    cage: { vertices: built.newVertices, faces: built.faces, creases: outCreases },
    bridgeFaceIndices: built.added.map((_, i) => built.baseCount + i),
    hubFaceIndex: built.baseCount + built.hubIndexInAdded,
    armCount: N,
    ringLength: built.ringLength,
  };
}

// One direction assignment's worth of hub geometry plus its winding search.
// Returns null if no winding of THIS assignment is consistent, so the caller can
// try another; every attempt allocates its OWN vertex array, so a rejected one
// leaves nothing behind in the cage — the vertex leak this construction has
// already been caught by a count assertion for once.
function buildHubForSeq(cage, topology, seq, m, N, centre, segments, straightness) {
  const newVertices = cage.vertices.map((p) => p.slice());
  const push = (p) => { newVertices.push(p); return newVertices.length - 1; };
  const innerPos = seq.map((run) => run.map((vi) => lerp3(cage.vertices[vi], centre, HUB_INSET)));
  // Consecutive inner runs SHARE their touching endpoint — that single weld
  // per junction is what turns N separate inner chains into one closed ring,
  // and it is why the ring has N(m-1) vertices rather than N*m.
  const corner = [];
  for (let k = 0; k < N; k++) corner.push(push(lerp3(innerPos[k][m - 1], innerPos[(k + 1) % N][0], 0.5)));
  const inner = seq.map((_, k) => {
    const row = [corner[(k - 1 + N) % N]];
    for (let j = 1; j < m - 1; j++) row.push(push(innerPos[k][j]));
    row.push(corner[k]);
    return row;
  });

  // The ring, walked once: each arm contributes its own start corner and its
  // interior vertices, and the next arm's start corner continues it.
  const ring = [];
  for (let k = 0; k < N; k++) for (let j = 0; j < m - 1; j++) ring.push(inner[k][j]);

  // Every vertex this junction needs, built ONCE PER DIRECTION ATTEMPT. Only the
  // WINDING is searched below, and winding is a property of the face lists alone
  // — so building geometry inside THAT search would append a fresh copy of every
  // interior row per rejected trial and leave the discarded ones orphaned in the
  // cage. Caught by a vertex-count test rather than by reading.
  const armRows = seq.map((run, k) => {
    const outer = run.map((vi) => cage.vertices[vi]);
    const innerPts = inner[k].map((vi) => newVertices[vi]);
    const tan = rimTangents(cage, topology, run, innerPts, false, 1);
    const rows = [run];
    for (let r = 1; r < segments; r++) {
      const t = r / segments;
      const row = [];
      for (let j = 0; j < m; j++) {
        // The hub end gets a straight tangent on purpose: there is no surface
        // there for a rung to leave along, so claiming one would be inventing
        // a direction rather than reading one.
        row.push(push(bridgeSpanPoint(outer[j], innerPts[j], tan[j], sub(innerPts[j], outer[j]), t, straightness)));
      }
      rows.push(row);
    }
    rows.push(inner[k]);
    return rows;
  });
  const build = (armFlip, capReversed) => {
    const faces = cage.faces.map((f) => [...f]);
    const added = [];
    for (let k = 0; k < N; k++) {
      const rows = armRows[k];
      for (let r = 0; r + 1 < rows.length; r++) {
        const near = rows[r], far = rows[r + 1];
        for (let j = 0; j + 1 < m; j++) {
          added.push(armFlip ? [near[j + 1], near[j], far[j], far[j + 1]] : [near[j], near[j + 1], far[j + 1], far[j]]);
        }
      }
    }
    added.push(capReversed ? [...ring].reverse() : [...ring]);
    return { faces: [...faces, ...added], added, hubIndexInAdded: added.length - 1, baseCount: faces.length };
  };
  let chosen = null;
  for (const armFlip of [false, true]) {
    for (const capReversed of [false, true]) {
      const trial = build(armFlip, capReversed);
      if (directedEdgeReuseCount(trial.faces) === 0) { chosen = trial; break; }
    }
    if (chosen) break;
  }
  if (!chosen) return null;
  return { ...chosen, newVertices, seq, ringLength: ring.length };
}

// ─────────────────────────────────────────────────────────────────────────
// N-WAY CLOSED-RIM HUB — N >= 3 CLOSED rims (tube ends) welded into one
// junction. Build-order step 2: a pipe network's junction is N
// tube rims meeting, and nothing in this file could do it.
//
// WHY NEITHER EXISTING BRIDGE FITS, stated precisely because the difference is
// the whole reason this exists. bridgeBoundaryLoops joins two closed loops
// into a tunnel and stops at two. bridgeEdgeRunsHub joins N OPEN runs, and an
// open run is a chain with two ENDS — which is exactly what its hub relies on:
// consecutive runs weld end to end into one closed ring, which a single n-gon
// then caps. A closed rim has no ends to weld, so that construction has
// nothing to build on. Handing it one is not a refusal either — it builds
// cleanly, winds consistently, and silently leaves each rim's own closing edge
// unattached (see checkOpenRunChain above, which now refuses it by name).
//
// THE CONSTRUCTION — a sphere with N holes, which is exactly what a junction
// surface is. N boundary circles on a genus-0 surface gives Euler
// characteristic 2 - N, and the smallest honest realization of it here adds
// exactly TWO new vertices, the junction's own poles:
//
//   - Each rim is split at two vertices into an arc on the +normal side of the
//     junction plane and an arc on the -normal side.
//   - Every +side arc, closed through the top pole, is one face; every -side
//     arc through the bottom pole is another. That is 2N faces.
//   - Between neighboring arms, one triangle per pole closes the gap: the
//     "crotch" edge joining two adjacent rims is shared by exactly those two
//     triangles. That is 2N more.
//
//   V = Nm + 2, E = Nm + 5N, F = 4N, so V - E + F = 2 - N exactly, for every
//   N and every m. Checked as a real Euler count in this module's own tests,
//   not asserted here.
//
// NO SEGMENTS DIAL, and that is a scope cut rather than an oversight: the hub
// faces attach DIRECTLY to the rims the caller supplies, so the only new
// vertices are the two poles. A pipe network is expected to have already
// stopped its tubes short of the junction — that inset is the caller's job
// (HUB_INSET is the constant to generalize for it), and doing it
// here would mean this function moving vertices that already exist, which a
// bridge may never do.
//
// WHAT IS DERIVED AND WHAT IS VERIFIED:
//
//  - FILL ORDER is taken from the CAGE, never from the caller. A closed rim is
//    a cycle: the given array can start anywhere and run either way. The
//    canonical direction is boundaryLoopFromSeed's own — the order a single
//    face filling this rim would use, which traverses each rim edge opposite
//    to that edge's one existing face. Normalizing to it here absorbs every
//    direction flip and every rotation of the input, once, instead of leaving
//    a search to find them later.
//
//  - THE SPLIT is chosen by maximizing how cleanly it separates the rim's own
//    +normal side from its -normal side, over every candidate pair of split
//    vertices. Exact ties are ordinary on a symmetric rim (a facets-8 ring
//    whose vertices straddle the equator has a three-way tie), so a tie is
//    broken toward the most BALANCED split — which on that same fixture is
//    also the geometrically right one, the two vertices nearest the equator.
//
//  - WHICH NEIGHBOR EACH ARM CONNECTS TO is derived and then VERIFIED, not
//    trusted. The derivation: a rim's fill order runs counter-clockwise about
//    its own outward direction d, and writing the rim in the orthonormal basis
//    e1 = normalize(n - (n.d)d), e2 = d x e1 gives dot(v - centre, n) =
//    rho * cos(phi) with e2 contributing nothing, so the +n arc runs from
//    phi = -90 to +90 and its START sits at -rho*e2 = rho*(n x d)/|..| — which
//    is exactly the direction of the NEXT arm around n. That is a proof, not a
//    guess, but a winding check cannot discriminate a wrong answer here (both
//    pairings close a valid ring), so the choice is settled by measurement
//    instead: total crotch-edge length under each pairing, the shorter wins.
//    The derivation is why the first one essentially always does.
//
//  - WINDING is not searched at all, and does not need to be: every hub face
//    walks its rim arc in fill order, which is opposite to the arm face that
//    already owns that edge. Consistency follows by construction, and is then
//    checked against directedEdgeReuseCount as a real gate rather than a
//    belief.
//
// THE REAL LIMITATION, named rather than discovered later: this is a junction
// ordered around ONE plane. An arm pointing along the junction normal itself
// has no equator to divide between the two poles — its whole rim lies flat in
// the junction plane — and is refused by name. That rules out a genuinely
// three-dimensional hub such as +X/-X/+Y/-Y/+Z, which needs a different
// construction, not a bigger version of this one.
export function bridgeClosedRimsHub(cage, rims, opts = {}) {
  const creaseWeight = opts.creaseWeight ?? 0;
  if (!Array.isArray(rims) || rims.some((r) => !Array.isArray(r))) throw new Error('bridgeClosedRimsHub: rims must be an array of vertex-index arrays');
  const N = rims.length;
  if (N < 3) throw new Error(`bridgeClosedRimsHub: a hub junction needs at least 3 closed rims (got ${N}) — two rims is an ordinary tunnel, use bridgeBoundaryLoops`);
  const m = rims[0].length;
  if (m < 3) throw new Error(`bridgeClosedRimsHub: each rim needs at least 3 vertices (got ${m}) — fewer cannot close into a real ring`);
  if (rims.some((r) => r.length !== m)) throw new Error(`bridgeClosedRimsHub: the rims have different vertex counts (${rims.map((r) => r.length).join(', ')}) — every arm meets the same hub, and matching one rim to another by resampling would move vertices that already exist, which a bridge may never do`);
  for (let i = 0; i < N; i++) {
    if (new Set(rims[i]).size !== m) throw new Error(`bridgeClosedRimsHub: rim ${i} repeats a vertex — a closed rim is listed once around, without writing its first vertex again at the end`);
  }
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const shared = rims[i].find((vi) => rims[j].includes(vi));
    if (shared !== undefined) throw new Error(`bridgeClosedRimsHub: rims ${i} and ${j} share vertex ${shared} — they already touch, with no real gap to bridge across`);
  }
  if (!(creaseWeight >= 0)) throw new Error('bridgeClosedRimsHub: creaseWeight must be zero or positive');

  const topology = buildTopology(cage);
  // EVERY edge of the ring, the CLOSING one included. That closing edge is the
  // entire difference between a rim and a run, so it is checked first-class
  // rather than left to fall out of a loop that stops one short.
  rims.forEach((rim, i) => {
    for (let k = 0; k < m; k++) {
      const a = rim[k], b = rim[(k + 1) % m];
      const key = edgeKey(a, b);
      const edge = topology.edgeMap.get(key);
      if (!edge) throw new Error(`bridgeClosedRimsHub: rim ${i} is not a closed ring — vertices ${a} and ${b} are not joined by a real cage edge${k === m - 1 ? ' (this is the rim\'s own closing edge; an OPEN run belongs in bridgeEdgeRunsHub)' : ''}`);
      if (edge.faces.length !== 1) throw new Error(`bridgeClosedRimsHub: rim ${i}'s edge "${key}" is not a naked (open) edge — it already has ${edge.faces.length} faces, so there is no opening there to bridge from`);
    }
  });

  // Fill order, from the cage. buildTopology records an edge as (v0 -> v1) in
  // the direction the owning face visits it, so a rim listed the same way the
  // arm walks it is the REVERSE of fill order.
  const oriented = rims.map((rim) => {
    const edge = topology.edgeMap.get(edgeKey(rim[0], rim[1]));
    return edge.v0 === rim[0] ? [...rim].reverse() : [...rim];
  });

  const mids = oriented.map((rim) => scale(rim.reduce((acc, vi) => add(acc, cage.vertices[vi]), [0, 0, 0]), 1 / m));
  const centre = scale(mids.reduce((acc, p) => add(acc, p), [0, 0, 0]), 1 / N);
  const { nrm, order } = junctionPlaneOrder(mids, centre, 'bridgeClosedRimsHub', 'rim');

  // Per rim: where to cut it in two. `s` is each vertex's own height above the
  // junction plane, measured from that rim's OWN center — using the hub center
  // instead would bias a whole rim to one side and destroy the split.
  const arms = order.map((ri) => {
    const rim = oriented[ri];
    const c = mids[ri];
    const s = rim.map((vi) => dot(sub(cage.vertices[vi], c), nrm));
    const radii = rim.map((vi) => length(sub(cage.vertices[vi], c)));
    const ringScale = radii.reduce((a, r) => a + r, 0) / m;
    if (!(ringScale > 0)) throw new Error(`bridgeClosedRimsHub: rim ${ri} has collapsed to a single point — there is no ring there to bridge`);
    const spread = Math.max(...s) - Math.min(...s);
    if (!(spread > ringScale * 1e-9)) {
      throw new Error(`bridgeClosedRimsHub: rim ${ri} lies flat IN the junction plane — its arm points along the junction's own normal, so the ring has no side above or below to divide between the two hub poles. A junction with an arm perpendicular to the plane the others share needs a different construction, not this one`);
    }
    const pre = [0];
    for (let i = 0; i < m; i++) pre.push(pre[i] + s[i]);
    const total = pre[m];
    // Sum of s over the vertices STRICTLY between p and q, walking forward.
    const fwdInterior = (p, q) => (p < q ? pre[q] - pre[p + 1] : total - pre[p + 1] + pre[q]);
    const steps = (p, q) => (q - p + m) % m;
    const tol = Math.max(1e-12, spread * m * 1e-9);
    let best = null;
    for (let p = 0; p < m; p++) for (let q = 0; q < m; q++) {
      if (p === q) continue;
      const score = fwdInterior(p, q) - fwdInterior(q, p);
      const balance = Math.abs(steps(p, q) - steps(q, p));
      if (!best || score > best.score + tol || (Math.abs(score - best.score) <= tol && balance < best.balance)) {
        best = { p, q, score, balance };
      }
    }
    const arc = (from, to) => {
      const out = [];
      for (let i = from; ; i = (i + 1) % m) { out.push(rim[i]); if (i === to) break; }
      return out;
    };
    return { ri, rim, arcPlus: arc(best.p, best.q), arcMinus: arc(best.q, best.p), P: rim[best.p], Q: rim[best.q] };
  });

  // The poles. Each sits at the center of the vertices it caps — a mean over a
  // set, so no arm's position in the array can influence it. A rim whose +side
  // arc is a single edge contributes no interior vertex, which is legal; only
  // if NO arm contributes does the fallback along the normal apply.
  const ringScaleAll = arms.reduce((acc, a) => acc + a.rim.reduce((r, vi) => r + length(sub(cage.vertices[vi], mids[a.ri])), 0) / m, 0) / N;
  const interiorOf = (list) => list.flatMap((a) => a.slice(1, -1));
  const meanOf = (idxs) => scale(idxs.reduce((acc, vi) => add(acc, cage.vertices[vi]), [0, 0, 0]), 1 / idxs.length);
  const plusInterior = interiorOf(arms.map((a) => a.arcPlus));
  const minusInterior = interiorOf(arms.map((a) => a.arcMinus));
  const polePlus = plusInterior.length ? meanOf(plusInterior) : add(centre, scale(nrm, ringScaleAll));
  const poleMinus = minusInterior.length ? meanOf(minusInterior) : sub(centre, scale(nrm, ringScaleAll));

  // Which neighbor each arm's P vertex reaches across to. The derivation in
  // this function's own header says NEXT; both choices close a valid ring, so a
  // winding check cannot tell them apart, and the shorter total crotch is what
  // actually decides. On a symmetric junction the two totals differ by a wide
  // margin — the near neighbor against the far one — so this is not a tie.
  const crotchTotal = (nb) => arms.reduce((acc, a, k) => acc + Math.sqrt(distSq(cage.vertices[a.P], cage.vertices[arms[(k + nb + N) % N].Q])), 0);
  const nb = crotchTotal(1) <= crotchTotal(-1) ? 1 : -1;

  const newVertices = cage.vertices.map((v) => v.slice());
  const xi = newVertices.push(polePlus) - 1;
  const yi = newVertices.push(poleMinus) - 1;
  const added = [];
  for (const a of arms) {
    added.push([...a.arcPlus, xi]);
    added.push([...a.arcMinus, yi]);
  }
  for (let k = 0; k < N; k++) {
    added.push([arms[k].P, xi, arms[(k + nb + N) % N].Q]);
    added.push([arms[k].Q, yi, arms[(k - nb + N) % N].P]);
  }

  const faces = [...cage.faces.map((f) => [...f]), ...added];
  const reuseBefore = directedEdgeReuseCount(cage.faces);
  if (directedEdgeReuseCount(faces) !== reuseBefore) {
    throw new Error('bridgeClosedRimsHub: the assembled junction is not consistently wound — the rims may not all bound genuinely separate openings of one consistently-oriented cage');
  }

  const outCreases = { ...(cage.creases || {}) };
  for (const a of arms) creaseChain(outCreases, a.rim, creaseWeight, true);

  return {
    cage: { vertices: newVertices, faces, creases: outCreases },
    hubFaceIndices: added.map((_, i) => cage.faces.length + i),
    poleIndices: [xi, yi],
    armCount: N,
    rims: arms.map((a) => a.rim),
  };
}
