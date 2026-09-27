// SuperB cage topology edits: ExtrudeSubD, InsertEdge, Bridge, Stitch,
// Delete Faces, FillSubDHole, Subdivide. Builds on kernel/subdselect.mjs
// and kernel/subd.mjs's buildTopology/edgeKey. Kernel only: pure cage-in,
// cage-out topology surgery, no app-layer object, UI or undo.
//
// Extrude, InsertEdge and Bridge grow the cage (new vertices/edges/faces);
// Delete Faces shrinks it (removes faces, prunes any orphaned vertex);
// Stitch welds two separate boundary vertex chains into one (the vertex
// count shrinks, no faces are removed — the two chains' edges become shared
// interior edges); FillSubDHole adds one new face and no vertices. None of
// these mutate the input cage — every function returns a new cage object,
// as subdivideCatmullClark does.

function distSq(a, b) { const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2]; return dx * dx + dy * dy + dz * dz; }

import { add, sub, scale, length, cross, dot } from './vec3.mjs';
import { edgeKey, buildTopology, subdivideCatmullClark } from './subd.mjs';
import { oppositeEdgeInFace } from './subdselect.mjs';
import { cubicHermiteSegment } from './interpolate.mjs';

function lerp3(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }

// Bridge crease — Rhino's fourth Bridge Option. It creases the rim, not the
// rungs: the edges where the new wall meets the existing surface, so the
// bridge reads as a distinct tube joining two forms rather than blending
// smoothly into both.
//
// Takes a weight rather than a boolean so the kernel holds no app-side "how
// hard is a hard crease" constant, and a partial crease uses the same 0..N
// semi-sharp range as the rest of the crease machinery. 0 writes no key.
function creaseChain(creases, seq, weight, closed) {
  if (!(weight > 0)) return;
  const n = seq.length;
  const last = closed ? n : n - 1;
  for (let i = 0; i < last; i++) creases[edgeKey(seq[i], seq[(i + 1) % n])] = weight;
}

// Bridge straightness
// Rhino's Bridge Options carries a Straightness percentage alongside
// Segments: at 100% a rung's interior rows are a straight lerp between the
// two rims; at 0% they follow a cubic that leaves each rim along that rim's
// own outgoing surface direction, so the tunnel meets the cage tangentially
// instead of at a crease. Anything between is a blend of the two.
//
// The cubic is `cubicHermiteSegment` (kernel/interpolate.mjs), evaluated at
// a parameter. It returns a degree-3 Bezier whose single span is [0,1], so
// evaluating it is the cubic Bernstein basis, no knot machinery.
//
// Two identities, both asserted in the test file: straightness 1 returns
// the lerp bit-identically (an early return), and at segments=1 there are
// no interior rows, so straightness cannot change the result — which is
// why Rhino defaults Segments to 2.
function bridgeSpanPoint(pA, pB, mA, mB, t, straightness) {
  const straight = lerp3(pA, pB, t);
  if (straightness >= 1 || !mA || !mB) return straight;
  const b = cubicHermiteSegment(pA, pB, mA, mB).ctrlPts;
  const u = 1 - t;
  const w0 = u * u * u, w1 = 3 * u * u * t, w2 = 3 * u * t * t, w3 = t * t * t;
  const herm = [0, 1, 2].map((k) => b[0][k] * w0 + b[1][k] * w1 + b[2][k] * w2 + b[3][k] * w3);
  return lerp3(straight, herm, 1 - straightness);
}

// The outgoing surface direction at one rim vertex, read off the owner
// face's loop rather than a face normal or a guessed axis. A rim vertex
// sits on one naked edge pair of its chain; its owner face gives it one
// neighbor along the rim (the partner passed in) and one inward, into the
// surface the rim bounds. `vertex - inward` points away from that surface,
// along the direction the surface travels as it reaches the rim — the
// tangent a tangent-continuous bridge leaves on.
//
// Returns null for a degenerate neighborhood (a zero-length inward edge).
// Callers fall back to the straight chord there, so a degenerate cage
// produces a straight rung rather than a NaN.
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

// Per-vertex outgoing tangents for a whole rim, scaled to that rung's
// chord length (the standard Hermite magnitude choice — a tangent scaled to
// the span it crosses gives a blend proportional to the gap, so a narrow
// bridge bulges less than a wide one for the same straightness value).
// `closed` distinguishes a Bridge rim (a closed loop, every vertex has a
// rim partner on both sides) from an open edge run (the last vertex's only
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

// A consistently wound 2-manifold never traverses the same directed edge
// twice. Used here to decide an orientation rather than verify one after
// the fact: a candidate rung set that would traverse any directed edge a
// second time is not the correct winding.
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

// Face normal — Newell's method (robust for a non-planar or bowed n-gon, not
// just a 3-point cross product), normalized. Returns [0,0,0] for a
// degenerate (zero-area) face; callers decide how to react —
// computeAverageNormal below throws for a selection whose normals sum to zero.
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

// Average normal of a set of selected faces — the default ExtrudeSubD
// direction. A plain per-face-normal average, not area-weighted. Throws if
// the normals cancel out (e.g. two opposite faces of a box selected
// together) rather than returning a zero vector; the caller then needs an
// explicit picked direction.
export function computeAverageNormal(cage, faceIndices) {
  if (!faceIndices || !faceIndices.length) throw new Error('computeAverageNormal: faceIndices must be a non-empty array');
  let sum = [0, 0, 0];
  for (const fi of faceIndices) sum = add(sum, computeFaceNormal(cage, fi));
  const len = length(sum);
  if (len < 1e-9) throw new Error('computeAverageNormal: the selected faces\' own normals cancel out (e.g. two directly opposite faces) — pick an explicit direction instead');
  return scale(sum, 1 / len);
}

// ExtrudeSubD — extrude the selected face(s) along `direction` (any nonzero
// vector; normalized here) by `distance` (signed — a negative distance
// extrudes into the surface). This is topology surgery, not a point move:
// boundary edges of the selected region (edges touching exactly one
// selected face — whether the other side is a non-selected face or a cage
// boundary) gain new side faces; a vertex touching any such edge is
// duplicated (the old copy stays for the non-selected geometry; a new copy
// at `position + direction*distance` goes to the selected faces); a vertex
// used only by selected faces is translated in place, since nothing
// outside the selection references it. This is the standard "extrude
// region" algorithm, applied to an arbitrary (not necessarily singly
// connected) face set: the boundary/interior classification is per edge.
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

  // A boundary-region edge touches exactly one selected face — shared with
  // a non-selected face or on the cage boundary, both need a new side face
  // and duplicated endpoints. An interior-region edge (two selected faces)
  // needs neither: the two faces move together and the edge follows its
  // endpoints.
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

  // Degenerate direction: if the extrude direction runs exactly parallel
  // to a boundary-region edge (e.g. extruding a face a second time
  // sideways), that edge's new side face is four collinear points with zero
  // area. Every number involved is finite; only the cross product shows it.
  // The whole extrude is refused, since skipping the one side face would
  // leave a hole in the boundary.
  for (const [key, edge] of topology.edgeMap) {
    if (!boundaryEdgeKeys.has(key)) continue;
    const edgeDir = sub(cage.vertices[edge.v1], cage.vertices[edge.v0]);
    const edgeLen = length(edgeDir);
    if (edgeLen < 1e-12) continue; // a zero-length edge is a defect of the input cage, not this guard's to diagnose
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
  // Interior vertices — used by >=1 selected face, touching no
  // boundary-region edge — translate in place (same index, new position).
  const interiorSet = new Set();
  for (const fi of selectedSet) for (const vi of cage.faces[fi]) if (!boundaryVertexSet.has(vi)) interiorSet.add(vi);
  for (const vi of interiorSet) newVertices[vi] = add(cage.vertices[vi], offset);

  const newFaces = cage.faces.map((f) => f.slice());
  for (const fi of selectedSet) {
    newFaces[fi] = newFaces[fi].map((vi) => (boundaryVertexSet.has(vi) ? newIndexForOld.get(vi) : vi));
  }
  // One new side face per boundary-region edge, wound using the original
  // (pre-remap) vertex order in the selected face that owns that edge — a
  // boundary-region edge is owned by exactly one selected face, so this loop
  // visits each once.
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

  // Crease remap. A boundary-region (selCount===1) edge's crease survives:
  // its new side face reuses the edge's original vertex pair as its near
  // edge, so the old key is still a valid edge. An interior-region
  // (selCount===2) edge whose endpoints are both duplicated moves onto the
  // new vertex copies, so its key in the output is the remapped one;
  // without the remap the old key would dangle and the moved edge would
  // lose its weight. `mapVertex` mirrors the remap `newFaces` applies above.
  const mapVertex = (vi) => (boundaryVertexSet.has(vi) ? newIndexForOld.get(vi) : vi);
  const newCreases = { ...(cage.creases || {}) };
  for (const [key, weight] of Object.entries(cage.creases || {})) {
    const edge = topology.edgeMap.get(key);
    if (!edge) continue; // a stale key in the input — nothing to remap
    const selCount = edge.faces.filter((f) => selectedSet.has(f)).length;
    if (selCount !== 2) continue; // boundary/exterior edges keep their key
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

// ExtrudeSubD on edges — extrude selected naked (boundary) edges into a new
// strip of faces. Extruding a face moves that face and walls in the gap
// behind it (the surface keeps its boundary); extruding an edge leaves
// every existing vertex where it is and grows new surface outward from the
// open edge — pulling a plane or a hole's rim into a longer sheet.
//
// Naked edges only. An interior edge (two faces) has no free side to grow
// into; extruding one means tearing the surface open along it and deciding
// which side each existing face follows, which is a different operation.
// It is refused by name.
//
// A chain of edges extrudes as one strip, not as detached quads: an
// endpoint shared by two selected edges is duplicated once and reused, so
// the new faces share their rung and the result stays one manifold surface
// — a whole boundary loop extrudes into a collar.
//
// `direction` may be null, meaning each edge grows the way its surface
// points — perpendicular to the edge, in its owner face's plane, away from
// that face. This is the default a whole boundary loop needs: a single
// shared direction would push one side of the rim out and the opposite side
// through the sheet, and averaging the loop's outward directions cancels to
// nothing. A vertex shared by two selected edges moves by the average of
// their two outward directions, which keeps the collar's corners closed.
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
  // a selected edge collapses that edge's new face to four collinear
  // points. Refused entirely rather than per edge, so the result never
  // has a hole where one bad quad was skipped.
  if (offset) {
    for (const key of keys) {
      const edge = topology.edgeMap.get(key);
      const edgeDir = sub(cage.vertices[edge.v1], cage.vertices[edge.v0]);
      const edgeLen = length(edgeDir);
      if (edgeLen < 1e-12) continue; // a zero-length edge is a defect of the input cage
      const cr = length(cross(edgeDir, offset));
      if (cr < 1e-9 * edgeLen * length(offset)) {
        throw new Error(`extrudeEdges: the extrude direction runs exactly parallel to edge "${key}" — its new face would collapse to four collinear points (zero area); pick a direction with some component across the edge`);
      }
    }
  }
  // Per-edge outward (the `direction == null` default) — one vector per
  // selected edge, then per vertex by averaging the selected edges that
  // touch it, so a shared corner gets one offset and one copy.
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

  // Winding — the new face traverses the shared edge opposite to the way its
  // owner face does, which keeps the two consistently oriented (as
  // extrudeFaces' side faces do). The direction is read off the owner's
  // vertex loop, not the key's sorted order.
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

// Insert Point — split one edge by putting a new vertex on it. Unlike
// InsertEdge, which threads a new loop across a strip of faces, this touches
// only the picked edge. Every face using that edge gains one vertex in its
// loop, so a quad becomes a 5-gon, which Catmull-Clark accepts (the cylinder
// and cone cage caps are n-gons too).
//
// No T-junction: the new vertex goes into the loop of every face that used
// the edge, as per-face subdivide does when it widens a neighbor. Skipping
// the second face would leave the new vertex used by one face only, a cage
// that renders and subdivides into a crack.
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

  // Crease — the split edge is gone, so a weight left on its key would
  // dangle while the two halves went smooth. Both halves inherit it, which
  // keeps a hard edge hard through the edit.
  const creases = { ...(cage.creases || {}) };
  if (creases[edgeKeyStr] !== undefined) {
    const w = creases[edgeKeyStr];
    delete creases[edgeKeyStr];
    creases[edgeKey(edge.v0, newIndex)] = w;
    creases[edgeKey(newIndex, edge.v1)] = w;
  }
  return { cage: { vertices, faces, creases }, newVertexIndex: newIndex, widenedFaceIndices: [...touched] };
}

// Weld Vertices — collapse a selected set of cage vertices into one. The
// same vertex-remap discipline as stitchEdgeRuns rather than a second,
// subtly different merge.
//
// Where the weld lands is the caller's choice, in Stitch's vocabulary:
// 'average' (the centroid of the selection — the default, and the only one
// that treats every picked vertex equally), or 'first' (hold the
// first-picked vertex still and pull the rest onto it).
//
// A weld that would make the cage non-manifold (an edge with 3+ faces) is
// refused: such a cage renders and counts right, then subdivides into
// garbage. Checked on the result, not predicted from the input.
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
    // A face can also fold onto itself non-consecutively (welding two
    // opposite corners of a quad) — a bowtie, not a triangle — and it is
    // dropped rather than kept as a repeated-vertex face.
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
  // Non-manifold check on the result.
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

// Slide Edge — move a selected edge (or a whole loop of them) along the
// surface without changing topology: every vertex of the selection travels
// down one of its incident rail edges — the edges running across the loop
// rather than along it — by a fraction t. Pure geometry: no face is added,
// removed or rewritten, only positions move.
//
// Each loop vertex has two rails, one on each side, and choosing per vertex
// independently gives a zig-zag rather than a slide. The choice is
// propagated instead: the first vertex picks a rail, and every later vertex
// takes the rail pointing most nearly the same way as the previous choice.
// A vertex with no incident edge outside the selection stays put.
//
// At |t| = 1 a sliding vertex lands on the neighbor it slides toward,
// producing a zero-length edge and a degenerate face, so |t| above 0.95 is
// refused rather than clamped.
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
  // One pass. The first vertex with a choice seeds the direction — rail 0
  // for a positive t, the rail most opposite to it for a negative one — and
  // every vertex after it takes the rail that agrees best with the previous
  // choice.
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

// Offset a cage — the SubD counterpart of the NURBS Offset. A NURBS offset
// moves a tensor surface's control net along Greville normals; here every
// cage vertex moves along its vertex normal — the normalized sum of the
// normals of the faces meeting there, the standard vertex-normal estimate.
//
// Offsetting a control cage does not produce an exact offset of its limit
// surface (no polynomial surface has one in general). It produces a cage
// whose limit surface runs roughly `distance` away from the original's.
//
// Known limitation: self-intersection is not guarded. A large enough
// offset on a concave region folds the cage through itself; the NURBS
// side's safeOffsetMagnitude search has no cage counterpart.
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

// Thicken — give an open cage thickness: the original, an offset copy wound
// the other way, and a rim of quads closing the gap between the two
// boundaries, so a sheet becomes a slab.
//
// Winding: the offset copy is reversed (its faces read backwards) so its
// outward side faces away from the original — otherwise the two sheets
// face the same way and the slab is inside-out on one side. The rim quads
// are wound from the original's boundary direction, so every shared edge
// is traversed once each way.
//
// A closed cage is refused. The rim is built from naked edges — one wall
// quad per edge with a single owner face — and a closed cage has none, so
// the result would be the original plus a reversed offset copy with nothing
// joining them: two disconnected shells, one nested inside the other. That
// is a valid B-rep hollow solid but not a single connected SubD cage, and
// every later cage command would see two components. The message names the
// way through: delete a face first, and the wall is built around that
// opening.
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
  // Rim — one quad per naked edge, wound against the owner face's
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
  // Creases are copied to both sheets — a hard edge stays hard on the copy.
  const creases = {};
  for (const [key, w] of Object.entries(cage.creases || {})) {
    creases[key] = w;
    const [a, b] = key.split('_').map(Number);
    creases[edgeKey(a + n, b + n)] = w;
  }
  // The shell must face out whichever way it grew. The construction above
  // can come out inside-out (a wall grown outward, distance > 0, on an open
  // box does), so the winding is set from the finished shell's signed
  // volume rather than from the sign of distance.
  let allFaces = [...faces, ...rimFaces];
  let vol = 0;
  for (const f of allFaces) for (let i = 1; i + 1 < f.length; i++) {
    const a = vertices[f[0]], b = vertices[f[i]], c = vertices[f[i + 1]];
    vol += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  }
  if (vol < 0) allFaces = allFaces.map((f) => f.slice().reverse());
  return {
    cage: { vertices, faces: allFaces, creases },
    offsetFaceIndices: cage.faces.map((_, i) => cage.faces.length + i),
    rimFaceIndices: rimFaces.map((_, i) => 2 * cage.faces.length + i),
  };
}

// InsertEdge — insert a new edge loop parallel to and offset from a picked
// existing loop (Rhino's InsertEdge). `seedKey` is any cage edge; `side`
// (0 or 1) picks which of its (up to 2) adjacent faces the new loop is
// inserted toward; the neighbor loop on the other side is reached by
// seeding from an edge there. `t` in (0,1) exclusive is the position
// fraction: t->0 sits near the picked loop, t->1 near its neighbor on the
// chosen side. 0 or 1 would coincide with an existing loop and produce
// zero-area faces.
//
// Algorithm (the test file carries worked numbers for a cylinder side
// band): a directed BFS walks the quad strip between the picked loop and
// its chosen neighbor loop. Each strip face is entered knowing its "near"
// edge (an edge of the picked loop, or — deeper into the strip — the
// corresponding edge of the nearest loop); within that face,
// `oppositeEdgeInFace` gives the "far" edge (the corresponding edge of the
// neighbor loop), and the face's other two edges are its two rungs
// (connecting a near vertex to its corresponding far vertex). Crossing a
// rung into the next face requires knowing, at that vertex, which of the
// new face's two other incident edges continues the same loop direction
// (not the rung, and not the far-side continuation) — read from the new
// face's vertex-index neighbors, since "opposite" needs a known near edge.
// An interior rung is shared by two strip faces; its inserted vertex is
// computed once and reused, which joins the per-face new edges into one
// continuous loop.
//
// Local mode (`opts.local`) — the same cut, stopped after the seed face:
// whole-strip propagation tightens a whole band, local refines one face.
// It is a scope restriction of the walk above — same `t`, `side`,
// near/far orientation and crease rules — that never enqueues the
// neighbors across its rungs, so it is an option here rather than a
// separate function that would restate the orientation logic.
// `recomputeInsertedLoopPositions` works on the result, since it is driven
// by the returned rung pairs.
//
// What local leaves behind: the seed face becomes two quads separated by
// the new edge. That edge's endpoints sit inside the seed face's two rungs,
// the only edges this operation splits — the near edge (the seed) and the
// far edge are reused verbatim by the two replacement quads. Each rung's
// other face must therefore receive the same new vertex on its boundary,
// growing from n sides to n+1: a vertex present on one side of an edge and
// absent on the other is a crack. The resulting n-gon is a legal
// Catmull-Clark cage face (computeFacePoint is a centroid for any n>=3), so
// the limit surface stays well defined; that n-gon plus the extraordinary
// vertices at the new T-vertices is what a T-spline-style local workflow
// needs.
//
// The seed edge's other face is not split. `side` names which of the seed
// edge's faces is refined, and the seed edge itself is not split, so the
// other face has no missing vertex to repair. Splitting it too would start
// a second, independent parallel cut running away from the shared edge,
// doubling the new faces and the T-junctions; that is one more local insert
// with the other `side`.
//
// N-gons, in either mode. The opposite edge is well defined only for a
// quad, so a strip cannot pass through an n-gon. It ends at one: reached
// mid-strip, a non-quad stops that direction of the walk as a cage boundary
// or a non-manifold rung does, and the loop keeps every face it has
// crossed. The rung shared with the n-gon is still split, and the
// T-junction repair below widens the n-gon from n to n+1 so no crack is
// left. The n-gons a run stopped at come back in
// `stoppedAtNonQuadFaceIndices`, so a partial loop is stated rather than
// inferred.
//
// The seed face is the exception, in both modes: a non-quad there has no
// cut in it and no earlier progress to stop after, so it is refused by a
// message naming the face, its side count, and the adjacent quad to seed
// from. A face carrying a T-junction vertex (an n-gon, n>=5) therefore
// cannot seed the next insert, but it is widened again as the neighbor of
// one, which is the case that recurs when refining locally.
//
// On an all-quad cage a whole-loop insert leaves no T-junctions (the walk
// crosses every face whose rungs it splits); on a cage carrying an n-gon it
// leaves one widened n-gon at each end where the strip stopped.
export function insertEdgeLoop(cage, seedKey, t = 0.5, side = 0, opts = {}) {
  const local = !!(opts && opts.local);
  if (!Number.isFinite(t) || t <= 0 || t >= 1) throw new Error('insertEdgeLoop: t must be a number strictly between 0 and 1');
  if (side !== 0 && side !== 1) throw new Error('insertEdgeLoop: side must be 0 or 1');
  const topology = buildTopology(cage);
  const seedEdge = topology.edgeMap.get(seedKey);
  if (!seedEdge) throw new Error(`insertEdgeLoop: "${seedKey}" is not a real edge of this cage`);
  const face0 = seedEdge.faces[side];
  if (face0 === undefined) throw new Error(`insertEdgeLoop: no face on side ${side} of edge "${seedKey}" (it has ${seedEdge.faces.length} adjacent face${seedEdge.faces.length === 1 ? '' : 's'})`);
  // A non-quad seed face is refused, and the refusal names what it hit. The
  // cut inside a face runs from one of the near edge's two neighboring sides
  // to the other, leaving across the edge opposite the near one — which
  // needs exactly one side on each hand and a single opposite edge, true
  // only of a quad. Met mid-strip that is a place to stop (see the enqueue
  // below); met as the seed there is no earlier progress to stop after.
  if (cage.faces[face0].length !== 4) {
    const otherSide = side === 0 ? 1 : 0;
    const otherFace = seedEdge.faces[otherSide];
    const alt = otherFace === undefined
      ? 'this edge has only one face'
      : `side ${otherSide} of this same edge is face ${otherFace}, with ${cage.faces[otherFace].length} sides`;
    throw new Error(`insertEdgeLoop: the face this insert would start from, face ${face0} on side ${side} of edge "${seedKey}", has ${cage.faces[face0].length} sides, not 4 — a cut leaves a face across the edge opposite the one it entered by, and only a quad has an opposite edge, so there is no cut to make in this face. Seed from an edge of an adjacent quad instead (${alt}); a loop seeded on a quad runs up to this face and stops at it.`);
  }

  const faceInfos = new Map(); // faceIdx -> { na, nb, farNa, farNb }
  const visited = new Set();
  const stoppedAtNonQuad = new Set(); // input face indices the strip declined to enter
  const queue = [{ faceIdx: face0, nearKey: seedKey }];

  while (queue.length) {
    const { faceIdx, nearKey } = queue.shift();
    if (visited.has(faceIdx)) continue;
    visited.add(faceIdx);
    // Every face dequeued here is a quad: the seed was checked above, and the
    // enqueue below only ever enqueues quads.
    const face = cage.faces[faceIdx];
    const nearEdge = topology.edgeMap.get(nearKey);
    const farKey = oppositeEdgeInFace(cage, faceIdx, nearKey);
    if (!farKey) throw new Error(`insertEdgeLoop: face ${faceIdx} has no well-defined opposite edge for "${nearKey}"`);
    // Orient the near pair to this face's traversal, not to the edge's
    // canonical (sorted) order. An edgeKey is order-independent — "3_7"
    // says nothing about which way either of its two faces walks it, and
    // the two walk it in opposite directions because the cage is
    // consistently wound. The replacement quads below are built in
    // traversal order ([na, nb, Nb, Na]), so na/nb taken from the edge
    // record would give a correctly shaped but backwards-wound pair for
    // every face whose traversal runs against the canonical order: a cage
    // with the right counts that renders and is non-orientable. On a box,
    // side 0 happens to follow the canonical order throughout and side 1
    // runs against it at every step.
    const ia0 = face.indexOf(nearEdge.v0);
    if (ia0 === -1) throw new Error(`insertEdgeLoop: face ${faceIdx} does not contain vertex ${nearEdge.v0}`);
    let na, nb, farNa, farNb;
    if (face[(ia0 + 1) % 4] === nearEdge.v1) {
      // this face walks v0 -> v1
      na = nearEdge.v0; nb = nearEdge.v1;
      farNb = face[(ia0 + 2) % 4]; farNa = face[(ia0 + 3) % 4];
    } else if (face[(ia0 + 3) % 4] === nearEdge.v1) {
      // this face walks v1 -> v0: relabel so na is the corner this face
      // reaches first, carrying each one's far partner with it (the rung
      // pairing is unchanged — only which is called "a").
      na = nearEdge.v1; nb = nearEdge.v0;
      farNa = face[(ia0 + 2) % 4]; farNb = face[(ia0 + 1) % 4];
    } else throw new Error(`insertEdgeLoop: face ${faceIdx} does not contain edge "${nearKey}" as one of its own sides`);

    faceInfos.set(faceIdx, { na, nb, farNa, farNb });

    if (local) continue; // local: the cut stops at this face's rungs, so no neighbor is enqueued

    for (const [nearV, farV] of [[na, farNa], [nb, farNb]]) {
      const rKey = edgeKey(nearV, farV);
      const rEdge = topology.edgeMap.get(rKey);
      if (!rEdge) continue; // unreachable for a well-formed quad; skipped rather than thrown
      const others = rEdge.faces.filter((f) => f !== faceIdx);
      if (others.length !== 1) continue; // a cage boundary (0 left) or non-manifold (2+ left) — stop expanding this direction
      const nextFaceIdx = others[0];
      if (visited.has(nextFaceIdx)) continue;
      const nextFace = cage.faces[nextFaceIdx];
      // A non-quad ends the strip here, not the operation — the same answer
      // the two lines above give a cage boundary (no face beyond the rung)
      // and a non-manifold rung (two or more): stop this direction and keep
      // what the walk has.
      //
      // kernel/subdselect.mjs reads it the same way: edgeRingFromSeed and
      // faceLoopFromSeed — the walks that show which strip an insert will
      // act on — return a partial ring at an n-gon. A pick that highlights a
      // strip has to be a pick the insert can run.
      //
      // Crossing has no rule to follow. The two new vertices land inside the
      // near edge's two neighboring sides; an n-gon has (n-2)/2 sides on each
      // hand rather than one, and nothing chooses among them. For odd n there
      // is no opposite edge at all. The rung shared with the n-gon is still
      // split (it is a rung of a face being split), and the T-junction repair
      // below splices that new vertex into the n-gon's boundary, widening it
      // from n to n+1, the same mechanism local mode uses.
      if (nextFace.length !== 4) { stoppedAtNonQuad.add(nextFaceIdx); continue; }
      const idx2 = nextFace.indexOf(nearV);
      if (idx2 === -1) throw new Error(`insertEdgeLoop: rung "${rKey}" traversal error at face ${nextFaceIdx}`);
      const n2 = nextFace.length;
      const neighbors2 = [nextFace[(idx2 + 1) % n2], nextFace[(idx2 - 1 + n2) % n2]];
      const other2 = neighbors2.find((x) => x !== farV);
      if (other2 === undefined) throw new Error(`insertEdgeLoop: rung "${rKey}" traversal error at face ${nextFaceIdx} (degenerate face)`);
      queue.push({ faceIdx: nextFaceIdx, nearKey: edgeKey(nearV, other2) });
    }
  }

  // Materialize one new vertex per distinct rung touched by the strip (a
  // rung shared by 2 strip faces is discovered twice but built once — both
  // discoveries look up the same `newVertexOf` map).
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
  // T-junction repair — a rung this insert split may be shared with a face
  // that is not itself being split, whose boundary still walks the rung as
  // a single side. Left alone it would omit a vertex its neighbor has, which
  // is a crack: the two faces would agree on the endpoints and disagree
  // about everything between them, and subdivideCatmullClark would refine
  // the mismatch rather than close it. So the new vertex is spliced into
  // that face's loop, in traversal order, between the rung's two endpoints —
  // turning an n-gon into an (n+1)-gon, still legal input to kernel/subd.mjs.
  //
  // A face is rebuilt only if one of its sides is a split rung; otherwise it
  // is copied through unchanged. This repair is reached by a cut that stops
  // — local mode by construction, and a whole-loop insert wherever its
  // strip stopped at an n-gon or a cage boundary. On an all-quad closed band
  // every face across every split rung was itself enqueued, so nothing is
  // rebuilt.
  const splitRungMid = new Map(); // rung edgeKey -> the new vertex sitting in its interior
  rungPairs.forEach(([nearIdx, farIdx], i) => splitRungMid.set(edgeKey(nearIdx, farIdx), insertedVertexIndices[i]));

  const keptFaces = [];
  const tJunctionFaceIndices = [];
  const stoppedAtNonQuadFaceIndices = [];
  cage.faces.forEach((face, fi) => {
    if (facesToRemove.has(fi)) return;
    // Recorded here rather than during the walk because the walk knows input
    // indices and every other index this function returns is an output one.
    // keptFaces leads the output array, so its current length is that index,
    // whichever of the two branches below does the push.
    if (stoppedAtNonQuad.has(fi)) stoppedAtNonQuadFaceIndices.push(keptFaces.length);
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
    tJunctionFaceIndices.push(keptFaces.length); // index into the output faces array, which starts with keptFaces
    keptFaces.push(widened);
  });

  // Crease remap, as in extrudeFaces: the near/far edges of every strip face
  // (na-nb, farNa-farNb — the loop-direction edges) are reused verbatim by
  // the replacement faces above, so a crease on either survives unchanged. A
  // rung (na-farNa or nb-farNb — the edges the loop crosses) is split into
  // two new edges (near-to-inserted, inserted-to-far) and is gone from the
  // output, so its weight moves onto both halves; otherwise the old key
  // would dangle and both halves would go smooth. `rungPairs` and
  // `insertedVertexIndices` hold one entry per rung crossed.
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
    // untouched but widened neighbors that carry a T-junction vertex.
    splitFaceIndices: replacementFaces.map((_, i) => keptFaces.length + i),
    tJunctionFaceIndices,
    // The n-gons the strip stopped at, so a partial loop is reported rather
    // than inferred from a face count. Empty means the loop ran to its
    // natural ends (a closed band, or a cage boundary at each end); an
    // all-quad cage always returns empty.
    stoppedAtNonQuadFaceIndices,
  };
}

// The local edge insert, named. Same arguments as insertEdgeLoop's first
// four and the same return shape; it holds no logic of its own, so the two
// cannot disagree.
export function insertEdgeLocal(cage, seedKey, t = 0.5, side = 0) {
  return insertEdgeLoop(cage, seedKey, t, side, { local: true });
}

// Re-positions an inserted loop's vertices to a new `t`, given the
// (nearIdx, farIdx) rung pairs insertEdgeLoop returned — no topology change
// (same vertex/face/edge counts before and after). The topology surgery
// above runs once, at insertion; a position slider dragged afterwards calls
// only this.
export function recomputeInsertedLoopPositions(cage, insertedVertexIndices, rungPairs, t) {
  if (!Number.isFinite(t) || t <= 0 || t >= 1) throw new Error('recomputeInsertedLoopPositions: t must be a number strictly between 0 and 1');
  const vertices = cage.vertices.map((v) => v.slice());
  insertedVertexIndices.forEach((vi, i) => {
    const [nearIdx, farIdx] = rungPairs[i];
    vertices[vi] = lerp3(cage.vertices[nearIdx], cage.vertices[farIdx], t);
  });
  return { vertices, faces: cage.faces, creases: cage.creases };
}

// Delete Faces — removes the given faces from the cage (making holes) and
// prunes any vertex left with no remaining face (an orphan would break
// subdivideCatmullClark's vertex-point rule, which assumes every vertex has
// at least one incident face), compacting the surviving vertex indices. A
// crease is kept only when the edge it names exists in the output topology
// (both endpoints survive and some remaining face uses that exact edge);
// two vertices can each survive, used by different faces, without the edge
// between them surviving. Returns `vertexRemap` (old index -> new index,
// for surviving vertices only) so a caller (Bridge, below) holding
// original indices can translate them.
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
    if (!newTopology.edgeMap.has(newKey)) continue; // both endpoints survive, but this edge does not
    newCreases[newKey] = weight;
  }

  return {
    cage: { vertices: newVertices, faces: newFaces, creases: newCreases },
    vertexRemap,
    removedVertexCount: cage.vertices.length - newVertices.length,
  };
}

// The single boundary edge at `vIdx` other than `excludeKey` — used by
// boundaryLoopFromSeed's walk below. Returns null when there is not exactly
// one (a dead end, or a non-manifold boundary vertex where 2+ boundary
// edges meet), which stops the walk.
function otherBoundaryEdgeAt(topology, vIdx, excludeKey) {
  const candidates = topology.vertexEdges[vIdx].filter((e) => e.faces.length === 1 && edgeKey(e.v0, e.v1) !== excludeKey);
  return candidates.length === 1 ? candidates[0] : null;
}

// Walks a cage boundary (edges used by exactly one face) all the way
// around, starting from any one boundary edge, returning the ordered
// vertex-index loop. Unlike kernel/subdselect.mjs's edgeLoopFromSeed (which
// needs a regular valence-4 vertex at every step), a hole's rim vertices
// can have any valence, so the only constraint is exactly one other
// boundary edge at each vertex (otherBoundaryEdgeAt above).
//
// Orientation: `seedKey`'s one adjacent face traverses the edge as
// (v0 -> v1) (buildTopology records an edge in the direction its first
// owning face visits it). For the cage to stay consistently oriented once
// the hole is filled, the fill face must traverse this edge as (v1 -> v0).
// Starting at v0 and stepping away from v1 (via v0's other boundary edge)
// gives [v0, x2, ..., xk, v1]; read as a face's cyclic vertex order, its
// wrap-around edge is (v1 -> v0), with no reversal step. The test file
// checks that a face built from this loop shares every edge with its
// neighbor in the opposite direction.
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

// FillSubDHole — the inverse of Delete Faces. Fills an open boundary loop
// with one new n-gon face over the loop's vertices (no multi-face or
// centroid-fan fill). Refuses if the loop is not an open boundary (each
// consecutive pair must be an edge of the cage used by exactly one face)
// rather than building an invalid or overlapping face.
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

// The ordered boundary loop of a face set (as opposed to
// boundaryLoopFromSeed's single open boundary edge) — Bridge's
// prerequisite: before the two selected patches are deleted, find what
// their rim will be once they are. A boundary-region edge of the set
// (exactly one of its adjacent faces is in the set, as in extrudeFaces) is
// usable only when its other side is a different, unselected face
// (edge.faces.length === 2). A selection whose rim touches an open cage
// edge is refused by name: deleting the selection there would erase that
// edge rather than create a new hole edge.
//
// Orientation: the selected face's surviving neighbor across a rim edge
// (va,vb) traverses that edge as (vb,va), as any two consistently oriented
// faces sharing an edge do. A new face filling this rim must traverse the
// edge opposite the surviving neighbor, i.e. as (va,vb) — the direction the
// selected face used. So `nextOf` records the selected face's forward
// direction (va -> vb) verbatim, not reversed. The test file checks a
// delete-and-refill round trip against the same rim walked by
// boundaryLoopFromSeed.
export function orderedBoundaryLoopOfFaceSet(cage, faceIndices) {
  const topology = buildTopology(cage);
  const selSet = new Set(faceIndices);
  const nextOf = new Map(); // va -> vb, the direction the selected face traverses this rim edge
  for (const fi of faceIndices) {
    const face = cage.faces[fi];
    const n = face.length;
    for (let c = 0; c < n; c++) {
      const va = face[c], vb = face[(c + 1) % n];
      const key = edgeKey(va, vb);
      const edge = topology.edgeMap.get(key);
      const selCount = edge.faces.filter((f) => selSet.has(f)).length;
      if (selCount !== 1) continue; // interior to the selection (2 selected faces share it) — not part of the rim
      if (edge.faces.length !== 2) throw new Error(`orderedBoundaryLoopOfFaceSet: the selection's own rim touches an already-open cage edge ("${key}") — a face group to bridge must be surrounded by other faces on every side`);
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
  /* Closing is not the same as being the whole rim. Every check above is
     local — each rim vertex has exactly one continuation, so the walk closes
     even when the patch has several separate boundaries, and it returns
     whichever one its start vertex sat on. A band of faces wrapped around a
     tube is that shape: an annulus, two rim loops, no single face that can
     stand for it. Every caller treats this loop as the outline of the patch
     — mergeFaces replaces the patch with one n-gon over it, bridgeFaces
     builds a tunnel from it — so a dropped second loop would be left as a
     hole. `nextOf` holds one entry per rim edge, and a single cycle of
     length L consumes exactly L of them, so comparing the two tells one
     loop from several. */
  if (loop.length !== nextOf.size) {
    throw new Error(`orderedBoundaryLoopOfFaceSet: this patch has ${nextOf.size} rim edges but its outline closes after ${loop.length} — it has more than one boundary (a band around the body rather than a disc), so no single face can stand in for it`);
  }
  return loop;
}

// Bridge — connect two face selections with a tunnel of faces (Segments).
// Each selection must be a single simply connected interior patch (see
// orderedBoundaryLoopOfFaceSet's refusals); two open holes are bridged by
// bridgeBoundaryLoops below. Rims of unequal vertex count are joined by
// the monotone correspondence in bridgeRims.
//
// Correspondence — which vertex of loop A lines up with which of loop B —
// is two questions, one deterministic and one heuristic:
//   Direction is fixed. Both loops come out of orderedBoundaryLoopOfFaceSet
//   in the same "fill with one face" convention, and a tunnel between two
//   such loops needs exactly one of them walked in reverse (a cylinder
//   wall's two rims run opposite senses relative to how each would cap off
//   alone). Loop B is always the one reversed; reversing A would work the
//   same by symmetry. The test file checks the result is a consistent
//   2-manifold (every shared edge traversed in opposite directions by its
//   two faces) on several fixtures.
//   Rotation (which vertex of the reversed loop B lines up with loop A's
//   start) has no single correct answer for two independently selected
//   patches. It is resolved by the standard "bridge two loops" heuristic
//   (as in Blender's Bridge Edge Loops): search every rotation, keep the
//   one minimizing the total squared distance between corresponding vertex
//   pairs. `spin` offsets the result by whole rim steps, for a pair of
//   loops where the nearest-distance answer reads as twisted.
export function bridgeFaces(cage, faceIndicesA, faceIndicesB, segments = 1, straightness = 1, creaseWeight = 0, spin = 0) {
  if (!faceIndicesA || !faceIndicesA.length || !faceIndicesB || !faceIndicesB.length) throw new Error('bridgeFaces: faceIndicesA and faceIndicesB must both be non-empty arrays');
  if (!Number.isInteger(segments) || segments < 1) throw new Error('bridgeFaces: segments must be a positive integer');
  const setB = new Set(faceIndicesB);
  for (const fi of faceIndicesA) if (setB.has(fi)) throw new Error(`bridgeFaces: face ${fi} is in both selections — they must be disjoint`);

  const loopA0 = orderedBoundaryLoopOfFaceSet(cage, faceIndicesA);
  const loopB0raw = orderedBoundaryLoopOfFaceSet(cage, faceIndicesB);
  // The two patches are deleted first — that turns a face selection into a
  // pair of open rims, the only shape the tunnel builder works on. An
  // already-open pair of rims skips this step and calls the same builder:
  // see bridgeBoundaryLoops below.
  const del = deleteFaces(cage, [...faceIndicesA, ...faceIndicesB]);
  return bridgeRims(cage, del.cage, del.vertexRemap, loopA0, loopB0raw, segments, 'bridgeFaces', straightness, creaseWeight, spin);
}

// Bridge on two open edge loops. Each seed is one boundary edge of its
// hole; the rest of that hole's rim is walked from it by
// boundaryLoopFromSeed.
//
// No new orientation reasoning is needed: boundaryLoopFromSeed and
// orderedBoundaryLoopOfFaceSet both return their loop in the same
// convention — the order a single new face filling this rim would use — so
// both paths hand the tunnel builder loops of identical meaning, and its
// reverse-B-then-rotate correspondence applies unchanged.
export function bridgeBoundaryLoops(cage, seedKeyA, seedKeyB, segments = 1, straightness = 1, creaseWeight = 0, spin = 0) {
  if (!Number.isInteger(segments) || segments < 1) throw new Error('bridgeBoundaryLoops: segments must be a positive integer');
  const loopA0 = boundaryLoopFromSeed(cage, seedKeyA);
  const loopB0raw = boundaryLoopFromSeed(cage, seedKeyB);
  // Two seeds on the same rim would walk the identical loop and bridge a
  // hole to itself. Caught here by name rather than left to the
  // shared-vertex refusal in bridgeRims, whose message (two openings
  // touching) would misdescribe it.
  if (loopA0.length === loopB0raw.length && loopA0.every((v) => loopB0raw.includes(v))) {
    throw new Error(`bridgeBoundaryLoops: "${seedKeyA}" and "${seedKeyB}" are both on the SAME open boundary loop — Bridge needs one edge from each of TWO different holes`);
  }
  // Nothing is deleted here: the rims are already open, so the tunnel is
  // built onto this cage and every rim vertex keeps its index (an identity
  // remap, where the face path needs a real one).
  const workCage = { vertices: cage.vertices.map((v) => v.slice()), faces: cage.faces.map((f) => [...f]), creases: { ...(cage.creases || {}) } };
  const identity = new Map(cage.vertices.map((_, i) => [i, i]));
  return bridgeRims(cage, workCage, identity, loopA0, loopB0raw, segments, 'bridgeBoundaryLoops', straightness, creaseWeight, spin);
}

// The tunnel builder both Bridge paths share (bridgeRims, below): given two
// rims (in fill-face order), the cage those rims are open on, and how each
// rim's vertex indices map into it, build the ring-by-ring wall between
// them. The correspondence search, the duplicate-edge refusal and the
// winding are one definition, so the two paths cannot drift.
//
// Unequal rims: the monotone correspondence. The resampling happens inside
// the bridge, never on a rim. Both rims keep every vertex they had, in
// place — nothing is inserted, removed or moved, since a bridge may not
// move existing vertices — and the reconciliation lives entirely in the
// new faces.
//
// `c[j] = floor(j * nCoarse / nFine)` is monotone and advances by exactly 0
// or 1 per step, so walking the fine rim once emits exactly `nCoarse` quads
// and `nFine - nCoarse` triangles. A 12-rim bridged to an 8-rim gives 8
// quads and 4 triangles.
//
// Each triangle becomes a valence-3 extraordinary vertex under
// Catmull-Clark, so the limit surface is slightly less even there than an
// all-quad bridge between matched rims; the alternatives are refusing, or
// editing one of the two rims.
//
// The triangles are distributed evenly around the ring by the floor(), not
// bunched at the seam, because `c` advances on a uniform schedule.
function rimCorrespondence(nFine, nCoarse) {
  const c = [];
  for (let j = 0; j <= nFine; j++) c.push(Math.floor((j * nCoarse) / nFine));
  return c; // c[nFine] === nCoarse, i.e. the wrap back to coarse[0]
}
// One band of the tunnel wall, between two rings of possibly different
// counts. Equal counts take the plain quad path.
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
  // repeated far (or near) vertex dropped — a triangle is that quad
  // degenerating, not a separately derived orientation.
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
// The open-run counterpart of rimCorrespondence. A rim wraps, so its
// correspondence is a uniform schedule that lands back on coarse[0] after a
// full turn and has no distinguished starting vertex. A run has ends, and
// they are not free — the first vertex of one run must meet the first of
// the other and the last the last, or the wall would run off the end of the
// shorter run and leave a dangling stub. So this map is monotone with
// c[0] === 0 and c[p - 1] === q - 1 (both ends pinned), and the interior is
// spread evenly between them by the round().
//
// `p >= 2` is guaranteed by the caller's "at least 2 vertices" refusal, so
// the p - 1 divisor is never zero.
function openRunCorrespondence(p, q) {
  const c = [];
  for (let j = 0; j < p; j++) c.push(Math.round((j * (q - 1)) / (p - 1)));
  return c;
}
// One band of an open bridge wall, between two rows of possibly different
// counts — the open-chain counterpart of bridgeBandFaces, which wraps with
// `% p` because a rim closes. Nothing here wraps: the walk stops one short
// of the end of the finer row, so each of its edges gets exactly one face
// and the coarse row stalls where the correspondence does.
//
// Equal counts take the plain quad path, in whichever of the two windings
// the caller's orientation search chose.
function bridgeRunBandFaces(rowNear, rowFar, flip) {
  const p = rowNear.length, q = rowFar.length;
  const out = [];
  if (p === q) {
    for (let i = 0; i + 1 < p; i++) {
      out.push(flip ? [rowNear[i + 1], rowNear[i], rowFar[i], rowFar[i + 1]] : [rowNear[i], rowNear[i + 1], rowFar[i + 1], rowFar[i]]);
    }
    return out;
  }
  // A stalled step repeats a vertex, and the face it produces is that quad
  // with the repeat dropped — a triangle here is the quad degenerating, not
  // a separately derived orientation that could disagree with its
  // neighbors' winding.
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
// Spin is a whole number of rim steps added to the correspondence the
// distance search settles on: +1 pairs each A vertex with the B vertex one
// step further round, so the tunnel twists by one edge; 0 is the search's
// answer. A spun correspondence is held to the same duplicate-edge rule as
// the searched one, and refused by name rather than built non-manifold.
function bridgeRims(sourceCage, workCage, vertexRemap, loopA0, loopB0raw, segments, label, straightness = 1, creaseWeight = 0, spin = 0) {
  const cage = sourceCage;
  if (!Number.isInteger(spin)) throw new Error(`${label}: spin must be a whole number of rim steps`);
  // Shared-vertex refusal. The rungA===rungB check below only rejects a
  // rotation that pairs a shared vertex with itself. Two failure modes
  // remain if the loops share a vertex: (1) the rotation search can accept
  // a rotation where the shared vertex is in no rung pair — the tunnel
  // builds, but that vertex's face neighborhood is not a topological disk
  // (a vertex-non-manifold pinch where the limit surface collapses to
  // a point), invisible to the checks here, which reason about edges, not
  // vertex link-disks; (2) if the two loops share two vertices without
  // sharing an edge, some rotation can pair them crosswise (A[i]=u,B[i]=v
  // and A[j]=v,B[j]=u), creating the same rung edge (u,v) twice — an edge
  // with 4 incident faces, caught by the edgeMap check only if (u,v)
  // already exists elsewhere. Both are prevented by refusing whenever the
  // two loops' vertex sets intersect: two openings that already touch have
  // no gap to tunnel through. `stitchEdgeRuns` refuses its shared-vertex
  // case the same way.
  const sharedVertex = loopA0.find((vi) => loopB0raw.includes(vi));
  if (sharedVertex !== undefined) {
    throw new Error(`${label}: the two boundary loops share vertex ${sharedVertex} — they already touch, with no real gap to tunnel through (pick two openings separated by real, uninterrupted cage surface)`);
  }
  const nA = loopA0.length, nB = loopB0raw.length;
  // Every ring the tunnel builds internally carries the fine count, so the
  // reconciliation happens in exactly one band — the one touching the coarse
  // rim — rather than being spread through every intermediate ring.
  const nF = Math.max(nA, nB);
  const loopB0 = [...loopB0raw].reverse(); // deterministic — see bridgeFaces' header
  // The rung pairs for one candidate alignment: which A vertex each B vertex
  // reaches across to. With equal counts this is index-to-index.
  function rungPairs(seqB) {
    const pairs = [];
    if (nA === nB) { for (let i = 0; i < nA; i++) pairs.push([i, i]); return pairs; }
    if (nA < nB) { const c = rimCorrespondence(nB, nA); for (let j = 0; j < nB; j++) pairs.push([c[j], j]); return pairs; }
    const c = rimCorrespondence(nA, nB);
    for (let i = 0; i < nA; i++) pairs.push([i, c[i]]);
    return pairs;
  }

  // Rotation validity: on a cage with internal symmetry (e.g. a plain box),
  // some rotation offsets pair loopA[i] with a loopB[i] that already sits
  // one edge away via a different, kept face (the two rims share a
  // "diagonal" neighbor). Building the tunnel with that offset would
  // duplicate an existing edge — an edge used by 4 faces — and the distance
  // heuristic cannot see it (a duplicate can be the shortest rung). So every
  // rotation candidate is checked against the cage's topology after
  // deletion, and candidates are tried in ascending-distance order: the
  // closest rotation that duplicates nothing wins.
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
      if (rungA === undefined || rungB === undefined) return true; // unreachable for a well-formed selection; treated as invalid
      // When the two rims share a cage vertex without sharing an edge (e.g.
      // two diagonal faces of a 2x2 grid, meeting only at the center
      // vertex), that vertex survives deleteFaces (the unselected faces still
      // use it) and rungA===rungB for the rotation that pairs it — a
      // degenerate rung from a vertex to itself, which the edgeMap check
      // cannot catch (a cage has no self-loop edge). Its distSq is 0, so the
      // closest-rotation heuristic would prefer it over every other rotation.
      // Rejected explicitly, as stitchEdgeRuns refuses runs sharing a vertex.
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
  let alignedLoopB0 = validCandidate.candidate;
  if (spin) {
    const spunBy = (k) => rotate(alignedLoopB0, ((k % nB) + nB) % nB);
    const spun = spunBy(spin);
    if (wouldDuplicateExistingEdge(spun)) {
      let near = null;
      for (let k = 1; k < nB && near === null; k++) { if (!wouldDuplicateExistingEdge(spunBy(spin - k))) near = spin - k; else if (!wouldDuplicateExistingEdge(spunBy(spin + k))) near = spin + k; }
      throw new Error(`${label}: spin ${spin} pairs rim vertices the cage already joins${near === null ? '' : `; ${near} is the nearest spin that builds`}`);
    }
    alignedLoopB0 = spun;
  }

  const loopA = loopA0.map((vi) => del.vertexRemap.get(vi));
  const loopB = alignedLoopB0.map((vi) => del.vertexRemap.get(vi));
  if (loopA.some((v) => v === undefined) || loopB.some((v) => v === undefined)) {
    throw new Error(`${label}: internal error — a rim vertex did not survive face deletion (this should not happen for a well-formed selection)`);
  }

  const newVertices = del.cage.vertices.map((v) => v.slice());
  const ptsA = loopA0.map((vi) => cage.vertices[vi]);
  const ptsB = alignedLoopB0.map((vi) => cage.vertices[vi]);
  // Tangents are read off workCage (post-deletion, where the rims are
  // naked) using the remapped indices, while positions come from the source
  // cage — deleteFaces preserves every surviving vertex position exactly, so
  // the two agree.
  // The corresponded partner point for each rim vertex — with equal counts
  // this is ptsB/ptsA verbatim. With unequal counts a coarse vertex has
  // several partners; the first is used, since this only aims the tangent
  // and any of them points the same way.
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
    // An intermediate ring carries the fine count, so the tunnel keeps its
    // full resolution all the way along and only the single band against the
    // coarse rim reconciles. At t < 1 the several fine points sharing one
    // coarse partner are still distinct, so no face degenerates.
    for (let j = 0; j < nF; j++) {
      const [ai, bi] = pairs[j % pairs.length];
      ring.push(newVertices.length);
      newVertices.push(bridgeSpanPoint(ptsA[ai], ptsB[bi], tanA[ai], tanB[bi], t, straightness));
    }
    rings.push(ring);
  }
  rings.push(loopB);

  // Face winding [a, b, bFar, aFar], both rings walked forward (loopB was
  // reversed once above, so no second reversal belongs here). The test file
  // checks the result on two separate boxes with a gap between them, one
  // opened face each, for consistent winding (every edge shared by exactly
  // 2 faces is traversed in opposite directions).
  const newFaces = [...del.cage.faces];
  for (let r = 0; r < rings.length - 1; r++) {
    for (const f of bridgeBandFaces(rings[r], rings[r + 1])) newFaces.push(f);
  }
  // Rim edges only, and on the remapped indices — the crease map is keyed by the
  // indices the returned cage uses, which after deleteFaces are not the
  // source cage's.
  const outCreases = { ...del.cage.creases };
  creaseChain(outCreases, loopA, creaseWeight, true);
  creaseChain(outCreases, loopB, creaseWeight, true);
  return {
    cage: { vertices: newVertices, faces: newFaces, creases: outCreases },
    tunnelFaceIndices: newFaces.map((_, i) => i).slice(del.cage.faces.length),
    spinSteps: nB,
  };
}

// Bridge on two open edge runs — any run of edges to any other: one edge
// to one edge, N to N, across bodies, or two runs of the same hole.
//
//   A closed loop has rotational freedom, which is why bridgeRims runs an
//   n-way rotation search scored by total squared distance, and needs a
//   duplicate-edge check to stop the closest rotation from being a
//   non-manifold one. Two open runs have ends. There is no rotation — only
//   a direction choice, forward or reversed, 2 candidates.
//
// Orientation is decided by topology: a consistently wound 2-manifold never
// traverses the same directed edge twice, so a candidate rung set that
// would reuse one is the wrong winding, and directedEdgeReuseCount decides
// it. Correspondence (which end of runA meets which end of runB) is the
// geometric question, and total squared distance answers it.
//
// Additive only — unlike bridgeFaces, nothing is deleted (the runs are
// already naked edges), so there is no vertexRemap and every existing
// vertex index stays valid.
//
// Bridging two runs of the same hole splits that boundary loop into two,
// where the closed-loop Bridge merges two loops into a tunnel. Both are
// legal on a Catmull-Clark cage; the result here is still open (the two
// ends of the new wall are naked edges), not a closed solid.
//
// Unequal counts are supported as in the closed-loop Bridge: nothing is
// resampled, because a bridge may not move an existing vertex. The fine run
// drives the walk, the coarse run stalls where openRunCorrespondence says
// it does, and each stalled step's quad degenerates to a triangle — `p - q`
// of them for a p-to-q bridge, spread evenly along the wall. Each triangle
// becomes a valence-3 extraordinary vertex under Catmull-Clark. The ends
// are pinned first-to-first and last-to-last.
//
// An open run is not a closed rim. A run's precondition — every
// consecutive pair is a naked edge — is also met by a closed loop handed
// over as a plain vertex list. Bridged as a run, it builds a junction that
// winds consistently, has no repeated-vertex face, and leaves every rim's
// closing edge unattached — a slit down each arm that no structural check
// here sees.
//
// Two shapes of closed input, both refused by name:
//   - the bare cycle [v0..v_{m-1}], detected by its closing edge existing
//     in the cage as a naked edge between the run's two ends. An open run
//     can only have that if it spans its whole rim bar one edge — the same
//     loop under another name.
//   - the wrapped list [v0..v_{m-1}, v0], which is not a simple chain.
// Length 2 is exempt from the first check: a 2-vertex run's single edge is
// the edge between its ends, and no loop can close in two vertices.
//
// `wording` exists because the consequence of handing a closed rim to an
// open-run function differs by caller, and a refusal should name the right
// one. A bridge builds faces onto the run and leaves the rim's closing edge
// unattached; a weld builds nothing and fails a different way (see
// stitchEdgeRuns). The checks themselves are shared; the defaults are the
// bridge wording.
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

  // Two runs that already touch have no gap to bridge, and a shared vertex
  // would pinch the result vertex-non-manifold — the same refusal
  // bridgeRims and stitchEdgeRuns make.
  const shared = runA.find((vi) => runB.includes(vi));
  if (shared !== undefined) throw new Error(`bridgeEdgeRuns: the two edge runs share vertex ${shared} — they already touch, with no real gap to bridge across`);

  const topology = buildTopology(cage);
  checkOpenRunChain(cage, topology, runA, 'runA', 'bridgeEdgeRuns');
  checkOpenRunChain(cage, topology, runB, 'runB', 'bridgeEdgeRuns');

  const nA = runA.length, nB = runB.length;
  // Which runB vertex each runA vertex reaches across to, and vice versa —
  // the same pairing bridgeRunBandFaces walks, needed here for the distance
  // score and again below for the tangents and the interior rows. With equal
  // counts it is index-to-index. An open run has no rotational freedom, so
  // unlike bridgeRims' rungPairs this depends on the counts alone and is
  // built once.
  const pairs = [];
  if (nA === nB) { for (let i = 0; i < nA; i++) pairs.push([i, i]); }
  else if (nA < nB) { const c = openRunCorrespondence(nB, nA); for (let j = 0; j < nB; j++) pairs.push([c[j], j]); }
  else { const c = openRunCorrespondence(nA, nB); for (let i = 0; i < nA; i++) pairs.push([i, c[i]]); }

  const candidates = [];
  for (const reversed of [false, true]) {
    const seqB = reversed ? [...runB].reverse() : [...runB];
    // Scored over corresponding pairs, not over a shared index — with unequal
    // counts a plain index walk would score the two runs against vertices that
    // never meet, and would not even visit the longer run's tail.
    let dist = 0;
    for (const [ai, bi] of pairs) dist += distSq(cage.vertices[runA[ai]], cage.vertices[seqB[bi]]);
    for (const flip of [false, true]) {
      // Probed through the band helper itself, so directedEdgeReuseCount
      // judges the topology the bridge will build — including the degenerate
      // triangles, which carry the rim edges that decide the flip.
      const faces = bridgeRunBandFaces(runA, seqB, flip);
      candidates.push({ seqB, flip, dist, reuse: directedEdgeReuseCount([...cage.faces, ...faces]) });
    }
  }
  const valid = candidates.filter((c) => c.reuse === 0);
  if (!valid.length) throw new Error('bridgeEdgeRuns: no correspondence between these two edge runs produces a consistently-wound result — the runs may not both bound the same open region, or already meet through existing geometry');
  valid.sort((x, y) => x.dist - y.dist);
  const chosen = valid[0];
  const seqB = chosen.seqB;

  // Twisted bridge: winding validity is a hard constraint and distance only
  // a preference, so when the two runs' surfaces are oppositely oriented
  // (face normals pointing the same way rather than continuing one surface),
  // the nearest correspondence is not manifold-legal and the only legal one
  // pairs each rim vertex with the far end of the other run — a twisted wall
  // whose rungs cross in the middle. That is valid cage topology and a
  // legitimate join between two independently built bodies, but rarely the
  // intent, so it is built and reported (`twisted`) for the caller to name.
  const nearestDist = Math.min(...candidates.map((c) => c.dist));
  const twisted = chosen.dist > nearestDist * (1 + 1e-9);

  const ptsA = runA.map((vi) => cage.vertices[vi]);
  const ptsB = seqB.map((vi) => cage.vertices[vi]);
  // rimTangents wants one partner point per vertex of the run it is aiming.
  // With equal counts that is ptsB/ptsA verbatim. With unequal counts a
  // coarse vertex has several partners; the first is used, since this only
  // aims the tangent and any of them points the same way — the same choice
  // bridgeRims makes.
  const partnerOfA = new Array(nA), partnerOfB = new Array(nB);
  for (const [ai, bi] of pairs) {
    if (partnerOfA[ai] === undefined) partnerOfA[ai] = ptsB[bi];
    if (partnerOfB[bi] === undefined) partnerOfB[bi] = ptsA[ai];
  }
  const tanA = rimTangents(cage, topology, runA, partnerOfA, false, 1);
  const tanB = rimTangents(cage, topology, seqB, partnerOfB, false, -1);

  const newVertices = cage.vertices.map((v) => v.slice());
  const rows = [runA];
  // Every interior row carries the fine count (pairs.length), so the
  // reconciliation happens in the single band touching the coarse run and
  // every other band is plain quads. At t < 1 the several fine points
  // sharing one coarse partner are still distinct, so no interior face
  // degenerates.
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
  // multi-segment one repeats the band pattern per row pair, but at unequal
  // counts only one of those bands is the reconciling one and the rest are
  // plain quads, so the assembled wall is checked again rather than assumed.
  const finalFaces = [...newFaces, ...added];
  if (directedEdgeReuseCount(finalFaces) !== 0) throw new Error('bridgeEdgeRuns: internal error — the assembled bridge is not consistently wound (this should not happen for two well-formed open runs)');

  // Rim edges only — an open run is not closed, so its last vertex has no edge
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

// Stitch — merge two edge runs into one (First/Second/Average position,
// after Rhino's Stitch). `runA`/`runB` are each an ordered array of vertex
// indices tracing a connected boundary chain — an open run of consecutive
// boundary edges, as Rhino's Stitch expects two nearby, disconnected
// boundary edges. Matching vertex counts are required. Bridge reconciles
// unequal counts by letting a quad degenerate to a triangle, which costs
// only a face; a weld reconciles by identifying vertices, so pairing two
// fine vertices onto one coarse one would merge them into each other and
// move existing geometry.
//
// Mechanism — a topology merge, not a face rebuild: once each corresponding
// pair of vertices is identified as one vertex (moved to the position
// `position` dictates) and every face reference to the losing vertex is
// remapped onto the keeping one, the edge along one run (one adjacent face,
// e.g. runA[i]-runA[i+1]) and the edge along the other (runB[i]-runB[i+1],
// also one face) become the same edgeKey, used by both original faces: a
// 2-face interior edge, closing the gap. No new faces are built.
//
// Correspondence — an open run has no rotational ambiguity (unlike Bridge's
// closed loops); the only question is which end matches which, forward vs
// reversed — resolved by minimal total squared distance, as Bridge
// resolves its larger (rotation + direction) search.
//
// That is why a closed rim is refused here, and the failure differs from
// the bridges'. A weld builds no faces, so there is no unattached closing
// edge: two tube rims weld along every edge, including each rim's closing
// one. What it cannot find is the rotation. Forward-vs-reversed is the
// whole search, so vertex i of one rim is paired with vertex i of the other
// wherever that lands: on two coaxial rims, diametrically opposite vertices
// can be averaged onto the axis, crushing the merged ring, and the welded
// edges can be traversed twice in the same direction, leaving the result
// inconsistently wound. A closed rim's rotation is what bridgeRims' n-way
// rotation search is for, and the refusal names it.
//
// checkOpenRunChain also enforces the three preconditions shared with the
// bridges — a run is a simple chain, its consecutive pairs are cage edges,
// and those edges are naked. Without them a list of non-adjacent vertices
// welds unrelated points into a vertex pinch, and a run of interior (2-face)
// vertices produces edges with three faces, a non-manifold cage.
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
  // Defensive cleanup (not expected to fire for a well-formed stitch of two
  // separate runs, since a bijective per-index merge cannot collapse two
  // distinct face vertices onto one index): drop any consecutive-duplicate
  // vertex within a face, and drop any face that collapses below 3 sides.
  // `collapsedFaceCount` reports how many were dropped.
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
    if (na === undefined || nb === undefined || na === nb) continue; // a dropped/merged-to-self edge — nothing left to crease
    newCreases[edgeKey(na, nb)] = weight;
  }

  return {
    cage: { vertices: newVertices, faces: newFaces, creases: newCreases },
    mergedVertexIndices: runA.map((vi) => finalIndexOf(vi)),
    collapsedFaceCount,
  };
}

// Subdivide (global) — one refinement of the whole cage.
// subdivideCatmullClark (kernel/subd.mjs) produces a finer control cage
// whose limit surface is exactly the input cage's limit surface — the
// defining property of a subdivision surface, not an approximation. So
// "subdivide the whole cage one level" and "commit that finer cage as the
// editable cage" is one call: every quad becomes 4, an n-gon becomes n
// quads, the shape is unchanged, and there are more control vertices to
// push and pull. Semi-sharp crease weights decrement by exactly 1 per level
// (the DeRose semi-sharp decay), which is what keeps the limit surface
// identical across the refinement. Returns a new cage (the input is never
// mutated). Refining only selected faces is subdivideFaces, below.
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

// MergeFaces — dissolves the shared internal edges between a connected
// group of 2+ edge-adjacent selected faces, replacing the group with one
// new n-gon face spanning their combined outer boundary (Rhino's
// MergeFaces). The dissolved internal edges' creases disappear with the
// edges; every surviving (rim) edge keeps its crease.
//
// orderedBoundaryLoopOfFaceSet gives the new face's ordered vertex loop,
// wound in the direction the selected faces traverse it — opposite the
// surviving neighbor — so the new n-gon is consistently oriented with its
// neighbors. deleteFaces does the removal, orphaned-vertex prune,
// compaction and crease rebuild: it drops any crease whose edge is gone
// from the output (the dissolved internal edges) and keeps every surviving
// edge's crease (the rim edges still exist via their unselected neighbor).
// The new n-gon is then appended, its loop translated through deleteFaces'
// vertexRemap.
//
// The selected faces must form a single edge-connected group. A
// disconnected selection (e.g. two faces on opposite sides of the cage) has
// no single merged n-gon, and orderedBoundaryLoopOfFaceSet would walk only
// one component's rim, so it is refused by name before that function is
// called; the app layer partitions a multi-group selection and merges one
// group at a time. The rim may not touch an open cage boundary edge
// (orderedBoundaryLoopOfFaceSet's refusal, as for Bridge).
export function mergeFaces(cage, faceIndices) {
  if (!faceIndices || faceIndices.length < 2) throw new Error('mergeFaces: select at least 2 edge-adjacent faces to merge (merging a single face is a no-op)');
  const seen = new Set();
  for (const fi of faceIndices) {
    if (!Number.isInteger(fi) || fi < 0 || fi >= cage.faces.length) throw new Error(`mergeFaces: face index ${fi} is out of range`);
    if (seen.has(fi)) throw new Error(`mergeFaces: face index ${fi} was selected more than once`);
    seen.add(fi);
  }

  // Single edge-connected group check — must run before
  // orderedBoundaryLoopOfFaceSet, which would return one component's rim for
  // a disconnected selection. Uses the same shared-edge adjacency as the app
  // layer's superbFaceGroupsFromSelection grouping.
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
  // a non-simple rim (see orderedBoundaryLoopOfFaceSet).
  const loop = orderedBoundaryLoopOfFaceSet(cage, faceIndices);

  // Remove the group, prune the orphaned interior vertices, compact, and
  // rebuild creases (dropping the dissolved internal edges' creases along
  // with their edges).
  const del = deleteFaces(cage, faceIndices);
  const newLoop = loop.map((vi) => del.vertexRemap.get(vi));
  if (newLoop.some((v) => v === undefined)) {
    // A rim vertex is shared with a surviving neighbor face, so deleteFaces
    // cannot orphan it — an internal-consistency guard, not an expected path.
    throw new Error('mergeFaces: internal error — a rim vertex did not survive the merge (this should not happen for a well-formed connected selection)');
  }
  const newFaces = [...del.cage.faces, newLoop];
  const newCage = { vertices: del.cage.vertices, faces: newFaces, creases: del.cage.creases };

  // Manifold check — the same last-resort structural check
  // kernel/subdconvert.mjs runs on its output (no face with a repeated
  // vertex; no edge shared by more than 2 faces). Not expected to fire for a
  // well-formed connected selection; it keeps a corrupt cage from being
  // returned, and it is cheap.
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

// Subdivide selected faces — refine only the picked faces and leave the
// rest of the cage where it is.
//
// Refining one face puts a new vertex at the middle of an edge that an
// unrefined neighbor still describes as a single side: a T-junction. The
// neighbor's face loop never mentions the new vertex, so the two faces
// disagree about their shared boundary, and subdivideCatmullClark, Reflect
// and Bridge would all inherit a non-manifold cage — a crack that renders.
//
// Catmull-Clark here accepts any face size (superbCylinderCage's n-gon caps
// rely on it), so the neighbor does not need splitting into quads: the new
// midpoint is inserted into the neighbor's vertex loop. A quad bordering
// one refined face becomes a 5-gon, one bordering two becomes a 6-gon, and
// there is no T-junction by construction rather than by tolerance. Every
// edge still has exactly two faces that agree on it.
//
// Unlike the global subdivide, this moves no existing vertex, and it places
// new vertices at plain edge midpoints and face centroids rather than
// through the smooth Catmull-Clark rules. Applying those rules locally
// would move the selection's boundary vertices, which belong equally to
// faces not being refined. The consequence: a local subdivide changes the
// limit surface near the refined region, where the global one is exactly
// limit-preserving. Rhino's local subdivide behaves the same way, for the
// same reason.
export function subdivideFaces(cage, faceIndices) {
  const selected = [...new Set(faceIndices)];
  if (!selected.length) throw new Error('subdivideFaces: no faces selected');
  for (const fi of selected) {
    if (!Number.isInteger(fi) || fi < 0 || fi >= cage.faces.length) throw new Error(`subdivideFaces: ${fi} is not a face of this cage`);
    if (cage.faces[fi].length < 3) throw new Error(`subdivideFaces: face ${fi} is degenerate`);
  }
  const selectedSet = new Set(selected);
  const newVertices = cage.vertices.map((v) => v.slice());

  // One edge point per distinct edge of the selection — an edge shared by
  // two selected faces is discovered twice and built once, which keeps those
  // two faces agreeing on their shared boundary.
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
    // An unselected face keeps its shape and gains only the midpoints that
    // exist on its sides — the T-junction repair, and why a neighbor can
    // come out a 5-gon or a 6-gon.
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

  // Crease remap, as in insertEdgeLoop: a split edge is gone from the
  // output topology, so its weight would dangle on a dead key while both
  // halves came out smooth. It transfers to both halves. No decay: this is
  // a topology refinement, not a subdivision step.
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
  // Last-resort manifold check, the same one subdconvert.mjs runs on its
  // output: no face may repeat a vertex, and no edge may be shared by more
  // than two faces. The construction above should make both impossible;
  // this checks it.
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

// N-way bridge — a Y, a cross, or any N ≥ 3 open edge runs joined through a
// single hub. Rhino's SubD Bridge joins two runs only.
//
// Bridging two runs is a quad grid — a strip between two chains, with one
// correspondence to choose. Three or more runs meet at a branching
// junction, and there is no canonical quad topology for one: something has
// to occupy the middle. Here that is an explicit hub — each run is bridged
// inward to its own shrunken copy, those copies are welded corner to corner
// into one closed ring, and the ring is capped by a single n-gon.
//
// Catmull-Clark refines a face of any size, and the SuperBCylinder/
// SuperBCone cages already have n-gon caps and an n-valence apex, so a hub
// face and its extraordinary corner vertices are ordinary geometry here.
// A tensor-product NURBS patch cannot represent a branching junction as
// one surface.
//
// Decided rather than derived:
//  - Cyclic order. The runs arrive as an unordered set; a hub needs to know
//    which run sits next to which. Taken from each run's angle about the
//    hub center, measured in the junction's best-fit plane.
//  - Run direction. Each run is traversed so that walking the ring is
//    consistent: a run's far end is whichever of its two endpoints lies
//    nearer the next run around the ring.
//  - Winding. Both arm windings and both cap directions are built and
//    tested against `directedEdgeReuseCount`, and the one that produces a
//    consistently wound 2-manifold is kept, as the two-run bridge does.
//
// Straightness applies at the rim only: a rung leaves each run along that
// run's outgoing surface direction and arrives at the hub flat, because
// there is no surface at the hub to be tangent to.
const HUB_INSET = 0.62; // how far in from its run each inner ring sits, as a fraction of the way to the hub center
function norm3(a) { const L = length(a); return L > 1e-12 ? scale(a, 1 / L) : [0, 0, 0]; }

// The junction plane and the cyclic order around it — shared by both hub
// constructions in this file.
//
// The plane is taken from the single most spread pair, not a sum. Summing
// the signed cross products over array-ordered pairs is order-dependent —
// swap two arms and terms change sign — and for a symmetric junction (N
// arms evenly spaced, the ordinary case) the terms cancel to zero and a
// coplanar, well-spread set would be refused as collinear. For coplanar
// centroids every pair's cross product is parallel to the true normal, so
// taking the largest is exact and independent of the order the arms arrive
// in; for a non-coplanar set it is a best available estimate. The same
// technique as kernel/selfintersect.mjs's bestFitPlane.
//
// Its sign does not matter to either caller: it only fixes which way round
// the ring is numbered, and each construction absorbs that (the run hub
// searches both windings; the closed-rim hub is invariant, its two poles
// swapping roles).
//
// Known limitation: the cyclic order is a ring in one plane. Three arms are
// always coplanar, so N = 3 is safe. A three-dimensional N >= 4 junction —
// the ±X/±Y/+Z frame corner — has arms that no single plane orders, and two
// of them can project to nearly the same angle here and be mis-ordered. The
// closed-rim hub below refuses that case by name via its equator test; the
// run hub does not.
function junctionPlaneOrder(mids, centre, who, noun) {
  let nrm = [0, 0, 0], nrmLen = 0;
  for (let i = 0; i < mids.length; i++) for (let j = i + 1; j < mids.length; j++) {
    const c = cross(sub(mids[i], centre), sub(mids[j], centre));
    const cl = length(c);
    if (cl > nrmLen) { nrm = c; nrmLen = cl; }
  }
  nrm = norm3(nrm);
  if (!length(nrm)) throw new Error(`${who}: the ${noun}s are in a line, so there is no junction plane — move one ${noun} off it`);
  // An in-plane reference direction: the first arm's offset from the
  // center, with any out-of-plane part removed. Which arm is first only
  // rotates every measured angle by a constant, so the cyclic sequence the
  // sort produces is the same one cut at a different place.
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
// Same scale-free contract as subdnetwork.mjs: a one-plane angular ordering
// holds only while the worst arm center sits within 25% of the arms' mean
// spread from that plane. Three centers are always coplanar, so this applies
// only for N >= 4. Kept local so subdnetwork -> subdedit does not become a
// circular dependency.
const RUN_HUB_PLANARITY_TOLERANCE = 0.25;
function runHubPlanarityResidual(pts) {
  const centre = scale(pts.reduce((acc, p) => add(acc, p), [0, 0, 0]), 1 / pts.length);
  let nrm = [0, 0, 0], best = 0;
  for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
    const c = cross(sub(pts[i], centre), sub(pts[j], centre)), cl = length(c);
    if (cl > best) { best = cl; nrm = c; }
  }
  if (!(best > 0)) return 0;
  const n = norm3(nrm);
  const spread = pts.reduce((acc, p) => acc + length(sub(p, centre)), 0) / pts.length;
  return spread > 0 ? Math.max(...pts.map((p) => Math.abs(dot(sub(p, centre), n)))) / spread : 0;
}
export function bridgeEdgeRunsHub(cage, runs, segments = 1, straightness = 1, creaseWeight = 0) {
  if (!Array.isArray(runs) || runs.some((r) => !Array.isArray(r))) throw new Error('bridgeEdgeRunsHub: runs must be an array of vertex-index arrays');
  if (runs.length < 3) throw new Error(`bridgeEdgeRunsHub: a hub junction needs at least 3 edge runs (got ${runs.length}) — two runs is an ordinary bridge, use bridgeEdgeRuns`);
  if (runs.some((r) => r.length < 2)) throw new Error('bridgeEdgeRunsHub: each run needs at least 2 vertices (at least one edge)');
  if (!Number.isInteger(segments) || segments < 1) throw new Error('bridgeEdgeRunsHub: segments must be a positive integer');
  if (!(straightness >= 0 && straightness <= 1)) throw new Error('bridgeEdgeRunsHub: straightness must be between 0 and 1');
  for (let i = 0; i < runs.length; i++) for (let j = i + 1; j < runs.length; j++) {
    const shared = runs[i].find((vi) => runs[j].includes(vi));
    if (shared !== undefined) throw new Error(`bridgeEdgeRunsHub: runs ${i} and ${j} share vertex ${shared} — they already touch, with no real gap to bridge across`);
  }
  const topology = buildTopology(cage);
  runs.forEach((run, i) => checkOpenRunChain(cage, topology, run, `run ${i}`, 'bridgeEdgeRunsHub'));

  // The hub's center and plane — see junctionPlaneOrder, which both hub
  // constructions share.
  const mids = runs.map((run) => scale(run.reduce((acc, vi) => add(acc, cage.vertices[vi]), [0, 0, 0]), 1 / run.length));
  const centre = scale(mids.reduce((acc, p) => add(acc, p), [0, 0, 0]), 1 / mids.length);
  const planarResidual = mids.length >= 4 ? runHubPlanarityResidual(mids) : 0;
  if (planarResidual > RUN_HUB_PLANARITY_TOLERANCE) throw new Error(`bridgeEdgeRunsHub: the ${mids.length} runs sit ${(planarResidual * 100).toFixed(0)}% of their spread off one plane — move them nearer one, or bridge the faces`);
  const { order } = junctionPlaneOrder(mids, centre, 'bridgeEdgeRunsHub', 'run');

  // Each run is walked so its far end is the one nearer the next run around
  // the ring — that makes "run i's end meets run i+1's start" true for every
  // i, which is why the ring closes.
  //
  // The rule fails under an exact tie, which is ordinary: on a symmetric
  // junction (three arcs evenly spaced around one rim) a run's two endpoints
  // are equidistant from the next run to the bit, so `<` falls through to
  // "keep" and the direction is decided by the order the caller handed the
  // run in rather than by geometry, and some of those orders give a ring
  // that cannot be wound consistently.
  //
  // Under a tie every direction is geometrically equally good, so the
  // assignment is searched, as the winding below is. The derived assignment
  // is tried first and is what every non-degenerate junction uses, so the
  // search costs nothing in the ordinary case.
  const N = order.length;
  const derivedFlips = order.map((ri, k) => {
    const run = runs[ri];
    const nextMid = mids[order[(k + 1) % N]];
    return distSq(cage.vertices[run[0]], nextMid) < distSq(cage.vertices[run[run.length - 1]], nextMid);
  });
  const seqFor = (flips) => order.map((ri, k) => (flips[k] ? [...runs[ri]].reverse() : [...runs[ri]]));
  // The derived assignment first, then every other combination. Flipping all
  // of them at once only reverses the ring's orientation, which the winding
  // search absorbs, so this converges early in practice. Capped so a
  // pathological arm count cannot turn a refusal into a hang; beyond the cap
  // only the derived assignment is tried.
  const MAX_SEARCHED_ARMS = 8;
  const directionCandidates = [derivedFlips];
  if (N <= MAX_SEARCHED_ARMS) {
    for (let bits = 0; bits < (1 << N); bits++) {
      const flips = Array.from({ length: N }, (_, k) => !!((bits >> k) & 1));
      if (flips.every((f, k) => f === derivedFlips[k])) continue;
      directionCandidates.push(flips);
    }
  }
  const attempt = (seq) => buildHubForSeq(cage, topology, seq, N, centre, segments, straightness);
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
// Returns null if no winding of this assignment is consistent, so the caller
// can try another; every attempt allocates its own vertex array, so a
// rejected one leaves no orphaned vertices in the cage.
function buildHubForSeq(cage, topology, seq, N, centre, segments, straightness) {
  const newVertices = cage.vertices.map((p) => p.slice());
  const push = (p) => { newVertices.push(p); return newVertices.length - 1; };
  const innerPos = seq.map((run) => run.map((vi) => lerp3(cage.vertices[vi], centre, HUB_INSET)));
  // Consecutive inner runs share their touching endpoint — that single weld
  // per junction turns N separate inner chains into one closed ring. Each
  // arm contributes its (m[k]-1) steps, so unequal run counts need neither
  // resampling nor a transition patch: the central n-gon has Σ(m[k]-1)
  // vertices.
  const corner = [];
  for (let k = 0; k < N; k++) corner.push(push(lerp3(innerPos[k][innerPos[k].length - 1], innerPos[(k + 1) % N][0], 0.5)));
  const inner = seq.map((_, k) => {
    const m = seq[k].length;
    const row = [corner[(k - 1 + N) % N]];
    for (let j = 1; j < m - 1; j++) row.push(push(innerPos[k][j]));
    row.push(corner[k]);
    return row;
  });

  // The ring, walked once: each arm contributes its start corner and its
  // interior vertices, and the next arm's start corner continues it.
  const ring = [];
  for (let k = 0; k < N; k++) for (let j = 0; j + 1 < inner[k].length; j++) ring.push(inner[k][j]);

  // Every vertex this junction needs, built once per direction attempt. Only
  // the winding is searched below, and winding is a property of the face
  // lists alone — building geometry inside that search would append a fresh
  // copy of every interior row per rejected trial and leave the discarded
  // ones orphaned in the cage.
  const armRows = seq.map((run, k) => {
    const m = run.length;
    const outer = run.map((vi) => cage.vertices[vi]);
    const innerPts = inner[k].map((vi) => newVertices[vi]);
    const tan = rimTangents(cage, topology, run, innerPts, false, 1);
    const rows = [run];
    for (let r = 1; r < segments; r++) {
      const t = r / segments;
      const row = [];
      for (let j = 0; j < m; j++) {
        // The hub end gets a straight tangent: there is no surface there for
        // a rung to leave along.
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
        for (let j = 0; j + 1 < near.length; j++) {
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

// N-way closed-rim hub — N >= 3 closed rims (tube ends) welded into one
// junction, as at a pipe network's junction.
//
// bridgeBoundaryLoops joins two closed loops into a tunnel and stops at
// two. bridgeEdgeRunsHub joins N open runs, and relies on each run having
// two ends: consecutive runs weld end to end into one closed ring, which a
// single n-gon caps. A closed rim has no ends to weld, and handed to that
// construction it would leave each rim's closing edge unattached
// (checkOpenRunChain refuses it by name).
//
// The construction — a sphere with N holes, which is what a junction
// surface is. N boundary circles on a genus-0 surface give Euler
// characteristic 2 - N, and the smallest realization of it here adds
// exactly two new vertices, the junction's poles:
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
//   N and every m. The tests check the Euler count.
//
// No Segments option: the hub faces attach directly to the rims the caller
// supplies, so the only new vertices are the two poles. A pipe network is
// expected to have stopped its tubes short of the junction — that inset is
// the caller's job (HUB_INSET is the constant to generalize for it); doing
// it here would move existing vertices, which a bridge may not do.
//
// Derived and verified:
//
//  - Fill order is taken from the cage, never from the caller. A closed rim
//    is a cycle: the given array can start anywhere and run either way. The
//    canonical direction is boundaryLoopFromSeed's — the order a single face
//    filling this rim would use, which traverses each rim edge opposite to
//    that edge's one existing face. Normalizing to it absorbs every
//    direction flip and rotation of the input once.
//
//  - The split is chosen by maximizing how cleanly it separates the rim's
//    +normal side from its -normal side, over every candidate pair of split
//    vertices. Exact ties are ordinary on a symmetric rim (a facets-8 ring
//    whose vertices straddle the equator has a three-way tie), so a tie is
//    broken toward the most balanced split — there, the two vertices
//    nearest the equator.
//
//  - Which neighbor each arm connects to is derived and then measured. The
//    derivation: a rim's fill order runs counter-clockwise about its outward
//    direction d, and writing the rim in the orthonormal basis
//    e1 = normalize(n - (n.d)d), e2 = d x e1 gives dot(v - center, n) =
//    rho * cos(phi) with e2 contributing nothing, so the +n arc runs from
//    phi = -90 to +90 and its start sits at -rho*e2 = rho*(n x d)/|..| — the
//    direction of the next arm around n. A winding check cannot tell the
//    two pairings apart (both close a valid ring), so the choice is settled
//    by total crotch-edge length under each pairing, the shorter winning;
//    by the derivation that is almost always the next arm.
//
//  - Winding is not searched: every hub face walks its rim arc in fill
//    order, which is opposite to the arm face that owns that edge.
//    Consistency follows by construction and is checked against
//    directedEdgeReuseCount.
//
// Known limitation: the junction is ordered around one plane. An arm
// pointing along the junction normal has no equator to divide between the
// two poles — its whole rim lies flat in the junction plane — and is
// refused by name. That rules out a three-dimensional hub such as
// +X/-X/+Y/-Y/+Z, which needs a different construction.
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
  // Every edge of the ring, the closing one included — that edge is the
  // difference between a rim and a run.
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
  // arm walks it is the reverse of fill order.
  const oriented = rims.map((rim) => {
    const edge = topology.edgeMap.get(edgeKey(rim[0], rim[1]));
    return edge.v0 === rim[0] ? [...rim].reverse() : [...rim];
  });

  const mids = oriented.map((rim) => scale(rim.reduce((acc, vi) => add(acc, cage.vertices[vi]), [0, 0, 0]), 1 / m));
  const centre = scale(mids.reduce((acc, p) => add(acc, p), [0, 0, 0]), 1 / N);
  const { nrm, order } = junctionPlaneOrder(mids, centre, 'bridgeClosedRimsHub', 'rim');

  // Per rim: where to cut it in two. `s` is each vertex's height above the
  // junction plane, measured from that rim's own center — the hub center
  // would bias a whole rim to one side and destroy the split.
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
    // Sum of s over the vertices strictly between p and q, walking forward.
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
  // if no arm contributes does the fallback along the normal apply.
  const ringScaleAll = arms.reduce((acc, a) => acc + a.rim.reduce((r, vi) => r + length(sub(cage.vertices[vi], mids[a.ri])), 0) / m, 0) / N;
  const interiorOf = (list) => list.flatMap((a) => a.slice(1, -1));
  const meanOf = (idxs) => scale(idxs.reduce((acc, vi) => add(acc, cage.vertices[vi]), [0, 0, 0]), 1 / idxs.length);
  const plusInterior = interiorOf(arms.map((a) => a.arcPlus));
  const minusInterior = interiorOf(arms.map((a) => a.arcMinus));
  const polePlus = plusInterior.length ? meanOf(plusInterior) : add(centre, scale(nrm, ringScaleAll));
  const poleMinus = minusInterior.length ? meanOf(minusInterior) : sub(centre, scale(nrm, ringScaleAll));

  // Which neighbor each arm's P vertex reaches across to. The derivation in
  // the header says next; both choices close a valid ring, so a winding
  // check cannot tell them apart, and the shorter total crotch decides. On a
  // symmetric junction the two totals differ widely — the near neighbor
  // against the far one — so this is not a tie.
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

// N-way face bridge — N >= 3 face groups, on one cage or on bodies merged
// into one, joined through a single junction: a Y, a T, an X. Each group is
// opened into a rim as bridgeFaces opens its two, each rim grows an arm of
// `segments` bands toward the junction, and the arms' inner ends are closed
// by the junction below.
//
// Ports. Each arm ends on a port ring: the rim's vertices carried `reach` of
// the way toward the junction center. That map is a similarity about the
// center, so N rims that do not overlap give N ports that do not overlap
// either, at 1 - reach of their size. FACE_HUB_REACH is 0.25, not the run
// hub's 0.62: on a Y of equal boxes the limit surface's neck against the rim
// is 0.77 at 0.2 and 0.42 at 0.62 — the second reads as three balloons tied
// at a knot. The junction center is the mean of the rim centers.
//
// Unequal rims. Every port carries the fine count M (the largest rim), so
// the junction sees equal rings and the reconciliation happens in one band,
// the one on a coarser rim, as in bridgeRims: port point j sits at
// parameter j * m / M in that rim's edge-index space, so its rim partner is
// floor(j * m / M) — the correspondence bridgeBandFaces derives, met by
// construction. A rim of the fine count gets its own vertices back
// verbatim. The M - m triangles in that band are spread round the ring by
// the floor.
//
// The junction is the shape a modeler builds by hand for a pipe Y: each
// port is cut into a contiguous chain above the junction plane and one
// below, the cut edges of neighboring arms are joined by one crotch quad
// each, and the upper chains, taken round the ring, bound one cap face, the
// lower chains another. No new vertex: V = NM, F = N + 2, and with the 2N
// crotch edges V - E + F = 2 - N, a sphere with N holes.
// bridgeClosedRimsHub's two poles are not used here because a port of four
// vertices has no equator vertex to split at, and its pole faces then span
// both hemispheres — a crease Catmull-Clark never smooths out.
//
// The cut is the contiguous split that puts the most height above and the
// least below; the ring order of the arms is junctionPlaneOrder's, run in
// the direction that lands each arm's upper→lower cut edge on its neighbor;
// the winding is derived from the port's array order (an arm band walks its
// far ring backward, so the caps walk it forward) and then checked by
// directedEdgeReuseCount.
//
// Spin, per arm, is a whole number of port steps: arm k's rim meets its port
// rotated by spins[k], twisting that arm alone. 0 everywhere is the
// untwisted result.
//
// Two junctions, chosen by the arms' spread. Arms round one plane — a Y, a
// T, an X, a star — take the ring junction above. Arms that leave that
// plane — the ±X/±Y/+Z corner, a tetrahedral four, a six-way cross — take
// the spatial junction: the convex hull of every port vertex with each
// port's facet removed, a sphere with N holes in triangles, wound by the
// arm bands. The choice is by the largest rise of one arm, seen from the
// junction center, out of the plane the other arms share (a plane fitted to
// all of them tilts with the raised arm and never sees it; three arms are
// always coplanar and take the ring). Worst dihedral after two
// Catmull-Clark steps, on an X of four boxes with one arm raised: the ring
// gives 37° at a 45° raise and folds to 87° at 60°; the hull holds 44°
// throughout, 35° on a tetrahedral four, 36° on a six-way cross, 49° on a
// flat Y where the ring gives 30° — so the ring up to HUB_ARM_TILT_MAX and
// the hull past it.
//
// Refused by name: fewer than three groups (two is bridgeFaces); a face in
// two groups; two rims sharing a vertex (they already touch); a group whose
// rim is not one simple loop (orderedBoundaryLoopOfFaceSet's refusal);
// group centers in a line (no junction plane); an arm whose port is not a
// facet of the hull (it ends inside the junction the others make). Two
// arms pointing nearly the same way are built and reported: `crossing`
// lists the pairs whose port rings come closer than their radii sum, since
// a larger reach pulls the ports apart and the result is undoable where a
// refusal is not.
const HUB_ARM_TILT_MAX = 42; // degrees an arm may rise, seen from the junction center, out of the plane the other arms share before the junction is built on the hull
const FACE_HUB_REACH = 0.25; // where the arms meet, as a fraction of the way from each rim to the junction center
export function bridgeFacesHub(cage, groups, opts = {}) {
  if (!Array.isArray(groups) || groups.some((g) => !Array.isArray(g) || !g.length)) throw new Error('bridgeFacesHub: groups must be an array of non-empty face-index arrays');
  if (groups.length < 3) throw new Error(`bridgeFacesHub: a hub needs at least 3 face groups (got ${groups.length}) — two groups is an ordinary tunnel, use bridgeFaces`);
  return bridgeOpenings(cage, groups.map((faces) => ({ faces })), opts);
}

// Any openings, any number: each is a group of faces (deleted first, as
// bridgeFaces does) or an already-open rim named by one of its edges
// (boundaryLoopFromSeed), on one body or on bodies merged into one. Two
// openings are the ordinary tunnel (bridgeRims); three or more are the
// junction below. `opts.spins[k]` / the returned `rims[k]` follow the
// openings' order.
export function bridgeOpenings(cage, openings, opts = {}) {
  const segments = opts.segments ?? 1, straightness = opts.straightness ?? 1, creaseWeight = opts.creaseWeight ?? 0;
  // Reach — how far along the way from each rim to the junction center the
  // arms meet; the ports are that fraction in and 1 - reach of their rim's
  // size, so a small reach is a wide junction close to the rims and a large
  // one is long necks meeting at a small core.
  const reach = opts.reach ?? FACE_HUB_REACH;
  if (!(reach >= 0.05 && reach <= 0.95)) throw new Error('bridgeOpenings: reach must be between 0.05 and 0.95 of the way to the junction centre');
  if (!Array.isArray(openings) || openings.some((o) => !o || (!(Array.isArray(o.faces) && o.faces.length) && typeof o.rim !== 'string'))) throw new Error('bridgeOpenings: each opening is a non-empty face list or the key of one edge on an open rim');
  const N = openings.length;
  if (N < 2) throw new Error(`bridgeOpenings: a bridge needs at least 2 openings (got ${N})`);
  if (!Number.isInteger(segments) || segments < 1) throw new Error('bridgeOpenings: segments must be a positive integer');
  const spins = openings.map((_, k) => (opts.spins && opts.spins[k]) || 0);
  if (spins.some((v) => !Number.isInteger(v))) throw new Error('bridgeOpenings: each spin must be a whole number of port steps');
  const seen = new Map();
  openings.forEach((o, k) => { for (const fi of o.faces || []) { if (seen.has(fi)) throw new Error(`bridgeOpenings: face ${fi} is in openings ${seen.get(fi) + 1} and ${k + 1} — the groups must be disjoint`); seen.set(fi, k); } });

  const loops0 = openings.map((o) => (o.faces ? orderedBoundaryLoopOfFaceSet(cage, o.faces) : boundaryLoopFromSeed(cage, o.rim)));
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    if (loops0[i].length === loops0[j].length && loops0[i].every((v) => loops0[j].includes(v))) throw new Error(`bridgeOpenings: openings ${i + 1} and ${j + 1} are the same rim`);
    const shared = loops0[i].find((vi) => loops0[j].includes(vi));
    if (shared !== undefined) throw new Error(`bridgeOpenings: openings ${i + 1} and ${j + 1} share vertex ${shared} — they already touch, with no real gap to bridge across`);
  }

  const deleted = openings.flatMap((o) => o.faces || []);
  const del = deleted.length ? deleteFaces(cage, deleted) : { cage: { vertices: cage.vertices.map((v) => v.slice()), faces: cage.faces.map((f) => [...f]), creases: { ...(cage.creases || {}) } }, vertexRemap: new Map(cage.vertices.map((_, i) => [i, i])) };
  if (N === 2) {
    const r = bridgeRims(cage, del.cage, del.vertexRemap, loops0[0], loops0[1], segments, 'bridgeOpenings', straightness, creaseWeight, spins[0] - spins[1]);
    return { ...r, bridgeFaceIndices: r.tunnelFaceIndices, hubFaceIndices: [], armCount: 2, ringLength: Math.max(loops0[0].length, loops0[1].length), crossing: [], spatial: false };
  }
  const work = del.cage;
  const topology = buildTopology(work);
  const loops = loops0.map((loop) => loop.map((vi) => del.vertexRemap.get(vi)));
  if (loops.some((loop) => loop.some((v) => v === undefined))) throw new Error('bridgeOpenings: internal error — a rim vertex did not survive face deletion');

  const M = Math.max(...loops.map((l) => l.length));
  const mids = loops.map((loop) => scale(loop.reduce((acc, vi) => add(acc, work.vertices[vi]), [0, 0, 0]), 1 / loop.length));
  const centre = scale(mids.reduce((acc, p) => add(acc, p), [0, 0, 0]), 1 / N);
  const { nrm, order } = junctionPlaneOrder(mids, centre, 'bridgeOpenings', 'opening');
  let spatial = false;
  if (N >= 4) mids.forEach((mid, k) => {
    const others = mids.filter((_, j) => j !== k);
    const oc = scale(others.reduce((acc, p) => add(acc, p), [0, 0, 0]), 1 / others.length);
    let on = [0, 0, 0], onLen = 0;
    for (let i = 0; i < others.length; i++) for (let j = i + 1; j < others.length; j++) {
      const c = cross(sub(others[i], oc), sub(others[j], oc)), cl = length(c);
      if (cl > onLen) { on = c; onLen = cl; }
    }
    if (!onLen) return;
    const tilt = Math.asin(Math.min(1, Math.abs(dot(norm3(sub(mid, centre)), norm3(on))))) * 180 / Math.PI;
    if (tilt > HUB_ARM_TILT_MAX) spatial = true;
  });
  if (opts.junction === 'ring') spatial = false; else if (opts.junction === 'hull') spatial = true;

  const newVertices = work.vertices.map((v) => v.slice());
  const push = (p) => newVertices.push(p) - 1;
  const added = [];
  const ports = [];
  loops.forEach((loop, k) => {
    const m = loop.length;
    const inset = loop.map((vi) => lerp3(work.vertices[vi], centre, reach));
    const portPos = [];
    for (let j = 0; j < M; j++) {
      const u = (j * m) / M, e = Math.floor(u);
      portPos.push(lerp3(inset[e], inset[(e + 1) % m], u - e));
    }
    const port = portPos.map((p) => push(p));
    const spin = ((spins[k] % M) + M) % M;
    const portSpun = port.map((_, j) => port[(j + spin) % M]);
    const c = rimCorrespondence(M, m);
    const rimPts = loop.map((vi) => work.vertices[vi]);
    const partner = new Array(m);
    for (let j = 0; j < M; j++) if (partner[c[j]] === undefined) partner[c[j]] = newVertices[portSpun[j]];
    const tan = rimTangents(work, topology, loop, partner, true, 1);
    const rings = [loop];
    for (let r = 1; r < segments; r++) {
      const t = r / segments;
      const ring = [];
      for (let j = 0; j < M; j++) {
        const a = rimPts[c[j]], b = newVertices[portSpun[j]];
        // The port end is straight: there is no surface there for a band to
        // leave along.
        ring.push(push(bridgeSpanPoint(a, b, tan[c[j]], sub(b, a), t, straightness)));
      }
      rings.push(ring);
    }
    rings.push(portSpun);
    for (let r = 0; r + 1 < rings.length; r++) for (const f of bridgeBandFaces(rings[r], rings[r + 1])) added.push(f);
    ports.push(port);
  });

  let junction, arms;
  if (!spatial) {
    // The cut: the contiguous run of the port with the greatest height above
    // the junction plane, measured from the port's own center, is the upper
    // chain; the rest is the lower. Both are non-empty.
    arms = ports.map((port, k) => {
      const pc = scale(port.reduce((acc, vi) => add(acc, newVertices[vi]), [0, 0, 0]), 1 / M);
      const s = port.map((vi) => dot(sub(newVertices[vi], pc), nrm));
      let best = null;
      for (let start = 0; start < M; start++) {
        let sum = 0;
        for (let len = 1; len < M; len++) {
          sum += s[(start + len - 1) % M];
          if (!best || sum > best.sum) best = { start, len, sum };
        }
      }
      const upper = [], lower = [];
      for (let j = 0; j < M; j++) (j < best.len ? upper : lower).push(port[(best.start + j) % M]);
      return { k, upper, lower, centre: pc };
    });

    // Ring order, in the direction that lands each arm's upper→lower cut edge
    // (the one that follows the upper chain in array order) on its neighbor.
    const cutMid = (a) => lerp3(newVertices[a.upper[a.upper.length - 1]], newVertices[a.lower[0]], 0.5);
    const at = (i) => arms[order[((i % N) + N) % N]];
    let vote = 0;
    for (let i = 0; i < N; i++) {
      const a = at(i), mid = cutMid(a);
      vote += distSq(mid, at(i + 1).centre) < distSq(mid, at(i - 1).centre) ? 1 : -1;
    }
    const ring = vote >= 0 ? order.map((k) => arms[k]) : order.map((k) => arms[k]).reverse();
    junction = [];
    for (let i = 0; i < N; i++) {
      const a = ring[i], b = ring[(i + 1) % N];
      junction.push([a.upper[a.upper.length - 1], a.lower[0], b.lower[b.lower.length - 1], b.upper[0]]);
    }
    junction.push(ring.flatMap((a) => a.upper));
    junction.push([...ring].reverse().flatMap((a) => a.lower));
  } else {
    // The spatial junction — the convex hull of every port vertex, with each
    // port's facet removed: a sphere with N holes for arms in any
    // directions, in triangles. Each port must be a facet of the hull, which
    // is what "the arms come from around the junction" means geometrically;
    // an arm whose port sinks inside the hull the others make is named.
    // A port is a planar polygon, and four coplanar corners are what an
    // incremental hull cannot triangulate cleanly. So each port also gets a
    // peak: its center pushed a hair outward from the junction center. The
    // facet becomes a strictly convex pyramid the hull triangulates without
    // ties; every triangle touching a peak belongs to one port and is
    // dropped, and the peaks never enter the cage.
    const pts = ports.flat();
    const P = pts.map((vi) => newVertices[vi]);
    let extent = 0; for (const q of P) extent = Math.max(extent, Math.abs(q[0]), Math.abs(q[1]), Math.abs(q[2]));
    const portOf = new Map(); ports.forEach((port, k) => port.forEach((vi) => portOf.set(vi, k)));
    const peakOf = [];
    ports.forEach((port, k) => {
      const c = scale(port.reduce((acc, vi) => add(acc, newVertices[vi]), [0, 0, 0]), 1 / M);
      const away = norm3(sub(c, centre));
      peakOf.push(P.length); P.push(add(c, scale(away, extent * 1e-4)));
    });
    const hull = convexHullFaces(P);
    const onHull = new Set();
    junction = [];
    for (const f of hull) {
      if (f.some((i) => i >= pts.length)) { f.forEach((i) => { if (i < pts.length) onHull.add(pts[i]); }); continue; }
      const vs = f.map((i) => pts[i]);
      vs.forEach((vi) => onHull.add(vi));
      if (new Set(vs.map((vi) => portOf.get(vi))).size > 1) junction.push(vs);
    }
    for (let k = 0; k < N; k++) if (ports[k].some((vi) => !onHull.has(vi))) throw new Error(`bridgeOpenings: opening ${k + 1}'s arm ends inside the junction the others make — move it outward or raise Reach`);
    // A peak that reached an inter-port triangle would mean a port facet is
    // not a face of the hull — the same case, caught the same way.
    for (const f of hull) if (f.some((i) => i >= pts.length) && new Set(f.map((i) => (i >= pts.length ? peakOf.indexOf(i) : portOf.get(pts[i])))).size > 1) throw new Error(`bridgeOpenings: opening ${peakOf.indexOf(f.find((i) => i >= pts.length)) + 1}'s arm ends inside the junction the others make — move it outward or raise Reach`);
    arms = ports.map((port, k) => ({ k, centre: scale(port.reduce((acc, vi) => add(acc, newVertices[vi]), [0, 0, 0]), 1 / M) }));
    // The hull is wound one way throughout; the arm bands fix which.
    const base = directedEdgeReuseCount(work.faces);
    if (directedEdgeReuseCount([...work.faces.map((f) => [...f]), ...added, ...junction]) !== base) junction = junction.map((f) => [...f].reverse());
  }
  const faces = [...work.faces.map((f) => [...f]), ...added, ...junction];
  if (directedEdgeReuseCount(faces) !== directedEdgeReuseCount(work.faces)) {
    throw new Error('bridgeOpenings: the junction is not consistently wound — the openings may not all be separate openings of one consistently-oriented cage');
  }
  const crossing = [];
  const portRadius = (k) => ports[k].reduce((acc, vi) => acc + Math.sqrt(distSq(newVertices[vi], arms[k].centre)), 0) / M;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    if (Math.sqrt(distSq(arms[i].centre, arms[j].centre)) < portRadius(i) + portRadius(j)) crossing.push([i, j]);
  }
  const outCreases = { ...(work.creases || {}) };
  for (const loop of loops) creaseChain(outCreases, loop, creaseWeight, true);
  const base = work.faces.length;
  return {
    cage: { vertices: newVertices, faces, creases: outCreases },
    bridgeFaceIndices: added.map((_, i) => base + i),
    hubFaceIndices: junction.map((_, i) => base + added.length + i),
    armCount: N,
    ringLength: M,
    rims: loops,
    crossing,
    spatial,
  };
}

// The convex hull of a point set, as outward-wound triangles of point indices
// — incremental: a starting tetrahedron, then each further point removes the
// faces that see it and closes the horizon with new ones. A few dozen points
// at most here, so the plain O(n²) form is the right one.
export function convexHullFaces(P) {
  const n = P.length;
  if (n < 4) throw new Error('convexHullFaces: at least four points');
  let scaleRef = 0;
  for (const p of P) scaleRef = Math.max(scaleRef, Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2]));
  const eps = Math.max(1e-12, scaleRef * 1e-9);
  const faceNormal = (a, b, c) => cross(sub(P[b], P[a]), sub(P[c], P[a]));
  // A starting tetrahedron of four points not in one plane — each the point
  // that spans the most, so four coplanar corners of one port can never be it.
  const i0 = 0;
  let i1 = -1, i2 = -1, i3 = -1, best = 0;
  for (let i = 1; i < n; i++) { const d = length(sub(P[i], P[i0])); if (d > best) { best = d; i1 = i; } }
  if (best <= eps) throw new Error('convexHullFaces: the points coincide');
  best = 0;
  for (let i = 1; i < n; i++) { if (i === i1) continue; const a = length(faceNormal(i0, i1, i)); if (a > best) { best = a; i2 = i; } }
  if (i2 < 0 || best <= eps * eps) throw new Error('convexHullFaces: the points lie on one line');
  const n012 = norm3(faceNormal(i0, i1, i2));
  best = 0;
  for (let i = 1; i < n; i++) { if (i === i1 || i === i2) continue; const h = Math.abs(dot(n012, sub(P[i], P[i0]))); if (h > best) { best = h; i3 = i; } }
  if (i3 < 0 || best <= eps) throw new Error('convexHullFaces: the points lie in one plane');
  let faces = [[i0, i1, i2], [i0, i2, i3], [i0, i3, i1], [i1, i3, i2]];
  const centroid = scale(add(add(P[i0], P[i1]), add(P[i2], P[i3])), 0.25);
  faces = faces.map((f) => (dot(faceNormal(f[0], f[1], f[2]), sub(P[f[0]], centroid)) < 0 ? [f[0], f[2], f[1]] : f));
  const used = new Set([i0, i1, i2, i3]);
  for (let p = 0; p < n; p++) {
    if (used.has(p)) continue;
    const visible = faces.filter((f) => dot(norm3(faceNormal(f[0], f[1], f[2])), sub(P[p], P[f[0]])) > eps);
    if (!visible.length) continue;
    const edgeCount = new Map();
    for (const f of visible) for (let k = 0; k < 3; k++) { const a = f[k], b = f[(k + 1) % 3]; edgeCount.set(`${a}>${b}`, (edgeCount.get(`${a}>${b}`) || 0) + 1); }
    const horizon = [];
    for (const f of visible) for (let k = 0; k < 3; k++) { const a = f[k], b = f[(k + 1) % 3]; if (!edgeCount.has(`${b}>${a}`)) horizon.push([a, b]); }
    faces = faces.filter((f) => !visible.includes(f));
    for (const [a, b] of horizon) faces.push([a, b, p]);
  }
  return faces;
}
