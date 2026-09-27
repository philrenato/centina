// SuperB cage mirror symmetry — Reflect (after Rhino's live-symmetry
// command: set a mirror plane, edits on either side propagate live,
// RemoveExistingReflectSymmetry to bake). Cage/plane in, correspondence or
// position out; no app-layer object, UI or undo here. Reuses subd.mjs's
// `buildTopology`/`edgeKey` rather than re-deriving cage adjacency.
//
// Reflect needs a vertex correspondence — for every vertex on one side of
// the plane, which vertex (if any) is its mirror partner on the other side —
// so that an edit to one can be replayed, mirrored, onto the other. The
// approach is nearest vertex after reflecting through the plane, with a
// tolerance: for each vertex, reflect its position across the plane and
// find the unique cage vertex already sitting there. This only works if the
// cage is already symmetric about the chosen plane when Reflect is set,
// which matches how Rhino's command is used: build a symmetric cage (or one
// whose halves already match, e.g. any SuperBBox/Sphere/Cylinder/Plane
// primitive about its own center planes), then turn Reflect on to maintain
// that symmetry through further edits. A vertex with no partner within
// tolerance, or an ambiguous one (two candidates equally close), is refused
// by throwing with the offending vertex named, rather than guessed or
// dropped from the correspondence.
//
// A vertex whose own reflection lands back on itself (within tolerance) is
// classified as on-plane (partner[i] === i) — a distinct case, not an
// error: it is the cage's own seam, and the app layer's edit mirroring never
// double-applies a delta to a self-paired vertex (it already received the
// user's edit directly; there is no other-side copy of it to move).

import { sub, scale, dot, length, normalize } from './vec3.mjs';
import { buildTopology, edgeKey } from './subd.mjs';

// Reflects a single point across the plane through `planeOrigin` with unit
// (or any nonzero — normalized here) normal `planeNormal`. The standard
// `p - 2*((p-origin).n)*n` construction — the same formula the app's Mirror
// command uses for a whole-object copy (finishMirror), generalized to an
// arbitrary plane (Mirror's plane always passes through a picked point with
// a picked-line-derived normal; Reflect's plane is stored and reused call
// after call).
export function reflectPoint(point, planeOrigin, planeNormal) {
  const n = normalize(planeNormal);
  const d = dot(sub(point, planeOrigin), n);
  return sub(point, scale(n, 2 * d));
}

// Reflects a direction/delta vector across a plane's normal — no origin
// needed (a vector has no position; only the plane's orientation matters).
// Live propagation calls this on every mirrored push-pull edit: the user's
// drag delta is reflected and applied to the partner vertex, rather than
// reflecting two absolute positions and subtracting (equivalent, but this is
// the direct computation).
export function reflectVector(vector, planeNormal) {
  const n = normalize(planeNormal);
  const d = dot(vector, n);
  return sub(vector, scale(n, 2 * d));
}

function bboxDiagonal(vertices) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const v of vertices) {
    for (let k = 0; k < 3; k++) {
      if (v[k] < min[k]) min[k] = v[k];
      if (v[k] > max[k]) max[k] = v[k];
    }
  }
  return length(sub(max, min));
}

// Given a cage and a plane, finds every vertex's mirror partner (or itself,
// if it sits on the plane) — see this file's header. Throws, naming the
// offending vertex, its position, and — for the ambiguous case — its
// candidate partners, rather than guessing (as computeAverageNormal refuses
// a canceled normal and extrudeFaces a degenerate direction).
// `opts.tolerance` defaults to a scale-relative epsilon (the cage's
// bounding-box diagonal * 1e-4, floored at 1e-6) rather than a fixed
// absolute number, so this behaves consistently whether the cage is
// millimeter- or meter-scale.
export function findCageMirrorPartners(cage, planeOrigin, planeNormal, opts = {}) {
  const n = normalize(planeNormal);
  const diag = bboxDiagonal(cage.vertices);
  const tol = opts.tolerance ?? Math.max(diag * 1e-4, 1e-6);
  const nv = cage.vertices.length;
  const partner = new Array(nv).fill(-1);
  const reflected = cage.vertices.map((v) => reflectPoint(v, planeOrigin, n));

  // Tie epsilon. Checking "does vertex i's own reflection land back on
  // itself within tolerance" before searching for a distinct partner would
  // let two correctly mirror-paired vertices that both sit close to the
  // plane (each within tolerance/2 of it) each satisfy that self-check and
  // be self-paired ("on-plane"), even though an exact, far closer partner
  // exists a few thousandths of a unit away — and the mutual-bijection check
  // below cannot see that failure (partner[i]=i trivially satisfies
  // partner[partner[i]]===i). So for every vertex this finds the globally
  // nearest cage vertex to its reflected position, itself included as one
  // more candidate rather than a privileged first check; self-pairing wins
  // only when self is the nearest. `tieEps` (tiny relative to `tol`)
  // distinguishes a true ambiguity (two candidates equally close) from one
  // candidate merely being near another that is closer.
  const tieEps = Math.max(tol * 1e-6, 1e-12);

  for (let i = 0; i < nv; i++) {
    const r = reflected[i];
    let bestDist = Infinity;
    for (let j = 0; j < nv; j++) {
      const d = length(sub(cage.vertices[j], r));
      if (d < bestDist) bestDist = d;
    }
    if (bestDist > tol) {
      throw new Error(`findCageMirrorPartners: vertex ${i} at [${cage.vertices[i].map((c) => c.toFixed(3)).join(', ')}] has no mirror partner within tolerance ${tol.toFixed(6)} — its nearest candidate is ${bestDist.toFixed(6)} away, so the cage would have to close a ${(bestDist - tol).toFixed(6)} discrepancy — this cage is not symmetric about the given plane (Reflect requires an already-symmetric cage at set time)`);
    }
    const candidates = [];
    for (let j = 0; j < nv; j++) {
      if (length(sub(cage.vertices[j], r)) <= bestDist + tieEps) candidates.push(j);
    }
    if (candidates.length > 1) {
      throw new Error(`findCageMirrorPartners: vertex ${i} has ${candidates.length} ambiguous candidate mirror partners within tolerance ${tol.toFixed(6)} (vertex indices ${JSON.stringify(candidates)}) — too coarse/degenerate relative to this tolerance for a clean pairing`);
    }
    partner[i] = candidates[0]; // may equal i — on-plane, only when self is the nearest
  }
  // Mutual bijection check: the nearest-within-tolerance search above finds
  // each vertex's best candidate independently, so verify that the pairing
  // is an involution (partner[partner[i]] === i for every i) rather than
  // trusting it.
  for (let i = 0; i < nv; i++) {
    if (partner[partner[i]] !== i) {
      throw new Error(`findCageMirrorPartners: vertex ${i}'s own partner ${partner[i]} does not pair back to it (partner[${partner[i]}]=${partner[partner[i]]}) — not a clean mirror involution`);
    }
  }
  const onPlaneCount = partner.filter((p, i) => p === i).length;
  return { partner, onPlaneCount, tolerance: tol };
}

// Given a `partner` correspondence (findCageMirrorPartners' return), finds
// the cage face on the other side that a selected face's vertex loop maps
// onto, so a face-based edit (ExtrudeSubD) can find its mirrored face
// selection, not just a mirrored vertex. Matched as a set (not a cyclic
// sequence) since a reflection flips winding — two faces that are mirror
// images of each other do not share vertex order, only vertex identity once
// mapped through `partner`. Returns null ("no counterpart"), never a guess,
// for a face with no clean match (e.g. one whose mapped vertex set collapses
// — two of its vertices sharing one partner — or one with no counterpart
// face in the cage).
export function mirrorFaceIndex(cage, partner, faceIdx) {
  const face = cage.faces[faceIdx];
  const mappedSet = new Set(face.map((vi) => partner[vi]));
  if (mappedSet.size !== face.length) return null; // degenerate mapping — two of this face's own vertices share one partner
  for (let fi = 0; fi < cage.faces.length; fi++) {
    const f = cage.faces[fi];
    if (f.length !== face.length) continue;
    if (f.every((vi) => mappedSet.has(vi))) return fi; // same length + full coverage of a same-size set is set equality
  }
  return null;
}

// Same idea as mirrorFaceIndex, for an edge — given a cage edge key
// (subd.mjs's edgeKey format), maps both endpoints through `partner` and
// confirms the mapped key is an edge of this cage (never returns a
// synthesized key that might not exist), so a Crease/SoftCrease edit can
// find its mirrored edge. Returns null when there is no counterpart edge
// (e.g. an edge with one on-plane and one off-plane endpoint, whose mapped
// pair corresponds to no edge).
export function mirrorEdgeKey(cage, partner, key) {
  const [a, b] = key.split('_').map(Number);
  if (!Number.isInteger(a) || !Number.isInteger(b) || partner[a] === undefined || partner[b] === undefined) return null;
  const topology = buildTopology(cage);
  if (!topology.edgeMap.has(edgeKey(a, b))) return null; // the input itself is not an edge of this cage
  const newKey = edgeKey(partner[a], partner[b]);
  return topology.edgeMap.has(newKey) ? newKey : null;
}
