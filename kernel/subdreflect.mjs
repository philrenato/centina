// SuperB CAGE MIRROR SYMMETRY — REFLECT ("Rhino's live-symmetry
// command, and the one SubD feature that must not slip: set a mirror plane,
// edits on either side propagate live, RemoveExistingReflectSymmetry to
// bake"). KERNEL ONLY: pure cage/plane-in, correspondence-or-position-out
// math, no app-layer object/UI/undo here, matching every other
// kernel/*.mjs module's own discipline exactly. Reuses subd.mjs's own
// `buildTopology`/`edgeKey` directly rather than re-deriving cage adjacency
// a second time (this project's own standing rule).
//
// THE CORE PROBLEM, stated plainly before any code: Reflect needs a
// VERTEX CORRESPONDENCE — for every vertex on one side of the plane, which
// vertex (if any) is its mirror partner on the other side — so that an
// edit to one can be replayed, mirrored, onto the other. The natural, v1-
// honest approach (the explicit instruction) is
// NEAREST-VERTEX-AFTER-REFLECTING-THROUGH-THE-PLANE, WITH A TOLERANCE: for
// each vertex, reflect its position across the plane and find the (unique)
// real cage vertex sitting there already. This only works if the cage is
// ALREADY symmetric about the chosen plane at SET-time — a real, honest v1
// scope cut (matching Rhino's own actual usage pattern: you build a
// symmetric cage — or one whose two halves already match, e.g. any of this
// app's own SuperBBox/Sphere/Cylinder/Plane primitives about their own
// center planes — THEN turn Reflect on to lock and maintain that symmetry
// through further edits). A vertex with NO clean partner within tolerance
// (or an AMBIGUOUS one — two candidates equally close) is refused
// honestly, by throwing with the offending vertex named, rather than
// silently guessing wrong or dropping it from the correspondence.
//
// A vertex whose own reflection lands back on ITSELF (within tolerance) is
// classified as ON-PLANE (partner[i] === i) — a real, distinct case, not
// an error: it is the cage's own "seam," and the app layer's edit-mirroring
// logic deliberately never double-applies a delta to a self-paired vertex
// (it already received the user's own edit directly; there is no separate
// "other side" copy of it to also move).

import { add, sub, scale, dot, length, normalize } from './vec3.mjs';
import { buildTopology, edgeKey } from './subd.mjs';

// Reflects a single point across the plane through `planeOrigin` with unit
// (or any nonzero — normalized here) normal `planeNormal`. The standard
// `p - 2*((p-origin).n)*n` construction — the same formula this app's own
// already-shipped Mirror command already uses for a whole-object copy (see
// the app's own finishMirror), just generalized here to an
// arbitrary plane (Mirror's own plane always passes through a picked point
// with a picked-line-derived normal; Reflect's plane is stored and reused
// call after call, so this is factored out as its own real, testable
// primitive rather than re-inlined every call site).
export function reflectPoint(point, planeOrigin, planeNormal) {
  const n = normalize(planeNormal);
  const d = dot(sub(point, planeOrigin), n);
  return sub(point, scale(n, 2 * d));
}

// Reflects a DIRECTION/DELTA vector across a plane's own normal — no
// origin needed at all (a vector has no position, only the plane's
// ORIENTATION matters for how it reflects, unlike a point). This is the
// one the LIVE PROPAGATION actually calls on every
// mirrored push-pull edit: the user's raw drag delta gets reflected by
// this function and applied to the partner vertex, rather than reflecting
// two absolute positions and subtracting (equivalent, but this is the
// direct, minimal computation the app layer actually needs).
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

// THE REAL, TESTABLE PRIMITIVE Reflect needed beyond plain point/
// vector reflection: given a cage and a plane, find every vertex's own
// mirror partner (or itself, if it sits on the plane) — see this file's
// own header for the full derivation. Throws honestly (naming the
// offending vertex, its real position, and — for the ambiguous case — its
// candidate partners) rather than silently guessing wrong, matching this
// project's "refuse rather than guess" standard (computeAverageNormal's
// own canceled-normal refusal, extrudeFaces' own degenerate-direction
// refusal). `opts.tolerance` defaults to a scale-relative epsilon (the
// cage's own bounding-box diagonal * 1e-4, floored at 1e-6) rather than a
// fixed absolute number, so this behaves consistently whether the cage is
// millimeter- or meter-scale.
export function findCageMirrorPartners(cage, planeOrigin, planeNormal, opts = {}) {
  const n = normalize(planeNormal);
  const diag = bboxDiagonal(cage.vertices);
  const tol = opts.tolerance ?? Math.max(diag * 1e-4, 1e-6);
  const nv = cage.vertices.length;
  const partner = new Array(nv).fill(-1);
  const reflected = cage.vertices.map((v) => reflectPoint(v, planeOrigin, n));

  // TIE EPSILON — a real bug a review caught, fixed here: the ORIGINAL
  // version checked "does vertex i's OWN reflection land back on itself
  // within tolerance" FIRST, before ever searching for a real distinct
  // partner elsewhere in the cage. That meant two genuinely different,
  // correctly-mirror-paired vertices that both happened to sit close to
  // the plane (each within tolerance/2 of it) would EACH independently
  // satisfy that self-check and get silently self-paired ("on-plane"),
  // even though an exact, far-closer real partner existed a few
  // thousandths of a unit away — and the "mutual bijection" safety net
  // below is structurally blind to this failure (partner[i]=i trivially
  // satisfies partner[partner[i]]===i). The fix: for every vertex, find
  // the GLOBALLY NEAREST real cage vertex to its reflected position
  // (itself included, as just one more candidate, not a privileged first
  // check) — self-pairing only wins when self genuinely IS the nearest.
  // `tieEps` (tiny relative to `tol`) is what distinguishes a genuine
  // ambiguity (two candidates truly equally close) from one candidate
  // merely being in the neighborhood of another that's actually closer.
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
    partner[i] = candidates[0]; // may legitimately equal i — ON-PLANE, only when self really is the nearest
  }
  // MUTUAL BIJECTION CHECK — a real, cheap defensive test, not assumed safe:
  // the nearest-within-tolerance search above finds each vertex's own
  // best candidate independently, so verify directly that the pairing is a
  // genuine involution (partner[partner[i]] === i for every i) rather than
  // silently trusting it — matching subdivideCatmullClark's own "verify,
  // don't assume" precedent for a structural claim like this.
  for (let i = 0; i < nv; i++) {
    if (partner[partner[i]] !== i) {
      throw new Error(`findCageMirrorPartners: vertex ${i}'s own partner ${partner[i]} does not pair back to it (partner[${partner[i]}]=${partner[partner[i]]}) — not a clean mirror involution`);
    }
  }
  const onPlaneCount = partner.filter((p, i) => p === i).length;
  return { partner, onPlaneCount, tolerance: tol };
}

// Given an already-computed `partner` correspondence (findCageMirrorPartners'
// own return), finds the cage FACE on the "other side" that a selected
// face's own vertex loop maps onto — needed so a face-based edit
// (EXTRUDESUBD) can find its own mirrored face selection, not just a
// mirrored vertex. Matched as a SET (not a cyclic sequence) since a
// reflection flips winding/handedness — two faces that are true mirror
// images of each other do NOT share vertex order, only vertex identity
// once mapped through `partner`. Returns null (an honest "no counterpart"),
// never a guess, for a face with no clean match (e.g. one whose own
// mapped vertex set collapses — two of its vertices sharing one partner —
// or one that simply has no real counterpart face in the cage at all).
export function mirrorFaceIndex(cage, partner, faceIdx) {
  const face = cage.faces[faceIdx];
  const mappedSet = new Set(face.map((vi) => partner[vi]));
  if (mappedSet.size !== face.length) return null; // degenerate mapping — two of this face's own vertices share one partner
  for (let fi = 0; fi < cage.faces.length; fi++) {
    const f = cage.faces[fi];
    if (f.length !== face.length) continue;
    if (f.every((vi) => mappedSet.has(vi))) return fi; // same length + full coverage of a same-size set IS set equality
  }
  return null;
}

// Same idea as mirrorFaceIndex, for an EDGE — given a real cage edge key
// (subd.mjs's own edgeKey format), maps both endpoints through `partner`
// and confirms the mapped key is a REAL edge of this cage (never just
// returns a synthesized key that might not exist) — needed so a Crease/
// SoftCrease edit can find its own mirrored edge. Returns null for an
// honest "no real counterpart edge" (e.g. an edge with one on-plane and
// one off-plane endpoint, whose mapped pair doesn't correspond to any
// actual edge).
export function mirrorEdgeKey(cage, partner, key) {
  const [a, b] = key.split('_').map(Number);
  if (!Number.isInteger(a) || !Number.isInteger(b) || partner[a] === undefined || partner[b] === undefined) return null;
  const topology = buildTopology(cage);
  if (!topology.edgeMap.has(edgeKey(a, b))) return null; // the INPUT itself isn't a real edge of this cage — nothing honest to map
  const newKey = edgeKey(partner[a], partner[b]);
  return topology.edgeMap.has(newKey) ? newKey : null;
}
