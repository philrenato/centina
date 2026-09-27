// SuperB cage radial symmetry — the cyclic-group sibling of
// subdreflect.mjs. Cage/axis in, correspondence out; no app-layer object, UI
// or undo here.
//
// Reflect needs a vertex correspondence so an edit on one side can be
// replayed, mirrored, on the other. This needs the same thing for a
// rotation: for every vertex, which vertex sits where this one lands after
// turning `angle` about the axis. The technique is the same —
// nearest vertex after transforming, with a scale-relative tolerance,
// refusing rather than guessing — so this module follows
// findCageMirrorPartners line for line, and uses `rotatePoint` from
// transform.mjs for the rotation.
//
// Two things differ:
//
// 1. A mirror is an involution; a rotation is a cycle. Reflect asserts
//    partner[partner[i]] === i. That check is meaningless here: a rotation
//    has a direction, so the correspondence is a permutation `next`
//    (vertex -> the vertex one step around), and the structural check is
//    that applying `next` exactly `order` times returns every vertex to
//    itself while no smaller number of steps does. That is a strictly
//    stronger claim than the involution check, and it is what shows the cage
//    has the symmetry it was asked to have.
//
// 2. A mirror has on-plane points; a rotation has on-axis points. Reflect's
//    self-paired vertices lie on the plane. Here they lie on the axis, and
//    they differ: an on-plane vertex is fixed by the one reflection, whereas
//    an on-axis vertex is fixed by every rotation at once. So an on-axis
//    vertex forms an orbit of size 1, and an order check that asked "does
//    `next` return everything after k steps" would conclude the order is 1
//    the moment any on-axis vertex exists. They are excluded from the order
//    check explicitly. A SuperB torus has none; a SuperB cone or sphere has
//    them (its apex/poles).
//
// The tie-epsilon rule is the same as the mirror module's: checking "does
// this vertex map back onto itself" first, as a privileged case, would let a
// vertex self-pair even when a closer partner exists. The radial case is a
// vertex sitting near the axis: it maps close to itself, and must not claim
// self-pairing merely for being near. So the search finds the globally
// nearest candidate with self included as one more candidate, never checked
// first.
import { sub, length, normalize } from './vec3.mjs';
import { rotatePoint } from './transform.mjs';

function bboxDiagonal(vertices) {
  if (!vertices.length) return 0;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const v of vertices) for (let c = 0; c < 3; c++) { if (v[c] < lo[c]) lo[c] = v[c]; if (v[c] > hi[c]) hi[c] = v[c]; }
  return length(sub(hi, lo));
}

// gcd, and the derived orbit order. This is the arithmetic behind the one
// surprising behavior of the feature: "every 3rd of 12" is a 4-member orbit,
// but "every 3rd of 10" wraps and hits all ten. Exported so the app layer
// and the tests read the same number rather than each deriving it, and so a
// status line can state it before it surprises the user.
export function gcd(a, b) { a = Math.abs(a); b = Math.abs(b); while (b) { const t = a % b; a = b; b = t; } return a; }
export function orbitOrder(ringCount, stride) {
  const N = Math.round(ringCount), s = Math.round(stride);
  if (!(N > 0) || !Number.isFinite(s) || s === 0) throw new Error('orbitOrder: ringCount must be positive and stride nonzero');
  return N / gcd(N, s);
}
// The divisors of N, for a refusal message that tells the user which orbit
// sizes their cage can support instead of only saying no.
export function divisorsOf(n) {
  const out = [];
  for (let d = 1; d <= n; d++) if (n % d === 0) out.push(d);
  return out;
}

// The other direction: given a desired member count, the stride that
// produces it. This makes Count and Stride two live, two-way views of one
// relationship rather than one input and one readout.
//
// They are not exact inverses, and that is correct: orbitOrder(12, 5) is 12,
// but strideForCount(12, 12) returns 1, not 5. Stride 1 and stride 5 on a
// 12-ring select the identical member set at identical angular positions.
// Stride only chooses which members are in the orbit; once that set is
// fixed, each member's rotation comes from its own angular offset, never
// from its slot label. So returning the canonical (smallest) stride for a
// count is indistinguishable downstream from returning any other stride that
// yields the same count, which is what makes a two-way pair of editable rows
// safe.
//
// Refuses a count the ring cannot make, naming the ones it can.
export function strideForCount(ringCount, count) {
  const N = Math.round(ringCount), c = Math.round(count);
  if (!(N > 0)) throw new Error('strideForCount: ringCount must be positive');
  if (!(c >= 2)) throw new Error('strideForCount: an orbit needs at least 2 members');
  if (N % c !== 0) {
    throw new Error(`strideForCount: ${N} positions around the axis cannot make a ${c}-member orbit — the sizes this ring supports are ${divisorsOf(N).filter((d) => d >= 2).join(', ')}`);
  }
  return N / c;
}

// Returns `next` (vertex -> vertex one rotation step around), the on-axis
// vertex count, and the tolerance used. Throws — never guesses — when the
// cage is not symmetric about this axis at this angle, or when a vertex's
// partner is ambiguous.
export function findCageRotationPartners(cage, axisOrigin, axisDir, angleRad, opts = {}) {
  const axis = normalize(axisDir);
  if (!Number.isFinite(angleRad) || Math.abs(angleRad) < 1e-12) {
    throw new Error('findCageRotationPartners: angle must be a nonzero finite number — a zero rotation pairs every vertex with itself, which is not a symmetry relationship');
  }
  const diag = bboxDiagonal(cage.vertices);
  const tol = opts.tolerance ?? Math.max(diag * 1e-4, 1e-6);
  const nv = cage.vertices.length;
  const next = new Array(nv).fill(-1);
  const rotated = cage.vertices.map((v) => rotatePoint(v, axisOrigin, axis, angleRad));

  // See the module header: self is one candidate among many, never a
  // privileged first check. tieEps distinguishes a true ambiguity (two
  // candidates equally close) from one candidate merely being in the
  // neighborhood of another that is closer.
  const tieEps = Math.max(tol * 1e-6, 1e-12);

  for (let i = 0; i < nv; i++) {
    const r = rotated[i];
    let bestDist = Infinity;
    for (let j = 0; j < nv; j++) {
      const d = length(sub(cage.vertices[j], r));
      if (d < bestDist) bestDist = d;
    }
    if (bestDist > tol) {
      throw new Error(`findCageRotationPartners: vertex ${i} at [${cage.vertices[i].map((c) => c.toFixed(3)).join(', ')}] has no rotational partner within tolerance ${tol.toFixed(6)} — its nearest candidate is ${bestDist.toFixed(6)} away, so the cage would have to close a ${(bestDist - tol).toFixed(6)} discrepancy — this cage is not rotationally symmetric about the given axis at this angle`);
    }
    const candidates = [];
    for (let j = 0; j < nv; j++) {
      if (length(sub(cage.vertices[j], r)) <= bestDist + tieEps) candidates.push(j);
    }
    if (candidates.length > 1) {
      throw new Error(`findCageRotationPartners: vertex ${i} has ${candidates.length} ambiguous candidate rotational partners within tolerance ${tol.toFixed(6)} (vertex indices ${JSON.stringify(candidates)}) — too coarse/degenerate relative to this tolerance for a clean correspondence`);
    }
    next[i] = candidates[0]; // may equal i — on-axis, only when self is the nearest
  }

  // Permutation check. The nearest-search resolves each vertex
  // independently, so verify that the result is a bijection rather than
  // trusting it — two vertices mapping onto the same target would otherwise
  // be a silent corruption.
  const hit = new Array(nv).fill(-1);
  for (let i = 0; i < nv; i++) {
    if (hit[next[i]] !== -1) {
      throw new Error(`findCageRotationPartners: vertices ${hit[next[i]]} and ${i} both map onto vertex ${next[i]} — not a clean permutation, so this is not a symmetry of the cage`);
    }
    hit[next[i]] = i;
  }
  const onAxisCount = next.filter((p, i) => p === i).length;
  return { next, onAxisCount, tolerance: tol };
}

// Verifies that `next` has the claimed order — the cyclic analog of the
// mirror module's involution check, and a strictly stronger statement:
// applying it exactly `order` times must return every vertex to itself, and
// no smaller positive number of steps may do so for any vertex that is not
// on the axis.
//
// On-axis vertices are excluded explicitly — they are fixed by every
// rotation, so they satisfy "returns to itself" at every step count and
// would otherwise drag the measured order down to 1. See the module header.
export function verifyRotationOrder(next, order) {
  const n = Math.round(order);
  if (!(n >= 2)) throw new Error('verifyRotationOrder: order must be at least 2');
  const nv = next.length;
  const onAxis = new Set();
  for (let i = 0; i < nv; i++) if (next[i] === i) onAxis.add(i);

  for (let i = 0; i < nv; i++) {
    if (onAxis.has(i)) continue;
    let cur = i;
    for (let step = 1; step <= n; step++) {
      cur = next[cur];
      if (cur === i && step < n) {
        throw new Error(`verifyRotationOrder: vertex ${i} returns to itself after ${step} steps, not ${n} — the real orbit order here is ${step}, so the requested order is wrong for this cage`);
      }
    }
    if (cur !== i) {
      throw new Error(`verifyRotationOrder: vertex ${i} does not return to itself after ${n} steps (landed on ${cur}) — the correspondence does not close into an orbit of that order`);
    }
  }
  return { order: n, onAxisCount: onAxis.size };
}

// Walks the chain from one seed element and returns its ordered orbit.
// Position in the returned array is the group element: slot k sits k
// rotation steps around from the seed. Refuses if the chain does not close
// in exactly `order` steps, rather than returning a partial ring.
export function orbitFromSeed(next, seed, order) {
  const n = Math.round(order);
  if (!(n >= 2)) throw new Error('orbitFromSeed: order must be at least 2');
  if (!(seed >= 0 && seed < next.length)) throw new Error(`orbitFromSeed: seed ${seed} is not a vertex of this cage`);
  if (next[seed] === seed) {
    throw new Error(`orbitFromSeed: vertex ${seed} lies ON the rotation axis, so it is fixed by every rotation and has no orbit of size ${n} — pick an element off the axis`);
  }
  const slots = [seed];
  let cur = seed;
  for (let step = 1; step < n; step++) {
    cur = next[cur];
    if (cur === seed) throw new Error(`orbitFromSeed: the chain from vertex ${seed} closed after ${step} steps, not ${n}`);
    slots.push(cur);
  }
  if (next[cur] !== seed) throw new Error(`orbitFromSeed: the chain from vertex ${seed} did not close back after ${n} steps`);
  return slots;
}

// The rotation applied to a delta (a free vector, not a point) — what an
// orbit member `steps` positions around receives when the master is
// dragged. This is the conjugation rule: for a pure translation,
// R_k . T . R_k^-1 collapses to "rotate the delta by k steps." Expressed as
// a rotation about the origin because a delta has no position of its own.
export function rotateDelta(delta, axisDir, angleRad, steps = 1) {
  return rotatePoint(delta, [0, 0, 0], normalize(axisDir), angleRad * steps);
}

// The rotational counterpart of mirrorFaceIndex. Matched as a vertex set,
// the mirror module's approach — a rotation preserves winding (unlike a
// reflection), so a cyclic match would also work, but set-matching carries
// the degenerate-collapse guard. Returns null ("no counterpart"), never a
// guess.
export function rotationFaceIndex(cage, next, faceIdx) {
  const face = cage.faces[faceIdx];
  const mappedSet = new Set(face.map((vi) => next[vi]));
  if (mappedSet.size !== face.length) return null; // the mapping collapsed — no clean counterpart
  for (let f = 0; f < cage.faces.length; f++) {
    const other = cage.faces[f];
    if (other.length !== face.length) continue;
    let all = true;
    for (const vi of other) if (!mappedSet.has(vi)) { all = false; break; }
    if (all) return f;
  }
  return null;
}
