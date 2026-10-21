// Self-collision hold for a displaced control net.
//
// A deformer moves control points from a base net to a deformed net. Where the
// deformed surface passes through itself, the hold pulls each responsible
// control point back toward the base, by as little as it can: the displacement
// is scaled per control point, never as one global amplitude, so the rest of
// the effect stands at full strength.
//
// The surface is sampled on a grid in (u,v), welded where samples coincide on
// the base (closed seams, poles), and triangulated. Two triangles collide when
// an edge of one crosses the other and they share no welded vertex. Where the
// base already crosses itself, the crossing is the base's own and is never
// held: the hold answers for what the deformer did.
//
// Each round, every control point under a colliding triangle takes HOLD_STEP
// more hold, and the hold spreads to its net neighbors at HOLD_SPREAD, so the
// pull-back fades out rather than stepping. A sample is linear in the control
// points (the weights are fixed), so a round re-evaluates by sums alone.
//
// strength 0 is none (the deformed net, unchanged); 1 is strict (held until no
// pair collides, within HOLD_ROUNDS). Between, the strict hold is applied by
// that fraction.

import { findSpan, basisFuns } from './basis.mjs';

export const HOLD_STEP = 0.2;
export const HOLD_SPREAD = 0.6;
export const HOLD_ROUNDS = 30;
export const MIN_SAMPLES_PER_DIRECTION = 48;
export const MAX_SAMPLES_PER_DIRECTION = 128;

// The sample parameters along one direction: at least two per nonzero knot
// span and MIN_SAMPLES_PER_DIRECTION in all, spread evenly within each span;
// evenly over the domain once that passes the cap.
function sampleParams(knots, deg, count) {
  const lo = knots[deg], hi = knots[count];
  const breaks = [];
  for (let k = deg; k <= count; k++) if (!breaks.length || knots[k] > breaks[breaks.length - 1] + 1e-12) breaks.push(knots[k]);
  const spans = Math.max(1, breaks.length - 1);
  const per = Math.max(2, Math.ceil((MIN_SAMPLES_PER_DIRECTION - 1) / spans));
  const n = spans * per + 1;
  if (n > MAX_SAMPLES_PER_DIRECTION) {
    const out = [];
    for (let a = 0; a < MAX_SAMPLES_PER_DIRECTION; a++) out.push(lo + ((hi - lo) * a) / (MAX_SAMPLES_PER_DIRECTION - 1));
    return out;
  }
  const out = [];
  for (let s = 0; s < spans; s++) for (let a = 0; a < per; a++) out.push(breaks[s] + ((breaks[s + 1] - breaks[s]) * a) / per);
  out.push(hi);
  return out;
}

// The sampled, welded, triangulated surface and each sample's linear weights
// over the control points. Both nets share knots, degrees and weights.
export function collisionMesh(base, deformed) {
  const { degU: p, degV: q, knotsU: U, knotsV: V } = base;
  const nu = base.ctrlNet.length, nv = base.ctrlNet[0].length;
  const us = sampleParams(U, p, nu), vs = sampleParams(V, q, nv);
  const su = us.length, sv = vs.length, S = su * sv;
  const width = (p + 1) * (q + 1);
  const ctrlIdx = new Int32Array(S * width), coef = new Float64Array(S * width);
  const basePos = new Float64Array(S * 3);
  const spanU = new Int32Array(su), spanV = new Int32Array(sv);
  const Nu = [], Nv = [];
  for (let a = 0; a < su; a++) { spanU[a] = findSpan(nu - 1, p, us[a], U); Nu.push(basisFuns(spanU[a], us[a], p, U)); }
  for (let b = 0; b < sv; b++) { spanV[b] = findSpan(nv - 1, q, vs[b], V); Nv.push(basisFuns(spanV[b], vs[b], q, V)); }
  for (let a = 0; a < su; a++) for (let b = 0; b < sv; b++) {
    const s = a * sv + b, o = s * width;
    let wsum = 0, k = 0;
    for (let i = 0; i <= p; i++) for (let j = 0; j <= q; j++, k++) {
      const ci = spanU[a] - p + i, cj = spanV[b] - q + j;
      const w = Nu[a][i] * Nv[b][j] * (base.ctrlNet[ci][cj][3] ?? 1);
      ctrlIdx[o + k] = ci * nv + cj; coef[o + k] = w; wsum += w;
    }
    for (k = 0; k < width; k++) {
      coef[o + k] /= wsum;
      const c = ctrlIdx[o + k], P = base.ctrlNet[Math.floor(c / nv)][c % nv];
      basePos[s * 3] += coef[o + k] * P[0]; basePos[s * 3 + 1] += coef[o + k] * P[1]; basePos[s * 3 + 2] += coef[o + k] * P[2];
    }
  }
  const disp = new Float64Array(nu * nv * 3);
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const B = base.ctrlNet[i][j], D = deformed.ctrlNet[i][j], c = (i * nv + j) * 3;
    disp[c] = D[0] - B[0]; disp[c + 1] = D[1] - B[1]; disp[c + 2] = D[2] - B[2];
  }
  // Weld coincident base samples so a seam or a pole is shared, not a gap.
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let s = 0; s < S; s++) for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], basePos[s * 3 + c]); hi[c] = Math.max(hi[c], basePos[s * 3 + c]); }
  const eps = Math.max(1e-9, 1e-7 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
  const weld = new Int32Array(S), seen = new Map();
  for (let s = 0; s < S; s++) {
    const key = `${Math.round(basePos[s * 3] / eps)},${Math.round(basePos[s * 3 + 1] / eps)},${Math.round(basePos[s * 3 + 2] / eps)}`;
    const hit = seen.get(key);
    if (hit === undefined) { seen.set(key, s); weld[s] = s; } else weld[s] = hit;
  }
  const tris = [], support = [];
  for (let a = 0; a + 1 < su; a++) for (let b = 0; b + 1 < sv; b++) {
    const v00 = weld[a * sv + b], v10 = weld[(a + 1) * sv + b], v01 = weld[a * sv + b + 1], v11 = weld[(a + 1) * sv + b + 1];
    const cell = [];
    for (let i = 0; i <= p; i++) for (let j = 0; j <= q; j++) cell.push((spanU[a] - p + i) * nv + (spanV[b] - q + j));
    for (const t of [[v00, v10, v11], [v00, v11, v01]]) {
      if (t[0] === t[1] || t[1] === t[2] || t[0] === t[2]) continue;
      tris.push(t[0], t[1], t[2]); support.push(cell);
    }
  }
  return { nu, nv, S, width, ctrlIdx, coef, basePos, disp, tris: Int32Array.from(tris), support, eps };
}

// The sample positions with each control point's displacement scaled by
// scale[c].
export function collisionPositions(mesh, scale) {
  const { S, width, ctrlIdx, coef, basePos, disp } = mesh;
  const pos = new Float64Array(basePos);
  for (let s = 0; s < S; s++) {
    const o = s * width;
    let x = 0, y = 0, z = 0;
    for (let k = 0; k < width; k++) {
      const c = ctrlIdx[o + k], f = coef[o + k] * scale[c];
      if (f === 0) continue;
      x += f * disp[c * 3]; y += f * disp[c * 3 + 1]; z += f * disp[c * 3 + 2];
    }
    pos[s * 3] += x; pos[s * 3 + 1] += y; pos[s * 3 + 2] += z;
  }
  return pos;
}

// Does segment p0-p1 cross triangle a,b,c strictly inside both?
function segmentCrossesTriangle(P, i0, i1, a, b, c) {
  const ox = P[i0 * 3], oy = P[i0 * 3 + 1], oz = P[i0 * 3 + 2];
  const dx = P[i1 * 3] - ox, dy = P[i1 * 3 + 1] - oy, dz = P[i1 * 3 + 2] - oz;
  const ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
  const e1x = P[b * 3] - ax, e1y = P[b * 3 + 1] - ay, e1z = P[b * 3 + 2] - az;
  const e2x = P[c * 3] - ax, e2y = P[c * 3 + 1] - ay, e2z = P[c * 3 + 2] - az;
  const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  const scale = Math.hypot(e1x, e1y, e1z) * Math.hypot(e2x, e2y, e2z) * Math.hypot(dx, dy, dz);
  if (Math.abs(det) <= 1e-12 * scale) return false;
  const inv = 1 / det;
  const tx = ox - ax, ty = oy - ay, tz = oz - az;
  const u = (tx * px + ty * py + tz * pz) * inv;
  if (u <= 0 || u >= 1) return false;
  const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
  const v = (dx * qx + dy * qy + dz * qz) * inv;
  if (v <= 0 || u + v >= 1) return false;
  const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
  return t > 0 && t < 1;
}

function trianglesCross(P, T, i, j) {
  const a0 = T[i * 3], a1 = T[i * 3 + 1], a2 = T[i * 3 + 2];
  const b0 = T[j * 3], b1 = T[j * 3 + 1], b2 = T[j * 3 + 2];
  return segmentCrossesTriangle(P, a0, a1, b0, b1, b2) || segmentCrossesTriangle(P, a1, a2, b0, b1, b2) || segmentCrossesTriangle(P, a2, a0, b0, b1, b2)
    || segmentCrossesTriangle(P, b0, b1, a0, a1, a2) || segmentCrossesTriangle(P, b1, b2, a0, a1, a2) || segmentCrossesTriangle(P, b2, b0, a0, a1, a2);
}

// Every pair of triangles that cross and share no vertex, as i * T + j (i < j).
// Broad phase: a uniform grid sized to the mean triangle box.
export function collidingPairs(pos, tris) {
  const T = tris.length / 3;
  const box = new Float64Array(T * 6);
  let extent = 0;
  for (let t = 0; t < T; t++) {
    const o = t * 6;
    box[o] = box[o + 1] = box[o + 2] = Infinity; box[o + 3] = box[o + 4] = box[o + 5] = -Infinity;
    for (let k = 0; k < 3; k++) {
      const v = tris[t * 3 + k];
      for (let c = 0; c < 3; c++) { const x = pos[v * 3 + c]; if (x < box[o + c]) box[o + c] = x; if (x > box[o + 3 + c]) box[o + 3 + c] = x; }
    }
    extent += Math.max(box[o + 3] - box[o], box[o + 4] - box[o + 1], box[o + 5] - box[o + 2]);
  }
  const cell = Math.max(1e-9, (2 * extent) / Math.max(1, T));
  let gx = Infinity, gy = Infinity, gz = Infinity, hx = -Infinity, hy = -Infinity, hz = -Infinity;
  for (let t = 0; t < T; t++) {
    const o = t * 6;
    gx = Math.min(gx, box[o]); gy = Math.min(gy, box[o + 1]); gz = Math.min(gz, box[o + 2]);
    hx = Math.max(hx, box[o + 3]); hy = Math.max(hy, box[o + 4]); hz = Math.max(hz, box[o + 5]);
  }
  const NY = Math.floor((hy - gy) / cell) + 1, NZ = Math.floor((hz - gz) / cell) + 1;
  const cx = (x) => Math.floor((x - gx) / cell), cy = (y) => Math.floor((y - gy) / cell), cz = (z) => Math.floor((z - gz) / cell);
  const grid = new Map(), big = [];
  for (let t = 0; t < T; t++) {
    const o = t * 6;
    const x0 = cx(box[o]), y0 = cy(box[o + 1]), z0 = cz(box[o + 2]);
    const x1 = cx(box[o + 3]), y1 = cy(box[o + 4]), z1 = cz(box[o + 5]);
    if ((x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) > 512) { big.push(t); continue; }
    for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) {
      const key = (x * NY + y) * NZ + z;
      let list = grid.get(key);
      if (!list) grid.set(key, (list = []));
      list.push(t);
    }
  }
  const found = new Set();
  const isBig = new Uint8Array(T);
  for (const t of big) isBig[t] = 1;
  // `key` is the cell the pair is met in; the pair is tested only in the cell
  // holding the low corner of the two boxes' overlap, so once.
  const test = (i, j, key) => {
    const oi = i * 6, oj = j * 6;
    if (box[oi] > box[oj + 3] || box[oj] > box[oi + 3] || box[oi + 1] > box[oj + 4] || box[oj + 1] > box[oi + 4] || box[oi + 2] > box[oj + 5] || box[oj + 2] > box[oi + 5]) return;
    if (key >= 0 && (cx(Math.max(box[oi], box[oj])) * NY + cy(Math.max(box[oi + 1], box[oj + 1]))) * NZ + cz(Math.max(box[oi + 2], box[oj + 2])) !== key) return;
    const a0 = tris[i * 3], a1 = tris[i * 3 + 1], a2 = tris[i * 3 + 2];
    const b0 = tris[j * 3], b1 = tris[j * 3 + 1], b2 = tris[j * 3 + 2];
    if (a0 === b0 || a0 === b1 || a0 === b2 || a1 === b0 || a1 === b1 || a1 === b2 || a2 === b0 || a2 === b1 || a2 === b2) return;
    if (trianglesCross(pos, tris, i, j)) found.add(i < j ? i * T + j : j * T + i);
  };
  for (const [key, list] of grid) for (let m = 0; m < list.length; m++) for (let n = m + 1; n < list.length; n++) test(list[m], list[n], key);
  // A triangle too large for the grid is tested against every other, once.
  for (const t of big) for (let u = 0; u < T; u++) if (u !== t && !(isBig[u] && u < t)) test(t, u, -1);
  return found;
}

// The hold itself, over any mesh whose positions are linear in per-point
// scales: `mesh` gives C points, `tris`, each triangle's `support` (the points
// that move it), `neighbors` (per point, the points the hold spreads to),
// `basePos` and `positionsAt(scale)`. Returns the hold per point (0..1).
export function collisionHold(mesh) {
  const { C, tris, support, neighbors } = mesh;
  const T = tris.length / 3;
  // The base's own crossings are a region, not a list of pairs: a small move
  // inside it changes which pairs cross. Every triangle sharing a vertex with
  // a triangle the base already crosses is left out of the hold.
  const ownVert = new Uint8Array(tris.length ? Math.max(...tris) + 1 : 0);
  for (const key of collidingPairs(mesh.basePos, tris)) {
    const i = Math.floor(key / T), j = key - i * T;
    for (let m = 0; m < 3; m++) { ownVert[tris[i * 3 + m]] = 1; ownVert[tris[j * 3 + m]] = 1; }
  }
  const inOwn = (t) => ownVert[tris[t * 3]] || ownVert[tris[t * 3 + 1]] || ownVert[tris[t * 3 + 2]];
  const hold = new Float64Array(C), scale = new Float64Array(C).fill(1);
  const fresh = (pairs) => { const out = []; for (const key of pairs) { const i = Math.floor(key / T); if (!inOwn(i) && !inOwn(key - i * T)) out.push(key); } return out; };
  let pairs = fresh(collidingPairs(mesh.positionsAt(scale), tris));
  const collided = pairs.length;
  let rounds = 0;
  while (pairs.length && rounds < HOLD_ROUNDS) {
    rounds++;
    const bump = new Uint8Array(C);
    for (const key of pairs) {
      const i = Math.floor(key / T), j = key - i * T;
      for (const c of support[i]) bump[c] = 1;
      for (const c of support[j]) bump[c] = 1;
    }
    for (let c = 0; c < C; c++) if (bump[c]) hold[c] = Math.min(1, hold[c] + HOLD_STEP);
    const spread = new Float64Array(hold);
    for (let c = 0; c < C; c++) {
      let m = 0;
      for (const n of neighbors[c]) if (hold[n] > m) m = hold[n];
      spread[c] = Math.max(hold[c], m * HOLD_SPREAD);
    }
    hold.set(spread);
    for (let c = 0; c < C; c++) scale[c] = 1 - hold[c];
    pairs = fresh(collidingPairs(mesh.positionsAt(scale), tris));
  }
  return { hold, collided, held: pairs.length === 0, rounds };
}

// holdSelfCollision(base, deformed, strength) — the deformed surface with its
// displacement held back where it collides with itself. `base` and `deformed`
// share knots, degrees and weights; only positions differ.
// Returns { srf, collided, held, rounds, maxHold, hold }: `collided` counts
// the pairs the deformer made, `held` is true when none remain at full
// strength, `hold[i][j]` is what was taken off each control point's move.
export function holdSelfCollision(base, deformed, strength) {
  const k = Math.max(0, Math.min(1, Number(strength) || 0));
  const none = () => ({ srf: deformed, collided: 0, held: true, rounds: 0, maxHold: 0, hold: null });
  if (k === 0) return none();
  const mesh = collisionMesh(base, deformed);
  const { nu, nv } = mesh;
  const neighbors = [];
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const n = [];
    if (i > 0) n.push((i - 1) * nv + j);
    if (i + 1 < nu) n.push((i + 1) * nv + j);
    if (j > 0) n.push(i * nv + j - 1);
    if (j + 1 < nv) n.push(i * nv + j + 1);
    neighbors.push(n);
  }
  const r = collisionHold({ C: nu * nv, tris: mesh.tris, support: mesh.support, neighbors, basePos: mesh.basePos, positionsAt: (scale) => collisionPositions(mesh, scale) });
  if (!r.collided) return none();
  let maxHold = 0;
  const net = deformed.ctrlNet.map((row) => row.map((cp) => [...cp]));
  const hold = [];
  for (let i = 0; i < nu; i++) {
    const row = [];
    for (let j = 0; j < nv; j++) {
      const h = r.hold[i * nv + j] * k;
      row.push(h);
      if (h > maxHold) maxHold = h;
      if (h === 0) continue;
      const B = base.ctrlNet[i][j], P = net[i][j];
      for (let c = 0; c < 3; c++) P[c] = B[c] + (P[c] - B[c]) * (1 - h);
    }
    hold.push(row);
  }
  return { srf: { ...deformed, ctrlNet: net }, collided: r.collided, held: r.held, rounds: r.rounds, maxHold, hold };
}

// holdCageSelfCollision(base, deformed, strength) — the same hold over a
// polygon cage ({ vertices, faces }), read as its own fan-triangulated faces.
// Both cages share faces; only vertex positions differ. Returns { cage,
// collided, held, rounds, maxHold, hold } with `hold[v]` per vertex.
export function holdCageSelfCollision(base, deformed, strength) {
  const k = Math.max(0, Math.min(1, Number(strength) || 0));
  const none = () => ({ cage: deformed, collided: 0, held: true, rounds: 0, maxHold: 0, hold: null });
  if (k === 0) return none();
  const V = base.vertices.length;
  const basePos = new Float64Array(V * 3), disp = new Float64Array(V * 3);
  for (let v = 0; v < V; v++) for (let c = 0; c < 3; c++) {
    basePos[v * 3 + c] = base.vertices[v][c];
    disp[v * 3 + c] = deformed.vertices[v][c] - base.vertices[v][c];
  }
  const tris = [], support = [], nb = Array.from({ length: V }, () => new Set());
  for (const face of base.faces) {
    if (!face || face.length < 3) continue;
    for (let m = 0; m < face.length; m++) { const a = face[m], b = face[(m + 1) % face.length]; nb[a].add(b); nb[b].add(a); }
    for (let m = 1; m + 1 < face.length; m++) { tris.push(face[0], face[m], face[m + 1]); support.push(face); }
  }
  const positionsAt = (scale) => {
    const pos = new Float64Array(basePos);
    for (let v = 0; v < V; v++) if (scale[v] !== 0) for (let c = 0; c < 3; c++) pos[v * 3 + c] += disp[v * 3 + c] * scale[v];
    return pos;
  };
  const r = collisionHold({ C: V, tris: Int32Array.from(tris), support, neighbors: nb.map((s) => [...s]), basePos, positionsAt });
  if (!r.collided) return none();
  let maxHold = 0;
  const hold = Array.from(r.hold, (h) => h * k);
  const vertices = deformed.vertices.map((p, v) => {
    const h = hold[v];
    if (h > maxHold) maxHold = h;
    if (h === 0) return [...p];
    const B = base.vertices[v];
    return [B[0] + (p[0] - B[0]) * (1 - h), B[1] + (p[1] - B[1]) * (1 - h), B[2] + (p[2] - B[2]) * (1 - h)];
  });
  return { cage: { ...deformed, vertices }, collided: r.collided, held: r.held, rounds: r.rounds, maxHold, hold };
}
