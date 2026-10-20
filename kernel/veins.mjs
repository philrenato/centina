// VEINS: channels traced through a solid by the flow from an inlet to an outlet.
//
// The body (a closed triangle mesh, welded: { positions, indices }) is put on a
// grid of cubic cells, at most `resolution` cells across its longest side. A cell
// is inside by scanline parity: one ray per (x, y) column along +z, the body's
// triangles binned by their footprint, the crossings sorted and filled in pairs.
//
// Pressure p solves div(k grad p) = 0 inside, by red-black successive
// over-relaxation: p = 1 on the inlet port (the inside cells within `port` of
// the inlet point), p = 0 on the outlet port, no flux through the walls (a wall
// neighbor is left out of the average). k = exp(wander * n(x)) with n a seeded
// value noise one sixth of the body across: wander 0 is a uniform medium, larger
// values make the flow meander.
//
// Veins are streamlines of -grad p, traced by RK4 from seeds on the inlet port's
// rim spaced by `separation` (Jobard and Lefer 1997): a vein that comes within
// separation / 2 of one already traced ends there, a tributary of it, unless the
// pressure there is below VEINS_MEET_PRESSURE: near the outlet, where every vein
// is meant to arrive, they may run together. A vein ends in an outlet cell. A vein is
// held off the wall: where it comes within radius + one cell of it, it is pushed
// inward along the gradient of the distance to the wall (a 3D Euclidean distance
// transform of the inside mask, Felzenszwalb and Huttenlocher 2012). Each vein
// is then smoothed until it bends no tighter than its radius, so a pipe on it
// cannot fold.
//
// Returns { ok, veins: [{ points, joins }], grid: { n, h, origin, inside }, stats }
// or { ok: false, reason } naming the change to make.

export const VEINS_MEET_PRESSURE = 0.15;
export const VEINS_DEFAULTS = { resolution: 40, separation: 6, radius: 1, port: 4, wander: 0, seed: 1, maxIterations: 4000 };

export function traceVeins(body, inlet, outlet, opts = {}) {
  const o = { ...VEINS_DEFAULTS, ...opts };
  const P = body && body.positions, I = body && body.indices;
  if (!P || !I || I.length < 12) return { ok: false, reason: 'pick a closed body' };
  if (!vnFinite3(inlet) || !vnFinite3(outlet)) return { ok: false, reason: 'pick an inlet and an outlet on the body' };
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], P[i + k]); hi[k] = Math.max(hi[k], P[i + k]); }
  const ext = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  if (!(ext > 0)) return { ok: false, reason: 'the body has no extent' };
  const h = ext / Math.max(8, Math.round(o.resolution));
  const origin = [lo[0] - h, lo[1] - h, lo[2] - h];
  const n = [0, 1, 2].map((k) => Math.ceil((hi[k] - lo[k]) / h) + 3);
  const N = n[0] * n[1] * n[2];
  const idx = (i, j, k) => i + n[0] * (j + n[1] * k);
  const center = (i, j, k) => [origin[0] + (i + 0.5) * h, origin[1] + (j + 0.5) * h, origin[2] + (k + 0.5) * h];

  const inside = scanlineInside(P, I, origin, h, n);
  let insideCount = 0; for (let c = 0; c < N; c++) insideCount += inside[c];
  if (insideCount < 8) return { ok: false, reason: 'the body is too thin for veins at this resolution; raise Resolution' };
  if (o.port < h) o.port = h;

  // Ports: discs on the surface. The inside cells within `port` of the point and
  // within a cell and a half of the surface's tangent plane there (its normal
  // read from the nearest triangle), so a port wider than its face covers the
  // whole face and the flow leaves it square.
  const portCells = (pt) => {
    const out = [];
    const r = o.port, r2 = r * r;
    const nrm = nearestOnSurface(P, I, pt).normal;
    const c0 = pt.map((v, k) => Math.floor((v - origin[k]) / h));
    const m = Math.ceil(r / h) + 2;
    for (let dk = -m; dk <= m; dk++) for (let dj = -m; dj <= m; dj++) for (let di = -m; di <= m; di++) {
      const i = c0[0] + di, j = c0[1] + dj, k = c0[2] + dk;
      if (i < 0 || j < 0 || k < 0 || i >= n[0] || j >= n[1] || k >= n[2]) continue;
      const c = idx(i, j, k); if (!inside[c]) continue;
      const q = center(i, j, k);
      const off = nrm ? Math.abs(nrm[0] * (q[0] - pt[0]) + nrm[1] * (q[1] - pt[1]) + nrm[2] * (q[2] - pt[2])) : 0;
      if (vnDist2(q, pt) <= r2 && off <= 1.5 * h) out.push(c);
    }
    return out;
  };
  // A port is the nearest point of the surface to the one given, so a point
  // typed on a face the body has since rounded away from (a SuperB of a box)
  // still lands on it; one farther than a quarter of the body away is refused.
  const snapIn = nearestOnSurface(P, I, inlet), snapOut = nearestOnSurface(P, I, outlet);
  if (!snapIn.point || snapIn.distance > ext / 4) return { ok: false, reason: 'the inlet is not on the body; pick a point on its surface' };
  if (!snapOut.point || snapOut.distance > ext / 4) return { ok: false, reason: 'the outlet is not on the body; pick a point on its surface' };
  inlet = snapIn.point; outlet = snapOut.point;
  const inCells = portCells(inlet), outCells = portCells(outlet);
  if (!inCells.length) return { ok: false, reason: 'the inlet is not on the body; pick a point on its surface' };
  if (!outCells.length) return { ok: false, reason: 'the outlet is not on the body; pick a point on its surface' };
  const fixed = new Int8Array(N); // 1 inlet, 2 outlet
  for (const c of inCells) fixed[c] = 1;
  for (const c of outCells) { if (fixed[c] === 1) return { ok: false, reason: 'the inlet and outlet overlap; move them apart or lower Port' }; fixed[c] = 2; }

  // Conductivity.
  const noise = valueNoise3(o.seed, ext / 6);
  const kf = new Float64Array(N);
  for (let k = 0; k < n[2]; k++) for (let j = 0; j < n[1]; j++) for (let i = 0; i < n[0]; i++) {
    const c = idx(i, j, k); if (inside[c]) kf[c] = o.wander ? Math.exp(o.wander * noise(center(i, j, k))) : 1;
  }

  // Pressure, red-black SOR.
  const p = new Float64Array(N);
  for (let c = 0; c < N; c++) p[c] = fixed[c] === 1 ? 1 : fixed[c] === 2 ? 0 : 0.5;
  const omega = 2 / (1 + Math.sin(Math.PI / Math.max(...n)));
  const nb = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  let iterations = 0, change = Infinity;
  for (; iterations < o.maxIterations && change > 1e-6; iterations++) {
    change = 0;
    for (let color = 0; color < 2; color++) {
      for (let k = 1; k < n[2] - 1; k++) for (let j = 1; j < n[1] - 1; j++) for (let i = 1 + ((j + k + color) & 1); i < n[0] - 1; i += 2) {
        const c = idx(i, j, k);
        if (!inside[c] || fixed[c]) continue;
        let s = 0, w = 0;
        for (const [a, b, d] of nb) {
          const e = idx(i + a, j + b, k + d);
          if (!inside[e]) continue;
          const ke = 2 * kf[c] * kf[e] / (kf[c] + kf[e]);
          s += ke * p[e]; w += ke;
        }
        if (!w) continue;
        const next = p[c] + omega * (s / w - p[c]);
        change = Math.max(change, Math.abs(next - p[c]));
        p[c] = next;
      }
    }
  }

  const wallDist = insideDistance(inside, n, h);
  const sample = (field, x) => vnTrilinear(field, x, origin, h, n, idx);
  const grad = (field, x) => {
    const e = h * 0.5;
    return [0, 1, 2].map((k) => { const a = x.slice(), b = x.slice(); a[k] += e; b[k] -= e; return (sample(field, a) - sample(field, b)) / (2 * e); });
  };
  const fixedAt = (x) => {
    const c = x.map((v, k) => Math.floor((v - origin[k]) / h));
    if (c.some((v, k) => v < 0 || v >= n[k])) return 0;
    return fixed[idx(c[0], c[1], c[2])];
  };
  const insideAt = (x) => {
    const c = x.map((v, k) => Math.floor((v - origin[k]) / h));
    if (c.some((v, k) => v < 0 || v >= n[k])) return false;
    return !!inside[idx(c[0], c[1], c[2])];
  };

  // Seeds: inside cells next to the inlet port, spaced by separation.
  const rim = [];
  for (const c of inCells) {
    const i = c % n[0], j = Math.floor(c / n[0]) % n[1], k = Math.floor(c / (n[0] * n[1]));
    for (const [a, b, d] of nb) { const e = idx(i + a, j + b, k + d); if (inside[e] && !fixed[e]) { rim.push(center(i + a, j + b, k + d)); break; } }
  }
  const clearance = o.radius + h;
  // The port is on the wall, so its rim is too: a seed starts one clearance in
  // from the port along the port's inward normal, and is kept where it is that
  // far from every wall (so a vein never starts inside a side wall), spaced by
  // separation from the port's middle outward.
  const nIn = snapIn.normal ? (() => { const t = vnAdd(inlet, vnScale(snapIn.normal, 2 * h)); return insideAt(t) ? snapIn.normal : vnScale(snapIn.normal, -1); })() : null;
  const seeds = [];
  for (const r of rim.sort((a, b) => vnDist2(a, inlet) - vnDist2(b, inlet))) {
    const q = nIn ? vnAdd(r, vnScale(nIn, clearance)) : r;
    if (!insideAt(q) || vnTrilinear(wallDist, q, origin, h, n, idx) < clearance * 0.9) continue;
    if (seeds.every((s) => vnDist2(s, q) >= o.separation * o.separation)) seeds.push(q);
  }
  if (!seeds.length) return { ok: false, reason: 'the inlet port is too small for a vein this wide; raise Port or lower Radius' };

  const step = h * 0.5;
  const traced = [];
  const hash = new Map(); // a spatial hash of traced points, cell = separation
  const hkey = (x) => x.map((v) => Math.floor(v / o.separation)).join(',');
  const near = (x, r) => {
    const r2 = r * r, c = x.map((v) => Math.floor(v / o.separation));
    for (let dk = -1; dk <= 1; dk++) for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
      const l = hash.get(`${c[0] + di},${c[1] + dj},${c[2] + dk}`); if (!l) continue;
      for (const q of l) if (vnDist2(q.p, x) < r2) return q;
    }
    return null;
  };
  const velocity = (x) => {
    const g = grad(p, x);
    let v = [-g[0], -g[1], -g[2]];
    const lv = Math.hypot(...v);
    if (!(lv > 1e-12)) return null;
    v = v.map((t) => t / lv);
    const wd = sample(wallDist, x);
    if (wd < clearance) {
      const gw = grad(wallDist, x), lw = Math.hypot(...gw);
      if (lw > 1e-12) { const push = (clearance - wd) / clearance; v = [0, 1, 2].map((k) => v[k] + 1.5 * push * gw[k] / lw); const l2 = Math.hypot(...v); v = v.map((t) => t / l2); }
    }
    return v;
  };
  const maxSteps = Math.ceil((8 * ext) / step);
  // A vein runs surface to surface: it starts on the inlet's plane (its seed
  // carried back along the inlet normal) and, when it reaches the outlet, ends
  // on the outlet's plane, so a channel cut along it opens at both ports. Near
  // the ports the wall is where a vein is meant to be; the clearance is held,
  // and measured, away from them.
  const nOutIn = snapOut.normal ? (() => { const t = vnAdd(outlet, vnScale(snapOut.normal, 2 * h)); return insideAt(t) ? snapOut.normal : vnScale(snapOut.normal, -1); })() : null;
  const onPlane = (x, c, nrm) => (nrm ? vnSub(x, vnScale(nrm, vnDot(vnSub(x, c), nrm))) : x);
  const nearPort = (x) => vnDist2(x, inlet) < (o.port + clearance) ** 2 || vnDist2(x, outlet) < (o.port + clearance) ** 2;
  for (const s of seeds) {
    const pts = [onPlane(s, inlet, nIn), s.slice()];
    let x = s.slice(), joins = null;
    for (let it = 0; it < maxSteps; it++) {
      const k1 = velocity(x); if (!k1) break;
      const k2 = velocity(vnAdd(x, vnScale(k1, step / 2))); if (!k2) break;
      const k3 = velocity(vnAdd(x, vnScale(k2, step / 2))); if (!k3) break;
      const k4 = velocity(vnAdd(x, vnScale(k3, step))); if (!k4) break;
      const nx = vnAdd(x, vnScale(vnAdd(vnAdd(k1, vnScale(k2, 2)), vnAdd(vnScale(k3, 2), k4)), step / 6));
      if (!insideAt(nx)) break;
      x = nx; pts.push(x.slice());
      if (fixedAt(x) === 2 || sample(p, x) < 0.002) { pts.push(onPlane(x, outlet, nOutIn)); break; }
      if (sample(p, x) > VEINS_MEET_PRESSURE) {
        const hit = near(x, o.separation / 2);
        if (hit) { joins = hit.vein; pts.push(hit.p.slice()); break; }
      }
    }
      const len = pts.reduce((a, q, i) => (i ? a + Math.sqrt(vnDist2(q, pts[i - 1])) : 0), 0);
    if (len < o.separation) continue;
    const smooth = keepOffWall(relaxToBend(pts, o.radius * 1.25, insideAt), (x) => sample(wallDist, x), (x) => grad(wallDist, x), o.radius + h * 0.25, nearPort);
    const vein = traced.length;
    traced.push({ points: smooth, joins, length: len });
    for (const q of smooth) { const kk = hkey(q); if (!hash.has(kk)) hash.set(kk, []); hash.get(kk).push({ p: q, vein }); }
  }
  if (!traced.length) return { ok: false, reason: 'no vein reached the outlet; lower Separation or raise Port' };
  let minWall = Infinity;
  for (const v of traced) for (const q of v.points) if (!nearPort(q)) minWall = Math.min(minWall, sample(wallDist, q));
  // Unmeasured (every point lies by a port) is null, not a distance.
  return { ok: true, veins: traced, grid: { n, h, origin }, stats: { iterations, residual: change, seeds: seeds.length, insideCells: insideCount, minWall: Number.isFinite(minWall) ? minWall : null } };
}

// The nearest point of the body's surface to `pt`, its distance, and the normal
// of the triangle it lies on.
function nearestOnSurface(P, I, pt) {
  let best = Infinity, nrm = null, at = null;
  for (let t = 0; t < I.length; t += 3) {
    const a = [P[I[t] * 3], P[I[t] * 3 + 1], P[I[t] * 3 + 2]], b = [P[I[t + 1] * 3], P[I[t + 1] * 3 + 1], P[I[t + 1] * 3 + 2]], c = [P[I[t + 2] * 3], P[I[t + 2] * 3 + 1], P[I[t + 2] * 3 + 2]];
    const q = vnClosestOnTriangle(pt, a, b, c), d = vnDist2(q, pt);
    if (d < best) { best = d; at = q; const cr = vnCross(vnSub(b, a), vnSub(c, a)), l = Math.hypot(...cr); if (l > 0) nrm = cr.map((v) => v / l); }
  }
  return { point: at, distance: Math.sqrt(best), normal: nrm };
}
// Ericson, Real-Time Collision Detection, 5.1.5.
function vnClosestOnTriangle(p, a, b, c) {
  const ab = vnSub(b, a), ac = vnSub(c, a), ap = vnSub(p, a);
  const d1 = vnDot(ab, ap), d2 = vnDot(ac, ap); if (d1 <= 0 && d2 <= 0) return a;
  const bp = vnSub(p, b), d3 = vnDot(ab, bp), d4 = vnDot(ac, bp); if (d3 >= 0 && d4 <= d3) return b;
  const vc = d1 * d4 - d3 * d2; if (vc <= 0 && d1 >= 0 && d3 <= 0) return vnAdd(a, vnScale(ab, d1 / (d1 - d3)));
  const cp = vnSub(p, c), d5 = vnDot(ab, cp), d6 = vnDot(ac, cp); if (d6 >= 0 && d5 <= d6) return c;
  const vb = d5 * d2 - d1 * d6; if (vb <= 0 && d2 >= 0 && d6 <= 0) return vnAdd(a, vnScale(ac, d2 / (d2 - d6)));
  const va = d3 * d6 - d5 * d4; if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) return vnAdd(b, vnScale(vnSub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
  const den = 1 / (va + vb + vc); return vnAdd(a, vnAdd(vnScale(ab, vb * den), vnScale(ac, vc * den)));
}
// Inside cells by scanline parity along +z, triangles binned by footprint.
function scanlineInside(P, I, origin, h, n) {
  const inside = new Uint8Array(n[0] * n[1] * n[2]);
  const bins = new Map();
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
    const x0 = Math.min(P[a], P[b], P[c]), x1 = Math.max(P[a], P[b], P[c]);
    const y0 = Math.min(P[a + 1], P[b + 1], P[c + 1]), y1 = Math.max(P[a + 1], P[b + 1], P[c + 1]);
    for (let j = Math.floor((y0 - origin[1]) / h - 0.5); j <= Math.ceil((y1 - origin[1]) / h - 0.5); j++)
      for (let i = Math.floor((x0 - origin[0]) / h - 0.5); i <= Math.ceil((x1 - origin[0]) / h - 0.5); i++) {
        if (i < 0 || j < 0 || i >= n[0] || j >= n[1]) continue;
        const key = i + n[0] * j;
        if (!bins.has(key)) bins.set(key, []);
        bins.get(key).push(t);
      }
  }
  for (const [key, tris] of bins) {
    const i = key % n[0], j = Math.floor(key / n[0]);
    // A tiny fixed offset keeps the ray off mesh edges and vertices on the grid.
    const px = origin[0] + (i + 0.5) * h + h * 1.3e-4, py = origin[1] + (j + 0.5) * h + h * 0.7e-4;
    const zs = [];
    for (const t of tris) {
      const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
      const z = rayZ(px, py, P[a], P[a + 1], P[a + 2], P[b], P[b + 1], P[b + 2], P[c], P[c + 1], P[c + 2]);
      if (z !== null) zs.push(z);
    }
    zs.sort((u, v) => u - v);
    for (let q = 0; q + 1 < zs.length; q += 2) {
      const k0 = Math.ceil((zs[q] - origin[2]) / h - 0.5), k1 = Math.floor((zs[q + 1] - origin[2]) / h - 0.5);
      for (let k = Math.max(0, k0); k <= Math.min(n[2] - 1, k1); k++) inside[i + n[0] * (j + n[1] * k)] = 1;
    }
  }
  return inside;
}
function rayZ(px, py, ax, ay, az, bx, by, bz, cx, cy, cz) {
  const d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
  if (Math.abs(d) < 1e-300) return null;
  const l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / d;
  const l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / d;
  const l3 = 1 - l1 - l2;
  if (l1 < 0 || l2 < 0 || l3 < 0) return null;
  return l1 * az + l2 * bz + l3 * cz;
}

// Euclidean distance from each inside cell's center to the nearest outside cell's
// center, minus half a cell: the distance to the wall.
function insideDistance(inside, n, h) {
  const N = inside.length, INF = 1e20;
  let f = new Float64Array(N);
  for (let c = 0; c < N; c++) f[c] = inside[c] ? INF : 0;
  const idx = (i, j, k) => i + n[0] * (j + n[1] * k);
  for (let axis = 0; axis < 3; axis++) {
    const len = n[axis];
    const line = new Float64Array(len), out = new Float64Array(len);
    const others = [0, 1, 2].filter((a) => a !== axis);
    for (let b = 0; b < n[others[1]]; b++) for (let a = 0; a < n[others[0]]; a++) {
      const at = (t) => { const v = [0, 0, 0]; v[axis] = t; v[others[0]] = a; v[others[1]] = b; return idx(v[0], v[1], v[2]); };
      for (let t = 0; t < len; t++) line[t] = f[at(t)];
      edt1d(line, out, len);
      for (let t = 0; t < len; t++) f[at(t)] = out[t];
    }
  }
  const d = new Float64Array(N);
  for (let c = 0; c < N; c++) d[c] = inside[c] ? Math.max(0, Math.sqrt(f[c]) * h - h / 2) : 0;
  return d;
}
function edt1d(f, d, n) {
  const v = new Int32Array(n), z = new Float64Array(n + 1);
  let k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
    k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]; }
}

function vnTrilinear(field, x, origin, h, n, idx) {
  const g = [0, 1, 2].map((k) => (x[k] - origin[k]) / h - 0.5);
  const b = g.map((v, k) => Math.max(0, Math.min(n[k] - 2, Math.floor(v))));
  const t = g.map((v, k) => Math.max(0, Math.min(1, v - b[k])));
  let s = 0;
  for (let dz = 0; dz < 2; dz++) for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
    const w = (dx ? t[0] : 1 - t[0]) * (dy ? t[1] : 1 - t[1]) * (dz ? t[2] : 1 - t[2]);
    s += w * field[idx(b[0] + dx, b[1] + dy, b[2] + dz)];
  }
  return s;
}

// Every point closer to the wall than `clear` lifted along the distance's
// gradient until it is not, a few passes; the end points, and any point
// `exempt` names (one by a port), stay where they are.
function keepOffWall(pts, dist, gradDist, clear, exempt = () => false) {
  for (let pass = 0; pass < 8; pass++) {
    let moved = false;
    for (let i = 1; i + 1 < pts.length; i++) {
      if (exempt(pts[i])) continue;
      const d = dist(pts[i]);
      if (d >= clear) continue;
      const g = gradDist(pts[i]), l = Math.hypot(...g);
      if (!(l > 1e-12)) continue;
      pts[i] = [0, 1, 2].map((k) => pts[i][k] + ((clear - d) * g[k]) / (l * l));
      moved = true;
    }
    if (!moved) break;
  }
  return pts;
}
// A polyline relaxed (Laplacian, ends fixed) until its tightest turn has a radius
// of at least `minRadius`, staying inside; at most 200 passes.
export function relaxToBend(pts, minRadius, insideAt = () => true) {
  let q = vnResample(pts, Math.max(minRadius * 0.75, vnPolyLength(pts) / 400));
  for (let pass = 0; pass < 200 && q.length > 2; pass++) {
    if (minTurnRadius(q) >= minRadius) break;
    const next = q.map((x) => x.slice());
    for (let i = 1; i + 1 < q.length; i++) {
      const m = [0, 1, 2].map((k) => (q[i - 1][k] + q[i + 1][k]) / 2);
      const c = [0, 1, 2].map((k) => q[i][k] + 0.5 * (m[k] - q[i][k]));
      if (insideAt(c)) next[i] = c;
    }
    q = next;
  }
  return q;
}
function minTurnRadius(q) {
  let r = Infinity;
  for (let i = 1; i + 1 < q.length; i++) {
    const a = q[i - 1], b = q[i], c = q[i + 1];
    const ab = Math.sqrt(vnDist2(a, b)), bc = Math.sqrt(vnDist2(b, c)), ca = Math.sqrt(vnDist2(c, a));
    const cr = vnCross(vnSub(b, a), vnSub(c, a)), area2 = Math.hypot(...cr);
    if (area2 > 1e-12) r = Math.min(r, (ab * bc * ca) / (2 * area2));
  }
  return r;
}
function vnResample(pts, spacing) {
  const out = [pts[0].slice()];
  let carry = 0;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i], L = Math.sqrt(vnDist2(a, b));
    let t = spacing - carry;
    while (t <= L) { out.push([0, 1, 2].map((k) => a[k] + ((b[k] - a[k]) * t) / L)); t += spacing; }
    carry = L - (t - spacing);
  }
  const last = pts[pts.length - 1];
  if (vnDist2(out[out.length - 1], last) > 1e-12) out.push(last.slice());
  return out;
}
function vnPolyLength(pts) { let s = 0; for (let i = 1; i < pts.length; i++) s += Math.sqrt(vnDist2(pts[i], pts[i - 1])); return s; }

// Smooth value noise in [-1, 1] with feature size `size`, seeded.
function valueNoise3(seed, size) {
  const hash = (i, j, k) => { let x = (i * 374761393 + j * 668265263 + k * 2147483647 + seed * 144269504) | 0; x = Math.imul(x ^ (x >>> 13), 1274126177); x ^= x >>> 16; return ((x >>> 0) / 4294967295) * 2 - 1; };
  const fade = (t) => t * t * (3 - 2 * t);
  return (p) => {
    const g = p.map((v) => v / size), b = g.map(Math.floor), t = g.map((v, k) => fade(v - b[k]));
    let s = 0;
    for (let dz = 0; dz < 2; dz++) for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++)
      s += (dx ? t[0] : 1 - t[0]) * (dy ? t[1] : 1 - t[1]) * (dz ? t[2] : 1 - t[2]) * hash(b[0] + dx, b[1] + dy, b[2] + dz);
    return s;
  };
}

function vnFinite3(v) { return Array.isArray(v) && v.length >= 3 && v.slice(0, 3).every(Number.isFinite); }
function vnAdd(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function vnSub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function vnScale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function vnCross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function vnDot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function vnDist2(a, b) { const x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2]; return x * x + y * y + z * z; }
