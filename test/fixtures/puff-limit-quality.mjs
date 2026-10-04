// What a puff's limit surface looks like, as numbers: how sharply its crown bends anywhere, and how
// far its rims stand off the drawn curves, both ways.
import { subdivideCatmullClark } from '../../kernel/subd.mjs';
import { buildAnnularPuff, puffLimitRimMiss } from '../../kernel/puff.mjs';

const segDist = (P, x, y) => { let b = Infinity; for (let i = 0; i < P.length; i++) { const a = P[i], c = P[(i + 1) % P.length]; const ex = c[0] - a[0], ey = c[1] - a[1], L2 = ex * ex + ey * ey; let t = L2 > 0 ? ((x - a[0]) * ex + (y - a[1]) * ey) / L2 : 0; t = Math.max(0, Math.min(1, t)); b = Math.min(b, Math.hypot(a[0] + t * ex - x, a[1] + t * ey - y)); } return b; };
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

export function limitOf(r, levels) {
  const vertices = []; for (let i = 0; i < r.nv; i++) vertices.push([r.positions[3 * i], r.positions[3 * i + 1], r.positions[3 * i + 2]]);
  const faces = []; for (let i = 0; i < r.quads.length; i += 4) faces.push([r.quads[i], r.quads[i + 1], r.quads[i + 2], r.quads[i + 3]]);
  let c = { vertices, faces };
  for (let k = 0; k < levels; k++) c = subdivideCatmullClark(c);
  return c;
}

/* The highlight's continuity, edge by edge. On the three-times-subdivided front sheet, every edge
   between two faces is scored by the angle between their normals per unit distance between their
   centers (how fast a highlight slides across it), times the half-width of the crown or bridge it
   sits on (half the sum of the distances to the two nearest drawn curves, capped at the crown's
   depth), so a round tube of any width scores the same. Edges low on the rim walls (below a
   quarter of the peak) or near a rim (within a quarter of the depth) are left out: a rim's own
   turn is the drawing, not a seam. A round single-hole ring reads 1.2 at most; a construction line
   reads several times its surroundings. */
export function creaseScore(r, loops, levels = 3) {
  const { vertices: V, faces: F } = limitOf(r, levels);
  let zmax = 0; for (const p of V) zmax = Math.max(zmax, p[2]);
  const two = (x, y) => { const d = loops.map((L) => segDist(L, x, y)).sort((u, v) => u - v); return [d[0], (d[0] + d[1]) / 2]; };
  let depth = 0; for (const p of V) if (p[2] > 0) depth = Math.max(depth, two(p[0], p[1])[0]);
  const n = F.map((f) => unit(cross(sub(V[f[2]], V[f[0]]), sub(V[f[3]], V[f[1]]))));
  const ctr = F.map((f) => [0, 1, 2].map((k) => (V[f[0]][k] + V[f[1]][k] + V[f[2]][k] + V[f[3]][k]) / 4));
  const owner = new Map(), scores = [];
  F.forEach((f, fi) => {
    for (let k = 0; k < 4; k++) {
      const a = f[k], b = f[(k + 1) % 4], key = a < b ? a * 1e7 + b : b * 1e7 + a, o = owner.get(key);
      if (o === undefined) { owner.set(key, fi); continue; }
      if (V[a][2] < zmax / 4 || V[b][2] < zmax / 4) continue;
      const mx = (V[a][0] + V[b][0]) / 2, my = (V[a][1] + V[b][1]) / 2, [d0, half] = two(mx, my);
      if (d0 < depth / 4) continue;
      const c = n[o][0] * n[fi][0] + n[o][1] * n[fi][1] + n[o][2] * n[fi][2];
      const len = Math.hypot(...sub(ctr[o], ctr[fi])) || 1;
      scores.push({ k: Math.acos(Math.max(-1, Math.min(1, c))) / len * Math.min(depth, half), x: mx, y: my });
    }
  });
  scores.sort((u, v) => v.k - u.k);
  return { max: scores[0].k, p99: scores[Math.floor(scores.length / 100)].k, at: [scores[0].x, scores[0].y] };
}

/* The rims against the drawing, both ways: the farthest any limit-rim point stands from its curve,
   and the farthest any point of a curve stands from the limit rim that should follow it (a turn
   the rim cuts across shows only this way), as fractions of the outline's span. The limit rim is
   every edge of the three-times-subdivided cage with both ends on the drawing plane. */
export function rimMissBothWays(r, loops, levels = 3) {
  const { vertices: V, faces: F } = limitOf(r, levels);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const L of loops) for (const [x, y] of L) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  const span = Math.max(x1 - x0, y1 - y0), flat = V.map((p) => Math.abs(p[2]) < span * 1e-7);
  const segs = loops.map(() => []), seen = new Set();
  for (const f of F) for (let k = 0; k < 4; k++) {
    const a = f[k], b = f[(k + 1) % 4]; if (!flat[a] || !flat[b]) continue;
    const key = a < b ? a * 1e7 + b : b * 1e7 + a; if (seen.has(key)) continue; seen.add(key);
    const mx = (V[a][0] + V[b][0]) / 2, my = (V[a][1] + V[b][1]) / 2;
    let best = 0, bd = Infinity; loops.forEach((L, i) => { const d = segDist(L, mx, my); if (d < bd) { bd = d; best = i; } });
    segs[best].push([V[a], V[b]]);
  }
  let worst = 0;
  loops.forEach((L, i) => {
    for (const [a, b] of segs[i]) for (const p of [a, b]) worst = Math.max(worst, segDist(L, p[0], p[1]));
    for (let j = 0; j < L.length; j++) for (let s = 0; s < 4; s++) {
      const p = L[j], q = L[(j + 1) % L.length], x = p[0] + (q[0] - p[0]) * s / 4, y = p[1] + (q[1] - p[1]) * s / 4;
      let d = Infinity; for (const [a, b] of segs[i]) d = Math.min(d, segDist([a, b], x, y));
      worst = Math.max(worst, d);
    }
  });
  return worst / span;
}

export const circle = (r, n, cx = 0, cy = 0, sx = 1) => { const p = []; for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; p.push([cx + Math.cos(a) * r * sx, cy + Math.sin(a) * r]); } return p; };
const wavy = (r, n, k, amp) => { const p = []; for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2, rr = r * (1 + amp * Math.sin(k * a)); p.push([Math.cos(a) * rr, Math.sin(a) * rr]); } return p; };
// A single-hole puff's typical rim miss, the median over five plain drawings at the default
// density: two circles, an ellipse, two wavy outlines (0.2% to 7% of the span).
export function singleHoleMiss() {
  const m = [
    [circle(40, 128), [circle(8, 64, 12, 0)]], [circle(40, 128), [circle(15, 64)]],
    [circle(40, 128, 0, 0, 1.4), [circle(10, 64, 10, 5, 1.5)]], [wavy(40, 128, 3, 0.15), [circle(10, 64, 5, 0)]], [wavy(40, 128, 5, 0.1), [circle(12, 64, -5, 3)]],
  ].map(([o, h]) => puffLimitRimMiss(buildAnnularPuff(o, h), [o, ...h]).worst).sort((a, b) => a - b);
  return m[2];
}
