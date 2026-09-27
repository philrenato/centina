// A closed outline from a handful of discs.
//
// Draw two or three overlapping circles and the contour around them is a
// single soft shape — the crotch where they meet is rounded rather than
// notched, and a small gap is bridged. It is the cheapest way to get an
// organic closed outline, which Puff turns into a solid.
//
// The blend is the fuse's, not a second one. `solidwrap.mjs` carries this
// app's one smooth minimum, and its Blend row promises "two surfaces a gap g
// apart meet at Blend g". A different 2D blend here — a metaball field summed
// to an iso level, the usual way — would give the same word two meanings, and
// keeping that promise would need a per-frame bisection of the iso level.
// Folding the same `smoothMinPoly` over 2D disc distances keeps the promise by
// construction: two r = 10 discs bridge at a gap of exactly 2.00, 5.00 and
// 10.00mm at Blend 2, 5 and 10.
//
// The contour is marching squares over a grid, then every vertex is pushed onto
// the true zero set by Newton steps along the gradient — so the grid decides
// where the contour is found and not where it sits, and its resolution does
// not show in the result.
import { smoothMinPoly, fuseBlendRadius } from './solidwrap.mjs';

/* The signed distance to a disc, and the fold that joins them. Exact outside
   the blend region, which is what makes a lone disc's contour land on its own
   drawn radius at every Blend. */
export function blobFieldAt(discs, k, x, y) {
  let acc = Infinity;
  for (let i = 0; i < discs.length; i += 1) {
    const d = Math.hypot(x - discs[i][0], y - discs[i][1]) - discs[i][2];
    acc = i === 0 ? d : smoothMinPoly(acc, d, k);
  }
  return acc;
}

/* A per-ball melt, and the fold order is the mechanism. Each ball may carry its
   own melt (element 3; null or absent means "follow the global"), and what two
   balls do where they meet is governed by the softer of the two — a hard ball
   stays hard against anything.
   Folding in decreasing melt makes that hold: by the time ball i is folded in,
   every ball already in the accumulator has a melt at least as large, so the
   step's k is min(b_i, b_j) for every pair it stands for. Bridging then
   happens at min(b_i,b_j) equal to the gap to within 1.1e-3mm across mixed
   settings.
   Alternatives that fail: a per-ball multiplier misses the bridge by +100% at
   0.5 and +400% at 0.2; a running min(k) lets a hard ball 130mm away, merely
   listed between two others, pull them from one piece into two; a
   partition-of-unity k breaks scale neutrality by 1.0937x.
   The sort is conditional. smoothMinPoly is not associative, so reordering the
   balls moves the outline, and sorting unconditionally would move saved blobs
   by up to 3.2mm. It runs only when the melts differ, which is exactly when it
   changes anything, so a blob with uniform melt is bit-identical. */
export function blobFieldMixed(discs, k, ks, x, y) {
  let acc = Infinity;
  for (let i = 0; i < discs.length; i += 1) {
    const d = Math.hypot(x - discs[i][0], y - discs[i][1]) - discs[i][2];
    // ks[i] is the pairwise minimum, because the fold order guarantees every
    // ball already in `acc` has a melt at least this large.
    acc = i === 0 ? d : smoothMinPoly(acc, d, ks[i]);
  }
  return acc;
}

/* The per-ball melt each ball actually gets: its own if it has one, otherwise
   the global — and never more than the global, because the minimum governs and a
   ball asking for more than the set allows would be a slider with a dead half.
   Returns the balls in fold order (softest first) with the blend radii to match. */
export function blobMeltOrder(discs, blend) {
  const melts = discs.map((d) => {
    const own = d.length > 3 && d[3] != null && Number.isFinite(d[3]) ? Math.max(0, d[3]) : null;
    return own == null ? blend : Math.min(own, blend);
  });
  const varied = melts.some((m) => m !== melts[0]);
  if (!varied) return { discs, ks: melts.map(() => fuseBlendRadius(blend)), varied: false };
  const idx = discs.map((_, i) => i).sort((a, b) => melts[b] - melts[a] || a - b);
  return {
    discs: idx.map((i) => discs[i]),
    ks: idx.map((i) => fuseBlendRadius(melts[i])),
    varied: true,
  };
}

function blobGradAt(discs, k, x, y, h, at = blobFieldAt) {
  return [
    (at(discs, k, x + h, y) - at(discs, k, x - h, y)) / (2 * h),
    (at(discs, k, x, y + h) - at(discs, k, x, y - h)) / (2 * h),
  ];
}

const AREA = (pts) => {
  let a = 0;
  for (let i = 0; i < pts.length; i += 1) {
    const j = (i + 1) % pts.length;
    a += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
  }
  return a / 2;
};

/**
 * The outline of a set of discs, as closed rings in the plane.
 *
 * `discs` is [[x, y, r], ...] in millimeters. Returns every ring the field
 * produces, not only the biggest: separate pieces and holes are answers — two
 * discs far apart are two pieces — and returning only the largest would drop
 * part of the drawing.
 */
function blobOutlineOneCluster(discs, opts = {}) {
  if (!Array.isArray(discs) || discs.length === 0) {
    return { ok: false, reason: 'no-discs', why: 'a blob needs at least one disc' };
  }
  for (const d of discs) {
    if (!Array.isArray(d) || d.length < 3 || !Number.isFinite(d[0]) || !Number.isFinite(d[1]) || !(d[2] > 0)) {
      return { ok: false, reason: 'bad-disc', why: 'every disc needs a centre and a radius above zero' };
    }
  }
  const asked = Math.max(0, opts.blend ?? 0);
  const tolerance = opts.tolerance ?? 0.01;

  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity], rMin = Infinity, rMax = 0;
  for (const [x, y, r] of discs) {
    lo[0] = Math.min(lo[0], x - r); lo[1] = Math.min(lo[1], y - r);
    hi[0] = Math.max(hi[0], x + r); hi[1] = Math.max(hi[1], y + r);
    rMin = Math.min(rMin, r); rMax = Math.max(rMax, r);
  }
  const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1]);

  const blend = asked;
  const k = fuseBlendRadius(blend);
  /* The cell is tied to the smallest disc, and the pad is part of the box.
     A count taken over the drawing alone cannot see a disc much smaller than a
     cell, and the failure is silent — the disc is not in the outline. Four
     cells across the smallest radius is the floor; the clamp keeps a
     pathological ratio from asking for an unaffordable grid. The pad is
     counted too: Blend widens the domain, and sizing the grid before adding
     it would let a large Blend coarsen every cell. */
  const pad = blend + 0.05 * extent;
  lo = [lo[0] - pad, lo[1] - pad]; hi = [hi[0] + pad, hi[1] + pad];
  const boxed = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
  const want = Math.round((4 * boxed) / Math.max(rMin, 1e-9));
  const cells = Math.max(192, Math.min(384, Number.isFinite(want) ? want : 192));
  const nx = cells;
  const ny = Math.max(8, Math.round((cells * (hi[1] - lo[1])) / Math.max(hi[0] - lo[0], 1e-9)));
  const hx = (hi[0] - lo[0]) / nx, hy = (hi[1] - lo[1]) / ny;

  /* A Blend within a hair of a gap draws a neck thinner than one cell, which no
     grid can draw. The zero set pinches, and the contour through the pinch is
     cut into sub-cell islands (six circles with a 12mm gap: whole at 4439 and
     4482mm2 either side, but inside a band of about 0.015mm around Blend 12 it
     breaks into 22 contours totaling 2982). The saddle rule and the segment
     chaining are not the cause.
     The condition is a closed-form question about the discs: it is degenerate
     when Blend is within a cell of some pair's gap. Blend is then nudged to the
     joined side by a fraction of one cell — far below the document tolerance,
     and the direction the Blend row promises: at Blend g, a gap of g is
     bridged. */
  /* Which field this blob uses. With one melt for every ball the plain fold is
     exact and is used unchanged; only a blob whose balls disagree pays for the
     sorted mixed fold. */
  const order = blobMeltOrder(discs, blend);
  const fieldAt = order.varied
    ? (dd, kk, x, y) => blobFieldMixed(order.discs, kk, order.ks, x, y)
    : blobFieldAt;
  const nudge = Math.min(hx, hy) * 0.02;
  let kBlend = blend;
  for (let i = 0; i < discs.length; i += 1) {
    for (let j = i + 1; j < discs.length; j += 1) {
      const gap = Math.hypot(discs[i][0] - discs[j][0], discs[i][1] - discs[j][1]) - discs[i][2] - discs[j][2];
      if (Math.abs(gap - blend) < nudge) kBlend = Math.max(kBlend, gap + nudge);
    }
  }
  const kUse = kBlend === blend ? k : fuseBlendRadius(kBlend);

  const F = new Float64Array((nx + 1) * (ny + 1));
  for (let j = 0; j <= ny; j += 1) {
    for (let i = 0; i <= nx; i += 1) F[j * (nx + 1) + i] = fieldAt(discs, kUse, lo[0] + i * hx, lo[1] + j * hy);
  }
  const at = (i, j) => F[j * (nx + 1) + i];
  const px = (i, j) => [lo[0] + i * hx, lo[1] + j * hy];
  const lerp = (a, b, fa, fb) => {
    const t = fa === fb ? 0.5 : fa / (fa - fb);
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  };

  /* Marching squares, emitting segments. The saddle case is resolved on the
     cell's center value rather than by a fixed choice, so a neck that is
     joined does not come apart at one cell. */
  const segs = [];
  for (let j = 0; j < ny; j += 1) for (let i = 0; i < nx; i += 1) {
    const f0 = at(i, j), f1 = at(i + 1, j), f2 = at(i + 1, j + 1), f3 = at(i, j + 1);
    let code = 0;
    if (f0 < 0) code |= 1; if (f1 < 0) code |= 2; if (f2 < 0) code |= 4; if (f3 < 0) code |= 8;
    if (code === 0 || code === 15) continue;
    const p0 = px(i, j), p1 = px(i + 1, j), p2 = px(i + 1, j + 1), p3 = px(i, j + 1);
    const eB = () => lerp(p0, p1, f0, f1), eR = () => lerp(p1, p2, f1, f2);
    const eT = () => lerp(p3, p2, f3, f2), eL = () => lerp(p0, p3, f0, f3);
    const push = (a, b) => segs.push([a, b]);
    switch (code) {
      case 1: push(eL(), eB()); break;
      case 2: push(eB(), eR()); break;
      case 3: push(eL(), eR()); break;
      case 4: push(eR(), eT()); break;
      case 6: push(eB(), eT()); break;
      case 7: push(eL(), eT()); break;
      case 8: push(eT(), eL()); break;
      case 9: push(eT(), eB()); break;
      case 11: push(eT(), eR()); break;
      case 12: push(eR(), eL()); break;
      case 13: push(eR(), eB()); break;
      case 14: push(eB(), eL()); break;
      case 5: case 10: {
        const c = fieldAt(discs, kUse, lo[0] + (i + 0.5) * hx, lo[1] + (j + 0.5) * hy);
        if ((code === 5) === (c < 0)) { push(eL(), eT()); push(eR(), eB()); }
        else { push(eL(), eB()); push(eR(), eT()); }
        break;
      }
      default: break;
    }
  }
  if (!segs.length) {
    return { ok: false, reason: 'empty', why: 'those discs enclose nothing to draw a curve around' };
  }

  // Chain the segments into rings by welding endpoints at a fraction of a cell.
  const q = Math.min(hx, hy) * 1e-3;
  const key = (p) => `${Math.round(p[0] / q)},${Math.round(p[1] / q)}`;
  const next = new Map();
  for (const [a, b] of segs) {
    const ka = key(a);
    if (!next.has(ka)) next.set(ka, []);
    next.get(ka).push([a, b]);
  }
  const used = new Set();
  const rings = [];
  /* At a shared vertex, take the tightest turn. With Blend exactly the gap
     between two circles they touch at a point, and four contour segments meet
     at one vertex. Taking whichever segment came first walks a figure-eight
     through that vertex, consuming the segments of both loops into one
     traversal and discarding the rest, so whole rings vanish. Whether two
     circles touching at a point are one shape or two is ambiguous at that
     measure-zero value, and no invisible epsilon can settle it — the join is
     thinner than any affordable grid. What is not ambiguous is that nothing
     may disappear. Turning as tightly as possible keeps each loop on its own
     side of the vertex, so the pieces stay whole either way. */
  const dirOf = (a, b) => { const dx = b[0] - a[0], dy = b[1] - a[1]; const L = Math.hypot(dx, dy) || 1; return [dx / L, dy / L]; };
  for (const [a0, b0] of segs) {
    const id = `${key(a0)}>${key(b0)}`;
    if (used.has(id)) continue;
    const pts = [a0];
    let cur = b0, inDir = dirOf(a0, b0), guard = 0;
    used.add(id);
    while (guard < segs.length + 4) {
      guard += 1;
      pts.push(cur);
      const outs = (next.get(key(cur)) || []).filter(([a, b]) => !used.has(`${key(a)}>${key(b)}`));
      if (!outs.length) break;
      let best = null, bestTurn = Infinity;
      for (const seg of outs) {
        const d = dirOf(seg[0], seg[1]);
        // Signed turn in (-pi, pi]; the smallest one hugs this loop's own side.
        const turn = Math.atan2(inDir[0] * d[1] - inDir[1] * d[0], inDir[0] * d[0] + inDir[1] * d[1]);
        if (Math.abs(turn) < bestTurn) { bestTurn = Math.abs(turn); best = seg; }
      }
      used.add(`${key(best[0])}>${key(best[1])}`);
      inDir = dirOf(best[0], best[1]);
      cur = best[1];
      if (key(cur) === key(a0)) break;
    }
    if (pts.length >= 4) rings.push(pts);
  }
  if (!rings.length) return { ok: false, reason: 'open', why: 'the contour did not close' };

  /* The grid finds the contour; Newton puts it on the zero set, and it runs
     last. Snapping before the stations are laid out fixes the wrong points:
     the stations are then chords between snapped points (0.1mm off the contour
     on a 1.2m drawing).
     It gains nothing on a lone disc, because marching squares interpolates a
     linear field exactly and a distance field is very nearly linear across one
     cell. It matters where the field curves: 6x across a blend crease, 15x
     around an annulus, and 160-226x on a wide scale spread. A test for it has
     to use those shapes; a lone circle cannot detect it. */
  const hgrad = Math.max(extent * 1e-6, 1e-12);
  const snapTo = Math.max(tolerance * 0.05, extent * 1e-12);
  const snap = (pts) => pts.map(([x, y]) => {
    let px2 = x, py2 = y;
    for (let s2 = 0; s2 < 12; s2 += 1) {
      const f = fieldAt(discs, kUse, px2, py2);
      if (Math.abs(f) <= snapTo) break;
      const g = blobGradAt(discs, kUse, px2, py2, hgrad, fieldAt);
      const g2 = g[0] * g[0] + g[1] * g[1];
      if (!(g2 > 1e-18)) break;
      px2 -= (f * g[0]) / g2; py2 -= (f * g[1]) / g2;
    }
    return [px2, py2];
  });

  /* Stations by arc length, dense enough that the chord never leaves the true
     arc by more than the document tolerance. For a circle of radius r sampled n
     ways the sagitta is r(1 - cos(pi/n)), so n = pi*sqrt(r / (2*tol)). */
  const sag = Math.max(tolerance, 1e-6);
  const byTol = Math.ceil(Math.PI * Math.sqrt(Math.max(rMax, 1e-9) / (2 * sag)));
  /* The cap is 512 so the tolerance still governs large blobs: at r=200 and
     above, a cap of 256 binds for tolerances of both 0.001 and 0.01, leaving
     a fixed 1.48 degrees of turn per station whatever the size; 512 gives 0.70
     degrees. It is still a cap, because these stations are the interpolated
     curve's control points and a blob handed downstream to Puff or Trim has
     to stay a workable curve. */
  const n = Math.max(24, Math.min(512, Number.isFinite(byTol) ? byTol : 64));
  const resample = (pts) => {
    const cum = [0];
    for (let i = 1; i <= pts.length; i += 1) {
      const a = pts[i - 1], b = pts[i % pts.length];
      cum.push(cum[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1]));
    }
    const per = cum[pts.length];
    if (!(per > 0)) return pts.slice();
    const res = [];
    let i = 0;
    for (let s = 0; s < n; s += 1) {
      const target = (s * per) / n;
      while (i < pts.length - 1 && cum[i + 1] < target) i += 1;
      const seg = cum[i + 1] - cum[i];
      const t = seg > 0 ? (target - cum[i]) / seg : 0;
      const a = pts[i], b = pts[(i + 1) % pts.length];
      res.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
    return res;
  };

  /* Outer or hole is decided by nesting, not by winding. Which way marching
     squares emits a ring is a property of the case table, and reading it as
     the answer can make a lone disc report zero pieces. Counting how many
     other rings a ring sits inside cannot be wrong that way: even depth is
     material, odd is a hole, and separate pieces both come out at depth 0.
     Winding is then set from that answer — outer counter-clockwise, holes
     clockwise — so a consumer can rely on it. */
  const inside = (ring, x, y) => {
    let hit = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  };
  /* A ring smaller than a cell is the instrument, not the drawing. With Blend
     exactly the gap between two circles the field grazes zero along the whole
     neck instead of crossing it, and marching squares finds a shower of specks
     there rather than one join (six circles with a 12mm gap: 22 pieces at
     Blend 12, one piece either side). Blend is a slider, so that value is
     passed through on the way to any other. A contour enclosing less than one
     cell of area is below what this grid can resolve, so it is dropped — the
     disc-too-small argument applied to the output instead of the input. */
  const cellArea = hx * hy;
  const shaped = rings.map((r) => snap(resample(r)))
    .filter((r) => r.length >= 3 && Math.abs(AREA(r)) > cellArea);
  if (!shaped.length) return { ok: false, reason: 'open', why: 'the contour did not close' };
  const out = shaped.map((pts, i) => {
    let depth = 0;
    for (let k = 0; k < shaped.length; k += 1) {
      if (k !== i && Math.abs(AREA(shaped[k])) > Math.abs(AREA(pts)) && inside(shaped[k], pts[0][0], pts[0][1])) depth += 1;
    }
    const outer = depth % 2 === 0;
    const ccw = AREA(pts) > 0;
    return { pts: outer === ccw ? pts : pts.slice().reverse(), area: Math.abs(AREA(pts)), outer };
  });
  out.sort((a, b) => b.area - a.area);

  /* Which discs did not make it. A disc far smaller than the rest can be
     smaller than the grid resolves, and then contributes nothing; it is
     reported by name. Asking whether its center landed inside the material
     catches it wherever it sits. */
  const undrawn = [];
  for (let i = 0; i < discs.length; i += 1) {
    let d = 0;
    for (const r of out) if (inside(r.pts, discs[i][0], discs[i][1])) d += 1;
    if (d % 2 === 0) undrawn.push(i);
  }
  const pieces = out.filter((r) => r.outer).length;
  return {
    ok: true, rings: out, pieces, holes: out.length - pieces,
    cells: nx, stations: n, stationsWanted: byTol, undrawn, blend, tolerance,
  };
}

/* Balls that cannot reach each other get their own grid.
   The grid is sized to hold every ball, and its cell count is capped — so one
   ball dragged far enough stretches the same lattice over the whole span and
   the cell grows past the size of a ball (three 12mm balls with one moved out:
   correct at 500mm and 2000mm; at 9m the far ball is smaller than a cell and
   drops out of the outline; at 50m the whole blob refuses).
   Two balls can only affect one another within the reach of the melt, which
   is a closed-form question about the discs: centers closer than r_i + r_j
   plus the widest melt in play. Grouping on that and contouring each group
   over its own tight box keeps every group's resolution, and is faster —
   several small grids instead of one enormous one. */
export function blobClusters(discs, reach) {
  const parent = discs.map((_, i) => i);
  const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
  for (let i = 0; i < discs.length; i += 1) {
    for (let j = i + 1; j < discs.length; j += 1) {
      const d = Math.hypot(discs[i][0] - discs[j][0], discs[i][1] - discs[j][1]);
      if (d <= discs[i][2] + discs[j][2] + reach) {
        const a = find(i), b = find(j);
        if (a !== b) parent[a] = b;
      }
    }
  }
  const groups = new Map();
  for (let i = 0; i < discs.length; i += 1) {
    const r = find(i);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(i);
  }
  return [...groups.values()];
}

export function blobOutline(discs, opts = {}) {
  if (!Array.isArray(discs) || discs.length === 0) {
    return { ok: false, reason: 'no-discs', why: 'a blob needs at least one disc' };
  }
  for (const d of discs) {
    if (!Array.isArray(d) || d.length < 3 || !Number.isFinite(d[0]) || !Number.isFinite(d[1]) || !(d[2] > 0)) {
      return { ok: false, reason: 'bad-disc', why: 'every disc needs a centre and a radius above zero' };
    }
  }
  const blend = Math.max(0, opts.blend ?? 0);
  // The widest melt anything might use, so the grouping never separates two
  // balls that would have joined.
  let reach = blend;
  for (const d of discs) if (d.length > 3 && d[3] != null && Number.isFinite(d[3])) reach = Math.max(reach, d[3]);
  const groups = blobClusters(discs, reach * 2 + 1e-9);
  if (groups.length === 1) return blobOutlineOneCluster(discs, opts);

  const rings = [];
  const undrawn = [];
  let cells = 0, stations = 0, stationsWanted = 0, refusals = 0, lastWhy = '';
  for (const g of groups) {
    const sub = g.map((i) => discs[i]);
    const res = blobOutlineOneCluster(sub, opts);
    if (!res.ok) { refusals += 1; lastWhy = res.why; for (const i of g) undrawn.push(i); continue; }
    rings.push(...res.rings);
    for (const k of res.undrawn) undrawn.push(g[k]);
    cells = Math.max(cells, res.cells);
    stations = Math.max(stations, res.stations);
    stationsWanted = Math.max(stationsWanted, res.stationsWanted);
  }
  /* A group that refuses does not take the others with it: one unusable ball
     leaves the rest of the drawing standing; only if every group fails is
     there nothing to draw. */
  if (!rings.length) return { ok: false, reason: 'empty', why: lastWhy || 'those discs enclose nothing to draw a curve around' };
  rings.sort((a, b) => b.area - a.area);
  const pieces = rings.filter((r) => r.outer).length;
  return {
    ok: true, rings, pieces, holes: rings.length - pieces,
    cells, stations, stationsWanted, undrawn: undrawn.sort((a, b) => a - b),
    blend, tolerance: opts.tolerance ?? 0.01, groups: groups.length, refusedGroups: refusals,
  };
}
