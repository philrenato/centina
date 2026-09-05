// A CLOSED OUTLINE FROM A HANDFUL OF DISCS.
//
// Draw two or three circles that overlap and the contour around them is a
// single soft shape — the crotch where they meet is rounded rather than
// notched, and a small gap is bridged. It is the cheapest way to get an organic
// closed outline, and this app wants one because Puff turns a closed outline
// into a solid.
//
// ⚠⚠ THE BLEND IS THE FUSE'S, NOT A SECOND ONE. `solidwrap.mjs` already carries
// this app's one smooth minimum, and its Blend row already makes a promise a
// reader has learned: "two surfaces a gap g apart meet at Blend g". Inventing a
// different 2D blend here — a metaball field summed to an iso level, which is
// the usual way — would mean the same word meant two things in two panels, and
// the sibling app it was copied from has to bisect an iso level per frame to
// keep its own promise. Folding the same `smoothMinPoly` over 2D disc distances
// keeps the promise by construction, and it was measured to hold: two r = 10
// discs bridge at a gap of exactly 2.00, 5.00 and 10.00mm at Blend 2, 5 and 10.
//
// The contour is marching squares over a grid, then every vertex is pushed onto
// the true zero set by Newton steps along the gradient — so the grid decides
// where the contour is FOUND and not where it SITS, and its resolution stops
// being visible in the result.
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

/* ⚠⚠ A PER-BALL MELT, AND THE ORDER IS THE MECHANISM. Each ball may carry its
   own melt (element 3; null or absent means "follow the global"), and what two
   balls should do where they meet is governed by the SOFTER of the two — a hard
   ball stays hard against anything.
   Folding in DECREASING melt makes that true for free: by the time ball i is
   folded in, every ball already in the accumulator has a melt at least as large,
   so the step's own k IS min(b_i, b_j) for every pair it stands for. Measured,
   bridging then happens at min(b_i,b_j) equal to the gap to within 1.1e-3mm
   across mixed settings.
   Three spellings were measured and rejected. A per-ball MULTIPLIER misses the
   bridge by +100% at 0.5 and +400% at 0.2. A running min(k) lets a hard ball
   130mm away, merely listed between two others, pull them from one piece into
   two. A partition-of-unity k breaks scale neutrality by 1.0937x.
   ⚠ AND THE SORT IS CONDITIONAL. smoothMinPoly is not associative, so reordering
   the balls moves the outline — measured at 1.46mm today. Sorting fixes that
   (0.00mm), but applying it unconditionally would move every blob already saved
   by up to 3.2mm. It runs only when the melts actually differ, which is exactly
   when it changes anything, so an untouched document stays bit-identical. */
export function blobFieldMixed(discs, k, ks, x, y) {
  let acc = Infinity;
  for (let i = 0; i < discs.length; i += 1) {
    const d = Math.hypot(x - discs[i][0], y - discs[i][1]) - discs[i][2];
    // ks[i] IS the pairwise minimum, because the fold order guarantees every
    // ball already in `acc` has a melt at least this large.
    acc = i === 0 ? d : smoothMinPoly(acc, d, ks[i]);
  }
  return acc;
}

/* The per-ball melt each ball actually gets: its own if it has one, otherwise
   the global — and never MORE than the global, because the minimum governs and a
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
 * `discs` is [[x, y, r], ...] in millimetres. Returns every ring the field
 * produces, not only the biggest: separate pieces and holes are real answers —
 * two discs far apart ARE two pieces — and a caller that only received the
 * largest would silently drop what a reader drew.
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
  /* ⚠ THE GRID IS SIZED OFF THE SMALLEST DISC, not off the drawing. A fixed
     count over the bounding box cannot see a disc much smaller than a cell, and
     the failure is silent — the disc simply is not in the outline. Four cells
     across the smallest radius is the floor; the clamp keeps a pathological
     ratio from asking for a grid nobody can afford. */
  /* ⚠ THE CELL IS TIED TO THE SMALLEST DISC, AND THE PAD IS PART OF THE BOX.
     A count taken over the drawing alone cannot see a disc much smaller than a
     cell, and the failure is silent — the disc is simply not in the outline. The
     pad has to be counted too: Blend widens the domain, and sizing the grid
     before adding it let a large Blend quietly coarsen every cell. */
  const pad = blend + 0.05 * extent;
  lo = [lo[0] - pad, lo[1] - pad]; hi = [hi[0] + pad, hi[1] + pad];
  const boxed = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
  const want = Math.round((4 * boxed) / Math.max(rMin, 1e-9));
  const cells = Math.max(192, Math.min(384, Number.isFinite(want) ? want : 192));
  const nx = cells;
  const ny = Math.max(8, Math.round((cells * (hi[1] - lo[1])) / Math.max(hi[0] - lo[0], 1e-9)));
  const hx = (hi[0] - lo[0]) / nx, hy = (hi[1] - lo[1]) / ny;

  /* ⚠⚠ A BLEND WITHIN A HAIR OF A GAP DRAWS A NECK THINNER THAN ONE CELL, and
     no grid can draw that. The zero set stops being a curve and pinches; the
     contour through the pinch is cut into sub-cell islands and fragments.
     Measured on six circles with a 12mm gap: whole at 4439 and 4482mm2 either
     side, and inside a band of about 0.015mm around Blend 12 it shatters into
     22 contours totalling 2982 — a third of the drawing gone. Neither the
     saddle rule nor the segment chaining is at fault; both were replaced and
     measured, and neither moved the number by a millimetre.
     The condition is a closed-form question about the discs and needs no grid
     to ask: it is degenerate when Blend is within a cell of some pair's gap. It
     is nudged to the JOINED side, by a fraction of one cell — far below the
     document tolerance, and the direction the Blend row's own sentence already
     promises: at Blend g, a gap of g is bridged. */
  /* WHICH FIELD THIS BLOB USES. With one melt for every ball the original fold
     is exact and is left alone byte for byte; only a blob whose balls disagree
     pays for the sorted mixed fold. */
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
     cell's own centre value rather than by a fixed choice, so a neck that is
     genuinely joined does not come apart at one cell. */
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
  /* ⚠ AT A SHARED VERTEX, TAKE THE TIGHTEST TURN. Set Blend to exactly the gap
     between two circles and they touch at a POINT — four contour segments meet
     at one vertex there. Taking whichever segment came first walks a
     figure-eight through that vertex, consuming the segments of BOTH loops into
     one traversal and discarding what is left: measured on six circles with a
     12mm gap, four rings totalling 2982mm2 where either side of 12.0 measures
     ~4400. Two whole rings simply vanished from a drawing at one slider value.
     Whether two circles that touch at a point are one shape or two is genuinely
     ambiguous at that measure-zero value, and no epsilon small enough to stay
     invisible can settle it — the join is thinner than any grid this can afford
     to sample. What is NOT ambiguous is that nothing may disappear. Turning as
     tightly as possible keeps each loop walking its own side of the vertex, so
     the pieces stay whole whichever way the count falls. */
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

  /* THE GRID FINDS THE CONTOUR; NEWTON PUTS IT WHERE IT BELONGS — AND IT RUNS
     LAST. Snapping BEFORE the stations are laid out fixes the wrong points: the
     stations are then chords between snapped points, and on a 1.2m drawing they
     sat 0.1mm off a contour whose own vertices were exact.
     ⚠ It earns nothing on a lone disc — measured identical to five figures at
     both 0.5mm and 10mm — because marching squares interpolates a linear field
     exactly and a distance field is very nearly linear across one cell. What it
     buys is where the field CURVES: 6x across a blend crease, 15x around an
     annulus, and 160-226x on a wide scale spread. A gate for it therefore has to
     use those shapes; a lone circle cannot tell whether this loop is here. */
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
  const n = Math.max(24, Math.min(256, Number.isFinite(byTol) ? byTol : 64));
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

  /* ⚠ OUTER OR HOLE IS DECIDED BY NESTING, NOT BY WINDING. Which way marching
     squares emits a ring is a property of the case table, and reading it as the
     answer made a lone disc report zero pieces. Counting how many other rings a
     ring sits inside cannot be wrong that way: even depth is material, odd is a
     hole, and separate pieces both come out at depth 0. Winding is then SET from
     that answer — outer counter-clockwise, holes clockwise — so a consumer can
     rely on it. */
  const inside = (ring, x, y) => {
    let hit = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  };
  /* ⚠ A RING SMALLER THAN A CELL IS THE INSTRUMENT, NOT THE DRAWING. Set Blend
     to exactly the gap between two circles and the field grazes zero along the
     whole neck instead of crossing it; marching squares then finds a shower of
     specks there rather than one join. Measured: six circles with a 12mm gap
     came out as 22 pieces at Blend 12, one piece either side of it. Blend is a
     slider, so a reader drags THROUGH that value on the way to any other, and
     the shatter is what they would see. A contour enclosing less than one cell
     of area is below what this grid can resolve at all, so it is dropped —
     which is the same argument the disc-too-small report already makes, applied
     to the output instead of the input. */
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

  /* WHICH DISCS DID NOT MAKE IT. A disc far smaller than the rest is smaller
     than the grid can resolve, and it then contributes nothing at all — the
     honest thing is to name it rather than leave a reader wondering why their
     circle did nothing. Asking whether its centre landed inside the material is
     the test that catches it wherever it sits. */
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

/* ⚠⚠ BALLS THAT CANNOT REACH EACH OTHER GET THEIR OWN GRID, and without this the
   tool breaks the moment a reader drags one ball away.
   The grid is sized to hold every ball, and its cell count is capped — so one
   ball dragged far enough stretches the same lattice over the whole span and the
   cell grows past the size of a ball. Measured, three 12mm balls with one moved
   out: at 500mm and 2000mm everything is correct; at 9m the far ball is smaller
   than a cell and VANISHES from the outline entirely; at 50m the whole blob
   refuses and there is no shape at all. A reader dragging a ball would watch the
   thing stop working with no way to know why.
   Two balls can only ever affect one another within the reach of the melt, which
   is a closed-form question about the discs: centres closer than r_i + r_j plus
   the widest melt in play. Grouping on that and contouring each group over its
   own tight box means a distant ball keeps its own resolution, the balls it left
   behind keep theirs, and nothing is capped over a span nobody asked for. It is
   also faster — several small grids instead of one enormous one. */
function blobClusters(discs, reach) {
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
  // The widest melt anything might use, so the grouping can never separate two
  // balls that would in fact have joined.
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
  /* ⚠ A GROUP THAT REFUSES DOES NOT TAKE THE OTHERS WITH IT. The whole point of
     separating them is that one unusable ball leaves the rest of the drawing
     standing; only if EVERY group fails is there nothing to draw. */
  if (!rings.length) return { ok: false, reason: 'empty', why: lastWhy || 'those discs enclose nothing to draw a curve around' };
  rings.sort((a, b) => b.area - a.area);
  const pieces = rings.filter((r) => r.outer).length;
  return {
    ok: true, rings, pieces, holes: rings.length - pieces,
    cells, stations, stationsWanted, undrawn: undrawn.sort((a, b) => a - b),
    blend, tolerance: opts.tolerance ?? 0.01, groups: groups.length, refusedGroups: refusals,
  };
}
