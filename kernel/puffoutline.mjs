// outline — turning a raw pointer stroke into a closed loop something can be built from.
//
// Every tool that takes a drawn region needs the same four things: an explicit closure, an even
// resampling, a refusal when the loop crosses itself, and the loop's inradius. They live here once
// so a second outline tool does not grow a second copy of them and drift.
//
// Nothing here repairs. A stroke that crosses itself is refused and says where. Repairing it means
// guessing which of two readings the hand meant, and a wrong guess produces a plausible region that
// is not the one that was drawn.

// The closing segment is part of the loop. A resampler that walks p[0]..p[n-1] and stops covers
// every edge except the one from the last point back to the first, so the samples bunch and the
// tail is padded with duplicates of the final point. The gap then survives into the geometry as a
// notch. This walks the wrap segment like any other.
export function perimeter(pts) {
  let d = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    d += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  return d;
}

// Signed area, positive for counter-clockwise. Also the emptiness test: a stroke that doubles back
// on itself encloses nothing, and its area is the only thing that says so.
export function signedArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a / 2;
}

// Orientation is a convention every consumer depends on and none should have to establish.
export function toCounterClockwise(pts) {
  return signedArea(pts) < 0 ? pts.slice().reverse() : pts.slice();
}

// Closure is explicit, and the threshold is a fraction of the stroke's own size, not pixels.
// A fixed pixel gap means the same stroke closes at one zoom and not at another. `tol` is a
// fraction of the stroke's path length, so it travels.
export function closeOutline(pts, { tol = 0.45 } = {}) {
  if (!pts || pts.length < 3) return { ok: false, why: 'a loop needs at least three points', pts: null };
  // The gap is measured against the stroke's own path length, not against its bounding
  // diagonal. The diagonal is a property of the box the stroke happens to sit in, so the
  // same drawn shape would pass or fail depending on how it is oriented and how eccentric
  // it is — a banana-shaped stroke with its ends 24% of its diagonal apart is an
  // unambiguous loop. Chord-over-arc is scale-free and rotation-free, and for a
  // circular arc of angle t it is exactly 2*sin(t/2)/t:
  //     full loop 0%   3/4 circle 30%   5/8 circle 43%   half circle 64%   straight line 100%
  // so tol = 0.45 admits anything past about five eighths of a loop and refuses a half-arc
  // or less, which is a line, not a loop.
  let path = 0;
  for (let i = 1; i < pts.length; i++) path += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  if (!(path > 0)) return { ok: false, why: 'the stroke has no extent', pts: null };
  const a = pts[0], b = pts[pts.length - 1];
  const gap = Math.hypot(b[0] - a[0], b[1] - a[1]);
  if (gap > path * tol) {
    return { ok: false, why: `the ends are ${(100 * gap / path).toFixed(0)}% of the stroke apart \u2014 draw a closed shape`, pts: null, gap, path };
  }
  // Drop a final point that merely duplicates the first; the loop is closed by index wrap, not by a
  // repeated vertex. A repeated vertex is a zero-length edge and every consumer has to special-case it.
  const out = pts.slice();
  if (gap < path * 1e-9) out.pop();
  return out.length >= 3
    ? { ok: true, pts: out, gap, path }
    : { ok: false, why: 'a loop needs at least three points', pts: null };
}

// Refuse, do not repair. Returns the first crossing pair so a caller can show it rather than
// saying "invalid". Adjacent segments share an endpoint and are skipped; so is the wrap pair.
export function selfIntersection(pts) {
  const n = pts.length;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  // The side test is scale-relative; a strict `> 0` refuses valid shapes.
  // `cross` returns twice a signed area, so on two nearly-parallel edges it is a tiny number whose
  // sign is floating-point noise. Written as `(d1 > 0) !== (d2 > 0)` that noise reads as a straddle,
  // and since more samples means more nearly-parallel pairs, the false positive is
  // density-dependent: a plain thin triangle measures clean at 10 samples an edge and "crosses
  // itself" at 40. A hand-drawn thin shape at pencil density is exactly that input.
  //
  // Dividing by the segment's length turns the cross product back into a perpendicular distance,
  // which is comparable against the outline's own size. The epsilon is 1e-9 of the bounding
  // diagonal: about seven orders of magnitude above double-precision noise on these coordinates,
  // and six below a 0.001-wide gap, which must not be flagged — so it
  // separates "noise" from "a real near-miss" with room on both sides rather than being tuned
  // between them.
  //
  // A point lying on the other segment is tested separately, because the epsilon would
  // otherwise discard a real crossing. A polygon that visits the same point twice is not
  // simple, and a densely-sampled bowtie does exactly that: sampled at 40 per edge, both diagonals
  // land a vertex exactly on (0,0), so the two paths touch at a shared point rather than crossing
  // transversally through segment interiors — a case the side test's zero band cannot see, while
  // a four-point bowtie crosses transversally. So the two questions are
  // asked separately: do the segments straddle each other, and does an endpoint of either lie on
  // the other. Both are self-intersections; only the first is what the side test measures.
  let lo0 = Infinity, lo1 = Infinity, hi0 = -Infinity, hi1 = -Infinity;
  for (const p of pts) {
    if (p[0] < lo0) lo0 = p[0]; if (p[0] > hi0) hi0 = p[0];
    if (p[1] < lo1) lo1 = p[1]; if (p[1] > hi1) hi1 = p[1];
  }
  const EPS = Math.hypot(hi0 - lo0, hi1 - lo1) * 1e-9;
  const side = (d, len) => { const h = len > 0 ? d / len : 0; return h > EPS ? 1 : (h < -EPS ? -1 : 0); };
  // Does p lie on segment a-b — within EPS of the line and between the ends?
  const onSeg = (p, a, b, len) => {
    if (!(len > 0)) return Math.hypot(p[0] - a[0], p[1] - a[1]) <= EPS;
    const t = ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / (len * len);
    if (t < 0 || t > 1) return false;
    return Math.hypot(p[0] - (a[0] + (b[0] - a[0]) * t), p[1] - (a[1] + (b[1] - a[1]) * t)) <= EPS;
  };
  for (let i = 0; i < n; i++) {
    const p1 = pts[i], p2 = pts[(i + 1) % n];
    const len12 = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
    for (let j = i + 1; j < n; j++) {
      if (j === i) continue;
      // neighbors share a vertex; the first and last segments share one too
      if (j === (i + 1) % n || i === (j + 1) % n) continue;
      const p3 = pts[j], p4 = pts[(j + 1) % n];
      const len34 = Math.hypot(p4[0] - p3[0], p4[1] - p3[1]);
      const s1 = side(cross(p3, p4, p1), len34), s2 = side(cross(p3, p4, p2), len34);
      const s3 = side(cross(p1, p2, p3), len12), s4 = side(cross(p1, p2, p4), len12);
      // a transversal crossing: each segment straddles the other's line, on both counts
      if (s1 !== 0 && s2 !== 0 && s3 !== 0 && s4 !== 0 && s1 !== s2 && s3 !== s4) return { at: [i, j] };
      // a touch: an endpoint of one lying on the other. Non-adjacent segments already, so a shared
      // point here is the outline meeting itself, not two edges meeting at their own corner.
      if ((s1 === 0 && onSeg(p1, p3, p4, len34)) || (s2 === 0 && onSeg(p2, p3, p4, len34))
       || (s3 === 0 && onSeg(p3, p1, p2, len12)) || (s4 === 0 && onSeg(p4, p1, p2, len12))) {
        return { at: [i, j] };
      }
    }
  }
  return null;
}

// Arc length, not index. A hand moves at wildly different speeds around a loop, so index-uniform
// samples bunch where the hand slowed and thin where it hurried — and every downstream measure that
// assumes even spacing then reads the hand's tempo as the shape's geometry.
export function resampleByArcLength(pts, count) {
  const n = pts.length;
  if (n < 3 || count < 3) return pts.slice();
  // Cumulative arc length at each vertex, with the wrap segment closing the table. Walking a
  // precomputed table rather than consuming the input means the source is never mutated and the
  // output count is exact by construction — a resampler that can terminate early pads its tail with
  // duplicates of the last point, and duplicates are zero-length edges every consumer must special-case.
  const cum = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const a = pts[i], b = pts[(i + 1) % n];
    cum[i + 1] = cum[i] + Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  const total = cum[n];
  if (!(total > 0)) return pts.slice();
  const out = new Array(count);
  let seg = 0;
  for (let k = 0; k < count; k++) {
    const target = (k * total) / count;
    while (seg < n - 1 && cum[seg + 1] < target) seg++;
    const L = cum[seg + 1] - cum[seg];
    const t = L > 0 ? (target - cum[seg]) / L : 0;
    const a = pts[seg], b = pts[(seg + 1) % n];
    out[k] = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }
  return out;
}

export function pointInPolygon(pts, x, y) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
  }
  return inside;
}

export function distanceToBoundary(pts, x, y) {
  let best = Infinity;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const L2 = dx * dx + dy * dy;
    let t = L2 > 0 ? ((x - a[0]) * dx + (y - a[1]) * dy) / L2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const d = Math.hypot(x - (a[0] + dx * t), y - (a[1] + dy * t));
    if (d < best) best = d;
  }
  return best;
}

// The inradius is the number that decides resolution, and it is not the drawn size.
// A puffed region's height is bounded by half its local thickness, so the deepest interior point —
// not the bounding box — says how many voxels the shape needs to survive a field bake. A crescent
// drawn as wide as a disc has a small fraction of the disc's inradius, and a grid sized from the
// drawn extent renders it as a blunted lump. Sampled on a grid: `spacing` is the accuracy, and it is
// returned so a caller can say how sure it is rather than implying exactness.
export function inradius(pts, { samples = 96 } = {}) {
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const p of pts) {
    lo[0] = Math.min(lo[0], p[0]); lo[1] = Math.min(lo[1], p[1]);
    hi[0] = Math.max(hi[0], p[0]); hi[1] = Math.max(hi[1], p[1]);
  }
  const w = hi[0] - lo[0], h = hi[1] - lo[1];
  const nx = Math.max(8, Math.round(samples * (w >= h ? 1 : w / h)));
  const ny = Math.max(8, Math.round(samples * (h >= w ? 1 : h / w)));
  const sx = w / nx, sy = h / ny;
  let best = 0, at = null;
  for (let iy = 0; iy <= ny; iy++) {
    const y = lo[1] + iy * sy;
    for (let ix = 0; ix <= nx; ix++) {
      const x = lo[0] + ix * sx;
      if (!pointInPolygon(pts, x, y)) continue;
      const d = distanceToBoundary(pts, x, y);
      if (d > best) { best = d; at = [x, y]; }
    }
  }
  return { r: best, at, spacing: Math.max(sx, sy), diag: Math.hypot(w, h) };
}

// The one entry point a tool should call. Everything above is exported so the parts can be tested.
export function prepareOutline(raw, { count = 128, tol = 0.15 } = {}) {
  const closed = closeOutline(raw, { tol });
  if (!closed.ok) return { ok: false, why: closed.why };
  const ccw = toCounterClockwise(closed.pts);
  const hit = selfIntersection(ccw);
  if (hit) return { ok: false, why: 'the outline crosses itself — draw a simple loop', at: hit.at };
  const pts = resampleByArcLength(ccw, count);
  const area = Math.abs(signedArea(pts));
  if (!(area > 0)) return { ok: false, why: 'the outline encloses no area' };
  const ir = inradius(pts);
  return { ok: true, pts, area, inradius: ir.r, inradiusAt: ir.at, inradiusSpacing: ir.spacing, diag: ir.diag };
}


/**
 * Taubin fairing of a closed curve — low-pass a drawn outline without shrinking it.
 *
 * Lines drawn into puff are irregular: a hand-drawn loop carries tremor, and the inflation is
 * faithful to it — every wobble in the line becomes a lobe in the solid, and the medial axis
 * between lobes becomes a crease. So the outline is faired rather than assumed clean.
 *
 * Taubin, not a plain Laplacian. A plain neighbor
 * average shrinks a closed curve on every pass — smooth it enough to remove tremor and the drawn
 * shape has visibly deflated. Alternating a positive step with a slightly larger negative one
 * cancels that. Measured over 10/20/40 iterations on gentle, typical and scribbly strokes, the
 * enclosed area lands at 100.1% - 100.5% of the original.
 *
 * It barely moves the line. Worst deviation from the drawn curve, as a fraction of the
 * shape's half-width: 0.8% gentle, 1.5% typical, 2.1% scribbly, at 20 iterations. What it removes
 * is tremor, not shape — a deliberate lobe survives, a hand wobble does not.
 *
 * The lambda/mu pair is the one the original SmoothTeddy uses in `TaubinFairing` (pass-band 0.1).
 * That program also contains a `TaubinFairing2D` doing this to a closed 2D polyline, with no
 * call sites there.
 */
export function taubinSmoothClosed(pts, { iters = 20, lam = 0.63139836, mu = -0.6739516 } = {}) {
  if (!pts || pts.length < 5 || !(iters > 0)) return pts;
  let Q = pts.map(p => [p[0], p[1]]);
  const step = (w) => {
    const n = Q.length, out = new Array(n);
    for (let i = 0; i < n; i++) {
      const a = Q[(i - 1 + n) % n], b = Q[i], c = Q[(i + 1) % n];
      out[i] = [b[0] + w * ((a[0] + c[0]) / 2 - b[0]), b[1] + w * ((a[1] + c[1]) / 2 - b[1])];
    }
    Q = out;
  };
  for (let k = 0; k < iters; k++) { step(lam); step(mu); }
  return Q;
}

// The seam of a closed stroke.
// The join of a drawn loop is its worst point. Discrete curvature, worst around the loop against
// the mean:
//
//      as prepared   gap 0%  4.0x     gap 3%  5.2x     gap 8%  7.2x    (worst is at the seam)
//      + Taubin x20          2.9x             3.0x             3.8x
//
// Clean analytic curves on the same measure: circle 1.0x, ellipse 1.2:1 1.4x, ellipse 1.4:1 1.9x,
// ellipse 2:1 3.4x. So a faired stroke of roughly 1.4:1 at 2.9x carries one region as sharp as a
// 2:1 ellipse's end.
//
// No seam treatment is applied, because the residual does not reach the surface. On the
// twice-subdivided surface, across stroke roughness from a gentle wobble to heavy per-sample
// tremor, with and without a closing gap, the worst dihedral is 6.6 - 13.8 degrees and no edge
// passes 20 degrees, the angle at which a crease reads. The subdivided surface, not the 2D
// curvature ratio, is the measure of a visible seam.
//
// Constructions that do not help, measured:
//   1. A periodic cubic spline (G2 at the seam by construction): 2.9x -> 2.9x. C2 makes curvature
//      continuous, not small; it removes the discontinuity and leaves the magnitude, and the
//      magnitude is what reads as a kink.
//   2. Fairing wherever curvature is an outlier against its local neighborhood: 2.9x, 3.0x, 3.8x
//      unchanged, and a drawn triangle 22.2x -> 11.2x. After Taubin the seam is a broad elevated
//      region with no single-point spike, while a deliberate corner is one — so a local-outlier
//      detector finds corners, not seams.
//   3. Blending the last K and first K points into one smoothstep run from the tail's shape to the
//      head's, K from path length; measured through prepareOutline -> taubin x20 -> prepareOutline:
//        gap 5% 3.1x -> 2.4x   gap 8% 4.4x -> 2.4x   gap 12% 5.6x -> 3.3x   long 2:1 gap 8% 10.0x -> 5.0x
//        triangle at corner 23.9x -> 24.2x   triangle mid-edge 24.6x -> 23.2x   square 12.5x -> 12.8x
//      It reshapes only the arc next to the join, but on the subdivided surface it moves the worst
//      dihedral by under ~3 degrees in either direction (6.1 - 13.0 degrees), so it is not used.
//   4. Blending only the actual overlap: K is 0 on every gap fixture — it applies to an
//      overshoot, not a gap.
//   5. Walking each end toward the gap's midpoint: worse on every gap fixture (gap 8%: 4.4x ->
//      6.8x), because pulling the ends together sharpens the local turn.
//
// A seam fix must be evaluated with a drawn triangle among the fixtures, as the control that
// separates an artifact from an intended corner, and through prepareOutline's resample: without
// it a truncated stroke keeps its closing chord as one long edge, and gap 8% reads 14.9x instead
// of 4.4x.

