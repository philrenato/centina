// outline — turning a raw pointer stroke into a closed loop something can be built from.
//
// Every tool that takes a drawn region needs the same four things: an explicit closure, an even
// resampling, a refusal when the loop crosses itself, and the loop's INRADIUS. They live here once
// so a second outline tool does not grow a second copy of them and drift.
//
// ⚠ NOTHING HERE REPAIRS. A stroke that crosses itself is refused and says where. Repairing it means
// guessing which of two readings the hand meant, and a wrong guess produces a plausible region that
// is not the one that was drawn — which is the failure this module exists to make impossible.

// ⚠ THE CLOSING SEGMENT IS PART OF THE LOOP. A resampler that walks p[0]..p[n-1] and stops covers
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

// ⚠ CLOSURE IS EXPLICIT, AND THE THRESHOLD IS A FRACTION OF THE STROKE'S OWN SIZE, not pixels.
// A fixed pixel gap means the same stroke closes at one zoom and not at another. `tol` is a
// fraction of the bounding diagonal, so it travels.
export function closeOutline(pts, { tol = 0.45 } = {}) {
  if (!pts || pts.length < 3) return { ok: false, why: 'a loop needs at least three points', pts: null };
  // The gap is measured against the stroke's OWN PATH LENGTH, not against its bounding
  // diagonal. The diagonal is a property of the box the stroke happens to sit in, so the
  // same drawn shape passed or failed depending on how it was oriented and how eccentric
  // it was — a banana-shaped stroke was refused at 24% of its diagonal while being a
  // perfectly unambiguous loop. Chord-over-arc is scale-free AND rotation-free, and for a
  // circular arc of angle t it is exactly 2*sin(t/2)/t:
  //     full loop 0%   3/4 circle 30%   5/8 circle 43%   half circle 64%   straight line 100%
  // so tol = 0.45 admits anything past about five eighths of a loop and refuses a half-arc
  // or less, which is a line, not a loop. (The old message already read "% of the stroke
  // apart" — it described this test while the code measured the box.)
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

// ⚠ REFUSE, DO NOT REPAIR. Returns the first crossing pair so a caller can SHOW it rather than
// saying "invalid". Adjacent segments share an endpoint and are skipped; so is the wrap pair.
export function selfIntersection(pts) {
  const n = pts.length;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  // ⚠⚠ THE SIDE TEST IS SCALE-RELATIVE, AND A STRICT `> 0` WAS FALSELY REFUSING REAL SHAPES.
  // `cross` returns twice a signed area, so on two NEARLY-PARALLEL edges it is a tiny number whose
  // SIGN is floating-point noise. Written as `(d1 > 0) !== (d2 > 0)` that noise reads as a straddle,
  // and since more samples means more nearly-parallel pairs, the false positive is
  // DENSITY-DEPENDENT: a plain thin triangle measured clean at 10 samples an edge and "crosses
  // itself" at 40. A hand-drawn thin shape at pencil density is exactly that input, and the puff
  // tool it feeds refused it by name.
  //
  // Dividing by the segment's length turns the cross product back into a PERPENDICULAR DISTANCE,
  // which is comparable against the outline's own size. The epsilon is 1e-9 of the bounding
  // diagonal: about seven orders of magnitude above double-precision noise on these coordinates,
  // and six below the 0.001-wide gap section D of the gate requires NOT to be flagged — so it
  // separates "noise" from "a real near-miss" with room on both sides rather than being tuned
  // between them.
  //
  // ⚠⚠ AND A POINT LYING ON THE OTHER SEGMENT IS TESTED SEPARATELY, BECAUSE THE EPSILON WOULD
  // OTHERWISE HAVE THROWN AWAY A REAL CROSSING. A polygon that visits the same point twice is not
  // simple, and a densely-sampled bowtie does exactly that: sampled at 40 per edge, both diagonals
  // land a vertex EXACTLY on (0,0), so the two paths TOUCH at a shared point rather than crossing
  // transversally through segment interiors. The strict test caught that by accident of sign
  // convention; suppressing the zero side lost it, and the gate could not see the loss because its
  // own bowtie is four raw points, where the crossing IS transversal. So the two questions are
  // asked separately: do the segments straddle each other, and does an endpoint of either lie on
  // the other. Both are self-intersections; only the first is what the side test measures.
  let lo0 = Infinity, lo1 = Infinity, hi0 = -Infinity, hi1 = -Infinity;
  for (const p of pts) {
    if (p[0] < lo0) lo0 = p[0]; if (p[0] > hi0) hi0 = p[0];
    if (p[1] < lo1) lo1 = p[1]; if (p[1] > hi1) hi1 = p[1];
  }
  const EPS = Math.hypot(hi0 - lo0, hi1 - lo1) * 1e-9;
  const side = (d, len) => { const h = len > 0 ? d / len : 0; return h > EPS ? 1 : (h < -EPS ? -1 : 0); };
  // Does p lie ON segment a-b — within EPS of the line AND between the ends?
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
      // neighbours share a vertex; the first and last segments share one too
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

// ⚠ ARC LENGTH, NOT INDEX. A hand moves at wildly different speeds around a loop, so index-uniform
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

// ⚠ THE INRADIUS IS THE NUMBER THAT DECIDES RESOLUTION, and it is not the drawn size.
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

// The one entry point a tool should call. Everything above is exported so a gate can test the parts.
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
 * TAUBIN FAIRING OF A CLOSED CURVE — low-pass a drawn outline WITHOUT shrinking it.
 *
 * ⚠⚠⚠ WHY THIS EXISTS, in the user's own words: "Lines drawn into puff will always be irregular,
 * blobby." That is the design constraint, not a caveat. A hand-drawn loop carries tremor, and the
 * inflation is faithful to it — every wobble in the line becomes a lobe in the solid, and the
 * medial axis between lobes becomes a crease. The fix cannot be to hope for a clean stroke.
 *
 * ⚠ TAUBIN, NOT A PLAIN LAPLACIAN, AND THE DIFFERENCE IS THE WHOLE POINT. A plain neighbour
 * average shrinks a closed curve on every pass — smooth it enough to remove tremor and the drawn
 * shape has visibly deflated. Alternating a positive step with a slightly LARGER negative one
 * cancels that. Measured over 10/20/40 iterations on gentle, typical and scribbly strokes, the
 * enclosed AREA lands at 100.1% - 100.5% of the original: it does not shrink at all.
 *
 * ⚠ AND IT BARELY MOVES THE LINE. Worst deviation from the drawn curve, as a fraction of the
 * shape's half-width: 0.8% gentle, 1.5% typical, 2.1% scribbly, at 20 iterations. What it removes
 * is tremor, not shape — a deliberate lobe survives, a hand wobble does not.
 *
 * The lambda/mu pair is the one the original SmoothTeddy uses in `TaubinFairing` (pass-band 0.1).
 * That program also ships a `TaubinFairing2D` doing exactly this to a closed 2D polyline; it is
 * dead code there, with zero call sites. It is not dead here.
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

// ── A MEASURED NEGATIVE, KEPT AS A NOTE RATHER THAN AS CODE ──────────────────────────────────────
// A PERIODIC CUBIC SPLINE (G2 at the seam by construction) WAS BUILT HERE AND REMOVED, because it
// changed nothing. The seam of a drawn loop is its worst point — discrete curvature, worst around
// the loop against the mean:
//
//      as prepared   gap 0%  4.0x     gap 3%  5.2x     gap 8%  7.2x    (worst is AT the seam)
//      + Taubin x20          2.9x             3.0x             3.8x
//      + periodic spline     2.9x             3.0x             3.8x    <- no change at all
//
// ⚠ C2 MEANS THE CURVATURE IS CONTINUOUS, NOT SMALL. An interpolating spline through points that
// genuinely turn at the seam still turns there; it removes the DISCONTINUITY and leaves the
// MAGNITUDE. What a person sees as a kink is the magnitude.
//
// ⚠ AND THE CONTROL SAYS THE RESIDUAL IS MODEST. Clean analytic curves, same instrument:
// circle 1.0x, ellipse 1.2:1 1.4x, ellipse 1.4:1 1.9x, ellipse 2:1 3.4x. A faired stroke of
// roughly 1.4:1 sitting at 2.9x is elevated — it carries one region as sharp as a 2:1 ellipse's
// end — but it is not the 7.2x it started at.
//
// WHAT WOULD ACTUALLY FIX IT: reduce the curvature magnitude AT the join specifically — either by
// weighting the fairing toward the seam, or by blending the stroke's ends over an overlap when it
// is closed, so the join is smooth by construction rather than faired-toward-smooth afterwards.
// Neither is built. Do not reach for a higher continuity class again; that has been measured.


// ── THE SECOND MEASURED NEGATIVE ON THE SEAM. BOTH ATTEMPTS ARE RECORDED SO NEITHER IS REPEATED ──
// A CURVATURE-OUTLIER FAIRING was written here and removed. The idea: do not look for the seam,
// smooth wherever curvature is an outlier against its own LOCAL neighbourhood, so the seam is found
// because it is an outlier and a deliberate corner is left alone because a shape that is sharp all
// the way round has none. Measured, worst-over-mean curvature:
//
//                        taubin only    + outlier fairing
//      gap 0%               2.9x              2.9x        <- did not fire at all
//      gap 3%               3.0x              3.0x        <- did not fire at all
//      gap 8%               3.8x              3.8x        <- did not fire at all
//      TRIANGLE (control)  22.2x             11.2x        <- FIRED, and rounded the corners
//
// ⚠ IT DID EXACTLY THE OPPOSITE OF ITS PURPOSE, and the reason is worth keeping: after Taubin the
// seam is a BROAD ELEVATED REGION spread over several points, so no single point stands out from
// its neighbours — while a deliberate corner IS a single-point spike. "Local outlier" is a detector
// for sharp corners, which is the one thing that must not be touched. The control caught it; a
// seam-only fixture set would have shown a harmless no-op and hidden the corner damage entirely.
//
// So the two things tried and measured, neither of which works:
//   1. a PERIODIC CUBIC SPLINE (G2 by construction) — 2.9x -> 2.9x. C2 makes curvature CONTINUOUS,
//      not SMALL, and what a person sees is the magnitude.
//   2. curvature-outlier fairing — no effect on the seam, damages real corners.
//
// WHAT IS LEFT TO TRY, and why it is different: both attempts operated on the CLOSED loop, after
// the join already exists. Blend the stroke's ENDS as it is closed instead — overlap the last few
// samples with the first few and average them — so the join is smooth by construction and there is
// never a spike to detect. That is a change to `closeOutline`, not to a fairing pass.
// ⚠ Whatever is tried, keep a DRAWN TRIANGLE in the fixtures. It is the control that says the fix
// distinguishes an artefact from an intention.

// ── A THIRD CONSTRUCTION, KEPT UNUSED ───────────────────────────────────────────────────────────
// The blend above was built exactly as this note asked — the last K points and the first K replaced
// by ONE run ramping from the tail's shape to the head's, smoothstep so the TANGENT matches and not
// only the position, K taken from PATH LENGTH so sampling density does not change how much curve is
// touched. It does what nothing before it did. Measured through the real chain
// (prepareOutline -> taubin x20 -> prepareOutline), worst curvature over mean:
//
//      gap 5% 3.1x -> 2.4x   gap 8% 4.4x -> 2.4x   gap 12% 5.6x -> 3.3x   long 2:1 gap 8% 10.0x -> 5.0x
//      TRIANGLE at corner 23.9x -> 24.2x    TRIANGLE mid-edge 24.6x -> 23.2x    SQUARE 12.5x -> 12.8x
//
// The three CONTROLS hold within 2% — it reshapes only the arc adjacent to the JOIN, so a
// deliberate corner survives wherever the pen started. On its own terms it is the fix.
//
// ⚠⚠ IT WAS NOT SHIPPED BECAUSE THE 2D NUMBER NO LONGER CORRESPONDS TO ANYTHING VISIBLE. This whole
// note was written when a seam was a crease a person could see. Measured on the SUBDIVIDED SURFACE
// — worst dihedral and edges past 20 degrees, twice-subdivided, which is what is on screen — across
// stroke roughness from a gentle wobble to heavy per-sample tremor, with and without a closing gap:
//
//      control (no blend)   worst dihedral 6.6 - 13.8 deg,  edges over 20 deg: ZERO, every fixture
//      with the blend       worst dihedral 6.1 - 13.0 deg,  edges over 20 deg: ZERO, every fixture
//
// Better in five fixtures, WORSE in three, all inside ~3 degrees, and never near the 20 degrees at
// which a crease reads. The causes that made the seam visible were removed elsewhere — the apex fan
// the apex fan, the pole-free grid cap and the curve fairing itself — and what remains is
// absorbed by subdivision. Shipping it would have added an option, a code path and a maintenance
// claim in exchange for a number no one looks at.
//
// ⚠ THE 2D CURVATURE RATIO IS A PROXY AND IT HAS NOW OUTLIVED ITS ORACLE. If the seam is ever
// reported again, measure the SURFACE first and only reopen this if edges past 20 degrees actually
// appear. The implementation is recoverable from this note; it is thirty lines.
// ⚠ AND A HARNESS WARNING PAID FOR TWICE: measuring taubin WITHOUT prepareOutline's resample ranks
// the candidates wrongly. A truncated stroke keeps its closing chord as one long straight edge, so
// discrete curvature at its ends is enormous for a reason the app never sees — gap 8% reads 14.9x
// instead of 4.4x, and two of the four candidates swap places. Measure the chain the app runs.
//
// TWO FURTHER VARIANTS BUILT AND REJECTED, so a fourth attempt does not repeat them:
//   3. BLEND ONLY THE ACTUAL OVERLAP — the wording of the note above. Fires only where the stroke
//      overshoots its own start; K came out 0 on every gap fixture. Right idea for an overshoot,
//      no idea at all for a gap.
//   4. DISTRIBUTE THE CLOSING GAP — walk each end toward the gap's midpoint by a ramp, so the
//      correction is proportional to the artefact and a corner (which has no gap) cannot be
//      touched. Elegant, and WORSE than doing nothing on every gap fixture (gap 8%: 4.4x -> 6.8x):
//      pulling the ends together sharpens the local turn.

