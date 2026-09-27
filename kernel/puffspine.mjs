// Spine — the medial axis of a drawn outline, as an inscribed-ball march.
//
// `buildPuff` calls buildSpine/attachToSpine whenever a boundary point cannot see the deepest
// point in a straight line, and uses the per-point target instead of the single center. A
// star-shaped outline never reaches this file.
//
// Lerping every boundary point straight toward one interior point leaves the shape on a crescent
// — 23 of 96 boundary points against 0 of 96 on a disc. A crescent, a C, an S, a comma, a bean
// and a boomerang are ordinary things to draw, and a closed loop should give a solid for each.
//
// Cap dihedrals depend on the profile. `family: 'tangent'` has slope 0 at the rim (a thin flaring
// edge) and slope 2 at the peak (a cone tip); with it a disc measures 52 degrees worst off-rim,
// with the Lamé hemisphere 4.5.
//
// The apex cap does not suit a spined shape. Ring M is a long thin loop hugging the spine, and
// collapsing it to one apex puts near-coplanar faces back to back — worst 179.7 degrees on a C.
// Fairing makes the count worse: it is a topological degeneracy, not a smoothing problem. It is
// under 1% of the surface (C 0.43%, comma 0.36%), and the body is clean below 60% of the apex
// height. The grid cap (puffgridcap.mjs) closes that loop with a ridge patch instead.
//
// The march is balls, not a ridge scan. The obvious approach is to sample the distance field on a
// grid and extract its ridge, but a sampled distance field's worst-case error is 46-49% of a cell
// at every resolution, and it lands on the medial axis: an axis midway between two grid columns
// gives both nodes the same value. Ridge extraction estimates precisely the locus where grid
// sampling is structurally worst. This marches instead, on the exact polygon distance, and never
// builds a grid.
//
// The design rests on one fact: the closed ball B(s, r(s)) with r(s) = the distance from
// s to the boundary lies entirely inside the polygon. So a segment from a boundary point p to s is
// guaranteed inside iff |p − s| <= r(s). That single inequality is what makes every spoke safe,
// and it is why the attachment rule below minimizes |p − s| − r(s) rather than |p − s|: nearest
// point on the spine carries no containment guarantee at all.

import { distanceToBoundary, pointInPolygon } from './puffoutline.mjs';

/** Signed: positive inside, negative outside. distanceToBoundary is unsigned, so at the boundary
 *  its field rises in both directions and its gradient is degenerate. */
export function signedDistance(pts, x, y) {
  return (pointInPolygon(pts, x, y) ? 1 : -1) * distanceToBoundary(pts, x, y);
}

/** Refine an interior point onto the local maximum of the distance field — the center of a
 *  maximal inscribed ball. A 16-direction pattern search with a halving step: no derivatives, and
 *  the field is not differentiable where we are going. */
function refineBall(pts, p, step) {
  let [x, y] = p, r = signedDistance(pts, x, y), h = step;
  for (let it = 0; it < 24 && h > step * 1e-3; it++) {
    let best = r, bx = x, by = y;
    for (let k = 0; k < 16; k++) {
      const a = 2 * Math.PI * k / 16;
      const nx = x + h * Math.cos(a), ny = y + h * Math.sin(a);
      const nr = signedDistance(pts, nx, ny);
      if (nr > best) { best = nr; bx = nx; by = ny; }
    }
    if (best > r) { x = bx; y = by; r = best; } else h *= 0.5;
  }
  return { p: [x, y], r };
}

/**
 * March outward from the root along one direction, re-centering onto the ridge at every step.
 * The stop condition that matters is the redundant-ball test, and it is against the root.
 * A ball wholly contained in the root's ball adds nothing to the medial axis. Tested against the
 * previous step instead it never fires, and a disc marches all the way to its own rim, growing a
 * spurious spine.
 */
function marchRun(pts, root, r0, dir, opts) {
  const { minR, maxSteps, turnMax } = opts;
  const out = [];
  let p = root.slice(), u = dir.slice(), r = r0;
  for (let step = 0; step < maxSteps; step++) {
    const h = 0.35 * Math.max(r, minR);
    let q = [p[0] + u[0] * h, p[1] + u[1] * h];
    if (!pointInPolygon(pts, q[0], q[1])) break;
    // re-center perpendicular to the direction of travel: scan for the local max of the field
    const nx = -u[1], ny = u[0];
    let bestT = 0, bestR = signedDistance(pts, q[0], q[1]);
    const span = 0.6 * Math.max(r, minR);
    for (let k = 1; k <= 12; k++) {
      for (const sgn of [1, -1]) {
        const t = sgn * span * k / 12;
        const rr = signedDistance(pts, q[0] + nx * t, q[1] + ny * t);
        if (rr > bestR) { bestR = rr; bestT = t; }
      }
    }
    q = [q[0] + nx * bestT, q[1] + ny * bestT];
    const rq = bestR;
    if (!(rq > minR)) break;
    // Redundant against the root — a ball inside the root's ball is not part of the axis.
    // The tolerance is load-bearing. On a disc this quantity is r0 exactly at
    // every step — the field is r0 − |q − root| by construction — so an exact `<=` is a coin flip
    // on the last bit, and a false result grows a 16-sample spine and 32 runs where there should
    // be none. A disc must return a single sample, so that it builds exactly as the single-center
    // construction does.
    if (Math.hypot(q[0] - root[0], q[1] - root[1]) + rq <= r0 * (1 + 1e-6)) break;
    const nu = [q[0] - p[0], q[1] - p[1]];
    const L = Math.hypot(nu[0], nu[1]);
    if (!(L > 1e-12)) break;
    nu[0] /= L; nu[1] /= L;
    if (nu[0] * u[0] + nu[1] * u[1] < Math.cos(turnMax)) break;   // the ridge turned too sharply
    out.push({ p: q, r: rq });
    p = q; u = nu; r = rq;
  }
  return out;
}

/**
 * @returns { ok, reason, pts:[[x,y],...], r:Float64Array, root, rootR, runs, thirdLimb }
 *   `pts` is an ordered open polyline. A disc returns a single sample — the degenerate case
 *   falls out of the pruning rather than being special-cased, which keeps a disc bit-identical
 *   to the single-center construction.
 */
export function buildSpine(pts, { seed, samples = 24, minRFrac = 0.03, turnMax = Math.PI / 3.2 } = {}) {
  const root0 = refineBall(pts, seed.at, seed.r * 0.25);
  const root = root0.p, r0 = root0.r;
  if (!(r0 > 0)) return { ok: false, reason: 'no-interior', pts: [root], r: Float64Array.from([0]), root, rootR: 0, runs: 0, thirdLimb: 0 };
  const minR = r0 * minRFrac;

  // Candidate directions: local maxima of the field on a small circle about the root. On a disc
  // the field is flat there and every direction ties, which is exactly why the redundant-ball test
  // has to be the thing that stops them.
  const A = 64, ring = new Float64Array(A);
  for (let k = 0; k < A; k++) {
    const a = 2 * Math.PI * k / A;
    ring[k] = signedDistance(pts, root[0] + 0.25 * r0 * Math.cos(a), root[1] + 0.25 * r0 * Math.sin(a));
  }
  const dirs = [];
  for (let k = 0; k < A; k++) {
    const prev = ring[(k - 1 + A) % A], next = ring[(k + 1) % A];
    if (ring[k] >= prev && ring[k] >= next) {
      const a = 2 * Math.PI * k / A;
      dirs.push([Math.cos(a), Math.sin(a)]);
    }
  }
  const marchOpts = { minR, maxSteps: 200, turnMax };
  let runs = dirs.map(d => marchRun(pts, root, r0, d, marchOpts)).filter(rn => rn.length > 0);

  // The prune is what collapses a disc to a point, and the root ball must be one of the
  // candidates or it cannot. A sample whose ball is contained in another's is not on the medial
  // axis — and on a disc every marched ball is contained in the root's, in no other. Pruning only
  // against sibling samples leaves a disc with a 16-sample spine and 32 runs.
  // The tolerance: a drawn "disc" is a 96-gon, so
  // |s − root| + r(s) = r0 holds exactly only along the perpendicular to the nearest edge; toward
  // a vertex the marched ball is legitimately a little larger. The slack is what makes a polygon
  // behave like the circle it is drawn as. 2% of the root radius: far above that discretization,
  // far below any real limb, and the crescent (whose limbs run 3.3 root-radii) is untouched by it.
  const flat = [];
  for (const rn of runs) for (const s of rn) flat.push(s);
  const rootBall = { p: root, r: r0 };
  const cands = [rootBall, ...flat];
  const keep = flat.filter((s) => !cands.some((t) =>
    t !== s && Math.hypot(s.p[0] - t.p[0], s.p[1] - t.p[1]) + s.r <= t.r * (1 + 0.02)));
  runs = runs.map(rn => rn.filter(s => keep.includes(s))).filter(rn => rn.length > 0);

  const lenOf = (rn) => { let L = 0, q = root; for (const s of rn) { L += Math.hypot(s.p[0]-q[0], s.p[1]-q[1]); q = s.p; } return L; };
  runs.sort((a, b) => lenOf(b) - lenOf(a));
  const thirdLimb = runs.length > 2 ? lenOf(runs[2]) : 0;

  // Two limbs only; a third is reported rather than quietly dropped. An unbranched pruned
  // axis is the scope: a Y, a T or a plus needs a spine tree and a cap that is not a slit,
  // and silently flattening the ignored limb into a wedge would give a plausible shape that is
  // not what was drawn.
  const A1 = runs[0] || [], B1 = runs[1] || [];
  const line = [...A1.slice().reverse(), { p: root, r: r0 }, ...B1];
  return {
    ok: true, reason: null,
    pts: line.map(s => s.p), r: Float64Array.from(line.map(s => s.r)),
    root, rootR: r0, runs: runs.length, thirdLimb,
  };
}

/**
 * Attach every boundary point to the spine sample whose maximal ball reaches it best.
 * Minimizes |p − s| − r(s), not |p − s|. The residual it returns is the safety statement: it is
 * how far the point lies outside the ball assigned to it, so zero means the spoke is provably
 * inside the polygon and a large value means the spine does not describe this part of the shape.
 * A branching form passes every containment test and fails here — which is the only place it fails.
 */
export function attachToSpine(pts, spine, { smoothPasses = 8 } = {}) {
  const N = pts.length, tau = new Int32Array(N), residual = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let best = Infinity, bj = 0;
    for (let j = 0; j < spine.pts.length; j++) {
      const d = Math.hypot(pts[i][0] - spine.pts[j][0], pts[i][1] - spine.pts[j][1]) - spine.r[j];
      if (d < best) { best = d; bj = j; }
    }
    tau[i] = bj; residual[i] = best;
  }

  // The target is smoothed around the loop.
  // The attachment above is a nearest-ball argmin, and an argmin is a step function: on a
  // hand-drawn outline neighboring boundary points land on different spine samples. Measured on a
  // wobbly pencil-drawn loop — 8 places where tau jumps by more than one sample, by as much as
  // four — against a clean map on a smooth disc. Unsmoothed, adjacent spokes point in different
  // directions, the ring lattice zigzags, and the surface comes out as a crumpled starfish with
  // radial creases converging on a point. That surface is closed, genus 0, Euler 2, with zero
  // degenerate quads, every directed edge once, positive signed volume and a one-to-one XY
  // projection: topology and projection checks cannot see a crease in Z.
  //
  // So the target positions are averaged around the loop rather than the indices. Positions,
  // because the result has to be continuous: smoothing an integer index still steps. The targets lie
  // on the medial axis and their neighbors lie along it, so averaging keeps them there — asserted
  // by the caller, not assumed here.
  const target = new Array(N);
  for (let i = 0; i < N; i++) target[i] = spine.pts[tau[i]].slice();
  for (let pass = 0; pass < smoothPasses; pass++) {
    const next = target.map((t, i) => {
      const a = target[(i - 1 + N) % N], b = target[(i + 1) % N];
      return [(a[0] + 2 * t[0] + b[0]) / 4, (a[1] + 2 * t[1] + b[1]) / 4];
    });
    for (let i = 0; i < N; i++) target[i] = next[i];
  }
  return { tau, residual, target };
}
