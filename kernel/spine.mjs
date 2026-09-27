// A centerline measured from a body, which then drives it.
//
// Every flow tool asks for an authored base curve. Here the base curve is
// measured: Spine on a shape gives the curve down its middle, with a handful of
// control points. Bending those bends the body with them.
//
// Two ideas carry the whole file.
//
// Sections move rigidly. A point is stored as an arc length along the rest
// spine plus an offset in the frame there, and replayed against the same curve
// after it has been bent. Because a whole cross-section shares one frame, the
// section is carried by a rotation and a translation and nothing else — so
// thickness is preserved to 5e-5 of itself at bend amplitudes from 5mm to
// 100mm, the residue being that two mirrored points can land at slightly
// different arc lengths and so take slightly different frames. Displacing each
// point along one axis by a height field instead shears: it thins the body to
// cos(atan|grad h|).
//
// The frame's seed is carried from the rest curve. Rotation-minimizing frames
// need a starting normal, and any rule that picks one from the tangent alone
// has a discontinuity somewhere: the usual one (cross the tangent with X, or
// with Y when the tangent is too close to X) switches reference vector at
// |Tx| = 0.9. Choosing independently per curve lets a drag that walks the
// start tangent across that threshold roll every downstream frame at once (a
// 5mm drag of the first control point can roll a section at the far end by
// 45.6 degrees). Carrying the rest curve's seed across by the minimal rotation
// between the two start tangents gives 0.00 degrees over the same sweep.
// A test for this has to put the start tangent at the threshold: away from
// 0.9 the same drag without the carry rolls only 0.99 degrees.
import { curvePoint, curvePointAndTangent, closestPointOnCurve } from './curve.mjs';
import { leastSquaresFit, centripetalParams } from './fitcurve.mjs';
import { jacobiEigenSym3 } from './refit.mjs';

const SAMPLES = 16000;   // 4k leaves 1.45mm of noise on a fish's stations; 16k and 32k both leave 0.55mm
const SLABS = 48;
const STATION_RESAMPLE = 64;
const FRAME_SAMPLES = 256;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => { const L = len(a) || 1; return [a[0] / L, a[1] / L, a[2] / L]; };

/* A deterministic low-discrepancy point, so the same body always measures to the
   same centerline. A random sequence would make the spine a different curve
   each time. */
function halton(i, base) {
  let f = 1, r = 0, n = i + 1;
  while (n > 0) { f /= base; r += f * (n % base); n = Math.floor(n / base); }
  return r;
}

/* Area-uniform samples of a triangle soup. Uniform over area rather than over
   vertices, because a body's centroid is a property of its surface and a dense
   corner would otherwise pull the whole spine toward itself. */
export function sampleSurface(positions, count = SAMPLES) {
  const triCount = Math.floor(positions.length / 9);
  if (triCount < 1) return { pts: [], areas: [], total: 0 };
  const cum = new Float64Array(triCount);
  let total = 0;
  for (let t = 0; t < triCount; t += 1) {
    const o = t * 9;
    const a = [positions[o], positions[o + 1], positions[o + 2]];
    const b = [positions[o + 3], positions[o + 4], positions[o + 5]];
    const c = [positions[o + 6], positions[o + 7], positions[o + 8]];
    total += 0.5 * len(cross(sub(b, a), sub(c, a)));
    cum[t] = total;
  }
  if (!(total > 0)) return { pts: [], areas: [], total: 0 };
  const pts = [];
  for (let i = 0; i < count; i += 1) {
    const target = ((i + 0.5) / count) * total;
    let lo = 0, hi = triCount - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < target) lo = m + 1; else hi = m; }
    const o = lo * 9;
    const a = [positions[o], positions[o + 1], positions[o + 2]];
    const b = [positions[o + 3], positions[o + 4], positions[o + 5]];
    const c = [positions[o + 6], positions[o + 7], positions[o + 8]];
    let u = halton(i, 2), v = halton(i, 3);
    if (u + v > 1) { u = 1 - u; v = 1 - v; }
    pts.push([
      a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0]),
      a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1]),
      a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2]),
    ]);
  }
  return { pts, total };
}

/* The long axis, and how confident it is. `aniso` is the ratio of the two
   largest spreads: near 1 the body has no long axis (a sphere, a cube, a disc);
   the direction is still deterministic but not meaningful, and the caller is
   expected to report that. */
export function principalAxisOf(pts) {
  const n = pts.length;
  if (n < 3) return null;
  const c = [0, 0, 0];
  for (const p of pts) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
  const centroid = mul(c, 1 / n);
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const p of pts) {
    const d = sub(p, centroid);
    for (let i = 0; i < 3; i += 1) for (let j = 0; j < 3; j += 1) M[i][j] += d[i] * d[j];
  }
  for (let i = 0; i < 3; i += 1) for (let j = 0; j < 3; j += 1) M[i][j] /= n;
  // jacobiEigenSym3 returns ascending, with a deterministic tie-break for an
  // exactly degenerate spectrum — which is what a sphere gives, and why the
  // direction is repeatable even where it is not meaningful.
  const eig = jacobiEigenSym3(M);
  const axis = norm(eig[2].vector);
  const l0 = Math.max(eig[2].value, 0), l1 = Math.max(eig[1].value, 0);
  return { axis, centroid, aniso: Math.sqrt(l0 / Math.max(l1, 1e-12)), values: [eig[2].value, eig[1].value, eig[0].value] };
}

/* The measured centerline. Slabs perpendicular to the long axis, each station
   the area-weighted centroid of its slab.
   The slabs stay perpendicular to the mean axis, as a deliberate limit.
   Re-slicing each station perpendicular to the running tangent (iterated,
   damped, or single pass) oscillates on a flat body (a fish zig-zags 3-8mm per
   pass and its spine shrinks from 94 to 63mm), and a single pass makes a
   plate's ends worse, 2.2mm to 10mm. The cost of the stable definition is that
   an already-bent body reads straighter than it is: a banana's ends are off by
   about the tube radius, and the user bends it back. */
export function deriveSpineStations(positions, opts = {}) {
  const count = opts.samples ?? SAMPLES;
  const { pts } = sampleSurface(positions, count);
  if (pts.length < 16) return { ok: false, reason: 'no-surface', why: 'this shape has no surface to measure' };
  const pr = opts.axis
    ? { axis: norm(opts.axis), centroid: principalAxisOf(pts).centroid, aniso: principalAxisOf(pts).aniso }
    : principalAxisOf(pts);
  if (!pr) return { ok: false, reason: 'no-surface', why: 'this shape has no surface to measure' };
  const { axis, centroid } = pr;

  let tMin = Infinity, tMax = -Infinity;
  const proj = new Float64Array(pts.length);
  for (let i = 0; i < pts.length; i += 1) {
    const t = dot(sub(pts[i], centroid), axis);
    proj[i] = t;
    if (t < tMin) tMin = t;
    if (t > tMax) tMax = t;
  }
  const span = tMax - tMin;
  if (!(span > 0)) return { ok: false, reason: 'no-extent', why: 'this shape has no length along any direction' };

  const slabs = opts.slabs ?? SLABS;
  const sums = Array.from({ length: slabs }, () => [0, 0, 0]);
  const counts = new Int32Array(slabs);
  const bins = Array.from({ length: slabs }, () => new Set());
  const members = Array.from({ length: slabs }, () => []);
  for (let i = 0; i < pts.length; i += 1) {
    let k = Math.floor(((proj[i] - tMin) / span) * slabs);
    if (k < 0) k = 0; if (k >= slabs) k = slabs - 1;
    sums[k] = add(sums[k], pts[i]);
    counts[k] += 1;
    members[k].push(i);
  }

  /* A frame in the slab plane, to ask whether a station sits in material or in
     air. A ring's centroid is in its hole; that is a legitimate answer (the
     spine becomes a bending axis rather than a centerline), but it is measured
     and reported rather than assumed away. */
  const ref = Math.abs(axis[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = norm(cross(axis, ref));
  const e2 = cross(axis, e1);

  const stations = [];
  let inAir = 0;
  for (let k = 0; k < slabs; k += 1) {
    if (!counts[k]) continue;
    const centre = mul(sums[k], 1 / counts[k]);
    // Put the station on the slab's own mid-plane, so stations are evenly
    // spaced along the axis even where a slab's samples lean to one end.
    const mid = tMin + ((k + 0.5) / slabs) * span;
    const drift = dot(sub(centre, centroid), axis) - mid;
    const station = sub(centre, mul(axis, drift));
    for (const i of members[k]) {
      const d = sub(pts[i], station);
      bins[k].add(((Math.floor((Math.atan2(dot(d, e2), dot(d, e1)) + Math.PI) / (2 * Math.PI) * 16)) % 16 + 16) % 16);
    }
    if (bins[k].size < 8) inAir += 1;
    stations.push(station);
  }
  if (stations.length < 4) return { ok: false, reason: 'too-short', why: 'this shape is too short to measure a spine along' };

  // The ends sit at the extremes, not at the last slab's mid-plane, so the
  // spine spans the whole body before any overhang is added.
  stations[0] = add(centroid, add(mul(axis, tMin), sub(stations[0], add(centroid, mul(axis, dot(sub(stations[0], centroid), axis))))));
  const last = stations.length - 1;
  stations[last] = add(centroid, add(mul(axis, tMax), sub(stations[last], add(centroid, mul(axis, dot(sub(stations[last], centroid), axis))))));

  return {
    ok: true,
    stations,
    axis,
    centroid,
    aniso: pr.aniso,
    outside: inAir / stations.length,
    from: add(centroid, mul(axis, tMin)),
    to: add(centroid, mul(axis, tMax)),
    length: span,
  };
}

/* Even spacing along the measured polyline, so fairing and fitting both see
   stations that carry equal weight rather than whatever the slabs produced. */
function resamplePolyline(pts, n) {
  const cum = [0];
  for (let i = 1; i < pts.length; i += 1) cum.push(cum[i - 1] + len(sub(pts[i], pts[i - 1])));
  const total = cum[cum.length - 1];
  if (!(total > 0)) return pts.slice();
  const out = [];
  let j = 0;
  for (let i = 0; i < n; i += 1) {
    const target = (i / (n - 1)) * total;
    while (j < pts.length - 2 && cum[j + 1] < target) j += 1;
    const segLen = cum[j + 1] - cum[j];
    const t = segLen > 0 ? (target - cum[j]) / segLen : 0;
    out.push(add(pts[j], mul(sub(pts[j + 1], pts[j]), t)));
  }
  return out;
}

/* The noise filter, exposed as a control. The stations carry the sampling's
   wobble; without this the fit chases it. At 8 control points the fit's
   deviation is 2.9mm unfaired and 0.24mm at Smoothness 1; at 16 points the
   unfaired fit's peak curvature is 0.31/mm — a 3mm-radius wobble in a body
   100mm long — against 0.044/mm faired. It costs 5% of the amplitude of a
   gentle curve at 1. Ends stay pinned so the spine still spans the body. */
export function fairStations(stations, amount, iterationsAt1 = 20, lambda = 0.5) {
  const iterations = Math.round(Math.max(0, Math.min(1, amount)) * iterationsAt1);
  let pts = stations.map((p) => p.slice());
  for (let it = 0; it < iterations; it += 1) {
    const next = pts.map((p) => p.slice());
    for (let i = 1; i < pts.length - 1; i += 1) {
      const avg = mul(add(pts[i - 1], pts[i + 1]), 0.5);
      next[i] = add(pts[i], mul(sub(avg, pts[i]), lambda));
    }
    pts = next;
  }
  return pts;
}

/* The spine runs past the body at both ends. With the spine cut exactly to
   the body, the last span of the fit lies inside it, so an end handle puts the
   curvature of a short clamped span into the tip: on a 20mm end drag, the
   tightest bend radius inside the body is 6.9mm against a half-thickness of
   13.3 — folded, Jacobian -1.05. With 20% of the body's length of overhang it
   is 18.1mm and the Jacobian 0.73. Sections never bunch (they stay rigid);
   this is only about where the fit's last span sits. */
function withOverhang(stations, overhang) {
  if (!(overhang > 0)) return stations;
  const n = stations.length;
  const t0 = norm(sub(stations[0], stations[1]));
  const t1 = norm(sub(stations[n - 1], stations[n - 2]));
  return [add(stations[0], mul(t0, overhang)), ...stations, add(stations[n - 1], mul(t1, overhang))];
}

/* Degree 3, clamped, ends interpolated, a fixed number of control points — a
   fit, not an interpolation. Interpolating 12 noisy stations gives a peak
   curvature of 0.14/mm (a 7mm-radius wobble) against 0.05/mm for the 12-point
   fit and 0.014/mm for the 5-point one.
   The fit's accuracy does not affect the deformation's correctness, so there
   is no second, finer, hidden curve. Rest coordinates are measured against the
   fitted spine itself, so a spine that misses the centroids by 0.75mm still
   leaves the body bit-identical at rest and still bends it rigidly. The miss
   only decides how natural the bending axis looks. */
export function fitSpine(stations, opts = {}) {
  const points = Math.max(4, Math.min(16, Math.round(opts.points ?? 5)));
  const smoothness = opts.smoothness ?? 0.5;
  const overhang = opts.overhang ?? 0;
  const dense = resamplePolyline(stations, opts.resample ?? STATION_RESAMPLE);
  const faired = fairStations(dense, smoothness);
  const work = withOverhang(faired, overhang);
  const p = 3;
  const n = points - 1;
  if (work.length < points + 2) return { ok: false, reason: 'too-few', why: 'there are not enough measured stations for that many control points' };
  let crv;
  try { crv = leastSquaresFit(work, p, n, centripetalParams(work)); }
  catch (e) { return { ok: false, reason: 'fit-failed', why: `the centerline could not be fitted (${e.message})` }; }
  let deviation = 0;
  for (const q of faired) {
    const hit = closestPointOnCurve(crv, q);
    deviation = Math.max(deviation, len(sub(q, hit.point ?? curvePoint(crv, hit.u))));
  }
  return { ok: true, crv, deviation, stations: faired };
}

/* Rotation-minimizing frames by double reflection (Wang 2008), with the seed
   passed in — see this file's header for why the seed is carried rather than
   chosen per curve. Two reflections per step carry the previous frame onto the
   next tangent without accumulating the twist a naive cross-product frame
   does. */
export function spineFrameTable(crv, seedNormal, samples = FRAME_SAMPLES) {
  const uMin = crv.knots[0], uMax = crv.knots[crv.knots.length - 1];
  const us = [], pts = [], Ts = [];
  for (let i = 0; i < samples; i += 1) {
    const u = uMin + ((uMax - uMin) * i) / (samples - 1);
    const { point, tangent } = curvePointAndTangent(crv, u);
    us.push(u); pts.push(point); Ts.push(norm(tangent));
  }
  let r = seedNormal ? norm(seedNormal) : null;
  if (!r) { const a = Math.abs(Ts[0][0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]; r = norm(cross(Ts[0], a)); }
  // Make the seed perpendicular to the first tangent without letting it flip.
  r = norm(sub(r, mul(Ts[0], dot(r, Ts[0]))));
  if (!(len(r) > 1e-9)) { const a = Math.abs(Ts[0][0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]; r = norm(cross(Ts[0], a)); }
  const Ns = [r], Bs = [cross(Ts[0], r)];
  for (let i = 0; i < samples - 1; i += 1) {
    const v1 = sub(pts[i + 1], pts[i]);
    const c1 = dot(v1, v1);
    let rL = Ns[i], tL = Ts[i];
    if (c1 > 1e-18) {
      rL = sub(Ns[i], mul(v1, (2 / c1) * dot(v1, Ns[i])));
      tL = sub(Ts[i], mul(v1, (2 / c1) * dot(v1, Ts[i])));
    }
    const v2 = sub(Ts[i + 1], tL);
    const c2 = dot(v2, v2);
    const rNext = c2 > 1e-18 ? sub(rL, mul(v2, (2 / c2) * dot(v2, rL))) : rL;
    const nn = norm(sub(rNext, mul(Ts[i + 1], dot(rNext, Ts[i + 1]))));
    Ns.push(nn);
    Bs.push(cross(Ts[i + 1], nn));
  }
  const arc = [0];
  for (let i = 1; i < samples; i += 1) arc.push(arc[i - 1] + len(sub(pts[i], pts[i - 1])));
  return { us, pts, Ts, Ns, Bs, arc, total: arc[arc.length - 1] };
}

export function frameAtArc(table, s) {
  const { arc, pts, Ts, Ns, total } = table;
  const target = Math.max(0, Math.min(total, s));
  let lo = 0, hi = arc.length - 1;
  while (lo < hi - 1) { const m = (lo + hi) >> 1; if (arc[m] <= target) lo = m; else hi = m; }
  const seg = arc[hi] - arc[lo];
  const t = seg > 1e-12 ? (target - arc[lo]) / seg : 0;
  const P = add(pts[lo], mul(sub(pts[hi], pts[lo]), t));
  const T = norm(add(Ts[lo], mul(sub(Ts[hi], Ts[lo]), t)));
  let N = add(Ns[lo], mul(sub(Ns[hi], Ns[lo]), t));
  N = norm(sub(N, mul(T, dot(N, T))));
  return { P, T, N, B: cross(T, N), beyond: s < 0 ? s : (s > total ? s - total : 0) };
}

/* Rest coordinates — measured once, against the fitted spine itself.
   Arc length is measured from the spine's midpoint, not from its start. An
   edit at one end changes the arc length of the end span (lifting a start
   control point can add ~5mm); anchored at the start, the entire body would
   slide along the spine by that amount, so an edit at the nose would move the
   tail. Anchored at the middle, the two ends share the change. */
export function spineRestCoords(points, crv, seedNormal) {
  const table = spineFrameTable(crv, seedNormal);
  const half = table.total / 2;
  const rest = [];
  for (const P of points) {
    const hit = closestPointOnCurve(crv, P);
    const u = hit.u ?? hit;
    // Arc length at that parameter, from the same table the flow will use.
    let lo = 0;
    while (lo < table.us.length - 2 && table.us[lo + 1] < u) lo += 1;
    const du = table.us[lo + 1] - table.us[lo];
    const t = du > 1e-12 ? (u - table.us[lo]) / du : 0;
    const arcAt = table.arc[lo] + (table.arc[lo + 1] - table.arc[lo]) * t;
    const f = frameAtArc(table, arcAt);
    const d = sub(P, f.P);
    rest.push({ s: arcAt - half, y: dot(d, f.N), z: dot(d, f.B), a: dot(d, f.T) });
  }
  return { rest, table, seedNormal: table.Ns[0], restLength: table.total };
}

/* The flow. A section is carried by the frame at its own arc length and
   nothing else, which is what makes thickness exact — see this file's header. */
export function spineFlow(restInfo, crv, opts = {}) {
  const stretch = !!opts.stretch;
  // The seed rides across from the rest curve by the minimal rotation between
  // the two start tangents (the anti-roll carry described in the header).
  const restT = restInfo.table.Ts[0];
  const { tangent } = curvePointAndTangent(crv, crv.knots[0]);
  const curT = norm(tangent);
  let seed = restInfo.seedNormal;
  const axis = cross(restT, curT);
  const sinA = len(axis), cosA = dot(restT, curT);
  if (sinA > 1e-12) {
    const k = norm(axis), ang = Math.atan2(sinA, cosA), c = Math.cos(ang), sn = Math.sin(ang);
    seed = add(add(mul(seed, c), mul(cross(k, seed), sn)), mul(k, dot(k, seed) * (1 - c)));
  }
  const table = spineFrameTable(crv, seed);
  const half = table.total / 2;
  const scale = stretch && restInfo.restLength > 1e-12 ? table.total / restInfo.restLength : 1;
  const out = [];
  for (const r of restInfo.rest) {
    const f = frameAtArc(table, half + r.s * scale);
    const along = r.a + (f.beyond || 0);
    out.push([
      f.P[0] + along * f.T[0] + r.y * f.N[0] + r.z * f.B[0],
      f.P[1] + along * f.T[1] + r.y * f.N[1] + r.z * f.B[1],
      f.P[2] + along * f.T[2] + r.y * f.N[2] + r.z * f.B[2],
    ]);
  }
  return { points: out, length: table.total };
}

/* How close the body is to folding, as a percentage. The Jacobian in the
   bending plane is 1 - kappa*d, with d the offset toward the turn's center, so
   the inside of a bend folds where the offset reaches the center of curvature.
   Reported, never refused: a folded body is drawn folded, saves, and reopens
   folded; the user is told what to ease. A fish's natural wriggle sits at
   40-80%. Evaluate this on the display points, not the cage: on a hard bend
   the cage can read +0.15 while the limit surface is at -1.87. */
export function spineReach(restInfo, crv) {
  const table = spineFrameTable(crv, restInfo.seedNormal);
  const half = table.total / 2;
  let worst = 0;
  for (const r of restInfo.rest) {
    const s = half + r.s;
    const h = Math.max(table.total * 1e-3, 1e-6);
    const a = frameAtArc(table, Math.max(0, s - h));
    const b = frameAtArc(table, s);
    const c = frameAtArc(table, Math.min(table.total, s + h));
    // Curvature from the turn of the tangent over arc length, and the offset
    // component pointing at the center of that turn.
    const dT = sub(c.T, a.T);
    const ds = Math.min(table.total, s + h) - Math.max(0, s - h);
    if (!(ds > 1e-12)) continue;
    const kappa = len(dT) / ds;
    if (!(kappa > 1e-12)) continue;
    const toCentre = norm(dT);
    const d = r.y * dot(b.N, toCentre) + r.z * dot(b.B, toCentre);
    worst = Math.max(worst, kappa * d);
  }
  return worst;
}
