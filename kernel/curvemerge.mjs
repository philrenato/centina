// MERGE: a group of curves that lie on one another (Dedupe's groups) as one
// curve, their mean.
//
// Each member is sampled at `samples` points evenly spaced by arc length. The
// members are put in step with the first before they are averaged: a member
// running the other way is reversed (whichever pairing of ends is nearer), and
// on a closed group each member's samples are turned so its start is the sample
// nearest the first member's start, then reversed if that runs closer. The mean
// of the samples is fitted by least squares (P&T 9.4.1, ends held) to the
// first member's degree and control-point count, so the merged curve is as
// light as what it replaces; a closed group's two end samples are one point, so
// the fit closes.
//
// Returns { crv, deviation } (the farthest any member's sample lies from the
// mean) or null for fewer than two usable curves.

import { curvePoint, buildArcLengthTable, paramAtArcLength, isCurveClosed } from './curve.mjs';
import { leastSquaresFit } from './fitcurve.mjs';

export const MERGE_SAMPLES = 96;

export function mergeCurves(curves, opts = {}) {
  const list = (curves || []).filter((c) => c && Array.isArray(c.ctrlPts) && c.ctrlPts.length >= 2);
  if (list.length < 2) return null;
  const n = Math.max(16, opts.samples ?? MERGE_SAMPLES);
  const closed = list.every((c) => isCurveClosed(c));
  const sets = list.map((c) => evenSamples(c, n));
  const ref = sets[0];
  const aligned = [ref];
  for (let k = 1; k < sets.length; k++) aligned.push(closed ? alignClosed(ref, sets[k]) : alignOpen(ref, sets[k]));
  const mean = ref.map((_, i) => [0, 1, 2].map((d) => aligned.reduce((s, a) => s + a[i][d], 0) / aligned.length));
  if (closed) mean[mean.length - 1] = mean[0].slice();
  let deviation = 0;
  for (const a of aligned) for (let i = 0; i < a.length; i++) deviation = Math.max(deviation, dist(a[i], mean[i]));
  const first = list[0];
  const p = Math.max(1, first.degree);
  const count = Math.max(p + 1, first.ctrlPts.length);
  const ubar = mean.map((_, i) => i / (mean.length - 1));
  const fit = leastSquaresFit(mean, p, count - 1, ubar);
  const crv = { degree: fit.degree ?? p, knots: fit.knots, ctrlPts: fit.ctrlPts.map((q) => [q[0], q[1], q[2], q.length > 3 ? q[3] : 1]) };
  return { crv, deviation };
}

function evenSamples(crv, n) {
  const u0 = crv.knots[0], u1 = crv.knots[crv.knots.length - 1];
  const tbl = buildArcLengthTable(crv, u0, u1);
  const out = [];
  for (let i = 0; i <= n; i++) out.push(curvePoint(crv, paramAtArcLength(tbl, (tbl.total * i) / n)).slice(0, 3));
  return out;
}
function alignOpen(ref, s) {
  const same = dist(ref[0], s[0]) + dist(ref[ref.length - 1], s[s.length - 1]);
  const flip = dist(ref[0], s[s.length - 1]) + dist(ref[ref.length - 1], s[0]);
  return flip < same ? s.slice().reverse() : s;
}
// A closed member: its seam moved to the sample nearest the reference's start,
// then the direction that lies closer along the whole loop.
function alignClosed(ref, s) {
  const m = s.length - 1; // the last sample repeats the first
  let at = 0;
  for (let i = 1; i < m; i++) if (dist(s[i], ref[0]) < dist(s[at], ref[0])) at = i;
  const fwd = [], back = [];
  for (let i = 0; i <= m; i++) { fwd.push(s[(at + i) % m]); back.push(s[((at - i) % m + m) % m]); }
  const cost = (a) => a.reduce((t, q, i) => t + dist(q, ref[i]), 0);
  return cost(back) < cost(fwd) ? back : fwd;
}
function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
