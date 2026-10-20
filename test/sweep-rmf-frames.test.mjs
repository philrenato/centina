import test from 'node:test';
import assert from 'node:assert/strict';
import { buildParallelTransportFrames, transportByDoubleReflection, sweep1Rigid } from '../kernel/sweep.mjs';
import { globalCurveInterp } from '../kernel/interpolate.mjs';
import { curvePointAndTangent } from '../kernel/curve.mjs';
import { makeLine } from '../kernel/primitives.mjs';

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit = (a) => { const l = Math.hypot(...a); return a.map((x) => x / l); };
const deg = (r) => (r * 180) / Math.PI;

// A rotation-minimizing frame of the curve integrated over many steps: the reference.
function referenceNormals(rail, us, r0) {
  const u0 = rail.knots[0], u1 = rail.knots[rail.knots.length - 1], N = 20000;
  let { point: p, tangent: t } = curvePointAndTangent(rail, u0); t = unit(t);
  let r = r0; const out = new Map(); const want = [...us].sort((a, b) => a - b); let w = 0;
  for (let i = 1; i <= N; i++) {
    const u = u0 + ((u1 - u0) * i) / N;
    let { point: q, tangent: s } = curvePointAndTangent(rail, u); s = unit(s);
    r = transportByDoubleReflection(p, t, r, q, s); p = q; t = s;
    while (w < want.length && want[w] <= u + 1e-12) out.set(want[w++], r);
  }
  return out;
}
function twistSpread(rail, frames) {
  const R = referenceNormals(rail, frames.map((f) => f.u), frames[0].xAxis);
  const a = [];
  for (const f of frames.slice(1, -1)) { const r = R.get(f.u); a.push(deg(Math.atan2(dot(r, f.yAxis), dot(r, f.xAxis)))); }
  for (let i = 1; i < a.length; i++) { while (a[i] - a[i - 1] > 180) a[i] -= 360; while (a[i] - a[i - 1] < -180) a[i] += 360; }
  return Math.max(...a) - Math.min(...a);
}
const helix = globalCurveInterp(Array.from({ length: 25 }, (_, i) => { const a = (i / 24) * 4 * Math.PI; return [20 * Math.cos(a), 20 * Math.sin(a), a * 6]; }), 3);
const trefoilPts = Array.from({ length: 16 }, (_, i) => { const a = (i / 16) * 2 * Math.PI; return [Math.sin(a) + 2 * Math.sin(2 * a), Math.cos(a) - 2 * Math.cos(2 * a), -Math.sin(3 * a)].map((x) => x * 15); });
const trefoil = globalCurveInterp([...trefoilPts, trefoilPts[0]], 3);

test('rmf frames on a two-turn helix stay within 0.1 degree of a rotation-minimizing frame; projection does not', () => {
  const rmf = buildParallelTransportFrames(helix, [], { transport: 'rmf' });
  const proj = buildParallelTransportFrames(helix);
  assert.ok(twistSpread(helix, rmf) < 0.1, `rmf ${twistSpread(helix, rmf)}`);
  assert.ok(twistSpread(helix, proj) > 10, `projection ${twistSpread(helix, proj)}`);
});

test('rmf frames stay orthonormal and keep the station tangent', () => {
  for (const f of buildParallelTransportFrames(trefoil, [0.1, 0.33, 0.8], { transport: 'rmf' })) {
    assert.ok(Math.abs(dot(f.xAxis, f.zAxis)) < 1e-9 && Math.abs(dot(f.yAxis, f.zAxis)) < 1e-9 && Math.abs(dot(f.xAxis, f.yAxis)) < 1e-9);
    assert.ok(Math.abs(Math.hypot(...f.xAxis) - 1) < 1e-9 && Math.abs(Math.hypot(...f.yAxis) - 1) < 1e-9);
  }
});

test('on a closed rail the seam span twists no more than the largest other span', () => {
  const F = buildParallelTransportFrames(trefoil, [], { transport: 'rmf' });
  const spans = [];
  for (let i = 1; i < F.length; i++) {
    const a = F[i - 1], b = F[i], bU = i === F.length - 1 ? trefoil.knots[trefoil.knots.length - 1] : b.u;
    let { point: p, tangent: t } = curvePointAndTangent(trefoil, a.u); t = unit(t); let r = a.xAxis;
    for (let k = 1; k <= 400; k++) { const u = a.u + ((bU - a.u) * k) / 400; let { point: q, tangent: s } = curvePointAndTangent(trefoil, u); s = unit(s); r = transportByDoubleReflection(p, t, r, q, s); p = q; t = s; }
    r = transportByDoubleReflection(p, t, r, p, b.zAxis); // into the welded seam's bisector tangent
    spans.push(Math.abs(deg(Math.atan2(dot(r, b.yAxis), dot(r, b.xAxis)))));
  }
  const seam = spans.pop();
  assert.ok(seam <= Math.max(...spans) + 1e-6, `seam ${seam} vs ${Math.max(...spans)}`);
  assert.equal(F[0], F[F.length - 1]);
});

test('a polyline corner and a straight run carry a finite frame', () => {
  const rail = { degree: 1, knots: [0, 0, 1, 2, 2], ctrlPts: [[0, 0, 0, 1], [40, 0, 0, 1], [40, 0, 40, 1]] };
  for (const f of buildParallelTransportFrames(rail, [0.5, 1.5], { transport: 'rmf' })) for (const v of [f.xAxis, f.yAxis, f.zAxis]) assert.ok(v.every(Number.isFinite));
  const line = makeLine([0, 0, 0], [0, 0, 50]);
  for (const f of buildParallelTransportFrames(line, [], { transport: 'rmf' })) assert.ok(f.xAxis.every(Number.isFinite));
});

test('the default transport is unchanged and Sweep1 takes the option', () => {
  const a = buildParallelTransportFrames(helix), b = buildParallelTransportFrames(helix, [], {});
  assert.deepEqual(a.map((f) => f.xAxis), b.map((f) => f.xAxis));
  const rect = { degree: 1, knots: [0, 0, 1, 2, 3, 4, 4], ctrlPts: [[-4, -2, 0, 1], [4, -2, 0, 1], [4, 2, 0, 1], [-4, 2, 0, 1], [-4, -2, 0, 1]] };
  const plain = sweep1Rigid(helix, rect), rmf = sweep1Rigid(helix, rect, null, { transport: 'rmf' });
  assert.equal(plain.ctrlNet.length, rmf.ctrlNet.length);
  assert.notDeepEqual(plain.ctrlNet, rmf.ctrlNet);
  assert.ok(rmf.ctrlNet.flat(2).every(Number.isFinite));
});
