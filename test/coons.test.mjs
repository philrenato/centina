import test from 'node:test';
import assert from 'node:assert/strict';
import { coonsPatch } from '../kernel/coons.mjs';
import { curvePoint } from '../kernel/curve.mjs';
import { surfacePoint } from '../kernel/surface.mjs';

const line = (A, B) => ({ degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[...A, 1], [...B, 1]] });
const bez = (pts) => ({ degree: pts.length - 1, knots: [...pts.map(() => 0), ...pts.map(() => 1)], ctrlPts: pts.map((p) => [...p, 1]) });
const gap = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const at = (c, t) => curvePoint(c, c.knots[0] + (c.knots[c.knots.length - 1] - c.knots[0]) * t);

test('coonsPatch: every boundary is its curve to machine precision, at unequal degrees and knots', () => {
  // c0 cubic with an interior knot, c1 a line, c2 a curved quadratic, c3 a quartic — a mixed loop.
  const c0 = { degree: 3, knots: [0, 0, 0, 0, 0.4, 1, 1, 1, 1], ctrlPts: [[0, 0, 0, 1], [3, 2, 1, 1], [6, -1, 0, 1], [8, 1, 2, 1], [10, 0, 0, 1]] };
  const c1 = line([10, 0, 0], [10, 8, 3]);
  const c2 = bez([[10, 8, 3], [5, 11, 4], [0, 8, 1]]);
  const c3 = bez([[0, 8, 1], [-1, 6, 2], [1, 4, 0], [-2, 2, 1], [0, 0, 0]]);
  const S = coonsPatch(c0, c1, c2, c3);
  const u0 = S.knotsU[0], u1 = S.knotsU[S.knotsU.length - 1], v0 = S.knotsV[0], v1 = S.knotsV[S.knotsV.length - 1];
  let worst = 0;
  for (let i = 0; i <= 50; i++) {
    const t = i / 50;
    worst = Math.max(worst,
      gap(surfacePoint(S, u0 + (u1 - u0) * t, v0), at(c0, t)),
      gap(surfacePoint(S, u1, v0 + (v1 - v0) * t), at(c1, t)),
      gap(surfacePoint(S, u0 + (u1 - u0) * (1 - t), v1), at(c2, t)),
      gap(surfacePoint(S, u0, v0 + (v1 - v0) * (1 - t)), at(c3, t)));
  }
  assert.ok(worst < 1e-9, `boundary deviation ${worst.toExponential(2)}`);
  assert.equal(S.degU, 3); assert.equal(S.degV, 4);
});

test('coonsPatch: four straight sides give the bilinear sheet — the interior point is the bilinear point', () => {
  const S = coonsPatch(line([0, 0, 0], [10, 0, 0]), line([10, 0, 0], [10, 10, 5]), line([10, 10, 5], [0, 10, 0]), line([0, 10, 0], [0, 0, 0]));
  const m = surfacePoint(S, 0.5, 0.5);
  assert.ok(gap(m, [5, 5, 1.25]) < 1e-12, `${m}`);
});

test('coonsPatch: a side of zero length is a pole and the patch still holds its other three boundaries', () => {
  const C = [5, 5, 3];
  const c0 = bez([[0, 0, 0], [5, -2, 0], [10, 0, 0]]);
  const S = coonsPatch(c0, line([10, 0, 0], C), line(C, C), line(C, [0, 0, 0]));
  let worst = 0;
  for (let i = 0; i <= 20; i++) { const t = i / 20; worst = Math.max(worst, gap(surfacePoint(S, t, 0), at(c0, t)), gap(surfacePoint(S, t, 1), C)); }
  assert.ok(worst < 1e-9, `${worst.toExponential(2)}`);
});
