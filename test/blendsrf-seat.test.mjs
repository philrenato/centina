// A fitted One-surface junction is seated on its edges: its trimmed boundary
// lies within the seat tolerance of the tube rims and of the free edges the
// face above shares with the face below, close enough for Join to pair them
// (JOIN_TOLERANCE 0.001 mm), and still meets the rims tangentially.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { blendSurfaceJunction, patchFromSurfaceEdges } from '../kernel/blendsrf.mjs';
import { makeCircle, extrude } from '../kernel/primitives.mjs';
import { surfacePoint } from '../kernel/surface.mjs';
import { readFileSync } from 'node:fs';

const tube = (deg, r) => { const t = deg * Math.PI / 180, a = [Math.cos(t), Math.sin(t), 0], b = [-Math.sin(t), Math.cos(t), 0]; return extrude(makeCircle([25 * a[0], 25 * a[1], 0], b, [0, 0, 1], r, 4), a, 40); };
const opts = (n) => ({ continuity: 2, bulge: 1, reach: 0.25, stations: 16, flips: Array(n).fill(false), square: true, continuities: Array(n).fill(null), onePatch: true, edgeTolerance: 3e-4 });

for (const [label, degs, r] of [['a cross', [0, 90, 180, 270], 10], ['a T', [0, 90, 180], 10], ['five tubes', [0, 72, 144, 216, 288], 8]]) {
  test(`One surface over ${label}: the trim lies on the rims and the shared free edges`, () => {
    const j = blendSurfaceJunction(degs.map((d) => ({ srf: tube(d, r), edge: 'v0' })), opts(degs.length));
    assert.ok(j.ok, j.reason);
    assert.ok(j.patched.trimGap != null && j.patched.trimGap <= 4e-4, `trim ${j.patched.trimGap} mm off its edges`);
    assert.ok(j.patched.edgeAngle < 2, `${j.patched.edgeAngle} degrees at the rims`);
    // Every trim vertex of the face above that lies on the waist (z = 0) is a
    // free edge the face below runs along too.
    const [up, down] = j.centre.map((f, i) => j.centreTrims[i].map(([u, v]) => surfacePoint(f, u, v)));
    let worst = 0;
    for (const p of up) {
      if (Math.abs(p[2]) > 1e-3) continue;
      let best = Infinity;
      for (let k = 0; k < down.length; k++) { const a = down[k], b = down[(k + 1) % down.length], d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], L2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2; const t = L2 > 0 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1] + (p[2] - a[2]) * d[2]) / L2)) : 0; best = Math.min(best, Math.hypot(a[0] + d[0] * t - p[0], a[1] + d[1] * t - p[1], a[2] + d[2] * t - p[2])); }
      worst = Math.max(worst, best);
    }
    assert.ok(worst < 1e-3, `the faces above and below part by ${worst} mm along their shared free edges`);
  });
}

test('weights in fitPatch hold a target: a weighted target is met more closely', async () => {
  const { fitPatch } = await import('../kernel/patch.mjs');
  const pts = [], params = [];
  for (let i = 0; i <= 10; i++) for (let k = 0; k <= 10; k++) { const u = i / 10, v = k / 10; pts.push([u * 10, v * 10, Math.sin(3 * u) * Math.cos(4 * v) * 3]); params.push([u, v]); }
  const plain = fitPatch(pts, { uCount: 5, vCount: 5, params, stiffness: 0.01 });
  const w = pts.map((_, i) => (i === 60 ? 50 : 1));
  const held = fitPatch(pts, { uCount: 5, vCount: 5, params, stiffness: 0.01, weights: w });
  const off = (f) => { const s = surfacePoint(f.srf, ...params[60]); return Math.hypot(s[0] - pts[60][0], s[1] - pts[60][1], s[2] - pts[60][2]); };
  assert.ok(off(held) < off(plain) / 10, `${off(held)} against ${off(plain)}`);
});

test('Patch on four half tubes is seated on their rims', () => {
  const doc = JSON.parse(readFileSync(new URL('./fixtures/half_tubes_cross.unreason', import.meta.url), 'utf8')).document;
  const r = patchFromSurfaceEdges(doc.objects.map((o) => ({ srf: o.surface.srf, edge: 'v0' })), { maxCount: 48, stiffness: 0.01, edgeTolerance: 3e-4 });
  assert.ok(r.ok, r.reason);
  assert.ok(r.trimGap != null && r.trimGap <= 4e-4 && r.edgeGap <= 4e-4, `trim ${r.trimGap}, edges ${r.edgeGap} mm`);
  assert.ok(r.edgeAngleInner < 2, `${r.edgeAngleInner} degrees along the rims`);
});
