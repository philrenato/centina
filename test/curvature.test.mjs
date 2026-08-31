// CURVATURE — asserted against CLOSED FORMS, not against itself.
//
// Curvature is the easiest quantity in this kernel to get plausibly wrong: a
// version that ignores the weights of a rational surface produces numbers that
// vary smoothly, look reasonable on a plot, and are simply not the curvature.
// So every test here compares against a value known in advance — a sphere of
// radius R has K = 1/R^2 and |H| = 1/R everywhere, a cylinder has K = 0 and
// |H| = 1/(2R), a plane has both zero, a saddle has K < 0. Every primitive
// sphere and cylinder in this app is RATIONAL, which is exactly where a
// weight mistake shows.
import test from 'node:test';
import assert from 'node:assert/strict';
import { makeArc, makeLine, revolve, extrude } from '../kernel/primitives.mjs';
import { surfaceCurvature, surfaceDerivs2, minimumRadiusOfCurvature } from '../kernel/curvature.mjs';

const dom = (knots, deg) => [knots[deg], knots[knots.length - 1 - deg]];
function sampleInterior(srf, fn, n = 5) {
  const [uMin, uMax] = dom(srf.knotsU, srf.degU);
  const [vMin, vMax] = dom(srf.knotsV, srf.degV);
  for (let i = 1; i < n; i++) {
    for (let j = 1; j < n; j++) {
      const u = uMin + ((uMax - uMin) * i) / n;
      const v = vMin + ((vMax - vMin) * j) / n;
      fn(surfaceCurvature(srf, u, v), u, v);
    }
  }
}

function sphere(R) {
  const profile = makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], R, -Math.PI / 2, Math.PI);
  return revolve(profile, [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
}
function cylinder(R, h) {
  return revolve(makeLine([R, 0, 0], [R, 0, h]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI);
}
function plane(size) {
  return extrude(makeLine([0, 0, 0], [size, 0, 0]), [0, 1, 0], size);
}

test('⭐ a RATIONAL sphere gives its own closed-form curvature', () => {
  for (const R of [10, 50, 137.5]) {
    const s = sphere(R);
    let worstK = 0, worstH = 0, n = 0;
    sampleInterior(s, (c) => {
      if (!c.ok) return; // the poles are genuinely degenerate and reported as such
      n++;
      worstK = Math.max(worstK, Math.abs(c.K - 1 / (R * R)) * R * R);
      worstH = Math.max(worstH, Math.abs(Math.abs(c.H) - 1 / R) * R);
    });
    assert.ok(n > 8, `enough interior samples were evaluated on R=${R} (${n})`);
    assert.ok(worstK < 1e-6, `R=${R}: Gaussian curvature is 1/R^2, worst relative error ${worstK}`);
    assert.ok(worstH < 1e-6, `R=${R}: mean curvature magnitude is 1/R, worst relative error ${worstH}`);
  }
});

test('⭐ a RATIONAL cylinder is developable — K = 0, |H| = 1/(2R)', () => {
  const R = 25;
  const s = cylinder(R, 80);
  let worstK = 0, worstH = 0, n = 0;
  sampleInterior(s, (c) => {
    if (!c.ok) return;
    n++;
    worstK = Math.max(worstK, Math.abs(c.K) * R * R);
    worstH = Math.max(worstH, Math.abs(Math.abs(c.H) - 1 / (2 * R)) * R);
  });
  assert.ok(n > 8, `interior samples evaluated (${n})`);
  assert.ok(worstK < 1e-6, `a cylinder rolls flat, so K must be 0 — worst ${worstK}`);
  assert.ok(worstH < 1e-6, `and its mean curvature is 1/(2R) — worst relative error ${worstH}`);
});

test('a plane has no curvature at all', () => {
  const s = plane(60);
  sampleInterior(s, (c) => {
    assert.equal(c.ok, true);
    assert.ok(Math.abs(c.K) < 1e-9, `K is 0 on a plane, got ${c.K}`);
    assert.ok(Math.abs(c.H) < 1e-9, `H is 0 on a plane, got ${c.H}`);
  });
});

test('⭐ a saddle reads NEGATIVE Gaussian curvature — the sign carries meaning', () => {
  // z = (x^2 - y^2)/k is a genuine saddle: curving up one way and down the other.
  const rows = [];
  for (let i = 0; i < 4; i++) {
    const row = [];
    for (let j = 0; j < 4; j++) {
      const x = -30 + (60 * i) / 3, y = -30 + (60 * j) / 3;
      row.push([x, y, (x * x - y * y) / 120, 1]);
    }
    rows.push(row);
  }
  const clamped = (n, d) => {
    const k = [];
    for (let i = 0; i <= d; i++) k.push(0);
    for (let i = 1; i <= n - d - 1; i++) k.push(i / (n - d));
    for (let i = 0; i <= d; i++) k.push(1);
    return k;
  };
  const s = { degU: 3, degV: 3, knotsU: clamped(4, 3), knotsV: clamped(4, 3), ctrlNet: rows };
  let sawNegative = 0, count = 0;
  sampleInterior(s, (c) => { if (c.ok) { count++; if (c.K < -1e-9) sawNegative++; } });
  assert.ok(count > 8, `samples evaluated (${count})`);
  assert.equal(sawNegative, count, `every interior point of a saddle has K < 0 (${sawNegative}/${count})`);
});

test('a pole is REFUSED rather than returned as a huge number', () => {
  const s = sphere(40);
  const [uMin] = dom(s.knotsU, s.degU);
  const [vMin, vMax] = dom(s.knotsV, s.degV);
  const c = surfaceCurvature(s, uMin, (vMin + vMax) / 2);
  assert.equal(c.ok, false, 'the pole has no tangent plane');
  assert.equal(c.K, null);
  assert.match(c.reason, /pole|degenerate/i);
});

test('principal curvatures bracket the mean and multiply to the Gaussian', () => {
  const s = sphere(30);
  sampleInterior(s, (c) => {
    if (!c.ok) return;
    assert.ok(Math.abs(c.k1 * c.k2 - c.K) < 1e-9 * Math.max(1, Math.abs(c.K)), 'k1*k2 = K');
    assert.ok(Math.abs((c.k1 + c.k2) / 2 - c.H) < 1e-9 * Math.max(1, Math.abs(c.H)), '(k1+k2)/2 = H');
    assert.ok(c.k1 >= c.k2 - 1e-12, 'k1 is the larger root');
  });
});

test('⭐ minimum radius of curvature finds the tightest bend, and says nothing on a plane', () => {
  const R = 18;
  const s = sphere(R);
  const worst = minimumRadiusOfCurvature(s, 16, 16);
  assert.ok(worst, 'a sphere has a tightest bend');
  assert.ok(Math.abs(worst.radius - R) < 1e-6 * R,
    `on a sphere the tightest bend is R everywhere — got ${worst.radius}, expected ${R}`);

  const flat = plane(50);
  assert.equal(minimumRadiusOfCurvature(flat, 8, 8), null,
    'a plane has no tightest bend, and saying "infinity" would be a number nobody can use');
});

test('second partials of a plane vanish, and of a sphere do not', () => {
  const flat = plane(40);
  const [uMin, uMax] = dom(flat.knotsU, flat.degU);
  const [vMin, vMax] = dom(flat.knotsV, flat.degV);
  const d = surfaceDerivs2(flat, (uMin + uMax) / 2, (vMin + vMax) / 2);
  for (const k of ['Suu', 'Suv', 'Svv']) {
    assert.ok(Math.hypot(...d[k]) < 1e-9, `${k} vanishes on a plane, got ${d[k]}`);
  }
  const s = sphere(20);
  const ds = surfaceDerivs2(s, (dom(s.knotsU, s.degU)[0] + dom(s.knotsU, s.degU)[1]) / 2, (dom(s.knotsV, s.degV)[0] + dom(s.knotsV, s.degV)[1]) / 2);
  assert.ok(Math.hypot(...ds.Suu) > 1e-6, 'a sphere genuinely bends, so Suu does not vanish');
});
