import test from 'node:test';
import assert from 'node:assert/strict';
import {
  nSidedTangentPatch, nSidedPatchFolds, cornerNormalSpread,
  meanValueCoords, regularDomain,
} from '../kernel/tangentpatch.mjs';
import { sideVertexPatch } from '../kernel/cornerblend.mjs';

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const crs = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const nrm3 = (v) => { const L = Math.hypot(v[0], v[1], v[2]); return L > 0 ? [v[0] / L, v[1] / L, v[2] / L] : null; };
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const angAbs = (a, b) => Math.acos(Math.max(-1, Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])))) * 180 / Math.PI;

/* ═══ THE FIXTURE ═══════════════════════════════════════════════════════════
   An n-sided hole cut in a TWISTED analytic surface. The hole's boundary is a
   polygon in the surface's own parameters, so every boundary curve lies on the
   surface exactly and the cross-boundary tangent field is the surface's own
   tangent plane pushed inward — which makes the surface itself the thing the
   patch has to be tangent to, at every sample, with no fitting anywhere.

   ⚠ THE SURFACE HAS A NON-ZERO MIXED SECOND DERIVATIVE ON PURPOSE. A fixture
   built on a cylinder or a sphere of revolution has vanishing or symmetric
   twist at the hole's corners, which is exactly the condition the twist
   question is about, so such a fixture cannot ask it.                        */
function holeInSurface(n, { R = 1.2, u0 = 0.4, v0 = -0.3, wobble = 0, warp = 1 } = {}) {
  const f = (u, v) => warp * (0.30 * Math.sin(1.3 * u) * Math.cos(1.1 * v) + 0.17 * u * v);
  const fu = (u, v) => warp * (0.39 * Math.cos(1.3 * u) * Math.cos(1.1 * v) + 0.17 * v);
  const fv = (u, v) => warp * (-0.33 * Math.sin(1.3 * u) * Math.sin(1.1 * v) + 0.17 * u);
  const S = (u, v) => [u, v, f(u, v)];
  const Su = (u, v) => [1, 0, fu(u, v)];
  const Sv = (u, v) => [0, 1, fv(u, v)];
  const P = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    const rr = R * (1 + wobble * Math.sin((i + 1) * 2.1));
    P.push([u0 + rr * Math.cos(a), v0 + rr * Math.sin(a)]);
  }
  const cx = P.reduce((s, p) => s + p[0], 0) / n, cy = P.reduce((s, p) => s + p[1], 0) / n;
  const uvAt = (i, s) => {
    const a = P[i], b = P[(i + 1) % n];
    return [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s];
  };
  const boundary = [], tangent = [];
  for (let i = 0; i < n; i++) {
    boundary.push((s) => { const [u, v] = uvAt(i, s); return S(u, v); });
    tangent.push((s) => {
      const [u, v] = uvAt(i, s);
      let dx = cx - u, dy = cy - v; const L = Math.hypot(dx, dy); dx /= L; dy /= L;
      const su = Su(u, v), sv = Sv(u, v);
      return [su[0] * dx + sv[0] * dy, su[1] * dx + sv[1] * dy, su[2] * dx + sv[2] * dy];
    });
  }
  const surfaceNormalAt = (i, s) => { const [u, v] = uvAt(i, s); return nrm3(crs(Su(u, v), Sv(u, v))); };
  return { n, boundary, tangent, surfaceNormalAt };
}

/** The same hole, with a large component ADDED ALONG each boundary to the
 *  cross-tangent field.
 *
 *  That component is legitimate: it lies in the same tangent plane, so the G1
 *  request is unchanged and only the interior shape moves. What it does change
 *  is the ORIENTATION POLL — the chord from a boundary point to the far center
 *  swings past perpendicular partway along a side, so the sign test genuinely
 *  disagrees with itself along that side.
 */
function shearedHole(n, k, { R = 1.2, u0 = 0.4, v0 = -0.3, wobble = 0.25 } = {}) {
  const f = (u, v) => 0.30 * Math.sin(1.3 * u) * Math.cos(1.1 * v) + 0.17 * u * v;
  const fu = (u, v) => 0.39 * Math.cos(1.3 * u) * Math.cos(1.1 * v) + 0.17 * v;
  const fv = (u, v) => -0.33 * Math.sin(1.3 * u) * Math.sin(1.1 * v) + 0.17 * u;
  const S = (u, v) => [u, v, f(u, v)], Su = (u, v) => [1, 0, fu(u, v)], Sv = (u, v) => [0, 1, fv(u, v)];
  const P = [];
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    const rr = R * (1 + wobble * Math.sin((i + 1) * 2.1));
    P.push([u0 + rr * Math.cos(a), v0 + rr * Math.sin(a)]);
  }
  const cx = P.reduce((s, p) => s + p[0], 0) / n, cy = P.reduce((s, p) => s + p[1], 0) / n;
  const uvAt = (i, s) => { const a = P[i], b = P[(i + 1) % n]; return [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s]; };
  const boundary = [], tangent = [];
  for (let i = 0; i < n; i++) {
    const a = P[i], b = P[(i + 1) % n];
    const ex = b[0] - a[0], ey = b[1] - a[1], L = Math.hypot(ex, ey);
    boundary.push((s) => { const [u, v] = uvAt(i, s); return S(u, v); });
    tangent.push((s) => {
      const [u, v] = uvAt(i, s);
      let dx = cx - u, dy = cy - v; const M = Math.hypot(dx, dy); dx /= M; dy /= M;
      const w = k * Math.cos(Math.PI * s);
      const gx = dx + (w * ex) / L, gy = dy + (w * ey) / L;
      const su = Su(u, v), sv = Sv(u, v);
      return [su[0] * gx + sv[0] * gy, su[1] * gx + sv[1] * gy, su[2] * gx + sv[2] * gy];
    });
  }
  const surfaceNormalAt = (i, s) => { const [u, v] = uvAt(i, s); return nrm3(crs(Su(u, v), Sv(u, v))); };
  return { n, boundary, tangent, surfaceNormalAt };
}

/** A flat n-gon with straight sides and cross-tangents that lie in the plane. */
function flatNGon(n, { rx = 3, ry = 2.2 } = {}) {
  const V = [];
  for (let i = 0; i < n; i++) { const a = (2 * Math.PI * i) / n; V.push([rx * Math.cos(a), ry * Math.sin(a), 0]); }
  const boundary = [], tangent = [];
  for (let i = 0; i < n; i++) {
    const a = V[i], b = V[(i + 1) % n];
    boundary.push((s) => [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, 0]);
    tangent.push((s) => {
      const px = a[0] + (b[0] - a[0]) * s, py = a[1] + (b[1] - a[1]) * s;
      return [-px, -py, 0];
    });
  }
  return { V, boundary, tangent };
}

/** Domain-space walk along side i of the regular N-gon, `depth` inward. */
function domainPoint(D, i, t, depth) {
  const n = D.length, a = D[i], b = D[(i + 1) % n];
  const ex = b[0] - a[0], ey = b[1] - a[1], L = Math.hypot(ex, ey);
  let nx = ey / L, ny = -ex / L;
  if (nx * (0 - a[0]) + ny * (0 - a[1]) < 0) { nx = -nx; ny = -ny; }
  return [a[0] + ex * t + nx * depth, a[1] + ey * t + ny * depth];
}

function patchNormalAt(patch, x, y, h) {
  const p = patch.evaluateXY(x, y), px = patch.evaluateXY(x + h, y), py = patch.evaluateXY(x, y + h);
  if (!p || !px || !py) return null;
  return nrm3(crs(sub(px, p), sub(py, p)));
}

// ── THE CORRECTNESS ANCHOR ──────────────────────────────────────────────────

test('⚠ N=3 IS THE SAME SURFACE AS sideVertexPatch, not an approximation of it', () => {
  /* `sideVertexPatch` files a side by the corner OPPOSITE it: its boundary[j]
     runs from corners[j+1] to corners[j+2]. This module files a side by the
     corner it STARTS at. The two orderings differ by one index and nothing
     else, so the same hole handed to both must come back as the same points —
     if it does not, one of the two schemes is wrong and the fillet chain would
     have a discontinuity at every valence-3 corner. */
  const fx = holeInSurface(3, { wobble: 0.2 });
  const V = [0, 1, 2].map((i) => fx.boundary[i](0));
  const svp = sideVertexPatch({
    corners: V,
    boundary: [0, 1, 2].map((j) => fx.boundary[(j + 1) % 3]),
    tangent: [0, 1, 2].map((j) => fx.tangent[(j + 1) % 3]),
  });
  assert.equal(svp.ok, true, svp.reason);
  const mine = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent });
  assert.equal(mine.ok, true, mine.reason);

  let worst = 0, compared = 0;
  for (let i = 0; i <= 24; i++) {
    for (let j = 0; i + j <= 24; j++) {
      const l0 = i / 24, l1 = j / 24, l2 = 1 - l0 - l1;
      const p = mine.evaluate([l0, l1, l2]), q = svp.evaluate(l0, l1, l2);
      assert.equal(!!p, !!q, `the two disagree about whether (${l0}, ${l1}, ${l2}) is on the patch`);
      if (!p) continue;
      worst = Math.max(worst, dist(p, q)); compared += 1;
    }
  }
  assert.ok(compared > 300, `only ${compared} points compared`);
  /* The two write the Hermite's far end differently — one as a fixed vertex,
     one as a coordinate-weighted sum that reduces to it — so they agree to
     float rounding rather than bit for bit. The scale here is about 3 units. */
  assert.ok(worst < 1e-13, `the three-sided patches differ by ${worst.toExponential(3)}`);
});

test('the domain coordinates ARE barycentric coordinates at N=3', () => {
  // Nothing above would catch a coordinate scheme that is wrong on a triangle
  // if the anchor test only ever feeds coordinates in directly.
  const D = regularDomain(3);
  let worst = 0;
  for (let i = 1; i < 20; i++) {
    for (let j = 1; i + j < 20; j++) {
      const b = [i / 20, j / 20, 1 - i / 20 - j / 20];
      const x = b[0] * D[0][0] + b[1] * D[1][0] + b[2] * D[2][0];
      const y = b[0] * D[0][1] + b[1] * D[1][1] + b[2] * D[2][1];
      const l = meanValueCoords(D, x, y);
      for (let k = 0; k < 3; k++) worst = Math.max(worst, Math.abs(l[k] - b[k]));
    }
  }
  assert.ok(worst < 1e-14, `mean value coordinates must BE barycentric on a triangle, off by ${worst.toExponential(3)}`);
});

test('the domain is the regular N-gon it is documented to be', () => {
  // The parameterisation is a stated choice, so it is checkable rather than
  // whatever the vertex loop happens to emit.
  assert.equal(regularDomain(2), null);
  assert.equal(regularDomain(3.5), null);
  for (const n of [3, 4, 5, 7, 12]) {
    const D = regularDomain(n);
    assert.equal(D.length, n);
    let minE = Infinity, maxE = 0, area = 0;
    for (let i = 0; i < n; i++) {
      assert.ok(Math.abs(Math.hypot(D[i][0], D[i][1]) - 1) < 1e-14, `n=${n}: vertex ${i} is not on the unit circle`);
      const a = D[i], b = D[(i + 1) % n];
      const e = Math.hypot(b[0] - a[0], b[1] - a[1]);
      minE = Math.min(minE, e); maxE = Math.max(maxE, e);
      area += a[0] * b[1] - b[0] * a[1];
    }
    assert.ok(maxE - minE < 1e-14, `n=${n}: edges run ${minE} to ${maxE} — the domain is not regular`);
    assert.ok(area > 0, `n=${n}: the domain must wind counter-clockwise, or the coordinates come back negative`);
    assert.deepEqual(D[0], [1, 0], `n=${n}: vertex 0 is fixed so a caller sampling the domain gets the same points every time`);
  }
});

test('⚠ THE COORDINATE FORMULA IS ANSWERED AT ITS OWN SINGULARITIES, not approached', () => {
  /* At a domain vertex the weight divides by a zero radius; on a domain edge
     the half-angle tangent of an angle at pi is infinite. Both have exact
     answers, and a scheme that only ever gets asked about interior points can
     carry a broken branch for either indefinitely — the patch's corners and
     boundary are exactly where a caller sampling the domain will land. */
  for (const n of [3, 4, 6]) {
    const D = regularDomain(n);
    for (let i = 0; i < n; i++) {
      const atV = meanValueCoords(D, D[i][0], D[i][1]);
      assert.ok(atV, `n=${n}: no coordinates at domain vertex ${i}`);
      const want = new Array(n).fill(0); want[i] = 1;
      assert.deepEqual(atV, want, `n=${n}: vertex ${i} must be the Lagrange value, got ${atV}`);
      // Just off the vertex, the answer has to be near it rather than anywhere.
      const off = meanValueCoords(D, D[i][0] * 0.999999 + 1e-7, D[i][1] * 0.999999);
      assert.ok(off[i] > 0.99, `n=${n}: a step of 1e-6 off vertex ${i} gave coordinate ${off[i]}`);
      // And on the open edge, exactly linear between its two ends.
      for (const t of [0.2, 0.5, 0.83]) {
        const [x, y] = domainPoint(D, i, t, 0);
        const l = meanValueCoords(D, x, y);
        assert.ok(Math.abs(l[i] - (1 - t)) < 1e-12 && Math.abs(l[(i + 1) % n] - t) < 1e-12,
          `n=${n}: edge ${i} at ${t} gave ${l[i]}, ${l[(i + 1) % n]}`);
        for (let m = 0; m < n; m++) if (m !== i && m !== (i + 1) % n) assert.equal(l[m], 0, `n=${n}: edge ${i} leaks onto corner ${m}`);
      }
    }
  }
  /* LINEAR REPRODUCTION, which is the defining property and the one that fails
     silently. sum(lambda_m V_m) must be the query point itself — everywhere the
     coordinates are defined, including OUTSIDE the polygon and on the extension
     of an edge past its ends, where the half-angle tangent is a 0/0 of the
     other kind (an angle of zero rather than of pi) and its correct limit is
     also zero. Nothing in the patch reaches those points; a wrong branch there
     is still a wrong formula. */
  for (const n of [3, 4, 5, 6]) {
    const Dn = regularDomain(n);
    const pts = [];
    for (let i = 0; i <= 16; i++) for (let j = 0; j <= 16; j++) pts.push([-1.6 + (3.2 * i) / 16, -1.6 + (3.2 * j) / 16]);
    for (let i = 0; i < n; i++) {
      const a = Dn[i], b = Dn[(i + 1) % n];
      for (const t of [-0.7, -0.3, -0.05, 1.05, 1.3, 1.7]) pts.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
    let worst = 0, judged = 0;
    for (const [x, y] of pts) {
      const l = meanValueCoords(Dn, x, y);
      if (!l) continue;
      assert.ok(l.every(Number.isFinite), `n=${n}: non-finite coordinates at ${x}, ${y}`);
      let rx = 0, ry = 0;
      for (let m = 0; m < n; m++) { rx += l[m] * Dn[m][0]; ry += l[m] * Dn[m][1]; }
      worst = Math.max(worst, Math.hypot(rx - x, ry - y)); judged += 1;
    }
    assert.ok(judged > 250, `n=${n}: only ${judged} points`);
    assert.ok(worst < 1e-11, `n=${n}: the coordinates fail to reproduce their own point by ${worst.toExponential(3)}`);
  }

  // Which is what makes a domain-vertex query on the patch itself exact.
  const fx = holeInSurface(5, { wobble: 0.3 });
  const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent });
  const D = regularDomain(5);
  for (let i = 0; i < 5; i++) {
    assert.deepEqual(p.evaluateXY(D[i][0], D[i][1]), fx.boundary[i](0), `domain vertex ${i} must land on patch corner ${i}`);
  }
});

test('the coordinates are a positive partition of unity inside any N-gon', () => {
  // Positivity is what makes the weight sum non-zero, which is what makes the
  // claim "there is no central pole" true rather than hoped for.
  for (const n of [3, 4, 5, 6, 9, 12]) {
    const D = regularDomain(n);
    let worstSum = 0, minL = 1, judged = 0;
    for (let i = 0; i <= 40; i++) {
      for (let j = 0; j <= 40; j++) {
        const x = -0.95 + (1.9 * i) / 40, y = -0.95 + (1.9 * j) / 40;
        let inside = true;
        for (let k = 0; k < n; k++) {
          const a = D[k], b = D[(k + 1) % n];
          if ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) < 1e-3) inside = false;
        }
        if (!inside) continue;
        const l = meanValueCoords(D, x, y);
        assert.ok(l, `no coordinates at an interior point of a ${n}-gon`);
        worstSum = Math.max(worstSum, Math.abs(l.reduce((s, v) => s + v, 0) - 1));
        minL = Math.min(minL, Math.min(...l)); judged += 1;
      }
    }
    assert.ok(judged > 100, `n=${n}: only ${judged} interior samples`);
    assert.ok(worstSum < 1e-14, `n=${n}: coordinates sum to 1 only within ${worstSum.toExponential(3)}`);
    assert.ok(minL > 0, `n=${n}: a coordinate reached ${minL} inside the domain — the weight sum can vanish there`);
  }
});

// ── EXACTNESS ───────────────────────────────────────────────────────────────

test('a FLAT n-gon comes back EXACTLY flat, for every N — not approximately', () => {
  for (const n of [3, 4, 5, 6, 8, 10]) {
    const fx = flatNGon(n);
    const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent });
    assert.equal(p.ok, true, `n=${n}: ${p.reason}`);
    const D = regularDomain(n);
    let worstZ = 0, judged = 0;
    for (let i = 0; i <= 30; i++) {
      for (let j = 0; j <= 30; j++) {
        const l = meanValueCoords(D, -1 + (2 * i) / 30, -1 + (2 * j) / 30);
        if (!l || l.some((v) => v < 0)) continue;
        const q = p.evaluate(l);
        if (!q) continue;
        worstZ = Math.max(worstZ, Math.abs(q[2])); judged += 1;
      }
    }
    assert.ok(judged > 200, `n=${n}: only ${judged} samples`);
    assert.equal(worstZ, 0, `n=${n}: a patch on straight coplanar sides must be flat, got |z| up to ${worstZ}`);
  }
});

test('⚠ WITH STRAIGHT SIDES AND NO CROSS-TANGENT THE PATCH IS THE LINEAR MAP EXACTLY', () => {
  /* Every side's interpolant collapses to sum(lambda_m V_m) independently of
     the side, so the blend of them is that same sum whatever the weights are.
     This is linear precision, and it holds for corners placed anywhere in
     space — not only coplanar ones, which is why it is a stronger statement
     than the flatness test above and catches a weight scheme that is merely
     normalized rather than correct. */
  for (const n of [3, 5, 7, 9]) {
    const V = [];
    for (let i = 0; i < n; i++) {
      const a = (2 * Math.PI * i) / n;
      V.push([3 * Math.cos(a), 2.2 * Math.sin(a), Math.sin(3 * i) * 1.7]);
    }
    const boundary = [];
    for (let i = 0; i < n; i++) {
      const a = V[i], b = V[(i + 1) % n];
      boundary.push((s) => [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s]);
    }
    const p = nSidedTangentPatch({ boundary, validate: false });
    assert.equal(p.ok, true, `n=${n}: ${p.reason}`);
    assert.equal(p.continuity, 'G0', 'without a cross-tangent field it must not claim G1');
    const D = regularDomain(n);
    let worst = 0, judged = 0;
    for (let i = 0; i <= 24; i++) {
      for (let j = 0; j <= 24; j++) {
        const l = meanValueCoords(D, -1 + (2 * i) / 24, -1 + (2 * j) / 24);
        if (!l || l.some((v) => v < 0)) continue;
        const q = p.evaluate(l);
        if (!q) continue;
        const w = [0, 0, 0];
        for (let m = 0; m < n; m++) { w[0] += l[m] * V[m][0]; w[1] += l[m] * V[m][1]; w[2] += l[m] * V[m][2]; }
        worst = Math.max(worst, dist(q, w)); judged += 1;
      }
    }
    assert.ok(judged > 150, `n=${n}: only ${judged} samples`);
    assert.ok(worst < 1e-14, `n=${n}: linear precision off by ${worst.toExponential(3)}`);
  }
});

test('every boundary is reproduced EXACTLY, which is what watertight depends on', () => {
  for (const n of [3, 4, 5, 6, 7, 8]) {
    const fx = holeInSurface(n, { wobble: 0.25 });
    const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent });
    assert.equal(p.ok, true, `n=${n}: ${p.reason}`);
    const D = regularDomain(n);
    let worst = 0;
    for (let i = 0; i < n; i++) {
      for (let k = 0; k <= 32; k++) {
        const t = k / 32;
        const [x, y] = domainPoint(D, i, t, 0);
        const q = p.evaluateXY(x, y), want = fx.boundary[i](t);
        assert.ok(q, `n=${n}: side ${i} at ${t} did not evaluate`);
        worst = Math.max(worst, dist(q, want));
      }
    }
    assert.ok(worst < 1e-12, `n=${n}: boundary reproduced to ${worst.toExponential(3)}, which is the seam a fill would leak through`);
  }
});

test('the N corners are returned outright, not approached through a 0/0', () => {
  for (const n of [3, 5, 8]) {
    const fx = holeInSurface(n, { wobble: 0.3 });
    const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent });
    assert.equal(p.ok, true, p.reason);
    const D = regularDomain(n);
    for (let m = 0; m < n; m++) {
      const l = new Array(n).fill(0); l[m] = 1;
      const q = p.evaluate(l);
      assert.deepEqual(q, fx.boundary[m](0), `n=${n}: corner ${m} must be exact — every weight vanishes there`);
      /* ⚠ AND ONLY AT THE CORNER. Naming the corner over a small DISC instead
         of at the point flattens the patch there and leaves a region whose
         partials are both zero, so the normal is 0/0 over an area rather than
         at a point — which is the pole this construction exists without. The
         patch has to still move at a domain radius of 1e-8. */
      const seen = [];
      const prev = D[(m + n - 1) % n], next = D[(m + 1) % n];
      for (const w of [0.2, 0.4, 0.6, 0.8]) {
        // Inside the interior angle, so the step is into the domain and not out of it.
        let dx = (1 - w) * (next[0] - D[m][0]) + w * (prev[0] - D[m][0]);
        let dy = (1 - w) * (next[1] - D[m][1]) + w * (prev[1] - D[m][1]);
        const L = Math.hypot(dx, dy); dx /= L; dy /= L;
        const v = p.evaluateXY(D[m][0] + 1e-8 * dx, D[m][1] + 1e-8 * dy);
        assert.ok(v, `n=${n}: no value 1e-8 from corner ${m}`);
        assert.ok(dist(v, q) > 0, `n=${n}: the patch is constant 1e-8 from corner ${m} — the corner has been given a disc`);
        seen.push(v);
      }
      for (let a = 0; a < seen.length; a++) {
        for (let b = a + 1; b < seen.length; b++) {
          assert.ok(dist(seen[a], seen[b]) > 0, `n=${n}: corner ${m} returns one point for every direction of approach`);
        }
      }
    }
  }
});

// ── TANGENCY: THE RESIDUAL IS THE RULER ─────────────────────────────────────

test('⚠ THE PATCH IS TANGENT TO THE SURFACE AROUND THE HOLE, AND THE RESIDUAL IS THE RULER', () => {
  /* The normal is measured by finite differences taken a small distance off the
     boundary. Both the offset and the difference step are errors OF THE
     MEASUREMENT. Refine them and a measurement error falls with them; a genuine
     tangent defect sits still. Asserting an absolute number at a coarse step
     pins the instrument, not the geometry, so the absolute bound is taken only
     at the fine end and the CONVERGENCE is the real statement. */
  for (const n of [3, 4, 5, 6, 7, 8]) {
    const fx = holeInSurface(n, { wobble: 0.25 });
    const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent });
    assert.equal(p.ok, true, `n=${n}: ${p.reason}`);
    assert.equal(p.continuity, 'G1 along the open boundaries, G0 at the corners');
    const D = regularDomain(n);
    const measure = (eps, h) => {
      let worst = 0, judged = 0;
      for (let i = 0; i < n; i++) {
        for (let k = 1; k < 20; k++) {
          const t = k / 20;
          const [x, y] = domainPoint(D, i, t, eps);
          const pn = patchNormalAt(p, x, y, h);
          const bn = fx.surfaceNormalAt(i, t);
          if (pn && bn) { worst = Math.max(worst, angAbs(pn, bn)); judged += 1; }
        }
      }
      assert.ok(judged > 15 * n, `n=${n}: only ${judged} tangency samples`);
      return worst;
    };
    const coarse = measure(2e-4, 1e-5), fine = measure(1e-5, 1e-6), finer = measure(2e-6, 2e-7);
    assert.ok(fine < coarse / 5,
      `n=${n}: the angle must fall with the measurement — coarse ${coarse.toExponential(3)}, fine ${fine.toExponential(3)}. A real tangent defect would sit still.`);
    assert.ok(finer < fine / 3,
      `n=${n}: and it must keep falling — fine ${fine.toExponential(3)}, finer ${finer.toExponential(3)}`);
    assert.ok(finer < 5e-3, `n=${n}: tangency error ${finer.toExponential(3)} degrees at the finest step is a visible crease`);
  }
});

test('⚠ THE PATCH AGREES IN DIRECTION WITH THE SURFACE, not merely in plane', () => {
  /* An everted patch has the RIGHT tangent plane everywhere and is folded flat
     against its neighbor; an unsigned angle reads 0.00 degrees for it. Only
     the signed comparison can see it. */
  for (const n of [3, 5, 6]) {
    const fx = holeInSurface(n, { wobble: 0.25 });
    const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent });
    const D = regularDomain(n);
    let worstSigned = 1;
    for (let i = 0; i < n; i++) {
      for (let k = 1; k < 12; k++) {
        const t = k / 12;
        const [x, y] = domainPoint(D, i, t, 1e-4);
        const pn = patchNormalAt(p, x, y, 1e-5);
        const bn = fx.surfaceNormalAt(i, t);
        if (!pn || !bn) continue;
        worstSigned = Math.min(worstSigned, pn[0] * bn[0] + pn[1] * bn[1] + pn[2] * bn[2]);
      }
    }
    assert.ok(Math.abs(worstSigned) > 0.999, `n=${n}: the normals must stay in one plane, got ${worstSigned}`);
    assert.ok(worstSigned > 0, `n=${n}: the patch is everted against the surface it fills — signed agreement ${worstSigned}`);
  }
});

// ── THE CORNERS: TWIST, MEASURED RATHER THAN ASSUMED ────────────────────────

test('⚠ AT A CORNER THE LIMIT NORMAL EXISTS WHEN THE INPUT IS COMPATIBLE — measured, by refining the approach', () => {
  /* A corner is a 0/0: every weight vanishes. Whether the patch is G1 there is
     therefore a question about a LIMIT, and the only honest way to ask it is to
     shrink the approach radius and watch. A spread that falls linearly with the
     radius means the limit exists and the fan is merely sampling a curved
     region; a spread that stands still means there is no limit. */
  for (const n of [4, 5, 6]) {
    const fx = holeInSurface(n, { wobble: 0.2 });
    const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent });
    assert.equal(p.ok, true, p.reason);
    for (let m = 0; m < n; m++) {
      const a = cornerNormalSpread(p, m, { radius: 1e-2 });
      const b = cornerNormalSpread(p, m, { radius: 1e-3 });
      const c = cornerNormalSpread(p, m, { radius: 1e-4 });
      assert.equal(a.ok && b.ok && c.ok, true, `n=${n} corner ${m} could not be sampled`);
      assert.ok(b.spreadDeg < a.spreadDeg / 5,
        `n=${n} corner ${m}: the spread must fall with the approach — ${a.spreadDeg} at 1e-2, ${b.spreadDeg} at 1e-3`);
      assert.ok(c.spreadDeg < b.spreadDeg / 5,
        `n=${n} corner ${m}: and keep falling — ${b.spreadDeg} at 1e-3, ${c.spreadDeg} at 1e-4`);
      assert.ok(c.spreadDeg < 0.05, `n=${n} corner ${m}: limit normal still spread over ${c.spreadDeg} degrees`);
    }
  }
});

test('⚠ AND WHEN THE INPUT CREASES AT A CORNER THE SPREAD DOES NOT FALL — so the measurement can fail', () => {
  /* Without this the test above is decorative: any measurement that always
     converges is measuring the ruler. One side's cross-tangent is tilted out of
     its neighbor's plane near a shared corner, which is a genuine crease in
     the DATA, and the spread must then stand still at every radius. */
  const n = 5;
  const fx = holeInSurface(n, { wobble: 0.2 });
  const bad = fx.tangent.slice();
  const t0 = fx.tangent[0];
  bad[0] = (s) => { const d = t0(s); return [d[0], d[1], d[2] + 1.2 * s * s]; };
  const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: bad, cornerAngleToleranceDeg: 180, validate: false });
  assert.equal(p.ok, true, p.reason);
  const spreads = [1e-2, 1e-3, 1e-4, 1e-5].map((r) => cornerNormalSpread(p, 1, { radius: r }).spreadDeg);
  assert.ok(spreads.every((s) => s > 20),
    `a corner with no limit normal must keep its spread at every radius, got ${spreads.map((s) => s.toFixed(2)).join(', ')}`);
  assert.ok(spreads[3] > spreads[0] * 0.5, 'and it must not be quietly converging either');
});

test('⚠ A CORNER WHOSE TWO SIDES NAME DIFFERENT TANGENT PLANES IS REFUSED AT BUILD TIME', () => {
  // The refusal has to name the corner, or a caller cannot act on it.
  const fx = holeInSurface(5, { wobble: 0.2 });
  const bad = fx.tangent.slice();
  const t0 = fx.tangent[0];
  bad[0] = (s) => { const d = t0(s); return [d[0], d[1], d[2] + 1.2 * s * s]; };
  const r = nSidedTangentPatch({ boundary: fx.boundary, tangent: bad });
  assert.equal(r.ok, false, 'a fill that cannot be tangent to both neighbors must refuse, not reproduce the crease silently');
  assert.match(r.reason, /at corner 1 .*crease/);
  assert.ok(r.cornerAngleDeg > 20, `the reported disagreement ${r.cornerAngleDeg} must be the real one`);
  // And a caller who genuinely wants a crease running into that corner can say so.
  assert.equal(nSidedTangentPatch({ boundary: fx.boundary, tangent: bad, cornerAngleToleranceDeg: 180, validate: false }).ok, true);
  // A well-formed hole is NOT refused by the same gate — otherwise it is a wall.
  assert.equal(nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent }).ok, true);
});

// ── FOLDING ─────────────────────────────────────────────────────────────────

test('the validator passes well-formed holes at every N it is given', () => {
  for (const n of [3, 4, 5, 6, 7, 8, 10]) {
    const fx = holeInSurface(n, { wobble: 0.25 });
    const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent, validate: false });
    assert.equal(p.ok, true, `n=${n}: ${p.reason}`);
    const v = nSidedPatchFolds(p);
    assert.equal(v.folds, false, `n=${n}: ${v.reason}`);
    assert.ok(v.samples > 500, `n=${n}: only ${v.samples} samples judged`);
  }
});

test('⚠ A FOLD IS DETECTED, AND THE BUILDER HONOURS THE VERDICT rather than reporting it', () => {
  // Past the cross-tangent reach the interpolant can survive, the rays cross
  // and the surface everts. The builder must refuse it by default.
  const fx = holeInSurface(5, { wobble: 0.3, R: 1.6 });
  for (const ts of [3, 4, 6]) {
    const loose = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent, tangentScale: ts, validate: false });
    const v = nSidedPatchFolds(loose);
    assert.equal(v.folds, true, `tangentScale ${ts} folds and the validator must catch it — it reported ${v.reversals} reversals, worst ${v.worstAdjacentDeg.toFixed(1)} degrees`);
    assert.ok(v.reversals > 0);
    const built = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent, tangentScale: ts });
    assert.equal(built.ok, false, `tangentScale ${ts}: a folded patch must be refused, not returned with a note`);
    assert.match(built.reason, /folds over itself/);
    assert.equal(typeof built.evaluate, 'undefined', 'a refusal must not hand back an evaluable patch');
  }
  // And the default scale is comfortably inside the working range for the same hole.
  assert.equal(nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent }).ok, true);
});

test('⚠ A BOUNDARY-HUGGING FOLD IS CAUGHT — the interior grid alone CERTIFIES this one', () => {
  /* The interpolant is exact on the boundary and blends hardest just inside it,
     so a fold can live nearer a side than any grid coarse enough to run ever
     samples. This hole folds in exactly that band, and the two sweeps disagree
     about whether it is a surface at all: the interior grid on its own reports
     no reversal and passes it, while the ribbon finds thirty and refuses. */
  const fx = holeInSurface(4, { wobble: 0.4, R: 1.4, warp: 16 });
  const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent, validate: false });
  assert.equal(p.ok, true, p.reason);
  const full = nSidedPatchFolds(p);
  const gridOnly = nSidedPatchFolds(p, { ribbon: 0, depths: [] });
  assert.equal(gridOnly.folds, false,
    'the interior grid must be blind to this one, or the fixture is not exercising the ribbon');
  assert.equal(gridOnly.reversals, 0);
  assert.ok(gridOnly.samples > 100, `the blind sweep must still be a real sweep, got ${gridOnly.samples} samples`);
  assert.equal(full.folds, true, `the shipped sweep must catch it: ${JSON.stringify(full)}`);
  assert.ok(full.reversals >= 10, `only ${full.reversals} reversals were found in the boundary band`);
  assert.match(full.reason, /folds over itself/);
  // And the builder refuses it, which is the point of finding it at all.
  assert.equal(nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent }).ok, false);
});

test('a patch too small to judge is NOT passed', () => {
  // A sweep that finds nothing must not read as a clean bill of health.
  const fx = flatNGon(5);
  const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent, validate: false });
  const v = nSidedPatchFolds(p, { grid: 1, ribbon: 0, depths: [] });
  assert.equal(v.ok, false);
  assert.match(v.reason, /cannot be judged/);
  assert.equal(nSidedPatchFolds(null).ok, false);
  assert.equal(nSidedPatchFolds({ domain: { vertices: [] } }).ok, false);
});

// ── REFUSALS ────────────────────────────────────────────────────────────────

test('the patch refuses a malformed request rather than building something plausible', () => {
  const fx = holeInSurface(4);
  assert.match(nSidedTangentPatch({ boundary: fx.boundary.slice(0, 2) }).reason, /at least three boundary curves/);
  assert.match(nSidedTangentPatch({ boundary: [1, 2, 3] }).reason, /not an evaluable curve/);
  assert.match(nSidedTangentPatch({}).reason, /at least three boundary curves/);
  assert.match(nSidedTangentPatch({ boundary: fx.boundary, tangentScale: -1 }).reason, /non-negative/);
  assert.match(nSidedTangentPatch({ boundary: [() => [0, 0, 0], () => [0, 0, 0], () => [0, 0, 0]] }).reason, /no loop to fill/);
});

test('⚠ AN OPEN LOOP IS REFUSED WITH ITS GAP, not welded over', () => {
  /* Every side would still be reproduced exactly — on a boundary that is not
     the hole's. The result looks like a patch and leaves a slot. */
  const fx = holeInSurface(4);
  const open = fx.boundary.slice();
  const b3 = fx.boundary[3];
  open[3] = (s) => { const p = b3(s); return [p[0], p[1] + 0.01 * s, p[2]]; };
  const r = nSidedTangentPatch({ boundary: open, tangent: fx.tangent });
  assert.equal(r.ok, false);
  assert.match(r.reason, /do not close into a loop/);
  assert.match(r.reason, /1\.000e-2/);
  /* ⚠ THE TOLERANCE IS PINNED FROM BOTH SIDES, or it can be widened a
     millionfold and this test still passes on a gap as gross as the one above.
     A gap of 1e-6 on a hole three units across is a leak a weld will not close
     and must be refused; float noise from a caller's own arithmetic must not
     be. */
  const gapped = (g) => {
    const b = fx.boundary.slice();
    const b1 = fx.boundary[1];
    b[1] = (s) => { const p = b1(s); return [p[0], p[1] + g * s, p[2]]; };
    return nSidedTangentPatch({ boundary: b, tangent: fx.tangent, validate: false });
  };
  assert.equal(gapped(1e-6).ok, false, 'a 1e-6 gap on a three-unit hole is a leak, not noise');
  assert.equal(gapped(1e-13).ok, true, 'float noise from a caller\'s own arithmetic must not be refused');
  // And a caller whose curves come from a fit can say what "closed" means.
  assert.equal(gapped(1e-6).ok, false);
  assert.equal(nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent, loopTolerance: 1e-3, validate: false }).ok, true);
});

test('⚠ A MALFORMED TANGENT ARRAY IS REFUSED, NOT SILENTLY DEMOTED TO A G0 FILL', () => {
  /* Building the positional-only patch instead hands the caller a creased fill
     that reads as a geometry problem rather than as their own bad argument.
     Absent is a request; wrong is a defect. */
  const fx = holeInSurface(4);
  assert.match(nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent.slice(0, 3) }).reason, /4 evaluable fields/);
  assert.match(nSidedTangentPatch({ boundary: fx.boundary, tangent: [fx.tangent[0], null, fx.tangent[2], fx.tangent[3]] }).reason, /4 evaluable fields/);
  assert.match(nSidedTangentPatch({ boundary: fx.boundary, tangent: 'inward' }).reason, /4 evaluable fields/);
  // Omitting it entirely is a legitimate request for a G0 fill.
  const g0 = nSidedTangentPatch({ boundary: fx.boundary });
  assert.equal(g0.ok, true, g0.reason);
  assert.equal(g0.continuity, 'G0');
  // A field that is zero everywhere names no direction and is refused by side.
  const dead = fx.tangent.map((t, i) => (i === 2 ? () => [0, 0, 0] : t));
  assert.match(nSidedTangentPatch({ boundary: fx.boundary, tangent: dead }).reason, /side\(s\) 2: no usable cross-tangent/);
});

test('supplied corners are checked against the curves, not trusted', () => {
  const fx = holeInSurface(4);
  const right = [0, 1, 2, 3].map((i) => fx.boundary[i](0));
  assert.equal(nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent, corners: right }).ok, true);
  const wrong = right.map((p, i) => (i === 2 ? [p[0] + 0.5, p[1], p[2]] : p));
  const r = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent, corners: wrong });
  assert.equal(r.ok, false);
  assert.match(r.reason, /corner 2 is .* from where boundary 2 actually starts/);
  assert.match(nSidedTangentPatch({ boundary: fx.boundary, corners: right.slice(0, 3) }).reason, /one point per side/);
});

test('off-domain coordinates get no answer at all', () => {
  const fx = holeInSurface(4);
  const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent });
  assert.equal(p.evaluate([0.5, 0.5, 0.5, 0.5]), null, 'coordinates that do not sum to one are not a point of the domain');
  assert.equal(p.evaluate([1.2, -0.2, 0, 0]), null, 'a negative coordinate extrapolates the Hermite outside the curve it came from');
  assert.equal(p.evaluate([0.5, 0.5, 0]), null, 'the wrong number of coordinates');
  assert.equal(p.evaluate([0.25, 0.25, 0.25, NaN]), null);
  assert.equal(p.evaluate('middle'), null);
  /* ⚠ AND A COORDINATE VECTOR CAN BE PERFECTLY WELL FORMED AND STILL NAME NO
     POINT. Two OPPOSITE corners at a half each is non-negative and sums to one,
     yet every side has one of its two own coordinates at zero, so every weight
     vanishes and the blend is 0/0. Nothing in the domain produces it, and a
     caller doing their own arithmetic can. The answer is that there is no
     point, not a vector of NaN that propagates into a mesh. */
  assert.equal(p.evaluate([0.5, 0, 0.5, 0]), null);
  assert.equal(p.evaluate([0, 0.5, 0, 0.5]), null);
  // Float drift of a few ulps from a caller's own arithmetic is fine.
  assert.ok(p.evaluate([0.25, 0.25, 0.25, 0.25 + 1e-12]));
  // And an ordinary edge point, where exactly one weight survives, is a point.
  assert.ok(p.evaluate([0.4, 0.6, 0, 0]));
});

// ── THE POLE QUESTION, ASKED DIRECTLY ───────────────────────────────────────

test('⚠ THERE IS NO CENTRAL POLE AND NO BOUNDARY 0/0 — the interior is a value, not a limit', () => {
  /* The far point of each side's Hermite is 0/0 on that side, and the
     construction removes the division algebraically instead of guarding it. If
     a guard had been left in, the patch would be discontinuous or NaN in a thin
     band, and the way to see that is to walk INTO each side from a distance the
     guard would have to be smaller than. */
  for (const n of [3, 5, 7]) {
    const fx = holeInSurface(n, { wobble: 0.3 });
    const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent });
    const D = regularDomain(n);
    for (let i = 0; i < n; i++) {
      let prev = null;
      for (const depth of [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 0]) {
        const [x, y] = domainPoint(D, i, 0.37, depth);
        const q = p.evaluateXY(x, y);
        assert.ok(q && q.every(Number.isFinite), `n=${n} side ${i} at depth ${depth}: no finite value`);
        if (prev) assert.ok(dist(q, prev) < 0.05, `n=${n} side ${i}: the patch jumps ${dist(q, prev)} between depths — a guard band`);
        prev = q;
      }
      const onEdge = domainPoint(D, i, 0.37, 0);
      assert.ok(dist(p.evaluateXY(onEdge[0], onEdge[1]), fx.boundary[i](0.37)) < 1e-12);
    }
    // And the center of the domain, where a split-into-quads scheme would put
    // its extraordinary point, is an ordinary evaluation with an ordinary normal.
    const centre = patchNormalAt(p, 0, 0, 1e-6);
    assert.ok(centre, `n=${n}: the center of the domain has no normal`);
    const near = patchNormalAt(p, 1e-4, 1e-4, 1e-6);
    assert.ok(angAbs(centre, near) < 0.01, `n=${n}: the normal swings ${angAbs(centre, near)} degrees across the domain center`);
  }
});

// ── WHAT SURVIVED THE MUTATIONS ─────────────────────────────────────────────
// Each of these closes a mutation of the module that passed every other test
// in this file.

test('⚠ THE WEIGHT MUST VANISH ON THE OTHER SIDES TO SECOND ORDER, not merely vanish', () => {
  /* Dropping the square from (lambda_i lambda_i+1)^2 leaves a scheme that still
     reproduces every boundary exactly, still reproduces a flat n-gon exactly,
     and still has linear precision — because on the boundary the rival weights
     are zero either way. What it loses is G1: the rivals then fall only
     linearly with the distance from a side, so the patch's tangent plane there
     is a blend of the wanted one and its neighbors'. It shows up ONLY in a
     tangency measurement, and only because that measurement converges.

     Held here by requiring the convergence to be first-order in the offset: the
     unsquared weight leaves a residual that stalls instead. */
  const fx = holeInSurface(5, { wobble: 0.25 });
  const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent });
  const D = regularDomain(5);
  const at = (eps) => {
    let worst = 0;
    for (let i = 0; i < 5; i++) {
      for (let k = 1; k < 12; k++) {
        const t = k / 12;
        const [x, y] = domainPoint(D, i, t, eps);
        const pn = patchNormalAt(p, x, y, eps * 0.05);
        const bn = fx.surfaceNormalAt(i, t);
        if (pn && bn) worst = Math.max(worst, angAbs(pn, bn));
      }
    }
    return worst;
  };
  const a = at(1e-3), b = at(1e-4), c = at(1e-5);
  assert.ok(b < a / 6 && c < b / 6,
    `the tangency residual must be proportional to the offset — ${a.toExponential(2)}, ${b.toExponential(2)}, ${c.toExponential(2)}. A weight that vanishes only to first order stalls here.`);
});

test('⚠ THE CROSS-TANGENT ORIENTATION IS ONE DECISION PER SIDE, and a per-sample one everts the fill', () => {
  /* The field is already smooth. Deciding its sign per sample injects a
     discontinuity it never had, wherever the chord to the far center swings
     past perpendicular — which an asymmetric hole does. The symptom is a fold,
     so the guard is that a hole built to have exactly that swing still passes
     the validator, and that its patch still agrees in direction with the
     surface all the way along the side where the swing happens. */
  const n = 5;
  const fx = shearedHole(n, 8);
  const p = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent });
  assert.equal(p.ok, true, p.reason);
  const D = regularDomain(n);

  /* THE PRECONDITION, CHECKED BEFORE THE CLAIM. A fixture on which the sign
     test never changes its mind cannot distinguish a per-side decision from a
     per-sample one, so the poll is re-run here and the split demanded. A 4:3
     split is the knife edge: a single mid-side sample would be arbitrary there
     and an input difference of 1e-9 would evert that whole side. */
  let swinging = 0, worstSplit = 99;
  for (let i = 0; i < n; i++) {
    let plus = 0, minus = 0, far = [0, 0, 0], k = 0;
    for (let m = 0; m < n; m++) {
      if (m === i || m === (i + 1) % n) continue;
      const V = fx.boundary[m](0);
      far = [far[0] + V[0], far[1] + V[1], far[2] + V[2]]; k += 1;
    }
    far = far.map((v) => v / k);
    for (let s = 1; s < 8; s++) {
      const d = nrm3(fx.tangent[i](s / 8));
      const chord = sub(far, fx.boundary[i](s / 8));
      if (d[0] * chord[0] + d[1] * chord[1] + d[2] * chord[2] >= 0) plus += 1; else minus += 1;
    }
    if (plus > 0 && minus > 0) { swinging += 1; worstSplit = Math.min(worstSplit, Math.max(plus, minus)); }
  }
  assert.ok(swinging >= 2, `only ${swinging} sides change their mind — the case is not exercised`);
  assert.ok(worstSplit <= 4, `the closest split is ${worstSplit}:${7 - worstSplit}, which is not the knife edge this guards`);

  /* THE CLAIM. A sign decided per sample flips mid-side and everts the patch
     from there on, which shows in the SIGNED agreement — not in the unsigned
     angle, which stays near zero for an everted surface. */
  let worstSigned = 1;
  for (let i = 0; i < n; i++) {
    for (let t = 1; t < 24; t++) {
      const [x, y] = domainPoint(D, i, t / 24, 1e-4);
      const pn = patchNormalAt(p, x, y, 1e-5);
      const bn = fx.surfaceNormalAt(i, t / 24);
      if (pn && bn) worstSigned = Math.min(worstSigned, pn[0] * bn[0] + pn[1] * bn[1] + pn[2] * bn[2]);
    }
  }
  assert.ok(worstSigned > 0.999, `a side reverses against the surface — signed agreement ${worstSigned}`);
  assert.equal(nSidedPatchFolds(p).folds, false);

  /* And the component along the boundary must have moved the interior shape
     WITHOUT moving the tangent plane, which is the reason it is allowed at all:
     the tangency residual still falls with the measuring step. */
  const measure = (eps, h) => {
    let worst = 0;
    for (let i = 0; i < n; i++) {
      for (let k = 1; k < 16; k++) {
        const [x, y] = domainPoint(D, i, k / 16, eps);
        const pn = patchNormalAt(p, x, y, h);
        const bn = fx.surfaceNormalAt(i, k / 16);
        if (pn && bn) worst = Math.max(worst, angAbs(pn, bn));
      }
    }
    return worst;
  };
  const coarse = measure(2e-4, 1e-5), fine = measure(1e-5, 1e-6);
  assert.ok(fine < coarse / 5, `the sheared field must still be exactly tangent — coarse ${coarse.toExponential(2)}, fine ${fine.toExponential(2)}`);
});

test('⚠ THE FIXTURE ITSELF IS NOT FLAT AND NOT UNTWISTED — otherwise most of this file proves nothing', () => {
  /* A hole cut in a plane, or in a surface with no mixed second derivative,
     passes every tangency and corner test here for reasons that have nothing to
     do with the patch. The fixture is checked before it is believed. */
  const fx = holeInSurface(5, { wobble: 0.25 });
  const ns = [];
  for (let i = 0; i < 5; i++) for (let k = 0; k < 5; k++) ns.push(fx.surfaceNormalAt(i, k / 5));
  let spread = 0;
  for (const a of ns) for (const b of ns) spread = Math.max(spread, angAbs(a, b));
  assert.ok(spread > 20, `the fixture surface must actually curve around the hole, normals spread only ${spread.toFixed(1)} degrees`);
  // Non-zero twist: the mixed partial of the height field, read numerically
  // over the region the hole actually occupies. A single point can sit at a
  // node of f_uv and read near zero on a surface that is genuinely twisted.
  const f = (u, v) => 0.30 * Math.sin(1.3 * u) * Math.cos(1.1 * v) + 0.17 * u * v;
  const h = 1e-4;
  let maxTwist = 0;
  for (let u = -1.2; u <= 2.0; u += 0.05) {
    for (let v = -1.9; v <= 1.3; v += 0.05) {
      const fuv = (f(u + h, v + h) - f(u + h, v - h) - f(u - h, v + h) + f(u - h, v - h)) / (4 * h * h);
      maxTwist = Math.max(maxTwist, Math.abs(fuv));
    }
  }
  assert.ok(maxTwist > 0.3, `the fixture must have real twist across the hole, largest |f_uv| = ${maxTwist}`);
});

test('⚠ THE FOLD VALIDATOR CAN STILL FAIL — a widened limit or a blind sweep must be visible', () => {
  /* A detector that never says no is decorative. Both of its criteria are
     exercised on the same known-bad patch: the reversal count and the crease
     limit. */
  const fx = holeInSurface(5, { wobble: 0.3, R: 1.6 });
  const bad = nSidedTangentPatch({ boundary: fx.boundary, tangent: fx.tangent, tangentScale: 6, validate: false });
  const v = nSidedPatchFolds(bad);
  assert.equal(v.folds, true);
  assert.ok(v.reversals > 5, `${v.reversals} reversals is too few to be a robust signal`);
  assert.ok(v.worstAdjacentDeg > 150, `worst adjacent turn ${v.worstAdjacentDeg}`);
  assert.match(v.reason, /folds over itself/);
  // The crease criterion catches what the reversal criterion does not: raise the
  // limit past the observed turn and the same patch passes, which is what a
  // widened tolerance would silently do.
  assert.equal(nSidedPatchFolds(bad, { creaseLimitDeg: 200 }).folds, true, 'reversals alone must still catch this one');
  const creasedNotFolded = nSidedPatchFolds(
    nSidedTangentPatch({ boundary: holeInSurface(5, { wobble: 0.3, R: 1.6 }).boundary, tangent: fx.tangent, tangentScale: 2.5, validate: false }),
    { creaseLimitDeg: 40 },
  );
  assert.equal(creasedNotFolded.folds, true, 'the crease criterion must be able to fire on its own');
  assert.equal(creasedNotFolded.reversals, 0, 'and it must be firing without any reversal, or it is not being tested');
  assert.match(creasedNotFolded.reason, /creases/);
});
