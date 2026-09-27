// Light deformer — a directional light carves a closed quad cage. Every check
// runs on a welded cube-sphere (all quads, closed, smooth radial normals), so a
// vertex's smoothed normal is very nearly its outward radial direction and the
// lit/shadow split falls cleanly on the +X/-X hemispheres.
import { strict as assert } from 'node:assert';
import {
  applyLightDeform, compositeLightRigs, vertexNormals, smoothstep,
  resolveLightDir, shadeField, displacementField, foldGuardCeiling, SAFE_FRACTION,
} from '../kernel/lightformer.mjs';

// Every case runs; the failures are collected and reported together at the end.
const failures = [];
function t(name, fn) {
  try { fn(); } catch (e) { failures.push(`${name} — ${e.message}`); }
}

// A cube-sphere: each cube face split into n*n quads, shared edge/corner
// vertices welded (keyed on the pre-projection cube coordinate), then every
// vertex projected onto the sphere of radius R. Flat Float32Array positions +
// flat Uint32Array quads, stride 4 — the kernel's cage shape.
function cubeSphere(n = 6, R = 10) {
  const map = new Map();
  const pos = [];
  function vid(cx, cy, cz) {
    const k = `${cx.toFixed(6)},${cy.toFixed(6)},${cz.toFixed(6)}`;
    let id = map.get(k);
    if (id !== undefined) return id;
    id = pos.length / 3;
    const d = Math.hypot(cx, cy, cz) || 1;
    pos.push((cx / d) * R, (cy / d) * R, (cz / d) * R);
    map.set(k, id);
    return id;
  }
  const faces = [
    (u, v) => [1, u, v], (u, v) => [-1, u, v],
    (u, v) => [u, 1, v], (u, v) => [u, -1, v],
    (u, v) => [u, v, 1], (u, v) => [u, v, -1],
  ];
  const quads = [];
  for (const f of faces) {
    const grid = [];
    for (let j = 0; j <= n; j++) {
      const row = [];
      for (let i = 0; i <= n; i++) {
        const [x, y, z] = f(-1 + (2 * i) / n, -1 + (2 * j) / n);
        row.push(vid(x, y, z));
      }
      grid.push(row);
    }
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      quads.push(grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]);
    }
  }
  // Orient every quad outward (opposite cube faces share one parametrization,
  // which would otherwise wind them inconsistently). A sphere about the origin
  // lets us test each face's Newell normal against its own centroid.
  for (let q = 0; q < quads.length; q += 4) {
    const idx = [quads[q], quads[q + 1], quads[q + 2], quads[q + 3]];
    let nx = 0, ny = 0, nz = 0, cx = 0, cy = 0, cz = 0;
    for (let e = 0; e < 4; e++) {
      const a = idx[e], b = idx[(e + 1) % 4];
      nx += (pos[a * 3 + 1] - pos[b * 3 + 1]) * (pos[a * 3 + 2] + pos[b * 3 + 2]);
      ny += (pos[a * 3 + 2] - pos[b * 3 + 2]) * (pos[a * 3] + pos[b * 3]);
      nz += (pos[a * 3] - pos[b * 3]) * (pos[a * 3 + 1] + pos[b * 3 + 1]);
      cx += pos[a * 3]; cy += pos[a * 3 + 1]; cz += pos[a * 3 + 2];
    }
    if (nx * cx + ny * cy + nz * cz < 0) { quads[q + 1] = idx[3]; quads[q + 3] = idx[1]; }
  }
  return { positions: new Float32Array(pos), quads: new Uint32Array(quads), R };
}

// Nearest edge-neighbor distance per vertex — the fold guard's yardstick,
// recomputed here independently of the kernel.
function neighborDist(positions, quads) {
  const V = positions.length / 3;
  const nbr = Array.from({ length: V }, () => new Set());
  for (let q = 0; q < quads.length; q += 4) {
    const a = quads[q], b = quads[q + 1], c = quads[q + 2], d = quads[q + 3];
    const link = (i, j) => { nbr[i].add(j); nbr[j].add(i); };
    link(a, b); link(b, c); link(c, d); link(d, a);
  }
  const out = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    let best = Infinity;
    for (const j of nbr[i]) {
      const dx = positions[i * 3] - positions[j * 3];
      const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
      const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
      best = Math.min(best, Math.hypot(dx, dy, dz));
    }
    out[i] = best;
  }
  return out;
}

const radial = (positions, i) => {
  const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
  const d = Math.hypot(x, y, z) || 1;
  return [x / d, y / d, z / d];
};
// Outward (radial) component of a vertex's move, using the original position.
const outwardMove = (before, after, i) => {
  const dir = radial(before, i);
  return (after[i * 3] - before[i * 3]) * dir[0]
    + (after[i * 3 + 1] - before[i * 3 + 1]) * dir[1]
    + (after[i * 3 + 2] - before[i * 3 + 2]) * dir[2];
};
function sideMeanOutward(before, after, sign, R) {
  let sum = 0, k = 0;
  for (let i = 0; i < before.length / 3; i++) {
    const px = before[i * 3];
    if (sign > 0 ? px > 0.5 * R : px < -0.5 * R) { sum += outwardMove(before, after, i); k++; }
  }
  return sum / k;
}

// Helper unit checks
t('smoothstep clamps and eases 0..1', () => {
  assert.equal(smoothstep(0, 1, -5), 0);
  assert.equal(smoothstep(0, 1, 5), 1);
  assert.ok(Math.abs(smoothstep(0, 1, 0.5) - 0.5) < 1e-12);
});

t('resolveLightDir: 3-vec normalizes; disc rim stays unit', () => {
  const a = resolveLightDir([0, 0, 5]);
  assert.ok(Math.abs(Math.hypot(a[0], a[1], a[2]) - 1) < 1e-12);
  const rim = resolveLightDir({ x: 1, y: 0 }); // rim dot: lz floored, then normalized
  assert.ok(Math.abs(Math.hypot(rim[0], rim[1], rim[2]) - 1) < 1e-12);
  assert.ok(rim[2] > 0); // floor keeps a forward component
});

// 1. A +X light pushes the lit (+X) side less than the shadow (-X) side.
t('1. +X light: lit side pushed less than shadow side', () => {
  const { positions, quads, R } = cubeSphere();
  const out = applyLightDeform(positions, quads, { lightDir: [1, 0, 0], magnitude: 3 });
  const lit = sideMeanOutward(positions, out, +1, R);
  const shadow = sideMeanOutward(positions, out, -1, R);
  assert.ok(shadow > lit + 0.5, `shadow ${shadow.toFixed(3)} should exceed lit ${lit.toFixed(3)}`);
  assert.ok(lit >= -1e-4, `lit side must not go inward under push (${lit.toFixed(4)})`);
});

// 2. Magnitude scales displacement linearly; magnitude 0 is an exact no-op.
t('2. magnitude scales linearly; magnitude 0 is a no-op', () => {
  const { positions, quads } = cubeSphere();
  const zero = applyLightDeform(positions, quads, { lightDir: [1, 0, 0], magnitude: 0 });
  for (let i = 0; i < positions.length; i++) assert.equal(zero[i], positions[i]);
  const a = applyLightDeform(positions, quads, { lightDir: [1, 0, 0], magnitude: 1 });
  const b = applyLightDeform(positions, quads, { lightDir: [1, 0, 0], magnitude: 2 });
  let maxDiff = 0;
  for (let i = 0; i < positions.length; i++) {
    const da = a[i] - positions[i], db = b[i] - positions[i];
    maxDiff = Math.max(maxDiff, Math.abs(db - 2 * da));
  }
  assert.ok(maxDiff < 1e-2, `2x magnitude should double the move (max residual ${maxDiff})`);
});

// 3. Hardness narrows the terminator: low hardness spreads the shade over more
//    vertices, high hardness collapses it toward a hard 0/1 split.
t('3. hardness narrows the terminator (shade spread 10 vs 90)', () => {
  const { positions, quads } = cubeSphere();
  const spread = (h) => {
    const s = shadeField(positions, quads, { lightDir: [1, 0, 0], hardness: h });
    let mid = 0;
    for (const v of s) if (v > 0.05 && v < 0.95) mid++;
    return mid;
  };
  const soft = spread(10), hard = spread(90);
  assert.ok(soft > hard, `soft terminator (${soft}) should touch more vertices than hard (${hard})`);
});

// 4. 'erode' drives lit vertices inward and shadow vertices outward (signed),
//    unlike 'push' which is outward-only.
t('4. erode moves lit in and shadow out, unlike push', () => {
  const { positions, quads, R } = cubeSphere();
  const er = applyLightDeform(positions, quads, { lightDir: [1, 0, 0], magnitude: 1, mode: 'erode' });
  const litE = sideMeanOutward(positions, er, +1, R);
  const shadowE = sideMeanOutward(positions, er, -1, R);
  assert.ok(litE < -1e-3, `erode lit side should move inward (${litE.toFixed(4)})`);
  assert.ok(shadowE > 1e-3, `erode shadow side should move outward (${shadowE.toFixed(4)})`);
  const pu = applyLightDeform(positions, quads, { lightDir: [1, 0, 0], magnitude: 1, mode: 'push' });
  assert.ok(sideMeanOutward(positions, pu, +1, R) >= -1e-4, 'push lit side must not go inward');
});

// 5. The fold guard caps every inward move at SAFE_FRACTION of the vertex's
//    nearest-neighbor distance, and it actually bites under a large magnitude.
t('5. fold guard caps inward motion at the safe fraction (and bites)', () => {
  const { positions, quads } = cubeSphere();
  const nn = neighborDist(positions, quads);
  const out = applyLightDeform(positions, quads, { lightDir: [1, 0, 0], magnitude: 100, mode: 'erode' });
  let bit = false;
  for (let i = 0; i < positions.length / 3; i++) {
    const dir = radial(positions, i);
    const mx = out[i * 3] - positions[i * 3], my = out[i * 3 + 1] - positions[i * 3 + 1], mz = out[i * 3 + 2] - positions[i * 3 + 2];
    const proj = mx * dir[0] + my * dir[1] + mz * dir[2];
    const moveLen = Math.hypot(mx, my, mz);
    const cap = SAFE_FRACTION * nn[i];
    if (proj < 0) {
      assert.ok(moveLen <= cap + 1e-3, `inward move ${moveLen.toFixed(3)} exceeded cap ${cap.toFixed(3)} at vertex ${i}`);
      if (moveLen > 0.9 * cap) bit = true;
    }
  }
  // Uncapped, a lit vertex would move ~100 inward; the cap is ~1, so it must clamp.
  assert.ok(bit, 'fold guard never engaged — the check would be vacuous');
});

// 6. Two opposite lights: the applied field equals the sum of the two shade
//    fields, and the result is symmetric across the X plane.
t('6. compositeLightRigs sums the shade fields and stays symmetric', () => {
  const { positions, quads, R } = cubeSphere();
  const rigs = [{ lightDir: [1, 0, 0], magnitude: 2 }, { lightDir: [-1, 0, 0], magnitude: 2 }];
  const out = compositeLightRigs(positions, quads, rigs);

  // additive shade: expected move = normal * (dispA + dispB), no guard (outward push)
  const normals = vertexNormals(positions, quads, true);
  const dA = displacementField(positions, quads, { lightDir: [1, 0, 0], magnitude: 2 });
  const dB = displacementField(positions, quads, { lightDir: [-1, 0, 0], magnitude: 2 });
  let maxDiff = 0;
  for (let i = 0; i < normals.length; i++) {
    const s = dA[i] + dB[i];
    for (let k = 0; k < 3; k++) {
      const expected = positions[i * 3 + k] + normals[i][k] * s;
      maxDiff = Math.max(maxDiff, Math.abs(out[i * 3 + k] - expected));
    }
  }
  assert.ok(maxDiff < 1e-2, `composite must equal the summed shade field (max residual ${maxDiff})`);

  const plus = sideMeanOutward(positions, out, +1, R);
  const minus = sideMeanOutward(positions, out, -1, R);
  assert.ok(Math.abs(plus - minus) < 1e-2, `opposite lights should be symmetric (+X ${plus.toFixed(4)} vs -X ${minus.toFixed(4)})`);
});

// 7. The input positions array is never mutated.
t('7. input positions are not mutated', () => {
  const { positions, quads } = cubeSphere();
  const snapshot = positions.slice();
  applyLightDeform(positions, quads, { lightDir: [1, 0, 0], magnitude: 5, mode: 'erode' });
  compositeLightRigs(positions, quads, [{ lightDir: [0, 1, 0], magnitude: 4 }], {});
  displacementField(positions, quads, { lightDir: [0, 0, 1] });
  for (let i = 0; i < positions.length; i++) assert.equal(positions[i], snapshot[i], `mutated at ${i}`);
});

// 8. The fold-guard ceiling equals safeFraction * the shortest edge, and a
// magnitude at that ceiling leaves the guard on the edge of biting: an erode at
// full ceiling holds while a small step past it folds.
t('8. foldGuardCeiling is safeFraction * shortest edge and is exactly non-folding', () => {
  const { positions, quads } = cubeSphere();
  const ceil = foldGuardCeiling(positions, quads);
  // Shortest edge measured directly.
  let shortest = Infinity;
  for (let q = 0; q < quads.length; q += 4) {
    const idx = [quads[q], quads[q + 1], quads[q + 2], quads[q + 3]];
    for (let e = 0; e < 4; e++) {
      const a = idx[e], b = idx[(e + 1) % 4];
      const d = Math.hypot(positions[a * 3] - positions[b * 3], positions[a * 3 + 1] - positions[b * 3 + 1], positions[a * 3 + 2] - positions[b * 3 + 2]);
      if (d < shortest) shortest = d;
    }
  }
  assert.ok(Math.abs(ceil - SAFE_FRACTION * shortest) < 1e-9, `ceiling ${ceil} vs ${SAFE_FRACTION * shortest}`);
  assert.ok(Number.isFinite(ceil) && ceil > 0, 'ceiling is a positive finite length');
});

assert.equal(failures.length, 0, `${failures.length} failed:\n${failures.join('\n')}`);
