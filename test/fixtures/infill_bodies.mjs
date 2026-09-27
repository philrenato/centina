// Closed triangle meshes for the Infill kernel tests and the shared
// cross-app fixtures. { positions: [x,y,z,...], indices: [a,b,c,...] },
// wound outward. `toObj` writes one as Wavefront OBJ.

export function uvSphere(r = 50, nu = 48, nv = 24, c = [0, 0, 0]) {
  const positions = [], indices = [];
  for (let j = 0; j <= nv; j++) {
    const th = (j / nv) * Math.PI;
    for (let i = 0; i < nu; i++) {
      const ph = (i / nu) * 2 * Math.PI;
      positions.push(c[0] + r * Math.sin(th) * Math.cos(ph), c[1] + r * Math.sin(th) * Math.sin(ph), c[2] + r * Math.cos(th));
    }
  }
  const at = (i, j) => j * nu + (i % nu);
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    const a = at(i, j), b = at(i + 1, j), cc = at(i + 1, j + 1), d = at(i, j + 1);
    if (j > 0) indices.push(a, d, b);
    if (j < nv - 1) indices.push(b, d, cc);
  }
  return { positions, indices };
}

export function torus(R = 60, r = 25, nu = 64, nv = 32) {
  const positions = [], indices = [];
  for (let i = 0; i < nu; i++) {
    const u = (i / nu) * 2 * Math.PI;
    for (let j = 0; j < nv; j++) {
      const v = (j / nv) * 2 * Math.PI;
      positions.push((R + r * Math.cos(v)) * Math.cos(u), (R + r * Math.cos(v)) * Math.sin(u), r * Math.sin(v));
    }
  }
  const at = (i, j) => (i % nu) * nv + (j % nv);
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
    indices.push(a, b, c, a, c, d);
  }
  return { positions, indices };
}

// A slide sandal as one genus-1 surface: a closed loop in the YZ plane (a
// flat sole run, two corners of radius 20, a strap arch over the foot)
// swept with a rounded-rectangle section that is 280 x 30 under the foot
// and 110 x 10 in the strap. About 280 x 110 x 90.
export function slide(nu = 120, nv = 32) {
  const sole = 35, cr = 20, zs = 15, arch = 55;
  const segs = [
    { L: 2 * sole, at: (t) => [0, -sole + 2 * sole * t, zs] },
    { L: (Math.PI / 2) * cr, at: (t) => { const a = -Math.PI / 2 + t * Math.PI / 2; return [0, sole + cr * Math.cos(a), zs + cr + cr * Math.sin(a)]; } },
    { L: Math.PI * arch, at: (t) => { const a = t * Math.PI; return [0, (sole + cr) * Math.cos(a), zs + cr + arch * Math.sin(a)]; } },
    { L: (Math.PI / 2) * cr, at: (t) => { const a = Math.PI + t * Math.PI / 2; return [0, -sole + cr * Math.cos(a), zs + cr + cr * Math.sin(a)]; } },
  ];
  const total = segs.reduce((s, g) => s + g.L, 0);
  const loop = (s) => {
    let d = ((s % 1) + 1) % 1 * total;
    for (const g of segs) { if (d <= g.L) return g.at(d / g.L); d -= g.L; }
    return segs[0].at(0);
  };
  // Sole weight: 1 along the sole run, 0 in the strap, eased over the corners.
  const weight = (s) => {
    const d = ((s % 1) + 1) % 1 * total;
    const a = segs[0].L, b = a + segs[1].L, c = b + segs[2].L;
    const ease = (x) => x * x * (3 - 2 * x);
    if (d <= a) return 1;
    if (d <= b) return 1 - ease((d - a) / segs[1].L);
    if (d <= c) return 0;
    return ease((d - c) / segs[3].L);
  };
  const sp = (x, e) => Math.sign(x) * Math.pow(Math.abs(x), e);
  const positions = [], indices = [];
  for (let i = 0; i < nu; i++) {
    const s = i / nu;
    const p = loop(s), q = loop(s + 1e-4), m = loop(s - 1e-4);
    const t = [q[0] - m[0], q[1] - m[1], q[2] - m[2]];
    const tl = Math.hypot(t[1], t[2]);
    const n = [0, t[2] / tl, -t[1] / tl]; // outward from the loop's inside
    const w = weight(s);
    const ax = w * 140 + (1 - w) * 55, an = w * 15 + (1 - w) * 5;
    for (let j = 0; j < nv; j++) {
      const v = (j / nv) * 2 * Math.PI;
      const x = ax * sp(Math.cos(v), 0.5), o = an * sp(Math.sin(v), 0.5);
      positions.push(p[0] + x, p[1] + n[1] * o, p[2] + n[2] * o);
    }
  }
  const at = (i, j) => (i % nu) * nv + (j % nv);
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
    indices.push(a, b, c, a, c, d);
  }
  return orientOutward({ positions, indices });
}

// The thin-wall breaker: a C in the XZ plane extruded along Y. Two prongs
// `wall` thick, `gap` apart, joined at x < 0. With a Kelvin Size of 20
// (struts 7.07) a strut can have both ends inside, one in each prong, and
// cross the gap through the air.
export function thinWallC({ length = 120, depth = 80, wall = 12, gap = 5, back = 20 } = {}) {
  const H = 2 * wall + gap;
  const poly = [[0, 0], [length, 0], [length, wall], [back, wall], [back, wall + gap], [length, wall + gap], [length, H], [0, H]];
  return orientOutward(extrudePolygon(poly, depth));
}

// A simple polygon (x, z) extruded along y from 0 to depth; caps by ear
// clipping.
export function extrudePolygon(poly, depth) {
  const n = poly.length, positions = [], indices = [];
  for (const [x, z] of poly) positions.push(x, 0, z);
  for (const [x, z] of poly) positions.push(x, depth, z);
  const tris = earClip(poly);
  for (const [a, b, c] of tris) { indices.push(a, c, b); indices.push(n + a, n + b, n + c); }
  for (let i = 0; i < n; i++) { const j = (i + 1) % n; indices.push(i, j, n + j, i, n + j, n + i); }
  return { positions, indices };
}
function earClip(poly) {
  const area = poly.reduce((s, p, i) => { const q = poly[(i + 1) % poly.length]; return s + p[0] * q[1] - q[0] * p[1]; }, 0);
  let idx = poly.map((_, i) => i);
  if (area < 0) idx.reverse();
  const tris = [];
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const inTri = (p, a, b, c) => cross(a, b, p) >= 0 && cross(b, c, p) >= 0 && cross(c, a, p) >= 0;
  let guard = 0;
  while (idx.length > 3 && guard++ < 10000) {
    for (let k = 0; k < idx.length; k++) {
      const i0 = idx[(k + idx.length - 1) % idx.length], i1 = idx[k], i2 = idx[(k + 1) % idx.length];
      const a = poly[i0], b = poly[i1], c = poly[i2];
      if (cross(a, b, c) <= 0) continue;
      if (idx.some((j) => j !== i0 && j !== i1 && j !== i2 && inTri(poly[j], a, b, c))) continue;
      tris.push([i0, i1, i2]);
      idx.splice(k, 1);
      break;
    }
  }
  tris.push(idx);
  return tris;
}

export function orientOutward(mesh) {
  const P = mesh.positions, I = mesh.indices;
  let v6 = 0;
  for (let t = 0; t < I.length; t += 3) {
    const a = 3 * I[t], b = 3 * I[t + 1], c = 3 * I[t + 2];
    v6 += P[a] * (P[b + 1] * P[c + 2] - P[b + 2] * P[c + 1]) - P[a + 1] * (P[b] * P[c + 2] - P[b + 2] * P[c]) + P[a + 2] * (P[b] * P[c + 1] - P[b + 1] * P[c]);
  }
  if (v6 >= 0) return mesh;
  const out = [];
  for (let t = 0; t < I.length; t += 3) out.push(I[t], I[t + 2], I[t + 1]);
  return { positions: P, indices: out };
}

export function scaleMesh(mesh, s, about = [0, 0, 0]) {
  return { positions: mesh.positions.map((v, i) => about[i % 3] + (v - about[i % 3]) * s), indices: mesh.indices.slice() };
}

export function toObj(mesh, header = '') {
  const lines = header ? header.split('\n').map((l) => `# ${l}`) : [];
  for (let i = 0; i < mesh.positions.length; i += 3) lines.push(`v ${mesh.positions[i].toFixed(6)} ${mesh.positions[i + 1].toFixed(6)} ${mesh.positions[i + 2].toFixed(6)}`);
  for (let t = 0; t < mesh.indices.length; t += 3) lines.push(`f ${mesh.indices[t] + 1} ${mesh.indices[t + 1] + 1} ${mesh.indices[t + 2] + 1}`);
  return lines.join('\n') + '\n';
}
