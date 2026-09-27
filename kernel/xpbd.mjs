// Coarse-cage XPBD. Coordinates are model millimeters, time seconds, mass kg.
export function createXPBDCage({ positions, faces, pinned = [], mass = 1 }) {
  if (!positions || positions.length % 3 || positions.length < 6 || !Array.from(positions).every(Number.isFinite))
    throw new Error('Invalid cage positions');
  if (!(mass > 0) || !Number.isFinite(mass)) throw new Error('Mass must be positive');
  const n = positions.length / 3;
  const pins = new Set(pinned);
  if ([...pins].some((i) => !Number.isInteger(i) || i < 0 || i >= n)) throw new Error('Invalid pinned vertex');
  const edges = new Map(),
    triangles = [],
    directed = new Map();
  const add = (a, b) => {
    if (a !== b) edges.set(a < b ? `${a}:${b}` : `${b}:${a}`, [a, b]);
  };
  for (const f of faces || []) {
    if (
      !Array.isArray(f) ||
      f.length < 3 ||
      new Set(f).size !== f.length ||
      f.some((i) => !Number.isInteger(i) || i < 0 || i >= n)
    )
      throw new Error('Invalid cage face');
    for (let j = 0; j < f.length; j++) {
      const a = f[j],
        b = f[(j + 1) % f.length];
      add(a, b);
      const key = `${a}:${b}`;
      directed.set(key, (directed.get(key) || 0) + 1);
    }
    // Face diagonals prevent a quad from shearing at zero edge strain. These
    // are distance supports, not a claim of exact shell bending stiffness.
    for (let j = 0; j < f.length; j++) for (let k = j + 2; k < f.length; k++) add(f[j], f[k]);
    for (let j = 1; j < f.length - 1; j++) triangles.push(f[0], f[j], f[j + 1]);
  }
  if (!edges.size) throw new Error('The cage has no edges');
  // Flat typed links keep the solver's hottest loop allocation-free: a pair
  // array destructures one small array per edge, iteration and substep,
  // millions of times in a maximum-size settle.
  const linkPairs = [...edges.values()],
    links = Uint32Array.from(linkPairs.flat()),
    x = Float64Array.from(positions),
    rest = Float64Array.from(linkPairs, ([a, b]) =>
      Math.sqrt((x[3 * a] - x[3 * b]) ** 2 + (x[3 * a + 1] - x[3 * b + 1]) ** 2 + (x[3 * a + 2] - x[3 * b + 2]) ** 2)
    );
  const invMass = Float64Array.from({ length: n }, (_, i) => (pins.has(i) ? 0 : n / mass));
  const state = {
    x,
    previous: x.slice(),
    velocity: new Float64Array(x.length),
    backupVelocity: new Float64Array(x.length),
    invMass,
    links,
    rest,
    lambdas: new Float64Array(rest.length),
    triangles: Uint32Array.from(triangles),
    gradient: new Float64Array(x.length),
    pinned: Uint32Array.from(pins),
    time: 0,
    volume: 0,
    volumeLambda: 0,
  };
  state.closed = [...directed].every(([key, count]) => {
    const [a, b] = key.split(':');
    return count === 1 && directed.get(`${b}:${a}`) === 1;
  });
  state.volume = xpbdSignedVolume(state.x, state.triangles);
  if (Math.abs(state.volume) < 1e-12) state.closed = false;
  if (rest.some((v) => !Number.isFinite(v))) throw new Error('Cage edges exceed the numeric range');
  state.lengthScale = rest.reduce((a, b) => Math.max(a, b), 0);
  return state;
}

export function xpbdSignedVolume(x, triangles) {
  // A moving origin near the cage avoids catastrophic cancellation far from 0.
  const ox = x[0],
    oy = x[1],
    oz = x[2];
  let v = 0;
  for (let t = 0; t < triangles.length; t += 3) {
    const a = triangles[t] * 3,
      b = triangles[t + 1] * 3,
      c = triangles[t + 2] * 3;
    const ax = x[a] - ox,
      ay = x[a + 1] - oy,
      az = x[a + 2] - oz;
    const bx = x[b] - ox,
      by = x[b + 1] - oy,
      bz = x[b + 2] - oz;
    const cx = x[c] - ox,
      cy = x[c + 1] - oy,
      cz = x[c + 2] - oz;
    v += (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)) / 6;
  }
  return v;
}

function projectVolume(s, alpha) {
  const x = s.x,
    g = s.gradient;
  g.fill(0);
  const ox = x[0],
    oy = x[1],
    oz = x[2];
  for (let t = 0; t < s.triangles.length; t += 3) {
    const a = s.triangles[t] * 3,
      b = s.triangles[t + 1] * 3,
      c = s.triangles[t + 2] * 3;
    const ax = x[a] - ox,
      ay = x[a + 1] - oy,
      az = x[a + 2] - oz,
      bx = x[b] - ox,
      by = x[b + 1] - oy,
      bz = x[b + 2] - oz,
      cx = x[c] - ox,
      cy = x[c + 1] - oy,
      cz = x[c + 2] - oz;
    g[a] += (by * cz - bz * cy) / 6;
    g[a + 1] += (bz * cx - bx * cz) / 6;
    g[a + 2] += (bx * cy - by * cx) / 6;
    g[b] += (cy * az - cz * ay) / 6;
    g[b + 1] += (cz * ax - cx * az) / 6;
    g[b + 2] += (cx * ay - cy * ax) / 6;
    g[c] += (ay * bz - az * by) / 6;
    g[c + 1] += (az * bx - ax * bz) / 6;
    g[c + 2] += (ax * by - ay * bx) / 6;
  }
  let denom = alpha;
  for (let i = 0; i < s.invMass.length; i++) denom += s.invMass[i] * (g[3 * i] ** 2 + g[3 * i + 1] ** 2 + g[3 * i + 2] ** 2);
  if (!(denom > 1e-24)) return;
  const dl = (-(xpbdSignedVolume(x, s.triangles) - s.volume) - alpha * s.volumeLambda) / denom;
  s.volumeLambda += dl;
  for (let i = 0; i < x.length; i++) x[i] += s.invMass[(i / 3) | 0] * g[i] * dl;
}

// Atomic step: invalid output restores positions and velocities. Targets are
// flat world-space positions; only entries for pinned indices are read.
// `contact(x, i)` may move free vertex i in place after the constraint
// iterations; the velocity is then read from the moved position, so a contact
// is an impulse without any bookkeeping of its own.
export function stepXPBDCage(
  s,
  h,
  {
    targets = s.x,
    gravity = [0, 0, 0],
    compliance = 0.0001,
    damping = 2,
    preserveVolume = false,
    iterations = 6,
    contact = null,
  } = {}
) {
  if (!(h > 0 && h <= 1 / 30) || !Number.isFinite(h)) throw new Error('Invalid simulation time step');
  if (
    !(compliance >= 0) ||
    !Number.isFinite(compliance) ||
    !(damping >= 0) ||
    !Number.isFinite(damping) ||
    !Number.isInteger(iterations) ||
    iterations < 1 ||
    iterations > 32
  )
    throw new Error('Invalid solver parameters');
  if (!gravity || gravity.length !== 3 || !gravity.every(Number.isFinite)) throw new Error('Invalid gravity');
  if (!targets || targets.length !== s.x.length) throw new Error('Invalid pin targets');
  for (const i of s.pinned)
    for (let k = 0; k < 3; k++) if (!Number.isFinite(targets[3 * i + k])) throw new Error('Invalid pin target');
  const x = s.x,
    v = s.velocity,
    w = s.invMass;
  // Copy targets before prediction in case the caller passed s.x itself.
  const pinValues = s.pinValues || (s.pinValues = new Float64Array(s.pinned.length * 3));
  for (let p = 0; p < s.pinned.length; p++) for (let k = 0; k < 3; k++) pinValues[3 * p + k] = targets[3 * s.pinned[p] + k];
  s.previous.set(x);
  s.backupVelocity.set(v);
  s.lambdas.fill(0);
  s.volumeLambda = 0;
  const decay = Math.exp(-damping * h),
    alpha = compliance / (h * h);
  for (let i = 0; i < w.length; i++)
    if (w[i] > 0)
      for (let k = 0; k < 3; k++) {
        const j = 3 * i + k;
        v[j] = v[j] * decay + gravity[k] * h;
        x[j] += h * v[j];
      }
  for (let p = 0; p < s.pinned.length; p++) for (let k = 0; k < 3; k++) x[3 * s.pinned[p] + k] = pinValues[3 * p + k];
  for (let it = 0; it < iterations; it++) {
    for (let e = 0, pair = 0; e < s.links.length; e += 2, pair++) {
      const a = s.links[e], b = s.links[e + 1],
        wa = w[a],
        wb = w[b];
      if (wa + wb === 0) continue;
      let dx = x[3 * a] - x[3 * b],
        dy = x[3 * a + 1] - x[3 * b + 1],
        dz = x[3 * a + 2] - x[3 * b + 2];
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len < 1e-12) continue;
      const dl = (-(len - s.rest[pair]) - alpha * s.lambdas[pair]) / (wa + wb + alpha);
      s.lambdas[pair] += dl;
      dx *= dl / len;
      dy *= dl / len;
      dz *= dl / len;
      x[3 * a] += wa * dx;
      x[3 * a + 1] += wa * dy;
      x[3 * a + 2] += wa * dz;
      x[3 * b] -= wb * dx;
      x[3 * b + 1] -= wb * dy;
      x[3 * b + 2] -= wb * dz;
    }
    if (preserveVolume && s.closed) projectVolume(s, alpha * s.lengthScale ** 4);
  }
  if (contact) for (let i = 0; i < w.length; i++) if (w[i] > 0) contact(x, i);
  let finite = true;
  for (let j = 0; j < x.length; j++) {
    v[j] = (x[j] - s.previous[j]) / h;
    if (!Number.isFinite(x[j]) || !Number.isFinite(v[j])) finite = false;
  }
  // Reject a global orientation flip; this is not a self-collision detector.
  if (preserveVolume && s.closed && xpbdSignedVolume(x, s.triangles) * s.volume <= 0) finite = false;
  if (!finite) {
    x.set(s.previous);
    v.set(s.backupVelocity);
    throw new Error('Simulation could not produce a finite, consistently oriented cage');
  }
  s.time += h;
  return s;
}
