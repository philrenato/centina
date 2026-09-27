import { createXPBDCage, stepXPBDCage } from './xpbd.mjs';

const finite3 = (v) => v && v.length === 3 && v.every(Number.isFinite);
export function dynamicBondFrame(point, su, sv) {
  if (![point, su, sv].every(finite3)) throw new Error('Invalid host frame');
  const a = Math.hypot(...su),
    b = Math.hypot(...sv);
  if (!(a > 1e-12 && b > 1e-12)) throw new Error('The attachment is at a singular surface point');
  const x = su.map((v) => v / a),
    v = sv.map((v) => v / b);
  let z = [x[1] * v[2] - x[2] * v[1], x[2] * v[0] - x[0] * v[2], x[0] * v[1] - x[1] * v[0]];
  const l = Math.hypot(...z);
  if (l < 1e-8) throw new Error('The attachment tangents are parallel');
  z = z.map((v) => v / l);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return { point: point.slice(), axes: [x, y, z] };
}
export function dynamicBondLocal(point, frame) {
  return frame.axes.map((a) => a.reduce((v, c, k) => v + c * (point[k] - frame.point[k]), 0));
}
export function dynamicBondWorld(point, frame) {
  return frame.point.map((v, k) => v + point.reduce((sum, c, j) => sum + c * frame.axes[j][k], 0));
}
export function dynamicBondTopology(cage) {
  return JSON.stringify([cage.vertices.length, cage.faces]);
}

// One-sided contact with the host: a free vertex stays `margin` off the host
// on the side the cage started. `closest(point, maxDistance)` answers
// {point, normal} or null; the side is fixed once from the cage centroid (the
// pin offset breaks a tie). Each vertex remembers how far it can move before
// the host could possibly be reached again, so a vertex hanging in the air
// asks nothing of the tree while it falls.
export function createHostContact({ closest, cage, pinOffset = null, margin, reach }) {
  const n = cage.vertices.length;
  const centroid = cage.vertices.reduce((a, p) => a.map((v, k) => v + p[k] / n), [0, 0, 0]);
  const at = closest(centroid, Infinity);
  if (!at) return null;
  let side =
    (centroid[0] - at.point[0]) * at.normal[0] +
    (centroid[1] - at.point[1]) * at.normal[1] +
    (centroid[2] - at.point[2]) * at.normal[2];
  if (Math.abs(side) < 1e-9 && pinOffset)
    side = pinOffset[0] * at.normal[0] + pinOffset[1] * at.normal[1] + pinOffset[2] * at.normal[2];
  side = side < 0 ? -1 : 1;
  const last = new Float64Array(n * 3).fill(NaN),
    slack = new Float64Array(n);
  return {
    side,
    margin,
    contact(x, i) {
      const j = 3 * i,
        dx = x[j] - last[j],
        dy = x[j + 1] - last[j + 1],
        dz = x[j + 2] - last[j + 2];
      if (dx * dx + dy * dy + dz * dz < slack[i] * slack[i]) return;
      const hit = closest([x[j], x[j + 1], x[j + 2]], reach);
      last[j] = x[j];
      last[j + 1] = x[j + 1];
      last[j + 2] = x[j + 2];
      if (!hit) {
        slack[i] = reach - margin;
        return;
      }
      const s =
        side *
        ((x[j] - hit.point[0]) * hit.normal[0] +
          (x[j + 1] - hit.point[1]) * hit.normal[1] +
          (x[j + 2] - hit.point[2]) * hit.normal[2]);
      if (s < margin) {
        const push = side * (margin - s);
        x[j] += hit.normal[0] * push;
        x[j + 1] += hit.normal[1] * push;
        x[j + 2] += hit.normal[2] * push;
        last[j] = x[j];
        last[j + 1] = x[j + 1];
        last[j + 2] = x[j + 2];
        slack[i] = 0;
      } else slack[i] = s - margin;
    },
    // Diagnostic: the least clearance any free vertex has from the host.
    clearance(x, pinned = []) {
      const skip = new Set(pinned);
      let min = Infinity;
      for (let i = 0; i < n; i++) {
        if (skip.has(i)) continue;
        const j = 3 * i;
        const hit = closest([x[j], x[j + 1], x[j + 2]], Infinity);
        if (!hit) continue;
        const s =
          side *
          ((x[j] - hit.point[0]) * hit.normal[0] +
            (x[j + 1] - hit.point[1]) * hit.normal[1] +
            (x[j + 2] - hit.point[2]) * hit.normal[2]);
        if (s < min) min = s;
      }
      return min;
    },
  };
}

export function createDynamicBondSimulation({ cage, frame, pinned, mass = 1, step = 1 / 240 }) {
  if (!(step > 0 && step <= 1 / 30) || !Number.isFinite(step)) throw new Error('Invalid simulation time step');
  const local = cage.vertices.map((v) => dynamicBondLocal(v, frame));
  const solver = createXPBDCage({ positions: cage.vertices.flat(), faces: cage.faces, pinned, mass });
  return {
    solver,
    local,
    targets: solver.x.slice(),
    lastTargets: solver.x.slice(),
    nextTargets: solver.x.slice(),
    accumulator: 0,
    step,
  };
}
// Fixed steps, bounded catch-up. Pin positions are interpolated to substep
// times instead of jumping to the end-of-frame target in every substep.
export function advanceDynamicBond(sim, elapsed, frame, options = {}) {
  if (!Number.isFinite(elapsed) || elapsed < 0) throw new Error('Invalid elapsed time');
  if (elapsed > 0.25) {
    sim.accumulator = 0;
    return 0;
  } // background pause
  const dt = Math.min(elapsed, 0.05);
  for (const i of sim.solver.pinned) {
    const p = dynamicBondWorld(sim.local[i], frame);
    sim.nextTargets.set(p, 3 * i);
  }
  const prior = sim.accumulator;
  sim.accumulator += dt;
  let count = 0;
  while (sim.accumulator + 1e-12 >= sim.step && count < 12) {
    const fraction = dt > 0 ? Math.min(1, ((count + 1) * sim.step - prior) / dt) : 1;
    for (const i of sim.solver.pinned)
      for (let k = 0; k < 3; k++) {
        const j = 3 * i + k;
        sim.targets[j] = sim.lastTargets[j] + fraction * (sim.nextTargets[j] - sim.lastTargets[j]);
      }
    stepXPBDCage(sim.solver, sim.step, { ...options, targets: sim.targets });
    sim.accumulator -= sim.step;
    count++;
  }
  sim.lastTargets.set(sim.nextTargets);
  return count;
}

// A recorded settle runs until the cage is still, or this long in simulated
// time. A pin of three or more vertices (the seat's face ring) holds a
// stationary anchor and steps at 120 Hz; one or two pins swing farther between
// steps and keep the 240 Hz default. The page and the settle worker both run
// this, so a batch settles exactly as a single child does.
export const DYNAMIC_BOND_SETTLE_SECONDS = 3;
export const DYNAMIC_BOND_FRAME = 1 / 60;
export const DYNAMIC_BOND_REST_SPEED = 0.5; // mm/s, the fastest vertex of a cage that counts as still
export const dynamicBondSettleStep = (pins) => (pins.length >= 3 ? 1 / 120 : undefined);
export function dynamicBondOptions(params, contact) {
  return { gravity: [0, 0, -1000 * params.gravity], compliance: params.softness ** 2 * .001, damping: params.damping,
    preserveVolume: params.volume, contact: contact ? contact.contact : null };
}
export function dynamicBondAtRest(sim) {
  const v = sim.solver.velocity; let m = 0;
  for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]));
  return m < DYNAMIC_BOND_REST_SPEED;
}
// frames, when given, receives the solver positions after every frame.
export function settleDynamicBond(sim, frame, options, frames = null) {
  for (let t = 0; t < DYNAMIC_BOND_SETTLE_SECONDS; t += DYNAMIC_BOND_FRAME) {
    advanceDynamicBond(sim, DYNAMIC_BOND_FRAME, frame, options);
    if (frames) frames.push(sim.solver.x.slice());
    if (t > 0.25 && dynamicBondAtRest(sim)) break;
  }
  return sim;
}
