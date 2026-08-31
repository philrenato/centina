// SURFACE WAVE — the
// surface modifier chain's 6th member, joining
// Rebuild -> Fair -> Point-Edits -> Cage -> Noise -> Wave. Same operand
// as Noise (the control net, interior points only, boundary rows/
// columns and every rational weight left exactly untouched, amplitude 0
// = exact identity), but a genuinely different displacement law: Noise
// is seeded pseudo-random texture; Wave is a smooth, fully deterministic
// PERIODIC ripple along one chosen grid axis, carrying an explicit
// PHASE parameter — the doc's own instruction was to cut Wave's stated
// hard dependency on the not-yet-built Animation Toolset ("the (future)
// Animator drives phase, same as it would drive any other numeric
// param — no special coupling needed"), so phase is just an ordinary
// param here, ready for that later wiring without needing it now.
//
// Reuses noise.mjs's own generic (non-noise-specific) helpers directly
// rather than re-deriving them a second time: grevilleFromKnots (a
// control point's own parameter of maximum influence, for normal-frame
// direction), surfaceNormalAtParam (the honest pole-degenerate-safe
// local normal), and refineSurface (knot-insertion density boost before
// displacement, shape-preserving by construction).

import { grevilleFromKnots, surfaceNormalAtParam, refineSurface, maxSafeDisplacementScale, boundaryFalloffGrid } from './noise.mjs';

export const WAVE_AXES = ['u', 'v', 'diagonal'];
export const WAVE_DIRECTIONS = ['normal', 'world-x', 'world-y', 'world-z'];

// Fill defaults + clamp, mirroring normalizeNoiseParams' own shape.
// No seed field at all — unlike Noise, Wave is a pure closed-form
// formula, deterministic by construction with nothing to seed.
export function normalizeWaveParams(params) {
  const p = params || {};
  return {
    axis: WAVE_AXES.includes(p.axis) ? p.axis : 'u',
    amplitude: Number.isFinite(p.amplitude) ? Math.max(0, p.amplitude) : 0,
    frequency: Number.isFinite(p.frequency) && p.frequency > 0 ? p.frequency : 1,
    phase: Number.isFinite(p.phase) ? p.phase : 0,
    direction: WAVE_DIRECTIONS.includes(p.direction) ? p.direction : 'normal',
    refine: Number.isFinite(p.refine) ? Math.max(0, Math.round(p.refine)) : 0,
    // 0 keeps the displacement at full strength to the boundary, which is what
    // this did before falloff existed — so the default is byte-identical.
    falloff: Number.isFinite(p.falloff) ? Math.max(0, Math.min(1, p.falloff)) : 0,
  };
}

// The per-control-point signed scalar (exactly [-1,1], a real sine —
// never clamped/hashed) for the whole grid. `axis` picks which grid
// direction's own fractional position (0..1 across the net) drives the
// wave; 'diagonal' averages both fractions, giving a wave that runs
// across both directions at once.
function waveScalarGrid(p, nu, nv) {
  const g = Array.from({ length: nu }, () => new Array(nv).fill(0));
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      const fu = nu > 1 ? i / (nu - 1) : 0;
      const fv = nv > 1 ? j / (nv - 1) : 0;
      const t = p.axis === 'u' ? fu : p.axis === 'v' ? fv : (fu + fv) / 2;
      g[i][j] = Math.sin(2 * Math.PI * p.frequency * t + p.phase);
    }
  }
  return g;
}

// waveControlNet(srf, params) — the public surface, mirroring
// noiseControlNet's own (srf, params) -> new srf shape exactly.
export function waveControlNet(srf, params) {
  const p = normalizeWaveParams(params);
  if (p.amplitude === 0) return srf; // EXACT identity, same amplitude=0 tween baseline as Noise/Fair
  let s = p.refine > 0 ? refineSurface(srf, p.refine) : srf;
  const nu = s.ctrlNet.length, nv = s.ctrlNet[0].length;
  if (nu < 3 || nv < 3) return srf; // no genuine interior control point to displace
  const grid = waveScalarGrid(p, nu, nv);
  const fade = boundaryFalloffGrid(nu, nv, p.falloff);
  const useNormal = p.direction === 'normal';
  const axisVec = p.direction === 'world-x' ? [1, 0, 0] : p.direction === 'world-y' ? [0, 1, 0] : p.direction === 'world-z' ? [0, 0, 1] : null;
  const gU = useNormal ? grevilleFromKnots(s.knotsU, s.degU, nu) : null;
  const gV = useNormal ? grevilleFromKnots(s.knotsV, s.degV, nv) : null;

  // PHASE 1 — the UNIT (amplitude=1) displacement field, cached so phase 3
  // never re-derives a pole's own normal a second time. Same shape as
  // noise.mjs's own noiseControlNet — see its header comment for the full
  // reasoning behind this two-phase split.
  const unitDisp = Array.from({ length: nu }, () => new Array(nv).fill(null).map(() => [0, 0, 0]));
  for (let i = 1; i < nu - 1; i++) {
    for (let j = 1; j < nv - 1; j++) {
      let dir = axisVec;
      if (useNormal) { dir = surfaceNormalAtParam(s, gU[i], gV[j]); if (!dir) continue; } // a true pole — no defined normal, displace nothing rather than a NaN
      const g = grid[i][j] * fade[i][j];
      unitDisp[i][j] = [dir[0] * g, dir[1] * g, dir[2] * g];
    }
  }

  // PHASE 2 — the SELF-INTERSECTION-SAFE CLAMP, reusing noise.mjs's own
  // shared primitive directly (the same math applies unchanged: Wave's own
  // amplitude is also floored non-negative by normalizeWaveParams above).
  const safeMax = maxSafeDisplacementScale(s.ctrlNet, unitDisp);
  const appliedAmplitude = Math.min(p.amplitude, safeMax);

  // PHASE 3 — apply the (possibly clamped) amplitude to the cached unit field.
  const net = s.ctrlNet.map((row) => row.map((cp) => [...cp]));
  for (let i = 1; i < nu - 1; i++) {
    for (let j = 1; j < nv - 1; j++) {
      const d = unitDisp[i][j];
      net[i][j][0] += d[0] * appliedAmplitude;
      net[i][j][1] += d[1] * appliedAmplitude;
      net[i][j][2] += d[2] * appliedAmplitude;
    }
  }
  return {
    ...s,
    ctrlNet: net,
    ampClamp: { requested: p.amplitude, applied: appliedAmplitude, safeMax, clamped: appliedAmplitude < p.amplitude },
  };
}
