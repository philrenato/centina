// SPLITS AS AN ORDERED FEATURE LIST ON THE SURFACE
// ================================================================
// A surface carries a LIST of split entries — `{ direction: 'u'|'v', frac }` —
// rather than being wrapped in a container per split. Evaluating the list cuts
// the surface into a grid of pieces.
//
// WHY A LIST AND NOT NESTED CONTAINERS. A container per split nests containers
// for a second split, and a container's pieces are not independently
// addressable surfaces with their own history — so "split it again over there"
// has nowhere to live. A list gives many entries, each independently editable,
// composing with whatever recipe made the surface in the first place (a Loft's
// curves keep driving it), with nothing nested.
//
// ⚠⚠ EVERY FRACTION IS MEASURED ON THE ORIGINAL SURFACE, AND THAT IS NOT A
// PREFERENCE. `splitSurface` does not cut a NURBS surface exactly — it RESAMPLES
// each half and refits it (networkCorrectionSurface over a sample grid; exact at
// the stations, an honest approximation between them). So splitting a piece that
// is itself already a refit compounds the approximation, and a fifth split would
// be a refit of a refit of a refit. Cutting the ORIGINAL at every fraction makes
// each piece exactly ONE refit deep, however many entries the list holds.
//
// It also gives the property the list is FOR: entries are independent. Dragging
// one split does not move any other, because none of them is defined relative to
// another. Two entries in the same direction give strips; two crossed give a
// grid. Order is presentation order, not evaluation order — the geometry is a
// set of cut lines, and saying so is more honest than implying a sequence that
// does not exist.
import { surfacePoint, surfaceClosure } from './surface.mjs';
import { networkCorrectionSurface } from './loft.mjs';

function linspace(a, b, n) {
  if (n === 1) return [a];
  const out = [];
  for (let i = 0; i < n; i++) out.push(a + (b - a) * (i / (n - 1)));
  return out;
}

function rescaleKnots(knots, oldMin, oldMax, newMin, newMax) {
  const span = oldMax - oldMin;
  return knots.map((k) => newMin + ((k - oldMin) / span) * (newMax - newMin));
}

// The cut PARAMETERS in one direction, from the fractions, sorted and with
// near-duplicates collapsed. Two entries at the same place would otherwise ask
// for a zero-width piece, which is not a piece.
function cutParamsFor(features, direction, domainMin, domainMax, eps) {
  const params = features
    .filter((f) => f.direction === direction)
    .map((f) => domainMin + Math.min(1, Math.max(0, f.frac)) * (domainMax - domainMin))
    .filter((t) => t > domainMin + eps && t < domainMax - eps)
    .sort((a, b) => a - b);
  const kept = [];
  for (const t of params) if (!kept.length || t - kept[kept.length - 1] > eps) kept.push(t);
  return kept;
}

// opts: { sampleCount, crossSampleCount, degU, degV }
// Returns { ok, pieces, cuts: { u, v }, stats } or { ok: false, reason }.
// Each piece is an ordinary `{ degU, degV, knotsU, knotsV, ctrlNet }` surface
// plus `uStations`/`vStations` — the real (u,v) grid it is EXACT at, exposed for
// the same reason loft() and splitSurface() expose theirs: so a caller never has
// to guess the internal sample grid to check exactness correctly.
export function applySplitFeatures(srf, features, opts = {}) {
  if (!srf || !srf.ctrlNet || !srf.knotsU || !srf.knotsV) {
    return { ok: false, reason: 'applySplitFeatures needs a surface' };
  }
  const list = Array.isArray(features) ? features.filter((f) => f && (f.direction === 'u' || f.direction === 'v')) : [];
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const uEps = (uMax - uMin) * 1e-6, vEps = (vMax - vMin) * 1e-6;

  // ⚠ A CLOSED DIRECTION REFUSES BY NAME, exactly as `splitSurface` does and for
  // its reason: cutting a closed loop at ONE parameter unrolls it into a single
  // open piece, not two. Offering a slider that cannot do what it says would be
  // worse than not offering one.
  const { closedU, closedV } = surfaceClosure(srf);
  for (const [dir, closed] of [['u', closedU], ['v', closedV]]) {
    if (closed && list.some((f) => f.direction === dir)) {
      return {
        ok: false,
        reason: `the ${dir}-direction is CLOSED (a seam, not a free edge) — cutting it at one parameter would unroll it into one piece, not two. `
          + 'Split the other direction, or cut this one twice (not supported yet).',
      };
    }
  }

  const uCuts = cutParamsFor(list, 'u', uMin, uMax, uEps);
  const vCuts = cutParamsFor(list, 'v', vMin, vMax, vEps);
  const dropped = list.length - (uCuts.length + vCuts.length);
  if (!uCuts.length && !vCuts.length) {
    // Not a failure: a surface with no usable cuts IS the surface. Saying so
    // lets a caller treat "no splits yet" and "splits that all collapsed" the
    // same way without a special case.
    return { ok: true, pieces: [{ ...srf, uStations: null, vStations: null }], cuts: { u: [], v: [] }, stats: { pieces: 1, dropped, entries: list.length } };
  }

  const sampleCount = opts.sampleCount ?? 12;
  const crossSampleCount = opts.crossSampleCount ?? 16;
  const uEdges = [uMin, ...uCuts, uMax];
  const vEdges = [vMin, ...vCuts, vMax];

  // ONE refit per piece, always sampling the ORIGINAL — see the header. The
  // sample counts are per PIECE, so a finely-cut surface keeps the same
  // per-piece fidelity rather than sharing one budget across the grid.
  function buildPatch(u0, u1, v0, v1) {
    const uParams = linspace(u0, u1, sampleCount);
    const vParams = linspace(v0, v1, crossSampleCount);
    const uNorm = uParams.map((u) => (u - u0) / (u1 - u0));
    const vNorm = vParams.map((v) => (v - v0) / (v1 - v0));
    const degU = Math.min(opts.degU ?? 3, uParams.length - 1);
    const degV = Math.min(opts.degV ?? 3, vParams.length - 1);
    const grid = uParams.map((u) => vParams.map((v) => surfacePoint(srf, u, v)));
    const built = networkCorrectionSurface(grid, uNorm, vNorm, degU, degV);
    return {
      degU: built.degU, knotsU: rescaleKnots(built.knotsU, 0, 1, u0, u1),
      degV: built.degV, knotsV: rescaleKnots(built.knotsV, 0, 1, v0, v1),
      ctrlNet: built.ctrlNet, uStations: uParams, vStations: vParams,
    };
  }

  const pieces = [];
  try {
    for (let i = 0; i < uEdges.length - 1; i++) {
      for (let j = 0; j < vEdges.length - 1; j++) {
        pieces.push(buildPatch(uEdges[i], uEdges[i + 1], vEdges[j], vEdges[j + 1]));
      }
    }
  } catch (err) {
    return { ok: false, reason: `a split piece could not be built: ${err && err.message ? err.message : err}` };
  }

  return {
    ok: true,
    pieces,
    cuts: { u: uCuts, v: vCuts },
    stats: { pieces: pieces.length, entries: list.length, dropped, uCuts: uCuts.length, vCuts: vCuts.length },
  };
}
