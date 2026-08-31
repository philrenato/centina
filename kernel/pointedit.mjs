// SURFACE POINT EDITS — the modifier chain's
// third stage. Each edit is anchored by a control point's RELATIVE
// (rowFrac, colFrac) position in the net at the moment it was made
// (rowFrac = i/(nu-1), colFrac = j/(nv-1)) rather than a raw (u,v) surface
// parameter — this kernel's control nets are not interpolatory in (u,v)
// (a Greville-abscissa grid, not a parametric one), so "the control point
// at 40% along each direction" is the honest, already-meaningful anchor
// this app's own CV-drag UI already indexes by (surfaceHandleRowCol),
// not a fabricated new coordinate system.
//
// Each entry is either:
//   { rowFrac, colFrac, delta:[dx,dy,dz] }                 — a WORLD-frame
//     edit: `delta` is a fixed world vector added to the control point.
//     This is the original CV-drag shape (a human visually repositioning a
//     control point); `frame` absent === 'world', so every already-shipped
//     entry keeps its exact meaning with zero migration.
//   { rowFrac, colFrac, frame:'normal', amount:<scalar> }  — a NORMAL-frame
//     edit: `amount` is a scalar displacement along the surface's OWN local
//     normal AT that (rowFrac,colFrac), recomputed FRESH from the current
//     (post-Rebuild/Fair) surface on every re-apply (surfacePointAndPartials
//     -> normalize(su × sv)). This is the sculpt-brush shape: a stroke rides
//     the surface, so an upstream edit that reorients the parent (a changed
//     Revolve axis) makes the stored bump follow the NEW normal, not the old
//     world direction a fixed vector would keep pointing in.

import { findSpan, basisFuns } from './basis.mjs';
import { surfacePointAndPartials } from './surface.mjs';

const EDIT_EPS = 1e-9; // below this magnitude an accumulated resample cell is treated as untouched (no edit emitted)

// The Greville abscissa of each control point in ONE direction — the average
// of its p consecutive interior knots, the parameter where that control
// point has the most influence. Matches curve.mjs's own grevilleAbscissae
// exactly, split out per-direction here so a surface can ask for its U and V
// abscissae independently (no mixed-basis evaluation is ever needed).
function grevilleFromKnots(knots, p, count) {
  const g = [];
  for (let i = 0; i < count; i++) {
    if (p <= 0) { g.push(knots[i] ?? 0); continue; }
    let s = 0;
    for (let k = i + 1; k <= i + p; k++) s += knots[k];
    g.push(s / p);
  }
  return g;
}

// The surface's local unit normal at parameter (u,v): normalize(Su × Sv).
// Falls back to a slightly interior parameter once (a Greville abscissa can
// land exactly on a degenerate POLE, where Su or Sv collapses and the cross
// product is ~zero); if still degenerate there is genuinely no defined
// normal (a true pole), so it returns null and the caller applies zero
// displacement rather than a NaN — the same honest-degrade posture every
// other stage in this chain already takes.
function surfaceNormalAtParam(srf, u, v) {
  const uMin = srf.knotsU[srf.degU], uMax = srf.knotsU[srf.knotsU.length - 1 - srf.degU];
  const vMin = srf.knotsV[srf.degV], vMax = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  const tries = [
    [u, v],
    // nudge a hair toward the domain centre, so a Greville abscissa sitting
    // exactly on a pole/edge singularity gets a real normal from just inside
    [u + (u < (uMin + uMax) / 2 ? 1 : -1) * (uMax - uMin) * 1e-4, v + (v < (vMin + vMax) / 2 ? 1 : -1) * (vMax - vMin) * 1e-4],
  ];
  for (const [uu, vv] of tries) {
    const cu = Math.max(uMin, Math.min(uMax, uu)), cv = Math.max(vMin, Math.min(vMax, vv));
    const { su, sv } = surfacePointAndPartials(srf, cu, cv);
    const nx = su[1] * sv[2] - su[2] * sv[1];
    const ny = su[2] * sv[0] - su[0] * sv[2];
    const nz = su[0] * sv[1] - su[1] * sv[0];
    const len = Math.hypot(nx, ny, nz);
    if (len > 1e-9) return [nx / len, ny / len, nz / len];
  }
  return null;
}

// Re-apply a stored edit list ON TOP of a freshly rebuilt natural surface.
// A WORLD-frame edit adds its fixed `delta` to the rounded control point.
// A NORMAL-frame edit adds `amount` along the surface's CURRENT local normal
// at that control point's own Greville (u,v) — computed fresh from `srf`
// (the natural post-Rebuild/Fair surface, never the partially-edited net),
// so the whole stored stroke is consistently anchored to one coherent
// surface. Re-applying against a net with a DIFFERENT point count rounds each
// fraction to the nearest surviving row/column — an honest nearest-match.
// (resamplePointEditsToNet below is the real answer to a density change; this
// function is the final splat once the edits already sit on the target net.)
//
// BLEND (0..1, default 0) — a genuine linear interpolation between the fully
// EDITED shape (blend=0, every delta/amount applied at full strength) and the
// pure SOURCE/natural shape `srf` already IS (blend=1, zero displacement),
// matching this app's own already-shipped Instance-blend convention (0=Edited,
// 1=Source) applied to control-point edits instead of a whole-object point
// snapshot. Scaling each edit's own delta/amount by `(1-blend)` is exactly
// the same lerp as `edited.lerp(natural, blend)` per touched control point,
// since `natural + delta*(1-blend) = (natural+delta)*(1-blend) + natural*blend`
// — i.e. it is a real, provable geometric lerp, not merely a label, and it
// stays a true no-op at the default blend=0 (scale factor exactly 1, byte-
// identical to every pre-existing call site that never passes a third
// argument at all).
export function applyPointEdits(srf, edits, blend = 0) {
  if (!edits || !edits.length) return srf;
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  if (nu < 1 || nv < 1) return srf;
  const scale = 1 - Math.max(0, Math.min(1, blend));
  if (scale === 0) return srf; // fully Source — zero displacement, honestly a no-op rather than a wasted clone
  const net = srf.ctrlNet.map((row) => row.map((cp) => [...cp]));
  const hasNormal = edits.some((e) => e.frame === 'normal');
  const gU = hasNormal ? grevilleFromKnots(srf.knotsU, srf.degU, nu) : null;
  const gV = hasNormal ? grevilleFromKnots(srf.knotsV, srf.degV, nv) : null;
  // A CLOSED SURFACE'S SEAM IS NOT DEFENDED HERE. The control points that
  // must move together are named by surfaceSeamBandGroup and expanded into
  // the DRAG'S OWN TARGET LIST, the same way a pole's collapsed row and a
  // PolyCurve's shared joint already are, so every one of them arrives here
  // as its own edit. Tying them a second time in this function would apply
  // each band member's delta to the whole band and move the seam by several
  // times what the drag asked for.
  for (const e of edits) {
    const i = Math.max(0, Math.min(nu - 1, Math.round(e.rowFrac * (nu - 1))));
    const j = Math.max(0, Math.min(nv - 1, Math.round(e.colFrac * (nv - 1))));
    if (e.frame === 'normal') {
      const n = surfaceNormalAtParam(srf, gU[i], gV[j]);
      if (!n) continue; // true pole — no defined normal, apply nothing rather than a NaN
      const a = e.amount * scale;
      net[i][j][0] += n[0] * a;
      net[i][j][1] += n[1] * a;
      net[i][j][2] += n[2] * a;
    } else {
      net[i][j][0] += e.delta[0] * scale;
      net[i][j][1] += e.delta[1] * scale;
      net[i][j][2] += e.delta[2] * scale;
    }
  }
  return { ...srf, ctrlNet: net };
}

// A clamped, uniform DEGREE-1 knot vector over `count` control points on the
// fractional domain [0,1] — control point i then sits at Greville parameter
// i/(count-1), exactly the (rowFrac, colFrac) an edit is anchored by.
function clampedUniformKnots(count) {
  const p = 1;
  const U = [];
  for (let k = 0; k < count + p + 1; k++) {
    if (k <= p) U.push(0);
    else if (k >= count) U.push(1);
    else U.push((k - p) / (count - p));
  }
  return U;
}

// The (control-index, weight) distribution of a fractional coordinate
// `frac` in [0,1] over `count` control points, via the DEGREE-1 basisFuns.
// Degree 1 is the deliberate, load-bearing choice — it is the unique
// low-degree B-spline basis that is simultaneously:
//   (a) INTERPOLATING at every grid node (N_i(node_i)=1, all others 0) — so
//       a value landing exactly on a control-point position keeps its full
//       magnitude there, which is what makes a resample to an identical net
//       shape reproduce an edit EXACTLY (evaluating a degree-1 B-spline at
//       its own control-point parameters returns the control points);
//   (b) a PARTITION OF UNITY (Σ_k N_k = 1) — so an interpolated value is a
//       true weighted average of its neighbours, never amplified or lost;
//   (c) NON-NEGATIVE with NO overshoot — so a sculpt bump reconstructed on a
//       finer net never grows the spurious negative side-lobes a cubic-basis
//       reconstruction (which also fails (a) at interior nodes and rings)
//       would. It is genuinely basisFuns weighting, at the degree the
//       resampling math requires.
function basisDistribution1D(count, frac) {
  if (count <= 1) return [{ index: 0, weight: 1 }];
  const p = 1;
  const U = clampedUniformKnots(count);
  const s = Math.max(0, Math.min(1, frac));
  const span = findSpan(count - 1, p, s, U);
  const N = basisFuns(span, s, p, U);
  const out = [];
  for (let a = 0; a <= p; a++) {
    const w = N[a];
    if (w !== 0) out.push({ index: span - p + a, weight: w });
  }
  return out;
}

// Build a dense source grid (oldNu x oldNv) of a per-cell numeric payload
// (world edits -> a [dx,dy,dz] vector; normal edits -> a [scalar]) from the
// stored edits, each landing on its own exact grid node.
function buildSourceGrid(entries, oldNu, oldNv, dim, pick) {
  const grid = Array.from({ length: oldNu }, () => Array.from({ length: oldNv }, () => new Array(dim).fill(0)));
  for (const e of entries) {
    const i = Math.max(0, Math.min(oldNu - 1, Math.round(e.rowFrac * (oldNu - 1))));
    const j = Math.max(0, Math.min(oldNv - 1, Math.round(e.colFrac * (oldNv - 1))));
    const val = pick(e);
    for (let c = 0; c < dim; c++) grid[i][j][c] += val[c];
  }
  return grid;
}

// RESAMPLE the whole delta FIELD onto a new control-net density — the real
// answer to a Rebuild that changes the surface's own point count. NOT the
// old nearest-round-and-stack (which lumps a coarsening Rebuild — many
// entries collapsing additively onto one surviving control point — and gaps
// a refining one — most new control points left with nothing to splat into).
//
// Treats the stored sparse edits as samples of a displacement FIELD over the
// surface's own fractional-index (rowFrac, colFrac) domain and RE-PROJECTS
// that field onto the new net: it reconstructs the field as a DEGREE-1
// tensor B-spline whose control values ARE the old net's per-node displaced
// amounts (buildSourceGrid), then EVALUATES that field (via basisFuns, the
// same weighting every other resampling operation in this kernel uses) at
// each new control point's own (rowFrac, colFrac) position. Because degree-1
// evaluation is interpolating + partition-of-unity + non-negative:
//   - a coarsening Rebuild SUB-samples the smooth field — each new control
//     point gets ONE bounded interpolated value, never a stack of collided
//     deltas, so it NEVER lumps;
//   - a refining Rebuild FILLS every intermediate new control point with the
//     interpolated field value, so it NEVER gaps;
//   - resampling to the SAME net shape is the exact identity (evaluating the
//     field at its own control-point parameters returns them);
//   - the AREA-WEIGHTED field integral (Σ value · Δu · Δv, the honest
//     "displaced volume/energy" — NOT the raw per-node sum, which correctly
//     scales with node count under a refinement) is conserved across the
//     resample, so a bump is neither amplified nor lost.
// World and normal entries are resampled independently (a vec3 field and a
// scalar field) and re-emitted with their own frame preserved.
export function resamplePointEditsToNet(edits, oldNetShape, newNetShape) {
  if (!edits || !edits.length) return [];
  const oldNu = oldNetShape.nu, oldNv = oldNetShape.nv;
  const newNu = newNetShape.nu, newNv = newNetShape.nv;
  if (oldNu < 1 || oldNv < 1 || newNu < 1 || newNv < 1) return [];

  const worldEntries = edits.filter((e) => e.frame !== 'normal');
  const normalEntries = edits.filter((e) => e.frame === 'normal');

  // Precompute, for every new-net fractional row/col, its basis distribution
  // over the OLD net (this is where the field reconstruction happens — a new
  // node's value is the old grid interpolated at that node's parameter).
  const rowDist = [];
  for (let a = 0; a < newNu; a++) rowDist.push(basisDistribution1D(oldNu, newNu > 1 ? a / (newNu - 1) : 0));
  const colDist = [];
  for (let b = 0; b < newNv; b++) colDist.push(basisDistribution1D(oldNv, newNv > 1 ? b / (newNv - 1) : 0));

  const out = [];
  const emit = (entries, dim, pick, make) => {
    if (!entries.length) return;
    const grid = buildSourceGrid(entries, oldNu, oldNv, dim, pick);
    for (let a = 0; a < newNu; a++) {
      for (let b = 0; b < newNv; b++) {
        const acc = new Array(dim).fill(0);
        for (const ru of rowDist[a]) {
          for (const cv of colDist[b]) {
            const w = ru.weight * cv.weight;
            const cell = grid[ru.index][cv.index];
            for (let c = 0; c < dim; c++) acc[c] += cell[c] * w;
          }
        }
        let mag = 0;
        for (let c = 0; c < dim; c++) mag += Math.abs(acc[c]);
        if (mag <= EDIT_EPS) continue;
        out.push(make(newNu > 1 ? a / (newNu - 1) : 0, newNv > 1 ? b / (newNv - 1) : 0, acc));
      }
    }
  };

  emit(worldEntries, 3, (e) => e.delta, (rowFrac, colFrac, v) => ({ rowFrac, colFrac, delta: [v[0], v[1], v[2]] }));
  emit(normalEntries, 1, (e) => [e.amount], (rowFrac, colFrac, v) => ({ rowFrac, colFrac, frame: 'normal', amount: v[0] }));
  return out;
}

// The AREA-WEIGHTED field integral of an edit list anchored to a net of the
// given shape — the honest "total displaced volume/energy" invariant a
// resample should approximately conserve (a plain per-node sum would
// spuriously scale with node count under a refinement). Returns { world:
// [x,y,z], normal:<scalar> } — world components summed as vectors, normal
// amounts summed as scalars, each weighted by that net's own fractional cell
// area. Independent of applyPointEdits — a real, checkable ground truth for
// the resample tests, not a self-consistency tautology.
export function pointEditFieldMass(edits, shape) {
  const nu = shape.nu, nv = shape.nv;
  const du = nu > 1 ? 1 / (nu - 1) : 1;
  const dv = nv > 1 ? 1 / (nv - 1) : 1;
  const world = [0, 0, 0];
  let normal = 0;
  for (const e of edits || []) {
    if (e.frame === 'normal') normal += e.amount * du * dv;
    else { world[0] += e.delta[0] * du * dv; world[1] += e.delta[1] * du * dv; world[2] += e.delta[2] * du * dv; }
  }
  return { world, normal };
}
