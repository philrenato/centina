// NURBS surface evaluation — Piegl & Tiller Ch. 3/4, the 2D analog of
// curve.mjs. A NurbsSrf here is { degU, degV, knotsU, knotsV, ctrlNet }
// where ctrlNet[i][j] = [x, y, z, w] (real point + weight, U-direction first
// index, V-direction second — matches this kernel's data model).
//
// Surface derivatives (A4.4, for shading normals / isocurve tangents) are
// deferred — not needed until P1's viewport shading. Point evaluation is
// enough to prove P0's revolve/extrude/sweep surfaces are well-formed.

import { findSpan, basisFuns, dersBasisFuns } from './basis.mjs';

function toHomogeneousNet(ctrlNet) {
  return ctrlNet.map(row => row.map(([x, y, z, w]) => [x * w, y * w, z * w, w]));
}

export function surfacePointHomogeneous(srf, u, v) {
  assertSurface(srf, 'surfacePoint');
  const { degU: p, degV: q, knotsU: U, knotsV: V } = srf;
  const Pw = toHomogeneousNet(srf.ctrlNet);
  const n = Pw.length - 1;
  const m = Pw[0].length - 1;
  const uspan = findSpan(n, p, u, U);
  const Nu = basisFuns(uspan, u, p, U);
  const vspan = findSpan(m, q, v, V);
  const Nv = basisFuns(vspan, v, q, V);

  const temp = [];
  for (let l = 0; l <= q; l++) {
    const s = [0, 0, 0, 0];
    for (let k = 0; k <= p; k++) {
      const cp = Pw[uspan - p + k][vspan - q + l];
      for (let c = 0; c < 4; c++) s[c] += Nu[k] * cp[c];
    }
    temp.push(s);
  }
  const Sw = [0, 0, 0, 0];
  for (let l = 0; l <= q; l++) {
    for (let c = 0; c < 4; c++) Sw[c] += Nv[l] * temp[l][c];
  }
  return Sw;
}

/* WHAT A SURFACE IS — the surface half of `assertCurve`, and there for the same
   reason: `surfacePoint({}, 0.5, 0.5)` used to answer "Cannot read properties of
   undefined (reading 'map')", which tells a newcomer nothing about which of the
   three arguments they got wrong. */
export function assertSurface(srf, fn = 'this function') {
  if (!srf || typeof srf !== 'object') throw new Error(`${fn}: expected a surface object { degU, knotsU, degV, knotsV, ctrlNet }, got ${srf === null ? 'null' : typeof srf}`);
  if (!Array.isArray(srf.knotsU) || !Array.isArray(srf.knotsV)) throw new Error(`${fn}: the surface has no usable knot vectors — expected { degU, knotsU, degV, knotsV, ctrlNet }`);
  if (!Array.isArray(srf.ctrlNet) || !Array.isArray(srf.ctrlNet[0])) throw new Error(`${fn}: the surface has no control net — expected ctrlNet indexed [u][v]`);
  if (!Number.isFinite(srf.degU) || !Number.isFinite(srf.degV)) throw new Error(`${fn}: the surface has no degrees — expected { degU, degV }`);
  return srf;
}

/* THE (u, v) RANGES A SURFACE IS DEFINED OVER, published for the same reason as
   `curveDomain`: neither is [0,1] in general, and every consumer would otherwise
   re-derive it from the knot vectors by hand. */
export function surfaceDomain(srf) {
  assertSurface(srf, 'surfaceDomain');
  return {
    u: [srf.knotsU[0], srf.knotsU[srf.knotsU.length - 1]],
    v: [srf.knotsV[0], srf.knotsV[srf.knotsV.length - 1]],
  };
}

export function surfacePoint(srf, u, v) {
  const Sw = surfacePointHomogeneous(srf, u, v);
  return [Sw[0] / Sw[3], Sw[1] / Sw[3], Sw[2] / Sw[3]];
}

// True if every control point in the net has finite, defined coordinates and
// a positive weight — the basic well-formedness gate every primitive/history
// command should pass before an object enters the document (04, 14 Note 2's
// "refuse silent bad export" ethic starts here, at construction time).
// Expects the NESTED surface shape (ctrlNet[i][j] = [x,y,z,w]) — the
// hand-pasted twin in the app is deliberately a DIFFERENT, flat-
// array-only version for curves (ctrlPts = [x,y,z,w][]), not a literal
// copy; the app flattens a surface's own net (srf.ctrlNet.flat()) before
// calling ITS version, precisely so both stay single-purpose and correct
// for their own callers rather than one trying to handle both shapes.
export function isFiniteNet(ctrlNet) {
  for (const row of ctrlNet) {
    for (const [x, y, z, w] of row) {
      if (![x, y, z, w].every(Number.isFinite)) return false;
      if (w <= 0) return false;
    }
  }
  return true;
}

// Whether a surface's own control net WRAPS AROUND in U and/or V — the
// first and last U-rows (or V-columns) of the net coincide within
// tolerance, meaning that parametric boundary is an internal SEAM, not a
// real free edge. Genuinely computable from the control net alone, no
// Brep/topology structure needed — this app's only surface
// representation is still a single untrimmed face (no polysurface Join
// exists yet), so per-surface closure is as far
// as "naked edge" analysis can honestly go until a real Brep lands.
export function surfaceClosure(srf, tol = 1e-6) {
  const net = srf.ctrlNet;
  const nu = net.length, nv = net[0].length;
  const same = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) <= tol && Math.abs(a[3] - b[3]) <= tol;
  let closedU = true;
  for (let j = 0; j < nv; j++) if (!same(net[0][j], net[nu - 1][j])) { closedU = false; break; }
  let closedV = true;
  for (let i = 0; i < nu; i++) if (!same(net[i][0], net[i][nv - 1])) { closedV = false; break; }
  return { closedU, closedV };
}

// THE SEAM BAND — the control points a drag near a closed surface's seam must
// carry with it, the companion to surfaceStructuralGroup's "same physical
// point" rule. These are DIFFERENT points, deliberately: they move together
// not because they coincide but because the seam's smoothness is a relation
// among them.
//
// A closed direction is stored CLAMPED, its first and last control column
// holding the same point with nothing structural tying the two ends. Writing
// one column alone fails twice over, and only the first failure is obvious.
// Measured on a degree-3 editable torus, moving the seam column by 12 against
// an untouched seam of 0.0033 degrees:
//   · the seam column alone         — an 8.0 mm gap AND a 64.5 degree break
//   · the coincident pair together  — no gap, still a 129 degree crease
//   · one column either side too    — no gap, 0.0434 degrees
//   · the full `degree`-wide band   — no gap, 0.0033 degrees, its own value
// So closing the gap is not enough on its own, and the band is where it stops
// improving: every difference INSIDE a rigidly moved band is unchanged, and
// the continuity relations across the seam are relations among exactly those
// differences, so they survive the edit exactly rather than approximately.
//
// Returns every OTHER (i, j) in the band — matching surfaceStructuralGroup's
// own convention of excluding the point asked about. An OPEN direction
// contributes nothing, so an open surface's behavior is unchanged.
export function surfaceSeamBandGroup(srf, i0, j0, closure) {
  const c = closure || surfaceClosure(srf);
  const rows = seamBandIndices(i0, srf.ctrlNet.length, srf.degU, c.closedU);
  const cols = seamBandIndices(j0, srf.ctrlNet[0].length, srf.degV, c.closedV);
  const out = [];
  for (const i of rows) for (const j of cols) if (i !== i0 || j !== j0) out.push({ i, j });
  return out;
}

// The band is `degree` wide at EACH end, capped at half the net so a net too
// small to hold two disjoint bands moves as a whole rather than producing an
// overlapping one.
function seamBandIndices(index, count, degree, closed) {
  if (!closed || count < 2) return [index];
  const deg = Math.max(1, Math.min(degree, Math.floor(count / 2)));
  if (index >= deg && index < count - deg) return [index];
  const band = new Set();
  for (let k = 0; k < deg; k++) { band.add(k); band.add(count - 1 - k); }
  return [...band].sort((a, b) => a - b);
}

// DIVIDESRF — the surface analog of curve.mjs's divideByArcLength, but a
// genuinely SIMPLER operation: real Rhino's own DivideSrf places points on a
// plain PARAMETER-SPACE grid (equal U/V domain fractions), not an
// arc-length-even one. That's not a corner cut — a general surface patch has
// no single well-defined "even by real length" grid at all without
// arbitrarily privileging one direction (a curve has exactly one own arc
// length to divide evenly; a surface's physical spacing along an
// isoparametric grid line varies row-to-row and column-to-column in
// general), so a parameter-uniform grid IS the honest default here, matching
// the real tool this mirrors. uCount/vCount name SEGMENT counts (Rhino's own
// "number of divisions" language, and the same semantics divideByArcLength's
// own `count` already uses). In a direction the surface is OPEN in, that
// gives uCount+1 (or vCount+1) points, BOTH domain ends included EXACTLY (not
// derived from a fractional step) so a bilinear (already parameter-
// proportional) patch reproduces exact grid corners/edges bit-for-bit. In a
// direction the surface is CLOSED in (surfaceClosure below — the same test
// ExtractBorder already uses), the domain's own two ends are the SAME
// physical seam, so only uCount (or vCount) points are placed around the
// full loop, never repeating that seam column/row — the exact curve-level
// rule divideByArcLength applies to a closed curve's own seam, applied here
// per-direction. Reuses surfacePoint directly — no new evaluation math, only
// the grid walk is new.
export function divideSrfGrid(srf, uCount, vCount) {
  if (!Number.isInteger(uCount) || uCount < 1) throw new Error('divideSrfGrid: uCount must be a positive integer');
  if (!Number.isInteger(vCount) || vCount < 1) throw new Error('divideSrfGrid: vCount must be a positive integer');
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const { closedU, closedV } = surfaceClosure(srf);
  const uN = closedU ? uCount : uCount + 1;
  const vN = closedV ? vCount : vCount + 1;
  const results = [];
  for (let i = 0; i < uN; i++) {
    const u = i === 0 ? uMin : (!closedU && i === uCount) ? uMax : uMin + (uMax - uMin) * (i / uCount);
    for (let j = 0; j < vN; j++) {
      const v = j === 0 ? vMin : (!closedV && j === vCount) ? vMax : vMin + (vMax - vMin) * (j / vCount);
      results.push({ u, v, point: surfacePoint(srf, u, v) });
    }
  }
  return results;
}

// The number of NAKED (free, unshared) boundary edges of a single
// untrimmed-face surface — up to 4 parametric sides, minus 2 for each
// direction that wraps (a seam is internal, not a boundary). A pole
// (profile point touching the revolve axis) degenerates that boundary to
// a single point rather than removing it — still counted, same as Rhino
// still lists a zero-length singular edge in its own WHAT/EDGES report.
export function nakedEdgeCount(srf) {
  const { closedU, closedV } = surfaceClosure(srf);
  return (closedU ? 0 : 2) + (closedV ? 0 : 2);
}

// SEAM/POLE STRUCTURAL SIBLINGS — given a
// control-net (row, col) index, returns every OTHER (row, col) index that
// is the SAME physical point and must therefore move together with it, so
// a control-point drag can never open a seam or tear a pole apart by
// moving only one of its duplicated copies. Two structural sources, both
// reused rather than re-derived:
//   - SEAM: a direction the surface is CLOSED in (surfaceClosure above —
//     the exact same test INSPECT/naked-edge reporting already trusts)
//     makes its own first and last control row (or column) the SAME
//     physical seam — moving row 0 without row (nu-1) at the same column
//     opens a gap that was never really there.
//   - POLE: revolve()'s own pole detection (a profile point ON the axis
//     collapses its whole row to one point, via the same coincidence test
//     revolve() applies at 1e-9) is reused HERE post-hoc, on the already-
//     built net, rather than re-derived from a profile — an entire row
//     (or, defensively, an entire column, for a future surface-producing
//     command that might collapse the other direction) sharing one exact
//     point is a pole, wherever in the net it sits, not just at a
//     boundary (an hourglass/goblet profile can touch the axis at an
//     INTERIOR control point too).
// Both checks are STRUCTURAL — whole-row/whole-column/whole-boundary
// coincidence, the surface's own construction guaranteeing near-exact
// float agreement — never a raw "these two individual points happen to be
// near each other" distance check, which could accidentally weld two
// genuinely distinct control points that merely touch. A small BFS closure
// (not just one hop) is required because a direction closed in BOTH U and
// V (a torus) identifies a CORNER control point across all four of its own
// index combinations, transitively, not just one pairing away.
//
// The pole check compares POSITION only (x, y, z), deliberately not
// weight — revolve()'s own pole row carries the correct ALTERNATING
// rational weight shape across its columns (arcSpanPoints' tangent-line
// weight, needed for the surface's own row-blend exactness — see
// revolve()'s own header comment), even though every column's real 3D
// POSITION already collapsed to the exact same point (radius 0). Weight
// is exactly what setControlHandle already leaves untouched on a write
// (matching Circle/a solid's shared-vertex write path), so pairing on
// position alone is what makes every pole column move to the SAME
// physical spot while each keeps its own distinct rational weight.
export function surfaceStructuralGroup(srf, i0, j0, tol = 1e-6) {
  const net = srf.ctrlNet;
  const nu = net.length, nv = net[0].length;
  const samePos = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) <= tol;
  const { closedU, closedV } = surfaceClosure(srf, tol);
  const key = (i, j) => `${i}|${j}`;
  const seen = new Set([key(i0, j0)]);
  const queue = [[i0, j0]];
  while (queue.length) {
    const [i, j] = queue.shift();
    const add = (ni, nj) => { const k = key(ni, nj); if (!seen.has(k)) { seen.add(k); queue.push([ni, nj]); } };
    if (closedU) {
      if (i === 0) add(nu - 1, j);
      if (i === nu - 1) add(0, j);
    }
    if (closedV) {
      if (j === 0) add(i, nv - 1);
      if (j === nv - 1) add(i, 0);
    }
    let rowIsPole = true;
    for (let jj = 1; jj < nv && rowIsPole; jj++) if (!samePos(net[i][0], net[i][jj])) rowIsPole = false;
    if (rowIsPole) for (let jj = 0; jj < nv; jj++) add(i, jj);
    let colIsPole = true;
    for (let ii = 1; ii < nu && colIsPole; ii++) if (!samePos(net[0][j], net[ii][j])) colIsPole = false;
    if (colIsPole) for (let ii = 0; ii < nu; ii++) add(ii, j);
  }
  seen.delete(key(i0, j0));
  return [...seen].map((k) => { const [i, j] = k.split('|').map(Number); return { i, j }; });
}

// STRUCTURED U/V ROW-COLUMN SELECTION —
// "select the whole U row / V column a control point belongs to." A
// tensor-product control net makes this pure arithmetic (every point
// sharing a row or column index), never the BFS graph-walk a SubD cage's
// SelEdgeLoop/SelEdgeRing genuinely needs over arbitrary topology — the
// ONE real subtlety is the SAME one item 1d already solved: on a CLOSED
// surface a row/column selection must include BOTH physically-coincident
// seam copies (and a full pole row), or row-select becomes a fresh way to
// tear the seam the instant a student drags the whole row. Reuses
// surfaceStructuralGroup DIRECTLY, per (i,j), rather than re-deriving the
// seam/pole pairing a second time — every produced pair is expanded
// through it too, so a torus corner (row AND column both closed) still
// resolves to its full seam-complete set from either surfaceRowGroup or
// surfaceColGroup.
export function surfaceRowGroup(srf, i0, tol = 1e-6) {
  const nv = srf.ctrlNet[0].length;
  const seen = new Set();
  const out = [];
  const add = (i, j) => { const k = `${i}|${j}`; if (!seen.has(k)) { seen.add(k); out.push({ i, j }); } };
  for (let j = 0; j < nv; j++) {
    add(i0, j);
    for (const sib of surfaceStructuralGroup(srf, i0, j, tol)) add(sib.i, sib.j);
  }
  return out;
}
export function surfaceColGroup(srf, j0, tol = 1e-6) {
  const nu = srf.ctrlNet.length;
  const seen = new Set();
  const out = [];
  const add = (i, j) => { const k = `${i}|${j}`; if (!seen.has(k)) { seen.add(k); out.push({ i, j }); } };
  for (let i = 0; i < nu; i++) {
    add(i, j0);
    for (const sib of surfaceStructuralGroup(srf, i, j0, tol)) add(sib.i, sib.j);
  }
  return out;
}

// ---------------------------------------------------------------
// FACE (ISOCURVE-CELL) CHAINS — the ordinary-surface counterpart of a
// SubD face loop, so "grab the strip of faces running this way" means the
// same thing on both surface families.
//
// A face here is what a student actually sees: one CELL bounded by four
// adjacent drawn isocurves. Those isocurves sit at GREVILLE ABSCISSAE —
// one per control point — so an nu x nv control net draws an
// (nu-1) x (nv-1) grid of cells, and cell (i, j) spans grevilleU[i]..
// grevilleU[i+1] by grevilleV[j]..grevilleV[j+1].
//
// WHY THIS IS ARITHMETIC AND THE SUBD SIBLING IS A GRAPH WALK: a tensor-
// product surface's cells are a genuine rectangular grid, so the strip
// through one is every cell sharing its row or column — no traversal, no
// termination cases. The SubD walk exists because a cage has arbitrary
// topology; this one would be dishonest to write that way.
//
// AND WHY NO SEAM SPECIAL CASE, which the control-point groups above DO
// need: a closed direction makes the first and last control ROWS
// coincide, so surfaceRowGroup has to pair them. Cells sit BETWEEN
// Grevilles, so there is no duplicated cell to pair — a closed surface's
// cell strip is simply every cell in that row. Checked rather than
// assumed; it is why these two functions look so much simpler than their
// neighbors directly above.
// ---------------------------------------------------------------

// Cell counts for a surface, as [uCells, vCells].
export function surfaceCellCounts(srf) {
  return [srf.ctrlNet.length - 1, srf.ctrlNet[0].length - 1];
}

// The strip of cells running through cell (i0, j0).
//   dir 'u' — every cell along U (i varies, j0 fixed)
//   dir 'v' — every cell along V (i0 fixed, j varies)
// Refuses an out-of-range cell or an unknown direction by name rather
// than returning a plausible-looking empty strip.
export function surfaceCellStrip(srf, i0, j0, dir) {
  const [uCells, vCells] = surfaceCellCounts(srf);
  if (!(i0 >= 0 && i0 < uCells && j0 >= 0 && j0 < vCells)) {
    throw new Error(`surfaceCellStrip: cell (${i0}, ${j0}) is outside this surface's ${uCells}x${vCells} cell grid`);
  }
  if (dir !== 'u' && dir !== 'v') throw new Error(`surfaceCellStrip: direction must be 'u' or 'v', got "${dir}"`);
  const out = [];
  if (dir === 'u') for (let i = 0; i < uCells; i++) out.push({ i, j: j0 });
  else for (let j = 0; j < vCells; j++) out.push({ i: i0, j });
  return out;
}

// WHICH WAY DID THE PRESS MEAN — the direction whose bounding isocurve
// the press landed nearest, given its position INSIDE the cell expressed
// as fractions (fu, fv) of that cell's own span.
//
// Same rule as the SubD sibling (nearestFaceEdgeToPoint): press near an
// edge and you have pointed at the strip that CROSSES it. Crossing a
// constant-u isocurve steps to the next cell in i, so a press near one
// yields 'u'.
//
// COMPARED IN CELL FRACTIONS, NOT RAW PARAMETERS, and that is the whole
// reason this takes fractions at all: u and v domains have unrelated
// scales (a revolve's sweep parameter against its profile's), so raw
// parameter distances would let whichever domain happens to be numerically
// larger win nearly every press regardless of where it actually landed.
//
// A dead-center press is a real tie and resolves to 'u' every time —
// stable across repeated reads, so a caller re-reading the same press
// gets the same answer and an explicit re-read control does not appear
// dead on its second press.
export function nearestCellDirection(fu, fv) {
  const du = Math.min(fu, 1 - fu); // distance to the nearer constant-u edge
  const dv = Math.min(fv, 1 - fv);
  return du <= dv ? 'u' : 'v';
}

// ExtractIsocurve click-to-pick — surface point INVERSION:
// given an arbitrary 3D point known to lie ON (or very near) the surface
// — the real raycast hit point from a viewport click — find the (u, v)
// parameter it corresponds to. curve.mjs's closestPointOnCurve already
// solves the 1D analog for Sweep1 N-profiles' own rail-stationing need
// (coarse search, then Newton-Raphson refinement on the real derivatives)
// — this is the same two-stage recipe generalized to 2 parameters. The
// one real difference from the curve case: a rational SURFACE'S closest-
// point problem is a genuine 2-variable least-squares minimization of
// |S(u,v) - P|^2, not a 1-variable root find, so the refinement step is
// Gauss-Newton (solve the 2x2 normal-equations system J^T J * delta =
// -J^T r using the real first partials Su/Sv, dropping the curvature
// term full Newton would need from second partials — the standard,
// well-known simplification for this exact class of problem, valid
// because the residual r shrinks toward zero near the true closest
// point, which is exactly where the dropped term would matter least)
// rather than curve.mjs's own 1D Newton on f'(u)/f''(u).
//
// surfacePointAndPartials is the 2-variable sibling of curve.mjs's
// rationalCurveDerivs: dersBasisFuns (already used by isocurve.mjs's own
// extractIsocurveU/V machinery indirectly via basisFuns) gives the
// value+first-derivative basis functions in EACH direction independently
// (no mixed d^2/dudv term is needed here, only Su and Sv, so this never
// needs a true mixed-partial basis evaluation); the SAME quotient-rule
// trick rationalCurveDerivs uses (a rational curve's homogeneous form IS
// an ordinary B-spline in 4D) applies per-direction: Su = (Swu - wu*S)/w,
// Sv = (Swv - wv*S)/w.
export function surfacePointAndPartials(srf, u, v) {
  const { degU: p, degV: q, knotsU: U, knotsV: V } = srf;
  const Pw = toHomogeneousNet(srf.ctrlNet);
  const n = Pw.length - 1;
  const m = Pw[0].length - 1;
  const uspan = findSpan(n, p, u, U);
  const dersU = dersBasisFuns(uspan, u, p, 1, U); // dersU[0]=N, dersU[1]=N'
  const vspan = findSpan(m, q, v, V);
  const dersV = dersBasisFuns(vspan, v, q, 1, V);
  const Sw = [0, 0, 0, 0], Swu = [0, 0, 0, 0], Swv = [0, 0, 0, 0];
  for (let k = 0; k <= p; k++) {
    for (let l = 0; l <= q; l++) {
      const cp = Pw[uspan - p + k][vspan - q + l];
      const Nu0 = dersU[0][k], Nu1 = dersU[1][k];
      const Nv0 = dersV[0][l], Nv1 = dersV[1][l];
      for (let c = 0; c < 4; c++) {
        Sw[c] += Nu0 * Nv0 * cp[c];
        Swu[c] += Nu1 * Nv0 * cp[c];
        Swv[c] += Nu0 * Nv1 * cp[c];
      }
    }
  }
  const w = Sw[3];
  const point = [Sw[0] / w, Sw[1] / w, Sw[2] / w];
  const su = [(Swu[0] - Swu[3] * point[0]) / w, (Swu[1] - Swu[3] * point[1]) / w, (Swu[2] - Swu[3] * point[2]) / w];
  const sv = [(Swv[0] - Swv[3] * point[0]) / w, (Swv[1] - Swv[3] * point[1]) / w, (Swv[2] - Swv[3] * point[2]) / w];
  return { point, su, sv };
}

// Wraps t into [tMin,tMax) when `closed`, otherwise clamps to the domain —
// the shared domain-boundary rule every Gauss-Newton step below and in
// refineClosestPointOnSurface (kernel/trim.mjs) uses identically. A surface
// closed in a direction (a full torus/revolve/sphere sweep) has no real
// domain edge there at all — u=uMin and u=uMax are the SAME physical
// point — so a search must be able to continue past the seam rather than
// pin against a boundary that isn't geometrically real.
export function wrapParam(t, tMin, tMax, closed) {
  const span = tMax - tMin;
  if (!closed || span <= 0) return Math.max(tMin, Math.min(tMax, t));
  let r = (t - tMin) % span;
  if (r < 0) r += span;
  return tMin + r;
}

// A POLE IS RANK-1, NOT RANK-0 — and a search that treats it as rank-0 cannot
// leave it. Where a revolve's profile meets its axis, every v names the SAME
// physical point: `sv` collapses, and the Gauss-Newton determinant with it,
// while `su` along the profile stays as healthy as anywhere else on the
// surface (3e+4 against 5e-31 on a real one). Returning the current point
// there — the only safe move for a 2-D solve that cannot be formed — makes the
// pole ABSORBING, and a coarse seed grid puts searches there far more often
// than the geometry warrants: on a strongly shaped revolve the pole is
// genuinely the nearest sample of the whole grid, because the true minimum
// hides inside the first cell beside it, where the surface expands fastest per
// unit of u. A target lying ON such a surface then reads over a millimeter
// away from it.
//
// The escape is a 1-D move, but the live direction is only half of it: WHICH
// meridian to leave along is decided by the collapsed parameter, whose value
// at a pole is arbitrary. Stepping along the live direction while keeping that
// arbitrary value walks away from the target on any surface whose pole is not
// a long way from it — measured on a sphere, where the step is then rejected
// and the search freezes exactly as before. So the dead parameter is CHOSEN,
// not kept: step just off the degenerate point (a thousandth of the domain,
// where the collapsed direction has opened up but the geometry is still the
// pole's own neighborhood) and scan the dead parameter across its whole
// domain there, at the same 24 samples the seed grid caps itself at.
//
// A step is taken only if it genuinely improves, so a target whose nearest
// point really IS the pole — anything on the axis — keeps the pole and this
// returns null, leaving the caller to stop where it is.
export function escapeDegeneratePoint(srf, targetPt, u, v, curDistSq, Juu, Jvv, closedU, closedV) {
  if (Juu < 1e-14 && Jvv < 1e-14) return null; // both directions collapsed — no move exists
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const liveIsU = Juu >= Jvv;
  const liveMin = liveIsU ? uMin : vMin, liveMax = liveIsU ? uMax : vMax;
  const deadMin = liveIsU ? vMin : uMin, deadMax = liveIsU ? vMax : uMax;
  const here = liveIsU ? u : v;
  // A pole sits at an end of its own direction's domain, so the step goes
  // toward the interior; from the middle (which no pole occupies) either way
  // is as good.
  const inward = here - liveMin < (liveMax - liveMin) * 0.5 ? 1 : -1;
  const stepped = wrapParam(here + inward * (liveMax - liveMin) * 1e-3, liveMin, liveMax, liveIsU ? closedU : closedV);
  let bestDead = null, bestDistSq = curDistSq;
  for (let k = 0; k < 24; k++) {
    const dead = deadMin + (deadMax - deadMin) * (k / 24);
    const p = liveIsU ? surfacePoint(srf, stepped, dead) : surfacePoint(srf, dead, stepped);
    const dx = p[0] - targetPt[0], dy = p[1] - targetPt[1], dz = p[2] - targetPt[2];
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bestDistSq) { bestDistSq = d; bestDead = dead; }
  }
  if (bestDead === null) return null;
  return liveIsU
    ? { u: stepped, v: bestDead, distSq: bestDistSq }
    : { u: bestDead, v: stepped, distSq: bestDistSq };
}

// Closest (u, v) on the surface to an arbitrary 3D point (the real click-
// raycast hit). Stage 1: a coarse parameter-space grid search (resolution
// scales with the control net's own size, clamped to a sane [8,24] range
// per direction — a finer net can plausibly need a finer coarse seed, an
// arbitrarily fixed constant here would silently under-seed a dense net
// exactly like divideByArcLength's own tolerance comment warns against
// for a fixed absolute number). Stage 2: bounded Gauss-Newton refinement.
// A direction the surface is CLOSED in wraps at the domain boundary
// instead of clamping (via wrapParam above), so the search can cross a
// torus/revolve seam instead of pinning dead against it. A step that
// worsens the distance is backtracked (halved, up to 5 times) rather than
// rejected outright — the original hard-reject-on-first-worsening-step
// only ever produced the exact right answer on an AFFINE surface (a flat
// plane or a straight-profile ruled extrusion), where Gauss-Newton
// converges in one exact step from any seed; on anything genuinely curved
// the full step routinely overshoots (the algorithm drops the curvature
// term (S-P).Suu from the Jacobian), and a hard reject there just freezes
// the search at its current point forever — which is silently how a
// caller warm-starting sample-to-sample (refineClosestPointOnSurface)
// produced long runs of byte-identical points. A smaller step along the
// same Newton direction is very often still an improvement.
export function closestPointOnSurface(srf, targetPt, opts = {}) {
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const gridU = opts.gridU ?? Math.max(8, Math.min(24, srf.ctrlNet.length * 3));
  const gridV = opts.gridV ?? Math.max(8, Math.min(24, srf.ctrlNet[0].length * 3));
  const { closedU, closedV } = surfaceClosure(srf);
  const distSq = (p) => {
    const dx = p[0] - targetPt[0], dy = p[1] - targetPt[1], dz = p[2] - targetPt[2];
    return dx * dx + dy * dy + dz * dz;
  };
  /* ⚠⚠ THE GRID'S BEST SAMPLE IS NOT ALWAYS IN THE RIGHT BASIN, and descending
     from it alone turns that into a wrong answer rather than a slow one. A
     surface that comes back near itself has several places that are locally
     nearest, the grid resolves each only to its own spacing, and where a tight
     lobe is sampled closer than the true nearest region the solve descends into
     the lobe and converges there, perfectly, somewhere else on the shape.

     It is not a density problem and raising the grid does not fix it: measured on
     a freeform rim, seeds of 12, 30, 48, 96 and 192 all find the right point at
     0.005mm and 24 — the value that surface's control net happens to ask for —
     returns 6.57mm, because at exactly that spacing one sample lands on the
     tightest lobe.

     So every local minimum of the grid is a candidate and the best few are each
     descended from, keeping whichever converges nearest. The extra cost falls
     only on surfaces that genuinely have more than one candidate. */
  const cells = [];
  for (let i = 0; i <= gridU; i++) {
    const u = uMin + (uMax - uMin) * (i / gridU);
    for (let j = 0; j <= gridV; j++) {
      const v = vMin + (vMax - vMin) * (j / gridV);
      cells.push({ i, j, u, v, d: distSq(surfacePoint(srf, u, v)) });
    }
  }
  const cellAt = (i, j) => {
    let ii = i, jj = j;
    if (ii < 0) ii = closedU ? gridU - 1 : 0; else if (ii > gridU) ii = closedU ? 1 : gridU;
    if (jj < 0) jj = closedV ? gridV - 1 : 0; else if (jj > gridV) jj = closedV ? 1 : gridV;
    return cells[ii * (gridV + 1) + jj];
  };
  const seeds = cells.filter((c) => {
    const d = c.d;
    return d <= cellAt(c.i - 1, c.j).d && d <= cellAt(c.i + 1, c.j).d
      && d <= cellAt(c.i, c.j - 1).d && d <= cellAt(c.i, c.j + 1).d;
  }).sort((a, b) => a.d - b.d).slice(0, 3);
  if (!seeds.length) seeds.push(cells.reduce((a, b) => (b.d < a.d ? b : a), cells[0]));
  const refineFrom = (seed) => {
  let u = seed.u, v = seed.v, curDistSq = seed.d;
  for (let iter = 0; iter < 20; iter++) {
    const { point, su, sv } = surfacePointAndPartials(srf, u, v);
    const r = [point[0] - targetPt[0], point[1] - targetPt[1], point[2] - targetPt[2]];
    const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const Juu = dot3(su, su), Juv = dot3(su, sv), Jvv = dot3(sv, sv);
    const bu = dot3(su, r), bv = dot3(sv, r);
    const det = Juu * Jvv - Juv * Juv;
    if (Math.abs(det) < 1e-14) {
      const esc = escapeDegeneratePoint(srf, targetPt, u, v, curDistSq, Juu, Jvv, closedU, closedV);
      if (!esc) break;
      u = esc.u; v = esc.v; curDistSq = esc.distSq;
      continue;
    }
    const du = -(Jvv * bu - Juv * bv) / det;
    const dv = -(Juu * bv - Juv * bu) / det;
    let accepted = false, didConverge = false;
    for (let bt = 0, step = 1; bt < 5; bt++, step *= 0.5) {
      const uNext = wrapParam(u + du * step, uMin, uMax, closedU);
      const vNext = wrapParam(v + dv * step, vMin, vMax, closedV);
      const nextDistSq = distSq(surfacePoint(srf, uNext, vNext));
      if (nextDistSq <= curDistSq + 1e-12) {
        didConverge = Math.abs(uNext - u) < 1e-10 && Math.abs(vNext - v) < 1e-10;
        u = uNext; v = vNext; curDistSq = nextDistSq;
        accepted = true;
        break;
      }
    }
    if (!accepted || didConverge) break;
  }
  return { u, v, point: surfacePoint(srf, u, v), distance: Math.sqrt(curDistSq), distSq: curDistSq };
  };
  let best = null;
  for (const seed of seeds) {
    const got = refineFrom(seed);
    if (!best || got.distSq < best.distSq) best = got;
  }
  return best;
}

// ADAPTIVE V-DIRECTION RENDER-MESH DENSITY — a real, SEPARATE
// gap alongside the just-shipped rounded-Pipe-
// corner fix (kernel/sweep.mjs's own "PER-RAIL-SPAN COMPOSED FIT" comment):
// that fix proves the swept NURBS SURFACE itself is genuinely fold-free
// per real rail span (worst adjacent-face-normal angle under 0.0003 degrees,
// sampled densely WITHIN each span) — but the app's own
// `tessellateSurface`, the function that actually turns that surface into
// the triangle mesh on screen, used to walk the V direction with a
// PLAIN UNIFORM loop (`v = vMin + (j/vRes)*(vMax-vMin)`), completely blind
// to where the surface's own real span boundaries are. On an irregular
// rail (a short fillet-arc span next to much longer straight runs) one
// span's own v-fraction footprint can be under 0.01 of the whole domain —
// at the shipped default `vRes=96`, the uniform step (≈0.0104) can land
// ZERO samples strictly inside a span that narrow, so two samples several
// degrees of real surface curvature apart end up connected by a single
// straight-line mesh edge. Confirmed directly (not assumed): on the
// irregular reproduction fixture with a MILDER, non-self-intersecting
// corner radius (15mm, comfortably larger than the swept tube's own 5mm
// cross-section — see the self-intersection paragraph below for why that
// qualifier matters), the shipped uniform-V mesh at `vRes=96` measured a
// genuine 16.9-degree adjacent-face-normal fold; a real, visible facet,
// even though the underlying surface has zero genuine C0-or-worse
// discontinuity there (checked directly with the SAME derivative-dot-
// product test `tessellateSurface`'s own existing U-direction hard-break
// logic already uses — every one of these V-direction span-boundary knots
// reads as fully G1-continuous in practice, dot >= 0.999 every time; this
// is a pure SAMPLING gap, never a genuine kink, so the fix below adds
// DENSITY, never a duplicate/kinked vertex row the way the existing
// U-direction logic does for a real break).
//
// THE FIX: `tessellationVSamples` finds every GENUINE span boundary in
// `srf.knotsV` (full-multiplicity knots, >= degV, excluding the two domain
// ends — the exact same "real joint vs. an ordinary smooth interior knot"
// distinction `kernel/sweep.mjs`'s own `railHardBreakParams` and
// `tessellateSurface`'s own U-direction gate both already use) and, for any
// such span whose plain uniform allotment falls under `minSamplesPerSpan`
// (default 48 — reusing `kernel/sweep.mjs`'s own established
// `MIN_SPAN_SAMPLES` constant/precedent for "a narrow span needs a real
// density floor," not a newly-invented number), merges in extra evenly-
// spaced samples so that span is genuinely resolved. A surface with no
// such span at all (the overwhelming common case — a plain Line/Arc/
// SketchCurve rail, or a sharp/no-corner Pipe) takes the exact plain
// uniform array, BYTE-IDENTICAL to before this fix, zero regression risk.
// Verified directly on the SAME milder 15mm-corner-radius fixture above:
// worst adjacent-face-normal fold drops from 16.9 degrees (OLD) to 2.4
// degrees (NEW, `minSamplesPerSpan=48`) — a genuine, large improvement,
// not asserted-away to zero (an honest residual of finite density,
// unavoidable for any fixed-budget adaptive tessellation, the same
// character as MIN_SPAN_SAMPLES/MAX_SPAN_SAMPLES's own tradeoff).
//
// HONEST, SEPARATE LIMITATION, measured and named directly rather than
// silently left in place: on the EXACT severe reproduction fixture
// (a 2mm fillet corner radius next to a swept 5mm-radius tube), this fix
// does NOT close the gap — the worst adjacent-face-normal fold STAYS large
// (176-179 degrees) no matter how much density is added, and actually
// INCREASES slightly as density increases (176.3 at `minSamplesPerSpan=0`
// -> 179.4 at `minSamplesPerSpan=48`), the opposite direction an aliasing
// artifact would move. That signature, cross-checked directly by sweeping
// `cornerRadius` from 2mm to 40mm (fold stays 150-179 degrees whenever
// `cornerRadius` < the swept tube's own cross-section radius, then drops
// sharply to single digits once `cornerRadius` exceeds it), confirms this
// is a REAL, physical surface self-intersection — a tube literally cannot
// bend around a fillet whose own radius of curvature is smaller than the
// tube's cross-section radius without overlapping itself — not a
// tessellation defect at all, and not something any render-mesh-density
// fix can resolve. An exact fix would mean actual self-intersection
// detection/avoidance at Pipe-build time (e.g. refusing or clamping a
// `cornerRadius` smaller than `radius`), real, separate, harder scope,
// deliberately not attempted.
export function tessellationVSamples(srf, vRes, minSamplesPerSpan = 48) {
  const vMin = srf.knotsV[srf.degV], vMax = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  const vSamples = [];
  for (let j = 0; j <= vRes; j++) vSamples.push(vMin + (j / vRes) * (vMax - vMin));
  const mult = new Map();
  for (const k of srf.knotsV) mult.set(k, (mult.get(k) || 0) + 1);
  const bounds = [vMin];
  for (const [k, m] of mult) {
    if (k <= vMin + 1e-9 || k >= vMax - 1e-9 || m < srf.degV) continue; // domain boundary, or provably-smooth-by-multiplicity — no real span break here
    bounds.push(k);
  }
  bounds.push(vMax);
  bounds.sort((a, b) => a - b);
  if (bounds.length <= 2 || !minSamplesPerSpan) return vSamples; // single span (or explicitly disabled) — byte-identical to plain uniform, the overwhelming common case
  const merged = new Set(vSamples);
  const uniformStep = (vMax - vMin) / vRes;
  for (let s = 0; s < bounds.length - 1; s++) {
    const lo = bounds[s], hi = bounds[s + 1];
    let inside = 0;
    for (const v of vSamples) if (v > lo + 1e-9 && v < hi - 1e-9) inside++;
    if (inside >= minSamplesPerSpan) continue;
    /* ⚠ A FORCED SAMPLE THAT LANDS ALMOST ON TOP OF A UNIFORM ONE IS NOT
       DENSITY, IT IS A SLIVER. `Set` dedupes on exact equality, so two
       parameters a millionth of a span apart both survive and the grid gets a
       row strip that width — triangles whose circumradius-to-inradius ratio
       runs into the hundreds on a surface that is otherwise evenly meshed, and
       the worst-shaped triangles in the whole mesh on every closed revolve.
       The two lists nearly coincide by construction whenever this fires: a span
       one sample short of the floor is allotted 47 uniform samples at k/48 of
       the span and then handed 48 forced ones at k/49, which agree to
       1/(48*49) at the span ends and only separate toward the middle.
       A forced sample within a quarter of the local spacing of a uniform one is
       therefore DROPPED, not added: the uniform sample already resolves that
       part of the span, so the span still clears `minSamplesPerSpan` counting
       it, and the guarantee this function exists for is untouched. */
    const gap = Math.min((hi - lo) / (minSamplesPerSpan + 1), uniformStep) * 0.25;
    for (let k = 1; k <= minSamplesPerSpan; k++) {
      const v = lo + (hi - lo) * k / (minSamplesPerSpan + 1);
      if (Math.abs(v - (vMin + Math.round((v - vMin) / uniformStep) * uniformStep)) <= gap) continue;
      merged.add(v);
    }
  }
  return [...merged].sort((a, b) => a - b);
}

// ARC-LENGTH GRID RESOLUTION — a declared `uRes x vRes` is a COUNT, and a
// count says nothing about the SHAPE it is counting across. The same 96x192
// grid that is well proportioned on a sphere puts 192 divisions along the
// 7mm straight wall of a disc extrusion whose circumference is 283mm, so its
// cells measure 1.47 x 0.036 units and its triangles carry interior angles
// under 3 degrees. A rasteriser hides that; a path tracer draws it.
//
// THE MEASUREMENT IS CHORD DEVIATION IN WORLD UNITS, NOT PARAMETER SPAN.
// A coarse probe grid (`probe` cells per direction, capped by the declared
// count so a small grid is never probed more finely than it is drawn) is
// evaluated once, and three deviations are read off it:
//
//   devU  the sagitta of an isocurve in U — the perpendicular distance of a
//         sample from the chord joining its two neighbours along U. Zero on
//         a straight ruling, largest where the surface turns tightest.
//   devV  the same one direction over.
//   devT  the TWIST: the distance of a cell's fourth corner from the plane
//         of its other three. This is the term a per-direction sagitta
//         cannot see, and it is the one that matters for a bilinear patch
//         with a corner lifted — every isocurve of that saddle is a straight
//         line, devU and devV are both exactly zero, and the patch still
//         needs subdividing both ways. Planarity of the control net is the
//         wrong test for the same reason; developability is the right one,
//         and devT is what measures it.
//
// A chord's sagitta scales as the square of the sample spacing and a
// bilinear cell's twist error scales as the product of its two spacings, so
// each deviation can be re-evaluated at any division count without
// re-sampling: dev(n) = devU * (su/n)^2, and devT(nu,nv) = devT * (su/nu) *
// (sv/nv).
//
// THE TOLERANCE IS THE DECLARED GRID'S OWN WORST DEVIATION, not a new
// constant. Evaluate all three at the declared counts, take the largest, and
// solve each direction for the count that meets exactly that. Three
// properties follow, and they are the whole reason this is safe to put under
// every surface in an application:
//
//   1. NOTHING GETS COARSER THAN IT ALREADY WAS. The direction that set the
//      tolerance solves back to its own declared count; every other
//      direction was already finer than the tolerance and can only shrink.
//      The mesh's worst chord deviation is therefore unchanged, by
//      construction.
//   2. THE COUNT NEVER RISES. Both results are clamped to the declared
//      counts, so no surface can cost more triangles than it costs today —
//      including under the aspect guard below. A ruled wall declared with
//      one division across stays at one division across, which is exact.
//   3. IT COMPOSES WITH A GLOBAL DENSITY MULTIPLIER. Applied AFTER that
//      multiplier has scaled the declared counts, doubling the multiplier
//      quarters the tolerance and doubles both solved counts — linear
//      density scaling, preserved exactly.
//
// CURVATURE FALLS OUT OF THE SAME MECHANISM RATHER THAN NEEDING A SECOND
// ONE. Equal deviation in both directions means cell edges in the ratio
// 1/sqrt(curvature), so a tight fillet gets shorter edges than a slack one
// of the same arc length, and a flat span collapses to a single division,
// with no separate curvature term and no second set of constants to tune.
//
// THE ASPECT GUARD is the one place arc length itself is read. Equal
// deviation does not imply a square cell: on a surface that is curved one
// way and straight the other, the straight direction solves to one division
// and the cell is as long as the surface. That is geometrically EXACT and
// shades exactly — a ruled quad is planar and its normal is constant along
// the ruling — but a cell tens of times longer than it is wide still yields
// slivers a tracer can find. So a direction is subdivided further, up to but
// never past its declared count, until no cell edge exceeds `maxCellAspect`
// times the other. On a ruled wall whose declared count across is already 1
// the clamp holds it at 1, which is why a ruled extrusion cannot be inflated
// by this.
//
// A SURFACE THAT IS ALREADY EXACT IS LEFT ALONE. When all three deviations
// sit at the numerical floor the patch is planar (or bilinear-flat) and the
// declared counts carry information this function does not have — a plane in
// a modelling application is resolved for the shape it is about to be
// deformed into, not the flat one it starts as — so the declared counts are
// returned untouched rather than collapsed to a single quad.
//
// Below `minCells` the mechanism is skipped entirely: a grid that small has
// no waste to reclaim, the probe would cost more than the mesh, and the
// per-object budgets that produce those grids are deliberate.
export function tessellationGridResolution(srf, uRes, vRes, opts = {}) {
  const minCells = opts.minCells == null ? 1024 : opts.minCells;
  const maxAspect = opts.maxCellAspect == null ? 3 : opts.maxCellAspect;
  const probeCells = opts.probe == null ? 16 : opts.probe;
  const out = (u, v, engaged, note) => ({ uRes: u, vRes: v, engaged, note });
  if (!Number.isFinite(uRes) || !Number.isFinite(vRes) || uRes < 1 || vRes < 1) return out(uRes, vRes, false, 'no declared grid to reallocate');
  if (uRes * vRes < minCells) return out(uRes, vRes, false, 'declared grid is below the reallocation floor');
  const uMin = srf.knotsU[srf.degU], uMax = srf.knotsU[srf.knotsU.length - 1 - srf.degU];
  const vMin = srf.knotsV[srf.degV], vMax = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
  const su = Math.max(2, Math.min(probeCells, Math.floor(uRes)));
  const sv = Math.max(2, Math.min(probeCells, Math.floor(vRes)));
  const P = [];
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i <= su; i++) {
    const u = uMin + (i / su) * (uMax - uMin);
    const row = [];
    for (let j = 0; j <= sv; j++) {
      const p = surfacePoint(srf, u, vMin + (j / sv) * (vMax - vMin));
      if (!Number.isFinite(p[0]) || !Number.isFinite(p[1]) || !Number.isFinite(p[2])) return out(uRes, vRes, false, 'probe hit a non-finite surface point');
      for (let k = 0; k < 3; k++) { if (p[k] < lo[k]) lo[k] = p[k]; if (p[k] > hi[k]) hi[k] = p[k]; }
      row.push(p);
    }
    P.push(row);
  }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  if (!(diag > 0)) return out(uRes, vRes, false, 'the probe collapsed to a point');
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const len = (a) => Math.hypot(a[0], a[1], a[2]);
  // The perpendicular distance of p1 from the chord p0->p2 — the sagitta the
  // mesh would carry if p1 were skipped and that chord drawn instead. That
  // chord spans TWO probe steps, and a sagitta scales as the square of its
  // chord, so the reading is divided by four to express it at ONE probe step —
  // the same unit the twist term below is already measured in, and the unit
  // every scaling here assumes.
  const sagitta = (p0, p1, p2) => {
    const c = sub(p2, p0), L = len(c);
    if (L < 1e-15) return len(sub(p1, p0)) / 4; // a closed or collapsed triple — the whole excursion is the error
    return len(cross(sub(p1, p0), c)) / L / 4;
  };
  // The distance of d from the plane through a, b, c — zero exactly when the
  // cell is planar, which is what makes a ruled direction free.
  const planeDev = (a, b, c, d) => {
    const n = cross(sub(b, a), sub(c, a)), L = len(n);
    if (L < 1e-15) return 0; // three collinear corners carry no plane to measure against
    const e = sub(d, a);
    return Math.abs((e[0] * n[0] + e[1] * n[1] + e[2] * n[2]) / L);
  };
  let devU = 0, devV = 0, devT = 0, lenU = 0, lenV = 0;
  for (let j = 0; j <= sv; j++) {
    let L = 0;
    for (let i = 0; i < su; i++) L += len(sub(P[i + 1][j], P[i][j]));
    if (L > lenU) lenU = L;
    for (let i = 1; i < su; i++) { const d = sagitta(P[i - 1][j], P[i][j], P[i + 1][j]); if (d > devU) devU = d; }
  }
  for (let i = 0; i <= su; i++) {
    let L = 0;
    for (let j = 0; j < sv; j++) L += len(sub(P[i][j + 1], P[i][j]));
    if (L > lenV) lenV = L;
    for (let j = 1; j < sv; j++) { const d = sagitta(P[i][j - 1], P[i][j], P[i][j + 1]); if (d > devV) devV = d; }
  }
  for (let i = 0; i < su; i++) for (let j = 0; j < sv; j++) {
    const d = planeDev(P[i][j], P[i + 1][j], P[i][j + 1], P[i + 1][j + 1]);
    if (d > devT) devT = d;
  }
  const flatEps = 1e-9 * diag;
  if (devU <= flatEps && devV <= flatEps && devT <= flatEps) return out(uRes, vRes, false, 'planar within tolerance — the declared grid carries the intent');

  /* THE TARGET DEVIATION, and why there are two ways to set it.
     ==================================================================
     By default the target is the DECLARED grid's own worst chord deviation, so
     this function only ever REDISTRIBUTES density between the two directions
     and can never call a grid finer than it needs to be. That is right when the
     declared grid means something -- the untrimmed path derives it from the
     surface, so it carries the caller's intent and is worth preserving.

     It is wrong when the declared grid is a CONSTANT. The trimmed path hands
     every piece a flat 64x64 with no relation to its size, and a self-
     referential target can only ever agree with it: a 5mm fillet band on a 90mm
     box edge solved back to 64x64 and cost 8192 triangles, of which the arc
     needed about 15. Measured on a filleted box, that was 110,592 of its
     116,320 triangles, and the cells came out 11,145:1 -- needles, which is
     what makes a fillet show its own tessellation under a shiny material.

     `relTolerance` sets the target instead as a fraction of arc length, PER
     DIRECTION. Per direction is the load-bearing part: a band's bounding box is
     dominated by its 90mm length while the curvature a viewer can see is the
     5mm arc, so one tolerance taken from the diagonal is set by the direction
     that is already straight. Each direction is allowed a sagitta proportional
     to its own extent, which is what makes the result scale-invariant.

     Neither mode may return a count HIGHER than the one handed in, so this
     stays a reduction in both. */
  const relTol = opts.relTolerance == null ? 0 : opts.relTolerance;
  const relative = relTol > 0 && lenU > 0 && lenV > 0;
  const tolU = relative ? lenU * relTol : Math.max(devU * (su / uRes) ** 2, devV * (sv / vRes) ** 2, devT * (su / uRes) * (sv / vRes));
  const tolV = relative ? lenV * relTol : tolU;
  const tolT = Math.min(tolU, tolV);
  const tol = tolU;
  if (!(tolU > 0 && tolV > 0)) return out(uRes, vRes, false, 'the declared grid already meets the surface exactly');
  let nu = devU > flatEps ? su * Math.sqrt(devU / tolU) : 1;
  let nv = devV > flatEps ? sv * Math.sqrt(devV / tolV) : 1;
  if (devT > flatEps) {
    const at = devT * (su / nu) * (sv / nv);
    if (at > tolT) { const s = Math.sqrt(at / tolT); nu *= s; nv *= s; }
  }
  nu = Math.max(1, Math.min(Math.round(uRes), Math.ceil(nu - 1e-9)));
  nv = Math.max(1, Math.min(Math.round(vRes), Math.ceil(nv - 1e-9)));
  // THE ASPECT GUARD, and the clamp is load-bearing: it is what keeps a ruled
  // wall declared at one division across from being inflated back up.
  if (maxAspect > 0 && lenU > 0 && lenV > 0) {
    const hu = lenU / nu, hv = lenV / nv;
    if (hu > maxAspect * hv) nu = Math.min(Math.round(uRes), Math.ceil(lenU / (maxAspect * hv)));
    else if (hv > maxAspect * hu) nv = Math.min(Math.round(vRes), Math.ceil(lenV / (maxAspect * hu)));
  }
  return { uRes: nu, vRes: nv, engaged: true,
    note: relative
      ? `solved for a chord deviation of ${relTol} of each direction's own arc length`
      : 'solved for equal chord deviation at the declared grid\'s own worst',
    devU, devV, devT, lenU, lenV, tol, probeU: su, probeV: sv };
}
