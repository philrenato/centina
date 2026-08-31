// CURVE / SURFACE INTERSECTION — every point where a NURBS curve crosses a
// NURBS surface, not just the nearest one.
//
// Named as its own primitive by both the trimming and the boolean specs
// (Patrikalakis & Maekawa
// 5.7). It is a strictly SMALLER problem than surface-surface intersection
// and worth having on its own terms: one parameter along the curve, two on
// the surface, three equations, three unknowns. Unlike SSI's own seed snap
// — which is 3 equations in 4 unknowns and therefore RANK-DEFICIENT along
// the intersection curve's own tangent, the reason that one needs
// Levenberg-Marquardt damping — this system is square and generically
// well-conditioned, so a plain Newton solves it.
//
// WHY IT EXISTS HERE: SSI seeds from a coarse grid over one surface and
// keeps whichever point is globally closest, so it finds ONE component. A
// boolean needs all of them. Sederberg & Meyers 1988 gives the completeness
// argument in two halves — every branch that reaches a patch BOUNDARY is
// found by intersecting that boundary against the opposing surface (this
// module), and a normal-cone test decides whether any branch could avoid
// every boundary by closing into an interior loop. This is the first half.

import { curvePoint, rationalCurveDerivs } from './curve.mjs';
import { surfacePoint, surfacePointAndPartials, closestPointOnSurface, surfaceClosure } from './surface.mjs';
import { sub, cross, dot, length, normalize } from './vec3.mjs';
import { solveSquareSystem } from './ssi.mjs';

function curveDomain(crv) {
  return { min: crv.knots[crv.degree], max: crv.knots[crv.knots.length - 1 - crv.degree] };
}
function surfaceDomain(srf) {
  return {
    uMin: srf.knotsU[0], uMax: srf.knotsU[srf.knotsU.length - 1],
    vMin: srf.knotsV[0], vMax: srf.knotsV[srf.knotsV.length - 1],
  };
}

// Signed side of the surface at the curve's own sample: positive on the
// side the surface normal points to, negative on the other. A crossing is a
// SIGN CHANGE between consecutive samples, which is what lets this find
// several crossings along one curve instead of only the closest approach.
//
// The sign is only meaningful when the closest point is genuinely INTERIOR
// to the surface — at a clamped boundary the "closest point" is an edge, and
// which side of the edge's normal the curve sits on says nothing about
// whether it pierces the patch. Those samples report `side: 0`, which
// brackets nothing, and the near-distance test below is what covers them.
//
// A CLOSED direction has no such edge. Its domain ends are one internal
// seam, so a sample landing there is as interior as any other and must keep
// its sign — found by fixture: a line through a full-revolve cylinder had
// its closest point land on the seam (v=0 on one side of the crossing,
// v=vMax on the other, the SAME place), which discarded the sign across the
// whole far half and left one of the two crossings unbracketed. Same
// distinction extractBorderCurves and SSI's own wrapParam already make.
function sampleSide(crv, srf, t, dom, closed) {
  const P = curvePoint(crv, t);
  const cp = closestPointOnSurface(srf, P);
  const edgeTol = 1e-9;
  const interior = (closed.closedU || (cp.u > dom.uMin + edgeTol && cp.u < dom.uMax - edgeTol))
    && (closed.closedV || (cp.v > dom.vMin + edgeTol && cp.v < dom.vMax - edgeTol));
  let side = 0;
  if (interior) {
    const sp = surfacePointAndPartials(srf, cp.u, cp.v);
    const n = cross(sp.su, sp.sv);
    if (length(n) > 1e-12) side = Math.sign(dot(sub(P, sp.point), normalize(n)));
  }
  return { t, u: cp.u, v: cp.v, distance: cp.distance, side, point: P };
}

// Newton on F(t,u,v) = C(t) - S(u,v) = 0. Columns of the Jacobian are
// dC/dt, -Su, -Sv. Rejects a step that makes the residual worse rather than
// letting it wander, the same guard closestPointOnSurface and SSI's own seed
// snap already use.
function refine(crv, srf, t, u, v, cd, sd) {
  let best = { t, u, v, residual: length(sub(curvePoint(crv, t), surfacePoint(srf, u, v))) };
  for (let iter = 0; iter < 40; iter++) {
    const [, ct] = rationalCurveDerivs(crv, best.t, 1);
    const sp = surfacePointAndPartials(srf, best.u, best.v);
    const r = sub(curvePoint(crv, best.t), sp.point);
    const cols = [ct, [-sp.su[0], -sp.su[1], -sp.su[2]], [-sp.sv[0], -sp.sv[1], -sp.sv[2]]];
    const rows = [0, 1, 2].map((eq) => cols.map((c) => c[eq]));
    const delta = solveSquareSystem(rows, r.map((x) => -x));
    if (!delta) break;
    const nt = Math.max(cd.min, Math.min(cd.max, best.t + delta[0]));
    const nu = Math.max(sd.uMin, Math.min(sd.uMax, best.u + delta[1]));
    const nv = Math.max(sd.vMin, Math.min(sd.vMax, best.v + delta[2]));
    const nr = length(sub(curvePoint(crv, nt), surfacePoint(srf, nu, nv)));
    if (nr > best.residual + 1e-12) break;
    best = { t: nt, u: nu, v: nv, residual: nr };
    if (nr < 1e-11) break;
  }
  return best;
}

// Every parameter where `crv` crosses `srf`, as
// `[{ t, u, v, point, residual }, ...]` sorted by t. `tolerance` is the
// distance below which a refined candidate counts as a genuine hit — a
// candidate that refines to a still-large residual is a near miss and is
// dropped rather than reported as an intersection.
export function curveSurfaceIntersections(crv, srf, opts = {}) {
  const cd = curveDomain(crv);
  const sd = surfaceDomain(srf);
  if (!(cd.max > cd.min)) return [];
  const scan = opts.scanSamples ?? Math.max(64, crv.ctrlPts.length * 8);
  const tolerance = opts.tolerance ?? 1e-6;
  // A hit is the same point as a previous one when their curve parameters
  // are within this fraction of the curve's own domain — parameter space,
  // not 3D space, so a curve that genuinely returns to the same spot in
  // space at a different parameter still reports both.
  const dedupeFrac = opts.dedupeFrac ?? 1e-4;

  const closed = surfaceClosure(srf);
  const samples = [];
  for (let i = 0; i <= scan; i++) samples.push(sampleSide(crv, srf, cd.min + (cd.max - cd.min) * (i / scan), sd, closed));

  const candidates = [];
  for (let i = 0; i < samples.length; i++) {
    const a = samples[i];
    if (a.distance < tolerance * 100) candidates.push(a); // already essentially on the surface
    const b = samples[i + 1];
    if (!b) continue;
    if (a.side !== 0 && b.side !== 0 && a.side !== b.side) {
      // Bracketed crossing — start the Newton from the nearer end, which is
      // inside the basin more often than the midpoint is.
      candidates.push(a.distance <= b.distance ? a : b);
    }
  }

  const hits = [];
  const tolSpan = (cd.max - cd.min) * dedupeFrac;
  for (const c of candidates) {
    const r = refine(crv, srf, c.t, c.u, c.v, cd, sd);
    if (r.residual > tolerance) continue;
    if (hits.some((h) => Math.abs(h.t - r.t) < tolSpan)) continue;
    hits.push({ t: r.t, u: r.u, v: r.v, residual: r.residual, point: curvePoint(crv, r.t) });
  }
  hits.sort((a, b) => a.t - b.t);
  return hits;
}
