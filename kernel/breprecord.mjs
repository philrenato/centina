// A SEWN SOLID, REDUCED TO PLAIN DATA A FILE WRITER CAN USE
// ================================================================
// The half-edge solid a boolean sews is the only thing that knows which faces
// SHARE which edge, and it is the one thing that cannot be kept: it is cyclic
// (a half-edge points at its twin, its next, and its vertex, and they point
// back), so it cannot be serialized, cloned, or carried through an autosave.
// This flattens it once, into ordinary arrays and numbers, while the topology
// is still available to be read.
//
// ⚠⚠ AND THE TOPOLOGY GENUINELY CANNOT BE RECOVERED LATER. Rebuilding it from
// the kept fragments by matching their boundaries in 3-D looks reasonable and
// does not work: each face's boundary is split at ITS OWN corners, found in ITS
// OWN parameters, and a corner in one face's (u,v) is not a corner in its
// neighbour's. Measured on the banked torus pair — of 24 runs across four
// fragments, 6 find a partner within 1e-2 and the other 18 have nothing closer
// than 95 units. The subdivision the two sides of an edge agree on exists only
// in the sew.
//
// WHAT THE RECORD IS FOR: ON_Brep has no join. A multi-face solid is ONE brep
// whose faces share their edges, so an exporter needs an edge list, a face
// list, and — per trim — which edge it runs along and in which direction. That
// is exactly this shape, and nothing in it is specific to .3dm.
import { fitSolidEdgeCurves, fitFaceLoops } from './brepfit.mjs';
import { surfacePoint } from './surface.mjs';

// ⚠⚠ A GAP IN A TRIM LOOP IS SOMETIMES A POLE, AND A POLE IS A REAL LOOP MEMBER.
//
// A revolved disc (a cylinder cap) collapses its whole u=0 edge to ONE point.
// The sew builds face boundaries from 3-D points and welds consecutive
// duplicates — correctly, since a run of identical points is not an edge — so
// the stretch along that pole vanishes from the half-edge structure and the
// face's loop arrives here with a hole in it. Measured on a box-and-cylinder
// union: a cap's loop carried (0,0)->(1,0), (1,0)->(1,1.3333),
// (1,1.3333)->(0,1.3333) and simply stopped, missing the run back along u=0.
//
// ON represents exactly this as a SINGULAR TRIM: a loop member with real (u,v)
// extent and no 3-D length, sitting on one side of the parametric rectangle. So
// the gap is not damage to be refused — it is a boundary the sew could not
// express, and it can be recovered here because the SURFACE still knows.
//
// The test is the honest one: walk the straight (u,v) path across the gap and
// ask whether the surface collapses to a single point along all of it. Anything
// else is a genuine hole and still refuses.
function poleAcrossGap(srf, fromUV, toUV, tolerance, samples = 8) {
  const p0 = surfacePoint(srf, fromUV[0], fromUV[1]);
  let worst = 0;
  for (let i = 1; i <= samples; i++) {
    const t = i / samples;
    const u = fromUV[0] + (toUV[0] - fromUV[0]) * t;
    const v = fromUV[1] + (toUV[1] - fromUV[1]) * t;
    const p = surfacePoint(srf, u, v);
    worst = Math.max(worst, Math.hypot(p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]));
  }
  if (worst > Math.max(tolerance, 1e-9)) return null;
  return { point: [p0[0], p0[1], p0[2]], deviation: worst };
}

// Which side of the parametric rectangle a degenerate stretch runs along, in
// ON_Surface::ISO terms (W=3, S=4, E=5, N=6). ON needs this because it cannot
// infer it, and a wrong answer is a trim that claims to be somewhere it is not.
// Returns null when the stretch is not on a domain edge at all — which would
// mean an interior degeneracy, something this kernel does not produce and
// should not pretend to describe.
function isoSideFor(srf, fromUV, toUV) {
  const uMin = srf.knotsU[0], uMax = srf.knotsU[srf.knotsU.length - 1];
  const vMin = srf.knotsV[0], vMax = srf.knotsV[srf.knotsV.length - 1];
  const uEps = (uMax - uMin) * 1e-6, vEps = (vMax - vMin) * 1e-6;
  const constU = Math.abs(fromUV[0] - toUV[0]) <= uEps;
  const constV = Math.abs(fromUV[1] - toUV[1]) <= vEps;
  if (constU) {
    if (Math.abs(fromUV[0] - uMin) <= uEps) return 3; // W_iso
    if (Math.abs(fromUV[0] - uMax) <= uEps) return 5; // E_iso
  }
  if (constV) {
    if (Math.abs(fromUV[1] - vMin) <= vEps) return 4; // S_iso
    if (Math.abs(fromUV[1] - vMax) <= vEps) return 6; // N_iso
  }
  return null;
}

function samePoint(a, b, tol) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) <= tol;
}

// Traverse the same point set from the other end — control point order and the
// knot vector flip together, re-based onto the same domain so span lookups stay
// valid (P&T's own reversal; kernel/curve.mjs states the identity in full).
// Named apart from `reverseCurve` because this returns a NEW curve and never
// touches the one it was given, which the fitted record depends on.
function reverseCurveInPlace(crv) {
  const { degree, knots, ctrlPts } = crv;
  const m = knots.length - 1;
  const a = knots[0], b = knots[m];
  const newKnots = [];
  for (let j = 0; j <= m; j++) newKnots.push(a + b - knots[m - j]);
  return { degree, knots: newKnots, ctrlPts: ctrlPts.slice().reverse().map((cp) => cp.slice()) };
}

// opts: { tolerance, vertexTolerance }
// Returns { ok, surfaces, edges, faces, stats } or { ok: false, reason }.
//
//   surfaces  [srf, ...]                       — plain NURBS surfaces, one per face
//   edges     [{ curve, start, end, closed, deviation }]
//   faces     [{ surfaceIndex, loops: [{ loopType, trims: [{ edgeIndex, reversed, curve }] }] }]
//
// Every curve is `{ degree, knots, ctrlPts }`; every point is `[x, y, z]`.
// There is not one object reference in the result, by construction — which is
// the property that makes it storable at all.
export function solidToBrepRecord(solid, opts = {}) {
  const tolerance = opts.tolerance ?? 1e-3;
  // ⚠ EVERY FIELD THE MERGE READS, not just the two that name the shape. The
  // first guard here checked `edges` and `shells` only, so a solid carrying
  // those and nothing else got past it and `fitSolidEdgeCurves` then threw on
  // `solid.vertices.length`. A refusal is a RETURN VALUE in this kernel; a
  // throw out of a flattening step reaches the app as an unexplained failure of
  // whatever called it.
  if (!solid || !Array.isArray(solid.edges) || !Array.isArray(solid.shells) || !Array.isArray(solid.vertices)) {
    return { ok: false, reason: 'solidToBrepRecord needs a sewn solid with edges, vertices and shells' };
  }
  // How close two chain ends must be to be the same vertex. Deliberately
  // looser than the fit bound: this is an identity question about points the
  // sew already welded, not an accuracy one.
  const vertexTolerance = opts.vertexTolerance ?? Math.max(tolerance, 1e-9) * 10;

  const fitted = fitSolidEdgeCurves(solid, { tolerance });
  if (!fitted || !Array.isArray(fitted.edges)) {
    return { ok: false, reason: `the solid's edges could not be merged: ${fitted && fitted.reason}` };
  }
  const loops = fitFaceLoops(solid, fitted, { tolerance });
  if (!loops || !Array.isArray(loops.faces)) {
    return { ok: false, reason: `the face loops could not be built: ${loops && loops.reason}` };
  }

  // ⚠ AN EDGE WITHOUT A CURVE IS NOT AN EDGE A FILE CAN CARRY, and dropping it
  // silently would leave the trims that run along it pointing at nothing. The
  // index is kept stable by emitting null and refusing any trim that lands on
  // one, so a partial fit degrades to a NAMED refusal rather than to a brep
  // with a hole in its topology.
  // ⚠⚠ THE EDGE CURVE MUST RUN THE WAY ITS CHAIN DOES, and it is not born that
  // way. A trim's `reversed` flag is computed against the CHAIN's traversal
  // order, so an edge curve running the other way makes every trim on it point
  // at the wrong end — and a B-rep validator reports that as a trim sitting a
  // full edge-length away from its own edge, which reads like a broken trim
  // rather than a backwards curve.
  //
  // It happens because `fitSolidEdgeCurves` does not ask for exact endpoints,
  // and a STRAIGHT chain is recognised by `fitLine`, which CANONICALIZES its
  // direction (largest component positive) so that near-identical input cannot
  // flicker between opposite directions. A chain travelling the other way
  // therefore comes back as a line running backwards. Curved chains are
  // unaffected, which is exactly why a torus pair never caught this and a box
  // — every edge of it straight — fails on the first one.
  const edges = fitted.edges.map((e) => {
    if (!e.curve) return null;
    const pts = e.points;
    const first = pts[0], last = pts[pts.length - 1];
    const cps = e.curve.ctrlPts;
    const c0 = cps[0], cN = cps[cps.length - 1];
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    // Only a genuinely reversed curve is turned — an ambiguous one (a closed
    // chain, whose ends are the same point) is left exactly as it is.
    const backwards = dist(c0, first) > dist(c0, last) && dist(cN, last) > dist(cN, first);
    const curve = backwards ? reverseCurveInPlace(e.curve) : e.curve;
    return {
      curve,
      // Read off the CURVE that will actually be written, not off the chain it
      // was fitted from: the vertex a B-rep hangs on the edge has to be where
      // the edge geometry really starts.
      start: [curve.ctrlPts[0][0], curve.ctrlPts[0][1], curve.ctrlPts[0][2]],
      end: [curve.ctrlPts[curve.ctrlPts.length - 1][0], curve.ctrlPts[curve.ctrlPts.length - 1][1], curve.ctrlPts[curve.ctrlPts.length - 1][2]],
      closed: !!e.closed || samePoint(first, last, vertexTolerance),
      deviation: e.maxDeviation ?? null,
    };
  });
  const unfittedEdges = edges.filter((e) => !e).length;
  // The reason the FIRST unfittable edge gave. An edge with no curve takes every
  // trim that runs along it out of its face's loop, so the loop then has a hole
  // the width of that edge — and the refusal downstream describes the hole while
  // this describes what made it.
  const firstEdgeFailure = (fitted.edges.find((e) => !e.curve) || {}).reason || null;

  const surfaces = [];
  const faces = [];
  let droppedTrims = 0, trimCount = 0, refusedFaces = 0, singularTrims = 0;
  let unclosedGaps = 0, worstUnclosed = 0;
  // ⚠ THE PER-FACE REASON IS THE ONLY USEFUL PART OF A TOTAL REFUSAL. Every
  // face failing for the same reason reads as "this does not work", and the
  // caller can do nothing with that; the reason `fitFaceLoops` gave for the
  // FIRST one is what names a way forward.
  let firstRefusal = null;
  for (const rec of loops.faces) {
    if (!rec.loops || !rec.loops.length) {
      refusedFaces++;
      if (!firstRefusal) {
        // ⚠ NAME THE SURFACE, not just the complaint. "Curve is not on the
        // surface" is true of a projection failure on any surface at all, and
        // a reader cannot tell a pole from a coarse net from a genuinely wrong
        // face without knowing what the surface IS. Degree and net size are
        // the two numbers that separate those cases, and they are free here.
        const s = rec.srf;
        const shape = s
          ? ` [face surface: degree ${s.degU}x${s.degV}, net ${s.ctrlNet.length}x${s.ctrlNet[0].length}]`
          : '';
        firstRefusal = (rec.reason || (s ? 'the face produced no loop' : 'the face carries no surface')) + shape;
      }
    }
    if (!rec.srf || !rec.loops || !rec.loops.length) continue;
    const surfaceIndex = surfaces.length;
    surfaces.push(rec.srf);
    const outLoops = [];
    for (const loop of rec.loops) {
      const trims = [];
      const sourceUV = [];
      for (const t of loop.trims) {
        if (t.chainIndex == null || !edges[t.chainIndex] || !t.curve) { droppedTrims++; continue; }
        trims.push({ edgeIndex: t.chainIndex, reversed: !!t.reversed, curve: t.curve });
        sourceUV.push(t.uv);   // the (u,v) the pole check needs; NOT carried into the record itself, which stays minimal
        trimCount++;
      }
      // CLOSE THE LOOP WITH ITS POLES before anything downstream judges it. A
      // gap that the surface itself collapses across is a SINGULAR trim the sew
      // could not carry, not a defect — recovered here while the surface is
      // still in hand, and left alone otherwise so a real hole still refuses.
      const closed = [];
      for (let i = 0; i < trims.length; i++) {
        closed.push(trims[i]);
        const mine = sourceUV[i], nextUV = sourceUV[(i + 1) % trims.length];
        if (!mine || !nextUV) continue;
        const endUV = mine[mine.length - 1];
        const startUV = nextUV[0];
        const gap = Math.hypot(endUV[0] - startUV[0], endUV[1] - startUV[1]);
        if (gap <= 1e-9) continue;
        const pole = poleAcrossGap(rec.srf, endUV, startUV, tolerance);
        const iso = pole ? isoSideFor(rec.srf, endUV, startUV) : null;
        if (!pole || iso == null) {
          // A genuine hole: the surface does NOT collapse across it, so no
          // singular trim can honestly describe it. Counted, and named below.
          unclosedGaps++;
          worstUnclosed = Math.max(worstUnclosed, gap);
          continue;
        }
        closed.push({
          singular: true,
          iso,
          point: pole.point,
          // A straight run along the domain edge, which is exactly what the
          // stretch is: the pcurve is real even though the 3-D edge is a point.
          curve: { degree: 1, knots: [0, 0, 1, 1], ctrlPts: [[endUV[0], endUV[1], 0, 1], [startUV[0], startUV[1], 0, 1]] },
          deviation: pole.deviation,
        });
        singularTrims++;
      }
      if (closed.length) outLoops.push({ loopType: loop.loopType === 'inner' ? 'inner' : 'outer', trims: closed });
    }
    if (!outLoops.length) { surfaces.pop(); continue; }
    faces.push({ surfaceIndex, loops: outLoops });
  }

  if (!faces.length) {
    return {
      ok: false,
      reason: `no face of the solid produced a usable trim loop (${refusedFaces} of ${loops.faces.length} refused`
        + (firstRefusal ? `; the first says: ${firstRefusal}` : '') + ')',
    };
  }

  // ⚠⚠ A PARTIAL JOIN IS WORSE THAN NO JOIN, and this is the refusal that
  // matters most here. A brep missing even one face is still VALID by
  // OpenNURBS' own check — it is simply open, and an open brep looks exactly
  // like a solid on screen while failing everything a solid is for: another
  // boolean, a shell, an STL, a volume. Measured: a box-and-cylinder union
  // whose record dropped 3 of 13 faces assembled into one brep that Rhino
  // reported as OPEN, with nothing anywhere naming the missing faces.
  //
  // So an incomplete record refuses, and the caller falls back to exporting
  // the loose trimmed faces — which is worse geometry honestly labelled,
  // rather than a solid that is quietly not one.
  // ⚠⚠ A DROPPED DEGENERATE TRIM LEAVES A HOLE THIS CODE CANNOT SEE.
  // `fitFaceLoops` removes a trim whose (u,v) run collapses to a point. When the
  // 3-D run collapses with it that is a POLE, and a cylinder cap (a revolved
  // disc) has one at its centre. Dropping it is right for the fitter and wrong
  // for a B-rep: ON represents a pole as a SINGULAR TRIM, a real member of the
  // loop with no 3-D length, and a loop that simply omits it does not close.
  // OpenNURBS says "brep.m_L[10] loop is not valid" and nothing upstream had
  // noticed.
  //
  // Those drops happen inside fitFaceLoops, so they never reach `droppedTrims`
  // below and the record arrives looking complete. Counted from its own stats
  // instead, and refused — writing a loop with a hole is the "valid but quietly
  // wrong" outcome this whole path exists to avoid.
  //
  // ⚠ THE FIX IS A SINGULAR TRIM, AND THIS MODULE ALREADY AUTHORS ONE — the
  // pole recovery above emits `singular: true` members and `brepRecordToRhino`
  // writes them through `ON_Brep::NewSingularTrim`, which
  // vendor/rhino3dm/brep_authoring.patch binds. What it cannot recover is a
  // trim the FITTER already removed. That recovery reads the (u,v) gap between
  // the trims that SURVIVED, and it is the two thresholds that make it blind
  // here: `fitFaceLoops` drops a run spanning 1e-12 or less, and the recovery
  // ignores a gap of 1e-9 or less. A dropped run is three orders of magnitude
  // inside that floor, so the hole it leaves is indistinguishable from a loop
  // that genuinely closes, and there is nothing left to author a singular trim
  // across. The solid refuses here and exports as loose trimmed faces.
  //
  // ⚠ AND THE TWO COUNTS ARE NOT THE SAME DEFECT. `degenerateTrimsDropped` is a
  // run that collapses in 3-D as well — a genuine pole. `collapsedInUV` has real
  // 3-D length and squashes only in parameters, so it is NOT a pole and no
  // singular trim would describe it honestly; `fitFaceLoops` measures the
  // separation and its own comment names the two suspects still unresolved.
  const poleDrops = (loops.stats && (loops.stats.degenerateTrimsDropped || 0));
  const uvDrops = (loops.stats && (loops.stats.collapsedInUV || 0));
  const lostTrims = poleDrops + uvDrops;
  if (lostTrims > 0) {
    return {
      ok: false,
      reason: `${lostTrims} trim(s) were removed by the loop fitter for collapsing in (u,v) `
        + `(${poleDrops} collapse in 3-D as well, which is a POLE; ${uvDrops} keep real 3-D length and are not), `
        + 'so a loop is missing a boundary the face has. The pole recovery here works from the gap between the '
        + 'trims that survived, and a collapsed run leaves no measurable gap, so there is nothing left to author '
        + 'a singular trim across. Exported as loose trimmed faces instead of a solid with a hole in it.',
    };
  }

  // ⚠⚠ A LOOP THAT DOES NOT CLOSE CANNOT BECOME A BREP, and `fitFaceLoops`
  // already measures exactly that — `worstLoopJoin` is the largest (u,v) gap
  // between one trim's end and the next one's start. Handing a broken loop to
  // OpenNURBS produces "brep.m_L[14] loop is not valid ... end of m_T[72]=
  // (1.1e-16, 1.33333) and start of m_T[70]=(0,0) do not match", which is a
  // true statement about a defect three layers upstream and reads like a bug in
  // the writer.
  //
  // ⚠ THE 1.33333 IS A KNOWN, OPEN KERNEL DEFECT, not a pole and not anything
  // this module can repair: `splitAtSeam`'s own internal cut line at v = span/3
  // emits a loop that is missing the stretch that cut line should have
  // contributed, and 1.33333 is that cut line's own v. So
  // this refuses by name and the caller exports loose trimmed faces — the same
  // geometry, honestly labelled — instead of a solid with a hole in its
  // topology.
  // ⚠⚠ JUDGED ON WHAT WAS ACTUALLY BUILT, not on the upstream stat. The first
  // version refused using `fitFaceLoops`'s own `loopJoinBreaks`, which is
  // measured BEFORE the pole recovery above — so a loop this module had just
  // closed correctly was still reported as broken. A check that reads a number
  // computed before the fix cannot see the fix.
  if (unclosedGaps > 0) {
    return {
      ok: false,
      reason: `${unclosedGaps} trim loop gap(s) could not be closed — the worst is ${worstUnclosed.toPrecision(6)} in (u,v), `
        + (unfittedEdges
          ? `and ${unfittedEdges} edge(s) of ${edges.length} could not be fitted at all, which is what takes their trims out of the loop${firstEdgeFailure ? ` — the first says: ${String(firstEdgeFailure).slice(0, 200)}` : ''}. `
          : `${droppedTrims} trim(s) were dropped for want of an edge. `)
        + 'and the surface does NOT collapse across it, so it is a real hole rather than a pole. '
        + 'A loop with a gap cannot be a Brep face, so this is refused here rather than written and rejected by OpenNURBS.',
    };
  }

  if (refusedFaces > 0 || droppedTrims > 0) {
    return {
      ok: false,
      reason: `the solid would be incomplete — ${faces.length} of ${loops.faces.length} faces usable`
        + (refusedFaces ? `, ${refusedFaces} refused` : '')
        + (droppedTrims ? `, ${droppedTrims} trims dropped` : '')
        + (firstRefusal ? `; the first says: ${firstRefusal}` : '')
        + '. A brep missing a face is valid but OPEN, which is not a solid.',
    };
  }

  return {
    ok: true,
    surfaces,
    edges,
    faces,
    stats: {
      sourceEdges: solid.edges.length,
      edges: edges.length,
      unfittedEdges,
      faces: faces.length,
      refusedFaces,
      trims: trimCount,
      singularTrims,
      droppedTrims,
      worstEdgeDeviation: fitted.stats ? fitted.stats.worstDeviation ?? null : null,
      worstLoopJoin: loops.stats ? loops.stats.worstLoopJoin ?? null : null,
    },
  };
}

// A record is geometry in world coordinates, so a container carrying a rigid
// mesh transform has to bake it before the record can be written — the same
// affine-invariance identity the boolean and .3dm export paths already rely on
// (a NURBS point is an affine combination of its own control points, and an
// affine map commutes with an affine combination; the WEIGHT is untouched and
// only the euclidean position moves).
//
// ⚠ THE PCURVES ARE NOT TRANSFORMED, and must not be: a trim lives in its own
// face's PARAMETERS, which a rigid motion of the face does not change. Moving
// them would shift every boundary within its own surface — a file that opens
// and is quietly the wrong shape.
export function transformBrepRecord(record, applyToPoint) {
  if (!record || !record.ok) return record;
  const cp = (p) => { const q = applyToPoint([p[0], p[1], p[2]]); return p.length > 3 ? [q[0], q[1], q[2], p[3]] : [q[0], q[1], q[2]]; };
  return {
    ...record,
    surfaces: record.surfaces.map((s) => ({
      degU: s.degU, degV: s.degV,
      knotsU: [...s.knotsU], knotsV: [...s.knotsV],
      ctrlNet: s.ctrlNet.map((row) => row.map(cp)),
    })),
    edges: record.edges.map((e) => (e ? {
      ...e,
      curve: { degree: e.curve.degree, knots: [...e.curve.knots], ctrlPts: e.curve.ctrlPts.map(cp) },
      start: applyToPoint(e.start),
      end: applyToPoint(e.end),
    } : null)),
    faces: record.faces.map((f) => ({
      surfaceIndex: f.surfaceIndex,
      loops: f.loops.map((l) => ({
        loopType: l.loopType,
        trims: l.trims.map((t) => ({ edgeIndex: t.edgeIndex, reversed: t.reversed, curve: t.curve })),
      })),
    })),
  };
}
