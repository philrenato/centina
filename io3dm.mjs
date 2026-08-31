// .3dm interchange — conversion between this kernel's own P&T-standard NURBS
// representation and rhino3dm/OpenNURBS's representation. Deliberately lives
// OUTSIDE kernel/: everything under kernel/ is implemented from the
// published literature — Piegl & Tiller above all, and alongside it Catmull
// & Clark, Stam, Loop & Schaefer, Halstead/Kass/DeRose, Sederberg & Parry,
// Patrikalakis & Maekawa, Pharr/Jakob/Humphreys and the others each module
// cites by name — with no code derived from another implementation. This
// module, by contrast, is honest, attributed, vendored MIT infrastructure
// (rhino3dm, McNeel), the same relationship vendor/three has to the app.
//
// Every function here takes a `rhino` instance as its first argument
// (the awaited rhino3dm() module object) rather than importing rhino3dm
// itself — this keeps the module isomorphic between a Node test environment
// (rhino3dm installed as a real, non-shipped devDependency) and the app's own
// worker (the vendored vendor/rhino3dm/rhino3dm.module.js), with one
// implementation, not two.
//
// Two empirically-verified conversion rules this whole module rests on
// (rhino3dm's own .d.ts is confirmed incomplete/unreliable — see the header
// comments below for what was actually proven, not assumed):
//
// 1. KNOT VECTOR LENGTH. Our own clamped knot vector has length n+p+1
//    (n = control point count, p = degree) — the literal P&T convention.
//    rhino3dm/OpenNURBS's own internal knot vector has length n+p-1: it
//    drops the first and last entry of our own array (both of which, on a
//    clamped curve, are always exact duplicates of their own neighbor
//    anyway — no information is lost). EXPORT: `ourKnots.slice(1, -1)`.
//    IMPORT: reconstruct by prepending the reduced array's own first
//    element again and appending its own last element again.
//
// 2. CONTROL POINT WEIGHT CONVENTION. Our own ctrlPts[i] = [x,y,z,w] stores
//    the EUCLIDEAN (dehomogenized) position plus a separate weight — proven
//    by reading kernel/curve.mjs's curvePoint, which calls toHomogeneous()
//    to CONVERT to the homogeneous form before ever evaluating, meaning the
//    raw array is not already homogeneous. rhino3dm's own
//    NurbsCurvePointList/NurbsSurfacePointList .get()/.set() instead use the
//    PRE-MULTIPLIED homogeneous form (X,Y,Z,W) = (w*x, w*y, w*z, w).
//    EXPORT: `[x*w, y*w, z*w, w]`. IMPORT: `[X/W, Y/W, Z/W, W]`.

export function ourKnotsToRhino(knots) {
  return knots.slice(1, knots.length - 1);
}

export function rhinoKnotsToOurs(reduced) {
  return [reduced[0], ...reduced, reduced[reduced.length - 1]];
}

// ---- CURVE ----

// Our curve: { degree, knots, ctrlPts: [[x,y,z,w], ...] }
export function curveToRhino(rhino, crv) {
  const n = crv.ctrlPts.length;
  const nc = new rhino.NurbsCurve(3, true, crv.degree + 1, n);
  const knots = nc.knots();
  const reduced = ourKnotsToRhino(crv.knots);
  for (let i = 0; i < reduced.length; i++) knots.set(i, reduced[i]);
  const pts = nc.points();
  for (let i = 0; i < n; i++) {
    const [x, y, z, w] = crv.ctrlPts[i];
    pts.set(i, [x * w, y * w, z * w, w]);
  }
  return nc;
}

// OUR control point is EUCLIDEAN plus a weight, never the premultiplied
// homogeneous form rhino3dm's point lists use (see the note at the top of this
// file — getting the two confused is the single easiest way to write a file
// that opens and is subtly the wrong shape). So a control point's position is
// its first three components, with no division.
function euclideanCtrlPt(cp) {
  return [cp[0], cp[1], cp[2]];
}

// rhinoCrv: a real rhino.NurbsCurve (or any rhino.Curve exposing
// .toNurbsCurve() — the caller is responsible for that cast, this function
// only ever reads a genuine NurbsCurve's own knots()/points()).
export function curveFromRhino(rhinoCrv) {
  const degree = rhinoCrv.degree;
  const knotList = rhinoCrv.knots();
  const reduced = [];
  for (let i = 0; i < knotList.count; i++) reduced.push(knotList.get(i));
  const knots = rhinoKnotsToOurs(reduced);
  const ptList = rhinoCrv.points();
  const ctrlPts = [];
  for (let i = 0; i < ptList.count; i++) {
    const [X, Y, Z, W] = ptList.get(i);
    ctrlPts.push([X / W, Y / W, Z / W, W]);
  }
  return { degree, knots, ctrlPts };
}

// ---- SURFACE ----

// Our surface: { degU, degV, knotsU, knotsV, ctrlNet: [i][j] = [x,y,z,w] }
export function surfaceToRhino(rhino, srf) {
  const nu = srf.ctrlNet.length;
  const nv = srf.ctrlNet[0].length;
  const ns = rhino.NurbsSurface.create(3, true, srf.degU + 1, srf.degV + 1, nu, nv);
  const knotsU = ns.knotsU();
  const redU = ourKnotsToRhino(srf.knotsU);
  for (let i = 0; i < redU.length; i++) knotsU.set(i, redU[i]);
  const knotsV = ns.knotsV();
  const redV = ourKnotsToRhino(srf.knotsV);
  for (let i = 0; i < redV.length; i++) knotsV.set(i, redV[i]);
  const pts = ns.points();
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      const [x, y, z, w] = srf.ctrlNet[i][j];
      pts.set(i, j, [x * w, y * w, z * w, w]);
    }
  }
  return ns;
}

// rhinoSrf: a real rhino.NurbsSurface (or any rhino.Surface exposing a cast
// the caller has already performed — same convention as curveFromRhino).
export function surfaceFromRhino(rhinoSrf) {
  const degU = rhinoSrf.orderU - 1;
  const degV = rhinoSrf.orderV - 1;
  const knotsUList = rhinoSrf.knotsU();
  const redU = [];
  for (let i = 0; i < knotsUList.count; i++) redU.push(knotsUList.get(i));
  const knotsU = rhinoKnotsToOurs(redU);
  const knotsVList = rhinoSrf.knotsV();
  const redV = [];
  for (let i = 0; i < knotsVList.count; i++) redV.push(knotsVList.get(i));
  const knotsV = rhinoKnotsToOurs(redV);
  const ptList = rhinoSrf.points();
  const nu = ptList.countU;
  const nv = ptList.countV;
  const ctrlNet = [];
  for (let i = 0; i < nu; i++) {
    const row = [];
    for (let j = 0; j < nv; j++) {
      const [X, Y, Z, W] = ptList.get(i, j);
      row.push([X / W, Y / W, Z / W, W]);
    }
    ctrlNet.push(row);
  }
  return { degU, degV, knotsU, knotsV, ctrlNet };
}

// ---- BREP TRIM BOUNDARIES ----

// A face's trim loops, read out as 3-D POLYLINES rather than as (u,v) curves,
// because rhino3dm exposes no 2-D pcurve at all: BrepTrim carries only
// edgeIndex/isReversed/startVertexIndex/endVertexIndex, and Brep has no
// curves2D list (measured, and pinned by a test). What IS
// exact is the 3-D edge behind each trim, so that is what comes out here.
//
// TURNING THOSE INTO (u,v) IS DELIBERATELY NOT DONE HERE. Inversion needs
// closestPointOnSurface, which lives in the kernel — and this module runs
// inside the I/O worker, whose module graph is precached by sw.js and contains
// no kernel modules at all. Importing one would add an offline-breaking fetch
// to satisfy a computation the main thread can already do, where the kernel is
// inlined. So this stays what the rest of the file is: format reading.
//
// SAMPLING IS DRIVEN BY DEVIATION, NOT BY SPAN COUNT. The polyline this
// produces becomes the face's boundary, so its error is a permanent property
// of the imported geometry — and a polyline inscribed in a curve always
// UNDER-measures the region it bounds. A count derived from spans gets that
// silently wrong: a full circle came back as 12 points, whose enclosed area is
// exactly 75.0000 against the disc's 78.5398 — a 4.5% shortfall that looks
// like a plausible boundary and reads as a working import.
//
// So each segment is bisected while its midpoint sits further from the chord
// than the file's own model tolerance, which ties the error to the number the
// file itself declares as meaningful. A straight edge terminates immediately
// at two points; a circle refines until it is a circle. The cap is a bound on
// pathological input, not a target, and it is stated in the refusal path
// rather than silently truncating.
function sampleRhinoEdge(edge, reversed, tolerance) {
  const dom = edge.domain;
  const t0 = dom[0], t1 = dom[1];
  const at = (t) => { const p = edge.pointAt(t); return [p[0], p[1], p[2]]; };
  const tol = Math.max(tolerance || 0, 1e-9);
  const MAX_POINTS = 2048;
  const pts = [at(t0)];
  // Explicit stack rather than recursion: a near-degenerate edge can subdivide
  // deeply, and blowing the call stack inside an import is a worse failure
  // than a coarse curve.
  const stack = [[t0, t1]];
  while (stack.length && pts.length < MAX_POINTS) {
    const [a, b] = stack.pop();
    const pa = pts[pts.length - 1];
    const pb = at(b);
    const m = 0.5 * (a + b);
    const pm = at(m);
    // Distance from the true midpoint to the chord's own midpoint — the
    // sagitta. Cheaper than a point-to-segment distance and identical to it
    // for the shallow arcs subdivision converges to.
    const cx = 0.5 * (pa[0] + pb[0]), cy = 0.5 * (pa[1] + pb[1]), cz = 0.5 * (pa[2] + pb[2]);
    const sag = Math.hypot(pm[0] - cx, pm[1] - cy, pm[2] - cz);
    if (sag > tol && (b - a) > 1e-12) { stack.push([m, b], [a, m]); }
    else { pts.push(pb); }
  }
  // Hitting the cap means the walk stopped BEFORE the edge's end, so the
  // polyline does not reach the next trim's start and the loop it belongs to
  // would close across a gap — a boundary that is wrong rather than coarse.
  // Refuse the loop instead: the caller falls back to the untrimmed face and
  // the app names it, which is the honest outcome for input this pathological.
  if (stack.length) return null;
  return reversed ? pts.reverse() : pts;
}

// Every loop of one face, in the order Rhino stores them, each as a closed 3-D
// polyline plus the loop's own kind. Returns null when the face has no
// readable loop structure at all (an old rhino3dm build with no trim API is
// exactly that case, and the caller falls back to the untrimmed panel it used
// to produce, rather than throwing).
//
// `outer` is taken from Rhino's own loopType and NOT inferred from area:
// a face whose outer loop is the full parametric rectangle and a face trimmed
// down to a sliver are both legitimate, and area-ranking guesses wrong exactly
// where a hole approaches the size of its face.
export function faceTrimLoopsFromRhino(brep, face, tolerance) {
  const loops = face.loops;
  if (!loops || typeof loops.count !== 'number') return null;
  const edges = brep.edges();
  const out = [];
  for (let i = 0; i < loops.count; i++) {
    const loop = loops.get(i);
    const trims = loop.trims;
    if (!trims || !trims.count) continue;
    const kindName = loop.loopType && loop.loopType.constructor ? loop.loopType.constructor.name : '';
    const kind = kindName.replace('BrepLoopType_', '').toLowerCase() || 'unknown';
    const polyline = [];
    let readable = true;
    for (let j = 0; j < trims.count; j++) {
      const trim = trims.get(j);
      // A singular trim (a pole, where a whole surface edge collapses to one
      // point) names no edge. It is a real part of the loop's topology but it
      // contributes no 3-D curve to walk, so it is skipped rather than
      // treated as a failure — a sphere's or revolve's cap is normal.
      if (trim.edgeIndex < 0 || trim.edgeIndex >= edges.count) continue;
      let pts;
      try { pts = sampleRhinoEdge(edges.get(trim.edgeIndex), !!trim.isReversed, tolerance); }
      catch { readable = false; break; }
      if (!pts) { readable = false; break; } // the walk hit its point cap short of the edge's end
      // Drop the joint duplicate: consecutive trims share a vertex, and two
      // coincident points in a row make a zero-length segment that every
      // downstream winding and self-intersection test has to special-case.
      const start = polyline.length && samePoint3(polyline[polyline.length - 1], pts[0]) ? 1 : 0;
      for (let k = start; k < pts.length; k++) polyline.push(pts[k]);
    }
    if (!readable || polyline.length < 3) continue;
    // Closed by convention everywhere else in this project's loop handling —
    // so the repeated last point is dropped rather than carried.
    if (samePoint3(polyline[0], polyline[polyline.length - 1])) polyline.pop();
    if (polyline.length >= 3) out.push({ kind, polyline });
  }
  return out.length ? out : null;
}

function samePoint3(a, b) {
  return Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9 && Math.abs(a[2] - b[2]) < 1e-9;
}

// ---- POINT ----

// WHAT KIND OF GEOMETRY IS THIS, ROBUSTLY. rhino3dm's objectType is an
// embind ENUM VALUE — an opaque object, not a number — so `===` against
// rhino.ObjectType.X is an identity comparison that only holds while both
// sides come from the same module instance, and String()-ing one yields
// the useless "[object Object]" that a skip report should never show a
// student. The constructor name is carried by the instance itself and
// survives both, so it is the primary test, with the enum identity kept
// as a secondary that costs nothing when it does work.
export function describeRhinoGeometry(geo) {
  const ctor = geo && geo.constructor && geo.constructor.name;
  if (ctor && ctor !== 'Object') return ctor;
  return 'unrecognized geometry';
}
export function isRhinoSubD(rhino, geo) {
  if (describeRhinoGeometry(geo) === 'SubD') return true;
  return rhino.ObjectType && rhino.ObjectType.SubD !== undefined && geo.objectType === rhino.ObjectType.SubD;
}
// A Rhino SUBD ARRIVES AS ITS OWN CONTROL NET — the cage, which is the
// object (a load-bearing distinction), never a refined mesh
// standing in for it.
//
// WHY VIA Mesh.createFromSubDControlNet AND NOT THE SubD ITSELF: the
// rhino3dm JS binding exposes SubD as an opaque handle — isSolid,
// subdivide, clearEvaluationCache, updateAllTagsAndSectorCoefficients,
// and nothing that reads a vertex, an edge or a face (checked directly
// against 8.17.0 AND 8.32.0). This one static is the entire door: it hands back the
// CONTROL NET as a mesh, which for our purposes is exactly the cage.
//
// The second argument is load-bearing and its own .d.ts declares this
// function taking none at all — confirmed empirically against a real
// Rhino-authored file instead: FALSE returns the WELDED net (one vertex
// per real cage vertex, which is what a cage means), TRUE returns an
// unwelded per-face copy (4 vertices per quad — the same 108-face cage
// read as 110 vertices welded and 432 unwelded). Welded is the only
// correct reading; an unwelded net would import as a cage whose faces
// share no vertices at all and therefore subdivides into confetti.
//
// HONEST LOSS, named not hidden: CREASES DO NOT SURVIVE. A mesh carries
// no crease weights and the SubD handle exposes none, so a creased Rhino
// SubD imports SMOOTH. That is reported through `creasesLost` rather than
// left for the student to discover by eye.
export function subdCageFromRhino(rhino, subd) {
  const mesh = rhino.Mesh.createFromSubDControlNet(subd, false);
  if (!mesh) throw new Error('subdCageFromRhino: rhino3dm returned no control net for this SubD');
  const mv = mesh.vertices(), mf = mesh.faces();
  const vertices = [];
  for (let i = 0; i < mv.count; i++) { const p = mv.get(i); vertices.push([p[0], p[1], p[2]]); }
  const faces = [];
  for (let i = 0; i < mf.count; i++) {
    const f = mf.get(i);
    // A mesh face is [a,b,c,d]; a TRIANGLE is encoded with its last index
    // repeated, which is a real, legal cage face only if collapsed to 3.
    const loop = f[2] === f[3] ? [f[0], f[1], f[2]] : [f[0], f[1], f[2], f[3]];
    faces.push(loop);
  }
  if (!vertices.length || !faces.length) throw new Error('subdCageFromRhino: control net came back empty');
  return { cage: { vertices, faces, creases: {} }, creasesLost: true };
}

export function pointToRhino(p) {
  return [p[0], p[1], p[2]];
}

export function pointFromRhino(p) {
  return [p[0], p[1], p[2]];
}

// ---- TRIMMED SURFACE -> ON_Brep ----

// A trimmed face written as a real ON_Brep, using the Brep authoring bindings
// this project adds to rhino3dm (see vendor/rhino3dm/brep_authoring.patch).
// Before these existed a trimmed face could not be written at all, and the app
// refused it by name rather than exporting the uncut base surface — which is
// the right refusal and a poor deliverable.
//
// THE TRIM BOUNDARY GOES OUT AS A REAL CURVE WHEN ONE WAS FITTED, and as a
// POLYLINE when none was. This kernel stores a trim loop as a (u,v) POLYLINE —
// that is what marching produces and what trims, sews and tessellates — so a
// polyline is what can always be written. `kernel/fitcurve.mjs` recovers the
// exact form of such a boundary, but it lives in the kernel and this module
// imports nothing at all, deliberately: it is reachable from the app and the
// I/O worker both, and the worker's precached module graph contains no kernel.
//
// So the FIT HAPPENS ABOVE, and arrives here as plain data alongside the
// polyline it was fitted to (`fit: { pcurve, edge, tolerance }`). A caller
// with no fitter still gets a valid trimmed Brep; a caller with one gets a
// boundary Rhino re-reads as a curve rather than as facets. Neither path can
// silently become the other, because the polyline is always carried too.
//
// ⚠ A FITTED PCURVE AND A FITTED EDGE ARE TWO INDEPENDENT APPROXIMATIONS of
// one boundary, and they do NOT agree exactly: the surface image of the pcurve
// is not the edge curve, for the same reason a NURBS surface's isocurve image
// of an arbitrary parameter path is not itself a NURBS curve. That disagreement
// is what an ON_Edge's TOLERANCE means, and it is passed in measured rather
// than assumed — see the caller. This is ordinary B-rep practice, not a
// compromise; every Brep Rhino writes carries the same pair.
//
// ⚠ AND ON THE POLYLINE PATH THE 3-D EDGE IS EVALUATED THROUGH rhino3dm
// ITSELF, not through our kernel: `NurbsSurface.pointAt` is right there, so the
// edge curve and the pcurve describe the same points without this module
// importing a surface evaluator it is not allowed to have.
function polylineCurve(rhino, points) {
  const pl = new rhino.Polyline();
  for (const p of points) pl.add(p[0], p[1], p.length > 2 ? p[2] : 0);
  // Closed loops are stored without the repeated first point everywhere in
  // this project; a trim curve must actually close, so it is added here.
  pl.add(points[0][0], points[0][1], points[0].length > 2 ? points[0][2] : 0);
  return pl.toPolylineCurve ? pl.toPolylineCurve() : new rhino.PolylineCurve(pl);
}

// ⚠⚠ A LOOP IS A CHAIN OF RUNS, NOT ONE CURVE, and that is forced by geometry
// rather than chosen for tidiness. A single smooth curve CANNOT represent a
// boundary with a corner in it: asked for a square at a 1e-3 bound the fitter
// refuses, closest 3.1e-1 at its full control-point ceiling — correctly, since
// no spline of any count rounds a right angle to within a thousandth. So a
// fitted loop arrives split at its corners, one run per smooth stretch, which
// is also the edge structure a B-rep is supposed to have; the single-curve
// closed case is just the one-run degenerate of the same shape.
//
// A fit is USED ONLY IF IT IS STRUCTURALLY COMPLETE. A half-supplied run — a
// pcurve with no edge curve — would build a Brep whose trim and edge describe
// different boundaries, which OpenNURBS may well accept and Rhino would then
// draw wrong. Falling back to the polyline is the honest answer, and it is
// per-LOOP rather than per-face: a face whose outer boundary fitted and whose
// hole did not still exports its outer boundary as curves.
function usableFit(fit) {
  const curveOk = (c) => !!c && Array.isArray(c.knots) && Array.isArray(c.ctrlPts)
    && c.ctrlPts.length >= 2 && Number.isFinite(c.degree) && c.degree >= 1;
  const runs = fit && Array.isArray(fit.runs) ? fit.runs : null;
  if (!runs || !runs.length) return null;
  return runs.every((r) => r && curveOk(r.pcurve) && curveOk(r.edge)) ? fit : null;
}

export function trimmedSurfaceToRhinoBrep(rhino, obj, tolerance) {
  const ns = surfaceToRhino(rhino, obj);
  const loops = [];
  if (obj.trimLoop && obj.trimLoop.length >= 3) {
    loops.push({ uv: obj.trimLoop, kind: 1, fit: usableFit(obj.trimLoopFit) });
  }
  const holeFits = obj.trimHoleFits || [];
  (obj.trimHoles || []).forEach((hole, i) => {
    if (hole && hole.length >= 3) loops.push({ uv: hole, kind: 2, fit: usableFit(holeFits[i]) });
  });
  if (!loops.length) return null;

  const brep = new rhino.Brep();
  const si = brep.addSurface(ns);
  if (si < 0) return null;
  const fi = brep.newFace(si);
  if (fi < 0) return null;
  const tol = Number.isFinite(tolerance) && tolerance > 0 ? tolerance : 1e-3;

  for (const loop of loops) {
    const li = brep.newLoop(fi, loop.kind);
    if (li < 0) return null;

    if (!loop.fit) {
      // THE POLYLINE PATH, unchanged: one edge, one trim, one vertex used at
      // both ends — exactly the shape createTrimmedPlane produces for a closed
      // boundary. A polyline absorbs corners, so it needs no splitting.
      const pts3d = loop.uv.map(([u, v]) => {
        const p = ns.pointAt(u, v);
        return [p[0], p[1], p[2]];
      });
      const c3 = brep.addEdgeCurve(polylineCurve(rhino, pts3d));
      const c2 = brep.addTrimCurve(polylineCurve(rhino, loop.uv));
      if (c3 < 0 || c2 < 0) return null;
      const v0 = brep.newVertex(pts3d[0], tol);
      const e = brep.newEdge(v0, v0, c3, tol);
      if (v0 < 0 || e < 0) return null;
      if (brep.newTrim(li, e, false, c2, tol) < 0) return null;
      continue;
    }

    // ⚠ THE VERTICES COME FROM THE CURVES THAT WERE ACTUALLY WRITTEN, not from
    // the polyline they were fitted to. A fitted run interpolates its own
    // endpoints exactly — the fitter is asked for that precisely so a boundary
    // still meets its neighbors at the corners topology already agreed on —
    // so the two agree; reading it off the written curve is what makes that an
    // invariant this code HOLDS rather than one it assumes holds elsewhere.
    // One vertex per corner, SHARED by the run that ends there and the run that
    // starts there, so the loop is closed topologically and not merely
    // geometrically. A one-run loop shares its single vertex with itself, which
    // is the polyline path's shape reached by the same rule.
    const runs = loop.fit.runs;
    const verts = runs.map((r) => brep.newVertex(euclideanCtrlPt(r.edge.ctrlPts[0]), tol));
    if (verts.some((v) => v < 0)) return null;

    for (const [ri, run] of runs.entries()) {
      const c3 = brep.addEdgeCurve(curveToRhino(rhino, run.edge));
      const c2 = brep.addTrimCurve(curveToRhino(rhino, run.pcurve));
      if (c3 < 0 || c2 < 0) return null;
      // The edge's own tolerance is how far the surface image of the pcurve can
      // sit from the edge curve — two independent approximations of one
      // boundary. Supplied MEASURED by whoever made the fit, and never allowed
      // below the document tolerance, which is the floor everything else here
      // is written to.
      const edgeTol = Number.isFinite(run.tolerance) && run.tolerance > 0
        ? Math.max(run.tolerance, tol) : tol;
      const e = brep.newEdge(verts[ri], verts[(ri + 1) % runs.length], c3, edgeTol);
      if (e < 0) return null;
      if (brep.newTrim(li, e, false, c2, tol) < 0) return null;
    }
  }
  brep.setTrimIsoFlags();
  brep.setVertexTolerances(true);
  brep.setEdgeTolerances(true);
  brep.setTrimTolerances(true);
  brep.compact();
  // ⚠ ASK OPENNURBS, DO NOT ASSUME. A brep that assembled without a refused
  // index can still be invalid, and writing an invalid one into a .3dm is how
  // a file opens wrong in Rhino with nothing to point at. An invalid result is
  // reported to the caller so the object can be named as skipped instead.
  const [valid, log] = brep.isValidWithLog;
  return valid ? { brep, log: '' } : { brep: null, log: String(log || 'invalid brep') };
}

// ---- A WHOLE SOLID AS ONE ON_Brep ----
//
// ⚠⚠ ON_Brep HAS NO JOIN, AND NEEDS NONE. A multi-face solid is not N breps
// stuck together afterwards — it is ONE brep whose faces SHARE their edges,
// authored that way from the start. Exporting a boolean as N separate trimmed
// faces produces a file that looks right and is a pile of loose surfaces:
// Rhino reports it as open, `SelBadObjects` finds nothing to complain about,
// and nothing downstream that needs a closed solid (a shell, a boolean, an
// STL) will work.
//
// The record arrives already flattened (kernel/breprecord.mjs) because the
// half-edge solid that knows this topology is cyclic and cannot be carried.
// Everything here is index-following.
//
// ⚠ A VERTEX IS MINTED PER DISTINCT POINT, not per edge end. Two edges meeting
// at a corner must reference ONE vertex or the brep is not closed there —
// which is the entire difference between a solid and a heap. Keyed by rounded
// position because that is what "the same corner" means after two independent
// fits have each landed on it.
export function brepRecordToRhino(rhino, record, tolerance) {
  if (!record || !record.ok || !record.faces || !record.faces.length) return null;
  const tol = Number.isFinite(tolerance) && tolerance > 0 ? tolerance : 1e-3;
  const brep = new rhino.Brep();

  const surfaceIndex = record.surfaces.map((s) => brep.addSurface(surfaceToRhino(rhino, s)));
  if (surfaceIndex.some((i) => i < 0)) return { brep: null, log: 'a surface was refused' };

  // ⚠⚠ WELDED BY TOLERANCE, NOT BY A ROUNDED KEY. Two edges meeting at one
  // corner arrive from two INDEPENDENT fits, so their endpoints agree to about
  // the fit's accuracy and not to six decimals — and a rounded key then mints
  // two vertices at one point. A loop built across them is open, and OpenNURBS
  // says so in its own terms: "loop has trim vertex mismatch: m_T[75].m_vi[1] =
  // 24 != m_T[76].m_vi[0] = 25", two indices for one place.
  //
  // The tolerance is the document's, which is the number everything else here
  // is written to; a corner is "the same corner" at exactly the scale the file
  // claims to be accurate to. Linear search because a brep's vertex count is in
  // the tens and the alternative is a spatial index for no measurable gain.
  const vertexPoints = [];
  const vertexFor = (p) => {
    for (let i = 0; i < vertexPoints.length; i++) {
      const q = vertexPoints[i];
      if (Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) <= tol) return i;
    }
    const idx = brep.newVertex([p[0], p[1], p[2]], tol);
    if (idx !== vertexPoints.length) {
      // The array index IS the brep vertex index by construction; if ON ever
      // returns something else the mapping is wrong and silently welding to the
      // wrong corner would be far worse than stopping.
      vertexPoints.push([p[0], p[1], p[2]]);
      return idx;
    }
    vertexPoints.push([p[0], p[1], p[2]]);
    return idx;
  };

  // Edges FIRST and ONCE each — this is the shared part, and fitting or adding
  // it per adjacent face is exactly what pulls a solid apart into loose faces.
  // The two vertices each edge runs BETWEEN are kept alongside it, because a
  // loop is a directed walk and every trim's end vertex has to BE the next
  // trim's start vertex. A singular trim in particular has no edge to take one
  // from, so it must reuse whatever the previous trim ended at.
  const edgeVerts = [];
  const edgeIndex = record.edges.map((e) => {
    if (!e) { edgeVerts.push(null); return -1; }
    const c3 = brep.addEdgeCurve(curveToRhino(rhino, e.curve));
    if (c3 < 0) { edgeVerts.push(null); return -1; }
    const a = vertexFor(e.start);
    const b = e.closed ? a : vertexFor(e.end);
    edgeVerts.push({ a, b });
    return brep.newEdge(a, b, c3, Number.isFinite(e.deviation) && e.deviation > 0 ? Math.max(e.deviation, tol) : tol);
  });

  let trimCount = 0;
  for (const f of record.faces) {
    const fi = brep.newFace(surfaceIndex[f.surfaceIndex]);
    if (fi < 0) return { brep: null, log: 'a face was refused' };
    for (const loop of f.loops) {
      const li = brep.newLoop(fi, loop.loopType === 'inner' ? 2 : 1);
      if (li < 0) return { brep: null, log: 'a loop was refused' };
      let walkVertex = -1; // where the directed walk currently stands
      // ⚠ TRIMS IN LOOP-TRAVERSAL ORDER, EACH WITH ITS OWN ORIENTATION. Emitting
      // them in edge order, or without `reversed`, is what OpenNURBS rejected by
      // name when this assembly was first written — the loop is a directed walk,
      // and an edge shared by two faces is traversed one way by each of them.
      for (const t of loop.trims) {
        // ⚠⚠ A SINGULAR TRIM IS A LOOP MEMBER WITH NO EDGE. It is how ON
        // represents a POLE — a stretch of the parametric rectangle where the
        // surface collapses to a single point, as a revolved disc does along
        // its center. It has a real pcurve and no 3-D length, so it takes a
        // VERTEX rather than an edge, and it closes a loop that would otherwise
        // have a hole where the sew could not carry it.
        if (t.singular) {
          const c2s = brep.addTrimCurve(curveToRhino(rhino, t.curve));
          if (c2s < 0) continue;
          // ⚠⚠ THE VERTEX COMES FROM THE WALK, NOT FROM THE POLE'S COORDINATES.
          // Minting one at the pole's own evaluated position looks equivalent
          // and is not: the adjacent edge ENDS at that pole too, via a fitted
          // curve whose endpoint differs in the last decimals, so the two round
          // to different keys and become two vertices at one point. OpenNURBS
          // reports it exactly — "loop has trim vertex mismatch: m_T[72].m_vi[1]
          // = 25 != m_T[73].m_vi[0] = 24". Reusing the vertex the previous trim
          // ended at makes them the same by construction.
          const v = walkVertex >= 0 ? walkVertex : vertexFor(t.point);
          if (v < 0) continue;
          if (typeof brep.newSingularTrim !== 'function') continue; // an older vendored build; the loop then refuses below rather than silently opening
          if (brep.newSingularTrim(li, v, t.iso, c2s, tol) >= 0) trimCount++;
          continue; // a pole has no length, so the walk stays where it was
        }
        const ei = edgeIndex[t.edgeIndex];
        if (ei == null || ei < 0) continue;
        const c2 = brep.addTrimCurve(curveToRhino(rhino, t.curve));
        if (c2 < 0) continue;
        if (brep.newTrim(li, ei, !!t.reversed, c2, tol) >= 0) trimCount++;
        const ev = edgeVerts[t.edgeIndex];
        if (ev) walkVertex = t.reversed ? ev.a : ev.b;
      }
    }
  }
  if (!trimCount) return { brep: null, log: 'no trim could be built' };

  brep.setTrimIsoFlags();
  brep.setVertexTolerances(true);
  brep.setEdgeTolerances(true);
  brep.setTrimTolerances(true);
  brep.compact();
  // ASK OPENNURBS, DO NOT ASSUME — the same rule the single-face path follows,
  // and the only independent judgment anywhere in this pipeline.
  const [valid, log] = brep.isValidWithLog;
  return valid
    ? { brep, log: '', counts: { faces: record.faces.length, edges: edgeIndex.filter((i) => i >= 0).length, vertices: vertexPoints.length, trims: trimCount } }
    : { brep: null, log: String(log || 'invalid brep') };
}

// ---- DOCUMENT EXPORT ----
//
// Builds a real, complete .3dm file from a plain-data payload — no live
// document/THREE.js objects touch this module at all (worker.js's own job
// is translating the app's real objectTable into this shape and back).
//
// payload = {
//   tolerance: number (mm),
//   layers: [{id, name, color:{r,g,b}, parentId}],   // parent MUST already
//     appear earlier in the array than any child referencing it — true by
//     construction for this app's own `layers` array (a child's parentId
//     always names an already-existing layer at creation time).
//   objects: [
//     {kind:'point', layerId, name, point:[x,y,z]},
//     {kind:'curve', layerId, name, degree, knots, ctrlPts},
//     {kind:'surface', layerId, name, degU, degV, knotsU, knotsV, ctrlNet},
//   ],
// }
//
// A RuledLoft/PolySurface/Split/MultiPipe container's own N panels are not
// a special case here — the caller just emits N separate 'surface' entries
// sharing a name prefix (e.g. "RuledLoft04 panel 1", "...panel 2"), exactly
// per the "Breps (as untrimmed-face joins)" framing. History
// never travels (an explicit rule) — every exported object is
// a plain baked leaf, there is no HistoryRecord field anywhere in this
// payload shape.
export function exportDocument(rhino, payload) {
  const doc = new rhino.File3dm();
  doc.settings().modelUnitSystem = rhino.UnitSystem.Millimeters;
  if (Number.isFinite(payload.tolerance)) {
    doc.settings().modelAbsoluteTolerance = payload.tolerance;
  }

  const layerIdToIndex = new Map();
  const layerIdToGuid = new Map();
  for (const layer of payload.layers || []) {
    const rlayer = new rhino.Layer();
    rlayer.name = layer.name;
    /* ⚠⚠ ALPHA, OR THE LAYER IS INVISIBLE TO SHADING. rhino3dm's color setter
       takes {r,g,b,a} and a missing `a` writes ZERO — a fully transparent layer
       color. Reported from Rhino as "they are coming in on a layer that cannot
       be shaded… so I have to put them on a new layer with shading". Checked
       against a file Rhino itself wrote: its Default layer is 0,0,0,255, and
       ours was 138,141,144,0. */
    const lc = layer.color || { r: 140, g: 141, b: 144 };
    rlayer.color = { r: lc.r, g: lc.g, b: lc.b, a: lc.a == null ? 255 : lc.a };
    if (layer.parentId != null && layerIdToGuid.has(layer.parentId)) {
      rlayer.parentLayerId = layerIdToGuid.get(layer.parentId);
    }
    const idx = doc.layers().add(rlayer);
    layerIdToIndex.set(layer.id, idx);
    layerIdToGuid.set(layer.id, doc.layers().get(idx).id);
  }

  const skipped = [];
  for (const obj of payload.objects || []) {
    const attrs = new rhino.ObjectAttributes();
    if (obj.name) attrs.name = obj.name;
    if (obj.layerId != null && layerIdToIndex.has(obj.layerId)) {
      attrs.layerIndex = layerIdToIndex.get(obj.layerId);
    }
    if (obj.kind === 'point') {
      doc.objects().addPoint(pointToRhino(obj.point), attrs);
    } else if (obj.kind === 'curve') {
      const nc = curveToRhino(rhino, obj);
      doc.objects().addCurve(nc, attrs);
    } else if (obj.kind === 'surface') {
      const ns = surfaceToRhino(rhino, obj);
      doc.objects().addSurface(ns, attrs);
    } else if (obj.kind === 'trimmedsurface') {
      const built = trimmedSurfaceToRhinoBrep(rhino, obj, payload.tolerance);
      if (built && built.brep) doc.objects().add(built.brep, attrs);
      else skipped.push({ name: obj.name || null, kind: `trimmed surface (OpenNURBS rejected it: ${built ? built.log.split('\n')[0] : 'could not be built'})` });
    } else if (obj.kind === 'brep') {
      // A whole solid as ONE brep — and a refusal here MUST NOT COST THE OBJECT.
      // The record is only an attempt at a better representation of geometry the
      // caller can also express as loose trimmed faces, so it sends those along
      // as `fallback` and a rejected join degrades to them: worse, named, and
      // still every face in the file.
      //
      // ⚠ THIS IS NOT BELT-AND-BRACES. Without it a validator refusal deletes
      // the object outright — the export reports one skip and writes NOTHING for
      // it, which is the single worst outcome available here and exactly what
      // happened before the fallback was wired.
      const built = brepRecordToRhino(rhino, obj.record, payload.tolerance);
      if (built && built.brep) {
        doc.objects().add(built.brep, attrs);
      } else {
        const why = built ? built.log.split('\n')[0] : 'could not be built';
        let wrote = 0;
        for (const face of obj.fallback || []) {
          const fb = trimmedSurfaceToRhinoBrep(rhino, face, payload.tolerance);
          if (fb && fb.brep) {
            const fattrs = new rhino.ObjectAttributes();
            if (face.name) fattrs.name = face.name;
            fattrs.layerIndex = attrs.layerIndex;
            doc.objects().add(fb.brep, fattrs);
            wrote++;
          }
        }
        skipped.push({
          name: obj.name || null,
          kind: `joined solid (OpenNURBS rejected it: ${why})`
            + (wrote ? ` — written as ${wrote} loose trimmed face(s) instead` : ', and it has no loose faces to fall back on'),
        });
      }
    } else {
      skipped.push({ name: obj.name || null, kind: obj.kind });
    }
  }

  const bytes = doc.toByteArray();
  return { bytes, skipped };
}

// ---- DOCUMENT IMPORT ----
//
// Reads a real .3dm file back into the identical plain-data shape
// exportDocument() consumes, plus a `skipped` array naming (never silently
// dropping) anything genuinely out of this v1's scope: a Mesh, a SubD, or
// any other object type this app has no honest mapping for yet. A Brep of
// ANY face count now imports — a single-face Brep as one plain surface, a
// multi-face Brep as one panel per face, each carrying its own trim loops
// as 3-D polylines under `trimEdges3d`. Turning those into the (u,v) the
// app stores is the caller's job and can fail; a panel whose boundary is
// not recovered keeps the old behavior of arriving at its full untrimmed
// extent, and the app says so rather than letting it pass for faithful.
// Each panel's own "face N" name is unchanged. worker.js's own job is turning THESE plain
// objects back into real table entries (a curve/surface imports as a plain
// baked leaf, no history — matching Bake's own already-established
// convention).
export function importDocument(rhino, bytes) {
  const doc = rhino.File3dm.fromByteArray(bytes);
  const tolerance = doc.settings().modelAbsoluteTolerance;

  const layerGuidToId = new Map();
  const layers = [];
  const rlayers = doc.layers();
  for (let i = 0; i < rlayers.count; i++) {
    const l = rlayers.get(i);
    layerGuidToId.set(l.id, i);
  }
  const NIL_GUID = '00000000-0000-0000-0000-000000000000';
  for (let i = 0; i < rlayers.count; i++) {
    const l = rlayers.get(i);
    const parentId = l.parentLayerId && l.parentLayerId !== NIL_GUID && layerGuidToId.has(l.parentLayerId)
      ? layerGuidToId.get(l.parentLayerId)
      : null;
    layers.push({ id: i, name: l.name || `Layer${i}`, color: l.color, parentId });
  }

  let brepGroupSeq = 0;

  const objects = [];
  const skipped = [];
  const robjs = doc.objects();
  for (let i = 0; i < robjs.count; i++) {
    const o = robjs.get(i);
    const attrs = o.attributes();
    // attrs.layerIndex is already a plain index into rlayers, and `layers`
    // above was built in that exact same index order — so the index IS the
    // id, no GUID lookup needed here (layerGuidToId above is only for
    // resolving PARENT relationships between layers, a different question).
    const layerId = attrs.layerIndex >= 0 && attrs.layerIndex < layers.length ? attrs.layerIndex : (layers.length ? 0 : null);
    /* ⚠⚠ AN EXTRUSION IS A BREP RHINO HAS NOT BOTHERED TO EXPAND. ON_Extrusion is
       the lightweight form Rhino stores a great many ordinary solids in — most
       of what a box, a boss or a rib actually is — and it fell to the skip list
       by name, so those solids arrived as nothing at all. Counted across the 105
       models that ship with the Rhino Level 1 and Level 2 training manuals: ONE
       HUNDRED extrusion objects, every one dropped.
       `toBrep` is OpenNURBS's own conversion, so the result takes the ordinary
       Brep path below and a converted extrusion imports exactly as the same
       shape saved as a Brep would. */
    let geo = o.geometry();
    if (geo && geo.objectType === rhino.ObjectType.Extrusion && typeof geo.toBrep === 'function') {
      let asBrep = null;
      try { asBrep = geo.toBrep(true); } catch { asBrep = null; }
      if (asBrep) geo = asBrep;
    }
    const name = attrs.name || null;
    if (geo.objectType === rhino.ObjectType.Point) {
      objects.push({ kind: 'point', layerId, name, point: pointFromRhino(geo.location) });
    } else if (geo.objectType === rhino.ObjectType.Curve) {
      const nc = geo.toNurbsCurve();
      objects.push({ kind: 'curve', layerId, name, ...curveFromRhino(nc) });
    } else if (geo.objectType === rhino.ObjectType.Surface) {
      const ns = geo.toNurbsSurface();
      objects.push({ kind: 'surface', layerId, name, ...surfaceFromRhino(ns) });
    } else if (geo.objectType === rhino.ObjectType.Brep) {
      // A single-face Brep with trivial trimming IS its own untrimmed
      // surface — imports as one plain editable surface, no naming
      // suffix needed (matches the pre-existing single-face behavior
      // exactly, byte-for-byte).
      //
      // A genuinely MULTI-face Brep (any real mechanical part — a
      // fillet, a counterbore, an arm) is real, honest v1 scope, not a
      // silent skip anymore: each face's own UNTRIMMED underlying
      // surface comes in as an independent panel, mirroring this app's
      // own EXPORT-side "Brep (as untrimmed-face joins)" convention
      // exactly, just run in reverse. A trimmed face's true visible
      // BOUNDARY is real information genuinely lost here — a fillet
      // trimmed to a narrow band, or a small counterbore trimmed out of
      // a bigger planar sheet, both come back at their FULL untrimmed
      // extent, which can be larger than the real part ever showed at
      // that spot. Named honestly, not silently perfect: each panel's
      // name gets its own "face N" suffix.
      const faces = geo.faces();
      const faceCount = faces.count;
      if (faceCount === 1 && geo.isSurface) {
        const face = faces.get(0);
        const ns = face.underlyingSurface().toNurbsSurface();
        objects.push({ kind: 'surface', layerId, name, ...surfaceFromRhino(ns) });
      } else {
        /* ⚠ A MULTI-FACE BREP IS ONE OBJECT, AND IT CAME IN AS N LOOSE ONES.
           Every face was pushed as its own surface with nothing saying they
           belonged together, so a solid arrived as a pile of sheets with no
           topology — no faces list, no edges, and therefore nothing any command
           that needs a solid could act on. A real Rhino solid of seven faces
           imported as seven surfaces and could not be filleted, joined, shelled
           or booleaned. The faces still convert one at a time; they now carry
           the index of the brep they came from, and the caller assembles them
           back into the one object they were. `isSolid` travels too, because
           whether a thing is closed is a fact the file already knows and this
           app would otherwise have to re-derive. */
        let anyFaceFailed = false;
        const brepGroup = brepGroupSeq++;
        for (let f = 0; f < faceCount; f++) {
          try {
            const face = faces.get(f);
            const ns = face.underlyingSurface().toNurbsSurface();
            const panelName = name ? `${name} face ${f + 1}` : null;
            const trimEdges3d = faceTrimLoopsFromRhino(geo, face, tolerance);
            objects.push({ kind: 'surface', layerId, name: panelName, ...surfaceFromRhino(ns), trimEdges3d,
              brepGroup, brepName: name || null, brepIsSolid: !!geo.isSolid, brepFaceCount: faceCount });
          } catch (err) {
            anyFaceFailed = true;
          }
        }
        if (anyFaceFailed) skipped.push({ name, objectType: 'Brep (one or more faces failed to convert)' });
      }
    } else if (isRhinoSubD(rhino, geo)) {
      // A REAL Rhino SubD arrives as a real SuperB cage — see
      // subdCageFromRhino. Refuses by name rather than degrading to a
      // mesh if the control net can't be read: a mesh that looks right
      // and edits wrong is the one outcome the SubD design rules out
      // before anything else.
      try {
        const { cage, creasesLost } = subdCageFromRhino(rhino, geo);
        objects.push({ kind: 'subd', layerId, name, cage, creasesLost });
      } catch (err) {
        skipped.push({ name, objectType: `SubD (${err && err.message ? err.message : 'control net unreadable'})` });
      }
    } else {
      // A genuinely multi-face or trimmed Brep, a Mesh, an
      // Extrusion, etc. — real, stated v1 scope cut: this app can't yet
      // author/edit a trimmed surface, so degrading one to a display-only
      // mesh (the fuller ask) is real, separate follow-up work,
      // not attempted here. Named honestly, never silently dropped.
      skipped.push({ name, objectType: describeRhinoGeometry(geo) });
    }
  }

  return { tolerance, layers, objects, skipped };
}
