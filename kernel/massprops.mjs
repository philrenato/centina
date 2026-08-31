// MASS PROPERTIES — volume, surface area, centroid and second moments of a
// closed shell, by the divergence theorem over its own boundary faces.
//
// THE REASON THIS EXISTS IS NOT THE FEATURE. Volume is a BOOLEAN VALIDITY
// ORACLE that needs no reference answer and no human looking at a picture:
//
//     volume(A u B) + volume(A n B) == volume(A) + volume(B)
//     volume(A - B) + volume(A n B) == volume(A)
//
// Both hold for ANY two solids, whatever shape, with nothing to compare
// against — which is exactly what a fuzz harness needs. A boolean that keeps
// a fragment it should have dropped, or drops one it should have kept, still
// sews into something closed with a perfectly ordinary Euler characteristic
// and zero naked edges; chi and the naked-edge count both pass it. The volume
// identity does not.
//
// A FACE IS A POINT LOOP, AND A TRIANGLE IS JUST A 3-POINT LOOP — so the same
// function measures a tessellated triangle soup and a B-rep's own face
// boundaries with no second code path. Every integral below is accumulated as
// a signed sum of tetrahedra fanned from the ORIGIN, which is what makes that
// unification exact rather than convenient:
//
//   - The origin needs no relationship to the solid. A tetrahedron behind the
//     origin contributes a negative volume and the far face's own positive
//     contribution cancels it, so the total is the enclosed volume wherever
//     the solid sits. No centering pass, no bounding-box trick.
//   - A fan is exact for a PLANAR loop even when it is not convex. A fan
//     triangle that escapes the polygon is traversed the other way round by
//     its neighbors and cancels — the 3D generalization of the shoelace
//     formula. Convexity is not required and is not checked.
//   - A face's inner rings need no special handling. A ring wound opposite to
//     its outer loop subtracts, so a face with holes is measured by iterating
//     all of its loops and summing.
//
// WHAT THIS IS EXACT FOR, STATED PLAINLY. Every result is exact for a shell
// whose faces are genuinely planar polygons — a box, any polyhedron, any
// triangle soup. For a CURVED face handed over as its own boundary loop, the
// loop is a polygon inscribed in the true surface, so the answer is the
// inscribed polyhedron's, slightly under the true one. Measure a curved solid
// from its TESSELLATION rather than its boundary loops and the error is the
// tessellation's, which is the caller's own resolution choice.
//
// CLOSURE IS REPORTED, NOT ASSUMED. A shell that is genuinely closed and
// consistently wound satisfies the closed-surface theorem exactly: the sum of
// every face's own area vector is zero. `closureResidual` is that sum's
// magnitude, relative to total area — a real, cheap check that an open shell
// or a flipped face fails, and one no volume number can tell you on its own.
//
// NOT BUILT, NAMED RATHER THAN IMPLIED: principal axes. They are one
// eigen-decomposition of the inertia tensor this already computes, and
// kernel/refit.mjs already carries a symmetric 3x3 Jacobi solver to reuse —
// but a symmetric solid (a sphere, a cube) has degenerate eigenvalues, where
// "the principal axes" are not a well-defined answer at all, and deciding
// whether to refuse, flag, or return an arbitrary orthonormal triple is its
// own call rather than an oversight of this one.

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

function finitePoint(p, where) {
  if (!Array.isArray(p) || p.length < 3) throw new Error(`massProperties: ${where} is not an [x,y,z] point`);
  for (let k = 0; k < 3; k++) {
    if (!Number.isFinite(p[k])) throw new Error(`massProperties: ${where} has a non-finite coordinate`);
  }
}

/**
 * Mass properties of a closed shell given as an array of point LOOPS.
 *
 * `loops` — [[[x,y,z], ...], ...]. Each entry is one face boundary (or one
 *   ring of a face), given once, NOT repeating its first point at the end.
 *   A triangle soup is simply an array of 3-point loops. Every loop must be
 *   wound consistently with every other — outward for a positive volume.
 *
 * Returns { volume, area, centroid, bbox, areaVector, closureResidual,
 *           inertiaOrigin, inertiaCentroid }, where:
 *
 *   volume            SIGNED. Negative means the shell is wound inside-out,
 *                     which is a real possible outcome and is reported rather
 *                     than absorbed by an abs().
 *   area              Sum of each loop's own planar area. Exact for a planar
 *                     loop of any shape; for a triangle it is the triangle.
 *   centroid          Volume-weighted, null when the volume is degenerate.
 *   areaVector        Sum of every loop's own area vector; zero for a closed,
 *                     consistently wound shell.
 *   closureResidual   |areaVector| / area — dimensionless, ~1e-16 when closed.
 *   inertiaOrigin     Second-moment (inertia) tensor about the world origin,
 *                     unit density, as a flat 9-element row-major matrix.
 *   inertiaCentroid   The same tensor moved to the centroid by the parallel
 *                     axis theorem; null when the centroid is.
 */
export function massProperties(loops) {
  if (!Array.isArray(loops) || !loops.length) {
    throw new Error('massProperties: needs at least one face loop');
  }

  let volume = 0, area = 0;
  const areaVector = [0, 0, 0];
  const cWeighted = [0, 0, 0];
  // Second-moment accumulator, the integral of x_i x_j over the solid.
  const M = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];

  for (let li = 0; li < loops.length; li++) {
    const loop = loops[li];
    if (!Array.isArray(loop) || loop.length < 3) {
      throw new Error(`massProperties: loop ${li} has fewer than 3 points`);
    }
    for (let i = 0; i < loop.length; i++) {
      finitePoint(loop[i], `loop ${li} point ${i}`);
      for (let k = 0; k < 3; k++) {
        if (loop[i][k] < lo[k]) lo[k] = loop[i][k];
        if (loop[i][k] > hi[k]) hi[k] = loop[i][k];
      }
    }

    // Area vector of the loop: half the sum of consecutive cross products.
    // Its magnitude IS the planar polygon's area, with no convexity
    // assumption and no need to know the plane in advance.
    const av = [0, 0, 0];
    for (let i = 0; i < loop.length; i++) {
      const p = loop[i], q = loop[(i + 1) % loop.length];
      const c = cross(p, q);
      av[0] += c[0] / 2; av[1] += c[1] / 2; av[2] += c[2] / 2;
    }
    area += Math.hypot(av[0], av[1], av[2]);
    areaVector[0] += av[0]; areaVector[1] += av[1]; areaVector[2] += av[2];

    // Fan the loop from its own first point, then each fan triangle with the
    // origin makes a signed tetrahedron.
    const a = loop[0];
    for (let i = 1; i + 1 < loop.length; i++) {
      const b = loop[i], c = loop[i + 1];
      const det = dot(a, cross(b, c));       // 6 * signed tetrahedron volume
      const v = det / 6;
      volume += v;

      // Centroid of the tetrahedron (0,a,b,c) is (a+b+c)/4.
      cWeighted[0] += v * (a[0] + b[0] + c[0]) / 4;
      cWeighted[1] += v * (a[1] + b[1] + c[1]) / 4;
      cWeighted[2] += v * (a[2] + b[2] + c[2]) / 4;

      // Second moments of the same tetrahedron, about the origin:
      //   integral of x_i x_j = (det / 120) * (sum p_k p_k^T + (sum p)(sum p)^T)
      // Checked against the canonical tetrahedron (0, e1, e2, e3), whose own
      // integrals are 1/60 on the diagonal and 1/120 off it.
      const s = [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]];
      for (let r = 0; r < 3; r++) {
        for (let q = 0; q < 3; q++) {
          M[r * 3 + q] += (det / 120) * (a[r] * a[q] + b[r] * b[q] + c[r] * c[q] + s[r] * s[q]);
        }
      }
    }
  }

  const degenerate = Math.abs(volume) < 1e-15;
  const centroid = degenerate ? null : [cWeighted[0] / volume, cWeighted[1] / volume, cWeighted[2] / volume];

  // Inertia from the second moments: I = trace(M) * Id - M.
  const tr = M[0] + M[4] + M[8];
  const inertiaOrigin = [
    tr - M[0], -M[1], -M[2],
    -M[3], tr - M[4], -M[5],
    -M[6], -M[7], tr - M[8],
  ];

  let inertiaCentroid = null;
  if (centroid) {
    // Parallel axis theorem, subtracting the point-mass tensor of the whole
    // volume sitting at the centroid.
    const c = centroid;
    const c2 = dot(c, c);
    inertiaCentroid = inertiaOrigin.slice();
    for (let r = 0; r < 3; r++) {
      for (let q = 0; q < 3; q++) {
        inertiaCentroid[r * 3 + q] -= volume * ((r === q ? c2 : 0) - c[r] * c[q]);
      }
    }
  }

  return {
    volume,
    area,
    centroid,
    bbox: { lo, hi },
    areaVector,
    closureResidual: area > 0 ? Math.hypot(areaVector[0], areaVector[1], areaVector[2]) / area : 0,
    inertiaOrigin,
    inertiaCentroid,
  };
}

/**
 * Every face boundary of a B-rep solid, as point loops, ready for
 * `massProperties`. Includes each face's inner rings alongside its outer
 * loop — a ring is wound opposite to the loop that contains it, so it
 * subtracts on its own, with no hole-specific arithmetic anywhere.
 *
 * A vertex with no point is a real possibility in the half-edge structure
 * (the Euler operators can create one before it is positioned), so a loop
 * carrying one is refused by name rather than measured as if it sat at the
 * origin.
 */
export function brepFaceLoops(solid) {
  if (!solid?.shells?.length) throw new Error('brepFaceLoops: solid has no shells');
  const loops = [];
  for (const shell of solid.shells) {
    for (const face of shell.faces) {
      for (const loop of face.loops) {
        const pts = [];
        let he = loop.halfEdge;
        if (!he) continue;
        const start = he;
        do {
          if (!he.vertex?.point) throw new Error(`brepFaceLoops: face ${face.id} has a vertex with no position`);
          pts.push(he.vertex.point);
          he = he.next;
        } while (he && he !== start);
        if (pts.length >= 3) loops.push(pts);
      }
    }
  }
  if (!loops.length) throw new Error('brepFaceLoops: solid has no usable face loops');
  return loops;
}

/** Mass properties straight from a B-rep solid. See the exactness note above. */
export function massPropertiesOfBrep(solid) {
  return massProperties(brepFaceLoops(solid));
}

/**
 * The boolean validity oracle, as a function rather than a comment.
 *
 * Given the four measured volumes, reports the residual of both identities
 * relative to the scale of the operands — dimensionless, so a fixture's own
 * size does not change what "close" means. `ok` is a plain threshold on the
 * worst of the two; a caller wanting a different threshold reads the
 * residuals directly.
 *
 * Volumes are taken as absolute, since an inside-out shell is a separate
 * failure with its own separate report (`volume < 0`) and folding the two
 * together would let one mask the other.
 */
export function volumeIdentityResidual({ a, b, union, intersect, difference = null }, tol = 1e-9) {
  const A = Math.abs(a), B = Math.abs(b), U = Math.abs(union), I = Math.abs(intersect);
  const scale = Math.max(A, B, 1e-12);
  const unionResidual = Math.abs((U + I) - (A + B)) / scale;
  const out = { unionResidual, differenceResidual: null, worst: unionResidual, ok: unionResidual <= tol, tol };
  if (difference !== null) {
    const D = Math.abs(difference);
    out.differenceResidual = Math.abs((D + I) - A) / scale;
    out.worst = Math.max(unionResidual, out.differenceResidual);
    out.ok = out.worst <= tol;
  }
  return out;
}
