// SuperB primitive cage generators (the SubDBox/SubDSphere/SubDCylinder/
// SubDPlane commands — Rhino's names, kept as a cross-reference; the
// app-layer object type is SuperB). Kernel only: pure cage-in-data-out
// math, no app-layer object or UI. Each function returns a plain cage
// { vertices:[[x,y,z],...], faces:[[i0,i1,...],...], creases:{} } — the
// shape kernel/subd.mjs's subdivideCatmullClark expects.
//
// Facet count is the typed option each command takes: an integer setting
// how finely the cage is divided before any Catmull-Clark refinement runs
// (control-net density, not display resolution — see superbDisplayMesh
// for the refinement-level choice).
//
// No primitive is creased by default: a facet-count-1 SuperBBox subdivided
// with zero creases converges toward a rounded, sphere-like shape (the
// textbook "a cube smooths into a sphere" Catmull-Clark demonstration),
// which is what the Box/Smooth toggle shows. Creasing every box edge by
// default would hide that behind an already-sharp shape.

// Dedupes coincident vertices by a rounded-coordinate key so a cage built
// from several independently parametrized face grids (a box's 6 faces)
// welds into one watertight manifold cage rather than 6 disjoint islands.
// The key rounds to 6 decimals as a number first and canonicalizes -0 to
// +0 before formatting: `(-1.4695761589768238e-15).toFixed(6)` is
// `"-0.000000"`, not `"0.000000"`, so a seam pair differing by ~1.5e-15 (a
// trig-cancellation residue at v=2*PI on a revolved surface) would
// otherwise stay two vertices and leave the seam unwelded.
function roundKeyComponent(v) {
  let r = Math.round(v * 1e6) / 1e6;
  if (r === 0) r = 0; // canonicalize -0 -> +0 (Math.round/division can produce -0 for a tiny negative input)
  return r.toFixed(6);
}
export function makeVertexWelder() {
  const map = new Map();
  const vertices = [];
  function vid(x, y, z) {
    const key = `${roundKeyComponent(x)}_${roundKeyComponent(y)}_${roundKeyComponent(z)}`;
    let idx = map.get(key);
    if (idx === undefined) { idx = vertices.length; vertices.push([x, y, z]); map.set(key, idx); }
    return idx;
  }
  return { vid, vertices };
}

// SuperBBox (Rhino: SubDBox) — an axis-aligned box cage, `facets`
// subdivisions per edge on each of its 6 faces (facets=1 is the plain
// 8-vertex/6-face/12-edge cube). Each face is its own uniform point grid,
// welded at shared edges/corners by the vertex welder above; the test file
// checks the result is watertight (zero boundary edges) via buildTopology.
// The second count (`facetsH`) runs along Z, and the six faces do not all
// use it the same way: a side face is (around) x (up), while the two caps
// are (around) x (around) in both directions. Two faces meeting along an
// edge whose sides carry different point counts have nothing to weld and
// the box leaks, so each face names its own pair and every shared edge is
// named the same on both sides. `facetsH` omitted uses `facets`.
export function superbBoxCage(center = [0, 0, 0], halfExtents = [25, 25, 25], facets = 1, facetsH = null) {
  const n = Math.max(1, Math.round(facets));
  const nz = Math.max(1, Math.round(facetsH == null ? facets : facetsH));
  const [cx, cy, cz] = center;
  const [hx, hy, hz] = halfExtents;
  const { vid, vertices } = makeVertexWelder();
  const faces = [];
  function buildFace(originFn, nu = n, nv = n) {
    const grid = [];
    for (let j = 0; j <= nv; j++) {
      const row = [];
      for (let i = 0; i <= nu; i++) {
        const u = i / nu, v = j / nv;
        const [x, y, z] = originFn(u, v);
        row.push(vid(x, y, z));
      }
      grid.push(row);
    }
    for (let j = 0; j < nv; j++) {
      for (let i = 0; i < nu; i++) {
        faces.push([grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]]);
      }
    }
  }
  // Six faces, each parametrized (u,v)->world point, ordered so every
  // face's own loop winds consistently outward (CCW viewed from outside)
  // — matters for correct THREE.js shading normals downstream, not for
  // the subdivision math itself (which is winding-direction agnostic).
  buildFace((u, v) => [cx + hx, cy - hy + 2 * hy * u, cz - hz + 2 * hz * v], n, nz); // +X
  buildFace((u, v) => [cx - hx, cy + hy - 2 * hy * u, cz - hz + 2 * hz * v], n, nz); // -X
  buildFace((u, v) => [cx + hx - 2 * hx * u, cy + hy, cz - hz + 2 * hz * v], n, nz); // +Y
  buildFace((u, v) => [cx - hx + 2 * hx * u, cy - hy, cz - hz + 2 * hz * v], n, nz); // -Y
  buildFace((u, v) => [cx - hx + 2 * hx * u, cy - hy + 2 * hy * v, cz + hz]); // +Z — a cap, both directions horizontal
  buildFace((u, v) => [cx - hx + 2 * hx * u, cy + hy - 2 * hy * v, cz - hz]); // -Z
  return { vertices, faces, creases: {} };
}

// SuperBSphere (Rhino: SubDSphere) — reuses superbBoxCage's topology (a
// cube-sphere cage; it does not reproduce Rhino's internal SubDSphere
// construction), then projects every vertex radially onto the sphere.
// facets=1 gives the 6-face cage.
export function superbSphereCage(center = [0, 0, 0], radius = 25, facets = 1, facetsH = null) {
  const box = superbBoxCage(center, [radius, radius, radius], facets, facetsH);
  const [cx, cy, cz] = center;
  const vertices = box.vertices.map(([x, y, z]) => {
    const dx = x - cx, dy = y - cy, dz = z - cz;
    const d = Math.hypot(dx, dy, dz) || 1;
    const s = radius / d;
    return [cx + dx * s, cy + dy * s, cz + dz * s];
  });
  return { vertices, faces: box.faces, creases: {} };
}

// SuperBCylinder (Rhino: SubDCylinder) — `facetsH + 1` rings of `facets`
// vertices, evenly spaced from z=center.z to z=center.z+height, the side
// quads between them, plus one n-gon cap at each end (ngons are legal
// Catmull-Clark faces — kernel/subd.mjs handles any face size). `facets` is
// the radial count (>=3) and `facetsH` the height count: a form whose
// profile is edited along its height cannot be shaped through the radial
// count alone. `facetsH` omitted means one ring pair.
export function superbCylinderCage(center = [0, 0, 0], radius = 25, height = 50, facets = 8, facetsH = null) {
  const n = Math.max(3, Math.round(facets));
  const rings = Math.max(1, Math.round(facetsH == null ? 1 : facetsH));
  const [cx, cy, cz] = center;
  const vertices = [];
  for (let r = 0; r <= rings; r++) {
    const z = cz + height * (r / rings);
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      vertices.push([cx + radius * Math.cos(a), cy + radius * Math.sin(a), z]);
    }
  }
  const faces = [];
  for (let r = 0; r < rings; r++) {
    const near = r * n, far = (r + 1) * n;
    for (let i = 0; i < n; i++) {
      const i1 = (i + 1) % n;
      faces.push([near + i, near + i1, far + i1, far + i]); // side quad
    }
  }
  const bottomFace = []; for (let i = n - 1; i >= 0; i--) bottomFace.push(i); // reversed for outward (-Z) winding
  const topFace = []; for (let i = 0; i < n; i++) topFace.push(rings * n + i);
  faces.push(bottomFace, topFace);
  return { vertices, faces, creases: {} };
}

// SuperBPlane (Rhino: SubDPlane) — a flat, open (it has a boundary, unlike
// the closed cages above) grid of quads in the XY plane at the center's Z,
// `facets` along X and `facetsH` (default `facets`) along Y.
export function superbPlaneCage(center = [0, 0, 0], width = 50, height = 50, facets = 1, facetsH = null) {
  const n = Math.max(1, Math.round(facets));
  const m = Math.max(1, Math.round(facetsH == null ? facets : facetsH));
  const [cx, cy, cz] = center;
  const vertices = [];
  const idx = (i, j) => j * (n + 1) + i;
  for (let j = 0; j <= m; j++) {
    for (let i = 0; i <= n; i++) {
      vertices.push([cx - width / 2 + width * (i / n), cy - height / 2 + height * (j / m), cz]);
    }
  }
  const faces = [];
  for (let j = 0; j < m; j++) {
    for (let i = 0; i < n; i++) {
      faces.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
    }
  }
  return { vertices, faces, creases: {} };
}

// SuperBCone (Rhino's SubD toolset has no cone primitive) — a ring of
// `facets` base vertices plus one apex vertex: SuperBCylinder's ring
// construction with the top ring collapsed to a point. Unlike SuperBSphere
// (the box's welded grid, every vertex valence 3 or 4), the apex is an
// extraordinary vertex of valence `facets` — the pole of a NURBS revolve
// (kernel/primitives.mjs) expressed as cage topology instead of a
// degenerate control row. computeVertexPoint/smoothVertexRule
// (kernel/subd.mjs) handle any valence and any face size (the base cap is
// an n-gon), so no new subdivision math is needed.
// The height count adds rings below the apex, never at it: the apex is a
// single vertex by construction, so the topmost band stays a ring of
// triangles and every band under it is quads. A ring at the apex would be a
// ring of coincident points, which is a pinch, not a denser cone.
export function superbConeCage(center = [0, 0, 0], radius = 25, height = 50, facets = 8, facetsH = null) {
  const n = Math.max(3, Math.round(facets));
  const rings = Math.max(1, Math.round(facetsH == null ? 1 : facetsH));
  const [cx, cy, cz] = center;
  const vertices = [];
  for (let r = 0; r < rings; r++) {
    const t = r / rings;
    const z = cz + height * t, rad = radius * (1 - t);
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      vertices.push([cx + rad * Math.cos(a), cy + rad * Math.sin(a), z]);
    }
  }
  const apexIdx = vertices.length;
  vertices.push([cx, cy, cz + height]);
  const faces = [];
  for (let r = 0; r + 1 < rings; r++) {
    const near = r * n, far = (r + 1) * n;
    for (let i = 0; i < n; i++) {
      const i1 = (i + 1) % n;
      faces.push([near + i, near + i1, far + i1, far + i]);
    }
  }
  const last = (rings - 1) * n;
  for (let i = 0; i < n; i++) {
    const i1 = (i + 1) % n;
    faces.push([last + i, last + i1, apexIdx]); // side triangle — the winding of a SuperBCylinder side quad [i0,i1,j1,j0] with j0 and j1 collapsed onto the apex
  }
  const baseFace = []; for (let i = n - 1; i >= 0; i--) baseFace.push(i); // reversed for outward (-Z) winding, matching SuperBCylinder's bottom cap
  faces.push(baseFace);
  return { vertices, faces, creases: {} };
}

// SuperBTorus — the genus-1 case. A torus has no boundary and must close
// in both its ring (major) and tube (minor) directions at once. The
// construction does not use the welder: every vertex is generated once
// into a single flat nU x nV array, addressed with a wrapping modulo index
// in both directions, so vertex (i,j) and vertex (i+nU,j) are the same
// array slot and both seams close exactly, with no rounded-coordinate
// coincidence to rely on. The genus follows from this closure: an
// nU x nV toroidal grid has V=nU*nV, F=nU*nV quads, E=2*nU*nV (nU*nV
// ring-direction edges + nU*nV tube-direction edges) — so chi = V-E+F = 0,
// and chi=2-2*genus gives genus 1 for any pair of counts.
//
// The two directions take separate counts — around the ring (U) and
// around the tube (V), like the U and V control-point counts of a NURBS
// surface. `facetsV` defaults to `facetsU` when omitted, so a single-count
// call and a stored single `facets` param give the same cage.
export function superbTorusCage(center = [0, 0, 0], majorRadius = 30, minorRadius = 10, facetsU = 8, facetsV = null) {
  const nU = Math.max(3, Math.round(facetsU));
  const nV = Math.max(3, Math.round(facetsV == null ? facetsU : facetsV));
  const [cx, cy, cz] = center;
  const vertices = new Array(nU * nV);
  const idx = (i, j) => ((i % nU + nU) % nU) * nV + ((j % nV + nV) % nV);
  for (let i = 0; i < nU; i++) {
    const theta = (i / nU) * Math.PI * 2; // ring (major) angle
    const ct = Math.cos(theta), st = Math.sin(theta);
    for (let j = 0; j < nV; j++) {
      const phi = (j / nV) * Math.PI * 2; // tube (minor) angle
      const rho = majorRadius + minorRadius * Math.cos(phi); // distance from the ring axis at this tube angle
      vertices[idx(i, j)] = [cx + rho * ct, cy + rho * st, cz + minorRadius * Math.sin(phi)];
    }
  }
  const faces = [];
  for (let i = 0; i < nU; i++) {
    for (let j = 0; j < nV; j++) {
      // Same corner order as SuperBCylinder's side quad [i0,i1,j1,j0]
      // (curr-ring/next-ring at curr-tube-angle, then next-ring/curr-ring at
      // next-tube-angle) — outward-normal winding, generalized from one
      // periodic direction (the cylinder's ring) to two.
      faces.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
    }
  }
  return { vertices, faces, creases: {} };
}

// SuperBEllipsoid — reuses superbBoxCage's welded topology, like
// superbSphereCage, then projects every vertex radially onto the
// ellipsoid along the line from center through that vertex. With
// rx=ry=rz=R the scale factor s below reduces to R/|d|, superbSphereCage's
// formula. Solving for s such that center + s*(x-center,y-center,z-center)
// satisfies ((s*dx)/rx)^2+((s*dy)/ry)^2+((s*dz)/rz)^2=1 gives
// s = 1/sqrt((dx/rx)^2+(dy/ry)^2+(dz/rz)^2) — exact for every vertex,
// wherever on the box it started.
export function superbEllipsoidCage(center = [0, 0, 0], radii = [25, 25, 25], facets = 1, facetsH = null) {
  const box = superbBoxCage(center, radii, facets, facetsH);
  const [cx, cy, cz] = center;
  const [rx, ry, rz] = radii;
  const vertices = box.vertices.map(([x, y, z]) => {
    const dx = x - cx, dy = y - cy, dz = z - cz;
    const q = (dx / rx) * (dx / rx) + (dy / ry) * (dy / ry) + (dz / rz) * (dz / rz);
    const s = q > 0 ? 1 / Math.sqrt(q) : 1;
    return [cx + dx * s, cy + dy * s, cz + dz * s];
  });
  return { vertices, faces: box.faces, creases: {} };
}
