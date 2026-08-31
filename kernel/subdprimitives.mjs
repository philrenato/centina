// SuperB PRIMITIVE CAGE GENERATORS (the SUBDBOX/SUBDSPHERE/
// SUBDCYLINDER/SUBDPLANE creation commands — Rhino names, kept as Rhino's
// own cross-reference; the app-layer object type is OUR name, SuperB, see
// the app's own SuperB section) — KERNEL ONLY: pure cage-in-data-
// out math, no app-layer object/UI, matching kernel/subd.mjs's own
// discipline exactly. Each function returns a plain cage
// { vertices:[[x,y,z],...], faces:[[i0,i1,...],...], creases:{} } — the
// EXACT shape kernel/subd.mjs's own subdivideCatmullClark already expects,
// so a fresh primitive cage is immediately valid input to it with zero
// adaptation.
//
// FACET COUNT is the one typed option every one of these 4 commands takes
// ("primitive cages with facet-count chips") — a single
// integer controlling how finely the cage is subdivided BEFORE any
// Catmull-Clark refinement ever runs (this is control-net density, not
// display resolution — see the app's own superbDisplayMesh for
// the separate, later, adaptive refinement-level choice).
//
// NO CREASES ON ANY PRIMITIVE BY DEFAULT — a real, deliberate, honestly-
// stated choice, not an oversight: a facet-count-1 SuperBBox subdivided
// with zero creases converges toward a rounded, sphere-like blob (the
// textbook "a cube smooths into a sphere" Catmull-Clark demonstration) —
// exactly the comprehension device the Box/Smooth toggle
// exists to show a student. Creasing every box edge by default would hide
// that demonstration behind an already-sharp shape. Semi-sharp/full-crease
// EDITING is a later milestone's scope (this one explicitly does not
// build vertex/edge/face selection or editing), so there is no user-
// reachable way to change this yet either way.

// Dedupes coincident vertices by a rounded-coordinate key so a cage built
// from several independently-parametrized face grids (a box's 6 faces)
// welds into one real, watertight manifold cage rather than 6 disjoint
// islands — the SAME "weld by rounded key" idiom this app's own dedupe/
// snap-candidate code already uses elsewhere, applied here at cage-
// construction time instead of after the fact.
// ⚠ ROUNDED-KEY HAZARD, reachable from any revolve conversion:
// `(-1.4695761589768238e-15).
// toFixed(6)` returns the STRING `"-0.000000"`, genuinely different from
// plain `(0).toFixed(6)` -> `"0.000000"` — so two points that coincide to
// well within 1e-6 (an actual seam pair differing by ~1.5e-15, twelve
// orders of magnitude tighter, produced by an ordinary trig-cancellation
// residue at v=2*PI on a revolved surface) could silently weld into TWO
// separate vertices purely because one sample landed a hair on the
// negative side of exact zero — quietly breaking the "watertight, not 6
// disjoint islands" guarantee this function's own header comment promises
// (confirmed live: a 5x9-control-point revolve surface's own seam column
// failed to weld at all, 45 vertices surviving where 40 should have,
// before this fix). Fixed by rounding to the SAME 6-decimal precision
// FIRST (as a number, not a string) and explicitly canonicalizing a
// resulting -0 to +0 before formatting — every EXISTING caller (SuperBBox/
// Sphere/Cylinder/Plane) is unaffected: their own coordinates never
// produce this exact "near-zero from the negative side" residue in the
// first place (built from clean +/- half-extent arithmetic, not trig
// cancellation), so this is a strict superset fix, not a behavior change
// for any case that worked before.
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

// SUPERBBOX (Rhino: SubDBox) — an axis-aligned box cage, `facets`
// subdivisions per edge on every one of its 6 faces (facets=1 is the
// plain 8-vertex/6-face/12-edge cube). Each face is its own uniform
// (facets+1)x(facets+1) point grid, welded at shared edges/corners via
// the shared vertex welder above — proven watertight (zero boundary
// edges) directly in this module's own test file via buildTopology, not
// assumed from construction alone.
// THE SECOND COUNT RUNS ALONG Z, and the six faces do not all mean the same
// thing by it: a side face is (around) x (up), while the two caps are (around)
// x (around) in both of their own directions. Getting that wrong does not
// produce a wrong-looking box, it produces a LEAKING one — two faces meeting
// along an edge whose two sides carry different point counts have nothing to
// weld. So each face names its own pair, and every shared edge is named the
// same on both sides of it. `facetsV` omitted reproduces the previous cage
// bit-for-bit rather than needing a migration, exactly as the torus's own
// second count already does.
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

// SUPERBSPHERE (Rhino: SubDSphere) — reuses superbBoxCage's own topology
// wholesale (a "boxy sphere"/cube-sphere cage — a defensible, simple v1
// construction, not required to match Rhino's own internal SubDSphere
// construction exactly, per the parallel/sovereign-
// implementation decision), then projects every vertex radially onto the
// true sphere. facets=1 gives the familiar 6-face "beach ball" cage.
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

// SUPERBCYLINDER (Rhino: SubDCylinder) — `facetsV + 1` rings of `facets`
// vertices, evenly spaced from z=center.z to z=center.z+height, the side
// quads between them, plus one N-GON cap at each end (ngons are legal
// Catmull-Clark faces — computeFacePoint/the vertex rules in kernel/subd.mjs
// already handle any face size, no special-casing needed here). `facets` is
// the radial count (>=3) and `facetsV` the height count: the two directions
// are separately meaningful here for the same reason they are on a torus, and
// a form whose profile is edited along its height cannot be shaped at all
// through the radial count alone. `facetsV` omitted means one ring pair,
// which is what this cage was before it had a second count.
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

// SUPERBPLANE (Rhino: SubDPlane) — a flat, OPEN (has a real boundary —
// genuinely different topology from the 3 closed-solid cages above) NxN
// grid of quads in the cage's own local XY plane at the center's own Z.
// A single typed facet count sets both grid axes uniformly (no separate
// U/V facet counts this v1 — matching the milestone's own single "typed
// facet-count option" per command, not DivideSrf's own two-axis "UxV"
// convention).
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

// SUPERBCONE (no direct Rhino SubD-primitive precedent — Rhino's own SubD
// toolset has no built-in cone; this is a genuine new construction, not a
// port) — a ring of `facets` base vertices plus ONE apex vertex, exactly
// SuperBCylinder's own ring construction with the top ring collapsed to a
// single point. This is the real "pole" case named in this module's own
// milestone doc: unlike SuperBSphere's own cube-topology construction
// (which reuses the box's welded-corner grid and has no singular vertex at
// all — every vertex there keeps an ordinary valence-3-or-4 corner/edge
// role, unlike SuperBBox), a cone's apex is a genuine EXTRAORDINARY vertex
// of valence `facets` — `facets` triangles fan into one shared point, the
// same "one control row collapsed to a point" idea the NURBS revolve pole
// already uses one representation over (kernel/primitives.mjs's own
// on-axis pole handling), just expressed here as cage TOPOLOGY instead of
// a degenerate NURBS control row. `computeVertexPoint`/`smoothVertexRule`
// (kernel/subd.mjs) already generalize over any vertex valence and any
// face size (the base cap is itself an N-gon, exactly like a cylinder's own
// cap) — nothing about this construction needs new subdivision math, only
// a cage with a genuine extraordinary vertex to feed it, which no earlier
// primitive here has produced until now.
// The height count adds rings BELOW the apex, never at it: the apex is a
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
    faces.push([last + i, last + i1, apexIdx]); // side triangle — the SAME winding a SuperBCylinder side quad [i0,i1,j1,j0] would have if j0 and j1 both collapsed onto the apex
  }
  const baseFace = []; for (let i = n - 1; i >= 0; i--) baseFace.push(i); // reversed for outward (-Z) winding, matching SuperBCylinder's own bottom cap
  faces.push(baseFace);
  return { vertices, faces, creases: {} };
}

// SUPERBTORUS — the genus-1 case, and genuinely NOT a variation on the
// box/sphere/cylinder pattern: those three all either weld several
// independently-parametrized patches at shared corners (box/sphere) or
// leave real open boundaries at their two flat ends (cylinder, absent its
// own caps). A torus has NO boundary anywhere and must close up in BOTH
// its own ring (major) and tube (minor) directions at once — the doubly-
// periodic case this module's own header comment already flags as the hard
// one. The construction below sidesteps the welder entirely (unlike
// box/sphere, which weld several independent face grids together after
// the fact): every vertex is generated ONCE into a single flat n x n
// array, addressed with a WRAPPING modulo index in both directions —
// vertex (i,j) and vertex (i+n,j) are, by construction, literally the
// SAME array slot, so the ring seam and the tube seam are both closed
// EXACTLY, with no rounded-coordinate coincidence to rely on (the box/
// sphere welder's whole reason to exist — two independently-parametrized
// patches meeting only approximately, in floating point — never arises
// here, since there is only ever one patch). The resulting genus is a
// direct, checkable consequence of this closure, not asserted: a full
// nU x nV toroidal grid has V=nU*nV, F=nU*nV quads, E=2*nU*nV (nU*nV
// ring-direction edges + nU*nV tube-direction edges, each already closed by
// the same wraparound indexing) — so chi = V-E+F = 0, and chi=2-2*genus
// makes genus exactly 1 for ANY pair of counts, not just a particular one.
//
// THE TWO DIRECTIONS ARE INDEPENDENT, and this is the one place in this
// module where that is true. Every other cage builder here takes a single
// `facets` because its own construction genuinely has one density knob (a
// welded box/sphere grid, a cylinder's ring). A torus has two real,
// separately-meaningful directions — around the ring (U) and around the
// tube (V) — exactly like the U and V control-point counts a NURBS surface
// already exposes through Surface Rebuild, so it takes two. `facetsV`
// defaults to `facetsU` when omitted, which makes the old single-count
// call shape (and every stored `facets` param predating this) reproduce
// its previous cage bit-for-bit rather than needing migration.
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
      // Same corner ORDER as SuperBCylinder's own side quad [i0,i1,j1,j0]
      // (curr-ring/next-ring at curr-tube-angle, then next-ring/curr-ring at
      // next-tube-angle) — outward-normal winding by the identical construction,
      // generalized from one periodic direction (cylinder's ring) to two.
      faces.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
    }
  }
  return { vertices, faces, creases: {} };
}

// SUPERBELLIPSOID — reuses SuperBBoxCage's own welded topology wholesale,
// exactly like SuperBSphereCage already does (a "boxy ellipsoid" cage,
// the identical simple v1 construction), then projects every vertex
// RADIALLY onto the true ellipsoid along the line from center through that
// vertex — the direct affine generalization of SuperBSphereCage's own
// "project onto radius R" step (SuperBSphereCage is exactly the case
// radii=[R,R,R] of this same projection: with rx=ry=rz=R, the scale factor
// s below reduces algebraically to R/|d|, SuperBSphereCage's own formula,
// bit-for-bit). Solving for the scale factor s such that
// center + s*(x-center,y-center,z-center) satisfies the true ellipsoid
// equation ((s*dx)/rx)^2+((s*dy)/ry)^2+((s*dz)/rz)^2=1 gives
// s = 1/sqrt((dx/rx)^2+(dy/ry)^2+(dz/rz)^2) directly — exact for every
// vertex regardless of where on the box it started (a face-center sample,
// not just a true box corner), the same "affine map of a unit sphere"
// identity this app's own NURBS Ellipsoid primitive already relies on one
// representation over (a non-uniform scale of a revolved half-circle).
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
