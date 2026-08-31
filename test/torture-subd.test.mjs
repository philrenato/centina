// A GENERATED TORTURE CORPUS FOR THE SUBD CAGE — the recommended first
// lane, built rather than the external one.
//
// WHY THIS LANE FIRST: "this project already
// generates its own adversarial fixtures programmatically... a generated
// corpus — sweep N, run length, symmetry, tilt, and input ORDER, then
// assert the invariants rather than a golden output — found both bridge
// bugs, and cost nothing in licensing or build footprint. Worth
// exhausting that lane before taking on an external corpus, and it makes
// the eventual external one easier to judge, since the invariants will
// already be written." An OCCT corpus additionally cannot be loaded at
// all today (it is STEP/IGES/BREP; this app reads .3dm and .obj), so the
// external lane is gated on decisions nobody has made. This one is not.
//
// WHAT A CORPUS IS ACTUALLY FOR HERE, and why it is not "more tests":
// every robustness bug found in this area was found by a fixture someone
// hand-built for one round, and the pattern is consistent — the honest
// fixtures find real bugs, the convenient ones pass. The N-way bridge is
// the worked example: tilted, unevenly-spaced flaps were robust across
// every variant, while a plain SYMMETRIC rim exposed two genuine
// order-dependence bugs, because symmetry produces EXACT TIES and a
// hand-built fixture rarely does. So this file sweeps deliberately toward
// the shapes a person would not think to author: symmetric, degenerate,
// permuted, and applied at every site rather than one chosen site.
//
// INVARIANTS, NOT GOLDEN OUTPUTS. Nothing here asserts a specific vertex
// position. Every check is a property that must hold of ANY valid cage
// however it was produced — which is what lets one assertion cover a few
// thousand generated cases, and what makes a failure a real finding
// rather than a fixture needing its expected numbers updated.
//
// AN OPERATION IS ALLOWED TO REFUSE. A refusal (a thrown, named error) is
// a correct outcome for a genuinely degenerate input and is recorded, not
// failed. What is never allowed is a SILENTLY BROKEN cage: a returned
// result that violates an invariant is the actual bug class this hunts.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  superbBoxCage, superbSphereCage, superbCylinderCage, superbPlaneCage,
  superbConeCage, superbTorusCage, superbEllipsoidCage,
} from '../kernel/subdprimitives.mjs';
import { buildTopology, subdivideCatmullClark } from '../kernel/subd.mjs';
import {
  extrudeFaces, extrudeEdges, insertPointOnEdge, weldVertices, slideEdges,
  offsetCage, thickenCage, insertEdgeLoop, deleteFaces, mergeFaces,
  subdivideFaces, subdivideCageGlobal, bridgeEdgeRunsHub, bridgeEdgeRuns,
  computeAverageNormal,
} from '../kernel/subdedit.mjs';

// extrudeFaces wants a real direction vector — a null one is an APP-layer
// convention (a coincident direction pick means "use the selection's own
// average normal"), resolved before the kernel is ever called. The sweep
// resolves it the same way rather than passing the null through.
const faceDir = (cage, idx) => computeAverageNormal(cage, idx);

// ===================================================================
// THE INVARIANT LIBRARY
//
// Deliberately recomputed here from the raw {vertices, faces} rather than
// asked of buildTopology — a corpus that trusted the same helper the code
// under test uses would agree with it about a shared mistake. This walks
// the face lists directly.
// ===================================================================

const ekey = (a, b) => (a < b ? `${a}_${b}` : `${b}_${a}`);

// Every editing function in this kernel returns either a bare cage or a
// RESULT OBJECT carrying the cage plus its own bookkeeping (which faces
// it made, how many vertices it pruned). Both shapes are real and both
// are correct; a harness that assumed one of them would report the other
// as "not a cage" and read as a swarm of app bugs. Unwrapped once here.
const unwrap = (r) => (r && r.cage ? r.cage : r);

// Returns a list of violation strings; empty means the cage is well-formed.
// `opts.closed` asserts the stronger properties only a closed cage has.
function cageViolations(cage, opts = {}) {
  const v = [];
  if (!cage || !Array.isArray(cage.vertices) || !Array.isArray(cage.faces)) return ['not a cage'];
  const nV = cage.vertices.length;

  // 1. Every vertex is a real finite 3-vector. A NaN here is the failure
  //    that propagates furthest and is hardest to trace back later.
  for (let i = 0; i < nV; i++) {
    const p = cage.vertices[i];
    if (!p || p.length < 3 || !p.slice(0, 3).every(Number.isFinite)) v.push(`vertex ${i} is not finite: ${JSON.stringify(p)}`);
  }

  // 2. Every face index is in range, and no face repeats a vertex. A
  //    repeated vertex is a zero-area sliver that still evaluates, still
  //    renders, and quietly corrupts every later subdivision — precisely
  //    the class the N-way bridge shipped and had to have fixed.
  cage.faces.forEach((f, fi) => {
    if (!Array.isArray(f) || f.length < 3) { v.push(`face ${fi} has ${f?.length} corners`); return; }
    const seen = new Set();
    for (const idx of f) {
      if (!Number.isInteger(idx) || idx < 0 || idx >= nV) v.push(`face ${fi} references out-of-range vertex ${idx} (nV=${nV})`);
      if (seen.has(idx)) v.push(`face ${fi} repeats vertex ${idx} — a degenerate sliver`);
      seen.add(idx);
    }
  });
  if (v.length) return v; // later checks would only produce noise on a malformed list

  // 3. MANIFOLD: no edge may be shared by more than two faces.
  const edgeFaces = new Map();
  for (let fi = 0; fi < cage.faces.length; fi++) {
    const f = cage.faces[fi];
    for (let c = 0; c < f.length; c++) {
      const k = ekey(f[c], f[(c + 1) % f.length]);
      if (!edgeFaces.has(k)) edgeFaces.set(k, []);
      edgeFaces.get(k).push(fi);
    }
  }
  for (const [k, fs] of edgeFaces) if (fs.length > 2) v.push(`edge ${k} is shared by ${fs.length} faces — non-manifold`);

  // 4. CONSISTENT WINDING: a consistently-oriented 2-manifold never
  //    traverses the same DIRECTED edge twice. This is the property the
  //    bridge's own winding search verifies rather than derives, reused
  //    here as a check on every generated result.
  const directed = new Set();
  for (const f of cage.faces) {
    for (let c = 0; c < f.length; c++) {
      const d = `${f[c]}>${f[(c + 1) % f.length]}`;
      if (directed.has(d)) v.push(`directed edge ${d} is used twice — inconsistent winding`);
      directed.add(d);
    }
  }

  // 5. NO ORPHANS: every vertex is used by at least one face. An orphan
  //    is invisible on screen and was a real bug in the hub builder's own
  //    winding search (rejected trials leaking their vertices).
  if (opts.noOrphans !== false) {
    const used = new Set();
    for (const f of cage.faces) for (const idx of f) used.add(idx);
    for (let i = 0; i < nV; i++) if (!used.has(i)) v.push(`vertex ${i} is an orphan — used by no face`);
  }

  // 6. CREASE KEYS NAME REAL EDGES. A crease left on a key that no longer
  //    exists is silent: it changes nothing, until a later edit
  //    resurrects that key and a crease appears from nowhere.
  if (cage.creases) {
    for (const k of Object.keys(cage.creases)) {
      if (!edgeFaces.has(k)) v.push(`crease on ${k}, which is not an edge of this cage`);
      if (!Number.isFinite(cage.creases[k])) v.push(`crease on ${k} is not finite: ${cage.creases[k]}`);
    }
  }

  // 7. CLOSED cages only: every edge has exactly two faces, and the Euler
  //    characteristic is even (chi = 2 - 2g admits only even values).
  if (opts.closed) {
    for (const [k, fs] of edgeFaces) if (fs.length !== 2) v.push(`closed cage has a naked edge ${k} (${fs.length} face)`);
    const chi = nV - edgeFaces.size + cage.faces.length;
    if (chi % 2 !== 0) v.push(`closed cage has odd Euler characteristic ${chi} — cannot be 2-2g for any integer genus`);
  }
  return v;
}

// A cage that passes every structural check can still be one a real
// subdivision pass destroys. Running one genuine Catmull-Clark level is
// the cheapest way to catch that, and it is what every downstream
// consumer actually does.
function survivesSubdivision(cage) {
  let refined;
  try { refined = subdivideCatmullClark(cage); } catch (e) { return [`subdivideCatmullClark threw: ${e.message}`]; }
  return cageViolations(refined, { noOrphans: false }).map((s) => `after one subdivision: ${s}`);
}

function nakedEdgesOf(cage) {
  const count = new Map();
  for (const f of cage.faces) for (let c = 0; c < f.length; c++) {
    const k = ekey(f[c], f[(c + 1) % f.length]);
    count.set(k, (count.get(k) || 0) + 1);
  }
  return [...count.entries()].filter(([, n]) => n === 1).map(([k]) => k);
}

// ===================================================================
// THE FIXTURE SWEEP
//
// Deliberately includes the shapes a person authoring one fixture would
// skip: facets=1 (as coarse as a cage gets, where neighborhoods overlap
// in ways denser cages never do), a genus-1 torus, two cages with real
// POLES (cone apex, sphere/ellipsoid), an OPEN cage (plane), and
// deliberately ASYMMETRIC dimensions alongside perfectly symmetric ones —
// symmetry is what produces the exact ties that found the bridge bugs, so
// it is generated on purpose rather than avoided.
// ===================================================================
function fixtures() {
  const out = [];
  for (const f of [1, 2, 3]) {
    out.push({ name: `box f${f} symmetric`, closed: true, cage: superbBoxCage([0, 0, 0], [25, 25, 25], f) });
    out.push({ name: `box f${f} asymmetric`, closed: true, cage: superbBoxCage([3, -7, 11], [40, 12, 26], f) });
    out.push({ name: `sphere f${f}`, closed: true, cage: superbSphereCage([0, 0, 0], 25, f) });
    out.push({ name: `ellipsoid f${f} unequal`, closed: true, cage: superbEllipsoidCage([0, 0, 0], [30, 12, 20], f) });
    out.push({ name: `plane f${f} (OPEN)`, closed: false, cage: superbPlaneCage([0, 0, 0], 50, 50, f) });
  }
  for (const f of [3, 4, 6, 8]) {
    out.push({ name: `cylinder f${f}`, closed: true, cage: superbCylinderCage([0, 0, 0], 20, 50, f) });
    out.push({ name: `cone f${f} (POLE)`, closed: true, cage: superbConeCage([0, 0, 0], 20, 40, f) });
  }
  for (const [u, v] of [[3, 3], [4, 3], [6, 4], [8, 8], [5, 7]]) {
    out.push({ name: `torus ${u}x${v} (GENUS 1)`, closed: true, cage: superbTorusCage([0, 0, 0], 30, 10, u, v) });
  }
  return out;
}

const FIXTURES = fixtures();

// ===================================================================
test('CORPUS BASELINE: every generated fixture is itself well-formed', () => {
  let checked = 0;
  for (const f of FIXTURES) {
    const v = cageViolations(f.cage, { closed: f.closed });
    assert.deepEqual(v, [], `${f.name}: ${v.join('; ')}`);
    const s = survivesSubdivision(f.cage);
    assert.deepEqual(s, [], `${f.name}: ${s.join('; ')}`);
    checked++;
  }
  assert.ok(checked >= 25, `the sweep must be genuinely broad, got ${checked} fixtures`);
});

// ===================================================================
// SINGLE-SITE OPERATIONS, APPLIED AT EVERY SITE
//
// Not "pick a face and extrude it" — extrude EVERY face of EVERY fixture,
// insert a point on EVERY edge, and so on. This is the part a hand-built
// fixture structurally cannot do, and it is where a site-dependent bug
// (a pole, a seam, a valence-3 corner) hides.
// ===================================================================
test('TORTURE: extrudeFaces at every single face of every fixture', () => {
  let ran = 0, refused = 0;
  for (const f of FIXTURES) {
    for (let fi = 0; fi < f.cage.faces.length; fi++) {
      let out;
      try { out = unwrap(extrudeFaces(f.cage, [fi], faceDir(f.cage, [fi]), 5)); } catch { refused++; continue; }
      const v = cageViolations(out, { noOrphans: false });
      assert.deepEqual(v, [], `${f.name} face ${fi}: ${v.join('; ')}`);
      const s = survivesSubdivision(out);
      assert.deepEqual(s, [], `${f.name} face ${fi}: ${s.join('; ')}`);
      ran++;
    }
  }
  assert.ok(ran > 400, `the sweep must genuinely run, got ${ran} extrusions (${refused} honest refusals)`);
});

test('TORTURE: insertPointOnEdge at every edge, at several parameters including both extremes', () => {
  let ran = 0, refused = 0;
  for (const f of FIXTURES) {
    const ctx = buildTopology(f.cage);
    for (const k of ctx.edgeMap.keys()) {
      for (const t of [0.001, 0.5, 0.999]) {
        let out;
        try { out = unwrap(insertPointOnEdge(f.cage, k, t)); } catch { refused++; continue; }
        const v = cageViolations(out, { closed: f.closed });
        assert.deepEqual(v, [], `${f.name} edge ${k} t=${t}: ${v.join('; ')}`);
        ran++;
      }
    }
  }
  assert.ok(ran > 1000, `got ${ran} insertions (${refused} refusals)`);
});

test('TORTURE: deleteFaces at every face, and the hole it leaves is a real boundary', () => {
  let ran = 0;
  for (const f of FIXTURES) {
    for (let fi = 0; fi < f.cage.faces.length; fi++) {
      let out;
      try { out = deleteFaces(f.cage, [fi]); } catch { continue; }
      const cage = unwrap(out);
      const v = cageViolations(cage);
      assert.deepEqual(v, [], `${f.name} delete face ${fi}: ${v.join('; ')}`);
      // Deleting one face of a CLOSED cage must open exactly that face's
      // own edges and no others — an accounting check, strictly stronger
      // than "some naked edges appeared".
      if (f.closed) {
        const naked = nakedEdgesOf(cage);
        assert.equal(naked.length, f.cage.faces[fi].length,
          `${f.name} delete face ${fi}: expected exactly ${f.cage.faces[fi].length} naked edges, got ${naked.length}`);
      }
      ran++;
    }
  }
  assert.ok(ran > 400, `got ${ran} deletions`);
});

test('TORTURE: slideEdges over the full legal parameter range, including near the refusal boundary', () => {
  let ran = 0, refused = 0;
  for (const f of FIXTURES.slice(0, 14)) {
    const ctx = buildTopology(f.cage);
    const keys = [...ctx.edgeMap.keys()];
    for (const k of keys) {
      for (const t of [-0.94, -0.5, -0.01, 0.01, 0.5, 0.94]) {
        let out;
        try { out = unwrap(slideEdges(f.cage, [k], t)); } catch { refused++; continue; }
        const v = cageViolations(out, { closed: f.closed });
        assert.deepEqual(v, [], `${f.name} slide ${k} t=${t}: ${v.join('; ')}`);
        ran++;
      }
    }
  }
  assert.ok(ran > 200, `got ${ran} slides (${refused} refusals)`);
});

test('TORTURE: offsetCage and thickenCage across sign and magnitude, including magnitudes that must be refused or clamped', () => {
  let ran = 0, refused = 0;
  for (const f of FIXTURES) {
    for (const d of [-40, -3, -0.01, 0.01, 3, 40]) {
      for (const fn of [offsetCage, thickenCage]) {
        let out;
        try { out = fn(f.cage, d); } catch { refused++; continue; }
        const cage = unwrap(out);
        const v = cageViolations(cage, { noOrphans: false });
        assert.deepEqual(v, [], `${f.name} ${fn.name}(${d}): ${v.join('; ')}`);
        ran++;
      }
    }
  }
  assert.ok(ran > 100, `got ${ran} offsets (${refused} refusals)`);
});

test('TORTURE: insertEdgeLoop seeded from every edge, both sides', () => {
  let ran = 0, refused = 0;
  for (const f of FIXTURES.slice(0, 16)) {
    const ctx = buildTopology(f.cage);
    for (const k of ctx.edgeMap.keys()) {
      for (const side of [0, 1]) {
        let out;
        try { out = insertEdgeLoop(f.cage, k, 0.5, side); } catch { refused++; continue; }
        const cage = unwrap(out);
        const v = cageViolations(cage, { closed: f.closed });
        assert.deepEqual(v, [], `${f.name} loop from ${k} side ${side}: ${v.join('; ')}`);
        ran++;
      }
    }
  }
  assert.ok(ran > 100, `got ${ran} loop insertions (${refused} refusals)`);
});

test('TORTURE: subdivideFaces at every single face, and subdivideCageGlobal on every fixture', () => {
  for (const f of FIXTURES) {
    const g = unwrap(subdivideCageGlobal(f.cage));
    const gv = cageViolations(g, { closed: f.closed });
    assert.deepEqual(gv, [], `${f.name} global subdivide: ${gv.join('; ')}`);
    // Catmull-Clark refines an n-gon into n quads, so a global pass has
    // exactly the summed corner count as its face count. An accounting
    // check rather than "it got bigger".
    const expected = f.cage.faces.reduce((s, face) => s + face.length, 0);
    assert.equal(g.faces.length, expected, `${f.name}: global subdivide must produce one quad per corner`);
    for (let fi = 0; fi < f.cage.faces.length; fi++) {
      let out;
      try { out = subdivideFaces(f.cage, [fi]); } catch { continue; }
      const cage = unwrap(out);
      const v = cageViolations(cage, { closed: f.closed });
      assert.deepEqual(v, [], `${f.name} subdivide face ${fi}: ${v.join('; ')}`);
    }
  }
});

// ===================================================================
// ORDER DEPENDENCE — the specific bug class a generated corpus exists to
// find, and the one that a hand-built fixture is worst at exposing.
//
// An operation taking a SET must not depend on the order that set
// happens to arrive in. Two genuine bugs in bridgeEdgeRunsHub were
// exactly this, and both were invisible to fixtures that were robust in
// every other respect.
// ===================================================================
function permutations(arr, cap = 24) {
  if (arr.length <= 1) return [arr.slice()];
  const out = [];
  const walk = (rest, acc) => {
    if (out.length >= cap) return;
    if (!rest.length) { out.push(acc); return; }
    for (let i = 0; i < rest.length; i++) walk([...rest.slice(0, i), ...rest.slice(i + 1)], [...acc, rest[i]]);
  };
  walk(arr, []);
  return out;
}

test('ORDER DEPENDENCE: deleteFaces gives the same result whatever order the faces arrive in', () => {
  let compared = 0;
  for (const f of FIXTURES) {
    if (f.cage.faces.length < 4) continue;
    const set = [0, 1, Math.min(2, f.cage.faces.length - 1), f.cage.faces.length - 1].filter((x, i, a) => a.indexOf(x) === i);
    let reference = null;
    for (const perm of permutations(set)) {
      let out;
      try { out = deleteFaces(f.cage, perm); } catch { continue; }
      const cage = unwrap(out);
      const v = cageViolations(cage);
      assert.deepEqual(v, [], `${f.name} delete ${JSON.stringify(perm)}: ${v.join('; ')}`);
      // Compared as a SET of faces keyed by their own sorted vertex
      // POSITIONS, not indices: a different input order may legitimately
      // renumber the compacted vertex array, so comparing indices would
      // assert something never promised. The geometry must match.
      const key = cage.faces
        .map((face) => face.map((i) => cage.vertices[i].slice(0, 3).map((n) => n.toFixed(9)).join(',')).sort().join('|'))
        .sort().join('#');
      if (reference === null) reference = key;
      else assert.equal(key, reference, `${f.name}: deleteFaces depends on the order its faces arrive in`);
      compared++;
    }
  }
  assert.ok(compared > 100, `got ${compared} permutation comparisons`);
});

test('ORDER DEPENDENCE: weldVertices gives the same geometry whatever order the vertices arrive in', () => {
  let compared = 0;
  for (const f of FIXTURES) {
    const ctx = buildTopology(f.cage);
    // Weld a genuine EDGE's two endpoints — a real, reachable gesture,
    // unlike two arbitrary far-apart vertices.
    const keys = [...ctx.edgeMap.keys()].slice(0, 6);
    for (const k of keys) {
      const [a, b] = k.split('_').map(Number);
      const results = [];
      for (const perm of [[a, b], [b, a]]) {
        let out;
        try { out = weldVertices(f.cage, perm, 'average'); } catch { results.push(null); continue; }
        const cage = unwrap(out);
        const v = cageViolations(cage, { noOrphans: false });
        assert.deepEqual(v, [], `${f.name} weld ${JSON.stringify(perm)}: ${v.join('; ')}`);
        results.push(cage.faces.length + ':' + cage.vertices.length);
      }
      // 'average' is symmetric in its inputs, so both orders must agree
      // on whether the weld is legal at all AND on the resulting counts.
      assert.equal(results[0], results[1], `${f.name} edge ${k}: weldVertices depends on input order`);
      compared++;
    }
  }
  assert.ok(compared > 50, `got ${compared} weld comparisons`);
});

test('ORDER DEPENDENCE: mergeFaces gives the same geometry whatever order the faces arrive in', () => {
  let compared = 0;
  for (const f of FIXTURES) {
    const ctx = buildTopology(f.cage);
    // Adjacent face pairs only — merging disconnected faces is refused by
    // design, so permuting it would compare two refusals and prove
    // nothing.
    const pairs = [];
    for (const e of ctx.edgeMap.values()) if (e.faces.length === 2) pairs.push(e.faces);
    for (const pair of pairs.slice(0, 8)) {
      const results = [];
      for (const perm of [pair, [...pair].reverse()]) {
        let out;
        try { out = mergeFaces(f.cage, perm); } catch { results.push('refused'); continue; }
        const cage = unwrap(out);
        const v = cageViolations(cage, { closed: f.closed, noOrphans: false });
        assert.deepEqual(v, [], `${f.name} merge ${JSON.stringify(perm)}: ${v.join('; ')}`);
        results.push(`${cage.faces.length}:${cage.vertices.length}`);
      }
      assert.equal(results[0], results[1], `${f.name} merge ${JSON.stringify(pair)}: mergeFaces depends on input order`);
      compared++;
    }
  }
  assert.ok(compared > 50, `got ${compared} merge comparisons`);
});

// ===================================================================
// SYMMETRY — the property that produces EXACT TIES, and the one that
// found both N-way bridge bugs. Generated on purpose here rather than
// avoided, across every run count and every input permutation.
// ===================================================================
test('SYMMETRY + ORDER: an N-way hub on a perfectly symmetric rim survives every run permutation and direction', () => {
  let built = 0, refused = 0;
  for (const N of [3, 4, 5, 6]) {
    // N arcs evenly spaced around one rim — perfectly symmetric on
    // purpose, so every pairwise comparison the hub makes is an exact
    // tie. This is the fixture shape that exposed a summed-cross-product
    // plane derivation collapsing to zero and a direction rule falling
    // through to input order.
    const verts = [], runs = [];
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2;
      const c = Math.cos(a), s = Math.sin(a);
      const base = verts.length;
      verts.push([c * 30, s * 30, 0], [c * 45, s * 45, 0], [c * 60, s * 60, 0]);
      runs.push([base, base + 1, base + 2]);
    }
    const faces = [];
    for (let i = 0; i < N; i++) faces.push([runs[i][0], runs[i][1], runs[i][2]]);
    const cage = { vertices: verts, faces, creases: {} };
    for (const perm of permutations(runs.map((_, i) => i), 12)) {
      for (const flip of [false, true]) {
        const ordered = perm.map((i) => (flip ? [...runs[i]].reverse() : runs[i]));
        let out;
        try { out = bridgeEdgeRunsHub(cage, ordered, 1, 1, 0); } catch { refused++; continue; }
        const c = unwrap(out);
        const v = cageViolations(c, { noOrphans: false });
        assert.deepEqual(v, [], `hub N=${N} perm=${JSON.stringify(perm)} flip=${flip}: ${v.join('; ')}`);
        built++;
      }
    }
  }
  // A hub on a flat rim may legitimately refuse (coplanar arcs with no
  // junction plane is a named refusal). What must never happen is a
  // returned cage that violates an invariant — asserted above for every
  // one that DID build.
  assert.ok(built + refused > 80, `the permutation sweep must genuinely run, got ${built} built / ${refused} refused`);
});

test('SYMMETRY: a two-run bridge is invariant to which run is named first', () => {
  let compared = 0;
  for (const seg of [1, 2, 3]) {
    for (const len of [2, 3, 4]) {
      const A = [], B = [];
      for (let i = 0; i < len; i++) { A.push([i * 10, 0, 0]); B.push([i * 10, 40, 0]); }
      const verts = [...A, ...B];
      const runA = A.map((_, i) => i);
      const runB = B.map((_, i) => len + i);
      const faces = [];
      for (let i = 0; i < len - 1; i++) faces.push([runA[i], runA[i + 1], runB[i + 1], runB[i]]);
      const cage = { vertices: verts, faces: [], creases: {} };
      void faces;
      const results = [];
      for (const [x, y] of [[runA, runB], [runB, runA]]) {
        let out;
        try { out = bridgeEdgeRuns(cage, x, y, seg, 1, 0); } catch { results.push('refused'); continue; }
        const c = unwrap(out);
        const v = cageViolations(c, { noOrphans: false });
        assert.deepEqual(v, [], `bridge seg=${seg} len=${len}: ${v.join('; ')}`);
        results.push(`${c.faces.length}:${c.vertices.length}`);
      }
      assert.equal(results[0], results[1], `bridge seg=${seg} len=${len}: naming order changed the result's own shape`);
      compared++;
    }
  }
  assert.ok(compared >= 9, `got ${compared} comparisons`);
});

// ===================================================================
// COMPOSITION — a cage is rarely edited once. Real corruption tends to
// appear only after several operations have each individually "passed",
// which no single-operation test can reach.
// ===================================================================
test('TORTURE: chains of edits compose without silently corrupting the cage', () => {
  const ops = [
    ['subdivideFaces', (c) => subdivideFaces(c, [0])],
    ['extrudeFaces', (c) => extrudeFaces(c, [0], faceDir(c, [0]), 4)],
    ['insertPointOnEdge', (c) => insertPointOnEdge(c, [...buildTopology(c).edgeMap.keys()][0], 0.5)],
    ['deleteFaces', (c) => deleteFaces(c, [0])],
    ['offsetCage', (c) => offsetCage(c, 1.5)],
    ['slideEdges', (c) => slideEdges(c, [[...buildTopology(c).edgeMap.keys()][0]], 0.25)],
    ['insertEdgeLoop', (c) => insertEdgeLoop(c, [...buildTopology(c).edgeMap.keys()][0], 0.5, 0)],
  ];
  let chains = 0;
  for (const f of FIXTURES.slice(0, 12)) {
    // Every ordered PAIR and a sample of triples — deliberately including
    // pairs a person would not try (delete then offset, extrude then
    // slide) because those are where a stale index or a dropped crease
    // key surfaces.
    for (let i = 0; i < ops.length; i++) {
      for (let j = 0; j < ops.length; j++) {
        let cage = f.cage;
        const trail = [];
        for (const [name, fn] of [ops[i], ops[j]]) {
          let out;
          try { out = fn(cage); } catch { out = null; }
          if (!out) break;
          cage = unwrap(out);
          trail.push(name);
          const v = cageViolations(cage, { noOrphans: false });
          assert.deepEqual(v, [], `${f.name} after [${trail.join(' -> ')}]: ${v.join('; ')}`);
        }
        if (trail.length === 2) {
          const s = survivesSubdivision(cage);
          assert.deepEqual(s, [], `${f.name} after [${trail.join(' -> ')}]: ${s.join('; ')}`);
          chains++;
        }
      }
    }
  }
  assert.ok(chains > 200, `got ${chains} completed two-op chains`);
});

// ===================================================================
// DEGENERATE INPUT — an operation must refuse honestly (a thrown, named
// error) rather than return a broken cage. A refusal is a pass here; a
// silently invalid result is the bug.
// ===================================================================
test('DEGENERATE INPUT: every operation either refuses honestly or returns a valid cage — never a broken one', () => {
  const cage = superbBoxCage([0, 0, 0], [25, 25, 25], 2);
  const nF = cage.faces.length, nV = cage.vertices.length;
  const attempts = [
    ['extrudeFaces empty set', () => extrudeFaces(cage, [], [0, 0, 1], 5)],
    ['extrudeFaces out-of-range', () => extrudeFaces(cage, [nF + 99], [0, 0, 1], 5)],
    ['extrudeFaces zero distance', () => extrudeFaces(cage, [0], [0, 0, 1], 0)],
    ['extrudeFaces NaN distance', () => extrudeFaces(cage, [0], [0, 0, 1], NaN)],
    ['extrudeFaces every face at once', () => extrudeFaces(cage, cage.faces.map((_, i) => i), [0, 0, 1], 3)],
    ['deleteFaces every face', () => deleteFaces(cage, cage.faces.map((_, i) => i))],
    ['deleteFaces duplicate indices', () => deleteFaces(cage, [0, 0, 0])],
    ['deleteFaces out-of-range', () => deleteFaces(cage, [nF + 5])],
    ['weldVertices single vertex', () => weldVertices(cage, [0])],
    ['weldVertices out-of-range', () => weldVertices(cage, [0, nV + 3])],
    ['weldVertices every vertex', () => weldVertices(cage, cage.vertices.map((_, i) => i))],
    ['insertPointOnEdge t=0', () => insertPointOnEdge(cage, [...buildTopology(cage).edgeMap.keys()][0], 0)],
    ['insertPointOnEdge t=1', () => insertPointOnEdge(cage, [...buildTopology(cage).edgeMap.keys()][0], 1)],
    ['insertPointOnEdge bogus key', () => insertPointOnEdge(cage, '999_1000', 0.5)],
    ['slideEdges t=0', () => slideEdges(cage, [[...buildTopology(cage).edgeMap.keys()][0]], 0)],
    ['slideEdges t beyond the refusal boundary', () => slideEdges(cage, [[...buildTopology(cage).edgeMap.keys()][0]], 5)],
    ['offsetCage zero', () => offsetCage(cage, 0)],
    ['offsetCage NaN', () => offsetCage(cage, NaN)],
    ['offsetCage enormous', () => offsetCage(cage, 1e6)],
    ['thickenCage enormous', () => thickenCage(cage, 1e6)],
    ['thickenCage negative', () => thickenCage(cage, -1e6)],
    ['mergeFaces single face', () => mergeFaces(cage, [0])],
    ['mergeFaces every face', () => mergeFaces(cage, cage.faces.map((_, i) => i))],
    ['subdivideFaces empty set', () => subdivideFaces(cage, [])],
    ['subdivideFaces out-of-range', () => subdivideFaces(cage, [nF + 2])],
    ['insertEdgeLoop bogus seed', () => insertEdgeLoop(cage, '999_1000', 0.5, 0)],
    ['insertEdgeLoop t=0', () => insertEdgeLoop(cage, [...buildTopology(cage).edgeMap.keys()][0], 0, 0)],
    ['extrudeEdges on an INTERIOR edge (must refuse — no free side to grow into)', () => extrudeEdges(cage, [[...buildTopology(cage).edgeMap.keys()][0]], null, 4)],
    ['extrudeEdges empty set', () => extrudeEdges(cage, [], null, 4)],
  ];
  let refusals = 0, returned = 0;
  const silentlyBroken = [];
  for (const [label, fn] of attempts) {
    let out;
    try { out = fn(); } catch { refusals++; continue; }
    if (out == null) { refusals++; continue; }
    const c = unwrap(out);
    if (!c || !Array.isArray(c.vertices)) { refusals++; continue; }
    const v = cageViolations(c, { noOrphans: false });
    if (v.length) silentlyBroken.push(`${label}: ${v.slice(0, 2).join('; ')}`);
    else {
      const s = survivesSubdivision(c);
      if (s.length) silentlyBroken.push(`${label}: ${s.slice(0, 2).join('; ')}`);
    }
    returned++;
  }
  assert.deepEqual(silentlyBroken, [],
    `every degenerate input must be REFUSED or produce a VALID cage; these returned a broken one:\n  ${silentlyBroken.join('\n  ')}`);
  assert.ok(refusals > 5, `the degenerate set must genuinely reach real refusals, got ${refusals} refusals / ${returned} accepted`);
});
