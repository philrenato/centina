// SIMPLIFY IS A CLAIM ABOUT A CAGE, NOT ABOUT A FACE COUNT.
//
// Every assertion here is about the thing handed to the subdivider and then to
// subdToPatches: all quads, closed if it started closed, no interior pinch, the
// same Euler characteristic, and — for the exact tier — the same cage back,
// creases included. A simplifier that lands the face count and loses the
// winding, the crease, or the genus has produced something no later command
// names as broken; it just stops converting.
//
// THE FIXTURES ARE CHOSEN AGAINST THE WAYS THIS GOES WRONG:
//   · `bump` is ASYMMETRIC and carries valence 3, 4 AND 5 — a smooth dense
//     symmetric cage cannot catch a winding-order or wheel-walk bug, because
//     the reversed answer is the same answer.
//   · `plane` is OPEN. Its corners are valence 2, which is an ordinary corner
//     on a boundary and a pinch in the interior; a fixture set of closed cages
//     only would have let one floor stand in for the other.
//   · the creased cube is run at five weights, including the two that leave NO
//     entry at all in the subdivided cage.
//   · `twoShells` is disconnected, which is the only input that reaches the
//     per-component path.
import { strict as assert } from 'node:assert';
import { subdivideCatmullClark, edgeKey } from '../kernel/subd.mjs';
import { superbBoxCage, superbTorusCage, superbPlaneCage, superbCylinderCage, superbConeCage, superbSphereCage } from '../kernel/subdprimitives.mjs';
import { puffCage } from '../kernel/puff.mjs';
import {
  unsubdivideCage, subdivisionSignature, isSubdivisionOfSomething,
  polychordsOf, collapsePolychord, rankPolychords, polychordDrift,
  refitCageToLimitTargets, simplifySubD,
  cageInvariants, checkSimplifyInvariants, cageExtent,
} from '../kernel/subdsimplify.mjs';

// EVERY CHECK IS COUNTED AND NONE OF THEM STOPS THE RUN. A suite that dies on
// the first failure reports one defect however many it has, and the score is
// what says whether a change fixed one thing or moved the deficit.
let CHECKS = 0;
const FAILURES = [];
const record = (fn, msg) => {
  CHECKS += 1;
  try { fn(); } catch (e) { FAILURES.push(`${msg} — ${e.message.split('\n')[0]}`); }
};
const ok = (cond, msg) => record(() => assert.ok(cond, msg), msg);
// A SECTION THAT THROWS MUST STILL LEAVE A SCORE. A defect anywhere in this
// module reaches most of these checks through a value that is now undefined, and
// a run that dies on the stack trace reports NOTHING about the other 300.
const section = (id, fn) => { try { fn(); } catch (e) { FAILURES.push(`section ${id} threw — ${String(e.message).split('\n')[0]}`); } };
const eq = (a, b, msg) => record(() => assert.equal(a, b, msg), msg);
const deep = (a, b, msg) => record(() => assert.deepEqual(a, b, msg), msg);

// --------------------------------------------------------------------------
// FIXTURES
// --------------------------------------------------------------------------

const cube = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
const box2 = superbBoxCage([0, 0, 0], [10, 7, 4], 2);
const torus = superbTorusCage([0, 0, 0], 30, 10, 8, 6);
const plane = superbPlaneCage([0, 0, 0], 40, 40, 3);
const cylinder = superbCylinderCage([0, 0, 0], 10, 20, 8);
const coneFine = subdivideCatmullClark(superbConeCage([0, 0, 0], 10, 20, 6));

// An asymmetric closed all-quad cage carrying valence 3, 4 and 5. Built once by
// extruding one face of a two-facet box and displacing every point, then frozen
// here so the fixture cannot drift with the module that made it.
const bump = {
  vertices: [[10, -7, -4], [11.38, -0.31, -3.66], [11.15, 6.38, -3.32], [10.92, -7.93, 1.02], [10.69, 0, 1.36], [10.46, 6.69, 0.17], [10.23, -7.62, 4.51], [10, -0.93, 4.85], [11.38, 7, 5.19], [-8.85, 6.69, -4], [-9.08, -0.62, -3.66], [-9.31, -7.93, -3.32], [-9.54, 7, 1.02], [-9.77, -0.31, 1.36], [-10, -7.62, 0.17], [-8.62, 6.07, 4.51], [-8.85, 0, 4.85], [-9.08, -7.31, 5.19], [0.69, 6.38, -4], [0.46, 6.07, 0.34], [0.23, 7, 4.68], [0, -7.31, -2.98], [1.38, -7.62, 1.36], [1.15, -7.93, 4.17], [0.92, 0, 4.51], [0.69, -0.31, -3.15], [-14.54, -0.62, -2.81], [-14.77, -0.93, 0], [-15, -7, -3.66], [-13.62, -7.31, 0.68]],
  faces: [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7], [9, 10, 13, 12], [26, 28, 29, 27], [12, 13, 16, 15], [13, 14, 17, 16], [2, 18, 19, 5], [18, 9, 12, 19], [5, 19, 20, 8], [19, 12, 15, 20], [11, 21, 22, 14], [21, 0, 3, 22], [14, 22, 23, 17], [22, 3, 6, 23], [17, 23, 24, 16], [23, 6, 7, 24], [16, 24, 20, 15], [24, 7, 8, 20], [9, 18, 25, 10], [18, 2, 1, 25], [10, 25, 21, 11], [25, 1, 0, 21], [10, 11, 28, 26], [11, 14, 29, 28], [14, 13, 27, 29], [13, 10, 26, 27]],
  creases: {},
};

const twoShells = (() => {
  const off = cube.vertices.length;
  return {
    vertices: cube.vertices.concat(box2.vertices.map((v) => [v[0] + 60, v[1], v[2]])),
    faces: cube.faces.concat(box2.faces.map((f) => f.map((i) => i + off))),
    creases: {},
  };
})();

const puffOutline = (k) => { const p = []; for (let i = 0; i < 96; i += 1) { const a = (i / 96) * Math.PI * 2; p.push(Math.cos(a) * k, Math.sin(a)); } return p; };
const puffDisc = puffCage(puffOutline(1), { subdivide: subdivideCatmullClark });
const puffLong = puffCage(puffOutline(3), { subdivide: subdivideCatmullClark });
eq(puffDisc.ok, true, 'puff fixture did not build');
eq(puffLong.ok, true, 'puff fixture did not build');

const creasedCube = (w) => {
  const c = superbBoxCage([0, 0, 0], [10, 10, 10], 1);
  const f = c.faces[0];
  return { vertices: c.vertices, faces: c.faces, creases: { [edgeKey(f[0], f[1])]: w, [edgeKey(f[1], f[2])]: w, [edgeKey(f[2], f[3])]: w, [edgeKey(f[3], f[0])]: w } };
};

const maxCoordDiff = (a, b) => {
  let worst = 0;
  for (let i = 0; i < a.vertices.length; i += 1) for (let k = 0; k < 3; k += 1) worst = Math.max(worst, Math.abs(a.vertices[i][k] - b.vertices[i][k]));
  return worst;
};

// --------------------------------------------------------------------------
// 1. THE INSTRUMENT ITSELF — an invariant check that cannot go red is not a
//    check. Each of these is a cage that violates exactly one clause.
// --------------------------------------------------------------------------

section(1, () => {
  {
    const tri = { vertices: cube.vertices, faces: [[0, 1, 2]].concat(cube.faces.slice(1)), creases: {} };
    ok(checkSimplifyInvariants(cube, tri).problems.some((p) => p.includes('not all quads')), 'a triangle must be caught');

    // A doublet: two quads sharing two edges, so the shared vertices are interior
    // and valence 2.
    const doublet = { vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], faces: [[0, 1, 2, 3], [3, 2, 1, 0]], creases: {} };
    ok(cageInvariants(doublet).minInteriorValence === 2, 'a doublet must read as interior valence 2');
    ok(checkSimplifyInvariants(doublet, doublet).problems.some((p) => p.includes('valence 2')), 'a valence-2 interior vertex must be caught');

    // ...and the same number on a BOUNDARY is an ordinary corner, not a pinch.
    eq(cageInvariants(plane).minValence, 2, 'a plane cage has valence-2 corners');
    deep(checkSimplifyInvariants(plane, plane).problems, [], 'an open grid must not be condemned for its own corners');

    const opened = { vertices: cube.vertices, faces: cube.faces.slice(1), creases: {} };
    ok(checkSimplifyInvariants(cube, opened).problems.some((p) => p.includes('closed')), 'losing closure must be caught');
    ok(checkSimplifyInvariants(cube, opened).problems.some((p) => p.includes('Euler')), 'a genus change must be caught');
    console.log(`  invariants: five defective cages, five named problems`);
  }
});

// --------------------------------------------------------------------------
// 2. THE CUBE'S PREIMAGE IS NOT UNIQUE, and no solver fixes that.
//    Displacing the four corners of one inscribed tetrahedron by +d and the
//    other four by -d leaves the subdivision IDENTICAL. Every rule is a
//    weighted average; the vertex rule's own P coefficient is (n-3)/n, which is
//    zero at valence 3, and the two colors cancel everywhere else.
// --------------------------------------------------------------------------

section(2, () => {
  {
    const sgn = (v) => (Math.sign(v[0]) * Math.sign(v[1]) * Math.sign(v[2]) > 0 ? 1 : -1);
    const d = [3, 1.5, -0.6];
    const shifted = { vertices: cube.vertices.map((v) => v.map((c, k) => c + sgn(v) * d[k])), faces: cube.faces, creases: {} };
    const a = subdivideCatmullClark(cube);
    const b = subdivideCatmullClark(shifted);
    const diff = maxCoordDiff(a, b);
    ok(diff < 1e-12, `two different cubes must subdivide identically (got ${diff})`);
    ok(maxCoordDiff(cube, shifted) > 1, 'the two cubes must actually differ');

    const r = unsubdivideCage(a);
    eq(r.ok, true, 'the subdivided cube must still invert');
    eq(r.unique, false, 'the cube must REPORT that its preimage is a family, not a point');
    ok(r.message.includes('not unique'), 'and say so in words');
    ok(maxCoordDiff(r.cage, cube) < 1e-9, 'and return the symmetric member, which is the cube');

    const r2 = unsubdivideCage(subdivideCatmullClark(box2));
    eq(r2.unique, true, 'a cage with any valence other than 3 has ONE preimage');
    console.log(`  cube null space: two cages ${maxCoordDiff(cube, shifted).toFixed(1)} apart subdivide to within ${diff.toExponential(1)}`);
  }
});

// --------------------------------------------------------------------------
// 3. TIER 1 — EXACT ROUND TRIP. subdivide, un-subdivide, and get the SAME cage
//    back: same faces index-for-index (which pins the winding, since a wheel
//    walked the other way returns the reversed loop), same creases, positions
//    to floating point.
// --------------------------------------------------------------------------

// `cylinder` is here with `allowNgons` on deliberately: its preimage is a
// cylinder, n-gon caps and all, and the inversion is exact on it. What the
// default REFUSES is handing that preimage back as a simplification (section
// 5), which is a separate judgment from whether the arithmetic inverts.
const roundTrips = [['cube', cube, true], ['box2', box2, true], ['torus', torus, true], ['plane', plane, true], ['bump', bump, true], ['twoShells', twoShells, true], ['cylinder', cylinder, false]];
section(3, () => {
  for (const [name, coarse, quadPreimage] of roundTrips) {
    const fine = subdivideCatmullClark(coarse);
    const r = unsubdivideCage(fine, { allowNgons: true });
    eq(r.ok, true, `${name}: ${r.message}`);
    const diff = maxCoordDiff(r.cage, coarse);
    ok(diff < 1e-9 * cageExtent(coarse), `${name}: recovered positions off by ${diff}`);
    ok(r.residual < 1e-12, `${name}: residual ${r.residual}`);
    deep(r.cage.faces, coarse.faces, `${name}: faces did not come back index-for-index, in the same winding`);
    if (quadPreimage) deep(checkSimplifyInvariants(fine, r.cage).problems, [], `${name}: the preimage breaks an invariant`);
    eq(isSubdivisionOfSomething(fine), quadPreimage, `${name}: the boolean form disagrees`);
    console.log(`  round trip ${name.padEnd(10)} F ${String(fine.faces.length).padStart(4)} -> ${String(r.cage.faces.length).padStart(3)}  residual ${r.residual.toExponential(1)}  worst ${diff.toExponential(1)}`);
  }
});

// TWO LEVELS. Repeating the operation walks back another level.
section(4, () => {
  {
    const twice = subdivideCatmullClark(subdivideCatmullClark(bump));
    const r = simplifySubD(twice, { tier: 'unsubdivide', amount: 2 });
    eq(r.ok, true, 'two levels must come back');
    eq(r.steps.length, 2, 'two levels means two steps');
    ok(maxCoordDiff(r.cage, bump) < 1e-9 * cageExtent(bump), 'two levels must land on the original');
    // ...and asking for a third stops at what it could do rather than refusing.
    const r3 = simplifySubD(twice, { tier: 'unsubdivide', amount: 3 });
    eq(r3.steps.length, 2, 'a third level is not there and must not be invented');
    console.log('  two levels of subdivision come back exactly, a third stops');
  }
});

// --------------------------------------------------------------------------
// 4. CREASES. A crease silently dropped changes the limit surface and no
//    count-based check sees it, so the WEIGHTS and the EDGES are both asserted.
// --------------------------------------------------------------------------

section(5, () => {
  for (const w of [3, 2, 1.5, 1]) {
    const coarse = creasedCube(w);
    const fine = subdivideCatmullClark(coarse);
    const r = unsubdivideCage(fine);
    eq(r.ok, true, `crease ${w}: ${r.message}`);
    deep(r.cage.creases, coarse.creases, `crease ${w}: came back as ${JSON.stringify(r.cage.creases)}`);
    ok(maxCoordDiff(r.cage, coarse) < 1e-9 * cageExtent(coarse), `crease ${w}: positions moved`);
    // The whole point: the recovered cage must subdivide back to the input,
    // creases and all. A dropped crease changes the geometry, so this fails too.
    const again = subdivideCatmullClark(r.cage);
    deep(again.creases, fine.creases, `crease ${w}: the recovered cage does not re-subdivide to the same crease map`);
    ok(maxCoordDiff(again, fine) < 1e-9 * cageExtent(fine), `crease ${w}: the recovered cage does not re-subdivide to the same cage`);
    console.log(`  crease w=${w}: ${Object.keys(fine.creases).length} entries in the subdivided cage -> ${Object.keys(r.cage.creases).length} recovered at weight ${w}`);
  }
});

// WEIGHT 1 IS THE INTERESTING ONE. One pass decrements it to 0 and drops it, so
// the subdivided cage carries NO crease entry at all — the crease is recovered
// from the face-centroid equations, not from the map.
section(6, () => {
  {
    const fine = subdivideCatmullClark(creasedCube(1));
    eq(Object.keys(fine.creases).length, 0, 'a weight-1 crease must leave no entry behind');
    const r = unsubdivideCage(fine);
    eq(r.recoveredWeightOneCrease, true, 'and must be recovered from the geometry');
    eq(Object.keys(r.cage.creases).length, 4, 'all four of them');
  }
});

// A WEIGHT BETWEEN 0 AND 1 IS A PARTIAL BLEND AND LEAVES NOTHING TO READ. That
// is refused by name rather than returned as a smooth cage, which is the one
// answer that would look right and be wrong.
section(7, () => {
  {
    const fine = subdivideCatmullClark(creasedCube(0.5));
    const r = unsubdivideCage(fine);
    eq(r.ok, false, 'a weight-0.5 crease must not silently come back smooth');
    eq(r.reason, 'POSITIONS_DO_NOT_MATCH', `got ${r.reason}`);
    ok(r.message.includes('crease of weight between 0 and 1'), 'the refusal must name the cause');
    console.log(`  crease w=0.5: refused — ${r.message.slice(0, 96)}...`);
  }
});

// --------------------------------------------------------------------------
// 5. TIER 1 REFUSALS. Each one names what is wrong with the cage in front of it.
// --------------------------------------------------------------------------

section(8, () => {
  {
    const cases = [
      ['a raw cube', cube, 'NOT_A_SUBDIVISION'],
      ['a raw plane', plane, 'NOT_A_SUBDIVISION'],
      ['a cage with an n-gon', cylinder, 'NOT_ALL_QUADS'],
      ['a puff', puffDisc.cage, 'POSITIONS_DO_NOT_MATCH'],
      ['a two-facet sphere', superbSphereCage([0, 0, 0], 10, 2), 'POSITIONS_DO_NOT_MATCH'],
      ['a two-facet torus', superbTorusCage([0, 0, 0], 30, 10, 8, 6), 'POSITIONS_DO_NOT_MATCH'],
    ];
    for (const [name, cage, reason] of cases) {
      const r = unsubdivideCage(cage);
      eq(r.ok, false, `${name} must not invert`);
      eq(r.reason, reason, `${name}: got ${r.reason}`);
      ok(r.message.length > 40 && /[.]$/.test(r.message), `${name}: the refusal must be a sentence`);
      console.log(`  refuses ${name.padEnd(22)} ${r.reason}`);
    }

    // ⚠ THE ONE THAT MATTERS. A two-facet sphere HAS the topology of a subdivided
    // octahedron, and a residual-ranked answer would hand back a cage of the
    // right size and the wrong identity. It is refused, and the miss is quoted.
    const s = unsubdivideCage(superbSphereCage([0, 0, 0], 10, 2));
    eq(s.ok, false, 'a two-facet sphere must not be swapped for its dual');
    ok(s.residual > 1e-6, `and the miss must be reported (${s.residual})`);

    // A cage that WAS a subdivision and then had one point dragged is refused too.
    const moved = subdivideCatmullClark(bump);
    moved.vertices = moved.vertices.map((v, i) => (i === 7 ? [v[0] + 0.4, v[1], v[2]] : v));
    const m = unsubdivideCage(moved);
    eq(m.ok, false, 'a hand-edited subdivision must not come back as exact');
    eq(m.reason, 'POSITIONS_DO_NOT_MATCH', `got ${m.reason}`);

    // ...but the structural signature still holds, which is why the geometry has
    // to be the thing that decides.
    eq(subdivisionSignature(moved).ok, true, 'the structural signature alone cannot see a moved point');

    // The preimage of a subdivided cylinder is a cylinder — n-gon caps and all.
    const cy = unsubdivideCage(subdivideCatmullClark(cylinder));
    eq(cy.ok, false, 'an n-gon preimage must be refused by default');
    eq(cy.reason, 'PREIMAGE_NOT_ALL_QUADS', `got ${cy.reason}`);
    ok(cy.message.includes('NURBS'), 'and say why that matters');
  }
});

// A PRIMITIVE NOBODY EVER PRESSED SUBDIVIDE ON CAN STILL BE ONE. A two-facet
// box is EXACTLY one Catmull-Clark pass of a one-facet box whose twelve edges
// all carry weight 1 — the sharp rules hold every corner still, put every edge
// point at a midpoint and every face point at a centroid, which is precisely
// the uniform grid the primitive builds. The limit surface is unchanged,
// because a cage and its own subdivision have the same limit surface, so this
// is a real exact answer and not a coincidence of counts.
section(9, () => {
  {
    const r = unsubdivideCage(box2);
    eq(r.ok, true, 'a two-facet box is a subdivision after all');
    eq(r.recoveredWeightOneCrease, true, 'and it is the weight-1 crease retry that finds it');
    eq(Object.keys(r.cage.creases).length, 12, 'all twelve edges');
    eq(r.cage.faces.length, 6, 'six faces');
    const back = subdivideCatmullClark(r.cage);
    const canon = (c) => c.vertices.map((v) => v.map((x) => x.toFixed(9)).join(',')).sort().join('|');
    eq(canon(back), canon(box2), 'and re-subdividing it reproduces the primitive exactly');
    console.log(`  a two-facet box IS a subdivision: ${box2.faces.length} -> 6 faces, twelve weight-1 creases, re-subdivides exactly`);
  }
});

// --------------------------------------------------------------------------
// 6. THE POLYCHORD DECOMPOSITION IS COMPLETE. Every quad lies on exactly two
//    chords, so the chord lengths must sum to twice the face count. That one
//    identity catches a walk that stops early, one that double-counts, and one
//    that visits a face in the wrong direction.
// --------------------------------------------------------------------------

const quadCages = [['cube', cube], ['box2', box2], ['torus', torus], ['plane', plane], ['bump', bump], ['puff', puffDisc.cage], ['puffLong', puffLong.cage], ['coneFine', coneFine], ['twoShells', twoShells]];
section(10, () => {
  for (const [name, cage] of quadCages) {
    const r = polychordsOf(cage, { allowOpenChords: true });
    eq(r.ok, true, `${name}: ${r.message}`);
    const sum = r.chords.reduce((a, c) => a + c.length, 0);
    eq(sum, 2 * cage.faces.length, `${name}: chord lengths sum to ${sum}, not ${2 * cage.faces.length}`);
    const collapsible = r.chords.filter((c) => c.collapsible).length;
    console.log(`  chords ${name.padEnd(10)} ${String(r.chords.length).padStart(3)} chords, ${String(collapsible).padStart(3)} collapsible, lengths ${[...new Set(r.chords.map((c) => c.length))].sort((a, b) => a - b).join('/')}`);
  }
});

// --------------------------------------------------------------------------
// 7. TIER 2 — EVERY COLLAPSE, ON EVERY FIXTURE, HOLDS THE INVARIANT.
// --------------------------------------------------------------------------

let collapses = 0;
section(11, () => {
  for (const [name, cage] of quadCages) {
    const r = polychordsOf(cage, { allowOpenChords: true });
    for (const chord of r.chords) {
      if (!chord.collapsible) continue;
      const res = collapsePolychord(cage, chord, { allowOpenChords: true });
      eq(res.ok, true, `${name} chord ${chord.id}: ${res.message}`);
      const check = checkSimplifyInvariants(cage, res.cage);
      deep(check.problems, [], `${name} chord ${chord.id}: ${check.problems.join('; ')}`);
      eq(res.cage.faces.length, cage.faces.length - chord.length, `${name} chord ${chord.id}: wrong number of faces removed`);
      collapses += 1;
    }
  }
});
console.log(`  ${collapses} single collapses across ${quadCages.length} cages, every one all-quad, closed as it started, same Euler, no interior valence < 3`);

// AND REPEATEDLY, which is where a greedy loop walks a cage off a cliff.
section(12, () => {
  for (const [name, cage] of [['puff', puffDisc.cage], ['bump', bump], ['torus', torus]]) {
    const r = simplifySubD(cage, { amount: 20 });
    eq(r.ok, true, `${name}: ${r.message}`);
    deep(r.invariants.problems, [], `${name}: ${r.invariants.problems.join('; ')}`);
    ok(r.cage.faces.length >= 6, `${name}: went below the floor to ${r.cage.faces.length} faces`);
    console.log(`  greedy ${name.padEnd(8)} ${cage.faces.length} -> ${r.cage.faces.length} faces in ${r.steps.filter((s) => s.tier === 'polychord').length} collapses`);
  }
});

// --------------------------------------------------------------------------
// 8. TIER 2 REFUSALS.
// --------------------------------------------------------------------------

section(13, () => {
  {
    // AN OPEN CAGE. Every strip in a plane grid runs off the boundary, so by
    // default Simplify refuses the whole cage rather than moving the boundary;
    // `allowOpenChords` is the caller's way of saying it meant to.
    const r = polychordsOf(plane);
    eq(r.chords.every((c) => !c.closed), true, 'every chord of a grid is open');
    eq(r.chords.every((c) => c.refusal && c.refusal.reason === 'OPEN_CHORD'), true, 'and every one is refused by name');
    ok(r.chords[0].refusal.message.includes('boundary'), 'the refusal must name the boundary');
    const s = simplifySubD(plane, { minFaces: 1 });
    eq(s.ok, false, 'a plane must refuse by default');
    eq(s.reason, 'OPEN_CHORD', `got ${s.reason}`);
    const s2 = simplifySubD(plane, { allowOpenChords: true, minFaces: 1 });
    eq(s2.ok, true, 'and collapse when the caller asks for it');
    deep(checkSimplifyInvariants(plane, s2.cage).problems, [], 'an open collapse must still hold the invariant');
    eq(cageInvariants(s2.cage).boundaryEdgeCount > 0, true, 'and the result must still be open');
    console.log(`  open cage: ${r.chords.length} open chords refused by default, ${plane.faces.length} -> ${s2.cage.faces.length} faces when allowed`);
  }
});

section(14, () => {
  {
    // A CREASE ON A RUNG HAS NOWHERE TO GO — the collapse deletes that edge.
    const c = { vertices: torus.vertices, faces: torus.faces, creases: {} };
    const chords = polychordsOf(c).chords;
    const victim = chords.find((x) => x.collapsible);
    const creased = { vertices: c.vertices, faces: c.faces, creases: { [victim.rungKeys[0]]: 2 } };
    const again = polychordsOf(creased).chords.find((x) => x.id === victim.id);
    eq(again.collapsible, false, 'a chord crossing a creased edge must not be offered');
    eq(again.refusal.reason, 'CHORD_CREASE', `got ${again.refusal.reason}`);
    ok(again.refusal.message.includes('Remove the crease first'), 'and say what to change');
    const direct = collapsePolychord(creased, again, {});
    eq(direct.ok, false, 'and the direct call must refuse too, not just the offer');
    console.log(`  crease on a strip: ${again.refusal.message}`);
  }
});

section(14.5, () => {
  // THE SIDES OF A STRIP MERGE INTO ONE EDGE, and a crease has to survive that.
  // The two rails of every face in the strip land on the same pair of welded
  // points, so a crease on both carries across at that weight — and a crease on
  // only one of them has no right answer, which is a refusal rather than a guess.
  const chord = polychordsOf(torus).chords.find((x) => x.collapsible);
  const rungSet = new Set(chord.rungKeys);
  const rails = new Set();
  for (const fi of chord.faces) {
    const f = torus.faces[fi];
    for (let k = 0; k < 4; k += 1) { const key = edgeKey(f[k], f[(k + 1) % 4]); if (!rungSet.has(key)) rails.add(key); }
  }
  eq(rails.size, 2 * chord.length, 'a closed strip has two rails per face');

  const even = { vertices: torus.vertices, faces: torus.faces, creases: Object.fromEntries([...rails].map((k) => [k, 2])) };
  const r = collapsePolychord(even, polychordsOf(even).chords.find((x) => x.id === chord.id), {});
  eq(r.ok, true, `an evenly creased strip must still collapse: ${r.message}`);
  eq(Object.keys(r.cage.creases).length, rails.size / 2, 'the rails merge pairwise and the crease survives on every one');
  eq(Object.values(r.cage.creases).every((w) => w === 2), true, 'at the weight they carried');

  const odd = { vertices: torus.vertices, faces: torus.faces, creases: { [[...rails][0]]: 2 } };
  const o = polychordsOf(odd).chords.find((x) => x.id === chord.id);
  eq(o.refusal && o.refusal.reason, 'CREASE_COLLISION', `got ${o.refusal && o.refusal.reason}`);
  console.log(`  strip creases: ${rails.size} rail edges merge to ${Object.keys(r.cage.creases).length}, and a one-sided crease is refused`);
});

section(15, () => {
  {
    // A SELF-TOUCHING STRIP. Around a subdivided cone's apex, a strip comes back
    // to a vertex it has already crossed: welding both of those rungs would pull
    // three or more points onto one. Note WHICH of the two guards fires — the
    // walk itself completes, and it is the rung-vertex test that catches it, so a
    // fixture set that only exercised the walk would leave this open.
    const r = polychordsOf(coneFine);
    const self = r.chords.filter((c) => c.refusal && c.refusal.reason === 'SELF_TOUCHING');
    ok(self.length > 0, 'the subdivided cone must have self-touching chords');
    eq(self.every((c) => !c.selfTouching), true, 'and it is the shared-vertex guard, not the walk, that names them');
    ok(self[0].refusal.message.includes('weld'), 'the refusal must say what would happen');
    console.log(`  self-touching: ${self.length} of ${r.chords.length} chords on a subdivided cone, caught by the shared-rung-vertex guard`);
  }
});

section(16, () => {
  {
    // THE FLOOR, AND THE THING BEHIND IT. A 6-face cube is already at the default
    // floor, so every strip is refused on the count. Drop the floor to 1 and the
    // strips are STILL refused — collapsing one leaves two quads sharing all four
    // edges, whose vertices are interior and valence 2. That is the guard that
    // actually holds: all-quad, closed and Euler 2 are all satisfied by that
    // two-face pillow, so they are not stopping conditions and never fire.
    const r = polychordsOf(cube);
    eq(r.chords.every((c) => c.refusal && c.refusal.reason === 'BELOW_FLOOR'), true, 'a 6-face cube is already at the floor');
    const low = polychordsOf(cube, { minFaces: 1 });
    eq(low.chords.every((c) => c.refusal && c.refusal.reason === 'DEGENERATE_RESULT'), true, 'and below the floor the valence guard takes over');
    ok(low.chords[0].refusal.message.includes('valence 2'), 'naming the pinch');
    const pillow = { vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], faces: [[0, 1, 2, 3], [3, 2, 1, 0]], creases: {} };
    const pv = cageInvariants(pillow);
    eq(pv.allQuads && pv.closed && pv.euler === 2, true, 'the cage the guard refuses passes all-quad, closed and Euler 2');

    // AND A GREEDY LOOP WITH NO FLOOR AT ALL STILL STOPS, at a cage that is still
    // a surface. This is the check the floor alone could not make.
    for (const [name, cage, floor] of [['puff', puffDisc.cage, 6], ['torus', torus, 9]]) {
      const g = simplifySubD(cage, { amount: 50, minFaces: 1 });
      eq(g.cage.faces.length, floor, `${name}: unbounded greedy landed on ${g.cage.faces.length} faces`);
      deep(checkSimplifyInvariants(cage, g.cage).problems, [], `${name}: unbounded greedy broke an invariant`);
      ok(cageInvariants(g.cage).minInteriorValence >= 3, `${name}: unbounded greedy left a pinch`);
    }
    console.log('  the floor: 6 faces by default; with the floor removed the valence guard stops the loop anyway');
  }
});

section(17, () => {
  {
    // AN N-GON CAGE HAS NO STRIP AT ALL.
    const r = polychordsOf(cylinder);
    eq(r.ok, false, 'a cylinder must be refused whole');
    eq(r.reason, 'NOT_ALL_QUADS', `got ${r.reason}`);
    ok(r.message.includes('Merge Faces'), 'and name where n-gons come from');
  }
});

// --------------------------------------------------------------------------
// 9. THE REFIT. Moving the control points until their LIMIT positions land back
//    where they were is the puff's own technique; here it is what makes a
//    collapse invisible rather than merely legal.
// --------------------------------------------------------------------------

section(18, () => {
  for (const [name, cage] of [['puff', puffDisc.cage], ['puffLong', puffLong.cage], ['bump', bump]]) {
    const a = simplifySubD(cage, { amount: 2 });
    const b = simplifySubD(cage, { amount: 2, refit: false });
    eq(a.ok && b.ok, true, `${name}: simplify refused`);
    eq(a.cage.faces.length, b.cage.faces.length, `${name}: the refit must not change the topology`);
    deep(a.cage.faces, b.cage.faces, `${name}: the refit must not change the topology`);
    ok(a.steps.at(-1).drift < b.steps.at(-1).drift, `${name}: the refit did not help (${a.steps.at(-1).drift} vs ${b.steps.at(-1).drift})`);
    console.log(`  refit ${name.padEnd(9)} drift ${(b.steps.at(-1).drift * 100).toFixed(3)}% -> ${(a.steps.at(-1).drift * 100).toFixed(3)}% of extent at the surviving points`);
  }
});

section(19, () => {
  {
    // The refit keeps the best iterate, so a target it cannot reach leaves the
    // cage no worse than the pass that got closest.
    const targets = bump.vertices.map((v) => [v[0] * 3, v[1], v[2]]);
    const fit = refitCageToLimitTargets(bump, targets, { passes: 6 });
    eq(fit.vertices.length, bump.vertices.length, 'the refit must not add or drop points');
    ok(Number.isFinite(fit.maxError), 'and must report where it got to');
  }
});

// --------------------------------------------------------------------------
// 10. THE COMMAND — which tier answered, and what it cost.
// --------------------------------------------------------------------------

section(20, () => {
  {
    const exact = simplifySubD(subdivideCatmullClark(bump));
    eq(exact.tier, 'unsubdivide', 'a subdivided cage must get the exact answer');
    ok(exact.message.startsWith('exact —'), `got "${exact.message}"`);
    ok(maxCoordDiff(exact.cage, bump) < 1e-9 * cageExtent(bump), 'and it must be the cage it came from');

    const approx = simplifySubD(puffDisc.cage);
    eq(approx.tier, 'polychord', 'a puff must fall through to the strip collapse');
    ok(approx.message.includes('limit surface within'), `got "${approx.message}"`);
    eq(approx.steps[0].refused, 'POSITIONS_DO_NOT_MATCH', 'and the record must say why tier 1 declined');

    const ranked = rankPolychords(puffDisc.cage);
    ok(ranked.ranked.length > 1, 'a puff must offer a choice of strips');
    ok(ranked.ranked[0].worst <= ranked.ranked.at(-1).worst, 'and they must be ordered cheapest first');
    const drift = polychordDrift(puffDisc.cage, ranked.ranked[0].chord);
    ok(drift.worst >= 0 && drift.mean <= drift.worst, 'the drift proxy must be self-consistent');

    console.log(`  tiers: ${exact.message}`);
    console.log(`         ${approx.message}`);
  }
});

// --------------------------------------------------------------------------
// 11. THE CAGE-LEVEL REFUSALS, each on a cage that violates exactly one clause.
// --------------------------------------------------------------------------

section(21, () => {
  {
    const cases = [
      ['no faces', { vertices: [[0, 0, 0]], faces: [], creases: {} }, 'EMPTY_CAGE'],
      ['a face using a vertex twice', { vertices: cube.vertices, faces: [[0, 1, 1, 2]].concat(cube.faces.slice(1)), creases: {} }, 'DEGENERATE_FACE'],
      ['a vertex no face uses', { vertices: cube.vertices.concat([[99, 99, 99]]), faces: cube.faces, creases: {} }, 'UNUSED_VERTICES'],
      ['an edge on three faces', {
        vertices: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, -1, 0], [0, -1, 0], [1, 0, 1], [0, 0, 1]],
        faces: [[0, 1, 2, 3], [1, 0, 5, 4], [0, 1, 6, 7]], creases: {},
      }, 'NON_MANIFOLD'],
    ];
    for (const [name, cage, reason] of cases) {
      const r = unsubdivideCage(cage);
      eq(r.ok, false, `${name} must be refused`);
      eq(r.reason, reason, `${name}: got ${r.reason}`);
      eq(polychordsOf(cage).reason, reason, `${name}: the strip walk must refuse the same way`);
    }

    // A PINCHED PREIMAGE. Two quads sharing all four edges pass all-quad, closed
    // and Euler 2, and their subdivision is a perfectly ordinary-looking cage —
    // so the tier that inverts it exactly must still refuse to hand it back.
    const pillow = { vertices: [[0, 0, 0], [10, 0, 0], [10, 10, 3], [0, 10, 3]], faces: [[0, 1, 2, 3], [3, 2, 1, 0]], creases: {} };
    const p = unsubdivideCage(subdivideCatmullClark(pillow));
    eq(p.ok, false, 'a pinched preimage must not be handed back');
    eq(p.reason, 'PREIMAGE_PINCHED', `got ${p.reason}`);
    eq(unsubdivideCage(subdivideCatmullClark(pillow), { allowNgons: true }).ok, true, 'though the arithmetic inverts it');

    // THE TWO HALVES OF A CREASED EDGE ALWAYS CARRY THE SAME WEIGHT after one
    // pass, so a cage where they do not was not made by one.
    const fine = subdivideCatmullClark(creasedCube(3));
    const keys = Object.keys(fine.creases);
    const tampered = { vertices: fine.vertices, faces: fine.faces, creases: { ...fine.creases, [keys[0]]: 1.25 } };
    const t = unsubdivideCage(tampered);
    eq(t.ok, false, 'mismatched crease halves must be refused');
    eq(t.reason, 'NOT_A_SUBDIVISION', `got ${t.reason}`);
    ok(t.message.includes('different weights'), 'and say which fact is wrong');

    // A CREASE ONE PASS WOULD NEVER HAVE CREATED — on a spoke from an edge point
    // to a face point — rules the cage out too.
    const spoke = { vertices: fine.vertices, faces: fine.faces, creases: { ...fine.creases, [edgeKey(fine.faces[0][1], fine.faces[0][2])]: 2 } };
    const sp = unsubdivideCage(spoke);
    eq(sp.ok, false, 'a crease on a spoke must be refused');
    ok(sp.message.includes('smooth'), `got "${sp.message}"`);
    console.log('  cage-level refusals: empty, degenerate face, stray vertex, non-manifold, pinched preimage, two crease shapes');
  }
});

if (FAILURES.length) {
  for (const f of FAILURES) console.log(`  FAIL ${f}`);
  console.log(`subdsimplify: ${CHECKS - FAILURES.length}/${CHECKS} checks passed`);
  assert.fail(`${FAILURES.length} of ${CHECKS} checks failed`);
}
console.log(`subdsimplify: ok — ${CHECKS}/${CHECKS} checks passed`);
