import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  INFILL_CELLS, INFILL_CELL_TYPES, latticeGraph, infillBodyCheck, windingNumber, clipGraphToBody,
  skinNet, infillCurves, strutChordFrame, infillCvOffsets, applyInfillOverrides, buildInfill,
  infillDefaultSize, infillEstimate, infillStrutKnots, mergeShortStruts, pruneDangling,
  infillGraphHash, infillGraphArrays,
} from '../kernel/infill.mjs';
import { uvSphere, torus, slide, thinWallC, extrudePolygon, scaleMesh, orientOutward } from './fixtures/infill_bodies.mjs';

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
const BOX = { min: [-20, -20, -20], max: [20, 20, 20] };
const degrees = (g) => { const d = new Array(g.nodes.length).fill(0); for (const s of g.struts) { d[s.a]++; d[s.b]++; } return d; };
const inner = (p, r) => p.every((v) => Math.abs(v) < r);

// Distance from p to the closest triangle of a body, brute force.
function surfaceDistance(body, p) {
  const P = body.positions, I = body.indices;
  let best = Infinity;
  for (let t = 0; t < I.length; t += 3) {
    const a = [P[3 * I[t]], P[3 * I[t] + 1], P[3 * I[t] + 2]], b = [P[3 * I[t + 1]], P[3 * I[t + 1] + 1], P[3 * I[t + 1] + 2]], c = [P[3 * I[t + 2]], P[3 * I[t + 2] + 1], P[3 * I[t + 2] + 2]];
    // Sample the triangle's closest point by projection + edge clamps.
    const ab = a.map((v, i) => b[i] - v), ac = a.map((v, i) => c[i] - v), ap = a.map((v, i) => p[i] - v);
    const d00 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2], d01 = ab[0] * ac[0] + ab[1] * ac[1] + ab[2] * ac[2], d11 = ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2];
    const d20 = ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2], d21 = ap[0] * ac[0] + ap[1] * ac[1] + ap[2] * ac[2];
    const den = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / den, w = (d00 * d21 - d01 * d20) / den;
    let q;
    if (v >= 0 && w >= 0 && v + w <= 1) q = a.map((x, i) => x + ab[i] * v + ac[i] * w);
    else {
      const seg = (s, e) => { const d = s.map((x, i) => e[i] - x); const L = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]; const tt = Math.max(0, Math.min(1, ((p[0] - s[0]) * d[0] + (p[1] - s[1]) * d[1] + (p[2] - s[2]) * d[2]) / L)); return s.map((x, i) => x + d[i] * tt); };
      q = [seg(a, b), seg(b, c), seg(c, a)].sort((x, y) => dist(x, p) - dist(y, p))[0];
    }
    best = Math.min(best, dist(p, q));
  }
  return best;
}
// Kept core struts that pass through the air: a sample outside the body by
// more than a hair (a sample ON a face is not outside).
function strutsThroughAir(body, graph, samples = 8) {
  const hair = dist(body.bbox.min, body.bbox.max) * 1e-6;
  const bad = [];
  for (const s of graph.struts) {
    if (s.kind !== 'core') continue;
    const A = graph.nodes[s.a].p, B = graph.nodes[s.b].p;
    for (let k = 1; k < samples; k++) {
      const p = lerp(A, B, k / samples);
      if (windingNumber(body, p) < 0.5 && surfaceDistance(body, p) > hair) { bad.push(s.id); break; }
    }
  }
  return bad;
}

// Cells, frame, jitter

test('cell facts: valence and struts per conventional cell', () => {
  const expect = { kelvin: [12, 24, 4], voronoi: [12, 24, 4], diamond: [8, 16, 4], cubic: [1, 3, 6], gyroid: [8, 12, 3], octet: [4, 24, 12] };
  for (const k of INFILL_CELL_TYPES) {
    const c = INFILL_CELLS[k];
    assert.deepEqual([c.nodesPerCell, c.strutsPerCell, c.valence], expect[k], k);
  }
  assert.ok(Math.abs(INFILL_CELLS.kelvin.strutLength - Math.SQRT2 / 4) < 1e-12);
  assert.ok(Math.abs(INFILL_CELLS.diamond.strutLength - Math.sqrt(3) / 4) < 1e-12);
  assert.ok(Math.abs(INFILL_CELLS.gyroid.strutLength - Math.SQRT2 / 4) < 1e-12);
});

test('every interior node has the cell type\'s valence, every strut its length', () => {
  for (const cell of ['kelvin', 'diamond', 'cubic', 'gyroid', 'octet']) {
    const { graph } = latticeGraph(BOX, { cell, size: 10 });
    const deg = degrees(graph);
    let checked = 0;
    graph.nodes.forEach((n, i) => { if (inner(n.p, 12)) { assert.equal(deg[i], INFILL_CELLS[cell].valence, `${cell} node ${n.id}`); checked++; } });
    assert.ok(checked > 20, `${cell}: ${checked} interior nodes checked`);
    const L = INFILL_CELLS[cell].strutLength * 10;
    for (const s of graph.struts) assert.ok(Math.abs(dist(graph.nodes[s.a].p, graph.nodes[s.b].p) - L) < 1e-9, `${cell} strut ${s.id}`);
  }
});

test('ids are unique, and a bigger box keeps the ids and positions of the cells both cover', () => {
  const a = latticeGraph(BOX, { cell: 'kelvin', size: 10 }).graph;
  const b = latticeGraph({ min: [-40, -40, -40], max: [40, 40, 40] }, { cell: 'kelvin', size: 10 }).graph;
  assert.equal(new Set(a.nodes.map((n) => n.id)).size, a.nodes.length);
  assert.equal(new Set(a.struts.map((s) => s.id)).size, a.struts.length);
  const bNodes = new Map(b.nodes.map((n) => [n.id, n.p]));
  for (const n of a.nodes) assert.ok(bNodes.has(n.id) && dist(bNodes.get(n.id), n.p) < 1e-9, n.id);
  const bStruts = new Set(b.struts.map((s) => s.id));
  for (const s of a.struts) assert.ok(bStruts.has(s.id), s.id);
});

test('Voronoi of the BCC points at jitter 0 is Kelvin, node for node and strut for strut', () => {
  const size = 10;
  const kel = latticeGraph(BOX, { cell: 'kelvin', size }).graph;
  const vor = latticeGraph(BOX, { cell: 'voronoi', size, jitter: 0 }).graph;
  const key = (p) => p.map((v) => Math.round(v / (size * 1e-4))).join(',');
  const region = (p) => inner(p, 19.9);
  const kNodes = new Set(kel.nodes.filter((n) => region(n.p)).map((n) => key(n.p)));
  const vNodes = new Set(vor.nodes.filter((n) => region(n.p)).map((n) => key(n.p)));
  assert.ok(kNodes.size > 300, `${kNodes.size} Kelvin nodes compared`);
  assert.deepEqual([...vNodes].sort(), [...kNodes].sort());
  const ekey = (g, s) => [key(g.nodes[s.a].p), key(g.nodes[s.b].p)].sort().join('/');
  const inR = (g, s) => region(g.nodes[s.a].p) && region(g.nodes[s.b].p);
  const kE = new Set(kel.struts.filter((s) => inR(kel, s)).map((s) => ekey(kel, s)));
  const vE = new Set(vor.struts.filter((s) => inR(vor, s)).map((s) => ekey(vor, s)));
  assert.ok(kE.size > 500, `${kE.size} Kelvin struts compared`);
  assert.deepEqual([...vE].sort(), [...kE].sort());
});

test('Voronoi names a node by its four seeds and a strut by its three', () => {
  const vor = latticeGraph(BOX, { cell: 'voronoi', size: 10, jitter: 0.4, seed: 3 }).graph;
  for (const n of vor.nodes) assert.equal(n.id.slice(2).split('|').length, 4);
  for (const s of vor.struts) assert.equal(s.id.slice(2).split('|').length, 3);
  const deg = degrees(vor);
  vor.nodes.forEach((n, i) => { if (inner(n.p, 10)) assert.equal(deg[i], 4, n.id); });
});

test('jitter moves Voronoi nodes, a seed reproduces them exactly, another seed does not', () => {
  const g0 = latticeGraph(BOX, { cell: 'voronoi', size: 10, jitter: 0 }).graph;
  const g1 = latticeGraph(BOX, { cell: 'voronoi', size: 10, jitter: 0.5, seed: 7 }).graph;
  const g1b = latticeGraph(BOX, { cell: 'voronoi', size: 10, jitter: 0.5, seed: 7 }).graph;
  const g2 = latticeGraph(BOX, { cell: 'voronoi', size: 10, jitter: 0.5, seed: 8 }).graph;
  assert.deepEqual(g1, g1b);
  assert.notDeepEqual(g1.nodes.map((n) => n.p), g2.nodes.map((n) => n.p));
  const at0 = new Map(g0.nodes.map((n) => [n.id, n.p]));
  let moved = 0;
  for (const n of g1.nodes) if (at0.has(n.id) && dist(at0.get(n.id), n.p) > 0.1) moved++;
  assert.ok(moved > 50, `${moved} nodes moved`);
});

test('orientation keeps every strut length; origin shifts every node by exactly the offset', () => {
  const r = latticeGraph(BOX, { cell: 'kelvin', size: 10, orientation: [30, 20, 10] }).graph;
  for (const s of r.struts) assert.ok(Math.abs(dist(r.nodes[s.a].p, r.nodes[s.b].p) - INFILL_CELLS.kelvin.strutLength * 10) < 1e-9);
  const g = latticeGraph(BOX, { cell: 'cubic', size: 10 }).graph;
  const o = latticeGraph(BOX, { cell: 'cubic', size: 10, origin: [1, 2, 3] }).graph;
  const at = new Map(g.nodes.map((n) => [n.id, n.p]));
  for (const n of o.nodes) if (at.has(n.id)) assert.ok(dist(n.p, at.get(n.id).map((v, i) => v + [1, 2, 3][i])) < 1e-9);
});

test('a Size too small for the box is refused with the count', () => {
  const r = latticeGraph(BOX, { cell: 'kelvin', size: 0.2, maxCells: 1000 });
  assert.equal(r.ok, false);
  assert.match(r.reason, /raise Size/);
});

// Body, winding, clipping, skin

test('open, non-manifold and zero-volume bodies are refused by name; closed ones pass', () => {
  const sph = orientOutward(uvSphere());
  const ok = infillBodyCheck(sph);
  assert.equal(ok.ok, true);
  assert.ok(Math.abs(ok.volume - 4 / 3 * Math.PI * 50 ** 3) / ok.volume < 0.02, `sphere volume ${ok.volume}`);
  const open = { positions: sph.positions, indices: sph.indices.slice(3) };
  assert.match(infillBodyCheck(open).reason, /open: 3 naked edges/);
  // Two tetrahedra sharing one edge: every edge twice except the shared one, four times.
  const tet = (o) => ({ p: [o, [o[0] + 10, o[1], o[2]], [o[0], o[1] + 10, o[2]], [o[0], o[1], o[2] + 10]] });
  const t1 = tet([0, 0, 0]).p, t2 = [[0, 0, 0], [10, 0, 0], [5, -10, 3], [5, -3, -10]];
  const nm = { positions: [...t1.flat(), ...t2.flat()], indices: [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3, 4, 5, 6, 4, 7, 5, 4, 6, 7, 5, 7, 6] };
  assert.match(infillBodyCheck(nm).reason, /non-manifold/);
  // Four triangles folded flat onto one square: closed, manifold, no volume.
  const flat = { positions: [0, 0, 0, 10, 0, 0, 10, 10, 0, 0, 10, 0, 5, 5, 0], indices: [0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4, 1, 0, 4, 2, 1, 4, 3, 2, 4, 0, 3, 4] };
  assert.match(infillBodyCheck(flat).reason, /non-manifold|no volume/);
  const pillow = { positions: [0, 0, 0, 10, 0, 0, 0, 10, 0], indices: [0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1] };
  assert.ok(!infillBodyCheck(pillow).ok);
  const sliver = { positions: [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 1e-12], indices: [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3] };
  assert.match(infillBodyCheck(sliver).reason, /no volume/);
  // A refused body refuses the whole operation with the same reason.
  assert.match(buildInfill(open).reason, /naked/);
});

test('winding number: 1 inside, 0 outside, 0 in the hole of a torus, either winding', () => {
  const tb = infillBodyCheck(torus());
  assert.ok(Math.abs(windingNumber(tb, [60, 0, 0]) - 1) < 1e-6);
  assert.ok(Math.abs(windingNumber(tb, [0, 0, 0])) < 1e-6);
  assert.ok(Math.abs(windingNumber(tb, [200, 0, 0])) < 1e-6);
  const t = torus();
  const flipped = { positions: t.positions, indices: t.indices.map((_, i, a) => a[i - (i % 3) + [0, 2, 1][i % 3]]) };
  const fb = infillBodyCheck(flipped);
  assert.equal(fb.windingSign, -1);
  assert.ok(Math.abs(windingNumber(fb, [60, 0, 0]) - 1) < 1e-6);
});

test('clipping keeps no core strut through the air, and every skin node on the surface', () => {
  for (const [name, mesh, opts] of [['sphere', uvSphere(), {}], ['torus', torus(), {}], ['slide', slide(), {}]]) {
    const r = buildInfill(mesh, opts);
    assert.equal(r.ok, true, name);
    const body = infillBodyCheck(mesh);
    assert.deepEqual(strutsThroughAir(body, r.graph), [], name);
    const hair = dist(body.bbox.min, body.bbox.max) * 1e-6;
    let worst = 0;
    for (const n of r.graph.nodes) if (n.kind === 'skin') worst = Math.max(worst, surfaceDistance(body, n.p));
    assert.ok(worst < hair, `${name}: a skin node ${worst} off the surface`);
    const deg = degrees(r.graph);
    r.graph.nodes.forEach((n, i) => { if (n.kind === 'core') assert.ok(deg[i] >= 2, `${name}: core node ${n.id} dangles`); });
    assert.equal(r.report.components, 1, `${name}: one piece`);
  }
});

test('the thin-wall breaker can break, and the clip does not', () => {
  const mesh = thinWallC();
  const body = infillBodyCheck(mesh);
  const opts = { cell: 'kelvin', size: 20, origin: [0, 0, 3] };
  // The fixture produces the breaking input: struts with both ends inside,
  // one in each prong, crossing the gap.
  const lat = latticeGraph(body.bbox, opts).graph;
  const w = lat.nodes.map((n) => windingNumber(body, n.p));
  const naive = { nodes: lat.nodes, struts: lat.struts.filter((s) => w[s.a] > 0.5 && w[s.b] > 0.5) };
  const naiveBad = strutsThroughAir(body, naive);
  assert.ok(naiveBad.length >= 2, `an endpoint-only clip keeps ${naiveBad.length} struts across the gap`);
  const r = buildInfill(mesh, opts);
  assert.equal(r.ok, true);
  assert.ok(r.report.clip.reentrant >= 2, `${r.report.clip.reentrant} struts cut on both sides of the gap`);
  assert.deepEqual(strutsThroughAir(body, r.graph), []);
});

test('boundary "cells" keeps only whole struts and no skin', () => {
  const r = buildInfill(uvSphere(), { boundary: 'cells' });
  assert.equal(r.ok, true);
  assert.equal(r.report.skinNodes, 0);
  for (const s of r.graph.struts) assert.ok(!s.id.includes('@'), s.id);
  assert.deepEqual(strutsThroughAir(infillBodyCheck(uvSphere()), r.graph), []);
});

test('clearance: every core node sits at least that far inside', () => {
  const mesh = uvSphere();
  const body = infillBodyCheck(mesh);
  const r = buildInfill(mesh, { clearance: 6 });
  assert.ok(r.report.clip.cleared > 0, 'some nodes were moved to the skin');
  // On a convex body no move can carry a strut outside, so none is declined.
  assert.equal(r.report.clip.clearanceDeclined || 0, 0);
  for (const n of r.graph.nodes) if (n.kind === 'core') assert.ok(surfaceDistance(body, n.p) >= 6 - 1e-9, n.id);
  assert.deepEqual(strutsThroughAir(body, r.graph), []);
});

test('merge: no core strut shorter than mergeFraction x strut length unless a merge would leave the body', () => {
  const r = buildInfill(torus(), {});
  const L = r.report.strutLength * 0.25;
  const short = r.graph.struts.filter((s) => s.kind === 'core' && dist(r.graph.nodes[s.a].p, r.graph.nodes[s.b].p) < L);
  assert.ok(short.length <= r.report.clip.mergeDeclined, `${short.length} short struts, ${r.report.clip.mergeDeclined} merges declined`);
});

test('mergeShortStruts and pruneDangling on a hand-built graph', () => {
  const g = {
    nodes: [{ id: 'a', p: [0, 0, 0], kind: 'core' }, { id: 'b', p: [10, 0, 0], kind: 'core' }, { id: 'c', p: [10.5, 0, 0], kind: 'skin' }, { id: 'd', p: [0, 10, 0], kind: 'core' }, { id: 'e', p: [5, 5, 0], kind: 'core' }],
    struts: [{ id: 's1', a: 0, b: 1, kind: 'core' }, { id: 's2', a: 1, b: 2, kind: 'core' }, { id: 's3', a: 0, b: 3, kind: 'core' }, { id: 's4', a: 3, b: 4, kind: 'core' }],
  };
  const m = mergeShortStruts(g, 1);
  assert.equal(m.merged, 1);
  const c = m.graph.nodes.find((n) => n.id === 'c');
  assert.deepEqual(c.p, [10.5, 0, 0], 'the skin node keeps its point');
  assert.equal(m.graph.struts.length, 3);
  // e dangles, then d, then a (one strut left, to the skin), then the
  // skin node holds nothing: a spur into the body is removed to its root.
  assert.deepEqual(pruneDangling(m.graph).graph.nodes, []);
  // A core node between two skin nodes is held, not dangling.
  const held = {
    nodes: [{ id: 'k1', p: [0, 0, 0], kind: 'skin' }, { id: 'c', p: [5, 0, 0], kind: 'core' }, { id: 'k2', p: [10, 0, 0], kind: 'skin' }, { id: 'x', p: [5, 5, 0], kind: 'core' }],
    struts: [{ id: 's1', a: 0, b: 1, kind: 'core' }, { id: 's2', a: 1, b: 2, kind: 'core' }, { id: 's3', a: 1, b: 3, kind: 'core' }],
  };
  const p = pruneDangling(held);
  assert.equal(p.removed, 1);
  assert.deepEqual(p.graph.nodes.map((n) => n.id), ['k1', 'c', 'k2']);
});

test('face skin (Kelvin): every hit carries three skin struts and its core strut', () => {
  const r = buildInfill(uvSphere(), {});
  assert.equal(r.report.skin.source, 'faces');
  const deg = degrees(r.graph);
  const skinDeg = new Array(r.graph.nodes.length).fill(0);
  for (const s of r.graph.struts) if (s.kind === 'skin') { skinDeg[s.a]++; skinDeg[s.b]++; assert.ok(s.id.startsWith('f:'), s.id); }
  r.graph.nodes.forEach((n, i) => { assert.equal(deg[i], 4, n.id); if (n.kind === 'skin') assert.equal(skinDeg[i], 3, n.id); });
  // Each skin strut is a face's trace: its chord hugs the surface.
  const body = infillBodyCheck(uvSphere());
  for (const s of r.graph.struts) if (s.kind === 'skin') assert.ok(surfaceDistance(body, lerp(r.graph.nodes[s.a].p, r.graph.nodes[s.b].p, 0.5)) < 0.1 * r.report.size, s.id);
  // A wall thinner than a strut (the slide's strap) is crossed as a band;
  // the net stays one piece and no hit is left dangling or off the skin.
  const sl = buildInfill(slide(), {});
  assert.equal(sl.report.components, 1);
  const sd = new Array(sl.graph.nodes.length).fill(0), all = degrees(sl.graph);
  for (const s of sl.graph.struts) if (s.kind === 'skin') { sd[s.a]++; sd[s.b]++; }
  sl.graph.nodes.forEach((n, i) => { if (n.kind === 'skin') { assert.ok(sd[i] >= 1, n.id); assert.ok(all[i] >= 2, n.id); } });
});

test('the graph hash ignores list order and sees a moved node', () => {
  const r = buildInfill(uvSphere(), { size: 35 });
  const h = infillGraphHash(r.graph, 35);
  assert.match(h, /^[0-9a-f]{8}$/);
  const perm = r.graph.nodes.map((_, i) => i).reverse();
  const inv = new Array(perm.length); perm.forEach((p, i) => { inv[p] = i; });
  const shuffled = { nodes: perm.map((i) => r.graph.nodes[i]), struts: [...r.graph.struts].reverse().map((s) => ({ ...s, a: inv[s.b], b: inv[s.a] })) };
  assert.equal(infillGraphHash(shuffled, 35), h);
  const moved = { nodes: r.graph.nodes.map((n, i) => (i === 3 ? { ...n, p: [n.p[0] + 1e-3, n.p[1], n.p[2]] } : n)), struts: r.graph.struts };
  assert.notEqual(infillGraphHash(moved, 35), h);
  const arr = infillGraphArrays(r.graph);
  assert.equal(arr.nodes.length, 3 * r.graph.nodes.length);
  assert.equal(arr.edges.length, 2 * r.graph.struts.length);
  assert.equal(arr.nodeSkin.reduce((a, b) => a + b, 0), r.report.skinNodes);
});

test('woven skin: a net on the surface, woven into the core, with the rhythm of the lattice', () => {
  const r = buildInfill(uvSphere(), { skinSource: 'woven' });
  assert.equal(r.report.skin.source, 'woven');
  const g = r.graph;
  const skinDeg = new Array(g.nodes.length).fill(0), coreDeg = new Array(g.nodes.length).fill(0);
  for (const s of g.struts) { const d = s.kind === 'skin' ? skinDeg : coreDeg; d[s.a]++; d[s.b]++; }
  let sum = 0, n = 0;
  g.nodes.forEach((q, i) => { if (q.kind === 'skin') { assert.ok(skinDeg[i] >= 2, `skin node ${q.id} has ${skinDeg[i]} skin struts`); assert.ok(coreDeg[i] >= 1, `skin node ${q.id} holds no core strut`); sum += skinDeg[i]; n++; } });
  const mean = sum / n;
  assert.ok(mean >= 3 && mean <= 6, `mean skin valence ${mean}`);
  const longest = Math.max(...g.struts.filter((s) => s.kind === 'skin').map((s) => dist(g.nodes[s.a].p, g.nodes[s.b].p)));
  assert.ok(longest < 3 * r.report.size, `longest skin strut ${longest} at Size ${r.report.size}`);
  // Skin curves: every inner control point on the surface.
  const body = infillBodyCheck(uvSphere());
  for (const c of r.curves) if (c.kind === 'skin') for (const p of c.curve.ctrlPts) assert.ok(surfaceDistance(body, p) < 1e-6, c.id);
});

test('given skin: struts snap to the net nodes or split its edges', () => {
  const mesh = uvSphere(50, 48, 24);
  const nodes = [], edges = [];
  const nu = 12, nv = 6;
  for (let j = 1; j < nv; j++) for (let i = 0; i < nu; i++) {
    const th = (j / nv) * Math.PI, ph = (i / nu) * 2 * Math.PI;
    nodes.push([50 * Math.sin(th) * Math.cos(ph), 50 * Math.sin(th) * Math.sin(ph), 50 * Math.cos(th)]);
  }
  const at = (i, j) => (j - 1) * nu + (i % nu);
  for (let j = 1; j < nv; j++) for (let i = 0; i < nu; i++) { edges.push([at(i, j), at(i + 1, j)]); if (j < nv - 1) edges.push([at(i, j), at(i, j + 1)]); }
  const r = buildInfill(mesh, { boundary: 'snap', net: { nodes, edges } });
  assert.equal(r.ok, true);
  assert.ok(r.report.skin.snapped + r.report.skin.split > 0);
  for (const n of r.graph.nodes) if (n.kind === 'skin') assert.ok(n.id.startsWith('g:') || n.id.startsWith('h:'), n.id);
  // Every skin node carries at least two skin struts (it lies on the net).
  const skinDeg = new Array(r.graph.nodes.length).fill(0);
  for (const s of r.graph.struts) if (s.kind === 'skin') { skinDeg[s.a]++; skinDeg[s.b]++; }
  r.graph.nodes.forEach((n, i) => { if (n.kind === 'skin') assert.ok(skinDeg[i] >= 2, n.id); });
});

test('the defaults hold a few hundred struts on the slide', () => {
  const r = buildInfill(slide(), {});
  assert.equal(r.ok, true);
  assert.ok(r.report.coreStruts >= 150 && r.report.coreStruts <= 500, `${r.report.coreStruts} core struts`);
  assert.ok(r.report.struts <= 900, `${r.report.struts} struts in all`);
  const est = infillEstimate(r.report.body.volume, {});
  assert.ok(Math.abs(est.size - r.report.size) < 1e-9);
  assert.ok(est.struts > 0.5 * r.report.coreStruts && est.struts < 2 * r.report.coreStruts, `estimate ${est.struts} vs ${r.report.coreStruts}`);
});

test('the lattice is anchored on the body: a body moved off the origin carries its graph with it', () => {
  // A fixture centered on the world origin cancels any error in how the anchor
  // is taken from the box, so this one is moved well off it.
  const t = [13.7, -5.2, 8.9];
  const a = buildInfill(uvSphere(50, 48, 24), { size: 35 });
  const b = buildInfill(uvSphere(50, 48, 24, t), { size: 35 });
  assert.equal(b.graph.nodes.length, a.graph.nodes.length);
  const at = new Map(a.graph.nodes.map((n) => [n.id, n.p]));
  let worst = 0;
  for (const n of b.graph.nodes) { const p = at.get(n.id); assert.ok(p, n.id); worst = Math.max(worst, dist(n.p, p.map((v, i) => v + t[i]))); }
  assert.ok(worst < 1e-6, `moved by the translation to ${worst}`);
});

test('the same input builds the same graph and curves', () => {
  const a = buildInfill(torus(), { cell: 'voronoi', jitter: 0.3, seed: 5 });
  const b = buildInfill(torus(), { cell: 'voronoi', jitter: 0.3, seed: 5 });
  assert.deepEqual(a.graph, b.graph);
  assert.deepEqual(a.curves.map((c) => c.curve), b.curves.map((c) => c.curve));
});

test('every cell type builds on the sphere', () => {
  for (const cell of INFILL_CELL_TYPES) {
    const r = buildInfill(uvSphere(), { cell });
    assert.equal(r.ok, true, `${cell}: ${r.reason}`);
    assert.ok(r.report.coreStruts > 50, `${cell}: ${r.report.coreStruts}`);
    assert.deepEqual(strutsThroughAir(infillBodyCheck(uvSphere()), r.graph), [], cell);
  }
});

// Curves, frames, overrides

test('a strut is a clamped uniform cubic with Detail points, born straight and evenly spaced', () => {
  assert.deepEqual(infillStrutKnots(5), [0, 0, 0, 0, 0.5, 1, 1, 1, 1]);
  for (const detail of [4, 5, 6, 7]) {
    const r = buildInfill(uvSphere(), { detail });
    for (const c of r.curves.slice(0, 40)) {
      if (c.kind !== 'core') continue;
      const P = c.curve.ctrlPts;
      assert.equal(P.length, detail);
      assert.equal(c.curve.degree, 3);
      assert.deepEqual(c.curve.knots, infillStrutKnots(detail));
      assert.deepEqual(P[0], r.graph.nodes[c.a].p);
      assert.deepEqual(P[detail - 1], r.graph.nodes[c.b].p);
      for (let k = 0; k < detail; k++) assert.ok(dist(P[k], lerp(P[0], P[detail - 1], k / (detail - 1))) < 1e-9);
    }
  }
});

test('relax pairs the struts at a node into smooth fibers: paired ends leave in opposite directions', () => {
  const r = buildInfill(uvSphere(80, 64, 32), { relax: 1, size: 30 });
  const byNode = new Map();
  for (const c of r.curves) {
    if (c.kind !== 'core') continue;
    const P = c.curve.ctrlPts, n = P.length;
    const t = (a, b) => { const d = b.map((v, i) => v - a[i]); const L = Math.hypot(...d); return d.map((v) => v / L); };
    for (const [node, dir] of [[c.a, t(P[0], P[1])], [c.b, t(P[n - 1], P[n - 2])]]) {
      if (!byNode.has(node)) byNode.set(node, []);
      byNode.get(node).push(dir);
    }
  }
  let nodes = 0, bent = 0;
  for (const [node, dirs] of byNode) {
    if (r.graph.nodes[node].kind !== 'core' || dirs.length !== 4) continue;
    nodes++;
    // Each direction has an exact opposite among the others.
    for (const d of dirs) assert.ok(dirs.some((e) => Math.abs(d[0] + e[0]) + Math.abs(d[1] + e[1]) + Math.abs(d[2] + e[2]) < 1e-9), `node ${node}`);
  }
  for (const c of r.curves) if (c.kind === 'core') { const P = c.curve.ctrlPts; if (dist(P[2], lerp(P[0], P[4], 0.5)) > 0.05 * dist(P[0], P[4])) bent++; }
  assert.ok(nodes > 20, `${nodes} valence-4 nodes`);
  assert.ok(bent > 20, `${bent} struts bent into arcs`);
});

test('chord frame offsets round-trip a hand edit through an override', () => {
  const r0 = buildInfill(uvSphere(), {});
  const c = r0.curves.find((q) => q.kind === 'core');
  const A = r0.graph.nodes[c.a].p, B = r0.graph.nodes[c.b].p;
  const f = strutChordFrame(A, B);
  assert.ok(Math.abs(f.x[0] * f.y[0] + f.x[1] * f.y[1] + f.x[2] * f.y[2]) < 1e-12);
  const edited = c.curve.ctrlPts.map((p) => p.slice());
  edited[2] = edited[2].map((v, i) => v + [3, -2, 4][i]);
  const cv = infillCvOffsets(A, B, c.birth, edited);
  const r1 = buildInfill(uvSphere(), { overrides: { struts: { [c.id]: { cv } } } });
  const c1 = r1.curves.find((q) => q.id === c.id);
  for (let k = 0; k < 5; k++) assert.ok(dist(c1.curve.ctrlPts[k], edited[k]) < 1e-9, `point ${k}`);
  assert.deepEqual(r1.report.orphans, []);
});

test('an edit follows a scaled body, orphans when Size removes its strut, and comes back', () => {
  const mesh = slide();
  const r0 = buildInfill(mesh, { size: 45 });
  const c = r0.curves.find((q) => q.kind === 'core' && !q.id.includes('@'));
  const A = r0.graph.nodes[c.a].p, B = r0.graph.nodes[c.b].p;
  const edited = c.curve.ctrlPts.map((p) => p.slice());
  edited[2] = edited[2].map((v, i) => v + [0, 0, 5][i]);
  const overrides = { struts: { [c.id]: { cv: infillCvOffsets(A, B, c.birth, edited) } } };
  // Scale body and Size together about the body center: the same strut,
  // the same bend, 1.2 times bigger.
  const b = infillBodyCheck(mesh).bbox;
  const ctr = b.min.map((v, i) => (v + b.max[i]) / 2);
  const r1 = buildInfill(scaleMesh(mesh, 1.2, ctr), { size: 54, overrides });
  const c1 = r1.curves.find((q) => q.id === c.id);
  assert.ok(c1, 'the strut survives a scaled body');
  const bend0 = dist(edited[2], c.birth[2]), bend1 = dist(c1.curve.ctrlPts[2], c1.birth[2]);
  assert.ok(Math.abs(bend1 - 1.2 * bend0) < 1e-6, `bend ${bend0} -> ${bend1}`);
  // A Size that renames every cell orphans the edit, and says so.
  const r2 = buildInfill(mesh, { size: 30, overrides });
  assert.ok(!r2.curves.some((q) => q.id === c.id));
  assert.deepEqual(r2.report.orphans, [{ kind: 'strut', id: c.id }]);
  // Size back: re-attached, same shape.
  const r3 = buildInfill(mesh, { size: 45, overrides });
  const c3 = r3.curves.find((q) => q.id === c.id);
  assert.deepEqual(r3.report.orphans, []);
  for (let k = 0; k < 5; k++) assert.ok(dist(c3.curve.ctrlPts[k], edited[k]) < 1e-9);
});

test('overrides: node offset, delete, add between two nodes, merge, and each orphan kind', () => {
  const r0 = buildInfill(uvSphere(), {});
  const g = r0.graph;
  const deg = degrees(g);
  const ni = g.nodes.findIndex((n, i) => n.kind === 'core' && deg[i] === 4);
  const node = g.nodes[ni];
  const s0 = g.struts.find((s) => s.a === ni || s.b === ni);
  const far = g.nodes.find((n) => n.kind === 'core' && dist(n.p, node.p) > 20);
  const nbr = g.nodes[s0.a === ni ? s0.b : s0.a];
  const ov = {
    nodes: { [node.id]: { offset: [1, 2, 3] }, 'n:99,99,99:0': { offset: [1, 0, 0] } },
    struts: { [s0.id]: { deleted: true }, 'nope': { radius: 2 } },
    added: [{ id: 'my-strut', a: node.id, b: far.id }, { id: 'lost', a: node.id, b: 'n:98,98,98:0' }],
    merged: [[far.id, nbr.id]],
  };
  const r = applyInfillOverrides(g, ov);
  const moved = r.graph.nodes.find((n) => n.id === node.id);
  assert.deepEqual(moved.p, node.p.map((v, i) => v + [1, 2, 3][i]));
  assert.ok(!r.graph.struts.some((s) => s.id === s0.id));
  const added = r.graph.struts.find((s) => s.id === 'my-strut');
  assert.ok(added && r.graph.nodes[added.a].id === node.id && r.graph.nodes[added.b].id === far.id);
  assert.ok(!r.graph.nodes.some((n) => n.id === nbr.id), 'the merged node is gone');
  assert.deepEqual(r.orphans.map((o) => o.kind + ':' + o.id).sort(), ['added:lost', 'node:n:99,99,99:0', 'strut:nope'].sort());
});

test('a stored edit is read at another Detail through its offset spline', () => {
  const r0 = buildInfill(uvSphere(), {});
  const c = r0.curves.find((q) => q.kind === 'core');
  const A = r0.graph.nodes[c.a].p, B = r0.graph.nodes[c.b].p;
  const f = strutChordFrame(A, B);
  // A uniform lift of the three inner points: at detail 7 every inner point
  // lifts by the same amount (the offset spline is constant inside).
  const cv = [[0, 0, 0], [0, 0.1, 0], [0, 0.1, 0], [0, 0.1, 0], [0, 0, 0]];
  const r = buildInfill(uvSphere(), { detail: 7, overrides: { struts: { [c.id]: { cv } } } });
  const c7 = r.curves.find((q) => q.id === c.id);
  const lift = c7.curve.ctrlPts.map((p, k) => { const d = p.map((v, i) => v - c7.birth[k][i]); return d[0] * f.y[0] + d[1] * f.y[1] + d[2] * f.y[2]; });
  assert.ok(Math.abs(lift[0]) < 1e-9 && Math.abs(lift[6]) < 1e-9);
  assert.ok(lift[3] > 0.09 * f.length, `middle lift ${lift[3]}`);
});
