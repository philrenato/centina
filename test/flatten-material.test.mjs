// materialVerdict: which ForMatter materials can make a flattened surface.
// Fixtures are cutAndFlattenNurbsSurface layouts of whole revolves (the seam is
// cut, the pole welds) and an extruded plane; the bands are the bundled
// snapshot app/formability.json, read the way the app reads it.
import test from 'node:test';
import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { makeLine, makeArc, revolve, extrude } from '../kernel/primitives.mjs';
import { principalStrains, grainStrains, materialVerdict, formabilityMode, faceBendRadii, chebyshevDrape } from '../kernel/flatten.mjs';
import { cutAndFlattenNurbsSurface } from '../kernel/seam.mjs';
import { minimumRadiusOfCurvature } from '../kernel/curvature.mjs';

const DEG = Math.PI / 180;
const TABLE = JSON.parse(readFileSync(new URL('./fixtures/formability.json', import.meta.url), 'utf8'));
const ID = {
  cardstock: 'mat_paper_cardstock_index', aluminum: 'mat_aluminum_5052',
  jersey: 'mat_jersey_knit_cotton', neoprene: 'mat_neoprene_wetsuit',
  woven: 'mat_cotton_woven',
};
const record = (id) => TABLE.entries.find((e) => e.id === id).formability[0];

const FIXTURES = {
  plane: () => extrude(makeLine([0, 0, 0], [80, 0, 0]), [0, 1, 0], 50),
  cylinder: () => revolve(makeLine([30, 0, 0], [30, 0, 60]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI),
  thinCylinder: () => revolve(makeLine([0.5, 0, 0], [0.5, 0, 10]), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI),
  cap30: () => revolve(makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 50, 60 * DEG, 30 * DEG, 2), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI),
  hemisphere: () => revolve(makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 50, 0, 90 * DEG, 2), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI),
  sphere: () => revolve(makeArc([0, 0, 0], [1, 0, 0], [0, 0, 1], 50, -90 * DEG, 180 * DEG, 2), [0, 0, 0], [0, 0, 1], 0, 2 * Math.PI),
};
// Fixed in advance. The 30 degree cap needs about 8% of stretch once the
// pattern is cut small enough to compress nothing (jersey's wale band is 20%);
// the hemisphere needs about 133%, past 80% of neoprene's 150% band. Woven
// cotton locks at 35 degrees of shear: a developable surface drapes with none,
// a Chebyshev net on the 30 degree cap shears about 7.5 degrees, on the
// hemisphere about 52 and on the closed sphere close to 90.
const EXPECTED = {
  plane: { cardstock: 'ok', aluminum: 'ok', jersey: 'ok', neoprene: 'ok', woven: 'ok' },
  cylinder: { cardstock: 'ok', aluminum: 'ok', jersey: 'ok', neoprene: 'ok', woven: 'ok' },
  thinCylinder: { cardstock: 'ok', aluminum: 'fails', jersey: 'ok', neoprene: 'ok', woven: 'ok' },
  cap30: { cardstock: 'fails', aluminum: 'fails', jersey: 'ok', neoprene: 'ok', woven: 'ok' },
  hemisphere: { cardstock: 'fails', aluminum: 'fails', jersey: 'fails', neoprene: 'marginal', woven: 'fails' },
  sphere: { woven: 'fails' },
};
const WOVEN_ORIENTATIONS = [0, 45];
const drapesOf = (r) => WOVEN_ORIENTATIONS.map((o) => chebyshevDrape(r.positions, r.faces, { vertexOrigin: r.seam.vertexOrigin, orientationDeg: o }));

function judge(name, res = 32) {
  const srf = FIXTURES[name]();
  const r = cutAndFlattenNurbsSurface(srf, { uRes: res, vRes: res });
  const strains = principalStrains(r.positions, r.faces, r.uv);
  const grain = grainStrains(r.positions, r.faces, r.uv, [1, 0]);
  const faceRadius = faceBendRadii(r.positions, r.faces, r.seam.vertexOrigin);
  const mr = minimumRadiusOfCurvature(srf);
  const drape = drapesOf(r);
  const out = {};
  for (const [k, id] of Object.entries(ID)) {
    out[k] = materialVerdict(strains, record(id), TABLE, { grain, curvatureMin: mr ? mr.radius : null, faceRadius, drape });
  }
  return out;
}

for (const name of Object.keys(FIXTURES)) {
  test(`materialVerdict — ${name}: ${JSON.stringify(EXPECTED[name])}`, () => {
    const v = judge(name);
    for (const k of Object.keys(EXPECTED[name])) {
      const need = v[k].mode === 'shear' ? `${v[k].needShearDeg.toFixed(2)} deg of shear` : `${(100 * v[k].needStretch).toFixed(2)}%`;
      assert.strictEqual(v[k].verdict, EXPECTED[name][k], `${k} on ${name}: ${v[k].verdict} (needs ${need} against ${v[k].band}, bend ${v[k].bend})`);
      // A triangle is shaded exactly when the verdict is not a hold.
      const shaded = v[k].faceState.some((s) => s > 0);
      assert.strictEqual(shaded, v[k].verdict !== 'ok', `${k} on ${name}: shading ${shaded} with verdict ${v[k].verdict}`);
    }
  });
}

test('materialVerdict — the thin cylinder fails aluminum on the bend alone, and every triangle shows it', () => {
  const v = judge('thinCylinder');
  assert.strictEqual(v.aluminum.bend, 'fails');
  assert.ok(v.aluminum.needStretch < 1e-6, `no stretch needed (${v.aluminum.needStretch})`);
  assert.strictEqual(v.aluminum.binding, 'bend');
  assert.ok(Math.abs(v.aluminum.bendRadius - 0.5) < 1e-6 && v.aluminum.bendRequired === 1, `${v.aluminum.bendRadius} against ${v.aluminum.bendRequired}`);
  assert.ok(v.aluminum.faceState.every((s) => s === 2), 'every triangle of a 0.5 mm tube is bent past 1t');
  // Cardstock records no bend radius, so the rule does not bind it.
  assert.strictEqual(v.cardstock.bend, null);
  const wide = judge('cylinder');
  assert.strictEqual(wide.aluminum.bend, 'ok');
});

test('materialVerdict — the numbers behind the verdicts', () => {
  const cap = judge('cap30');
  assert.ok(cap.cardstock.needStretch > 0.03 && cap.cardstock.needStretch < 0.05, `cap rigid strain ${cap.cardstock.needStretch}`);
  assert.ok(cap.neoprene.needStretch > 0.06 && cap.neoprene.needStretch < 0.10, `cap stretch need ${cap.neoprene.needStretch}`);
  assert.ok(cap.cardstock.fracFail > 0.5, `cardstock fails on most of the cap (${cap.cardstock.fracFail})`);
  const hemi = judge('hemisphere');
  assert.ok(hemi.neoprene.needStretch > 1.2 && hemi.neoprene.needStretch < 1.5, `hemisphere stretch need ${hemi.neoprene.needStretch}`);
  assert.strictEqual(hemi.neoprene.fracFail, 0);
  assert.ok(hemi.neoprene.fracMarginal > 0);
  assert.ok(hemi.jersey.grain === 'u' || hemi.jersey.grain === 'v');
  assert.ok(hemi.jersey.thread === 'course' || hemi.jersey.thread === 'wale');
});

test('materialVerdict — a knit is read along both layout axes and the better one is reported', () => {
  // A plane stretched 30% along x only: courses along x (high, 50%) hold, wales
  // along x (moderate, 20%) do not, so the verdict must take grain u.
  const positions = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]];
  const faces = [[0, 1, 2], [1, 3, 2]];
  const uv = positions.map(([x, y]) => [x / 1.3, y]);
  const s = principalStrains(positions, faces, uv);
  const g = grainStrains(positions, faces, uv, [1, 0]);
  const v = materialVerdict(s, record(ID.jersey), TABLE, { grain: g });
  assert.strictEqual(v.grain, 'u');
  assert.ok(Math.abs(v.needStretch - 0.3) < 1e-9, `need ${v.needStretch}`);
  assert.strictEqual(v.thread, 'course');
  assert.strictEqual(v.verdict, 'ok');
  // Turned 90 degrees in the layout, the better orientation is v.
  const uvT = positions.map(([x, y]) => [x, y / 1.3]);
  const vT = materialVerdict(principalStrains(positions, faces, uvT), record(ID.jersey), TABLE, { grain: grainStrains(positions, faces, uvT, [1, 0]) });
  assert.strictEqual(vT.grain, 'v');
  assert.strictEqual(vT.verdict, 'ok');
  // Neoprene is isotropic: the same 30% reads the same either way.
  assert.ok(Math.abs(materialVerdict(s, record(ID.neoprene), TABLE).needStretch - 0.3) < 1e-9);
});

test('formabilityMode — a record with no valid band has no mode, and is never judged', () => {
  assert.strictEqual(formabilityMode(record(ID.cardstock), TABLE), 'inextensible');
  assert.strictEqual(formabilityMode(record(ID.aluminum), TABLE), 'inextensible');
  assert.strictEqual(formabilityMode(record(ID.jersey), TABLE), 'stretch');
  assert.strictEqual(formabilityMode(record(ID.neoprene), TABLE), 'stretch');
  assert.strictEqual(formabilityMode({ stretch: 'stiff', compression: 'rigid' }, TABLE), null);
  assert.strictEqual(formabilityMode({ stretch: { course: 'high' } }, TABLE), null);
  assert.strictEqual(materialVerdict({ s1: new Float64Array(0), s2: new Float64Array(0), area: new Float64Array(0) }, { stretch: 'stiff' }, TABLE), null);
});

test('formabilityMode — a woven record is judged on shear, and only with a valid shear band and no stretch', () => {
  assert.strictEqual(formabilityMode(record(ID.woven), TABLE), 'shear');
  assert.strictEqual(TABLE.shear_bands[record(ID.woven).shear].locking_deg, 35);
  assert.strictEqual(formabilityMode({ shear: 'stiff' }, TABLE), null);
  assert.strictEqual(formabilityMode({ shear: 'moderate', stretch: 'low', compression: 'rigid' }, TABLE), null);
  assert.strictEqual(formabilityMode({ shear: 'moderate' }, { ...TABLE, shear_bands: undefined }), null);
  // Without a drape of the mesh there is nothing to judge.
  const r = cutAndFlattenNurbsSurface(FIXTURES.plane(), { uRes: 8, vRes: 8 });
  assert.strictEqual(materialVerdict(principalStrains(r.positions, r.faces, r.uv), record(ID.woven), TABLE), null);
});

// The drape alone, on the same fixtures. Each bound sits on the far side of a
// number the drape can miss: a cylinder's threads shear 0 only when they keep
// to its rulings and circles, the cap is shallower than the hemisphere, and the
// hemisphere's diagonals pass the locking angle.
const drapeOf = (name, o = 0, res = 32) => {
  const r = cutAndFlattenNurbsSurface(FIXTURES[name](), { uRes: res, vRes: res });
  return chebyshevDrape(r.positions, r.faces, { vertexOrigin: r.seam.vertexOrigin, orientationDeg: o });
};

test('chebyshevDrape — a plane and a cylinder drape with no shear; a 30 degree cap shears less than a hemisphere, which passes 35', () => {
  for (const o of [0, 45]) {
    const p = drapeOf('plane', o);
    assert.ok(p.summary.maxDeg <= 0.1, `plane at ${o} deg: ${p.summary.maxDeg}`);
    assert.ok(p.nodeCount > 500 && [...p.shearDeg].every((s) => s === s), `plane at ${o} deg covers every triangle (${p.nodeCount} nodes)`);
  }
  const cyl = drapeOf('cylinder');
  assert.ok(cyl.summary.maxDeg <= 0.1, `cylinder: ${cyl.summary.maxDeg}`);
  assert.ok(Math.abs(cyl.center[2] - 30) < 2, `the cylinder's drape starts mid-height (${cyl.center})`);
  const cap = drapeOf('cap30'), hemi = drapeOf('hemisphere'), sph = drapeOf('sphere');
  assert.ok(Math.hypot(cap.center[0], cap.center[1]) < 1e-9 && Math.hypot(hemi.center[0], hemi.center[1]) < 1e-9, 'a revolved cap drapes from its pole: the seam is not a cut in the cloth');
  assert.ok(cap.summary.maxDeg > 5 && cap.summary.maxDeg < 10, `30 deg cap: ${cap.summary.maxDeg}`);
  assert.ok(hemi.summary.maxDeg > 45 && hemi.summary.maxDeg < 60, `hemisphere: ${hemi.summary.maxDeg}`);
  assert.ok(cap.summary.maxDeg < hemi.summary.maxDeg && cap.summary.meanDeg < hemi.summary.meanDeg, 'the cap shears less than the hemisphere');
  assert.ok(sph.summary.maxDeg > 80, `closed sphere: ${sph.summary.maxDeg}`);
});

test('chebyshevDrape — the cap and hemisphere numbers hold as the tessellation is refined', () => {
  const cap = drapeOf('cap30', 0, 64), hemi = drapeOf('hemisphere', 0, 64);
  assert.ok(Math.abs(cap.summary.maxDeg - drapeOf('cap30').summary.maxDeg) < 1, `cap at 64: ${cap.summary.maxDeg}`);
  assert.ok(Math.abs(hemi.summary.maxDeg - drapeOf('hemisphere').summary.maxDeg) < 2, `hemisphere at 64: ${hemi.summary.maxDeg}`);
});

test('materialVerdict — woven cotton: the hemisphere fails where the drape passes 35 degrees, and a tighter band fails the cap', () => {
  const hemi = judge('hemisphere').woven, cap = judge('cap30').woven;
  assert.strictEqual(hemi.lockingDeg, 35);
  assert.ok(hemi.needShearDeg > 35 && hemi.fracFail > 0.05 && hemi.fracFail < 0.5, `hemisphere: ${hemi.needShearDeg} deg, ${hemi.fracFail} of area`);
  assert.ok(hemi.binding === 'shear' && hemi.needStretch === null && WOVEN_ORIENTATIONS.includes(hemi.orientationDeg));
  assert.ok(cap.needShearDeg < 0.8 * 35 && cap.fracFail === 0 && cap.fracMarginal === 0, `cap: ${cap.needShearDeg} deg`);
  const r = cutAndFlattenNurbsSurface(FIXTURES.cap30(), { uRes: 32, vRes: 32 });
  const strains = principalStrains(r.positions, r.faces, r.uv), drape = drapesOf(r);
  const banded = (deg) => materialVerdict(strains, { shear: 'x' }, { ...TABLE, shear_bands: { x: { locking_deg: deg } } }, { drape });
  assert.strictEqual(banded(cap.needShearDeg / 0.9).verdict, 'marginal');
  assert.strictEqual(banded(cap.needShearDeg / 1.1).verdict, 'fails');
});
