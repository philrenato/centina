import { test } from 'node:test';
import assert from 'node:assert/strict';
import { applyPointEdits, resamplePointEditsToNet, pointEditFieldMass } from '../kernel/pointedit.mjs';

function makeNet(nu, nv) {
  const net = [];
  for (let i = 0; i < nu; i++) {
    const row = [];
    for (let j = 0; j < nv; j++) row.push([i * 10, j * 10, 0, 1]);
    net.push(row);
  }
  return { degU: 3, degV: 3, knotsU: [], knotsV: [], ctrlNet: net };
}

// ---- existing applyPointEdits (WORLD-frame) behaviour — must stay unchanged

test('applyPointEdits: empty/undefined edits is a true no-op (same object)', () => {
  const srf = makeNet(4, 4);
  assert.equal(applyPointEdits(srf, []), srf);
  assert.equal(applyPointEdits(srf, undefined), srf);
});

test('applyPointEdits: a single edit lands at the correct rounded row/col and leaves the rest untouched', () => {
  const srf = makeNet(5, 5); // rowFrac/colFrac in [0,1] over 4 spans
  const out = applyPointEdits(srf, [{ rowFrac: 0.5, colFrac: 0.25, delta: [1, 2, 3] }]);
  // 0.5*(5-1)=2, 0.25*(5-1)=1
  assert.deepEqual(out.ctrlNet[2][1], [21, 12, 3, 1]);
  // original untouched (a fresh clone)
  assert.deepEqual(srf.ctrlNet[2][1], [20, 10, 0, 1]);
  // a neighbor is unaffected
  assert.deepEqual(out.ctrlNet[2][2], [20, 20, 0, 1]);
});

test('applyPointEdits: multiple edits accumulate independently at distinct points', () => {
  const srf = makeNet(4, 4);
  const out = applyPointEdits(srf, [
    { rowFrac: 0, colFrac: 0, delta: [5, 0, 0] },
    { rowFrac: 1, colFrac: 1, delta: [0, 5, 0] },
  ]);
  assert.deepEqual(out.ctrlNet[0][0], [5, 0, 0, 1]);
  assert.deepEqual(out.ctrlNet[3][3], [30, 35, 0, 1]);
});

test('applyPointEdits: two edits landing on the SAME rounded point both apply (additive, not last-wins-overwrite)', () => {
  const srf = makeNet(4, 4);
  const out = applyPointEdits(srf, [
    { rowFrac: 0, colFrac: 0, delta: [1, 0, 0] },
    { rowFrac: 0, colFrac: 0, delta: [1, 0, 0] },
  ]);
  assert.deepEqual(out.ctrlNet[0][0], [2, 0, 0, 1]);
});

test('applyPointEdits: out-of-range fractions clamp to the nearest real row/col instead of throwing', () => {
  const srf = makeNet(3, 3);
  const out = applyPointEdits(srf, [{ rowFrac: -0.5, colFrac: 5, delta: [1, 1, 1] }]);
  assert.deepEqual(out.ctrlNet[0][2], [1, 21, 1, 1]);
});

test('applyPointEdits: rational weight (index 3) is never touched', () => {
  const srf = makeNet(3, 3);
  srf.ctrlNet[1][1][3] = 0.7071;
  const out = applyPointEdits(srf, [{ rowFrac: 0.5, colFrac: 0.5, delta: [1, 1, 1] }]);
  assert.equal(out.ctrlNet[1][1][3], 0.7071);
});

// ---- BLEND (0=Edited, 1=Source) ---------------------------------------------

test('applyPointEdits: default (no blend arg) is byte-identical to blend=0 (full Edited)', () => {
  const srf = makeNet(4, 4);
  const edits = [{ rowFrac: 0, colFrac: 0, delta: [5, 6, 7] }];
  const noArg = applyPointEdits(srf, edits);
  const explicit0 = applyPointEdits(srf, edits, 0);
  assert.deepEqual(noArg.ctrlNet[0][0], explicit0.ctrlNet[0][0]);
  assert.deepEqual(noArg.ctrlNet[0][0], [5, 6, 7, 1]);
});

test('applyPointEdits: blend=1 (fully Source) is a genuine, exact no-op — zero displacement', () => {
  const srf = makeNet(4, 4);
  const out = applyPointEdits(srf, [{ rowFrac: 0.5, colFrac: 0.5, delta: [9, 9, 9] }], 1);
  assert.equal(out, srf); // early-return identity, not just numerically close to unchanged
});

test('applyPointEdits: blend=0.5 is a real, provable LINEAR INTERPOLATION between Edited and Source, independently re-derived', () => {
  const srf = makeNet(4, 4);
  const edits = [{ rowFrac: 1 / 3, colFrac: 2 / 3, delta: [10, -20, 4] }];
  const edited = applyPointEdits(srf, edits, 0); // blend=0 == fully Edited
  const source = applyPointEdits(srf, edits, 1); // blend=1 == fully Source (== srf itself)
  const half = applyPointEdits(srf, edits, 0.5);
  // Independently compute the expected lerp in the TEST itself (never trust
  // the function's own internal math as its own proof) — the exact
  // component-wise midpoint between the Edited and Source control points.
  const i = Math.round((1 / 3) * 3), j = Math.round((2 / 3) * 3);
  const expected = [0, 1, 2].map((k) => (edited.ctrlNet[i][j][k] + source.ctrlNet[i][j][k]) / 2);
  assert.deepEqual(half.ctrlNet[i][j].slice(0, 3), expected);
  // A second, sharper cross-check: the delta actually applied at blend=0.5
  // must be EXACTLY half the full delta (5, -10, 2), not merely "between".
  assert.deepEqual(
    [0, 1, 2].map((k) => half.ctrlNet[i][j][k] - source.ctrlNet[i][j][k]),
    [5, -10, 2],
  );
});

test('applyPointEdits: blend scales a NORMAL-frame amount too, not just a WORLD-frame delta', () => {
  // A flat XY plane (bilinear, degree 1x1) with a known analytic normal (+Z).
  const srf = {
    degU: 1, degV: 1, knotsU: [0, 0, 1, 1], knotsV: [0, 0, 1, 1],
    ctrlNet: [[[0, 0, 0, 1], [0, 10, 0, 1]], [[10, 0, 0, 1], [10, 10, 0, 1]]],
  };
  const edits = [{ rowFrac: 0, colFrac: 0, frame: 'normal', amount: 8 }];
  const full = applyPointEdits(srf, edits, 0);
  const half = applyPointEdits(srf, edits, 0.5);
  const none = applyPointEdits(srf, edits, 1);
  assert.equal(none, srf);
  assert.ok(Math.abs(full.ctrlNet[0][0][2] - 8) < 1e-9, 'full amount applied along +Z at blend=0');
  assert.ok(Math.abs(half.ctrlNet[0][0][2] - 4) < 1e-9, 'exactly half the normal displacement at blend=0.5');
});

// ---- NORMAL-frame anchoring -------------------------------------------------

// A genuine bilinear (degree 1x1) surface, so surfacePointAndPartials returns
// a real, analytically-known normal. `xform` maps each planar (x,y,0) control
// point to world space, so we can reorient the SAME surface and prove the
// normal-frame edit rides the new orientation.
function makeFlatSrf(xform = (p) => p) {
  const grid = [];
  for (let i = 0; i < 3; i++) {
    const row = [];
    for (let j = 0; j < 3; j++) {
      const [x, y, z] = xform([i * 10, j * 10, 0]);
      row.push([x, y, z, 1]);
    }
    grid.push(row);
  }
  return { degU: 1, degV: 1, knotsU: [0, 0, 0.5, 1, 1], knotsV: [0, 0, 0.5, 1, 1], ctrlNet: grid };
}
const rotX90 = (p) => [p[0], -p[2], p[1]]; // +90 deg about world X

test('applyPointEdits (normal frame): a flat XY surface displaces the CP along +Z (its own normal)', () => {
  const srf = makeFlatSrf();
  const out = applyPointEdits(srf, [{ rowFrac: 0.5, colFrac: 0.5, frame: 'normal', amount: 5 }]);
  const cp = out.ctrlNet[1][1];
  // moved purely along Z (the flat surface's own normal), by exactly the amount
  assert.ok(Math.abs(cp[0] - 10) < 1e-9 && Math.abs(cp[1] - 10) < 1e-9, 'no in-plane drift');
  assert.ok(Math.abs(Math.abs(cp[2]) - 5) < 1e-9, `moved 5 along Z, got ${cp[2]}`);
  assert.equal(cp[3], 1);
});

test('normal frame RIDES a reoriented surface — the displacement follows the NEW normal, not the old world direction', () => {
  const edit = [{ rowFrac: 0.5, colFrac: 0.5, frame: 'normal', amount: 5 }];
  const flat = makeFlatSrf();
  const rotated = makeFlatSrf(rotX90); // same surface, reoriented into the XZ plane (normal now ±Y)

  const flatCp = applyPointEdits(flat, edit).ctrlNet[1][1];
  const rotCp = applyPointEdits(rotated, edit).ctrlNet[1][1];

  // displacement = edited CP - natural CP, in each surface's own world frame
  const flatDisp = [flatCp[0] - flat.ctrlNet[1][1][0], flatCp[1] - flat.ctrlNet[1][1][1], flatCp[2] - flat.ctrlNet[1][1][2]];
  const rotDisp = [rotCp[0] - rotated.ctrlNet[1][1][0], rotCp[1] - rotated.ctrlNet[1][1][1], rotCp[2] - rotated.ctrlNet[1][1][2]];

  const mag = (v) => Math.hypot(...v);
  assert.ok(Math.abs(mag(flatDisp) - 5) < 1e-9, 'flat displacement magnitude is the amount');
  assert.ok(Math.abs(mag(rotDisp) - 5) < 1e-9, 'reoriented displacement magnitude is still the amount');

  // flat surface: displacement is along Z (its normal)
  assert.ok(Math.abs(flatDisp[0]) < 1e-9 && Math.abs(flatDisp[1]) < 1e-9 && Math.abs(Math.abs(flatDisp[2]) - 5) < 1e-9,
    `flat rides +/-Z, got ${flatDisp}`);
  // reoriented surface: displacement is along Y (the NEW normal) — provably NOT the old world Z
  assert.ok(Math.abs(rotDisp[0]) < 1e-9 && Math.abs(Math.abs(rotDisp[1]) - 5) < 1e-9 && Math.abs(rotDisp[2]) < 1e-9,
    `reoriented rides +/-Y (the new normal), got ${rotDisp}`);
  // the whole point: the two displacements point in genuinely different directions
  const dot = flatDisp[0] * rotDisp[0] + flatDisp[1] * rotDisp[1] + flatDisp[2] * rotDisp[2];
  assert.ok(Math.abs(dot) < 1e-9, 'flat (Z) and reoriented (Y) displacements are orthogonal — the normal genuinely re-anchored');
});

test('world frame is BYTE-IDENTICAL under reorientation — the CV-drag case is unaffected (zero regression proof)', () => {
  const edit = [{ rowFrac: 0.5, colFrac: 0.5, delta: [0, 0, 5] }]; // no `frame` -> world
  const rotated = makeFlatSrf(rotX90);
  const out = applyPointEdits(rotated, edit);
  const cp = out.ctrlNet[1][1];
  const nat = rotated.ctrlNet[1][1];
  // a world edit adds the SAME fixed world vector regardless of the surface's
  // orientation (numeric compare; +0 vs -0 would trip deepEqual)
  assert.ok(Math.abs(cp[0] - nat[0]) < 1e-12 && Math.abs(cp[1] - nat[1]) < 1e-12
    && Math.abs(cp[2] - (nat[2] + 5)) < 1e-12 && cp[3] === 1, `world edit unchanged by reorientation, got ${cp}`);
});

// ---- resamplePointEditsToNet ------------------------------------------------

test('resample: empty field resamples to empty', () => {
  assert.deepEqual(resamplePointEditsToNet([], { nu: 5, nv: 5 }, { nu: 9, nv: 9 }), []);
  assert.deepEqual(resamplePointEditsToNet(undefined, { nu: 5, nv: 5 }, { nu: 9, nv: 9 }), []);
});

test('resample: a single edit onto the SAME net shape reproduces itself EXACTLY', () => {
  const edits = [{ rowFrac: 0.5, colFrac: 0.25, delta: [1, 2, 3] }];
  const out = resamplePointEditsToNet(edits, { nu: 5, nv: 5 }, { nu: 5, nv: 5 });
  assert.equal(out.length, 1);
  assert.ok(Math.abs(out[0].rowFrac - 0.5) < 1e-12 && Math.abs(out[0].colFrac - 0.25) < 1e-12);
  assert.ok(Math.abs(out[0].delta[0] - 1) < 1e-12 && Math.abs(out[0].delta[1] - 2) < 1e-12 && Math.abs(out[0].delta[2] - 3) < 1e-12);
});

test('resample: a NORMAL-frame edit onto the same shape reproduces its scalar amount exactly', () => {
  const edits = [{ rowFrac: 0.5, colFrac: 0.5, frame: 'normal', amount: 7.5 }];
  const out = resamplePointEditsToNet(edits, { nu: 5, nv: 5 }, { nu: 5, nv: 5 });
  assert.equal(out.length, 1);
  assert.equal(out[0].frame, 'normal');
  assert.ok(Math.abs(out[0].amount - 7.5) < 1e-12);
});

test('resample REFINE (coarse->fine) fills the gaps AND conserves the area-weighted field mass', () => {
  // a single central bump on a 3x3 net — the classic "a denser Rebuild leaves
  // nothing to splat into" case
  const edits = [{ rowFrac: 0.5, colFrac: 0.5, delta: [0, 0, 6] }];
  const oldShape = { nu: 3, nv: 3 }, newShape = { nu: 9, nv: 9 };
  const out = resamplePointEditsToNet(edits, oldShape, newShape);

  // NO GAP: an intermediate new control point (frac 0.375,0.375), which sits
  // strictly BETWEEN the coarse bump nodes, got a real interpolated value
  const between = out.find((e) => Math.abs(e.rowFrac - 0.375) < 1e-9 && Math.abs(e.colFrac - 0.375) < 1e-9);
  assert.ok(between && Math.abs(between.delta[2]) > 1e-6, 'the gap between coarse nodes is filled, not left empty');

  // NO OVERSHOOT/AMPLIFICATION: nothing on the fine net exceeds the coarse peak
  const peak = Math.max(...out.map((e) => Math.abs(e.delta[2])));
  assert.ok(peak <= 6 + 1e-9, `no amplification (fine peak ${peak} <= coarse 6)`);
  // the peak IS reproduced at the coinciding centre node (frac 0.5)
  const centre = out.find((e) => Math.abs(e.rowFrac - 0.5) < 1e-9 && Math.abs(e.colFrac - 0.5) < 1e-9);
  assert.ok(centre && Math.abs(centre.delta[2] - 6) < 1e-9, 'the true peak survives exactly at the coinciding node');

  // MASS conserved (area-weighted "displaced volume", not raw per-node sum)
  const m0 = pointEditFieldMass(edits, oldShape).world[2];
  const m1 = pointEditFieldMass(out, newShape).world[2];
  assert.ok(Math.abs(m1 - m0) < 1e-9, `field mass conserved across refine (${m0} -> ${m1})`);
});

test('resample COARSEN (fine->coarse) NEVER lumps — each new CP is one bounded interpolated value, no additive stacking', () => {
  // a dense multi-CP bump on a 9x9 net (what a real sculpt stroke produces),
  // resampled down to a 4x4 net (a coarser Rebuild)
  const edits = [];
  let maxIn = 0;
  for (let i = 3; i <= 5; i++) {
    for (let j = 3; j <= 5; j++) {
      const amt = 4 - (Math.abs(i - 4) + Math.abs(j - 4)); // 4 at centre, 3 on the ring
      edits.push({ rowFrac: i / 8, colFrac: j / 8, delta: [0, 0, amt] });
      maxIn = Math.max(maxIn, amt);
    }
  }
  const out = resamplePointEditsToNet(edits, { nu: 9, nv: 9 }, { nu: 4, nv: 4 });
  // NO LUMP: the coarse net's largest displacement is bounded by the input
  // field's own max — the old nearest-round-and-stack would ADD several
  // colliding entries and exceed it
  const maxOut = Math.max(...out.map((e) => Math.abs(e.delta[2])));
  assert.ok(maxOut <= maxIn + 1e-9, `coarse peak ${maxOut} bounded by input peak ${maxIn} — no additive lumping`);
  // and mass is still approximately conserved (a coarsening sub-samples, so
  // this is an APPROXIMATE invariant, unlike the exact refine case above)
  const m0 = pointEditFieldMass(edits, { nu: 9, nv: 9 }).world[2];
  const m1 = pointEditFieldMass(out, { nu: 4, nv: 4 }).world[2];
  assert.ok(Math.abs(m1 - m0) / Math.abs(m0) < 0.15, `field mass roughly conserved across coarsen (${m0} -> ${m1})`);
});

test('resample: a resampled field applied to the new net is smooth (monotone falloff, no jagged gaps)', () => {
  // resample the single central bump to a fine net, apply it, and confirm the
  // resulting displacements decrease monotonically outward from the centre —
  // the concrete "stays smooth/non-lumped" property the whole piece exists for
  const edits = [{ rowFrac: 0.5, colFrac: 0.5, delta: [0, 0, 6] }];
  const newNu = 9;
  const out = resamplePointEditsToNet(edits, { nu: 3, nv: 3 }, { nu: newNu, nv: newNu });
  // read the central row (colFrac 0.5) as a 1D profile
  const profile = [];
  for (let a = 0; a < newNu; a++) {
    const e = out.find((x) => Math.abs(x.rowFrac - a / (newNu - 1)) < 1e-9 && Math.abs(x.colFrac - 0.5) < 1e-9);
    profile.push(e ? e.delta[2] : 0);
  }
  const mid = (newNu - 1) / 2;
  // strictly rising toward the centre, strictly falling after it — no gaps/spikes
  for (let a = 1; a <= mid; a++) assert.ok(profile[a] >= profile[a - 1] - 1e-12, `rising to centre at ${a}`);
  for (let a = mid + 1; a < newNu; a++) assert.ok(profile[a] <= profile[a - 1] + 1e-12, `falling from centre at ${a}`);
  assert.ok(Math.abs(profile[mid] - 6) < 1e-9, 'centre peak intact');
});
