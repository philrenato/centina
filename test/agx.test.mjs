// A TRANSPOSED COLOR MATRIX IS INVISIBLE IN THE SOURCE AND INVISIBLE ON SCREEN.
//
// AgX's two matrices are near-identity: every row sums to about one either way
// round, every entry keeps its magnitude, and the transform still returns
// plausible colors. Transposing one of them puts a warm-pink cast on every
// pixel, and because the error is multiplicative, brightening a surface makes it
// worse rather than whiter — so the search goes to the lighting and stays there.
//
// The two checks that see it are here, and each is proved against the defect
// rather than assumed to catch it: the round trip must be the identity, and a
// neutral input must come back neutral.
import { strict as assert } from 'node:assert';
import { agx, agxExposed, agxContrast, AGX_M, AGX_MINV, AGX_MIN_EV, AGX_MAX_EV } from '../kernel/agx.mjs';

const transpose = (m) => [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];

/* An independently shaped evaluation of the shader's own expression: the
   matrices are turned into explicit ROWS and applied as dot products, rather
   than as the column-indexed multiply the module uses. Same arithmetic, written
   the other way round, so a convention error in one does not reproduce in the
   other. `minv` is a parameter precisely so the transposed variant can be run
   through the identical path. */
function agxRowwise(rgb, minv = AGX_MINV) {
  const rows = (m) => [[m[0], m[3], m[6]], [m[1], m[4], m[7]], [m[2], m[5], m[8]]];
  const apply = (r, v) => r.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
  const span = AGX_MAX_EV - AGX_MIN_EV;
  const lit = apply(rows(AGX_M), rgb.map((c) => Math.max(c, 1e-10)));
  const curved = lit.map((c) => {
    const ev = Math.min(Math.max(Math.log2(c), AGX_MIN_EV), AGX_MAX_EV);
    return agxContrast((ev - AGX_MIN_EV) / span);
  });
  return apply(rows(minv), curved).map((c) => Math.min(Math.max(c, 0), 1));
}

const spread = (c) => Math.max(c[0], c[1], c[2]) - Math.min(c[0], c[1], c[2]);

// ------------------------------------------------------- the round trip ----
/* M and its inverse are exact to machine precision, which is the tight form of
   the neutrality claim: this is the property the transpose destroys, and it
   destroys it by four orders of magnitude. */
{
  const mul = (A, B) => {
    const o = new Array(9).fill(0);
    for (let c = 0; c < 3; c += 1) for (let r = 0; r < 3; r += 1) {
      let s = 0; for (let k = 0; k < 3; k += 1) s += A[k * 3 + r] * B[c * 3 + k];
      o[c * 3 + r] = s;
    }
    return o;
  };
  const off = (p) => {
    let worst = 0;
    for (let c = 0; c < 3; c += 1) for (let r = 0; r < 3; r += 1) worst = Math.max(worst, Math.abs(p[c * 3 + r] - (c === r ? 1 : 0)));
    return worst;
  };
  const good = off(mul(AGX_M, AGX_MINV));
  const flipped = off(mul(AGX_M, transpose(AGX_MINV)));
  assert.ok(good < 1e-12, `M * Minv is off the identity by ${good} — the inverse does not invert`);
  assert.ok(flipped > 1e-2, 'a transposed Minv must be visibly not an inverse, or this check proves nothing');
  console.log(`  round trip:     M * Minv is the identity to ${good.toExponential(1)}; transposed it misses by ${flipped.toExponential(1)}`);
}

// ----------------------------------------------- neutral in, neutral out ----
/* ⚠ THE TOLERANCE IS 1e-3, NOT 1e-6, AND THE NUMBER IS MEASURED RATHER THAN
   CHOSEN. AgX's inset matrix is not row-normalized — its three row sums differ
   in the fourth decimal — and the sigmoid between the two rotations amplifies
   that into a channel spread the round trip cannot cancel. The transform is
   therefore neutral to about 2.3e-4 across its whole domain, and an assertion
   at 1e-6 would fail on the CORRECT matrices. The transposed defect misses by
   nearly seventy times that bound (printed by the check below), so the looser
   tolerance costs this nothing. */
const NEUTRAL_TOL = 1e-3;
{
  const grey = agx([0.18, 0.18, 0.18]);
  assert.ok(spread(grey) < 1e-4, `middle gray came back with a channel spread of ${spread(grey)}`);

  let worst = 0, worstAt = 0, samples = 0;
  for (let ev = -14; ev <= 6; ev += 0.02) {
    const v = 2 ** ev;
    const s = spread(agx([v, v, v]));
    if (s > worst) { worst = s; worstAt = v; }
    samples += 1;
  }
  assert.ok(worst < NEUTRAL_TOL, `a neutral input came back with a spread of ${worst} at ${worstAt}`);
  console.log(`  neutrality:     ${samples} neutral inputs, worst channel spread ${worst.toExponential(2)} at ${worstAt.toFixed(3)} (bound ${NEUTRAL_TOL})`);
}

// -------------------------------------- the transpose, run and confirmed ----
/* The defect reproduced exactly: the shader's own record of the incident says a
   neutral gray came back as (1.091, 0.956, 0.953) relative to correct. Matching
   those three numbers is what ties this module to the transform that actually
   shipped, rather than to a plausible reimplementation of it. */
{
  const good = agx([0.18, 0.18, 0.18]);
  const bad = agxRowwise([0.18, 0.18, 0.18], transpose(AGX_MINV));
  const ratio = bad.map((v, i) => v / good[i]);
  assert.ok(spread(bad) > NEUTRAL_TOL, 'the neutrality check does not fail on a transposed Minv — it is not a check');
  assert.ok(Math.abs(ratio[0] - 1.091) < 5e-4, `red should come back 9.1% hot, got ${ratio[0]}`);
  assert.ok(Math.abs(ratio[1] - 0.956) < 5e-4, `green should come back 4.4% cold, got ${ratio[1]}`);
  assert.ok(Math.abs(ratio[2] - 0.953) < 5e-4, `blue should come back 4.7% cold, got ${ratio[2]}`);
  console.log(`  the defect:     transposed Minv tints neutral gray by (${ratio.map((v) => v.toFixed(3)).join(', ')}) — spread ${spread(bad).toFixed(4)}, ${(spread(bad) / NEUTRAL_TOL).toFixed(0)}x the bound`);
}

// ----------------------------------------- non-neutral, against the WGSL ----
/* Symmetry alone would be satisfied by any transform that treats the three
   channels alike, including the wrong one. These are saturated and mixed
   colors, checked against the row-wise transcription of the shader's
   expression. */
{
  const cases = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [0.8, 0.2, 0.05], [0.05, 0.3, 0.9], [2.5, 1.2, 0.4],
    [0.0001, 0.5, 12], [64, 0.02, 0.02], [0, 0, 0], [1e-12, 1e-12, 1e-12],
  ];
  let worst = 0;
  for (const c of cases) {
    const a = agx(c), b = agxRowwise(c);
    for (let i = 0; i < 3; i += 1) {
      worst = Math.max(worst, Math.abs(a[i] - b[i]));
      assert.ok(a[i] >= 0 && a[i] <= 1, `agx(${c}) channel ${i} left [0,1] at ${a[i]}`);
    }
  }
  assert.ok(worst < 1e-12, `the column-indexed and row-wise evaluations disagree by ${worst}`);

  // Saturated inputs stay ordered: AgX desaturates, it does not re-hue.
  const red = agx([1, 0.02, 0.02]);
  assert.ok(red[0] > red[1] && red[0] > red[2], 'a red input must come back reddest');
  const blue = agx([0.02, 0.02, 1]);
  assert.ok(blue[2] > blue[0] && blue[2] > blue[1], 'a blue input must come back bluest');
  console.log(`  non-neutral:    ${cases.length} colors match the shader's expression to ${worst.toExponential(1)}, hue order preserved`);
}

// ------------------------------------------------ the curve and the window ----
{
  // Below the window everything is the same black; above it, the same white.
  const floor0 = agx([0, 0, 0]);
  assert.deepEqual(floor0, [0, 0, 0], 'zero radiance is black, not NaN — the input is floored before the log');
  const under = agx([2 ** (AGX_MIN_EV - 4), 2 ** (AGX_MIN_EV - 4), 2 ** (AGX_MIN_EV - 4)]);
  assert.ok(Math.max(...under) < 1e-3, 'below the window is black');
  const over = agx([2 ** (AGX_MAX_EV + 4), 2 ** (AGX_MAX_EV + 4), 2 ** (AGX_MAX_EV + 4)]);
  assert.ok(Math.min(...over) > 0.99, 'above the window is white');

  // Monotone: more light is never less display value.
  let prev = -1, steps = 0;
  for (let ev = AGX_MIN_EV; ev <= AGX_MAX_EV; ev += 0.05) {
    const v = agx([2 ** ev, 2 ** ev, 2 ** ev])[1];
    assert.ok(v >= prev - 1e-12, `the transform is not monotone at ${ev} EV (${v} after ${prev})`);
    prev = v; steps += 1;
  }
  assert.ok(agxContrast(0) < 0 && agxContrast(1) > 0.99, 'the sigmoid spans its unit domain');
  console.log(`  window:         black below ${AGX_MIN_EV} EV, white above ${AGX_MAX_EV}, monotone across ${steps} steps`);
}

// --------------------------------------------------------------- exposure ----
/* Exposure is applied in LINEAR light, before the transform. After it, the
   scale would multiply display-encoded values and flatten the rolloff the
   transform exists to provide. */
{
  for (const e of [0.25, 1, 4, 17.5]) {
    const a = agxExposed([0.3, 0.15, 0.6], e);
    const b = agx([0.3 * e, 0.15 * e, 0.6 * e]);
    for (let i = 0; i < 3; i += 1) assert.ok(Math.abs(a[i] - b[i]) < 1e-15, `exposure ${e} is not applied in linear light`);
  }
  // And it writes into the array it is given, so a caller may pass a pixel view.
  const buf = new Float32Array(3);
  const ret = agx([0.5, 0.5, 0.5], buf);
  assert.equal(ret, buf, 'agx returns the output it was handed');
  assert.ok(buf[0] > 0, 'agx wrote into the caller\'s array');
  console.log('  exposure:       four exposures equal a linear pre-multiply; the output array is the caller\'s');
}

console.log('agx: ok');
