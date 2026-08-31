// THE ENVIRONMENT'S GPU TABLES, CHECKED BY WALKING THEM THE WAY THE SHADER
// WALKS THEM.
//
// The failure this file exists for is `condOff` / `pdfOff` computed on one side
// only. The conditional binary search then walks numbers that are not that
// row's CDF — and a bisection always terminates, so directions still come out.
// The result is a plausible image, lit from slightly the wrong place,
// converging at the normal rate. Nothing in the picture says anything is wrong,
// and — proved below — the sampler/pdf agreement check does not see it either,
// because both halves recompute the same cell from the same `fu`.
//
// So the load-bearing check here is a WHITE FURNACE DRIVEN THROUGH THE PACKED
// BUFFER: sample by walking `aux` at the offsets that go into the frame
// uniform, and compare `mean(L/pdf)` with the same integral summed directly.
// Calling `envSample` instead would only re-test `kernel/envmap.mjs`, which has
// its own suite. Every furnace here is followed by a control that makes it red.
import { strict as assert } from 'node:assert';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { decodeRadianceHDR, buildEnvDistribution, envPdfDir } from '../kernel/envmap.mjs';
import {
  envAuxLayout, packEnvAux, packEnvTexture, packEnvForGPU,
  envSamplePacked, envPdfDirPacked, envDirToUVPacked, envRotTurns,
  halfFromFloat, HALF_FLOAT_MAX,
} from '../kernel/envpack.mjs';

const ROOT = path.resolve(import.meta.dirname, '..');
const REC709 = [0.2126, 0.7152, 0.0722];
const rng = (seed) => { let s = seed >>> 0; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; };

const img = decodeRadianceHDR(await readFile(path.join(ROOT, 'rendre_assets/studio_env.hdr')));
assert.equal(img.ok, true, `the studio environment did not decode: ${img.why}`);
const pack = packEnvForGPU(img);
const { cdfW, cdfH } = pack;
console.log(`  source ${img.width}x${img.height}, distribution ${cdfW}x${cdfH}`);

/* THE STRIDES, MEASURED FROM THE BUFFER RATHER THAN RESTATED FROM THE LAYOUT.
   Every conditional row is a CDF, so it starts at exactly 0 and ends at exactly
   1 — which makes the row stride findable by trying candidates and keeping the
   one under which every row has those markers. A stride restated from the same
   expression the packer used would agree with it by construction. */
{
  let measuredStride = -1;
  for (let s = cdfW; s <= cdfW + 4; s += 1) {
    let ok = true;
    for (let y = 0; y < cdfH && ok; y += 1) {
      const b = pack.condOff + y * s;
      if (b + cdfW >= pack.aux.length) { ok = false; break; }
      if (Math.abs(pack.aux[b]) > 1e-6 || Math.abs(pack.aux[b + cdfW] - 1) > 1e-6) ok = false;
    }
    if (ok) { measuredStride = s; break; }
  }
  assert.equal(measuredStride, cdfW + 1, `the conditional row stride measures ${measuredStride}, and the shader reads cdfW+1 = ${cdfW + 1}`);
  assert.equal(pack.condStride, cdfW + 1);

  /* And the pdf's offset by the identity that ties the pdf table to the two
     CDFs: the chance of landing in cell (y,x) is the marginal's step times that
     row's conditional step, and the pdf is that density on the unit square —
     so `pdf[y][x] == (marg[y+1]-marg[y]) * (cond[y][x+1]-cond[y][x]) * cw*ch`
     exactly, by construction.

     A weaker criterion is not enough: "the region of ch*cw floats whose mean is
     1" matches a window 82 floats early, straddling the tail of the
     conditionals and the head of the pdf.

     ⚠ THE TOLERANCE IS 2%, NOT AN EPSILON, and that is float32 rather than a
     packing error. Both factors are DIFFERENCES OF NORMALIZED CUMULATIVE SUMS:
     a dim cell contributes ~1e-5 of its row, f32 carries ~1e-7 relative, so the
     subtraction cancels most of the significant digits. Measured here: median
     0.013%, p99 0.37%, worst 0.78%. Tightening this asserts the precision of
     the buffer's own storage format. */
  const identityError = (o, cells) => {
    let worst = 0;
    for (const [y, x] of cells) {
      const b = pack.condOff + y * measuredStride;
      const want = (pack.aux[y + 1] - pack.aux[y]) * (pack.aux[b + x + 1] - pack.aux[b + x]) * cdfW * cdfH;
      const got = pack.aux[o + y * cdfW + x];
      if (want > 1e-12) worst = Math.max(worst, Math.abs(got - want) / want);
      else if (got > 1e-6) worst = Math.max(worst, 1);
    }
    return worst;
  };
  const probes = [];
  for (let i = 0; i < 97; i += 1) probes.push([(i * 37) % cdfH, (i * 61) % cdfW]);
  const matches = [];
  for (let o = pack.condOff + 1; o + cdfH * cdfW <= pack.aux.length; o += 1) {
    if (identityError(o, probes) < 2e-2) matches.push(o);
  }
  assert.equal(matches.length, 1, `${matches.length} offsets satisfy the pdf/CDF identity — the measurement does not name one table`);
  const measuredPdfOff = matches[0];
  assert.equal(measuredPdfOff, pack.pdfOff, `the pdf table measures at offset ${measuredPdfOff}, the uniform is told ${pack.pdfOff}`);

  const all = [];
  for (let y = 0; y < cdfH; y += 1) for (let x = 0; x < cdfW; x += 1) all.push([y, x]);
  const idErr = identityError(measuredPdfOff, all);
  assert.ok(idErr < 2e-2, `the pdf table and the CDFs disagree by ${(idErr * 100).toFixed(4)}% on some cell — the sampler's density is not the pdf it reports`);
  console.log(`  pdf[y][x] == marg-step * cond-step * cw*ch across all ${cdfH * cdfW} cells, worst ${(idErr * 100).toFixed(3)}% (f32 cancellation)`);

  const L = envAuxLayout(cdfW, cdfH);
  assert.equal(pack.margOff, 0, 'the shader bisects the marginal from index 0 with no offset in the uniform');
  assert.equal(pack.condOff, cdfH + 1);
  assert.equal(pack.pdfOff, (cdfH + 1) + cdfH * (cdfW + 1));
  assert.equal(pack.aux.length, L.length);
  assert.equal(pack.byteLength, L.length * 4);
  assert.equal(pack.byteLength % 4, 0, 'a storage buffer write must be a whole number of words');
  console.log(`  measured: marg[${cdfH + 1}]@0  cond[${cdfH}x${measuredStride}]@${pack.condOff}  pdf[${cdfH}x${cdfW}]@${measuredPdfOff}  = ${pack.aux.length} floats / ${pack.byteLength} B`);
}

/* THE PACKED WALK REPRODUCES THE SOURCE TABLES. A copy that dropped or shifted
   a row would still pass the marker checks above on the rows it kept. */
{
  const D = pack.distribution;
  let worst = 0;
  for (let i = 0; i < pack.aux.length; i += 1) worst = Math.max(worst, Math.abs(pack.aux[i] - D.aux[i]));
  assert.ok(worst === 0, `the packed buffer differs from the distribution by ${worst}`);
  console.log('  every float of the packed buffer matches the distribution it came from');
}

/* THE REFERENCE INTEGRAL, summed directly over the working grid. Written here
   rather than imported, so the furnace below compares the packed walk against
   something this file owns. dω = (2π/cw)(π/ch) sinθ. */
function directIrradiance(rgb, width, height, cw, ch) {
  let ref = 0;
  for (let y = 0; y < ch; y += 1) {
    const theta = (Math.PI * (y + 0.5)) / ch;
    const dOmega = ((2 * Math.PI) / cw) * (Math.PI / ch) * Math.sin(theta);
    for (let x = 0; x < cw; x += 1) {
      const sx = Math.min(width - 1, Math.floor(((x + 0.5) * width) / cw));
      const sy = Math.min(height - 1, Math.floor(((y + 0.5) * height) / ch));
      const o = (sy * width + sx) * 3;
      ref += (REC709[0] * rgb[o] + REC709[1] * rgb[o + 1] + REC709[2] * rgb[o + 2]) * dOmega;
    }
  }
  return ref;
}

/** mean(L / pdf) with every direction drawn by walking the PACKED buffer. */
function packedFurnace(p, rgb, width, height, samples, seed) {
  const rnd = rng(seed);
  let est = 0;
  for (let i = 0; i < samples; i += 1) {
    const sm = envSamplePacked(p, rnd(), rnd());
    if (!(sm.pdf > 0)) continue;
    const sx = Math.min(width - 1, Math.floor(sm.u * width));
    const sy = Math.min(height - 1, Math.floor(sm.v * height));
    const o = (sy * width + sx) * 3;
    est += (REC709[0] * rgb[o] + REC709[1] * rgb[o + 1] + REC709[2] * rgb[o + 2]) / sm.pdf;
  }
  return est / samples;
}

/* ⭐ THE WHITE FURNACE, THROUGH THE PACKED TABLES.
   mean(L/pdf) against the same integral summed directly. It agrees only if
   `pdfOff` addresses the table the packer wrote AND the pdf carries its
   `cw*ch/total` normalization AND the solid-angle conversion is right.

   ⚠⚠ AND IT IS STRUCTURALLY BLIND TO `condOff`, which is measured below
   rather than assumed. The algebra: with `condOff` off by a row, a sample in
   row y draws its column from row y+1's conditional, so the estimator picks up
   a factor P(x|y+1)/P(x|y) = [Lw(y+1,x)/rs(y+1)]·[rs(y)/Lw(y,x)]. The pdf is
   read at the SAME cell (y,x) and is proportional to Lw(y,x), so Lw(y,x)
   cancels, the sum over x collapses to rs(y+1), and rs(y+1) cancels too —
   leaving exactly the reference. The TOTAL energy is right; only WHERE the
   samples land is wrong, which is precisely "a plausible image lit from
   slightly the wrong place". The guard for that is the sampled-density check
   in the next block, and this furnace is not a substitute for it. */
const reference = directIrradiance(img.data, img.width, img.height, cdfW, cdfH);
{
  const est = packedFurnace(pack, img.data, img.width, img.height, 300000, 12345);
  const err = Math.abs(est - reference) / reference;
  assert.ok(err < 0.02, `furnace: the packed walk says ${est.toFixed(4)}, direct summation says ${reference.toFixed(4)} — ${(err * 100).toFixed(2)}% apart`);
  console.log(`  furnace through the packed buffer: ${est.toFixed(4)} against ${reference.toFixed(4)} — ${(err * 100).toFixed(2)}% apart`);

  /* ⚠⚠ THE CONTROLS. A furnace that has never been made to fail proves
     nothing, and "the estimator agrees with itself" is exactly the shape of a
     check that cannot fail. Each is an offset in the uniform that no longer
     names the table the packer wrote. */
  const shifted = (field, by) => ({ ...pack, [field]: pack[field] + by });

  const c1 = packedFurnace(shifted('pdfOff', cdfW), img.data, img.width, img.height, 120000, 777);
  const e1 = Math.abs(c1 - reference) / reference;
  assert.ok(e1 > 0.05, `the control failed: pdfOff off by one row still read ${(e1 * 100).toFixed(2)}% — this furnace cannot fail`);
  console.log(`  control, pdfOff off by one row (+${cdfW}): ${c1.toFixed(4)} = ${(e1 * 100).toFixed(1)}% off`);

  const c2 = packedFurnace(shifted('pdfOff', 1), img.data, img.width, img.height, 120000, 777);
  const e2 = Math.abs(c2 - reference) / reference;
  assert.ok(e2 > 0.05, `the control failed: pdfOff off by ONE FLOAT still read ${(e2 * 100).toFixed(2)}%`);
  console.log(`  control, pdfOff off by one float: ${(e2 * 100).toFixed(1)}% off`);

  /* And the normalization — the one that reads as an exposure error and gets
     compensated for in the exposure slider, after which nothing is ever right
     again. Nothing but a furnace sees it. */
  const scaled = { ...pack, aux: Float32Array.from(pack.aux) };
  for (let i = 0; i < cdfH * cdfW; i += 1) scaled.aux[scaled.pdfOff + i] *= 1.35;
  const c3 = packedFurnace(scaled, img.data, img.width, img.height, 120000, 777);
  const e3 = Math.abs(c3 - reference) / reference;
  assert.ok(e3 > 0.15, `the control failed: a pdf scaled by 1.35 still read ${(e3 * 100).toFixed(2)}%`);
  console.log(`  control, pdf scaled by 1.35 (the lost cw*ch/total): ${(e3 * 100).toFixed(1)}% off`);

  /* THE BLIND SPOT, MEASURED. If a later change ever makes the furnace notice a
     `condOff` shift, this assertion goes red and the comment above is wrong —
     which is the only way a documented blind spot stays honest. */
  const cBlind = packedFurnace(shifted('condOff', pack.condStride), img.data, img.width, img.height, 300000, 12345);
  const eBlind = Math.abs(cBlind - reference) / reference;
  assert.ok(eBlind < 0.02, `the furnace now sees a condOff shift (${(eBlind * 100).toFixed(2)}%) — the algebra above says it cannot, so one of the two is wrong`);
  console.log(`  and condOff off by one row reads ${(eBlind * 100).toFixed(2)}% — invisible here, by construction`);
}

/* ⭐⭐ THE GUARD FOR §F STAGE 2'S NAMED FAILURE: WHERE THE SAMPLES LAND.
   `condOff` computed on one side only leaves the total energy exactly right —
   proved above — and moves every sample to the wrong column. So the check is
   the sampled density itself: histogram the (row, col) cells the packed walk
   returns and compare with the pdf table it reports, as a total-variation
   distance. Zero, to Monte Carlo noise, iff the walk's addressing and the
   reported table are the same table.

   ⚠ THE FIXTURE HAS TO BE COARSE AND HIGH-CONTRAST. On the 384x192 studio
   environment the noise floor of this statistic is ~7% at 600k samples — 73728
   cells is too few samples each — and a whole-row `condOff` shift reads 8.9%,
   which is not a signal. A 64x32 image whose bright band MOVES with the row
   puts the floor at 0.7% and the same defect at 67%. A fixture that cannot
   express the defect is not a smaller test, it is no test. */
{
  const W = 64, H = 32, stripe = new Float32Array(W * H * 3).fill(0.01);
  for (let y = 0; y < H; y += 1) {
    const c = (y * 2) % W;
    for (let k = 0; k < 3; k += 1) { const o = (y * W + ((c + k) % W)) * 3; stripe[o] = stripe[o + 1] = stripe[o + 2] = 100; }
  }
  const sp = packEnvForGPU({ width: W, height: H, data: stripe });

  /* Total variation between where the walk puts its samples and the density it
     claims. `p` supplies the offsets the walk uses; the expectation is always
     read from the pdf table at the offset the packer reported. */
  const cellTV = (p, n, seed) => {
    const rnd = rng(seed);
    const obs = new Float64Array(H * W);
    for (let i = 0; i < n; i += 1) { const s = envSamplePacked(p, rnd(), rnd()); obs[s.row * W + s.col] += 1; }
    let t = 0;
    for (let i = 0; i < H * W; i += 1) t += Math.abs(obs[i] - (sp.aux[sp.pdfOff + i] / (W * H)) * n);
    return t / (2 * n);
  };

  const base = cellTV(sp, 400000, 2);
  assert.ok(base < 0.03, `the packed walk puts ${(base * 100).toFixed(2)}% of its samples in cells other than the ones its own pdf table names`);
  console.log(`  sampled density matches the reported pdf table: ${(base * 100).toFixed(2)}% total variation over ${H * W} cells`);

  for (const [label, by] of [['+ one row', sp.condStride], ['+ one float', 1], ['- one float', -1]]) {
    const c = cellTV({ ...sp, condOff: sp.condOff + by }, 400000, 2);
    assert.ok(c > 0.25, `the control failed: condOff ${label} still put ${(c * 100).toFixed(2)}% of samples elsewhere — this check cannot fail`);
    console.log(`  control, condOff ${label}: ${(c * 100).toFixed(1)}% of samples land in the wrong cell`);
  }

  /* AND THE FURNACE IS GREEN ON EVERY ONE OF THOSE. Stated as an assertion so
     the two guards are pinned as complementary rather than redundant. */
  const sref = directIrradiance(stripe, W, H, W, H);
  const sf = packedFurnace({ ...sp, condOff: sp.condOff + sp.condStride }, stripe, W, H, 200000, 3);
  assert.ok(Math.abs(sf - sref) / sref < 0.02, 'the furnace unexpectedly saw a condOff shift on the stripe fixture');
  console.log('  ...and the furnace is green on all three: total energy right, direction wrong');
}

/* ⚠⚠ THE SAMPLER AND THE PDF-FROM-DIRECTION ARE ONE EXPRESSION SPLIT IN TWO,
   and the tracer weights them against each other on every path that reaches the
   environment without being aimed at it. Disagreement errors nowhere: the
   highlights come out subtly too bright or too dim and stay that way.

   Note what this check does NOT see — either offset shift. Both halves recover
   the same cell from the same `fu` and read the pdf table through the same
   `pdfOff`, so they agree on a buffer whose tables are addressed wrongly. It is
   cheap and independent, and a substitute for neither guard above. */
{
  const check = (opts, label, tol) => {
    const rnd = rng(2468);
    let worst = 0, checked = 0;
    for (let i = 0; i < 5000; i += 1) {
      const sm = envSamplePacked(pack, rnd(), rnd(), opts);
      if (!(sm.pdf > 0)) continue;
      const back = envPdfDirPacked(pack, sm.dir, opts);
      worst = Math.max(worst, Math.abs(back - sm.pdf) / sm.pdf);
      checked += 1;
    }
    assert.ok(checked > 4000, `only ${checked} usable samples`);
    assert.ok(worst < tol, `${label}: the packed sampler's pdf and envPdfDirPacked disagree by ${(worst * 100).toFixed(4)}%`);
    console.log(`  ${label}: sampler pdf and pdf-from-direction agree to ${(worst * 100).toExponential(1)}% over ${checked} samples`);
    return worst;
  };
  check({}, 'unrotated', 1e-3);
  // A rotated, pitched dome exercises the inverse warp: `envSamplePacked` maps
  // texel -> world and `envDirToUVPacked` maps world -> texel, and they are
  // only inverses if the pitch solve is right.
  check({ rot: 0.37, height: 0.25 }, 'rot 0.37 turns, pitch 0.25', 2e-3);

  /* THE INDEPENDENT ARM: `envmap.mjs`'s own `envPdfDir`, which knows nothing
     about the packed layout, on directions drawn from the packed buffer. */
  const rnd = rng(13579);
  let worst = 0, checked = 0;
  for (let i = 0; i < 5000; i += 1) {
    const sm = envSamplePacked(pack, rnd(), rnd());
    if (!(sm.pdf > 0)) continue;
    worst = Math.max(worst, Math.abs(envPdfDir(pack.distribution, sm.dir) - sm.pdf) / sm.pdf);
    checked += 1;
  }
  assert.ok(worst < 1e-3, `the packed pdf and kernel/envmap.mjs's envPdfDir disagree by ${(worst * 100).toFixed(4)}%`);
  console.log(`  packed pdf against envmap.mjs envPdfDir: ${(worst * 100).toExponential(1)}% over ${checked} samples`);
}

/* THE ROTATION IS A ROTATION, and it is in TURNS. A whole turn is the identity;
   a quarter turn moves longitude by a quarter and leaves latitude alone. Fed
   degrees, `dirToUV` subtracts 90 from a [0,1] coordinate. */
{
  assert.equal(envRotTurns(360), 1);
  assert.ok(Math.abs(envRotTurns(90) - 0.25) < 1e-12);
  const d = [0.3, 0.5, -0.81];
  const a = envDirToUVPacked(d, { rot: 0 }), b = envDirToUVPacked(d, { rot: 1 }), q = envDirToUVPacked(d, { rot: 0.25 });
  assert.ok(Math.abs(a[0] - b[0]) < 1e-6 && Math.abs(a[1] - b[1]) < 1e-6, 'a whole turn is not the identity');
  // dirToUV SUBTRACTS the rotation, so a quarter turn moves u DOWN by a quarter.
  const moved = ((q[0] - a[0]) % 1 + 1) % 1;
  assert.ok(Math.abs(moved - 0.75) < 1e-6, `a quarter turn moved u by ${(q[0] - a[0]).toFixed(4)}, expected -0.25`);
  assert.ok(Math.abs(q[1] - a[1]) < 1e-9, 'rotation moved latitude');
  console.log('  envRot is in turns: one turn is the identity, a quarter turn moves u down by a quarter and leaves v alone');
}

/* ⚠ THE sin(theta) ROW WEIGHT MUST SURVIVE INTO THE PACKED TABLES. An
   equirectangular image gives polar rows the same texel count while they cover
   almost no solid angle — unweighted, a renderer aims at the poles, where there
   is nothing, and the picture is dim and noisy for a reason nothing in it
   explains.

   The discriminating fixture is a CONSTANT image: with the weight, packed
   samples are distributed by solid angle, so the two polar caps above
   |cos θ| = 0.9 — exactly 10% of the sphere — get 10% of them. Without it, v is
   uniform, θ is uniform, and the caps get 2·acos(0.9)/π = 28.7%. */
{
  const W = 64, H = 32, flat = new Float32Array(W * H * 3).fill(1);
  const weighted = packEnvAux(buildEnvDistribution(flat, W, H));

  /* The control: the same tables built WITHOUT the sinθ factor, laid out by the
     same `envAuxLayout` and walked by the same code, so the weight is the only
     variable between the two runs. */
  const L = envAuxLayout(W, H);
  const aux = new Float32Array(L.length);
  let total = 0;
  for (let y = 0; y < H; y += 1) {
    const base = L.condOff + y * L.condStride;
    let rs = 0;
    for (let x = 0; x < W; x += 1) { rs += 1; aux[base + x + 1] = rs; aux[L.pdfOff + y * W + x] = 1; }
    for (let x = 1; x <= W; x += 1) aux[base + x] /= rs;
    total += rs;
    aux[y + 1] = total;
  }
  for (let y = 1; y <= H; y += 1) aux[y] /= total;
  for (let i = 0; i < H * W; i += 1) aux[L.pdfOff + i] *= (W * H) / total;
  const control = { aux, cdfW: W, cdfH: H, margOff: 0, condOff: L.condOff, condStride: L.condStride, pdfOff: L.pdfOff, pdfStride: W };

  const capFraction = (p, seed) => {
    const rnd = rng(seed);
    const N = 40000;
    let hit = 0;
    for (let i = 0; i < N; i += 1) if (Math.abs(envSamplePacked(p, rnd(), rnd()).dir[1]) > 0.9) hit += 1;
    return hit / N;
  };
  const fw = capFraction(weighted, 31337), fc = capFraction(control, 31337);
  assert.ok(Math.abs(fw - 0.10) < 0.012, `sin(theta) weighting lost: ${(fw * 100).toFixed(1)}% of packed samples land in the polar caps, and solid angle says 10.0%`);
  assert.ok(fc > 0.25, `the control is not a control: the unweighted tables put ${(fc * 100).toFixed(1)}% in the caps, expected ~28.7%`);
  console.log(`  sin(theta) survives: polar caps (10.0% of the sphere) take ${(fw * 100).toFixed(1)}% of packed samples; unweighted control takes ${(fc * 100).toFixed(1)}%`);

  // And it holds under the furnace on the same fixture: a constant environment
  // of 1 integrates to 4π.
  const est = packedFurnace(weighted, flat, W, H, 120000, 24680);
  assert.ok(Math.abs(est - 4 * Math.PI) / (4 * Math.PI) < 0.01, `a constant environment integrates to ${est.toFixed(4)}, not 4pi = ${(4 * Math.PI).toFixed(4)}`);
  console.log(`  constant environment integrates to ${est.toFixed(4)} against 4pi = ${(4 * Math.PI).toFixed(4)}`);
}

/* SAMPLING FOLLOWS THE LIGHT — the point of the distribution. A uniform sampler
   would pass every structural check above. */
{
  const rnd = rng(97531);
  const lums = new Float64Array(img.width * img.height);
  for (let i = 0; i < lums.length; i += 1) {
    const o = i * 3;
    lums[i] = REC709[0] * img.data[o] + REC709[1] * img.data[o + 1] + REC709[2] * img.data[o + 2];
  }
  const cut = Array.from(lums).sort((a, b) => b - a)[Math.floor(lums.length * 0.1)];
  const N = 40000;
  let hit = 0;
  for (let i = 0; i < N; i += 1) {
    const sm = envSamplePacked(pack, rnd(), rnd());
    const x = Math.min(img.width - 1, Math.floor(sm.u * img.width));
    const y = Math.min(img.height - 1, Math.floor(sm.v * img.height));
    if (lums[y * img.width + x] >= cut) hit += 1;
  }
  const frac = hit / N;
  assert.ok(frac > 0.2, `only ${(frac * 100).toFixed(1)}% of packed samples land in the brightest tenth — barely better than the uniform 10%`);
  console.log(`  ${(frac * 100).toFixed(1)}% of packed samples land in the brightest 10% of the image (uniform control: 10.0%)`);
}

/* THE TEXTURE. Half-float bits, alpha forced to 1, and the row pitch
   `queue.writeTexture` wants — which is the tight width*8 here, NOT padded to
   256: that alignment belongs to copyBufferToTexture and padding it would shift
   every row. */
{
  assert.equal(halfFromFloat(0), 0x0000);
  assert.equal(halfFromFloat(1), 0x3c00);
  assert.equal(halfFromFloat(2), 0x4000);
  assert.equal(halfFromFloat(0.5), 0x3800);
  assert.equal(halfFromFloat(HALF_FLOAT_MAX), 0x7bff);
  /* ⚠ NOT Inf. A sun above 65504 clamps: an Inf texel bleeds through the linear
     sampler and turns into NaN downstream, and the accumulation buffer only
     adds — one NaN sample kills that pixel for the whole accumulation. */
  assert.equal(halfFromFloat(1e30), 0x7bff, 'a value past the f16 range became Infinity');
  assert.equal(halfFromFloat(NaN), 0);
  assert.equal(halfFromFloat(-3), 0);
  assert.equal(halfFromFloat(1e-9), 0);
  /* Round-to-nearest-EVEN, not truncation and not round-half-up. f16's step at
     1.0 is 1/1024, so 0.6 of a step rounds up where truncation would drop it,
     and an exact half goes to the even neighbor in BOTH directions — 0x3c00
     down, 0x3c02 up. Round-half-up would give 0x3c01 for the first of those. */
  assert.equal(halfFromFloat(1 + 0.6 / 1024), 0x3c01, 'f16 conversion truncates instead of rounding');
  assert.equal(halfFromFloat(1 + 0.5 / 1024), 0x3c00, 'an exact half did not go to the even neighbor');
  assert.equal(halfFromFloat(1 + 1.5 / 1024), 0x3c02, 'an exact half did not go to the even neighbor');

  const t = pack.tex;
  assert.equal(t.format, 'rgba16float');
  assert.equal(t.width, img.width);
  assert.equal(t.height, img.height);
  assert.equal(t.bytesPerRow, img.width * 8);
  assert.equal(t.rowsPerImage, img.height);
  assert.equal(t.data.length, img.width * img.height * 4);
  assert.equal(t.byteLength, t.bytesPerRow * t.height);
  for (let i = 3; i < t.data.length; i += 4) {
    if (t.data[i] !== 0x3c00) { assert.fail(`alpha at texel ${(i - 3) / 4} is ${t.data[i]}, not 1.0`); }
  }
  // The texel the shader reads must be the radiance that was decoded, to f16.
  const view = new DataView(new ArrayBuffer(2));
  let worst = 0;
  for (let i = 0; i < 4000; i += 1) {
    const px = (i * 7919) % (img.width * img.height);
    for (let c = 0; c < 3; c += 1) {
      view.setUint16(0, t.data[px * 4 + c], true);
      // decode f16 back by hand rather than trusting the encoder both ways
      const bits = t.data[px * 4 + c], e = (bits >> 10) & 0x1f, m = bits & 0x3ff;
      const val = e === 0 ? m * Math.pow(2, -24) : (1 + m / 1024) * Math.pow(2, e - 15);
      const want = Math.min(img.data[px * 3 + c], HALF_FLOAT_MAX);
      if (want > 1e-4) worst = Math.max(worst, Math.abs(val - want) / want);
    }
  }
  /* EXACTLY zero, and that is a fact about RGBE rather than a vacuous check:
     a `.hdr` texel is an 8-bit mantissa times a shared power of two, and f16
     carries 11 significant bits, so every decoded value in f16's exponent range
     is representable with nothing to round. A non-zero here means the encoder
     is losing bits it did not have to. The live-comparison proof is the
     synthetic image below, whose values f16 genuinely has to round. */
  assert.equal(worst, 0, `a decoded .hdr texel lost ${(worst * 100).toFixed(4)}% through f16, and RGBE fits inside f16 exactly`);
  console.log(`  texture ${t.width}x${t.height} rgba16float, ${t.bytesPerRow} B/row, ${(t.byteLength / 1048576).toFixed(2)} MB, ${t.clamped} components clamped, RGBE round trip exact`);

  // Values f16 must round, so the comparison above is demonstrably live.
  const fine = new Float32Array([Math.PI, Math.E, Math.SQRT2, 0.1, 1 / 3, 12345.6]);
  const ft = packEnvTexture(fine, 2, 1);
  let fw = 0, fmin = Infinity;
  for (let i = 0; i < 6; i += 1) {
    const bits = ft.data[(i / 3 | 0) * 4 + (i % 3)], e = (bits >> 10) & 0x1f, m = bits & 0x3ff;
    const val = e === 0 ? m * Math.pow(2, -24) : (1 + m / 1024) * Math.pow(2, e - 15);
    const r = Math.abs(val - fine[i]) / fine[i];
    fw = Math.max(fw, r); fmin = Math.min(fmin, r);
  }
  assert.ok(fmin > 0, 'the round-trip comparison is vacuous — even irrational values came back exact');
  assert.ok(fw < 5e-4, `f16 lost ${(fw * 100).toFixed(4)}%, more than its 11 significant bits allow`);
  console.log(`  values f16 must round come back within ${(fw * 100).toFixed(4)}%, so the comparison is live`);

  /* THE CLAMP COUNTER IS REAL. A sun past 65504 must be reported, not silently
     turned into Inf — an Inf texel bleeds through the linear sampler and one
     NaN sample kills that pixel for the whole accumulation. */
  const hot = new Float32Array(12);
  // Source is 3 floats per texel, the texture is 4 — texel i component c is
  // hot[i*3+c] and data[i*4+c], and conflating the two is how a packer ends up
  // writing green into red.
  hot[0] = 1e6; hot[4] = 2e5; hot[8] = 1;
  const ht = packEnvTexture(hot, 2, 2);
  assert.equal(ht.clamped, 2, `${ht.clamped} components reported clamped, expected 2`);
  assert.equal(ht.data[0], 0x7bff);
  assert.equal(ht.data[5], 0x7bff);
  assert.equal(ht.data[1], 0, 'a component the source left at zero came back non-zero');
  console.log(`  ${ht.clamped} components past the f16 range clamp to 65504 and are counted, not turned into Infinity`);
}

// REFUSALS: a table too small to bisect, and a short image, are named rather
// than packed into a buffer that reads as garbage on the GPU.
{
  assert.throws(() => packEnvAux({ aux: new Float32Array(4), cw: 1, ch: 1, condOff: 2, pdfOff: 3 }), /at least 2x2/);
  assert.throws(() => packEnvAux({ aux: new Float32Array(4), cw: 4, ch: 4, condOff: 5, pdfOff: 25 }), /too short/);
  assert.throws(() => packEnvTexture(new Float32Array(12), 4, 4), /short of/);
  assert.throws(() => packEnvTexture(new Float32Array(12), 0, 4), /positive size/);
  console.log('  a distribution too small to bisect and a short image are refused by name');
}

console.log('envpack: ok');
