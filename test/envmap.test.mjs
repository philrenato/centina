// THE ENVIRONMENT AND ITS SAMPLING DISTRIBUTION.
//
// Everything here runs on the CPU, which matters: the failures this catches are
// invisible in a rendered image. An unnormalised pdf makes every render off by
// one constant factor, which reads as an exposure error and gets
// compensated for in the exposure — after which the renderer is permanently and
// invisibly mis-weighted. A furnace test is how that gets caught, and it needs
// no renderer at all.
import { strict as assert } from 'node:assert';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import {
  decodeRadianceHDR, buildEnvDistribution, envSample, envPdfDir,
  envUVToDir, envDirToUV, envFurnaceError,
} from '../kernel/envmap.mjs';

const ROOT = path.resolve(import.meta.dirname, '..');
const img = decodeRadianceHDR(await readFile(path.join(ROOT, 'rendre_assets/studio_env.hdr')));
assert.equal(img.ok, true, `the studio environment did not decode: ${img.why}`);
assert.ok(img.width > 0 && img.height > 0);
/* ⚠ AN RGBE EXPONENT OF ZERO MEANS BLACK, not 2^-128. Read as a power it puts a
   floor of about 1e-38 under the whole image, which is invisible until something
   divides by it. */
let mn = Infinity, mx = -Infinity;
for (let i = 0; i < img.data.length; i += 1) { if (img.data[i] < mn) mn = img.data[i]; if (img.data[i] > mx) mx = img.data[i]; }
assert.ok(mn >= 0, `negative radiance ${mn}`);
assert.ok(mx > 1, `nothing in this environment is brighter than 1 (max ${mx}) — that is not an HDR`);
console.log(`  decoded ${img.width}x${img.height}, radiance ${mn.toFixed(4)}..${mx.toFixed(3)}`);

const D = buildEnvDistribution(img.data, img.width, img.height);
console.log(`  distribution ${D.cw}x${D.ch}, aux ${D.aux.length} floats`);

// THE CDFs ARE CDFs: monotone, 0 at the start, exactly 1 at the end. A CDF that
// does not reach 1 makes the last row or column unreachable.
{
  for (let y = 0; y <= D.ch; y += 1) {
    if (y) assert.ok(D.aux[y] >= D.aux[y - 1] - 1e-6, `marginal CDF decreases at row ${y}`);
  }
  assert.ok(Math.abs(D.aux[0]) < 1e-6, 'the marginal CDF does not start at 0');
  assert.ok(Math.abs(D.aux[D.ch] - 1) < 1e-6, `the marginal CDF ends at ${D.aux[D.ch]}, not 1`);
  for (let y = 0; y < D.ch; y += 1) {
    const b = D.condOff + y * (D.cw + 1);
    assert.ok(Math.abs(D.aux[b]) < 1e-6, `row ${y}'s CDF does not start at 0`);
    assert.ok(Math.abs(D.aux[b + D.cw] - 1) < 1e-6, `row ${y}'s CDF ends at ${D.aux[b + D.cw]}`);
    for (let x = 1; x <= D.cw; x += 1) assert.ok(D.aux[b + x] >= D.aux[b + x - 1] - 1e-6, `row ${y} CDF decreases at ${x}`);
  }
  console.log('  CDFs are monotone and span exactly 0 to 1');
}

// THE PDF INTEGRATES TO ONE over the unit square, which is what makes `L / pdf`
// an unbiased estimator at all.
{
  let sum = 0;
  for (let i = 0; i < D.ch * D.cw; i += 1) sum += D.aux[D.pdfOff + i];
  const integral = sum / (D.cw * D.ch);
  assert.ok(Math.abs(integral - 1) < 1e-3, `the pdf integrates to ${integral.toFixed(6)}, not 1`);
  console.log(`  pdf integrates to ${integral.toFixed(6)} over the unit square`);
}

// THE MAPPING ROUND-TRIPS. Direction to (u,v) and back is the shared expression
// the sampler and the pdf both stand on.
{
  let worst = 0;
  for (let i = 0; i < 2000; i += 1) {
    const t = (i / 2000) * Math.PI, ph = ((i * 7919) % 2000) / 2000 * 2 * Math.PI;
    const d = [Math.sin(t) * Math.cos(ph), Math.cos(t), Math.sin(t) * Math.sin(ph)];
    const [u, v] = envDirToUV(d);
    const b = envUVToDir(u, v);
    worst = Math.max(worst, Math.hypot(b[0] - d[0], b[1] - d[1], b[2] - d[2]));
  }
  assert.ok(worst < 1e-6, `direction round trip is off by ${worst}`);
  console.log(`  direction <-> uv round trips to ${worst.toExponential(1)}`);
}

/* ⚠⚠ THE SAMPLER'S PDF AND THE PDF-FROM-DIRECTION ARE ONE EXPRESSION SPLIT IN
   TWO, and a path tracer weights them against each other. Disagreement errors
   nowhere: the image comes out subtly too bright or too dark IN THE SPECULAR
   HIGHLIGHTS ONLY, and stays that way. */
{
  let worst = 0, checked = 0;
  let s = 987654321;
  const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  for (let i = 0; i < 5000; i += 1) {
    const sm = envSample(D, rnd(), rnd());
    const back = envPdfDir(D, sm.dir);
    if (!(sm.pdf > 0)) continue;
    worst = Math.max(worst, Math.abs(back - sm.pdf) / sm.pdf);
    checked += 1;
  }
  assert.ok(checked > 4000, `only ${checked} usable samples`);
  assert.ok(worst < 1e-3, `the sampler's pdf and envPdfDir disagree by ${(worst * 100).toFixed(3)}%`);
  console.log(`  sampler pdf and envPdfDir agree to ${(worst * 100).toExponential(1)}% over ${checked} samples`);
}

/* ⭐ THE WHITE FURNACE. mean(L/pdf) against the same integral summed directly.
   They agree only if the pdf is normalised AND the solid-angle conversion is
   right — and this is the only check in the file that would catch either. */
{
  const f = envFurnaceError(D, img.data, img.width, img.height, 300000);
  assert.ok(f.relativeError < 0.02,
    `furnace: Monte Carlo says ${f.estimate.toFixed(4)} and direct summation says ${f.reference.toFixed(4)} — ${(f.relativeError * 100).toFixed(2)}% apart`);
  console.log(`  furnace: estimate ${f.estimate.toFixed(4)} against reference ${f.reference.toFixed(4)} — ${(f.relativeError * 100).toFixed(2)}% apart`);

  /* THE CONTROL. Scale the pdf and the furnace must notice: without this, a
     furnace that always passed would prove nothing, and "the estimator agrees
     with itself" is exactly the shape of a check that cannot fail. */
  const broken = { ...D, aux: Float32Array.from(D.aux) };
  for (let i = 0; i < broken.ch * broken.cw; i += 1) broken.aux[broken.pdfOff + i] *= 1.35;
  const bf = envFurnaceError(broken, img.data, img.width, img.height, 60000);
  assert.ok(bf.relativeError > 0.15,
    `the control failed: a pdf scaled by 1.35 was still within ${(bf.relativeError * 100).toFixed(2)}% — the furnace cannot fail`);
  console.log(`  control: a pdf scaled by 1.35 reads ${(bf.relativeError * 100).toFixed(1)}% off, so the furnace can fail`);
}

/* SAMPLING FOLLOWS THE LIGHT. The point of the distribution is that it aims at
   the bright parts; a uniform sampler would pass every check above. */
{
  let s = 24680;
  const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  let hit = 0;
  const N = 20000;
  // The brightest tenth of the image by luminance.
  const lums = [];
  for (let y = 0; y < img.height; y += 1) for (let x = 0; x < img.width; x += 1) {
    const o = (y * img.width + x) * 3;
    lums.push(0.2126 * img.data[o] + 0.7152 * img.data[o + 1] + 0.0722 * img.data[o + 2]);
  }
  const cut = lums.slice().sort((a, b) => b - a)[Math.floor(lums.length * 0.1)];
  for (let i = 0; i < N; i += 1) {
    const sm = envSample(D, rnd(), rnd());
    const x = Math.min(img.width - 1, Math.floor(sm.u * img.width));
    const y = Math.min(img.height - 1, Math.floor(sm.v * img.height));
    if (lums[y * img.width + x] >= cut) hit += 1;
  }
  const frac = hit / N;
  assert.ok(frac > 0.2, `only ${(frac * 100).toFixed(1)}% of samples land in the brightest tenth — that is barely better than uniform`);
  console.log(`  ${(frac * 100).toFixed(1)}% of samples land in the brightest 10% of the image (uniform would be ~10%)`);
}

// REFUSALS: a file that is not an HDR must be named, not decoded into noise.
assert.equal(decodeRadianceHDR(new Uint8Array([1, 2, 3, 4])).ok, false);
assert.ok(/RADIANCE/.test(decodeRadianceHDR(new Uint8Array([1, 2, 3, 4])).why));
console.log('  a non-HDR file is refused by name');
console.log('envmap: ok');
