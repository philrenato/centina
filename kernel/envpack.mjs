// THE ENVIRONMENT, PACKED THE WAY THE TRACE SHADER READS IT.
//
// Two GPU resources come out of one equirectangular HDR: an `rgba16float`
// texture the shader samples for radiance, and ONE flat `array<f32>` — `envAux`
// — holding the importance-sampling distribution as three tables laid end to
// end. `envmap.mjs` builds the distribution; this module lays it out for the
// binding and reports the offsets the shader is told about.
//
// ⚠⚠ THE OFFSETS ARE COMPUTED ONCE AND HANDED TO THE SHADER. `condOff` and
// `pdfOff` travel in the frame uniform; the shader does NOT re-derive them from
// `cdfW`/`cdfH`. Re-deriving them on the shader side is the obvious
// simplification and it is exactly how the two sides drift apart — and the
// drift is silent. A wrong `condOff` makes the conditional binary search walk
// numbers that are not that row's CDF; the search still terminates, because a
// bisection always does, and directions still come out. The result is a
// plausible image lit from slightly the wrong place that converges at the
// normal rate. Nothing in the picture says anything is wrong.
//
// So `envAuxLayout` below is the single place the layout exists. The buffer is
// filled at the offsets it returns and the uniform is given the offsets it
// returns, and there is no second expression for either.
//
// ⚠ THE PDF IS NORMALIZED TO THE UNIT SQUARE, in `buildEnvDistribution`, by the
// `cw*ch/total` factor. Losing it makes every render off by ONE CONSTANT
// FACTOR — which reads as the exposure being wrong, gets compensated for in the
// exposure, and after that nothing is ever right again. A white furnace walked
// through these packed tables is the only check that catches it.
//
// ⚠⚠ AND A WHITE FURNACE CANNOT CATCH A WRONG `condOff`. It is worth knowing
// which instrument sees which failure, because the obvious one sees only half.
// With `condOff` off by a row, a sample in row y draws its column from row
// y+1's conditional, picking up a factor P(x|y+1)/P(x|y). The pdf is read at
// the SAME cell and is proportional to that row's weighted luminance, so the
// row's own normalization cancels and the estimator stays EXACTLY unbiased:
// the total energy is right and only the directions are wrong. That is the
// whole reason the failure reads as "lit from slightly the wrong place" rather
// than as an exposure error. Catching it takes a check on WHERE the samples
// land — the sampled cell histogram against the pdf table the walk reports —
// and it needs a coarse, high-contrast environment to have the resolution to
// see it. The two guards are complementary, not redundant: the furnace owns
// `pdfOff` and the normalization, the histogram owns `condOff`.
//
// ⚠ `envRot` IS NOT PACKED HERE. It lives in the frame uniform, in TURNS: the
// shader subtracts it directly from a `[0,1]` texture coordinate, so degrees or
// radians rotate the dome dozens of times. The tables themselves are
// rotation-independent — the sampler draws `(fu, fv)` in texture space and
// rotates on the way out — so re-packing on a rotation would be wasted work.
// `envRotTurns` converts, and is the whole of this module's involvement.

import { buildEnvDistribution } from './envmap.mjs';

/** The largest finite value `f16` can hold. */
export const HALF_FLOAT_MAX = 65504;

/**
 * The layout of `envAux`, and the ONLY expression of it.
 *
 * ```
 * [ marginal CDF   ch+1 floats ] at 0
 * [ conditional    ch*(cw+1)   ] at condOff, row y at condOff + y*(cw+1)
 * [ per-texel pdf  ch*cw       ] at pdfOff
 * ```
 *
 * ⚠ THE MARGINAL'S OFFSET IS STRUCTURALLY ZERO and is never communicated: the
 * shader indexes `envAux[mid]` bare while bisecting for the row. It is returned
 * as a field anyway so the fill loop reads it from here like the other two,
 * rather than from a literal that could disagree with the shader.
 */
export function envAuxLayout(cw, ch) {
  const margLen = ch + 1;
  const condStride = cw + 1;
  const condLen = ch * condStride;
  const pdfLen = ch * cw;
  const condOff = margLen;
  const pdfOff = condOff + condLen;
  const length = pdfOff + pdfLen;
  return {
    margOff: 0, margLen,
    condOff, condStride, condLen,
    pdfOff, pdfStride: cw, pdfLen,
    length, byteLength: length * 4,
  };
}

/**
 * A `buildEnvDistribution` result laid out as the storage buffer, with the
 * offsets the frame uniform must carry alongside it.
 *
 * Returns `{ aux, byteLength, cdfW, cdfH, margOff, condOff, condStride,
 * pdfOff, pdfStride }`. The four `*Off`/`*Stride` fields are `u32` values for
 * the uniform — `cdfW`, `cdfH`, `condOff` and `pdfOff` are the four the shader
 * actually binds; `condStride` and `pdfStride` are the shader's `cdfW+1` and
 * `cdfW`, returned so a CPU-side walk of the same buffer takes them from here
 * instead of restating them.
 *
 * The three regions are COPIED region by region, read at the source's own
 * offsets and written at this layout's. A source laid out differently still
 * lands correctly, and — the point — the written offsets and the reported
 * offsets cannot be two different numbers.
 */
export function packEnvAux(D) {
  const cw = D.cw, ch = D.ch;
  if (!(cw >= 2 && ch >= 2)) throw new Error(`an environment distribution needs at least 2x2 cells, got ${cw}x${ch}`);
  const L = envAuxLayout(cw, ch);
  /* The shader bisects the marginal starting at index 0 with no offset in the
     uniform, so this is not a convention that can be changed here alone. */
  if (L.margOff !== 0) throw new Error('the marginal CDF must start at index 0 — the shader has no offset for it');

  const src = D.aux;
  const srcCond = D.condOff, srcPdf = D.pdfOff;
  if (!(srcCond >= 0) || !(srcPdf >= srcCond)) throw new Error('the source distribution does not carry usable condOff/pdfOff');
  if (src.length < srcPdf + L.pdfLen) throw new Error(`the source aux is ${src.length} floats, too short for ${cw}x${ch}`);

  const aux = new Float32Array(L.length);
  aux.set(src.subarray(0, L.margLen), L.margOff);
  for (let y = 0; y < ch; y += 1) {
    aux.set(src.subarray(srcCond + y * (cw + 1), srcCond + (y + 1) * (cw + 1)), L.condOff + y * L.condStride);
  }
  aux.set(src.subarray(srcPdf, srcPdf + L.pdfLen), L.pdfOff);

  return {
    aux,
    byteLength: aux.byteLength,
    cdfW: cw, cdfH: ch,
    margOff: L.margOff,
    condOff: L.condOff, condStride: L.condStride,
    pdfOff: L.pdfOff, pdfStride: L.pdfStride,
  };
}

const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

/**
 * One non-negative float as `f16` bits, round-to-nearest-even.
 *
 * ⚠ ANYTHING ABOVE 65504 IS CLAMPED, NOT ROUNDED TO INFINITY. A sun in an HDR
 * reaches five and six figures, and an `Inf` texel does not stay in its texel:
 * the sampler filters linearly, so `Inf` bleeds into its neighbors, and
 * `Inf * 0` or `Inf - Inf` anywhere downstream is `NaN`. The accumulation
 * buffer only ever adds, so one NaN sample makes that pixel NaN for the whole
 * accumulation — a dead pixel that survives every further sample and clears
 * only on a full reset. Clamping loses the top of the sun's dynamic range,
 * which is invisible; the alternative is not.
 *
 * NaN and negatives become 0. Radiance is non-negative by construction and a
 * negative one would be an emitter with negative energy.
 */
export function halfFromFloat(v) {
  const c = v > HALF_FLOAT_MAX ? HALF_FLOAT_MAX : (v > 0 ? v : 0);
  _f32[0] = c;
  const x = _u32[0];
  const e = ((x >>> 23) & 0xff) - 127 + 15;
  const man = x & 0x7fffff;
  if (e <= 0) {
    // Subnormal or below: 6.1e-5 down to 6e-8 keeps some precision, under that
    // it is zero. Radiance this small is black either way.
    if (e < -10) return 0;
    const m = man | 0x800000;
    const shift = 14 - e;
    let h = m >>> shift;
    const rem = m & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rem > half || (rem === half && (h & 1))) h += 1;
    return h;
  }
  let h = (e << 10) | (man >>> 13);
  const rem = man & 0x1fff;
  // A carry out of the mantissa lands in the exponent, which is the correct
  // result of rounding up across a binade — not an overflow to guard against.
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h += 1;
  return h;
}

/**
 * The equirectangular image as `rgba16float` texel data, at full source
 * resolution, with everything `queue.writeTexture` needs.
 *
 * Returns `{ data, width, height, format, bytesPerRow, rowsPerImage,
 * byteLength, clamped }`. `clamped` counts the components that hit
 * `HALF_FLOAT_MAX` — a large count on a normal HDR means the source is scaled
 * far outside the range `f16` can carry, not that it has a bright sun.
 *
 * ⚠ ALPHA IS FORCED TO 1.0 (`0x3c00`), not left at zero. The shader reads
 * `.rgb` and never looks at alpha, so a zero there costs nothing today — but a
 * zero-alpha float texture is the kind of thing a later premultiply or a
 * different sampler quietly multiplies the sky away by.
 *
 * ⚠ `bytesPerRow` HAS NO 256-BYTE ALIGNMENT HERE. That requirement belongs to
 * `copyBufferToTexture`; `queue.writeTexture` from a typed array takes the
 * tight `width * 8`. Padding it to 256 would shift every row.
 *
 * The caller still owns the size check: an 8192-wide source is exactly the
 * default `maxTextureDimension2D`, and anything wider needs the adapter's real
 * limit or a downscale.
 */
export function packEnvTexture(rgb, width, height) {
  if (!(width > 0 && height > 0)) throw new Error(`an environment texture needs a positive size, got ${width}x${height}`);
  if (rgb.length < width * height * 3) throw new Error(`the image is ${rgb.length} floats, short of ${width * height * 3}`);
  const n = width * height;
  const data = new Uint16Array(n * 4);
  let clamped = 0;
  for (let i = 0; i < n; i += 1) {
    const s = i * 3, d = i * 4;
    if (rgb[s] > HALF_FLOAT_MAX) clamped += 1;
    if (rgb[s + 1] > HALF_FLOAT_MAX) clamped += 1;
    if (rgb[s + 2] > HALF_FLOAT_MAX) clamped += 1;
    data[d] = halfFromFloat(rgb[s]);
    data[d + 1] = halfFromFloat(rgb[s + 1]);
    data[d + 2] = halfFromFloat(rgb[s + 2]);
    data[d + 3] = 0x3c00;
  }
  return {
    data, width, height,
    format: 'rgba16float',
    bytesPerRow: width * 8,
    rowsPerImage: height,
    byteLength: data.byteLength,
    clamped,
  };
}

/**
 * A decoded HDR to both GPU resources in one call: the texture at full
 * resolution and `envAux` at the capped working resolution.
 *
 * `img` is a `decodeRadianceHDR` result (`{ width, height, data }`); `opts`
 * passes `maxW`/`maxH` through to `buildEnvDistribution`. Returns the
 * `packEnvAux` result with `tex` and `distribution` added — `distribution` is
 * the CPU-side object, kept so the same environment can be queried from JS
 * without unpacking the buffer again.
 */
export function packEnvForGPU(img, opts = {}) {
  const D = buildEnvDistribution(img.data, img.width, img.height, opts);
  const pack = packEnvAux(D);
  pack.tex = packEnvTexture(img.data, img.width, img.height);
  pack.distribution = D;
  return pack;
}

/**
 * Dome rotation in TURNS, from degrees.
 *
 * ⚠ THE SHADER SUBTRACTS THIS FROM A `[0,1]` TEXTURE COORDINATE. Feeding it
 * degrees rotates the dome ninety times for a quarter turn; feeding it radians
 * rotates it π times. This belongs in the frame uniform, not in the packed
 * tables — the distribution is built in texture space and the sampler applies
 * the rotation on the way out to a world direction, so changing the rotation
 * costs a uniform write and no repack.
 */
export function envRotTurns(degrees) {
  return degrees / 360;
}

const PI = Math.PI;
const clamp01 = (t) => (t < 0 ? 0 : (t > 1 ? 1 : t));

/**
 * The shader's `dirToUV`, walked on the CPU: dome pitch, then equirectangular
 * longitude minus the rotation, then latitude with `v = 0` straight UP.
 *
 * `opts.rot` is in turns and `opts.height` is the dome pitch, both as they sit
 * in the frame uniform.
 */
export function envDirToUVPacked(dir, opts = {}) {
  const rot = opts.rot || 0, h = opts.height || 0;
  let x = dir[0], y = dir[1] + h, z = dir[2];
  const L = Math.hypot(x, y, z) || 1;
  x /= L; y /= L; z /= L;
  const uu = (Math.atan2(z, x) + PI) / (2 * PI) - rot;
  return [uu - Math.floor(uu), Math.acos(y < -1 ? -1 : (y > 1 ? 1 : y)) / PI];
}

/**
 * Draw a direction by walking the PACKED buffer exactly as the shader walks it:
 * bisect the marginal for a row, bisect that row's conditional for a column,
 * interpolate inside the found cell, and convert to a world direction.
 *
 * Returns `{ dir, u, v, pdf, row, col }` with `pdf` in solid angle, because
 * `dω = 2π · π · sinθ · du dv`.
 *
 * ⚠ THIS READS `pack.condOff` AND `pack.pdfOff` — the numbers that go into the
 * uniform — and NOT the layout it could recompute from `cdfW`/`cdfH`. That is
 * the whole value of it as a check: if the buffer were ever filled at offsets
 * other than the ones reported, this walk reads the wrong table and a furnace
 * driven through it goes red. Recomputing the offsets here would make it agree
 * with the packer by construction and catch nothing.
 *
 * It is a transcription, not a second design. Where `envmap.mjs`'s `envSample`
 * and this one differ in an edge case — a zero-width CDF cell resolves to the
 * cell's midpoint there and to its far edge here — this one follows the shader.
 */
export function envSamplePacked(pack, r1, r2, opts = {}) {
  const { aux, cdfW, cdfH, condOff, condStride, pdfOff, pdfStride } = pack;
  const rot = opts.rot || 0, h = opts.height || 0;

  let lo = 0, hi = cdfH;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (aux[mid] <= r1) lo = mid; else hi = mid; }
  const y = lo;
  const my0 = aux[y], my1 = aux[y + 1];
  const fv = (y + clamp01((r1 - my0) / Math.max(my1 - my0, 1e-9))) / cdfH;

  const co = condOff + y * condStride;
  lo = 0; hi = cdfW;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (aux[co + mid] <= r2) lo = mid; else hi = mid; }
  const x = lo;
  const cx0 = aux[co + x], cx1 = aux[co + x + 1];
  const fu = (x + clamp01((r2 - cx0) / Math.max(cx1 - cx0, 1e-9))) / cdfW;

  const theta = fv * PI, phi = (fu + rot) * 2 * PI - PI, st = Math.sin(theta);
  let dx = st * Math.cos(phi), dy = Math.cos(theta), dz = st * Math.sin(phi);
  if (Math.abs(h) > 1e-4) {
    /* The inverse of the pitch `dirToUV` applies: solve for the scale s that
       puts `s*td - (0,h,0)` back on the unit sphere. Without it the sampler
       and the texture lookup disagree about which direction a texel is, by
       exactly the pitch. */
    const s = h * dy + Math.sqrt(Math.max(1 - h * h * (1 - dy * dy), 0));
    dx = dx * s; dy = dy * s - h; dz = dz * s;
    const L = Math.hypot(dx, dy, dz) || 1;
    dx /= L; dy /= L; dz /= L;
  }

  const p = aux[pdfOff + y * pdfStride + x];
  return { dir: [dx, dy, dz], u: fu, v: fv, pdf: p / (2 * PI * PI * Math.max(st, 1e-4)), row: y, col: x };
}

/**
 * The solid-angle pdf of a direction the sampler did not produce, read from the
 * packed table — the shader's `envPdfDir`.
 *
 * ⚠⚠ THIS AND `envSamplePacked` ARE ONE EXPRESSION SPLIT IN TWO, and the
 * tracer weights them against each other on every path that reaches the
 * environment without being aimed at it. Disagreement errors nowhere: the image
 * comes out subtly too bright or too dim IN THE SPECULAR HIGHLIGHTS ONLY, and
 * stays that way. A change to either mapping has to move both.
 */
export function envPdfDirPacked(pack, dir, opts = {}) {
  const { aux, cdfW, cdfH, pdfOff, pdfStride } = pack;
  const uv = envDirToUVPacked(dir, opts);
  const x = Math.min(cdfW - 1, Math.max(0, Math.floor(uv[0] * cdfW)));
  const y = Math.min(cdfH - 1, Math.max(0, Math.floor(uv[1] * cdfH)));
  return aux[pdfOff + y * pdfStride + x] / (2 * PI * PI * Math.max(Math.sin(uv[1] * PI), 1e-4));
}
