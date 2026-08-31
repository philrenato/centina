// AN ENVIRONMENT MAP, AND THE DISTRIBUTION THAT LETS A RENDERER AIM AT IT.
//
// Two things live here. Decoding a RADIANCE `.hdr` file into linear RGB, and
// building the piecewise-constant 2D distribution that turns "where is the light
// in this image" into a sampler — the construction in Pharr, Jakob & Humphreys,
// *Physically Based Rendering*, chapter 13.
//
// ⚠⚠ THE SAMPLER AND THE PDF-FROM-DIRECTION MUST AGREE EXACTLY. A path tracer
// draws a direction from the sampler and, on any path that arrives at the
// environment some other way, asks the pdf what the chances of that direction
// were — and weights the two estimates against each other. If the two
// expressions disagree, nothing errors: the image comes out subtly too bright or
// too dark, in the specular highlights only, and stays that way. They are
// derived from one mapping here for that reason.
//
// ⚠ AND AN UNNORMALIZED PDF IS THE SAME KIND OF FAILURE. Estimating an integral
// as the mean of `L / pdf` is only correct when the pdf integrates to one over
// its domain; a pdf off by a constant makes every render off by that constant,
// which reads as an exposure error and gets compensated for in the
// exposure, permanently. `envFurnaceError` below is the check that catches it,
// and it needs no renderer.

const REC709 = [0.2126, 0.7152, 0.0722];

/**
 * Decode a RADIANCE RGBE (`.hdr`) image into linear float RGB.
 * Returns `{ width, height, data }` with `data` as `width*height*3` floats.
 */
export function decodeRadianceHDR(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let p = 0;
  const line = () => {
    let s = '';
    while (p < u8.length && u8[p] !== 10) { s += String.fromCharCode(u8[p]); p += 1; }
    p += 1;
    return s;
  };
  const magic = line();
  if (!/^#\?(RADIANCE|RGBE)/.test(magic)) return { ok: false, why: 'that is not a RADIANCE .hdr file' };
  let fmt = null;
  for (;;) {
    const l = line();
    if (l === '') break;
    if (/^FORMAT=/.test(l)) fmt = l.slice(7).trim();
    if (p >= u8.length) return { ok: false, why: 'the header never ended' };
  }
  if (fmt && fmt !== '32-bit_rle_rgbe') return { ok: false, why: `unsupported RADIANCE format "${fmt}"` };
  const dim = line().trim();
  const m = /^-Y\s+(\d+)\s+\+X\s+(\d+)$/.exec(dim);
  // Only the standard orientation is handled, and the others are refused by name
  // rather than decoded into a silently flipped sky.
  if (!m) return { ok: false, why: `unsupported scanline order "${dim}" — only -Y +X is handled` };
  const height = +m[1], width = +m[2];
  const data = new Float32Array(width * height * 3);
  const row = new Uint8Array(width * 4);

  const emit = (y) => {
    for (let x = 0; x < width; x += 1) {
      const e = row[x * 4 + 3];
      /* ⚠ AN EXPONENT OF ZERO MEANS BLACK, not 2^-128. RGBE stores a shared
         exponent biased by 128, and the zero case is the encoding's own way of
         writing zero — reading it as a power produces a floor of 1e-38 across
         the whole image, which is invisible until something divides by it. */
      const f = e ? Math.pow(2, e - 136) : 0;
      const o = (y * width + x) * 3;
      data[o] = row[x * 4] * f;
      data[o + 1] = row[x * 4 + 1] * f;
      data[o + 2] = row[x * 4 + 2] * f;
    }
  };

  for (let y = 0; y < height; y += 1) {
    if (p + 4 > u8.length) return { ok: false, why: `the file ends after ${y} of ${height} scanlines` };
    const a = u8[p], b = u8[p + 1], c = u8[p + 2], d = u8[p + 3];
    if (a === 2 && b === 2 && ((c << 8) | d) === width && width >= 8 && width < 32768) {
      // Adaptive RLE: four separate component planes, each run-length coded.
      p += 4;
      for (let comp = 0; comp < 4; comp += 1) {
        let x = 0;
        while (x < width) {
          if (p >= u8.length) return { ok: false, why: 'the file ends inside a run-length scanline' };
          let n = u8[p]; p += 1;
          if (n > 128) {
            n -= 128;
            const v = u8[p]; p += 1;
            if (x + n > width) return { ok: false, why: 'a run overruns its scanline' };
            for (let k = 0; k < n; k += 1) { row[(x + k) * 4 + comp] = v; }
            x += n;
          } else {
            if (n === 0) return { ok: false, why: 'a run length of zero would not advance' };
            if (x + n > width) return { ok: false, why: 'a literal run overruns its scanline' };
            for (let k = 0; k < n; k += 1) { row[(x + k) * 4 + comp] = u8[p]; p += 1; }
            x += n;
          }
        }
      }
    } else {
      // Flat RGBE, four bytes per pixel.
      if (p + width * 4 > u8.length) return { ok: false, why: 'the file ends inside a flat scanline' };
      for (let x = 0; x < width * 4; x += 1) row[x] = u8[p + x];
      p += width * 4;
    }
    emit(y);
  }
  return { ok: true, width, height, data };
}

/**
 * The piecewise-constant 2D distribution over an equirectangular image, packed
 * as one flat array of three tables laid end to end — a marginal CDF over rows,
 * one conditional CDF per row, and a per-texel pdf.
 *
 * ⚠ THE ROWS ARE WEIGHTED BY `sin(theta)` BEFORE ANYTHING ELSE. An
 * equirectangular image gives every row the same number of texels, and the rows
 * near the poles cover almost no solid angle — sampling it without that weight
 * aims a renderer at the poles, where there is nothing, and the picture is dim
 * and noisy for a reason nothing in it explains.
 */
export function buildEnvDistribution(rgb, width, height, opts = {}) {
  const maxW = opts.maxW || 1024, maxH = opts.maxH || 512;
  const cw = Math.max(2, Math.min(width, maxW));
  const ch = Math.max(2, Math.min(height, maxH));
  const condOff = ch + 1;
  const pdfOff = condOff + ch * (cw + 1);
  const aux = new Float32Array(pdfOff + ch * cw);

  // Box-average the source into the working resolution rather than point-
  // sampling it: a small bright light dropped by decimation is a light the
  // renderer never aims at, and nothing downstream can recover it.
  const lumAt = (cx, cy) => {
    const x0 = Math.floor((cx * width) / cw), x1 = Math.max(x0 + 1, Math.floor(((cx + 1) * width) / cw));
    const y0 = Math.floor((cy * height) / ch), y1 = Math.max(y0 + 1, Math.floor(((cy + 1) * height) / ch));
    let acc = 0, n = 0;
    for (let y = y0; y < y1; y += 1) for (let x = x0; x < x1; x += 1) {
      const o = (y * width + x) * 3;
      acc += REC709[0] * rgb[o] + REC709[1] * rgb[o + 1] + REC709[2] * rgb[o + 2];
      n += 1;
    }
    return n ? acc / n : 0;
  };

  let total = 0;
  for (let y = 0; y < ch; y += 1) {
    const sinT = Math.sin((Math.PI * (y + 0.5)) / ch);
    const base = condOff + y * (cw + 1);
    let rs = 0;
    for (let x = 0; x < cw; x += 1) {
      /* THE EPSILON KEEPS EVERY TEXEL REACHABLE. A pdf of exactly zero anywhere
         makes that direction impossible to sample AND makes the weight on any
         path that reaches it by another route a division by zero. */
      const L = lumAt(x, y) * sinT + 1e-7;
      rs += L;
      aux[base + x + 1] = rs;
      aux[pdfOff + y * cw + x] = L;
    }
    if (rs > 0) for (let x = 1; x <= cw; x += 1) aux[base + x] /= rs;
    aux[base] = 0;
    aux[base + cw] = 1;
    total += rs;
    aux[y + 1] = total;
  }
  if (total > 0) for (let y = 1; y <= ch; y += 1) aux[y] /= total;
  aux[0] = 0;
  aux[ch] = 1;
  // p(u,v) on the unit square: integrates to 1 over [0,1]^2.
  if (total > 0) { const k = (cw * ch) / total; for (let i = 0; i < ch * cw; i += 1) aux[pdfOff + i] *= k; }
  return { aux, cw, ch, condOff, pdfOff, total };
}

// The first index whose CDF value exceeds `t`, over `[lo, lo+n]`.
function cdfFind(aux, lo, n, t) {
  let a = 0, b = n;
  while (a + 1 < b) {
    const mid = (a + b) >> 1;
    if (aux[lo + mid] <= t) a = mid; else b = mid;
  }
  return a;
}

/** `(u,v)` on the unit square is a direction, with v = 0 straight up. */
export function envUVToDir(u, v) {
  const phi = u * 2 * Math.PI - Math.PI;
  const theta = v * Math.PI;
  const st = Math.sin(theta);
  return [st * Math.cos(phi), Math.cos(theta), st * Math.sin(phi)];
}
export function envDirToUV(d) {
  const L = Math.hypot(d[0], d[1], d[2]) || 1;
  const x = d[0] / L, y = d[1] / L, z = d[2] / L;
  let u = (Math.atan2(z, x) + Math.PI) / (2 * Math.PI);
  u -= Math.floor(u);
  const v = Math.acos(Math.max(-1, Math.min(1, y))) / Math.PI;
  return [u, v];
}

/** Draw a direction from the distribution. Returns `{ dir, u, v, pdf }`. */
export function envSample(D, u1, u2) {
  const { aux, cw, ch, condOff, pdfOff } = D;
  const y = cdfFind(aux, 0, ch, u1);
  const dy = aux[y + 1] - aux[y];
  const fv = (y + (dy > 0 ? (u1 - aux[y]) / dy : 0.5)) / ch;
  const base = condOff + y * (cw + 1);
  const x = cdfFind(aux, base, cw, u2);
  const dx = aux[base + x + 1] - aux[base + x];
  const fu = (x + (dx > 0 ? (u2 - aux[base + x]) / dx : 0.5)) / cw;
  const p = aux[pdfOff + y * cw + x];
  const sinT = Math.max(Math.sin(fv * Math.PI), 1e-4);
  return { dir: envUVToDir(fu, fv), u: fu, v: fv, pdf: p / (2 * Math.PI * Math.PI * sinT) };
}

/**
 * The solid-angle pdf of a direction the sampler did not produce.
 * ⚠ THIS AND `envSample` ARE ONE EXPRESSION SPLIT IN TWO, and a renderer weights
 * them against each other — so a change to the mapping has to move both, and
 * changing only one is an image subtly wrong in its highlights alone.
 */
export function envPdfDir(D, dir) {
  const { aux, cw, ch, pdfOff } = D;
  const [u, v] = envDirToUV(dir);
  const x = Math.min(cw - 1, Math.max(0, Math.floor(u * cw)));
  const y = Math.min(ch - 1, Math.max(0, Math.floor(v * ch)));
  const sinT = Math.max(Math.sin(v * Math.PI), 1e-4);
  return aux[pdfOff + y * cw + x] / (2 * Math.PI * Math.PI * sinT);
}

/**
 * ⭐ THE WHITE FURNACE, AND IT NEEDS NO RENDERER.
 *
 * A Monte Carlo estimate of the environment's total irradiance, `mean(L/pdf)`,
 * against the same integral summed directly over the texels. They agree only if
 * the pdf is correctly normalized AND the solid-angle conversion is right.
 *
 * This is the single check that catches an unnormalized pdf. Left uncaught it is
 * not visible as an error anywhere: every render is off by one constant factor,
 * which reads as the exposure being wrong and gets compensated for in the
 * exposure — after which the renderer is permanently, invisibly mis-weighted.
 * Returns the relative error.
 */
export function envFurnaceError(D, rgb, width, height, samples = 200000, seed = 12345) {
  const { cw, ch } = D;
  // The reference: sum L * dOmega directly over the working grid.
  let ref = 0;
  for (let y = 0; y < ch; y += 1) {
    const theta = (Math.PI * (y + 0.5)) / ch;
    const dOmega = (2 * Math.PI / cw) * (Math.PI / ch) * Math.sin(theta);
    for (let x = 0; x < cw; x += 1) {
      const sx = Math.min(width - 1, Math.floor(((x + 0.5) * width) / cw));
      const sy = Math.min(height - 1, Math.floor(((y + 0.5) * height) / ch));
      const o = (sy * width + sx) * 3;
      ref += (REC709[0] * rgb[o] + REC709[1] * rgb[o + 1] + REC709[2] * rgb[o + 2]) * dOmega;
    }
  }
  let s = seed >>> 0;
  const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  let est = 0;
  for (let i = 0; i < samples; i += 1) {
    const sm = envSample(D, rnd(), rnd());
    if (!(sm.pdf > 0)) continue;
    const sx = Math.min(width - 1, Math.floor(sm.u * width));
    const sy = Math.min(height - 1, Math.floor(sm.v * height));
    const o = (sy * width + sx) * 3;
    const L = REC709[0] * rgb[o] + REC709[1] * rgb[o + 1] + REC709[2] * rgb[o + 2];
    est += L / sm.pdf;
  }
  est /= samples;
  return { reference: ref, estimate: est, relativeError: ref > 0 ? Math.abs(est - ref) / ref : Infinity };
}
