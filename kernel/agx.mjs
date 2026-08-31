// THE AgX DISPLAY TRANSFORM (Troy Sobotka), AS THE PRESENT SHADER APPLIES IT.
//
// A path tracer produces open-domain radiance and a display takes [0,1]. AgX
// gets from one to the other by rotating into a slightly desaturated working
// basis, compressing in log2 across a fixed exposure window, running a sigmoid
// in that space, and rotating back. The rotation is what keeps a bright
// saturated colour from clipping to a flat primary — the desaturation happens
// before the curve rather than after it.
//
// This is the CPU twin of the shader's `agx()`: same matrices, same window,
// same polynomial fit (after B. Wrensch). It exists so the transform can be
// checked without a GPU, and so anything that needs display-referred pixels in
// JS — a thumbnail, a colour swatch, a regression fixture — gets the identical
// answer rather than a second, drifting implementation.
//
// ⚠⚠⚠ THE MATRICES ARE COLUMNS, AND ONE OF THEM WAS ONCE WRITTEN AS ROWS.
// `Minv` was transposed, so `M * Minv` was not the identity: a neutral grey came
// back as (1.091, 0.956, 0.953) — red up 9%, green and blue down about 4.5%.
// That is a warm-pink cast on every pixel the renderer had ever produced, and
// because the error is MULTIPLICATIVE, brightening a surface made it worse
// rather than whiter, which is why a white backdrop could not be made to look
// white and why the search went to lighting for a long time. A transpose of a
// near-identity matrix breaks no invariant a reader would notice: the numbers
// look right, the rows sum to about one either way, and the output stays
// plausible. Only pushing a neutral through and demanding a neutral back can
// see it.
//
// ⚠ SO THESE NUMBERS ARE COPIED, NOT TYPED. They come from the shader source
// verbatim, in the shader's own column order and to the shader's own digit
// count. Retyping them, reflowing them into rows "for readability", or
// shortening them is how the transpose happened the first time.

/* Column-major, exactly as WGSL's mat3x3f(c0, c1, c2) stores it: entry
   [c*3 + r] is row r of column c, and `M * v` is c0*v.x + c1*v.y + c2*v.z. */
export const AGX_M = Object.freeze([
  0.842479062253094, 0.0423282422610123, 0.0423756549057051,   // column 0
  0.0784335999999992, 0.878468636469772, 0.0784336,   // column 1
  0.0792237451477643, 0.0791661274605434, 0.879142973793104,   // column 2
]);

export const AGX_MINV = Object.freeze([
  1.196879005120174, -0.052896851757456, -0.052971635514444,   // column 0
  -0.098020881140137, 1.151903129904173, -0.098043450117124,   // column 1
  -0.099029744079720, -0.098961176844843, 1.151073672641161,   // column 2
]);

/* The log2 window AgX compresses across: about 12.5 stops below middle grey to
   about 4 above. These bound the sigmoid's domain, so changing either one
   changes the look of every image, not just the extremes. */
export const AGX_MIN_EV = -12.47393;
export const AGX_MAX_EV = 4.026069;

/**
 * The AgX sigmoid, as a sixth-order polynomial fit on [0,1].
 *
 * A fit rather than the analytic curve because the shader evaluates this per
 * pixel per channel; the coefficients are the published minimal fit and are not
 * separately meaningful — changing one does not adjust one property of the
 * curve, it makes a different curve.
 */
export function agxContrast(x) {
  const x2 = x * x, x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

const clamp01 = (v) => (v < 0 ? 0 : (v > 1 ? 1 : v));

/* One column-major mat3 times one vec3. Written once and used for both
   directions, so the two rotations cannot pick up different conventions. */
function mul3(m, x, y, z, out) {
  out[0] = m[0] * x + m[3] * y + m[6] * z;
  out[1] = m[1] * x + m[4] * y + m[7] * z;
  out[2] = m[2] * x + m[5] * y + m[8] * z;
  return out;
}

/**
 * Linear scene-referred RGB → display-encoded RGB in [0,1].
 *
 * `out` may be any 3-element array-like, including a view into a pixel buffer;
 * omitted, a fresh 3-array comes back.
 *
 * ⚠ THE INPUT IS FLOORED AT 1e-10 BEFORE THE LOG, not clamped afterwards. A
 * channel of exactly zero is a legitimate result of a path tracer and log2(0)
 * is -Infinity, which the window clamp would turn into the correct answer on
 * most hardware and NaN on the rest.
 */
export function agx(rgb, out = [0, 0, 0]) {
  const span = AGX_MAX_EV - AGX_MIN_EV;
  mul3(AGX_M, Math.max(rgb[0], 1e-10), Math.max(rgb[1], 1e-10), Math.max(rgb[2], 1e-10), out);
  for (let i = 0; i < 3; i += 1) {
    let v = Math.log2(out[i]);
    if (v < AGX_MIN_EV) v = AGX_MIN_EV; else if (v > AGX_MAX_EV) v = AGX_MAX_EV;
    out[i] = agxContrast((v - AGX_MIN_EV) / span);
  }
  mul3(AGX_MINV, out[0], out[1], out[2], out);
  out[0] = clamp01(out[0]); out[1] = clamp01(out[1]); out[2] = clamp01(out[2]);
  return out;
}

/**
 * Exposure then AgX, which is the order the present pass uses.
 *
 * Exposure is applied in LINEAR light, before the transform — applying it after
 * would scale display-encoded values and crush the highlight rolloff the
 * transform exists to provide.
 */
export function agxExposed(rgb, exposure = 1, out = [0, 0, 0]) {
  out[0] = rgb[0] * exposure; out[1] = rgb[1] * exposure; out[2] = rgb[2] * exposure;
  return agx(out, out);
}
