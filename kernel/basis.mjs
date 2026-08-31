// B-spline basis functions — Piegl & Tiller, "The NURBS Book", 2nd ed.
// Algorithms A2.1 (FindSpan), A2.2 (BasisFuns), A2.3 (DersBasisFuns).
// n = index of last control point (n+1 control points total, 0-indexed).
// p = degree. U = knot vector, length n+p+2.
//
// ⚠ THESE THREE FUNCTIONS ARE THE FLOOR OF EVERY SURFACE EVALUATION IN THE
// PROJECT, so their allocation behavior is a whole-app performance fact rather
// than a local detail. Written the way the book prints them — `Array.from`,
// `new Array(p+1).fill(0)` — A2.3 allocates NINE arrays per call at degree 3,
// and a single fillet rebuild calls it often enough that it measured 36% of the
// total on a CPU profile, garbage collection on top. The arithmetic below is
// unchanged from the book; only the storage is. Scratch is module-level and
// reused, which is safe because none of these recurses or yields, and the
// buffers never escape: every public function still returns freshly allocated
// arrays, so no caller can be handed a buffer that will change underneath it.
// The `*Into` variants exist for hot loops that own their own output.

// Scratch, grown on demand. Indices are computed against `scratchP`, so a call
// at a higher degree than any before it re-sizes once and then never again.
let scratchP = -1;
let sLeft, sRight, sNdu, sA;
function ensureScratch(p) {
  if (p <= scratchP) return;
  scratchP = p;
  sLeft = new Float64Array(p + 1);
  sRight = new Float64Array(p + 1);
  sNdu = new Float64Array((p + 1) * (p + 1));
  sA = new Float64Array(2 * (p + 1));
}

// A2.1 — knot span containing u, via binary search.
export function findSpan(n, p, u, U) {
  if (u >= U[n + 1]) return n;
  if (u <= U[p]) return p;
  let low = p, high = n + 1;
  let mid = Math.floor((low + high) / 2);
  while (u < U[mid] || u >= U[mid + 1]) {
    if (u < U[mid]) high = mid; else low = mid;
    mid = Math.floor((low + high) / 2);
  }
  return mid;
}

// A2.2 — the p+1 nonzero basis functions at u, given its span i = findSpan(...).
// `out` must have room for p+1 entries; it is returned.
export function basisFunsInto(i, u, p, U, out) {
  ensureScratch(p);
  const left = sLeft, right = sRight;
  out[0] = 1.0;
  for (let j = 1; j <= p; j++) {
    left[j] = u - U[i + 1 - j];
    right[j] = U[i + j] - u;
    let saved = 0.0;
    for (let r = 0; r < j; r++) {
      const temp = out[r] / (right[r + 1] + left[j - r]);
      out[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    out[j] = saved;
  }
  return out;
}

export function basisFuns(i, u, p, U) {
  return basisFunsInto(i, u, p, U, new Array(p + 1).fill(0));
}

// A2.3 — basis functions AND their derivatives up to order `n` (n<=p), at
// span i. Returns ders[k][j] = d^k/du^k of the j-th nonzero basis function.
// `out` must be an array of n+1 rows, each with room for p+1 entries.
export function dersBasisFunsInto(i, u, p, n, U, out) {
  ensureScratch(p);
  const left = sLeft, right = sRight, ndu = sNdu, a = sA;
  const w = p + 1;                       // row stride of the ndu matrix
  ndu[0] = 1.0;
  for (let j = 1; j <= p; j++) {
    left[j] = u - U[i + 1 - j];
    right[j] = U[i + j] - u;
    let saved = 0.0;
    for (let r = 0; r < j; r++) {
      ndu[j * w + r] = right[r + 1] + left[j - r];
      const temp = ndu[r * w + (j - 1)] / ndu[j * w + r];
      ndu[r * w + j] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    ndu[j * w + j] = saved;
  }

  const row0 = out[0];
  for (let j = 0; j <= p; j++) row0[j] = ndu[j * w + p];

  for (let r = 0; r <= p; r++) {
    let s1 = 0, s2 = w;                  // the two alternating rows of `a`
    a[0] = 1.0;
    for (let k = 1; k <= n; k++) {
      let d = 0.0;
      const rk = r - k, pk = p - k;
      if (r >= k) {
        a[s2] = a[s1] / ndu[(pk + 1) * w + rk];
        d = a[s2] * ndu[rk * w + pk];
      }
      const j1 = rk >= -1 ? 1 : -rk;
      const j2 = (r - 1 <= pk) ? k - 1 : p - r;
      for (let j = j1; j <= j2; j++) {
        a[s2 + j] = (a[s1 + j] - a[s1 + j - 1]) / ndu[(pk + 1) * w + (rk + j)];
        d += a[s2 + j] * ndu[(rk + j) * w + pk];
      }
      if (r <= pk) {
        a[s2 + k] = -a[s1 + k - 1] / ndu[(pk + 1) * w + r];
        d += a[s2 + k] * ndu[r * w + pk];
      }
      out[k][r] = d;
      const tmp = s1; s1 = s2; s2 = tmp;
    }
  }

  let f = p;
  for (let k = 1; k <= n; k++) {
    const row = out[k];
    for (let j = 0; j <= p; j++) row[j] *= f;
    f *= (p - k);
  }
  return out;
}

export function dersBasisFuns(i, u, p, n, U) {
  const out = new Array(n + 1);
  for (let k = 0; k <= n; k++) out[k] = new Array(p + 1).fill(0);
  return dersBasisFunsInto(i, u, p, n, U, out);
}
