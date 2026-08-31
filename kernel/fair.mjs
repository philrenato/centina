// SURFACE FAIR — the second stage of the surface modifier
// chain (Rebuild -> Fair -> Point Edits). A
// real, well-known, honestly-scoped Laplacian CONTROL-NET relaxation —
// NOT Rhino's own energy-functional Fair (which minimizes a real
// curvature-based cost under G0/G1/G2 constraints), and it genuinely
// shrinks the net slightly toward the local average, a real side effect
// students should be able to see, not a claim of exact shape
// preservation the way refitSurfaceUV's own Rebuild can honestly make.
//
// A SINGLE discoverable "Smoothness" knob (`amount`, 0..1) is the whole
// public surface.
// A review finding: two coupled Laplacian parameters (iteration
// count, per-step blend weight) is a bad classroom experience for a
// design student. `fairParamsFromAmount` is the one place that maps the
// single knob to the two real internal numbers, so tuning that mapping
// later never touches call sites.

// amount=0 maps to 0 iterations (a byte-identical passthrough, matching
// Rebuild's own `srfRebuild:null`-is-inert convention exactly); amount=1
// maps to 20 iterations at a fixed, real 0.5 per-step blend weight —
// tuned to visibly, monotonically smooth a real control net without
// needing per-surface retuning.
const FAIR_MAX_ITERATIONS = 20;
const FAIR_LAMBDA = 0.5;
export function fairParamsFromAmount(amount) {
  const a = Math.max(0, Math.min(1, amount));
  return { iterations: Math.round(a * FAIR_MAX_ITERATIONS), lambda: FAIR_LAMBDA };
}

// fairControlNet(srf, amount) — relaxes every INTERIOR control point
// (never the U=0/U=max/V=0/V=max boundary row/column, which stay pinned
// exactly so the surface's own edges/corners never drift) toward the
// plain arithmetic average of its 4 grid neighbors, blended by `lambda`
// per iteration, repeated `iterations` times. Operates on CARTESIAN
// position only (indices 0/1/2) — a control point's own rational WEIGHT
// (index 3) is never touched, since weight expresses "pull," not a
// spatial position to smooth; a rational surface's own shape character
// (where its weights are non-uniform) survives fairing exactly as far as
// weight is concerned, only positions relax.
export function fairControlNet(srf, amount) {
  const { iterations, lambda } = fairParamsFromAmount(amount);
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  if (iterations <= 0 || nu < 3 || nv < 3) return srf; // no genuine interior point exists to relax at nu/nv<3 — an honest no-op, not a crash
  let net = srf.ctrlNet.map((row) => row.map((cp) => [...cp]));
  for (let iter = 0; iter < iterations; iter++) {
    const next = net.map((row) => row.map((cp) => [...cp]));
    for (let i = 1; i < nu - 1; i++) {
      for (let j = 1; j < nv - 1; j++) {
        const c = net[i][j];
        const n1 = net[i - 1][j], n2 = net[i + 1][j], n3 = net[i][j - 1], n4 = net[i][j + 1];
        next[i][j] = [
          c[0] + ((n1[0] + n2[0] + n3[0] + n4[0]) / 4 - c[0]) * lambda,
          c[1] + ((n1[1] + n2[1] + n3[1] + n4[1]) / 4 - c[1]) * lambda,
          c[2] + ((n1[2] + n2[2] + n3[2] + n4[2]) / 4 - c[2]) * lambda,
          c[3],
        ];
      }
    }
    net = next;
  }
  return { ...srf, ctrlNet: net };
}
