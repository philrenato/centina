// Surface fair — the second stage of the surface modifier
// chain (Rebuild -> Fair -> Point Edits). A Laplacian control-net
// relaxation, not an energy-functional fair (which minimizes a
// curvature-based cost under G0/G1/G2 constraints). It shrinks the net
// slightly toward the local average, so unlike Rebuild it does not
// preserve shape exactly.
//
// One "Smoothness" knob (`amount`, 0..1) is the whole public surface.
// `fairParamsFromAmount` is the one place that maps it to the two
// internal Laplacian parameters (iteration count, per-step blend weight),
// so tuning that mapping never touches call sites.

// amount=0 maps to 0 iterations (a byte-identical passthrough, matching
// Rebuild's `srfRebuild:null`-is-inert convention); amount=1 maps to 20
// iterations at a fixed 0.5 per-step blend weight, which smooths a
// control net monotonically without per-surface retuning.
const FAIR_MAX_ITERATIONS = 20;
const FAIR_LAMBDA = 0.5;
export function fairParamsFromAmount(amount) {
  const a = Math.max(0, Math.min(1, amount));
  return { iterations: Math.round(a * FAIR_MAX_ITERATIONS), lambda: FAIR_LAMBDA };
}

// fairControlNet(srf, amount) — relaxes every interior control point
// (never the U=0/U=max/V=0/V=max boundary row/column, which stay pinned
// so the surface's edges and corners never drift) toward the
// arithmetic average of its 4 grid neighbors, blended by `lambda`
// per iteration, repeated `iterations` times. Operates on Cartesian
// position only (indices 0/1/2); a control point's rational weight
// (index 3) is never touched, since weight expresses pull, not a
// spatial position to smooth.
export function fairControlNet(srf, amount) {
  const { iterations, lambda } = fairParamsFromAmount(amount);
  const nu = srf.ctrlNet.length, nv = srf.ctrlNet[0].length;
  if (iterations <= 0 || nu < 3 || nv < 3) return srf; // no interior point exists at nu/nv<3
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
