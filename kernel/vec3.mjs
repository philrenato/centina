// Minimal 3-vector helpers used throughout the kernel. Plain arrays [x,y,z],
// not objects — keeps the hot paths (basis eval, frame construction) free of
// allocation overhead from property lookups.

export function add(a, b) { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
export function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
export function scale(a, s) { return [a[0]*s, a[1]*s, a[2]*s]; }
export function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
export function cross(a, b) {
  return [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ];
}
export function length(a) { return Math.sqrt(dot(a, a)); }
export function normalize(a) {
  const l = length(a);
  if (l < 1e-12) throw new Error('normalize: zero-length vector');
  return [a[0]/l, a[1]/l, a[2]/l];
}

// Any unit vector perpendicular to `t` (t assumed unit length). Used to seed
// the first frame of a parallel-transport sweep, where no prior normal exists.
export function anyPerpendicular(t) {
  const ref = Math.abs(t[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  return normalize(cross(t, ref));
}
