// Eco: the ghost of the form before a dropped edit, and the words on it.
//
//   ecoShouldCapture(kind, handleMode)        does this drag kind change size or shape
//   ecoBoundsOf(parts)                        world bounds of copied positions
//   ecoPositionsChanged(before, after, tol)   did any copied position move
//   ecoRatio(before, after)                   the bounds axis that changed most
//   ecoFormatRatio(r)                         "1.2", "0.75", "3"
//   ecoHeldText(hold)                         "130 ml held", or null
//   ecoLabel(before, after, held)             "1.2x wide", "130 ml held · 1.2x tall"
//
// A part is { pos } with pos a flat xyz array (Float32Array or number[]).
// Bounds are { min: [x, y, z], max: [x, y, z] }, or null when nothing finite
// was measured. Axes are the world's, Z up: X is wide, Y is deep, Z is tall.

export const ECO_AXIS_WORDS = ['wide', 'deep', 'tall'];

// Kinds that move the whole object rigidly leave no ghost; in handle mode the
// same kinds move control points, which reshapes. A profile extrude makes a new
// object and leaves its profile as it was.
const ECO_RIGID_KINDS = new Set(['move', 'movePlane', 'rotate']);
const ECO_SHAPE_KINDS = new Set(['scale', 'scaleUniform', 'extrudeFace', 'extrudeEdge', 'point', 'cage']);

export function ecoShouldCapture(kind, handleMode) {
  if (kind === 'extrude') return false;
  if (handleMode) return ECO_RIGID_KINDS.has(kind) || ECO_SHAPE_KINDS.has(kind);
  return ECO_SHAPE_KINDS.has(kind);
}

export function ecoBoundsOf(parts) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  let any = false;
  for (const part of parts || []) {
    const p = part && part.pos;
    if (!p) continue;
    for (let i = 0; i + 2 < p.length; i += 3) {
      const x = p[i], y = p[i + 1], z = p[i + 2];
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      any = true;
      if (x < min[0]) min[0] = x; if (x > max[0]) max[0] = x;
      if (y < min[1]) min[1] = y; if (y > max[1]) max[1] = y;
      if (z < min[2]) min[2] = z; if (z > max[2]) max[2] = z;
    }
  }
  return any ? { min, max } : null;
}

// A different part count or vertex count is a change of topology (a face
// extrude), which is a change of shape whatever the positions say.
export function ecoPositionsChanged(before, after, tol = 1e-9) {
  const a = before || [], b = after || [];
  if (a.length !== b.length) return true;
  for (let k = 0; k < a.length; k += 1) {
    const p = a[k] && a[k].pos, q = b[k] && b[k].pos;
    if (!p || !q || p.length !== q.length) return true;
    for (let i = 0; i < p.length; i += 1) if (!(Math.abs(p[i] - q[i]) <= tol)) return true;
  }
  return false;
}

// The axis whose span changed by the largest factor, either way: a halving and
// a doubling are the same size of change. An axis with no span before (a flat
// curve's Z) has no ratio and is skipped. { axis: null, ratio: null } when no
// axis is measurable.
export function ecoRatio(before, after) {
  if (!before || !after) return { axis: null, ratio: null };
  const s0 = [0, 1, 2].map((i) => before.max[i] - before.min[i]);
  const s1 = [0, 1, 2].map((i) => after.max[i] - after.min[i]);
  const scale = Math.max(...s0, ...s1);
  if (!(scale > 0)) return { axis: null, ratio: null };
  const eps = scale * 1e-9;
  let axis = null, ratio = null, best = -1;
  for (let i = 0; i < 3; i += 1) {
    if (!(s0[i] > eps) || !(s1[i] > eps)) continue;
    const r = s1[i] / s0[i];
    const d = Math.abs(Math.log(r));
    if (d > best + 1e-12) { best = d; axis = i; ratio = r; }
  }
  return { axis, ratio };
}

export function ecoFormatRatio(r) {
  if (!Number.isFinite(r)) return null;
  return String(+r.toFixed(2));
}

export function ecoHeldText(hold) {
  const t = hold && hold.target;
  if (!t || !Number.isFinite(t.value) || typeof t.unit !== 'string' || !t.unit) return null;
  return `${+t.value.toPrecision(6)} ${t.unit} held`;
}

// A drop whose largest change rounds to 1x moved points without changing the
// bounds; it is named for what happened rather than as "1x".
export function ecoLabel(before, after, held = null) {
  const { axis, ratio } = ecoRatio(before, after);
  const f = ratio == null ? null : ecoFormatRatio(ratio);
  const size = axis == null || f == null || f === '1' ? 'reshaped' : `${f}x ${ECO_AXIS_WORDS[axis]}`;
  return held ? `${held} · ${size}` : size;
}
