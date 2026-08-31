// FIELD — a sampled scalar grid over a parametric domain, and the
// bilinear, wrap-aware sampler that reads it.
//
// WHAT THIS IS. A field is a rectangular grid of scalar values plus a
// rule for reading a continuous value between its nodes. It is the data
// half of "a position-dependent value flows down one wire" —
// lifted out of the app layer so it can have more than one producer and
// more than one consumer, which is exactly the limit of the app-layer form.
//
// WHAT THIS IS NOT, and the line is deliberate: this module says nothing
// about WHERE the grid lives. A painted field's domain is one surface's
// own UV rectangle; an attractor's would be a region of world space.
// Both are the same grid and the same sampler, and neither belongs in
// here. A consumer already knows which domain it is asking about — that
// is precisely the "the consumer declares its own sampling" contract
// Tessellate's density already proves in practice — so binding a domain
// into the field record would take a decision away from the only code
// with enough information to make it.
//
// THE RECORD SHAPE IS FROZEN BY PERSISTENCE, not by taste. A field is
// { uCount, vCount, values } where values is a PLAIN Array indexed
// i * vCount + j, u-major. Plain, not a typed array: a field is written
// straight into the document snapshot and read back through JSON, and a
// Float64Array round-trips through JSON as an object with numeric keys,
// not an array — a silent corruption that would only surface on reload.
//
// WRAP IS AN ARGUMENT, NEVER A STORED FIELD. Whether a direction closes
// is a property of the surface the field is being read against, and that
// surface can change under it (a Revolve's sweep angle is a live param).
// Storing it would be storing a copy that goes stale — the same
// derive-it-fresh rule this app's own seam handling already follows.

import { evalFalloffRamp } from './falloff.mjs';

// A grid node's continuous-index-to-parameter map. For a CLOSED
// direction the N nodes span [min,max) at fraction k/N — the wrap cell
// bridges node N-1 back to node 0, so a node at the seam is not stored
// twice. For an OPEN direction they span [min,max] inclusive at k/(N-1).
// Fractional k is legal and meaningful: a contour traced between nodes
// lands on the same parameter a node would.
export function fieldNodeParam(k, N, wrap, min, max) {
  const frac = wrap ? (k / N) : (N > 1 ? k / (N - 1) : 0);
  return min + frac * (max - min);
}

// Grid resolution for a field over a surface with an nu-by-nv control
// net. Denser than the net so a brush footprint and a traced contour
// both have real resolution to work with, and capped so painting and
// tracing stay responsive on a dense surface.
export const FIELD_MIN_DIM = 16;
export const FIELD_MAX_DIM = 48;
export function fieldDimsForNet(nu, nv) {
  const clamp = (n) => Math.max(FIELD_MIN_DIM, Math.min(FIELD_MAX_DIM, n * 4));
  return { uCount: clamp(nu), vCount: clamp(nv) };
}

export function makeField(uCount, vCount) {
  return { uCount, vCount, values: new Array(uCount * vCount).fill(0) };
}

// Well-formedness, in the same spirit as isFiniteNet: a field that is
// the wrong length or carries a non-finite value is refused HERE rather
// than producing a plausible-looking sample somewhere downstream.
export function isField(field) {
  if (!field || !Array.isArray(field.values)) return false;
  const { uCount, vCount, values } = field;
  if (!Number.isInteger(uCount) || !Number.isInteger(vCount)) return false;
  if (uCount < 2 || vCount < 2) return false;
  if (values.length !== uCount * vCount) return false;
  for (const v of values) if (!Number.isFinite(v)) return false;
  return true;
}

// An independent COPY of a field record. Needed the moment a field stops
// belonging to the one surface that produced it: lifting a grid into its
// own object, snapshotting it for undo, and writing it into a document
// record are all the same operation, and all three must produce data that
// cannot alias what it came from — a shared `values` array would make two
// records that look independent silently move together.
//
// REFUSES A MALFORMED FIELD by returning null rather than copying it.
// Copying is the exact moment the data crosses out of the code that
// produced it, so it is the right place to check: an object built from a
// grid of the wrong length would sample plausible-looking garbage forever
// afterward, with nothing downstream able to tell.
//
// The copy is a PLAIN Array, like the original, for the same persistence
// reason the record shape itself is frozen — see this module's header.
export function cloneField(field) {
  if (!isField(field)) return null;
  return { uCount: field.uCount, vCount: field.vCount, values: [...field.values] };
}

export function fieldPeak(field) {
  if (!field || !field.values) return 0;
  let m = 0;
  for (const v of field.values) if (v > m) m = v;
  return m;
}

// THE WHOLE GRID AS ONE NUMBER — the unweighted mean of every node.
//
// WHY THE MEAN AND NOT THE PEAK, which is the aggregate this module
// already had. Peak is the right normalizer and the wrong summary: every
// producer here tops out at 1 (the brush saturates, and an attractor's
// ramp starts at 1 on the attractor itself), so the peak of any field a
// student has actually made is ~1 no matter how much of the surface is
// marked. The minimum is 0 for the mirror-image reason — one untouched
// node is enough. The mean is the only reduction of the three that moves
// across its whole range as the field changes: marking more area raises
// it, pressing harder raises it, erasing lowers it.
//
// UNWEIGHTED, and that is a real claim about what the number means. Field
// nodes are evenly spaced in the surface's PARAMETER domain, not in area,
// so on a surface whose parameterization is uneven this mean weights
// parameter space rather than square millimetres. It is "the average
// value over the grid", which is a statement about the field, not about
// the geometry the field happens to be read against — the same stance the
// record's own missing domain already takes.
//
// Returns 0 for a malformed or empty field, matching fieldPeak: a caller
// that would divide by it is already guarding on the peak.
export function fieldMean(field) {
  if (!field || !field.values || !field.values.length) return 0;
  let s = 0;
  for (const v of field.values) s += v;
  return s / field.values.length;
}

// The index pair and blend weight a fraction resolves to along one
// direction. Split out because the wrap and open cases genuinely differ
// in BOTH the cell count and which cell a fraction of 1 lands in, and
// getting that wrong is invisible everywhere except at a seam.
function fieldCell(frac, N, wrap) {
  const f = frac < 0 ? 0 : frac > 1 ? 1 : frac;
  if (wrap) {
    const g = f * N;
    const i0 = ((Math.floor(g) % N) + N) % N;
    return [i0, (i0 + 1) % N, g - Math.floor(g)];
  }
  const g = f * (N - 1);
  const i0 = Math.max(0, Math.min(N - 2, Math.floor(g)));
  return [i0, i0 + 1, g - i0];
}

// Bilinear sample at a continuous UV FRACTION (fu, fv in [0,1], clamped
// here rather than by every caller). Returns the field's own value —
// a caller wanting a true normalized 0..1 divides by fieldPeak itself,
// which is the existing contract every current consumer already follows.
export function sampleFieldFraction(field, fu, fv, wrapU, wrapV) {
  const { uCount, vCount, values } = field;
  const [i0, i1, tu] = fieldCell(fu, uCount, wrapU);
  const [j0, j1, tv] = fieldCell(fv, vCount, wrapV);
  const v00 = values[i0 * vCount + j0], v10 = values[i1 * vCount + j0];
  const v01 = values[i0 * vCount + j1], v11 = values[i1 * vCount + j1];
  return (v00 * (1 - tu) + v10 * tu) * (1 - tv) + (v01 * (1 - tu) + v11 * tu) * tv;
}

// A COMPUTED field: every node's value is a falloff ramp evaluated
// against that node's own DISTANCE from something. This answers the
// "the field is PAINTED, never computed" limit without
// this module learning what an attractor is: the caller supplies
// distanceAt(i, j), so the same routine serves a point, an axis, a
// curve, or anything else that can answer "how far".
//
// THE NORMALIZATION IS krCharybdisPoint'S OWN, deliberately: full
// strength at or inside innerRadius, nothing at or outside outerRadius,
// the ramp across the band between. Reusing that convention is what
// lets an attractor and a Charybdis deform share one authored ramp and
// mean the same thing by it.
//
// ALL THREE CASES ROUTE THROUGH ONE RAMP CALL. evalFalloffRamp clamps
// its own argument, so there are no hardcoded weight=1 / weight=0 end
// branches here — which is what makes an EDITED ramp whose endpoints
// are not 1 and 0 actually honoured outside the band, rather than
// silently overridden. That was R2a's own structural improvement; it
// only holds for a consumer that declines to re-add the end branches.
export function fieldFromDistances(uCount, vCount, distanceAt, innerRadius, outerRadius, ramp) {
  const field = makeField(uCount, vCount);
  const span = outerRadius - innerRadius;
  for (let i = 0; i < uCount; i++) {
    for (let j = 0; j < vCount; j++) {
      const d = distanceAt(i, j);
      let t;
      // A degenerate node (a pole, a collapsed row) answers with a
      // non-finite distance. It reads as fully outside rather than
      // poisoning the whole grid with a NaN that isField would then
      // refuse — one bad node must not cost the entire field.
      if (!Number.isFinite(d)) t = 1;
      else if (span > 0) t = (d - innerRadius) / span;
      // A zero or inverted span is a legal way to author a hard edge at
      // innerRadius: there is no interior to interpolate across, so the
      // ramp is read at its own two ends and nowhere between.
      else t = d <= innerRadius ? 0 : 1;
      field.values[i * vCount + j] = evalFalloffRamp(ramp, t);
    }
  }
  return field;
}
