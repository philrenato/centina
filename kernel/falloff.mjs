// FALLOFF RAMP — a falloff curve as DATA rather than one hardcoded
// expression, evaluatable at any t in [0,1].
//
// Everything in this kernel that fades one influence out over a distance
// currently hardcodes `smoothstep` (kernel/cage.mjs's own zero-derivative-
// at-both-ends curve). That is a good default and stays the default here —
// what it is not is EDITABLE. A ramp expressed as data can be drawn,
// stored on a recipe, flowed down a wire, and shared between a deformer's
// own falloff and a future field's, without each consumer re-deriving its
// own curve.
//
// SHAPE: { interp: 'smooth' | 'linear', stops: [[t, v], ...] } — stops
// sorted ascending by t, each t and v in [0,1]. A segment interpolates
// between its two stops; 'smooth' runs the segment-local parameter through
// `smoothstep` first, so EVERY stop has zero derivative on both sides and a
// multi-stop smooth ramp is C1 throughout — the same property cage.mjs's
// own comment already gives as the reason a hard cutoff is unacceptable.
// Outside the first/last stop the ramp holds that stop's value flat.
//
// THE DEFAULT IS THE OLD EXPRESSION, EXACTLY. `null` (or a malformed ramp)
// evaluates as `1 - smoothstep(t)` — the literal expression Charybdis used
// before this module existed. The explicitly-written equivalent,
// DEFAULT_FALLOFF_RAMP, is proven BIT-IDENTICAL to it in this module's own
// tests rather than assumed: the segment form `v0 + (v1-v0)*smoothstep(u)`
// with v0=1, v1=0, t0=0, t1=1 reduces in IEEE-754 to `1 + (-1)*s`, i.e.
// exactly `1 - s`, and the segment-local parameter `(t-0)/(1-0)` is exactly
// `t`. Neither step introduces a rounding difference.

import { smoothstep } from './cage.mjs';

// The written-out equivalent of the hardcoded falloff every consumer used
// before this module: full strength at t=0, nothing at t=1, smoothstep in
// between. Frozen so a consumer cannot mutate the shared default in place.
export const DEFAULT_FALLOFF_RAMP = Object.freeze({
  interp: 'smooth',
  stops: Object.freeze([Object.freeze([0, 1]), Object.freeze([1, 0])]),
});

function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

// True only for a ramp this module can actually evaluate as written —
// anything else falls back to the default rather than throwing, because
// every caller here is a per-point geometry evaluation on a live drag and
// an honest fallback is worth more than a thrown error mid-gesture.
// A caller that wants a real refusal validates BEFORE committing, via
// falloffRampRefusalReason below.
export function isFalloffRamp(ramp) {
  return !!ramp && Array.isArray(ramp.stops) && ramp.stops.length >= 2;
}

// The commit-time gate: returns a plain-language reason string, or null if
// the ramp is genuinely well-formed. Separate from isFalloffRamp on
// purpose — the evaluator wants a cheap "can I use this", a commit path
// wants to be able to say WHY it refused.
export function falloffRampRefusalReason(ramp) {
  if (ramp == null) return null; // the default, always valid
  if (typeof ramp !== 'object') return 'a falloff ramp must be an object';
  if (ramp.interp !== 'smooth' && ramp.interp !== 'linear') return 'a falloff ramp\'s interp must be "smooth" or "linear"';
  if (!Array.isArray(ramp.stops)) return 'a falloff ramp needs a stops array';
  if (ramp.stops.length < 2) return 'a falloff ramp needs at least 2 stops';
  let prevT = -Infinity;
  for (let i = 0; i < ramp.stops.length; i++) {
    const s = ramp.stops[i];
    if (!Array.isArray(s) || s.length < 2) return `falloff stop ${i + 1} must be a [t, value] pair`;
    const [t, v] = s;
    if (!Number.isFinite(t) || !Number.isFinite(v)) return `falloff stop ${i + 1} has a non-finite number`;
    if (t < 0 || t > 1) return `falloff stop ${i + 1}'s position must be between 0 and 1`;
    if (v < 0 || v > 1) return `falloff stop ${i + 1}'s value must be between 0 and 1`;
    if (t < prevT) return 'falloff stops must be sorted by position';
    prevT = t;
  }
  return null;
}

// Evaluate a ramp at t. `t` is clamped into [0,1] first, so a caller can
// hand over a raw normalized distance without pre-clamping — which is what
// lets a consumer route EVERY case (inside, outside, and between) through
// one call instead of keeping its own hardcoded end branches.
export function evalFalloffRamp(ramp, t) {
  const c = clamp01(t);
  // The default path is the ORIGINAL expression itself, not a
  // reconstruction of it — so `null` can never drift from what shipped
  // before this module existed, whatever happens to the general path.
  if (!isFalloffRamp(ramp)) return 1 - smoothstep(c);

  const stops = ramp.stops;
  const last = stops.length - 1;
  if (c <= stops[0][0]) return stops[0][1];
  if (c >= stops[last][0]) return stops[last][1];

  let i = 0;
  while (i < last - 1 && c > stops[i + 1][0]) i++;
  const [t0, v0] = stops[i];
  const [t1, v1] = stops[i + 1];
  const span = t1 - t0;
  // Coincident stops are a legal way to author a hard step; the segment
  // between them has no interior, so jump straight to the later value.
  if (!(span > 0)) return v1;
  const local = (c - t0) / span;
  const eased = ramp.interp === 'linear' ? local : smoothstep(local);
  return v0 + (v1 - v0) * eased;
}
