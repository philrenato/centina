import test from 'node:test';
import assert from 'node:assert/strict';
import { makeEllipsoidProfile, makeCircle, revolve } from '../kernel/primitives.mjs';
import { surfacePointAndPartials, isFiniteNet } from '../kernel/surface.mjs';
import { refitSurfaceUV } from '../kernel/loft.mjs';

// REBUILDING A CLOSED SURFACE MUST NOT CREASE ITS SEAM.
//
// Reported as a NURBS sphere at degree 3 showing a hard crease down the seam
// where the same edit in Rhino stays smooth. The cause was not closure itself —
// a natively revolved sphere is smooth across its seam — but Rebuild:
// refitSurfaceUV interpolated the sample grid with a CLAMPED knot vector in
// each direction, so the first and last rows were independent of one another.
// On a closed surface the sample at fraction 1 is the same point as the one at
// 0, so closure came back as a coincidence of position with nothing at all
// holding the tangent across it — C0, which IS the crease.
//
// Every assertion here is stated against an INTERIOR CONTROL measured on the
// same surface with the same method. A bare threshold on the seam angle would
// pass a build whose surfaces had gone globally wrong, and would need re-tuning
// for every degree and station count; "the seam is no more discontinuous than
// an ordinary interior parameter" is the property actually wanted, and it is
// scale- and parameterisation-free.

const unit = (v) => { const L = Math.hypot(...v); return L > 0 ? v.map((c) => c / L) : v; };
const angleDeg = (a, b) => {
  const d = Math.max(-1, Math.min(1, unit(a)[0] * unit(b)[0] + unit(a)[1] * unit(b)[1] + unit(a)[2] * unit(b)[2]));
  return (Math.acos(d) * 180) / Math.PI;
};
const same = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-9;

function closure(srf) {
  const net = srf.ctrlNet;
  return {
    u: net[0].every((p, j) => same(p, net[net.length - 1][j])),
    v: net.every((row) => same(row[0], row[row.length - 1])),
  };
}
// The tangent turn across the seam, measured along the direction that closes —
// the other partial is continuous either way and would hide the kink.
function seamTurn(srf, dir, other, eps = 1e-6) {
  const K = dir === 'u' ? srf.knotsU : srf.knotsV;
  const at = (t) => (dir === 'u' ? surfacePointAndPartials(srf, t, other) : surfacePointAndPartials(srf, other, t));
  const a = at(K[0] + eps), b = at(K[K.length - 1] - eps);
  return angleDeg(dir === 'u' ? a.su : a.sv, dir === 'u' ? b.su : b.sv);
}
function interiorTurn(srf, dir, mid, other, eps = 1e-6) {
  const at = (t) => (dir === 'u' ? surfacePointAndPartials(srf, t, other) : surfacePointAndPartials(srf, other, t));
  const a = at(mid - eps), b = at(mid + eps);
  return angleDeg(dir === 'u' ? a.su : a.sv, dir === 'u' ? b.su : b.sv);
}
// A sphere by revolution: closed in one direction, genuinely curved in both,
// and rational — not a cylinder or a flat patch that could pass by accident.
const sphere = () => revolve(makeEllipsoidProfile([0, 0, 0], [1, 0, 0], [0, 0, 1], 25, 25), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);

test('the source sphere really is closed in exactly one direction — otherwise every seam assertion below is measuring an open edge', () => {
  const c = closure(sphere());
  assert.equal(c.u !== c.v, true, `expected exactly one closed direction, got u=${c.u} v=${c.v}`);
});

test('rebuilding a closed surface keeps it closed', () => {
  const src = sphere();
  const before = closure(src);
  const out = refitSurfaceUV(src, 16, 16, 3, 3);
  assert.equal(isFiniteNet(out.ctrlNet), true);
  const after = closure(out);
  assert.equal(after.u, before.u, 'u-direction closure changed across the rebuild');
  assert.equal(after.v, before.v, 'v-direction closure changed across the rebuild');
});

test('a rebuilt closed surface is no more discontinuous at its seam than at an ordinary interior parameter', () => {
  const src = sphere();
  const dir = closure(src).u ? 'u' : 'v';
  const out = refitSurfaceUV(src, 16, 16, 3, 3);
  const openK = dir === 'u' ? out.knotsV : out.knotsU;
  const closedK = dir === 'u' ? out.knotsU : out.knotsV;
  const lo = openK[0], hi = openK[openK.length - 1];
  const control = interiorTurn(out, dir, (closedK[0] + closedK[closedK.length - 1]) / 2, lo + (hi - lo) * 0.5);
  for (const f of [0.3, 0.5, 0.7]) {
    const turn = seamTurn(out, dir, lo + (hi - lo) * f);
    // Generous against the control rather than absolute: the point is that the
    // seam is ordinary, not that it hits a particular number. Before the fix
    // this read 4.91 degrees against a control of 0.0011 — four orders out, so
    // no plausible slack here lets the old behavior through.
    assert.ok(turn < Math.max(control * 20, 0.02),
      `seam turns ${turn.toFixed(4)} deg at ${f} along, against an interior control of ${control.toFixed(4)} deg`);
  }
});

test('the rebuild still reproduces the source surface at the seam itself, not merely smoothly across it', () => {
  const src = sphere();
  const dir = closure(src).u ? 'u' : 'v';
  const out = refitSurfaceUV(src, 16, 16, 3, 3);
  // Smoothing a seam by drifting the surface away from its source would satisfy
  // the continuity test above while quietly changing the shape.
  const sK = dir === 'u' ? src.knotsU : src.knotsV;
  const oK = dir === 'u' ? out.knotsU : out.knotsV;
  const sOther = dir === 'u' ? src.knotsV : src.knotsU;
  const oOther = dir === 'u' ? out.knotsV : out.knotsU;
  for (const f of [0.25, 0.5, 0.75]) {
    const so = sOther[0] + (sOther[sOther.length - 1] - sOther[0]) * f;
    const oo = oOther[0] + (oOther[oOther.length - 1] - oOther[0]) * f;
    const a = dir === 'u' ? surfacePointAndPartials(src, sK[0], so) : surfacePointAndPartials(src, so, sK[0]);
    const b = dir === 'u' ? surfacePointAndPartials(out, oK[0], oo) : surfacePointAndPartials(out, oo, oK[0]);
    const drift = Math.hypot(a.point[0] - b.point[0], a.point[1] - b.point[1], a.point[2] - b.point[2]);
    assert.ok(drift < 0.05, `seam point drifted ${drift.toFixed(4)} from the source surface at ${f} along (radius 25)`);
  }
});

// The closed direction can play either role, and both must work — a sphere
// closes in the REVOLVE direction with an open profile; a closed circle on a
// partial revolve closes in the PROFILE direction with an open revolve. Testing
// only the first would leave half the mechanism unexercised.
const closedProfile = () => revolve(makeCircle([30, 0, 0], [1, 0, 0], [0, 0, 1], 8, 4), [0, 0, 0], [0, 0, 1], 0, Math.PI * 0.8);

test('a closed PROFILE with an open revolve rebuilds smooth too — the closed direction is not always the revolve one', () => {
  const src = closedProfile();
  const c = closure(src);
  assert.equal(c.u !== c.v, true, `fixture must close in exactly one direction, got u=${c.u} v=${c.v}`);
  const out = refitSurfaceUV(src, 20, 18, 3, 3);
  const dir = closure(out).u ? 'u' : 'v';
  assert.equal(closure(out).u || closure(out).v, true, 'the rebuild dropped closure entirely');
  const openK = dir === 'u' ? out.knotsV : out.knotsU;
  const closedK = dir === 'u' ? out.knotsU : out.knotsV;
  const lo = openK[0], hi = openK[openK.length - 1];
  const control = interiorTurn(out, dir, (closedK[0] + closedK[closedK.length - 1]) / 2, lo + (hi - lo) * 0.5);
  const turn = seamTurn(out, dir, lo + (hi - lo) * 0.5);
  assert.ok(turn < Math.max(control * 20, 0.02), `seam turns ${turn.toFixed(4)} deg against an interior control of ${control.toFixed(4)}`);
});

// THE PARAMETERISATION MUST STAY HEALTHY, not merely the geometry. A seam can
// be made smooth on paper while the surface's speed collapses to zero there,
// which reads as degenerate to normals and tessellation and makes the tangent
// DIRECTION — what the continuity tests above compare — numerically meaningless.
// This is the check that caught the doubly-closed regression: the tangent
// magnitude fell from 218 to 0.013 at one seam while every angle still looked
// fine, and a 1e-4 nudge then swung that direction by 24 degrees.
function tangentMagnitudes(srf, dir) {
  const K = dir === 'u' ? srf.knotsU : srf.knotsV;
  const oK = dir === 'u' ? srf.knotsV : srf.knotsU;
  const o = oK[0] + (oK[oK.length - 1] - oK[0]) * 0.5;
  const at = (t) => (dir === 'u' ? surfacePointAndPartials(srf, t, o) : surfacePointAndPartials(srf, o, t));
  const g = (x) => Math.hypot(...(dir === 'u' ? x.su : x.sv));
  const eps = 1e-6;
  return { start: g(at(K[0] + eps)), end: g(at(K[K.length - 1] - eps)), interior: g(at((K[0] + K[K.length - 1]) / 2)) };
}

for (const [label, make, count] of [['sphere (revolve direction closed)', sphere, 16], ['closed profile, partial revolve', closedProfile, 18]]) {
  test(`${label}: the rebuilt seam is not a stalled parameterisation — tangent magnitude stays comparable to the interior`, () => {
    const out = refitSurfaceUV(make(), 16, count, 3, 3);
    const dir = closure(out).u ? 'u' : 'v';
    const m = tangentMagnitudes(out, dir);
    for (const end of ['start', 'end']) {
      assert.ok(m[end] > m.interior * 0.25,
        `|tangent| at the seam ${end} is ${m[end].toExponential(3)} against an interior ${m.interior.toExponential(3)} — the surface stalls there`);
    }
  });
}

// A TORUS IS CLOSED IN BOTH DIRECTIONS, and it is the only fixture here whose
// two counts also DIFFER — which is what makes it the one case able to catch a
// sub-range cut that misses its own knot. It did: the v cut wanted
// 0.800000000000000044 while the knot present was 0.800000000000000155, so the
// insertion left a span 1.1e-16 wide and the two control points spanning it
// collapsed. Geometry stayed right, parameterisation stalled. The u direction
// escaped only because its arithmetic happened to land bit-exact, which is
// precisely why a symmetric single-closed fixture could never have found it.
test('a surface closed in BOTH directions rebuilds smooth in both, with no stalled parameterisation', () => {
  const torus = revolve(makeCircle([30, 0, 0], [1, 0, 0], [0, 0, 1], 8, 4), [0, 0, 0], [0, 0, 1], 0, Math.PI * 2);
  const c0 = closure(torus);
  assert.equal(c0.u && c0.v, true, 'fixture must be closed in both directions or this test proves nothing');
  const out = refitSurfaceUV(torus, 34, 18, 3, 3);
  assert.equal(isFiniteNet(out.ctrlNet), true);
  const c = closure(out);
  assert.equal(c.u && c.v, true, 'a doubly-closed rebuild must stay closed in both directions');
  for (const dir of ['u', 'v']) {
    const K = dir === 'u' ? out.knotsU : out.knotsV;
    const oK = dir === 'u' ? out.knotsV : out.knotsU;
    const other = oK[0] + (oK[oK.length - 1] - oK[0]) * 0.5;
    const control = interiorTurn(out, dir, (K[0] + K[K.length - 1]) / 2, other);
    const turn = seamTurn(out, dir, other);
    assert.ok(turn < Math.max(control * 20, 0.02),
      `${dir} seam turns ${turn.toFixed(4)} deg against an interior control of ${control.toFixed(4)} deg`);
    // The magnitude check is the one that caught the sliver: the ANGLE looked
    // fine while the surface's speed had fallen to 0.013 against an interior
    // 218, which makes the angle itself meaningless.
    const m = tangentMagnitudes(out, dir);
    for (const end of ['start', 'end']) {
      assert.ok(m[end] > m.interior * 0.25,
        `${dir} |tangent| at the seam ${end} is ${m[end].toExponential(3)} against interior ${m.interior.toExponential(3)} — the surface stalls there`);
    }
  }
});

test('an OPEN surface is left on the ordinary path — the closed handling must not change what already worked', () => {
  const open = revolve(makeEllipsoidProfile([0, 0, 0], [1, 0, 0], [0, 0, 1], 25, 25), [0, 0, 0], [0, 0, 1], 0, Math.PI * 0.8);
  const c = closure(open);
  assert.equal(c.u || c.v, false, 'fixture is not open in both directions, so this test proves nothing');
  const out = refitSurfaceUV(open, 8, 7, 3, 3);
  assert.equal(isFiniteNet(out.ctrlNet), true);
  assert.equal(out.ctrlNet.length, 8, 'an open rebuild must still return exactly the requested station count');
  assert.equal(out.ctrlNet[0].length, 7);
});
