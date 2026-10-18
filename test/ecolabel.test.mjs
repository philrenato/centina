import test from 'node:test';
import assert from 'node:assert';
import { ecoShouldCapture, ecoBoundsOf, ecoPositionsChanged, ecoRatio, ecoFormatRatio, ecoHeldText, ecoLabel, ECO_AXIS_WORDS } from '../kernel/ecolabel.mjs';

// The eight corners of an axis-aligned box, as one part.
function boxPart(min, max) {
  const pos = [];
  for (const x of [min[0], max[0]]) for (const y of [min[1], max[1]]) for (const z of [min[2], max[2]]) pos.push(x, y, z);
  return { pos: new Float32Array(pos) };
}
const B = (min, max) => ecoBoundsOf([boxPart(min, max)]);

test('a rigid drag leaves no ghost; a size or shape drag does', () => {
  for (const k of ['move', 'movePlane', 'rotate', 'extrude']) assert.strictEqual(ecoShouldCapture(k, false), false, k);
  for (const k of ['scale', 'scaleUniform', 'extrudeFace', 'extrudeEdge', 'point', 'cage']) assert.strictEqual(ecoShouldCapture(k, false), true, k);
  // In handle mode a move or rotate moves control points, which reshapes.
  for (const k of ['move', 'movePlane', 'rotate', 'scale']) assert.strictEqual(ecoShouldCapture(k, true), true, k);
  assert.strictEqual(ecoShouldCapture('extrude', true), false);
  assert.strictEqual(ecoShouldCapture('nonsense', false), false);
});

test('bounds of copied positions', () => {
  const b = ecoBoundsOf([boxPart([-1, -2, -3], [4, 5, 6]), { pos: [10, 0, 0] }]);
  assert.deepStrictEqual(b, { min: [-1, -2, -3], max: [10, 5, 6] });
  // Non-finite samples contribute nothing; nothing finite is null, never a box at 0.
  assert.deepStrictEqual(ecoBoundsOf([{ pos: [NaN, 0, 0, 1, 1, 1] }]), { min: [1, 1, 1], max: [1, 1, 1] });
  assert.strictEqual(ecoBoundsOf([]), null);
  assert.strictEqual(ecoBoundsOf([{ pos: [Infinity, 0, 0] }]), null);
  assert.strictEqual(ecoBoundsOf(null), null);
});

test('position change: equal, moved, re-topologized', () => {
  const a = [boxPart([0, 0, 0], [1, 1, 1])];
  assert.strictEqual(ecoPositionsChanged(a, [boxPart([0, 0, 0], [1, 1, 1])]), false);
  assert.strictEqual(ecoPositionsChanged(a, [boxPart([0, 0, 0], [1, 1, 1.001])]), true);
  // A face extrude adds vertices: a change whatever the positions are.
  assert.strictEqual(ecoPositionsChanged(a, [{ pos: new Float32Array([...a[0].pos, 0, 0, 0]) }]), true);
  assert.strictEqual(ecoPositionsChanged(a, [...a, ...a]), true);
  // A NaN after is a change (the comparison can fail).
  const nan = new Float32Array(a[0].pos); nan[0] = NaN;
  assert.strictEqual(ecoPositionsChanged(a, [{ pos: nan }]), true);
  // Within the tolerance is not.
  assert.strictEqual(ecoPositionsChanged(a, [boxPart([0, 0, 0], [1, 1, 1 + 1e-7])], 1e-6), false);
});

test('the largest bounds-axis change, either way', () => {
  const b0 = B([0, 0, 0], [10, 20, 30]);
  assert.deepStrictEqual(ecoRatio(b0, B([0, 0, 0], [12, 20, 30])), { axis: 0, ratio: 1.2 });
  // Halving Y beats growing X by 1.5: |ln 0.5| > |ln 1.5|.
  const r = ecoRatio(b0, B([0, 0, 0], [15, 10, 30]));
  assert.strictEqual(r.axis, 1); assert.strictEqual(r.ratio, 0.5);
  assert.strictEqual(ecoRatio(b0, B([0, 0, 0], [10, 20, 60])).axis, 2);
  // Translated: no axis changed size, the ratio is 1.
  const t = ecoRatio(b0, B([1e5, 0, 0], [1e5 + 10, 20, 30]));
  assert.ok(Math.abs(t.ratio - 1) < 1e-9);
  // A flat curve has no Z span; Z is skipped rather than divided by zero.
  const flat = ecoRatio(B([0, 0, 0], [30, 60, 0]), B([0, 0, 0], [45, 60, 0]));
  assert.strictEqual(flat.axis, 0); assert.strictEqual(flat.ratio, 1.5);
  // A point has no axis at all.
  assert.deepStrictEqual(ecoRatio(B([1, 1, 1], [1, 1, 1]), B([1, 1, 1], [1, 1, 1])), { axis: null, ratio: null });
  assert.deepStrictEqual(ecoRatio(null, b0), { axis: null, ratio: null });
});

test('ratio words', () => {
  assert.strictEqual(ecoFormatRatio(1.2), '1.2');
  assert.strictEqual(ecoFormatRatio(1.25), '1.25');
  assert.strictEqual(ecoFormatRatio(3), '3');
  assert.strictEqual(ecoFormatRatio(0.7512), '0.75');
  assert.strictEqual(ecoFormatRatio(NaN), null);
  assert.deepStrictEqual(ECO_AXIS_WORDS, ['wide', 'deep', 'tall']);
});

test('labels, with and without a Hold', () => {
  const b0 = B([0, 0, 0], [10, 20, 30]);
  assert.strictEqual(ecoLabel(b0, B([0, 0, 0], [12, 20, 30])), '1.2x wide');
  assert.strictEqual(ecoLabel(b0, B([0, 0, 0], [10, 10, 30])), '0.5x deep');
  assert.strictEqual(ecoLabel(b0, B([0, 0, 0], [10, 20, 36])), '1.2x tall');
  // Points moved inside the bounds: named, not "1x".
  assert.strictEqual(ecoLabel(b0, B([0, 0, 0], [10, 20, 30.001])), 'reshaped');
  assert.strictEqual(ecoLabel(null, b0), 'reshaped');
  const held = ecoHeldText({ target: { value: 130, unit: 'ml' } });
  assert.strictEqual(held, '130 ml held');
  assert.strictEqual(ecoLabel(b0, B([0, 0, 0], [10, 20, 36]), held), '130 ml held · 1.2x tall');
  assert.strictEqual(ecoHeldText(null), null);
  assert.strictEqual(ecoHeldText({ target: { value: NaN, unit: 'ml' } }), null);
  assert.strictEqual(ecoHeldText({ target: { value: 6.2, unit: '' } }), null);
  assert.strictEqual(ecoHeldText({ target: { value: 0.1 + 0.2, unit: 'g' } }), '0.3 g held');
  // Every label fits a chip: the longest axis word with a Hold stays short.
  assert.ok(ecoLabel(b0, B([0, 0, 0], [10.25, 20, 30]), ecoHeldText({ target: { value: 1234.56, unit: 'mm3' } })).length <= 40);
});
