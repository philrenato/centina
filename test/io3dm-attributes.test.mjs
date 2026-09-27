// What a .3dm carries besides geometry.
//
// io3dm.mjs is tested on shape elsewhere — a curve, a surface, a trimmed Brep
// and a SubD cage all come back to within tolerance. These tests cover the
// attributes that travel alongside, and a round trip that keeps every control point while
// dropping a layer's visibility, a layer's lock, an object's own display color
// and the file's unit system is not a faithful one: the model opens looking
// right, and every organizing decision made about it is gone.
//
// Each test below pairs the claim with a control assertion
// (the layer's name and color, the point's coordinates), so a failure here is
// read as "this field did not survive", never as "the round trip did not run".
import { test } from 'node:test';
import assert from 'node:assert/strict';
import rhino3dmFactory from 'rhino3dm';
import { exportDocument, importDocument } from '../io3dm.mjs';

const rhino = await rhino3dmFactory();

// A layer that has been used rather than left at its defaults: renamed,
// recolored, hidden and locked. A pristine layer cannot show a dropped flag,
// because its default is the value a dropping writer produces.
const HIDDEN_LOCKED = { id: 7, name: 'Frame', color: { r: 199, g: 79, b: 43 }, parentId: null, visible: false, locked: true };
const DEFAULT_LAYER = { id: 0, name: 'Default', color: { r: 138, g: 141, b: 144 }, parentId: null, visible: true, locked: false };

function fileFrom(payload) {
  const { bytes } = exportDocument(rhino, payload);
  return { bytes, doc: rhino.File3dm.fromByteArray(bytes) };
}

test('a hidden, locked layer is still hidden and locked in the file it was written to', () => {
  const { doc } = fileFrom({
    tolerance: 0.004,
    layers: [DEFAULT_LAYER, HIDDEN_LOCKED],
    objects: [{ kind: 'point', layerId: 7, name: 'P1', point: [12.5, -7.25, 3] }],
  });
  const table = doc.layers();
  const frame = [...Array(table.count).keys()].map((i) => table.get(i)).find((l) => l.name === 'Frame');
  // Control — the layer itself was written, under its own name and color.
  assert.ok(frame, 'the Frame layer reached the file at all');
  assert.deepEqual(
    { r: frame.color.r, g: frame.color.g, b: frame.color.b },
    { r: 199, g: 79, b: 43 },
    'Control: the layer color survives, so this test exercised the layer writer',
  );
  assert.equal(frame.visible, false, 'a hidden layer must not be written visible');
  assert.equal(frame.locked, true, 'a locked layer must not be written unlocked');
});

test('reading a .3dm reports each layer’s visibility and lock, rather than leaving the caller to assume both', () => {
  // Authored with rhino3dm directly, so this is a real Rhino-shaped file and
  // not simply our own writer's output read back through our own reader.
  const doc = new rhino.File3dm();
  doc.settings().modelAbsoluteTolerance = 0.001;
  const l = new rhino.Layer();
  l.name = 'Frame';
  l.color = { r: 199, g: 79, b: 43, a: 255 };
  l.visible = false;
  l.locked = true;
  doc.layers().add(l);
  const back = importDocument(rhino, doc.toByteArray());

  const frame = back.layers.find((x) => x.name === 'Frame');
  // Control — the layer was read, with the fields the reader already carries.
  assert.ok(frame, 'the Frame layer was read back');
  assert.deepEqual(
    { r: frame.color.r, g: frame.color.g, b: frame.color.b },
    { r: 199, g: 79, b: 43 },
    'Control: the layer color is read, so this test exercised the layer reader',
  );
  assert.equal(frame.visible, false, 'a layer hidden in the file must come back hidden');
  assert.equal(frame.locked, true, 'a layer locked in the file must come back locked');
});

test('an object’s own display color survives the write and comes back on the read', () => {
  const colour = { r: 30, g: 127, b: 75 };
  const { bytes, doc } = fileFrom({
    tolerance: 0.004,
    layers: [DEFAULT_LAYER],
    objects: [{ kind: 'point', layerId: 0, name: 'P1', point: [12.5, -7.25, 3], color: colour }],
  });
  const attrs = doc.objects().get(0).attributes();
  // Control — the object and its name were written.
  assert.equal(attrs.name, 'P1', 'Control: the object name survives, so the attribute writer ran');

  assert.deepEqual(
    { r: attrs.objectColor.r, g: attrs.objectColor.g, b: attrs.objectColor.b },
    colour,
    'an object carrying its own color must not be written with the default black',
  );

  const back = importDocument(rhino, bytes);
  // Control — the object came back.
  assert.deepEqual(back.objects[0].point, [12.5, -7.25, 3], 'Control: the geometry came back, so the reader ran');
  assert.ok(back.objects[0].color, 'the read must report the object color the file carries');
  assert.deepEqual(
    { r: back.objects[0].color.r, g: back.objects[0].color.g, b: back.objects[0].color.b },
    colour,
    'and report it unchanged',
  );
});

test('a file authored in inches is not read as millimeters in silence', () => {
  // 25.4x is the quietest wrong answer available here: every number is
  // plausible, nothing is skipped, and the part is the wrong size.
  const doc = new rhino.File3dm();
  doc.settings().modelUnitSystem = rhino.UnitSystem.Inches;
  doc.settings().modelAbsoluteTolerance = 0.001;
  doc.objects().addPoint([1, 2, 3], new rhino.ObjectAttributes());
  const back = importDocument(rhino, doc.toByteArray());

  // Control — the file was read, and its tolerance came across.
  assert.equal(back.tolerance, 0.001, 'Control: the file was read, so the reader ran');
  assert.deepEqual(back.objects[0].point, [1, 2, 3], 'Control: the point was read');

  assert.ok('units' in back, 'the read must say what unit system the file declares');
  assert.equal(back.units, 'inches', 'and name it, so a millimeter-only caller can convert or refuse rather than silently mis-scale');
});
