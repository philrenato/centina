import test from 'node:test';
import assert from 'node:assert';
import {
  RING_MAX_STATES, ringHash, ringCreate, ringBlobOk, ringDocParts, ringRecord, ringMatch, ringTruncateAfter,
  ringDropState, ringStateDoc, ringCollect, ringBytes, ringStateBytes, ringCap, ringManifest, ringFromManifest,
  ringBlobDelta, ringFileBlockText, ringFromFileBlock, ringPrevKey, ringNextKey, ringKeys, ringLastKey,
} from '../kernel/undoring.mjs';

// A document in the shape the app's snapshot writer produces.
function doc(objects, extra = {}) {
  return {
    v: 1, ts: Date.now(), nextId: objects.length + 1, currentLayerId: 0,
    layers: [{ id: 0, name: 'Default' }],
    views: [{ kind: 'persp', theta: Math.random() }], activeViewIndex: 1, selectedIds: [1],
    documentSettings: { tolerance: 0.001 }, materialOverrides: {},
    objects, skippedAtSave: 0, ...extra,
  };
}
const point = (id, x) => ({ id, name: `Point ${id}`, layerId: 0, leaf: { type: 'Point', point: { x, y: 0, z: 0 } } });
const heavy = (id, n, seed) => ({ id, name: `Mesh ${id}`, layerId: 0, leaf: { type: 'Net', pts: Array.from({ length: n }, (_, i) => [i * 0.1 + seed, Math.sin(i + seed), i % 7]) } });
const strip = (d) => { const { ts, views, activeViewIndex, selectedIds, nextId, ...rest } = d; return rest; };

test('the hash is stable, 14 hex digits, and separates near texts', () => {
  assert.match(ringHash('abc'), /^[0-9a-f]{14}$/);
  assert.strictEqual(ringHash('abc'), ringHash('abc'));
  assert.notStrictEqual(ringHash('abc'), ringHash('abd'));
  assert.notStrictEqual(ringHash(''), ringHash(' '));
  assert.ok(ringBlobOk(ringHash('x'), 'x'));
  assert.ok(!ringBlobOk(ringHash('x'), 'y'));
  assert.ok(!ringBlobOk(ringHash('x'), undefined));
});

test('a state leaves the view keys out and rebuilds the model exactly', () => {
  const r = ringCreate();
  const d = doc([point(1, 0), point(2, 5)]);
  ringRecord(r, 0, d);
  const back = ringStateDoc(r, 0);
  assert.deepStrictEqual(back, strip(d));
  assert.ok(!('views' in back) && !('selectedIds' in back) && !('ts' in back));
  // Fresh objects: changing what came back does not reach the ring.
  back.objects[0].leaf.point.x = 99;
  assert.strictEqual(ringStateDoc(r, 0).objects[0].leaf.point.x, 0);
  assert.deepStrictEqual(ringDocParts({ objects: 'nope' }).records, []);
});

test('an unchanged object is stored once across states', () => {
  const r = ringCreate();
  const big = heavy(1, 2000, 0);
  for (let k = 0; k < 10; k++) ringRecord(r, k, doc([big, point(2, k)]));
  const one = ringStateBytes(r, 0);
  // Ten states cost about one heavy record plus ten small ones.
  assert.ok(ringBytes(r) < one * 1.5, `${ringBytes(r)} vs one state ${one}`);
  assert.strictEqual(new Set(r.states.map((s) => s.o[0])).size, 1);
});

test('re-recording the same content at a key reports no change; different content does', () => {
  const r = ringCreate();
  assert.strictEqual(ringRecord(r, 3, doc([point(1, 0)])).changed, true);
  assert.strictEqual(ringRecord(r, 3, doc([point(1, 0)])).changed, false); // views and ts differ, the model does not
  assert.strictEqual(ringRecord(r, 3, doc([point(1, 1)])).changed, true);
  assert.deepStrictEqual(ringKeys(r), [3]);
  assert.throws(() => ringRecord(r, 1.5, doc([])));
});

test('keys stay sorted, and stepping goes to the nearest key either side', () => {
  const r = ringCreate();
  for (const k of [5, 1, 3, -2]) ringRecord(r, k, doc([point(1, k)]));
  assert.deepStrictEqual(ringKeys(r), [-2, 1, 3, 5]);
  assert.strictEqual(ringPrevKey(r, 3), 1);
  assert.strictEqual(ringPrevKey(r, 2), 1);
  assert.strictEqual(ringPrevKey(r, -2), null);
  assert.strictEqual(ringNextKey(r, 3), 5);
  assert.strictEqual(ringNextKey(r, 5), null);
  assert.strictEqual(ringLastKey(r), 5);
});

test('three edits then three undos give back each earlier state; redo goes forward', () => {
  const r = ringCreate();
  const states = [doc([]), doc([point(1, 0)]), doc([point(1, 0), point(2, 1)]), doc([point(1, 4), point(2, 1)])];
  states.forEach((d, k) => ringRecord(r, k, d));
  // A reload hands the ring to a fresh store and back.
  const m = ringManifest(r, 3);
  const stored = new Map(r.blobs);
  const { ring, cursor, skipped } = ringFromManifest(structuredClone(m), (k) => stored.get(k));
  assert.strictEqual(skipped, 0);
  assert.strictEqual(cursor, 3);
  let at = cursor;
  for (let i = 2; i >= 0; i--) {
    at = ringPrevKey(ring, at);
    assert.deepStrictEqual(ringStateDoc(ring, at), strip(states[i]), `undo to ${i}`);
  }
  assert.strictEqual(ringPrevKey(ring, at), null);
  at = ringNextKey(ring, at);
  assert.deepStrictEqual(ringStateDoc(ring, at), strip(states[1]));
});

test('an edit after an undo truncates the redo side', () => {
  const r = ringCreate();
  for (let k = 0; k < 4; k++) ringRecord(r, k, doc([point(1, k)]));
  // Back at k=1, a new edit lands at k=2: everything above k=1 goes first.
  assert.strictEqual(ringTruncateAfter(r, 1), 2);
  ringRecord(r, 2, doc([point(1, 40)]));
  ringCollect(r);
  assert.deepStrictEqual(ringKeys(r), [0, 1, 2]);
  assert.strictEqual(ringNextKey(r, 2), null);
  assert.strictEqual(ringStateDoc(r, 2).objects[0].leaf.point.x, 40);
  // The dropped states' blobs are gone with them.
  assert.strictEqual(r.blobs.size, 1 + 3);
});

test('the cap drops the oldest first, never the kept state, then the far redo side', () => {
  const r = ringCreate();
  for (let k = 0; k < 60; k++) ringRecord(r, k, doc([point(1, k)]));
  const dropped = ringCap(r, 59, { maxStates: RING_MAX_STATES });
  assert.strictEqual(r.states.length, RING_MAX_STATES);
  assert.deepStrictEqual(dropped, Array.from({ length: 10 }, (_, i) => i));
  // Kept state at the bottom: the redo end goes instead.
  const r2 = ringCreate();
  for (let k = 0; k < 5; k++) ringRecord(r2, k, doc([point(1, k)]));
  ringCap(r2, 0, { maxStates: 3 });
  assert.deepStrictEqual(ringKeys(r2), [0, 1, 2]);
  // A byte cap smaller than one state keeps that state alone.
  const r3 = ringCreate();
  for (let k = 0; k < 5; k++) ringRecord(r3, k, doc([heavy(1, 500, k)]));
  ringCap(r3, 4, { maxBytes: 10 });
  assert.deepStrictEqual(ringKeys(r3), [4]);
  assert.ok(ringStateDoc(r3, 4));
  // The cap's byte count agrees with ringBytes.
  const r4 = ringCreate();
  for (let k = 0; k < 20; k++) ringRecord(r4, k, doc([heavy(1, 300, k % 3), point(2, k)]));
  const limit = ringBytes(r4) * 0.6;
  ringCap(r4, 19, { maxBytes: limit });
  assert.ok(ringBytes(r4) <= limit, `${ringBytes(r4)} <= ${limit}`);
});

test('a state with an altered blob or sum is skipped whole, never applied in part', () => {
  const r = ringCreate();
  ringRecord(r, 0, doc([point(1, 0), point(2, 0)]));
  ringRecord(r, 1, doc([point(1, 1), point(2, 0)]));
  const key = r.states[1].o[0];
  r.blobs.set(key, r.blobs.get(key).replace('"x":1', '"x":2'));
  assert.strictEqual(ringStateDoc(r, 1), null);
  assert.ok(ringStateDoc(r, 0));
  r.states[0].sum = '00000000000000';
  assert.strictEqual(ringStateDoc(r, 0), null);
  assert.strictEqual(ringStateDoc(r, 7), null);
});

test('a manifest that fails its checksum throws; a missing blob skips only its state', () => {
  const r = ringCreate();
  for (let k = 0; k < 3; k++) ringRecord(r, k, doc([point(1, 0), point(2, k)]));
  const m = ringManifest(r, 2);
  const blobs = new Map(r.blobs);
  assert.throws(() => ringFromManifest({ ...m, cursor: 1 }, (k) => blobs.get(k)), /checksum/);
  assert.throws(() => ringFromManifest(null, () => null));
  assert.throws(() => ringFromManifest({ v: 99, states: [] }, () => null));
  const gone = r.states[1].o[1];
  const { ring, skipped } = ringFromManifest(m, (k) => (k === gone ? undefined : blobs.get(k)));
  assert.strictEqual(skipped, 1);
  assert.deepStrictEqual(ringKeys(ring), [0, 2]);
  // A tampered blob reads as missing.
  const { skipped: s2 } = ringFromManifest(m, (k) => (k === gone ? '{"id":2}' : blobs.get(k)));
  assert.strictEqual(s2, 1);
});

test('the blob delta writes only new blobs and removes only orphans', () => {
  const r = ringCreate();
  ringRecord(r, 0, doc([point(1, 0), point(2, 0)]));
  const stored = new Set(r.blobs.keys());
  ringRecord(r, 1, doc([point(1, 1), point(2, 0)]));
  let d = ringBlobDelta(r, stored);
  assert.strictEqual(d.put.length, 1);
  assert.strictEqual(d.del.length, 0);
  ringTruncateAfter(r, 0);
  ringCollect(r);
  d = ringBlobDelta(r, new Set([...stored, 'feedfacefeedfa']));
  assert.deepStrictEqual(d.del, ['feedfacefeedfa']);
  assert.strictEqual(d.put.length, 0);
});

test('a match finds the state holding this content, preferring the named key', () => {
  const r = ringCreate();
  ringRecord(r, 0, doc([point(1, 0)]));
  ringRecord(r, 1, doc([point(1, 1)]));
  ringRecord(r, 2, doc([point(1, 0)]));
  assert.strictEqual(ringMatch(r, doc([point(1, 1)])), 1);
  assert.strictEqual(ringMatch(r, doc([point(1, 0)]), 0), 0);
  assert.strictEqual(ringMatch(r, doc([point(1, 0)])), 2);
  assert.strictEqual(ringMatch(r, doc([point(1, 3)])), null);
  const before = r.blobs.size;
  ringMatch(r, doc([point(9, 9)]));
  assert.strictEqual(r.blobs.size, before);
});

// What a file holds, built the way the app writes it: the envelope indented, the block appended compact.
function fileText(d, ring, cursor, budgetBytes) {
  const base = JSON.stringify({ format: 'unreason-document', version: 1, document: d }, null, 1);
  const out = ringFileBlockText(ring, cursor, d, { budgetBytes: budgetBytes ?? base.length });
  return { base, text: out ? `${base.slice(0, -2)},\n "history": ${out.text}\n}` : base, out };
}

test('a file block round-trips: open it, the cursor is the document, undo reaches every kept state', () => {
  const r = ringCreate();
  const states = [doc([heavy(1, 400, 0)]), doc([heavy(1, 400, 0), point(2, 1)]), doc([heavy(1, 400, 0), point(2, 2)])];
  states.forEach((d, k) => ringRecord(r, k, d));
  const { base, text, out } = fileText(states[2], r, 2);
  assert.strictEqual(out.states, 3);
  const env = JSON.parse(text);
  // What an older build reads is exactly the document.
  assert.deepStrictEqual(env.document, JSON.parse(base).document);
  const { ring, cursor, skipped } = ringFromFileBlock(env.history, env.document);
  assert.strictEqual(skipped, 0);
  assert.strictEqual(cursor, 2);
  for (let k = 0; k < 3; k++) assert.deepStrictEqual(ringStateDoc(ring, k), strip(states[k]), `state ${k}`);
  // The heavy record lives in the document only; the block does not repeat it.
  assert.ok(out.bytes < 1500, `block ${out.bytes} bytes`);
});

test('the file budget keeps fewer states rather than inflate', () => {
  const r = ringCreate();
  for (let k = 0; k < 30; k++) ringRecord(r, k, doc([heavy(1, 300, k)]));
  const d = doc([heavy(1, 300, 29)]);
  const { base, text, out } = fileText(d, r, 29);
  assert.ok(text.length <= base.length * 2, `${text.length} vs ${base.length}`);
  // The budget is a hard bound on what the block adds to the file, at any size.
  for (const frac of [0.5, 0.2, 0.05]) {
    const budget = Math.floor(base.length * frac);
    const f = fileText(d, r, 29, budget);
    assert.ok(f.text.length - base.length <= budget, `${frac}: added ${f.text.length - base.length} > ${budget}`);
  }
  assert.ok(out.states >= 2 && out.states < 30, `${out.states} states`);
  // The kept states are the nearest ones.
  const env = JSON.parse(text);
  const keys = env.history.states.map((s) => s.k);
  assert.deepStrictEqual(keys, Array.from({ length: keys.length }, (_, i) => 29 - keys.length + 1 + i));
  // Nothing to undo to: no block at all.
  const one = ringCreate();
  ringRecord(one, 0, d);
  assert.strictEqual(ringFileBlockText(one, 0, d), null);
  assert.strictEqual(ringFileBlockText(one, 5, d), null);
  // A budget too small for any other state: no block.
  assert.strictEqual(ringFileBlockText(r, 29, d, { budgetBytes: 10 }), null);
});

test('a corrupted file block throws as a whole, or skips the one bad state', () => {
  const r = ringCreate();
  for (let k = 0; k < 3; k++) ringRecord(r, k, doc([point(1, k), point(2, 7)]));
  const d = doc([point(1, 2), point(2, 7)]);
  const env = JSON.parse(fileText(d, r, 2).text);
  for (const bad of [null, 'x', [], { ...env.history, v: 2 }, { ...env.history, check: 'nope' }, { ...env.history, cursor: 0 }, { ...env.history, blobs: null }]) {
    assert.throws(() => ringFromFileBlock(bad, env.document), undefined, JSON.stringify(bad)?.slice(0, 40));
  }
  // An edited blob: its state goes, the rest open.
  const h = structuredClone(env.history);
  const k0 = h.states[0].o[0];
  h.blobs[k0].leaf.point.x = 123;
  const { ring, skipped } = ringFromFileBlock(h, env.document);
  assert.strictEqual(skipped, 1);
  assert.deepStrictEqual(ringKeys(ring), [1, 2]);
});

test('a hash collision is resolved by text, not trusted', () => {
  const r = ringCreate();
  const s = JSON.stringify(point(1, 0));
  // Plant a different text under the key the record would take.
  r.blobs.set(ringHash(s), 'not this record');
  ringRecord(r, 0, doc([point(1, 0)]));
  const key = r.states[0].o[0];
  assert.match(key, /^[0-9a-f]{14}\.1$/);
  assert.strictEqual(r.blobs.get(key), s);
  assert.deepStrictEqual(ringStateDoc(r, 0).objects[0], point(1, 0));
  assert.strictEqual(ringDropState(r, 0), true);
  assert.strictEqual(ringDropState(r, 0), false);
});

test('states that differ only in the id counter share one header blob', () => {
  const r = ringCreate();
  const bulky = { tolerance: 0.001, notes: 'x'.repeat(1500) };
  ringRecord(r, 0, doc([point(1, 0)], { documentSettings: bulky }));
  ringRecord(r, 1, doc([point(1, 0), point(2, 5)], { documentSettings: bulky }));
  assert.strictEqual(r.states[0].h, r.states[1].h);
  assert.ok(ringBytes(r) < 2200, `${ringBytes(r)} bytes for two states over one 1.5 kB header`);
});
