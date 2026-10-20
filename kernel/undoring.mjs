// A bounded history of whole-document states, for undo across a reload.
//
// A state is a header blob plus one blob per object record. A blob is the
// record's JSON text keyed by its content hash, so an object that did not
// change between two states is stored once however many states name it. A
// state is addressed by an integer `k`; keys need not be contiguous (an
// autosave that coalesces two edits records one state for both), and a step
// goes to the nearest key either side.
//
// Every read verifies: a blob whose text does not hash to its key, or a state
// whose blob list does not hash to its `sum`, is skipped whole, never applied
// in part. Pure: plain data in, plain data out; no storage, no DOM.

export const RING_VERSION = 1;
export const RING_MAX_STATES = 50;
/* Document keys that describe the view rather than the model: where the
   views look, what is selected, when it was written, the next free object id.
   A state leaves them out, so stepping through history keeps the camera, an
   orbit is not an edit, and states that differ only in the id counter share
   one header blob. A restore takes the counter past every id it rebuilds. */
export const RING_SESSION_KEYS = ['ts', 'views', 'activeViewIndex', 'selectedIds', 'nextId', 'objects'];

/* cyrb53: a 53-bit hash as 14 hex digits. At 53 bits a ring of tens of
   thousands of blobs has a collision chance near 1e-8; a collision is still
   resolved exactly, by comparing text and suffixing the key (`<hash>.<n>`). */
export function ringHash(str) {
  let h1 = 0xdeadbeef, h2 = 0x41c6ce57;
  for (let i = 0; i < str.length; i++) {
    const c = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 2654435761);
    h2 = Math.imul(h2 ^ c, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(16).padStart(14, '0');
}

export function ringCreate() {
  return { states: [], blobs: new Map() };
}

function ringKeyBase(key) {
  const i = key.indexOf('.');
  return i < 0 ? key : key.slice(0, i);
}

// True when `str` is the text the key names.
export function ringBlobOk(key, str) {
  return typeof key === 'string' && typeof str === 'string' && ringHash(str) === ringKeyBase(key);
}

// The key already holding exactly `str` ({ key }), or the free key it would take ({ free }).
function ringFindKey(blobs, str) {
  const base = ringHash(str);
  if (!blobs.has(base)) return { key: null, free: base };
  if (blobs.get(base) === str) return { key: base };
  for (let n = 1; ; n++) {
    const k = `${base}.${n}`;
    if (!blobs.has(k)) return { key: null, free: k };
    if (blobs.get(k) === str) return { key: k };
  }
}

function ringPutBlob(blobs, str) {
  const f = ringFindKey(blobs, str);
  if (f.key) return f.key;
  blobs.set(f.free, str);
  return f.free;
}

function ringStateSum(h, o) {
  return ringHash(`${h}|${o.join(',')}`);
}

// A document split into the part a state stores as its header and its object records.
export function ringDocParts(doc) {
  const header = {};
  for (const k of Object.keys(doc || {})) if (!RING_SESSION_KEYS.includes(k)) header[k] = doc[k];
  return { header, records: doc && Array.isArray(doc.objects) ? doc.objects : [] };
}

function ringIndexOf(ring, k) {
  return ring.states.findIndex((s) => s.k === k);
}

export function ringHasKey(ring, k) {
  return ringIndexOf(ring, k) >= 0;
}

export function ringKeys(ring) {
  return ring.states.map((s) => s.k);
}

export function ringLastKey(ring) {
  return ring.states.length ? ring.states[ring.states.length - 1].k : null;
}

// The nearest stored key strictly below / above `k`, or null.
export function ringPrevKey(ring, k) {
  let out = null;
  for (const s of ring.states) if (s.k < k) out = s.k;
  return out;
}

export function ringNextKey(ring, k) {
  for (const s of ring.states) if (s.k > k) return s.k;
  return null;
}

/* Stores `doc` as the state at `k`, replacing any state already there.
   `held` is how many of its records were held unrebuilt when it was taken; a
   restore that holds more than that did not reproduce the state. Returns
   whether the stored content changed. Blobs the replaced state alone used stay
   until ringCollect. */
export function ringRecord(ring, k, doc, { held = 0 } = {}) {
  if (!Number.isInteger(k)) throw new Error('ringRecord: state key must be an integer');
  const { header, records } = ringDocParts(doc);
  const h = ringPutBlob(ring.blobs, JSON.stringify(header));
  const o = records.map((r) => ringPutBlob(ring.blobs, JSON.stringify(r)));
  const st = { k, h, o, sum: ringStateSum(h, o), held: held | 0 };
  const at = ring.states.findIndex((s) => s.k >= k);
  let changed = true;
  if (at >= 0 && ring.states[at].k === k) {
    changed = ring.states[at].sum !== st.sum || ring.states[at].held !== st.held;
    ring.states[at] = st;
  } else if (at < 0) ring.states.push(st);
  else ring.states.splice(at, 0, st);
  return { state: st, changed };
}

/* The key of a stored state with exactly this document's content, or null.
   Nothing is inserted. `prefer` wins when several states match. */
export function ringMatch(ring, doc, prefer = null) {
  const { header, records } = ringDocParts(doc);
  const h = ringFindKey(ring.blobs, JSON.stringify(header)).key;
  if (!h) return null;
  const o = [];
  for (const r of records) {
    const key = ringFindKey(ring.blobs, JSON.stringify(r)).key;
    if (!key) return null;
    o.push(key);
  }
  const sum = ringStateSum(h, o);
  let found = null;
  for (const s of ring.states) {
    if (s.sum !== sum) continue;
    if (s.k === prefer) return s.k;
    found = s.k;
  }
  return found;
}

// Drops every state above `k`. Returns how many went.
export function ringTruncateAfter(ring, k) {
  const before = ring.states.length;
  ring.states = ring.states.filter((s) => s.k <= k);
  return before - ring.states.length;
}

export function ringDropState(ring, k) {
  const i = ringIndexOf(ring, k);
  if (i < 0) return false;
  ring.states.splice(i, 1);
  return true;
}

/* The document stored at `k`, rebuilt as fresh objects, or null when the state
   is absent or any part of it fails verification. */
export function ringStateDoc(ring, k) {
  const st = ring.states[ringIndexOf(ring, k)];
  if (!st || !Array.isArray(st.o) || typeof st.h !== 'string') return null;
  if (ringStateSum(st.h, st.o) !== st.sum) return null;
  const read = (key) => {
    const s = ring.blobs.get(key);
    if (!ringBlobOk(key, s)) throw new Error('blob');
    return JSON.parse(s);
  };
  try {
    const header = read(st.h);
    if (!header || typeof header !== 'object' || Array.isArray(header)) return null;
    const objects = st.o.map(read);
    return { ...header, objects };
  } catch (_) {
    return null;
  }
}

// Removes every blob no state names.
export function ringCollect(ring) {
  const used = new Set();
  for (const s of ring.states) { used.add(s.h); for (const k of s.o) used.add(k); }
  for (const k of [...ring.blobs.keys()]) if (!used.has(k)) ring.blobs.delete(k);
}

const RING_STATE_OVERHEAD = 64; // the state's own record: key, sum, list punctuation
function ringBlobBytes(ring, key) {
  const s = ring.blobs.get(key);
  return key.length + (s ? s.length : 0) + 4;
}

// Bytes the ring stores: each distinct blob once, plus every state's key list.
export function ringBytes(ring) {
  const seen = new Set();
  let n = 0;
  for (const s of ring.states) {
    n += RING_STATE_OVERHEAD + s.o.length * 17;
    for (const k of [s.h, ...s.o]) if (!seen.has(k)) { seen.add(k); n += ringBlobBytes(ring, k); }
  }
  return n;
}

// Bytes of one state on its own: the size of the document it holds.
export function ringStateBytes(ring, k) {
  const st = ring.states[ringIndexOf(ring, k)];
  if (!st) return 0;
  let n = RING_STATE_OVERHEAD;
  for (const key of new Set([st.h, ...st.o])) n += ringBlobBytes(ring, key);
  return n;
}

/* Holds the ring to `maxStates` and `maxBytes`, dropping the oldest state
   first and the newest redo state after that; the state at `keep` is never
   dropped. Returns the dropped keys. */
export function ringCap(ring, keep, { maxStates = RING_MAX_STATES, maxBytes = Infinity } = {}) {
  const refs = new Map();
  for (const s of ring.states) for (const k of new Set([s.h, ...s.o])) refs.set(k, (refs.get(k) || 0) + 1);
  let bytes = 0;
  for (const s of ring.states) bytes += RING_STATE_OVERHEAD + s.o.length * 17;
  for (const k of refs.keys()) bytes += ringBlobBytes(ring, k);
  const dropped = [];
  while (ring.states.length > 1 && (ring.states.length > maxStates || bytes > maxBytes)) {
    let i = 0;
    if (ring.states[0].k === keep) i = ring.states.length - 1;
    if (ring.states[i].k === keep) break;
    const [s] = ring.states.splice(i, 1);
    bytes -= RING_STATE_OVERHEAD + s.o.length * 17;
    for (const k of new Set([s.h, ...s.o])) {
      const n = refs.get(k) - 1;
      if (n <= 0) { refs.delete(k); bytes -= ringBlobBytes(ring, k); } else refs.set(k, n);
    }
    dropped.push(s.k);
  }
  ringCollect(ring);
  return dropped;
}

function ringStatesBody(ring) {
  return ring.states.map((s) => ({ k: s.k, h: s.h, o: s.o.slice(), sum: s.sum, held: s.held }));
}

// The manifest a store writes beside the blobs: states, cursor, and a checksum over both.
export function ringManifest(ring, cursor) {
  const body = { v: RING_VERSION, cursor, states: ringStatesBody(ring) };
  return { ...body, check: ringHash(JSON.stringify(body)) };
}

function ringStateShapeOk(s) {
  return s && Number.isInteger(s.k) && typeof s.h === 'string' && Array.isArray(s.o)
    && s.o.every((x) => typeof x === 'string') && typeof s.sum === 'string';
}

/* Takes verified blobs into `ring` and keeps each state whose blobs are all
   present and whose sum holds. Returns how many states were skipped. */
function ringAdoptStates(ring, states) {
  let skipped = 0;
  const seen = new Set();
  const kept = [];
  for (const s of states) {
    if (!ringStateShapeOk(s) || seen.has(s.k) || ringStateSum(s.h, s.o) !== s.sum
      || !ring.blobs.has(s.h) || !s.o.every((k) => ring.blobs.has(k))) { skipped++; continue; }
    seen.add(s.k);
    kept.push({ k: s.k, h: s.h, o: s.o.slice(), sum: s.sum, held: s.held | 0 });
  }
  kept.sort((a, b) => a.k - b.k);
  ring.states = kept;
  ringCollect(ring);
  return skipped;
}

/* A ring from a stored manifest and a blob reader (key -> text). Throws when
   the manifest itself is unreadable; a state with a missing or altered blob is
   skipped and counted. */
export function ringFromManifest(manifest, getBlob) {
  if (!manifest || typeof manifest !== 'object' || manifest.v !== RING_VERSION || !Array.isArray(manifest.states)) {
    throw new Error('undo history manifest is unreadable');
  }
  const body = { v: manifest.v, cursor: manifest.cursor, states: manifest.states };
  if (ringHash(JSON.stringify(body)) !== manifest.check) throw new Error('undo history manifest failed its checksum');
  const ring = ringCreate();
  for (const s of manifest.states) {
    if (!ringStateShapeOk(s)) continue;
    for (const key of [s.h, ...s.o]) {
      if (ring.blobs.has(key)) continue;
      const str = getBlob(key);
      if (ringBlobOk(key, str)) ring.blobs.set(key, str);
    }
  }
  const skipped = ringAdoptStates(ring, manifest.states);
  return { ring, cursor: Number.isInteger(manifest.cursor) ? manifest.cursor : ringLastKey(ring), skipped };
}

// The blobs a store must add and remove to hold exactly this ring.
export function ringBlobDelta(ring, storedKeys) {
  const put = [], del = [];
  for (const [k, s] of ring.blobs) if (!storedKeys.has(k)) put.push([k, s]);
  for (const k of storedKeys) if (!ring.blobs.has(k)) del.push(k);
  return { put, del };
}

/* The blobs a document already carries: its header and each record, by the key
   ringRecord gives them. A file stores these once, as its document record. */
function ringDocPool(doc) {
  const pool = new Map();
  const { header, records } = ringDocParts(doc);
  for (const v of [header, ...records]) {
    const s = JSON.stringify(v);
    const k = ringHash(s);
    if (!pool.has(k)) pool.set(k, s);
  }
  return pool;
}

/* The history block a file carries, as JSON text. The state at `cursor` is the
   document being saved, so its blobs are the document's own and are not written
   twice; every other state adds only the blobs it does not share. States go in
   nearest first (undo side, then redo side), each side stopping at the first
   that would take the block past `budgetBytes`. The count is exact: the block's
   text, plus the bytes that join it to the envelope, never exceeds the
   budget. Null when there is nothing to undo or redo to. */
export function ringFileBlockText(ring, cursor, doc, { budgetBytes = Infinity, maxStates = RING_MAX_STATES } = {}) {
  const cur = ring.states[ringIndexOf(ring, cursor)];
  if (!cur) return null;
  const pool = ringDocPool(doc);
  const inPool = (k) => pool.has(k) && pool.get(k) === ring.blobs.get(k);
  const chosen = new Set([cur.k]);
  const written = new Set();
  const stateText = (s) => JSON.stringify({ k: s.k, h: s.h, o: s.o, sum: s.sum, held: s.held });
  // Each state and blob costs its own text plus one separator; the frame is the rest of the block.
  const frame = `,\n "history": {"v":${RING_VERSION},"cursor":${JSON.stringify(cursor)},"states":[],"blobs":{},"check":"${'0'.repeat(14)}"}\n`.length;
  let bytes = frame;
  const cost = (s) => {
    let n = stateText(s).length + 1;
    for (const k of new Set([s.h, ...s.o])) if (!inPool(k) && !written.has(k)) n += JSON.stringify(k).length + 2 + ring.blobs.get(k).length;
    return n;
  };
  const take = (s) => {
    for (const k of [s.h, ...s.o]) if (!inPool(k)) written.add(k);
    chosen.add(s.k);
  };
  bytes += cost(cur);
  take(cur);
  const sides = [ring.states.filter((s) => s.k < cursor).reverse(), ring.states.filter((s) => s.k > cursor)];
  for (const side of sides) {
    for (const s of side) {
      if (chosen.size >= maxStates) break;
      const c = cost(s);
      if (bytes + c > budgetBytes) break;
      bytes += c;
      take(s);
    }
  }
  if (chosen.size < 2) return null;
  const states = ring.states.filter((s) => chosen.has(s.k)).map((s) => ({ k: s.k, h: s.h, o: s.o.slice(), sum: s.sum, held: s.held }));
  const body = { v: RING_VERSION, cursor, states };
  const check = ringHash(JSON.stringify(body));
  const blobText = [...written].sort().map((k) => `${JSON.stringify(k)}:${ring.blobs.get(k)}`).join(',');
  const text = `{"v":${RING_VERSION},"cursor":${JSON.stringify(cursor)},"states":${JSON.stringify(states)},"blobs":{${blobText}},"check":${JSON.stringify(check)}}`;
  return { text, states: states.length, bytes: text.length };
}

/* A ring from a file's history block and the document beside it (read before
   anything rebuilds from that document, which may migrate its records in
   place). Throws when the block is unreadable; a state with a missing or
   altered blob is skipped and counted. */
export function ringFromFileBlock(block, doc) {
  if (!block || typeof block !== 'object' || Array.isArray(block) || block.v !== RING_VERSION
    || !Array.isArray(block.states) || !block.blobs || typeof block.blobs !== 'object') {
    throw new Error('undo history block is unreadable');
  }
  const body = { v: block.v, cursor: block.cursor, states: block.states };
  if (ringHash(JSON.stringify(body)) !== block.check) throw new Error('undo history block failed its checksum');
  const ring = ringCreate();
  for (const [key, val] of Object.entries(block.blobs)) {
    const s = JSON.stringify(val);
    if (ringBlobOk(key, s)) ring.blobs.set(key, s);
  }
  for (const [k, s] of ringDocPool(doc)) if (!ring.blobs.has(k)) ring.blobs.set(k, s);
  const skipped = ringAdoptStates(ring, block.states);
  return { ring, cursor: Number.isInteger(block.cursor) ? block.cursor : ringLastKey(ring), skipped };
}
