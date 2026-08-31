// TEXT OUTLINES AS NURBS CURVES ("TEXT NODE —
// BUILD. Curves-only output, curated vetted font bank (no arbitrary system
// fonts), tracking+kerning-with-honest-fallback, internal-corner-fillet via
// Tier 1.")
// ================================================================
// This module is the whole maths half of the Text command: glyph coverage
// raster -> iso-contours -> corner-split spans -> ONE closed NURBS curve per
// contour, plus the line/word layout that places those contours in a plane.
// It is pure: no DOM, no three.js, no font file format. The CALLER supplies
// glyph outlines already normalized into em units, which is the one thing a
// browser can produce and this module cannot.
//
// WHERE THE OUTLINES COME FROM, stated plainly because it bounds the accuracy
// of everything below. There is no build step and no network at runtime, so
// the app has no font file to parse and no embedded outline bank. What a
// browser DOES expose is a rasteriser and real font metrics, so the shipped
// path renders one glyph at a time into a coverage raster and traces the 0.5
// coverage iso-line out of it. That is a MEASUREMENT of the typeface, not a
// copy of its Béziers:
//   - accuracy is set by the raster resolution (the Quality param) and by
//     antialiasing giving genuine sub-pixel coverage, not by the font's own
//     control points;
//   - a corner is recovered by detecting it, not by reading a knot, so a
//     corner sharper than the corner-angle threshold survives and one flatter
//     than it is rounded;
//   - a feature thinner than about one raster pixel (a hairline serif at a low
//     Quality setting) can close up or break.
// The curves themselves are ordinary NURBS — control points CARTESIAN with a
// separate weight, never premultiplied — so extrude, offset and fillet all
// read them as they read any other curve in this app.

import { marchingSquares } from './marchingsquares.mjs';
import { fitCurveToPoints } from './fitcurve.mjs';
import { joinCurvesC0 } from './knots.mjs';

// ----------------------------------------------------------------
// CONTOUR EXTRACTION
// ----------------------------------------------------------------

// The coverage raster is ROW-MAJOR (`coverage[y * width + x]`, the layout
// ImageData hands back), while marchingSquares indexes `values[i * vCount + j]`
// with i running U. Transposed here rather than asking every caller to hand
// over a column-major copy of a buffer it just read off a canvas.
//
// `opts.transform` maps grid coordinates to the caller's own space as
// `X = x * sx + tx`, `Y = y * sy + ty`. A raster's y grows DOWNWARD and a
// glyph's y grows upward, so sy is normally negative — doing the flip here
// keeps winding, area sign and corner turn direction all consistent with the
// final coordinates rather than with the raster's.
export function glyphCoverageToContours(coverage, width, height, opts = {}) {
  const threshold = opts.threshold ?? 0.5;
  const t = opts.transform || { sx: 1, sy: 1, tx: 0, ty: 0 };
  if (!(width > 1) || !(height > 1)) return [];
  const values = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) values[x * height + y] = coverage[y * width + x];
  }
  const raw = marchingSquares(values, width, height, threshold);
  const out = [];
  for (const poly of raw) {
    // An OPEN polyline means the iso-line ran off the edge of the raster,
    // which only happens when the glyph was drawn without enough margin.
    // Dropped and counted rather than closed with a straight chord across the
    // gap: a fabricated edge through the middle of a letter is worse than a
    // missing contour, because it looks like geometry.
    if (!poly.closed || poly.pts.length < 4) continue;
    // A CLOSED RING IS STORED WITHOUT REPEATING ITS START. The tracer closes a
    // loop by pushing the first point again before marking it closed, so every
    // contour arrives with `pts[0] === pts[n-1]`; measured across a 28-glyph
    // bank, 40 contours out of 40. Carried forward, that zero-length edge is a
    // coincident-point hazard in every chord-length parametrization downstream
    // (a repeated parameter is what makes a least-squares solve singular), and
    // `polylineCurve(pts, true)` would append a third copy of the same point.
    // The contract is stated in two places already — `simplifyClosedContour`'s
    // own comment and `fitCurveToPoints`' parameter doc — so this is where it
    // is made true rather than assumed.
    const ring = poly.pts.length > 1
      && poly.pts[0][0] === poly.pts[poly.pts.length - 1][0]
      && poly.pts[0][1] === poly.pts[poly.pts.length - 1][1]
      ? poly.pts.slice(0, -1)
      : poly.pts;
    // The length floor above was applied to the duplicate-terminated array, so
    // it asked for three distinct vertices. Re-applied to the ring so dropping
    // the repeat changes what is stored and not which contours survive.
    if (ring.length < 3) continue;
    const pts = ring.map(([ci, cj]) => [ci * t.sx + t.tx, cj * t.sy + t.ty]);
    const area = contourSignedArea(pts);
    if (Math.abs(area) < (opts.minArea ?? 0)) continue;
    out.push({ pts, area });
  }
  return orientContours(out);
}

export function contourSignedArea(pts) {
  let a = 0;
  for (let i = 0, n = pts.length; i < n; i++) {
    const p = pts[i], q = pts[(i + 1) % n];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a / 2;
}

// Standard ray-cast parity test. Used only for NESTING DEPTH, where the query
// point is a vertex of a contour that is either wholly inside or wholly
// outside the one being tested, so the usual on-the-boundary ambiguity cannot
// arise between two contours of the same glyph.
export function pointInContour(pt, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const yi = pts[i][1], yj = pts[j][1];
    if ((yi > pt[1]) === (yj > pt[1])) continue;
    const x = pts[i][0] + ((pt[1] - yi) / (yj - yi)) * (pts[j][0] - pts[i][0]);
    if (pt[0] < x) inside = !inside;
  }
  return inside;
}

// OUTER LOOPS COUNTER-CLOCKWISE, HOLES CLOCKWISE — the convention every
// downstream consumer of a closed profile in this app already expects, and the
// one a later trim/extrude needs in order to know which side is material.
// Nesting depth is counted rather than assumed from area sign: a glyph like
// "%" or a stencil "A" can nest more than one level, and marching squares'
// own output winding depends on which way the coverage gradient happened to
// run, not on what the contour means.
export function orientContours(contours) {
  return contours.map((c, i) => {
    let depth = 0;
    for (let j = 0; j < contours.length; j++) {
      if (i === j) continue;
      if (pointInContour(c.pts[0], contours[j].pts)) depth++;
    }
    const outer = depth % 2 === 0;
    const wantCCW = outer;
    const isCCW = c.area > 0;
    const pts = isCCW === wantCCW ? c.pts : c.pts.slice().reverse();
    return { pts, area: wantCCW ? Math.abs(c.area) : -Math.abs(c.area), depth, outer };
  });
}

// ----------------------------------------------------------------
// SIMPLIFICATION AND CORNERS
// ----------------------------------------------------------------

function perpDistance(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-15) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  return Math.abs((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) / len;
}

function douglasPeucker(pts, eps, first, last, keep) {
  let worst = -1, worstI = -1;
  for (let i = first + 1; i < last; i++) {
    const d = perpDistance(pts[i], pts[first], pts[last]);
    if (d > worst) { worst = d; worstI = i; }
  }
  if (worst > eps && worstI > 0) {
    douglasPeucker(pts, eps, first, worstI, keep);
    keep.add(worstI);
    douglasPeucker(pts, eps, worstI, last, keep);
  }
}

// A CLOSED loop has no natural pair of fixed endpoints for Douglas-Peucker to
// start from, so two anchors are chosen first: an extreme vertex (guaranteed
// to survive any correct simplification) and the vertex farthest from it. The
// loop is then simplified as two open runs between them. Anchoring on an
// arbitrary index instead — pts[0], the point marching squares happened to
// emit first — puts a permanent, meaningless vertex somewhere along a smooth
// edge and biases the fit around it.
export function simplifyClosedContour(pts, eps) {
  const n = pts.length;
  if (n < 5 || !(eps > 0)) return pts.slice();
  let a = 0;
  for (let i = 1; i < n; i++) if (pts[i][1] < pts[a][1] || (pts[i][1] === pts[a][1] && pts[i][0] < pts[a][0])) a = i;
  let b = a, far = -1;
  for (let i = 0; i < n; i++) {
    const d = Math.hypot(pts[i][0] - pts[a][0], pts[i][1] - pts[a][1]);
    if (d > far) { far = d; b = i; }
  }
  if (a === b) return pts.slice();
  // The rotated run is CLOSED by repeating its first point at index n, so the
  // second Douglas-Peucker half spans bIdx..n and the wrap-around edge is
  // simplified like any other. Forcing index n-1 to survive instead (the
  // vertex that merely happens to sit just before the wrap) plants a spurious
  // near-zero-length edge beside the seam, which the corner detector then
  // reads as a real corner and the fit turns into a degenerate span.
  const rot = pts.slice(a).concat(pts.slice(0, a));
  rot.push(rot[0]);
  const bIdx = (b - a + n) % n;
  const keep = new Set([0, bIdx]);
  douglasPeucker(rot, eps, 0, bIdx, keep);
  douglasPeucker(rot, eps, bIdx, n, keep);
  // Index n is the repeated first point and is dropped: the loop is stored
  // WITHOUT repeating its start, which is what fitCurveToPoints' own closed
  // mode expects.
  const out = [...keep].filter((i) => i < n).sort((x, y) => x - y).map((i) => rot[i]);
  return out.length >= 4 ? out : pts.slice();
}

// A CORNER IS A TURN, MEASURED ON THE SIMPLIFIED POLYGON. Running this on the
// raw traced contour instead measures antialiasing staircase noise, where
// every second vertex turns 90 degrees and every letter is all corner.
// Returns indices into `pts`, ascending.
export function contourCornerIndices(pts, minAngleDeg) {
  const n = pts.length;
  const out = [];
  if (n < 3 || !(minAngleDeg > 0)) return out;
  const cosLimit = Math.cos((minAngleDeg * Math.PI) / 180);
  for (let i = 0; i < n; i++) {
    const p = pts[(i - 1 + n) % n], c = pts[i], q = pts[(i + 1) % n];
    const ax = c[0] - p[0], ay = c[1] - p[1];
    const bx = q[0] - c[0], by = q[1] - c[1];
    const la = Math.hypot(ax, ay), lb = Math.hypot(bx, by);
    if (la < 1e-12 || lb < 1e-12) continue;
    // cos of the angle between the two edge DIRECTIONS: +1 is straight
    // through, -1 is a full reversal. A turn of `minAngleDeg` away from
    // straight is cos(minAngleDeg) — anything below that is a corner.
    const cosTurn = (ax * bx + ay * by) / (la * lb);
    if (cosTurn < cosLimit) out.push(i);
  }
  return out;
}

// ----------------------------------------------------------------
// LAYOUT
// ----------------------------------------------------------------

// TRACKING AND KERNING, WITH THE FALLBACK NAMED. Kerning is a PAIR table the
// caller measured off the real font; a pair the caller could not measure
// contributes 0, which is the honest fallback (the pair simply sets at its
// nominal advance) rather than a guessed correction. Tracking is a uniform
// extra advance applied to every pair alike, expressed as a fraction of the
// em so it means the same thing at every size.
//
// `\n` (backslash-n, two literal characters) is a line break. A single-line
// text field cannot carry a real newline, so the escape is what makes multi-
// line text reachable at all from a Properties row.
export function layoutTextGlyphs(text, metrics, opts = {}) {
  const advance = metrics.advance || {};
  const kern = metrics.kern || {};
  const tracking = opts.tracking ?? 0;
  const lineSpacing = opts.lineSpacing ?? 1.4;
  const align = opts.align || 'left';
  const source = String(text ?? '');
  const lines = [];
  let cur = [];
  for (let i = 0; i < source.length; i++) {
    if (source[i] === '\\' && source[i + 1] === 'n') { lines.push(cur); cur = []; i++; continue; }
    if (source[i] === '\n') { lines.push(cur); cur = []; continue; }
    cur.push(source[i]);
  }
  lines.push(cur);

  const placements = [];
  const lineWidths = [];
  const missing = [];
  for (let li = 0; li < lines.length; li++) {
    const chars = lines[li];
    let pen = 0;
    const row = [];
    for (let i = 0; i < chars.length; i++) {
      const ch = chars[i];
      const adv = advance[ch];
      if (adv == null) { if (!missing.includes(ch)) missing.push(ch); continue; }
      row.push({ ch, x: pen, line: li });
      pen += adv + tracking;
      const next = chars[i + 1];
      if (next != null && advance[next] != null) pen += kern[ch + next] || 0;
    }
    lineWidths.push(pen > 0 ? pen - tracking : 0);
    placements.push(row);
  }

  const width = Math.max(0, ...lineWidths);
  const flat = [];
  for (let li = 0; li < placements.length; li++) {
    const shift = align === 'center' ? (width - lineWidths[li]) / 2
      : align === 'right' ? width - lineWidths[li] : 0;
    // The FIRST line sits on y = 0, later lines run DOWNWARD. A student
    // placing text on the construction plane expects the first line where
    // they put the object, not the last one. Line 0 is written as a literal 0
    // rather than -0 * spacing, which is negative zero — indistinguishable on
    // screen and not equal to 0 under a strict comparison.
    const y = li ? -li * lineSpacing : 0;
    for (const g of placements[li]) flat.push({ ch: g.ch, x: g.x + shift, y, line: li });
  }
  return { placements: flat, lineWidths, lineCount: placements.length, width, missing };
}

// ----------------------------------------------------------------
// CONTOUR -> CURVE
// ----------------------------------------------------------------

// The always-available answer: a degree-1 NURBS curve straight through the
// points. Exact by construction (a degree-1 curve IS its control polygon), so
// it can never fail, which is what makes it the right floor under a fit that
// can. A student who lowers the corner threshold to zero, or who traces a
// glyph so intricate that the least-squares solve refuses, still gets real
// closed geometry they can extrude — reported honestly as a polyline rather
// than presented as a smooth fit.
export function polylineCurve(points, closed) {
  const pts = points.map((p) => [p[0], p[1], p[2], 1]);
  if (closed) pts.push([points[0][0], points[0][1], points[0][2], 1]);
  const n = pts.length - 1;
  const knots = [0];
  for (let i = 0; i <= n; i++) knots.push(i / n);
  knots.push(1);
  return { degree: 1, knots, ctrlPts: pts };
}

// ONE CLOSED CURVE PER CONTOUR, WITH ITS CORNERS INTACT.
//
// A single closed least-squares fit over a whole letter rounds every corner by
// roughly the fit tolerance — fine on an "O", wrong on an "L", and it destroys
// exactly the internal corners Tier 1's corner fillet is pointed at.
// So a contour carrying corners is split into corner-to-corner SPANS, each
// span fitted on its own (`exactEndpoints`, so the shared vertex is
// interpolated by both neighbors and the seam is watertight), and the spans
// chained with joinCurvesC0 into one curve whose knot multiplicity AT each
// corner is what makes the corner sharp.
//
// Every span is fitted OPEN, which also keeps the whole chain non-rational:
// fitCurveToPoints only ever emits a rational primitive (circle/ellipse) on
// the closed branch, and joinCurvesC0's degree elevation is not weight-aware.
// A cornerless contour takes the closed branch instead and keeps that exact
// circle when the glyph really is one.
export function contourToCurve(points3, corners, opts = {}) {
  const tolerance = opts.tolerance ?? 1e-3;
  const degree = opts.degree ?? 3;
  const n = points3.length;
  if (n < 4) return { crv: polylineCurve(points3, true), kind: 'polyline', maxDeviation: 0 };
  if (!corners || corners.length === 0) {
    const fit = fitCurveToPoints(points3, { tolerance, degree, closed: true });
    if (fit && fit.ok) return { crv: fit.curve, kind: fit.kind, maxDeviation: fit.maxDeviation };
    return { crv: polylineCurve(points3, true), kind: 'polyline', maxDeviation: 0 };
  }
  const sorted = [...corners].sort((a, b) => a - b);
  const spans = [];
  let worst = 0;
  let anyPolyline = false;
  for (let k = 0; k < sorted.length; k++) {
    const a = sorted[k], b = sorted[(k + 1) % sorted.length];
    // ONE CORNER IS ONE SPAN ALL THE WAY ROUND, and the walk has to start
    // before it can stop for that to be expressible. With a single detected
    // corner `a === b`, so a loop that tests its stop condition first breaks
    // on its own first vertex: the run comes back length 1, every span is
    // skipped, and the whole contour falls out to a degree-1 polyline —
    // measured on a smooth 20-gon with one corner, `kind='polyline' degree=1`
    // where two corners give `kind='spans' degree=3`.
    // A detected corner must survive, so the answer is a span, not the
    // cornerless closed fit: fitting the loop closed would round the one
    // corner away by the tolerance, which is the thing the corner split
    // exists to prevent. The run therefore ends where it began, and the open
    // fit's exact endpoints make that seam the sharp corner it was detected as.
    const run = [points3[a]];
    for (let i = (a + 1) % n; ; i = (i + 1) % n) {
      run.push(points3[i]);
      if (i === b) break;
    }
    if (run.length < 2) continue;
    if (run.length === 2) { spans.push(polylineCurve(run, false)); continue; }
    const fit = fitCurveToPoints(run, { tolerance, degree, closed: false, exactEndpoints: true });
    if (fit && fit.ok) { spans.push(fit.curve); worst = Math.max(worst, fit.maxDeviation); }
    else { spans.push(polylineCurve(run, false)); anyPolyline = true; }
  }
  if (!spans.length) return { crv: polylineCurve(points3, true), kind: 'polyline', maxDeviation: 0 };
  return {
    crv: joinCurvesC0(spans),
    kind: anyPolyline ? 'mixed' : 'spans',
    maxDeviation: worst,
    spanCount: spans.length,
    cornerCount: sorted.length,
  };
}

// ----------------------------------------------------------------
// THE ENTRY POINT
// ----------------------------------------------------------------

// `glyphs` maps a character to `{contours, advance}`, both in EM units with
// the glyph origin at (0,0) on the baseline and y running UP. `contours` is
// exactly what glyphCoverageToContours above returns for that glyph — a list
// of `{pts, outer}` — so the two halves of this module compose without the
// caller reshaping anything in between.
//
// `opts.size` is CAP HEIGHT in model units, not em size: "20mm text" should
// mean the capitals measure 20mm, which is the dimension a student can put a
// ruler on. `opts.capHeight` is the font's own measured cap height in em, so
// the em-to-model scale is size / capHeight.
export function buildTextCurves(glyphs, opts = {}) {
  const size = opts.size ?? 20;
  const capHeight = opts.capHeight > 0 ? opts.capHeight : 0.7;
  const scale = size / capHeight;
  const origin = opts.origin || [0, 0, 0];
  const xAxis = opts.xAxis || [1, 0, 0];
  const yAxis = opts.yAxis || [0, 1, 0];
  const layout = layoutTextGlyphs(opts.text ?? '', {
    advance: Object.fromEntries(Object.entries(glyphs).map(([ch, g]) => [ch, g.advance])),
    kern: opts.kern || {},
  }, { tracking: opts.tracking ?? 0, lineSpacing: opts.lineSpacing ?? 1.4, align: opts.align || 'left' });

  const pixel = (opts.pixelSize ?? 1 / 256) * scale;
  const simplifyEps = pixel * (opts.simplifyPixels ?? 0.55);
  const tolerance = pixel * (opts.tolerancePixels ?? 0.9);
  const cornerAngle = opts.cornerAngle ?? 32;

  const curves = [];
  let worst = 0, polylineCount = 0, cornerTotal = 0;
  for (const place of layout.placements) {
    const g = glyphs[place.ch];
    if (!g || !g.contours) continue;
    for (let ci = 0; ci < g.contours.length; ci++) {
      const rec = g.contours[ci];
      const em = rec && rec.pts ? rec.pts : rec;
      if (!em || em.length < 4) continue;
      // Into model units first, THEN simplified and fitted: eps and tolerance
      // are stated as real distances on the finished letter, so the same
      // Quality setting means the same accuracy whatever the size.
      const flat = em.map((p) => [(p[0] + place.x) * scale, (p[1] + place.y) * scale]);
      const simp = simplifyClosedContour(flat, simplifyEps);
      if (simp.length < 4) continue;
      const corners = contourCornerIndices(simp, cornerAngle);
      cornerTotal += corners.length;
      const pts3 = simp.map(([u, v]) => [
        origin[0] + u * xAxis[0] + v * yAxis[0],
        origin[1] + u * xAxis[1] + v * yAxis[1],
        origin[2] + u * xAxis[2] + v * yAxis[2],
      ]);
      const built = contourToCurve(pts3, corners, { tolerance, degree: opts.degree ?? 3 });
      if (built.kind === 'polyline' || built.kind === 'mixed') polylineCount++;
      worst = Math.max(worst, built.maxDeviation || 0);
      curves.push({
        crv: built.crv, closed: true, char: place.ch, contour: ci,
        outer: rec && rec.outer !== undefined ? !!rec.outer : true,
        kind: built.kind, cornerCount: built.cornerCount ?? 0,
      });
    }
  }
  return {
    curves,
    report: {
      glyphCount: layout.placements.length,
      curveCount: curves.length,
      cornerCount: cornerTotal,
      lineCount: layout.lineCount,
      missing: layout.missing,
      polylineFallbacks: polylineCount,
      maxDeviation: worst,
      widthEm: layout.width,
      tolerance,
    },
  };
}
