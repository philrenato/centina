// THE PARAMETRIC ENVIRONMENT LIBRARY, PORTED VERBATIM FROM THE RENDERER.
//
// Ported piece by piece: HDRI_STUDIO_RECIPES, the four strobe constructors
// _hSb/_hDc/_hRg/_hWs, the ambient bed _hdriAmbientImage, the direction
// convention _hdriDirFromUV, and the rasteriser rtHdriStrobeRaster. The
// renderer in turn took these from the offline environment generator's own
// BUILDERS dict, so these are the TRUE lights that generated the shipped
// .hdr files, not a fit.
//
// PURE MODULE. No DOM, no window, no three.js, no fetch, no imports. Numbers and
// typed arrays in, typed arrays out.
//
// ---- traps this file deliberately preserves ----
//
// 1. sizeU / sizeV ARE DEGREES OF ANGULAR HALF-WIDTH, not UV fractions
//    For 'wash' they mean something else again: sizeU is the
//    azimuthal HALF-span in degrees and sizeV the elevation HALF-height about
//    `elevation`. The generator's wash() takes el_lo/el_hi instead, so the
//    conversion is elevation=(lo+hi)/2, sizeV=(hi-lo)/2.
//
// 2. THE DIRECTION CONVENTION IS EQUIRECT, Y-UP, AZIMUTH 0 = +X, AZIMUTH +90 = +Z.
//    u=0 is azimuth -180deg, u=1 is +180deg; v=0 is the ZENITH (el=+90), v=1 the
//    nadir. A transposed or half-turn-offset convention produces a plausible
//    wrong picture that nobody ever reports as a bug — it just lights the model
//    from the wrong side. Do not "simplify" _hdriDirFromUV.
//
// 3. THE EXTRA `falloff` EXPONENT is the renderer's own addition;
//    the generator has no such term. Every cataloged recipe below sets falloff:1,
//    which makes it a no-op, so the two agree exactly. rtHdriDefaultStrobe
//    (a HAND-ADDED strobe, not a cataloged one) defaults it to 1.5 instead.
//
// 4. THE COMPOSITOR SKIPS A STROBE WITH A FALSY `enabled` (in
//    rtHdriComposite). The cataloged table below carries NO `enabled` field at
//    all — Rendre stamps enabled:true onto each strobe when it instantiates a
//    recipe into an edit session (rtHdriInstantiateRecipe). So the
//    rasteriser here treats "field absent" as enabled and only an EXPLICIT false
//    as off; feeding the raw table straight into Rendre's own compositor instead
//    would silently render a black dome.
//
// 5. Rendre's table shares the color arrays (HDRI_WARM et al) by reference across
//    every recipe, which is why its instantiator does color.slice(). Here every
//    strobe gets its own color array at construction, so a consumer that edits
//    one recipe cannot discolor the others.
//
// 6. The generator's panel() has a `tilt` argument that the strobe schema has
//    no field for. None of the 29 cataloged recipes uses it, so nothing is lost
//    here — but a NEW recipe that wants a rolled softbox cannot be expressed.

/* ---------------------------------------------------------------------------
   Color constants.
   --------------------------------------------------------------------------- */
const HDRI_WARM = [1.0, 0.93, 0.82], HDRI_COOL = [0.82, 0.90, 1.0], HDRI_WHITE = [1, 1, 1];
const HDRI_CINNABAR = [1.0, 0.22, 0.10], HDRI_SODIUM = [1.0, 0.72, 0.18];
const HDRI_TYRIAN = [0.62, 0.12, 0.72], HDRI_VERDIGRIS = [0.16, 0.85, 0.62];

/* ---------------------------------------------------------------------------
   The four strobe constructors.
   Argument ORDER differs per kind and is not guessable; keep the signatures.
   --------------------------------------------------------------------------- */

// softbox — mirrors the generator's panel(). su/sv = angular half-widths in degrees.
const _hSb = (az, el, su, sv, i, c, soft, name) => ({
  kind: 'softbox', azimuth: az, elevation: el, sizeU: su, sizeV: sv,
  intensity: i, color: c.slice(), edgeSoftness: soft != null ? soft : 0.18, falloff: 1, name,
});
// disc / pinhole — mirrors the generator's pinhole(). r = angular radius in degrees,
// written into BOTH sizeU and sizeV.
const _hDc = (az, el, r, i, c) => ({
  kind: 'disc', azimuth: az, elevation: el, sizeU: r, sizeV: r,
  intensity: i, color: c.slice(), edgeSoftness: 0.18, falloff: 1,
});
// ring — mirrors the generator's ring(). sizeU = angular radius from the axis
// direction, sizeV = annulus width, both degrees.
const _hRg = (az, el, r, w, i, c) => ({
  kind: 'ring', azimuth: az, elevation: el, sizeU: r, sizeV: w,
  intensity: i, color: c.slice(), edgeSoftness: 0.18, falloff: 1,
});
// wash — mirrors the generator's wash(). span = azimuthal half-span, elC = band
// center elevation, elHalf = band half-height. Default softness is 0.35 here,
// NOT the 0.18 the other three use.
const _hWs = (az, span, elC, elHalf, i, c, soft) => ({
  kind: 'wash', azimuth: az, elevation: elC, sizeU: span, sizeV: elHalf,
  intensity: i, color: c.slice(), edgeSoftness: soft != null ? soft : 0.35, falloff: 1,
});

/* Three-point key/fill/rim group.
   The fill is always 0.10 SOFTER than whatever hardness is asked for. */
function _hThreeSb(key, fill, rim, hard) {
  const h = hard != null ? hard : 0.18;
  const arr = [
    _hSb(-40, 28, 26, 18, key, HDRI_WARM, h),
    _hSb(55, 18, 18, 13, fill, HDRI_COOL, h + 0.10),
    _hSb(175, 34, 12, 22, rim, HDRI_WHITE, h),
  ];
  arr[0].name = 'Key'; arr[1].name = 'Fill'; arr[2].name = 'Rim';
  return arr;
}

/* N-light studio. The size shrinks as lights are added
   (su = 22-n, sv = 15-floor(n/2)), so three-/four-/five-light are NOT the same
   lights with entries removed. */
function _hNLight(n) {
  const az = [-40, 60, 165, -130, 5].slice(0, n),
    el = [30, 22, 38, 26, 60].slice(0, n),
    br = [9.5, 4.2, 5.5, 3.0, 6.5].slice(0, n),
    co = [HDRI_WARM, HDRI_COOL, HDRI_WHITE, HDRI_WHITE, HDRI_WHITE].slice(0, n);
  const su = 22 - n, sv = 15 - Math.floor(n / 2);
  return az.map((a, idx) => _hSb(a, el[idx], su, sv, br[idx], co[idx]));
}

/* ---------------------------------------------------------------------------
   HDRI_STUDIO_RECIPES — transcribed 1:1 from the renderer's own table.
   Keyed by the same slug the .hdr file and RENDRE_ENV_INDEX use.
   --------------------------------------------------------------------------- */
const RENDRE_ENV_RECIPES_BY_SLUG = {
  'studio': { ambient: { kind: 'bay', lvl: 0.014 }, strobes: [..._hThreeSb(10.5, 3.6, 5.0), _hSb(8, 75, 50, 15, 4.0, HDRI_WHITE, 0.35, 'Overhead')] },
  'rendre-studio': { ambient: { kind: 'bay', lvl: 0.012 }, strobes: [..._hThreeSb(11, 3.5, 6), _hSb(10, 78, 55, 16, 4.5, HDRI_WHITE, 0.3, 'Overhead')] },
  'annulus-low-contrast': { ambient: { kind: 'gray', lvl: 0.05 }, strobes: [_hRg(-25, 12, 24, 7, 3.2, HDRI_WHITE)] },
  'backwash-cinnabar': { ambient: { kind: 'bay', lvl: 0.006 }, strobes: [_hWs(180, 95, 10, 45, 2.6, HDRI_CINNABAR), _hWs(180, 50, 10, 30, 1.8, HDRI_CINNABAR), _hSb(-38, 30, 20, 14, 5.5, HDRI_WHITE)] },
  'backwash-sodium': { ambient: { kind: 'bay', lvl: 0.006 }, strobes: [_hWs(180, 95, 10, 45, 2.6, HDRI_SODIUM), _hWs(180, 50, 10, 30, 1.8, HDRI_SODIUM), _hSb(-38, 30, 20, 14, 5.5, HDRI_WHITE)] },
  'backwash-tyrian': { ambient: { kind: 'bay', lvl: 0.006 }, strobes: [_hWs(180, 95, 10, 45, 2.6, HDRI_TYRIAN), _hWs(180, 50, 10, 30, 1.8, HDRI_TYRIAN), _hSb(-38, 30, 20, 14, 5.5, HDRI_WHITE)] },
  'backwash-verdigris': { ambient: { kind: 'bay', lvl: 0.006 }, strobes: [_hWs(180, 95, 10, 45, 2.6, HDRI_VERDIGRIS), _hWs(180, 50, 10, 30, 1.8, HDRI_VERDIGRIS), _hSb(-38, 30, 20, 14, 5.5, HDRI_WHITE)] },
  'backwash-three-pinholes': {
    ambient: { kind: 'bay', lvl: 0.006 }, strobes: [
      _hWs(180, 95, 10, 45, 2.6, [0.9, 0.9, 1.0]), _hWs(180, 50, 10, 30, 1.8, [0.9, 0.9, 1.0]), _hSb(-38, 30, 20, 14, 5.5, HDRI_WHITE),
      _hDc(-60, 18, 2.2, 26, HDRI_WHITE), _hDc(-5, 40, 2.2, 26, HDRI_WHITE), _hDc(50, 18, 2.2, 26, HDRI_WHITE)],
  },
  'cage-room': {
    ambient: { kind: 'bay', lvl: 0.02 }, strobes: [
      ...[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map((a) => _hSb(a, 0, 1.1, 52, 5.0, HDRI_WHITE, 0.5)),
      _hRg(0, 90, 118, 1.6, 3.2, HDRI_WHITE), _hRg(0, 90, 68, 1.6, 3.2, HDRI_WHITE), _hRg(0, 90, 32, 1.6, 3.2, HDRI_WHITE)],
  },
  'coved': { ambient: { kind: 'gray', lvl: 0.10 }, strobes: [_hRg(0, 90, 90, 50, 1.7, HDRI_WHITE), _hSb(-30, 42, 40, 22, 3.4, HDRI_WHITE, 0.5)] },
  'five-light-studio': { ambient: { kind: 'bay', lvl: 0.012 }, strobes: _hNLight(5) },
  'four-light-studio': { ambient: { kind: 'bay', lvl: 0.012 }, strobes: _hNLight(4) },
  'three-light-studio': { ambient: { kind: 'bay', lvl: 0.012 }, strobes: _hNLight(3) },
  'floorlight-screen': { ambient: { kind: 'bay', lvl: 0.008 }, strobes: [_hDc(0, -90, 68, 7.0, [0.96, 1.0, 1.08]), _hSb(150, 35, 14, 10, 2.6, HDRI_WHITE)] },
  'four-square-mirrored': { ambient: { kind: 'bay', lvl: 0.012 }, strobes: [_hSb(-45, 26, 13, 13, 7.0, HDRI_WHITE), _hSb(45, 26, 13, 13, 7.0, HDRI_WARM), _hSb(135, 26, 13, 13, 7.0, HDRI_WHITE), _hSb(-135, 26, 13, 13, 7.0, HDRI_WARM)] },
  'full-wash-tyrian': { ambient: { kind: 'bay', lvl: 0.004 }, strobes: [_hRg(0, 90, 0, 180, 2.4, HDRI_TYRIAN), _hSb(-35, 30, 18, 13, 4.5, HDRI_WHITE)] },
  'low-contrast': { ambient: { kind: 'gray', lvl: 0.16 }, strobes: [_hSb(-30, 35, 34, 24, 1.7, HDRI_WHITE, 0.55), _hSb(150, 25, 28, 20, 1.1, HDRI_WHITE, 0.55)] },
  'paneled': { ambient: { kind: 'bay', lvl: 0.015 }, strobes: [_hSb(-75, 22, 9, 12, 5.2, HDRI_WHITE), _hSb(-45, 38, 9, 12, 5.2, HDRI_WARM), _hSb(-15, 22, 9, 12, 5.2, HDRI_WHITE), _hSb(15, 38, 9, 12, 5.2, HDRI_WARM), _hSb(45, 22, 9, 12, 5.2, HDRI_WHITE), _hSb(75, 38, 9, 12, 5.2, HDRI_WARM)] },
  'plush-gray': { ambient: { kind: 'gray', lvl: 0.12 }, strobes: [_hSb(-25, 55, 55, 30, 2.1, HDRI_WHITE, 0.6)] },
  'rimlit-three-point': { ambient: { kind: 'bay', lvl: 0.006 }, strobes: [..._hThreeSb(8.0, 2.6, 2.0), _hSb(172, 8, 2.6, 34, 22.0, HDRI_WHITE, 0.30), _hSb(-168, 6, 2.6, 30, 14.0, HDRI_COOL, 0.30)] },
  'soft-back': { ambient: { kind: 'bay', lvl: 0.008 }, strobes: [_hSb(180, 18, 44, 30, 6.5, HDRI_WHITE, 0.5), _hSb(-30, 24, 12, 9, 2.2, HDRI_WHITE)] },
  'soft-light': { ambient: { kind: 'bay', lvl: 0.02 }, strobes: [_hSb(-35, 38, 48, 34, 5.0, HDRI_WARM, 0.62)] },
  'studio-panel': { ambient: { kind: 'bay', lvl: 0.012 }, strobes: [_hSb(-32, 26, 30, 20, 9.0, HDRI_WHITE, 0.14)] },
  'three-point-high-contrast': { ambient: { kind: 'bay', lvl: 0.003 }, strobes: _hThreeSb(14, 2.2, 8, 0.08) },
  'three-point-two-pinholes': { ambient: { kind: 'bay', lvl: 0.012 }, strobes: [..._hThreeSb(11, 3.5, 6), _hSb(10, 78, 55, 16, 4.5, HDRI_WHITE, 0.3, 'Overhead'), _hDc(-95, 35, 1.8, 34, HDRI_WHITE), _hDc(120, 50, 1.8, 30, HDRI_WHITE)] },
  'top-and-bottom-center-pinhole': { ambient: { kind: 'bay', lvl: 0.006 }, strobes: [_hSb(0, 72, 60, 18, 6.0, HDRI_WHITE, 0.4), _hSb(0, -66, 60, 16, 4.0, HDRI_COOL, 0.4), _hDc(180, 4, 2.4, 30, HDRI_WHITE)] },
  'tubes': { ambient: { kind: 'bay', lvl: 0.008 }, strobes: [_hSb(-70, 18, 1.6, 38, 9.0, HDRI_COOL, 0.5), _hSb(-25, 32, 1.6, 38, 9.0, HDRI_WARM, 0.5), _hSb(20, 46, 1.6, 38, 9.0, HDRI_COOL, 0.5), _hSb(70, 18, 1.6, 38, 9.0, HDRI_WARM, 0.5), _hSb(160, 32, 1.6, 38, 9.0, HDRI_COOL, 0.5), _hSb(-150, 46, 1.6, 38, 9.0, HDRI_WARM, 0.5)] },
  'underlit-square': { ambient: { kind: 'bay', lvl: 0.006 }, strobes: [_hSb(-10, -52, 26, 20, 8.0, HDRI_WHITE, 0.16), _hSb(168, 30, 12, 9, 2.0, HDRI_WHITE)] },
  'vertical-bands': { ambient: { kind: 'bay', lvl: 0.008 }, strobes: [-180, -144, -108, -72, -36, 0, 36, 72, 108, 144].map((a) => _hSb(a + 8, 14, 3.2, 42, 6.0, HDRI_WHITE, 0.45)) },
};

/* Display names, from RENDRE_ENV_INDEX. Only the 29 slugs
   that have a recipe appear here; the 5 CC0-photographic studios (ballroom,
   photo-studio-sienna, empty-cell, snow-wood, disused-warehouse) and 'blank' are
   deliberately absent — a photo has no lights to recover. */
const RENDRE_ENV_NAMES = {
  'studio': 'Rendre Studio', 'rendre-studio': 'Rendre Studio',
  'annulus-low-contrast': 'Annulus, Low Contrast', 'backwash-cinnabar': 'Backwash, Cinnabar',
  'backwash-sodium': 'Backwash, Sodium', 'backwash-tyrian': 'Backwash, Tyrian',
  'backwash-verdigris': 'Backwash, Verdigris', 'backwash-three-pinholes': 'Backwash, Three Pinholes',
  'cage-room': 'Cage Room', 'coved': 'Coved', 'five-light-studio': 'Five-Light Studio',
  'four-light-studio': 'Four-Light Studio', 'three-light-studio': 'Three-Light Studio',
  'floorlight-screen': 'Floorlight, Screen', 'four-square-mirrored': 'Four-Square, Mirrored',
  'full-wash-tyrian': 'Full Wash, Tyrian', 'low-contrast': 'Low Contrast', 'paneled': 'Paneled',
  'plush-gray': 'Plush Gray', 'rimlit-three-point': 'Rimlit, Three-Point', 'soft-back': 'Soft Back',
  'soft-light': 'Soft Light', 'studio-panel': 'Studio, Panel',
  'three-point-high-contrast': 'Three-Point, High Contrast',
  'three-point-two-pinholes': 'Three-Point, Two Pinholes',
  'top-and-bottom-center-pinhole': 'Top-and-Bottom, Center Pinhole', 'tubes': 'Tubes',
  'underlit-square': 'Underlit, Square', 'vertical-bands': 'Vertical Bands',
};

/* The library as an ordered array. Order is HDRI_STUDIO_RECIPES' own key order,
   which is what Rendre's __rtHdriRecipeSlugs() reports. */
export const RENDRE_ENV_RECIPES = Object.keys(RENDRE_ENV_RECIPES_BY_SLUG).map((slug) => ({
  slug,
  name: RENDRE_ENV_NAMES[slug] || slug,
  ambient: RENDRE_ENV_RECIPES_BY_SLUG[slug].ambient,
  strobes: RENDRE_ENV_RECIPES_BY_SLUG[slug].strobes,
}));

/** Look one up by slug. Returns undefined for the photographic studios. */
export function rendreEnvRecipe(slug) {
  return RENDRE_ENV_RECIPES.find((r) => r.slug === slug);
}

/* ---------------------------------------------------------------------------
   Ambient bed (== the generator's base()).
   'bay'  = dark floor with a faint COOL glow BELOW the horizon (the blue tint is
            in the multipliers 0.98 / 1.06, not in a color constant).
   'gray' = an even dome brightening toward the top.
   Note the exponents 1.4 and 2.2, and that only 'gray' scales by lvl a second
   time (lvl + lvl*2.2*s^1.4); 'bay' adds a FIXED 0.05*s^2.2 regardless of lvl.
   --------------------------------------------------------------------------- */
export function rendreHdriAmbientImage(kind, lvl, W, H) {
  const rgb = new Float32Array(W * H * 3);
  for (let y = 0; y < H; y++) {
    const v = (y + 0.5) / H; const elRad = (0.5 - v) * Math.PI;
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 3;
      if (kind === 'gray') {
        const s = Math.max(0, Math.sin(elRad)); const g = lvl + lvl * 2.2 * Math.pow(s, 1.4);
        rgb[o] = g; rgb[o + 1] = g; rgb[o + 2] = g;
      } else {
        const s = Math.max(0, -Math.sin(elRad)); const g = lvl + 0.05 * Math.pow(s, 2.2);
        rgb[o] = g; rgb[o + 1] = g * 0.98; rgb[o + 2] = g * 1.06;
      }
    }
  }
  return rgb;
}

/* ---------------------------------------------------------------------------
   Direction convention. Equirect, Y-up, az 0 = +X.
   Same mapping as the trace shader's dirToUV, so a recipe translated either way
   lands in the same place. See trap 2 at the top of this file.
   --------------------------------------------------------------------------- */
function _hdriDirFromUV(u, v) {
  const az = (u * 2 - 1) * Math.PI, el = (0.5 - v) * Math.PI, ce = Math.cos(el);
  return [ce * Math.cos(az), Math.sin(el), ce * Math.sin(az)];
}
function _hdriDirFromAzEl(azDeg, elDeg) {
  const a = (azDeg || 0) * Math.PI / 180, e = (elDeg || 0) * Math.PI / 180, ce = Math.cos(e);
  return [ce * Math.cos(a), Math.sin(e), ce * Math.sin(a)];
}
/* Tangent frame about a direction. The up vector flips
   to +X near the poles (|d.y| >= 0.93) so the cross product never degenerates;
   this is why a softbox at elevation 90 does not spin its own axes to zero. The
   branch is written as two 0/1 scalars rather than a vector on purpose — that is
   what fixes WHICH tangent the frame picks, and therefore which way sizeU points. */
function _hdriFrame(d) {
  const upX = Math.abs(d[1]) < 0.93 ? 0 : 1, upY = Math.abs(d[1]) < 0.93 ? 1 : 0;
  let tx = upY * d[2], ty = -upX * d[2], tz = upX * d[1] - upY * d[0];
  const tl = Math.hypot(tx, ty, tz) || 1; tx /= tl; ty /= tl; tz /= tl;
  const bx = d[1] * tz - d[2] * ty, by = d[2] * tx - d[0] * tz, bz = d[0] * ty - d[1] * tx;
  return [[tx, ty, tz], [bx, by, bz]];
}

/* ---------------------------------------------------------------------------
   The rasteriser. Returns an intensity MULTIPLIER for
   one strobe at one (u,v); 0 means no contribution.

   'wash' is evaluated directly in (u,v) because it is not a localized emitter;
   the other three are evaluated in real 3D direction space, not a UV-plane
   approximation, so nothing warps near the poles. Every clamp, exponent and
   epsilon below is load-bearing:
     - disc:  (rad-ang)/(rad*0.45+0.15) clamped then SQUARED. The +0.15 is what
              keeps a sub-degree pinhole from having a zero-width edge.
     - ring:  (1-|ang-rad|/width) clamped then SQUARED.
     - softbox: atan2 against max(dot,1e-6), a smoothstep g*g*(3-2g) per axis,
              and a MINIMUM edge width of 1.0 degree (max(w*soft,1.0)) — so a
              1.1-degree-wide cage-room slat is almost entirely edge.
     - wash: ga^1.6 times an elevation band gate whose height is
              max((elHi-elLo)*soft, 1) degrees.
   --------------------------------------------------------------------------- */
export function rendreHdriStrobeRaster(u, v, s) {
  if (s.kind === 'wash') {
    const azDeg = (u * 2 - 1) * 180; let da = azDeg - (s.azimuth || 0); da = ((da + 180) % 360 + 360) % 360 - 180;
    const span = Math.max(1e-3, s.sizeU || 30);
    const ga = Math.max(0, Math.min(1, 1 - Math.abs(da) / span));
    const elDeg = (0.5 - v) * 180;
    const elLo = (s.elevation || 0) - (s.sizeV || 30), elHi = (s.elevation || 0) + (s.sizeV || 30);
    const soft = Math.max(0.02, s.edgeSoftness || 0.35);
    const bandH = Math.max((elHi - elLo) * soft, 1);
    const ge = Math.max(0, Math.min(1, (elDeg - elLo) / bandH)) * Math.max(0, Math.min(1, (elHi - elDeg) / bandH));
    const m = Math.pow(ga, 1.6) * ge;
    if (m <= 0) return 0;
    return Math.pow(m, Math.max(0.1, s.falloff || 1)) * (s.intensity || 0);
  }
  const D = _hdriDirFromUV(u, v);
  const d = _hdriDirFromAzEl(s.azimuth, s.elevation);
  const dot = Math.max(-1, Math.min(1, D[0] * d[0] + D[1] * d[1] + D[2] * d[2]));
  let m = 0;
  if (s.kind === 'disc') {
    const ang = Math.acos(dot) * 180 / Math.PI, rad = Math.max(0.1, s.sizeU || 3);
    m = Math.max(0, Math.min(1, (rad - ang) / (rad * 0.45 + 0.15))); m = m * m;
  } else if (s.kind === 'ring') {
    const ang = Math.acos(dot) * 180 / Math.PI, rad = s.sizeU || 30, width = Math.max(0.1, s.sizeV || 10);
    m = Math.max(0, Math.min(1, 1 - Math.abs(ang - rad) / width)); m = m * m;
  } else {   // 'softbox' — the default, and the fallback for any unrecognized kind
    if (dot <= 0) return 0;
    const [t, b] = _hdriFrame(d);
    const pt = D[0] * t[0] + D[1] * t[1] + D[2] * t[2], pb = D[0] * b[0] + D[1] * b[1] + D[2] * b[2];
    const xa = Math.atan2(pt, Math.max(dot, 1e-6)) * 180 / Math.PI;
    const yb = Math.atan2(pb, Math.max(dot, 1e-6)) * 180 / Math.PI;
    const wa = s.sizeU || 8, wb = s.sizeV || 8, soft = Math.max(0.02, s.edgeSoftness || 0.18);
    const sa = Math.max(wa * soft, 1.0), sb = Math.max(wb * soft, 1.0);
    const ga = Math.max(0, Math.min(1, (wa - Math.abs(xa)) / sa)), gb = Math.max(0, Math.min(1, (wb - Math.abs(yb)) / sb));
    m = (ga * ga * (3 - 2 * ga)) * (gb * gb * (3 - 2 * gb));
  }
  if (m <= 0) return 0;
  return Math.pow(m, Math.max(0.1, s.falloff || 1)) * (s.intensity || 0);
}

/* ---------------------------------------------------------------------------
   Recipe -> equirect radiance map. Mirrors rtHdriComposite,
   with the ambient bed standing in for its baseLayer, which is exactly what
   rtHdriEditOpen builds for a cataloged studio.

   Output is LINEAR radiance, RGB, 3 floats per pixel, row 0 = the ZENITH.
   Pixel centers: u=(x+0.5)/W, v=(y+0.5)/H — half-texel offsets, not (x/W).
   --------------------------------------------------------------------------- */
export function rendreRasterEnvRecipe(recipe, W, H) {
  W = W | 0; H = H | 0;
  if (!(W > 0) || !(H > 0)) throw new Error('rendreRasterEnvRecipe: W and H must be positive integers');
  const amb = recipe && recipe.ambient;
  const data = amb
    ? rendreHdriAmbientImage(amb.kind, amb.lvl, W, H)
    : new Float32Array(W * H * 3);
  for (const s of (recipe && recipe.strobes) || []) {
    if (s.enabled === false) continue;   // absent === enabled; see trap 4
    const c = s.color || [1, 1, 1];
    for (let y = 0; y < H; y++) {
      const v = (y + 0.5) / H;
      for (let x = 0; x < W; x++) {
        const u = (x + 0.5) / W;
        const m = rendreHdriStrobeRaster(u, v, s);
        if (m <= 0) continue;
        const o = (y * W + x) * 3;
        data[o] += m * c[0]; data[o + 1] += m * c[1]; data[o + 2] += m * c[2];
      }
    }
  }
  return { width: W, height: H, data };
}
