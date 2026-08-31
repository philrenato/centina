// PARAM SENSITIVITY — the GENERIC finite-difference fallback for direct
// manipulation.
//
// WHAT THIS IS FOR. Every operator in the app is a plain-data param bag plus a
// pure `evaluate(inputs, params)`. Any numeric param has to be
// DRAGGABLE on canvas. For most params the inverse is analytic and trivial
// (an offset distance is the perpendicular distance from the source curve to
// the cursor) and the author hand-writes `fromDrag`. This module is what
// happens for every OTHER param — the ones nobody wrote an inverse for:
// nudge the param, re-run the entry's own `evaluate`, measure how the
// sampled geometry moved, and least-squares that measured response against
// the geometric change the cursor is asking for. That is what makes
// draggability scale to every registry entry instead of the handful
// someone hand-writes.
//
// CITED SOURCE — "Direct Manipulation of Procedural Implicit Surfaces",
// Marzia Riso, Elie Michel, Axel Paris, Valentin Deschaintre, Mathieu
// Gaillard, Fabio Pellacini. SIGGRAPH Asia 2024, ACM TOG 43(6),
// DOI 10.1145/3687936. WHAT WE TOOK: the framing (a user drags the rendered
// geometry; the drag is propagated back into parameter updates) and the idea
// of a GENERIC fallback that works off the procedural definition itself
// rather than a per-parameter hand-written inverse. WHAT WE DID NOT TAKE:
// their machinery. They automatically differentiate a procedural graph over
// an IMPLICIT (SDF) surface and carry a co-parameterization to keep surface
// points identified across an edit. Our geometry is EXPLICIT NURBS and our
// params are plain scalars, so a one-scalar-at-a-time finite difference is
// sufficient — and, far more importantly here, it is verifiable: every claim
// this module makes can be cross-checked against a case whose analytic
// derivative is independently known (see test/sensitivity.test.mjs). No
// autodiff, no co-parameterization, no implicit surfaces.
//
// THIS IS AN ESTIMATOR. Nothing here is exact for a nonlinear param. Every
// entry point returns its own measured quality signal alongside its answer,
// and refuses honestly rather than returning a plausible wrong number — a
// wrong-but-plausible parameter value is worse than a refusal here, because
// downstream it becomes a silent geometry jump the student can't explain.
//
// THE LOAD-BEARING ASSUMPTION — CORRESPONDENCE. A finite difference compares
// "geometry before" against "geometry after", which is only meaningful once
// you have decided WHICH point corresponds to WHICH. This module's answer:
// the caller supplies a `sample(result)` function, and sample index i of the
// baseline is DEFINED to correspond to sample index i of every probe. That
// puts the assumption in one visible place instead of hiding it in the math.
// The caller's contract is therefore: sample() must return points in a
// STABLE ORDER and a STABLE COUNT across small changes of the param being
// probed. A changed count is detected and refused ('unstable-sampling') —
// that is exactly what a topology-changing param (a segment count, a sides
// count) does, so the check doubles as a real discrete-param detector.
// A changed count is detectable; a silently REORDERED or reparametrized
// sample set is NOT, and would produce a confidently wrong answer. That is
// the residual risk of this whole method, named here rather than buried:
// see sampleGeometry() below for the correspondence rule the default sampler
// uses, and its own honest limits.

import { curvePoint } from './curve.mjs';
import { surfacePoint } from './surface.mjs';

export const SENSITIVITY_DEFAULTS = Object.freeze({
  // STEP SIZE. A single absolute epsilon is wrong across params whose scales
  // differ by orders of magnitude — 1e-3 is a reasonable nudge for a radius
  // in mm and a catastrophic one for a normalized 0..1 softness. So the step
  // is RELATIVE to the param's own magnitude, with an absolute floor so a
  // param sitting at (or near) zero still gets probed at all.
  //
  //   h = max(relStep * |value|, floor)
  //   floor = max(absStepFloor, (max - min) * rangeStepFrac)   [range if declared]
  //
  // relStep 1e-3 is chosen from both ends: a NURBS evaluation chain
  // accumulates roughly 1e-12 relative float error, so a 1e-3 relative nudge
  // sits ~9 orders above the noise; and for a smooth function the central
  // difference's truncation error is O(h^2) ~ 1e-6 relative, far below the
  // precision any cursor drag actually needs. rangeStepFrac exists so a
  // declared [min,max] param that happens to sit at 0 (a draft angle, a
  // signed offset) is probed at a fraction of its own declared range rather
  // than at the generic 1e-6 floor, which for a degrees-valued param would
  // be pointlessly small.
  relStep: 1e-3,
  absStepFloor: 1e-6,
  rangeStepFrac: 1e-4,

  // Central differences (two probes, O(h^2)) by default. Set false to spend
  // one evaluate instead of two, at O(h) accuracy — worth it only if the
  // caller's evaluate is genuinely expensive at drag rate.
  central: true,

  // 1 = a single linear solve. >1 re-linearizes at the proposed value and
  // solves again (Gauss-Newton), which matters for a genuinely nonlinear
  // param; each extra iteration costs another probe pair.
  iterations: 1,

  // |cos| between the param's measured geometric response and the requested
  // displacement, below which the param is refused as unable to explain the
  // drag ('orthogonal'). Deliberately small: a param that explains even a
  // little of the drag is still a legitimate (if weak) handle, and the
  // returned `alignment` lets the caller apply a stricter rule.
  alignmentTol: 0.05,

  // Motion below motionTol * geomScale is treated as no motion at all.
  // geomScale is the largest sample coordinate magnitude, because absolute
  // float error in an evaluation scales with the magnitude of the
  // coordinates involved — a geometry sitting 10 meters from the origin
  // genuinely cannot resolve a nanometer of response.
  motionTol: 1e-9,

  // When the fine step produces no measurable motion, retry ONCE at
  // coarseFactor * h before declaring the param dead. This is what separates
  // "this param does nothing" from "this param only responds in discrete
  // jumps" — see the 'quantized' refusal. Since h is itself ~relStep*|value|,
  // coarseFactor 250 makes the coarse probe roughly a QUARTER of the param's
  // own magnitude: deliberately huge, because its only job is a yes/no
  // "does this param do anything at all". Anything smaller silently fails
  // the commonest real case — a param quantized to WHOLE units (a notch
  // count, an internally rounded index) sitting at a value of a few, where a
  // 3% nudge rounds straight back to where it started and the param reads as
  // inert. (planProbes still clamps the coarse step to the declared range.)
  coarseFactor: 250,

  // THE SMOOTHNESS GATE, and why it exists — this was found by probing, not
  // by reasoning. A param that rebuilds the geometry's own parametric
  // structure (a `segments` count feeding makeCircle, a `sides` count) does
  // NOT necessarily change the number of samples: a parameter-fraction
  // sampler happily returns the same 24 points off a curve whose knot vector
  // was just rebuilt from 9 control points to 11. The count check misses it
  // completely, and what comes back is a large, confident, WRONG derivative
  // — the single worst failure this module could have. The detector that
  // does catch it is smoothness: for any genuinely differentiable response,
  // the forward and backward halves of a central difference must agree to
  // O(h^2), i.e. (P(v+h) - P(v)) ~= (P(v) - P(v-h)). A param that jumps
  // structurally disagrees grossly (typically one side moves and the other
  // does not move at all). smoothTol is the fraction of the response allowed
  // to disagree between the two sides before the param is refused as
  // 'non-smooth'. 0.25 is deliberately permissive — a real nonlinear param
  // disagrees by only about h*|f''/f'|, which at a 1e-3 relative step is
  // parts-per-thousand, so this rejects structural jumps without touching
  // ordinary curvature.
  //
  // A param pinned at a declared bound can only be probed one-sided, and so
  // gets the same gate via a HALF-STEP probe on the one legal side instead
  // (see jacobianAt). HONEST LIMIT of both forms: they detect a jump that
  // lands INSIDE the probed interval. A param that is genuinely smooth
  // across the whole probe window but discontinuous somewhere further out is
  // correctly not refused here — the estimate is locally honest, and the
  // measured `residual` on the solve path is what catches the drag having
  // gone somewhere unexpected.
  smoothTol: 0.25,
});

function resolveOptions(o) {
  return { ...SENSITIVITY_DEFAULTS, ...(o || {}) };
}

// ---------------------------------------------------------------------------
// small vector helpers over a flat [x0,y0,z0, x1,y1,z1, ...] array
// ---------------------------------------------------------------------------

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function norm(a) {
  return Math.sqrt(dot(a, a));
}

function refuse(reason, message, extra) {
  return { ok: false, reason, message, ...(extra || {}) };
}

// ---------------------------------------------------------------------------
// DEFAULT SAMPLER (optional — a caller may always supply its own)
//
// CORRESPONDENCE RULE: samples are taken at FIXED FRACTIONS of each curve's /
// surface's own parametric domain, in a fixed traversal order. Sample i of
// one evaluation therefore corresponds to sample i of another IF the two
// share the same parametric structure — which is true for an ordinary
// continuous param (a radius, a distance, an angle) and FALSE for a param
// that rebuilds the knot vector (a segment/sides count). The latter usually
// changes the sample COUNT too and is refused, but not always: a param that
// keeps the count while reparametrizing would slide correspondence silently.
// A caller with a better correspondence (arc-length fractions, a real
// closest-point mapping, or the actual dragged control point's own index)
// should pass its own sampler; this one is the honest generic default, not a
// claim of correctness for every possible param.
// ---------------------------------------------------------------------------

function curveDomain(crv) {
  const p = crv.degree;
  return [crv.knots[p], crv.knots[crv.knots.length - 1 - p]];
}

export function sampleGeometry(result, opts = {}) {
  const nC = opts.curveSamples ?? 24;
  const nU = opts.surfaceSamplesU ?? 7;
  const nV = opts.surfaceSamplesV ?? 7;
  const out = [];

  const pushCurve = (crv) => {
    const [u0, u1] = curveDomain(crv);
    for (let i = 0; i < nC; i++) {
      const u = nC === 1 ? u0 : u0 + (u1 - u0) * (i / (nC - 1));
      const p = curvePoint(crv, u);
      out.push([p[0], p[1], p[2]]);
    }
  };
  const pushSrf = (srf) => {
    const u0 = srf.knotsU[srf.degU], u1 = srf.knotsU[srf.knotsU.length - 1 - srf.degU];
    const v0 = srf.knotsV[srf.degV], v1 = srf.knotsV[srf.knotsV.length - 1 - srf.degV];
    for (let i = 0; i < nU; i++) {
      for (let j = 0; j < nV; j++) {
        const u = nU === 1 ? u0 : u0 + (u1 - u0) * (i / (nU - 1));
        const v = nV === 1 ? v0 : v0 + (v1 - v0) * (j / (nV - 1));
        const p = surfacePoint(srf, u, v);
        out.push([p[0], p[1], p[2]]);
      }
    }
  };

  if (!result || typeof result !== 'object') {
    throw new Error('sampleGeometry: evaluate() returned no object to sample');
  }
  if (Array.isArray(result.panels)) {
    for (const panel of result.panels) if (panel && panel.srf) pushSrf(panel.srf);
  } else if (result.srf) {
    pushSrf(result.srf);
  } else if (result.crv) {
    pushCurve(result.crv);
  } else if (Array.isArray(result.curves)) {
    for (const c of result.curves) pushCurve(c.crv || c);
  } else if (Array.isArray(result.points)) {
    for (const p of result.points) out.push([p[0], p[1], p[2]]);
  } else if (result.start && result.end) {
    out.push([result.start[0], result.start[1], result.start[2]]);
    out.push([result.end[0], result.end[1], result.end[2]]);
  } else if (result.pos) {
    out.push([result.pos[0], result.pos[1], result.pos[2]]);
  } else {
    throw new Error('sampleGeometry: unrecognized evaluate() result shape — supply your own sample()');
  }
  if (!out.length) throw new Error('sampleGeometry: produced zero sample points');
  return out;
}

// ---------------------------------------------------------------------------
// probing
// ---------------------------------------------------------------------------

// Runs evaluate at ONE param value and flattens the sampled points. Never
// mutates the caller's params object. Returns either {ok:true, flat, points}
// or a refusal — an evaluate() that throws is a completely ordinary outcome
// (a degenerate recipe), not an exception to propagate.
function probe(req, value) {
  const { evaluate, inputs, params, key } = req;
  const sample = req.sample || sampleGeometry;
  let result;
  try {
    result = evaluate(inputs, { ...params, [key]: value });
  } catch (err) {
    return refuse('evaluate-failed', `evaluate() threw at ${key}=${value}: ${err && err.message}`);
  }
  let pts;
  try {
    pts = sample(result);
  } catch (err) {
    return refuse('sample-failed', `sample() threw at ${key}=${value}: ${err && err.message}`);
  }
  if (!Array.isArray(pts) || pts.length === 0) {
    return refuse('sample-failed', 'sample() returned no points');
  }
  const flat = new Array(pts.length * 3);
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    for (let c = 0; c < 3; c++) {
      const v = p && p[c];
      if (!Number.isFinite(v)) {
        return refuse('non-finite-geometry', `sampled geometry at ${key}=${value} contains a non-finite coordinate`);
      }
      flat[i * 3 + c] = v;
    }
  }
  return { ok: true, flat, points: pts };
}

// The largest sample coordinate magnitude — the scale float error in an
// evaluation is actually proportional to (see motionTol above).
function geomScaleOf(flat) {
  let m = 0;
  for (let i = 0; i < flat.length; i += 3) {
    m = Math.max(m, Math.hypot(flat[i], flat[i + 1], flat[i + 2]));
  }
  return Math.max(m, 1);
}

// The largest per-sample displacement between two flat sample arrays.
function maxPointMotion(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i += 3) {
    m = Math.max(m, Math.hypot(a[i] - b[i], a[i + 1] - b[i + 1], a[i + 2] - b[i + 2]));
  }
  return m;
}

function stepFor(value, spec, opt) {
  let floor = opt.absStepFloor;
  const min = spec && Number.isFinite(spec.min) ? spec.min : null;
  const max = spec && Number.isFinite(spec.max) ? spec.max : null;
  if (min !== null && max !== null && max > min) {
    floor = Math.max(floor, (max - min) * opt.rangeStepFrac);
  }
  return Math.max(opt.relStep * Math.abs(value), floor);
}

// Refuse a param this method structurally cannot handle, BEFORE spending any
// evaluate calls. A declared non-number (boolean/enum/vec3/array/string), a
// declared integer, or a non-finite current value are all honest refusals —
// finite-differencing a value that has no meaningful "slightly more" is not
// a hard problem to be solved, it is a category error.
function declaredRefusal(value, spec, key) {
  if (spec) {
    if (spec.type && spec.type !== 'number') {
      return refuse('discrete-param', `'${key}' is declared type '${spec.type}', not a continuous number — it has no meaningful derivative, so it cannot be dragged by this generic fallback`);
    }
    if (spec.enumValues) {
      return refuse('discrete-param', `'${key}' is an enum — it has no meaningful derivative`);
    }
    if (spec.integer === true || spec.discrete === true) {
      return refuse('discrete-param', `'${key}' is declared integer/discrete — it only takes whole values, so a finite difference on it is meaningless`);
    }
  }
  if (!Number.isFinite(value)) {
    return refuse('non-numeric-param', `'${key}' has no finite numeric value to differentiate (got ${JSON.stringify(value)})`);
  }
  return null;
}

// Where to place the probe(s), honouring a declared [min,max]. A param
// sitting exactly at its max cannot step forward, so it steps BACKWARD — a
// param pinned at a bound is still perfectly draggable in the direction that
// stays legal, and refusing there would make every clamped param dead.
function planProbes(value, h, spec, opt, key) {
  const min = spec && Number.isFinite(spec.min) ? spec.min : -Infinity;
  const max = spec && Number.isFinite(spec.max) ? spec.max : Infinity;
  if (max <= min) {
    return refuse('degenerate-range', `'${key}' declares an empty range [${min}, ${max}] — there is nothing to move it through`);
  }
  const step = Math.min(h, (max - min) / 2);
  if (!(step > 0)) {
    return refuse('degenerate-range', `'${key}' declares a range [${min}, ${max}] too narrow to place a probe inside`);
  }
  const canUp = value + step <= max;
  const canDown = value - step >= min;
  if (opt.central && canUp && canDown) return { ok: true, mode: 'central', step };
  if (canUp) return { ok: true, mode: 'forward', step };
  if (canDown) return { ok: true, mode: 'backward', step };
  return refuse('degenerate-range', `'${key}' has no legal probe position inside its declared range [${min}, ${max}]`);
}

// Builds the Jacobian (d sampled-geometry / d param) at `value`, plus the
// baseline samples it was measured around.
function jacobianAt(req, value, opt, spec) {
  const base = probe(req, value);
  if (!base.ok) return base;
  const geomScale = geomScaleOf(base.flat);
  const h0 = stepFor(value, spec, opt);

  const measure = (h) => {
    const plan = planProbes(value, h, spec, opt, req.key);
    if (!plan.ok) return plan;
    const step = plan.step;
    let jac, motion, asymmetry = null;
    if (plan.mode === 'central') {
      const up = probe(req, value + step);
      if (!up.ok) return up;
      const down = probe(req, value - step);
      if (!down.ok) return down;
      if (up.flat.length !== base.flat.length || down.flat.length !== base.flat.length) {
        return refuse('unstable-sampling', `nudging '${req.key}' changed the sample COUNT (${base.flat.length / 3} -> ${up.flat.length / 3}) — sample-to-sample correspondence is broken, which is the signature of a topology-changing (segment/count) param`);
      }
      jac = new Array(base.flat.length);
      for (let i = 0; i < jac.length; i++) jac[i] = (up.flat[i] - down.flat[i]) / (2 * step);
      motion = Math.max(maxPointMotion(base.flat, up.flat), maxPointMotion(base.flat, down.flat));
      // Forward vs backward half of the same central difference (see
      // smoothTol): they must agree to O(h^2) for any real derivative.
      let num = 0, fpN = 0, fmN = 0;
      for (let i = 0; i < jac.length; i++) {
        const fp = up.flat[i] - base.flat[i];
        const fm = base.flat[i] - down.flat[i];
        num += (fp - fm) * (fp - fm);
        fpN += fp * fp;
        fmN += fm * fm;
      }
      const den = Math.sqrt(Math.max(fpN, fmN));
      asymmetry = den > 0 ? Math.sqrt(num) / den : 0;
    } else {
      const sign = plan.mode === 'forward' ? 1 : -1;
      const other = probe(req, value + sign * step);
      if (!other.ok) return other;
      if (other.flat.length !== base.flat.length) {
        return refuse('unstable-sampling', `nudging '${req.key}' changed the sample COUNT (${base.flat.length / 3} -> ${other.flat.length / 3}) — sample-to-sample correspondence is broken, which is the signature of a topology-changing (segment/count) param`);
      }
      jac = new Array(base.flat.length);
      for (let i = 0; i < jac.length; i++) jac[i] = sign * (other.flat[i] - base.flat[i]) / step;
      motion = maxPointMotion(base.flat, other.flat);
      // One-sided probing has no opposite half to compare against, so the
      // smoothness gate uses a HALF-STEP probe on the same side instead:
      // for a differentiable response f(v+h)-f(v) must be about twice
      // f(v+h/2)-f(v) (both are h*f' to leading order; they differ only by
      // h^2*f''/4). A structural jump fails this the same way it fails the
      // two-sided check, so a param pinned at a declared bound is covered
      // too — at the cost of one extra evaluate on this path only.
      const half = probe(req, value + sign * step / 2);
      if (half.ok && half.flat.length === base.flat.length) {
        let num = 0, fullN = 0, dblHalfN = 0;
        for (let i = 0; i < jac.length; i++) {
          const full = other.flat[i] - base.flat[i];
          const dblHalf = 2 * (half.flat[i] - base.flat[i]);
          num += (full - dblHalf) * (full - dblHalf);
          fullN += full * full;
          dblHalfN += dblHalf * dblHalf;
        }
        const den = Math.sqrt(Math.max(fullN, dblHalfN));
        asymmetry = den > 0 ? Math.sqrt(num) / den : 0;
      }
    }
    for (const v of jac) {
      if (!Number.isFinite(v)) return refuse('non-finite-geometry', 'the finite difference produced a non-finite derivative');
    }
    return { ok: true, jac, motion, step, mode: plan.mode, asymmetry };
  };

  const fine = measure(h0);
  if (!fine.ok) return fine;

  const noiseFloor = opt.motionTol * geomScale;
  if (fine.motion <= noiseFloor) {
    // No measurable response at the fine step. Before calling the param dead,
    // try a much coarser one — this is what tells a genuinely inert param
    // apart from one that only ever responds in discrete jumps (an internally
    // rounded count, a thresholded mode switch).
    const coarse = measure(h0 * opt.coarseFactor);
    // A coarse probe that BREAKS CORRESPONDENCE is a more informative answer
    // than "inert": the param is discrete/topology-changing, it just happens
    // to round back to itself at a fine nudge. Report what was actually
    // found rather than the weaker fine-step conclusion.
    if (!coarse.ok && coarse.reason === 'unstable-sampling') return coarse;
    if (coarse.ok && coarse.motion > noiseFloor) {
      return refuse('quantized', `'${req.key}' does not move the geometry at a ${h0.toPrecision(3)} nudge but does at ${(h0 * opt.coarseFactor).toPrecision(3)} — it is either quantized/rounded internally (a discrete param in continuous clothing) or its influence is below numerical resolution at this scale. Either way there is no trustworthy derivative here.`, {
        step: h0,
        coarse: { step: coarse.step, motion: coarse.motion },
        geomScale,
      });
    }
    return refuse('insensitive', `'${req.key}' does not measurably change the geometry (largest sampled motion ${fine.motion.toExponential(2)} at a ${h0.toPrecision(3)} nudge, and none at ${opt.coarseFactor}x that) — dragging it would do nothing`, {
      step: h0,
      motion: fine.motion,
      geomScale,
    });
  }

  // The param genuinely moved the geometry — but did it move it CONTINUOUSLY?
  // (See smoothTol: this is the check that catches a structure-rebuilding
  // param whose sample count happens to stay stable.)
  if (fine.asymmetry !== null && fine.asymmetry > opt.smoothTol) {
    return refuse('non-smooth', `'${req.key}' responds inconsistently on either side of its current value (the two halves of the central difference disagree by ${(100 * fine.asymmetry).toFixed(0)}%) — its response is a structural jump, not a continuous motion, so no trustworthy derivative exists. This is the signature of a param that rebuilds the geometry (a segment/sides count, an internally rounded value) rather than moving it.`, {
      asymmetry: fine.asymmetry,
      step: fine.step,
      motion: fine.motion,
      geomScale,
    });
  }

  const jNorm = norm(fine.jac);
  const sampleCount = base.flat.length / 3;
  return {
    ok: true,
    key: req.key,
    baseValue: value,
    baseFlat: base.flat,
    basePoints: base.points,
    jacobian: fine.jac,
    jacobianNorm: jNorm,
    // RMS millimeters of geometric motion per unit change of the param —
    // the honest scalar answer to "how strongly does this param bite?"
    sensitivity: jNorm / Math.sqrt(sampleCount),
    asymmetry: fine.asymmetry,
    motion: fine.motion,
    step: fine.step,
    stepMode: fine.mode,
    sampleCount,
    geomScale,
  };
}

// ---------------------------------------------------------------------------
// PUBLIC: sensitivity only
// ---------------------------------------------------------------------------

// ESTIMATE how strongly one param moves the sampled geometry, with no target
// drag involved. Useful on its own for deciding whether to OFFER a drag
// handle at all, and for ranking which of several params best explains a
// gesture. Returns a refusal (ok:false, with a `reason` code) rather than a
// number whenever the param is structurally undraggable.
//
//   req = { evaluate, inputs, params, key, spec?, sample?, options? }
//
// Every returned figure is MEASURED, never assumed.
export function estimateParamSensitivity(req) {
  const opt = resolveOptions(req.options);
  const spec = req.spec || null;
  const value = req.params && req.params[req.key] !== undefined
    ? req.params[req.key]
    : (spec ? spec.default : undefined);
  const declined = declaredRefusal(value, spec, req.key);
  if (declined) return declined;
  return jacobianAt(req, value, opt, spec);
}

// ---------------------------------------------------------------------------
// PUBLIC: solve a param for a requested geometric change
// ---------------------------------------------------------------------------

// Normalizes the two accepted `targets` shapes into a flat desired-
// displacement vector d plus the list of flat component indices that are
// actually CONSTRAINED:
//   (a) an array parallel to the samples, entries [x,y,z] or null
//   (b) an array of { index, point:[x,y,z] } pairs
//
// THE `active` MASK IS LOAD-BEARING, not bookkeeping. A sample the caller
// said nothing about is UNCONSTRAINED — it must be excluded from the fit
// entirely, not folded in as a zero-displacement "stay exactly put" demand.
// Conflating the two is a real and easy mistake with a silently plausible
// result: drag one point of a 24-sample circle outward by 5mm while the
// other 23 are (wrongly) pinned, and the least-squares fit dutifully returns
// about 5/24th of the radius change the cursor asked for. The drag would
// simply feel weak and laggy, with nothing obviously broken to point at.
function buildDesired(targets, baseFlat, sampleCount) {
  const d = new Array(baseFlat.length).fill(0);
  const active = [];
  let touched = 0;
  const put = (idx, pt) => {
    if (!Number.isInteger(idx) || idx < 0 || idx >= sampleCount) {
      throw new Error(`target index ${idx} is outside the ${sampleCount} sampled points`);
    }
    for (let c = 0; c < 3; c++) {
      const v = pt[c];
      if (!Number.isFinite(v)) throw new Error('a target point contains a non-finite coordinate');
      d[idx * 3 + c] = v - baseFlat[idx * 3 + c];
      active.push(idx * 3 + c);
    }
    touched++;
  };
  if (!Array.isArray(targets)) throw new Error('targets must be an array');
  if (targets.length && targets[0] && typeof targets[0] === 'object' && !Array.isArray(targets[0]) && 'index' in targets[0]) {
    for (const t of targets) put(t.index, t.point);
  } else {
    if (targets.length !== sampleCount) {
      throw new Error(`a parallel targets array must have exactly ${sampleCount} entries (got ${targets.length})`);
    }
    for (let i = 0; i < targets.length; i++) if (targets[i]) put(i, targets[i]);
  }
  if (!touched) throw new Error('targets expressed no actual displacement');
  return { d, active };
}

// dot / norm restricted to the constrained components (see buildDesired).
function dotOn(a, b, active) {
  let s = 0;
  for (const i of active) s += a[i] * b[i];
  return s;
}
function normOn(a, active) {
  return Math.sqrt(dotOn(a, a, active));
}

function clampToSpec(v, spec) {
  let out = v, clamped = false;
  if (spec && Number.isFinite(spec.min) && out < spec.min) { out = spec.min; clamped = true; }
  if (spec && Number.isFinite(spec.max) && out > spec.max) { out = spec.max; clamped = true; }
  return { value: out, clamped };
}

// SOLVE for the param value that best moves the sampled geometry toward
// `targets`, by finite-differencing `evaluate` in that one param and taking
// the 1-D least-squares fit of the measured response against the requested
// displacement:
//
//   delta = (J . d) / (J . J)
//
// which is exactly the scalar minimizing |d - delta*J|. With `iterations` > 1
// this re-linearizes at the new value and repeats (Gauss-Newton), which
// matters when the param enters nonlinearly.
//
//   req = { evaluate, inputs, params, key, spec?, sample?, targets, options? }
//
// RETURNS on success:
//   { ok:true, value, delta, clamped,
//     sensitivity, alignment, residual, achieved,
//     step, stepMode, sampleCount, geomScale, iterations, verified:true }
//
//   alignment — |cos| of the angle between the param's measured response and
//     the requested displacement, in [-1, 1]. THIS is the quality signal that
//     answers "can this param explain what the cursor is asking for at all?".
//     1 means the param moves the geometry exactly the way the drag wants;
//     0 means the drag is orthogonal to everything this param can do.
//   residual — MEASURED, not predicted: the geometry is re-evaluated at the
//     proposed value and the leftover distance to the target is divided by
//     the distance that was there to begin with. 0 = the drag landed exactly;
//     1 = nothing was achieved; >1 = it got worse (reachable when a clamp
//     bites, or when a strongly nonlinear param overshoots at iterations=1).
//   achieved — 1 - residual, for callers that prefer the positive framing.
//
// RETURNS on refusal: { ok:false, reason, message, ...diagnostics }. Reason
// codes: 'discrete-param', 'non-numeric-param', 'degenerate-range',
// 'evaluate-failed', 'sample-failed', 'non-finite-geometry',
// 'unstable-sampling', 'non-smooth', 'quantized', 'insensitive',
// 'insensitive-at-target', 'no-target', 'orthogonal', 'bad-targets',
// 'non-finite-result'.
//
// Samples the caller gave no target for are UNCONSTRAINED and take no part
// in the fit — see buildDesired.
export function solveParamForDrag(req) {
  const opt = resolveOptions(req.options);
  const spec = req.spec || null;
  const baseValue = req.params && req.params[req.key] !== undefined
    ? req.params[req.key]
    : (spec ? spec.default : undefined);

  const declined = declaredRefusal(baseValue, spec, req.key);
  if (declined) return declined;

  let jac = jacobianAt(req, baseValue, opt, spec);
  if (!jac.ok) return jac;

  let desired, active;
  try {
    ({ d: desired, active } = buildDesired(req.targets, jac.baseFlat, jac.sampleCount));
  } catch (err) {
    return refuse('bad-targets', `targets could not be read: ${err.message}`);
  }
  const dNorm0 = normOn(desired, active);
  if (dNorm0 <= opt.motionTol * jac.geomScale) {
    return refuse('no-target', 'the requested displacement is zero (or below numerical resolution) — there is nothing to solve for');
  }

  // The fit — and the alignment quality signal — live entirely on the
  // CONSTRAINED samples. `sensitivity` below stays a whole-geometry figure;
  // these are deliberately different quantities.
  const jNormActive = normOn(jac.jacobian, active);
  if (jNormActive <= 0) {
    return refuse('insensitive-at-target', `'${req.key}' does move the geometry elsewhere (${jac.sensitivity.toPrecision(3)} mm per unit overall) but does not move the dragged sample(s) at all — this param is not the handle for this point`, {
      sensitivity: jac.sensitivity,
      step: jac.step,
      sampleCount: jac.sampleCount,
    });
  }
  const jd = dotOn(jac.jacobian, desired, active);
  const alignment = jd / (jNormActive * dNorm0);
  if (Math.abs(alignment) < opt.alignmentTol) {
    return refuse('orthogonal', `'${req.key}' does move the geometry (${jac.sensitivity.toPrecision(3)} mm per unit) but almost entirely ACROSS the requested drag (alignment ${alignment.toFixed(4)}) — solving it would produce a large parameter jump that does not go where the cursor asked`, {
      sensitivity: jac.sensitivity,
      alignment,
      step: jac.step,
      sampleCount: jac.sampleCount,
    });
  }

  // --- least-squares step(s) -------------------------------------------
  const iterations = Math.max(1, Math.floor(opt.iterations));
  let value = baseValue;
  let totalClamped = false;
  let used = 0;
  for (let it = 0; it < iterations; it++) {
    if (it > 0) {
      const next = jacobianAt(req, value, opt, spec);
      // A later iteration failing is not fatal — we already have a usable
      // answer from the earlier one. Stop refining, keep what we have.
      if (!next.ok) break;
      jac = next;
      try {
        ({ d: desired, active } = buildDesired(req.targets, jac.baseFlat, jac.sampleCount));
      } catch { break; }
    }
    const jj = dotOn(jac.jacobian, jac.jacobian, active);
    if (!(jj > 0)) break;
    const delta = dotOn(jac.jacobian, desired, active) / jj;
    if (!Number.isFinite(delta)) {
      return refuse('non-finite-result', `the least-squares fit for '${req.key}' produced a non-finite step`);
    }
    const c = clampToSpec(value + delta, spec);
    value = c.value;
    totalClamped = totalClamped || c.clamped;
    used = it + 1;
    if (Math.abs(delta) <= Math.abs(value) * 1e-12) break;
  }

  if (!Number.isFinite(value)) {
    return refuse('non-finite-result', `solving '${req.key}' produced a non-finite value`);
  }

  // --- MEASURED quality: re-evaluate at the answer and look --------------
  // A predicted residual (which for a single linear step is just
  // sqrt(1 - alignment^2)) would tell us nothing the alignment did not. The
  // point of re-evaluating is that it catches nonlinearity, clamping, and
  // any correspondence surprise for real, in the geometry itself.
  let residual = null;
  const finalProbe = probe(req, value);
  if (finalProbe.ok && finalProbe.flat.length === jac.baseFlat.length) {
    let leftover = null;
    try {
      leftover = buildDesired(req.targets, finalProbe.flat, jac.sampleCount).d;
    } catch { leftover = null; }
    if (leftover) residual = normOn(leftover, active) / dNorm0;
  }

  return {
    ok: true,
    key: req.key,
    baseValue,
    value,
    delta: value - baseValue,
    clamped: totalClamped,
    sensitivity: jac.sensitivity,
    alignment,
    residual,
    achieved: residual === null ? null : 1 - residual,
    verified: residual !== null,
    step: jac.step,
    stepMode: jac.stepMode,
    sampleCount: jac.sampleCount,
    geomScale: jac.geomScale,
    iterations: used,
    message: residual === null
      ? `Estimated '${req.key}' = ${value} (residual could not be verified)`
      : `Estimated '${req.key}' = ${value} — ${(100 * Math.max(0, 1 - residual)).toFixed(1)}% of the requested drag achieved`,
  };
}
