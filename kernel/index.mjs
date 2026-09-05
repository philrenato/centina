// CENTINA — the public surface of the kernel.
//
// Every module under `kernel/` is re-exported here, so `import { ... } from
// "centina"` reaches the whole kernel. Individual modules are ALSO importable
// directly — `import { curvePoint } from "centina/curve.mjs"` — and that is a
// supported way to use this, not a workaround: it is how you take one routine
// without pulling in the rest.
//
// ⚠⚠ THIS FILE IS COMPLETE ON PURPOSE, AND A GATE HOLDS IT THAT WAY. It used to
// re-export 13 of the 74 modules, because it was written to be hand-pasted into
// the single-file app rather than to be an API — which left `booleanSolids`,
// `loft`, `intersectSurfaces`, `fitCurveToPoints` and exact SubD limit
// evaluation unreachable by name from the package that exists to provide them.
// A hand-kept list of modules is exactly the thing this project has watched go
// stale over and over, so a gate asserts that every
// `kernel/*.mjs` is reachable from here and fails when one is not.
//
// ⚠ TWO NAMES COLLIDE ACROSS MODULES, and an ambiguous `export *` drops them
// SILENTLY rather than erroring — so both would simply vanish from the API with
// nothing said. They are re-exported explicitly under qualified names below,
// and the bare names are deliberately not exported at all: a bare `mergeFaces`
// would have to mean one of two unrelated things.

export * from './arrangement.mjs';
export * from './basis.mjs';
export * from './blend.mjs';
export * from './boolean.mjs';
export * from './booleansew.mjs';
export * from './brep.mjs';
export * from './brepbuild.mjs';
export * from './brepfit.mjs';
export * from './breprecord.mjs';
export * from './cage.mjs';
export * from './classify.mjs';
export * from './conform.mjs';
export * from './cornerblend.mjs';
export * from './curvature.mjs';
export * from './curve.mjs';
export * from './curvecurve.mjs';
export * from './curvegen.mjs';
export * from './curveonsurface.mjs';
export * from './curvesplit.mjs';
export * from './curvesurface.mjs';
export * from './extend.mjs';
export * from './facesplit.mjs';
export * from './fair.mjs';
export * from './falloff.mjs';
export * from './fiber.mjs';
export * from './field.mjs';
export * from './fillet.mjs';
export * from './fitcurve.mjs';
export * from './flatten.mjs';
export * from './flipseam.mjs';
export * from './interpolate.mjs';
export * from './isocurve.mjs';
export * from './knots.mjs';
export * from './loft.mjs';
export * from './marchingsquares.mjs';
export * from './massprops.mjs';
export * from './matchedge.mjs';
export * from './noise.mjs';
export * from './offset.mjs';
export * from './offsetcurve.mjs';
export * from './offsetrevolve.mjs';
export * from './patch.mjs';
export * from './pointedit.mjs';
export * from './primitives.mjs';
export * from './project.mjs';
export * from './refit.mjs';
export * from './seam.mjs';
export * from './selfintersect.mjs';
export * from './sensitivity.mjs';
export * from './simplify.mjs';
export * from './split.mjs';
export * from './splitfeatures.mjs';
export * from './ssi.mjs';
export * from './subd.mjs';
export * from './subdconvert.mjs';
export * from './subdedit.mjs';
export * from './subdlimit.mjs';
export * from './subdmatch.mjs';
export * from './subdnetwork.mjs';
export * from './subdpipe.mjs';
export * from './subdprimitives.mjs';
export * from './subdradial.mjs';
export * from './subdreflect.mjs';
export * from './subdselect.mjs';
export * from './subdsimplify.mjs';
export * from './surface.mjs';
export * from './surfaceknots.mjs';
export * from './sweep.mjs';
export * from './tangentpatch.mjs';
export * from './tessellate.mjs';
export * from './text.mjs';
export * from './transform.mjs';
export * from './trim.mjs';
export * from './trimfit.mjs';
export * from './trimtess.mjs';
export * from './varradius.mjs';
export * from './bvh.mjs';
export * from './rendrepack.mjs';
export * from './agx.mjs';
export * from './envmap.mjs';
export * from './envpack.mjs';
export * from './envrecipes.mjs';
export * from './puff.mjs';
export * from './puffgridcap.mjs';
export * from './puffoutline.mjs';
export * from './puffspine.mjs';
export * from './solidwrap.mjs';
export * from './blobcurve.mjs';
export * from './spine.mjs';
export * from './vec3.mjs';
export * from './wave.mjs';

// The two disambiguations. Import the module directly if you want the bare name.
export { mergeFaces as mergeArrangementFaces } from './arrangement.mjs';
export { mergeFaces as mergeSubDFaces } from './subdedit.mjs';
export { hash01 as hashCurveGen01 } from './curvegen.mjs';
export { hash01 as hashTessellate01 } from './tessellate.mjs';
