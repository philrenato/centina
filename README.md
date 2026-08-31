# Centina

A NURBS kernel in plain JavaScript. No dependencies, no build step, no renderer,
no DOM — just math over plain serializable data.

Centina is the geometry kernel of [Unreason3D](#the-app), a NURBS modeler built
to teach CAD to design students.

> **Pre-alpha.** `0.1.0-alpha.0`. The API is not frozen and the
> [capability map](#what-it-does-not-do) is not short. Read it before you build
> on this.

Not on npm. Clone this repository, or `npm install` it from its git URL.

## What it is

- **Curves and surfaces** — evaluation, derivatives, closest point, arc length,
  knot insertion, degree elevation, splitting, joining.
- **Construction** — line, arc, circle, ellipse, squircle; extrude, revolve,
  loft, sweep (1- and 2-rail), Gordon network surface, blend, offset.
- **Trimming and booleans** — surface-surface intersection, trimmed surfaces,
  B-rep sewing, union/intersect/difference on solids.
- **Fillets** — rolling-ball edge blends at constant and variable radius,
  chamfers, corner patches.
- **SubD** — Catmull-Clark, exact limit positions, and a ToNURBS bridge that
  converts a cage to bicubic patches.
- **Measurement** — mass properties, curvature, deviation across a shared edge.
- **Interchange** — `.3dm` (Rhino) read and write, from a **separate entry
  point**: `import { ... } from 'centina/io3dm'`.
- **Shading data**, which is not geometry and is here because the app it came
  from keeps it here: AgX tone mapping, HDRI environment recipes and their
  rasteriser, and a material pack. Numbers in, typed arrays out, like the rest —
  but nothing in this group renders anything, and none of it is what the package
  is for.

`io3dm` sits outside `kernel/`, and the boundary is a provenance one:
**`kernel/` is clean-room**, derived from Piegl & Tiller only, while `io3dm`
converts against rhino3dm/OpenNURBS's representation and is attributed
third-party infrastructure. It takes an awaited `rhino3dm()` instance as its
first argument rather than importing one, so Centina itself has **no runtime
dependencies** — supply rhino3dm if you want `.3dm`, and nothing is pulled in if
you do not.

The routines are hand-derived from Piegl & Tiller *The NURBS Book* and cited per
function.

## The data contract

There are no classes. A curve and a surface are plain objects, and control
points are `[x, y, z, w]` — homogeneous, weight last — everywhere without
exception.

```js
// a curve
{ degree: 2, knots: [0,0,0,1,1,2,2,3,3,4,4,4], ctrlPts: [[10,0,0,1], ...] }

// a surface
{ degU: 2, knotsU: [...], degV: 1, knotsV: [...], ctrlNet: [[[10,0,0,1], ...], ...] }
```

`ctrlNet` is indexed `[u][v]`, and everything is JSON-serializable — store it,
post it to a worker or diff it without a serializer.

⚠ **A curve's domain is not `[0,1]`.** It runs from `knots[0]` to
`knots[knots.length - 1]`, and for a circle built by `makeCircle` that is `0..4`
— one unit per quadrant. Ask the knot vector, never assume.

## A worked example

```js
import { makeCircle, extrude, curvePoint, surfacePoint } from 'centina';

// a circle of radius 10 in the XY plane
const circle = makeCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], 10);

// evaluate it — the domain runs 0..4, not 0..1
const [x, y, z] = curvePoint(circle, 1);        // quarter of the way round

// sweep it into a cylinder wall 20 tall
const wall = extrude(circle, [0, 0, 1], 20);
const p = surfacePoint(wall, 1, 0.5);           // (u, v) on that surface
```

Deep imports take one routine without pulling in the rest:

```js
import { curvePoint } from 'centina/curve.mjs';
```

## What it does **not** do

- **No STEP import.** It needs tolerant modeling, which is not built.
- **No tolerant modeling or geometry healing.** Input is expected to be clean.
- **No non-manifold topology.**
- **No face-face blend with hold lines.**
- **Booleans do not close on every placement.** A cylinder through a box
  intersects into one closed solid at all 24 seam angles measured, but unions
  and differences only at the 8 where the seam lines up with the box. Two
  spheres — both faces wrapped, both poled — close on all three operations. Two
  cylinder walls on crossed axes intersect and do nothing else.
- **The intersector's seed search misses cuts.** Sweeping a plane through a
  cylinder wall, 36 of 119 transversal crossings came back "no intersection
  found", at fixed parameters between seed nodes whatever the geometry. A
  boolean handed no curves then refuses for the wrong reason.
- **SubD exact limit evaluation is scoped to interior vertices** whose creases
  decay. Boundary vertices are excluded, which affects holed cages.
- **Stam's full arbitrary-(u,v) construction is not built.**
- **No global constraint solver.**

## When you get it wrong

Every public entry point refuses a malformed argument **by name** rather than
throwing a `TypeError` from three frames down:

```js
curvePoint({}, 0.5)
// curvePoint: the curve has no usable knot vector — expected { degree, knots, ctrlPts }

curvePoint(circle, 99)
// curvePoint: u = 99 is outside the curve's domain [0, 4] — a NURBS curve is not
// defined there, and evaluating anyway returns a plausible-looking point that means nothing
```

## Tests

```
npm test
```

From the repo root, in about a minute. A handful of interop tests need
`rhino3dm` (a dev dependency) and cover `.3dm` read/write; the rest is pure
kernel and needs nothing.

⚠ `node --test test/` does **not** work on Node ≥ 22 — positional arguments are
files and globs, not directories. Use `npm test`, or `node --test 'test/*.test.mjs'`.

## The app

Unreason3D is a single-file NURBS modeler — one `.html`, no build step. Centina
is what it is built on, and `kernel/` is the source of truth. The app carries
generated copies of some kernel modules inline because it has no build step;
those are artifacts, not forks.

## License

MIT — see [LICENSE](LICENSE). Names are covered separately and briefly in
[TRADEMARK.md](TRADEMARK.md); the command vocabulary is expressly released.
