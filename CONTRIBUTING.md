# Contributing to Centina

**Anything you contribute is MIT, same as the rest of the project.** That is the
whole legal part. There is no CLA to sign and nothing to email.

Fork it, commit, open a pull request. Unmerged pull requests are normal and are
not a judgment about your work — this is a teaching tool with a lot of opinions
baked into it, and "this is good but it isn't what this project is" is a real and
common answer. **A fork is a legitimate destination, not a failure state.**

## Running the tests

```
npm install     # one devDependency: rhino3dm, for the .3dm interop tests
npm test
```

About ninety seconds. Most of it is pure kernel and needs nothing installed.

⚠ `node --test test/` does **not** work on Node ≥ 22 — positional arguments are
files and globs, not directories. Use `npm test`, or
`node --test 'test/*.test.mjs'`.

## The data contract

There are no classes, and there is no builder. A curve and a surface are plain
objects, control points are `[x, y, z, w]` — homogeneous, weight last —
everywhere without exception, and `ctrlNet` is indexed `[u][v]`. The README has
the shapes. A change that introduces a class, a wrapper type or a non-serializable
field is a change to the contract, not an implementation detail; say so in the PR.

A curve's domain is `knots[0]` to `knots.at(-1)`, not `[0, 1]`. Code that assumes
otherwise passes its own tests and fails on a circle.

## `kernel/` is clean-room; `io3dm.mjs` is not

This boundary is the reason the two live in different places, and it is easy to
erase by accident.

- **Everything under `kernel/` is derived from Piegl & Tiller *The NURBS Book*
  only**, and cited per function. Do not paste, port or transcribe code from
  another geometry kernel into it — not OpenNURBS, not OCCT, not a GPL project,
  not a decompiled anything. If a routine needs a source, name the source.
- **`io3dm.mjs` sits at the repo root deliberately.** It converts against
  rhino3dm/OpenNURBS's own representation, so it is attributed third-party-adjacent
  infrastructure rather than clean-room. Do not move it under `kernel/` to tidy
  the packaging — that spends the distinction for nothing. It takes an awaited
  `rhino3dm()` instance as an argument rather than importing one, which is the
  only reason this package declares no runtime dependencies. Keep it that way.

**No wrapping a mature C++ kernel.** The geometry being hand-rolled from
published research is the point of the project, not an implementation detail. A
PR that replaces the kernel with a mature one is a fork, not a merge.

## Third-party code

`vendor/rhino3dm/` holds McNeel's openNURBS binding under its own MIT license,
built locally with a small patch — see `vendor/rhino3dm/NOTICE.txt`. That notice
travels with any copy you distribute.

## Names

The MIT license covers the code and says nothing about names. `TRADEMARK.md` says
the part MIT does not — briefly, and the command vocabulary is expressly released.
