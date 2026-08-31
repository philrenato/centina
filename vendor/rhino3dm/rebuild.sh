#!/bin/zsh
# Rebuild the vendored rhino3dm from source WITH this project's Brep authoring
# bindings, and install the result into vendor/rhino3dm/.
#
# The source tree is NOT in this repo — it is 668 MB and vendors draco and
# eigen as well as OpenNURBS. It is expected at $SRC below, which defaults to a
# sibling of this repository; set RHINO3DM_SRC to put it anywhere else. To get
# it:
#
#     git clone --branch 8.x --depth 1 --recurse-submodules \
#       --shallow-submodules https://github.com/mcneel/rhino3dm.git rhino3dm-src
#     cd rhino3dm-src && git apply <this-repo>/vendor/rhino3dm/brep_authoring.patch
#
# `brep_authoring.patch` is the only local change to that tree. The toolchain is
# emcc, cmake and python3 — no emsdk directory and no EMSDK env var needed.
#
# ⚠⚠ DO NOT USE `python3 script/build.py -p js`. It fails two ways, both paid
# for: without --overwrite it SILENTLY SKIPS the build and leaves the previous
# artifacts sitting there with their old timestamps, so a rebuild that never
# ran looks exactly like one that worked; with --overwrite it DELETES them
# first, so a compile error leaves nothing behind. Either way it EXITS 0 while
# printing errors in red, and it swallows the compiler output — six lines, none
# of them the actual error. `make` shows the real one.
set -e
DEST=${0:a:h}
SRC=${RHINO3DM_SRC:-${DEST}/../../../rhino3dm-src}
BUILD=$SRC/src/build/javascript

[ -d "$SRC" ] || { echo "REFUSE: no source tree at $SRC — see the clone command at the top of this script"; exit 1; }

# ⚠ CMake caches ABSOLUTE paths, so a tree that has been moved cannot build
# incrementally: CMakeCache.txt still names where it used to live. Regenerating
# is the whole fix and costs seconds; the compile after it is the slow part.
if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "CMAKE_HOME_DIRECTORY:INTERNAL=$SRC" "$BUILD/CMakeCache.txt"; then
  echo "cmake cache points somewhere else (moved tree) — regenerating"
  rm -rf "$BUILD"
fi
[ -d "$BUILD" ] || (cd "$SRC" && python3 script/setup.py -p js)

# ⚠ setup.py only CREATES draco's cmake project; it does not build it, and
# plain `make` for rhino3dm then dies with "No rule to make target
# draco_wasm/libdraco.a" because that path is a file DEPENDENCY and nothing
# produces it. build.py does this step and is otherwise unusable (see above),
# so it is done explicitly here.
# ⚠ Keyed on the LIBRARY, not on the build directory: a run that got as far as
# setup.py and then failed leaves the directory in place, and a guard testing
# the directory then skips the very step that was missing — which is how this
# script failed the same way twice.
[ -f "$BUILD/draco_wasm/libdraco.a" ] || (cd "$BUILD/draco_wasm" && emmake make draco_static)

(cd "$BUILD" && make)

# ⚠ CHECK THE WASM, NOT THE GLUE. embind names live in the binary: the glue was
# byte-identical at 127,787 bytes across a build that added ten methods, so
# grepping it reports an absence that is not real.
strings -a "$BUILD/rhino3dm.wasm" | grep -qx "newTrim" \
  || { echo "REFUSE: built wasm has no newTrim — the authoring patch is not in this build"; exit 1; }

# The worker imports an ES module; McNeel's output is UMD. One appended line is
# the whole difference.
{ cat "$BUILD/rhino3dm.js"; printf '\nexport default rhino3dm;\n'; } > "$DEST/rhino3dm.module.js"
cp "$BUILD/rhino3dm.wasm" "$DEST/rhino3dm.wasm"
# A plain CommonJS copy so NODE can load the same build the browser does —
# the ES module above takes emscripten's ENVIRONMENT_IS_NODE branch and calls a
# bare `require`, which an ES module does not have.
cp "$BUILD/rhino3dm.js" "$DEST/rhino3dm.cjs"

echo "installed:"
ls -la "$DEST"/rhino3dm.module.js "$DEST"/rhino3dm.wasm "$DEST"/rhino3dm.cjs
echo
echo "now re-run the .3dm tests: npm test"
