#!/bin/zsh
# Rebuild the vendored rhino3dm from source with this project's Brep authoring
# bindings, and install the result into vendor/rhino3dm/.
#
# The source tree is not in this repo — it is 668 MB and vendors draco and
# eigen as well as OpenNURBS. It is expected at $SRC below, which defaults to a
# sibling of this repository; set RHINO3DM_SRC to put it anywhere else. To get
# it:
#
#     git clone --branch 8.x --depth 1 --recurse-submodules \
#       --shallow-submodules https://github.com/mcneel/rhino3dm.git rhino3dm-src
#     cd rhino3dm-src && git apply <this-repo>/vendor/rhino3dm/brep_authoring.patch
#     git apply <this-repo>/vendor/rhino3dm/subd_authoring.patch
#
# `brep_authoring.patch` and `subd_authoring.patch` (SubD.addVertex/addFace4/
# addFace/setEdgeCrease/setEdgeSharpness/updateUnsetTagsAndSectorCoefficients/
# isValid — a SubD can be built and written, not only read) are the only local changes to that tree. The toolchain is emcc,
# cmake and python3 — no emsdk directory and no EMSDK env var needed.
# Homebrew's emscripten runs on whichever python3 is first on PATH and
# refuses 3.9 (the Xcode one): put /opt/homebrew/opt/python@3.14/bin first
# and set EMSDK_PYTHON to that python3.14.
#
# `python3 script/build.py -p js` is not used. Without --overwrite it skips the
# build silently and leaves the previous artifacts with their old timestamps,
# so a rebuild that never ran looks like one that worked; with --overwrite it
# deletes them first, so a compile error leaves nothing behind. Either way it
# exits 0 and hides the compiler output. `make` shows the actual error.
set -e
DEST=${0:a:h}
SRC=${RHINO3DM_SRC:-${DEST}/../../../rhino3dm-src}
BUILD=$SRC/src/build/javascript

[ -d "$SRC" ] || { echo "REFUSE: no source tree at $SRC — see the clone command at the top of this script"; exit 1; }

# CMake caches absolute paths, so a moved tree cannot build incrementally:
# CMakeCache.txt names the old location. Regenerating costs seconds; the
# compile after it is the slow part.
if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "CMAKE_HOME_DIRECTORY:INTERNAL=$SRC" "$BUILD/CMakeCache.txt"; then
  echo "cmake cache points somewhere else (moved tree) — regenerating"
  rm -rf "$BUILD"
fi
[ -d "$BUILD" ] || (cd "$SRC" && python3 script/setup.py -p js)

# setup.py only creates draco's cmake project; it does not build it, and
# plain `make` for rhino3dm then fails with "No rule to make target
# draco_wasm/libdraco.a" because that path is a file dependency and nothing
# produces it. build.py does this step and is otherwise unusable (see above),
# so it is done explicitly here.
# Keyed on the library, not on the build directory: a run that got as far as
# setup.py and then failed leaves the directory in place, and a guard testing
# the directory would skip the step that was missing.
[ -f "$BUILD/draco_wasm/libdraco.a" ] || (cd "$BUILD/draco_wasm" && emmake make draco_static)

(cd "$BUILD" && make)

# Check the wasm, not the glue. embind names live in the binary; the glue does
# not change when methods are added, so grepping it cannot see them.
strings -a "$BUILD/rhino3dm.wasm" | grep -qx "newTrim" \
  || { echo "REFUSE: built wasm has no newTrim — the brep authoring patch is not in this build"; exit 1; }
strings -a "$BUILD/rhino3dm.wasm" | grep -qx "addFace4" \
  || { echo "REFUSE: built wasm has no addFace4 — the subd authoring patch is not in this build"; exit 1; }

# The worker imports an ES module; McNeel's output is UMD. One appended line is
# the whole difference.
{ cat "$BUILD/rhino3dm.js"; printf '\nexport default rhino3dm;\n'; } > "$DEST/rhino3dm.module.js"
cp "$BUILD/rhino3dm.wasm" "$DEST/rhino3dm.wasm"
# A plain CommonJS copy so Node can load the same build the browser does —
# the ES module above takes emscripten's ENVIRONMENT_IS_NODE branch and calls a
# bare `require`, which an ES module does not have.
cp "$BUILD/rhino3dm.js" "$DEST/rhino3dm.cjs"

echo "installed:"
ls -la "$DEST"/rhino3dm.module.js "$DEST"/rhino3dm.wasm "$DEST"/rhino3dm.cjs
echo
echo "now re-run the .3dm tests: npm test"
