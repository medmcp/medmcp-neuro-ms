#!/usr/bin/env bash
# Build a linux/aarch64 picsl-greedy wheel into /wheels. No-op on any other arch.
#
# Why this exists: picsl-greedy publishes no linux aarch64 wheel, so `uv pip install
# lst-ai` cannot resolve on arm64 and this whole stack is amd64-only. The blocker is
# not greedy but Kitware's VTK wheel-SDK, which upstream's prebuild.sh downloads and
# which exists for x86_64 only (404 for aarch64 at every published version).
#
# Rather than reimplement the build, this runs upstream's own prebuild.sh from the
# branch proposed in pyushkevich/greedy_python#6, which adds a source build of VTK
# restricted to the ten modules greedy needs. When that PR merges, point GREEDY_PY_REPO
# and GREEDY_PY_REF at upstream and delete nothing else; when official aarch64 wheels
# ship, delete this script and the two lines in the Dockerfile that call it.
set -euo pipefail

WHEELHOUSE="${WHEELHOUSE:-/wheels}"
GREEDY_PY_REPO="${GREEDY_PY_REPO:-https://github.com/jqmcginnis/greedy_python}"
# Pinned to a commit, not a branch: a branch would let the contents of this image
# change without any commit here, which is exactly what reproducibility for a medical
# imaging stack must not allow. Update deliberately.
GREEDY_PY_REF="${GREEDY_PY_REF:-22570fe9ee8945b2d1b7a9de5c4f078ab3097850}"

mkdir -p "${WHEELHOUSE}"

if [ "$(dpkg --print-architecture)" != "arm64" ]; then
    echo "build-greedy-wheel: $(dpkg --print-architecture), using the published wheel"
    exit 0
fi

echo "build-greedy-wheel: arm64 — building picsl-greedy from source"

# Run the whole build inside a throwaway venv. Upstream's prebuild.sh is written for
# the manylinux images cibuildwheel uses, and assumes three things Ubuntu does not
# provide: `python` on PATH (Ubuntu ships only python3), an interpreter not marked
# externally-managed (PEP 668 refuses the unqualified `pip install`), and a pip that
# can upgrade itself (Debian's has no RECORD file, so `pip install --upgrade pip`
# fails). A venv satisfies all three at once, and keeps their script byte-identical to
# the branch under review upstream rather than patched locally.
uv venv --seed /opt/greedy-build
export PATH=/opt/greedy-build/bin:${PATH}
export VIRTUAL_ENV=/opt/greedy-build

# ITK 5.2.1 (2021) uses uint8_t without including <cstdint>, which GCC 13 no longer
# pulls in transitively. Upstream CI never sees this because cibuildwheel builds inside
# manylinux2014 (GCC 10.2.1), but this image is Ubuntu 24.04. Forcing the header
# globally is the least invasive fix and keeps greedy's pinned ITK version.
export CXXFLAGS="${CXXFLAGS:-} -include cstdint"

# Peak RAM, not core count, is the binding constraint: ITK's template-heavy translation
# units OOM a small runner when one compile job runs per core.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
# `git clone -b` only accepts branches and tags, so fetch the pinned commit directly.
# GitHub allows fetching an arbitrary reachable SHA, which keeps this shallow.
mkdir -p "${work}/greedy_python"
cd "${work}/greedy_python"
git init -q .
git remote add origin "${GREEDY_PY_REPO}"
git fetch -q --depth 1 origin "${GREEDY_PY_REF}"
git checkout -q FETCH_HEAD
git submodule update --init --recursive --depth 1
echo "build-greedy-wheel: greedy_python at $(git rev-parse HEAD)"

# Builds Eigen, VTK (from source on aarch64) and ITK into be/install, then greedy's
# C++ libraries. The argument selects the platform branch inside the script.
bash .github/workflows/prebuild.sh ubuntu-22.04-arm

# Same environment cibuildwheel passes (see build_wheels.yml CIBW_ENVIRONMENT); the
# pybind11 module itself is built here, from greedy_python's own CMakeLists.
pip install --no-cache-dir scikit-build-core setuptools wheel
FETCH_DEPENDENCIES=OFF \
CMAKE_PREFIX_PATH=be/install \
VTK_DIR=be/install/vtk/vtk-9.3.1.data/headers/cmake \
    pip wheel . --no-deps -w "${WHEELHOUSE}" --no-build-isolation

# Fail loudly rather than silently producing an empty wheelhouse that the runtime
# stage would then skip, leaving lst-ai to fail much later and far less legibly.
ls -1 "${WHEELHOUSE}"/picsl_greedy-*.whl
echo "build-greedy-wheel: done"
