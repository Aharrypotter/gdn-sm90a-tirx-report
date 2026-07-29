#!/usr/bin/env bash
# Copyright 2026 Aharrypotter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

TVM_COMMIT="acb1312de80b39340e09b0aaad818ff029e745d6"
TVM_FFI_COMMIT="3e8b2a860936fdacc85d65ec07bcebe150d23b5f"

usage() {
  echo "Usage: $0 --tvm-dir PATH --build-dir PATH --python-site-dir PATH \\"
  echo "  --python PATH --cuda-root PATH [--jobs N]"
}

fail() {
  echo "error: $*" >&2
  exit 1
}

TVM_DIR=""
BUILD_DIR=""
PYTHON_SITE_DIR=""
PYTHON_BIN=""
CUDA_ROOT=""
JOBS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tvm-dir) TVM_DIR="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --python-site-dir) PYTHON_SITE_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --cuda-root) CUDA_ROOT="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

for value in TVM_DIR BUILD_DIR PYTHON_SITE_DIR PYTHON_BIN CUDA_ROOT; do
  [[ -n "${!value}" ]] || fail "missing required option for ${value}"
done

for path in "$TVM_DIR" "$BUILD_DIR" "$PYTHON_SITE_DIR" "$PYTHON_BIN" "$CUDA_ROOT"; do
  [[ "$path" = /* ]] || fail "all paths must be absolute: $path"
done
[[ -d "$TVM_DIR/.git" || -f "$TVM_DIR/.git" ]] || fail "TVM path is not a Git worktree"
[[ -x "$PYTHON_BIN" ]] || fail "Python executable is not executable"
[[ -d "$CUDA_ROOT" ]] || fail "CUDA root is not a directory"
command -v cmake >/dev/null || fail "cmake is unavailable"
command -v ninja >/dev/null || fail "ninja is unavailable"

mapfile -t RESOLVED_PATHS < <(
  "$PYTHON_BIN" - "$TVM_DIR" "$BUILD_DIR" "$PYTHON_SITE_DIR" <<'PY'
from pathlib import Path
import sys

for value in sys.argv[1:]:
    print(Path(value).resolve(strict=False))
PY
)
TVM_DIR="${RESOLVED_PATHS[0]}"
BUILD_DIR="${RESOLVED_PATHS[1]}"
PYTHON_SITE_DIR="${RESOLVED_PATHS[2]}"

[[ "$BUILD_DIR" != "/" && "$PYTHON_SITE_DIR" != "/" ]] || fail "refusing filesystem-root output"
case "$BUILD_DIR/" in "$TVM_DIR/"*) fail "build directory must be outside the TVM tree" ;; esac
case "$PYTHON_SITE_DIR/" in "$TVM_DIR/"*) fail "Python site directory must be outside TVM" ;; esac
[[ "$BUILD_DIR" != "$PYTHON_SITE_DIR" ]] || fail "build and Python site directories must differ"

ACTUAL_HEAD="$(git -C "$TVM_DIR" rev-parse HEAD)"
[[ "$ACTUAL_HEAD" == "$TVM_COMMIT" ]] || fail "TVM HEAD is $ACTUAL_HEAD, expected $TVM_COMMIT"
[[ -z "$(git -C "$TVM_DIR" status --porcelain=v1 --untracked-files=all)" ]] \
  || fail "TVM worktree must be clean before build"

PINNED_FFI="$(git -C "$TVM_DIR" rev-parse HEAD:3rdparty/tvm-ffi)"
[[ "$PINNED_FFI" == "$TVM_FFI_COMMIT" ]] || fail "unexpected tvm-ffi Gitlink $PINNED_FFI"
FFI_STATUS="$(git -C "$TVM_DIR" submodule status --recursive 3rdparty/tvm-ffi)"
[[ -n "$FFI_STATUS" ]] || fail "tvm-ffi submodule is not available"
if grep -Eq '^[+-U]' <<<"$FFI_STATUS"; then
  fail "tvm-ffi submodule is uninitialized or does not match the pinned Gitlink"
fi
ACTUAL_FFI="$(git -C "$TVM_DIR/3rdparty/tvm-ffi" rev-parse HEAD)"
[[ "$ACTUAL_FFI" == "$TVM_FFI_COMMIT" ]] || fail "tvm-ffi checkout is $ACTUAL_FFI"

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  EXPECTED_HOME="CMAKE_HOME_DIRECTORY:INTERNAL=$TVM_DIR"
  grep -Fxq "$EXPECTED_HOME" "$BUILD_DIR/CMakeCache.txt" \
    || fail "existing CMake cache belongs to another source tree"
fi

if [[ -n "$JOBS" ]]; then
  [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "--jobs must be a positive integer"
else
  JOBS="$("$PYTHON_BIN" - <<'PY'
import os
print(max(1, os.cpu_count() or 1))
PY
)"
fi

mkdir -p "$BUILD_DIR"
cmake \
  -S "$TVM_DIR" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DUSE_CUDA="$CUDA_ROOT" \
  -DUSE_LLVM=OFF \
  -DUSE_OPENMP=OFF \
  -DUSE_RPC=ON
cmake --build "$BUILD_DIR" --parallel "$JOBS"

MARKER="$PYTHON_SITE_DIR/.gdn-sm90a-tvm-ffi-commit"
if [[ -d "$PYTHON_SITE_DIR" ]] && [[ -n "$(find "$PYTHON_SITE_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  [[ -f "$MARKER" ]] || fail "non-empty Python site lacks the reproduction marker"
  [[ "$(<"$MARKER")" == "$TVM_FFI_COMMIT" ]] || fail "Python site marker has wrong tvm-ffi"
else
  mkdir -p "$PYTHON_SITE_DIR"
  "$PYTHON_BIN" -m pip install \
    --no-build-isolation \
    --no-deps \
    --target "$PYTHON_SITE_DIR" \
    "$TVM_DIR/3rdparty/tvm-ffi"
  printf '%s\n' "$TVM_FFI_COMMIT" >"$MARKER"
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TVM_DIR/python:$PYTHON_SITE_DIR"
export TVM_LIBRARY_PATH="$BUILD_DIR/lib"
export EXPECTED_TVM_PYTHON="$TVM_DIR/python"
export EXPECTED_FFI_SITE="$PYTHON_SITE_DIR"
"$PYTHON_BIN" -s <<'PY'
import os
from pathlib import Path

import tvm
import tvm_ffi

tvm_file = Path(tvm.__file__).resolve()
ffi_file = Path(tvm_ffi.__file__).resolve()
tvm_root = Path(os.environ["EXPECTED_TVM_PYTHON"]).resolve()
ffi_root = Path(os.environ["EXPECTED_FFI_SITE"]).resolve()
if not tvm_file.is_relative_to(tvm_root):
    raise SystemExit(f"TVM import escaped the requested worktree: {tvm_file}")
if not ffi_file.is_relative_to(ffi_root):
    raise SystemExit(f"tvm-ffi import escaped the requested target: {ffi_file}")
target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
if target.arch != "sm_90a":
    raise SystemExit(f"unexpected target architecture: {target.arch}")
print(f"TVM import: {tvm_file}")
print(f"tvm-ffi import: {ffi_file}")
print(f"TIR target: {target}")
PY

echo "TVM build and non-editable tvm-ffi import gate: PASS"
