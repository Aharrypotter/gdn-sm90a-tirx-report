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
TIRX_COMMIT="12ce3721f7c62c5fbd911103ae373de689e58385"
TVM_FFI_COMMIT="3e8b2a860936fdacc85d65ec07bcebe150d23b5f"

usage() {
  echo "Usage: $0 --tvm-dir PATH --tvm-build-dir PATH --python-site-dir PATH \\"
  echo "  --tirx-dir PATH --python PATH"
}

fail() {
  echo "error: $*" >&2
  exit 1
}

TVM_DIR=""
TVM_BUILD_DIR=""
PYTHON_SITE_DIR=""
TIRX_DIR=""
PYTHON_BIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tvm-dir) TVM_DIR="$2"; shift 2 ;;
    --tvm-build-dir) TVM_BUILD_DIR="$2"; shift 2 ;;
    --python-site-dir) PYTHON_SITE_DIR="$2"; shift 2 ;;
    --tirx-dir) TIRX_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

for value in TVM_DIR TVM_BUILD_DIR PYTHON_SITE_DIR TIRX_DIR PYTHON_BIN; do
  [[ -n "${!value}" ]] || fail "missing required option for ${value}"
  [[ "${!value}" = /* ]] || fail "all paths must be absolute: ${!value}"
done
[[ -x "$PYTHON_BIN" ]] || fail "Python executable is not executable"
[[ "$(git -C "$TVM_DIR" rev-parse HEAD)" == "$TVM_COMMIT" ]] || fail "wrong TVM HEAD"
[[ "$(git -C "$TIRX_DIR" rev-parse HEAD)" == "$TIRX_COMMIT" ]] || fail "wrong TIRx HEAD"
[[ -z "$(git -C "$TVM_DIR" status --porcelain=v1 --untracked-files=all)" ]] \
  || fail "TVM worktree is dirty"
[[ -z "$(git -C "$TIRX_DIR" status --porcelain=v1 --untracked-files=all)" ]] \
  || fail "TIRx worktree is dirty"
[[ -f "$PYTHON_SITE_DIR/.gdn-sm90a-tvm-ffi-commit" ]] \
  || fail "Python site lacks the reproduction marker"
[[ "$(<"$PYTHON_SITE_DIR/.gdn-sm90a-tvm-ffi-commit")" == "$TVM_FFI_COMMIT" ]] \
  || fail "Python site has the wrong tvm-ffi marker"
[[ -d "$TVM_BUILD_DIR/lib" ]] || fail "TVM build library directory is missing"

export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TVM_DIR/python:$PYTHON_SITE_DIR:$TIRX_DIR"
export TVM_LIBRARY_PATH="$TVM_BUILD_DIR/lib"
export EXPECTED_TVM_PYTHON="$TVM_DIR/python"
export EXPECTED_FFI_SITE="$PYTHON_SITE_DIR"
export EXPECTED_TIRX_ROOT="$TIRX_DIR"

"$PYTHON_BIN" -s <<'PY'
import os
from pathlib import Path

import tirx_kernels
import tvm
import tvm_ffi

checks = (
    (Path(tvm.__file__).resolve(), Path(os.environ["EXPECTED_TVM_PYTHON"]).resolve(), "TVM"),
    (Path(tvm_ffi.__file__).resolve(), Path(os.environ["EXPECTED_FFI_SITE"]).resolve(), "tvm-ffi"),
    (
        Path(tirx_kernels.__file__).resolve(),
        Path(os.environ["EXPECTED_TIRX_ROOT"]).resolve(),
        "tirx-kernels",
    ),
)
for module_file, expected_root, label in checks:
    if not module_file.is_relative_to(expected_root):
        raise SystemExit(f"{label} import escaped its source lock: {module_file}")
PY

cd "$TIRX_DIR"
"$PYTHON_BIN" -s -m pytest \
  -p no:cacheprovider \
  -xvs \
  tests/gdn_sm90/test_auxiliary_semantics.py \
  tests/gdn_sm90/test_contract.py \
  tests/gdn_sm90/test_fixed_semantics.py \
  tests/gdn_sm90/test_mechanism_probes.py

echo "CPU-only GDN semantic/mechanism gate: PASS"
