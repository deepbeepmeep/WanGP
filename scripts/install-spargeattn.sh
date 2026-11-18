#!/usr/bin/env bash
# Installs SpargeAttn (spas_sage_attn) from source on Linux.
#
# The recommended Python 3.11 / PyTorch 2.10 / CUDA 13 environment does not
# need this: it installs the prebuilt cp311 wheel. Build from source when your
# Python has no wheel (3.10, 3.12) or when you need another PyTorch build.
#
# A plain `pip install git+https://github.com/woct0rdho/SpargeAttn.git` breaks
# on CUDA 13 in two ways:
#
#   1. The compile stops with
#        cuda_fp8.hpp(...): error: identifier "__assert_fail" is undefined
#      CUDA 13's cuda_fp8.hpp calls assert() inside its inline conversion
#      helpers, assert() expands to __assert_fail(), and nvcc's device
#      frontend does not know that symbol. nvcc reads the NVCC_PREPEND_FLAGS
#      environment variable and prepends it to every invocation, so exporting
#      NVCC_PREPEND_FLAGS='-DNDEBUG' switches the assert macro off for the
#      whole build and the compile completes. This is an nvcc feature rather
#      than a torch one, so it works with any torch version and needs no edit
#      of SpargeAttn's own flags.
#
#   2. SpargeAttn generates its kernel instantiations with
#        os.system(f"python {py_file}")
#      a bare `python` whose exit status is ignored. When `python` does not
#      resolve to the interpreter that is building the package (a pyenv shim,
#      an unactivated or the wrong virtual environment), the generators run
#      against the wrong torch or fail outright, the wheel silently ships
#      without kernels, and `from . import _qattn_sm89` fails on import, so
#      SM89_ENABLED stays False. This script puts the build environment's bin
#      directory first on PATH and replaces that call with one that uses
#      sys.executable and fails loudly.
#
# Usage:
#   scripts/install-spargeattn.sh
#
# The build directory is re-cloned on every run (an existing directory is
# removed first), so the script is safe to re-run even after an interrupted
# build or an upstream layout change.
#
# Optional environment variables:
#   SPARGEATTN_REPO          git URL (default: https://github.com/woct0rdho/SpargeAttn.git)
#   SPARGEATTN_BUILD_DIR     where to clone (default: ./SpargeAttn-build)
#   SPARGEATTN_PYTHON        python interpreter / pip (default: python)
#   TORCH_CUDA_ARCH_LIST     e.g. "8.6" (RTX 30xx), "8.9" (RTX 40xx) or
#                        "12.0" (RTX 50xx) to only build for your GPU and
#                        speed up the build
#   NVCC_PREPEND_FLAGS       extra nvcc flags, -DNDEBUG is added to them
set -euo pipefail

REPO="${SPARGEATTN_REPO:-https://github.com/woct0rdho/SpargeAttn.git}"
BUILD_DIR="${SPARGEATTN_BUILD_DIR:-$(pwd)/SpargeAttn-build}"
PYTHON="${SPARGEATTN_PYTHON:-python}"

# Make the interpreter that builds the package also the one a bare `python`
# resolves to, so the kernel generators cannot pick up a different torch.
# sys.prefix is the virtual environment, whose bin directory holds it.
PY_BIN_DIR="$("$PYTHON" -c 'import os, sys; print(os.path.join(sys.prefix, "bin"))')"
export PATH="$PY_BIN_DIR:$PATH"

# nvcc prepends this variable to its own command line. Keep whatever the
# caller already set, and add -DNDEBUG only once.
case " ${NVCC_PREPEND_FLAGS:-} " in
    *" -DNDEBUG "*) ;;
    *) export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -DNDEBUG" ;;
esac

"$PYTHON" -m pip install ninja wheel packaging

# Fresh clone every time: a leftover (possibly patched or half-built)
# directory would make `git clone` fail, and stale sources could mask
# upstream layout changes the patch below relies on.
rm -rf "$BUILD_DIR"
git clone --depth 1 "$REPO" "$BUILD_DIR"

# Run the autogen scripts with the interpreter that is building the package
# and stop on their first error: os.system reports nothing, and a wheel
# without generated kernels imports fine and fails later at runtime.
"$PYTHON" - "$BUILD_DIR/setup.py" <<'EOF'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
call = 'os.system(f"python {py_file}")'

if call not in source:
    sys.exit(
        "error: SpargeAttn's setup.py no longer contains " + call + "\n"
        "       Check how the kernel generators are invoked now before\n"
        "       installing, otherwise a broken wheel installs silently."
    )

source = source.replace(call, "subprocess.check_call([sys.executable, str(py_file)])")
missing = [line for line in ("import sys", "import subprocess")
           if not re.search("^" + re.escape(line) + "$", source, re.M)]
if missing:
    source = "\n".join(missing) + "\n" + source

path.write_text(source, encoding="utf-8")
print("patched: kernel generators run with sys.executable and fail loudly")
EOF

echo "[install-spargeattn] NVCC_PREPEND_FLAGS='$NVCC_PREPEND_FLAGS'"
"$PYTHON" -m pip install --no-build-isolation "$BUILD_DIR"

# The install can succeed with the kernels missing, so check for them.
"$PYTHON" - <<'EOF'
import spas_sage_attn.core as core

print("SpargeAttn kernels: sm80=%s sm89=%s sm90=%s"
      % (core.SM80_ENABLED, core.SM89_ENABLED, core.SM90_ENABLED))
if not (core.SM80_ENABLED or core.SM89_ENABLED or core.SM90_ENABLED):
    raise SystemExit(
        "error: SpargeAttn installed but enabled no kernel.\n"
        "       The kernel generators did not run for your GPU family;\n"
        "       re-run with TORCH_CUDA_ARCH_LIST set to your architecture."
    )
EOF
