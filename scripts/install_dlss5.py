#!/usr/bin/env python3
"""WanGP optional DLSS 5 installer for Linux.

Port of scripts/install_dlss5.ps1 for non-Windows platforms. The Windows
installer stages Windows-only binaries (D3D12 workers, ReShade/RenoDX hooks
and community DLSSNR builds); this installer instead fetches the equivalent
official resources that exist for Linux: the public NVIDIA DLSS SDK runtimes
(Super Resolution and Frame Generation, CUDA/Vulkan API) and the NVSDK NGX
API headers, pinned to a repository tag and SHA-256 verified.

All files are downloaded from NVIDIA's official public repository
(https://github.com/NVIDIA/DLSS, tag v310.7.0). No community mirrors are
used. The files are governed by the NVIDIA RTX SDKs License: no standalone
redistribution, NVIDIA-GPU-only use (see the installed
LICENSE-NVIDIA-DLSS.txt).

Installed layout under <WanGP root>/dlss5:

  dlss/libnvidia-ngx-dlss.so.310.7.0      DLSS Super Resolution runtime
  dlssg/libnvidia-ngx-dlssg.so.310.7.0    DLSS Frame Generation runtime
  sdk/include/...                          NVSDK NGX API headers
  sdk/lib/libnvsdk_ngx.a                   static import library
  sdk/lib/libnvidia-ngx-dlssd.so.310.7.0  legacy DLSS Dynamic feature library

Notes:
- DLSS 5 Neural Rendering has no Linux runtime (DLSSNR/ReShade/RenoDX are
  Windows-only) and stays disabled on Linux.
- Running DLSS Frame Generation additionally requires a Linux build of the
  WanGP DLSSG worker (MIT source:
  https://github.com/DeepBeepMeep/dlss5-visual-enhancer,
  native/WanGP-Adapter/dlssg_worker.cpp); place the resulting binary at
  dlss5/dlssg/dlssg-worker.

Usage (via scripts/install_dlss5.sh, which selects the venv interpreter):
  scripts/install_dlss5.sh [--force] [--wan-gp-root DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

USER_AGENT = "WanGP-DLSS5-Installer"
SDK_TAG = "v310.7.0"
SDK_URL = f"https://raw.githubusercontent.com/NVIDIA/DLSS/{SDK_TAG}"
RUNTIME_SO_SUFFIX = ".310.7.0"

# (repo path, destination relative to <root>/dlss5, pinned SHA-256). The
# license text is downloaded once and copied into the runtime directories.
SDK_DOWNLOADS = (
    ("lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.310.7.0", "dlss/libnvidia-ngx-dlss.so.310.7.0",
     "FC19B68CEFB4218E0954FAB812C782E0AA4D30526E0151C796E8887E0ECA3ACB"),
    ("lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so.310.7.0", "sdk/lib/libnvidia-ngx-dlssd.so.310.7.0",
     "EFD465933BF9A40B65F3C6D61AA079F4A4B188004E9C2B432E3375783B0029F3"),
    ("lib/Linux_x86_64/rel/libnvidia-ngx-dlssg.so.310.7.0", "dlssg/libnvidia-ngx-dlssg.so.310.7.0",
     "676CFEACE1BF675A281CF234DF619F24CEF16A1259F36119EBDF01138468A057"),
    ("lib/Linux_x86_64/libnvsdk_ngx.a", "sdk/lib/libnvsdk_ngx.a",
     "DAE18DCE6FDBAB45F7304B14901C93C41073D7CC31D555D6CB1FA145958937A6"),
    ("LICENSE.txt", "LICENSE-NVIDIA-DLSS.txt",
     "21B5DAEC892B12BEA692E66BC8FE45CF5902CCAF3A7B831E78050D8859881C37"),
    ("include/nvsdk_ngx.h", "sdk/include/nvsdk_ngx.h",
     "F6014A256F9D75CCEC1278AC6E23D596B398A76CC3960048CA1A274B378B1989"),
    ("include/nvsdk_ngx_defs.h", "sdk/include/nvsdk_ngx_defs.h",
     "EA23F33497CD274860D1C25A97644FCE807DCB0037C594547203343103FAD03E"),
    ("include/nvsdk_ngx_defs_dlssd.h", "sdk/include/nvsdk_ngx_defs_dlssd.h",
     "D2FDE340DB2189C89BCE093BC1EDD7B3579DF48DECAC329218A98A9C5FD46018"),
    ("include/nvsdk_ngx_defs_dlssg.h", "sdk/include/nvsdk_ngx_defs_dlssg.h",
     "5E76E5CF0397F0B093887D0392B427A6B3E3F722CEC5F5A4795357EDEB6DE4BA"),
    ("include/nvsdk_ngx_defs_vk.h", "sdk/include/nvsdk_ngx_defs_vk.h",
     "0A24D0861ACE7D6B9362A67B7F08BEA1B33EA123C5CE730B19133DE8A891D031"),
("include/nvsdk_ngx_helpers.h", "sdk/include/nvsdk_ngx_helpers.h",
     "2D5661F8B5AB55E1223E485F24146274D48077E09051873826B653D4384FE7D8"),
    ("include/nvsdk_ngx_helpers_dlssd.h", "sdk/include/nvsdk_ngx_helpers_dlssd.h",
     "6DF9D02C6F47FEEE3AFC8D12340A9F554A7DDCE4DDACB1B6A13A42B26C2C6293"),
    ("include/nvsdk_ngx_helpers_dlssd_cuda.h", "sdk/include/nvsdk_ngx_helpers_dlssd_cuda.h",
     "7E35CDE214D905346A487B9A28C05D35E1C55B480E237FE3DE69580A7E332730"),
    ("include/nvsdk_ngx_helpers_dlssd_vk.h", "sdk/include/nvsdk_ngx_helpers_dlssd_vk.h",
     "11DF8FBDA532A97FE867C8EF2B40058B49B7898183710F3468E55B1EEDB8F40B"),
    ("include/nvsdk_ngx_helpers_dlssg.h", "sdk/include/nvsdk_ngx_helpers_dlssg.h",
     "85894E5F44C3D1ADD74B3997C80F8D6A25DCBEB4DF5D092138E71DA5629B60E8"),
    ("include/nvsdk_ngx_helpers_dlssg_vk.h", "sdk/include/nvsdk_ngx_helpers_dlssg_vk.h",
     "B1B77DB963C6F1095EBC4A33E4DC06897C7EBECB5B582DC68B332E5EC85C01ED"),
    ("include/nvsdk_ngx_helpers_vk.h", "sdk/include/nvsdk_ngx_helpers_vk.h",
     "C192BFF72138F12F770DB48E12C1D8F712DFC81EA0E8E055EBB27D8DD1B31623"),
    ("include/nvsdk_ngx_params.h", "sdk/include/nvsdk_ngx_params.h",
     "943BC8CC5CDAE03B6303016FBAD3183636F2335AE27A2D18776798C3B4EFABBC"),
    ("include/nvsdk_ngx_params_dlssd.h", "sdk/include/nvsdk_ngx_params_dlssd.h",
     "D41E3BA8F1D72583B55273092C99BF426DC4A0D46DE827229E25D5B1325A370E"),
    ("include/nvsdk_ngx_params_dlssg.h", "sdk/include/nvsdk_ngx_params_dlssg.h",
     "C24FA5AA68FFA2808C828B75CFA32843F1606F80A145F84055AC450D7E6C74AD"),
    ("include/nvsdk_ngx_vk.h", "sdk/include/nvsdk_ngx_vk.h",
     "2D364CE7132881EB669E9498FD570D74CBA563B1FBEBC82C235F8E1AD2DD8B6D"),
)

LICENSE_COPY_DESTINATIONS = (
    "LICENSE-NVIDIA-DLSS.txt",
    "dlss/LICENSE-NVIDIA-DLSS.txt",
    "dlssg/LICENSE-NVIDIA-DLSS.txt",
    "sdk/LICENSE-NVIDIA-DLSS.txt",
)

README_LINUX = f"""\
DLSS 5 runtimes for Linux
=========================

Installed by scripts/install_dlss5.sh (Python port of the Windows
install_dlss5.ps1 for Linux hosts) from the official public NVIDIA DLSS SDK
(https://github.com/NVIDIA/DLSS, tag {SDK_TAG}). Every file is SHA-256
verified; no community mirrors are used.

Files
-----
  dlss/libnvidia-ngx-dlss.so{RUNTIME_SO_SUFFIX}      DLSS Super Resolution runtime (CUDA/Vulkan API)
  dlssg/libnvidia-ngx-dlssg.so{RUNTIME_SO_SUFFIX}    DLSS Frame Generation runtime (CUDA/Vulkan API)
  sdk/include/...                                    NVSDK NGX API headers
  sdk/lib/libnvsdk_ngx.a                             static import library
  sdk/lib/libnvidia-ngx-dlssd.so{RUNTIME_SO_SUFFIX}  legacy DLSS Dynamic feature library
  LICENSE-NVIDIA-DLSS.txt                            NVIDIA RTX SDKs License

Licensing
---------
These files are governed by the NVIDIA RTX SDKs License: no standalone
redistribution, NVIDIA-GPU-only use, and no implication of NVIDIA
sponsorship.

What this does and does not enable on Linux
-------------------------------------------
- DLSS 5 Neural Rendering (x1-x3): Windows-only. The DLSSNR runtime,
  ReShade and RenoDX have no Linux builds, so this mode remains disabled on
  Linux regardless of the files above.
- DLSS Frame Generation (x2-x4, x5/x6 on RTX 50): the official runtime is
  installed above, but WanGP's dlssg worker is a Windows D3D12 binary. Build
  a Linux (Vulkan or CUDA) worker from the MIT-licensed source
  (https://github.com/DeepBeepMeep/dlss5-visual-enhancer,
  native/WanGP-Adapter/dlssg_worker.cpp) and place the binary at
  dlss5/dlssg/dlssg-worker to enable it.
"""
def resolve_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        # scripts/install_dlss5.py -> WanGP root
        root = Path(__file__).resolve().parents[1]
    if not (root / "run.sh").is_file() and not (root / "src").is_dir():
        print(f"ERROR: no WanGP checkout found at {root}")
        print("       pass --wan-gp-root DIR pointing at the WanGP repository.")
        raise SystemExit(1)
    return root


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def download_file(url: str, destination: Path, expected_sha256: str, retries: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=900) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            actual = sha256_of(temporary)
            if actual != expected_sha256:
                last_error = f"SHA-256 mismatch: expected {expected_sha256}, got {actual}"
            else:
                temporary.replace(destination)
                return
        except Exception as error:  # noqa: BLE001 - reported to the user, retried
            last_error = str(error)
        print(f"  download attempt {attempt}/{retries} failed for {destination.name}: {last_error}")
        time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"could not download {url}: {last_error}")


def print_layout(dlss5_dir: Path) -> None:
    print(f"DLSS 5 Linux layout under {dlss5_dir}")
    print("-" * 60)
    for path in sorted(dlss5_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(dlss5_dir)}  ({path.stat().st_size:,} bytes)")
    print("-" * 60)
def main() -> int:
    parser = argparse.ArgumentParser(description="WanGP DLSS 5 installer (Linux)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite runtimes whose SHA-256 differs from the pin")
    parser.add_argument("--wan-gp-root", default=None,
                        help="WanGP checkout to install into (default: repository root)")
    args = parser.parse_args()

    root = resolve_root(args.wan_gp_root)
    dlss5_dir = root / "dlss5"
    print(f"WanGP root: {root}")
    print(f"Installing official NVIDIA DLSS SDK {SDK_TAG} Linux runtimes from raw.githubusercontent.com")
    print("Licensing: NVIDIA RTX SDKs License (NVIDIA-GPU-only use, no standalone redistribution).")
    print()

    failures = []
    with tempfile.TemporaryDirectory(prefix="dlss5-linux-") as stage:
        stage_dir = Path(stage)
        for repo_path, destination_rel, expected_sha256 in SDK_DOWNLOADS:
            destination = stage_dir / destination_rel
            print(f"downloading  {repo_path}")
            try:
                download_file(f"{SDK_URL}/{repo_path}", destination, expected_sha256)
                print(f"  sha256 ok   {sha256_of(destination)[:16]}...")
            except RuntimeError as error:
                failures.append(str(error))
                print(f"  FAILED      {error}")

        if failures:
            print(f"\nFAILED: {len(failures)} file(s) could not be downloaded or verified.")
            print("No files were installed. Check network access to raw.githubusercontent.com and retry.")
            return 1

        if dlss5_dir.exists() and not args.force:
            existing = [p for p in dlss5_dir.rglob("*") if p.is_file()
                        and p.name not in ("README-LINUX.txt", "LICENSE-NVIDIA-DLSS.txt")]
            if existing:
                print(f"NOTE: {dlss5_dir} already contains {len(existing)} file(s); "
                      "files matching the pin are kept, others need --force.")
        print("\nstaging -> install")
        for repo_path, destination_rel, _ in SDK_DOWNLOADS:
            source = stage_dir / destination_rel
            destination = dlss5_dir / destination_rel
            if destination.exists():
                if sha256_of(destination) == sha256_of(source):
                    print(f"  unchanged   {destination_rel}")
                    continue
                if not args.force:
                    print(f"  skipped     {destination_rel} (hash differs from pin; use --force)")
                    continue
                print(f"  replaced  {destination_rel}")
            else:
                print(f"  installed {destination_rel}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        license_source = stage_dir / "LICENSE-NVIDIA-DLSS.txt"
        for destination_rel in LICENSE_COPY_DESTINATIONS:
            destination = dlss5_dir / destination_rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or sha256_of(destination) != sha256_of(license_source):
                shutil.copy2(license_source, destination)

        readme = dlss5_dir / "README-LINUX.txt"
        readme.write_text(README_LINUX, encoding="utf-8")

    print_layout(dlss5_dir)
    print()
    print("Done. Official DLSS Super Resolution and Frame Generation runtimes are installed.")
    print("Note: DLSS 5 Neural Rendering is Windows-only, and the DLSSG worker")
    print("must be built for Linux (MIT source in the upstream fork).")
    print("See dlss5/README-LINUX.txt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())