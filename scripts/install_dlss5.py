#!/usr/bin/env python3
"""WanGP optional DLSS 5 installer for Linux.

Port of scripts/install_dlss5.ps1 for non-Windows platforms. The Windows
installer stages Windows-only binaries (D3D12 workers, ReShade/RenoDX hooks
and community DLSSNR builds); this installer instead fetches the equivalent
official resources that exist for Linux: the public NVIDIA DLSS SDK runtimes
(Super Resolution and Frame Generation, CUDA/Vulkan API) and the NVSDK NGX
API headers, pinned to a repository tag and SHA-256 verified.

All files are downloaded from NVIDIA's official public repository
(https://github.com/NVIDIA/DLSS, tag v310.9.16). No community mirrors are
used. The files are governed by the NVIDIA RTX SDKs License: no standalone
redistribution, NVIDIA-GPU-only use (see the installed
LICENSE-NVIDIA-DLSS.txt).

Installed layout under <WanGP root>/dlss5:

  dlss/libnvidia-ngx-dlss.so.310.9.16      DLSS Super Resolution runtime
  dlssg/libnvidia-ngx-dlssg.so.310.9.16    DLSS Frame Generation runtime
  sdk/include/...                          NVSDK NGX API headers
  sdk/lib/libnvsdk_ngx.a                   static import library
  sdk/lib/libnvidia-ngx-dlssd.so.310.9.16  legacy DLSS Dynamic feature library

Notes:
- DLSS 5 Neural Rendering has no Linux runtime (DLSSNR/ReShade/RenoDX are
  Windows-only) and stays disabled on Linux.
- Running DLSS Frame Generation additionally requires a Linux build of the
  WanGP DLSSG worker (MIT source:
  https://github.com/DeepBeepMeep/dlss5-visual-enhancer,
  native/WanGP-Adapter/dlssg_worker.cpp); place the resulting binary at
  dlss5/dlssg-native/dlssg-worker.

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
SDK_TAG = "v310.9.16"
SDK_URL = f"https://raw.githubusercontent.com/NVIDIA/DLSS/{SDK_TAG}"
RUNTIME_SO_SUFFIX = ".310.9.16"

# (repo path, destination relative to <root>/dlss5, pinned SHA-256). The
# license text is downloaded once and copied into the runtime directories.
SDK_DOWNLOADS = (
    ("lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.310.9.16", "dlss/libnvidia-ngx-dlss.so.310.9.16",
     "30982C937AFED85C96B8D313323DE9A650A2EEC6DE1BB0F714B21750F4CE98F6"),
    ("lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so.310.9.16", "sdk/lib/libnvidia-ngx-dlssd.so.310.9.16",
     "A5D59DE18841F15779039DDA567A15DD723BA737915F489271FDB5F109FBFB96"),
    ("lib/Linux_x86_64/rel/libnvidia-ngx-dlssg.so.310.9.16", "dlssg/libnvidia-ngx-dlssg.so.310.9.16",
     "574839CAE01C835DE4D5845C36CA9CF75361F8160BD52397142C36A24E29E6F7"),
    ("lib/Linux_x86_64/libnvsdk_ngx.a", "sdk/lib/libnvsdk_ngx.a",
     "D3C50BB0D9624FC35E502A8E929B8FC1E2B505017EBDB84C0D5B627E11A235DF"),
    ("LICENSE.txt", "LICENSE-NVIDIA-DLSS.txt",
     "D4216E39EBEF5F9B50A6712EBB37BEEB5379862A67733A9999C651F21592AAF0"),
    ("include/nvsdk_ngx.h", "sdk/include/nvsdk_ngx.h",
     "DC38E7467CF415379C9D12AE1B6E4A494C453ED92720FB53E92AECB523E7B848"),
    ("include/nvsdk_ngx_defs.h", "sdk/include/nvsdk_ngx_defs.h",
     "EA23F33497CD274860D1C25A97644FCE807DCB0037C594547203343103FAD03E"),
    ("include/nvsdk_ngx_defs_dlssd.h", "sdk/include/nvsdk_ngx_defs_dlssd.h",
     "2E97FE1595B1334F71B51724F2CB67496E8C296A7A2EEC5606DA53055882E716"),
    ("include/nvsdk_ngx_defs_dlssg.h", "sdk/include/nvsdk_ngx_defs_dlssg.h",
     "C29765AB9E0125F9D724AE9811CC2ABE03738D17486EC8EEE80D78A1504549E1"),
    ("include/nvsdk_ngx_defs_vk.h", "sdk/include/nvsdk_ngx_defs_vk.h",
     "0A24D0861ACE7D6B9362A67B7F08BEA1B33EA123C5CE730B19133DE8A891D031"),
    ("include/nvsdk_ngx_helpers.h", "sdk/include/nvsdk_ngx_helpers.h",
     "5BCBADFE7478B802CF6D3ACA4DC5DDD7D0889B99726E69C63F9E9BD555F44471"),
    ("include/nvsdk_ngx_helpers_dlssd.h", "sdk/include/nvsdk_ngx_helpers_dlssd.h",
     "A25FDDA925B479DFE869593AEBA311EF08793DA54E0391F2FE2FE756CBEF5532"),
    ("include/nvsdk_ngx_helpers_dlssd_cuda.h", "sdk/include/nvsdk_ngx_helpers_dlssd_cuda.h",
     "7E35CDE214D905346A487B9A28C05D35E1C55B480E237FE3DE69580A7E332730"),
    ("include/nvsdk_ngx_helpers_dlssd_vk.h", "sdk/include/nvsdk_ngx_helpers_dlssd_vk.h",
     "11DF8FBDA532A97FE867C8EF2B40058B49B7898183710F3468E55B1EEDB8F40B"),
    ("include/nvsdk_ngx_helpers_dlssg.h", "sdk/include/nvsdk_ngx_helpers_dlssg.h",
     "6881D21419C40C78EC948B326902ECE09FC112B0223144D51C4905F3FCED7A1F"),
    ("include/nvsdk_ngx_helpers_dlssg_vk.h", "sdk/include/nvsdk_ngx_helpers_dlssg_vk.h",
     "B371D90B9F4372B9A415E68A0AFBDC2F31B7B552E219A9637D848749BC965316"),
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
  dlss5/dlssg-native/dlssg-worker to enable it.
"""
def resolve_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        # scripts/install_dlss5.py -> WanGP root
        root = Path(__file__).resolve().parents[1]
    # wgp.py is what a checkout of the repository always carries. The launcher
    # scripts are personal and gitignored, so they cannot gate a fresh clone.
    if not (root / "wgp.py").is_file() and not (root / "src").is_dir():
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