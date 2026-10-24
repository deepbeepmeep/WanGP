#!/usr/bin/env bash
# Builds the Linux Vulkan worker for NVIDIA NGX DLSS Frame Generation.
#
# Prerequisites (see docs/DLSS5.md):
#   * an NVIDIA GPU with a recent driver providing the Vulkan ICD
#     (check: ls /etc/vulkan/icd.d/),
#   * the Vulkan loader (libvulkan.so.1),
#   * the official DLSS SDK installed by scripts/install_dlss5.py
#     (headers in dlss5/sdk/include, runtime in dlss5/dlssg),
#   * g++ (or a compatible C++17 compiler).
#
# The Vulkan *headers* are taken from /usr/include when the distro package
# (e.g. libvulkan-dev) is installed; otherwise they are downloaded from the
# official KhronosGroup/Vulkan-Headers repository into scripts/
# dlssg_worker_linux/.cache (matching the installed loader's major series).
#
# Output: dlss5/dlssg/dlssg-worker (this is the binary WanGP's runtime
# probes and spawns). The worker links the NGX runtime next to it via
# rpath $ORIGIN, so it must stay in dlss5/dlssg/.
set -euo pipefail

native="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${native}/.." && pwd)"
sdk_include="${root}/dlss5/sdk/include"
sdk_lib="${root}/dlss5/sdk/lib"
core_lib="${sdk_lib}/libnvsdk_ngx.a"
dlssg_dir="${root}/dlss5/dlssg"
output="${dlssg_dir}/dlssg-worker"
source="${native}/dlssg_worker_linux/dlssg_worker_linux.cpp"

fail() { echo "error: $*" >&2; exit 1; }

command -v g++ >/dev/null 2>&1 || fail "g++ not found; install a C++17 toolchain (e.g. 'apt install g++')"
[ -f "${source}" ] || fail "worker source missing: ${source}"
[ -f "${sdk_include}/nvsdk_ngx_vk.h" ] || fail "NGX SDK headers not found; run 'python scripts/install_dlss5.py' first"
[ -f "${core_lib}" ] || fail "NGX core library not found (${core_lib}); run 'python scripts/install_dlss5.py' first"
[ -d "${dlssg_dir}" ] || fail "dlss5/dlssg/ not found; run 'python scripts/install_dlss5.py' first"

runtime_so="$(ls "${dlssg_dir}"/libnvidia-ngx-dlssg.so.* 2>/dev/null | sort -V | tail -n 1 || true)"
[ -n "${runtime_so}" ] || fail "NGX DLSSG runtime not found in ${dlssg_dir}; run 'python scripts/install_dlss5.py' first"
# The feature runtime is dlopen'd by the NGX core at load time (via the
# worker's directory as PathListInfo); an unversioned symlink alongside the
# versioned .so makes that lookup robust.
ln -sf "$(basename "${runtime_so}")" "${dlssg_dir}/libnvidia-ngx-dlssg.so"

# --- Vulkan headers -------------------------------------------------------
vk_include=""
if [ -e /usr/include/vulkan/vulkan.h ]; then
    vk_include=/usr/include
else
    cache="${native}/dlssg_worker_linux/.cache"
    if [ -e "${cache}/vulkan-headers/include/vulkan/vulkan.h" ]; then
        vk_include="${cache}/vulkan-headers/include"
    else
        # No distro headers: use the official Khronos headers of the same
        # 1.3 series as the installed loader (loader soname carries no
        # version, so a 1.3.x tag is the safe default; 1.2+ core is all
        # the worker uses).
        tag="v1.3.275"
        echo "downloading Khronos Vulkan-Headers ${tag} (official) ..."
        mkdir -p "${cache}"
        curl -fL --retry 3 --max-time 300 \
            -o "${cache}/vulkan-headers.tar.gz" \
            "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/${tag}.tar.gz" \
            || fail "Vulkan-Headers download failed; 'apt install libvulkan-dev' also works"
        rm -rf "${cache}/vulkan-headers"
        mkdir -p "${cache}/vulkan-headers"
        tar -xzf "${cache}/vulkan-headers.tar.gz" -C "${cache}/vulkan-headers" --strip-components=1
        rm -f "${cache}/vulkan-headers.tar.gz"
        vk_include="${cache}/vulkan-headers/include"
    fi
fi
[ -e "${vk_include}/vulkan/vulkan.h" ] || fail "Vulkan headers not usable from ${vk_include}"

# --- Vulkan loader for link time ------------------------------------------
# Capture ldconfig output first so the grep runs without a pipeline: under
# `set -o pipefail` a `cmd | grep -q` can fail spuriously when grep exits on
# the first match and the upstream command is killed by SIGPIPE.
ldconfig_cache="$(ldconfig -p 2>/dev/null || true)"
vulkan_link=""
if grep -E -q 'libvulkan\.so( |$)' <<< "${ldconfig_cache}"; then
    vulkan_link="-lvulkan"
elif grep -E -q 'libvulkan\.so\.1' <<< "${ldconfig_cache}"; then
    vulkan_link="-l:libvulkan.so.1"
else
    fail "Vulkan loader (libvulkan.so.1) not found; install 'libvulkan1' (or the NVIDIA driver meta package)"
fi

echo "building ${output}"
echo "  source : ${source}"
echo "  NGX SDK: ${sdk_include}"
echo "  NGX core lib : ${core_lib}"
echo "  feature runtime: ${runtime_so} (loaded at runtime from the worker dir)"
echo "  Vulkan : ${vk_include} (link ${vulkan_link})"

g++ -O2 -std=c++17 -Wall \
    -I"${sdk_include}" -I"${vk_include}" \
    "${source}" \
    -o "${output}" \
    -L"${sdk_lib}" -lnvsdk_ngx \
    "${vulkan_link}" -ldl -lpthread \
    -Wl,-rpath,"\$ORIGIN"

chmod +x "${output}"
echo "built: ${output}"
echo "verify:  ${output} --probe"
