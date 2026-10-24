// Copyright (c) 2026 DeepBeepMeep contributors.
// SPDX-License-Identifier: MIT
//
// Linux (Vulkan) port of the open D3D12 host for NVIDIA NGX DLSS Frame
// Generation (dlssg_worker.cpp from the dlss5-visual-enhancer WanGP
// adapter, MIT). NVIDIA's NGX SDK headers and the libnvidia-ngx-dlssg.so
// runtime are separate proprietary components, not distributed by this
// project; install them with scripts/install_dlss5.py and build this file
// with scripts/build_dlssg_worker_linux.sh.
//
// The process intentionally keeps the original WanGP worker stdin/stdout
// protocol byte-for-byte so it replaces the Windows worker without Python
// changes. Vulkan differences from the Windows build:
//   * D3D12 device/adapter objects are replaced by a headless Vulkan
//     instance/device on an NVIDIA ICD; NGX is driven through
//     NVSDK_NGX_VULKAN_Init_with_ProjectID, NGX_VK_CREATE_DLSSG and
//     NGX_VK_EVALUATE_DLSSG.
//   * GPU selection uses --gpu <physical-device-index> instead of --adapter-luid.
//   * A scratch buffer (only if the runtime reports a non-zero requirement)
//     is a device-local VkBuffer, passed as a VkDeviceAddress.
//   * --probe always emits the capability JSON, even when initialization
//     fails, so WanGP can surface a structured unavailability reason.

#include <vulkan/vulkan.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_dlssg_vk.h>

#include <algorithm>
#include <cerrno>
#include <cstdarg>
#include <dirent.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <limits.h>
#include <unistd.h>

// File-local declarations and implementation; the only entry point is main().
namespace {

// Wire protocol constants; the magic values are little-endian ASCII tags.
constexpr uint32_t SETUP_MAGIC = 0x31534746; // "FGS1": setup request from the host
constexpr uint32_t SETUP_OUT_MAGIC = 0x31524746; // "FGR1": setup response to the host
constexpr uint32_t FRAME_MAGIC = 0x31464746; // "FGF1": one frame request (header + backbuffer + motion)
constexpr uint32_t FRAME_OUT_MAGIC = 0x314F4746; // "FGO1": frame response (header + generated pixels)
constexpr char PROJECT_ID[] = "6d648dba-bac0-44ef-8e49-d8291d756f37"; // project ID this worker registers with the NGX SDK on init
constexpr uint32_t NVIDIA_VENDOR_ID = 0x10DE; // PCI vendor ID of NVIDIA GPUs
constexpr uint64_t kQueueTimeoutNs = 60ull * 1000ull * 1000ull * 1000ull; // how long the worker waits for GPU work before declaring failure

// The setup request from the host: target resolution, how many input frames
// it will send, and how many frames to generate between each input pair.
struct SetupHeader {
    uint32_t magic, width, height, frame_count, generated_count;
};

// The worker's setup reply: status (0 ok, 1 frame generation unavailable,
// 2 request exceeds the runtime's limit, 3 feature creation failed) and
// the maximum number of generated frames the runtime supports.
struct SetupResult {
    uint32_t magic, status, maximum, reserved;
};

// One input frame: sequence id (used as the backbuffer frame id), a scene-
// reset flag, and the presentation timestamp as a rational number (carried
// for protocol compatibility; the worker does not consume it).
struct FrameHeader {
    uint32_t magic, index, reset, reserved;
    int64_t timestamp_numerator, timestamp_denominator;
};

// The worker's frame reply: status, how many frames were generated, and
// whether the runtime disabled (rejected) the output.
struct FrameResult {
    uint32_t magic, status, generated, disabled;
};

// The sizes are part of the wire protocol; a layout drift breaks the host.
static_assert(sizeof(SetupHeader) == 20);
static_assert(sizeof(SetupResult) == 16);
static_assert(sizeof(FrameHeader) == 32);
static_assert(sizeof(FrameResult) == 16);

// Reads exactly size bytes from stdin, retrying on EINTR and partial reads;
// returns false on EOF or error. The protocol is length-prefixed, so a
// short read must never be passed to the caller.
bool ReadExact(void *data, size_t size) {
    auto *target = static_cast<uint8_t *>(data);
    while (size != 0) {
        const ssize_t count = ::read(STDIN_FILENO, target, size);
        if (count < 0) {
            if (errno == EINTR) continue; // signal interruptions must not abort a protocol read
            return false;
        }
        if (count == 0) return false;
        target += static_cast<size_t>(count);
        size -= static_cast<size_t>(count);
    }
    return true;
}

// Writes exactly size bytes to stdout, retrying on EINTR and partial
// writes; returns false when the host goes away mid-response.
bool WriteExact(const void *data, size_t size) {
    const auto *source = static_cast<const uint8_t *>(data);
    while (size != 0) {
        const ssize_t count = ::write(STDOUT_FILENO, source, size);
        if (count < 0) {
            if (errno == EINTR) continue; // signal interruptions must not abort a protocol write
            return false;
        }
        source += static_cast<size_t>(count);
        size -= static_cast<size_t>(count);
    }
    return true;
}

// printf-style diagnostics on stderr; stdout is reserved for the wire
// protocol, and the flush keeps host-side log output live.
void Log(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputc('\n', stderr);
    fflush(stderr);
}

// Directory containing this executable (resolved via /proc/self/exe). The
// NGX runtime libraries are installed next to the worker binary and loaded
// from this path during initialization.
std::string WorkerDir() {
    char path[4096] = {};
    const ssize_t size = readlink("/proc/self/exe", path, sizeof(path) - 1); // /proc/self/exe is a symlink to this executable
    if (size <= 0) return ".";
    path[size] = '\0';
    char *slash = std::strrchr(path, '/');
    if (slash != nullptr && slash != path) *slash = '\0';
    else return ".";
    return path;
}

// Reports which NGX runtime version is installed next to the worker.
std::string RuntimeVersion(const std::string &directory) {
    // The worker directory holds e.g. libnvidia-ngx-dlssg.so.310.7.0.
    DIR *dir = opendir(directory.c_str());
    if (dir == nullptr) return "unknown";
    std::string version = "unknown";
    while (auto *entry = readdir(dir)) {
        const std::string name = entry->d_name;
        const std::string prefix = "libnvidia-ngx-dlssg.so.";
        if (name.rfind(prefix, 0) == 0 && name.size() > prefix.size()) { // only the versioned soname carries the runtime version
            version = name.substr(prefix.size());
            break;
        }
    }
    closedir(dir);
    return version;
}

// UTF-8 -> wchar_t conversion without locale support; the NGX API takes
// wide strings. Four-byte sequences and invalid bytes degrade to '?'.
std::wstring Utf8ToWide(const std::string &text) {
    std::wstring result;
    result.reserve(text.size());
    size_t index = 0;
    while (index < text.size()) {
        const uint8_t byte = static_cast<uint8_t>(text[index++]);
        if (byte < 0x80) {
            result.push_back(static_cast<wchar_t>(byte));
        } else if (byte >= 0xC0 && byte < 0xE0 && index < text.size()) {
            const uint8_t next = static_cast<uint8_t>(text[index++]);
            if ((next & 0xC0) == 0x80)
                result.push_back(static_cast<wchar_t>(((byte & 0x1F) << 6) | (next & 0x3F)));
        } else if (byte >= 0xE0 && byte < 0xF0 && index + 1 < text.size()) {
            const uint8_t a = static_cast<uint8_t>(text[index++]);
            const uint8_t b = static_cast<uint8_t>(text[index++]);
            if ((a & 0xC0) == 0x80 && (b & 0xC0) == 0x80)
                result.push_back(static_cast<wchar_t>(((byte & 0x0F) << 12) | ((a & 0x3F) << 6) | (b & 0x3F)));
        } else {
            result.push_back(L'?');
        }
    }
    return result;
}

// Returns the supported subset of `wanted` instance/device extensions.
std::vector<const char *> FilterExtensions(const char **wanted, unsigned int wanted_count, bool instance, VkPhysicalDevice physical) {
    std::vector<std::string> supported;
    uint32_t count = 0;
    const VkResult listed = instance
        ? vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr)
        : vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
    if (listed == VK_SUCCESS && count != 0) {
        std::vector<VkExtensionProperties> properties(count);
        if ((instance ? vkEnumerateInstanceExtensionProperties(nullptr, &count, properties.data())
                      : vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, properties.data())) == VK_SUCCESS)
            for (const auto &property : properties) supported.push_back(property.extensionName);
    }
    std::vector<const char *> enabled;
    for (unsigned int index = 0; index < wanted_count; ++index) {
        if (wanted[index] == nullptr) continue;
        const bool ok = std::find(supported.begin(), supported.end(), wanted[index]) != supported.end();
        if (ok) enabled.push_back(wanted[index]);
        else Log("extension %s is required by NGX but not available; continuing without it", wanted[index]);
    }
    return enabled;
}

// Returns the first memory-type index satisfying both the resource's
// memoryTypeBits and the requested property flags; UINT32_MAX when none.
uint32_t FindMemoryType(VkPhysicalDevice physical, const VkMemoryRequirements &requirements, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory = {};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    for (uint32_t index = 0; index < memory.memoryTypeCount; ++index)
        if ((requirements.memoryTypeBits & (1u << index)) != 0 &&
            (memory.memoryTypes[index].propertyFlags & properties) == properties)
            return index;
    return UINT32_MAX;
}

// Fills a 4x4 row-major matrix with the identity; used for the static
// camera projection matrices handed to the DLSSG evaluation.
void Identity(float matrix[4][4]) {
    std::memset(matrix, 0, sizeof(float) * 16);
    matrix[0][0] = matrix[1][1] = matrix[2][2] = matrix[3][3] = 1.0f;
}

// A GPU image plus its memory and view, and the NVSDK_NGX resource
// descriptor the runtime consumes; 'ready' marks the image as holding
// fresh data for the next evaluation.
struct ImageRes {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t width = 0, height = 0;
    NVSDK_NGX_Resource_VK resource = {};

    bool ready = false;
};

// Host-mapped staging buffer used as the CPU-side end of every transfer.
struct StagingBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
    uint64_t size = 0;
};

// Device-local buffer plus its NVSDK_NGX resource descriptor.
struct DeviceBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    NVSDK_NGX_Resource_VK resource = {};
    uint64_t size = 0;
};

// Owns the whole GPU/NGX session: the headless Vulkan instance/device on
// the selected GPU, the images and buffers that feed the DLSSG runtime,
// and the frame-generation feature itself. Per-session lifecycle:
// Initialize -> CreateFeature -> (Upload -> EvaluateGroup)* -> Shutdown.
class Worker {
public:
    ~Worker() { Shutdown(); } // guarantees teardown even after a partially failed Initialize

    // One-time setup shared by --probe and --serve:
    //   1. create a headless Vulkan instance/device (the GPU comes from
    //      --gpu, or the first NVIDIA GPU when omitted);
    //   2. point the NGX runtime at the worker directory so it finds
    //      libnvidia-ngx-dlssg.so, and initialize it with our project ID;
    //   3. query whether frame generation is available on this GPU at all.
    // Any step may fail; Probe() still emits a structured JSON report.
    bool Initialize(int gpu_index) {
        ::setenv("NGX_DISABLE_UPDATER", "1", 0); // a worker process must never let the SDK run its updater
        if (!CreateInstance()) return Fail("Vulkan instance creation failed");
        if (!SelectPhysicalDevice(gpu_index)) return Fail("no compatible Vulkan GPU found");
        if (!CreateDevice()) return Fail("Vulkan device creation failed");
        if (!CreateCommandObjects()) return Fail("Vulkan command objects could not be created");

        const std::string directory = WorkerDir(); // the runtime .so is installed next to this binary
        runtime_version_ = RuntimeVersion(directory);
        const std::wstring app_data = Utf8ToWide(directory);
        const wchar_t *paths[] = {app_data.c_str()};
        NVSDK_NGX_FeatureCommonInfo info = {};
        info.PathListInfo.Path = paths; // search path the SDK uses to locate the runtime
        info.PathListInfo.Length = 1;

        const NVSDK_NGX_Result init = NVSDK_NGX_VULKAN_Init_with_ProjectID( // loads the runtime and binds it to our Vulkan device
            PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", app_data.c_str(),
            instance_, physical_, device_, nullptr, nullptr, &info, NVSDK_NGX_Version_API);
        if (NVSDK_NGX_FAILED(init)) {
            Log("NGX initialization failed: 0x%08X", init);
            return false;
        }
        ngx_initialized_ = true; // guards the matching NVSDK_NGX_VULKAN_Shutdown1 in Shutdown()

        const NVSDK_NGX_Result capabilities = NVSDK_NGX_VULKAN_GetCapabilityParameters(&parameters_); // parameter block shared by capability queries and DLSSG configuration
        if (NVSDK_NGX_FAILED(capabilities) || parameters_ == nullptr) {
            Log("NGX capability query failed: 0x%08X", capabilities);
            return false;
        }
        int available = 0;
        parameters_->Get(NVSDK_NGX_Parameter_FrameGeneration_Available, &available); // 0/1: frame generation supported on this GPU
        parameters_->Get(NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &maximum_); // how many frames one evaluation may synthesize per input pair
        if (maximum_ == 0) maximum_ = 1; // treat an unreported limit as the minimum of one frame
        available_ = available != 0;
        Log("NGX initialized on %s (frame generation %s, multi-frame max %u)", device_name_.c_str(),
            available_ ? "available" : "unavailable", maximum_);
        return true;
    }

    // Builds all per-session GPU resources for one DLSSG context (backbuffer,
    // motion vectors, seeded depth, output image, reject-flag buffer, and
    // host-visible staging buffers), configures the NGX parameters, and
    // creates the frame-generation feature. Later Upload/EvaluateGroup calls
    // then only copy pixels and run the runtime.
    bool CreateFeature(uint32_t width, uint32_t height, uint32_t generated_count) {
        width_ = width;
        height_ = height;
        generated_count_ = generated_count;

        // Depth format/layout selection. The Windows build uses an R32_FLOAT
        // color image; the Vulkan reference app passes a true depth-format
        // image (D24S8/D32/S8) with the DEPTH aspect. DLSSG_DEBUG_DEPTH can
        // override the default for diagnosis:
        //   1 = R32F_SFLOAT (color aspect, GENERAL)
        //   2 = D32_SFLOAT_S8_UINT (depth aspect, DEPTH_STENCIL_OPTIMAL) [default]
        //   3 = D32_SFLOAT (depth aspect, DEPTH_STENCIL_OPTIMAL)
        //   4 = D24_UNORM_S8_UINT (depth aspect, DEPTH_STENCIL_OPTIMAL)
        int depth_mode = 2; // default: a true D32S8 depth image (override modes documented above)
        if (const char *env = std::getenv("DLSSG_DEBUG_DEPTH")) {
            const int value = std::atoi(env);
            if (value >= 1 && value <= 4) depth_mode = value;
        }
        VkFormat depth_format;
        VkImageLayout depth_layout;
        VkImageAspectFlags depth_aspect;
        switch (depth_mode) {
            case 1: depth_format = VK_FORMAT_R32_SFLOAT; depth_layout = VK_IMAGE_LAYOUT_GENERAL; depth_aspect = VK_IMAGE_ASPECT_COLOR_BIT; break;
            case 3: depth_format = VK_FORMAT_D32_SFLOAT; depth_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; depth_aspect = VK_IMAGE_ASPECT_DEPTH_BIT; break;
            case 4: depth_format = VK_FORMAT_D24_UNORM_S8_UINT; depth_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; depth_aspect = VK_IMAGE_ASPECT_DEPTH_BIT; break;
            default: depth_format = VK_FORMAT_D32_SFLOAT_S8_UINT; depth_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; depth_aspect = VK_IMAGE_ASPECT_DEPTH_BIT; break;
        }
        Log("depth image: mode %d (format 0x%X, layout %d, aspect 0x%X)", depth_mode,
            static_cast<unsigned>(depth_format), static_cast<int>(depth_layout), static_cast<unsigned>(depth_aspect));

        if (!CreateImage(color_, width, height, VK_FORMAT_R8G8B8A8_UNORM, false, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT)) // backbuffer: the host-rendered RGBA frame
            return Fail("backbuffer image creation failed");
        if (!CreateImage(motion_, width, height, VK_FORMAT_R16G16_SFLOAT, false, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT)) // motion vectors, RG16F, same size as the backbuffer
            return Fail("motion vector image creation failed");
        if (!CreateImage(depth_, width, height, depth_format, false, depth_layout, depth_aspect)) // depth: seeded once below, read-only for NGX afterwards
            return Fail("depth image creation failed");
        if (!CreateImage(output_, width, height, VK_FORMAT_R8G8B8A8_UNORM, true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT)) // output: where NGX writes each generated frame
            return Fail("output image creation failed");
        if (!CreateDeviceBuffer(disable_, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) // 4-byte flag the runtime writes to reject a generated frame
            return Fail("disable output buffer creation failed");
        if (!CreateStaging(color_upload_, static_cast<uint64_t>(width) * height * 4) ||
            !CreateStaging(motion_upload_, static_cast<uint64_t>(width) * height * 4) ||
            !CreateStaging(depth_upload_, static_cast<uint64_t>(width) * height * 4) ||
            !CreateStaging(disable_zero_, 4) ||
            !CreateStaging(output_readback_, static_cast<uint64_t>(width) * height * 4 * generated_count) || // readback slot for every generated frame of a group
            !CreateStaging(disable_readback_, static_cast<uint64_t>(generated_count) * 4))
            return Fail("staging buffer creation failed");

        const float mid_depth = 0.5f; // flat mid-plane so the first frame never samples undefined depth
        std::vector<float> depth(static_cast<size_t>(width) * height, mid_depth);
        std::memcpy(depth_upload_.mapped, depth.data(), depth.size() * sizeof(float));
        const uint8_t disable_zero[4] = {0, 0, 0, 0};
        std::memcpy(disable_zero_.mapped, disable_zero, sizeof(disable_zero));
        if (!Begin()) return Fail("command buffer begin failed");
        CopyStagingToImage(depth_upload_, depth_); // upload the seeded depth into the depth image
        if (!SubmitAndWait()) return Fail("initial depth upload failed");
        depth_.ready = true; // depth is never re-uploaded, so mark it provided up front
        const uint32_t always = NVSDK_NGX_DLSSG_ResourceFlags_Backbuffer | NVSDK_NGX_DLSSG_ResourceFlags_MVecs |
            NVSDK_NGX_DLSSG_ResourceFlags_Depth | NVSDK_NGX_DLSSG_ResourceFlags_HUDLess |
            NVSDK_NGX_DLSSG_ResourceFlags_OutputInterpolated | NVSDK_NGX_DLSSG_ResourceFlags_OutputDisableInterpolation; // the complete set of resources the host supplies every frame
        const uint32_t never = NVSDK_NGX_DLSSG_ResourceFlags_UI | NVSDK_NGX_DLSSG_ResourceFlags_UIAlpha |
            NVSDK_NGX_DLSSG_ResourceFlags_BidirectionalDistortionField | NVSDK_NGX_DLSSG_ResourceFlags_OutputReal; // overlays, distortion and "real" outputs are never provided
        parameters_->Set(NVSDK_NGX_DLSSG_Parameter_ResourceAlwaysProvided_Flags, always);
        parameters_->Set(NVSDK_NGX_DLSSG_Parameter_ResourceNeverProvided_Flags, never);
        parameters_->Set(NVSDK_NGX_DLSSG_Parameter_UserInterfaceRecompositionEnabled, 0u); // no HUD/UI recomposition in WanGP
        parameters_->Set(NVSDK_NGX_DLSSG_Parameter_Width, width_);
        parameters_->Set(NVSDK_NGX_DLSSG_Parameter_Height, height_);

        size_t scratch_size = 0;
        const NVSDK_NGX_Result scratch = NVSDK_NGX_VULKAN_GetScratchBufferSize(NVSDK_NGX_Feature_FrameGeneration, parameters_, &scratch_size); // the runtime may demand a device-local scratch buffer
        if (NVSDK_NGX_FAILED(scratch)) {
            Log("NGX scratch size query failed: 0x%08X (continuing without scratch)", scratch);
        }
        if (scratch_size != 0 && !CreateScratch(scratch_size)) {
            Log("NGX scratch buffer allocation failed");
            return false;
        }

        NVSDK_NGX_DLSSG_Create_Params create = {};
        create.Width = width_;
        create.Height = height_;
        create.NativeBackbufferFormat = static_cast<unsigned int>(VK_FORMAT_R8G8B8A8_UNORM);
        create.RenderWidth = width_;
        create.RenderHeight = height_;
        create.DynamicResolutionScaling = false;
        if (!Begin()) return false;
        const NVSDK_NGX_Result result = NGX_VK_CREATE_DLSSG(cmd_, 1, 1, &feature_, parameters_, &create); // instantiate the frame-generation feature
        if (NVSDK_NGX_FAILED(result)) {
            Log("DLSSG feature creation failed: 0x%08X", result);
            feature_ = nullptr;
            return false;
        }
        if (!SubmitAndWait()) return false; // feature creation is GPU work, so it must be submitted
        Log("DLSSG feature created at %ux%u, maximum generated frames %u", width_, height_, maximum_);
        return true;
    }

    // Uploads one host frame: the RGBA backbuffer and the RG16F motion
    // vectors (both 4 bytes per pixel) are memcpy'd into mapped staging
    // memory and then copied onto their GPU images.
    bool Upload(const uint8_t *rgba, const uint8_t *motion) {
        if (width_ == 0 || height_ == 0) return false;
        const size_t row = static_cast<size_t>(width_) * 4;
        std::memcpy(color_upload_.mapped, rgba, row * height_); // host-mapped coherent memory: a plain memcpy is the upload
        std::memcpy(motion_upload_.mapped, motion, row * height_);
        if (!Begin()) return false;
        CopyStagingToImage(color_upload_, color_);
        CopyStagingToImage(motion_upload_, motion_);
        color_.ready = true; // mark the backbuffer fresh for NGX
        motion_.ready = true; // mark the motion vectors fresh for NGX
        return SubmitAndWait();
    }

    // Runs the frame-generation feature for this backbuffer, synthesizing
    // generated_count_ intermediate frames as one group. Every generated
    // frame and its reject flag are copied back through staging and unpacked
    // into 'frames'; 'disabled' is set when the runtime refuses to produce
    // a frame (e.g. invalid motion vectors), in which case no pixels are
    // returned for it.
    NVSDK_NGX_Result EvaluateGroup(uint32_t frame_id, bool reset, std::vector<std::vector<uint8_t>> &frames, bool &disabled) {
        if (!Begin()) return static_cast<NVSDK_NGX_Result>(0x7FFFFFFF);
        NVSDK_NGX_Result result = NVSDK_NGX_Result_Success;
        for (uint32_t index = 1; index <= generated_count_; ++index) { // intermediate frames are numbered 1..N (0 is the real backbuffer)
            CopyBuffer(disable_zero_, disable_); // clear the per-frame reject flag before evaluating again
            parameters_->Set(NVSDK_NGX_DLSSG_Parameter_BackbufferFrameID, static_cast<unsigned long long>(frame_id)); // the host frame index keeps the runtime's continuity tracking correct

            NVSDK_NGX_VK_DLSSG_Eval_Params eval = {}; // wire the GPU resources into the evaluation call
            eval.pBackbuffer = &color_.resource;
            eval.pDepth = &depth_.resource;
            eval.pMVecs = &motion_.resource;
            eval.pHudless = &color_.resource;
            eval.pOutputInterpFrame = &output_.resource;
            eval.pOutputDisableInterpolation = &disable_.resource;

            NVSDK_NGX_DLSSG_Opt_Eval_Params options = {};
            options.multiFrameCount = generated_count_; // N frames are synthesized between the previous and this frame
            options.multiFrameIndex = index; // which intermediate frame this call produces
            Identity(options.cameraViewToClip); // static camera: all projection transforms are identity
            Identity(options.clipToCameraView);
            Identity(options.clipToLensClip);
            Identity(options.clipToPrevClip);
            Identity(options.prevClipToClip);
            options.mvecScale[0] = 1.0f / width_; // motion vectors are stored in pixels; scale them to UV space
            options.mvecScale[1] = 1.0f / height_;
            options.cameraNear = 0.1f; // placeholder camera constants (the camera never moves)
            options.cameraFar = 1000.0f;
            options.cameraFOV = 1.04719755f; // ~60 degrees vertical field of view
            options.cameraAspectRatio = static_cast<float>(width_) / height_;
            options.reset = reset; // scene-change flag: makes NGX drop its internal temporal state
            options.orthoProjection = true; // WanGP renders with an orthographic projection
            options.motionVectorsInvalidValue = -65500.0f; // sentinel the renderer writes where motion is undefined
            options.motionVectorsDilated = true;
            options.mvecsSubrectSize = {width_, height_};
            options.depthSubrectSize = {width_, height_};
            options.hudLessSubrectSize = {width_, height_};
            options.backbufferSubrectSize = {width_, height_};
            options.outputInterpSubrectSize = {width_, height_};

            result = NGX_VK_EVALUATE_DLSSG(cmd_, feature_, parameters_, &eval, &options); // the actual frame-generation call
            if (NVSDK_NGX_FAILED(result)) {
                Log("DLSSG evaluate failed for frame %u index %u/%u: 0x%08X", frame_id, index, generated_count_, result);
                vkEndCommandBuffer(cmd_);
                return result;
            }

            const uint64_t frame_offset = static_cast<uint64_t>(index - 1) * width_ * height_ * 4;
            CopyImageToStaging(output_, output_readback_, frame_offset, width_, height_); // pull this generated frame back to host memory
            CopyBuffer(disable_, disable_readback_, static_cast<uint64_t>(index - 1) * 4); // read this frame's flag into its own slot (the device buffer always holds the latest flag at offset 0)
        }
        if (!SubmitAndWait()) return static_cast<NVSDK_NGX_Result>(0x7FFFFFFF); // one submission covers the whole group of evaluations

        const uint8_t *disable_bytes = static_cast<const uint8_t *>(disable_readback_.mapped);
        disabled = false;
        for (uint32_t index = 0; index < generated_count_; ++index) disabled |= disable_bytes[index * 4] != 0; // the runtime may veto any frame of the group (e.g. bad motion)

        const uint8_t *output_bytes = static_cast<const uint8_t *>(output_readback_.mapped);
        const size_t row = static_cast<size_t>(width_) * 4;
        frames.assign(generated_count_, std::vector<uint8_t>(row * height_)); // allocate one output buffer per generated frame
        for (uint32_t index = 0; index < generated_count_; ++index) // unpack each frame from the readback buffer (row by row)
            for (uint32_t line = 0; line < height_; ++line)
                std::memcpy(frames[index].data() + static_cast<size_t>(line) * row,
                            output_bytes + static_cast<uint64_t>(index) * row * height_ + static_cast<size_t>(line) * row, row);
        return result;
    }

    // Introspection used by Probe() to build its JSON report.
    bool available() const { return available_; }
    uint32_t maximum() const { return maximum_; }
    const std::string &runtime_version() const { return runtime_version_; }
    const std::string &device_name() const { return device_name_; }
private:
    // Human-readable queue capability label for the startup diagnostics.
    static const char *QueueTypeName(VkQueueFlags flags) {
        if ((flags & VK_QUEUE_GRAPHICS_BIT) && (flags & VK_QUEUE_COMPUTE_BIT)) return "graphics+compute";
        if (flags & VK_QUEUE_GRAPHICS_BIT) return "graphics";
        if (flags & VK_QUEUE_COMPUTE_BIT) return "compute";
        if (flags & VK_QUEUE_TRANSFER_BIT) return "transfer";
        return "other";
    }

    // Human-readable GPU class label for the startup diagnostics.
    static const char *PhysicalDeviceTypeName(VkPhysicalDeviceType type) {
        switch (type) {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
            case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
            default: return "other";
        }
    }

    // Creates the headless Vulkan instance. The extension set is what NGX
    // says it requires, filtered to what the driver actually exposes;
    // missing optional extensions are tolerated with a warning.
    bool CreateInstance() {
        unsigned int instance_count = 0, device_count = 0;
        const char **instance_exts = nullptr, **device_exts = nullptr;
        const NVSDK_NGX_Result extensions = NVSDK_NGX_VULKAN_RequiredExtensions(&instance_count, &instance_exts, &device_count, &device_exts);
        if (NVSDK_NGX_FAILED(extensions)) {
            Log("NGX required-extensions query failed: 0x%08X (using none)", extensions);
        } else {
            ngx_device_exts_ = device_exts;
            ngx_device_ext_count_ = device_count;
        }

        VkApplicationInfo app = {};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "Wan2GP DLSSG worker";
        app.applicationVersion = 1;
        app.pEngineName = "Wan2GP";
        app.engineVersion = 1;
        app.apiVersion = VK_API_VERSION_1_3;

        const std::vector<const char *> instance_enabled =
            FilterExtensions(instance_exts, instance_count, true, VK_NULL_HANDLE);
        VkInstanceCreateInfo create = {};
        create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create.pApplicationInfo = &app;
        create.enabledExtensionCount = static_cast<uint32_t>(instance_enabled.size());
        create.ppEnabledExtensionNames = instance_enabled.empty() ? nullptr : instance_enabled.data();
        if (vkCreateInstance(&create, nullptr, &instance_) != VK_SUCCESS) return false;
        return true;
    }

    // Chooses the physical device: an explicit --gpu index wins; otherwise
    // the first NVIDIA discrete GPU is preferred, then any other NVIDIA
    // device, then any non-CPU GPU. Every device is logged so a wrong pick
    // shows up in the worker's stderr output.
    bool SelectPhysicalDevice(int gpu_index) {
        uint32_t count = 0;
        if (vkEnumeratePhysicalDevices(instance_, &count, nullptr) != VK_SUCCESS || count == 0) return false;
        std::vector<VkPhysicalDevice> devices(count);
        if (vkEnumeratePhysicalDevices(instance_, &count, devices.data()) != VK_SUCCESS) return false;

        VkPhysicalDevice chosen = VK_NULL_HANDLE;
        if (gpu_index >= 0 && gpu_index < static_cast<int>(devices.size())) { // an explicit --gpu index is trusted as-is
            chosen = devices[gpu_index];
        } else {
            for (size_t index = 0; index < devices.size(); ++index) {
                VkPhysicalDeviceProperties props = {};
                vkGetPhysicalDeviceProperties(devices[index], &props);
                Log("Device %zu: %s (%s, vendor 0x%04X)", index, props.deviceName,
                    PhysicalDeviceTypeName(props.deviceType), static_cast<unsigned>(props.vendorID));
                if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue; // skip software ICDs (llvmpipe & co)
                if (props.vendorID == NVIDIA_VENDOR_ID && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { // best case: a discrete NVIDIA GPU
                    chosen = devices[index];
                    break;
                }
                if (chosen == VK_NULL_HANDLE && props.vendorID == NVIDIA_VENDOR_ID) chosen = devices[index]; // fallback: any other NVIDIA device
            }
            if (chosen == VK_NULL_HANDLE)
                for (VkPhysicalDevice candidate : devices) {
                    VkPhysicalDeviceProperties props = {};
                    vkGetPhysicalDeviceProperties(candidate, &props);
                    if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) { // last resort: any hardware GPU (non-NVIDIA fails at NGX init)
                        chosen = candidate;
                        break;
                    }
                }
        }
        if (chosen == VK_NULL_HANDLE) return false;
        physical_ = chosen;
        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(physical_, &props);
        device_name_ = props.deviceName;
        if (props.vendorID != NVIDIA_VENDOR_ID) Log("warning: selected device is not an NVIDIA GPU; NGX DLSSG requires an NVIDIA GPU"); // warn early instead of failing opaquely inside the runtime
        return true;
    }

    // Creates the logical device: one graphics queue from the first
    // graphics-capable family plus the device extensions NGX requires. On
    // Vulkan 1.2+ the bufferDeviceAddress feature is enabled so the NGX
    // scratch buffer can be handed over as a VkDeviceAddress.
    bool CreateDevice() {
        const std::vector<const char *> device_enabled =
            FilterExtensions(ngx_device_exts_, ngx_device_ext_count_, false, physical_); // keep only the extensions this driver actually exposes

        uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, nullptr);
        std::vector<VkQueueFamilyProperties> family_props(families);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, family_props.data());
        uint32_t family = 0;
        bool found = false;
        for (uint32_t index = 0; index < families; ++index) {
            Log("Queue family %u: %s (%u queues)", index, QueueTypeName(family_props[index].queueFlags),
                static_cast<unsigned>(family_props[index].queueCount));
            if ((family_props[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) { // NGX runs on one graphics queue; transfers ride along on it
                family = index;
                found = true;
                break;
            }
        }
        if (!found) return false;

        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(physical_, &props);
        buffer_device_address_ = false;
        VkPhysicalDeviceVulkan12Features features12 = {};
        features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceFeatures2 features2 = {};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &features12;
        if (props.apiVersion >= VK_API_VERSION_1_2) {
            vkGetPhysicalDeviceFeatures2(physical_, &features2);
            buffer_device_address_ = features12.bufferDeviceAddress == VK_TRUE; // decides whether the scratch buffer can be a VkDeviceAddress
        }

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue = {};
        queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue.queueFamilyIndex = family;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;

        VkDeviceCreateInfo create = {};
        create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create.queueCreateInfoCount = 1;
        create.pQueueCreateInfos = &queue;
        create.enabledLayerCount = 0;
        create.enabledExtensionCount = static_cast<uint32_t>(device_enabled.size());
        create.ppEnabledExtensionNames = device_enabled.empty() ? nullptr : device_enabled.data();
        if (buffer_device_address_) {
            features12.bufferDeviceAddress = VK_TRUE;
            create.pNext = &features2;
        } else {
            VkPhysicalDeviceFeatures features = {};
            create.pEnabledFeatures = &features;
        }
        if (vkCreateDevice(physical_, &create, nullptr, &device_) != VK_SUCCESS) return false; // a single queue is all this worker needs
        queue_family_ = family;
        vkGetDeviceQueue(device_, family, 0, &queue_);
        return true;
    }

    // One command pool, one primary command buffer, one fence: every GPU
    // operation in this worker is recorded into the same command buffer
    // and submitted with the same fence (see Begin/SubmitAndWait).
    bool CreateCommandObjects() {
        VkCommandPoolCreateInfo pool = {};
        pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // the pool (and its command buffer) is reset on every Begin()
        pool.queueFamilyIndex = queue_family_;
        if (vkCreateCommandPool(device_, &pool, nullptr, &pool_) != VK_SUCCESS) return false;
        VkCommandBufferAllocateInfo allocate = {};
        allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate.commandPool = pool_;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device_, &allocate, &cmd_) != VK_SUCCESS) return false;
        VkFenceCreateInfo fence = {};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        return vkCreateFence(device_, &fence, nullptr, &fence_) == VK_SUCCESS;
    }

    // Creates a 2D image, allocates device-local memory, wraps it in an
    // image view, and fills the NVSDK_NGX resource descriptor the runtime
    // copies from; 'read_write' marks images the runtime may write (the
    // output image).
    bool CreateImage(ImageRes &item, uint32_t width, uint32_t height, VkFormat format, bool read_write,
                     VkImageLayout layout, VkImageAspectFlags aspect) {
        VkImageCreateInfo create = {};
        create.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        create.imageType = VK_IMAGE_TYPE_2D;
        create.format = format;
        create.extent = {width, height, 1};
        create.mipLevels = 1;
        create.arrayLayers = 1;
        create.samples = VK_SAMPLE_COUNT_1_BIT;
        const bool is_depth = format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D16_UNORM_S8_UINT || // depth formats need optimal tiling and extra usage bits
            format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
        // The NVIDIA driver rejects linear-tiling depth images
        // (VK_ERROR_FEATURE_NOT_PRESENT), so depth uses optimal tiling.
        create.tiling = is_depth ? VK_IMAGE_TILING_OPTIMAL : VK_IMAGE_TILING_LINEAR;
        create.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT;
        if (is_depth)
            create.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        create.initialLayout = layout;
        const VkResult create_result = vkCreateImage(device_, &create, nullptr, &item.image);
        if (create_result != VK_SUCCESS) {
            Log("vkCreateImage failed: result=%d format=0x%X layout=%d usage=0x%X tiling=%d size=%ux%u",
                static_cast<int>(create_result), static_cast<unsigned>(format), static_cast<int>(layout),
                static_cast<unsigned>(create.usage), static_cast<unsigned>(create.tiling), width, height);
            return false;
        }

        VkMemoryRequirements requirements = {};
        vkGetImageMemoryRequirements(device_, item.image, &requirements);
        const uint32_t memory_type = FindMemoryType(physical_, requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); // images live in device-local memory
        if (memory_type == UINT32_MAX) {
            Log("no device-local memory type for image (bits 0x%X, size %llu)", static_cast<unsigned>(requirements.memoryTypeBits),
                static_cast<unsigned long long>(requirements.size));
            return false;
        }
        VkMemoryAllocateInfo allocate = {};
        allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memory_type;
        const VkResult allocated = vkAllocateMemory(device_, &allocate, nullptr, &item.memory);
        if (allocated != VK_SUCCESS) {
            Log("vkAllocateMemory failed: result=%d size=%llu type=%u", static_cast<int>(allocated),
                static_cast<unsigned long long>(allocate.allocationSize), memory_type);
            return false;
        }
        if (vkBindImageMemory(device_, item.image, item.memory, 0) != VK_SUCCESS) return false;

        VkImageViewCreateInfo view = {};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = item.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format;
        view.subresourceRange = {aspect, 0, 1, 0, 1}; // full 2D view over the selected aspect
        const VkResult view_result = vkCreateImageView(device_, &view, nullptr, &item.view);
        if (view_result != VK_SUCCESS) {
            Log("vkCreateImageView failed: result=%d format=0x%X aspect=0x%X size=%ux%u", static_cast<int>(view_result),
                static_cast<unsigned>(format), static_cast<unsigned>(aspect), width, height);
            return false;
        }

        item.format = format;
        item.layout = layout;
        item.aspect = aspect;
        item.width = width;
        item.height = height;
        std::memset(&item.resource, 0, sizeof(item.resource));
        item.resource.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW; // expose the image view to the NGX runtime
        item.resource.ReadWrite = read_write;
        item.resource.Resource.ImageViewInfo.ImageView = item.view;
        item.resource.Resource.ImageViewInfo.Image = item.image;
        item.resource.Resource.ImageViewInfo.SubresourceRange = view.subresourceRange;
        item.resource.Resource.ImageViewInfo.Format = format;
        item.resource.Resource.ImageViewInfo.Width = width;
        item.resource.Resource.ImageViewInfo.Height = height;
        return true;
    }

    // Creates a host-visible, host-coherent buffer and keeps it mapped for
    // the process lifetime: uploads are plain memcpy() into item.mapped,
    // with no per-frame map/unmap.
    bool CreateStaging(StagingBuffer &item, uint64_t size) {
        VkBufferCreateInfo create = {};
        create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create.size = size;
        create.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device_, &create, nullptr, &item.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements = {};
        vkGetBufferMemoryRequirements(device_, item.buffer, &requirements);
        const uint32_t memory_type = FindMemoryType(physical_, requirements,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); // host-visible + coherent: plain memcpy() is the whole upload path
        if (memory_type == UINT32_MAX) return false;
        VkMemoryAllocateInfo allocate = {};
        allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memory_type;
        if (vkAllocateMemory(device_, &allocate, nullptr, &item.memory) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(device_, item.buffer, item.memory, 0) != VK_SUCCESS) return false;
        if (vkMapMemory(device_, item.memory, 0, size, 0, &item.mapped) != VK_SUCCESS) return false; // mapped once and kept mapped until DestroyStaging
        item.size = size;
        return true;
    }

    // Allocates the device-local scratch buffer NGX asked for (if any) and
    // publishes it to the SDK: as a VkDeviceAddress when the feature is
    // available, otherwise as the raw buffer handle.
    bool CreateScratch(uint64_t size) {
        scratch_.size = size;
        VkBufferCreateInfo create = {};
        create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create.size = size;
        create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device_, &create, nullptr, &scratch_.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements = {};
        vkGetBufferMemoryRequirements(device_, scratch_.buffer, &requirements);
        const uint32_t memory_type = FindMemoryType(physical_, requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memory_type == UINT32_MAX) return false;
        VkMemoryAllocateInfo allocate = {};
        allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memory_type;
        if (vkAllocateMemory(device_, &allocate, nullptr, &scratch_.memory) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(device_, scratch_.buffer, scratch_.memory, 0) != VK_SUCCESS) return false;

        if (buffer_device_address_) {
            VkBufferDeviceAddressInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            info.buffer = scratch_.buffer;
            scratch_.value = static_cast<uint64_t>(vkGetBufferDeviceAddress(device_, &info)); // preferred path: the runtime dereferences a device address
            Log("NGX scratch buffer: %llu bytes (device address)", static_cast<unsigned long long>(size));
        } else {
            scratch_.value = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(scratch_.buffer)); // older runtimes take the buffer handle itself
            Log("NGX scratch buffer: %llu bytes (buffer handle; device addresses unavailable)",
                static_cast<unsigned long long>(size));
        }
        parameters_->Set(NVSDK_NGX_Parameter_Scratch, static_cast<unsigned long long>(scratch_.value));
        parameters_->Set(NVSDK_NGX_Parameter_Scratch_SizeInBytes, static_cast<unsigned long long>(size));
        return true;
    }

    // Resets the command pool and starts recording into the single shared
    // command buffer; pairs with SubmitAndWait().
    bool Begin() {
        if (vkResetCommandPool(device_, pool_, 0) != VK_SUCCESS) return false; // resetting the pool re-arms the shared command buffer
        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        return vkBeginCommandBuffer(cmd_, &begin) == VK_SUCCESS;
    }

    // Creates a small device-local buffer (the "disable interpolation"
    // reject-flag buffer) and exposes it to NGX as a VK_BUFFER resource.
    bool CreateDeviceBuffer(DeviceBuffer &item, uint64_t size, VkBufferUsageFlags usage) {
        VkBufferCreateInfo create = {};
        create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create.size = size;
        create.usage = usage;
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device_, &create, nullptr, &item.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements = {};
        vkGetBufferMemoryRequirements(device_, item.buffer, &requirements);
        const uint32_t memory_type = FindMemoryType(physical_, requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memory_type == UINT32_MAX) return false;
        VkMemoryAllocateInfo allocate = {};
        allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = memory_type;
        if (vkAllocateMemory(device_, &allocate, nullptr, &item.memory) != VK_SUCCESS) return false;
        if (vkBindBufferMemory(device_, item.buffer, item.memory, 0) != VK_SUCCESS) return false;
        item.size = size;
        // The D3D12 build hands the runtime a 4-byte buffer for the
        // "disable interpolation" flag; mirror that with a Vulkan buffer
        // resource instead of an image.
        std::memset(&item.resource, 0, sizeof(item.resource));
        item.resource.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_BUFFER;
        item.resource.ReadWrite = true; // the runtime writes its reject flag into this buffer
        item.resource.Resource.BufferInfo.Buffer = item.buffer;
        item.resource.Resource.BufferInfo.SizeInBytes = static_cast<unsigned int>(size);
        return true;
    }

    // GPU-side copy of source.size bytes from a staging buffer into a device buffer.
    void CopyBuffer(const StagingBuffer &source, DeviceBuffer &destination) {
        VkBufferCopy copy = {0, 0, static_cast<uint64_t>(source.size)};
        vkCmdCopyBuffer(cmd_, source.buffer, destination.buffer, 1, &copy);
    }

    // Copies the 4-byte reject flag out of the device buffer (the runtime always
    // writes it at offset 0) into the readback staging buffer at destination_offset.
    void CopyBuffer(const DeviceBuffer &source, StagingBuffer &destination, uint64_t destination_offset) {
        VkBufferCopy copy = {0, destination_offset, 4};
        vkCmdCopyBuffer(cmd_, source.buffer, destination.buffer, 1, &copy);
    }

    // Ends the command buffer, submits it to the single queue, and blocks
    // until the fence signals (60 s timeout); the fence is then reset so the
    // same pipeline can be replayed on the next call.
    bool SubmitAndWait() {
        const VkResult end = vkEndCommandBuffer(cmd_);
        if (end != VK_SUCCESS) {
            Log("vkEndCommandBuffer failed: %d", static_cast<int>(end));
            return false;
        }
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd_;
        const VkResult submitted = vkQueueSubmit(queue_, 1, &submit, fence_); // one submission per Upload/EvaluateGroup; the fence serializes CPU and GPU
        if (submitted != VK_SUCCESS) {
            Log("vkQueueSubmit failed: %d", static_cast<int>(submitted));
            return false;
        }
        const VkResult wait = vkWaitForFences(device_, 1, &fence_, VK_TRUE, kQueueTimeoutNs); // bounded wait so a hung GPU cannot wedge the worker
        if (wait != VK_SUCCESS) {
            Log("vkWaitForFences failed: %d%s", static_cast<int>(wait), wait == VK_TIMEOUT ? " (timed out after 60s)" : "");
            return false;
        }
        const VkResult reset = vkResetFences(device_, 1, &fence_);
        if (reset != VK_SUCCESS) {
            Log("vkResetFences failed: %d", static_cast<int>(reset));
            return false;
        }
        return true;
    }

    // Uploads a fully mapped staging buffer into an image (rows are packed contiguously, so bufferRowLength == width).
    void CopyStagingToImage(const StagingBuffer &source, ImageRes &destination) {
        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = destination.width;
        region.bufferImageHeight = destination.height;
        region.imageSubresource = {destination.aspect, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {destination.width, destination.height, 1};
        vkCmdCopyBufferToImage(cmd_, source.buffer, destination.image, destination.layout, 1, &region);
    }

    // Downloads an image into the staging buffer at offset (used to collect all generated frames of a group in one buffer).
    void CopyImageToStaging(const ImageRes &source, StagingBuffer &destination, uint64_t offset, uint32_t row_length, uint32_t height) {
        VkBufferImageCopy region = {};
        region.bufferOffset = offset;
        region.bufferRowLength = row_length;
        region.bufferImageHeight = height;
        region.imageSubresource = {source.aspect, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {row_length, height, 1};
        vkCmdCopyImageToBuffer(cmd_, source.image, source.layout, destination.buffer, 1, &region);
    }

    // Frees view, image, and memory; safe to call on an already-freed resource.
    void DestroyImage(ImageRes &item) {
        if (item.view != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, item.view, nullptr);
            item.view = VK_NULL_HANDLE;
        }
        if (item.image != VK_NULL_HANDLE) {
            vkDestroyImage(device_, item.image, nullptr);
            item.image = VK_NULL_HANDLE;
        }
        if (item.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, item.memory, nullptr);
            item.memory = VK_NULL_HANDLE;
        }
    }

    // Unmaps (when still mapped) and frees the staging buffer and its memory.
    void DestroyStaging(StagingBuffer &item) {
        if (item.mapped != nullptr) {
            vkUnmapMemory(device_, item.memory);
            item.mapped = nullptr;
        }
        if (item.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, item.buffer, nullptr);
            item.buffer = VK_NULL_HANDLE;
        }
        if (item.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, item.memory, nullptr);
            item.memory = VK_NULL_HANDLE;
        }
    }

    // Tears everything down in reverse creation order: the NGX feature and
    // parameters first (the runtime must be released before the Vulkan
    // device it is bound to goes away), then images/buffers, command
    // objects, device, and instance. Idempotent.
    void Shutdown() {
        if (feature_ != nullptr) {
            NVSDK_NGX_VULKAN_ReleaseFeature(feature_);
            feature_ = nullptr;
        }
        if (parameters_ != nullptr) {
            NVSDK_NGX_VULKAN_DestroyParameters(parameters_);
            parameters_ = nullptr;
        }
        if (ngx_initialized_) NVSDK_NGX_VULKAN_Shutdown1(device_); // detach the runtime before the Vulkan device is destroyed
        ngx_initialized_ = false;
        DestroyImage(color_);
        DestroyImage(motion_);
        DestroyImage(depth_);
        DestroyImage(output_);
        if (disable_.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, disable_.buffer, nullptr);
            disable_.buffer = VK_NULL_HANDLE;
        }
        if (disable_.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, disable_.memory, nullptr);
            disable_.memory = VK_NULL_HANDLE;
        }
        DestroyStaging(color_upload_);
        DestroyStaging(motion_upload_);
        DestroyStaging(depth_upload_);
        DestroyStaging(disable_zero_);
        DestroyStaging(output_readback_);
        DestroyStaging(disable_readback_);
        if (scratch_.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, scratch_.buffer, nullptr);
            scratch_.buffer = VK_NULL_HANDLE;
        }
        if (scratch_.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, scratch_.memory, nullptr);
            scratch_.memory = VK_NULL_HANDLE;
        }
        if (cmd_ != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device_, pool_, 1, &cmd_);
            cmd_ = VK_NULL_HANDLE;
        }
        if (pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, pool_, nullptr);
            pool_ = VK_NULL_HANDLE;
        }
        if (fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(device_, fence_, nullptr);
            fence_ = VK_NULL_HANDLE;
        }
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
    }

    // Logs a failure reason and reports initialization failure to the caller.
    bool Fail(const char *message) {
        Log("%s", message);
        return false;
    }

    // Vulkan handles: instance -> device -> queue -> command objects.
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    // Capabilities and the NGX device-extension list cached by CreateInstance().
    bool buffer_device_address_ = false;
    const char **ngx_device_exts_ = nullptr;
    unsigned int ngx_device_ext_count_ = 0;
    // NGX runtime state.
    NVSDK_NGX_Parameter *parameters_ = nullptr;
    NVSDK_NGX_Handle *feature_ = nullptr;
    bool ngx_initialized_ = false;
    // Probe results and per-session parameters.
    bool available_ = false;
    std::string runtime_version_ = "unknown";
    std::string device_name_ = "unknown";
    uint32_t maximum_ = 1;
    uint32_t width_ = 0, height_ = 0, generated_count_ = 1;
    // Per-session GPU resources (created by CreateFeature).
    ImageRes color_, motion_, depth_, output_; // backbuffer (RGBA8), motion vectors (RG16F), seeded depth, generated output
    DeviceBuffer disable_; // 4-byte "reject this frame" flag written by the runtime
    StagingBuffer color_upload_, motion_upload_, depth_upload_, disable_zero_; // host-mapped upload paths; disable_zero_ seeds the flag buffer with zero
    StagingBuffer output_readback_, disable_readback_; // readback of generated frames and their reject flags
    // The NGX scratch buffer: handle plus the value published to the SDK (device address, or raw handle as a fallback).
    struct {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        uint64_t value = 0;
        uint64_t size = 0;
    } scratch_;
};

// Emits the capability report as one line of JSON on stdout and returns
// 0 (frame generation usable) or 2 (unavailable). The report is printed
// even when initialization failed, so WanGP always receives a structured
// unavailability reason.
int Probe(Worker &worker, bool initialized) {
    if (!initialized) { // a failed init still gets a structured JSON report
        printf("{\"available\":false,\"multi_frame_count_max\":%u,\"runtime_version\":\"%s\",\"worker_version\":\"2\",\"detail\":\"Vulkan/NGX initialization failed; see stderr for diagnostics.\"}\n",
            worker.maximum(), worker.runtime_version().c_str());
        return 2;
    }
    printf("{\"available\":%s,\"multi_frame_count_max\":%u,\"runtime_version\":\"%s\",\"worker_version\":\"2\",\"detail\":\"Open Vulkan NGX capability query completed on %s.\"}\n",
        worker.available() ? "true" : "false", worker.maximum(), worker.runtime_version().c_str(), worker.device_name().c_str());
    return worker.available() ? 0 : 2; // 0: frame generation usable, 2: unavailable
}

// The stdin/stdout serving loop. The host sends one SETUP request
// (resolution, number of input frames, frames to generate per input pair);
// the worker answers with a SETUP result and then processes frame_count
// FRAME exchanges (header, RGBA backbuffer, RG16F motion vectors each).
// Per-frame status: 5 upload failed, 6 evaluation failed; the process
// exits 4/7 on protocol read/write failures.
int Serve(Worker &worker) {
    SetupHeader setup = {};
    // Hard sanity bounds keep a malformed host from exhausting the worker.
    if (!ReadExact(&setup, sizeof(setup)) || setup.magic != SETUP_MAGIC || setup.width < 64 || setup.height < 64 ||
        setup.width > 7680 || setup.height > 4320 || setup.frame_count == 0 || setup.generated_count == 0)
        return 2;
    uint32_t status = 0;
    if (!worker.available()) status = 1; // 1: the runtime cannot do frame generation on this GPU
    else if (setup.generated_count > worker.maximum()) status = 2; // 2: the host asked for more generated frames than supported
    else if (!worker.CreateFeature(setup.width, setup.height, setup.generated_count)) status = 3; // 3: the DLSSG feature could not be created
    const SetupResult setup_result = {SETUP_OUT_MAGIC, status, worker.maximum(), 0}; // the host waits for this reply before sending frames
    if (!WriteExact(&setup_result, sizeof(setup_result)) || status != 0) return status ? static_cast<int>(status) : 3;

    const size_t frame_bytes = static_cast<size_t>(setup.width) * setup.height * 4; // both inputs are 4 bytes per pixel (RGBA backbuffer, RG16F motion)
    std::vector<uint8_t> rgba(frame_bytes), motion(frame_bytes);
    for (uint32_t frame_number = 0; frame_number < setup.frame_count; ++frame_number) { // one iteration per input frame
        FrameHeader frame = {};
        // A truncated or malformed frame ends the session (exit code 4).
        if (!ReadExact(&frame, sizeof(frame)) || frame.magic != FRAME_MAGIC || !ReadExact(rgba.data(), rgba.size()) ||
            !ReadExact(motion.data(), motion.size()))
            return 4;
        uint32_t frame_status = 0, generated_count = 0, disabled = 0;
        std::vector<std::vector<uint8_t>> outputs;
        if (!worker.Upload(rgba.data(), motion.data())) frame_status = 5; // 5: the GPU upload failed
        bool output_disabled = false;
        // Only evaluate when the upload succeeded; otherwise report the failure.
        const NVSDK_NGX_Result result = frame_status == 0
            ? worker.EvaluateGroup(frame.index, frame.reset != 0, outputs, output_disabled)
            : static_cast<NVSDK_NGX_Result>(0x7FFFFFFF);
        if (NVSDK_NGX_FAILED(result)) {
            frame_status = static_cast<uint32_t>(result);
            if (frame_status == 0 || frame_status == 1) frame_status = 6; // 6: the runtime evaluation failed (0/1 are not wire errors)
        }
        if (output_disabled) { // the runtime rejected this frame: send the flag, no pixels
            disabled = 1;
            outputs.clear();
        }
        if (frame.reset) outputs.clear(); // a reset frame only re-syncs the runtime; it produces no outputs
        generated_count = static_cast<uint32_t>(outputs.size());
        const FrameResult frame_result = {FRAME_OUT_MAGIC, frame_status, generated_count, disabled}; // result header first, then the pixels of every generated frame
        if (!WriteExact(&frame_result, sizeof(frame_result))) return 7;
        for (const auto &output : outputs)
            if (!WriteExact(output.data(), output.size())) return 7;
    }
    return 0;
}

} // namespace

// CLI entry point: exactly one of --probe (capability report) or --serve
// (frame serving) must be given; --gpu selects the physical device.
int main(int argc, char **argv) {
    bool probe = false, serve = false;
    int gpu_index = -1;
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--probe") == 0) probe = true;
        else if (strcmp(argv[index], "--serve") == 0) serve = true;
        else if (strcmp(argv[index], "--gpu") == 0 && index + 1 < argc) {
            char *end = nullptr;
            const long value = strtol(argv[++index], &end, 10);
            if (end == argv[index - 1] || *end != '\0') return 1;
            gpu_index = static_cast<int>(value);
        } else {
            fprintf(stderr, "Usage: dlssg-worker --probe|--serve [--gpu <index>]\n");
            return 1;
        }
    }
    if (probe == serve) { // exactly one mode (XOR) must be given
        fprintf(stderr, "Usage: dlssg-worker --probe|--serve [--gpu <index>]\n");
        return 1;
    }
    Worker worker;
    const bool initialized = worker.Initialize(gpu_index); // both modes share the same Vulkan/NGX setup
    if (probe) return Probe(worker, initialized); // --probe works even after a failed init (structured reason on stdout)
    if (!initialized) return 2;
    return Serve(worker);
}
