// Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
// CPU products of quantized linear weights read in place (their mmap or pinned storage): y_i[rows, n_i] = x[rows, k] . W_i[n_i, k]^T for the
// weights of a call, which share their input: the linears of the blocks that MMGP leaves on the CPU (cpuCompute).
// - GGUF Q4_K, Q5_K, Q6_K, Q8_0, IQ2_S, IQ3_S: ggml's dot products (ggml_x86_port.c, MIT) against activations quantized as ggml does;
// - Prism PTQ1_0 (ternary, groups of 128): decoded as ggml's TQ1_0, against the same Q8_K activations;
// - INT8 rows (Quanto qint8, INT8 ConvRot), FP8 e4m3 / e5m2 rows, NVFP4: against float activations, a scale per row or tensor.
// x86-64 with AVX2, FMA and F16C; a pool of persistent workers.
#include "ggml_port.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <intrin.h>
#define NOMINMAX
#include <windows.h>
#define CPU_LINEAR_API extern "C" __declspec(dllexport)
#else
#include <cpuid.h>
#include <dlfcn.h>
#include <sched.h>
#define CPU_LINEAR_API extern "C" __attribute__((visibility("default")))
#endif

enum Format { FORMAT_GGUF = 0, FORMAT_INT8_ROWS = 1, FORMAT_FP8_ROWS = 2, FORMAT_NVFP4 = 3 };
enum { GGML_Q8_0 = 8, GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_Q6_K = 14, GGML_IQ3_XXS = 18, GGML_IQ4_NL = 20, GGML_IQ3_S = 21, GGML_IQ2_S = 22, GGML_IQ4_XS = 23, GGML_Q2_0 = 42, PRISM_PTQ1_0 = 143 };
enum { FP8_E4M3 = 0, FP8_E5M2 = 1 };
enum { NVFP4_LEGACY = 0, NVFP4_TENSORCORE = 1 };
enum { ROWS_E5M2 = 12 };  // internal: FP8 e5m2 rows (FORMAT_FP8_ROWS rows are e4m3)

// One weight of a call (ctypes structure of shared/kernels/cpu_linear/__init__.py).
struct Weight {
    const void * data;       // rows [n, k] in the format (NVFP4: k / 2 bytes per row)
    const void * scales;     // INT8 / FP8: float32 per row, or null for `scale`; NVFP4: the e4m3 block scales, swizzled in 128 x 4 tiles
    float * out;             // float32 [rows, n]
    int64_t n;
    int32_t format;
    int32_t type;            // GGUF: ggml type; FP8: FP8_E4M3 / FP8_E5M2; NVFP4: NVFP4_LEGACY / NVFP4_TENSORCORE
    float scale;             // INT8 / FP8: scale of every row when `scales` is null; NVFP4: global factor
    int32_t reserved;
};

static inline int hsum8(__m256i a) {
    const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    const __m128i h = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    return _mm_cvtsi128_si32(_mm_add_epi32(h, _mm_shuffle_epi32(h, 1)));
}

static inline float hsum8f(__m256 a) {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    const __m128 h = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_movehdup_ps(h)));
}

// ---- activations quantized as ggml's reference quantize_row_q8_K (ggml-quants.c, MIT) ----------------------------------------------------
static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static void quantize_block_q8_K(const float * x, block_q8_K * y) {
    float max = 0, amax = 0;
    for (int j = 0; j < QK_K; ++j) {
        const float ax = fabsf(x[j]);
        if (ax > amax) {
            amax = ax;
            max = x[j];
        }
    }
    if (!amax) {
        memset(y, 0, sizeof(*y));
        return;
    }
    const float iscale = -127.f / max;
    for (int j = 0; j < QK_K; ++j) y->qs[j] = (int8_t)std::min(127, nearest_int(iscale * x[j]));
    for (int j = 0; j < QK_K / 16; ++j) {
        int sum = 0;
        for (int i = 0; i < 16; ++i) sum += y->qs[j * 16 + i];
        y->bsums[j] = (int16_t)sum;
    }
    y->d = 1 / iscale;
}

// ---- Prism PTQ1_0 ---------------------------------------------------------------------------------------------------------------------
struct block_ptq1_0 { uint8_t qs[24]; uint8_t qh[2]; ggml_half d; };  // 128 values: 5 trits per qs byte, 4 per qh byte
static_assert(sizeof(block_ptq1_0) == 28, "PTQ1_0 block");

// trits u in {0, 1, 2} (weight u - 1) of bytes multiplied by 3^n, as ggml's TQ1_0: the top 2 bits of 3 * (q * 3^n mod 256)
static inline __m128i trits(__m128i q, int n) {
    for (int i = 0; i < n; ++i) q = _mm_add_epi8(q, _mm_add_epi8(q, q));
    q = _mm_subs_epu8(q, _mm_set1_epi8(1));
    q = _mm_avg_epu8(q, _mm_avg_epu8(q, _mm_setzero_si128()));
    return _mm_and_si128(_mm_srli_epi16(q, 6), _mm_set1_epi8(3));
}

// a weight row decoded once (n trits in Prism's order, a scale per group of 128), then dotted with every activation row
static void ptq1_0_decode_row(const block_ptq1_0 * x, int64_t n, uint8_t * u, float * d) {
    alignas(16) uint8_t tail[16];
    static const uint8_t pow3[4] = {1, 3, 9, 27};
    for (int64_t g = 0; g < n / 128; ++g, u += 128) {
        const block_ptq1_0 & b = x[g];
        d[g] = GGML_CPU_FP16_TO_FP32(b.d);
        // 16 bytes x 5 trits (values 0-79), 8 bytes x 5 trits (80-119), 2 bytes x 4 trits (120-127)
        const __m128i lo = _mm_loadu_si128((const __m128i *)b.qs);
        for (int t = 0; t < 5; ++t) _mm_storeu_si128((__m128i *)(u + 16 * t), trits(lo, t));
        const __m128i mid = _mm_loadl_epi64((const __m128i *)(b.qs + 16));
        for (int t = 0; t < 5; ++t) {
            _mm_store_si128((__m128i *)tail, trits(mid, t));
            memcpy(u + 80 + 8 * t, tail, 8);
        }
        for (int t = 0; t < 4; ++t)
            for (int j = 0; j < 2; ++j) u[120 + 2 * t + j] = (uint8_t)(((uint16_t)(uint8_t)(b.qh[j] * pow3[t]) * 3) >> 8);
    }
}

static float ptq1_0_dot_decoded(const uint8_t * u, const float * d, int64_t n, const block_q8_K * y) {
    float sum = 0;
    for (int64_t g = 0; g < n / 128; ++g) {
        const block_q8_K & a = y[g / 2];
        const int8_t * q8 = a.qs + (g % 2) * 128;
        __m256i acc = _mm256_setzero_si256();
        for (int c = 0; c < 4; ++c) {
            const __m256i p = _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i *)(u + g * 128 + 32 * c)), _mm256_loadu_si256((const __m256i *)(q8 + 32 * c)));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
        }
        int q8sum = 0;
        for (int j = 0; j < 8; ++j) q8sum += a.bsums[(g % 2) * 8 + j];
        sum += d[g] * a.d * (float)(hsum8(acc) - q8sum);  // sum of (u - 1) * q8
    }
    return sum;
}

// ---- INT8 / FP8 rows against float activations ---------------------------------------------------------------------------------------
// 8 weights as floats; FP8 e4m3: its bits shifted into FP16 position, 2^8 below the value (folded into the row scale); e5m2: the top byte of FP16
template <int F>
static inline __m256 load8(const uint8_t * w) {
    if constexpr (F == FORMAT_INT8_ROWS) return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)w)));
    const __m128i v = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)w));
    if constexpr (F == ROWS_E5M2) return _mm256_cvtph_ps(_mm_slli_epi16(v, 8));
    return _mm256_cvtph_ps(_mm_or_si128(_mm_slli_epi16(_mm_and_si128(v, _mm_set1_epi16(0x7F)), 7), _mm_slli_epi16(_mm_and_si128(v, _mm_set1_epi16(0x80)), 8)));
}

template <int F>
static inline float load1(const uint8_t * w) {
    if constexpr (F == FORMAT_INT8_ROWS) return (float)(int8_t)w[0];
    uint8_t padded[8] = {w[0]};
    return _mm256_cvtss_f32(load8<F>(padded));
}

// out[r] = dot(weight row, x[r]) for R activation rows, each 8 weights decoded once
template <int R, int F>
static void rows_dot(const uint8_t * w, const float * x, int64_t k, float * out) {
    __m256 acc[R];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= k; i += 8) {
        const __m256 wv = load8<F>(w + i);
        for (int r = 0; r < R; ++r) acc[r] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + r * k + i), acc[r]);
    }
    for (int r = 0; r < R; ++r) {
        float s = hsum8f(acc[r]);
        for (int64_t j = i; j < k; ++j) s += load1<F>(w + j) * x[r * k + j];
        out[r] = s;
    }
}

// NVFP4: 16 values per e4m3 block scale, E2M1 values doubled into int8 by a byte shuffle
static float e4m3_table[256];

static void init_e4m3_table() {
    for (int b = 0; b < 256; ++b) {
        const int e = (b >> 3) & 15, m = b & 7;
        const float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(m / 8.f, -6);
        e4m3_table[b] = (b & 0x80) ? -v : v;
    }
}

template <int R>
static void nvfp4_rows_dot(const uint8_t * w, const uint8_t * scales, int64_t row, int64_t k, bool tensorcore, const float * x, float * out) {
    const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int64_t k_tiles = k / 64, row_base = (row / 128) * k_tiles, r32 = row % 32, r4 = (row % 128) / 32;
    __m256 acc[R];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    for (int64_t g = 0; g < k / 16; ++g) {
        const float s = 0.5f * e4m3_table[scales[(((row_base + g / 4) * 32 + r32) * 4 + r4) * 4 + g % 4]];
        const __m128i bytes = _mm_loadl_epi64((const __m128i *)(w + g * 8));
        const __m128i lo = _mm_and_si128(bytes, m4), hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), m4);
        const __m128i v = _mm_shuffle_epi8(lut, tensorcore ? _mm_unpacklo_epi8(hi, lo) : _mm_unpacklo_epi8(lo, hi));
        const __m256 scale = _mm256_set1_ps(s);
        const __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)), scale);
        const __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8))), scale);
        for (int r = 0; r < R; ++r) {
            acc[r] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + r * k + g * 16), acc[r]);
            acc[r] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + r * k + g * 16 + 8), acc[r]);
        }
    }
    for (int r = 0; r < R; ++r) out[r] = hsum8f(acc[r]);
}

// the dot products of one weight row with every activation row, by tiles of 8 rows (accumulators in registers)
template <int F>
static void row_any(const Weight & w, int64_t j, const float * x, int64_t k, int64_t rows, float * out) {
    const uint8_t * row = (const uint8_t *)w.data + j * (F == FORMAT_NVFP4 ? k / 2 : k);
    for (int64_t r0 = 0; r0 < rows; r0 += 8) {
        const float * xr = x + r0 * k;
        float * o = out + r0;
#define ROWS_CASE(N) case N: if constexpr (F == FORMAT_NVFP4) nvfp4_rows_dot<N>(row, (const uint8_t *)w.scales, j, k, w.type == NVFP4_TENSORCORE, xr, o); \
                             else rows_dot<N, F>(row, xr, k, o); break;
        switch (std::min<int64_t>(8, rows - r0)) {
            ROWS_CASE(1) ROWS_CASE(2) ROWS_CASE(3) ROWS_CASE(4) ROWS_CASE(5) ROWS_CASE(6) ROWS_CASE(7) default: ROWS_CASE(8)
        }
#undef ROWS_CASE
    }
}

// ---- Q2_0 (ggml's format; its CPU backend has only the scalar product) ------------------------------------------------------------------
// 64 two-bit codes per block, value (code - 1) * d, against two Q8_0 blocks of activations: integer dots by groups of 32, as the scalar product
static void vec_dot_q2_0_q8_0(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    const block_q2_0 * x = (const block_q2_0 *)vx;
    const block_q8_0 * y = (const block_q8_0 *)vy;
    const int nb = n / QK2_0;
    const __m256i repeat = _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
    const __m256i low = _mm256_set1_epi32(0x40100401), high = _mm256_set1_epi32((int)0x80200802), one = _mm256_set1_epi8(1), pairs = _mm256_set1_epi16(1);
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);
        for (int k = 0; k < 2; ++k) {
            const block_q8_0 & b = y[i * 2 + k];
            const __m128i codes = _mm_loadl_epi64((const __m128i *)(x[i].qs + k * 8));
            const __m256i spread = _mm256_shuffle_epi8(_mm256_set_m128i(codes, codes), repeat);  // each byte 4 times; the high half reads bytes 4..7 of its lane
            const __m256i b0 = _mm256_cmpeq_epi8(_mm256_and_si256(spread, low), low), b1 = _mm256_cmpeq_epi8(_mm256_and_si256(spread, high), high);
            const __m256i value = _mm256_sub_epi8(_mm256_sub_epi8(_mm256_setzero_si256(), _mm256_add_epi8(b0, _mm256_add_epi8(b1, b1))), one);  // code - 1 in {-1, 0, 1, 2}
            const __m256i q = _mm256_loadu_si256((const __m256i *)b.qs);
            const __m256i dots = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(value, value), _mm256_sign_epi8(q, value)), pairs);
            sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dots), _mm256_set1_ps(d0 * GGML_CPU_FP16_TO_FP32(b.d)), sum);
        }
    }
    *s = hsum8f(sum);
}

// ---- dispatch ---------------------------------------------------------------------------------------------------------------------------
typedef void (*vec_dot_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

struct GgufType {
    int type, block_values, block_bytes;
    bool q8_0;          // activations quantized as Q8_0 (else Q8_K)
    vec_dot_t dot;      // null: PTQ1_0
};

static const GgufType gguf_types[] = {
    {GGML_Q8_0, 32, 34, true, ggml_vec_dot_q8_0_q8_0},
    {GGML_Q4_K, 256, 144, false, ggml_vec_dot_q4_K_q8_K},
    {GGML_Q5_K, 256, 176, false, ggml_vec_dot_q5_K_q8_K},
    {GGML_Q6_K, 256, 210, false, ggml_vec_dot_q6_K_q8_K},
    {GGML_IQ3_S, 256, 110, false, ggml_vec_dot_iq3_s_q8_K},
    {GGML_IQ2_S, 256, 82, false, ggml_vec_dot_iq2_s_q8_K},
    {GGML_IQ3_XXS, 256, 98, false, ggml_vec_dot_iq3_xxs_q8_K},
    {GGML_IQ4_XS, 256, 136, false, ggml_vec_dot_iq4_xs_q8_K},
    {GGML_IQ4_NL, 32, 18, true, ggml_vec_dot_iq4_nl_q8_0},
    {GGML_Q2_0, 64, 18, true, vec_dot_q2_0_q8_0},
    {PRISM_PTQ1_0, 128, 28, false, nullptr},
};

static const GgufType * gguf_type(int type) {
    for (const GgufType & t : gguf_types)
        if (t.type == type) return &t;
    return nullptr;
}

static bool cpu_ok() {
    static int ok = -1;
    if (ok < 0) {
#ifdef _WIN32
        int info[4];
        __cpuid(info, 1);
        const bool fma = info[2] & (1 << 12), f16c = info[2] & (1 << 29), osxsave = info[2] & (1 << 27);
        __cpuidex(info, 7, 0);
        const bool avx2 = info[1] & (1 << 5);
#else
        unsigned a, b, c, d;
        __cpuid(1, a, b, c, d);
        const bool fma = c & (1 << 12), f16c = c & (1 << 29), osxsave = c & (1 << 27);
        __cpuid_count(7, 0, a, b, c, d);
        const bool avx2 = b & (1 << 5);
#endif
        ok = fma && f16c && osxsave && avx2;
        if (ok) init_e4m3_table();
    }
    return ok;
}

static bool weight_ok(const Weight & w, int64_t k) {
    switch (w.format) {
        case FORMAT_GGUF: {
            const GgufType * t = gguf_type(w.type);
            return t != nullptr && k % t->block_values == 0;
        }
        case FORMAT_INT8_ROWS: return true;
        case FORMAT_FP8_ROWS: return w.type == FP8_E4M3 || w.type == FP8_E5M2;
        case FORMAT_NVFP4: return k % 64 == 0 && (w.type == NVFP4_LEGACY || w.type == NVFP4_TENSORCORE);
    }
    return false;
}

// ---- workers ----------------------------------------------------------------------------------------------------------------------------
// Persistent workers, the caller being worker 0 (with OpenMP's, pinned to cores during each call, the caller too, the Python thread ran its
// host work 0.4 ms slower per block of a 27B). After a call they spin for SPIN, then sleep: the linears of a block come one after the other
// with GPU work in between (norms, attention), and waking sleeping threads at every call costs more than the spin. They are not pinned:
// pinned for good, they waited for the caller whenever it ran on their core (27B decoding 1.8x slower).
static const std::chrono::microseconds SPIN(2000);

static inline void cpu_relax() { _mm_pause(); }

struct Pool {
    std::atomic<uint64_t> generation{0};
    std::atomic<int> running{0}, sleeping{0};
    std::mutex mutex;
    std::condition_variable wake;
    void (*job)(void *, int) = nullptr;
    void * context = nullptr;
    int size = 0;  // workers + the caller
};

static void worker_main(Pool * pool, int index) {
    uint64_t seen = 0;  // the generation when the pool was created: a worker starting late still runs the first call
    for (;;) {
        uint64_t current = seen;
        const auto start = std::chrono::steady_clock::now();
        while (current == seen) {
            for (int i = 0; i < 64 && (current = pool->generation.load(std::memory_order_acquire)) == seen; ++i) cpu_relax();
            if (current == seen && std::chrono::steady_clock::now() - start > SPIN) break;
        }
        if (current == seen) {
            std::unique_lock<std::mutex> lock(pool->mutex);
            pool->sleeping.fetch_add(1);
            pool->wake.wait(lock, [&] { return (current = pool->generation.load()) != seen; });
            pool->sleeping.fetch_sub(1);
        }
        seen = current;
        pool->job(pool->context, index);
        pool->running.fetch_sub(1, std::memory_order_release);
    }
}

// the workers (created at the first call, never destroyed: they live until the process ends) and the caller
static Pool & workers(int threads) {
    static Pool * pool = nullptr;
    if (pool == nullptr) {
        pool = new Pool();
        pool->size = std::max(1, threads);
        for (int i = 1; i < pool->size; ++i) std::thread(worker_main, pool, i).detach();
    }
    return *pool;
}

static std::atomic<uint64_t> pool_wakes{0};  // calls that found sleeping workers
CPU_LINEAR_API uint64_t host_trace_wakes() { return pool_wakes.load(); }

// runs job(context, worker) on every worker of the pool and the caller (worker 0), returns once all are done
static void run_workers(Pool & pool, void (*job)(void *, int), void * context) {
    pool.job = job;
    pool.context = context;
    pool.running.store(pool.size - 1, std::memory_order_relaxed);
    pool.generation.fetch_add(1);
    if (pool.sleeping.load()) {
        pool_wakes.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(pool.mutex);
        pool.wake.notify_all();
    }
    job(context, 0);
    while (pool.running.load(std::memory_order_acquire)) cpu_relax();
}

// format of a weight (FORMAT_GGUF: type = ggml type): 1 when this CPU and library compute it
CPU_LINEAR_API int cpu_linear_supported(int format, int type) {
    if (!cpu_ok()) return 0;
    Weight w = {};
    w.format = format;
    w.type = type;
    return weight_ok(w, format == FORMAT_GGUF ? 256 * 128 : 64);
}

// one call: the activations quantized by every worker (a static split), then chunks of CHUNK output features of every weight taken in turn
static const int64_t CHUNK = 16;

struct Call {
    const float * x;
    int64_t rows, k;
    const Weight * weights;
    std::vector<int64_t> starts;  // first work item of each weight, then the item count
    int threads;
    bool need_q8_K, need_q8_0;
    block_q8_K * q8_K;
    block_q8_0 * q8_0;
    alignas(64) std::atomic<int64_t> next{0};  // own cache lines: the fields above are read by every worker at every item
    alignas(64) std::atomic<int> quantized{0};
};

static void run_call(void * context, int worker) {
    Call & c = *(Call *)context;
    if (worker >= c.threads) return;
    const int64_t rows = c.rows, k = c.k, nb_K = k / QK_K, nb_0 = k / QK8_0;
    const float * x = c.x;
    if (c.need_q8_K) {
        const int64_t blocks = rows * nb_K;
        for (int64_t b = blocks * worker / c.threads; b < blocks * (worker + 1) / c.threads; ++b) quantize_block_q8_K(x + b * QK_K, c.q8_K + b);
    }
    if (c.need_q8_0) {  // blocks of 32 values, contiguous across rows
        const int64_t blocks = rows * nb_0, b0 = blocks * worker / c.threads, b1 = blocks * (worker + 1) / c.threads;
        if (b1 > b0) quantize_row_q8_0(x + b0 * QK8_0, c.q8_0 + b0, (b1 - b0) * QK8_0);
    }
    if (c.need_q8_K || c.need_q8_0) {
        c.quantized.fetch_add(1, std::memory_order_acq_rel);
        while (c.quantized.load(std::memory_order_acquire) < c.threads) cpu_relax();
    }
    static thread_local std::vector<float> partial;
    static thread_local std::vector<uint8_t> trits_row;
    static thread_local std::vector<float> group_scales;
    partial.resize((size_t)rows);
    const int64_t items = c.starts.back(), * starts = c.starts.data();
    const Weight * weights = c.weights;
    const block_q8_K * q8_K = c.q8_K;
    const block_q8_0 * q8_0 = c.q8_0;
    for (int64_t item; (item = c.next.fetch_add(1, std::memory_order_relaxed)) < items;) {
        int i = 0;
        while (starts[i + 1] <= item) ++i;
        const Weight & w = weights[i];
        const int64_t j0 = (item - starts[i]) * CHUNK, j1 = std::min(w.n, j0 + CHUNK);
        if (w.format == FORMAT_GGUF) {
            const GgufType * t = gguf_type(w.type);
            const int64_t row_bytes = k / t->block_values * t->block_bytes;
            const uint8_t * data = (const uint8_t *)w.data;
            if (t->dot == nullptr) {  // PTQ1_0: each weight row decoded once, then dotted with every activation row
                trits_row.resize((size_t)k);
                group_scales.resize((size_t)(k / 128));
                for (int64_t j = j0; j < j1; ++j) {
                    ptq1_0_decode_row((const block_ptq1_0 *)(data + j * row_bytes), k, trits_row.data(), group_scales.data());
                    for (int64_t r = 0; r < rows; ++r) w.out[r * w.n + j] = ptq1_0_dot_decoded(trits_row.data(), group_scales.data(), k, q8_K + r * nb_K);
                }
                continue;
            }
            // as ggml: each activation row against the weight rows of the chunk, which stay in L2 (9% faster than the rows of each weight row)
            for (int64_t r = 0; r < rows; ++r) {
                const void * a = t->q8_0 ? (const void *)(q8_0 + r * nb_0) : (const void *)(q8_K + r * nb_K);
                for (int64_t j = j0; j < j1; ++j) t->dot((int)k, w.out + r * w.n + j, 0, data + j * row_bytes, 0, a, 0, 1);
            }
            continue;
        }
        for (int64_t j = j0; j < j1; ++j) {
            float scale = w.format != FORMAT_NVFP4 && w.scales != nullptr ? ((const float *)w.scales)[j] : w.scale;
            if (w.format == FORMAT_INT8_ROWS) {
                row_any<FORMAT_INT8_ROWS>(w, j, x, k, rows, partial.data());
            } else if (w.format == FORMAT_NVFP4) {
                row_any<FORMAT_NVFP4>(w, j, x, k, rows, partial.data());
            } else if (w.type == FP8_E5M2) {
                row_any<ROWS_E5M2>(w, j, x, k, rows, partial.data());
            } else {
                row_any<FORMAT_FP8_ROWS>(w, j, x, k, rows, partial.data());
                scale *= 256.f;  // e4m3 decoded 2^8 below its value
            }
            for (int64_t r = 0; r < rows; ++r) w.out[r * w.n + j] = partial[(size_t)r] * scale;
        }
    }
}

// x: float32 [rows, k], contiguous. Returns 0, or 1 for an unsupported weight (nothing computed). Calls from several threads take turns.
CPU_LINEAR_API int cpu_linear(const float * x, int64_t rows, int64_t k, int count, const Weight * weights, int threads) {
    if (!cpu_ok()) return 1;
    static std::mutex serial;
    std::lock_guard<std::mutex> guard(serial);
    Call c;
    c.x = x;
    c.rows = rows;
    c.k = k;
    c.weights = weights;
    c.need_q8_K = c.need_q8_0 = false;
    for (int i = 0; i < count; ++i) {
        if (!weight_ok(weights[i], k)) return 1;
        if (weights[i].format == FORMAT_GGUF) (gguf_type(weights[i].type)->q8_0 ? c.need_q8_0 : c.need_q8_K) = true;
    }
    static std::vector<block_q8_K> act_q8_K;
    static std::vector<block_q8_0> act_q8_0;
    if (c.need_q8_K) act_q8_K.resize((size_t)(rows * (k / QK_K)));
    if (c.need_q8_0) act_q8_0.resize((size_t)(rows * (k / QK8_0)));
    c.q8_K = act_q8_K.data();
    c.q8_0 = act_q8_0.data();
    c.starts.assign(count + 1, 0);
    for (int i = 0; i < count; ++i) c.starts[i + 1] = c.starts[i] + (weights[i].n + CHUNK - 1) / CHUNK;
    Pool & pool = workers(threads);
    c.threads = std::min(std::max(1, threads), pool.size);
    run_workers(pool, run_call, &c);
    return 0;
}

// ---- a diagnostic timeline of the host jobs -----------------------------------------------------------------------------------------------
// host_trace_start(buffer, capacity) fills caller's records until host_trace_stop(), which returns their count; nothing is recorded otherwise (one relaxed load per job).
struct HostStamp { int64_t start, end; int32_t kind, pairs; uint32_t thread, cpu; };  // ns of host_trace_now; kind: 0 routed experts, 1 table rows
static std::atomic<HostStamp *> host_stamps{nullptr};
static std::atomic<uint64_t> host_stamp_next{0};
static uint64_t host_stamp_capacity = 0;

static inline int64_t host_trace_clock() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

CPU_LINEAR_API int64_t host_trace_now() { return host_trace_clock(); }

CPU_LINEAR_API void host_trace_start(void * buffer, uint64_t capacity) {
    host_stamp_capacity = capacity;
    host_stamp_next.store(0);
    host_stamps.store((HostStamp *)buffer);
}

CPU_LINEAR_API uint64_t host_trace_stop() {
    host_stamps.store(nullptr);
    return std::min(host_stamp_next.load(), host_stamp_capacity);
}

static inline void host_trace(int kind, int64_t start, int64_t pairs) {
    HostStamp * stamps = host_stamps.load(std::memory_order_relaxed);
    if (stamps == nullptr) return;
    const uint64_t at = host_stamp_next.fetch_add(1);
    if (at >= host_stamp_capacity) return;
#ifdef _WIN32
    stamps[at] = {start, host_trace_clock(), kind, (int32_t)pairs, (uint32_t)GetCurrentThreadId(), (uint32_t)GetCurrentProcessorNumber()};
#else
    stamps[at] = {start, host_trace_clock(), kind, (int32_t)pairs, 0, (uint32_t)sched_getcpu()};
#endif
}

// A CUDA host function that spins for *(int64_t *)data nanoseconds: what a host job costs its graph, whatever it computes.
CPU_LINEAR_API void host_job_spin(void * data) {
    const int64_t until = host_trace_clock() + *(const int64_t *)data;
    while (host_trace_clock() < until) cpu_relax();
}

#include "host_server.inc"

// One linear as a job of the host server (ctypes structure of __init__.py): the same kernels as cpu_linear(), between a GPU kernel writing x to mapped RAM and another reading the result from it.
struct LinearJob {
    const float * x;
    int64_t rows, k;
    Weight weight;
    int32_t threads, status;
};

CPU_LINEAR_API int cpu_linear_server_job(void * data) {
    LinearJob & job = *(LinearJob *)data;
    const int64_t traced = host_stamps.load(std::memory_order_relaxed) == nullptr ? 0 : host_trace_clock();
    job.status = cpu_linear(job.x, job.rows, job.k, 1, &job.weight, job.threads);
    if (traced) host_trace(2, traced, job.rows);
    return 1;
}

#include "experts_gguf.inc"

static const int8_t kvalues_iq4nl_port[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};  // ggml's kvalues_iq4nl
#include "rows.inc"
