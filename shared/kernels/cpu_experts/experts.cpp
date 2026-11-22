// Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
// Routed experts on the CPU: float32 activations and accumulation, weights read from their checkpoint storage (no copies).
// Activations: SwiGLU (gate and up rows in w13), SwiGLU with soft caps (cap * tanh(v / cap) on the gate and on the up outputs) or
// squared ReLU (up rows only in w13).
// Formats: FP8 E4M3FN with float32 scales per block x block tile, symmetric INT4 (two values per byte, the even column in the low
// nibble, offset by 8) with FP16 scales per group of input columns, and INT8 rows with a float32 scale per row: gate rows then up
// rows. NVFP4 rows (two E2M1 codes per byte, the even column in the low nibble, an E4M3 scale per 16 values of a row and a float32
// factor per expert) and MXFP8 rows (E4M3 values, a scale 2^(byte - 127) per 32 values of a row): gate and up rows alternate. The
// scale block of w13 and of w2 may differ. An optional ConvRot rotation (a regular Hadamard transform of each group of input features, applied to the
// weights before quantization) is applied to the activations of both projections.
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <omp.h>
#ifdef _WIN32
#include <vector>
#include <intrin.h>
#include <windows.h>
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#if defined(__x86_64__)
// The oldest versions of the math functions: the library loads with any glibc PyTorch supports
__asm__(".symver expf,expf@GLIBC_2.2.5");
__asm__(".symver tanhf,tanhf@GLIBC_2.2.5");
#endif
#endif

enum { FP8_BLOCK = 0, INT4_GROUP = 1, INT8_ROWS = 2, NVFP4_ROWS = 3, MXFP8_ROWS = 4 };
enum { SWIGLU = 0, RELU2 = 1, SWIGLU_SOFTCAP = 2 };

#ifdef _WIN32
static std::vector<DWORD_PTR> worker_cpus() {
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    std::vector<char> buffer(bytes);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer.data(), &bytes)) return {};
    DWORD_PTR allowed, system;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &allowed, &system)) return {};
    std::vector<DWORD_PTR> primary, siblings;
    for (size_t offset = 0; offset < bytes;) {
        auto core = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer.data() + offset);
        for (WORD group = 0; group < core->Processor.GroupCount; ++group) {
            if (core->Processor.GroupMask[group].Group != 0) continue;
            DWORD_PTR mask = core->Processor.GroupMask[group].Mask & allowed;
            if (mask) {
                DWORD_PTR first = mask & (~mask + 1);
                primary.push_back(first);
                mask ^= first;
                while (mask) {
                    first = mask & (~mask + 1);
                    siblings.push_back(first);
                    mask ^= first;
                }
            }
        }
        offset += core->Size;
    }
    primary.insert(primary.end(), siblings.begin(), siblings.end());
    return primary;
}

// The logical processors (group 0) of the most performant cores, both threads of each: where the workers of a job may run when none is given a core of its own
static DWORD_PTR performance_cpus() {
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    std::vector<char> buffer(bytes);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer.data(), &bytes)) return 0;
    DWORD_PTR allowed, system, mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &allowed, &system)) return 0;
    int best = -1;
    for (int pass = 0; pass < 2; ++pass)
        for (size_t offset = 0; offset < bytes;) {
            auto core = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer.data() + offset);
            if (!pass) best = (std::max)(best, int(core->Processor.EfficiencyClass));
            else if (int(core->Processor.EfficiencyClass) == best)
                for (WORD group = 0; group < core->Processor.GroupCount; ++group)
                    if (core->Processor.GroupMask[group].Group == 0) mask |= core->Processor.GroupMask[group].Mask & allowed;
            offset += core->Size;
        }
    return mask;
}

// How the workers of a job are placed while it runs. 0: each on a core of its own (the first of each core, then their second threads). 1: all on the performance cores, on any
// of their threads. 2: wherever the system runs them. CPU_EXPERTS_AFFINITY=core|wide|none chooses for every format (a comparison); else `wide` is the format's own choice.
static int affinity_mode(bool wide) {
    static const int forced = [] {
        const char* value = std::getenv("CPU_EXPERTS_AFFINITY");
        return value == nullptr ? -1 : !std::strcmp(value, "core") ? 0 : !std::strcmp(value, "wide") ? 1 : !std::strcmp(value, "none") ? 2 : -1;
    }();
    return forced >= 0 ? forced : wide ? 1 : 0;
}
#endif

API int moe_supported() {
#ifdef _WIN32
    int r[4]; __cpuid(r, 1);
    if ((r[2] & ((1 << 12) | (1 << 27) | (1 << 28) | (1 << 29))) != ((1 << 12) | (1 << 27) | (1 << 28) | (1 << 29))) return 0;
    if ((_xgetbv(0) & 6) != 6) return 0;
    __cpuidex(r, 7, 0); return (r[1] & (1 << 5)) != 0;
#else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#endif
}

static inline __m256 decode8(const uint8_t* p) {
    // Every finite E4M3FN value embeds in FP16 (subnormals and signed zero included) as value / 256: the factor 256 is folded into
    // the block scale, a power of two, so results are those of exact decoding. NaN encodings (0x7F/0xFF) are not decoded: expert
    // checkpoints hold none (the prepared Drama 3 experts were verified).
    __m128i b = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)p));
    __m128i half = _mm_or_si128(_mm_slli_epi16(_mm_and_si128(b, _mm_set1_epi16(127)), 7), _mm_slli_epi16(_mm_and_si128(b, _mm_set1_epi16(128)), 8));
    return _mm256_cvtph_ps(half);
}

// Weight rows are read 32 columns at a time, as 4 x 8 floats: from their storage, or already decoded (Format::decode) when several
// tokens use them. Both run the same operations in the same order: the same results.
struct FloatRow {
    const float* p;
    void operator()(int k, __m256 (&w)[4]) const { for (int n = 0; n < 4; ++n) w[n] = _mm256_loadu_ps(p + k + 8 * n); }
};

struct Fp8Block {
    static const bool interleaved = false, wide_workers = false;
    using ScaleT = float;
    struct Row {
        const uint8_t* p;
        void operator()(int k, __m256 (&w)[4]) const { for (int n = 0; n < 4; ++n) w[n] = decode8(p + k + 8 * n); }
    };
    struct Scale { const float* p; float operator[](int g) const { return p[g] * 256.f; } };
    static size_t row_bytes(int cols) { return cols; }
    // Offset of the scales of a row: one scale per block x block tile
    static size_t scale_row(int row, int cols, int block) { return size_t(row / block) * (cols / block); }
};

struct Int4Group {
    static const bool interleaved = false, wide_workers = false;
    using ScaleT = uint16_t;  // FP16
    struct Row {
        const uint8_t* p;
        void operator()(int k, __m256 (&w)[4]) const {
            __m128i v = _mm_loadu_si128((const __m128i*)(p + k / 2)), mask = _mm_set1_epi8(15), eight = _mm_set1_epi8(8);
            __m128i lo = _mm_and_si128(v, mask), hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
            __m128i first = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), eight), second = _mm_sub_epi8(_mm_unpackhi_epi8(lo, hi), eight);
            w[0] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(first));
            w[1] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(first, 8)));
            w[2] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(second));
            w[3] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(second, 8)));
        }
    };
    struct Scale { const uint16_t* p; float operator[](int g) const { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(p[g]))); } };
    static size_t row_bytes(int cols) { return cols / 2; }
    // One scale per row and group of block columns
    static size_t scale_row(int row, int cols, int block) { return size_t(row) * (cols / block); }
};

struct Int8Rows {
    static const bool interleaved = false, wide_workers = false;
    using ScaleT = float;
    struct Row {
        const uint8_t* p;
        void operator()(int k, __m256 (&w)[4]) const {
            __m256i v = _mm256_loadu_si256((const __m256i*)(p + k));
            __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
            w[0] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
            w[1] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
            w[2] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
            w[3] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        }
    };
    struct Scale { const float* p; float operator[](int g) const { return p[g]; } };
    static size_t row_bytes(int cols) { return cols; }
    // One scale per row: the block is the row width
    static size_t scale_row(int row, int cols, int block) { return size_t(row) * (cols / block); }
};

// The projection of one expert (rows of cols columns): its weight rows and the scales of their sums by runs of dot_block() columns.
// Here the formats whose rows follow one another, with scales by tiles or groups of columns; below those whose rows carry their scales.
template <class F>
struct Matrix {
    const uint8_t* w; const typename F::ScaleT* s; int cols, block;
    typename F::Row row(int r) const { return {w + size_t(r) * F::row_bytes(cols)}; }
    typename F::Scale scale(int r) const { return {s + F::scale_row(r, cols, block)}; }
    static int dot_block(int, int block) { return block; }
    static Matrix of(const uint8_t* w, const void* s, const float*, size_t id, int rows, int cols, int block) {
        return {w + id * rows * F::row_bytes(cols), (const typename F::ScaleT*)s + id * F::scale_row(rows, cols, block), cols, block};
    }
};

// E4M3FN values, and the MXFP8 block scales 2^(byte - 127) times the 256 that decode8() leaves out
static const struct Tables {
    float fp8[256], mx[256];
    Tables() {
        for (int b = 0; b < 256; ++b) {
            int e = (b >> 3) & 15, m = b & 7;
            uint32_t bits = e ? uint32_t(e + 120) << 23 | uint32_t(m) << 20 : 0, scale = b + 8 < 255 ? uint32_t(b + 8) << 23 : 0x7F800000u;
            if ((b & 127) == 127) bits = 0x7FC00000u;
            float value; std::memcpy(&value, &bits, 4);
            if (!e) value = m * 0.001953125f;
            fp8[b] = b & 128 ? -value : value;
            std::memcpy(&mx[b], &scale, 4);
        }
    }
} tables;

struct Nvfp4Rows { static const bool interleaved = true, wide_workers = true; };
struct Mxfp8Rows { static const bool interleaved = true, wide_workers = true; };

template <>
struct Matrix<Nvfp4Rows> {
    const uint8_t* w; const uint8_t* s; float factor; int cols;
    // 32 columns: twice their E2M1 values (exact small integers) times the E4M3 scale of each half. The scale of the sums is the
    // expert's factor, halved.
    struct Row {
        const uint8_t* p; const uint8_t* s;
        void operator()(int k, __m256 (&w)[4]) const {
            const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12), mask = _mm_set1_epi8(15);
            __m128i v = _mm_loadu_si128((const __m128i*)(p + k / 2));
            __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(v, mask)), hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(v, 4), mask));
            __m128i first = _mm_unpacklo_epi8(lo, hi), second = _mm_unpackhi_epi8(lo, hi);
            __m256 s0 = _mm256_set1_ps(tables.fp8[s[k / 16]]), s1 = _mm256_set1_ps(tables.fp8[s[k / 16 + 1]]);
            w[0] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(first)), s0);
            w[1] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(first, 8))), s0);
            w[2] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(second)), s1);
            w[3] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(second, 8))), s1);
        }
    };
    struct Scale { float v; float operator[](int) const { return v; } };
    Row row(int r) const { return {w + size_t(r) * (cols / 2), s + size_t(r) * (cols / 16)}; }
    Scale scale(int) const { return {factor}; }
    static int dot_block(int cols, int) { return cols; }
    static Matrix of(const uint8_t* w, const void* s, const float* f, size_t id, int rows, int cols, int) {
        return {w + id * rows * (cols / 2), (const uint8_t*)s + id * rows * (cols / 16), f[id] * 0.5f, cols};
    }
};

template <>
struct Matrix<Mxfp8Rows> {
    const uint8_t* w; const uint8_t* s; int cols;
    using Row = Fp8Block::Row;
    struct Scale { const uint8_t* p; float operator[](int g) const { return tables.mx[p[g]]; } };
    Row row(int r) const { return {w + size_t(r) * cols}; }
    Scale scale(int r) const { return {s + size_t(r) * (cols / 32)}; }
    static int dot_block(int, int) { return 32; }
    static Matrix of(const uint8_t* w, const void* s, const float*, size_t id, int rows, int cols, int) {
        return {w + id * rows * cols, (const uint8_t*)s + id * rows * (cols / 32), cols};
    }
};

static inline float activate_relu2(float v) { v = v > 0.f ? v : 0.f; return v * v; }

static inline float activate(float gate, float up, int act, float cap_gate, float cap_up) {
    if (act == SWIGLU) return (gate / (1.f + std::exp(-gate))) * up;
    return cap_gate * std::tanh(gate / cap_gate) * (1.f / (1.f + std::exp(-gate))) * (cap_up * std::tanh(up / cap_up));
}

// An expert bank and its activation: block / block2 the scale blocks of w13 / w2, f13 / f2 the factors per expert (NVFP4 rows)
struct Bank {
    const uint8_t* w13; const uint8_t* w2; const void* s13; const void* s2; const float* f13; const float* f2;
    int format, rotation, act, h, i, e, block, block2; float cap_gate, cap_up;
};

template <class Row>
static inline void decode_row(Row row, float* out, int cols) {
    for (int k = 0; k < cols; k += 32) {
        __m256 v[4]; row(k, v);
        for (int n = 0; n < 4; ++n) _mm256_storeu_ps(out + k + 8 * n, v[n]);
    }
}

template <class Row, class Scale>
static inline float dot(Row w, const float* x, Scale scale, int cols, int block) {
    __m256 total = _mm256_setzero_ps();
    for (int k = 0; k < cols; k += block) {
        __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
        for (int j = 0; j < block; j += 32) {
            __m256 v[4]; w(k+j, v);
            a = _mm256_fmadd_ps(v[0], _mm256_loadu_ps(x+k+j), a);
            b = _mm256_fmadd_ps(v[1], _mm256_loadu_ps(x+k+j+8), b);
            c = _mm256_fmadd_ps(v[2], _mm256_loadu_ps(x+k+j+16), c);
            d = _mm256_fmadd_ps(v[3], _mm256_loadu_ps(x+k+j+24), d);
        }
        total = _mm256_fmadd_ps(_mm256_add_ps(_mm256_add_ps(a,b), _mm256_add_ps(c,d)), _mm256_set1_ps(scale[k/block]), total);
    }
    __m128 v = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
    v = _mm_hadd_ps(v,v); v = _mm_hadd_ps(v,v); return _mm_cvtss_f32(v);
}

// Two rows against the same activations (gate and up): each row accumulates exactly as dot() does.
template <class Row, class Scale>
static inline void dot2(Row w0, Row w1, const float* x, Scale s0, Scale s1, int cols, int block, float* r0, float* r1) {
    __m256 t0 = _mm256_setzero_ps(), t1 = t0;
    for (int k = 0; k < cols; k += block) {
        __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a, e = a, f = a, g = a, q = a;
        for (int j = 0; j < block; j += 32) {
            __m256 x0 = _mm256_loadu_ps(x+k+j), x1 = _mm256_loadu_ps(x+k+j+8), x2 = _mm256_loadu_ps(x+k+j+16), x3 = _mm256_loadu_ps(x+k+j+24);
            __m256 u[4], v[4]; w0(k+j, u); w1(k+j, v);
            a = _mm256_fmadd_ps(u[0], x0, a); b = _mm256_fmadd_ps(u[1], x1, b);
            c = _mm256_fmadd_ps(u[2], x2, c); d = _mm256_fmadd_ps(u[3], x3, d);
            e = _mm256_fmadd_ps(v[0], x0, e); f = _mm256_fmadd_ps(v[1], x1, f);
            g = _mm256_fmadd_ps(v[2], x2, g); q = _mm256_fmadd_ps(v[3], x3, q);
        }
        t0 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_add_ps(a,b), _mm256_add_ps(c,d)), _mm256_set1_ps(s0[k/block]), t0);
        t1 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_add_ps(e,f), _mm256_add_ps(g,q)), _mm256_set1_ps(s1[k/block]), t1);
    }
    __m128 v = _mm_add_ps(_mm256_castps256_ps128(t0), _mm256_extractf128_ps(t0, 1)); v = _mm_hadd_ps(v,v); v = _mm_hadd_ps(v,v); *r0 = _mm_cvtss_f32(v);
    v = _mm_add_ps(_mm256_castps256_ps128(t1), _mm256_extractf128_ps(t1, 1)); v = _mm_hadd_ps(v,v); v = _mm_hadd_ps(v,v); *r1 = _mm_cvtss_f32(v);
}

// ConvRot: v @ H in place for each group of n features, H = kron(h4, ..., h4) / sqrt(n) the regular Hadamard matrix (n a power of 4).
static void rotate(float* v, int features, int n) {
    const float norm = 1.f / std::sqrt(float(n));  // a power of two: exact
    for (int base = 0; base < features; base += n) {
        float* g = v + base;
        for (int stride = 1; stride < n; stride *= 4)
            for (int start = 0; start < n; start += 4 * stride)
                for (int j = start; j < start + stride; ++j) {
                    float a = g[j], b = g[j + stride], c = g[j + 2 * stride], d = g[j + 3 * stride];
                    g[j] = a + b + c - d; g[j + stride] = a + b - c + d; g[j + 2 * stride] = a - b + c + d; g[j + 3 * stride] = b + c + d - a;
                }
        for (int j = 0; j < n; ++j) g[j] *= norm;
    }
}

// ids: int32 [batch, topk]. An id of -1, or one marked in resident (uint8 per expert, may be null), is computed elsewhere.
// Returns the number of (token, expert) pairs computed here, or a negative error; out is left unwritten when there are none.
template <class F>
static int moe_forward_format(const Bank& b, const float* x, const int32_t* ids, const float* probabilities, const uint8_t* resident, float* out,
                              int batch, int topk, int threads) {
    using M = Matrix<F>;
    const int h = b.h, i = b.i, e = b.e, rotation = b.rotation, act = b.act;
    const int pairs = batch * topk, projections = act == RELU2 ? 1 : 2;
    // Rows of w13: gate rows then up rows, or gate and up alternating
    const int step = act != RELU2 && F::interleaved ? 2 : 1, up_row = F::interleaved ? 1 : i;
    const int block = M::dot_block(h, b.block), block2 = M::dot_block(i, b.block2);
    // Plain arrays rather than std::vector, whose GCC 11+ code needs a recent libstdc++
    try {
        std::unique_ptr<int[]> active(new int[pairs]);
        int count = 0;
        for (int p = 0; p < pairs; ++p) {
            if (ids[p] < -1 || ids[p] >= e) return -2;
            if (ids[p] >= 0 && !(resident && resident[ids[p]])) active[count++] = p;
        }
        if (!count) return 0;
        // When experts serve several tokens (prompt processing), the pairs go by expert: each weight row is read from memory once,
        // decoded once, and applied to all its tokens. A decoding step (one token) keeps reading the rows directly.
        std::unique_ptr<int[]> order(new int[count]), starts(new int[count + 1]);
        std::copy(active.get(), active.get() + count, order.get());
        std::stable_sort(order.get(), order.get() + count, [&](int a, int b) { return ids[a] < ids[b]; });
        int groups = 0;
        for (int k = 0; k < count; ++k)
            if (!k || ids[order[k]] != ids[order[k - 1]]) starts[groups++] = k;
        starts[groups] = count;
        // Every thread shares the rows of the pairs computed here; values stay zero for the pairs computed elsewhere.
        std::unique_ptr<float[]> mid(new float[size_t(count) * i]), values(new float[size_t(pairs) * h]()), rotated(new float[rotation ? size_t(batch) * h : 0]);
        // The decoded gate, up and down rows of each thread (grouped pairs)
        std::unique_ptr<float[]> scratch(new float[groups == count ? 0 : size_t(threads) * (2 * h + i)]);
        const float* xs = rotation ? rotated.get() : x;
#ifdef _WIN32
        static const auto cpus = worker_cpus();
        static const DWORD_PTR performance = performance_cpus();
        const int placed = affinity_mode(F::wide_workers);
#endif
        #pragma omp parallel num_threads(threads)
        {
#ifdef _WIN32
            int worker = omp_get_thread_num();
            DWORD_PTR previous = placed == 2 ? 0 : placed == 1 ? (performance ? SetThreadAffinityMask(GetCurrentThread(), performance) : 0) : worker < int(cpus.size()) ? SetThreadAffinityMask(GetCurrentThread(), cpus[worker]) : 0;
#endif
            if (rotation) {
                #pragma omp for schedule(static)
                for (int token = 0; token < batch; ++token) {
                    std::memcpy(rotated.get() + size_t(token) * h, x + size_t(token) * h, sizeof(float) * h);
                    rotate(rotated.get() + size_t(token) * h, h, rotation);
                }
            }
            if (groups == count) {
                #pragma omp for schedule(static)
                for (int task = 0; task < count * i; ++task) {
                    int pair = active[task / i], g = step * (task % i), u = g + up_row;
                    M m = M::of(b.w13, b.s13, b.f13, ids[pair], i * projections, h, b.block);
                    const float* input = xs + size_t(pair / topk) * h;
                    if (act == RELU2) {
                        mid[task] = activate_relu2(dot(m.row(g), input, m.scale(g), h, block));
                        continue;
                    }
                    float gate, up;
                    dot2(m.row(g), m.row(u), input, m.scale(g), m.scale(u), h, block, &gate, &up);
                    mid[task] = activate(gate, up, act, b.cap_gate, b.cap_up);
                }
            } else {
                float* rows = scratch.get() + size_t(omp_get_thread_num()) * (2 * h + i);
                // Experts serve different numbers of tokens: dynamic chunks balance the threads (each task writes its own outputs).
                #pragma omp for schedule(dynamic, 16)
                for (int task = 0; task < groups * i; ++task) {
                    int group = task / i, row = task % i, first = starts[group], g = step * row, u = g + up_row;
                    M m = M::of(b.w13, b.s13, b.f13, ids[order[first]], i * projections, h, b.block);
                    auto s0 = m.scale(g);
                    float gate, up;
                    if (act == RELU2) {
                        if (starts[group + 1] - first == 1) {
                            mid[size_t(first)*i + row] = activate_relu2(dot(m.row(g), xs + size_t(order[first] / topk) * h, s0, h, block));
                            continue;
                        }
                        decode_row(m.row(g), rows, h);
                        for (int k = first; k < starts[group + 1]; ++k)
                            mid[size_t(k)*i + row] = activate_relu2(dot(FloatRow{rows}, xs + size_t(order[k] / topk) * h, s0, h, block));
                        continue;
                    }
                    auto s1 = m.scale(u);
                    if (starts[group + 1] - first == 1) {  // an expert of one token reads its rows directly
                        dot2(m.row(g), m.row(u), xs + size_t(order[first] / topk) * h, s0, s1, h, block, &gate, &up);
                        mid[size_t(first)*i + row] = activate(gate, up, act, b.cap_gate, b.cap_up);
                        continue;
                    }
                    decode_row(m.row(g), rows, h);
                    decode_row(m.row(u), rows + h, h);
                    for (int k = first; k < starts[group + 1]; ++k) {
                        dot2(FloatRow{rows}, FloatRow{rows + h}, xs + size_t(order[k] / topk) * h, s0, s1, h, block, &gate, &up);
                        mid[size_t(k)*i + row] = activate(gate, up, act, b.cap_gate, b.cap_up);
                    }
                }
            }
            if (rotation) {
                #pragma omp for schedule(static)
                for (int k = 0; k < count; ++k) rotate(mid.get() + size_t(k) * i, i, rotation);
            }
            if (groups == count) {
                #pragma omp for schedule(static)
                for (int task = 0; task < count * h; ++task) {
                    int slot = task / h, pair = active[slot], row = task % h;
                    M m = M::of(b.w2, b.s2, b.f2, ids[pair], h, i, b.block2);
                    values[size_t(pair)*h + row] = dot(m.row(row), mid.get() + size_t(slot)*i, m.scale(row), i, block2);
                }
            } else {
                float* down = scratch.get() + size_t(omp_get_thread_num()) * (2 * h + i) + 2 * h;
                #pragma omp for schedule(dynamic, 16)
                for (int task = 0; task < groups * h; ++task) {
                    int group = task / h, row = task % h, first = starts[group];
                    M m = M::of(b.w2, b.s2, b.f2, ids[order[first]], h, i, b.block2);
                    auto scale = m.scale(row);
                    if (starts[group + 1] - first == 1) {
                        values[size_t(order[first])*h + row] = dot(m.row(row), mid.get() + size_t(first)*i, scale, i, block2);
                        continue;
                    }
                    decode_row(m.row(row), down, i);
                    for (int k = first; k < starts[group + 1]; ++k)
                        values[size_t(order[k])*h + row] = dot(FloatRow{down}, mid.get() + size_t(k)*i, scale, i, block2);
                }
            }
            #pragma omp for schedule(static)
            for (int task = 0; task < batch * h; ++task) {
                int token = task / h, col = task % h; float sum = 0;
                for (int slot = 0; slot < topk; ++slot) {
                    int pair = token * topk + slot;
                    sum += values[size_t(pair)*h + col] * probabilities[pair];
                }
                out[task] = sum;
            }
#ifdef _WIN32
            if (previous) SetThreadAffinityMask(GetCurrentThread(), previous);
#endif
        }
        return count;
    } catch (...) { return -3; }
}

// format: FP8_BLOCK (E4M3FN weights, float32 scales), INT4_GROUP (packed weights, FP16 scales), INT8_ROWS (int8 weights, float32
// scales per row), NVFP4_ROWS (packed weights, E4M3 scales, f13 / f2: a float32 factor per expert) or MXFP8_ROWS (E4M3 weights, uint8
// scales); block / block2: the scale block (FP8), group (INT4) or row width (INT8) of w13 / w2, multiples of 32 dividing their input
// columns h / i (FP8 tiles also divide the rows: i and h), 16 for NVFP4 rows and 32 for MXFP8 rows (h and i multiples of 32);
// rotation: the ConvRot group (a power of 4 dividing h and i) or 0; act: SWIGLU or SWIGLU_SOFTCAP (cap_gate, cap_up; w13 = gate and
// up, 2 * i rows) or RELU2 (w13 = up, i rows).
static int moe_run(const Bank& b, const float* x, const int32_t* ids, const float* probabilities, const uint8_t* resident, float* out, int batch, int topk, int threads) {
    const int h = b.h, i = b.i, block = b.block, block2 = b.block2, rotation = b.rotation;
    if (batch < 0 || h <= 0 || i <= 0 || b.e <= 0 || topk <= 0 || h % 32 || i % 32 || threads <= 0) return -1;
    if (b.format == NVFP4_ROWS ? block != 16 || block2 != 16 || !b.f13 || !b.f2 : b.format == MXFP8_ROWS ? block != 32 || block2 != 32
        : block <= 0 || block2 <= 0 || block % 32 || block2 % 32 || h % block || i % block2) return -1;
    if (b.format == FP8_BLOCK && (i % block || h % block2)) return -1;
    if (b.act != SWIGLU && b.act != RELU2 && !(b.act == SWIGLU_SOFTCAP && b.cap_gate > 0.f && b.cap_up > 0.f)) return -1;
    if (rotation && (rotation < 4 || (rotation & (rotation - 1)) || !(rotation & 0x55555555) || h % rotation || i % rotation)) return -1;
    if (b.format == FP8_BLOCK) return moe_forward_format<Fp8Block>(b, x, ids, probabilities, resident, out, batch, topk, threads);
    if (b.format == INT4_GROUP) return moe_forward_format<Int4Group>(b, x, ids, probabilities, resident, out, batch, topk, threads);
    if (b.format == INT8_ROWS) return moe_forward_format<Int8Rows>(b, x, ids, probabilities, resident, out, batch, topk, threads);
    if (b.format == NVFP4_ROWS) return moe_forward_format<Nvfp4Rows>(b, x, ids, probabilities, resident, out, batch, topk, threads);
    if (b.format == MXFP8_ROWS) return moe_forward_format<Mxfp8Rows>(b, x, ids, probabilities, resident, out, batch, topk, threads);
    return -4;
}

API int moe_forward(int format, int rotation, int act, const uint8_t* w13, const uint8_t* w2, const void* s13, const void* s2,
                    const float* x, const int32_t* ids, const float* probabilities, const uint8_t* resident, float* out,
                    int batch, int h, int i, int e, int topk, int block, int block2, int threads, const float* f13, const float* f2, float cap_gate, float cap_up) {
    return moe_run(Bank{w13, w2, s13, s2, f13, f2, format, rotation, act, h, i, e, block, block2, cap_gate, cap_up}, x, ids, probabilities, resident, out, batch, topk, threads);
}

// One routed call run by a CUDA host function (cuLaunchHostFunc), from a stream or a captured graph: the payload is written by the
// device-to-host copy queued before it ([batch*h float32 x | batch*topk float32 probabilities | batch*topk int32 ids]), and the
// full float32 result (zeros when every pair is computed elsewhere) is read by the copy back queued after it. A failure is kept in
// status (sticky) for the host to report once the stream is synchronized. usage (or null): per expert, the pairs routed to it are
// counted there, wherever they are computed (the placement follows the routing).
struct MoeJob {
    const uint8_t* w13; const uint8_t* w2; const void* s13; const void* s2;
    const int32_t* payload; const uint8_t* resident; uint32_t* usage; float* out;
    int32_t format, rotation, act, batch, h, i, e, topk, block, block2, threads, status;
    const float* f13; const float* f2; float cap_gate, cap_up;
};

static int run_job(const MoeJob* j, const float* x, const int32_t* ids, const float* probabilities) {
    return moe_run(Bank{j->w13, j->w2, j->s13, j->s2, j->f13, j->f2, j->format, j->rotation, j->act, j->h, j->i, j->e, j->block, j->block2, j->cap_gate, j->cap_up},
                   x, ids, probabilities, j->resident, j->out, j->batch, j->topk, j->threads);
}

static void count_usage(const MoeJob* j, const int32_t* ids) {
    if (j->usage == nullptr) return;
    for (size_t n = 0; n < size_t(j->batch) * j->topk; ++n)
        if (ids[n] >= 0) ++j->usage[ids[n]];
}

API void moe_host_job(void* data) {
    MoeJob* j = (MoeJob*)data;
    const float* x = (const float*)j->payload;
    const float* probabilities = x + size_t(j->batch) * j->h;
    const int32_t* ids = j->payload + size_t(j->batch) * (j->h + j->topk);
    count_usage(j, ids);
    int count = run_job(j, x, ids, probabilities);
    if (count == 0) std::fill(j->out, j->out + size_t(j->batch) * j->h, 0.f);
    if (count < 0) j->status = count;
}

// A job of the host server (shared.kernels.host_doorbell): returns whether it wrote a result.
API int moe_server_job(void* data) {
    MoeJob* j = (MoeJob*)data;
    const float* x = (const float*)j->payload;
    const float* probabilities = x + size_t(j->batch) * j->h;
    const int32_t* ids = j->payload + size_t(j->batch) * (j->h + j->topk);
    count_usage(j, ids);
    int count = run_job(j, x, ids, probabilities);
    if (count < 0) j->status = count;
    return count > 0;
}
