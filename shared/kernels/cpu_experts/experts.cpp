// Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
// Routed experts on the CPU: float32 activations and accumulation, weights read from their checkpoint storage (no copies).
// Activations: SwiGLU (gate then up rows in w13) or squared ReLU (up rows only in w13).
// Formats: FP8 E4M3FN with float32 scales per block x block tile, symmetric INT4 (two values per byte, the even column in the low
// nibble, offset by 8) with FP16 scales per group of input columns, and INT8 rows with a float32 scale per row. The scale block of
// w13 and of w2 may differ. An optional ConvRot rotation (a regular Hadamard transform of each group of input features, applied to the
// weights before quantization) is applied to the activations of both projections.
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <cmath>
#include <algorithm>
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
#endif
#endif

enum { FP8_BLOCK = 0, INT4_GROUP = 1, INT8_ROWS = 2 };
enum { SWIGLU = 0, RELU2 = 1 };

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

static inline float activate_relu2(float v) { v = v > 0.f ? v : 0.f; return v * v; }

template <class F>
static inline void decode_row(const uint8_t* w, float* out, int cols) {
    typename F::Row row{w};
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
// block: the scale block of w13 (input columns h), block2 that of w2 (input columns i).
template <class F>
static int moe_forward_format(const uint8_t* w13, const uint8_t* w2, const typename F::ScaleT* s13, const typename F::ScaleT* s2,
                              const float* x, const int32_t* ids, const float* probabilities, const uint8_t* resident, float* out,
                              int batch, int h, int i, int e, int topk, int block, int block2, int threads, int rotation, int act) {
    using Row = typename F::Row; using Scale = typename F::Scale;
    const int pairs = batch * topk, projections = act == SWIGLU ? 2 : 1;
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
        // Per expert: weight bytes and scales of the gate (then up) projection and of the down projection
        const size_t gate_bytes = size_t(i) * F::row_bytes(h), down_bytes = size_t(h) * F::row_bytes(i);
        const size_t gate_scales = F::scale_row(i, h, block), down_scales = F::scale_row(h, i, block2);
#ifdef _WIN32
        static const auto cpus = worker_cpus();
#endif
        #pragma omp parallel num_threads(threads)
        {
#ifdef _WIN32
            int worker = omp_get_thread_num();
            DWORD_PTR previous = worker < int(cpus.size()) ? SetThreadAffinityMask(GetCurrentThread(), cpus[worker]) : 0;
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
                    int pair = active[task / i], row = task % i; size_t id = ids[pair];
                    auto w = w13 + id * gate_bytes * projections; auto s = s13 + id * gate_scales * projections;
                    const float* input = xs + size_t(pair / topk) * h;
                    if (act == RELU2) {
                        mid[task] = activate_relu2(dot(Row{w + row * F::row_bytes(h)}, input, Scale{s + F::scale_row(row, h, block)}, h, block));
                        continue;
                    }
                    float gate, up;
                    dot2(Row{w + row * F::row_bytes(h)}, Row{w + gate_bytes + row * F::row_bytes(h)}, input, Scale{s + F::scale_row(row, h, block)},
                         Scale{s + gate_scales + F::scale_row(row, h, block)}, h, block, &gate, &up);
                    mid[task] = (gate / (1.f + std::exp(-gate))) * up;
                }
            } else {
                float* rows = scratch.get() + size_t(omp_get_thread_num()) * (2 * h + i);
                // Experts serve different numbers of tokens: dynamic chunks balance the threads (each task writes its own outputs).
                #pragma omp for schedule(dynamic, 16)
                for (int task = 0; task < groups * i; ++task) {
                    int group = task / i, row = task % i, first = starts[group]; size_t id = ids[order[first]];
                    auto w = w13 + id * gate_bytes * projections; auto s = s13 + id * gate_scales * projections;
                    Scale s0{s + F::scale_row(row, h, block)}, s1{s + gate_scales + F::scale_row(row, h, block)};
                    float gate, up;
                    if (act == RELU2) {
                        if (starts[group + 1] - first == 1) {
                            mid[size_t(first)*i + row] = activate_relu2(dot(Row{w + row * F::row_bytes(h)}, xs + size_t(order[first] / topk) * h, s0, h, block));
                            continue;
                        }
                        decode_row<F>(w + row * F::row_bytes(h), rows, h);
                        for (int k = first; k < starts[group + 1]; ++k)
                            mid[size_t(k)*i + row] = activate_relu2(dot(FloatRow{rows}, xs + size_t(order[k] / topk) * h, s0, h, block));
                        continue;
                    }
                    if (starts[group + 1] - first == 1) {  // an expert of one token reads its rows directly
                        dot2(Row{w + row * F::row_bytes(h)}, Row{w + gate_bytes + row * F::row_bytes(h)}, xs + size_t(order[first] / topk) * h, s0, s1, h, block, &gate, &up);
                        mid[size_t(first)*i + row] = (gate / (1.f + std::exp(-gate))) * up;
                        continue;
                    }
                    decode_row<F>(w + row * F::row_bytes(h), rows, h);
                    decode_row<F>(w + gate_bytes + row * F::row_bytes(h), rows + h, h);
                    for (int k = first; k < starts[group + 1]; ++k) {
                        dot2(FloatRow{rows}, FloatRow{rows + h}, xs + size_t(order[k] / topk) * h, s0, s1, h, block, &gate, &up);
                        mid[size_t(k)*i + row] = (gate / (1.f + std::exp(-gate))) * up;
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
                    int slot = task / h, pair = active[slot], row = task % h; size_t id = ids[pair];
                    values[size_t(pair)*h + row] = dot(Row{w2 + id*down_bytes + row * F::row_bytes(i)}, mid.get() + size_t(slot)*i,
                                                       Scale{s2 + id*down_scales + F::scale_row(row, i, block2)}, i, block2);
                }
            } else {
                float* down = scratch.get() + size_t(omp_get_thread_num()) * (2 * h + i) + 2 * h;
                #pragma omp for schedule(dynamic, 16)
                for (int task = 0; task < groups * h; ++task) {
                    int group = task / h, row = task % h, first = starts[group]; size_t id = ids[order[first]];
                    Scale scale{s2 + id*down_scales + F::scale_row(row, i, block2)};
                    const uint8_t* weights = w2 + id*down_bytes + row * F::row_bytes(i);
                    if (starts[group + 1] - first == 1) {
                        values[size_t(order[first])*h + row] = dot(Row{weights}, mid.get() + size_t(first)*i, scale, i, block2);
                        continue;
                    }
                    decode_row<F>(weights, down, i);
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

// format: FP8_BLOCK (E4M3FN weights, float32 scales), INT4_GROUP (packed weights, FP16 scales) or INT8_ROWS (int8 weights, float32
// scales per row); block / block2: the scale block (FP8), group (INT4) or row width (INT8) of w13 / w2, multiples of 32 dividing their
// input columns h / i (FP8 tiles also divide the rows: i and h); rotation: the ConvRot group (a power of 4 dividing h and i) or 0;
// act: SWIGLU (w13 = gate then up, 2 * i rows) or RELU2 (w13 = up, i rows).
API int moe_forward(int format, int rotation, int act, const uint8_t* w13, const uint8_t* w2, const void* s13, const void* s2,
                    const float* x, const int32_t* ids, const float* probabilities, const uint8_t* resident, float* out,
                    int batch, int h, int i, int e, int topk, int block, int block2, int threads) {
    if (batch < 0 || h <= 0 || i <= 0 || e <= 0 || topk <= 0 || block <= 0 || block2 <= 0 || block % 32 || block2 % 32 || h % block || i % block2 || threads <= 0) return -1;
    if (format == FP8_BLOCK && (i % block || h % block2)) return -1;
    if (act != SWIGLU && act != RELU2) return -1;
    if (rotation && (rotation < 4 || (rotation & (rotation - 1)) || !(rotation & 0x55555555) || h % rotation || i % rotation)) return -1;
    if (format == FP8_BLOCK)
        return moe_forward_format<Fp8Block>(w13, w2, (const float*)s13, (const float*)s2, x, ids, probabilities, resident, out, batch, h, i, e, topk, block, block2, threads, rotation, act);
    if (format == INT4_GROUP)
        return moe_forward_format<Int4Group>(w13, w2, (const uint16_t*)s13, (const uint16_t*)s2, x, ids, probabilities, resident, out, batch, h, i, e, topk, block, block2, threads, rotation, act);
    if (format == INT8_ROWS)
        return moe_forward_format<Int8Rows>(w13, w2, (const float*)s13, (const float*)s2, x, ids, probabilities, resident, out, batch, h, i, e, topk, block, block2, threads, rotation, act);
    return -4;
}

// One routed call run by a CUDA host function (cuLaunchHostFunc), from a stream or a captured graph: the payload is written by the
// device-to-host copy queued before it ([batch*h float32 x | batch*topk float32 probabilities | batch*topk int32 ids]), and the
// full float32 result (zeros when every pair is computed elsewhere) is read by the copy back queued after it. A failure is kept in
// status (sticky) for the host to report once the stream is synchronized.
struct MoeJob {
    const uint8_t* w13; const uint8_t* w2; const void* s13; const void* s2;
    const int32_t* payload; const uint8_t* resident; float* out;
    int32_t format, rotation, act, batch, h, i, e, topk, block, block2, threads, status;
};

API void moe_host_job(void* data) {
    MoeJob* j = (MoeJob*)data;
    const float* x = (const float*)j->payload;
    const float* probabilities = x + size_t(j->batch) * j->h;
    const int32_t* ids = j->payload + size_t(j->batch) * (j->h + j->topk);
    int count = moe_forward(j->format, j->rotation, j->act, j->w13, j->w2, j->s13, j->s2, x, ids, probabilities, j->resident, j->out,
                            j->batch, j->h, j->i, j->e, j->topk, j->block, j->block2, j->threads);
    if (count == 0) std::fill(j->out, j->out + size_t(j->batch) * j->h, 0.f);
    if (count < 0) j->status = count;
}
