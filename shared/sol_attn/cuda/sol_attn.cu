// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Sol sparse attention (routing of NVlabs Sana's Sol-Attn: a block of 64 keys is attended exactly or through its centroid),
// one CTA per block of 64 queries and head, tensor cores for every product:
//   centroid scores  Q8 . KC^T   INT8 queries, centroids as two INT8 digits (also the routing statistics and the mean term of
//                                exact scores)
//   exact scores     Q8 . K8^T   INT8 residual of the keys around their block centroid (QK_FP4 = 0), or
//                    Q4 . K4^T   NVFP4 (QK_FP4 = 1): query and residual rotated by one signed Hadamard matrix (exact in the
//                                product), E2M1 codes with an E4M3 scale per 16 channels; the query is packed in the kernel
//   values           P . V       E4M3 probabilities and values, FP16 sums added to the FP32 output before they can overflow
// K and V of a block are one contiguous record, stored in the order the tensor-core loads read it, as are the centroids of a
// group of 16 blocks: a record is fetched by one bulk copy (BULK = 1, SM90 and newer) or by 16-byte copies. Routing runs one
// group ahead of the exact blocks and the record of an exact block is fetched during the block before it, into the other half
// of a double buffer: one barrier per exact block.
// Self-contained: builds with nvcc --cubin and with NVRTC (no headers).
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
#define INFINITY __int_as_float(0x7f800000)

#ifndef QK_FP4
#define QK_FP4 0
#endif
#ifndef BULK
#define BULK 1
#endif
#ifndef P_BITS
#define P_BITS 8
#endif
#ifndef KERNEL_NAME
#define KERNEL_NAME sol_attn_i8
#endif

#define FI __device__ __forceinline__
constexpr int D = 128, BLOCK = 64, GROUP = 16, THREADS = 128;
constexpr float NEG = -1.0e30f, LOG2E = 1.4426950408889634f;
constexpr float P_SHIFT = P_BITS;                   // probabilities are scaled by 2^P_BITS before E4M3: normal down to 2^-(6 + P_BITS)
// E4M3 values are at most 2.25 (their channel's peak): the FP16 accumulators are added to the FP32 output before the sum of the
// probabilities they hold, plus one block of 64 at most, could reach FP16's largest number
constexpr float FLUSH_AT = 29000.f - BLOCK * (1 << P_BITS);
constexpr float VC_GAIN = 4.f;                      // block means are stored 4 times larger (E4M3 keeps two more bits of them)

// Records. A fragment of 512 bytes holds the four 8 x 16-byte matrices of one ldmatrix: lane l reads its 16 bytes at l * 16.
#if QK_FP4
constexpr int KV_K = 0;                             // E2M1 keys: fragment (64-channel part s, 16 keys p) at (s * 4 + p) * 512
constexpr int KV_KS = KV_K + BLOCK * D / 2;         // [64 keys][2 parts] four E4M3 scales (16 channels each)
constexpr int KV_UNIT = KV_KS + BLOCK * 8;          // FP32 unit of the block's codes
constexpr int KV_V = KV_UNIT + 16;
#else
constexpr int KV_K = 0;                             // INT8 keys: fragment (32-channel step s, 16 keys p) at (s * 4 + p) * 512
constexpr int KV_KS = KV_K + BLOCK * D;             // [64] FP32 key scales
constexpr int KV_V = KV_KS + BLOCK * 4;
#endif
// E4M3 values from KV_V: fragment (16 channels n, 32 keys j) at (n * 2 + j) * 512; inside
constexpr int KV_RECORD = KV_V + D * BLOCK;         // 16 keys the order is 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15
// Signs of the rotation, one bit per channel (the packer of the keys uses the same)
constexpr uint32_t SIGNS[4] = {0x9e3779b9u, 0x7f4a7c15u, 0xf39cc060u, 0x5cedc834u};
constexpr int C_KC = 0;                             // centroid digits: fragment (digit, step s) at (digit * 4 + s) * 512
constexpr int C_VC = C_KC + GROUP * D * 2;          // E4M3 block means [channel][16 centroids], same key order
constexpr int C_KCS = C_VC + D * GROUP;             // [16] FP32 centroid scales
constexpr int C_RECORD = C_KCS + GROUP * 4;

// Shared memory, in bytes (two CTAs per SM below 49,920)
constexpr int SM_KV = 0;                                    // two records; first the BF16 Q tile and its INT8 form
constexpr int SM_C = SM_KV + 2 * KV_RECORD;
constexpr int SM_STASH = SM_C + C_RECORD;                   // [2][64 rows][16 centroids] FP32 centroid scores; first the query
constexpr int SM_SUMS = SM_STASH + 2 * BLOCK * GROUP * 4;  //   centroid and scales. Then [16 centroids][4 warps] column sums
constexpr int SM_BARS = SM_SUMS + GROUP * 4 * 4;           // three bulk-copy barriers
constexpr int SM_TOTAL = SM_BARS + 32;

FI uint32_t swz(uint32_t row, uint32_t col, uint32_t stride) { return (row * stride + (col ^ (row & 7))) * 16; }
FI uint32_t shared_address(const char* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

FI void cp16(char* dst, const char* src, bool valid) {
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(shared_address(dst)), "l"(src), "r"(bytes));
}
FI void commit() { asm volatile("cp.async.commit_group;\n" ::); }
FI void wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }
FI void wait_but_one() { asm volatile("cp.async.wait_group 1;\n" ::); }
#if BULK
FI void bar_init(char* bar) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(shared_address(bar)) : "memory"); }
FI void bulk_copy(char* dst, const char* src, uint32_t bytes, char* bar) {  // by one thread
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(shared_address(bar)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(shared_address(dst)), "l"(src), "r"(bytes), "r"(shared_address(bar)) : "memory");
}
FI void bar_wait(char* bar, uint32_t parity) {
  asm volatile("{ .reg .pred ok; sol_wait: mbarrier.try_wait.parity.shared::cta.b64 ok, [%0], %1; @!ok bra.uni sol_wait; }"
               :: "r"(shared_address(bar)), "r"(parity) : "memory");
}
#endif

FI void ldm(uint32_t* r, const char* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(shared_address(p)));
}
FI void mma_s8(int32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
FI void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// 64 channels of E2M1 codes; a_scales: four E4M3 scales of this lane's row (lane / 4, + 8 for odd lanes), b_scales: of its key
FI void mma_fp4(float* c, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t a_scales, uint32_t b_scales) {
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%12,%12}, {%11}, {%12,%12};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(a_scales), "r"(b_scales), "h"(static_cast<unsigned short>(0)));
}
FI uint32_t e2m1x2(float lo, float hi) {  // one byte: the code of lo in its low half
  uint32_t r;
  asm volatile("{ .reg .b8 c; cvt.rn.satfinite.e2m1x2.f32 c, %2, %1; mov.b32 %0, {c, 0, 0, 0}; }" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}
FI uint32_t e4m3_round(float x, float& rounded) {  // the E4M3 code of x and its value
  uint32_t code;
  asm volatile("{ .reg .b16 c, z; .reg .b32 h; mov.b16 z, 0; cvt.rn.satfinite.e4m3x2.f32 c, %2, %2; mov.b32 %0, {c, z}; cvt.rn.f16x2.e4m3x2 h, c;"
               " mov.b32 {c, z}, h; cvt.f32.f16 %1, c; }" : "=r"(code), "=f"(rounded) : "f"(x));
  return code & 0xffu;
}
FI uint32_t hadamard_pair(uint32_t d, uint32_t col) {  // BF16 entries (d, col) and (d + 1, col) of signs * H, d even
  const uint32_t s0 = (__popc(d & col) ^ (SIGNS[d >> 5] >> (d & 31))) & 1u;
  const uint32_t s1 = (__popc((d + 1) & col) ^ (SIGNS[d >> 5] >> ((d + 1) & 31))) & 1u;
  return (0x3f80u | (s0 << 15)) | ((0x3f80u | (s1 << 15)) << 16);
}
FI void mma_e4m3_f16(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1, bool init) {
  if (init)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%8};\n"
                 : "=r"(c[0]), "=r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
  else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
FI void rowsum_e4m3(float& d0, float& d1, const uint32_t* a) {  // adds the sums of both rows of an E4M3 fragment
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, _, %1, _}, {%2,%3,%4,%5}, {%6,%6}, {%0, 0., %1, 0.};\n"
               : "+f"(d0), "+f"(d1) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(943208504));  // four 1.0 in E4M3
}
FI uint32_t e4m3x4(float a, float b, float c, float d) {
  uint32_t r;
  asm volatile("{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1; cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3; mov.b32 %0, {lo, hi}; }"
               : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}
FI void add_f16x2(float& lo, float& hi, uint32_t packed, float weight = 1.f) {
  float a, b;
  asm volatile("{ .reg .b16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h; }" : "=f"(a), "=f"(b) : "r"(packed));
  lo = fmaf(a, weight, lo); hi = fmaf(b, weight, hi);
}
FI float ex2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
FI uint32_t bf16_rne(float x) {  // round to nearest even, any sign (finite inputs)
  const uint32_t u = __float_as_uint(x);
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
FI uint32_t bf16x2(float lo, float hi) { return bf16_rne(lo) | (bf16_rne(hi) << 16); }
FI float bf16_at(const char* p) { return __uint_as_float(static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(p)) << 16); }
FI float row_max4(float x) {  // over the four lanes holding one row
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
  return fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
}
FI uint32_t bit_range(int lo, int hi) {  // bits lo .. hi - 1 of a group of 16
  lo = max(lo, 0); hi = min(hi, GROUP);
  return hi > lo ? (1u << hi) - (1u << lo) : 0u;
}

extern "C" __global__ void KERNEL_NAME(
    const char* Q, const char* KV, const char* C, const float* KC_MEAN, const float* KC_VAR, const float* VSCALE, char* O,
    double scale_arg, double tau_arg, int sink_start, int sink_end, int sink_q_start, int sink_q_end,
    int T, int H, int NB, int q_token_bytes, int q_head_bytes, int o_token_bytes, int o_head_bytes, int first_block) {
  extern __shared__ __align__(16) char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int qb = blockIdx.x + first_block, head = blockIdx.y;
  const int q0 = qb * BLOCK;
  const float q_len = static_cast<float>(min(BLOCK, T - q0));
  const float tau = static_cast<float>(tau_arg), scale_log2 = static_cast<float>(scale_arg) * LOG2E;
  const int groups = (NB + GROUP - 1) / GROUP;
  const bool q_in_sink = qb >= sink_q_start && qb < sink_q_end;
  float* const stash = reinterpret_cast<float*>(smem + SM_STASH);
  const int row0 = warp * 16 + (lane >> 2), row1 = row0 + 8;  // this thread's two rows in accumulator fragments
  const int pair = lane & 3;                                   // its columns in a tile of 8: 2 * pair, 2 * pair + 1
  const bool leader = tid == 0;
#if BULK
  if (leader) {
    #pragma unroll
    for (int i = 0; i < 3; ++i) bar_init(smem + SM_BARS + i * 8);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
#endif

  // ---- Q tile: BF16 rows, then the query centroid (one channel per thread) and the INT8 rows (one row per thread)
  char* const q_tile = smem + SM_KV;
  char* const q8_tile = smem + SM_KV + BLOCK * D * 2;
  {
    const char* src = Q + static_cast<int64_t>(head) * q_head_bytes;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = tid + THREADS * i, row = c >> 4, col = c & 15;
      cp16(q_tile + swz(row, col, 16), src + static_cast<int64_t>(q0 + row) * q_token_bytes + col * 16, q0 + row < T);
    }
    commit();
    wait_all();
    __syncthreads();
    float sum = 0.f;
    for (int r = 0; r < BLOCK; ++r) sum += bf16_at(q_tile + swz(r, tid >> 3, 16) + (tid & 7) * 2);
    stash[tid] = sum / q_len;
    if (tid < BLOCK) {
      float amax = 0.f;
      for (int d = 0; d < D; ++d) amax = fmaxf(amax, fabsf(bf16_at(q_tile + swz(tid, d >> 3, 16) + (d & 7) * 2)));
      const float qs = fmaxf(amax / 127.f, 1e-8f);
      for (int d = 0; d < D; ++d) {
        const float x = bf16_at(q_tile + swz(tid, d >> 3, 16) + (d & 7) * 2) / qs;
        q8_tile[swz(tid, d >> 4, 8) + (d & 15)] = static_cast<char>(static_cast<int>(x >= 0.f ? x + 0.5f : x - 0.5f));
      }
      stash[D + tid] = qs * scale_log2;
    }
    __syncthreads();
  }
  float threshold_sum;  // routing threshold of the sum of a centroid's scores over the query rows
  {
    const float* mean = KC_MEAN + head * D;
    const float* var = KC_VAR + head * D;
    float m = 0.f, v = 0.f;
    for (int d = 0; d < D; ++d) {
      const float c = stash[d];
      m += c * mean[d];
      v += c * c * var[d];
    }
    threshold_sum = (m * scale_log2 + tau * sqrtf(fmaxf(v * scale_log2 * scale_log2, 0.f) + 1.0e-6f)) * q_len;
  }
  uint32_t RQ8[4][4];  // INT8 query fragments of this warp's 16 rows
  #pragma unroll
  for (int s = 0; s < 4; ++s) ldm(RQ8[s], q8_tile + swz(warp * 16 + (lane & 15), 2 * s + (lane >> 4), 8));
  const float qs0 = stash[D + row0], qs1 = stash[D + row1];
#if QK_FP4
  // The NVFP4 query: X = (q * signs) @ H by BF16 tensor products (H: Sylvester Hadamard, +-1), then per row a unit (its
  // largest value is 96 units), an E4M3 scale per 16 channels and E2M1 codes.
  char* const q4_tile = smem + SM_KV + BLOCK * D * 3;   // [64 rows][64 code bytes]
  char* const q4_scales = q4_tile + BLOCK * D / 2;      // [64 rows][8]
  float qu0, qu1;                                       // score per unit product of this thread's rows
  {
    float X[16][4];
    #pragma unroll
    for (int i = 0; i < 16; ++i) X[i][0] = X[i][1] = X[i][2] = X[i][3] = 0.f;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
      uint32_t a[4];
      ldm(a, q_tile + swz(warp * 16 + (lane & 15), 2 * s + (lane >> 4), 16));
      #pragma unroll
      for (int n = 0; n < 16; ++n)
        mma_bf16(X[n], a, hadamard_pair(16 * s + 2 * pair, 8 * n + (lane >> 2)), hadamard_pair(16 * s + 8 + 2 * pair, 8 * n + (lane >> 2)));
    }
    float peak0[8], peak1[8], top0 = 0.f, top1 = 0.f;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
      peak0[b] = row_max4(fmaxf(fmaxf(fabsf(X[2 * b][0]), fabsf(X[2 * b][1])), fmaxf(fabsf(X[2 * b + 1][0]), fabsf(X[2 * b + 1][1]))));
      peak1[b] = row_max4(fmaxf(fmaxf(fabsf(X[2 * b][2]), fabsf(X[2 * b][3])), fmaxf(fabsf(X[2 * b + 1][2]), fabsf(X[2 * b + 1][3]))));
      top0 = fmaxf(top0, peak0[b]); top1 = fmaxf(top1, peak1[b]);
    }
    const float unit0 = fmaxf(top0 * (1.f / 96.f), 1e-30f), unit1 = fmaxf(top1 * (1.f / 96.f), 1e-30f);
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
      float s0, s1;
      const uint32_t c0 = e4m3_round(peak0[b] / (6.f * unit0), s0), c1 = e4m3_round(peak1[b] / (6.f * unit1), s1);
      if (pair == 0) { q4_scales[row0 * 8 + b] = static_cast<char>(c0); q4_scales[row1 * 8 + b] = static_cast<char>(c1); }
      const float i0 = s0 > 0.f ? 1.f / (s0 * unit0) : 0.f, i1 = s1 > 0.f ? 1.f / (s1 * unit1) : 0.f;
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int n = 2 * b + h;
        q4_tile[row0 * 64 + 4 * n + pair] = static_cast<char>(e2m1x2(X[n][0] * i0, X[n][1] * i0));
        q4_tile[row1 * 64 + 4 * n + pair] = static_cast<char>(e2m1x2(X[n][2] * i1, X[n][3] * i1));
      }
    }
    qu0 = unit0 * scale_log2 * (1.f / D); qu1 = unit1 * scale_log2 * (1.f / D);  // both rotations are unnormalized: / 128
  }
  __syncthreads();
  uint32_t RQ4[2][4], q_scales[2];
  #pragma unroll
  for (int s = 0; s < 2; ++s) {
    ldm(RQ4[s], q4_tile + (warp * 16 + (lane & 15)) * 64 + (2 * s + (lane >> 4)) * 16);
    q_scales[s] = *reinterpret_cast<const uint32_t*>(q4_scales + (warp * 16 + (lane >> 2) + 8 * (lane & 1)) * 8 + 4 * s);
  }
#endif
  __syncthreads();

  float RO[16][4];
  uint32_t acc[16][2];                            // FP16 sums of P . V since the last flush
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    RO[i][0] = RO[i][1] = RO[i][2] = RO[i][3] = 0.f;
    acc[i][0] = acc[i][1] = 0u;
  }
  float m0 = NEG, m1 = NEG;                       // running maxima of this thread's two rows
  float D0 = 0.f, D1 = 0.f, F0 = 0.f, F1 = 0.f;   // sums of their exact probabilities, now and at the last flush
  float A0 = 0.f, A1 = 0.f;                       // sums of their centroid probabilities, times 64

  const char* kv_head = KV + static_cast<int64_t>(head) * NB * KV_RECORD;
  const char* c_head = C + static_cast<int64_t>(head) * groups * C_RECORD;
  uint32_t uses[3] = {0u, 0u, 0u};  // completed fetches into each buffer (the phases of their barriers)
  // Fetch a record into a buffer (slot 0, 1: K / V, 2: centroids). Every thread calls request and await in the same order.
  auto request = [&](int slot, const char* record, bool valid) {
    char* dst = smem + (slot == 2 ? SM_C : SM_KV + slot * KV_RECORD);
    const int bytes = slot == 2 ? C_RECORD : KV_RECORD;
#if BULK
    if (leader && valid) bulk_copy(dst, record, bytes, smem + SM_BARS + slot * 8);
#else
    if (valid) {
      #pragma unroll
      for (int i = 0; i < (KV_RECORD / 16 + THREADS - 1) / THREADS; ++i) {
        const int c = tid + THREADS * i;
        if (c * 16 < bytes) cp16(dst + c * 16, record + c * 16, true);
      }
    }
    commit();
#endif
  };
  auto await = [&](int slot, bool latest) {  // latest: no other request since this one
#if BULK
    bar_wait(smem + SM_BARS + slot * 8, uses[slot] & 1);
    uses[slot] += 1;
#else
    if (latest) wait_all(); else wait_but_one();
    __syncthreads();
#endif
  };
  auto flush = [&]() {  // the FP16 sums into the FP32 output
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      add_f16x2(RO[i][0], RO[i][1], acc[i][0]); add_f16x2(RO[i][2], RO[i][3], acc[i][1]);
      acc[i][0] = acc[i][1] = 0u;
    }
    F0 = D0; F1 = D1;
  };
  auto rescale = [&](float new0, float new1) {  // online softmax: move both rows to their new maxima
    if (__any_sync(0xffffffffu, (new0 != m0) | (new1 != m1))) {
      flush();
      const float a0 = ex2(m0 - new0), a1 = ex2(m1 - new1);
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        RO[i][0] *= a0; RO[i][1] *= a0; RO[i][2] *= a1; RO[i][3] *= a1;
      }
      A0 *= a0; A1 *= a1; D0 *= a0; D1 *= a1;
      F0 = D0; F1 = D1;
      m0 = new0; m1 = new1;
    }
  };
  auto mask_tail = [&](float S[8][4], int valid_keys) {  // keys beyond the sequence (last block)
    asm volatile("" ::: "memory");  // keeps this rare case a branch
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
      if (t * 8 + pair * 2 >= valid_keys) S[t][0] = S[t][2] = -INFINITY;
      if (t * 8 + pair * 2 + 1 >= valid_keys) S[t][1] = S[t][3] = -INFINITY;
    }
  };

  // ---- one group of centroids: scores, routing, approximate attention to the blocks not routed. Returns the exact mask.
  auto score_group = [&](int g) -> uint32_t {
    await(2, true);
    const char* tile = smem + SM_C;
    // centroid = scale * (high + low / 128): two INT8 products per score
    int32_t SH[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}}, SL[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t hi[4], lo[4];
      ldm(hi, tile + C_KC + s * 512 + lane * 16);
      ldm(lo, tile + C_KC + (4 + s) * 512 + lane * 16);
      mma_s8(SH[0], RQ8[s], hi[0], hi[1]);
      mma_s8(SH[1], RQ8[s], hi[2], hi[3]);
      mma_s8(SL[0], RQ8[s], lo[0], lo[1]);
      mma_s8(SL[1], RQ8[s], lo[2], lo[3]);
    }
    float S[2][4];
    float* slot = stash + (g & 1) * (BLOCK * GROUP);
    float colsum[4];
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int col = t * 8 + pair * 2;
      const float2 cs = *reinterpret_cast<const float2*>(tile + C_KCS + col * 4);
      S[t][0] = fmaf(__int2float_rn(SL[t][0]), 1.f / 128.f, __int2float_rn(SH[t][0])) * (qs0 * cs.x);
      S[t][1] = fmaf(__int2float_rn(SL[t][1]), 1.f / 128.f, __int2float_rn(SH[t][1])) * (qs0 * cs.y);
      S[t][2] = fmaf(__int2float_rn(SL[t][2]), 1.f / 128.f, __int2float_rn(SH[t][2])) * (qs1 * cs.x);
      S[t][3] = fmaf(__int2float_rn(SL[t][3]), 1.f / 128.f, __int2float_rn(SH[t][3])) * (qs1 * cs.y);
      *reinterpret_cast<float2*>(slot + row0 * GROUP + col) = make_float2(S[t][0], S[t][1]);
      *reinterpret_cast<float2*>(slot + row1 * GROUP + col) = make_float2(S[t][2], S[t][3]);
      colsum[t * 2] = S[t][0] + S[t][2];
      colsum[t * 2 + 1] = S[t][1] + S[t][3];
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {  // over the eight row groups of the warp (lanes with the same column pair)
      colsum[i] += __shfl_xor_sync(0xffffffffu, colsum[i], 4);
      colsum[i] += __shfl_xor_sync(0xffffffffu, colsum[i], 8);
      colsum[i] += __shfl_xor_sync(0xffffffffu, colsum[i], 16);
    }
    float* sums = reinterpret_cast<float*>(smem + SM_SUMS);
    if (lane < 4) {
      sums[(pair * 2) * 4 + warp] = colsum[0]; sums[(pair * 2 + 1) * 4 + warp] = colsum[1];
      sums[(8 + pair * 2) * 4 + warp] = colsum[2]; sums[(8 + pair * 2 + 1) * 4 + warp] = colsum[3];
    }
    __syncthreads();
    const int b0 = g * GROUP;
    uint32_t exact = 0;
    #pragma unroll
    for (int c = 0; c < GROUP; ++c) {
      const float4 s = reinterpret_cast<const float4*>(sums)[c];
      exact |= static_cast<uint32_t>((s.x + s.y) + (s.z + s.w) > threshold_sum) << c;
    }
    const uint32_t valid = bit_range(0, NB - b0);
    exact |= bit_range(qb - 1 - b0, qb + 2 - b0) | bit_range(sink_start - b0, sink_end - b0);
    exact = (q_in_sink ? 0xffffu : exact) & valid;
    if (exact != valid) {
      // approximate attention: a routed-out block is one token of weight 64 (its length for the last one) with the mean value
      #pragma unroll
      for (int t = 0; t < 2; ++t) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int c = t * 8 + pair * 2 + i;
          if (!((exact ^ valid) >> c & 1)) S[t][i] = S[t][2 + i] = -INFINITY;
        }
      }
      if (g == groups - 1 && T < NB * BLOCK) {  // a shorter last block
        const float bias = log2f(static_cast<float>(T - (NB - 1) * BLOCK) * (1.f / BLOCK));
        const int c = NB - 1 - b0;
        if ((c >> 1 & 3) == pair) { S[c >> 3][c & 1] += bias; S[c >> 3][2 + (c & 1)] += bias; }
      }
      const float bm0 = fmaxf(fmaxf(S[0][0], S[0][1]), fmaxf(S[1][0], S[1][1])), bm1 = fmaxf(fmaxf(S[0][2], S[0][3]), fmaxf(S[1][2], S[1][3]));
      // nothing to add when every probability of this warp's rows rounds to zero in E4M3
      if (__any_sync(0xffffffffu, (bm0 - m0 > -10.f - P_SHIFT) | (bm1 - m1 > -10.f - P_SHIFT))) {
        rescale(row_max4(fmaxf(bm0, m0)), row_max4(fmaxf(bm1, m1)));
        const float e0 = P_SHIFT - m0, e1 = P_SHIFT - m1;
        // the upper 16 keys of the product are zeros
        const uint32_t P[4] = {e4m3x4(ex2(S[0][0] + e0), ex2(S[0][1] + e0), ex2(S[1][0] + e0), ex2(S[1][1] + e0)),
                               e4m3x4(ex2(S[0][2] + e1), ex2(S[0][3] + e1), ex2(S[1][2] + e1), ex2(S[1][3] + e1)), 0u, 0u};
        float t0 = 0.f, t1 = 0.f;
        rowsum_e4m3(t0, t1, P);
        A0 += t0 * BLOCK; A1 += t1 * BLOCK;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {  // 32 channels
          uint32_t b[4];
          ldm(b, tile + C_VC + (n * 32 + lane) * 16);
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            uint32_t sum[2];
            mma_e4m3_f16(sum, P, b[i], 0u, true);
            add_f16x2(RO[4 * n + i][0], RO[4 * n + i][1], sum[0], BLOCK / VC_GAIN);
            add_f16x2(RO[4 * n + i][2], RO[4 * n + i][3], sum[1], BLOCK / VC_GAIN);
          }
        }
      }
    }
    __syncthreads();  // every warp has read the centroid record
    request(2, c_head + static_cast<int64_t>(g + 1) * C_RECORD, g + 1 < groups);
    return exact;
  };

  request(2, c_head, true);
  uint32_t mask = score_group(0), mask_next = 0;
  int in_flight = -1;  // the exact block whose record is loaded or loading
  int buffer = 0;      // its half of the double buffer
  for (int g = 0; g < groups; ++g) {
    mask_next = g + 1 < groups ? score_group(g + 1) : 0;
    while (mask) {
      const int c = __ffs(mask) - 1, b = g * GROUP + c;
      mask &= mask - 1;
      const int next = mask ? g * GROUP + __ffs(mask) - 1 : (mask_next ? (g + 1) * GROUP + __ffs(mask_next) - 1 : -1);
      __syncthreads();  // every warp has left the previous block
      if (in_flight != b) {  // not known a block ahead: the first exact block, or after a group without one
        buffer ^= 1;
        request(buffer, kv_head + static_cast<int64_t>(b) * KV_RECORD, true);
      }
      request(buffer ^ 1, kv_head + static_cast<int64_t>(next) * KV_RECORD, next >= 0);
      await(buffer, false);
      const char* record = smem + SM_KV + buffer * KV_RECORD;
      const float* slot = stash + (g & 1) * (BLOCK * GROUP);
      const float mean0 = slot[row0 * GROUP + c], mean1 = slot[row1 * GROUP + c];
      float S[8][4];
#if QK_FP4
      // exact scores: NVFP4 residual + centroid score
      float R[8][4];
      #pragma unroll
      for (int i = 0; i < 8; ++i) R[i][0] = R[i][1] = R[i][2] = R[i][3] = 0.f;
      #pragma unroll
      for (int s = 0; s < 2; ++s) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
          uint32_t kb[4];
          ldm(kb, record + KV_K + (s * 4 + p) * 512 + lane * 16);
          #pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t k_scales = *reinterpret_cast<const uint32_t*>(record + KV_KS + (((2 * p + h) * 8 + (lane >> 2)) * 2 + s) * 4);
            mma_fp4(R[2 * p + h], RQ4[s], kb[2 * h], kb[2 * h + 1], q_scales[s], k_scales);
          }
        }
      }
      const float unit = *reinterpret_cast<const float*>(record + KV_UNIT);
      const float c0 = qu0 * unit, c1 = qu1 * unit;
      #define SCORES(base0, base1) \
        _Pragma("unroll") for (int t = 0; t < 8; ++t) { \
          S[t][0] = fmaf(R[t][0], c0, base0); S[t][1] = fmaf(R[t][1], c0, base0); \
          S[t][2] = fmaf(R[t][2], c1, base1); S[t][3] = fmaf(R[t][3], c1, base1); }
#else
      const float* ks = reinterpret_cast<const float*>(record + KV_KS);
      // exact scores: INT8 residual + centroid score
      int32_t SI[8][4];
      #pragma unroll
      for (int i = 0; i < 8; ++i) SI[i][0] = SI[i][1] = SI[i][2] = SI[i][3] = 0;
      #pragma unroll
      for (int s = 0; s < 4; ++s) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
          uint32_t kb[4];
          ldm(kb, record + KV_K + (s * 4 + p) * 512 + lane * 16);
          mma_s8(SI[2 * p], RQ8[s], kb[0], kb[1]);
          mma_s8(SI[2 * p + 1], RQ8[s], kb[2], kb[3]);
        }
      }
      #define SCORES(base0, base1) \
        _Pragma("unroll") for (int t = 0; t < 8; ++t) { \
          const float2 k = *reinterpret_cast<const float2*>(ks + t * 8 + pair * 2); \
          S[t][0] = fmaf(__int2float_rn(SI[t][0]), qs0 * k.x, base0); S[t][1] = fmaf(__int2float_rn(SI[t][1]), qs0 * k.y, base0); \
          S[t][2] = fmaf(__int2float_rn(SI[t][2]), qs1 * k.x, base1); S[t][3] = fmaf(__int2float_rn(SI[t][3]), qs1 * k.y, base1); }
#endif
      {
        // scores relative to the running maxima, as exponents of the scaled probabilities
        const float base0 = mean0 + (P_SHIFT - m0), base1 = mean1 + (P_SHIFT - m1);
        float mx0 = NEG, mx1 = NEG;
        SCORES(base0, base1)
        if (b == NB - 1) mask_tail(S, T - b * BLOCK);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
          mx0 = fmaxf(mx0, fmaxf(S[t][0], S[t][1])); mx1 = fmaxf(mx1, fmaxf(S[t][2], S[t][3]));
        }
        if (__any_sync(0xffffffffu, (mx0 > P_SHIFT) | (mx1 > P_SHIFT))) {
          // a row of this warp has a new maximum (always in its first block): true scores, then move the rows
          float bm0 = NEG, bm1 = NEG;
          SCORES(mean0, mean1)
          if (b == NB - 1) mask_tail(S, T - b * BLOCK);
          #pragma unroll
          for (int t = 0; t < 8; ++t) {
            bm0 = fmaxf(bm0, fmaxf(S[t][0], S[t][1])); bm1 = fmaxf(bm1, fmaxf(S[t][2], S[t][3]));
          }
          rescale(row_max4(fmaxf(bm0, m0)), row_max4(fmaxf(bm1, m1)));
          const float e0 = P_SHIFT - m0, e1 = P_SHIFT - m1;
          #pragma unroll
          for (int t = 0; t < 8; ++t) {
            S[t][0] += e0; S[t][1] += e0; S[t][2] += e1; S[t][3] += e1;
          }
        }
      }
      #undef SCORES
      uint32_t P[2][4];
      #pragma unroll
      for (int j = 0; j < 2; ++j) {  // 32 keys: tiles 4j .. 4j + 3
        P[j][0] = e4m3x4(ex2(S[4 * j][0]), ex2(S[4 * j][1]), ex2(S[4 * j + 1][0]), ex2(S[4 * j + 1][1]));
        P[j][1] = e4m3x4(ex2(S[4 * j][2]), ex2(S[4 * j][3]), ex2(S[4 * j + 1][2]), ex2(S[4 * j + 1][3]));
        P[j][2] = e4m3x4(ex2(S[4 * j + 2][0]), ex2(S[4 * j + 2][1]), ex2(S[4 * j + 3][0]), ex2(S[4 * j + 3][1]));
        P[j][3] = e4m3x4(ex2(S[4 * j + 2][2]), ex2(S[4 * j + 2][3]), ex2(S[4 * j + 3][2]), ex2(S[4 * j + 3][3]));
      }
      if (__any_sync(0xffffffffu, (D0 - F0 > FLUSH_AT) | (D1 - F1 > FLUSH_AT))) flush();
      rowsum_e4m3(D0, D1, P[0]);
      rowsum_e4m3(D0, D1, P[1]);
      #pragma unroll
      for (int n = 0; n < 8; ++n) {  // 16 channels
        uint32_t vb[2][4];
        #pragma unroll
        for (int j = 0; j < 2; ++j) ldm(vb[j], record + KV_V + (n * 2 + j) * 512 + lane * 16);
        mma_e4m3_f16(acc[2 * n], P[0], vb[0][0], vb[0][1], false);
        mma_e4m3_f16(acc[2 * n + 1], P[0], vb[0][2], vb[0][3], false);
        mma_e4m3_f16(acc[2 * n], P[1], vb[1][0], vb[1][1], false);
        mma_e4m3_f16(acc[2 * n + 1], P[1], vb[1][2], vb[1][3], false);
      }
      in_flight = next;
      buffer ^= 1;
    }
    mask = mask_next;
  }

  // ---- normalize and store
  flush();
  const float inv0 = 1.f / (A0 + D0), inv1 = 1.f / (A1 + D1);
  const float* vscale = VSCALE + head * D;
  char* out0 = O + static_cast<int64_t>(head) * o_head_bytes + static_cast<int64_t>(q0 + row0) * o_token_bytes;
  char* out1 = O + static_cast<int64_t>(head) * o_head_bytes + static_cast<int64_t>(q0 + row1) * o_token_bytes;
  #pragma unroll
  for (int t = 0; t < 16; ++t) {
    const float2 vs = *reinterpret_cast<const float2*>(vscale + t * 8 + pair * 2);
    if (q0 + row0 < T) *reinterpret_cast<uint32_t*>(out0 + (t * 8 + pair * 2) * 2) = bf16x2(RO[t][0] * (inv0 * vs.x), RO[t][1] * (inv0 * vs.y));
    if (q0 + row1 < T) *reinterpret_cast<uint32_t*>(out1 + (t * 8 + pair * 2) * 2) = bf16x2(RO[t][2] * (inv1 * vs.x), RO[t][3] * (inv1 * vs.y));
  }
}
