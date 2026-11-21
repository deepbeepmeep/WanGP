// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// Packers of the records read by the Sol attention kernels (sol_attn.cu, which describes the layouts). One CTA per block of
// 64 tokens and head, one thread per channel.
//   sol_pack_k   block centroid as two INT8 digits + scale (centroid record), INT8 residual keys + key scales (K / V record),
//                FP32 centroid for the routing statistics
//   sol_pack_k4  the same with NVFP4 residual keys: rotation by the signed Hadamard matrix of the kernel, E2M1 codes with an
//                E4M3 scale per 16 channels, one unit per block
//   sol_v_peak   largest magnitude of each value channel of a head (PEAK must be zeros)
//   sol_pack_v   E4M3 values scaled to 2.25 at their channel's peak (K / V record) and E4M3 block means (centroid record)
// Self-contained: builds with nvcc --cubin and with NVRTC (no headers).
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;

#define FI __device__ __forceinline__
constexpr int D = 128, BLOCK = 64, GROUP = 16;
constexpr float VC_GAIN = 4.f;
constexpr int K8_BYTES = BLOCK * D, K8_RECORD = K8_BYTES + BLOCK * 4 + D * BLOCK;    // keys, key scales, values
constexpr int K4_BYTES = BLOCK * D / 2, K4_SCALES = K4_BYTES, K4_UNIT = K4_SCALES + BLOCK * 8, K4_RECORD = K4_UNIT + 16 + D * BLOCK;
constexpr int C_KC = 0, C_VC = C_KC + GROUP * D * 2, C_KCS = C_VC + D * GROUP, C_RECORD = C_KCS + GROUP * 4;
constexpr uint32_t SIGNS[4] = {0x9e3779b9u, 0x7f4a7c15u, 0xf39cc060u, 0x5cedc834u};
// position of a key inside its 16 keys: the E4M3 products read them as 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15
constexpr int POS16[16] = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};

FI uint32_t shared_address(const char* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
FI float bf16_at(const char* p) { return __uint_as_float(static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(p)) << 16); }
FI unsigned short bf16_rne(float x) {
  const uint32_t u = __float_as_uint(x);
  return static_cast<unsigned short>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
FI int round_away(float x) { return static_cast<int>(x >= 0.f ? x + 0.5f : x - 0.5f); }
FI uint32_t swz(uint32_t row, uint32_t col, uint32_t stride) { return (row * stride + (col ^ (row & 7))) * 16; }
FI void ldm(uint32_t* r, const char* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(shared_address(p)));
}
FI void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
FI uint32_t e4m3x2(float lo, float hi) {  // two bytes: the code of lo in the low one
  uint32_t r;
  asm volatile("{ .reg .b16 c, z; mov.b16 z, 0; cvt.rn.satfinite.e4m3x2.f32 c, %2, %1; mov.b32 %0, {c, z}; }" : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}
#if FP4
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
FI float row_max4(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
  return fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
}
#endif
// offset of byte `byte` of row `row` in fragments of four 8 x 16-byte matrices over 16 rows and 32 bytes
FI int fragment_offset(int fragment, int row, int byte) { return fragment * 512 + ((row >> 3) * 2 + (byte >> 4)) * 128 + (row & 7) * 16 + (byte & 15); }
FI void copy_out(char* dst, const char* src, int bytes, int tid) {
  for (int c = tid; c * 16 < bytes; c += 128) *reinterpret_cast<uint4*>(dst + c * 16) = *reinterpret_cast<const uint4*>(src + c * 16);
}

// The centroid of this thread's channel: its two digits and scale go to the centroid record. Returns the decoded centroid.
FI float pack_centroid(float mean, const float* means, char* C, float* KC, int head, int block, int NB, int d) {
  KC[(static_cast<int64_t>(head) * NB + block) * D + d] = mean;
  float amax = 0.f;
  for (int i = 0; i < D; ++i) amax = fmaxf(amax, fabsf(means[i]));
  const float cs = fmaxf(amax / 127.f, 1e-30f);
  const float high = static_cast<float>(round_away(mean / cs));
  const float low = fminf(fmaxf(static_cast<float>(round_away((mean - high * cs) * (128.f / cs))), -127.f), 127.f);
  const int groups = (NB + GROUP - 1) / GROUP, ci = block % GROUP;
  char* record = C + (static_cast<int64_t>(head) * groups + block / GROUP) * C_RECORD;
  record[C_KC + fragment_offset(d >> 5, ci, d & 31)] = static_cast<char>(static_cast<int>(high));
  record[C_KC + fragment_offset(4 + (d >> 5), ci, d & 31)] = static_cast<char>(static_cast<int>(low));
  if (d == 0) *reinterpret_cast<float*>(record + C_KCS + ci * 4) = cs;
  return cs * (high + low * (1.f / 128.f));
}

#if !FP4
extern "C" __global__ void sol_pack_k(const char* K, char* KV, char* C, float* KC, int T, int H, int NB, int k_token_bytes, int k_head_bytes) {
  __shared__ float R[BLOCK][D];
  __shared__ __align__(16) char record[K8_BYTES + BLOCK * 4];
  __shared__ float means[D], scales[BLOCK];
  const int d = threadIdx.x, block = blockIdx.x, head = blockIdx.y;
  const int length = min(BLOCK, T - block * BLOCK);
  const char* src = K + static_cast<int64_t>(head) * k_head_bytes + static_cast<int64_t>(block) * BLOCK * k_token_bytes + d * 2;
  float x[BLOCK], sum = 0.f;
  #pragma unroll
  for (int key = 0; key < BLOCK; ++key) {
    x[key] = key < length ? bf16_at(src + static_cast<int64_t>(key) * k_token_bytes) : 0.f;
    sum += x[key];
  }
  const float mean = sum / static_cast<float>(length);
  means[d] = mean;
  __syncthreads();
  const float centroid = pack_centroid(mean, means, C, KC, head, block, NB, d);
  #pragma unroll
  for (int key = 0; key < BLOCK; ++key) R[key][d] = key < length ? x[key] - centroid : 0.f;
  __syncthreads();
  if (d < BLOCK) {
    float amax = 0.f;
    for (int i = 0; i < D; ++i) amax = fmaxf(amax, fabsf(R[d][i]));
    scales[d] = fmaxf(amax / 127.f, 1e-8f);
    *reinterpret_cast<float*>(record + K8_BYTES + d * 4) = d < length ? scales[d] : 0.f;
  }
  __syncthreads();
  #pragma unroll
  for (int key = 0; key < BLOCK; ++key)
    record[fragment_offset((d >> 5) * 4 + (key >> 4), key & 15, d & 31)] = static_cast<char>(round_away(R[key][d] / scales[key]));
  __syncthreads();
  copy_out(KV + (static_cast<int64_t>(head) * NB + block) * K8_RECORD, record, K8_BYTES + BLOCK * 4, d);
}
#else
extern "C" __global__ void sol_pack_k4(const char* K, char* KV, char* C, float* KC, int T, int H, int NB, int k_token_bytes, int k_head_bytes) {
  __shared__ __align__(16) char tile[BLOCK * D * 2];  // BF16 residual rows
  __shared__ __align__(16) char record[K4_UNIT + 16];
  __shared__ float means[D], tops[4];
  const int d = threadIdx.x, block = blockIdx.x, head = blockIdx.y, lane = d & 31, warp = d >> 5, pair = lane & 3;
  const int length = min(BLOCK, T - block * BLOCK);
  const char* src = K + static_cast<int64_t>(head) * k_head_bytes + static_cast<int64_t>(block) * BLOCK * k_token_bytes + d * 2;
  float x[BLOCK], sum = 0.f;
  #pragma unroll
  for (int key = 0; key < BLOCK; ++key) {
    x[key] = key < length ? bf16_at(src + static_cast<int64_t>(key) * k_token_bytes) : 0.f;
    sum += x[key];
  }
  const float mean = sum / static_cast<float>(length);
  means[d] = mean;
  __syncthreads();
  const float centroid = pack_centroid(mean, means, C, KC, head, block, NB, d);
  #pragma unroll
  for (int key = 0; key < BLOCK; ++key)
    *reinterpret_cast<unsigned short*>(tile + swz(key, d >> 3, 16) + (d & 7) * 2) = bf16_rne(key < length ? x[key] - centroid : 0.f);
  __syncthreads();
  // X = (residual * signs) @ H for the 16 rows of this warp
  float X[16][4];
  #pragma unroll
  for (int i = 0; i < 16; ++i) X[i][0] = X[i][1] = X[i][2] = X[i][3] = 0.f;
  #pragma unroll
  for (int s = 0; s < 8; ++s) {
    uint32_t a[4];
    ldm(a, tile + swz(warp * 16 + (lane & 15), 2 * s + (lane >> 4), 16));
    #pragma unroll
    for (int n = 0; n < 16; ++n)
      mma_bf16(X[n], a, hadamard_pair(16 * s + 2 * pair, 8 * n + (lane >> 2)), hadamard_pair(16 * s + 8 + 2 * pair, 8 * n + (lane >> 2)));
  }
  const int row0 = warp * 16 + (lane >> 2), row1 = row0 + 8;
  float peak0[8], peak1[8], top = 0.f;
  #pragma unroll
  for (int b = 0; b < 8; ++b) {
    peak0[b] = row_max4(fmaxf(fmaxf(fabsf(X[2 * b][0]), fabsf(X[2 * b][1])), fmaxf(fabsf(X[2 * b + 1][0]), fabsf(X[2 * b + 1][1]))));
    peak1[b] = row_max4(fmaxf(fmaxf(fabsf(X[2 * b][2]), fabsf(X[2 * b][3])), fmaxf(fabsf(X[2 * b + 1][2]), fabsf(X[2 * b + 1][3]))));
    top = fmaxf(top, fmaxf(peak0[b], peak1[b]));
  }
  #pragma unroll
  for (int shift = 4; shift < 32; shift *= 2) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, shift));
  if (lane == 0) tops[warp] = top;
  __syncthreads();
  const float unit = fmaxf(fmaxf(fmaxf(tops[0], tops[1]), fmaxf(tops[2], tops[3])) * (1.f / 96.f), 1e-30f);  // the largest value is 96 units
  if (d == 0) {
    *reinterpret_cast<float*>(record + K4_UNIT) = unit;
    *reinterpret_cast<float*>(record + K4_UNIT + 4) = 0.f; *reinterpret_cast<float*>(record + K4_UNIT + 8) = 0.f; *reinterpret_cast<float*>(record + K4_UNIT + 12) = 0.f;
  }
  #pragma unroll
  for (int b = 0; b < 8; ++b) {
    float s0, s1;
    const uint32_t c0 = e4m3_round(peak0[b] / (6.f * unit), s0), c1 = e4m3_round(peak1[b] / (6.f * unit), s1);
    if (pair == 0) {
      record[K4_SCALES + row0 * 8 + b] = static_cast<char>(c0);
      record[K4_SCALES + row1 * 8 + b] = static_cast<char>(c1);
    }
    const float i0 = s0 > 0.f ? 1.f / (s0 * unit) : 0.f, i1 = s1 > 0.f ? 1.f / (s1 * unit) : 0.f;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int n = 2 * b + h, byte = 4 * n + pair;  // of the row's 64 code bytes
      record[fragment_offset((byte >> 5) * 4 + (row0 >> 4), row0 & 15, byte & 31)] = static_cast<char>(e2m1x2(X[n][0] * i0, X[n][1] * i0));
      record[fragment_offset((byte >> 5) * 4 + (row1 >> 4), row1 & 15, byte & 31)] = static_cast<char>(e2m1x2(X[n][2] * i1, X[n][3] * i1));
    }
  }
  __syncthreads();
  copy_out(KV + (static_cast<int64_t>(head) * NB + block) * K4_RECORD, record, K4_UNIT + 16, d);
}
#endif

extern "C" __global__ void sol_v_peak(const char* V, float* PEAK, int T, int H, int NB, int v_token_bytes, int v_head_bytes) {
  const int d = threadIdx.x, block = blockIdx.x, head = blockIdx.y;
  const int length = min(BLOCK, T - block * BLOCK);
  const char* src = V + static_cast<int64_t>(head) * v_head_bytes + static_cast<int64_t>(block) * BLOCK * v_token_bytes + d * 2;
  float amax = 0.f;
  for (int key = 0; key < length; ++key) amax = fmaxf(amax, fabsf(bf16_at(src + static_cast<int64_t>(key) * v_token_bytes)));
  atomicMax(reinterpret_cast<int*>(PEAK + head * D + d), __float_as_int(amax));  // non-negative floats order as integers
}

extern "C" __global__ void sol_pack_v(const char* V, const float* PEAK, char* KV, char* C, int T, int H, int NB, int v_token_bytes, int v_head_bytes,
                                      int record_bytes, int value_offset) {
  __shared__ __align__(16) char record[D * BLOCK];
  const int d = threadIdx.x, block = blockIdx.x, head = blockIdx.y;
  const int length = min(BLOCK, T - block * BLOCK);
  const char* src = V + static_cast<int64_t>(head) * v_head_bytes + static_cast<int64_t>(block) * BLOCK * v_token_bytes + d * 2;
  const float peak = PEAK[head * D + d], scale = peak > 1e-30f ? 2.25f / peak : 0.f;
  float sum = 0.f;
  #pragma unroll
  for (int key = 0; key < BLOCK; key += 2) {
    const float x0 = key < length ? bf16_at(src + static_cast<int64_t>(key) * v_token_bytes) : 0.f;
    const float x1 = key + 1 < length ? bf16_at(src + static_cast<int64_t>(key + 1) * v_token_bytes) : 0.f;
    sum += x0 + x1;
    const uint32_t codes = e4m3x2(x0 * scale, x1 * scale);
    const int p0 = (key & ~15) + POS16[key & 15], p1 = (key & ~15) + POS16[(key + 1) & 15];
    record[fragment_offset((d >> 4) * 2 + (p0 >> 5), d & 15, p0 & 31)] = static_cast<char>(codes & 0xff);
    record[fragment_offset((d >> 4) * 2 + (p1 >> 5), d & 15, p1 & 31)] = static_cast<char>(codes >> 8);
  }
  const int groups = (NB + GROUP - 1) / GROUP;
  char* centroids = C + (static_cast<int64_t>(head) * groups + block / GROUP) * C_RECORD;
  centroids[C_VC + d * 16 + POS16[block % GROUP]] = static_cast<char>(e4m3x2(sum / static_cast<float>(length) * scale * VC_GAIN, 0.f) & 0xff);
  __syncthreads();
  copy_out(KV + (static_cast<int64_t>(head) * NB + block) * record_bytes + value_offset, record, D * BLOCK, d);
}
