// Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
// Routed experts in GGUF storage, included by linear.cpp: the selected experts of a bank, read in place in the checkpoint, with [experts, rows]
// rows of gate, up and down weights in ggml block formats (each with its activation quantization, as ggml). Pairs are grouped by expert, so that
// all the tokens routed to an expert run against its weight rows while these stay in the cache. Phases: the activations quantized, the gate and
// up rows, SwiGLU and its quantization, the down rows, the weighted sum.
struct MoeJob {  // ctypes structure of shared/qtypes/gguf.py, read when a CUDA host function runs it (graphs replay it at the same address)
    const void * gate;
    const void * up;
    const void * down;
    const void * payload;      // float32 activations [tokens, h], float32 probabilities [tokens, topk], int32 ids [tokens, topk]
    const uint8_t * resident;  // per expert, or null: nonzero experts are computed elsewhere
    uint32_t * usage;          // per expert, or null: the pairs routed to it are counted here, wherever they are computed
    float * out;               // float32 [tokens, h], zeros when no pair runs here
    int32_t gate_type, up_type, down_type, tokens, h, i, experts, topk, threads, tail_rank, status;  // tail_rank: see gguf_moe_forward
};

struct MoePair { int32_t token; float probability; };

struct MoeCall {
    const uint8_t * gate, * up, * down;
    const float * x;
    float * out;
    const GgufType * gate_t, * up_t, * down_t;
    int64_t tokens, h, i, gate_bytes, up_bytes, down_bytes, nb_x, nb_mid;
    bool x_q8_0;                                    // the activations of the gate and up rows as Q8_0 blocks, else Q8_K
    std::vector<int64_t> expert_ids, expert_start;  // the experts used, and the first pair of each
    std::vector<MoePair> pairs;
    std::vector<block_q8_K> x_q8_K;
    std::vector<block_q8_0> x_q8_0_blocks;
    std::vector<float> gate_up, down_out;           // [pairs, 2 * i], [pairs, h]
    std::vector<block_q8_0> mid_q8;                 // [pairs, i / 32]
    std::vector<uint8_t> token_used;
    int threads;
    int64_t gate_items, down_items, pair_count;
    alignas(64) std::atomic<int64_t> next_gate{0};
    alignas(64) std::atomic<int64_t> next_down{0};
    alignas(64) std::atomic<int> barrier{0};
};

static const int64_t MOE_GATE_ROWS = 16, MOE_DOWN_ROWS = 64;

static inline void moe_wait(MoeCall & c, int target) {
    c.barrier.fetch_add(1, std::memory_order_acq_rel);
    while (c.barrier.load(std::memory_order_acquire) < target) cpu_relax();
}

static void run_moe(void * context, int worker) {
    MoeCall & c = *(MoeCall *)context;
    if (worker >= c.threads) return;
    const int threads = c.threads;
    for (int64_t t = c.tokens * worker / threads; t < c.tokens * (worker + 1) / threads; ++t) {
        if (!c.token_used[t]) continue;
        if (c.x_q8_0) quantize_row_q8_0(c.x + t * c.h, c.x_q8_0_blocks.data() + t * c.nb_x, c.h);
        else for (int64_t b = 0; b < c.nb_x; ++b) quantize_block_q8_K(c.x + t * c.h + b * QK_K, c.x_q8_K.data() + t * c.nb_x + b);
    }
    moe_wait(c, threads);
    const int64_t * ids = c.expert_ids.data(), * start = c.expert_start.data();
    const int64_t gate_chunks = (c.i + MOE_GATE_ROWS - 1) / MOE_GATE_ROWS, down_chunks = (c.h + MOE_DOWN_ROWS - 1) / MOE_DOWN_ROWS;
    for (int64_t item; (item = c.next_gate.fetch_add(1, std::memory_order_relaxed)) < c.gate_items;) {
        const int64_t e = item / gate_chunks, j0 = (item % gate_chunks) * MOE_GATE_ROWS, j1 = std::min(c.i, j0 + MOE_GATE_ROWS);
        const uint8_t * gate = c.gate + ids[e] * c.i * c.gate_bytes, * up = c.up + ids[e] * c.i * c.up_bytes;
        for (int64_t p = start[e]; p < start[e + 1]; ++p) {
            const void * a = c.x_q8_0 ? (const void *)(c.x_q8_0_blocks.data() + c.pairs[p].token * c.nb_x) : (const void *)(c.x_q8_K.data() + c.pairs[p].token * c.nb_x);
            float * row = c.gate_up.data() + p * 2 * c.i;
            for (int64_t j = j0; j < j1; ++j) {
                c.gate_t->dot((int)c.h, row + j, 0, gate + j * c.gate_bytes, 0, a, 0, 1);
                c.up_t->dot((int)c.h, row + c.i + j, 0, up + j * c.up_bytes, 0, a, 0, 1);
            }
        }
    }
    moe_wait(c, 2 * threads);
    // silu(gate) * up, quantized as Q8_0 (blocks of 32 values)
    for (int64_t p = c.pair_count * worker / threads; p < c.pair_count * (worker + 1) / threads; ++p) {
        float * gate = c.gate_up.data() + p * 2 * c.i;
        const float * up = gate + c.i;
        for (int64_t j = 0; j < c.i; ++j) gate[j] = gate[j] / (1.f + std::exp(-gate[j])) * up[j];
        quantize_row_q8_0(gate, c.mid_q8.data() + p * c.nb_mid, c.i);
    }
    moe_wait(c, 3 * threads);
    for (int64_t item; (item = c.next_down.fetch_add(1, std::memory_order_relaxed)) < c.down_items;) {
        const int64_t e = item / down_chunks, j0 = (item % down_chunks) * MOE_DOWN_ROWS, j1 = std::min(c.h, j0 + MOE_DOWN_ROWS);
        const uint8_t * down = c.down + ids[e] * c.h * c.down_bytes;
        for (int64_t p = start[e]; p < start[e + 1]; ++p) {
            const block_q8_0 * a = c.mid_q8.data() + p * c.nb_mid;
            float * row = c.down_out.data() + p * c.h;
            for (int64_t j = j0; j < j1; ++j) c.down_t->dot((int)c.i, row + j, 0, down + j * c.down_bytes, 0, a, 0, 1);
        }
    }
    moe_wait(c, 4 * threads);
    // the sum of each token's experts, weighted: every worker takes its own output columns, in a fixed pair order
    const int64_t h0 = c.h * worker / threads, h1 = c.h * (worker + 1) / threads;
    for (int64_t p = 0; p < c.pair_count; ++p) {
        float * out = c.out + c.pairs[p].token * c.h;
        const float * value = c.down_out.data() + p * c.h;
        const float probability = c.pairs[p].probability;
        for (int64_t j = h0; j < h1; ++j) out[j] += value[j] * probability;
    }
}

// Returns the number of pairs computed (out is left unwritten when there are none), or a negative value for an unsupported weight.
// tail_rank > 0: an expert that no token routes above that rank (ids in descending order of weight) contributes nothing, as an idea of Strata (MIT License, https://github.com/Niko1221/Strata): faster, less precise.
CPU_LINEAR_API int64_t gguf_moe_forward(const void * gate, const void * up, const void * down, int gate_type, int up_type, int down_type, const float * x, const float * probabilities, const int32_t * ids,
                                        const uint8_t * resident, float * out, int64_t tokens, int64_t h, int64_t i, int64_t experts, int64_t topk, int threads, int tail_rank) {
    if (!cpu_ok()) return -1;
    const GgufType * gate_t = gguf_type(gate_type), * up_t = gguf_type(up_type), * down_t = gguf_type(down_type);
    if (gate_t == nullptr || up_t == nullptr || down_t == nullptr || gate_t->q8_0 != up_t->q8_0 || !down_t->q8_0 || gate_t->dot == nullptr || up_t->dot == nullptr || down_t->dot == nullptr)
        return -2;
    static std::mutex serial;
    std::lock_guard<std::mutex> guard(serial);
    static MoeCall * storage = new MoeCall();  // its vectors keep their capacity between the passes of a generation
    MoeCall & c = *storage;
    static std::vector<int32_t> counts, slot_of;
    static std::vector<uint8_t> above_tail;
    if (tail_rank > 0) {
        above_tail.assign((size_t)experts, 0);
        for (int64_t n = 0; n < tokens * topk; ++n)
            if (n % topk < tail_rank && ids[n] >= 0) above_tail[ids[n]] = 1;
    }
    const auto runs_here = [&](int32_t id) { return id >= 0 && (resident == nullptr || !resident[id]) && (tail_rank <= 0 || above_tail[id]); };
    counts.assign((size_t)experts, 0);
    int64_t count = 0;
    for (int64_t n = 0; n < tokens * topk; ++n) {
        const int32_t id = ids[n];
        if (runs_here(id)) {
            ++counts[id];
            ++count;
        }
    }
    if (!count) return 0;
    c.expert_ids.clear();
    c.expert_start.clear();
    slot_of.assign((size_t)experts, 0);
    for (int64_t e = 0, position = 0; e < experts; ++e)
        if (counts[e]) {
            c.expert_ids.push_back(e);
            c.expert_start.push_back(position);
            slot_of[e] = (int32_t)position;
            position += counts[e];
        }
    c.expert_start.push_back(count);
    c.pairs.resize((size_t)count);
    c.token_used.assign((size_t)tokens, 0);
    for (int64_t n = 0; n < tokens * topk; ++n) {
        const int32_t id = ids[n];
        if (runs_here(id)) {
            c.pairs[slot_of[id]++] = {(int32_t)(n / topk), probabilities[n]};
            c.token_used[n / topk] = 1;
        }
    }
    c.gate = (const uint8_t *)gate;
    c.up = (const uint8_t *)up;
    c.down = (const uint8_t *)down;
    c.x = x;
    c.out = out;
    c.gate_t = gate_t;
    c.up_t = up_t;
    c.down_t = down_t;
    c.tokens = tokens;
    c.h = h;
    c.i = i;
    c.gate_bytes = h / gate_t->block_values * gate_t->block_bytes;
    c.up_bytes = h / up_t->block_values * up_t->block_bytes;
    c.down_bytes = i / down_t->block_values * down_t->block_bytes;
    c.x_q8_0 = gate_t->q8_0;
    c.nb_x = h / (c.x_q8_0 ? QK8_0 : QK_K);
    c.nb_mid = i / QK8_0;
    c.pair_count = count;
    if (c.x_q8_0) c.x_q8_0_blocks.resize((size_t)(tokens * c.nb_x));
    else c.x_q8_K.resize((size_t)(tokens * c.nb_x));
    c.gate_up.resize((size_t)(count * 2 * i));
    c.mid_q8.resize((size_t)(count * c.nb_mid));
    c.down_out.resize((size_t)(count * h));
    memset(out, 0, (size_t)(tokens * h) * sizeof(float));
    const int64_t used = (int64_t)c.expert_ids.size();
    c.gate_items = used * ((i + MOE_GATE_ROWS - 1) / MOE_GATE_ROWS);
    c.down_items = used * ((h + MOE_DOWN_ROWS - 1) / MOE_DOWN_ROWS);
    c.next_gate.store(0);
    c.next_down.store(0);
    c.barrier.store(0);
    Pool & pool = workers(threads);
    c.threads = std::min(std::max(1, threads), pool.size);
    run_workers(pool, run_moe, &c);
    return count;
}

static std::atomic<uint64_t> moe_jobs{0}, moe_pairs{0}, moe_microseconds{0};

// jobs, pairs and microseconds of the host jobs so far
CPU_LINEAR_API void gguf_moe_stats(uint64_t * out) {
    out[0] = moe_jobs.load();
    out[1] = moe_pairs.load();
    out[2] = moe_microseconds.load();
}

// The computation of a job, from its routing payload into its result. Returns the pairs computed (out is left unwritten when there are none), 0 for a failure, which its status field reports.
static int64_t moe_job(MoeJob & job) {
    const auto started = std::chrono::steady_clock::now();
    const int64_t traced = host_stamps.load(std::memory_order_relaxed) == nullptr ? 0 : host_trace_clock();
    const int64_t tokens = job.tokens, h = job.h, topk = job.topk;
    const float * payload = (const float *)job.payload;
    if (job.usage != nullptr) {
        const int32_t * ids = (const int32_t *)(payload + tokens * (h + topk));
        for (int64_t n = 0; n < tokens * topk; ++n) ++job.usage[ids[n]];
    }
    const int64_t count = gguf_moe_forward(job.gate, job.up, job.down, job.gate_type, job.up_type, job.down_type, payload, payload + tokens * h, (const int32_t *)(payload + tokens * (h + topk)),
                                           job.resident, job.out, tokens, h, job.i, job.experts, topk, job.threads, job.tail_rank);
    if (count < 0) job.status = (int32_t)count;
    moe_jobs.fetch_add(1);
    moe_pairs.fetch_add((uint64_t)std::max<int64_t>(count, 0));
    moe_microseconds.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count());
    if (traced) host_trace(0, traced, count);
    return std::max<int64_t>(count, 0);
}

// A CUDA host function: the job, from its pinned routing payload into its pinned result. Without a pair, the result is zeroed.
CPU_LINEAR_API void gguf_moe_host_job(void * data) {
    MoeJob & job = *(MoeJob *)data;
    if (!moe_job(job)) memset(job.out, 0, (size_t)(job.tokens * job.h) * sizeof(float));
}

// A job of the host server: returns whether it wrote a result.
CPU_LINEAR_API int gguf_moe_server_job(void * data) {
    return moe_job(*(MoeJob *)data) > 0;
}
