// Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
// Routed experts in GGUF storage, included by linear.cpp: the selected experts of a bank, read in place in the checkpoint, with [experts, rows]
// rows of gate, up and down weights in ggml block formats (each with its activation quantization, as ggml). Pairs are grouped by expert, so that
// all the tokens routed to an expert run against its weight rows while these stay in the cache. Phases: the activations quantized, the gate and
// up rows, SwiGLU and its quantization, the down rows, the weighted sum.
static const int MOE_SHARE_MAX = 64;
struct MoeRing;

// The GPU's share of the experts of a layer that miss VRAM, and the experts of the layer kept in locked RAM (mmgp level 2): ctypes structure of
// shared/kernels/cpu_linear MoeShare, one per bank, written by mmgp (sources, fraction) while jobs read it.
struct MoeShare {
    const uint64_t * sources;   // per expert: the host address of its copy in locked RAM, the layout of a VRAM slot, 0 when it has none
    uint64_t offsets[3];        // of the gate, up and down rows in such a copy
    uint64_t slot_bytes;        // of a copy
    MoeRing * ring;             // the VRAM slots the copies land in (gguf_moe_ring_open), shared by the banks
    float fraction;             // of the layer's missing experts asked from the copy engine, those of them in locked RAM and read by the most tokens first
    int32_t experts, status, pad;
    uint64_t asked, landed, missing, jobs;  // since the start: copies asked for, landed in time (the GPU computed them), experts missing VRAM, jobs
    const uint64_t * remote_sources;  // with a ring of a copy helper (gguf_moe_ring_open_remote): per expert, the shared memory section of its copy (high 32 bits, its handle here) and the offset in it
    const uint64_t * device_sources;  // without a ring: per expert, the device address of its copy in mapped RAM, which the GPU reads in place (0: none)
    float live;                       // without a ring: the share asked now, following whether the GPU ends its share before the CPU its part (jobs only)
    int32_t pad2;
};

static const int64_t MOE_SHARE_DONE = 1 + 5 * MOE_SHARE_MAX + 1;  // the word of a job's rows that the GPU sets once it has computed the share (after the errors word)


struct MoeJob {  // ctypes structure of shared/qtypes/gguf.py, read when a CUDA host function runs it (graphs replay it at the same address)
    const void * gate;
    const void * up;
    const void * down;
    const void * payload;      // float32 activations [tokens, h], float32 probabilities [tokens, topk], int32 ids [tokens, topk]
    const uint8_t * resident;  // per expert, or null: nonzero experts are computed elsewhere
    uint32_t * usage;          // per expert, or null: the pairs routed to it are counted here, wherever they are computed
    float * out;               // float32 [tokens, h], zeros when no pair runs here
    int32_t gate_type, up_type, down_type, tokens, h, i, experts, topk, threads, tail_rank, status;  // tail_rank: see gguf_moe_forward
    MoeShare * share;          // or null: no GPU share, no copies in locked RAM
    int64_t * share_rows;      // with share, mapped RAM the GPU reads once the job has answered: the count, then (expert, gate, up, down addresses) per expert it computes
};

struct MoePair { int32_t token; float probability; };

struct MoeCall {
    const uint8_t * gate, * up, * down;
    const float * x;
    float * out;
    const GgufType * gate_t, * up_t, * down_t;
    const uint64_t * sources, * offsets;            // or null: see MoeShare, the rows of an expert with a copy in locked RAM are read there
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
        const uint64_t copy = c.sources == nullptr ? 0 : c.sources[ids[e]];
        const uint8_t * gate = copy ? (const uint8_t *)(copy + c.offsets[0]) : c.gate + ids[e] * c.i * c.gate_bytes, * up = copy ? (const uint8_t *)(copy + c.offsets[1]) : c.up + ids[e] * c.i * c.up_bytes;
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
        const uint64_t copy = c.sources == nullptr ? 0 : c.sources[ids[e]];
        const uint8_t * down = copy ? (const uint8_t *)(copy + c.offsets[2]) : c.down + ids[e] * c.h * c.down_bytes;
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
// sources, offsets: or null, see MoeShare. accumulate: out holds a result already, the pairs are added to it.
static int64_t moe_forward(const void * gate, const void * up, const void * down, int gate_type, int up_type, int down_type, const float * x, const float * probabilities, const int32_t * ids,
                           const uint8_t * resident, float * out, int64_t tokens, int64_t h, int64_t i, int64_t experts, int64_t topk, int threads, int tail_rank, const uint64_t * sources, const uint64_t * offsets,
                           bool accumulate = false) {
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
    c.sources = sources;
    c.offsets = offsets;
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
    if (!accumulate) memset(out, 0, (size_t)(tokens * h) * sizeof(float));
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

CPU_LINEAR_API int64_t gguf_moe_forward(const void * gate, const void * up, const void * down, int gate_type, int up_type, int down_type, const float * x, const float * probabilities, const int32_t * ids,
                                        const uint8_t * resident, float * out, int64_t tokens, int64_t h, int64_t i, int64_t experts, int64_t topk, int threads, int tail_rank) {
    return moe_forward(gate, up, down, gate_type, up_type, down_type, x, probabilities, ids, resident, out, tokens, h, i, experts, topk, threads, tail_rank, nullptr, nullptr);
}

// The CUDA driver calls of the copies, loaded once.
struct CudaDriver {
    int (*set_context)(void *) = nullptr;
    int (*stream_create)(void **, unsigned) = nullptr;
    int (*stream_destroy)(void *) = nullptr;
    int (*stream_synchronize)(void *) = nullptr;
    int (*to_device)(uint64_t, const void *, size_t, void *) = nullptr;
    int (*to_host)(void *, uint64_t, size_t, void *) = nullptr;
    int (*query)(void *) = nullptr;
};

static const CudaDriver & cuda_driver() {
    static const CudaDriver driver = [] {
        CudaDriver d;
#ifdef _WIN32
        HMODULE module = LoadLibraryA("nvcuda.dll");
        const auto get = [&](const char * name) { return module == nullptr ? nullptr : (void *)GetProcAddress(module, name); };
#else
        void * module = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        const auto get = [&](const char * name) { return module == nullptr ? nullptr : dlsym(module, name); };
#endif
        d.set_context = (int (*)(void *))get("cuCtxSetCurrent");
        d.stream_create = (int (*)(void **, unsigned))get("cuStreamCreate");
        d.stream_destroy = (int (*)(void *))get("cuStreamDestroy_v2");
        d.stream_synchronize = (int (*)(void *))get("cuStreamSynchronize");
        d.to_device = (int (*)(uint64_t, const void *, size_t, void *))get("cuMemcpyHtoDAsync_v2");
        d.to_host = (int (*)(void *, uint64_t, size_t, void *))get("cuMemcpyDtoHAsync_v2");
        d.query = (int (*)(void *))get("cuStreamQuery");
        return d;
    }();
    return driver;
}

// Whether the GPU's share can run in this process.
CPU_LINEAR_API int gguf_moe_share_supported() {
    const CudaDriver & d = cuda_driver();
    return d.set_context && d.stream_create && d.stream_destroy && d.stream_synchronize && d.to_device && d.to_host && d.query;
}

// The VRAM slots of the GPU's share, and the thread that asks the copy engine for them. No job waits for a driver call: a call may wait for another
// thread holding the driver (a launch that waits for room behind a GPU that waits for this job), so the jobs hand their copies to this thread and look,
// once their CPU part is done, at which ones have landed: a copy ends with a copy of a word holding 1 from VRAM into its slot's done word, which the
// job reads in RAM. The others are computed on the CPU. Its stream is its own: a stream of PyTorch's pool may hold another user's wait for the compute.
struct MoeCopy { uint64_t target; const void * source; uint64_t bytes; int32_t slot; int64_t asked; };  // asked: host_trace_clock() when the job asked for it

// On Windows a copy call waits while another thread of the process holds its memory manager (a page fault, a working set trimmed), up to 10-300 ms
// at a time when RAM is short and the CPU's experts fault: the copies are then asked of a helper process (gguf_moe_copy_helper), whose calls nothing
// of this process holds. The ring of VRAM is exported to it (the driver's virtual memory, an NT handle), the copies of the experts are shared memory
// sections, and the requests and the done words are in a section of their own.
static const int MOE_RING_SLOTS_MAX = 256, MOE_REMOTE_REQUESTS = 1024;
struct MoeRemoteRequest { uint64_t section, offset, target, bytes; int32_t slot, kind; int64_t asked; };  // kind 0: a copy, 1: forget a section (it leaves; asked: the fills
                                                                                                        // before it), 2: map a section ahead
// A copy of an expert from its checkpoint file into a slot of level 2 (mmgp.experts._Level2), made by a fill helper process (gguf_moe_fill_helper), which
// makes no driver call: its page faults and the drive's reads then hold none (a thread faulting pages holds the driver calls of its own process, by up to
// tens of ms: in this process, or in the copy helper, whose late copies the GPU waits for), and no thread of this process runs for it (a copying thread
// here cost about 8 % of decoding at 160 GB, where nothing faults). file: a read-only mapping of the whole file (a handle of this process), source: an
// offset there; section: a section of level 2 (a handle of this process), target: the slot's offset there, offset: a part's offset in the slot. drop: the
// file's pages leave the fill helper's working set once copied (RAM is short). A record with no move (count 0) forgets its section. result: 1 until done,
// then 0, or the step that failed.
struct MoeRemoteMove { uint64_t file, source, offset, bytes; };
struct MoeRemoteFill { uint64_t section, target; MoeRemoteMove moves[3]; int32_t count, drop, result, pad; };
static const int MOE_REMOTE_FILLS = 256;
struct MoeRemoteControl {
    uint32_t magic, ready, stop, status;  // ready: 1 once the helper serves, 2 once it has left
    std::atomic<int64_t> head, tail, forgotten;  // the next request written, the next request the helper reads, the requests up to which it forgot what they asked
    std::atomic<uint64_t> issued, call_nanoseconds, call_max_nanoseconds, wait_nanoseconds, buckets[5];
    volatile int32_t done[MOE_RING_SLOTS_MAX];
    MoeRemoteRequest requests[MOE_REMOTE_REQUESTS];
    std::atomic<int32_t> fill_ready, fill_pad;  // 1 once the fill helper serves, 2 once it has left
    std::atomic<int64_t> fill_head, fill_tail;  // the next fill written, the next fill the fill helper reads: those before are done, their results written
    std::atomic<uint64_t> fill_nanoseconds, fill_bytes;
    MoeRemoteFill fills[MOE_REMOTE_FILLS];
};

#ifdef _WIN32
struct CuMemLocation { int type; int id; };
struct CuMemAllocationProp { int type; int handle_types; CuMemLocation location; void * win32_metadata; unsigned char compression, rdma; unsigned short usage; unsigned char reserved[4]; };
struct CuMemAccessDesc { CuMemLocation location; int flags; };
struct WinObjectAttributes { unsigned long length; void * root; void * name; unsigned long attributes; void * security; void * quality; };

// The driver's virtual memory and host registration calls of the ring of a helper process.
struct CudaMemory {
    int (*init)(unsigned) = nullptr;
    int (*device_get)(int *, int) = nullptr;
    int (*primary_retain)(void **, int) = nullptr;
    int (*create)(uint64_t *, size_t, const CuMemAllocationProp *, uint64_t) = nullptr;
    int (*release)(uint64_t) = nullptr;
    int (*reserve)(uint64_t *, size_t, size_t, uint64_t, uint64_t) = nullptr;
    int (*address_free)(uint64_t, size_t) = nullptr;
    int (*map)(uint64_t, size_t, size_t, uint64_t, uint64_t) = nullptr;
    int (*unmap)(uint64_t, size_t) = nullptr;
    int (*set_access)(uint64_t, size_t, const CuMemAccessDesc *, size_t) = nullptr;
    int (*export_handle)(void *, uint64_t, int, uint64_t) = nullptr;
    int (*import_handle)(uint64_t *, void *, int) = nullptr;
    int (*granularity)(size_t *, const CuMemAllocationProp *, int) = nullptr;
    int (*memset32)(uint64_t, unsigned, size_t) = nullptr;
    int (*host_register)(void *, size_t, unsigned) = nullptr;
    int (*host_unregister)(void *) = nullptr;
    int (*synchronize)() = nullptr;
    int (*device_pointer)(uint64_t *, void *, unsigned) = nullptr;
};

static const CudaMemory & cuda_memory() {
    static const CudaMemory memory = [] {
        CudaMemory m;
        HMODULE module = LoadLibraryA("nvcuda.dll");
        const auto get = [&](const char * name) { return module == nullptr ? nullptr : (void *)GetProcAddress(module, name); };
        m.init = (int (*)(unsigned))get("cuInit");
        m.device_get = (int (*)(int *, int))get("cuDeviceGet");
        m.primary_retain = (int (*)(void **, int))get("cuDevicePrimaryCtxRetain");
        m.create = (int (*)(uint64_t *, size_t, const CuMemAllocationProp *, uint64_t))get("cuMemCreate");
        m.release = (int (*)(uint64_t))get("cuMemRelease");
        m.reserve = (int (*)(uint64_t *, size_t, size_t, uint64_t, uint64_t))get("cuMemAddressReserve");
        m.address_free = (int (*)(uint64_t, size_t))get("cuMemAddressFree");
        m.map = (int (*)(uint64_t, size_t, size_t, uint64_t, uint64_t))get("cuMemMap");
        m.unmap = (int (*)(uint64_t, size_t))get("cuMemUnmap");
        m.set_access = (int (*)(uint64_t, size_t, const CuMemAccessDesc *, size_t))get("cuMemSetAccess");
        m.export_handle = (int (*)(void *, uint64_t, int, uint64_t))get("cuMemExportToShareableHandle");
        m.import_handle = (int (*)(uint64_t *, void *, int))get("cuMemImportFromShareableHandle");
        m.granularity = (int (*)(size_t *, const CuMemAllocationProp *, int))get("cuMemGetAllocationGranularity");
        m.memset32 = (int (*)(uint64_t, unsigned, size_t))get("cuMemsetD32_v2");
        m.host_register = (int (*)(void *, size_t, unsigned))get("cuMemHostRegister_v2");
        m.host_unregister = (int (*)(void *))get("cuMemHostUnregister");
        m.synchronize = (int (*)())get("cuCtxSynchronize");
        m.device_pointer = (int (*)(uint64_t *, void *, unsigned))get("cuMemHostGetDevicePointer_v2");
        return m;
    }();
    return memory;
}
#endif

struct MoeRing {
    uint64_t base = 0, slot_bytes = 0, device_one = 0;
    int32_t slots = 0;
    volatile int32_t * done = nullptr;   // locked RAM, per slot: 1 once its last copy landed, 0 while a copy is on its way
    std::vector<uint8_t> asked;          // per slot: a copy was asked for it (the jobs' thread only)
    int32_t cursor = 0;
    void * context = nullptr;
    void * stream = nullptr;
    std::mutex guard;
    std::condition_variable wake;
    std::vector<MoeCopy> queue;
    bool stop = false;
    int32_t status = 0;
    std::thread thread;
    std::atomic<uint64_t> issued{0}, issue_microseconds{0}, wait_nanoseconds{0};  // copies issued, the thread's time issuing them, their time in the queue
    std::atomic<uint64_t> call_nanoseconds{0}, call_max_nanoseconds{0}, wakes{0}, wake_nanoseconds{0};  // the copy calls (the longest), the thread's wake-ups and their delay after a job's request
    std::atomic<int64_t> notified_at{0};
    std::atomic<uint64_t> call_buckets[5] = {};  // copy calls under 50 us, 200 us, 1 ms, 10 ms, and longer
    uint64_t parts = 0, part_nanoseconds = 0;  // the jobs' CPU parts while their copies were on their way (the jobs' thread only)
    bool remote = false;                       // the copies are asked of a helper process (gguf_moe_ring_open_remote)
    std::atomic<int32_t> paused{0};            // nonzero: no copies asked (this process's own copies hold the copy engine, which a GPU has only one of under WDDM)
    float fraction = 0.25f;                    // with a helper: of a job's missing experts, the share asked now, which follows whether its copies land within its CPU part (the
                                               // GPU waits for those that do not); at most the share's own fraction (the jobs' thread only)
    uint64_t header = 0;                       // with a helper: the ring's offset in its VRAM, after the word holding 1
    uint64_t done_device = 0;                  // with a helper: the device address of the done words, which this process's GPU waits on
    MoeRemoteControl * control = nullptr;
    std::mutex produce;                        // with a helper: the jobs' copies and the forgets of quiet points write requests
    std::mutex fill_produce;                   // with a helper: the fills of level 2
#ifdef _WIN32
    HANDLE control_section = nullptr, event = nullptr, vram_export = nullptr;
    uint64_t vram_handle = 0, vram_address = 0, vram_bytes = 0;
#endif

    int32_t take() {  // a slot no copy is on its way to, whose last copy the GPU has read (the job before this one ended before it was asked for)
        for (int32_t n = 0; n < slots; ++n) {
            const int32_t slot = (cursor + n) % slots;
            if (!asked[slot] || done[slot]) {
                cursor = (slot + 1) % slots;
                return slot;
            }
        }
        return -1;
    }
};

static void ring_main(MoeRing * r) {
    const CudaDriver & d = cuda_driver();
    if (d.set_context(r->context)) r->status = -10;
    std::vector<MoeCopy> batch;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(r->guard);
            r->wake.wait(lock, [&] { return r->stop || !r->queue.empty(); });
            if (r->queue.empty()) return;
            batch.swap(r->queue);
        }
        const int64_t notified = r->notified_at.exchange(0);
        if (notified) {
            r->wakes.fetch_add(1);
            r->wake_nanoseconds.fetch_add((uint64_t)std::max<int64_t>(0, host_trace_clock() - notified));
        }
        const auto started = std::chrono::steady_clock::now();
        const int64_t now = host_trace_clock();
        int status = 0;
        for (const MoeCopy & copy : batch) {
            r->wait_nanoseconds.fetch_add((uint64_t)std::max<int64_t>(0, now - copy.asked));
            const int64_t call = host_trace_clock();
            status |= d.to_device(copy.target, copy.source, (size_t)copy.bytes, r->stream);
            const uint64_t took = (uint64_t)std::max<int64_t>(0, host_trace_clock() - call);
            r->call_nanoseconds.fetch_add(took);
            if (took > r->call_max_nanoseconds.load()) r->call_max_nanoseconds.store(took);
            r->call_buckets[took < 50000 ? 0 : took < 200000 ? 1 : took < 1000000 ? 2 : took < 10000000 ? 3 : 4].fetch_add(1);
            status |= d.to_host((void *)(r->done + copy.slot), r->device_one, sizeof(int32_t), r->stream);
        }
        d.query(r->stream);  // submitted now: a driver that batches its work (WDDM) would keep it until a later call
        if (status) r->status = -11;
        r->issued.fetch_add(batch.size());
        r->issue_microseconds.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count());
        batch.clear();
    }
}

// base: device address of slots of slot_bytes; done: locked RAM of slots int32; device_one: device address of an int32 holding 1. Called with the CUDA context
// current. Returns null if the stream cannot be made.
CPU_LINEAR_API void * gguf_moe_ring_open(uint64_t base, uint64_t slot_bytes, int32_t slots, void * context, int32_t * done, uint64_t device_one) {
    const CudaDriver & d = cuda_driver();
    MoeRing * r = new MoeRing();
    r->base = base;
    r->slot_bytes = slot_bytes;
    r->slots = slots;
    r->done = done;
    r->device_one = device_one;
    r->context = context;
    r->asked.assign((size_t)slots, 0);
    for (int32_t slot = 0; slot < slots; ++slot) done[slot] = 0;
    if (d.stream_create(&r->stream, 1)) {  // CU_STREAM_NON_BLOCKING
        delete r;
        return nullptr;
    }
    r->thread = std::thread(ring_main, r);
    return r;
}

// Once nothing in flight asks for copies: the thread stops after the copies asked for, which are waited for. A helper's ring once the helper has left.
CPU_LINEAR_API void gguf_moe_ring_close(void * handle) {
    MoeRing * r = (MoeRing *)handle;
#ifdef _WIN32
    if (r->remote) {
        const CudaMemory & m = cuda_memory();
        m.synchronize();
        m.unmap(r->vram_address, (size_t)r->vram_bytes);
        m.address_free(r->vram_address, (size_t)r->vram_bytes);
        m.release(r->vram_handle);
        m.host_unregister((void *)r->control);
        UnmapViewOfFile(r->control);
        CloseHandle(r->control_section);
        CloseHandle(r->event);
        CloseHandle(r->vram_export);
        delete r;
        return;
    }
#endif
    {
        std::lock_guard<std::mutex> lock(r->guard);
        r->stop = true;
    }
    r->wake.notify_one();
    r->thread.join();
    const CudaDriver & d = cuda_driver();
    d.stream_synchronize(r->stream);
    d.stream_destroy(r->stream);
    delete r;
}

// copies issued, microseconds the thread spent issuing them, first failure, nanoseconds the copies waited in the queue, jobs with copies, nanoseconds of their CPU parts
CPU_LINEAR_API void gguf_moe_ring_stats(void * handle, int64_t * out) {
    MoeRing * r = (MoeRing *)handle;
    out[0] = (int64_t)r->issued.load();
    out[1] = (int64_t)r->issue_microseconds.load();
    out[2] = r->status;
    out[3] = (int64_t)r->wait_nanoseconds.load();
    out[4] = (int64_t)r->parts;
    out[5] = (int64_t)r->part_nanoseconds;
    out[6] = (int64_t)r->call_nanoseconds.load();
    out[7] = (int64_t)r->call_max_nanoseconds.load();
    out[8] = (int64_t)r->wakes.load();
    out[9] = (int64_t)r->wake_nanoseconds.load();
    for (int k = 0; k < 5; ++k) out[10 + k] = (int64_t)r->call_buckets[k].load();
    out[15] = (int64_t)std::lround(r->fraction * 1000);
    if (r->remote) {  // the helper's own counts
        MoeRemoteControl & c = *r->control;
        out[0] = (int64_t)c.issued.load();
        out[2] = (int32_t)c.status;
        out[3] = (int64_t)c.wait_nanoseconds.load();
        out[6] = (int64_t)c.call_nanoseconds.load();
        out[7] = (int64_t)c.call_max_nanoseconds.load();
        for (int k = 0; k < 5; ++k) out[10 + k] = (int64_t)c.buckets[k].load();
    }
}

#ifdef _WIN32
// A ring whose copies a helper process asks for: its VRAM (slots of slot_bytes after a word holding 1, the driver's virtual memory exported as an NT handle),
// the section of its requests and done words, and the event that wakes the helper. Called with the CUDA context current. out: the ring's device address,
// then the handles of the section, the event and the VRAM, and the VRAM's bytes, for the helper's command (gguf_moe_copy_helper). Null if refused.
CPU_LINEAR_API void * gguf_moe_ring_open_remote(uint64_t slot_bytes, int32_t slots, int32_t device, uint64_t * out) {
    const CudaMemory & m = cuda_memory();
    if (slots > MOE_RING_SLOTS_MAX || !m.create || !m.export_handle || !m.granularity) return nullptr;
    static WinObjectAttributes attributes{sizeof(WinObjectAttributes), nullptr, nullptr, 0, nullptr, nullptr};  // the default security of this process: its user only
    CuMemAllocationProp prop{};
    prop.type = 1;          // CU_MEM_ALLOCATION_TYPE_PINNED
    prop.handle_types = 2;  // CU_MEM_HANDLE_TYPE_WIN32
    prop.location = {1, device};
    prop.win32_metadata = &attributes;
    size_t granularity = 0;
    if (m.granularity(&granularity, &prop, 0) || !granularity) return nullptr;
    const uint64_t header = granularity, bytes = (header + (uint64_t)slots * slot_bytes + granularity - 1) / granularity * granularity;
    uint64_t generic = 0, vram = 0;
    if (m.create(&generic, (size_t)bytes, &prop, 0)) return nullptr;
    CuMemAccessDesc access{{1, device}, 3};
    void * exported = nullptr;
    if (m.reserve(&vram, (size_t)bytes, 0, 0, 0) || m.map(vram, (size_t)bytes, 0, generic, 0) || m.set_access(vram, (size_t)bytes, &access, 1) || m.memset32(vram, 1, 1) || m.synchronize()
        || m.export_handle(&exported, generic, 2, 0)) {
        if (vram) m.address_free(vram, (size_t)bytes);
        m.release(generic);
        return nullptr;
    }
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, (DWORD)sizeof(MoeRemoteControl), nullptr);
    MoeRemoteControl * control = section == nullptr ? nullptr : (MoeRemoteControl *)MapViewOfFile(section, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(MoeRemoteControl));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (control == nullptr || event == nullptr) return nullptr;
    memset((void *)control, 0, sizeof(MoeRemoteControl));
    control->magic = 0x4d4f4552;
    uint64_t control_device = 0;  // the GPU of this process waits on the done words: mapped for it
    if (m.host_register((void *)control, sizeof(MoeRemoteControl), 0x02) || m.device_pointer(&control_device, (void *)control, 0)) return nullptr;  // CU_MEMHOSTREGISTER_DEVICEMAP
    MoeRing * r = new MoeRing();
    r->remote = true;
    r->header = header;
    r->base = vram + header;
    r->slot_bytes = slot_bytes;
    r->slots = slots;
    r->done = control->done;
    r->asked.assign((size_t)slots, 0);
    r->control = control;
    r->control_section = section;
    r->event = event;
    r->vram_export = (HANDLE)exported;
    r->vram_handle = generic;
    r->vram_address = vram;
    r->vram_bytes = bytes;
    r->done_device = control_device + (uint64_t)((const volatile uint8_t *)control->done - (const volatile uint8_t *)control);
    out[0] = r->base;
    out[1] = (uint64_t)section;
    out[2] = (uint64_t)event;
    out[3] = (uint64_t)exported;
    out[4] = bytes;
    return r;
}

// While nonzero, the jobs ask for no copies: the placement's own copies hold the copy engine, and the share's would wait behind them.
CPU_LINEAR_API void gguf_moe_ring_pause(void * handle, int32_t paused) {
    ((MoeRing *)handle)->paused.store(paused, std::memory_order_relaxed);
}

// 1 once the helper serves, 2 once it has left, 0 before.
CPU_LINEAR_API int gguf_moe_ring_remote_state(void * handle) {
    return (int)((MoeRing *)handle)->control->ready;
}

// The helper leaves after the requests written: once nothing in flight asks for copies.
CPU_LINEAR_API void gguf_moe_ring_remote_stop(void * handle) {
    MoeRing * r = (MoeRing *)handle;
    r->control->stop = 1;
    SetEvent(r->event);
}

// A section of experts' copies leaves: the helper unmaps it before its RAM goes. Returns 1 once it did, 0 if it did not answer within timeout_ms.
CPU_LINEAR_API int gguf_moe_ring_forget(void * handle, uint64_t section, int32_t timeout_ms) {
    MoeRing * r = (MoeRing *)handle;
    int64_t head;
    {
        std::lock_guard<std::mutex> lock(r->produce);
        head = r->control->head.load(std::memory_order_relaxed);
        if (head - r->control->tail.load(std::memory_order_acquire) >= MOE_REMOTE_REQUESTS) return 0;
        r->control->requests[head % MOE_REMOTE_REQUESTS] = {section, 0, 0, 0, -1, 1, 0};
        r->control->head.store(head + 1, std::memory_order_release);
    }
    SetEvent(r->event);
    int64_t fill = -1;  // the fill helper's forget, after the fills asked before it (one of them may write into the section)
    for (int waited = 0; waited < timeout_ms; ++waited) {
        if (fill < 0 && r->control->fill_ready.load() == 1) {
            std::lock_guard<std::mutex> lock(r->fill_produce);
            const int64_t at = r->control->fill_head.load(std::memory_order_relaxed);
            if (at - r->control->fill_tail.load(std::memory_order_acquire) < MOE_REMOTE_FILLS) {
                MoeRemoteFill & record = r->control->fills[at % MOE_REMOTE_FILLS];
                record = {};
                record.section = section;
                record.result = 1;
                r->control->fill_head.store(at + 1, std::memory_order_release);
                fill = at;
            }
        }
        const bool copies = r->control->forgotten.load(std::memory_order_acquire) > head || r->control->ready == 2;
        const bool fills = r->control->fill_ready.load() != 1 || (fill >= 0 && r->control->fill_tail.load(std::memory_order_acquire) > fill);
        if (copies && fills) return 1;
        Sleep(1);
    }
    return 0;
}

// 1 once the fill helper serves, 2 once it has left, 0 before.
CPU_LINEAR_API int gguf_moe_ring_fill_state(void * handle) {
    return ((MoeRing *)handle)->control->fill_ready.load();
}

// Fills of level 2 asked of the helper (MoeRemoteFill): all of them, or none when its queue lacks room for them. Returns the sequence number of the first, -1 if refused.
CPU_LINEAR_API int64_t gguf_moe_ring_fill(void * handle, const MoeRemoteFill * fills, int32_t count) {
    MoeRing * r = (MoeRing *)handle;
    if (!r->remote || count <= 0 || count > MOE_REMOTE_FILLS) return -1;
    std::lock_guard<std::mutex> lock(r->fill_produce);
    const int64_t head = r->control->fill_head.load(std::memory_order_relaxed);
    if (head + count - r->control->fill_tail.load(std::memory_order_acquire) > MOE_REMOTE_FILLS) return -1;
    for (int32_t n = 0; n < count; ++n) {
        MoeRemoteFill & fill = r->control->fills[(head + n) % MOE_REMOTE_FILLS];
        fill = fills[n];
        fill.result = 1;
    }
    r->control->fill_head.store(head + count, std::memory_order_release);
    return head;
}

// The fills done so far (the sequence number after the last), and in results those of [first, first + count) that are done: 0 copied, else the step that failed.
// Read before asking for more: a fill done leaves its place to the next ones.
CPU_LINEAR_API int64_t gguf_moe_ring_filled(void * handle, int64_t first, int32_t count, int32_t * results, uint64_t * stats) {
    MoeRing * r = (MoeRing *)handle;
    const int64_t tail = r->control->fill_tail.load(std::memory_order_acquire);
    for (int64_t n = first; n < std::min<int64_t>(first + count, tail); ++n) results[n - first] = r->control->fills[n % MOE_REMOTE_FILLS].result;
    stats[0] = r->control->fill_nanoseconds.load();
    stats[1] = r->control->fill_bytes.load();
    return tail;
}

struct WinRangeEntry { void * address; size_t bytes; };  // WIN32_MEMORY_RANGE_ENTRY
using WinPrefetch = BOOL (WINAPI *)(HANDLE, ULONG_PTR, WinRangeEntry *, ULONG);

// A copy from a view of a file, which a drive failing to read it raises as an exception: -3 then.
static int fill_copy(void * target, const void * source, size_t bytes) {
    __try {
        memcpy(target, source, bytes);
    } __except (GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return -3;
    }
    return 0;
}

// The logical processors (group 0) of the least efficient cores of a hybrid CPU, 0 when its cores are all alike: there a spinning thread takes no time
// from the CPU's expert workers, which run on the others.
static DWORD_PTR efficient_processors() {
    DWORD size = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &size);
    std::vector<uint8_t> buffer(size);
    if (!size || !GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer.data(), &size)) return 0;
    DWORD_PTR masks[256] = {};
    int lowest = 256, highest = -1;
    for (DWORD offset = 0; offset < size;) {
        const auto * record = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buffer.data() + offset);
        const int efficiency = record->Processor.EfficiencyClass;
        for (WORD group = 0; group < record->Processor.GroupCount; ++group)
            if (record->Processor.GroupMask[group].Group == 0) masks[efficiency] |= record->Processor.GroupMask[group].Mask;
        lowest = std::min(lowest, efficiency);
        highest = std::max(highest, efficiency);
        offset += record->Size;
    }
    return highest > lowest ? masks[lowest] : 0;
}

// The sections the helper maps and registers ahead, after the requests written: at its start, while the prompt is read, the first copies of decoding
// would otherwise wait for it. Returns the sections asked for.
CPU_LINEAR_API int32_t gguf_moe_ring_prepare(void * handle, const uint64_t * sections, int32_t count) {
    MoeRing * r = (MoeRing *)handle;
    int32_t asked = 0;
    {
        std::lock_guard<std::mutex> lock(r->produce);
        for (; asked < count; ++asked) {
            const int64_t head = r->control->head.load(std::memory_order_relaxed);
            if (head - r->control->tail.load(std::memory_order_acquire) >= MOE_REMOTE_REQUESTS) break;
            r->control->requests[head % MOE_REMOTE_REQUESTS] = {sections[asked], 0, 0, 0, -1, 2, 0};
            r->control->head.store(head + 1, std::memory_order_release);
        }
    }
    SetEvent(r->event);
    return asked;
}

// The loop of a copy helper process (no PyTorch in it): the ring's VRAM and the section of its requests from the parent's handles, the experts' sections
// mapped and registered when first read, a stream of its own. Returns when asked to stop or when the parent has left; 0, or the step that failed.
CPU_LINEAR_API int gguf_moe_copy_helper(uint32_t parent_pid, uint64_t control_value, uint64_t event_value, uint64_t vram_value, uint64_t vram_bytes, int32_t device_index) {
    const CudaDriver & d = cuda_driver();
    const CudaMemory & m = cuda_memory();
    int device = 0;
    void * context = nullptr, * stream = nullptr;
    if (m.init(0) || m.device_get(&device, device_index) || m.primary_retain(&context, device) || d.set_context(context)) return -1;
    HANDLE parent = OpenProcess(PROCESS_DUP_HANDLE | SYNCHRONIZE, FALSE, parent_pid), self = GetCurrentProcess();
    if (parent == nullptr) return -2;
    const auto duplicate = [&](uint64_t value, HANDLE * local) { return DuplicateHandle(parent, (HANDLE)value, self, local, 0, FALSE, DUPLICATE_SAME_ACCESS) != 0; };
    HANDLE control_section = nullptr, event = nullptr, vram_export = nullptr;
    if (!duplicate(control_value, &control_section) || !duplicate(event_value, &event) || !duplicate(vram_value, &vram_export)) return -3;
    MoeRemoteControl * control = (MoeRemoteControl *)MapViewOfFile(control_section, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(MoeRemoteControl));
    if (control == nullptr || control->magic != 0x4d4f4552) return -4;
    if (m.host_register((void *)control, sizeof(MoeRemoteControl), 0)) return -5;  // the done words, written by the copies
    uint64_t generic = 0, vram = 0;
    CuMemAccessDesc access{{1, device_index}, 3};
    if (m.import_handle(&generic, (void *)vram_export, 2) || m.reserve(&vram, (size_t)vram_bytes, 0, 0, 0) || m.map(vram, (size_t)vram_bytes, 0, generic, 0)
        || m.set_access(vram, (size_t)vram_bytes, &access, 1)) return -6;
    if (d.stream_create(&stream, 1)) return -7;
    struct Mapped { uint64_t section; HANDLE local; void * view; };
    std::vector<Mapped> mapped;
    const auto view_of = [&](uint64_t section) -> void * {
        for (const Mapped & entry : mapped)
            if (entry.section == section) return entry.view;
        HANDLE local = nullptr;
        if (!duplicate(section, &local)) return nullptr;
        void * view = MapViewOfFile(local, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
        MEMORY_BASIC_INFORMATION info{};
        if (view == nullptr || !VirtualQuery(view, &info, sizeof(info)) || m.host_register(view, info.RegionSize, 0)) {
            if (view != nullptr) UnmapViewOfFile(view);
            CloseHandle(local);
            return nullptr;
        }
        mapped.push_back({section, local, view});
        return view;
    };
    const auto forget = [&](uint64_t section) {
        for (size_t n = 0; n < mapped.size(); ++n)
            if (mapped[n].section == section) {
                d.stream_synchronize(stream);
                m.host_unregister(mapped[n].view);
                UnmapViewOfFile(mapped[n].view);
                CloseHandle(mapped[n].local);
                mapped.erase(mapped.begin() + (long long)n);
                return;
            }
    };
    // On the efficient cores of a hybrid CPU it spins while requests come (a layer asks every few hundred us while decoding), waking on the event after
    // 20 ms without one: an event's wake-up took about 200 us, half the CPU part of a layer that its copies must land within.
    const DWORD_PTR efficient = efficient_processors();
    if (efficient) SetThreadAffinityMask(GetCurrentThread(), efficient);
    const int64_t spin = efficient ? 20000000 : 0;
    int64_t last = host_trace_clock();
    control->ready = 1;
    HANDLE waits[2] = {event, parent};
    for (;;) {
        const int64_t head = control->head.load(std::memory_order_acquire);
        int64_t tail = control->tail.load(std::memory_order_relaxed);
        if (tail == head) {
            if (control->stop) break;
            if (spin && host_trace_clock() - last < spin) {
                cpu_relax();
                continue;
            }
            if (WaitForMultipleObjects(2, waits, FALSE, 100) == WAIT_OBJECT_0 + 1) break;  // the parent has left
            last = host_trace_clock();
            continue;
        }
        const int64_t now = host_trace_clock();
        int status = 0;
        for (; tail < head; ++tail) {
            const MoeRemoteRequest request = control->requests[tail % MOE_REMOTE_REQUESTS];
            if (request.kind == 1) {
                forget(request.section);
                control->forgotten.store(tail + 1, std::memory_order_release);
                continue;
            }
            if (request.kind == 2) {  // mapped and registered before a job needs it (its first copy would wait for that)
                if (view_of(request.section) == nullptr) status = -12;
                continue;
            }
            void * view = view_of(request.section);
            if (view == nullptr) {  // never lands: the job computes the expert on the CPU (its slot stays taken)
                status = -12;
                continue;
            }
            control->wait_nanoseconds.fetch_add((uint64_t)std::max<int64_t>(0, now - request.asked));
            const int64_t call = host_trace_clock();
            status |= d.to_device(vram + request.target, (const uint8_t *)view + request.offset, (size_t)request.bytes, stream);
            const uint64_t took = (uint64_t)std::max<int64_t>(0, host_trace_clock() - call);
            control->call_nanoseconds.fetch_add(took);
            if (took > control->call_max_nanoseconds.load()) control->call_max_nanoseconds.store(took);
            control->buckets[took < 50000 ? 0 : took < 200000 ? 1 : took < 1000000 ? 2 : took < 10000000 ? 3 : 4].fetch_add(1);
            status |= d.to_host((void *)(control->done + request.slot), vram, sizeof(int32_t), stream);  // the word holding 1, after the copy
            control->issued.fetch_add(1);
        }
        control->tail.store(tail, std::memory_order_release);
        d.query(stream);  // submitted now (WDDM)
        if (status) control->status = (uint32_t)status;
        last = host_trace_clock();
    }
    d.stream_synchronize(stream);
    for (const Mapped & entry : mapped) {
        m.host_unregister(entry.view);
        UnmapViewOfFile(entry.view);
        CloseHandle(entry.local);
    }
    d.stream_destroy(stream);
    m.unmap(vram, (size_t)vram_bytes);
    m.address_free(vram, (size_t)vram_bytes);
    m.release(generic);
    m.host_unregister((void *)control);
    control->ready = 2;
    UnmapViewOfFile(control);
    CloseHandle(control_section);
    CloseHandle(event);
    CloseHandle(vram_export);
    CloseHandle(parent);
    return 0;
}

// The loop of a fill helper process (no CUDA and no PyTorch in it): the fills of level 2 (MoeRemoteFill) asked in the ring's section, from the parent's
// handles of the files and the sections, each mapped when first read, on the efficient cores of a hybrid CPU. Returns when asked to stop or when the
// parent has left; 0, or the step that failed.
CPU_LINEAR_API int gguf_moe_fill_helper(uint32_t parent_pid, uint64_t control_value) {
    HANDLE parent = OpenProcess(PROCESS_DUP_HANDLE | SYNCHRONIZE, FALSE, parent_pid), self = GetCurrentProcess();
    if (parent == nullptr) return -2;
    const auto duplicate = [&](uint64_t value, HANDLE * local) { return DuplicateHandle(parent, (HANDLE)value, self, local, 0, FALSE, DUPLICATE_SAME_ACCESS) != 0; };
    HANDLE control_section = nullptr;
    if (!duplicate(control_value, &control_section)) return -3;
    MoeRemoteControl * control = (MoeRemoteControl *)MapViewOfFile(control_section, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(MoeRemoteControl));
    if (control == nullptr || control->magic != 0x4d4f4552) return -4;
    struct View { uint64_t value; HANDLE local; uint8_t * view; };
    std::vector<View> files, sections;
    const auto view_of = [&](std::vector<View> & views, uint64_t value, DWORD access) -> uint8_t * {
        for (const View & entry : views)
            if (entry.value == value) return entry.view;
        HANDLE local = nullptr;
        if (!duplicate(value, &local)) return nullptr;
        uint8_t * view = (uint8_t *)MapViewOfFile(local, access, 0, 0, 0);
        if (view == nullptr) {
            CloseHandle(local);
            return nullptr;
        }
        views.push_back({value, local, view});
        return view;
    };
    const auto forget = [&](uint64_t value) {
        for (size_t n = 0; n < sections.size(); ++n)
            if (sections[n].value == value) {
                UnmapViewOfFile(sections[n].view);
                CloseHandle(sections[n].local);
                sections.erase(sections.begin() + (long long)n);
                return;
            }
    };
    const auto stopped = [&] { return *(volatile uint32_t *)&control->stop != 0; };
    const WinPrefetch prefetch = (WinPrefetch)(void *)GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    const DWORD_PTR cores = efficient_processors();
    if (cores) SetThreadAffinityMask(GetCurrentThread(), cores);
    control->fill_ready.store(1);
    while (!stopped()) {
        const int64_t head = control->fill_head.load(std::memory_order_acquire);
        int64_t tail = control->fill_tail.load(std::memory_order_relaxed);
        if (tail == head) {  // asked at the placement's rounds, a few times a second: no need to spin
            if (WaitForSingleObject(parent, 1) == WAIT_OBJECT_0) break;  // the parent has left
            continue;
        }
        for (; tail < head && !stopped(); ++tail) {
            MoeRemoteFill & fill = control->fills[tail % MOE_REMOTE_FILLS];
            const int64_t started = host_trace_clock();
            int32_t result = 0;
            if (fill.count == 0) {
                forget(fill.section);
            } else {
                uint8_t * target = view_of(sections, fill.section, FILE_MAP_READ | FILE_MAP_WRITE);
                if (target == nullptr) result = -1;
                for (int32_t k = 0; k < std::min(fill.count, 3) && result == 0; ++k) {
                    const MoeRemoteMove & move = fill.moves[k];
                    uint8_t * source = view_of(files, move.file, FILE_MAP_READ);
                    if (source == nullptr) {
                        result = -2;
                        break;
                    }
                    WinRangeEntry range{source + move.source, (size_t)move.bytes};
                    if (prefetch != nullptr) prefetch(self, 1, &range, 0);  // what a drive must read, read with large requests (touching it would read 64 KiB at a time)
                    result = fill_copy(target + fill.target + move.offset, source + move.source, (size_t)move.bytes);
                    if (fill.drop) {  // out of the working set, to the standby list: available, and read again without the drive if needed soon
                        const uint64_t first = (uint64_t)(source + move.source) & ~4095ull, last = ((uint64_t)(source + move.source) + move.bytes + 4095) & ~4095ull;
                        VirtualUnlock((void *)first, (size_t)(last - first));
                    }
                    control->fill_bytes.fetch_add(move.bytes);
                }
                control->fill_nanoseconds.fetch_add((uint64_t)std::max<int64_t>(0, host_trace_clock() - started));
            }
            fill.result = result;
            control->fill_tail.store(tail + 1, std::memory_order_release);
        }
    }
    for (std::vector<View> * views : {&files, &sections})
        for (const View & entry : *views) {
            UnmapViewOfFile(entry.view);
            CloseHandle(entry.local);
        }
    control->fill_ready.store(2);
    UnmapViewOfFile(control);
    CloseHandle(control_section);
    CloseHandle(parent);
    return 0;
}
#endif

// A measure of the ring (tests): count copies of bytes from sources (host addresses, or with a helper section << 32 | offset) asked for at once, as a job
// does, then waited for: latencies_ns of each, from the request to its done word. Returns the copies made.
CPU_LINEAR_API int32_t gguf_moe_ring_probe(void * handle, const uint64_t * sources, int32_t count, uint64_t bytes, int64_t * latencies_ns) {
    MoeRing & r = *(MoeRing *)handle;
    std::vector<int32_t> slots;
    const int64_t asked = host_trace_clock();
    {
        std::lock_guard<std::mutex> lock(r.remote ? r.produce : r.guard);
        for (int32_t n = 0; n < count; ++n) {
            const int32_t slot = r.take();
            if (slot < 0) break;
            r.done[slot] = 0;
            r.asked[slot] = 1;
            if (r.remote) {
                const int64_t head = r.control->head.load(std::memory_order_relaxed);
                r.control->requests[head % MOE_REMOTE_REQUESTS] = {sources[n] >> 32, sources[n] & 0xffffffffull, r.header + (uint64_t)slot * r.slot_bytes, bytes, slot, 0, asked};
                r.control->head.store(head + 1, std::memory_order_release);
            } else {
                r.queue.push_back({r.base + (uint64_t)slot * r.slot_bytes, (const void *)sources[n], bytes, slot, asked});
            }
            slots.push_back(slot);
        }
    }
#ifdef _WIN32
    if (r.remote) SetEvent(r.event);
    else r.wake.notify_one();
#else
    r.wake.notify_one();
#endif
    std::vector<uint8_t> seen(slots.size(), 0);
    for (size_t left = slots.size(); left;) {
        for (size_t n = 0; n < slots.size(); ++n)
            if (!seen[n] && r.done[slots[n]]) {
                seen[n] = 1;
                latencies_ns[n] = host_trace_clock() - asked;
                --left;
            }
        if (host_trace_clock() - asked > 2000000000) break;
        cpu_relax();
    }
    return (int32_t)slots.size();
}

// A job with a share: the copy engine is asked for part of its missing experts that have a copy in locked RAM (the most read first), the CPU computes the
// others meanwhile, then the share's rows go to the GPU for those that landed, and the CPU computes the rest.
static int64_t moe_share_job(MoeJob & job, const float * payload, const int32_t * ids) {
    MoeShare & s = *job.share;
    MoeRing & r = *s.ring;
    thread_local std::vector<int32_t> counts, candidates, chosen, slots;
    thread_local std::vector<uint8_t> flags, above_tail;
    const int64_t tokens = job.tokens, h = job.h, topk = job.topk, pairs = tokens * topk, experts = job.experts;
    const bool tail = job.tail_rank > 0;
    if (tail) {
        above_tail.assign((size_t)experts, 0);
        for (int64_t n = 0; n < pairs; ++n)
            if (n % topk < job.tail_rank && ids[n] >= 0) above_tail[ids[n]] = 1;
    }
    flags.resize((size_t)experts);
    if (job.resident != nullptr) memcpy(flags.data(), job.resident, (size_t)experts);
    else memset(flags.data(), 0, (size_t)experts);
    counts.assign((size_t)experts, 0);
    candidates.clear();
    int64_t missing = 0;
    for (int64_t n = 0; n < pairs; ++n) {
        const int32_t id = ids[n];
        if (id < 0 || flags[id] || (tail && !above_tail[id])) continue;
        if (!counts[id]++) {
            ++missing;
            if (s.sources[id] && (!r.remote || s.remote_sources[id])) candidates.push_back(id);
        }
    }
    // the CPU keeps one expert at least: the copies land while it computes, a copy with nothing to overlap would come too late and be computed here anyway
    const double fraction = r.paused.load(std::memory_order_relaxed) ? 0.0 : r.remote ? std::min(r.fraction, s.fraction) : s.fraction;
    const int64_t wanted = std::min<int64_t>({(int64_t)candidates.size(), (int64_t)MOE_SHARE_MAX, (int64_t)std::lround(fraction * (double)missing), missing - 1});
    std::partial_sort(candidates.begin(), candidates.begin() + std::max<int64_t>(wanted, 0), candidates.end(), [&](int32_t a, int32_t b) { return counts[a] != counts[b] ? counts[a] > counts[b] : a < b; });
    chosen.clear();
    slots.clear();
    if (wanted > 0) {
        std::lock_guard<std::mutex> lock(r.remote ? r.produce : r.guard);
        for (int64_t n = 0; n < wanted; ++n) {
            if (r.remote && r.control->head.load(std::memory_order_relaxed) - r.control->tail.load(std::memory_order_acquire) >= MOE_REMOTE_REQUESTS) break;
            const int32_t slot = r.take();
            if (slot < 0) break;
            const int32_t id = candidates[n];
            r.done[slot] = 0;
            r.asked[slot] = 1;
            if (r.remote) {
                const int64_t head = r.control->head.load(std::memory_order_relaxed);
                r.control->requests[head % MOE_REMOTE_REQUESTS] = {s.remote_sources[id] >> 32, s.remote_sources[id] & 0xffffffffull, r.header + (uint64_t)slot * r.slot_bytes, s.slot_bytes, slot, 0, host_trace_clock()};
                r.control->head.store(head + 1, std::memory_order_release);
            } else {
                r.queue.push_back({r.base + (uint64_t)slot * r.slot_bytes, (const void *)s.sources[id], s.slot_bytes, slot, host_trace_clock()});
            }
            chosen.push_back(id);
            slots.push_back(slot);
            flags[id] = 1;
        }
    }
    if (!chosen.empty()) {
        if (r.remote) {
#ifdef _WIN32
            SetEvent(r.event);
#endif
        } else {
            int64_t expected = 0;
            r.notified_at.compare_exchange_strong(expected, host_trace_clock());
            r.wake.notify_one();
        }
    }
    const int64_t part_started = chosen.empty() ? 0 : host_trace_clock();
    const uint64_t * offsets = s.offsets;
    if (r.remote) {  // its rows now, the done word of each copy after them: the GPU waits for the copies and computes them beside the CPU's part (a helper process
                     // issues them: no wait of the GPU on a driver call of this process)
        for (size_t n = 0; n < chosen.size(); ++n) {
            const uint64_t slot = r.base + (uint64_t)slots[n] * r.slot_bytes;
            int64_t * row = job.share_rows + 1 + 5 * (int64_t)n;
            row[0] = chosen[n];
            for (int k = 0; k < 3; ++k) row[1 + k] = (int64_t)(slot + offsets[k]);
            row[4] = (int64_t)(r.done_device + 4 * (uint64_t)slots[n]);
        }
        ((volatile int64_t *)job.share_rows)[MOE_SHARE_DONE] = 0;  // set by the GPU once it has computed its experts and the share
        std::atomic_thread_fence(std::memory_order_release);
        ((volatile int64_t *)job.share_rows)[0] = (int64_t)chosen.size();
    }
    int64_t count = moe_forward(job.gate, job.up, job.down, job.gate_type, job.up_type, job.down_type, payload, payload + tokens * h, ids, flags.data(), job.out, tokens, h, job.i, experts, topk,
                                job.threads, job.tail_rank, s.sources, offsets);
    if (count < 0) return count;
    if (part_started) {
        ++r.parts;
        r.part_nanoseconds += (uint64_t)std::max<int64_t>(0, host_trace_clock() - part_started);
    }
    int64_t landed = 0, late = 0;
    if (r.remote) {
        for (size_t n = 0; n < chosen.size(); ++n) landed += r.done[slots[n]] != 0;  // within the CPU's part
        // the share where the GPU ends its pass with the CPU's part: it computes its experts in VRAM and the share once the copies have landed, so copies
        // landing just in time still made the layer wait for the GPU's pass (thoughts 11 % slower with the share when it followed the copies alone)
        if (!chosen.empty()) {
            const bool ahead = ((volatile int64_t *)job.share_rows)[MOE_SHARE_DONE] != 0;
            r.fraction = ahead ? std::min(s.fraction, r.fraction + 0.01f) : std::max(0.02f, r.fraction - 0.02f);
        }
        chosen.clear();
    }
    for (size_t n = 0; n < chosen.size(); ++n) {
        if (r.done[slots[n]]) {
            const uint64_t slot = r.base + (uint64_t)slots[n] * r.slot_bytes;
            int64_t * row = job.share_rows + 1 + 4 * landed++;
            row[0] = chosen[n];
            for (int k = 0; k < 3; ++k) row[1 + k] = (int64_t)(slot + offsets[k]);
        } else {
            flags[chosen[n]] = 0;  // on its way still: the CPU computes it after the others
            ++late;
        }
    }
    if (!r.remote) job.share_rows[0] = landed;
    if (late) {
        for (size_t n = 0; n < chosen.size(); ++n) flags[chosen[n]] = flags[chosen[n]] ? 1 : 2;
        for (int64_t e = 0; e < experts; ++e) flags[e] = flags[e] == 2 ? 0 : 1;
        const int64_t more = moe_forward(job.gate, job.up, job.down, job.gate_type, job.up_type, job.down_type, payload, payload + tokens * h, ids, flags.data(), job.out, tokens, h, job.i, experts, topk,
                                         job.threads, job.tail_rank, s.sources, offsets, count > 0);
        if (more < 0) return more;
        count += more;
    }
    if (r.status) s.status = r.status;
    s.asked += r.remote ? (uint64_t)slots.size() : chosen.size();
    s.landed += (uint64_t)landed;
    s.missing += (uint64_t)missing;
    ++s.jobs;
    return count;
}

// A job with a share read in place: the GPU computes part of the job's missing experts that have a copy in mapped RAM (the most read first) straight from
// there, over the bus, beside the CPU's part; their rows are published at once. No copy and no driver call. The share follows whether the GPU ended
// its part before the CPU (the word it sets after the share), at most the share's fraction.
static int64_t moe_mapped_job(MoeJob & job, const float * payload, const int32_t * ids) {
    MoeShare & s = *job.share;
    thread_local std::vector<int32_t> counts, candidates;
    thread_local std::vector<uint8_t> flags, above_tail;
    const int64_t tokens = job.tokens, h = job.h, topk = job.topk, pairs = tokens * topk, experts = job.experts;
    const bool tail = job.tail_rank > 0;
    if (tail) {
        above_tail.assign((size_t)experts, 0);
        for (int64_t n = 0; n < pairs; ++n)
            if (n % topk < job.tail_rank && ids[n] >= 0) above_tail[ids[n]] = 1;
    }
    flags.resize((size_t)experts);
    if (job.resident != nullptr) memcpy(flags.data(), job.resident, (size_t)experts);
    else memset(flags.data(), 0, (size_t)experts);
    counts.assign((size_t)experts, 0);
    candidates.clear();
    int64_t missing = 0;
    for (int64_t n = 0; n < pairs; ++n) {
        const int32_t id = ids[n];
        if (id < 0 || flags[id] || (tail && !above_tail[id])) continue;
        if (!counts[id]++) {
            ++missing;
            if (s.sources[id] && s.device_sources[id]) candidates.push_back(id);
        }
    }
    if (s.live <= 0) s.live = std::min(0.25f, s.fraction);
    const int64_t wanted = std::max<int64_t>(0, std::min<int64_t>({(int64_t)candidates.size(), (int64_t)MOE_SHARE_MAX, (int64_t)std::lround(std::min(s.live, s.fraction) * (double)missing), missing - 1}));
    std::partial_sort(candidates.begin(), candidates.begin() + wanted, candidates.end(), [&](int32_t a, int32_t b) { return counts[a] != counts[b] ? counts[a] > counts[b] : a < b; });
    volatile int64_t * rows = job.share_rows;
    for (int64_t n = 0; n < wanted; ++n) {
        const int32_t id = candidates[n];
        volatile int64_t * row = rows + 1 + 5 * n;
        row[0] = id;
        for (int k = 0; k < 3; ++k) row[1 + k] = (int64_t)(s.device_sources[id] + s.offsets[k]);
        row[4] = 0;  // in place: nothing to wait for
        flags[id] = 1;
    }
    rows[MOE_SHARE_DONE] = 0;
    std::atomic_thread_fence(std::memory_order_release);
    rows[0] = wanted;
    const int64_t count = moe_forward(job.gate, job.up, job.down, job.gate_type, job.up_type, job.down_type, payload, payload + tokens * h, ids, flags.data(), job.out, tokens, h, job.i, experts, topk,
                                      job.threads, job.tail_rank, s.sources, s.offsets);
    if (count < 0) return count;
    if (wanted > 0) {
        const bool ahead = rows[MOE_SHARE_DONE] != 0;  // the GPU ended its share before the CPU its part: it can take more
        s.live = ahead ? std::min(s.fraction, s.live + 0.01f) : std::max(0.02f, s.live - 0.02f);
        if (ahead) ++s.landed;
    }
    s.asked += (uint64_t)wanted;
    s.missing += (uint64_t)missing;
    ++s.jobs;
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
        const int32_t * routed = (const int32_t *)(payload + tokens * (h + topk));
        for (int64_t n = 0; n < tokens * topk; ++n) ++job.usage[routed[n]];
    }
    const int32_t * ids = (const int32_t *)(payload + tokens * (h + topk));
    const int64_t count = job.share != nullptr ? (job.share->ring == nullptr ? moe_mapped_job(job, payload, ids) : moe_share_job(job, payload, ids))
                                               : moe_forward(job.gate, job.up, job.down, job.gate_type, job.up_type, job.down_type, payload, payload + tokens * h, ids, job.resident, job.out, tokens, h, job.i,
                                                             job.experts, topk, job.threads, job.tail_rank, nullptr, nullptr);
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
