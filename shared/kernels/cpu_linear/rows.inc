// Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
// Rows of a large table read from its file past the OS cache, included by linear.cpp: a table much larger than RAM of which a pass needs a few
// scattered rows (an n-gram embedding table). The rows of a call are grouped by aligned page and read concurrently (Windows: overlapped reads and a
// completion port; elsewhere: blocking reads on short-lived threads), and a small set-associative cache of rows keeps the ones in use, which common
// n-grams repeat. Only the rows a pass asks for are read. Decoding: IQ4_NL rows into float32.
#include <cstdio>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static const size_t ROW_PAGE = 4096;
static const int ROW_WINDOW = 64;  // concurrent reads
static const uint32_t ROW_WAYS = 8, ROW_EMPTY = 0xFFFFFFFFu;

struct RowRead {
#ifdef _WIN32
    OVERLAPPED overlapped;
#endif
    uint8_t * buffer;
    uint64_t offset;
    uint32_t bytes;
    int64_t first, last;  // the misses of this read, in the sorted list
    int status;
};

struct RowMiss { int64_t row; int64_t output; };

// Rows read ahead of a pass, on a thread of their own: they reach the cache of rows, where the pass finds them.
struct RowPrefetch {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable wake;
    std::vector<int64_t> rows, reading;
    std::vector<uint8_t> bytes;
    uint64_t pending = 0;
    bool stop = false;
};

struct RowTable {
    RowPrefetch * prefetch = nullptr;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE, port = nullptr;
#else
    int file = -1;
#endif
    uint64_t offset = 0, rows = 0;
    uint32_t row_bytes = 0;
    uint64_t sets = 0;
    uint32_t * keys = nullptr;
    uint8_t * data = nullptr, * next = nullptr;  // in cache_memory, or in own_cache without it
    std::vector<uint8_t> own_cache;
    uint8_t * slab = nullptr;
    RowRead reads[ROW_WINDOW];
    std::vector<RowMiss> misses;
    std::mutex serial;
    uint64_t lookups = 0, hits = 0, read_count = 0, read_bytes = 0, read_us = 0, calls = 0;
};

static inline uint64_t row_set(const RowTable & t, uint32_t row) {
    uint64_t x = row * 0x9E3779B97F4A7C15ull;
    return (x ^ (x >> 29)) % t.sets;
}

static inline const uint8_t * row_find(const RowTable & t, uint32_t row) {
    if (!t.sets) return nullptr;
    const uint64_t s = row_set(t, row);
    const uint32_t * keys = t.keys + s * ROW_WAYS;
    for (uint32_t w = 0; w < ROW_WAYS; ++w)
        if (keys[w] == row) return t.data + (s * ROW_WAYS + w) * t.row_bytes;
    return nullptr;
}

static inline void row_insert(RowTable & t, uint32_t row, const uint8_t * bytes) {
    if (!t.sets || row_find(t, row)) return;
    const uint64_t s = row_set(t, row);
    const uint32_t w = t.next[s];
    t.next[s] = (uint8_t)((w + 1) % ROW_WAYS);
    t.keys[s * ROW_WAYS + w] = row;
    memcpy(t.data + (s * ROW_WAYS + w) * t.row_bytes, bytes, t.row_bytes);
}

// Opens the table; null when the file cannot be opened. cache_bytes: the cache of rows (0: none), in cache_memory (RAM the caller locked, kept until row_table_close), or null for ordinary RAM.
CPU_LINEAR_API void * row_table_open(const char * utf8_path, uint64_t offset, uint32_t row_bytes, uint64_t rows, uint64_t cache_bytes, void * cache_memory) {
    RowTable * t = new RowTable();
#ifdef _WIN32
    wchar_t path[4096];
    if (!MultiByteToWideChar(CP_UTF8, 0, utf8_path, -1, path, 4096)) { delete t; return nullptr; }
    t->file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (t->file == INVALID_HANDLE_VALUE) { delete t; return nullptr; }
    // no concurrency limit of 1: a thread that once waited on the port counts as running on it, so a main thread spinning on the stream of a host job would keep that job from its completions
    t->port = CreateIoCompletionPort(t->file, nullptr, 0, 0);
    t->slab = (uint8_t *)VirtualAlloc(nullptr, ROW_WINDOW * 2 * ROW_PAGE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    t->file = open(utf8_path, O_RDONLY | O_DIRECT);
    if (t->file < 0) t->file = open(utf8_path, O_RDONLY);
    if (t->file < 0) { delete t; return nullptr; }
    void * slab = nullptr;
    if (posix_memalign(&slab, ROW_PAGE, ROW_WINDOW * 2 * ROW_PAGE)) slab = nullptr;
    t->slab = (uint8_t *)slab;
#endif
    t->offset = offset;
    t->row_bytes = row_bytes;
    t->rows = rows;
    t->sets = cache_bytes / ((uint64_t)ROW_WAYS * (row_bytes + 4) + 1);
    if (t->sets) {
        uint8_t * memory = (uint8_t *)cache_memory;
        if (memory == nullptr) {
            t->own_cache.resize((size_t)cache_bytes);
            memory = t->own_cache.data();
        }
        const size_t ways = (size_t)(t->sets * ROW_WAYS);
        t->keys = (uint32_t *)memory;
        t->data = memory + ways * sizeof(uint32_t);
        t->next = t->data + ways * row_bytes;
        std::fill_n(t->keys, ways, ROW_EMPTY);
        memset(t->next, 0, (size_t)t->sets);
    }
    for (int i = 0; i < ROW_WINDOW; ++i) t->reads[i].buffer = t->slab + (size_t)i * 2 * ROW_PAGE;
    return t;
}

CPU_LINEAR_API void row_table_close(void * table) {
    RowTable * t = (RowTable *)table;
    if (t->prefetch != nullptr) {
        {
            std::lock_guard<std::mutex> lock(t->prefetch->mutex);
            t->prefetch->stop = true;
        }
        t->prefetch->wake.notify_all();
        t->prefetch->thread.join();
        delete t->prefetch;
    }
#ifdef _WIN32
    CloseHandle(t->file);
    CloseHandle(t->port);
    VirtualFree(t->slab, 0, MEM_RELEASE);
#else
    close(t->file);
    free(t->slab);
#endif
    delete t;
}

// lookups, cache hits, reads, bytes read, microseconds spent reading, calls
CPU_LINEAR_API void row_table_stats(void * table, uint64_t * out) {
    RowTable * t = (RowTable *)table;
    out[0] = t->lookups; out[1] = t->hits; out[2] = t->read_count; out[3] = t->read_bytes; out[4] = t->read_us; out[5] = t->calls;
}

static void row_table_finish(RowTable & t, const RowRead & read, uint8_t * out) {
    const uint32_t rb = t.row_bytes;
    for (int64_t m = read.first; m < read.last; ++m) {
        const RowMiss & miss = t.misses[(size_t)m];
        const uint64_t at = t.offset + (uint64_t)miss.row * rb;
        const uint8_t * bytes = read.buffer + (at - read.offset);
        memcpy(out + (size_t)miss.output * rb, bytes, rb);
        row_insert(t, (uint32_t)miss.row, bytes);
    }
}

// The rows [count] into out [count, row_bytes]. Returns 0, or a nonzero system error.
CPU_LINEAR_API int row_table_read(void * table, const int64_t * rows, int64_t count, uint8_t * out) {
    RowTable & t = *(RowTable *)table;
    std::lock_guard<std::mutex> guard(t.serial);
    const uint32_t rb = t.row_bytes;
    t.misses.clear();
    t.lookups += (uint64_t)count;
    ++t.calls;
    for (int64_t n = 0; n < count; ++n) {
        const uint8_t * hit = row_find(t, (uint32_t)rows[n]);
        if (hit) memcpy(out + (size_t)n * rb, hit, rb);
        else t.misses.push_back({rows[n], n});
    }
    t.hits += (uint64_t)count - t.misses.size();
    if (t.misses.empty()) return 0;
    std::sort(t.misses.begin(), t.misses.end(), [](const RowMiss & a, const RowMiss & b) { return a.row < b.row; });
    const auto start = std::chrono::steady_clock::now();
    const int64_t total = (int64_t)t.misses.size();
    int64_t next = 0;
    int error = 0;
    // the next read: from the first waiting row, the rows that fit in its two pages
    auto prepare = [&](RowRead & read) {
        const uint64_t at = t.offset + (uint64_t)t.misses[(size_t)next].row * rb;
        read.offset = at / ROW_PAGE * ROW_PAGE;
        const uint64_t limit = read.offset + 2 * ROW_PAGE;
        read.first = next;
        uint64_t end = at + rb;
        for (++next; next < total; ++next) {
            const uint64_t row_end = t.offset + (uint64_t)t.misses[(size_t)next].row * rb + rb;
            if (row_end > limit) break;
            end = row_end;
        }
        read.last = next;
        read.bytes = (uint32_t)(((end - read.offset) + ROW_PAGE - 1) / ROW_PAGE * ROW_PAGE);
        t.read_bytes += read.bytes;
        ++t.read_count;
    };
#ifdef _WIN32
    RowRead * free_reads[ROW_WINDOW];
    int free_count = ROW_WINDOW, running = 0;
    for (int i = 0; i < ROW_WINDOW; ++i) free_reads[i] = &t.reads[i];
    while (next < total || running) {
        while (next < total && free_count && !error) {
            RowRead * read = free_reads[--free_count];
            prepare(*read);
            memset(&read->overlapped, 0, sizeof(read->overlapped));
            read->overlapped.Offset = (DWORD)read->offset;
            read->overlapped.OffsetHigh = (DWORD)(read->offset >> 32);
            if (!ReadFile(t.file, read->buffer, read->bytes, nullptr, &read->overlapped) && GetLastError() != ERROR_IO_PENDING) {
                error = (int)GetLastError();
                free_reads[free_count++] = read;
                break;
            }
            ++running;
        }
        if (!running) break;
        OVERLAPPED_ENTRY entries[ROW_WINDOW];
        ULONG got = 0;
        if (!GetQueuedCompletionStatusEx(t.port, entries, ROW_WINDOW, &got, INFINITE, FALSE)) { error = (int)GetLastError(); break; }
        for (ULONG e = 0; e < got; ++e) {
            RowRead * read = CONTAINING_RECORD(entries[e].lpOverlapped, RowRead, overlapped);
            if (entries[e].dwNumberOfBytesTransferred < (DWORD)(read->offset + read->bytes > 0 ? 1 : 0)) error = 38;  // past the end of the file
            else row_table_finish(t, *read, out);
            free_reads[free_count++] = read;
            --running;
        }
        if (error) next = total;
    }
#else
    while (next < total && !error) {
        int prepared = 0;
        while (next < total && prepared < ROW_WINDOW) prepare(t.reads[prepared++]);
        std::atomic<int> cursor{0}, failed{0};
        auto worker = [&] {
            for (int r; (r = cursor.fetch_add(1)) < prepared;)
                if (pread(t.file, t.reads[r].buffer, t.reads[r].bytes, (off_t)t.reads[r].offset) <= 0) failed.store(1);
        };
        std::vector<std::thread> threads;
        for (int n = 1; n < std::min(prepared, 16); ++n) threads.emplace_back(worker);
        worker();
        for (std::thread & thread : threads) thread.join();
        if (failed.load()) error = 5;
        else for (int r = 0; r < prepared; ++r) row_table_finish(t, t.reads[r], out);
    }
#endif
    t.read_us += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    return error;
}

// IQ4_NL rows (blocks of 32 values: FP16 scale and 16 bytes of nibbles) into float32
static void iq4nl_rows_to_float(const uint8_t * rows, int64_t count, int64_t values, float * out) {
    const int64_t blocks = values / 32;
    for (int64_t n = 0; n < count * blocks; ++n) {
        const block_iq4_nl & b = ((const block_iq4_nl *)rows)[n];
        const float d = GGML_CPU_FP16_TO_FP32(b.d);
        float * o = out + n * 32;
        for (int j = 0; j < 16; ++j) {
            o[j] = d * kvalues_iq4nl_port[b.qs[j] & 15];
            o[j + 16] = d * kvalues_iq4nl_port[b.qs[j] >> 4];
        }
    }
}

// The hash of a table of hashed n-grams: per n-gram order, heads_per_ngram rows of the table, each mixed % vocab + offset.
struct NgramJob {  // ctypes structure of shared/llm_engines/lmgp/layers/ngram_table.py
    void * table;
    const void * context;     // int32 [tokens, ngram_size]: each token, then its predecessors (the end token where the window was cut)
    float * out;              // float32 [tokens, heads, values]
    const uint64_t * multipliers;
    const uint64_t * offsets;
    const uint64_t * vocab_sizes;
    int32_t tokens, ngram_size, heads_per_ngram, values, status, reserved;
};

static void ngram_row_numbers(const NgramJob * job, std::vector<int64_t> & rows) {
    const int64_t tokens = job->tokens, grams = job->ngram_size, per = job->heads_per_ngram, heads = (grams - 1) * per;
    const int32_t * context = (const int32_t *)job->context;
    rows.resize((size_t)(tokens * heads));
    for (int64_t n = 0; n < tokens; ++n) {
        const int32_t * c = context + n * grams;
        uint64_t mixed = (uint64_t)(int64_t)c[0] * job->multipliers[0];
        for (int64_t g = 1; g < grams; ++g) {
            mixed ^= (uint64_t)(int64_t)c[g] * job->multipliers[g];
            for (int64_t h = 0; h < per; ++h) {
                const int64_t head = (g - 1) * per + h;
                rows[(size_t)(n * heads + head)] = (int64_t)(mixed % job->vocab_sizes[head] + job->offsets[head]);
            }
        }
    }
}

static void row_prefetch_main(RowTable * t) {
    RowPrefetch & p = *t->prefetch;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(p.mutex);
            p.wake.wait(lock, [&] { return p.stop || p.pending; });
            if (p.stop) return;
            p.pending = 0;
            p.reading.swap(p.rows);
        }
        p.bytes.resize(p.reading.size() * t->row_bytes);
        row_table_read(t, p.reading.data(), (int64_t)p.reading.size(), p.bytes.data());  // errors show in the pass, which reads again what is missing
    }
}

// The rows of a job (its context, its token count, read now) start reaching the cache of rows; a new call replaces what has not started.
CPU_LINEAR_API void ngram_rows_prefetch(NgramJob * job) {
    RowTable & t = *(RowTable *)job->table;
    if (t.prefetch == nullptr) {
        t.prefetch = new RowPrefetch();
        t.prefetch->thread = std::thread(row_prefetch_main, &t);
    }
    RowPrefetch & p = *t.prefetch;
    {
        std::lock_guard<std::mutex> lock(p.mutex);
        ngram_row_numbers(job, p.rows);
        p.pending = 1;
    }
    p.wake.notify_one();
}

CPU_LINEAR_API int ngram_rows_forward(NgramJob * job) {
    static thread_local std::vector<int64_t> rows;
    static thread_local std::vector<uint8_t> bytes;
    RowTable & t = *(RowTable *)job->table;
    const int64_t tokens = job->tokens, grams = job->ngram_size, per = job->heads_per_ngram, heads = (grams - 1) * per;
    ngram_row_numbers(job, rows);
    bytes.resize((size_t)(tokens * heads) * t.row_bytes);
    const int status = row_table_read(job->table, rows.data(), tokens * heads, bytes.data());
    if (!status) iq4nl_rows_to_float(bytes.data(), tokens * heads, job->values, job->out);
    return status;
}

// A CUDA host function: the embeddings of a job
CPU_LINEAR_API void ngram_rows_host_job(void * data) {
    NgramJob * job = (NgramJob *)data;
    const int64_t traced = host_stamps.load(std::memory_order_relaxed) == nullptr ? 0 : host_trace_clock();
    job->status = ngram_rows_forward(job);
    if (traced) host_trace(1, traced, job->tokens);
}
