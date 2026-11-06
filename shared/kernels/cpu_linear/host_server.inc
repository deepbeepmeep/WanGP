// Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
// Host jobs that GPU kernels ask for by writing a word to mapped RAM, an idea of Strata (MIT License, https://github.com/Niko1221/Strata), included by linear.cpp: a thread of this
// library spins on the word, runs the job and answers in another word that a GPU kernel reads (shared/kernels/host_doorbell.py). The GPU goes on with its own work meanwhile; no copy engine,
// no driver call, and the CUDA graphs around it stay one stream of compute kernels. A kernel waits for the answer of its job before asking for the next one.
// Long idle, the thread polls every millisecond: a GPU kernel waiting for it always gets its answer, a little later.
static const std::chrono::milliseconds HOST_SPIN(200);
static const int HOST_CALLS = 16384;

struct HostCall {
    int (*function)(void *);  // nonzero: a result to read; null: the job is a void function reporting nothing, whose result is always read
    void (*void_function)(void *);
    void * job;
};

struct HostServer {
    volatile uint32_t * mailbox = nullptr;   // the number of the call + 1, written by the GPU
    volatile uint32_t * answers = nullptr;   // per call: 0 until it is done, 1 nothing to read, 2 a result to read
    HostCall calls[HOST_CALLS];
    std::atomic<bool> stop{false};
    std::thread thread;
    std::atomic<uint64_t> served{0};
};

static HostServer * host_server = nullptr;

static void host_server_main(HostServer * s) {
    auto last = std::chrono::steady_clock::now();
    uint32_t spins = 0;
    for (;;) {
        const uint32_t ticket = *s->mailbox;
        if (!ticket) {
            cpu_relax();
            if (!(++spins & 4095u)) {
                if (s->stop.load(std::memory_order_relaxed)) return;
                if (std::chrono::steady_clock::now() - last > HOST_SPIN) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            continue;
        }
        *s->mailbox = 0;
        const HostCall & call = s->calls[ticket - 1];
        int read = 1;
        if (call.function != nullptr) read = call.function(call.job);
        else call.void_function(call.job);
        s->answers[ticket - 1] = read ? 2u : 1u;
        s->served.fetch_add(1, std::memory_order_relaxed);
        last = std::chrono::steady_clock::now();
        spins = 0;
    }
}

// mailbox: a word of mapped RAM, answers: HOST_CALLS words of mapped RAM, kept until host_server_close.
CPU_LINEAR_API void host_server_open(void * mailbox, void * answers) {
    host_server = new HostServer();
    host_server->mailbox = (volatile uint32_t *)mailbox;
    host_server->answers = (volatile uint32_t *)answers;
    *host_server->mailbox = 0;
    host_server->thread = std::thread(host_server_main, host_server);
}

// Nothing in flight may ask for a job any more. A kernel still waiting (the process leaving) is let go, with nothing to read.
CPU_LINEAR_API void host_server_close() {
    host_server->stop.store(true);
    host_server->thread.join();
    for (int call = 0; call < HOST_CALLS; ++call) host_server->answers[call] = 1;
    delete host_server;
    host_server = nullptr;
}

CPU_LINEAR_API uint64_t host_server_served() { return host_server == nullptr ? 0 : host_server->served.load(); }

// Call number index runs function(job) (returning whether there is a result to read), or void_function(job) (a result is always read).
CPU_LINEAR_API void host_server_set_call(int index, void * function, void * void_function, void * job) {
    host_server->calls[index] = {(int (*)(void *))function, (void (*)(void *))void_function, job};
}

CPU_LINEAR_API int host_server_capacity() { return HOST_CALLS; }

// The job of a bench: spins for *(int64_t *)data nanoseconds; a result to read when negative.
CPU_LINEAR_API int host_server_spin_job(void * data) {
    const int64_t ns = *(const int64_t *)data;
    const int64_t until = host_trace_clock() + (ns < 0 ? -ns : ns);
    while (host_trace_clock() < until) cpu_relax();
    return ns < 0;
}
