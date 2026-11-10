// p0/band.cpp -- the window's effective read bandwidth on the REAL expert access pattern.
//
// The campaign's per-token arithmetic needs ONE number the delivery's records do not have: what a
// GPU reading a registered window achieves when the reads look like a routed MoE decode (a handful of
// random, per-expert sized blocks per op), rather than the 1 GiB sequential sum the old probe measured
// (198-233 GB/s).  The two differ by up to 4x, because a 0.6-3.3 MiB block is latency-dominated at low
// concurrency -- which is precisely the regime a batch-1 decode is in.
//
// Backings, one process each (registration pins and is per-process, so never share):
//   anon-ro    anonymous mmap, filled, hipHostRegister(Mapped|ReadOnly)   <- the design's window
//   anon-rw    the same, registered without ReadOnly                      <- the 60 GB/s arm
//   dev        hipMalloc (the GTT buffer the arena is carved from)
//   file-ro    the real GGUF mmap, read-only, registered ReadOnly          <- a ZERO-COPY window
//
// Two launch shapes, both taken from the real path:
//   wide  ONE kernel of `reads` blocks (an op that merges the routed rows -- what MMB/wide-ubatch does)
//   layer `layers` kernels of 8 blocks each (what a per-layer routed op does at decode)
//
// Usage: band <arm> [bytes/expert] [reads] [footprint_GiB] [iters]
//   defaults: 610352 B (the 35B's Q4_K_M expert), 320 reads, 8 GiB, 60 iterations
//   the 122B's Q8_0 expert is 3.3 MiB -- pass it to model the stress model
//
// Build: hipcc -O2 --offload-arch=gfx1151 -o band band.cpp
#include <hip/hip_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(2); } \
} while (0)

// One block per routed row, exactly like the cold branch of `mul_mat_vec_q_moe`: the base is one
// window pointer and each row's byte offset comes from the id array.
__global__ void read_rows(const float4 * __restrict__ base, const long * __restrict__ offs,
                          long n4, float * __restrict__ sink) {
    const float4 * p = (const float4 *) ((const char *) base + offs[blockIdx.x]);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (long i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = __ldg(p + i);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    const float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 1e30f) {                     // never true for real data; keeps the loads alive
        sink[blockIdx.x] = s;
    }
}

static uint32_t g_seed = 12345;
static uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <anon-ro|anon-rw|dev|file-ro> [bytes/expert] [reads] [GiB] [iters]\n", argv[0]);
        return 2;
    }
    const std::string arm = argv[1];
    const long   eb    = argc > 2 ? atol(argv[2]) : 610352;
    const int    reads = argc > 3 ? atoi(argv[3]) : 320;
    const double gib   = argc > 4 ? atof(argv[4]) : 8.0;
    const int    iters = argc > 5 ? atoi(argv[5]) : 60;

    const char * gguf = getenv("GGUF");
    if (gguf == nullptr) {
        gguf = "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    }

    long span = (long) (gib * (1L << 30));
    if (span < (long) eb * reads * 8) {   // a read set 8x larger than one op, so "random" is real
        span = (long) eb * reads * 8;
    }
    span = (span / eb) * eb;

    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    printf("dev=%s integrated=%d canMapHost=%d threads=%s\n", prop.name, prop.integrated,
           prop.canMapHostMemory, getenv("BAND_THREADS") ? getenv("BAND_THREADS") : "256");

    void * host = nullptr;
    void * dev  = nullptr;
    bool   registered = false;
    bool   file_backed = false;

    if (arm == "dev") {
        CK(hipMalloc(&dev, span));
        CK(hipMemset(dev, 0x11, span));
    } else if (arm == "anon-ro" || arm == "anon-rw") {
        host = mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (host == MAP_FAILED) { perror("mmap anon"); return 2; }
        madvise(host, span, MADV_HUGEPAGE);
        memset(host, 0x11, span);                                  // FILL FIRST, then register
        unsigned flags = hipHostRegisterMapped | hipHostRegisterPortable;
        if (arm == "anon-ro") {
            flags |= hipHostRegisterReadOnly;
        }
        CK(hipHostRegister(host, span, flags));
        registered = true;
        CK(hipHostGetDevicePointer(&dev, host, 0));
    } else if (arm == "file-ro") {
        const int fd = open(gguf, O_RDONLY);
        if (fd < 0) { perror(gguf); return 2; }
        const off_t sz = lseek(fd, 0, SEEK_END);
        if (sz < (off_t) span) { span = ((long) sz / eb) * eb; }
        host = mmap(nullptr, span, PROT_READ, MAP_PRIVATE, fd, 0);
        if (host == MAP_FAILED) { perror("mmap file"); return 2; }
        madvise(host, span, MADV_WILLNEED);
        CK(hipHostRegister(host, span, hipHostRegisterMapped | hipHostRegisterPortable |
                                     hipHostRegisterReadOnly));
        registered = true;
        file_backed = true;
        CK(hipHostGetDevicePointer(&dev, host, 0));
    } else {
        fprintf(stderr, "unknown arm %s\n", arm.c_str());
        return 2;
    }
    const long n_experts = span / eb;
    const long n4        = eb / (long) sizeof(float4);

    long   * d_offs = nullptr;
    float  * d_sink = nullptr;
    CK(hipMalloc(&d_offs, sizeof(long) * (size_t) reads));
    CK(hipMalloc(&d_sink, sizeof(float) * (size_t) reads));

    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    const int threads = getenv("BAND_THREADS") ? atoi(getenv("BAND_THREADS")) : 256;

    std::vector<long> offs((size_t) reads);
    auto fill = [&]() {
        for (int k = 0; k < reads; k++) {
            offs[(size_t) k] = (long) (rnd() % (uint32_t) n_experts) * eb;
        }
    };

    // wide: one launch, one block per routed row (the merged-op shape).
    auto wide = [&](int it) -> double {
        fill();
        CK(hipMemcpy(d_offs, offs.data(), sizeof(long) * (size_t) reads, hipMemcpyHostToDevice));
        CK(hipEventRecord(e0));
        read_rows<<<reads, threads>>>((const float4 *) dev, d_offs, n4, d_sink);
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float t = 0; CK(hipEventElapsedTime(&t, e0, e1));
        (void) it;
        return (double) t;
    };

    // layer: `reads/8` launches of 8 blocks (the per-layer routed op shape at decode).
    auto layered = [&](int it) -> double {
        const int per = 8;
        const int nl  = reads / per;
        fill();
        CK(hipMemcpy(d_offs, offs.data(), sizeof(long) * (size_t) reads, hipMemcpyHostToDevice));
        CK(hipEventRecord(e0));
        for (int l = 0; l < nl; l++) {
            read_rows<<<per, threads>>>((const float4 *) dev, d_offs + (size_t) l * per, n4, d_sink);
        }
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float t = 0; CK(hipEventElapsedTime(&t, e0, e1));
        (void) it;
        return (double) t;
    };

    (void) wide(0);                                        // warm-up: page tables, TLB, caches
    CK(hipDeviceSynchronize());

    for (int mode = 0; mode < 2; mode++) {
        std::vector<double> ms;
        for (int i = 0; i < iters; i++) {
            ms.push_back(mode == 0 ? wide(i) : layered(i));
        }
        std::sort(ms.begin(), ms.end());
        const double med   = ms[ms.size() / 2];
        const double bytes = (double) eb * (double) reads;
        const double gbs   = bytes / (med / 1000.0) / 1e9;
        printf("%-8s expert=%-8ld reads/op=%-4d span=%5.2f GiB (%-6ld experts) %-26s  %s: %7.3f ms -> %6.1f GB/s"
               "   [min %.3f max %.3f]\n",
               arm.c_str(), eb, reads, (double) span / (1L << 30), n_experts,
               registered ? (file_backed ? "registered RO file(zero-copy)" : "registered host")
                          : "hipMalloc device",
               mode == 0 ? "wide " : "layer", med, gbs, ms.front(), ms.back());
    }
    return 0;
}
