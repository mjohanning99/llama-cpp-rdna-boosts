// p0/winprobe.cpp -- P2 step 1: a ZERO-COPY window built out of the model file's own pages.
//
// The design (README §2, as amended 2026-11-10) needs a fixed-stride, expert-indexed, read-only, GPU-readable
// region whose bytes are the MODEL'S OWN page-cache pages -- not a copy -- because a 96 GiB copy on a 116 GiB
// box swaps (measured: 22 GiB of swap, 0.8 t/s).
//
// This probe builds exactly that and measures it:
//   1. reserve `n_slots * stride` of VA (PROT_NONE, no pages);
//   2. MAP_FIXED each selected expert's *file range* into its slot -- so the pages are the page cache's,
//      shared with the CPU path, and the window is compacted to whatever slot order we choose;
//   3. hipHostRegister(Mapped|Portable|ReadOnly) over the whole window;
//   4. read it with the MoE shape (one block per routed row) and report GB/s;
//   5. report MemAvailable before/after -- a COPY costs `bytes`, this must cost ~0.
//
// Slot order is the whole point of the design: pass a slot->file-offset permutation (default: the hottest
// first, from the persisted MHT2 prior if --prior is given, else 0..n-1) and the window becomes a heat-ranked
// residency with a single base and a fixed stride, which is the shape the cold-read kernels already consume.
//
// Usage:
//   ./winprobe <gguf-shard> <stride-bytes> <n_slots> [n_reads] [prior-file]
//   e.g. ./winprobe shard2 3342336 7700 32 /llm/models/.../Qwen3.5-122B-A10B-Q8_0.moe-heatmap
//        (7700 x 3.19 MiB = 24.0 GiB)
//
// Build: hipcc -O2 --offload-arch=gfx1151 -o winprobe winprobe.cpp
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
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(2); } } while (0)

__global__ void read_rows(const float4 * __restrict__ base, const long * __restrict__ offs,
                          long n4, float * __restrict__ sink) {
    const float4 * p = (const float4 *) ((const char *) base + offs[blockIdx.x]);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (long i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = __ldg(p + i);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x + acc.y + acc.z + acc.w == 1e30f) { sink[blockIdx.x] = acc.x; }
}

static long memavail_mib() {
    FILE * f = fopen("/proc/meminfo", "r");
    char line[256]; long v = -1;
    while (f && fgets(line, sizeof line, f)) {
        if (sscanf(line, "MemAvailable: %ld kB", &v) == 1) { break; }
    }
    if (f) { fclose(f); }
    return v / 1024;
}

// Minimal MHT2 reader: returns the per-expert decode-band counts in file order (table-major, expert-minor).
static bool load_prior(const char * path, std::vector<uint32_t> & out, std::vector<uint64_t> & bytes) {
    FILE * f = fopen(path, "rb");
    if (!f) { return false; }
    char magic[4]; uint32_t ver = 0, n_tab = 0, r0 = 0, n_ten = 0, r1 = 0; uint64_t mbytes = 0;
    if (fread(magic, 1, 4, f) != 4 || fread(&ver, 4, 1, f) != 1 || fread(&n_tab, 4, 1, f) != 1 ||
        fread(&r0, 4, 1, f) != 1 || fread(&mbytes, 8, 1, f) != 1 || fread(&n_ten, 4, 1, f) != 1 ||
        fread(&r1, 4, 1, f) != 1) { fclose(f); return false; }
    if (memcmp(magic, "MHT2", 4) != 0 && memcmp(magic, "MHT1", 4) != 0) { fclose(f); return false; }
    for (uint32_t t = 0; t < n_tab; t++) {
        int32_t layer; char role[64]; uint32_t n_exp; uint64_t host_b;
        if (fread(&layer, 4, 1, f) != 1 || fread(role, 1, 64, f) != 64 ||
            fread(&n_exp, 4, 1, f) != 1 || fread(&host_b, 8, 1, f) != 1) { fclose(f); return false; }
        std::vector<uint32_t> all(n_exp), dec(n_exp, 0);
        if (fread(all.data(), 4, n_exp, f) != n_exp) { fclose(f); return false; }
        if (ver >= 2 && fread(dec.data(), 4, n_exp, f) != n_exp) { fclose(f); return false; }
        for (uint32_t e = 0; e < n_exp; e++) {
            out.push_back(ver >= 2 ? dec[e] : all[e]);
            bytes.push_back(host_b);
        }
    }
    fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <gguf-shard> <stride-bytes> <n_slots> [n_reads] [prior]\n", argv[0]);
        return 2;
    }
    const char * path   = argv[1];
    const long   stride = atol(argv[2]);
    const long   n_slot = atol(argv[3]);
    const int    n_read = argc > 4 ? atoi(argv[4]) : 32;
    const char * prior  = argc > 5 ? argv[5] : nullptr;

    if (stride % 4096 != 0) {
        fprintf(stderr, "stride %ld is not a multiple of the page size -- MAP_FIXED cannot express it\n", stride);
        return 2;
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return 2; }
    const off_t fsz = lseek(fd, 0, SEEK_END);

    // The file region we will scatter into the window: as many experts as the file holds at this stride,
    // starting from a page-aligned offset (a real GGUF tensor starts at a 32-byte-aligned offset; the
    // residue is handled by the caller passing a stride that keeps the congruence).
    const off_t base_off = 4096;                       // page-aligned, inside shard data
    const long  n_file   = (long) ((fsz - base_off) / stride);

    // Slot order: hottest first when a prior is given (that is the design's selection), else file order.
    std::vector<uint32_t> counts; std::vector<uint64_t> pbytes;
    std::vector<long> order((size_t) std::min(n_slot, n_file));
    for (size_t i = 0; i < order.size(); i++) { order[i] = (long) i; }
    bool ranked = false;
    if (prior != nullptr && load_prior(prior, counts, pbytes)) {
        // rank the first n_file file slots by the prior (a shard-local approximation: the same table order)
        std::sort(order.begin(), order.end(), [&](long a, long b) {
            const uint32_t ca = a < (long) counts.size() ? counts[(size_t) a] : 0;
            const uint32_t cb = b < (long) counts.size() ? counts[(size_t) b] : 0;
            return ca != cb ? ca > cb : a < b;
        });
        ranked = true;
    }

    const size_t bytes = (size_t) n_slot * (size_t) stride;
    const long avail0 = memavail_mib();
    void * win = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (win == MAP_FAILED) { perror("reserve"); return 2; }
    long mapped = 0;
    for (long s = 0; s < n_slot; s++) {
        const long fi = order[(size_t) s];
        void * dst = (char *) win + (size_t) s * (size_t) stride;
        void * got = mmap(dst, (size_t) stride, PROT_READ, MAP_SHARED | MAP_FIXED, fd, base_off + fi * stride);
        if (got == MAP_FAILED) { perror("mmap slot"); break; }
        mapped++;
    }
    const long avail1 = memavail_mib();     // mapped-but-not-faulted pages cost nothing yet

    // Touch lightly so the pages are present, then register.  (In the delivery the fill is the read itself;
    // here we only want the registration+read cost, so one byte per page is enough to fault them in.)
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long vol = 0;
    for (long s = 0; s < mapped; s++) {
        volatile char * p = (volatile char *) win + (size_t) s * (size_t) stride;
        for (size_t o = 0; o < (size_t) stride; o += 4096) { vol += p[o]; }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double fault_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    hipError_t re = hipHostRegister(win, (size_t) mapped * (size_t) stride,
                                    hipHostRegisterMapped | hipHostRegisterPortable | hipHostRegisterReadOnly);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const double reg_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    if (re != hipSuccess) {
        fprintf(stderr, "hipHostRegister(ReadOnly) over %ld mapped slots -> %s\n", mapped, hipGetErrorString(re));
        return 2;
    }
    void * dev = nullptr;
    CK(hipHostGetDevicePointer(&dev, win, 0));
    const long avail2 = memavail_mib();

    printf("window: %ld/%ld slots mapped, %.2f GiB, stride %ld, %s\n", mapped, n_slot,
           (double) mapped * stride / (1u << 30), stride, ranked ? "heat-ranked" : "file order");
    printf("  fault-in   %6.2f s (%.1f MiB/s)\n", fault_s, mapped * stride / (1u << 20) / (fault_s > 0 ? fault_s : 1));
    printf("  register   %6.2f s (%.1f MiB/s)   [Mapped|Portable|ReadOnly]\n", reg_s,
           mapped * stride / (1u << 20) / (reg_s > 0 ? reg_s : 1));
    printf("  MemAvailable %ld -> %ld -> %ld MiB (fault -> registered); a COPY of %.2f GiB would have cost that much\n",
           avail0, avail1, avail2, (double) mapped * stride / (1u << 30));
    printf("  device alias %p (window %p)\n", dev, win);

    // ---- the read band, in the MoE shape -----------------------------------------------------------------
    const long n4 = stride / (long) sizeof(float4);
    long * d_offs = nullptr; float * d_sink = nullptr;
    CK(hipMalloc(&d_offs, sizeof(long) * (size_t) n_read));
    CK(hipMalloc(&d_sink, sizeof(float) * (size_t) n_read));
    std::vector<long> offs((size_t) n_read);
    uint32_t seed = 12345;
    auto fill = [&]() {
        for (int k = 0; k < n_read; k++) {
            seed = seed * 1664525u + 1013904223u;
            offs[(size_t) k] = (long) ((seed >> 8) % (uint32_t) mapped) * stride;
        }
    };
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    for (int mode = 0; mode < 2; mode++) {
        std::vector<double> ms;
        for (int it = 0; it < 20; it++) {
            fill();
            CK(hipMemcpy(d_offs, offs.data(), sizeof(long) * (size_t) n_read, hipMemcpyHostToDevice));
            CK(hipEventRecord(e0));
            if (mode == 0) {
                read_rows<<<n_read, 1024>>>((const float4 *) dev, d_offs, n4, d_sink);
            } else {
                for (int l = 0; l + 8 <= n_read; l += 8) {
                    read_rows<<<8, 1024>>>((const float4 *) dev, d_offs + l, n4, d_sink);
                }
            }
            CK(hipEventRecord(e1));
            CK(hipEventSynchronize(e1));
            float t = 0; CK(hipEventElapsedTime(&t, e0, e1));
            ms.push_back((double) t);
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        const double gbs = (double) stride * n_read / (med / 1000.0) / 1e9;
        const int rows = mode == 0 ? n_read : (n_read / 8) * 8;
        printf("  read %-6s %4d rows/launch, 1024 thr: %7.3f ms -> %6.1f GB/s\n",
               mode == 0 ? "wide" : "layer", rows, med, gbs);
    }
    return 0;
}
