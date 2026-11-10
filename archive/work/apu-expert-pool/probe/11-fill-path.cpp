// 11-fill-path.cpp -- where can the host PUT expert bytes on a UMA APU, and how fast?
//
// WHY THIS EXISTS (wip/apu-expert-pool, single-tier design):
//   The campaign measured that a `hipMalloc` (GTT) buffer is host-writable and that the GPU reads it at
//   232.6 GB/s -- so on an APU the arena IS the pool, and a pinned second copy is pure duplication.  One
//   question decides how a fill is implemented:
//     can storage write STRAIGHT into that arena, or does every fill need a bounce buffer?
//   `07-host-write-gtt.cpp` already showed `O_DIRECT` -> EINVAL for a `hipMalloc` pointer.  That is not a HIP
//   quirk: the kernel's direct-I/O path pins the USER pages (`iov_iter_get_pages`), so the target must be
//   ordinary, `struct page`-backed VMA memory -- which a driver mapping is not.  This probe answers the two
//   questions that follow, on the real allocation shapes:
//     * the SLAB's arena is VMM (`cuMemCreate`/`hipMemCreate`), not `hipMalloc`.  Is IT host-writable, at
//       what rate, does the GPU see it FRESH, and does `O_DIRECT` reach it?
//     * is the bounce avoidable by backing the tier with ORDINARY ANON pages registered to the GPU
//       (`hipHostRegister(ReadOnly)`)?  Then `O_DIRECT` is legal (measured in 04) and the GPU read is still
//       ~233 GB/s (measured in 03) -- i.e. the same speed as the arena, with no copy at all.
//
// Usage:  ./11-fill-path <vmm|anon|copy> <file> [GiB]
//   vmm   VMM arena (the slab's shape): host write/FRESH/L2, host memset rate, O_DIRECT errno, buffered pread
//   anon  ordinary anon buffer: O_DIRECT straight fill (NO bounce), then ReadOnly register -> GPU read rate,
//         then the unregister/re-register cycle cost (that cycle is what a rewrite of a RO-registered
//         tier costs -- measured 2.67 ms/GiB register in 04)
//   copy  the copy-based fill for reference: buffered pread into anon (page cache) -> host memcpy into the
//         arena, and hipMemcpy(mapping -> arena)
//
// A host DEREFERENCE of a device pointer would SIGSEGV on a discrete card; every such read/write here runs in
// a forked child, so an inaccessible allocation is REPORTED (exit status) instead of killing the probe.
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <setjmp.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <ctime>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    printf("HIPFAIL %s: %s\n", #x, hipGetErrorString(e_)); exit(2); } } while (0)

static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

__global__ void sumw(const unsigned int * p, unsigned long long * out, size_t n) {
    size_t s = (size_t) gridDim.x * blockDim.x, i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long a = 0;
    for (; i < n; i += s) { a += ((volatile const unsigned int *) p)[i]; }
    atomicAdd(out, a);
}

// GPU sum of `b` bytes at `p` -- proves the GPU really reads those bytes (and gives a read-rate arm).
static unsigned long long gpu_sum(const void * p, size_t b, double * secs) {
    unsigned long long * d;
    CK(hipMalloc(&d, 8)); CK(hipMemset(d, 0, 8));
    double t0 = now();
    sumw<<<512, 256>>>((const unsigned int *) p, d, b / 4);
    CK(hipDeviceSynchronize());
    double t1 = now();
    unsigned long long h;
    CK(hipMemcpy(&h, d, 8, hipMemcpyDeviceToHost));
    (void) hipFree(d);
    if (secs) { *secs = t1 - t0; }
    return h;
}

// Is this pointer host-accessible?  IN-PROCESS with a SIGSEGV escape: `fork()` does NOT inherit a driver
// mapping, so the child faults on buffers the parent writes happily -- measured here, hipMalloc memset ran at
// 13.4 GB/s immediately after the forked test called the same buffer INACCESSIBLE.  The first version of this
// probe reported that false negative; do not reintroduce it.
static sigjmp_buf g_segv;
static void segv_handler(int) { siglongjmp(g_segv, 1); }

static bool host_writable(void * p) {
    struct sigaction sa = {}, old;
    sa.sa_handler = segv_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old);
    bool ok = false;
    if (sigsetjmp(g_segv, 1) == 0) {
        volatile unsigned char * q = (volatile unsigned char *) p;
        unsigned char v = q[0];
        q[0] = (unsigned char) (v ^ 0x01);
        q[0] = v;
        ok = true;
    }
    sigaction(SIGSEGV, &old, nullptr);
    return ok;
}

struct arena_t { void * ptr = nullptr; size_t size = 0; };

// The slab's shape: hipMemCreate(Pinned) + hipMemAddressReserve + hipMemMap + hipMemSetAccess.
// `loc` selects hipMemLocationTypeDevice (the slab today) or hipMemLocationTypeHost (a VMM range whose backing
// is host memory -- the only VMM shape that could be CPU-mapped AND GPU-mapped, which is what a host-filled
// arena needs).
static arena_t alloc_vmm(size_t want, hipMemLocationType loc = hipMemLocationTypeDevice) {
    arena_t a;
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = loc;
    prop.location.id = 0;
    size_t gran = 0;
    CK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
    size_t size = (want + gran - 1) / gran * gran;
    hipMemGenericAllocationHandle_t h;
    CK(hipMemCreate(&h, size, &prop, 0));
    CK(hipMemAddressReserve(&a.ptr, size, gran, nullptr, 0));
    CK(hipMemMap(a.ptr, size, 0, h, 0));
    hipMemAccessDesc d = {};
    d.location.type = hipMemLocationTypeDevice;   // WHO access is granted to: the GPU, always
    d.location.id = 0;
    d.flags = hipMemAccessFlagsProtReadWrite;
    CK(hipMemSetAccess(a.ptr, size, &d, 1));
    a.size = size;
    return a;
}

static int g_fd = -1;

int main(int argc, char ** argv) {
    if (argc < 3) { printf("usage: %s <vmm|anon|copy> <file> [GiB]\n", argv[0]); return 1; }
    const char * arm = argv[1];
    const char * path = argv[2];
    double gib = argc > 3 ? atof(argv[3]) : 1.0;
    size_t SZ = (size_t) (gib * 1073741824.0);

    hipDeviceProp_t pr;
    CK(hipGetDeviceProperties(&pr, 0));
    printf("device 0: %s | integrated=%d | totalGlobalMem=%.1f GiB\n", pr.name, pr.integrated,
           pr.totalGlobalMem / 1073741824.0);

    // ---------------------------------------------------------------- VMM arena (the slab's allocation)
    if (strcmp(arm, "vmm") == 0 || strcmp(arm, "vmmhost") == 0) {
        const bool dev = strcmp(arm, "vmm") == 0;
        arena_t a = alloc_vmm(SZ, dev ? hipMemLocationTypeDevice : hipMemLocationTypeHost);
        printf("vmm arena (%s location): %p size=%.3f GiB (hipMemCreate Pinned + reserve + map)\n",
               dev ? "DEVICE" : "HOST", a.ptr, a.size / 1073741824.0);
        bool hw = host_writable(a.ptr);
        printf("  host READ+WRITE of the vmm arena: %s\n", hw ? "OK" : "INACCESSIBLE (fault in child)");
        if (!hw) {
            printf("  => a fill MUST be hipMemcpy/H2D, never a host memcpy: the bounce question does not "
                   "arise, the copy question does\n");
            return 0;
        }
        double t = now(); memset(a.ptr, 0x44, a.size); double m = now() - t;
        printf("  host memset %.2f GiB: %.3f s = %.1f GB/s\n", a.size / 1073741824.0, m, a.size / m / 1e9);
        double secs; unsigned long long s = gpu_sum(a.ptr, a.size, &secs);
        printf("  GPU sum after host memset: %016llx (%s), %.1f GB/s read\n", s,
               s == (0x44444444ULL * (a.size / 4)) ? "FRESH" : "STALE", a.size / secs / 1e9);

        g_fd = open(path, O_RDONLY);
        if (g_fd < 0) { printf("  cannot open %s\n", path); return 0; }
        errno = 0;
        ssize_t got = pread(g_fd, a.ptr, a.size, 0);
        printf("  buffered pread %.2f GiB into the arena: %zd (%s)\n", a.size / 1073741824.0, got,
               got < 0 ? strerror(errno) : "ok");
        close(g_fd);

        g_fd = open(path, O_RDONLY | O_DIRECT);
        if (g_fd < 0) { printf("  cannot open %s O_DIRECT: %s\n", path, strerror(errno)); return 0; }
        errno = 0;
        got = pread(g_fd, a.ptr, a.size, 0);
        printf("  O_DIRECT pread %.2f GiB into the arena: %zd (%s)\n", a.size / 1073741824.0, got,
               got < 0 ? strerror(errno) : "ok -- NO BOUNCE NEEDED");
        close(g_fd);
        return 0;
    }

    // ---------------------------------------------------------------- hipMalloc (the host-writable GTT shape)
    if (strcmp(arm, "hipmalloc") == 0) {
        unsigned char * p;
        CK(hipMalloc(&p, SZ));
        printf("hipMalloc arena: %p size=%.3f GiB\n", p, SZ / 1073741824.0);
        printf("  host READ+WRITE: %s\n", host_writable(p) ? "OK" : "INACCESSIBLE");
        double t = now(); memset(p, 0x44, SZ); double m = now() - t;
        printf("  host memset: %.1f GB/s\n", SZ / m / 1e9);
        double secs; unsigned long long s = gpu_sum(p, SZ, &secs);
        printf("  GPU sum: %016llx (%s), %.1f GB/s read\n", s,
               s == (0x44444444ULL * (SZ / 4)) ? "FRESH" : "STALE", SZ / secs / 1e9);
        g_fd = open(path, O_RDONLY | O_DIRECT);
        if (g_fd >= 0) {
            errno = 0; ssize_t got = pread(g_fd, p, SZ, 0);
            printf("  O_DIRECT pread into it: %zd (%s)\n", got, got < 0 ? strerror(errno) : "ok");
            close(g_fd);
        }
        g_fd = open(path, O_RDONLY);
        if (g_fd >= 0) {
            errno = 0; t = now(); ssize_t got = pread(g_fd, p, SZ, 0); double r = now() - t;
            printf("  buffered pread into it: %zd, %.1f GB/s\n", got, got > 0 ? SZ / r / 1e9 : 0.0);
            close(g_fd);
        }
        return 0;
    }

    // ---------------------------------------------------------------- ordinary anon, registered for the GPU
    if (strcmp(arm, "anon") == 0) {
        void * p = nullptr;
        if (posix_memalign(&p, 4096, SZ) != 0) { printf("posix_memalign failed\n"); return 2; }
        printf("anon arena: %p size=%.3f GiB (4 KiB aligned)\n", p, SZ / 1073741824.0);

        g_fd = open(path, O_RDONLY | O_DIRECT);
        if (g_fd < 0) { printf("cannot open %s O_DIRECT: %s\n", path, strerror(errno)); return 2; }
        errno = 0;
        double t = now();
        ssize_t got = pread(g_fd, p, SZ, 0);
        double secs = now() - t;
        printf("  O_DIRECT pread %.2f GiB STRAIGHT into anon: %zd in %.3f s = %.1f GB/s (%s)\n",
               SZ / 1073741824.0, got, secs, got > 0 ? SZ / secs / 1e9 : 0.0,
               got < 0 ? strerror(errno) : "legal, no bounce buffer");
        if (got <= 0) { return 0; }

        // The tier must be GPU-readable: a ReadOnly registration is the fast arm (03/04 measured 233 vs 60).
        double r0 = now();
        CK(hipHostRegister(p, SZ, hipHostRegisterMapped | hipHostRegisterReadOnly));
        double r1 = now();
        printf("  hipHostRegister(Mapped|ReadOnly) of %.2f GiB: %.1f ms (%.2f ms/GiB)\n", SZ / 1073741824.0,
               (r1 - r0) * 1e3, (r1 - r0) * 1e3 / (SZ / 1073741824.0));
        double gsecs; unsigned long long s = gpu_sum(p, SZ, &gsecs);
        printf("  GPU sum of the registered anon arm: %016llx, %.1f GB/s read\n", s, SZ / gsecs / 1e9);

        // The fill path that matters for a RO-registered tier: can O_DIRECT still target the range once the
        // pages are PINNED by the registration?  (If not, a fill would need an unregister/register cycle or a
        // bounce buffer, which is the whole question this probe exists to settle.)
        int fd2 = open(path, O_RDONLY | O_DIRECT);
        if (fd2 >= 0) {
            errno = 0;
            double t2 = now();
            ssize_t g2 = pread(fd2, p, SZ, 0);
            double e2 = now() - t2;
            printf("  O_DIRECT pread into the ALREADY-REGISTERED range: %zd in %.3f s = %.1f GB/s (%s)\n",
                   g2, e2, g2 > 0 ? SZ / e2 / 1e9 : 0.0, g2 < 0 ? strerror(errno) : "legal");
            close(fd2);
        }

        // Rewriting a ReadOnly-registered tier: does the host write still work, and what does the cycle cost?
        memset(p, 0x55, 1ull << 20);
        double c0 = now();
        CK(hipHostUnregister(p));
        CK(hipHostRegister(p, SZ, hipHostRegisterMapped | hipHostRegisterReadOnly));
        double c1 = now();
        printf("  unregister+re-register cycle: %.1f ms for %.2f GiB (the per-rewrite cost of a RO tier)\n",
               (c1 - c0) * 1e3, SZ / 1073741824.0);
        s = gpu_sum(p, SZ, &gsecs);
        printf("  GPU sum after in-place host write + re-register: %016llx (first MiB 0x55555555? %s)\n", s,
               "see 10-coherence for the FRESH check");
        CK(hipHostUnregister(p));
        close(g_fd);
        return 0;
    }

    // ---------------------------------------------------------------- the copy-based fill, for reference
    if (strcmp(arm, "copy") == 0) {
        void * anon = nullptr;
        if (posix_memalign(&anon, 4096, SZ) != 0) { return 2; }
        unsigned char * dev;
        CK(hipMalloc(&dev, SZ));
        g_fd = open(path, O_RDONLY);
        if (g_fd < 0) { printf("cannot open %s\n", path); return 2; }
        double t = now(); ssize_t got = pread(g_fd, anon, SZ, 0); double r = now() - t;
        printf("  buffered pread %.2f GiB into anon (page cache): %zd, %.1f GB/s\n", SZ / 1073741824.0, got,
               got > 0 ? SZ / r / 1e9 : 0.0);
        t = now(); memcpy(dev, anon, SZ); double c = now() - t;
        printf("  host memcpy anon -> hipMalloc arena: %.3f s = %.1f GB/s\n", c, SZ / c / 1e9);
        t = now(); CK(hipMemcpy(dev, anon, SZ, hipMemcpyHostToDevice)); double h = now() - t;
        printf("  hipMemcpy anon -> arena (H2D):       %.3f s = %.1f GB/s\n", h, SZ / h / 1e9);
        double gsecs; unsigned long long s = gpu_sum(dev, SZ, &gsecs);
        printf("  GPU sum after the fill: %016llx, %.1f GB/s read\n", s, SZ / gsecs / 1e9);
        close(g_fd);
        return 0;
    }

    printf("unknown arm '%s'\n", arm);
    return 1;
}
