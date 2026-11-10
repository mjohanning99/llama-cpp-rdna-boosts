// p0/pageref.cpp -- can the GPU read the model mapping's pages IN PLACE, with NO registration?
//
// This is the question a PARTIAL window lives or dies on.  A window that covers only the hot experts of each
// table leaves the complement to be read somewhere, and every "somewhere" except the pageable model mapping costs
// a copy of the same bytes (the arena) -- which on a UMA box is 26.8 GiB the budget does not have.  If the GPU can
// dereference a plain `MAP_SHARED|PROT_READ` model mapping, the complement costs nothing and is shared with the
// CPU, which is the design's own premise.
//
// `p0/regprobe.cpp` already shows a 4.7 MiB unregistered file range reads correctly at full band.  It does NOT
// answer the only question that matters: what happens when a page of that range is **NOT resident** at the moment
// the kernel touches it.  A GPU fault is a queue hang, not a retry, so:
//
//   arm 1  resident    the whole range read warm (the control)
//   arm 2  reclaimed   `POSIX_FADV_DONTNEED` the whole range, then read it again -- the whole design depends on
//                      this arm, and a fault here is the answer "no"
//   arm 3  sampled     after the reclaim, compare a few KiB against `pread` on the HOST side (ground truth)
//
// Anything that hangs the queue kills THIS process only (run it in its own session).  Correct bytes on arm 2 mean
// the driver faults/pins on demand -- which also makes the *rate* matter, so arm 2 is timed.
//
// Usage: ./pageref <gguf> <offset> <GiB>
// Build: hipcc -O2 --offload-arch=gfx1151 -o pageref pageref.cpp
#include <hip/hip_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(2); } } while (0)

__global__ void copy_out(const char * __restrict__ src, char * __restrict__ dst, long n) {
    for (long i = (long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long) gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

static long memavail_mib() {
    FILE * f = fopen("/proc/meminfo", "r");
    char line[256]; long v = -1;
    while (f && fgets(line, sizeof line, f)) { if (sscanf(line, "MemAvailable: %ld kB", &v) == 1) { break; } }
    if (f) { fclose(f); }
    return v / 1024;
}

// Read `bytes` of `base` with the GPU and report wall time + GB/s.  Returns false if the read failed.
static bool gpu_read_all(const char * arm, const char * base, size_t bytes, char * dev, char * host_copy) {
    const long chunk = 64L << 20;                       // 64 MiB per pass, so a partial failure is locatable
    const long nchunk = (long) ((bytes + (size_t) chunk - 1) / (size_t) chunk);
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0));
    printf("%-10s src %p dst %p, %ld chunk(s) of %ld MiB\n", arm, (const void *) base, (void *) dev, nchunk, chunk >> 20);
    for (long c = 0; c < nchunk; c++) {
        const size_t off = (size_t) c * (size_t) chunk;
        const long   n   = (long) ((bytes - off) < (size_t) chunk ? (bytes - off) : (size_t) chunk);
        copy_out<<<256, 256>>>((const char *) base + off, dev, n);
        const hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) {
            printf("%-10s chunk %ld of %ld FAILED: %s\n", arm, c, nchunk, hipGetErrorString(e));
            (void) hipGetLastError();
            return false;
        }
    }
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms = 0.f;
    CK(hipEventElapsedTime(&ms, e0, e1));
    printf("%-10s read %.2f GiB in %8.3f s -> %6.1f GB/s\n", arm, (double) bytes / (1024.0 * 1024.0 * 1024.0),
           ms / 1000.0, ((double) bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0));
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    (void) host_copy;
    return true;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 4) {
        fprintf(stderr, "usage: %s <gguf> <offset> <GiB>\n", argv[0]);
        return 2;
    }
    const char * path  = argv[1];
    const size_t alo   = (size_t) strtoull(argv[2], nullptr, 10) & ~(size_t) 4095;
    const size_t bytes = (size_t) strtoull(argv[3], nullptr, 10) << 30;
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return 2; }

    printf("range: file %s offset %zu, %.2f GiB, NOT registered\n", path, alo,
           (double) bytes / (1024.0 * 1024.0 * 1024.0));
    printf("MemAvailable before mapping: %ld MiB\n", memavail_mib());
    char * m = (char *) mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, (off_t) alo);
    if (m == MAP_FAILED) { perror("mmap"); return 2; }

    char * dev = nullptr;
    CK(hipMalloc((void **) &dev, 64 << 20));   // one chunk's worth: the read loop copies chunk-by-chunk
    std::vector<char> host_copy(1 << 20);

    // ---- arm 1: resident (the control; this is what regprobe already showed at 4.7 MiB)
    // Touch it from the CPU first so the pages are known-resident and pinned by nothing.
    {   volatile char acc = 0;
        for (size_t i = 0; i < bytes; i += 4096) { acc = (char) (acc ^ m[i]); }
        printf("resident: faulted in by the CPU, MemAvailable now %ld MiB\n", memavail_mib());
    }
    if (!gpu_read_all("resident", m, bytes, dev, host_copy.data())) { printf("RESULT: resident arm failed\n"); return 1; }

    // ---- arm 2: reclaim the WHOLE range, then read it with the GPU again.
    const int fa = posix_fadvise(fd, (off_t) alo, (off_t) bytes, POSIX_FADV_DONTNEED);
    printf("reclaimed: posix_fadvise(DONTNEED) -> %d (%s), MemAvailable %ld MiB\n",
           fa, fa == 0 ? "ok" : strerror(fa), memavail_mib());
    if (!gpu_read_all("reclaimed", m, bytes, dev, host_copy.data())) { printf("RESULT: reclaimed arm FAILED -> "
                                                                             "the pages must be resident\n"); return 1; }

    // ---- arm 3: ground truth on the HOST for four sampled spots (a wrong-but-fast read is the other failure mode)
    {
        int bad = 0;
        for (int k = 0; k < 4; k++) {
            const size_t off = (bytes / 5) * (size_t) k;
            std::vector<char> want(4096), got(4096);
            if (pread(fd, want.data(), want.size(), (off_t) (alo + off)) != 4096) { printf("pread failed\n"); return 2; }
            copy_out<<<4, 256>>>(m + off, dev, 4096);
            if (hipDeviceSynchronize() != hipSuccess) { printf("sampled read %d failed\n", k); (void) hipGetLastError(); return 1; }
            CK(hipMemcpy(got.data(), dev, 4096, hipMemcpyDeviceToHost));
            if (memcmp(want.data(), got.data(), 4096) != 0) { printf("sampled spot %d: MISMATCH\n", k); bad++; }
        }
        printf("sampled: %d of 4 spots mismatch\n", bad);
    }
    printf("RESULT: the GPU read a %.2f GiB pageable file mapping CORRECTLY both resident and after DONTNEED\n",
           (double) bytes / (1024.0 * 1024.0 * 1024.0));
    (void) munmap(m, bytes);
    return 0;
}
