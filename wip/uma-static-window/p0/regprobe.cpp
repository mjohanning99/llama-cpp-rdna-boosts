// p0/regprobe.cpp -- does a hipHostRegister'd range actually hold its bytes FOR THE DEVICE?
//
// `p0/band.cpp` and `p0/winprobe.cpp` measured the window's BAND only: a registration that makes the device read
// ZEROS is just as fast as one that makes it read the expert's bytes, so "the transport is free" was measured
// content-blind.  This probe adds the missing half: read the registered range with a KERNEL (a `cudaMemcpy` out
// of a host pointer is short-circuited into a CPU memcpy under UVA and proves nothing) and compare it with the
// bytes the CPU sees at the same address -- and with the file itself.
//
// Arms (each read through a kernel into a device buffer, then memcpy'd back):
//   anon-rw   anonymous MAP_PRIVATE, filled with a pattern, hipHostRegister(writable)
//   anon-ro   the same range, hipHostRegister(ReadOnly)
//   file-rw   the GGUF's own pages (MAP_SHARED|PROT_READ), hipHostRegister(writable)
//   file-ro   the same, hipHostRegister(ReadOnly)    <- what the P2 window mode uses
//
// Usage: ./regprobe <gguf> <file-offset> <bytes>
// Build: hipcc -O2 --offload-arch=gfx1151 -o regprobe regprobe.cpp
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

static void report(const char * arm, const char * want_desc,
                   const char * want, const char * got_host_view, const char * got_dev, long n) {
    // `got_host_view` = the same VA read by the CPU; `got_dev` = what the GPU reads through the registration.
    long d_host = -1, d_want = -1;
    for (long i = 0; i < n; i++) {
        if (d_host < 0 && got_host_view[i] != want[i]) { d_host = i; }
        if (d_want < 0 && got_dev[i]      != want[i]) { d_want = i; }
    }
    long zeros = 0;
    for (long i = 0; i < n; i++) { zeros += (got_dev[i] == 0); }
    printf("%-8s vs %-22s cpu-view %s (first diff %ld)   device %s (first diff %ld, %ld/%ld zero bytes)\n",
           arm, want_desc,
           d_host < 0 ? "MATCH" : "MISMATCH", d_host,
           d_want < 0 ? "MATCH" : "MISMATCH", d_want, zeros, n);
    if (d_want >= 0) {
        printf("         device[0..16] = ");
        for (int i = 0; i < 16; i++) { printf("%02x", (unsigned char) got_dev[i]); }
        printf("   want[0..16] = ");
        for (int i = 0; i < 16; i++) { printf("%02x", (unsigned char) want[i]); }
        printf("\n");
    }
}

// The same device read, with NO registration: the pointer the kernel gets is a plain host VA.
static void trial_raw(const char * arm, void * base, size_t bytes, const char * want, const char * want_desc,
                      char * devbuf_dev, char * devbuf_host) {
    CK(hipMemset(devbuf_dev, 0xAA, bytes));
    copy_out<<<64, 256>>>((const char *) base, devbuf_dev, (long) bytes);
    const hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("%-10s UNREGISTERED device read FAILED: %s\n", arm, hipGetErrorString(e));
        (void) hipGetLastError();
        return;
    }
    CK(hipMemcpy(devbuf_host, devbuf_dev, bytes, hipMemcpyDeviceToHost));
    report(arm, want_desc, want, (const char *) base, devbuf_host, (long) bytes);
}

static void trial(const char * arm, void * base, size_t bytes, unsigned flags, const char * want,
                  const char * want_desc, char * devbuf_dev, char * devbuf_host) {
    if (hipHostRegister(base, bytes, hipHostRegisterMapped | hipHostRegisterPortable | flags) != hipSuccess) {
        printf("%-8s hipHostRegister failed: %s\n", arm, hipGetErrorString(hipGetLastError()));
        return;
    }
    void * dev = nullptr;
    if (hipHostGetDevicePointer(&dev, base, 0) != hipSuccess || dev == nullptr) {
        printf("%-8s hipHostGetDevicePointer failed\n", arm);
        (void) hipHostUnregister(base);
        return;
    }
    printf("%-8s registered: host %p dev %p (same=%d)\n", arm, base, dev, dev == base);
    CK(hipMemset(devbuf_dev, 0xAA, bytes));
    copy_out<<<64, 256>>>((const char *) dev, devbuf_dev, (long) bytes);
    const hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("%-8s device read FAILED: %s\n", arm, hipGetErrorString(e));
        (void) hipGetLastError();
        (void) hipHostUnregister(base);
        return;
    }
    CK(hipMemcpy(devbuf_host, devbuf_dev, bytes, hipMemcpyDeviceToHost));
    report(arm, want_desc, want, (const char *) base, devbuf_host, (long) bytes);
    (void) hipHostUnregister(base);
}


// The multi-slot layout the P2 window actually builds: a PROT_NONE reserve, one MAP_FIXED per slot from a
// SCATTERED file offset (slot stride == expert stride), then ONE hipHostRegister over the whole reserve.  This
// is the arm that isolates the pool's device-zero read from the single-range arm above.
static void window_trial(const char * path, size_t foff, size_t stride, int n_slots, const int * order,
                         char * devbuf_dev, char * devbuf_host) {
    const size_t page = 4096;
    const size_t resid = foff % page;
    const size_t alo0  = foff - resid;
    const size_t len   = (stride + resid + page - 1) & ~(page - 1);
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return; }
    char * base = (char *) mmap(nullptr, (size_t) n_slots * stride + page, PROT_NONE,
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) { perror("reserve"); return; }
    for (int s = 0; s < n_slots; s++) {
        const size_t alo = alo0 + (size_t) order[s] * stride;
        if (mmap(base + (size_t) s * stride, len, PROT_READ, MAP_SHARED | MAP_FIXED, fd, (off_t) alo) == MAP_FAILED) {
            perror("MAP_FIXED"); return;
        }
    }
    printf("window   %d slots of %zu B (stride %zu, residue %zu), %.1f MiB registered as one range\n",
           n_slots, stride, stride, resid, (double) ((size_t) n_slots * stride) / (1024.0 * 1024.0));
    if (hipHostRegister(base, (size_t) n_slots * stride + page,
                        hipHostRegisterMapped | hipHostRegisterPortable | hipHostRegisterReadOnly) != hipSuccess) {
        printf("window   hipHostRegister failed: %s\n", hipGetErrorString(hipGetLastError()));
        return;
    }
    void * dev = nullptr;
    (void) hipHostGetDevicePointer(&dev, base, 0);
    const char * dbase = (const char *) (dev != nullptr ? dev : base);
    // Slot 0's content, and slot (n_slots-1)'s, each against the file itself (pread, not the mapping).
    for (int s : { 0, n_slots - 1 }) {
        std::vector<char> want(stride);
        if (pread(fd, want.data(), stride, (off_t) (alo0 + (size_t) order[s] * stride + resid)) != (ssize_t) stride) {
            perror("pread"); return;
        }
        CK(hipMemset(devbuf_dev, 0xAA, stride));
        copy_out<<<64, 256>>>(dbase + resid + (size_t) s * stride, devbuf_dev, (long) stride);
        if (hipDeviceSynchronize() != hipSuccess) {
            printf("slot %-4d device read FAILED: %s\n", s, hipGetErrorString(hipGetLastError()));
            continue;
        }
        CK(hipMemcpy(devbuf_host, devbuf_dev, stride, hipMemcpyDeviceToHost));
        char label[32]; snprintf(label, sizeof label, "win%d", s);
        report(label, "the file (pread)", want.data(), base + resid + (size_t) s * stride, devbuf_host, (long) stride);
    }
    (void) hipHostUnregister(base);
    munmap(base, (size_t) n_slots * stride + page);
    close(fd);
}

// Band of the same range through each mapping kind, at the MoE launch shape: `n_rows` blocks of 1024 threads,
// each row `row_bytes` long, `iters` repetitions.  A read of a pointer the GPU cannot translate shows up here as
// either an error or a number far below DRAM.
__global__ void read_rows_bw(const char * __restrict__ base, long row_bytes, long n_rows, int iters,
                             float * __restrict__ sink) {
    const char * p = base + (long) blockIdx.x * row_bytes;
    float acc = 0.f;
    for (int it = 0; it < iters; it++) {
        for (long i = (long) threadIdx.x * 4; i + 3 < row_bytes; i += (long) blockDim.x * 4) {
            const float4 v = *(const float4 *) (p + i);
            acc += v.x + v.y + v.z + v.w;
        }
    }
    if (acc == 1e30f) { sink[blockIdx.x] = acc; }
}

static void band(const char * arm, const void * base, long row_bytes, long n_rows, int iters,
                 float * sink_dev);

// Register a range read-only, then band it, then unregister.
static void trial_band_reg(const char * arm, void * base, long row, float * sink) {
    const size_t bytes = (size_t) row * 8;
    if (hipHostRegister(base, bytes, hipHostRegisterMapped | hipHostRegisterPortable | hipHostRegisterReadOnly) != hipSuccess) {
        printf("%-10s hipHostRegister failed\n", arm);
        return;
    }
    void * dev = nullptr;
    (void) hipHostGetDevicePointer(&dev, base, 0);
    band(arm, dev != nullptr ? dev : base, row, 8, 32, sink);
    (void) hipHostUnregister(base);
}

static void band(const char * arm, const void * base, long row_bytes, long n_rows, int iters,
                 float * sink_dev) {
    // warm: translate the pages once (an unregistered read may fault them in)
    read_rows_bw<<<(unsigned) n_rows, 1024>>>(  (const char *) base, row_bytes, n_rows, 1, sink_dev);
    if (hipDeviceSynchronize() != hipSuccess) {
        printf("%-10s band: device read FAILED: %s\n", arm, hipGetErrorString(hipGetLastError()));
        (void) hipGetLastError();
        return;
    }
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0));
    read_rows_bw<<<(unsigned) n_rows, 1024>>>(  (const char *) base, row_bytes, n_rows, iters, sink_dev);
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms = 0.f;
    CK(hipEventElapsedTime(&ms, e0, e1));
    const double gib = (double) row_bytes * (double) n_rows * (double) iters / (1024.0 * 1024.0 * 1024.0);
    printf("%-10s band %2ld rows x 1024 thr: %8.2f MiB in %7.3f ms -> %6.1f GB/s\n",
           arm, n_rows, gib * 1024.0, ms, gib / (ms / 1000.0));
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // an arm can take the GPU down: never lose a line to buffering
    if (argc < 4) {
        fprintf(stderr, "usage: %s <gguf> <file-offset> <bytes>\n", argv[0]);
        return 2;
    }
    const char * path   = argv[1];
    const size_t foff   = (size_t) strtoull(argv[2], nullptr, 10);
    const size_t bytes  = (size_t) strtoull(argv[3], nullptr, 10);
    const size_t page   = 4096;
    const size_t alo    = foff & ~(page - 1);
    const size_t shift  = foff - alo;
    const size_t maplen = (bytes + shift + page - 1) & ~(page - 1);

    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    printf("device: %s (UVA=%d, unmapped-host-access=%d)\n", prop.name, prop.unifiedAddressing,
           prop.canMapHostMemory);

    char * devbuf_dev = nullptr;
    char * devbuf_host = (char *) malloc(bytes);
    CK(hipMalloc(&devbuf_dev, bytes));

    // 1. The anonymous control: a range we filled ourselves.
    char * anon = (char *) mmap(nullptr, maplen, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (anon == MAP_FAILED) { perror("mmap anon"); return 2; }
    for (size_t i = 0; i < maplen; i++) { anon[i] = (char) (i * 31 + 7); }
    std::vector<char> anon_want(bytes);
    memcpy(anon_want.data(), anon + shift, bytes);
    trial("anon-rw", anon + shift, bytes, 0, anon_want.data(), "the pattern we wrote", devbuf_dev, devbuf_host);
    trial("anon-ro", anon + shift, bytes, hipHostRegisterReadOnly, anon_want.data(), "the pattern we wrote",
          devbuf_dev, devbuf_host);

    // 2. The GGUF's own pages -- exactly what the P2 window registers.
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return 2; }
    char * fm = (char *) mmap(nullptr, maplen, PROT_READ, MAP_SHARED, fd, (off_t) alo);
    if (fm == MAP_FAILED) { perror("mmap file"); return 2; }
    // The independent reference: the same bytes read from the FILE, not from the mapping.
    std::vector<char> file_want(bytes);
    if (pread(fd, file_want.data(), bytes, (off_t) foff) != (ssize_t) bytes) { perror("pread"); return 2; }
    long fm_diff = -1;
    for (size_t i = 0; i < bytes; i++) { if (fm[i + shift] != file_want[i]) { fm_diff = (long) i; break; } }
    printf("file mapping vs pread: %s\n", fm_diff < 0 ? "MATCH" : "MISMATCH");
    trial("file-rw", fm + shift, bytes, 0, file_want.data(), "the file (pread)", devbuf_dev, devbuf_host);
    trial("file-ro", fm + shift, bytes, hipHostRegisterReadOnly, file_want.data(), "the file (pread)",
          devbuf_dev, devbuf_host);

    // 2b. NO REGISTRATION AT ALL: can the GPU read pageable host pages on this box (HMM / ATS)?  If it can, the
    //     whole "register the pages ReadOnly" step -- and its pinning cost -- is unnecessary, and a PARTIAL window
    //     becomes expressible (the complement is the same mapping the CPU reads).  If it cannot, the read faults
    //     or returns zeros, which is what makes registration mandatory.
    {
        trial_raw("file-noreg", fm + shift, bytes, file_want.data(), "the file (pread)", devbuf_dev, devbuf_host);
        trial_raw("anon-noreg", anon + shift, bytes, anon_want.data(), "the pattern we wrote", devbuf_dev, devbuf_host);
    }

    // 3. A writable anonymous window whose HOST pages we then fill from the file, registered RO before the fill.
    //    This is the shape a "fill-then-register is impossible, so register first" window would need.
    char * anon2 = (char *) mmap(nullptr, maplen, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (anon2 != MAP_FAILED) {
        memcpy(anon2 + shift, file_want.data(), bytes);
        trial("anonfile", anon2 + shift, bytes, hipHostRegisterReadOnly, file_want.data(), "the file (pread)",
              devbuf_dev, devbuf_host);
    }
    // 3b. BAND through each mapping kind on the SAME bytes, at the per-layer launch shape.  Its own mapping,
    //     sized for the widest arm: reading past the end of a mapping is a real GPU fault (that is how this
    //     probe first died), so the arm's extent and the mapping's extent must be the same thing.
    {
        const long row = 589824;   // one 35B expert as the "row": 8 rows = one layer's 8 routed experts
        const size_t band_map_len = (size_t) row * 32;
        char * fm_band = (char *) mmap(nullptr, band_map_len + page, PROT_READ, MAP_SHARED, fd, (off_t) alo);
        if (fm_band == MAP_FAILED) { perror("mmap band"); return 2; }
        // The anonymous band arm needs its own full-size range too (reading past a mapping is a GPU fault, and
        // reading past a 0.6 MiB one is exactly how this probe first died).
        char * anon_band = (char *) mmap(nullptr, band_map_len + page, PROT_READ | PROT_WRITE,
                                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (anon_band != MAP_FAILED) {
            memcpy(anon_band, fm_band, band_map_len);
        }
        float * sink = nullptr;
        CK(hipMalloc(&sink, 1024 * sizeof(float)));
        char * halloc = nullptr;
        CK(hipMalloc((void **) &halloc, row * 8));
        CK(hipMemcpy(halloc, fm_band + shift, (size_t) row * 8, hipMemcpyHostToDevice));
        band("hipMalloc", halloc, row, 8, 32, sink);
        trial_band_reg("file-ro", fm_band + shift, row, sink);
        band("file-noreg", fm_band + shift, row, 8, 32, sink);
        if (anon_band != MAP_FAILED) {
            band("anon-noreg", anon_band, row, 8, 32, sink);
        }
        CK(hipFree(sink));
        (void) halloc;
    }

    // 4. The window shape itself, at three sizes, scattered order (stride = the 35B's expert stride).
    {
        const size_t stride = 589824;
        int order_lin[64], order_scr[64];
        for (int i = 0; i < 64; i++) { order_lin[i] = i; order_scr[i] = (i * 37) % 256; }
        window_trial(path, foff, stride, 64, order_lin, devbuf_dev, devbuf_host);
        window_trial(path, foff, stride, 64, order_scr, devbuf_dev, devbuf_host);
        // A larger window needs matching device buffers: read slot content in stride-sized pieces above, so only
        // the reserve grows.  (devbuf_* are `bytes` long; ask for stride == bytes from the caller.)
    }
    printf("\n");
    return 0;
}
