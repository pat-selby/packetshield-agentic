// packetshield-agentic — src/gpu_matcher.cu
//
// CUDA implementation of the PacketShield DPI engine.
//
// Pipeline (all on one dedicated CUDA stream):
//   1. Host packet batch already lives in pinned memory (cudaHostAlloc).
//   2. cudaMalloc device buffers for payloads, offset/length tables,
//      the concatenated signature blob, and the result array.
//   3. cudaMemcpyAsync (host->device) for every input buffer.
//   4. match_signatures_kernel<<<grid, block, 0, stream>>>().
//   5. cudaMemcpyAsync (device->host) for the MatchResult array.
//   6. cudaStreamSynchronize + CUDA-event timing around steps 3-5.
//
// Threading model: one thread per packet. Threads are grouped into
// 256-thread blocks (8 warps); warps whose packets share a block read the
// signature blob from the read-only data cache (__ldg), and consecutive
// threads read consecutive packets so global-memory accesses coalesce.

#include "gpu_matcher.cuh"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace packetshield {
namespace {

// RAII wrapper for device memory: cudaMalloc on construction, cudaFree on
// destruction. Non-copyable, movable.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes) { allocate(bytes); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), bytes_(other.bytes_) {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t bytes) {
        release();
        if (bytes == 0) return;
        cudaError_t err = cudaMalloc(&ptr_, bytes);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                     cudaGetErrorString(err));
        bytes_ = bytes;
    }

    void*       get()   const { return ptr_; }
    std::size_t bytes() const { return bytes_; }

private:
    void release() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            bytes_ = 0;
        }
    }

    void*       ptr_   = nullptr;
    std::size_t bytes_ = 0;
};

// RAII wrapper for a CUDA stream.
class CudaStream {
public:
    CudaStream() {
        cudaError_t err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("cudaStreamCreate failed: ") +
                                     cudaGetErrorString(err));
    }
    ~CudaStream() {
        if (stream_) cudaStreamDestroy(stream_);
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

// Throw on CUDA error with context.
inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(err));
}

// Device-side exact substring test. __ldg() routes signature reads through
// the read-only data cache since every thread scans the same signatures.
__device__ __forceinline__ bool device_contains(const uint8_t* __restrict__ payload,
                                               int len,
                                               const uint8_t* __restrict__ sig,
                                               int slen) {
    if (slen == 0 || slen > len) return false;
    const uint8_t first = __ldg(sig);
    for (int i = 0; i + slen <= len; ++i) {
        if (payload[i] != first) continue;
        int j = 1;
        for (; j < slen; ++j) {
            if (payload[i + j] != __ldg(sig + j)) break;
        }
        if (j == slen) return true;
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// Kernel: parallel multi-string search, one thread per packet.
// ---------------------------------------------------------------------------
__global__ void match_signatures_kernel(const uint8_t* __restrict__ payloads,
                                        const int* __restrict__ offsets,
                                        const int* __restrict__ lengths,
                                        const uint8_t* __restrict__ sig_data,
                                        const int* __restrict__ sig_offsets,
                                        const int* __restrict__ sig_lengths,
                                        int num_packets, int num_signatures,
                                        MatchResult* __restrict__ results) {
    const int pkt = blockIdx.x * blockDim.x + threadIdx.x;
    if (pkt >= num_packets) return;

    const uint8_t* payload = payloads + offsets[pkt];
    const int len = lengths[pkt];

    int found = -1;
    for (int s = 0; s < num_signatures && found < 0; ++s) {
        const uint8_t* sig = sig_data + sig_offsets[s];
        if (device_contains(payload, len, sig, sig_lengths[s])) found = s;
    }

    results[pkt].packet_index = pkt;
    results[pkt].signature_id = found;
    results[pkt].matched = (found >= 0);
}

// ---------------------------------------------------------------------------
// Host-side pinned allocation (page-locked memory for DMA-speed transfers).
// ---------------------------------------------------------------------------
void* alloc_pinned(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) return nullptr;
    return ptr;
}

void free_pinned(void* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

// ---------------------------------------------------------------------------
// Host wrapper: full async inspection pipeline on a dedicated stream.
// ---------------------------------------------------------------------------
bool run_gpu_inspection(const PacketBatch& batch, MatchResult* out,
                        double* ms_out) {
    if (batch.count <= 0 || out == nullptr) return false;

    try {
        check(cudaSetDevice(0), "cudaSetDevice");

        // --- Pack the *live* signature table (agent-updatable) into
        // contiguous device-friendly blobs. Rules pushed by the Remediation
        // agent take effect on the very next batch with no restart.
        const std::vector<std::string> live_sigs = current_signatures();
        const int num_sigs = static_cast<int>(live_sigs.size());
        if (num_sigs == 0)
            throw std::runtime_error("signature table is empty");
        std::vector<uint8_t> sig_blob;
        std::vector<int> sig_offsets;
        std::vector<int> sig_lengths;
        sig_blob.reserve(256);
        for (int s = 0; s < num_sigs; ++s) {
            sig_offsets.push_back(static_cast<int>(sig_blob.size()));
            const std::string& str = live_sigs[static_cast<std::size_t>(s)];
            const int slen = static_cast<int>(str.size());
            sig_lengths.push_back(slen);
            sig_blob.insert(sig_blob.end(), str.begin(), str.end());
        }

        const std::size_t payload_bytes = batch.total_bytes();
        const std::size_t result_bytes =
            static_cast<std::size_t>(batch.count) * sizeof(MatchResult);

        // --- Device allocations (RAII: freed automatically on any throw).
        DeviceBuffer d_payloads(payload_bytes == 0 ? 1 : payload_bytes);
        DeviceBuffer d_offsets(static_cast<std::size_t>(batch.count) * sizeof(int));
        DeviceBuffer d_lengths(static_cast<std::size_t>(batch.count) * sizeof(int));
        DeviceBuffer d_sig_data(sig_blob.empty() ? 1 : sig_blob.size());
        DeviceBuffer d_sig_offsets(static_cast<std::size_t>(num_sigs) * sizeof(int));
        DeviceBuffer d_sig_lengths(static_cast<std::size_t>(num_sigs) * sizeof(int));
        DeviceBuffer d_results(result_bytes);

        CudaStream stream;
        cudaStream_t st = stream.get();

        // --- CUDA events for end-to-end timing (transfers + kernel).
        cudaEvent_t start = nullptr, stop = nullptr;
        check(cudaEventCreate(&start), "cudaEventCreate");
        check(cudaEventCreate(&stop), "cudaEventCreate");

        check(cudaEventRecord(start, st), "cudaEventRecord");

        // --- Async host->device copies (pinned source => DMA, no staging).
        if (payload_bytes > 0)
            check(cudaMemcpyAsync(d_payloads.get(), batch.data, payload_bytes,
                                  cudaMemcpyHostToDevice, st),
                  "H2D payloads");
        check(cudaMemcpyAsync(d_offsets.get(), batch.offsets,
                              static_cast<std::size_t>(batch.count) * sizeof(int),
                              cudaMemcpyHostToDevice, st),
              "H2D offsets");
        check(cudaMemcpyAsync(d_lengths.get(), batch.lengths,
                              static_cast<std::size_t>(batch.count) * sizeof(int),
                              cudaMemcpyHostToDevice, st),
              "H2D lengths");
        check(cudaMemcpyAsync(d_sig_data.get(), sig_blob.data(), sig_blob.size(),
                              cudaMemcpyHostToDevice, st),
              "H2D signatures");
        check(cudaMemcpyAsync(d_sig_offsets.get(), sig_offsets.data(),
                              sig_offsets.size() * sizeof(int),
                              cudaMemcpyHostToDevice, st),
              "H2D sig offsets");
        check(cudaMemcpyAsync(d_sig_lengths.get(), sig_lengths.data(),
                              sig_lengths.size() * sizeof(int),
                              cudaMemcpyHostToDevice, st),
              "H2D sig lengths");

        // --- Kernel launch: 256 threads/block = 8 warps/block.
        constexpr int kThreadsPerBlock = 256;
        const int blocks = (batch.count + kThreadsPerBlock - 1) / kThreadsPerBlock;
        match_signatures_kernel<<<blocks, kThreadsPerBlock, 0, st>>>(
            static_cast<const uint8_t*>(d_payloads.get()),
            static_cast<const int*>(d_offsets.get()),
            static_cast<const int*>(d_lengths.get()),
            static_cast<const uint8_t*>(d_sig_data.get()),
            static_cast<const int*>(d_sig_offsets.get()),
            static_cast<const int*>(d_sig_lengths.get()),
            batch.count, num_sigs,
            static_cast<MatchResult*>(d_results.get()));
        check(cudaGetLastError(), "kernel launch");

        // --- Async device->host copy of results, then synchronize.
        check(cudaMemcpyAsync(out, d_results.get(), result_bytes,
                              cudaMemcpyDeviceToHost, st),
              "D2H results");
        check(cudaEventRecord(stop, st), "cudaEventRecord");
        check(cudaStreamSynchronize(st), "cudaStreamSynchronize");

        if (ms_out) {
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
            *ms_out = static_cast<double>(ms);
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace packetshield
