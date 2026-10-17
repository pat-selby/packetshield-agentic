// packetshield-agentic — include/gpu_matcher.cuh
//
// Public interface for the PacketShield-Agentic deep packet inspection engine.
// Data plane: C++20 + CUDA. Control plane: Python/FastAPI/LangGraph agents
// (see agent/orchestrator.py) that update the signature table at runtime.
//
// The header is includable from both host C++ (.cpp) and CUDA (.cu)
// translation units; CUDA-specific declarations are guarded by __CUDACC__.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace packetshield {

// ---------------------------------------------------------------------------
// Intrusion signature database (compiled-in defaults).
// ---------------------------------------------------------------------------

// Default signature strings. kIntrusionSignatureId marks the signature that
// the benchmark harness injects into packet #4200.
inline constexpr const char* kSignatures[] = {
    "EXPLOIT_CVE_2024-8811",  // 0 — injected test intrusion signature
    "DROP TABLE users;--",    // 1 — SQL injection
    "../../etc/passwd",       // 2 — path traversal
    "UNION SELECT * FROM",    // 3 — SQL injection
    "eval(base64_decode(",    // 4 — PHP code injection
    "cmd.exe /c",             // 5 — Windows command execution
    "wget http://evil",       // 6 — malicious downloader
    "%2e%2e%2f",              // 7 — encoded path traversal
    "<script>alert(",         // 8 — cross-site scripting
    "/bin/sh -c",             // 9 — shell command execution
};

inline constexpr int kNumSignatures =
    static_cast<int>(sizeof(kSignatures) / sizeof(kSignatures[0]));
inline constexpr int kIntrusionSignatureId = 0;

// ---------------------------------------------------------------------------
// Dynamic signature table (agentic control plane integration).
// ---------------------------------------------------------------------------
//
// The Remediation agent synthesizes new block rules for zero-day /
// polymorphic payloads and pushes them to the data plane without service
// interruption. The table below is the live, mutable rule set inspected by
// both cpu_inspect() and run_gpu_inspection(); it starts as a copy of
// kSignatures and can be replaced or extended at runtime.

// Snapshot of the currently active signature table.
std::vector<std::string> current_signatures();

// Replace the active table from a rules file: one signature per line,
// '#' starts a comment, blank lines are ignored. Returns false on I/O error
// (the previous table is left untouched).
bool reload_signatures(const std::string& path);

// Append a single signature to the active table (deduplicated).
void append_signature(const std::string& signature);

// Restore the compiled-in default table.
void reset_signatures();

// ---------------------------------------------------------------------------
// Batch and result structures.
// ---------------------------------------------------------------------------

// A batch of packet payloads stored as one contiguous buffer plus per-packet
// offset/length tables. The payload buffer is allocated with pinned
// (page-locked) host memory via cudaHostAlloc when CUDA is available, which
// lets cudaMemcpyAsync transfer it to the device at full PCIe bandwidth
// without an extra staging copy inside the driver.
struct PacketBatch {
    uint8_t* data    = nullptr;  // concatenated payloads, pinned host memory
    int*     offsets = nullptr;  // byte offset of packet i inside data
    int*     lengths = nullptr;  // payload length of packet i in bytes
    int      count   = 0;        // number of packets in the batch

    // Total bytes across all payloads.
    std::size_t total_bytes() const {
        std::size_t n = 0;
        for (int i = 0; i < count; ++i) n += static_cast<std::size_t>(lengths[i]);
        return n;
    }
};

// Per-packet inspection verdict. Plain-old-data so it can be memcpy'd
// between host and device without marshalling.
struct MatchResult {
    int  packet_index = -1;  // index of the packet in the batch
    int  signature_id = -1;  // id of the first matching signature, -1 if none
    bool matched      = false;
};

// ---------------------------------------------------------------------------
// Memory management.
// ---------------------------------------------------------------------------

// Allocate page-locked (pinned) host memory. Falls back to 64-byte aligned
// malloc when CUDA is unavailable so CPU-only builds still run.
void* alloc_pinned(std::size_t bytes);
void  free_pinned(void* ptr);

// Allocate and populate a PacketBatch in pinned memory from a vector of
// payloads. The caller owns the batch and must call destroy_batch().
PacketBatch create_batch(const std::vector<std::vector<uint8_t>>& payloads);
void        destroy_batch(PacketBatch& batch);

// ---------------------------------------------------------------------------
// Inspection entry points.
// ---------------------------------------------------------------------------

// Multi-threaded CPU baseline: inspects every packet against every signature
// in the *current* signature table using std::thread over
// hardware_concurrency() workers. Returns the number of matched packets;
// fills `out` (must hold batch.count entries).
int cpu_inspect(const PacketBatch& batch, MatchResult* out);

// GPU inspection pipeline: pinned staging -> async H2D copy on a dedicated
// CUDA stream -> match_signatures_kernel -> async D2H copy of results ->
// stream synchronize. The kernel receives the *current* signature table, so
// agent-pushed rules take effect on the next batch with no restart.
// On success returns true and, when ms_out is non-null, writes the
// end-to-end elapsed time in milliseconds (CUDA-event measured).
// Returns false when CUDA is unavailable or any CUDA call fails.
bool run_gpu_inspection(const PacketBatch& batch, MatchResult* out,
                        double* ms_out = nullptr);

// Human-readable name of the injected test signature.
inline std::string intrusion_signature() { return kSignatures[kIntrusionSignatureId]; }

}  // namespace packetshield

#ifdef __CUDACC__
// CUDA kernel: one thread per packet. Each thread scans its payload against
// every signature and records the first match.
__global__ void match_signatures_kernel(const uint8_t* __restrict__ payloads,
                                        const int* __restrict__ offsets,
                                        const int* __restrict__ lengths,
                                        const uint8_t* __restrict__ sig_data,
                                        const int* __restrict__ sig_offsets,
                                        const int* __restrict__ sig_lengths,
                                        int num_packets, int num_signatures,
                                        packetshield::MatchResult* __restrict__ results);
#endif
