// packetshield-agentic — src/cpu_matcher.cpp
//
// Multi-threaded CPU baseline for the PacketShield-Agentic DPI engine, plus
// the live signature table shared with the CUDA data plane.
//
// The table is guarded by a std::shared_mutex: inspection workers take the
// read side (many concurrent readers, no copies in the hot loop beyond one
// snapshot per cpu_inspect call), while the agentic control plane's
// reload_signatures()/append_signature() take the write side. This gives the
// Remediation agent lock-free-style rule pushes with no service interruption.

#include "gpu_matcher.cuh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

namespace packetshield {
namespace {

// ---------------------------------------------------------------------------
// Live signature table.
// ---------------------------------------------------------------------------
class SignatureTable {
public:
    SignatureTable() { reset(); }

    std::vector<std::string> snapshot() const {
        std::shared_lock lock(mutex_);
        return signatures_;
    }

    bool reload_from_file(const std::string& path) {
        std::vector<std::string> fresh;
        std::ifstream in(path);
        if (!in.is_open()) return false;
        std::string line;
        while (std::getline(in, line)) {
            // Strip comments and surrounding whitespace.
            const auto hash = line.find('#');
            if (hash != std::string::npos) line.erase(hash);
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            const auto last = line.find_last_not_of(" \t\r\n");
            line = line.substr(first, last - first + 1);
            if (!line.empty() &&
                std::find(fresh.begin(), fresh.end(), line) == fresh.end())
                fresh.push_back(line);
        }
        if (fresh.empty()) return false;  // never install an empty rule set
        std::unique_lock lock(mutex_);
        signatures_ = std::move(fresh);
        return true;
    }

    void append(const std::string& sig) {
        if (sig.empty()) return;
        std::unique_lock lock(mutex_);
        if (std::find(signatures_.begin(), signatures_.end(), sig) ==
            signatures_.end())
            signatures_.push_back(sig);
    }

    void reset() {
        std::unique_lock lock(mutex_);
        signatures_.assign(kSignatures, kSignatures + kNumSignatures);
    }

private:
    mutable std::shared_mutex mutex_;
    std::vector<std::string> signatures_;
};

SignatureTable& table() {
    static SignatureTable instance;
    return instance;
}

// ---------------------------------------------------------------------------
// Matching core (same semantics as the CUDA kernel).
// ---------------------------------------------------------------------------
inline bool contains_signature(const uint8_t* payload, int len,
                               const uint8_t* sig, int slen) {
    if (slen == 0 || slen > len) return false;
    const uint8_t first = sig[0];
    for (int i = 0; i + slen <= len; ++i) {
        if (payload[i] != first) continue;              // cheap first-byte filter
        if (std::memcmp(payload + i, sig, static_cast<std::size_t>(slen)) == 0)
            return true;
    }
    return false;
}

struct SigView {
    const uint8_t* data;
    int len;
};

void inspect_range(const PacketBatch& batch, MatchResult* out,
                   const std::vector<SigView>& sigs, int begin, int end) {
    for (int p = begin; p < end; ++p) {
        const uint8_t* payload = batch.data + batch.offsets[p];
        const int len = batch.lengths[p];
        int found = -1;
        for (int s = 0; s < static_cast<int>(sigs.size()) && found < 0; ++s) {
            if (contains_signature(payload, len, sigs[s].data, sigs[s].len))
                found = s;
        }
        out[p].packet_index = p;
        out[p].signature_id = found;
        out[p].matched = (found >= 0);
    }
}

}  // namespace

// --- Dynamic table API -------------------------------------------------------
std::vector<std::string> current_signatures() { return table().snapshot(); }
bool reload_signatures(const std::string& path) {
    return table().reload_from_file(path);
}
void append_signature(const std::string& signature) { table().append(signature); }
void reset_signatures() { table().reset(); }

// --- CPU inspection -----------------------------------------------------------
int cpu_inspect(const PacketBatch& batch, MatchResult* out) {
    if (batch.count <= 0 || out == nullptr) return 0;

    // One snapshot per call; workers share read-only views into it.
    const std::vector<std::string> owned = table().snapshot();
    std::vector<SigView> sigs;
    sigs.reserve(owned.size());
    for (const auto& s : owned)
        sigs.push_back({reinterpret_cast<const uint8_t*>(s.data()),
                        static_cast<int>(s.size())});

    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 4;
    nthreads = std::min<unsigned>(nthreads, static_cast<unsigned>(batch.count));

    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    const int chunk = (batch.count + static_cast<int>(nthreads) - 1) /
                      static_cast<int>(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) {
        const int begin = static_cast<int>(t) * chunk;
        const int end = std::min(begin + chunk, batch.count);
        if (begin >= end) break;
        workers.emplace_back(inspect_range, std::cref(batch), out,
                             std::cref(sigs), begin, end);
    }
    for (auto& w : workers) w.join();

    int matched = 0;
    for (int i = 0; i < batch.count; ++i)
        if (out[i].matched) ++matched;
    return matched;
}

#ifndef HAS_CUDA
// CPU-only build: pinned helpers fall back to aligned heap memory.
void* alloc_pinned(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    constexpr std::size_t kAlign = 64;
    if (::posix_memalign(&ptr, kAlign, (bytes + kAlign - 1) & ~(kAlign - 1)) != 0)
        return nullptr;
    return ptr;
}

void free_pinned(void* ptr) { ::free(ptr); }

bool run_gpu_inspection(const PacketBatch&, MatchResult*, double*) {
    return false;  // no CUDA runtime in this build
}
#endif

PacketBatch create_batch(const std::vector<std::vector<uint8_t>>& payloads) {
    PacketBatch batch;
    batch.count = static_cast<int>(payloads.size());
    if (batch.count == 0) return batch;

    batch.offsets = static_cast<int*>(alloc_pinned(
        static_cast<std::size_t>(batch.count) * sizeof(int)));
    batch.lengths = static_cast<int*>(alloc_pinned(
        static_cast<std::size_t>(batch.count) * sizeof(int)));

    std::size_t total = 0;
    for (int i = 0; i < batch.count; ++i) {
        batch.offsets[i] = static_cast<int>(total);
        batch.lengths[i] = static_cast<int>(payloads[i].size());
        total += payloads[i].size();
    }

    batch.data = static_cast<uint8_t*>(alloc_pinned(total == 0 ? 1 : total));
    for (int i = 0; i < batch.count; ++i) {
        if (!payloads[i].empty())
            std::memcpy(batch.data + batch.offsets[i], payloads[i].data(),
                        payloads[i].size());
    }
    return batch;
}

void destroy_batch(PacketBatch& batch) {
    free_pinned(batch.data);
    free_pinned(batch.offsets);
    free_pinned(batch.lengths);
    batch.data = nullptr;
    batch.offsets = nullptr;
    batch.lengths = nullptr;
    batch.count = 0;
}

}  // namespace packetshield
