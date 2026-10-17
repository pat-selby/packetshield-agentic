// packetshield-agentic — src/main.cpp
//
// Benchmark harness: generates 100,000 synthetic packet payloads averaging
// 512 bytes, injects the intrusion test signature into packet #4200, then
// times the multi-threaded CPU baseline against the CUDA GPU pipeline with
// std::chrono and prints a comparison table.

#include "gpu_matcher.cuh"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kNumPackets = 100000;
constexpr int kInjectIndex = 4200;      // packet carrying the test intrusion
constexpr int kAvgPayloadBytes = 512;  // target average payload size

// Deterministic synthetic traffic: payloads drawn from [A-Za-z0-9 ] so no
// signature can appear by accident; lengths uniform in [64, 960] (mean 512).
std::vector<std::vector<uint8_t>> generate_traffic() {
    std::mt19937_64 rng(0xC10C4A11u);  // fixed seed: reproducible benchmarks
    std::uniform_int_distribution<int> len_dist(64, 960);
    std::uniform_int_distribution<int> char_dist(0, 62);
    constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ";

    std::vector<std::vector<uint8_t>> payloads;
    payloads.reserve(kNumPackets);
    for (int i = 0; i < kNumPackets; ++i) {
        const int len = len_dist(rng);
        std::vector<uint8_t> p(static_cast<std::size_t>(len));
        for (int j = 0; j < len; ++j)
            p[static_cast<std::size_t>(j)] =
                static_cast<uint8_t>(kAlphabet[char_dist(rng)]);
        payloads.push_back(std::move(p));
    }

    // Inject the intrusion signature into packet #4200 at a fixed offset.
    const std::string sig = packetshield::intrusion_signature();
    auto& target = payloads[kInjectIndex];
    constexpr std::size_t kInjectOffset = 128;
    if (target.size() >= kInjectOffset + sig.size()) {
        std::copy(sig.begin(), sig.end(),
                  target.begin() + static_cast<std::ptrdiff_t>(kInjectOffset));
    }
    return payloads;
}

template <typename F>
double time_ms(F&& fn) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

int main() {
    using packetshield::MatchResult;
    using packetshield::PacketBatch;

    std::cout << "PacketShield DPI benchmark\n"
              << "=========================\n";
    std::printf("Packets        : %d (avg ~%d bytes)\n", kNumPackets, kAvgPayloadBytes);
    std::printf("Signatures     : %d\n", packetshield::kNumSignatures);
    std::printf("Injected at    : packet #%d  (\"%s\")\n\n", kInjectIndex,
                packetshield::intrusion_signature().c_str());

    PacketBatch batch = packetshield::create_batch(generate_traffic());
    std::printf("Batch bytes    : %zu\n\n", batch.total_bytes());

    std::vector<MatchResult> cpu_results(static_cast<std::size_t>(batch.count));
    std::vector<MatchResult> gpu_results(static_cast<std::size_t>(batch.count));

    // Warm-up (page in memory, spin up the thread pool).
    packetshield::cpu_inspect(batch, cpu_results.data());

    // --- CPU baseline timing.
    int cpu_hits = 0;
    const double cpu_ms = time_ms([&] {
        cpu_hits = packetshield::cpu_inspect(batch, cpu_results.data());
    });

    const bool cpu_ok = cpu_results[kInjectIndex].matched &&
                        cpu_results[kInjectIndex].signature_id ==
                            packetshield::kIntrusionSignatureId;

    std::printf("[CPU]  %8.2f ms   hits=%d   intrusion@%d %s\n", cpu_ms, cpu_hits,
                kInjectIndex, cpu_ok ? "DETECTED" : "MISSED");

    // --- GPU pipeline timing.
    double gpu_ms = 0.0;
    const bool gpu_ran =
        packetshield::run_gpu_inspection(batch, gpu_results.data(), &gpu_ms);

    if (gpu_ran) {
        int gpu_hits = 0;
        for (const auto& r : gpu_results)
            if (r.matched) ++gpu_hits;
        const bool gpu_ok = gpu_results[kInjectIndex].matched &&
                            gpu_results[kInjectIndex].signature_id ==
                                packetshield::kIntrusionSignatureId;
        std::printf("[GPU]  %8.2f ms   hits=%d   intrusion@%d %s\n", gpu_ms,
                    gpu_hits, kInjectIndex, gpu_ok ? "DETECTED" : "MISSED");

        // Cross-check GPU vs CPU verdicts.
        bool agree = true;
        for (int i = 0; i < batch.count; ++i) {
            if (cpu_results[i].matched != gpu_results[i].matched ||
                cpu_results[i].signature_id != gpu_results[i].signature_id) {
                agree = false;
                std::printf("MISMATCH at packet %d\n", i);
                break;
            }
        }
        std::printf("\nCPU/GPU agreement: %s\n", agree ? "YES" : "NO");
        std::printf("Speedup (CPU/GPU): %.2fx\n", cpu_ms / gpu_ms);
    } else {
        std::printf("[GPU]  unavailable in this build (no CUDA runtime)\n");
        std::printf("\nRebuild with nvcc / CUDA Toolkit for GPU numbers.\n");
    }

    packetshield::destroy_batch(batch);
    const bool ok = cpu_ok && (!gpu_ran || gpu_results[kInjectIndex].matched);
    return ok ? 0 : 1;
}
