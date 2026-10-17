// packetshield-agentic — tests/test_matcher.cpp
//
// CTest regression suite for the PacketShield DPI engine.
// Verifies detection correctness on standard intrusion payloads and memory
// safety (empty payloads, empty batches, pinned allocation round-trips).
// GPU-vs-CPU parity tests compile only when HAS_CUDA is defined.

#include "gpu_matcher.cuh"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                          \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

std::vector<uint8_t> bytes(const std::string& s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

// Deterministic clean payload: printable noise that cannot contain any
// signature (alphabet restricted to [A-Za-z0-9 ]).
std::vector<uint8_t> clean_payload(std::size_t len, unsigned seed) {
    constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ";
    std::vector<uint8_t> p(len);
    unsigned x = seed;
    for (std::size_t i = 0; i < len; ++i) {
        x = x * 1103515245u + 12345u;
        p[i] = static_cast<uint8_t>(kAlphabet[(x >> 16) % 63]);
    }
    return p;
}

// ---------------------------------------------------------------------------
// 1. Detection correctness: each signature is found at the right index.
// ---------------------------------------------------------------------------
void test_detection_correctness() {
    std::vector<std::vector<uint8_t>> payloads;
    for (int i = 0; i < 2000; ++i)
        payloads.push_back(clean_payload(256, static_cast<unsigned>(i + 1)));

    // Embed every signature into a distinct packet.
    for (int s = 0; s < packetshield::kNumSignatures; ++s) {
        const int idx = 100 + s * 37;
        const std::string sig = packetshield::kSignatures[s];
        auto& p = payloads[static_cast<std::size_t>(idx)];
        std::copy(sig.begin(), sig.end(), p.begin() + 40);
    }

    packetshield::PacketBatch batch = packetshield::create_batch(payloads);
    std::vector<packetshield::MatchResult> out(
        static_cast<std::size_t>(batch.count));
    const int hits = packetshield::cpu_inspect(batch, out.data());

    CHECK(hits == packetshield::kNumSignatures);
    for (int s = 0; s < packetshield::kNumSignatures; ++s) {
        const int idx = 100 + s * 37;
        CHECK(out[static_cast<std::size_t>(idx)].matched);
        CHECK(out[static_cast<std::size_t>(idx)].signature_id == s);
    }
    // A packet with no embedded signature must not match.
    CHECK(!out[0].matched);
    CHECK(out[0].signature_id == -1);

    packetshield::destroy_batch(batch);
    std::printf("PASS test_detection_correctness\n");
}

// ---------------------------------------------------------------------------
// 2. The benchmark's headline case: intrusion at packet #4200.
// ---------------------------------------------------------------------------
void test_injected_intrusion_4200() {
    constexpr int kN = 10000;
    constexpr int kIdx = 4200;
    std::vector<std::vector<uint8_t>> payloads;
    for (int i = 0; i < kN; ++i)
        payloads.push_back(clean_payload(512, static_cast<unsigned>(i + 7)));

    const std::string sig = packetshield::intrusion_signature();
    auto& target = payloads[kIdx];
    std::copy(sig.begin(), sig.end(), target.begin() + 128);

    packetshield::PacketBatch batch = packetshield::create_batch(payloads);
    std::vector<packetshield::MatchResult> out(
        static_cast<std::size_t>(batch.count));
    const int hits = packetshield::cpu_inspect(batch, out.data());

    CHECK(hits == 1);
    CHECK(out[kIdx].matched);
    CHECK(out[kIdx].signature_id == packetshield::kIntrusionSignatureId);
    CHECK(out[kIdx].packet_index == kIdx);

    packetshield::destroy_batch(batch);
    std::printf("PASS test_injected_intrusion_4200\n");
}

// ---------------------------------------------------------------------------
// 3. Memory safety: empty payloads, empty batch, zero-length handling.
// ---------------------------------------------------------------------------
void test_memory_safety() {
    // Empty payloads interleaved with normal ones.
    std::vector<std::vector<uint8_t>> payloads = {
        {}, clean_payload(128, 1), {}, bytes("hello"),
        bytes(packetshield::kSignatures[5]), {}};
    packetshield::PacketBatch batch = packetshield::create_batch(payloads);
    std::vector<packetshield::MatchResult> out(
        static_cast<std::size_t>(batch.count));
    const int hits = packetshield::cpu_inspect(batch, out.data());
    CHECK(hits == 1);
    CHECK(out[4].matched && out[4].signature_id == 5);
    CHECK(!out[0].matched && !out[2].matched && !out[5].matched);
    packetshield::destroy_batch(batch);
    CHECK(batch.data == nullptr && batch.count == 0);

    // Empty batch: must not crash, must report zero hits.
    std::vector<std::vector<uint8_t>> empty;
    packetshield::PacketBatch b2 = packetshield::create_batch(empty);
    CHECK(b2.count == 0);
    CHECK(packetshield::cpu_inspect(b2, nullptr) == 0);
    packetshield::destroy_batch(b2);

    // Pinned allocation round-trip.
    void* p = packetshield::alloc_pinned(4096);
    CHECK(p != nullptr);
    packetshield::free_pinned(p);
    packetshield::free_pinned(nullptr);  // freeing null is a no-op

    std::printf("PASS test_memory_safety\n");
}

// ---------------------------------------------------------------------------
// 4. Standard intrusion payloads (IDS-style attack strings).
// ---------------------------------------------------------------------------
void test_standard_intrusion_payloads() {
    const std::vector<std::pair<std::string, int>> cases = {
        {"GET /index.php?id=1 UNION SELECT * FROM users-- HTTP/1.1", 3},
        {"POST /upload.php?f=../../etc/passwd", 2},
        {"GET /x?cmd=" + std::string("%2e%2e%2f") + "win.ini", 7},
        {std::string("ping; /bin/sh -c 'id'"), 9},
        {"<script>alert(document.cookie)</script>", 8},
    };
    std::vector<std::vector<uint8_t>> payloads;
    for (const auto& c : cases) payloads.push_back(bytes(c.first));

    packetshield::PacketBatch batch = packetshield::create_batch(payloads);
    std::vector<packetshield::MatchResult> out(
        static_cast<std::size_t>(batch.count));
    packetshield::cpu_inspect(batch, out.data());

    for (std::size_t i = 0; i < cases.size(); ++i) {
        CHECK(out[i].matched);
        CHECK(out[i].signature_id == cases[i].second);
    }
    packetshield::destroy_batch(batch);
    std::printf("PASS test_standard_intrusion_payloads\n");
}

#ifdef HAS_CUDA
// ---------------------------------------------------------------------------
// 5. GPU/CPU parity: identical verdicts across a 20k-packet batch.
// ---------------------------------------------------------------------------
void test_gpu_cpu_parity() {
    constexpr int kN = 20000;
    std::vector<std::vector<uint8_t>> payloads;
    for (int i = 0; i < kN; ++i)
        payloads.push_back(clean_payload(384, static_cast<unsigned>(i * 3 + 11)));
    for (int s = 0; s < packetshield::kNumSignatures; ++s) {
        const std::string sig = packetshield::kSignatures[s];
        auto& p = payloads[static_cast<std::size_t>(500 + s * 613)];
        std::copy(sig.begin(), sig.end(), p.begin() + 64);
    }

    packetshield::PacketBatch batch = packetshield::create_batch(payloads);
    std::vector<packetshield::MatchResult> cpu_out(
        static_cast<std::size_t>(batch.count));
    std::vector<packetshield::MatchResult> gpu_out(
        static_cast<std::size_t>(batch.count));

    packetshield::cpu_inspect(batch, cpu_out.data());
    double gpu_ms = 0.0;
    const bool ok =
        packetshield::run_gpu_inspection(batch, gpu_out.data(), &gpu_ms);
    CHECK(ok);

    int mismatches = 0;
    for (int i = 0; i < batch.count; ++i) {
        if (cpu_out[static_cast<std::size_t>(i)].matched !=
                gpu_out[static_cast<std::size_t>(i)].matched ||
            cpu_out[static_cast<std::size_t>(i)].signature_id !=
                gpu_out[static_cast<std::size_t>(i)].signature_id)
            ++mismatches;
    }
    CHECK(mismatches == 0);
    std::printf("PASS test_gpu_cpu_parity (gpu %.2f ms)\n", gpu_ms);
    packetshield::destroy_batch(batch);
}
#endif

// ---------------------------------------------------------------------------
// 5. Agentic control plane contract: the Remediation agent can push a new
//    block rule at runtime and the data plane enforces it on the next batch.
// ---------------------------------------------------------------------------
void test_dynamic_signature_reload() {
    namespace fs = std::filesystem;
    const fs::path rules_file =
        fs::temp_directory_path() / "packetshield_test_rules.txt";

    // A zero-day payload the default table does not know about.
    const std::string zero_day = "ZERODAY_POLYMORPHIC_X9Q2";
    std::vector<std::vector<uint8_t>> payloads = {
        clean_payload(256, 42),
        bytes(std::string("prefix-") + zero_day + "-suffix"),
        clean_payload(256, 43),
    };
    packetshield::PacketBatch batch = packetshield::create_batch(payloads);
    std::vector<packetshield::MatchResult> out(
        static_cast<std::size_t>(batch.count));

    // Before the push: no detection.
    CHECK(packetshield::cpu_inspect(batch, out.data()) == 0);

    // The Remediation agent writes an updated rules file; the data plane
    // reloads it with zero downtime.
    {
        std::ofstream rules(rules_file);
        for (int s = 0; s < packetshield::kNumSignatures; ++s)
            rules << packetshield::kSignatures[s] << "\n";
        rules << "# pushed by Remediation agent\n" << zero_day << "\n";
    }
    CHECK(packetshield::reload_signatures(rules_file.string()));
    CHECK(static_cast<int>(packetshield::current_signatures().size()) ==
          packetshield::kNumSignatures + 1);

    // After the push: the zero-day is caught with the new rule's id.
    const int hits = packetshield::cpu_inspect(batch, out.data());
    CHECK(hits == 1);
    CHECK(out[1].matched);
    CHECK(out[1].signature_id == packetshield::kNumSignatures);

    // append_signature is deduplicated and also takes effect immediately.
    packetshield::append_signature(zero_day);
    CHECK(static_cast<int>(packetshield::current_signatures().size()) ==
          packetshield::kNumSignatures + 1);

    // A missing file must not clobber the live table.
    CHECK(!packetshield::reload_signatures("/nonexistent/rules.txt"));
    CHECK(static_cast<int>(packetshield::current_signatures().size()) ==
          packetshield::kNumSignatures + 1);

    // Restore defaults so later tests see the stock table.
    packetshield::reset_signatures();
    CHECK(static_cast<int>(packetshield::current_signatures().size()) ==
          packetshield::kNumSignatures);
    std::error_code ec;
    fs::remove(rules_file, ec);

    packetshield::destroy_batch(batch);
    std::printf("PASS test_dynamic_signature_reload\n");
}

}  // namespace

int main() {
    test_detection_correctness();
    test_injected_intrusion_4200();
    test_memory_safety();
    test_standard_intrusion_payloads();
    test_dynamic_signature_reload();
#ifdef HAS_CUDA
    test_gpu_cpu_parity();
#endif

    if (g_failures == 0) {
        std::printf("\nALL TESTS PASSED\n");
        return 0;
    }
    std::printf("\n%d TEST(S) FAILED\n", g_failures);
    return 1;
}
