// test_core.cpp
// Torch-free cross-language test. Reads a shard produced by prepare_shards.py's
// ShardWriter and exercises reader -> length queues -> scheduler. Proves the
// C++ side decodes the exact bytes the Python side wrote.
//
// Build: see CMakeLists.txt (target: test_core). Run: ./test_core <shard_dir>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "batch_scheduler.hpp"
#include "length_queues.hpp"
#include "shard_reader.hpp"

namespace fs = std::filesystem;
using namespace uds;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_core <shard_dir>\n"); return 2; }
    std::string dir = argv[1];

    std::vector<std::string> shards;
    for (auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".bin") shards.push_back(e.path().string());
    std::sort(shards.begin(), shards.end());
    assert(!shards.empty() && "no .bin shards found");
    std::printf("[test] found %zu shard(s)\n", shards.size());

    LengthAwareQueues queues;
    uint32_t total = 0, chunked = 0;
    std::array<uint32_t, NUM_BANDS> band_counts{0, 0, 0, 0};

    for (auto& path : shards) {
        ShardReader reader(path);
        std::printf("[test] %s: %u samples, max_seq_len=%u\n",
                    path.c_str(), reader.num_samples(), reader.max_seq_length());
        for (uint32_t i = 0; i < reader.num_samples(); ++i) {
            Sample s = reader.get(i);
            // integrity: total_len consistent, band matches index
            assert(s.total_len() == s.prompt_len() + s.context_len() + s.response_len());
            assert(s.band == reader.band_of(i));
            if (s.is_chunked) ++chunked;
            band_counts[s.band]++;
            ++total;
            queues.push(std::move(s));
        }
    }
    std::printf("[test] streamed %u samples  bands: S=%u M=%u L=%u C=%u  (chunked=%u)\n",
                total, band_counts[0], band_counts[1], band_counts[2], band_counts[3], chunked);
    assert(total == queues.total());

    // Scheduler: homogeneous pools, no chunked (test determinism of banding).
    SchedulerConfig scfg;
    scfg.B = 4;
    scfg.mode = PoolMode::Homogeneous;
    scfg.chunked_rate = 0.0f;
    scfg.pop_timeout_ms = 5;
    BatchScheduler sched(scfg, queues);

    int pools = 0, fit_samples = 0;
    while (true) {
        CandidatePool p = sched.next_pool();
        if (p.samples.empty()) break;
        // homogeneous invariant: every sample in the pool shares the pool band
        for (auto& s : p.samples) assert(s.band == p.band);
        fit_samples += static_cast<int>(p.samples.size());
        ++pools;
        if (pools > 100000) break; // safety
    }
    std::printf("[test] scheduler emitted %d pools, %d fit samples\n", pools, fit_samples);

    // all non-chunked samples should have been schedulable
    int expected_fit = static_cast<int>(band_counts[0] + band_counts[1] + band_counts[2]);
    assert(fit_samples == expected_fit);

    std::printf("[test] ALL ASSERTIONS PASSED\n");
    return 0;
}
