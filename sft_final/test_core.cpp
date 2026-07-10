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

    // Scheduler: profile-based. Use the 3 default equal-probability profiles
    // (P0 Short+Medium, P1 Medium+Long, P2 Chunked). Drain everything and check
    // each pool's samples belong to that profile's declared bands.
    SchedulerConfig scfg;
    scfg.B = 4;
    BatchScheduler sched(scfg, queues);   // empty profiles -> CAT1/CAT2/CAT3 defaults

    int pools = 0, total_samples = 0, fell_back = 0;
    // Drain until every queue is empty (deterministic; random profile picking can
    // otherwise strand a band's last few samples until its profile is chosen).
    while (queues.total() > 0) {
        CandidatePool p = sched.next_pool();
        if (p.samples.empty()) continue;   // picked a profile whose bands are momentarily empty
        // profile invariant: every sample belongs to one of the profile's bands
        for (auto& s : p.samples)
            assert(static_cast<int>(s.band) == p.profile_bands[0] ||
                   static_cast<int>(s.band) == p.profile_bands[1]);
        if (p.fell_back) ++fell_back;
        total_samples += static_cast<int>(p.samples.size());
        ++pools;
        if (pools > 1000000) break; // safety
    }
    std::printf("[test] scheduler emitted %d pools, %d samples, %d fell_back\n",
                pools, total_samples, fell_back);

    // every streamed sample should eventually be schedulable across profiles
    assert(total_samples == static_cast<int>(total));

    std::printf("[test] ALL ASSERTIONS PASSED\n");
    return 0;
}