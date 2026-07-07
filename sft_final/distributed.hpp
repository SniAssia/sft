// distributed.hpp
// Phase 14 — each rank sees the SAME seeded shard order, then takes its subset
// by striding. Deterministic across ranks so no shard is trained twice / missed.
#pragma once
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace uds {

struct DistConfig {
    int rank = 0;
    int world_size = 1;
    uint64_t seed = 1234;
};

// Global seeded shuffle of shard indices, identical on every rank, then this
// rank keeps indices where (position % world_size == rank).
inline std::vector<std::string> assign_shards(
        const std::vector<std::string>& all_shards, const DistConfig& dc, int epoch) {
    std::vector<size_t> order(all_shards.size());
    std::iota(order.begin(), order.end(), 0);
    // seed folds in epoch so shard order changes each epoch but stays identical across ranks
    std::mt19937_64 rng(dc.seed ^ (0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(epoch)));
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<std::string> mine;
    for (size_t pos = 0; pos < order.size(); ++pos)
        if (static_cast<int>(pos % dc.world_size) == dc.rank)
            mine.push_back(all_shards[order[pos]]);
    return mine;
}

} // namespace uds
