// sample.hpp
// A decoded training sample. Deliberately torch-free so the reader / queues /
// scheduler compile and unit-test without libtorch. Only the collator turns
// these into tensors.
#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include "shard_format.hpp"

namespace uds {

struct Sample {
    // token ids, stored per-zone; full sequence == prompt ++ context ++ response
    std::vector<int32_t> prompt_ids;
    std::vector<int32_t> context_ids;
    std::vector<int32_t> response_ids;

    uint32_t is_chunked = 0;
    uint32_t case_code  = CASE_FIT;
    uint32_t band       = BAND_SHORT;
    std::vector<uint32_t> chunk_bounds;

    uint32_t prompt_len()   const { return static_cast<uint32_t>(prompt_ids.size()); }
    uint32_t context_len()  const { return static_cast<uint32_t>(context_ids.size()); }
    uint32_t response_len() const { return static_cast<uint32_t>(response_ids.size()); }
    uint32_t total_len()    const { return prompt_len() + context_len() + response_len(); }
    uint32_t input_len()    const { return prompt_len() + context_len(); }
};

// A candidate pool: B samples the scheduler emits for UDS scoring.
struct CandidatePool {
    std::vector<Sample> samples;
    uint32_t band = BAND_SHORT;   // primary band (bookkeeping)
    bool is_chunked = false;      // true => Option-B representative-window collation
    // profile-based scheduling metadata
    int profile_index = -1;             // which profile produced this pool
    std::array<int, 2> profile_bands = {BAND_SHORT, BAND_SHORT};
    bool mixed = false;                 // true for two-band profiles (P0/P1)
    bool fell_back = false;             // a band was empty -> filled 100% from the other
};

} // namespace uds