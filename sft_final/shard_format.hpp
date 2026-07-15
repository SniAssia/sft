// shard_format.hpp
// Binary format constants — MUST stay in sync with python/prepare_shards.py.
#pragma once
#include <cstdint>
#include <vector>

namespace uds {

// Compile-time ceiling on band count. Arrays are sized by this; loops are
// bounded by the RUNTIME count (queues_.num_bands()). Raise if you need >8.
constexpr int MAX_BANDS = 8;

// Legacy fixed names (band 3 == chunked only in the default 4-band setup).
enum Band : uint32_t { BAND_SHORT = 0, BAND_MEDIUM = 1, BAND_LONG = 2, BAND_CHUNKED = 3 };

// k length bands (from k-1 internal cutoffs) + 1 chunked band = k+1 total.
inline uint32_t num_bands_for(const std::vector<uint32_t>& cutoffs) {
    return static_cast<uint32_t>(cutoffs.size()) + 2;
}

// Compute band from total length + cutoffs at LOAD time, so cutoffs can change
// without rebuilding shards. CHUNKED is always the LAST band index.
//   cutoffs = [c0, c1, ...] ascending:
//     len <  c0            -> band 0
//     c0 <= len < c1       -> band 1   ... etc
//     len >= last cutoff   -> band (num_bands-2)
//     chunked / > max_seq  -> band (num_bands-1)
inline uint32_t band_from_len(uint32_t total_len, bool is_chunked,
                              const std::vector<uint32_t>& cutoffs,
                              uint32_t max_seq_len) {
    const uint32_t nb = num_bands_for(cutoffs);
    if (is_chunked || total_len > max_seq_len) return nb - 1;
    for (uint32_t i = 0; i < cutoffs.size(); ++i)
        if (total_len < cutoffs[i]) return i;
    return nb - 2;
}

constexpr char       MAGIC[4]        = {'U', 'D', 'S', 'S'};
constexpr uint32_t   FORMAT_VERSION  = 1;
constexpr int        HEADER_SIZE     = 64;
constexpr int        INDEX_ENTRY_SIZE = 16;
constexpr int        RECORD_HEADER_FIELDS = 7;
constexpr int        RECORD_HEADER_SIZE   = RECORD_HEADER_FIELDS * 4;

enum Case : uint32_t { CASE_FIT = 0, CASE_LONG_CTX = 1, CASE_LONG_RESP = 2, CASE_MIXED = 3 };

} // namespace uds