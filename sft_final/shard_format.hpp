// shard_format.hpp
// Binary format constants — MUST stay in sync with python/prepare_shards.py.
// See the SHARD FORMAT block in that file for the authoritative spec.
#pragma once
#include <cstdint>

namespace uds {

enum Band : uint32_t { BAND_SHORT = 0, BAND_MEDIUM = 1, BAND_LONG = 2, BAND_CHUNKED = 3 };
constexpr int NUM_BANDS = 4;

// Compute band from total length + cutoffs at LOAD time (so cutoffs can change
// without rebuilding shards). is_chunked overrides to CHUNKED.
inline uint32_t band_from_len(uint32_t total_len, bool is_chunked,
                              uint32_t short_max, uint32_t medium_max,
                              uint32_t max_seq_len) {
    if (is_chunked || total_len > max_seq_len) return BAND_CHUNKED;
    if (total_len <  short_max)  return BAND_SHORT;
    if (total_len <  medium_max) return BAND_MEDIUM;
    return BAND_LONG;
}

constexpr char       MAGIC[4]        = {'U', 'D', 'S', 'S'};
constexpr uint32_t   FORMAT_VERSION  = 1;
constexpr int        HEADER_SIZE     = 64;
constexpr int        INDEX_ENTRY_SIZE = 16;   // uint64 offset + uint32 nbytes + uint32 band
constexpr int        RECORD_HEADER_FIELDS = 7; // pl,cl,rl,tl,is_chunked,case,ncb
constexpr int        RECORD_HEADER_SIZE   = RECORD_HEADER_FIELDS * 4;

// Band codes (precomputed offline, stored in the index).
enum Band : uint32_t { BAND_SHORT = 0, BAND_MEDIUM = 1, BAND_LONG = 2, BAND_CHUNKED = 3 };
constexpr int NUM_BANDS = 4;

// Long-zone case codes.
enum Case : uint32_t { CASE_FIT = 0, CASE_LONG_CTX = 1, CASE_LONG_RESP = 2, CASE_MIXED = 3 };

} // namespace uds
