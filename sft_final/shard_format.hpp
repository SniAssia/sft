// shard_format.hpp
// Binary format constants — MUST stay in sync with python/prepare_shards.py.
// See the SHARD FORMAT block in that file for the authoritative spec.
#pragma once
#include <cstdint>

namespace uds {

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
