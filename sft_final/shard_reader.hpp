// shard_reader.hpp
// Memory-maps a shard_XXXXX.bin and provides zero-copy random access to records.
// Mirrors python/prepare_shards.py exactly. Torch-free.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sample.hpp"
#include "shard_format.hpp"

namespace uds {

struct IndexEntry {
    uint64_t offset;   // byte offset of record from start of file
    uint32_t nbytes;
    uint32_t band;
};

class ShardReader {
public:
    explicit ShardReader(const std::string& path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("ShardReader: cannot open " + path);
        struct stat st{};
        if (::fstat(fd_, &st) != 0) { ::close(fd_); throw std::runtime_error("fstat failed"); }
        size_ = static_cast<size_t>(st.st_size);
        base_ = static_cast<const uint8_t*>(::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (base_ == MAP_FAILED) { ::close(fd_); throw std::runtime_error("mmap failed"); }
        parse_header_();
    }

    ~ShardReader() {
        if (base_ && base_ != MAP_FAILED) ::munmap(const_cast<uint8_t*>(base_), size_);
        if (fd_ >= 0) ::close(fd_);
    }

    ShardReader(const ShardReader&) = delete;
    ShardReader& operator=(const ShardReader&) = delete;

    uint32_t num_samples()    const { return num_samples_; }
    uint32_t max_seq_length() const { return max_seq_length_; }
    uint32_t band_of(uint32_t i) const { return index_[i].band; }

    // Decode sample i into a Sample (copies token ids out of the mmap).
    Sample get(uint32_t i) const {
        if (i >= num_samples_) throw std::out_of_range("shard index");
        const uint8_t* p = base_ + index_[i].offset;
        uint32_t h[RECORD_HEADER_FIELDS];
        std::memcpy(h, p, RECORD_HEADER_SIZE);
        const uint32_t pl = h[0], cl = h[1], rl = h[2], /*tl=*/ _tl = h[3],
                       isk = h[4], cc = h[5], ncb = h[6];
        (void)_tl;
        p += RECORD_HEADER_SIZE;

        Sample s;
        s.is_chunked = isk;
        s.case_code  = cc;
        s.band       = index_[i].band;

        auto read_ids = [&](std::vector<int32_t>& out, uint32_t n) {
            out.resize(n);
            if (n) std::memcpy(out.data(), p, n * sizeof(int32_t));
            p += n * sizeof(int32_t);
        };
        read_ids(s.prompt_ids, pl);
        read_ids(s.context_ids, cl);
        read_ids(s.response_ids, rl);

        s.chunk_bounds.resize(ncb);
        if (ncb) std::memcpy(s.chunk_bounds.data(), p, ncb * sizeof(uint32_t));
        return s;
    }

private:
    void parse_header_() {
        if (size_ < static_cast<size_t>(HEADER_SIZE))
            throw std::runtime_error("shard too small for header");
        if (std::memcmp(base_, MAGIC, 4) != 0)
            throw std::runtime_error("bad magic in " + path_);
        uint32_t hdr[5];
        std::memcpy(hdr, base_ + 4, 20);
        // hdr = {version, num_samples, max_seq_length, token_dtype, record_align}
        if (hdr[0] != FORMAT_VERSION)
            throw std::runtime_error("shard version mismatch");
        num_samples_    = hdr[1];
        max_seq_length_ = hdr[2];

        index_.resize(num_samples_);
        const uint8_t* ip = base_ + HEADER_SIZE;
        for (uint32_t i = 0; i < num_samples_; ++i) {
            std::memcpy(&index_[i].offset, ip, 8);
            std::memcpy(&index_[i].nbytes, ip + 8, 4);
            std::memcpy(&index_[i].band, ip + 12, 4);
            ip += INDEX_ENTRY_SIZE;
        }
    }

    std::string path_;
    int fd_ = -1;
    const uint8_t* base_ = nullptr;
    size_t size_ = 0;
    uint32_t num_samples_ = 0;
    uint32_t max_seq_length_ = 0;
    std::vector<IndexEntry> index_;
};

} // namespace uds
