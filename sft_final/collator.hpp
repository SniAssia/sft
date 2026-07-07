// collator.hpp
// Turns a CandidatePool into padded libtorch tensors. This is the ONLY core
// component that depends on libtorch (per the design: libtorch for tensors only).
//
// Fit pools  -> [P, T] input_ids/attention_mask/labels, right-padded to the
//               pool's max length. Loss mask = response tokens only (prompt +
//               context + pad => ignore_index).
// Chunked pools (Option B) -> a REPRESENTATIVE WINDOW that fits: Instruction +
//               start of Context (long-ctx) or Instruction + start of Response
//               (long-resp). Scored no_grad in Python; full SeCO training on the
//               selected few is Phase 13 (deferred).
#pragma once
#include <algorithm>
#include <vector>
#include <torch/torch.h>

#include "sample.hpp"
#include "shard_format.hpp"

namespace uds {

struct CollatorConfig {
    int64_t pad_id = 0;
    int64_t ignore_index = -100;
    int64_t option_b_window = 2048;   // budget for the chunked scoring window
    int64_t pad_to_multiple = 8;      // pad T up to a multiple (tensor-core friendly)
};

// The tensor bundle handed to Python. torch::Tensor auto-converts to torch.Tensor.
struct CollatedPool {
    torch::Tensor input_ids;       // [P, T] long
    torch::Tensor attention_mask;  // [P, T] long (1 real, 0 pad)
    torch::Tensor labels;          // [P, T] long (targets; ignore_index elsewhere)
    bool is_chunked = false;
    int64_t band = 0;
    // per-sample true zone lengths (pre-pad), for UDS bookkeeping / SeCO later
    std::vector<int64_t> prompt_len, context_len, response_len;
    std::vector<int64_t> is_chunked_flags, case_codes;
    double formation_seconds = 0.0;  // filled by the prefetcher
};

class Collator {
public:
    explicit Collator(CollatorConfig cfg) : cfg_(cfg) {}

    CollatedPool operator()(const CandidatePool& pool) const {
        return pool.is_chunked ? collate_chunked_(pool) : collate_fit_(pool);
    }

private:
    static int64_t round_up_(int64_t x, int64_t m) {
        return (m <= 1) ? x : ((x + m - 1) / m) * m;
    }

    CollatedPool collate_fit_(const CandidatePool& pool) const {
        const int64_t P = static_cast<int64_t>(pool.samples.size());
        int64_t maxT = 1;
        for (const auto& s : pool.samples) maxT = std::max<int64_t>(maxT, s.total_len());
        const int64_t T = round_up_(maxT, cfg_.pad_to_multiple);

        auto opts = torch::TensorOptions().dtype(torch::kInt64);
        CollatedPool out;
        out.is_chunked = false;
        out.band = pool.band;
        out.input_ids      = torch::full({P, T}, cfg_.pad_id, opts);
        out.attention_mask = torch::zeros({P, T}, opts);
        out.labels         = torch::full({P, T}, cfg_.ignore_index, opts);

        auto in_a  = out.input_ids.accessor<int64_t, 2>();
        auto am_a  = out.attention_mask.accessor<int64_t, 2>();
        auto lb_a  = out.labels.accessor<int64_t, 2>();

        for (int64_t i = 0; i < P; ++i) {
            const Sample& s = pool.samples[i];
            int64_t t = 0;
            auto emit = [&](const std::vector<int32_t>& ids, bool is_target) {
                for (int32_t id : ids) {
                    in_a[i][t] = id;
                    am_a[i][t] = 1;
                    if (is_target) lb_a[i][t] = id;   // loss on response tokens only
                    ++t;
                }
            };
            emit(s.prompt_ids, false);
            emit(s.context_ids, false);
            emit(s.response_ids, true);
            record_lengths_(out, s);
        }
        return out;
    }

    // Option-B: build a fit window per chunked sample for RANKING ONLY.
    CollatedPool collate_chunked_(const CandidatePool& pool) const {
        const int64_t P = static_cast<int64_t>(pool.samples.size());
        const int64_t W = round_up_(cfg_.option_b_window, cfg_.pad_to_multiple);

        auto opts = torch::TensorOptions().dtype(torch::kInt64);
        CollatedPool out;
        out.is_chunked = true;
        out.band = BAND_CHUNKED;
        out.input_ids      = torch::full({P, W}, cfg_.pad_id, opts);
        out.attention_mask = torch::zeros({P, W}, opts);
        out.labels         = torch::full({P, W}, cfg_.ignore_index, opts);

        auto in_a = out.input_ids.accessor<int64_t, 2>();
        auto am_a = out.attention_mask.accessor<int64_t, 2>();
        auto lb_a = out.labels.accessor<int64_t, 2>();

        for (int64_t i = 0; i < P; ++i) {
            const Sample& s = pool.samples[i];
            std::vector<int32_t> win;
            std::vector<char> is_tgt;
            build_window_(s, cfg_.option_b_window, win, is_tgt);
            int64_t t = 0;
            for (size_t k = 0; k < win.size() && t < W; ++k, ++t) {
                in_a[i][t] = win[k];
                am_a[i][t] = 1;
                if (is_tgt[k]) lb_a[i][t] = win[k];
            }
            record_lengths_(out, s);
        }
        return out;
    }

    // Representative window: always keep the full Instruction; then fill the
    // remaining budget with the start of Context (long-ctx) or start of Response
    // (long-resp / mixed). Consistent fit-window => valid relative ranking.
    void build_window_(const Sample& s, int64_t budget,
                       std::vector<int32_t>& win, std::vector<char>& is_tgt) const {
        for (int32_t id : s.prompt_ids) { win.push_back(id); is_tgt.push_back(0); }

        auto fill_from = [&](const std::vector<int32_t>& ids, char tgt) {
            for (int32_t id : ids) {
                if (static_cast<int64_t>(win.size()) >= budget) break;
                win.push_back(id); is_tgt.push_back(tgt);
            }
        };

        if (s.case_code == CASE_LONG_CTX) {
            fill_from(s.context_ids, 0);
            fill_from(s.response_ids, 1);   // include a little response if room
        } else { // LONG_RESP or MIXED: prioritize response continuation
            fill_from(s.context_ids, 0);
            fill_from(s.response_ids, 1);
        }
    }

    static void record_lengths_(CollatedPool& out, const Sample& s) {
        out.prompt_len.push_back(s.prompt_len());
        out.context_len.push_back(s.context_len());
        out.response_len.push_back(s.response_len());
        out.is_chunked_flags.push_back(s.is_chunked);
        out.case_codes.push_back(s.case_code);
    }

    CollatorConfig cfg_;
};

} // namespace uds
