// data_pipeline.hpp
// Facade that wires Phases 5-11 into one object Python drives:
//   streamer -> length queues -> scheduler -> collator -> prefetcher.
// Python calls next_pool() and gets padded tensors; timing() exposes the
// C++-side benchmark counters (formation + stall).
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "batch_scheduler.hpp"
#include "collator.hpp"
#include "length_queues.hpp"
#include "prefetcher.hpp"
#include "shard_streamer.hpp"

namespace uds {

struct PipelineConfig {
    // data / distribution
    std::vector<std::string> shards;
    int rank = 0, world_size = 1;
    uint64_t seed = 1234;
    int num_epochs = -1;
    size_t max_queue_occupancy = 50000;

    // scheduler
    size_t B = 64;
    bool homogeneous = true;
    std::array<float, 3> fit_band_weights = {1.f, 1.f, 1.f};
    float chunked_rate = 0.1f;

    // collator
    int64_t pad_id = 0;
    int64_t ignore_index = -100;
    int64_t option_b_window = 2048;
    int64_t pad_to_multiple = 8;

    // prefetch
    int prefetch_workers = 2;
    size_t ring_capacity = 3;
};

class DataPipeline {
public:
    explicit DataPipeline(PipelineConfig cfg) : cfg_(std::move(cfg)) {
        queues_ = std::make_unique<LengthAwareQueues>();

        StreamerConfig scfg;
        scfg.shards = cfg_.shards;
        scfg.dist = {cfg_.rank, cfg_.world_size, cfg_.seed};
        scfg.max_queue_occupancy = cfg_.max_queue_occupancy;
        scfg.num_epochs = cfg_.num_epochs;
        scfg.shuffle_seed = cfg_.seed;
        streamer_ = std::make_unique<ShardStreamer>(std::move(scfg), *queues_);

        SchedulerConfig schcfg;
        schcfg.B = cfg_.B;
        schcfg.mode = cfg_.homogeneous ? PoolMode::Homogeneous : PoolMode::Mixed;
        schcfg.fit_band_weights = cfg_.fit_band_weights;
        schcfg.chunked_rate = cfg_.chunked_rate;
        schcfg.seed = cfg_.seed;
        scheduler_ = std::make_unique<BatchScheduler>(schcfg, *queues_);

        CollatorConfig ccfg;
        ccfg.pad_id = cfg_.pad_id;
        ccfg.ignore_index = cfg_.ignore_index;
        ccfg.option_b_window = cfg_.option_b_window;
        ccfg.pad_to_multiple = cfg_.pad_to_multiple;
        collator_ = std::make_unique<Collator>(ccfg);

        PrefetchConfig pcfg;
        pcfg.num_workers = cfg_.prefetch_workers;
        pcfg.ring_capacity = cfg_.ring_capacity;
        prefetcher_ = std::make_unique<Prefetcher>(pcfg, *scheduler_, *collator_);
    }

    ~DataPipeline() { stop(); }

    void start() { streamer_->start(); prefetcher_->start(); started_ = true; }

    void stop() {
        if (!started_) return;
        prefetcher_->stop();
        streamer_->stop();
        queues_->close_all();
        started_ = false;
    }

    std::shared_ptr<CollatedPool> next_pool() { return prefetcher_->next(); }

    // C++-side benchmark counters.
    double formation_total_s() const { return prefetcher_->formation_stat().total_s(); }
    double formation_mean_ms() const { return prefetcher_->formation_stat().mean_ms(); }
    uint64_t formation_count()  const { return prefetcher_->formation_stat().n(); }
    double stall_total_s()      const { return prefetcher_->stall_stat().total_s(); }
    double stall_mean_ms()      const { return prefetcher_->stall_stat().mean_ms(); }
    uint64_t samples_streamed() const { return streamer_->samples_streamed(); }

    size_t queue_size(uint32_t band) const { return queues_->size(band); }

private:
    PipelineConfig cfg_;
    std::unique_ptr<LengthAwareQueues> queues_;
    std::unique_ptr<ShardStreamer> streamer_;
    std::unique_ptr<BatchScheduler> scheduler_;
    std::unique_ptr<Collator> collator_;
    std::unique_ptr<Prefetcher> prefetcher_;
    bool started_ = false;
};

} // namespace uds
