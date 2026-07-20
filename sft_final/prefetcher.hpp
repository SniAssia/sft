// Phase 9 — while the GPU trains on pool k, background threads build pool k+1
// (scheduler -> collator) and park it in a bounded ring. The consumer's
// next_pool() then pops instantly; any time it *does* wait is recorded as the
// STALL (formation not hidden), which is the number the benchmark cares about.
//
// TERMINATION FIX:
//   The epoch used to end only when streamer_.done() && all_categories_dry().
//   all_categories_dry() depends on per-band `exhausted` flags, which are a
//   refill hint and can go stale at end of stream — leaving one band empty but
//   "not dry" forever. The worker then never sets finished_, next() never
//   returns nullptr, and the whole thing livelocks (skipped_ climbs, no pool).
//   We now:
//     1. tell the scheduler the stream is done (so empty == dry at drain), and
//     2. gate epoch-end on the authoritative signal: stream done AND every queue
//        actually empty (queues_total() == 0), which no stale flag can fake.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "shard_streamer.hpp"
#include "batch_scheduler.hpp"
#include "collator.hpp"
#include "timer.hpp"

namespace uds {

struct PrefetchConfig {
    int num_workers = 2;
    size_t ring_capacity = 3;   // >=2 to overlap; depth of the double/triple buffer
};

class Prefetcher {
public:
    Prefetcher(PrefetchConfig cfg, BatchScheduler& sched, Collator& collator, ShardStreamer& streamer)
        : cfg_(cfg), sched_(sched), collator_(collator), streamer_(streamer) {}

    ~Prefetcher() { stop(); }

    void start() {
        running_ = true;
        finished_ = false;
        for (int i = 0; i < cfg_.num_workers; ++i)
            workers_.emplace_back([this] { worker_loop_(); });
    }

    void stop() {
        running_ = false;
        cv_not_full_.notify_all();
        cv_not_empty_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
        workers_.clear();
    }

    // Consumer side. Blocks until a pool is ready, the pipeline is stopped, or
    // the epoch is genuinely finished (streamer done + all queues drained).
    // Returns nullptr in the latter two cases — never blocks forever.
    std::shared_ptr<CollatedPool> next() {
        double t0 = now_s();
        std::unique_lock<std::mutex> lk(m_);
        while (true) {
            if (!ring_.empty()) {
                auto p = ring_.front();
                ring_.pop_front();
                lk.unlock();
                cv_not_full_.notify_one();
                stall_stat_.add(now_s() - t0);
                return p;
            }
            if (!running_ || finished_.load()) {
                return nullptr;
            }
            // timed wait — wakes on notify, or every 50ms to re-check the loop
            // conditions above (covers the case where worker_loop_ sets
            // finished_ and notifies right as we're about to sleep)
            cv_not_empty_.wait_for(lk, std::chrono::milliseconds(50));
        }
    }

    bool finished() const { return finished_.load(); }

    const TimeStat& formation_stat() const { return formation_stat_; }
    const TimeStat& stall_stat() const { return stall_stat_; }

private:
    void worker_loop_() {
        int empty_streak = 0;
        while (running_) {
            // Build (formation = scheduler pop + collate).
            double t0 = now_s();
            CandidatePool cp = sched_.next_pool();
            if (cp.samples.empty()) {
                const bool stream_done = streamer_.done();

                // Once the stream is finished there is no "more coming": let the
                // scheduler treat empty bands as dry so the tail drains 100% from
                // whatever remains instead of endlessly re-checking the mix.
                if (stream_done) sched_.notify_stream_done();

                // Authoritative end-of-epoch test: stream finished AND nothing is
                // left in any queue. queues_total()==0 cannot be faked by a stale
                // per-band exhausted flag, so this always terminates. The streak
                // is a short confirmation window to avoid a false trip during a
                // resident-window swap (queues momentarily empty mid-stream).
                if (stream_done && sched_.queues_total() == 0) {
                    if (++empty_streak >= 25) {            // ~50ms confirmation
                        finished_.store(true);
                        cv_not_empty_.notify_all();        // wake next() -> nullptr
                        return;                            // this worker exits; others follow
                    }
                } else {
                    empty_streak = 0;                      // still mid-stream, reset
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            empty_streak = 0;
            auto collated = std::make_shared<CollatedPool>(collator_(cp));
            collated->formation_seconds = now_s() - t0;
            formation_stat_.add(collated->formation_seconds);

            // Park in ring (backpressure when GPU is the bottleneck).
            std::unique_lock<std::mutex> lk(m_);
            cv_not_full_.wait(lk, [&] { return ring_.size() < cfg_.ring_capacity || !running_; });
            if (!running_) return;
            ring_.push_back(std::move(collated));
            lk.unlock();
            cv_not_empty_.notify_one();
        }
    }

    PrefetchConfig cfg_;
    BatchScheduler& sched_;
    Collator& collator_;
    ShardStreamer& streamer_;

    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};
    std::atomic<bool> finished_{false};

    std::mutex m_;
    std::condition_variable cv_not_full_, cv_not_empty_;
    std::deque<std::shared_ptr<CollatedPool>> ring_;

    TimeStat formation_stat_;  // raw build cost (overlapped with training)
    TimeStat stall_stat_;      // consumer wait (formation NOT hidden)
};

} // namespace uds