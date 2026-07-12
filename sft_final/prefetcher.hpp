// prefetcher.hpp
// Phase 9 — while the GPU trains on pool k, background threads build pool k+1
// (scheduler -> collator) and park it in a bounded ring. The consumer's
// next_pool() then pops instantly; any time it *does* wait is recorded as the
// STALL (formation not hidden), which is the number the benchmark cares about.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

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
    Prefetcher(PrefetchConfig cfg, BatchScheduler& sched, Collator& collator)
        : cfg_(cfg), sched_(sched), collator_(collator) {}

    ~Prefetcher() { stop(); }

    void start() {
        running_ = true;
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
    // Consumer side. Returns nullptr on timeout when epoch is done.
    std::shared_ptr<CollatedPool> next() {
        double t0 = now_s();
        std::unique_lock<std::mutex> lk(m_);
        // Use wait_for so we can return nullptr when the streamer is done + ring empty
        while (true) {
            if (!ring_.empty()) {
                auto p = ring_.front();
                ring_.pop_front();
                lk.unlock();
                cv_not_full_.notify_one();
                stall_stat_.add(now_s() - t0);
                return p;
            }
            if (!running_) {
                return nullptr;
            }
            // timed wait — lets the caller check streamer_done() periodically
            cv_not_empty_.wait_for(lk, std::chrono::milliseconds(50));
        }
    }

    const TimeStat& formation_stat() const { return formation_stat_; }
    const TimeStat& stall_stat() const { return stall_stat_; }

private:
    void worker_loop_() {
        while (running_) {
            // Build (formation = scheduler pop + collate).
            double t0 = now_s();
            CandidatePool cp = sched_.next_pool();
            if (cp.samples.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
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

    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};

    std::mutex m_;
    std::condition_variable cv_not_full_, cv_not_empty_;
    std::deque<std::shared_ptr<CollatedPool>> ring_;

    TimeStat formation_stat_;  // raw build cost (overlapped with training)
    TimeStat stall_stat_;      // consumer wait (formation NOT hidden)
};

} // namespace uds
