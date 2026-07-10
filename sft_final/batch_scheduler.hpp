// batch_scheduler.hpp
// Phase 8 — Profile-based batch scheduler with demand signaling.
//
// The scheduler samples one batch profile according to profile probabilities.
// When a requested band is empty it:
//
//   1. Marks the band as hungry.
//   2. Gives the streamer a few chances to refill it.
//   3. Falls back to the other band if necessary.
//
// Every hungry event and fallback is counted for benchmarking.

#pragma once

#include <array>
#include <atomic>
#include <random>
#include <vector>

#include "length_queues.hpp"
#include "sample.hpp"

namespace uds {
struct BatchProfile {
    std::array<int,2> bands = {BAND_SHORT, BAND_MEDIUM};
    std::array<float,2> mix = {0.5f, 0.5f};
    float prob = 1.f / 3.f;
    bool is_chunked = false;
};
struct SchedulerConfig {
    size_t B = 64;

    std::vector<BatchProfile> profiles;

    int pop_timeout_ms = 50;
    int hungry_retries = 2;

    uint64_t seed = 777;
};
class BatchScheduler {
public:

    BatchScheduler(
        SchedulerConfig cfg,
        LengthAwareQueues& queues)
        : cfg_(std::move(cfg)),
          queues_(queues),
          rng_(cfg_.seed)
    {
        if (cfg_.profiles.empty())
            install_default_profiles_();

        build_cdf_();
    }

    CandidatePool next_pool()
    {
        const int profile_index = pick_profile_();
        const BatchProfile& profile = cfg_.profiles[profile_index];

        CandidatePool pool;

        pool.profile_index = profile_index;
        pool.profile_bands = profile.bands;
        pool.is_chunked    = profile.is_chunked;
        pool.mixed         = !profile.is_chunked &&
                             profile.bands[0] != profile.bands[1];
        pool.band = static_cast<uint32_t>(profile.bands[0]);
        if (profile.is_chunked ||
            profile.bands[0] == profile.bands[1])
        {
            fill_band_(
                static_cast<uint32_t>(profile.bands[0]),
                cfg_.B,
                pool);

            return pool;
        }
        size_t target0 =
            static_cast<size_t>(profile.mix[0] * cfg_.B + 0.5f);

        if (target0 > cfg_.B)
            target0 = cfg_.B;

        size_t target1 = cfg_.B - target0;

        size_t got0 =
            fill_band_(
                static_cast<uint32_t>(profile.bands[0]),
                target0,
                pool);

        size_t got1 =
            fill_band_(
                static_cast<uint32_t>(profile.bands[1]),
                target1,
                pool);
        size_t remaining = cfg_.B - pool.samples.size();

        if (remaining > 0)
        {
            // Band 0 starved
            if (got0 < target0 && got1 == target1)
            {
                queues_.set_hungry(
                    static_cast<uint32_t>(profile.bands[0]));

                empty_alerts_[profile.bands[0]]
                    .fetch_add(1, std::memory_order_relaxed);

                if (fill_band_(
                        static_cast<uint32_t>(profile.bands[1]),
                        remaining,
                        pool) > 0)
                {
                    pool.fell_back = true;
                }
            }
            else if (got1 < target1)
            {
                queues_.set_hungry(
                    static_cast<uint32_t>(profile.bands[1]));

                empty_alerts_[profile.bands[1]]
                    .fetch_add(1, std::memory_order_relaxed);

                if (fill_band_(
                        static_cast<uint32_t>(profile.bands[0]),
                        remaining,
                        pool) > 0)
                {
                    pool.fell_back = true;
                }
            }
        }

        if (pool.fell_back)
        {
            fallback_pools_.fetch_add(
                1,
                std::memory_order_relaxed);
        }

        return pool;
    }
    uint64_t empty_alerts(uint32_t band) const
    {
        return empty_alerts_[band].load(std::memory_order_relaxed);
    }

    uint64_t fallback_pools() const
    {
        return fallback_pools_.load(std::memory_order_relaxed);
    }

private:
    void install_default_profiles_()
    {
        cfg_.profiles = {

            BatchProfile{
                {BAND_SHORT, BAND_MEDIUM},
                {0.5f,0.5f},
                1.f/3.f,
                false
            },

            BatchProfile{
                {BAND_MEDIUM, BAND_LONG},
                {0.5f,0.5f},
                1.f/3.f,
                false
            },

            BatchProfile{
                {BAND_CHUNKED, BAND_CHUNKED},
                {1.f,0.f},
                1.f/3.f,
                true
            }
        };
    }
    void build_cdf_()
    {
        cdf_.clear();

        float acc = 0.f;

        for (const auto& p : cfg_.profiles)
        {
            acc += p.prob;
            cdf_.push_back(acc);
        }

        total_prob_ = (acc > 0.f) ? acc : 1.f;
    }

    int pick_profile_()
    {
        std::uniform_real_distribution<float> dist(0.f, total_prob_);

        const float r = dist(rng_);

        for (size_t i = 0; i < cdf_.size(); ++i)
        {
            if (r <= cdf_[i])
                return static_cast<int>(i);
        }

        return static_cast<int>(cfg_.profiles.size() - 1);
    }
    size_t fill_band_(
        uint32_t band,
        size_t target,
        CandidatePool& pool)
    {
        size_t taken = 0;
        int retries = 0;

        while (taken < target)
        {
            auto sample = queues_.band(band).try_pop();

            if (!sample)
            {
                if (retries >= cfg_.hungry_retries)
                    break;

                ++retries;

                queues_.set_hungry(band);

                queues_.band(band).pop_wait(
                    cfg_.pop_timeout_ms);

                continue;
            }

            pool.samples.push_back(std::move(*sample));
            ++taken;
        }

        return taken;
    }

private:

    SchedulerConfig cfg_;

    LengthAwareQueues& queues_;

    std::mt19937_64 rng_;

    std::vector<float> cdf_;

    float total_prob_ = 1.f;

    std::array<std::atomic<uint64_t>, NUM_BANDS> empty_alerts_{};

    std::atomic<uint64_t> fallback_pools_{0};
};

} // namespace uds