#pragma once

// include/mm/feature_pipeline.hpp
//
// FeaturePipeline — converts raw BookSnapshot + net position into a fixed-length
// MarketFeatures observation vector consumed by the PPO agent.
//
// Design goals:
//   - O(1) rolling accumulators for realized volatility (circular buffer of
//     mid-price returns, running sum + sum-of-squares)  (Req 2.12)
//   - NaN/Inf guard on all normalization divisions         (Req 2.9, 11.4)
//   - 5-sigma clamp: |obs[i]| <= 5.0                      (Req 2.10)
//   - Zero-copy observe() into caller-supplied buffer      (Req 2.11)
//   - reset() clears all window state                      (Req 2.13)
//
// Requirements: 2.1–2.13, 10.4, 11.4

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

#include "mm/lob_types.hpp"    // BookSnapshot
#include "mm/data_models.hpp"  // MarketFeatures

namespace mm {

// ---------------------------------------------------------------------------
// FeaturePipeline
// ---------------------------------------------------------------------------
class FeaturePipeline {
public:
    /// @param max_position  Hard position limit used to normalize inventory fields.
    ///                      Must be > 0.
    /// @param vol_window    Rolling window size for realized-volatility accumulator.
    ///                      Default 60 (typically 60 seconds of 1-per-second ticks).
    /// @param ofi_levels    Number of top levels used for ofi_l5 computation (≤ 10).
    ///                      Default 5.
    /// @param session_duration_s  Nominal trading session length in seconds.
    ///                      Default 23400 = 6.5 hours (NSE equities session).
    explicit FeaturePipeline(int32_t max_position,
                              int32_t vol_window        = 60,
                              int32_t ofi_levels        = 5,
                              int64_t session_duration_s = 23400) noexcept;

    // Not copyable — large circular buffer; move is fine.
    FeaturePipeline(const FeaturePipeline&)            = delete;
    FeaturePipeline& operator=(const FeaturePipeline&) = delete;
    FeaturePipeline(FeaturePipeline&&)                 = default;
    FeaturePipeline& operator=(FeaturePipeline&&)      = default;

    // -----------------------------------------------------------------------
    // Core hot-path methods
    // -----------------------------------------------------------------------

    /// Update all rolling accumulators from a new book snapshot + position.
    ///
    /// Preconditions (soft — violations produce 0-filled features, not UB):
    ///   - snap.best_bid < snap.best_ask (valid, non-crossed book)
    ///   - |net_position| <= max_position_
    ///   - snap.timestamp_ns > last_update_ns_ (strictly increasing)
    ///
    /// Requirement 2.1, 10.4
    void update(const BookSnapshot& snap, int64_t net_position) noexcept;

    /// Write current MarketFeatures into caller-supplied buffer (zero-copy).
    ///
    /// Thread-safety: safe for one writer (update) + one reader (observe) if the
    /// caller ensures no concurrent update() during observe().
    ///
    /// Requirement 2.11
    void observe(MarketFeatures& out) const noexcept;

    /// Clear all rolling window state and session tracking.
    ///
    /// Requirement 2.13
    void reset() noexcept;

private:
    // -----------------------------------------------------------------------
    // Configuration (set at construction, immutable thereafter)
    // -----------------------------------------------------------------------
    int32_t max_position_;          ///< normalization denominator for inventory
    int32_t vol_window_;            ///< size of realized-vol circular buffer
    int32_t ofi_levels_;            ///< top-K levels for ofi_l5 (clamped to ≤10)
    int64_t session_duration_ns_;   ///< session length in nanoseconds

    // -----------------------------------------------------------------------
    // Realized-volatility rolling window  (Req 2.12)
    //
    // Stores log mid-price returns: r_t = log(mid_t / mid_{t-1}).
    // Uses O(1) accumulators: running sum and sum-of-squares.
    // Circular buffer: vol_buf_[vol_head_] is always the slot being overwritten.
    // -----------------------------------------------------------------------
    std::vector<float> vol_buf_;    ///< circular buffer of log-returns, size vol_window_
    int32_t   vol_head_;            ///< index of the oldest / next-write slot
    int32_t   vol_count_;           ///< number of valid entries (saturates at vol_window_)
    double    vol_sum_;             ///< running sum of returns
    double    vol_sum_sq_;          ///< running sum of squared returns
    double    vol_mean_baseline_;   ///< long-run mean (updated each window fill)
    double    vol_std_baseline_;    ///< long-run std  (updated each window fill)
    int64_t   last_mid_price_;      ///< previous mid-price for return computation

    // -----------------------------------------------------------------------
    // Session progress tracking  (Req 2.8)
    // -----------------------------------------------------------------------
    int64_t   session_start_ns_;    ///< timestamp of first update() call (0 = unset)
    int64_t   last_update_ns_;      ///< timestamp of last update() call

    // -----------------------------------------------------------------------
    // Current feature snapshot (written by update, read by observe)
    // -----------------------------------------------------------------------
    MarketFeatures current_;

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Clamp a float to [-5.0, 5.0] and replace NaN/Inf with 0.  (Req 2.10, 11.4)
    static float sanitize(float v) noexcept;

    /// Compute O(1) variance from sum and sum-of-squares over n samples.
    /// Returns 0 if n < 2 or variance would be negative (numerical noise).
    static double online_variance(double sum, double sum_sq, int32_t n) noexcept;

    /// Update the volatility circular buffer with a new log-return.
    void push_return(float r) noexcept;

    /// Compute the z-scored realized volatility from current accumulator state.
    float compute_vol_regime() const noexcept;
};

} // namespace mm
