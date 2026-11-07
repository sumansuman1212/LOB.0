// src/cpp/feature_pipeline.cpp
//
// FeaturePipeline implementation.
//
// All arithmetic is float32 for the final observation vector but uses double
// internally for the rolling accumulators to avoid catastrophic cancellation
// in sum-of-squares updates.
//
// Requirements: 2.1–2.13, 10.4, 11.4

#include "mm/feature_pipeline.hpp"

#include <algorithm>   // std::clamp, std::min
#include <cmath>       // std::log, std::sqrt, std::isfinite, std::isinf, std::isnan
#include <cstring>     // std::memset
#include <limits>

namespace mm {

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
static constexpr float  kSigmaClamp     = 5.0f;
static constexpr double kEpsilon        = 1e-10;   // division-by-zero guard
static constexpr int    kMaxLobLevels   = 10;      // kMaxDepth from types.hpp

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
FeaturePipeline::FeaturePipeline(int32_t max_position,
                                  int32_t vol_window,
                                  int32_t ofi_levels,
                                  int64_t session_duration_s) noexcept
    : max_position_      (max_position > 0 ? max_position : 1)
    , vol_window_        (vol_window > 1 ? vol_window : 2)
    , ofi_levels_        (std::min(std::max(ofi_levels, 1), kMaxLobLevels))
    , session_duration_ns_(session_duration_s * 1'000'000'000LL)
    , vol_buf_           (static_cast<size_t>(vol_window > 1 ? vol_window : 2), 0.0f)
    , vol_head_          (0)
    , vol_count_         (0)
    , vol_sum_           (0.0)
    , vol_sum_sq_        (0.0)
    , vol_mean_baseline_ (0.0)
    , vol_std_baseline_  (1.0)   // avoid divide-by-zero before baseline is set
    , last_mid_price_    (0)
    , session_start_ns_  (0)
    , last_update_ns_    (std::numeric_limits<int64_t>::min())
    , current_           {}
{}

// ---------------------------------------------------------------------------
// reset()  — Req 2.13
// ---------------------------------------------------------------------------
void FeaturePipeline::reset() noexcept {
    std::fill(vol_buf_.begin(), vol_buf_.end(), 0.0f);
    vol_head_          = 0;
    vol_count_         = 0;
    vol_sum_           = 0.0;
    vol_sum_sq_        = 0.0;
    vol_mean_baseline_ = 0.0;
    vol_std_baseline_  = 1.0;
    last_mid_price_    = 0;
    session_start_ns_  = 0;
    last_update_ns_    = std::numeric_limits<int64_t>::min();
    current_           = MarketFeatures{};
}

// ---------------------------------------------------------------------------
// sanitize()  — NaN/Inf guard + 5-sigma clamp  (Req 2.10, 11.4)
// ---------------------------------------------------------------------------
/*static*/
float FeaturePipeline::sanitize(float v) noexcept {
    if (!std::isfinite(v)) return 0.0f;
    if (v >  kSigmaClamp) return  kSigmaClamp;
    if (v < -kSigmaClamp) return -kSigmaClamp;
    return v;
}

// ---------------------------------------------------------------------------
// online_variance()
//
// Sample variance computed from accumulated sum / sum-of-squares:
// Var = (sum_sq - sum^2 / n) / (n - 1)
// Returns 0 for n < 2 or negative result (numerical noise).
// ---------------------------------------------------------------------------
/*static*/
double FeaturePipeline::online_variance(double sum, double sum_sq, int32_t n) noexcept {
    if (n < 2) return 0.0;
    double mean = sum / static_cast<double>(n);
    double var  = (sum_sq / static_cast<double>(n)) - (mean * mean);
    return var > 0.0 ? var : 0.0;
}

// ---------------------------------------------------------------------------
// push_return()
//
// Push a new log-return into the circular buffer and update O(1) accumulators.
// When the buffer is full the oldest entry is subtracted before the new one is
// added — this keeps sum/sum_sq O(1) regardless of window size.
// ---------------------------------------------------------------------------
void FeaturePipeline::push_return(float r) noexcept {
    // If buffer is full, subtract the oldest entry that will be overwritten.
    if (vol_count_ == vol_window_) {
        float old_r = vol_buf_[static_cast<size_t>(vol_head_)];
        vol_sum_    -= static_cast<double>(old_r);
        vol_sum_sq_ -= static_cast<double>(old_r) * static_cast<double>(old_r);
    }

    // Write the new return at the head position.
    vol_buf_[static_cast<size_t>(vol_head_)] = r;
    vol_sum_    += static_cast<double>(r);
    vol_sum_sq_ += static_cast<double>(r) * static_cast<double>(r);

    // Advance circular-buffer head.
    vol_head_ = (vol_head_ + 1) % vol_window_;

    // Saturate count at vol_window_.
    if (vol_count_ < vol_window_) ++vol_count_;

    // Update baseline stats when the window first fills (or each refill).
    // This provides a stable denominator for z-scoring even early in a session.
    if (vol_count_ == vol_window_) {
        double var = online_variance(vol_sum_, vol_sum_sq_, vol_count_);
        vol_std_baseline_  = (var > kEpsilon) ? std::sqrt(var) : 1.0;
        vol_mean_baseline_ = vol_sum_ / static_cast<double>(vol_count_);
    }
}

// ---------------------------------------------------------------------------
// compute_vol_regime()
//
// Realized vol = sqrt(mean of squared returns) over the current window.
// z-score = (realized_vol - baseline_mean_vol) / baseline_std_vol
//
// Both the baseline mean and std are derived from the same accumulator
// window, giving a rolling z-score that naturally converges to 0-mean, 1-std
// once the window is full.
// ---------------------------------------------------------------------------
float FeaturePipeline::compute_vol_regime() const noexcept {
    if (vol_count_ < 2) return 0.0f;

    // Realized vol = sqrt(E[r^2])  (standard HFT definition, no mean subtraction)
    double mean_sq = vol_sum_sq_ / static_cast<double>(vol_count_);
    if (mean_sq < 0.0) mean_sq = 0.0;
    double realized_vol = std::sqrt(mean_sq);

    // z-score against running baseline
    double std_b = (vol_std_baseline_ > kEpsilon) ? vol_std_baseline_ : 1.0;
    double z     = (realized_vol - vol_mean_baseline_) / std_b;

    return sanitize(static_cast<float>(z));
}

// ---------------------------------------------------------------------------
// update()  — main hot-path method  (Req 2.1, 10.4)
//
// Compute all 8 features and store in current_.
// All divisions are guarded; NaN/Inf → 0; result clamped to [-5, 5].
// ---------------------------------------------------------------------------
void FeaturePipeline::update(const BookSnapshot& snap,
                              int64_t net_position) noexcept {
    // ------------------------------------------------------------------
    // Session progress  (Req 2.8)
    //
    // Use wall-clock timestamps from the snapshot.
    // On first call, record session_start_ns_.
    // ------------------------------------------------------------------
    int64_t ts = snap.timestamp_ns;

    if (session_start_ns_ == 0) {
        session_start_ns_ = ts;
    }
    last_update_ns_ = ts;

    float session_progress = 0.0f;
    if (session_duration_ns_ > 0) {
        int64_t elapsed = ts - session_start_ns_;
        if (elapsed < 0) elapsed = 0;
        double progress = static_cast<double>(elapsed) /
                          static_cast<double>(session_duration_ns_);
        // Clamp to [0, 1] before the 5-sigma clamp so it stays in a natural range.
        if (progress > 1.0) progress = 1.0;
        session_progress = static_cast<float>(progress);
    }

    // ------------------------------------------------------------------
    // OFI level-1  (Req 2.2)
    //
    // ofi_l1 = (bid_qty_l1 - ask_qty_l1) / (bid_qty_l1 + ask_qty_l1)
    // ------------------------------------------------------------------
    double bid_l1 = static_cast<double>(snap.bids[0].total_qty);
    double ask_l1 = static_cast<double>(snap.asks[0].total_qty);
    double ofi_l1_denom = bid_l1 + ask_l1;
    float  ofi_l1 = (ofi_l1_denom > kEpsilon)
                        ? static_cast<float>((bid_l1 - ask_l1) / ofi_l1_denom)
                        : 0.0f;

    // ------------------------------------------------------------------
    // OFI level-5  (Req 2.2)
    //
    // ofi_l5 = (sum bid_qty[0..K-1] - sum ask_qty[0..K-1])
    //        / (sum bid_qty[0..K-1] + sum ask_qty[0..K-1])
    // ------------------------------------------------------------------
    double bid_l5_sum = 0.0;
    double ask_l5_sum = 0.0;
    int32_t levels_to_use = std::min(ofi_levels_, kMaxLobLevels);
    for (int i = 0; i < levels_to_use; ++i) {
        bid_l5_sum += static_cast<double>(snap.bids[i].total_qty);
        ask_l5_sum += static_cast<double>(snap.asks[i].total_qty);
    }
    double ofi_l5_denom = bid_l5_sum + ask_l5_sum;
    float  ofi_l5 = (ofi_l5_denom > kEpsilon)
                        ? static_cast<float>((bid_l5_sum - ask_l5_sum) / ofi_l5_denom)
                        : 0.0f;

    // ------------------------------------------------------------------
    // Inventory  (Req 2.3, 2.4)
    //
    // inventory_skew    = net_position / max_position        in [-1, 1]
    // inventory_urgency = |net_position| / max_position      in [0, 1]
    // ------------------------------------------------------------------
    double pos_norm = static_cast<double>(net_position) /
                      static_cast<double>(max_position_);
    float inventory_skew    = static_cast<float>(pos_norm);
    float inventory_urgency = static_cast<float>(pos_norm < 0.0 ? -pos_norm : pos_norm);

    // Clamp to [-1, 1] / [0, 1] before the final 5-sigma clamp.
    inventory_skew    = (inventory_skew >  1.0f) ?  1.0f : (inventory_skew < -1.0f) ? -1.0f : inventory_skew;
    inventory_urgency = (inventory_urgency > 1.0f) ? 1.0f : (inventory_urgency < 0.0f) ? 0.0f : inventory_urgency;

    // ------------------------------------------------------------------
    // Volatility regime  (Req 2.5)
    //
    // Push a new log-return (log mid / prev_mid) into the circular buffer
    // and compute the z-scored realized vol.
    // ------------------------------------------------------------------
    int64_t mid_price_ticks = snap.best_bid > 0 && snap.best_ask > 0
                                ? (snap.best_bid + snap.best_ask) / 2
                                : 0;

    if (last_mid_price_ > 0 && mid_price_ticks > 0) {
        double log_ret = std::log(static_cast<double>(mid_price_ticks) /
                                  static_cast<double>(last_mid_price_));
        if (std::isfinite(log_ret)) {
            push_return(static_cast<float>(log_ret));
        }
    }
    if (mid_price_ticks > 0) last_mid_price_ = mid_price_ticks;

    float vol_regime = compute_vol_regime();

    // ------------------------------------------------------------------
    // Micro-price offset  (Req 2.6)
    //
    // micro_price = (ask_price * bid_qty_l1 + bid_price * ask_qty_l1)
    //             / (bid_qty_l1 + ask_qty_l1)
    //
    // micro_price_offset = micro_price - mid_price
    // (in raw tick units; normalized by spreading over tick_size=1 for now)
    // ------------------------------------------------------------------
    float micro_price_offset = 0.0f;
    {
        double bid_p = static_cast<double>(snap.best_bid);
        double ask_p = static_cast<double>(snap.best_ask);
        double denom = bid_l1 + ask_l1;  // same as ofi_l1_denom
        if (denom > kEpsilon && snap.best_bid > 0 && snap.best_ask > 0) {
            double micro_price = (ask_p * bid_l1 + bid_p * ask_l1) / denom;
            double mid_p       = (bid_p + ask_p) * 0.5;
            double raw_offset  = micro_price - mid_p;
            // Normalize by half-spread so the value is dimensionless ~[-1, 1].
            double half_spread = (ask_p - bid_p) * 0.5;
            double normalized  = (half_spread > kEpsilon)
                                     ? raw_offset / half_spread
                                     : raw_offset;
            micro_price_offset = static_cast<float>(normalized);
        }
    }

    // ------------------------------------------------------------------
    // Spread in ticks  (Req 2.7)
    //
    // spread_ticks = best_ask - best_bid  (integer tick distance)
    // Normalize by dividing by a typical spread reference.
    // We z-score against the running baseline derived from the vol window,
    // or simply log-scale it for a compact feature.
    // Using log1p(spread_ticks - 1) keeps it near 0 for 1-tick spreads
    // and grows slowly.  Then we clamp with sanitize().
    // ------------------------------------------------------------------
    float spread_ticks = 0.0f;
    if (snap.best_ask > 0 && snap.best_bid > 0) {
        int64_t raw_spread = snap.best_ask - snap.best_bid;
        if (raw_spread > 0) {
            // log1p(raw_spread - 1): spread=1 → 0.0, spread=2 → ~0.69, etc.
            spread_ticks = static_cast<float>(std::log1p(
                static_cast<double>(raw_spread) - 1.0));
        }
    }

    // ------------------------------------------------------------------
    // Assemble and store — apply NaN guard + 5-sigma clamp to every field
    // (Req 2.9, 2.10)
    // ------------------------------------------------------------------
    current_.ofi_l1             = sanitize(ofi_l1);
    current_.ofi_l5             = sanitize(ofi_l5);
    current_.inventory_skew     = sanitize(inventory_skew);
    current_.inventory_urgency  = sanitize(inventory_urgency);
    current_.vol_regime         = sanitize(vol_regime);
    current_.micro_price_offset = sanitize(micro_price_offset);
    current_.spread_ticks       = sanitize(spread_ticks);
    current_.session_progress   = sanitize(session_progress);
}

// ---------------------------------------------------------------------------
// observe()  — zero-copy  (Req 2.11)
// ---------------------------------------------------------------------------
void FeaturePipeline::observe(MarketFeatures& out) const noexcept {
    out = current_;
}

} // namespace mm
