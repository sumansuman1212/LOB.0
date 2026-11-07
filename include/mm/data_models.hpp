#pragma once

#include <cstdint>
#include <cassert>
#include <cmath>

#include "mm/types.hpp"   // VolRegime, Side, kObsDim

// ============================================================================
// data_models.hpp — Market-making data structures used across C++ components
//
// Implements: MarketFeatures, QuoteAction, QuoteParams, RewardSignal,
//             VolatilityState (VolRegime defined in types.hpp)
//
// Requirements: 2.1, 3.1, 5.1, 8.1, 8.2
// ============================================================================

namespace mm {

// ---------------------------------------------------------------------------
// MarketFeatures
//
// Fixed-length observation vector produced by FeaturePipeline after each LOB
// event. All fields are normalized to approximately [-1, 1] for stable neural
// network training.
//
// Requirement 2.1: 8-dimensional observation vector.
// ---------------------------------------------------------------------------
struct MarketFeatures {
    // Order-flow imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty)
    float ofi_l1;             ///< OFI at level-1 depth
    float ofi_l5;             ///< OFI over top-5 depth levels

    // Inventory state (normalized to [-1, 1])
    float inventory_skew;     ///< signed net_position / max_position
    float inventory_urgency;  ///< |net_position| / max_position, in [0, 1]

    // Volatility
    float vol_regime;         ///< z-scored rolling realized volatility

    // Micro-price relative to mid
    float micro_price_offset; ///< (ask*bid_qty + bid*ask_qty)/(bid_qty+ask_qty) - mid

    // Spread state
    float spread_ticks;       ///< current bid-ask spread in ticks

    // Session timing
    float session_progress;   ///< fraction of trading session elapsed, in [0, 1]

    /// Number of observation dimensions — matches PPO observation space shape.
    static constexpr int DIM = 8;

    /// Write all DIM floats into caller-supplied buffer (zero-copy, no allocation).
    /// @param buf  Must point to at least DIM contiguous floats.
    /// @return     Always returns 0.0f (unused; signature kept for C-ABI
    ///             compatibility where a return value may be expected).
    ///
    /// Requirement 2.11: zero-copy observe method without dynamic allocation.
    float to_array(float* buf) const noexcept {
        buf[0] = ofi_l1;
        buf[1] = ofi_l5;
        buf[2] = inventory_skew;
        buf[3] = inventory_urgency;
        buf[4] = vol_regime;
        buf[5] = micro_price_offset;
        buf[6] = spread_ticks;
        buf[7] = session_progress;
        return 0.0f;
    }
};

static_assert(MarketFeatures::DIM == 8,
              "MarketFeatures::DIM must equal the number of declared fields");

// ---------------------------------------------------------------------------
// QuoteAction
//
// Continuous action output from the PPO actor network. All three dimensions
// are produced by the same Gaussian head and subsequently clipped/scaled by
// QuoteGenerator before use.
//
// Requirement 3.1: action space (delta_bid, delta_ask, size_frac).
// ---------------------------------------------------------------------------
struct QuoteAction {
    float delta_bid;   ///< bid offset from mid in ticks (positive = further from mid)
    float delta_ask;   ///< ask offset from mid in ticks (positive = further from mid)
    float size_frac;   ///< quote size as fraction of max_order_size, in (0, 1]
};

// ---------------------------------------------------------------------------
// QuoteParams
//
// Concrete bid/ask limit-order parameters emitted by QuoteGenerator.
// Prices are in absolute integer ticks; quantities in lots.
//
// Requirement 5.1: output of QuoteGenerator::generate().
// ---------------------------------------------------------------------------
struct QuoteParams {
    int64_t  bid_price;      ///< absolute bid price in ticks
    int64_t  ask_price;      ///< absolute ask price in ticks
    uint64_t bid_qty;        ///< bid quantity in lots
    uint64_t ask_qty;        ///< ask quantity in lots
    int64_t  valid_until_ns; ///< TTL: nanoseconds since epoch at which quote expires
};

// ---------------------------------------------------------------------------
// RewardSignal
//
// Per-step reward decomposition used by LobEnv during PPO training.
// Separating components enables diagnostic logging and independent tuning.
//
// Sign invariants (Requirement 8.2):
//   - inventory_penalty <= 0   (penalises large positions)
//   - spread_capture    >= 0   (earned only on completed round-trips)
//   - realized_pnl      unconstrained (mark-to-market can go negative)
// ---------------------------------------------------------------------------
struct RewardSignal {
    float realized_pnl;      ///< mark-to-market change in ticks (can be negative)
    float inventory_penalty; ///< -γ · |net_position| / max_position  (always ≤ 0)
    float spread_capture;    ///< half-spread earned on fills          (always ≥ 0)

    /// Sum of all reward components.
    /// Asserts sign constraints at runtime in debug builds.
    ///
    /// Requirement 8.1: total = realized_pnl + inventory_penalty + spread_capture.
    /// Requirement 8.2: sign constraints on individual components.
    float total() const noexcept {
        // Sign-constraint assertions (stripped in release / NDEBUG builds).
        assert(inventory_penalty <= 0.0f &&
               "inventory_penalty must be non-positive");
        assert(spread_capture >= 0.0f &&
               "spread_capture must be non-negative");
        return realized_pnl + inventory_penalty + spread_capture;
    }
};

// ---------------------------------------------------------------------------
// VolRegime
//
// Categorical volatility regime label derived from z-scored realized vol.
// Used as a discrete conditioning input for policy decisions and logging.
//
// NOTE: VolRegime is defined in types.hpp (included transitively via
// lob_types.hpp). The duplicate definition previously here has been removed.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// VolatilityState
//
// Rolling realized-volatility snapshot maintained by FeaturePipeline.
// ---------------------------------------------------------------------------
struct VolatilityState {
    float     realized_vol_1min; ///< 1-minute rolling annualized realized vol
    float     realized_vol_5min; ///< 5-minute rolling annualized realized vol
    float     vol_z_score;       ///< z-score vs. 20-day baseline

    VolRegime regime;            ///< discrete regime label

    /// Map a z-score to a VolRegime using fixed thresholds.
    /// Thresholds match conventional vol-regime definitions used in practice:
    ///   z < -1  → Low
    ///   z in [-1, 1) → Normal
    ///   z in [1, 2)  → High
    ///   z >= 2  → Extreme
    static VolRegime classify(float z_score) noexcept {
        if (z_score >= 2.0f)  return VolRegime::Extreme;
        if (z_score >= 1.0f)  return VolRegime::High;
        if (z_score >= -1.0f) return VolRegime::Normal;
        return VolRegime::Low;
    }
};

} // namespace mm
