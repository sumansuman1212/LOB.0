#pragma once

// include/mm/quote_generator.hpp
//
// QuoteGenerator — Translates policy action and alpha signals into concrete
// limit order parameters with hard inventory, spread, and quantity constraints.
//
// Requirements: 5.1–5.9, 9.5, 10.1, 12.1–12.6

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "mm/data_models.hpp"      // QuoteParams, QuoteAction
#include "mm/lob_types.hpp"        // BookSnapshot
#include "mm/policy_inference.hpp"  // IPolicyInference

namespace mm {

// ---------------------------------------------------------------------------
// IAlphaSignal — Hot-swappable plug-in interface (Req 12.5)
// ---------------------------------------------------------------------------
struct IAlphaSignal {
    virtual ~IAlphaSignal() = default;

    /// Unique 32-bit identifier for the alpha signal
    virtual uint32_t id() const noexcept = 0;

    /// Additive adjustment to (delta_bid, delta_ask) in ticks
    virtual std::pair<float, float> adjust(const BookSnapshot& snap) const noexcept = 0;

    /// Optional signal name for logging
    virtual const char* name() const noexcept { return "AlphaSignal"; }
};

// ---------------------------------------------------------------------------
// QuoteGenerator (Req 5.1–5.9)
// ---------------------------------------------------------------------------
class QuoteGenerator {
public:
    /// @param tick_size         Minimum price increment in ticks (must be > 0).
    /// @param max_order_size    Maximum order quantity allowed per side.
    /// @param max_inventory     Hard position limit for inventory halt rule.
    /// @param latency_budget_ns Maximum allowed budget for generate() (default 800 ns).
    QuoteGenerator(int64_t  tick_size,
                   uint64_t max_order_size,
                   int64_t  max_inventory,
                   int64_t  latency_budget_ns = 800) noexcept;

    // Movable, non-copyable
    QuoteGenerator(const QuoteGenerator&) = delete;
    QuoteGenerator& operator=(const QuoteGenerator&) = delete;
    QuoteGenerator(QuoteGenerator&&) noexcept = default;
    QuoteGenerator& operator=(QuoteGenerator&&) noexcept = default;

    /// Generate concrete bid/ask quote parameters on the hot path in < 800 ns (Req 5.1, 10.1).
    /// @param obs          Pointer to 8-dimensional observation float array.
    /// @param policy       Policy inference engine (Neural or Simplified).
    /// @param mid_price    Current mid-price in ticks.
    /// @param net_position Current signed net inventory position.
    /// @param snap         Optional current book snapshot for alpha signal evaluation.
    /// @param current_ts_ns Timestamp in nanoseconds for TTL calculation.
    QuoteParams generate(const float*      obs,
                         IPolicyInference& policy,
                         int64_t           mid_price,
                         int64_t           net_position,
                         const BookSnapshot* snap = nullptr,
                         int64_t           current_ts_ns = 0) noexcept;

    /// Attach a hot-swappable alpha signal atomically (Req 5.8, 12.2, 12.6).
    void attach_signal(std::shared_ptr<IAlphaSignal> signal) noexcept;

    /// Detach an alpha signal by ID atomically (Req 5.8, 12.3, 12.4).
    /// Detaching an unknown ID is a no-op.
    void detach_signal(uint32_t signal_id) noexcept;

    /// Accessors
    int64_t  tick_size()          const noexcept { return tick_size_; }
    uint64_t max_order_size()     const noexcept { return max_order_size_; }
    int64_t  max_inventory()      const noexcept { return max_inventory_; }
    int64_t  latency_budget_ns()  const noexcept { return latency_budget_ns_; }

private:
    int64_t  tick_size_;
    uint64_t max_order_size_;
    int64_t  max_inventory_;
    int64_t  latency_budget_ns_;

    // Atomic signal list swap (hazard pointer / shared_ptr swap pattern for lock-free read on hot path)
    std::shared_ptr<const std::vector<std::shared_ptr<IAlphaSignal>>> signals_;
    mutable std::atomic_flag signal_lock_ = ATOMIC_FLAG_INIT;
};

} // namespace mm
