#pragma once

// include/mm/lob_engine.hpp
//
// LobEngine — price-time priority limit order book for a single instrument.
//
// Design choices:
//   - std::map<int64_t, PriceLevel> for bid side (std::greater → highest price first)
//   - std::map<int64_t, PriceLevel> for ask side (std::less   → lowest  price first)
//   - std::unordered_map<uint64_t, OrderEntry> for O(1) order_id → {price,qty,side}
//   - Sentinel values: best_bid() returns INT64_MIN when bid side empty,
//                      best_ask() returns INT64_MAX when ask side empty
//
// Requirement coverage:
//   1.1–1.13  LOB management correctness
//   9.1       Reject crossed-book events with warning log
//   11.1      Reject price <= 0
//   11.2      Reject Add/Fill with qty == 0
//   11.3      Reject events with timestamp_ns < last processed timestamp

#include <cstdint>
#include <limits>
#include <map>
#include <unordered_map>
#include <functional>   // std::greater

#include "lob_types.hpp"  // PriceLevel, OrderEvent, BookSnapshot
#include "types.hpp"      // Side, kMaxDepth

namespace mm {

// ---------------------------------------------------------------------------
// OrderEntry — internal per-order state kept for O(1) Cancel/Modify/Fill
// ---------------------------------------------------------------------------
struct OrderEntry {
    int64_t  price;   ///< price level this order sits at
    uint32_t qty;     ///< remaining quantity (matches OrderEvent::qty type)
    Side     side;    ///< Bid or Ask
};

// ---------------------------------------------------------------------------
// LobEngine
// ---------------------------------------------------------------------------
class LobEngine {
public:
    /// @param tick_size   Minimum price increment in ticks (must be > 0).
    /// @param max_depth   Maximum levels captured in snapshot (default 10).
    explicit LobEngine(int64_t tick_size, uint32_t max_depth = 10) noexcept;

    // Not copyable — internal maps are large; move is fine.
    LobEngine(const LobEngine&)            = delete;
    LobEngine& operator=(const LobEngine&) = delete;
    LobEngine(LobEngine&&)                 = default;
    LobEngine& operator=(LobEngine&&)      = default;

    // -----------------------------------------------------------------------
    // Core hot-path method
    // -----------------------------------------------------------------------

    /// Process one order event.
    ///
    /// Returns true  if the best bid or best ask changed (book state changed).
    /// Returns false if the event was rejected (validation failure) or the
    ///               book state was unchanged (e.g. mid-book add).
    ///
    /// Rejection conditions (all leave book state unchanged):
    ///   - price <= 0                          (Req 11.1)
    ///   - qty == 0 for Add or Fill            (Req 11.2)
    ///   - timestamp_ns < last_timestamp_ns_   (Req 11.3)
    ///   - unknown order_id for Cancel/Modify  (Req 1.8)
    ///   - event would produce a crossed book  (Req 1.6, 1.7, 9.1)
    bool process(const OrderEvent& ev) noexcept;

    // -----------------------------------------------------------------------
    // Snapshot (Req 1.5)
    // -----------------------------------------------------------------------

    /// Fill a pre-allocated BookSnapshot with the current top-max_depth levels.
    /// Zero-copy: writes directly into caller's struct.
    void snapshot(BookSnapshot& out) const noexcept;

    // -----------------------------------------------------------------------
    // O(1) accessors (Req 1.12)
    // -----------------------------------------------------------------------

    /// Highest active bid price in ticks.
    /// Returns std::numeric_limits<int64_t>::min() when bid side is empty.
    int64_t best_bid()  const noexcept;

    /// Lowest active ask price in ticks.
    /// Returns std::numeric_limits<int64_t>::max() when ask side is empty.
    int64_t best_ask()  const noexcept;

    /// (best_bid + best_ask) / 2; returns 0 when either side is empty.
    int64_t mid_price() const noexcept;

    /// best_ask - best_bid; returns 0 when either side is empty.
    int64_t spread()    const noexcept;

    // -----------------------------------------------------------------------
    // Reset (Req 1.13)
    // -----------------------------------------------------------------------

    /// Clear all book state: orders, price levels, timestamps.
    void reset() noexcept;

private:
    // -----------------------------------------------------------------------
    // Internal event handlers (each called only after validation passes)
    // -----------------------------------------------------------------------
    bool handle_add         (const OrderEvent& ev) noexcept;
    bool handle_cancel      (const OrderEvent& ev) noexcept;
    bool handle_modify      (const OrderEvent& ev) noexcept;
    bool handle_fill        (const OrderEvent& ev) noexcept;
    bool handle_partial_fill(const OrderEvent& ev) noexcept;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Returns true if adding an order at `price` on `side` would cross.
    bool would_cross(int64_t price, Side side) const noexcept;

    /// Remove qty from a price level; erase the level if total_qty reaches 0.
    /// Returns true if the level was fully removed (affects best price).
    bool remove_qty(int64_t price, Side side, uint32_t qty, bool decrement_order_count = true) noexcept;

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    // Bid side: highest price = best bid → use std::greater for descending order
    std::map<int64_t, PriceLevel, std::greater<int64_t>> bids_;

    // Ask side: lowest price = best ask → use default std::less for ascending order
    std::map<int64_t, PriceLevel> asks_;

    // O(1) order lookup
    std::unordered_map<uint64_t, OrderEntry> orders_;

    int64_t  tick_size_;           ///< minimum price increment
    uint32_t max_depth_;           ///< levels written to snapshot
    int64_t  last_timestamp_ns_;   ///< monotonicity guard (Req 11.3)
};

} // namespace mm
