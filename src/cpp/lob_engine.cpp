// src/cpp/lob_engine.cpp
//
// LobEngine implementation — limit order book with price-time priority.
//
// All methods are noexcept. No heap allocation occurs on the hot path after
// construction; std::map and std::unordered_map nodes may allocate during
// inserts, but the design treats per-event O(log N) tree operations as the
// expected cost (Req 1.10 / 1.11).
//
// Requirements: 1.1–1.13, 9.1, 11.1, 11.2, 11.3

#include "mm/lob_engine.hpp"

#include <algorithm>   // std::min
#include <cstdio>      // std::fprintf (structured logging)
#include <cinttypes>   // PRId64, PRIu64
#include <limits>

namespace mm {

// ---------------------------------------------------------------------------
// Sentinel values
// ---------------------------------------------------------------------------
static constexpr int64_t kEmptyBid = std::numeric_limits<int64_t>::min();
static constexpr int64_t kEmptyAsk = std::numeric_limits<int64_t>::max();

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
LobEngine::LobEngine(int64_t tick_size, uint32_t max_depth) noexcept
    : tick_size_(tick_size)
    , max_depth_(max_depth == 0 ? 1u : max_depth)
    , last_timestamp_ns_(std::numeric_limits<int64_t>::min())
{}

// ---------------------------------------------------------------------------
// reset()  — Req 1.13
// ---------------------------------------------------------------------------
void LobEngine::reset() noexcept {
    bids_.clear();
    asks_.clear();
    orders_.clear();
    last_timestamp_ns_ = std::numeric_limits<int64_t>::min();
}

// ---------------------------------------------------------------------------
// Accessors  — Req 1.12
// ---------------------------------------------------------------------------
int64_t LobEngine::best_bid() const noexcept {
    return bids_.empty() ? kEmptyBid : bids_.begin()->first;
}

int64_t LobEngine::best_ask() const noexcept {
    return asks_.empty() ? kEmptyAsk : asks_.begin()->first;
}

int64_t LobEngine::mid_price() const noexcept {
    if (bids_.empty() || asks_.empty()) return 0;
    return (best_bid() + best_ask()) / 2;
}

int64_t LobEngine::spread() const noexcept {
    if (bids_.empty() || asks_.empty()) return 0;
    return best_ask() - best_bid();
}

// ---------------------------------------------------------------------------
// snapshot()  — Req 1.5
//
// Writes top-max_depth_ levels into caller-supplied BookSnapshot.
// Levels beyond active depth are zeroed.
// ---------------------------------------------------------------------------
void LobEngine::snapshot(BookSnapshot& out) const noexcept {
    out.best_bid     = best_bid();
    out.best_ask     = best_ask();
    out.timestamp_ns = last_timestamp_ns_;

    // Bid levels (descending — bids_ already in descending key order)
    uint32_t i = 0;
    for (auto it = bids_.begin();
         it != bids_.end() && i < static_cast<uint32_t>(kMaxDepth);
         ++it, ++i) {
        out.bids[i] = it->second;
    }
    for (; i < static_cast<uint32_t>(kMaxDepth); ++i) {
        out.bids[i] = PriceLevel{};  // zero-fill unused levels
    }

    // Ask levels (ascending — asks_ already in ascending key order)
    i = 0;
    for (auto it = asks_.begin();
         it != asks_.end() && i < static_cast<uint32_t>(kMaxDepth);
         ++it, ++i) {
        out.asks[i] = it->second;
    }
    for (; i < static_cast<uint32_t>(kMaxDepth); ++i) {
        out.asks[i] = PriceLevel{};
    }
}

// ---------------------------------------------------------------------------
// would_cross()
//
// An Add event on the Bid side crosses if bid_price >= best_ask.
// An Add event on the Ask side crosses if ask_price <= best_bid.
// ---------------------------------------------------------------------------
bool LobEngine::would_cross(int64_t price, Side side) const noexcept {
    if (side == Side::Bid) {
        // New bid would cross if it meets or beats the best ask
        return !asks_.empty() && (price >= asks_.begin()->first);
    } else {
        // New ask would cross if it meets or undercuts the best bid
        return !bids_.empty() && (price <= bids_.begin()->first);
    }
}

// ---------------------------------------------------------------------------
// remove_qty()
//
// Removes `qty` from the PriceLevel at `price` on `side`.
// Erases the level when total_qty drops to zero.
// Returns true if the erased level was the best price on that side
// (i.e. best price may have changed).
// ---------------------------------------------------------------------------
bool LobEngine::remove_qty(int64_t price, Side side, uint32_t qty, bool decrement_order_count) noexcept {
    if (side == Side::Bid) {
        auto it = bids_.find(price);
        if (it == bids_.end()) return false;

        bool is_best = (it == bids_.begin());
        if (qty >= it->second.total_qty) {
            bids_.erase(it);
        } else {
            it->second.total_qty -= qty;
            if (decrement_order_count && it->second.order_count > 0) {
                --it->second.order_count;
            }
        }
        return is_best;
    } else {
        auto it = asks_.find(price);
        if (it == asks_.end()) return false;

        bool is_best = (it == asks_.begin());
        if (qty >= it->second.total_qty) {
            asks_.erase(it);
        } else {
            it->second.total_qty -= qty;
            if (decrement_order_count && it->second.order_count > 0) {
                --it->second.order_count;
            }
        }
        return is_best;
    }
}

// ---------------------------------------------------------------------------
// process()  — main hot-path dispatcher
//
// Validation order (all checks before any state mutation):
//   1. Timestamp monotonicity              (Req 11.3)
//   2. price > 0                           (Req 11.1)
//   3. qty > 0 for Add / Fill              (Req 11.2)
//   4. order_id exists for Cancel / Modify (Req 1.8)
//   5. no crossed book for Add             (Req 1.6, 1.7, 9.1)
//
// Returns true iff best_bid or best_ask changed.
// ---------------------------------------------------------------------------
bool LobEngine::process(const OrderEvent& ev) noexcept {
    using Type = OrderEvent::Type;

    // Reject an invalid enum value before it can affect the timestamp or book.
    if (static_cast<uint8_t>(ev.type) > static_cast<uint8_t>(Type::PartialFill)) {
        return false;
    }

    // --- 1. Timestamp monotonicity (Req 11.3) ---
    if (ev.timestamp_ns < last_timestamp_ns_) {
        return false;
    }

    // --- 2. Price validation (Req 11.1) ---
    // Cancel events carry price=0 in some implementations; only enforce for
    // Add, Modify, Fill, PartialFill where price is meaningful.
    if (ev.type != Type::Cancel) {
        if (ev.price <= 0) return false;
    }

    // --- 3. Qty validation (Req 11.2) ---
    if ((ev.type == Type::Add || ev.type == Type::Fill ||
         ev.type == Type::PartialFill) && ev.qty == 0) {
        return false;
    }

    // --- 4. Order identity validation ---
    // A duplicate Add must be rejected.  Replacing the entry in orders_ after
    // adding its quantity to a level would orphan the original resting
    // quantity, leaving the per-order index and aggregate levels inconsistent.
    if (ev.type == Type::Add && orders_.find(ev.order_id) != orders_.end()) {
        return false;
    }

    // Cancel and Modify require an existing resting order.  Keep the iterator
    // for the Modify preflight below; no mutation has occurred at this point.
    auto existing = orders_.end();
    if (ev.type == Type::Cancel || ev.type == Type::Modify) {
        existing = orders_.find(ev.order_id);
        if (existing == orders_.end()) return false;
    }

    // --- 5. Crossed-book guard for Add (Req 1.6, 1.7, 9.1) ---
    if (ev.type == Type::Add && would_cross(ev.price, ev.side)) {
        // Structured log entry (Req 9.1, 13.3)
        std::fprintf(stderr,
            "[LobEngine] CROSSED-BOOK REJECT ts=%" PRId64 " order_id=%" PRIu64
            " price=%" PRId64 " side=%s best_bid=%" PRId64 " best_ask=%" PRId64 "\n",
            ev.timestamp_ns,
            ev.order_id,
            ev.price,
            (ev.side == Side::Bid ? "Bid" : "Ask"),
            best_bid(),
            best_ask());
        return false;
    }

    // A Modify can move a resting order onto the opposite best price.  This
    // must be checked before advancing last_timestamp_ns_ or removing the old
    // level, so a rejected modification is fully transaction-safe.
    if (ev.type == Type::Modify && ev.price != existing->second.price &&
        would_cross(ev.price, existing->second.side)) {
        return false;
    }

    // --- Advance timestamp only after all validation passes ---
    last_timestamp_ns_ = ev.timestamp_ns;

    // --- Dispatch ---
    switch (ev.type) {
        case Type::Add:         return handle_add(ev);
        case Type::Cancel:      return handle_cancel(ev);
        case Type::Modify:      return handle_modify(ev);
        case Type::Fill:        return handle_fill(ev);
        case Type::PartialFill: return handle_partial_fill(ev);
        default:                return false;
    }
}

// ---------------------------------------------------------------------------
// handle_add()  — Req 1.1
//
// Insert order into price level.  Returns true if best price changed.
// ---------------------------------------------------------------------------
bool LobEngine::handle_add(const OrderEvent& ev) noexcept {
    // Capture best prices before insertion to detect change.
    int64_t old_best_bid = best_bid();
    int64_t old_best_ask = best_ask();

    if (ev.side == Side::Bid) {
        auto& level = bids_[ev.price];
        if (level.price == 0 && level.total_qty == 0) {
            // New level — initialise price field
            level.price = ev.price;
        }
        level.total_qty   += ev.qty;
        level.order_count += 1;
    } else {
        auto& level = asks_[ev.price];
        if (level.price == 0 && level.total_qty == 0) {
            level.price = ev.price;
        }
        level.total_qty   += ev.qty;
        level.order_count += 1;
    }

    // Track order for later Cancel/Modify/Fill
    orders_[ev.order_id] = OrderEntry{ev.price, ev.qty, ev.side};

    return (best_bid() != old_best_bid) || (best_ask() != old_best_ask);
}

// ---------------------------------------------------------------------------
// handle_cancel()  — Req 1.2
//
// Remove order from price level.  Returns true if best price changed.
// ---------------------------------------------------------------------------
bool LobEngine::handle_cancel(const OrderEvent& ev) noexcept {
    auto it = orders_.find(ev.order_id);
    // Guaranteed to exist (validated in process()).
    OrderEntry& entry = it->second;

    int64_t old_best_bid = best_bid();
    int64_t old_best_ask = best_ask();

    remove_qty(entry.price, entry.side, entry.qty, true);
    orders_.erase(it);

    return (best_bid() != old_best_bid) || (best_ask() != old_best_ask);
}

// ---------------------------------------------------------------------------
// handle_modify()  — Req 1.4
//
// Update quantity or price.  If price changes, move the order.
// Time priority relative to remaining orders at the same level is preserved
// (we reduce/increase total_qty; individual FIFO queues are not tracked
// since the LOB simulator treats each price level as an aggregate bucket).
//
// Returns true if best price changed.
// ---------------------------------------------------------------------------
bool LobEngine::handle_modify(const OrderEvent& ev) noexcept {
    auto it = orders_.find(ev.order_id);
    OrderEntry& entry = it->second;

    int64_t old_best_bid = best_bid();
    int64_t old_best_ask = best_ask();

    // Price change: remove from old level, insert at new level.
    // Qty change only: just update the level's total_qty.
    bool price_changed = (ev.price > 0 && ev.price != entry.price);

    if (price_changed) {
        // Check the new price wouldn't cross the book.
        if (would_cross(ev.price, entry.side)) {
            return false;
        }
        remove_qty(entry.price, entry.side, entry.qty, true);

        // Insert at new price level
        uint32_t new_qty = (ev.qty > 0) ? ev.qty : entry.qty;
        if (entry.side == Side::Bid) {
            auto& level = bids_[ev.price];
            if (level.price == 0 && level.total_qty == 0) level.price = ev.price;
            level.total_qty   += new_qty;
            level.order_count += 1;
        } else {
            auto& level = asks_[ev.price];
            if (level.price == 0 && level.total_qty == 0) level.price = ev.price;
            level.total_qty   += new_qty;
            level.order_count += 1;
        }
        entry.price = ev.price;
        entry.qty   = new_qty;
    } else if (ev.qty > 0 && ev.qty != entry.qty) {
        // Quantity-only change
        int64_t price = entry.price;
        if (entry.side == Side::Bid) {
            auto bit = bids_.find(price);
            if (bit != bids_.end()) {
                // Adjust delta
                if (ev.qty > entry.qty) {
                    bit->second.total_qty += (ev.qty - entry.qty);
                } else {
                    uint32_t delta = entry.qty - ev.qty;
                    if (delta >= bit->second.total_qty) {
                        bids_.erase(bit);
                    } else {
                        bit->second.total_qty -= delta;
                    }
                }
            }
        } else {
            auto ait = asks_.find(price);
            if (ait != asks_.end()) {
                if (ev.qty > entry.qty) {
                    ait->second.total_qty += (ev.qty - entry.qty);
                } else {
                    uint32_t delta = entry.qty - ev.qty;
                    if (delta >= ait->second.total_qty) {
                        asks_.erase(ait);
                    } else {
                        ait->second.total_qty -= delta;
                    }
                }
            }
        }
        entry.qty = ev.qty;
    }

    return (best_bid() != old_best_bid) || (best_ask() != old_best_ask);
}

// ---------------------------------------------------------------------------
// handle_fill()  — Req 1.3 (full fill)
//
// A full fill: remove the entire order from the book.
// ---------------------------------------------------------------------------
bool LobEngine::handle_fill(const OrderEvent& ev) noexcept {
    // For Fill events the order_id may not be in our map if it's an aggressor
    // order that never had an Add event (market order).  We use the provided
    // price and side to locate and reduce the level directly.
    auto it = orders_.find(ev.order_id);
    if (it != orders_.end()) {
        // Known resting order — remove it fully
        OrderEntry& entry = it->second;
        int64_t old_best_bid = best_bid();
        int64_t old_best_ask = best_ask();

        remove_qty(entry.price, entry.side, entry.qty, true);
        orders_.erase(it);

        return (best_bid() != old_best_bid) || (best_ask() != old_best_ask);
    }

    // Unknown order_id (aggressor / market order) — use ev.price and ev.side
    // to reduce the opposite side level by ev.qty.
    if (ev.price <= 0) return false;

    int64_t old_best_bid = best_bid();
    int64_t old_best_ask = best_ask();

    remove_qty(ev.price, ev.side, ev.qty, false);

    return (best_bid() != old_best_bid) || (best_ask() != old_best_ask);
}

// ---------------------------------------------------------------------------
// handle_partial_fill()  — Req 1.3 (partial fill)
//
// Reduce quantity by ev.qty; leave remainder in the book.
// ---------------------------------------------------------------------------
bool LobEngine::handle_partial_fill(const OrderEvent& ev) noexcept {
    auto it = orders_.find(ev.order_id);
    if (it != orders_.end()) {
        OrderEntry& entry = it->second;
        int64_t old_best_bid = best_bid();
        int64_t old_best_ask = best_ask();

        uint32_t fill_qty = ev.qty;

        if (fill_qty >= entry.qty) {
            // Treat over-fill as full fill
            remove_qty(entry.price, entry.side, entry.qty, true);
            orders_.erase(it);
        } else {
            // Partial fill leaves order resting in book -> do not decrement order_count
            remove_qty(entry.price, entry.side, fill_qty, false);
            entry.qty -= fill_qty;
        }

        return (best_bid() != old_best_bid) || (best_ask() != old_best_ask);
    }

    // Unknown order_id — reduce level directly
    if (ev.price <= 0) return false;

    int64_t old_best_bid = best_bid();
    int64_t old_best_ask = best_ask();

    remove_qty(ev.price, ev.side, ev.qty, false);

    return (best_bid() != old_best_bid) || (best_ask() != old_best_ask);
}

} // namespace mm
