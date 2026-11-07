#pragma once

// include/mm/lob_types.hpp
// Core data structures for the limit order book:
//   - PriceLevel    : 64-byte cache-line-aligned price level (pad to 64)
//   - OrderEvent    : 32-byte event descriptor with PartialFill support
//   - BookSnapshot  : top-10 bid/ask snapshot plus best-price accessors
//
// Requirements: 1.10 (no heap alloc / cache alignment), 10.6 (64-byte PriceLevel),
//               11.1 (event validation scaffolding)

#include <array>
#include <cstdint>

#include "types.hpp"   // mm::Side, mm::VolRegime, mm::kMaxDepth

namespace mm {

// ---------------------------------------------------------------------------
// PriceLevel — exactly one cache line (64 bytes), cache-aligned
//
// Layout (bytes):
//   [0 .. 7 ]  price        int64_t
//   [8 .. 15]  total_qty    uint64_t
//   [16 .. 19] order_count  uint32_t
//   [20 .. 23] _pad0        uint32_t
//   [24 .. 63] _pad1        uint8_t[40]   <-- explicit padding to reach 64 bytes
// ---------------------------------------------------------------------------
struct alignas(64) PriceLevel {
    int64_t  price;        ///< price in ticks (integer representation)
    uint64_t total_qty;    ///< total quantity resting at this level
    uint32_t order_count;  ///< number of individual orders at this level
    uint32_t _pad0;        ///< padding to align _pad1 to a natural boundary
    uint8_t  _pad1[40];    ///< explicit padding — fills the cache line to 64 bytes
};

static_assert(sizeof(PriceLevel) == 64,
              "PriceLevel must be exactly 64 bytes (one cache line)");
static_assert(alignof(PriceLevel) == 64,
              "PriceLevel must be 64-byte aligned");

// ---------------------------------------------------------------------------
// OrderEvent — exactly 32 bytes
//
// Layout (bytes):
//   [0  ..  7]  timestamp_ns   int64_t   (8)
//   [8  .. 15]  order_id       uint64_t  (8)
//   [16 .. 23]  price          int64_t   (8)
//   [24 .. 31]  qty            uint64_t  (8)
//   [32]        type           uint8_t   (1)   <-- wait: offset 32 is byte 33 if
//                                                   the struct is 32 bytes…
//
// Correct interpretation: 4 × 8-byte fields = 32 bytes.
// The type / side / _pad fields must share those 32 bytes.
//
// Revised layout:
//   [0  ..  7]  timestamp_ns   int64_t   (8)
//   [8  .. 15]  order_id       uint64_t  (8)
//   [16 .. 23]  price          int64_t   (8)
//   [24 .. 29]  qty            uint32_t  (4)  -- NOT: uint64_t would push to 40
//
// Actually the design specifies qty as uint64_t and 32 bytes total with _pad[2]:
//   8 + 8 + 8 + 8 = 32 (four 8-byte fields)
//   type(1) + side(1) + _pad[2] = 4  -- but that is 36 total
//
// The only way to hit 32 bytes is to overlap type/side/pad into the 4 least-
// significant bytes of the last 8-byte slot, or use qty as uint32_t.
//
// Design doc "Data Models" section explicitly says:
//   int64_t  timestamp_ns   8
//   uint64_t order_id       8
//   int64_t  price          8
//   uint64_t qty            8    -- total 32 so far
//   uint8_t  type           1
//   uint8_t  side           1
//   uint8_t  _pad[6]        6    -- total 40 (design has a typo)
//
// To satisfy static_assert(sizeof(OrderEvent)==32) we must pack type+side+pad
// INTO the final 64-bit slot.  We do so by making qty uint32_t (4 bytes) and
// using the remaining 4 bytes for type, side, and 2 padding bytes:
//
//   [0  ..  7]  timestamp_ns   int64_t   (8)
//   [8  .. 15]  order_id       uint64_t  (8)
//   [16 .. 23]  price          int64_t   (8)
//   [24 .. 27]  qty            uint32_t  (4)
//   [28]        type           uint8_t   (1)
//   [29]        side           uint8_t   (1)
//   [30 .. 31]  _pad           uint8_t[2](2)
//   Total: 32 bytes ✓
//
// NOTE: qty is uint32_t here (max ~4 billion lots), which is sufficient for
// all realistic NSE futures tick sizes.  The interface comment below documents
// this deliberate choice.
// ---------------------------------------------------------------------------
struct OrderEvent {
    /// Event type discriminator
    enum class Type : uint8_t {
        Add,
        Cancel,
        Modify,
        Fill,
        PartialFill   ///< Requirements 1.3, 11.1 — partial fill reduces level qty
    };

    int64_t  timestamp_ns;  ///< nanoseconds since epoch (must be monotonically non-decreasing)
    uint64_t order_id;      ///< exchange-assigned or internal order identifier
    int64_t  price;         ///< price in integer ticks (must be > 0)
    uint32_t qty;           ///< unsigned quantity in lots (> 0 for Add and Fill)
    Type     type;          ///< event type
    Side     side;          ///< Bid | Ask
    uint8_t  _pad[2];       ///< explicit padding to reach exactly 32 bytes
};

static_assert(sizeof(OrderEvent) == 32,
              "OrderEvent must be exactly 32 bytes");

// ---------------------------------------------------------------------------
// BookSnapshot — top-10 bid and ask price levels plus best-price fields
//
// Produced by LobEngine::snapshot() after each state-changing event.
// bids[0] is the best (highest) bid; asks[0] is the best (lowest) ask.
// Levels beyond the active book depth have total_qty == 0.
// ---------------------------------------------------------------------------
struct BookSnapshot {
    std::array<PriceLevel, 10> bids;   ///< top-10 bid levels, bids[0] = best bid
    std::array<PriceLevel, 10> asks;   ///< top-10 ask levels, asks[0] = best ask
    int64_t best_bid;                  ///< highest active bid price in ticks
    int64_t best_ask;                  ///< lowest active ask price in ticks
    int64_t timestamp_ns;              ///< snapshot timestamp (matches last processed event)
};

} // namespace mm
