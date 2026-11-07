#pragma once

// include/mm/types.hpp
// Shared enumerations and compile-time constants used across all mm:: components.

#include <cstdint>

namespace mm {

// ---------------------------------------------------------------------------
// Side: order / quote side
// ---------------------------------------------------------------------------
enum class Side : uint8_t { Bid = 0, Ask = 1 };

// ---------------------------------------------------------------------------
// VolRegime: volatility regime classification
// ---------------------------------------------------------------------------
enum class VolRegime : uint8_t { Low = 0, Normal = 1, High = 2, Extreme = 3 };

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------

/// Maximum number of price levels captured in a BookSnapshot (per side)
constexpr int kMaxDepth = 10;

/// Observation vector dimension (must match MarketFeatures::DIM)
constexpr int kObsDim = 8;

/// Action vector dimension: (delta_bid, delta_ask, size_frac)
constexpr int kActionDim = 3;

} // namespace mm
