// src/cpp/quote_generator.cpp
//
// Implementation of QuoteGenerator and Alpha Signal aggregation.
//
// Requirements: 5.1–5.9, 9.5, 10.1, 12.1–12.6

#include "mm/quote_generator.hpp"

#include <algorithm>
#include <cmath>

namespace mm {

namespace {

template <typename T>
constexpr const T& clamp_val(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

} // namespace

QuoteGenerator::QuoteGenerator(int64_t  tick_size,
                               uint64_t max_order_size,
                               int64_t  max_inventory,
                               int64_t  latency_budget_ns) noexcept
    : tick_size_(tick_size > 0 ? tick_size : 1),
      max_order_size_(max_order_size > 0 ? max_order_size : 1),
      max_inventory_(max_inventory > 0 ? max_inventory : 1),
      latency_budget_ns_(latency_budget_ns > 0 ? latency_budget_ns : 800),
      signals_(std::make_shared<const std::vector<std::shared_ptr<IAlphaSignal>>>()) {
}

QuoteParams QuoteGenerator::generate(const float*      obs,
                                     IPolicyInference& policy,
                                     int64_t           mid_price,
                                     int64_t           net_position,
                                     const BookSnapshot* snap,
                                     int64_t           current_ts_ns) noexcept {
    // 1. Invoke policy inference (Req 5.1)
    QuoteAction action = policy.act(obs);

    // 2. Continuous action to tick offset translation (Req 5.3)
    float delta_bid = std::max(0.0f, action.delta_bid);
    float delta_ask = std::max(0.0f, action.delta_ask);
    float size_frac = clamp_val(action.size_frac, 0.01f, 1.0f);

    int64_t bid_offset = static_cast<int64_t>(std::round(delta_bid * static_cast<float>(tick_size_)));
    int64_t ask_offset = static_cast<int64_t>(std::round(delta_ask * static_cast<float>(tick_size_)));

    int64_t bid_price = mid_price - bid_offset;
    int64_t ask_price = mid_price + ask_offset;

    // 3. Aggregate attached Alpha Signals (Req 5.6, 12.1)
    auto sig_list = std::atomic_load(&signals_);
    if (sig_list && !sig_list->empty() && snap) {
        for (const auto& sig : *sig_list) {
            if (sig) {
                auto adj = sig->adjust(*snap);
                float adj_bid = adj.first;
                float adj_ask = adj.second;
                bid_price -= static_cast<int64_t>(std::round(adj_bid * static_cast<float>(tick_size_)));
                ask_price += static_cast<int64_t>(std::round(adj_ask * static_cast<float>(tick_size_)));
            }
        }
    }

    // 4. Enforce minimum spread of at least 1 tick (Req 5.2)
    if (ask_price - bid_price < tick_size_) {
        ask_price = bid_price + tick_size_;
    }

    // 5. Scale quote sizes to [1, max_order_size] (Req 5.4)
    uint64_t bid_qty = clamp_val(
        static_cast<uint64_t>(std::round(size_frac * static_cast<float>(max_order_size_))),
        1ULL,
        max_order_size_
    );
    uint64_t ask_qty = clamp_val(
        static_cast<uint64_t>(std::round(size_frac * static_cast<float>(max_order_size_))),
        1ULL,
        max_order_size_
    );

    // 6. Inventory halt rule (Req 5.5, 9.5)
    // When |net_position| >= max_inventory * 0.9, zero out offending-side quantity
    // and widen the opposite side quote distance by 2x while preserving alpha signal adjustments.
    int64_t halt_threshold = static_cast<int64_t>(static_cast<double>(max_inventory_) * 0.9);

    if (net_position >= halt_threshold) {
        // Overly long -> halt buying, widen ask quote distance by 2x (preserving alpha adjustment)
        bid_qty = 0;
        int64_t current_ask_dist = std::max<int64_t>(tick_size_, ask_price - mid_price);
        ask_price = mid_price + current_ask_dist * 2;
        if (ask_price - bid_price < tick_size_) {
            ask_price = bid_price + tick_size_;
        }
    } else if (net_position <= -halt_threshold) {
        // Overly short -> halt selling, widen bid quote distance by 2x (preserving alpha adjustment)
        ask_qty = 0;
        int64_t current_bid_dist = std::max<int64_t>(tick_size_, mid_price - bid_price);
        bid_price = mid_price - current_bid_dist * 2;
        if (ask_price - bid_price < tick_size_) {
            bid_price = ask_price - tick_size_;
        }
    }

    // 7. Assign TTL valid_until_ns (default 100 ms in future) (Req 5.9)
    int64_t valid_until_ns = (current_ts_ns > 0) ? (current_ts_ns + 100'000'000LL) : 0;

    return QuoteParams{
        bid_price,
        ask_price,
        bid_qty,
        ask_qty,
        valid_until_ns,
    };
}

void QuoteGenerator::attach_signal(std::shared_ptr<IAlphaSignal> signal) noexcept {
    if (!signal) return;
    while (signal_lock_.test_and_set(std::memory_order_acquire)) {}
    auto current = std::atomic_load(&signals_);
    auto updated = std::make_shared<std::vector<std::shared_ptr<IAlphaSignal>>>(*current);
    
    // Remove if already exists with same ID, then append
    updated->erase(
        std::remove_if(updated->begin(), updated->end(),
                       [&](const auto& s) { return s && s->id() == signal->id(); }),
        updated->end()
    );
    updated->push_back(signal);
    std::atomic_store(&signals_, std::shared_ptr<const std::vector<std::shared_ptr<IAlphaSignal>>>(updated));
    signal_lock_.clear(std::memory_order_release);
}

void QuoteGenerator::detach_signal(uint32_t signal_id) noexcept {
    while (signal_lock_.test_and_set(std::memory_order_acquire)) {}
    auto current = std::atomic_load(&signals_);
    auto updated = std::make_shared<std::vector<std::shared_ptr<IAlphaSignal>>>(*current);
    
    auto it = std::remove_if(updated->begin(), updated->end(),
                             [signal_id](const auto& s) { return s && s->id() == signal_id; });
    if (it != updated->end()) {
        updated->erase(it, updated->end());
        std::atomic_store(&signals_, std::shared_ptr<const std::vector<std::shared_ptr<IAlphaSignal>>>(updated));
    }
    signal_lock_.clear(std::memory_order_release);
}

} // namespace mm
