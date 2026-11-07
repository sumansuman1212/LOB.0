// src/cpp/tick_replayer.cpp
//
// Implementation of TickReplayer with gap detection.
//
// Requirements: 8.6, 9.6, 10.8, 11.7

#include "mm/tick_replayer.hpp"

#include <algorithm>
#include <iostream>

namespace mm {

constexpr int64_t kOneSecondNs = 1'000'000'000LL;

TickReplayer::TickReplayer(size_t /*queue_capacity*/) noexcept {
}

void TickReplayer::load_events(std::vector<OrderEvent> events) noexcept {
    events_ = std::move(events);
    // Guarantee chronological order (Req 8.6, 11.7)
    std::stable_sort(events_.begin(), events_.end(), [](const OrderEvent& a, const OrderEvent& b) {
        return a.timestamp_ns < b.timestamp_ns;
    });
    reset();
}

void TickReplayer::reset() noexcept {
    cursor_ = 0;
    last_timestamp_ns_ = 0;
    OrderEvent dummy;
    while (queue_.pop(dummy)) {}
}

size_t TickReplayer::pump_to_queue(size_t max_events) noexcept {
    size_t count = 0;
    while (cursor_ < events_.size() && count < max_events) {
        const auto& ev = events_[cursor_];

        // Gap detection > 1 second (Req 9.6)
        if (last_timestamp_ns_ > 0 && (ev.timestamp_ns - last_timestamp_ns_) > kOneSecondNs) {
            if (gap_callback_) {
                gap_callback_(last_timestamp_ns_, ev.timestamp_ns);
            }
        }

        if (!queue_.push(ev)) {
            break; // SPSC queue full
        }

        last_timestamp_ns_ = ev.timestamp_ns;
        ++cursor_;
        ++count;
    }
    return count;
}

bool TickReplayer::next_event(OrderEvent& out_event) noexcept {
    if (queue_.pop(out_event)) {
        return true;
    }

    // In threaded mode, consumer NEVER pumps into the SPSC queue, preserving the Single-Producer invariant.
    if (threaded_mode_.load(std::memory_order_acquire)) {
        return false;
    }

    // Synchronous single-threaded fallback: pump on-demand
    if (cursor_ < events_.size()) {
        pump_to_queue(256);
        return queue_.pop(out_event);
    }

    return false;
}

bool TickReplayer::has_events() const noexcept {
    if (threaded_mode_.load(std::memory_order_acquire)) {
        return !queue_.empty() || (cursor_ < events_.size());
    }
    return (cursor_ < events_.size()) || !queue_.empty();
}

} // namespace mm
