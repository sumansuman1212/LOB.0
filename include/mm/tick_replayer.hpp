#pragma once

// include/mm/tick_replayer.hpp
//
// TickReplayer — Lock-free SPSC queue and chronological event replayer
// with 1-second gap detection and data integrity validation.
//
// Requirements: 8.6, 9.6, 10.8, 11.7

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mm/lob_types.hpp"  // OrderEvent

namespace mm {

// ---------------------------------------------------------------------------
// SPSCQueue — Lock-free Single-Producer Single-Consumer bounded ring buffer
//
// Eliminates lock contention between the data ingestion thread (producer)
// and the strategy event loop thread (consumer) on the hot path (Req 10.8).
// ---------------------------------------------------------------------------
template <typename T, size_t Capacity = 65536>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

public:
    SPSCQueue() : head_(0), tail_(0), buffer_(std::make_unique<T[]>(Capacity)) {}

    // Non-copyable
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    /// Producer: push one item into the queue. Returns true if queued, false if full.
    bool push(const T& item) noexcept {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if ((current_tail - current_head) >= Capacity) {
            return false; // Queue full
        }

        buffer_[current_tail & (Capacity - 1)] = item;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    /// Consumer: pop one item from the queue. Returns true if retrieved, false if empty.
    bool pop(T& item) noexcept {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return false; // Queue empty
        }

        item = buffer_[current_head & (Capacity - 1)];
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }

    /// Number of items currently queued.
    /// Unsigned modular difference guarantees correct size across integer wraparound.
    size_t size() const noexcept {
        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t t = tail_.load(std::memory_order_relaxed);
        return t - h;
    }

private:
    // Separate cache lines for head and tail to prevent false sharing
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    std::unique_ptr<T[]> buffer_;
};

// ---------------------------------------------------------------------------
// TickReplayer
// ---------------------------------------------------------------------------
class TickReplayer {
public:
    using GapCallback = std::function<void(int64_t prev_ts, int64_t next_ts)>;

    explicit TickReplayer(size_t queue_capacity = 65536) noexcept;
    ~TickReplayer() = default;

    /// Load events from a pre-allocated vector (e.g. from Parquet reader or test fixtures)
    void load_events(std::vector<OrderEvent> events) noexcept;

    /// Produce next batch into SPSC queue; called by ingestion thread
    size_t pump_to_queue(size_t max_events = 1024) noexcept;

    /// Pop next event from queue; called by strategy consumer thread (non-blocking)
    bool next_event(OrderEvent& out_event) noexcept;

    /// Check if more events remain in dataset or queue
    bool has_events() const noexcept;

    /// Total events loaded
    size_t total_events() const noexcept { return events_.size(); }

    /// Enable decoupled multi-threaded mode where an ingestion thread calls pump_to_queue()
    /// and the consumer thread strictly pops from the SPSC queue without synchronous pumping.
    void set_threaded_mode(bool enable) noexcept { threaded_mode_.store(enable, std::memory_order_release); }
    bool is_threaded_mode() const noexcept { return threaded_mode_.load(std::memory_order_acquire); }

    /// Register callback for data gaps > 1 second (1e9 ns) (Req 9.6)
    void on_gap(GapCallback cb) noexcept { gap_callback_ = std::move(cb); }

    /// Reset replayer cursor
    void reset() noexcept;

private:
    std::vector<OrderEvent> events_;
    size_t cursor_{0};
    int64_t last_timestamp_ns_{0};
    SPSCQueue<OrderEvent, 65536> queue_;
    GapCallback gap_callback_;
    std::atomic<bool> threaded_mode_{false};
};

} // namespace mm
