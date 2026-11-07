// src/cpp/strategy_framework.cpp
//
// Implementation of StrategyFramework and OrderGateway.
//
// Requirements: 7.1–7.10, 9.3, 9.4, 9.6, 9.8, 10.3, 10.7, 13.1–13.5

#include "mm/strategy_framework.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>

namespace mm {

// ---------------------------------------------------------------------------
// OrderGateway Implementation
// ---------------------------------------------------------------------------
OrderGateway::OrderGateway(int32_t max_position) noexcept
    : max_position_(max_position > 0 ? max_position : 500) {
}

void OrderGateway::submit(const QuoteParams& params) noexcept {
    active_quote_ = params;
    has_active_quotes_ = true;
}

void OrderGateway::cancel_stale() noexcept {
    has_active_quotes_ = false;
}

void OrderGateway::cancel_all() noexcept {
    has_active_quotes_ = false;
    active_quote_ = QuoteParams{};
}

void OrderGateway::on_fill(int64_t price, uint32_t qty, Side side) noexcept {
    if (side == Side::Bid) {
        net_position_ += qty;
        realized_pnl_ -= (price * qty); // cash outflow
    } else {
        net_position_ -= qty;
        realized_pnl_ += (price * qty); // cash inflow
    }

    total_volume_ += qty;
    last_fill_price_ = price;

    // Hard position limit invariant (Req 7.5)
    if (std::abs(net_position_) > max_position_) {
        std::cerr << "[OrderGateway] FATAL: Hard position limit breached! Position: "
                  << net_position_ << ", Max: " << max_position_ << std::endl;
        // Clamp to limit
        if (net_position_ > max_position_) net_position_ = max_position_;
        if (net_position_ < -max_position_) net_position_ = -max_position_;
    }
}

// ---------------------------------------------------------------------------
// StrategyFramework Implementation
// ---------------------------------------------------------------------------
StrategyFramework::StrategyFramework(StrategyConfig cfg)
    : config_(std::move(cfg)),
      lob_(config_.tick_size),
      features_(config_.max_position),
      active_policy_(&fallback_policy_),
      quoter_(config_.tick_size, config_.max_order_size, config_.max_position, config_.latency_budget_ns),
      gateway_(config_.max_position) {
    latency_samples_.reserve(10000);

    if (!config_.model_path.empty()) {
        if (neural_policy_.reload(config_.model_path)) {
            active_policy_ = &neural_policy_;
        }
    }
}

StrategyFramework::~StrategyFramework() {
    shutdown();
}

void StrategyFramework::shutdown() noexcept {
    is_running_.store(false);
    // Cancel all open orders before terminating event loop (Req 9.8)
    gateway_.cancel_all();
}

void StrategyFramework::reload_model(const std::string& path) noexcept {
    if (neural_policy_.reload(path)) {
        active_policy_ = &neural_policy_;
        std::cout << "[StrategyFramework] Hot-reloaded model: " << path << std::endl;
    }
}

void StrategyFramework::attach_signal(std::shared_ptr<IAlphaSignal> sig) noexcept {
    quoter_.attach_signal(std::move(sig));
}

void StrategyFramework::detach_signal(uint32_t sig_id) noexcept {
    quoter_.detach_signal(sig_id);
}

void StrategyFramework::handle_data_gap(int64_t prev_ts, int64_t next_ts) noexcept {
    // Gap > 1 second: cancel all quotes immediately and suspend quoting (Req 9.6)
    int64_t gap_ms = (next_ts - prev_ts) / 1'000'000LL;
    std::cout << "[StrategyFramework] Data gap detected: " << gap_ms << " ms -> canceling quotes and suspending" << std::endl;
    gateway_.cancel_all();
    suspended_due_to_gap_ = true;
}

void StrategyFramework::record_latency(int64_t latency_ns) noexcept {
    // 1. Update max latency
    int64_t current_max = stats_.max_latency_ns.load(std::memory_order_relaxed);
    while (latency_ns > current_max &&
           !stats_.max_latency_ns.compare_exchange_weak(current_max, latency_ns, std::memory_order_relaxed)) {
    }

    // 2. Check latency budget breach (Req 9.3)
    if (latency_ns > config_.latency_budget_ns) {
        stats_.latency_breaches.fetch_add(1, std::memory_order_relaxed);
    }

    // 3. Track sample for P99 calculation
    if (latency_samples_.size() < 10000) {
        latency_samples_.push_back(latency_ns);
    } else {
        // Compute P99 when window fills
        std::vector<int64_t> sorted = latency_samples_;
        std::sort(sorted.begin(), sorted.end());
        size_t p99_idx = static_cast<size_t>(static_cast<double>(sorted.size()) * 0.99);
        stats_.p99_latency_ns.store(sorted[p99_idx], std::memory_order_relaxed);
        latency_samples_.clear();
    }

    // 4. Auto-switch to SimplifiedPolicyInference on persistent breaches > 1% (Req 9.4)
    uint64_t quotes = stats_.quotes_generated.load(std::memory_order_relaxed);
    uint64_t breaches = stats_.latency_breaches.load(std::memory_order_relaxed);
    if (quotes >= 100 && (breaches * 100 > quotes)) {
        if (active_policy_ != &fallback_policy_) {
            std::cerr << "[StrategyFramework] ALERT: Persistent latency breaches > 1% ("
                      << breaches << "/" << quotes << ")! Switching to SimplifiedPolicyInference fallback." << std::endl;
            active_policy_ = &fallback_policy_;
            stats_.fallback_activations.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

bool StrategyFramework::step(const OrderEvent& ev) noexcept {
    stats_.events_processed.fetch_add(1, std::memory_order_relaxed);

    // 1. Check if we should resume from gap suspension
    if (suspended_due_to_gap_) {
        BookSnapshot snap{};
        lob_.snapshot(snap);
        // Resume only when book has at least 3 levels on each side (Req 9.6)
        if (snap.bids[2].total_qty > 0 && snap.asks[2].total_qty > 0) {
            suspended_due_to_gap_ = false;
            std::cout << "[StrategyFramework] Book replenished (>= 3 levels) -> resuming quoting" << std::endl;
        }
    }

    // 2. Process event in LOB Engine (Req 7.2)
    bool book_changed = lob_.process(ev);

    // 3. If event did not change LOB state, skip quote generation (Req 7.3)
    if (!book_changed || suspended_due_to_gap_) {
        return false;
    }

    // 4. Update feature pipeline
    BookSnapshot snapshot{};
    lob_.snapshot(snapshot);
    features_.update(snapshot, gateway_.net_position());

    MarketFeatures features{};
    features_.observe(features);

    // 5. Measure latency on quote generation hot path (Req 7.7, 10.1, 10.3)
    auto t0 = std::chrono::high_resolution_clock::now();

    float obs_buf[MarketFeatures::DIM];
    features.to_array(obs_buf);

    QuoteParams quote_params = quoter_.generate(
        obs_buf,
        *active_policy_,
        lob_.mid_price(),
        gateway_.net_position(),
        &snapshot,
        ev.timestamp_ns
    );

    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    record_latency(latency_ns);

    // 6. Cancel stale quotes & submit new quote (Req 7.4)
    gateway_.cancel_stale();
    gateway_.submit(quote_params);

    stats_.quotes_generated.fetch_add(1, std::memory_order_relaxed);

    // Update net PnL (in ticks)
    int64_t mid = lob_.mid_price();
    int64_t unrealized = gateway_.net_position() * mid;
    stats_.net_pnl_ticks.store(gateway_.realized_pnl() + unrealized, std::memory_order_relaxed);

    return true;
}

void StrategyFramework::run(TickReplayer& replayer) {
    replayer.on_gap([this](int64_t prev, int64_t next) {
        this->handle_data_gap(prev, next);
    });

    OrderEvent ev{};
    while (is_running_.load(std::memory_order_relaxed) && replayer.has_events()) {
        if (replayer.next_event(ev)) {
            step(ev);
        }
    }
}

} // namespace mm
