#pragma once

// include/mm/strategy_framework.hpp
//
// StrategyFramework — Orchestrates the full market-making event loop:
// tick replay → LOB update → feature extraction → policy inference →
// quote generation → order submission → fill processing.
//
// Requirements: 7.1–7.10, 9.3, 9.4, 9.6, 9.8, 10.3, 10.7, 13.1–13.5

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mm/data_models.hpp"       // MarketFeatures, QuoteParams
#include "mm/feature_pipeline.hpp"  // FeaturePipeline
#include "mm/lob_engine.hpp"        // LobEngine
#include "mm/policy_inference.hpp"   // PolicyInference, SimplifiedPolicyInference
#include "mm/quote_generator.hpp"   // QuoteGenerator, IAlphaSignal
#include "mm/tick_replayer.hpp"     // TickReplayer

namespace mm {

// ---------------------------------------------------------------------------
// StrategyConfig (Req 7.10)
// ---------------------------------------------------------------------------
struct StrategyConfig {
    std::string instrument_id{"NSE_NIFTY_FUT"};
    std::string model_path{""};
    int32_t     max_position{500};
    uint64_t    max_order_size{50};
    int64_t     tick_size{5};
    int64_t     latency_budget_ns{800};
    float       inventory_penalty{0.1f};
};

// ---------------------------------------------------------------------------
// OrderGateway — Mock/Adapter order execution interface
// ---------------------------------------------------------------------------
class OrderGateway {
public:
    explicit OrderGateway(int32_t max_position = 500) noexcept;

    void submit(const QuoteParams& params) noexcept;
    void cancel_stale() noexcept;
    void cancel_all() noexcept;

    void on_fill(int64_t price, uint32_t qty, Side side) noexcept;

    int64_t net_position()    const noexcept { return net_position_; }
    int64_t total_volume()    const noexcept { return total_volume_; }
    int64_t realized_pnl()    const noexcept { return realized_pnl_; }
    bool    has_active_quotes() const noexcept { return has_active_quotes_; }

    const QuoteParams& active_quote() const noexcept { return active_quote_; }

private:
    int32_t     max_position_;
    int64_t     net_position_{0};
    int64_t     total_volume_{0};
    int64_t     realized_pnl_{0};
    int64_t     last_fill_price_{0};
    bool        has_active_quotes_{false};
    QuoteParams active_quote_{};
};

// ---------------------------------------------------------------------------
// PerfStats — Lock-free performance monitoring counters (Req 7.7, 13.1, 13.2)
// ---------------------------------------------------------------------------
struct PerfStats {
    std::atomic<uint64_t> events_processed{0};
    std::atomic<uint64_t> quotes_generated{0};
    std::atomic<int64_t>  net_pnl_ticks{0};
    std::atomic<int64_t>  max_latency_ns{0};
    std::atomic<int64_t>  p99_latency_ns{0};
    std::atomic<uint64_t> latency_breaches{0};
    std::atomic<uint64_t> fallback_activations{0};

    void reset() noexcept {
        events_processed.store(0);
        quotes_generated.store(0);
        net_pnl_ticks.store(0);
        max_latency_ns.store(0);
        p99_latency_ns.store(0);
        latency_breaches.store(0);
        fallback_activations.store(0);
    }
};

// ---------------------------------------------------------------------------
// StrategyFramework
// ---------------------------------------------------------------------------
class StrategyFramework {
public:
    explicit StrategyFramework(StrategyConfig cfg);
    ~StrategyFramework();

    /// Process a single event through the entire strategy pipeline.
    /// Returns true if a quote was generated.
    bool step(const OrderEvent& ev) noexcept;

    /// Blocking event loop consuming from TickReplayer (Req 7.1).
    void run(TickReplayer& replayer);

    /// Hot-reload model asynchronously from another thread (Req 7.6, 7.9).
    void reload_model(const std::string& path) noexcept;

    /// Attach a hot-swappable alpha signal (Req 7.6, 12.2).
    void attach_signal(std::shared_ptr<IAlphaSignal> sig) noexcept;

    /// Detach an alpha signal by ID (Req 7.6, 12.3).
    void detach_signal(uint32_t sig_id) noexcept;

    /// Graceful shutdown: cancel all open orders before exit (Req 9.8).
    void shutdown() noexcept;

    /// Accessors
    const PerfStats& stats() const noexcept { return stats_; }
    const LobEngine& lob() const noexcept { return lob_; }
    const FeaturePipeline& features() const noexcept { return features_; }
    const OrderGateway& gateway() const noexcept { return gateway_; }
    const StrategyConfig& config() const noexcept { return config_; }

private:
    StrategyConfig config_;
    LobEngine lob_;
    FeaturePipeline features_;
    PolicyInference neural_policy_;
    SimplifiedPolicyInference fallback_policy_;
    IPolicyInference* active_policy_;

    QuoteGenerator quoter_;
    OrderGateway gateway_;
    PerfStats stats_;

    std::atomic<bool> is_running_{true};
    bool suspended_due_to_gap_{false};
    std::vector<int64_t> latency_samples_;

    void record_latency(int64_t latency_ns) noexcept;
    void handle_data_gap(int64_t prev_ts, int64_t next_ts) noexcept;
};

} // namespace mm
