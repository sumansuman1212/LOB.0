// src/cpp/mm_backtest.cpp
//
// Standalone Backtest Runner for RL-Driven Adaptive Market Making Engine.
//
// Requirements: 7.1–7.10, 10.1–10.8, 17.1

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "mm/strategy_framework.hpp"
#include "mm/tick_replayer.hpp"

namespace {

// Generate realistic synthetic L2 tick event stream
std::vector<mm::OrderEvent> generate_synthetic_tick_stream(size_t n_events, int64_t start_price = 20000) {
    std::vector<mm::OrderEvent> events;
    events.reserve(n_events);

    std::mt19937_64 rng(42);
    std::normal_distribution<double> price_delta(0.0, 1.5);
    std::uniform_int_distribution<uint32_t> qty_dist(5, 50);
    std::uniform_int_distribution<int> type_dist(0, 100);

    int64_t current_ts = 1'700'000'000'000'000'000LL; // ns
    int64_t current_mid = start_price;
    uint64_t order_id = 1;

    // Initial book depth
    for (int i = 1; i <= 10; ++i) {
        events.push_back({current_ts, order_id++, current_mid - i * 5, 20, mm::OrderEvent::Type::Add, mm::Side::Bid, {0, 0}});
        events.push_back({current_ts, order_id++, current_mid + i * 5, 20, mm::OrderEvent::Type::Add, mm::Side::Ask, {0, 0}});
        current_ts += 1000;
    }

    for (size_t i = 0; i < n_events; ++i) {
        current_ts += 50'000; // 50 microseconds between events
        double delta = price_delta(rng);
        if (std::abs(delta) > 1.0) {
            current_mid += static_cast<int64_t>(std::round(delta)) * 5;
        }

        int p = type_dist(rng);
        mm::Side side = (p % 2 == 0) ? mm::Side::Bid : mm::Side::Ask;
        uint32_t qty = qty_dist(rng);

        if (p < 60) {
            // Add event
            int64_t price = (side == mm::Side::Bid) ? (current_mid - 5) : (current_mid + 5);
            events.push_back({current_ts, order_id++, price, qty, mm::OrderEvent::Type::Add, side, {0, 0}});
        } else if (p < 85) {
            // Fill event on passive quote
            int64_t price = (side == mm::Side::Bid) ? (current_mid - 5) : (current_mid + 5);
            events.push_back({current_ts, order_id++, price, qty, mm::OrderEvent::Type::Fill, side, {0, 0}});
        } else {
            // Cancel event
            int64_t price = (side == mm::Side::Bid) ? (current_mid - 10) : (current_mid + 10);
            events.push_back({current_ts, order_id++, price, qty, mm::OrderEvent::Type::Cancel, side, {0, 0}});
        }
    }

    return events;
}

} // namespace

int main(int argc, char** argv) {
    size_t event_count = 50000;
    if (argc > 1) {
        event_count = std::stoull(argv[1]);
    }

    std::cout << "================================================================" << std::endl;
    std::cout << " RL-Driven Adaptive Market Making Engine — Backtest Runner" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Generating " << event_count << " synthetic tick events..." << std::endl;

    auto events = generate_synthetic_tick_stream(event_count);
    std::cout << "Loaded " << events.size() << " tick events." << std::endl;

    mm::StrategyConfig cfg;
    cfg.instrument_id = "NSE_NIFTY_FUT";
    cfg.model_path = "models/ppo_actor.pt";
    cfg.max_position = 500;
    cfg.max_order_size = 50;
    cfg.tick_size = 5;
    cfg.latency_budget_ns = 800;
    cfg.inventory_penalty = 0.1f;

    mm::StrategyFramework strategy(cfg);
    mm::TickReplayer replayer;
    replayer.load_events(std::move(events));

    std::cout << "Starting backtest replay..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    strategy.run(replayer);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    const auto& stats = strategy.stats();
    const auto& gateway = strategy.gateway();

    std::cout << "\n================================================================" << std::endl;
    std::cout << " Backtest Execution Summary" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Events Processed:    " << stats.events_processed.load() << std::endl;
    std::cout << "Quotes Generated:    " << stats.quotes_generated.load() << std::endl;
    std::cout << "Total Volume Traded: " << gateway.total_volume() << " lots" << std::endl;
    std::cout << "Final Net Position:  " << gateway.net_position() << " lots (Limit: ±" << cfg.max_position << ")" << std::endl;
    std::cout << "Net PnL (ticks):     " << stats.net_pnl_ticks.load() << std::endl;
    std::cout << "Max Latency:         " << stats.max_latency_ns.load() << " ns" << std::endl;
    std::cout << "Latency Breaches:    " << stats.latency_breaches.load() << std::endl;
    std::cout << "Total Replay Time:   " << duration_ms << " ms" << std::endl;
    if (duration_ms > 0) {
        double throughput = static_cast<double>(stats.events_processed.load()) / (static_cast<double>(duration_ms) / 1000.0);
        std::cout << "Throughput:          " << std::fixed << std::setprecision(0) << throughput << " events/sec" << std::endl;
    }
    std::cout << "================================================================" << std::endl;

    // Invariant assertions
    if (std::abs(gateway.net_position()) > cfg.max_position) {
        std::cerr << "FAILED: Net position exceeded max_position limit!" << std::endl;
        return 1;
    }

    std::cout << "SUCCESS: All backtest invariants satisfied." << std::endl;
    return 0;
}
