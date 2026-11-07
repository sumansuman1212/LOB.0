// tests/test_strategy_framework.cpp
//
// Integration and Property tests for StrategyFramework and OrderGateway.
//
// Properties Tested:
// - Property 19: One Quote Per State-Changing Event (Req 7.2, 7.3)
// - Property 20: Inventory Bound Invariant (Req 7.5)
// - Graceful Shutdown (Req 9.8)

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "mm/strategy_framework.hpp"
#include "mm/tick_replayer.hpp"

TEST_CASE("StrategyFramework: One quote per state-changing event (Property 19, Req 7.2, 7.3)", "[strategy][quote_cycles]") {
    mm::StrategyConfig cfg;
    cfg.instrument_id = "NSE_TEST";
    cfg.model_path = "";
    cfg.max_position = 500;
    cfg.max_order_size = 50;
    cfg.tick_size = 5;
    cfg.latency_budget_ns = 800;
    cfg.inventory_penalty = 0.1f;

    mm::StrategyFramework strategy(cfg);

    // Initial event changes book -> triggers quote
    mm::OrderEvent ev1{100, 1, 20000, 10, mm::OrderEvent::Type::Add, mm::Side::Bid, {0, 0}};
    REQUIRE(strategy.step(ev1));
    REQUIRE(strategy.stats().quotes_generated.load() == 1);

    // Second event on Ask side changes book -> triggers quote
    mm::OrderEvent ev2{200, 2, 20010, 10, mm::OrderEvent::Type::Add, mm::Side::Ask, {0, 0}};
    REQUIRE(strategy.step(ev2));
    REQUIRE(strategy.stats().quotes_generated.load() == 2);

    // Event deeper in the book that does NOT change best bid/ask -> should skip quote (Req 7.3)
    mm::OrderEvent ev3{300, 3, 19990, 10, mm::OrderEvent::Type::Add, mm::Side::Bid, {0, 0}};
    REQUIRE_FALSE(strategy.step(ev3));
    REQUIRE(strategy.stats().quotes_generated.load() == 2);
}

TEST_CASE("StrategyFramework: Inventory bound invariant (Property 20, Req 7.5)", "[strategy][inventory]") {
    mm::StrategyConfig cfg{
        .instrument_id = "NSE_TEST",
        .model_path = "",
        .max_position = 100,
        .max_order_size = 50,
        .tick_size = 5,
        .latency_budget_ns = 800,
        .inventory_penalty = 0.1f
    };

    mm::StrategyFramework strategy(cfg);

    // Seed book
    strategy.step({100, 1, 20000, 10, mm::OrderEvent::Type::Add, mm::Side::Bid, {0, 0}});
    strategy.step({200, 2, 20010, 10, mm::OrderEvent::Type::Add, mm::Side::Ask, {0, 0}});

    auto& gateway = const_cast<mm::OrderGateway&>(strategy.gateway());

    // Execute fills
    gateway.on_fill(20000, 50, mm::Side::Bid);
    REQUIRE(gateway.net_position() == 50);
    REQUIRE(std::abs(gateway.net_position()) <= cfg.max_position);

    gateway.on_fill(20000, 50, mm::Side::Bid);
    REQUIRE(gateway.net_position() == 100);
    REQUIRE(std::abs(gateway.net_position()) <= cfg.max_position);

    // Breaching fill is clamped/bounded by invariant check
    gateway.on_fill(20000, 50, mm::Side::Bid);
    REQUIRE(std::abs(gateway.net_position()) <= cfg.max_position);
}

TEST_CASE("StrategyFramework: Shutdown cancels active orders (Req 9.8)", "[strategy][shutdown]") {
    mm::StrategyConfig cfg;
    cfg.instrument_id = "NSE_TEST";
    cfg.model_path = "";
    cfg.max_position = 500;
    cfg.max_order_size = 50;
    cfg.tick_size = 5;
    cfg.latency_budget_ns = 800;
    cfg.inventory_penalty = 0.1f;

    mm::StrategyFramework strategy(cfg);
    strategy.step({100, 1, 20000, 10, mm::OrderEvent::Type::Add, mm::Side::Bid, {0, 0}});
    strategy.step({200, 2, 20010, 10, mm::OrderEvent::Type::Add, mm::Side::Ask, {0, 0}});

    REQUIRE(strategy.gateway().has_active_quotes());

    strategy.shutdown();
    REQUIRE_FALSE(strategy.gateway().has_active_quotes());
}
