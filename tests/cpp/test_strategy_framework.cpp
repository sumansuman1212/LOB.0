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
    mm::StrategyConfig cfg;
    cfg.instrument_id = "NSE_TEST";
    cfg.model_path = "";
    cfg.max_position = 100;
    cfg.max_order_size = 50;
    cfg.tick_size = 5;
    cfg.latency_budget_ns = 800;
    cfg.inventory_penalty = 0.1f;

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

TEST_CASE("OrderGateway: Cash flow ledger and mark-to-market equity accounting", "[strategy][accounting]") {
    mm::OrderGateway gateway(100);

    // Initial flat position
    REQUIRE(gateway.net_position() == 0);
    REQUIRE(gateway.cash_balance() == 0);
    REQUIRE(gateway.total_pnl(20000) == 0);

    // Buy 20 @ 20000 -> cash = -400,000, pos = +20
    gateway.on_fill(20000, 20, mm::Side::Bid);
    REQUIRE(gateway.net_position() == 20);
    REQUIRE(gateway.cash_balance() == -400000);
    // At price 20000, MTM PnL = -400000 + 20*20000 = 0
    REQUIRE(gateway.total_pnl(20000) == 0);
    // If mid moves up to 20050, MTM PnL = -400000 + 20*20050 = +1000
    REQUIRE(gateway.total_pnl(20050) == 1000);

    // Sell 10 @ 20060 -> cash = -400000 + 200600 = -199400, pos = +10
    gateway.on_fill(20060, 10, mm::Side::Ask);
    REQUIRE(gateway.net_position() == 10);
    REQUIRE(gateway.cash_balance() == -199400);
    // At price 20060, MTM PnL = -199400 + 10*20060 = +1200
    REQUIRE(gateway.total_pnl(20060) == 1200);

    // Close remaining 10 @ 20040 -> cash = -199400 + 200400 = +1000, pos = 0
    gateway.on_fill(20040, 10, mm::Side::Ask);
    REQUIRE(gateway.net_position() == 0);
    REQUIRE(gateway.cash_balance() == 1000);
    REQUIRE(gateway.total_pnl(20040) == 1000);
}
