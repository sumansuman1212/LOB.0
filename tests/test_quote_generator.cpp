// tests/test_quote_generator.cpp
//
// Unit and Property tests for QuoteGenerator.
//
// Properties Tested:
// - Property 14: Minimum Spread Enforcement (Req 5.2, 5.3)
// - Property 15: Inventory Halt Rule (Req 5.5, 9.5)
// - Property 16: Quote Quantity Bounds (Req 5.4)
// - Property 17: Alpha Signal Aggregation (Req 5.6, 5.8, 12.2, 12.3)

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "mm/quote_generator.hpp"
#include "mm/policy_inference.hpp"

namespace {

class MockAlphaSignal : public mm::IAlphaSignal {
public:
    MockAlphaSignal(uint32_t id, float bid_adj, float ask_adj)
        : id_(id), bid_adj_(bid_adj), ask_adj_(ask_adj) {}

    uint32_t id() const noexcept override { return id_; }
    std::pair<float, float> adjust(const mm::BookSnapshot&) const noexcept override {
        return {bid_adj_, ask_adj_};
    }

private:
    uint32_t id_;
    float bid_adj_;
    float ask_adj_;
};

} // namespace

TEST_CASE("QuoteGenerator: Minimum spread enforcement (Property 14, Req 5.2)", "[quoter][spread]") {
    mm::QuoteGenerator quoter(5, 50, 500); // tick_size = 5
    mm::SimplifiedPolicyInference policy;

    float obs[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    int64_t mid_price = 20000;

    auto params = quoter.generate(obs, policy, mid_price, 0);

    REQUIRE(params.bid_price < params.ask_price);
    REQUIRE(params.ask_price - params.bid_price >= quoter.tick_size());
}

TEST_CASE("QuoteGenerator: Quote quantity bounds (Property 16, Req 5.4)", "[quoter][quantity]") {
    uint64_t max_order_size = 50;
    mm::QuoteGenerator quoter(5, max_order_size, 500);
    mm::SimplifiedPolicyInference policy;

    float obs[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};

    auto params = quoter.generate(obs, policy, 20000, 0);

    REQUIRE(params.bid_qty >= 1);
    REQUIRE(params.bid_qty <= max_order_size);
    REQUIRE(params.ask_qty >= 1);
    REQUIRE(params.ask_qty <= max_order_size);
}

TEST_CASE("QuoteGenerator: Inventory halt rule (Property 15, Req 5.5, 9.5)", "[quoter][inventory_halt]") {
    int64_t max_inventory = 500;
    mm::QuoteGenerator quoter(5, 50, max_inventory);
    mm::SimplifiedPolicyInference policy;

    float obs[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};

    SECTION("Long inventory breach (pos >= 0.9 * max_inv) halts buying") {
        int64_t high_long_pos = 460; // > 450
        auto params = quoter.generate(obs, policy, 20000, high_long_pos);

        REQUIRE(params.bid_qty == 0);
        REQUIRE(params.ask_qty > 0);
    }

    SECTION("Short inventory breach (pos <= -0.9 * max_inv) halts selling") {
        int64_t high_short_pos = -460; // < -450
        auto params = quoter.generate(obs, policy, 20000, high_short_pos);
        REQUIRE(params.ask_qty == 0);
        REQUIRE(params.bid_qty > 0);
    }

    SECTION("Inventory halt preserves alpha signal adjustments") {
        mm::BookSnapshot snap{};
        // Base quote under halt without alpha
        auto base_halt = quoter.generate(obs, policy, 20000, 460, &snap);
        REQUIRE(base_halt.bid_qty == 0);

        // Attach alpha signal adjusting ask by +2 ticks
        auto signal = std::make_shared<MockAlphaSignal>(202, 0.0f, 2.0f);
        quoter.attach_signal(signal);

        auto alpha_halt = quoter.generate(obs, policy, 20000, 460, &snap);
        REQUIRE(alpha_halt.bid_qty == 0);
        // Alpha adjustment must not be discarded: ask price must reflect the additional alpha offset
        REQUIRE(alpha_halt.ask_price > base_halt.ask_price);

        quoter.detach_signal(202);
    }
}

TEST_CASE("QuoteGenerator: Alpha signal aggregation and hot-swap (Property 17, Req 5.6, 12.2, 12.3)", "[quoter][alpha]") {
    mm::QuoteGenerator quoter(5, 50, 500);
    mm::SimplifiedPolicyInference policy;
    float obs[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    mm::BookSnapshot snap{};

    auto base_params = quoter.generate(obs, policy, 20000, 0, &snap);

    auto signal1 = std::make_shared<MockAlphaSignal>(101, 1.0f, 2.0f); // adjusts bid and ask by ticks
    quoter.attach_signal(signal1);

    auto adj_params = quoter.generate(obs, policy, 20000, 0, &snap);
    REQUIRE(adj_params.bid_price <= base_params.bid_price);
    REQUIRE(adj_params.ask_price >= base_params.ask_price);

    // Detach signal
    quoter.detach_signal(101);
    auto after_detach_params = quoter.generate(obs, policy, 20000, 0, &snap);
    REQUIRE(after_detach_params.bid_price == base_params.bid_price);
    REQUIRE(after_detach_params.ask_price == base_params.ask_price);

    // Detach non-existent signal ID is a no-op (Req 12.4)
    quoter.detach_signal(999);
}

TEST_CASE("QuoteGenerator: TTL assignment (Req 5.9)", "[quoter][ttl]") {
    mm::QuoteGenerator quoter(5, 50, 500);
    mm::SimplifiedPolicyInference policy;
    float obs[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};

    int64_t ts_ns = 1'700'000'000'000'000'000LL;
    auto params = quoter.generate(obs, policy, 20000, 0, nullptr, ts_ns);

    REQUIRE(params.valid_until_ns > ts_ns);
    REQUIRE(params.valid_until_ns == ts_ns + 100'000'000LL);
}
