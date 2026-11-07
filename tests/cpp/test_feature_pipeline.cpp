#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>

#include "mm/feature_pipeline.hpp"

namespace {

mm::BookSnapshot make_snapshot(int64_t timestamp, int64_t bid, int64_t ask,
                               uint64_t bid_qty, uint64_t ask_qty) {
    mm::BookSnapshot snap{};
    snap.timestamp_ns = timestamp;
    snap.best_bid = bid;
    snap.best_ask = ask;
    snap.bids[0].price = bid;
    snap.bids[0].total_qty = bid_qty;
    snap.bids[0].order_count = bid_qty == 0 ? 0U : 1U;
    snap.asks[0].price = ask;
    snap.asks[0].total_qty = ask_qty;
    snap.asks[0].order_count = ask_qty == 0 ? 0U : 1U;
    return snap;
}

mm::MarketFeatures observe(mm::FeaturePipeline& pipeline) {
    mm::MarketFeatures result{};
    pipeline.observe(result);
    return result;
}

} // namespace

TEST_CASE("feature observations are finite and bounded for valid snapshots",
          "[features][task-4.2]") {
    mm::FeaturePipeline pipeline{100};
    for (int64_t i = 0; i < 100; ++i) {
        const auto snap = make_snapshot(i + 1, 100 + i, 102 + i,
                                        static_cast<uint64_t>(i + 1),
                                        static_cast<uint64_t>(100 - i));
        pipeline.update(snap, (i % 201) - 100);
        const auto features = observe(pipeline);
        float values[mm::MarketFeatures::DIM]{};
        features.to_array(values);
        for (float value : values) {
            REQUIRE(std::isfinite(value));
            REQUIRE(std::abs(value) <= 5.0f);
        }
    }
}

TEST_CASE("OFI matches level-one and top-five formulas", "[features][task-4.3]") {
    mm::FeaturePipeline pipeline{100, 60, 5};
    auto snap = make_snapshot(1, 100, 102, 30, 10);
    for (int i = 1; i < 5; ++i) {
        snap.bids[i].total_qty = 30;
        snap.asks[i].total_qty = 10;
    }
    pipeline.update(snap, 0);
    const auto features = observe(pipeline);
    REQUIRE(features.ofi_l1 == 0.5f);
    REQUIRE(features.ofi_l5 == 0.5f);
}

TEST_CASE("inventory features preserve normalized sign and magnitude",
          "[features][task-4.4]") {
    const auto snap = make_snapshot(1, 100, 102, 10, 10);
    for (const int64_t position : {-100, -25, 0, 25, 100}) {
        mm::FeaturePipeline pipeline{100};
        pipeline.update(snap, position);
        const auto features = observe(pipeline);
        REQUIRE(features.inventory_skew == static_cast<float>(position) / 100.0f);
        REQUIRE(features.inventory_urgency == std::abs(static_cast<float>(position)) / 100.0f);
    }
}

TEST_CASE("feature reset clears rolling and emitted state", "[features][task-4.5]") {
    mm::FeaturePipeline pipeline{100};
    pipeline.update(make_snapshot(1, 100, 102, 40, 10), 50);
    pipeline.reset();
    const auto features = observe(pipeline);
    float values[mm::MarketFeatures::DIM]{};
    features.to_array(values);
    for (float value : values) {
        REQUIRE(value == 0.0f);
    }
}
