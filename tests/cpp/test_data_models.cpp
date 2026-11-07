#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "mm/data_models.hpp"
#include "mm/lob_types.hpp"

TEST_CASE("data-model binary layout and reward invariants", "[data-models][task-2.3]") {
    STATIC_REQUIRE(sizeof(mm::PriceLevel) == 64);
    STATIC_REQUIRE(alignof(mm::PriceLevel) == 64);
    STATIC_REQUIRE(sizeof(mm::OrderEvent) == 32);

    const mm::RewardSignal signal{12.5f, -1.5f, 2.0f};
    REQUIRE(signal.total() == 13.0f);

    const mm::MarketFeatures features{1, 2, 3, 4, 5, 6, 7, 8};
    float output[mm::MarketFeatures::DIM]{};
    features.to_array(output);
    for (int i = 0; i < mm::MarketFeatures::DIM; ++i) {
        REQUIRE(output[i] == static_cast<float>(i + 1));
    }
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    return CatchShim::run_all_tests();
}

