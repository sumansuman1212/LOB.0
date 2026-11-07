// tests/test_policy_inference.cpp
//
// Unit tests for PolicyInference, SimplifiedPolicyInference, and SHA-256 verification.
//
// Requirements: 6.5–6.9, 9.2, 9.4, 10.5, 11.8

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <fstream>
#include <vector>

#include "mm/policy_inference.hpp"

TEST_CASE("SimplifiedPolicyInference produces valid and bounded actions", "[policy][simplified]") {
    mm::SimplifiedPolicyInference policy;

    SECTION("Normal market observation") {
        float obs[8] = {0.2f, 0.1f, 0.0f, 0.0f, 0.5f, 0.0f, 2.0f, 0.3f};
        mm::QuoteAction action = policy.act(obs);

        REQUIRE(std::isfinite(action.delta_bid));
        REQUIRE(std::isfinite(action.delta_ask));
        REQUIRE(std::isfinite(action.size_frac));
        REQUIRE(action.delta_bid >= 0.5f);
        REQUIRE(action.delta_ask >= 0.5f);
        REQUIRE(action.size_frac > 0.0f);
        REQUIRE(action.size_frac <= 1.0f);
    }

    SECTION("Zero inventory produces symmetric base quotes") {
        float obs_zero[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        mm::QuoteAction action = policy.act(obs_zero);

        REQUIRE(action.delta_bid == 1.0f);
        REQUIRE(action.delta_ask == 1.0f);
    }

    SECTION("Long inventory increases bid distance and decreases ask distance") {
        // High positive inventory_skew (long inventory: skew = +0.8)
        float obs_long[8] = {0.0f, 0.0f, 0.8f, 0.8f, 0.0f, 0.0f, 1.0f, 0.0f};
        mm::QuoteAction action = policy.act(obs_long);

        // delta_bid should widen (> 1.0) to discourage buying
        // delta_ask should tighten (< 1.0) to encourage selling
        REQUIRE(action.delta_bid > 1.0f);
        REQUIRE(action.delta_ask < 1.0f);
        REQUIRE(action.delta_bid > action.delta_ask);
    }

    SECTION("Short inventory increases ask distance and decreases bid distance") {
        // High negative inventory_skew (short inventory: skew = -0.8)
        float obs_short[8] = {0.0f, 0.0f, -0.8f, 0.8f, 0.0f, 0.0f, 1.0f, 0.0f};
        mm::QuoteAction action = policy.act(obs_short);

        // delta_ask should widen (> 1.0) to discourage selling
        // delta_bid should tighten (< 1.0) to encourage buying
        REQUIRE(action.delta_ask > 1.0f);
        REQUIRE(action.delta_bid < 1.0f);
        REQUIRE(action.delta_ask > action.delta_bid);
    }

    SECTION("NaN or Inf observation triggers safe fallback") {
        float obs_nan[8] = {0.0f, 0.0f, NAN, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        mm::QuoteAction action = policy.act(obs_nan);

        REQUIRE(action.delta_bid == mm::PolicyInference::kFallbackDeltaBid);
        REQUIRE(action.delta_ask == mm::PolicyInference::kFallbackDeltaAsk);
        REQUIRE(action.size_frac == mm::PolicyInference::kFallbackSizeFrac);
    }
}

TEST_CASE("PolicyInference fallback on null / invalid inputs (Req 9.2)", "[policy][fallback]") {
    mm::PolicyInference policy;

    SECTION("Null observation") {
        mm::QuoteAction action = policy.act(nullptr);
        REQUIRE(action.delta_bid == mm::PolicyInference::kFallbackDeltaBid);
        REQUIRE(action.delta_ask == mm::PolicyInference::kFallbackDeltaAsk);
        REQUIRE(action.size_frac == mm::PolicyInference::kFallbackSizeFrac);
    }

    SECTION("Invalid model reload gracefully fails and leaves state safe (Req 6.8, 9.7)") {
        REQUIRE_FALSE(policy.reload("non_existent_model_file.pt"));
        REQUIRE_FALSE(policy.is_loaded());

        // Still evaluates safely using fallback
        float obs[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        mm::QuoteAction action = policy.act(obs);
        REQUIRE(std::isfinite(action.delta_bid));
        REQUIRE(std::isfinite(action.delta_ask));
        REQUIRE(std::isfinite(action.size_frac));
    }
}

TEST_CASE("SHA-256 computation verifies file integrity (Req 11.8)", "[policy][sha256]") {
    // Create temporary file
    std::string test_file = "test_sha256_temp.txt";
    {
        std::ofstream out(test_file);
        out << "hello market maker";
    }

    std::string hash = mm::PolicyInference::compute_sha256(test_file);
    REQUIRE(!hash.empty());
    REQUIRE(hash.length() == 64);

    // Empty for non-existent file
    REQUIRE(mm::PolicyInference::compute_sha256("missing_file.txt").empty());

    // Clean up
    std::remove(test_file.c_str());
}
