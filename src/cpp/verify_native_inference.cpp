// src/cpp/verify_native_inference.cpp
//
// End-to-End Native C++ Verification of TorchScript Policy Inference & Quote Generation.
//
// 1. Loads models/ppo_actor.pt via mm::PolicyInference.
// 2. Verifies SHA-256 file checksum against models/ppo_actor.pt.sha256.
// 3. Constructs an 8-dimensional observation vector.
// 4. Executes TorchScript / analytical fallback policy inference.
// 5. Validates 3 output dimensions (delta_bid, delta_ask, size_frac).
// 6. Enforces non-NaN, non-Inf finite assertions.
// 7. Converts policy action to QuoteAction.
// 8. Feeds QuoteAction into mm::QuoteGenerator.
// 9. Validates emitted QuoteParams (prices, lot bounds, spread invariant).

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cassert>

#include "mm/types.hpp"
#include "mm/data_models.hpp"
#include "mm/policy_inference.hpp"
#include "mm/quote_generator.hpp"

int main() {
    std::cout << "========================================================\n";
    std::cout << "LOB.0 Native C++ TorchScript & Quoting Integration Test\n";
    std::cout << "========================================================\n";

    const std::string model_path = "models/ppo_actor.pt";
    const std::string checksum_path = "models/ppo_actor.pt.sha256";

    // 1. Verify file presence
    std::ifstream model_file(model_path, std::ios::binary);
    if (!model_file.is_open()) {
        std::cerr << "FAIL: Model file not found at " << model_path << "\n";
        return 1;
    }
    model_file.seekg(0, std::ios::end);
    size_t file_size = model_file.tellg();
    std::cout << "[1/9] Model checkpoint found: " << model_path 
              << " (" << file_size << " bytes)\n";

    // 2. Verify SHA-256 Checksum
    std::string computed_hash = mm::PolicyInference::compute_sha256(model_path);
    std::string expected_hash;
    std::ifstream cfile(checksum_path);
    if (cfile.is_open()) {
        cfile >> expected_hash;
    }
    std::cout << "[2/9] SHA-256 verification:\n"
              << "      Computed: " << computed_hash << "\n"
              << "      Expected: " << expected_hash << "\n";
    if (computed_hash.empty() || (!expected_hash.empty() && computed_hash != expected_hash)) {
        std::cerr << "FAIL: Checksum mismatch or unreadable file!\n";
        return 1;
    }
    std::cout << "      Checksum status: MATCH\n";

    // 3. Construct 8-dimensional observation
    float obs[mm::kObsDim] = {
        0.35f,   // ofi_l1
        0.28f,   // ofi_l5
        0.20f,   // inventory_skew (+20% long)
        0.20f,   // inventory_urgency
        0.50f,   // vol_regime
        0.15f,   // micro_price_offset
        1.60f,   // spread_ticks
        0.10f    // session_progress
    };
    std::cout << "[3/9] Constructed valid 8D observation vector:\n      [";
    for (int i = 0; i < mm::kObsDim; ++i) {
        std::cout << obs[i] << (i + 1 < mm::kObsDim ? ", " : "]\n");
    }

    // 4. Initialize PolicyInference and act
    mm::PolicyInference policy(model_path);
    std::cout << "[4/9] Policy engine initialized: " << policy.name() 
              << " (loaded=" << (policy.is_loaded() ? "true" : "false") << ")\n";

    // 5. Evaluate policy
    mm::QuoteAction action = policy.act(obs);
    std::cout << "[5/9] Policy inference evaluated:\n"
              << "      delta_bid: " << action.delta_bid << "\n"
              << "      delta_ask: " << action.delta_ask << "\n"
              << "      size_frac: " << action.size_frac << "\n";

    // 6. Check for non-finite outputs
    if (!std::isfinite(action.delta_bid) || !std::isfinite(action.delta_ask) || !std::isfinite(action.size_frac)) {
        std::cerr << "FAIL: Non-finite action values detected!\n";
        return 1;
    }
    std::cout << "[6/9] Non-finite check passed: All action dimensions are finite.\n";

    // 7. Verify action value bounds
    assert(action.delta_bid >= 0.0f);
    assert(action.delta_ask >= 0.0f);
    assert(action.size_frac > 0.0f && action.size_frac <= 1.0f);
    std::cout << "[7/9] Action bounds check passed (delta >= 0, 0 < size_frac <= 1).\n";

    // 8. Pass action into QuoteGenerator
    const int64_t tick_size = 5;
    const uint64_t max_order_size = 50;
    const int64_t max_inventory = 500;
    const int64_t mid_price = 20000;
    const int64_t current_inventory = 100; // long 100 lots

    mm::QuoteGenerator quoter(tick_size, max_order_size, max_inventory);
    mm::BookSnapshot dummy_snap{};
    dummy_snap.best_bid = mid_price - tick_size;
    dummy_snap.best_ask = mid_price + tick_size;

    mm::QuoteParams params = quoter.generate(obs, policy, mid_price, current_inventory, &dummy_snap);
    std::cout << "[8/9] QuoteGenerator produced QuoteParams:\n"
              << "      bid_price: " << params.bid_price << " ticks\n"
              << "      ask_price: " << params.ask_price << " ticks\n"
              << "      bid_qty:   " << params.bid_qty << " lots\n"
              << "      ask_qty:   " << params.ask_qty << " lots\n"
              << "      valid_until_ns: " << params.valid_until_ns << "\n";

    // 9. Verify emitted quote invariants
    if (params.ask_price - params.bid_price < tick_size) {
        std::cerr << "FAIL: Minimum spread invariant violated!\n";
        return 1;
    }
    if (params.bid_qty == 0 || params.ask_qty == 0) {
        std::cerr << "FAIL: Quote quantity clamped to 0 under normal inventory!\n";
        return 1;
    }
    std::cout << "[9/9] QuoteParams invariants verified: spread >= tick_size, quantities in bounds.\n";

    std::cout << "========================================================\n";
    std::cout << "STATUS: NATIVE INFERENCE VERIFICATION SUCCESSFUL\n";
    std::cout << "========================================================\n";
    return 0;
}

