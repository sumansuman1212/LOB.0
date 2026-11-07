// src/cpp/policy_inference.cpp
//
// Implementation of PolicyInference, SimplifiedPolicyInference, and SHA-256 verification.
//
// Requirements: 3.10, 6.5, 6.6, 6.7, 6.8, 6.9, 9.2, 9.4, 9.7, 10.5, 11.8, 13.5

#include "mm/policy_inference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef USE_TORCH
#include <torch/script.h>
#include <torch/torch.h>
#endif

namespace mm {

// ---------------------------------------------------------------------------
// Portable SHA-256 Implementation (No external OpenSSL dependency required)
// ---------------------------------------------------------------------------
namespace {

struct SHA256Context {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[64];
};

inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

inline uint32_t choose(uint32_t e, uint32_t f, uint32_t g) {
    return (e & f) ^ (~e & g);
}

inline uint32_t majority(uint32_t a, uint32_t b, uint32_t c) {
    return (a & b) ^ (a & c) ^ (b & c);
}

inline uint32_t sig0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
inline uint32_t sig1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
inline uint32_t theta0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
inline uint32_t theta1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

void sha256_transform(uint32_t state[8], const uint8_t data[64]) {
    uint32_t W[64];
    for (int i = 0; i < 16; ++i) {
        W[i] = (static_cast<uint32_t>(data[i * 4]) << 24) |
               (static_cast<uint32_t>(data[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(data[i * 4 + 2]) << 8) |
               (static_cast<uint32_t>(data[i * 4 + 3]));
    }
    for (int i = 16; i < 64; ++i) {
        W[i] = theta1(W[i - 2]) + W[i - 7] + theta0(W[i - 15]) + W[i - 16];
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + sig1(e) + choose(e, f, g) + K[i] + W[i];
        uint32_t t2 = sig0(a) + majority(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void sha256_init(SHA256Context* ctx) {
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
    ctx->count = 0;
}

void sha256_update(SHA256Context* ctx, const uint8_t* data, size_t len) {
    size_t index = (ctx->count >> 3) & 63;
    ctx->count += (len << 3);
    size_t part_len = 64 - index;

    size_t i = 0;
    if (len >= part_len) {
        std::memcpy(&ctx->buffer[index], data, part_len);
        sha256_transform(ctx->state, ctx->buffer);
        for (i = part_len; i + 63 < len; i += 64) {
            sha256_transform(ctx->state, &data[i]);
        }
        index = 0;
    }
    std::memcpy(&ctx->buffer[index], &data[i], len - i);
}

void sha256_final(SHA256Context* ctx, uint8_t digest[32]) {
    uint8_t bits[8];
    for (int i = 0; i < 8; ++i) {
        bits[i] = static_cast<uint8_t>((ctx->count >> ((7 - i) * 8)) & 0xFF);
    }
    size_t index = (ctx->count >> 3) & 63;
    size_t pad_len = (index < 56) ? (56 - index) : (120 - index);
    static const uint8_t PADDING[64] = { 0x80 };
    sha256_update(ctx, PADDING, pad_len);
    sha256_update(ctx, bits, 8);
    for (int i = 0; i < 8; ++i) {
        digest[i * 4] = static_cast<uint8_t>((ctx->state[i] >> 24) & 0xFF);
        digest[i * 4 + 1] = static_cast<uint8_t>((ctx->state[i] >> 16) & 0xFF);
        digest[i * 4 + 2] = static_cast<uint8_t>((ctx->state[i] >> 8) & 0xFF);
        digest[i * 4 + 3] = static_cast<uint8_t>(ctx->state[i] & 0xFF);
    }
}

} // namespace

std::string PolicyInference::compute_sha256(const std::string& filepath) noexcept {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }

    SHA256Context ctx;
    sha256_init(&ctx);

    std::vector<uint8_t> buffer(65536);
    while (file.read(reinterpret_cast<char*>(buffer.data()), buffer.size()) || file.gcount() > 0) {
        sha256_update(&ctx, buffer.data(), static_cast<size_t>(file.gcount()));
    }

    uint8_t hash[32];
    sha256_final(&ctx, hash);

    std::ostringstream oss;
    for (int i = 0; i < 32; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return oss.str();
}

template <typename T>
constexpr const T& clamp_val(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

// ---------------------------------------------------------------------------
// SimplifiedPolicyInference Implementation (Req 9.4)
// ---------------------------------------------------------------------------
QuoteAction SimplifiedPolicyInference::act(const float* obs) noexcept {
    if (!obs) {
        return {PolicyInference::kFallbackDeltaBid, PolicyInference::kFallbackDeltaAsk, PolicyInference::kFallbackSizeFrac};
    }

    // Verify all 8 dimensions are finite
    for (int i = 0; i < kObsDim; ++i) {
        if (!std::isfinite(obs[i])) {
            return {PolicyInference::kFallbackDeltaBid, PolicyInference::kFallbackDeltaAsk, PolicyInference::kFallbackSizeFrac};
        }
    }

    float ofi_l1 = obs[0];
    float inventory_skew = obs[2];
    float inventory_urgency = obs[3];
    float vol_regime = obs[4];

    // Adaptive quote skew formula:
    // If long inventory (skew > 0), widen bid distance (delta_bid higher) to avoid buying
    // and tighten ask distance (delta_ask lower) to encourage selling.
    // If short inventory (skew < 0), widen ask distance (delta_ask higher) to avoid selling
    // and tighten bid distance (delta_bid lower) to encourage buying.
    float delta_bid = 1.0f + 1.5f * std::max(0.0f, inventory_skew) - 0.5f * std::max(0.0f, -inventory_skew) - 0.3f * ofi_l1 + 0.2f * std::max(0.0f, vol_regime);
    float delta_ask = 1.0f + 1.5f * std::max(0.0f, -inventory_skew) - 0.5f * std::max(0.0f, inventory_skew) + 0.3f * ofi_l1 + 0.2f * std::max(0.0f, vol_regime);
    float size_frac = clamp_val(0.2f * (1.0f - 0.5f * inventory_urgency), 0.05f, 1.0f);

    delta_bid = clamp_val(delta_bid, 0.5f, 20.0f);
    delta_ask = clamp_val(delta_ask, 0.5f, 20.0f);

    return {delta_bid, delta_ask, size_frac};
}

// ---------------------------------------------------------------------------
// PolicyInference::Impl
// ---------------------------------------------------------------------------
struct PolicyInference::Impl {
#ifdef USE_TORCH
    std::shared_ptr<torch::jit::Module> module;
#endif
    std::string current_model_path;
    std::string current_checksum;
    bool loaded{false};
};

PolicyInference::~PolicyInference() = default;

PolicyInference::PolicyInference() noexcept
    : impl_(std::make_unique<Impl>()) {
}

PolicyInference::PolicyInference(const std::string& model_path)
    : impl_(std::make_unique<Impl>()) {
    reload(model_path);
}

PolicyInference::PolicyInference(PolicyInference&&) noexcept = default;
PolicyInference& PolicyInference::operator=(PolicyInference&&) noexcept = default;

bool PolicyInference::is_loaded() const noexcept {
    return impl_ && impl_->loaded;
}

bool PolicyInference::reload(const std::string& new_model_path) noexcept {
    try {
        if (new_model_path.empty()) {
            return false;
        }

        // Verify SHA-256 against .sha256 file if present (Req 6.9, 11.8)
        std::string computed_hash = compute_sha256(new_model_path);
        if (computed_hash.empty()) {
            std::cerr << "[PolicyInference] Model file not found: " << new_model_path << std::endl;
            return false;
        }

        std::string checksum_file = new_model_path + ".sha256";
        std::ifstream cfile(checksum_file);
        if (cfile.is_open()) {
            std::string expected_hash;
            cfile >> expected_hash;
            if (!expected_hash.empty() && expected_hash != computed_hash) {
                std::cerr << "[PolicyInference] Checksum mismatch! Expected "
                          << expected_hash << ", got " << computed_hash << std::endl;
                return false;
            }
        }

#ifdef USE_TORCH
        auto new_module = std::make_shared<torch::jit::Module>(torch::jit::load(new_model_path, torch::kCPU));
        new_module->eval();
        std::atomic_store(&impl_->module, new_module);
#endif
        impl_->current_model_path = new_model_path;
        impl_->current_checksum = computed_hash;
        impl_->loaded = true;

        std::cout << "[PolicyInference] Successfully loaded model: " << new_model_path
                  << " (SHA-256: " << computed_hash.substr(0, 8) << "...)" << std::endl;
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[PolicyInference] Error loading TorchScript model: " << ex.what() << std::endl;
        return false;
    }
}

QuoteAction PolicyInference::act(const float* obs) noexcept {
    // 1. Guard against null or non-finite inputs (Req 9.2, 13.5)
    if (!obs) {
        std::cerr << "[PolicyInference::act] Error: Null observation pointer" << std::endl;
        return {kFallbackDeltaBid, kFallbackDeltaAsk, kFallbackSizeFrac};
    }

    for (int i = 0; i < kObsDim; ++i) {
        if (!std::isfinite(obs[i])) {
            std::cerr << "[PolicyInference::act] Warning: Non-finite feature at index " << i << ": " << obs[i] << std::endl;
            return {kFallbackDeltaBid, kFallbackDeltaAsk, kFallbackSizeFrac};
        }
    }

    // 2. If TorchScript is loaded and compiled with USE_TORCH, perform neural inference
#ifdef USE_TORCH
    if (impl_->loaded) {
        try {
            auto mod = std::atomic_load(&impl_->module);
            if (mod) {
                torch::NoGradGuard no_grad;
                auto options = torch::TensorOptions().dtype(torch::kFloat32);
                torch::Tensor input_tensor = torch::from_blob(const_cast<float*>(obs), {1, kObsDim}, options);
                
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                torch::Tensor output = mod->forward(inputs).toTensor().squeeze(0);
                auto ptr = output.data_ptr<float>();

                float delta_bid = std::clamp(ptr[0], 0.0f, 20.0f);
                float delta_ask = std::clamp(ptr[1], 0.0f, 20.0f);
                float size_frac = std::clamp(ptr[2], 0.01f, 1.0f);

                return {delta_bid, delta_ask, size_frac};
            }
        } catch (const std::exception& ex) {
            std::cerr << "[PolicyInference::act] TorchScript exception: " << ex.what() << " -> using fallback" << std::endl;
            return {kFallbackDeltaBid, kFallbackDeltaAsk, kFallbackSizeFrac};
        }
    }
#endif

    // 3. Fallback to analytical policy
    return fallback_engine_.act(obs);
}

} // namespace mm
