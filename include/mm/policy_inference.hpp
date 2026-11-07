#pragma once

// include/mm/policy_inference.hpp
//
// PolicyInference — C++ wrapper around a frozen TorchScript actor model
// with zero-copy inference, SHA-256 integrity validation, atomic hot-reload,
// and safe analytical fallback.
//
// Requirements: 3.10, 6.5, 6.6, 6.7, 6.8, 6.9, 9.2, 9.4, 9.7, 10.5, 11.8, 13.5

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include "mm/data_models.hpp"  // QuoteAction, MarketFeatures
#include "mm/types.hpp"        // kObsDim, kActionDim

namespace mm {

// ---------------------------------------------------------------------------
// IPolicyInference — Abstract interface for policy evaluation
// ---------------------------------------------------------------------------
class IPolicyInference {
public:
    virtual ~IPolicyInference() = default;

    /// Synchronous inference mapping 8D observation to 3D QuoteAction.
    /// Must be noexcept and never throw.
    virtual QuoteAction act(const float* obs) noexcept = 0;

    /// Return policy identifier/name
    virtual const char* name() const noexcept = 0;
};

// ---------------------------------------------------------------------------
// SimplifiedPolicyInference — Fast analytical fallback (Req 9.4)
//
// Heuristic mapping from OFI, inventory skew, and volatility to quote skew
// without running neural network inference. Used as a safe fallback when
// neural model fails or latency breaches exceed 1%.
// ---------------------------------------------------------------------------
class SimplifiedPolicyInference : public IPolicyInference {
public:
    QuoteAction act(const float* obs) noexcept override;
    const char* name() const noexcept override { return "SimplifiedPolicyInference"; }
};

// ---------------------------------------------------------------------------
// PolicyInference — Neural TorchScript inference wrapper (Req 6.5–6.9)
// ---------------------------------------------------------------------------
class PolicyInference : public IPolicyInference {
public:
    /// Default constructor with fallback policy active
    PolicyInference() noexcept;

    /// Construct and load TorchScript model from file (Req 6.5, 6.9, 11.8).
    /// @param model_path Path to the exported .pt file.
    explicit PolicyInference(const std::string& model_path);

    ~PolicyInference() override;

    // Non-copyable, movable
    PolicyInference(const PolicyInference&) = delete;
    PolicyInference& operator=(const PolicyInference&) = delete;
    PolicyInference(PolicyInference&&) noexcept;
    PolicyInference& operator=(PolicyInference&&) noexcept;

    /// Synchronous inference.
    /// Wraps body in try-catch; on any exception or NaN input returns safe fallback:
    /// {delta_bid=2.0f, delta_ask=2.0f, size_frac=0.1f} (Req 9.2, 13.5).
    QuoteAction act(const float* obs) noexcept override;

    /// Hot-swap model atomically at runtime without downtime (Req 6.7, 6.8, 9.7).
    /// Returns true on successful load, false on verification/load failure.
    bool reload(const std::string& new_model_path) noexcept;

    /// Check if a neural model is currently loaded
    bool is_loaded() const noexcept;

    /// Compute SHA-256 hex string for a file (Req 11.8)
    static std::string compute_sha256(const std::string& filepath) noexcept;

    /// Safe fallback action constants (Req 9.2)
    static constexpr float kFallbackDeltaBid = 2.0f;
    static constexpr float kFallbackDeltaAsk = 2.0f;
    static constexpr float kFallbackSizeFrac = 0.1f;

    const char* name() const noexcept override { return "TorchScriptPolicyInference"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    SimplifiedPolicyInference fallback_engine_;
};

} // namespace mm
