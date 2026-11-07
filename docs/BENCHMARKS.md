# Benchmarking Methodology & Performance Engineering

This document describes the benchmarking methodology, timing instrumentation, latency considerations, and performance profiling across the **LOB.0** engine.

---

## 1. Latency Budgets & Measurement Strategy

In high-frequency algorithmic market making, execution delays translate directly to adverse selection risk (stale quotes getting picked off by informed traders).

### Latency Budget Allocation (Target: $\le 800\text{ ns}$)
| Pipeline Stage | Target Budget | Mechanism |
| :--- | :---: | :--- |
| **LOB Event Ingestion** | $\le 200\text{ ns}$ | $O(\log N)$ price level lookup via balanced BST, $O(1)$ order cancellation hash map (plus $O(\log M)$ if price level empties) |
| **Feature Extraction** | $\le 100\text{ ns}$ | Zero-heap sliding-window sum and sum-of-squares accumulators, $O(1)$ circular buffer, stack-allocated observation vector |
| **Policy Evaluation** | $\le 400\text{ ns}$ | Zero-copy TorchScript inference (`torch::NoGradGuard`, `from_blob`) or analytical lookup |
| **Quote Generation & Risk** | $\le 100\text{ ns}$ | Discrete integer arithmetic, atomic alpha signal read, minimum spread clamp |

---

## 2. Timing Instrumentation (`PerfStats`)

Timing measurements in C++ are tracked using low-overhead high-resolution clocks:
```cpp
auto t0 = std::chrono::high_resolution_clock::now();
// Hot path quoting logic
auto t1 = std::chrono::high_resolution_clock::now();
int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
```

The `PerfStats` structure maintains thread-safe atomic counters with relaxed memory orderings (`std::memory_order_relaxed`) on the hot path:
- `events_processed`: Total order book messages processed.
- `quotes_generated`: Total quote cycles executed.
- `max_latency_ns`: Global maximum quote cycle latency.
- `p99_latency_ns`: 99th-percentile latency sampled over sliding windows of 10,000 events.
- `latency_breaches`: Count of cycles exceeding the configured `latency_budget_ns`.

---

## 3. Cache Line Alignment & Data Structure Packing

Modern x86-64 CPUs load memory from RAM into CPU L1/L2/L3 caches in discrete 64-byte chunks (cache lines).

- **Cacheline Alignment:** `PriceLevel` is forced to 64 bytes using `alignas(64)` and explicit byte padding. This prevents an individual price level struct from straddling a 64-byte cache line boundary. However, in `LobEngine`, `PriceLevel` structs reside as nodes within `std::map` (allocated on the heap), meaning traversals incur pointer chasing rather than contiguous memory streaming. Furthermore, `alignas(64)` does not automatically eliminate multi-threaded false sharing unless price levels are partitioned into distinct, non-shared cachelines across CPU cores. Contiguous cache locality is realized in `BookSnapshot` (`std::array<PriceLevel, 10>`).
- **OrderEvent Packing:** `OrderEvent` is packed to exactly 32 bytes (four 8-byte slots), allowing two full order events to fit into a single 64-byte L1 cache line.

---

## 4. Reproducing Benchmarks

To run the standalone C++ backtest and latency reporter:
```bash
# Build standalone backtest runner
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target mm_backtest

# Run backtest with simulated L2 tick stream
./build/src/cpp/mm_backtest
```

### Interpretation of Results
When evaluating benchmarks, differentiate between:
1. **LOB Simulation Throughput:** Event messages processed per second by `LobEngine::process` without quote generation.
2. **End-to-End Quoting Throughput:** Full cycle throughput including feature extraction, inference, risk clamping, and quote dispatch.
3. **Hardware Context:** Latency figures measured on consumer laptops without CPU core isolation (`isolcpus`) or RT-Linux kernels will exhibit occasional jitter due to OS scheduler interrupts.

