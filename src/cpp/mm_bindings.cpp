// src/cpp/mm_bindings.cpp
//
// pybind11 bindings — exposes C++ mm:: components to Python as the `mm` module.
//
// Exposed Python API:
//   mm.LobEngine       — wraps mm::LobEngine
//   mm.FeaturePipeline — wraps mm::FeaturePipeline
//   mm.OrderEvent      — wraps mm::OrderEvent (Python dict ↔ struct)
//   mm.MarketFeatures  — wraps mm::MarketFeatures (read-only field access)
//   mm.BookSnapshot    — wraps mm::BookSnapshot (bid/ask array views)
//   mm.step(lob, fp, event_dict, net_position) → (changed: bool, obs: np.ndarray | None)
//   mm.reset(lob, fp)  → None
//
// Boundary validation (Req 6.10, 11.6):
//   - Observation or action arrays containing NaN → ValueError
//   - Action array shape != (3,)                 → ValueError
//   - Observation array shape != (8,)            → ValueError
//   - Action size_frac not in (0, 1]             → ValueError
//
// Requirements: 6.1, 6.2, 6.10, 11.6

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <array>

#include "mm/lob_engine.hpp"       // LobEngine, OrderEntry
#include "mm/feature_pipeline.hpp" // FeaturePipeline
#include "mm/lob_types.hpp"        // OrderEvent, BookSnapshot, PriceLevel
#include "mm/data_models.hpp"      // MarketFeatures, QuoteAction
#include "mm/types.hpp"            // Side, kObsDim, kActionDim

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

/// Check a float32 numpy array for NaN values.
/// Raises ValueError on violation (Req 11.6).
void validate_no_nan(const py::array_t<float>& arr, const char* name) {
    auto buf = arr.unchecked<1>();
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        if (std::isnan(buf(i)) || std::isinf(buf(i))) {
            throw std::invalid_argument(
                std::string(name) + ": array contains NaN or Inf at index " +
                std::to_string(i));
        }
    }
}

/// Validate that a 1-D float32 array has the expected shape.
/// Raises ValueError on violation (Req 11.6).
void validate_shape(const py::array_t<float>& arr,
                    py::ssize_t expected_size,
                    const char* name) {
    if (arr.ndim() != 1 || arr.shape(0) != expected_size) {
        throw std::invalid_argument(
            std::string(name) + ": expected shape (" +
            std::to_string(expected_size) + ",) but got shape (" +
            std::to_string(arr.ndim() == 1 ? arr.shape(0) : -1) + ",)");
    }
}

/// Validate an action array: shape (3,), no NaN/Inf, size_frac in (0, 1].
/// Raises ValueError on violation (Req 6.10, 11.6).
void validate_action(const py::array_t<float>& action) {
    validate_shape(action, mm::kActionDim, "action");
    validate_no_nan(action, "action");

    auto buf = action.unchecked<1>();
    float size_frac = buf(2);
    if (size_frac <= 0.0f || size_frac > 1.0f) {
        throw std::invalid_argument(
            "action: size_frac must be in (0, 1], got " +
            std::to_string(size_frac));
    }
}

/// Validate an observation array: shape (8,), no NaN/Inf.
/// Raises ValueError on violation (Req 6.10, 11.6).
void validate_observation(const py::array_t<float>& obs) {
    validate_shape(obs, mm::kObsDim, "observation");
    validate_no_nan(obs, "observation");
}

/// Build an OrderEvent from a Python dict.
/// Raises ValueError for invalid or missing fields.
mm::OrderEvent dict_to_order_event(const py::dict& d) {
    mm::OrderEvent ev{};

    // Required fields
    if (!d.contains("timestamp_ns"))
        throw std::invalid_argument("event dict missing 'timestamp_ns'");
    if (!d.contains("order_id"))
        throw std::invalid_argument("event dict missing 'order_id'");
    if (!d.contains("price"))
        throw std::invalid_argument("event dict missing 'price'");
    if (!d.contains("qty"))
        throw std::invalid_argument("event dict missing 'qty'");
    if (!d.contains("type"))
        throw std::invalid_argument("event dict missing 'type'");
    if (!d.contains("side"))
        throw std::invalid_argument("event dict missing 'side'");

    ev.timestamp_ns = d["timestamp_ns"].cast<int64_t>();
    ev.order_id     = d["order_id"].cast<uint64_t>();
    ev.price        = d["price"].cast<int64_t>();
    ev.qty          = d["qty"].cast<uint32_t>();

    // Parse type string
    std::string type_str = d["type"].cast<std::string>();
    if      (type_str == "Add")         ev.type = mm::OrderEvent::Type::Add;
    else if (type_str == "Cancel")      ev.type = mm::OrderEvent::Type::Cancel;
    else if (type_str == "Modify")      ev.type = mm::OrderEvent::Type::Modify;
    else if (type_str == "Fill")        ev.type = mm::OrderEvent::Type::Fill;
    else if (type_str == "PartialFill") ev.type = mm::OrderEvent::Type::PartialFill;
    else throw std::invalid_argument(
            "event dict 'type' must be one of: Add, Cancel, Modify, Fill, PartialFill");

    // Parse side string
    std::string side_str = d["side"].cast<std::string>();
    if      (side_str == "Bid" || side_str == "bid") ev.side = mm::Side::Bid;
    else if (side_str == "Ask" || side_str == "ask") ev.side = mm::Side::Ask;
    else throw std::invalid_argument(
            "event dict 'side' must be 'Bid' or 'Ask'");

    return ev;
}

/// Convert the current MarketFeatures snapshot to a float32 numpy array of shape (8,).
py::array_t<float> features_to_numpy(const mm::MarketFeatures& feats) {
    py::array_t<float> result(mm::kObsDim);
    auto buf = result.mutable_unchecked<1>();
    buf(0) = feats.ofi_l1;
    buf(1) = feats.ofi_l5;
    buf(2) = feats.inventory_skew;
    buf(3) = feats.inventory_urgency;
    buf(4) = feats.vol_regime;
    buf(5) = feats.micro_price_offset;
    buf(6) = feats.spread_ticks;
    buf(7) = feats.session_progress;
    return result;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(mm, m) {
    m.doc() = R"pbdoc(
        mm — C++ market-making engine bindings
        ======================================
        Exposes LobEngine and FeaturePipeline to Python for use in the
        Gym-compatible training environment (LobEnv) and backtesting.

        Requirements: 6.1, 6.2, 6.10, 11.6
    )pbdoc";

    // -----------------------------------------------------------------------
    // OrderEvent::Type enum
    // -----------------------------------------------------------------------
    py::enum_<mm::OrderEvent::Type>(m, "OrderEventType")
        .value("Add",         mm::OrderEvent::Type::Add)
        .value("Cancel",      mm::OrderEvent::Type::Cancel)
        .value("Modify",      mm::OrderEvent::Type::Modify)
        .value("Fill",        mm::OrderEvent::Type::Fill)
        .value("PartialFill", mm::OrderEvent::Type::PartialFill)
        .export_values();

    // -----------------------------------------------------------------------
    // Side enum
    // -----------------------------------------------------------------------
    py::enum_<mm::Side>(m, "Side")
        .value("Bid", mm::Side::Bid)
        .value("Ask", mm::Side::Ask)
        .export_values();

    // -----------------------------------------------------------------------
    // OrderEvent struct
    //
    // Exposed so that Python callers can construct events programmatically.
    // Also constructible from a dict via mm.step()'s event_dict parameter.
    // -----------------------------------------------------------------------
    py::class_<mm::OrderEvent>(m, "OrderEvent",
        R"pbdoc(
            32-byte order event descriptor.

            Fields
            ------
            timestamp_ns : int   nanoseconds since epoch
            order_id     : int   exchange-assigned order ID
            price        : int   price in integer ticks (must be > 0)
            qty          : int   unsigned quantity in lots (> 0 for Add/Fill)
            type         : OrderEventType
            side         : Side
        )pbdoc")
        .def(py::init<>())
        .def_readwrite("timestamp_ns", &mm::OrderEvent::timestamp_ns)
        .def_readwrite("order_id",     &mm::OrderEvent::order_id)
        .def_readwrite("price",        &mm::OrderEvent::price)
        .def_readwrite("qty",          &mm::OrderEvent::qty)
        .def_readwrite("type",         &mm::OrderEvent::type)
        .def_readwrite("side",         &mm::OrderEvent::side)
        .def("__repr__", [](const mm::OrderEvent& ev) {
            return "<OrderEvent order_id=" + std::to_string(ev.order_id) +
                   " price=" + std::to_string(ev.price) +
                   " qty="   + std::to_string(ev.qty)   + ">";
        });

    // -----------------------------------------------------------------------
    // PriceLevel struct (read-only from Python)
    // -----------------------------------------------------------------------
    py::class_<mm::PriceLevel>(m, "PriceLevel",
        R"pbdoc(
            64-byte cache-aligned price level.

            Fields
            ------
            price        : int   price in ticks
            total_qty    : int   total resting quantity at this level
            order_count  : int   number of individual orders
        )pbdoc")
        .def_readonly("price",       &mm::PriceLevel::price)
        .def_readonly("total_qty",   &mm::PriceLevel::total_qty)
        .def_readonly("order_count", &mm::PriceLevel::order_count)
        .def("__repr__", [](const mm::PriceLevel& pl) {
            return "<PriceLevel price=" + std::to_string(pl.price) +
                   " qty=" + std::to_string(pl.total_qty) + ">";
        });

    // -----------------------------------------------------------------------
    // BookSnapshot struct
    //
    // bids / asks are exposed as Python lists of PriceLevel so the caller
    // can iterate the ladder without copying large structs.
    // -----------------------------------------------------------------------
    py::class_<mm::BookSnapshot>(m, "BookSnapshot",
        R"pbdoc(
            Point-in-time capture of the top-10 bid and ask price levels.

            Attributes
            ----------
            bids         : list[PriceLevel]   top-10 bids, bids[0] = best bid
            asks         : list[PriceLevel]   top-10 asks, asks[0] = best ask
            best_bid     : int   highest active bid price in ticks
            best_ask     : int   lowest  active ask price in ticks
            timestamp_ns : int   snapshot timestamp
        )pbdoc")
        .def(py::init<>())
        .def_readonly("best_bid",     &mm::BookSnapshot::best_bid)
        .def_readonly("best_ask",     &mm::BookSnapshot::best_ask)
        .def_readonly("timestamp_ns", &mm::BookSnapshot::timestamp_ns)
        // Expose bids as a Python list of PriceLevel objects (read-only views)
        .def_property_readonly("bids", [](const mm::BookSnapshot& snap) {
            py::list lst;
            for (const auto& lvl : snap.bids) lst.append(lvl);
            return lst;
        })
        .def_property_readonly("asks", [](const mm::BookSnapshot& snap) {
            py::list lst;
            for (const auto& lvl : snap.asks) lst.append(lvl);
            return lst;
        })
        .def("__repr__", [](const mm::BookSnapshot& snap) {
            return "<BookSnapshot best_bid=" + std::to_string(snap.best_bid) +
                   " best_ask=" + std::to_string(snap.best_ask) + ">";
        });

    // -----------------------------------------------------------------------
    // MarketFeatures struct (read-only from Python)
    // -----------------------------------------------------------------------
    py::class_<mm::MarketFeatures>(m, "MarketFeatures",
        R"pbdoc(
            8-dimensional observation vector produced by FeaturePipeline.

            All fields are normalized to approximately [-1, 1] with a
            5-sigma hard clamp (|obs[i]| <= 5.0).

            Fields
            ------
            ofi_l1            : float  level-1 order-flow imbalance
            ofi_l5            : float  top-5 order-flow imbalance
            inventory_skew    : float  net_position / max_position
            inventory_urgency : float  |net_position| / max_position
            vol_regime        : float  z-scored rolling realized volatility
            micro_price_offset: float  micro-price offset from mid
            spread_ticks      : float  bid-ask spread in ticks
            session_progress  : float  fraction of session elapsed
        )pbdoc")
        .def(py::init<>())
        .def_readwrite("ofi_l1",             &mm::MarketFeatures::ofi_l1)
        .def_readwrite("ofi_l5",             &mm::MarketFeatures::ofi_l5)
        .def_readwrite("inventory_skew",     &mm::MarketFeatures::inventory_skew)
        .def_readwrite("inventory_urgency",  &mm::MarketFeatures::inventory_urgency)
        .def_readwrite("vol_regime",         &mm::MarketFeatures::vol_regime)
        .def_readwrite("micro_price_offset", &mm::MarketFeatures::micro_price_offset)
        .def_readwrite("spread_ticks",       &mm::MarketFeatures::spread_ticks)
        .def_readwrite("session_progress",   &mm::MarketFeatures::session_progress)
        .def("to_numpy", [](const mm::MarketFeatures& feats) {
            return features_to_numpy(feats);
        },
        R"pbdoc(
            Convert to a float32 numpy array of shape (8,).
        )pbdoc")
        .def("__repr__", [](const mm::MarketFeatures& f) {
            return "<MarketFeatures ofi_l1=" + std::to_string(f.ofi_l1) +
                   " inv_skew=" + std::to_string(f.inventory_skew) +
                   " vol=" + std::to_string(f.vol_regime) + ">";
        });

    // -----------------------------------------------------------------------
    // LobEngine class (Req 6.1)
    // -----------------------------------------------------------------------
    py::class_<mm::LobEngine>(m, "LobEngine",
        R"pbdoc(
            Price-time priority limit order book for a single instrument.

            Parameters
            ----------
            tick_size : int   minimum price increment in ticks (must be > 0)
            max_depth : int   maximum levels captured in snapshot (default 10)
        )pbdoc")
        .def(py::init<int64_t, uint32_t>(),
             py::arg("tick_size"),
             py::arg("max_depth") = static_cast<uint32_t>(10))
        .def("process",
             [](mm::LobEngine& lob, const mm::OrderEvent& ev) -> bool {
                 return lob.process(ev);
             },
             py::arg("event"),
             R"pbdoc(
                 Process one order event.

                 Returns True if the book state changed (best bid or ask moved).
                 Returns False if the event was rejected or produced no change.
             )pbdoc")
        .def("process_dict",
             [](mm::LobEngine& lob, const py::dict& d) -> bool {
                 mm::OrderEvent ev = dict_to_order_event(d);
                 return lob.process(ev);
             },
             py::arg("event_dict"),
             R"pbdoc(
                 Process one order event supplied as a Python dict.

                 The dict must contain: timestamp_ns, order_id, price, qty,
                 type (str: 'Add'|'Cancel'|'Modify'|'Fill'|'PartialFill'),
                 side (str: 'Bid'|'Ask').
             )pbdoc")
        .def("snapshot",
             [](const mm::LobEngine& lob) {
                 mm::BookSnapshot snap{};
                 lob.snapshot(snap);
                 return snap;
             },
             R"pbdoc(
                 Return a BookSnapshot of the current top-10 bid/ask levels.
             )pbdoc")
        .def("best_bid",  &mm::LobEngine::best_bid,
             "Highest active bid price in ticks (INT64_MIN if empty).")
        .def("best_ask",  &mm::LobEngine::best_ask,
             "Lowest  active ask price in ticks (INT64_MAX if empty).")
        .def("mid_price", &mm::LobEngine::mid_price,
             "(best_bid + best_ask) / 2; returns 0 when either side is empty.")
        .def("spread",    &mm::LobEngine::spread,
             "best_ask - best_bid; returns 0 when either side is empty.")
        .def("reset",     &mm::LobEngine::reset,
             "Clear all book state to empty.")
        .def("__repr__", [](const mm::LobEngine& lob) {
            return "<LobEngine best_bid=" + std::to_string(lob.best_bid()) +
                   " best_ask=" + std::to_string(lob.best_ask()) + ">";
        });

    // -----------------------------------------------------------------------
    // FeaturePipeline class (Req 6.1)
    // -----------------------------------------------------------------------
    py::class_<mm::FeaturePipeline>(m, "FeaturePipeline",
        R"pbdoc(
            Converts BookSnapshot + net position into a fixed-length
            observation vector for the PPO agent.

            Parameters
            ----------
            max_position       : int   hard position limit (used for normalization)
            vol_window         : int   rolling window size for realized vol (default 60)
            ofi_levels         : int   top-K levels for OFI computation (default 5)
            session_duration_s : int   session length in seconds (default 23400)
        )pbdoc")
        .def(py::init<int32_t, int32_t, int32_t, int64_t>(),
             py::arg("max_position"),
             py::arg("vol_window")         = 60,
             py::arg("ofi_levels")         = 5,
             py::arg("session_duration_s") = static_cast<int64_t>(23400))
        .def("update",
             [](mm::FeaturePipeline& fp,
                const mm::BookSnapshot& snap,
                int64_t net_position) {
                 fp.update(snap, net_position);
             },
             py::arg("snapshot"),
             py::arg("net_position"),
             R"pbdoc(
                 Update all rolling accumulators from a new book snapshot and
                 current net position.
             )pbdoc")
        .def("observe",
             [](const mm::FeaturePipeline& fp) {
                 mm::MarketFeatures out{};
                 fp.observe(out);
                 return out;
             },
             R"pbdoc(
                 Return the current MarketFeatures observation struct.
             )pbdoc")
        .def("observe_numpy",
             [](const mm::FeaturePipeline& fp) {
                 mm::MarketFeatures out{};
                 fp.observe(out);
                 return features_to_numpy(out);
             },
             R"pbdoc(
                 Return the current observation as a float32 numpy array of shape (8,).
             )pbdoc")
        .def("reset", &mm::FeaturePipeline::reset,
             "Clear all rolling window state.")
        .def("__repr__", [](const mm::FeaturePipeline& /*fp*/) {
            return "<FeaturePipeline>";
        });

    // -----------------------------------------------------------------------
    // Free function: mm.step(lob, features, event_dict, net_position)
    //
    // This is the main hot-path entry point called from LobEnv.step().
    //
    // Behaviour:
    //   1. Parse event_dict → OrderEvent (raises ValueError on malformed input)
    //   2. Call lob.process(event)
    //   3. If book changed: call features.update(snap, net_position),
    //      retrieve observation via features.observe(), return (True, obs_array)
    //   4. If book unchanged: return (False, None)
    //
    // Boundary validation (Req 6.10, 11.6):
    //   - Validates all numeric fields of the event dict before forwarding.
    //
    // Returns
    // -------
    // tuple[bool, np.ndarray | None]
    // -----------------------------------------------------------------------
    m.def("step",
        [](mm::LobEngine&      lob,
           mm::FeaturePipeline& features,
           const py::dict&     event_dict,
           int64_t             net_position)
        -> py::tuple
        {
            // Parse and validate the event from Python dict
            mm::OrderEvent ev = dict_to_order_event(event_dict);

            // Validate price and qty (Req 11.1, 11.2)
            if (ev.price <= 0) {
                throw std::invalid_argument(
                    "step: event price must be > 0, got " +
                    std::to_string(ev.price));
            }
            if ((ev.type == mm::OrderEvent::Type::Add ||
                 ev.type == mm::OrderEvent::Type::Fill ||
                 ev.type == mm::OrderEvent::Type::PartialFill) && ev.qty == 0) {
                throw std::invalid_argument(
                    "step: Add/Fill/PartialFill events must have qty > 0");
            }

            // Process event — C++ LOB decides if book changed
            bool changed = lob.process(ev);

            if (!changed) {
                return py::make_tuple(false, py::none());
            }

            // Book changed: get snapshot and update feature pipeline
            mm::BookSnapshot snap{};
            lob.snapshot(snap);
            features.update(snap, net_position);

            // Retrieve observation
            mm::MarketFeatures obs_struct{};
            features.observe(obs_struct);

            py::array_t<float> obs_arr = features_to_numpy(obs_struct);
            return py::make_tuple(true, obs_arr);
        },
        py::arg("lob"),
        py::arg("features"),
        py::arg("event_dict"),
        py::arg("net_position"),
        R"pbdoc(
            Process one market event through the LOB and feature pipeline.

            Parameters
            ----------
            lob          : LobEngine
            features     : FeaturePipeline
            event_dict   : dict
                Must contain: timestamp_ns (int), order_id (int), price (int),
                qty (int), type (str: 'Add'|'Cancel'|'Modify'|'Fill'|'PartialFill'),
                side (str: 'Bid'|'Ask').
            net_position : int  current signed net position

            Returns
            -------
            (changed: bool, obs: np.ndarray[float32, shape=(8,)] | None)
                If the LOB state changed, returns (True, observation_array).
                If the event produced no state change, returns (False, None).

            Raises
            ------
            ValueError
                If event_dict is malformed, price <= 0, or qty == 0 for
                Add/Fill events.
        )pbdoc");

    // -----------------------------------------------------------------------
    // Free function: mm.reset(lob, features)
    // -----------------------------------------------------------------------
    m.def("reset",
        [](mm::LobEngine& lob, mm::FeaturePipeline& features) {
            lob.reset();
            features.reset();
        },
        py::arg("lob"),
        py::arg("features"),
        R"pbdoc(
            Reset both the LOB engine and feature pipeline to their initial state.

            This clears all book state, order history, rolling windows, and
            session tracking. Equivalent to starting a fresh episode.
        )pbdoc");

    // -----------------------------------------------------------------------
    // Free function: mm.validate_action(action)
    //
    // Validates an action array before it is used (Req 6.10, 11.6).
    // Raises ValueError on: shape != (3,), NaN/Inf, size_frac not in (0, 1].
    // -----------------------------------------------------------------------
    m.def("validate_action",
        [](const py::array_t<float>& action) {
            validate_action(action);
        },
        py::arg("action"),
        R"pbdoc(
            Validate an action array before forwarding to C++.

            Raises ValueError if:
              - shape != (3,)
              - any element is NaN or Inf
              - size_frac (action[2]) is not in (0, 1]

            Requirements: 6.10, 11.6
        )pbdoc");

    // -----------------------------------------------------------------------
    // Free function: mm.validate_observation(obs)
    //
    // Validates an observation array (Req 6.10, 11.6).
    // Raises ValueError on: shape != (8,), NaN/Inf.
    // -----------------------------------------------------------------------
    m.def("validate_observation",
        [](const py::array_t<float>& obs) {
            validate_observation(obs);
        },
        py::arg("obs"),
        R"pbdoc(
            Validate an observation array.

            Raises ValueError if:
              - shape != (8,)
              - any element is NaN or Inf

            Requirements: 6.10, 11.6
        )pbdoc");

    // -----------------------------------------------------------------------
    // Module-level constants matching the C++ compile-time values
    // -----------------------------------------------------------------------
    m.attr("OBS_DIM")    = mm::kObsDim;    // 8
    m.attr("ACTION_DIM") = mm::kActionDim; // 3
}
