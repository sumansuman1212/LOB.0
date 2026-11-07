// tests/test_tick_replayer.cpp
//
// Unit tests for TickReplayer and SPSCQueue.
//
// Requirements: 8.6, 9.6, 10.8, 11.7

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "mm/tick_replayer.hpp"

TEST_CASE("SPSCQueue: Lock-free push and pop (Req 10.8)", "[replayer][queue]") {
    mm::SPSCQueue<mm::OrderEvent, 16> queue;

    REQUIRE(queue.empty());
    REQUIRE(queue.size() == 0);

    mm::OrderEvent ev1{100, 1, 20000, 10, mm::OrderEvent::Type::Add, mm::Side::Bid, {0, 0}};
    mm::OrderEvent ev2{200, 2, 20005, 10, mm::OrderEvent::Type::Add, mm::Side::Ask, {0, 0}};

    REQUIRE(queue.push(ev1));
    REQUIRE(queue.push(ev2));
    REQUIRE(queue.size() == 2);
    REQUIRE_FALSE(queue.empty());

    mm::OrderEvent out1{};
    mm::OrderEvent out2{};

    REQUIRE(queue.pop(out1));
    REQUIRE(out1.order_id == 1);
    REQUIRE(out1.price == 20000);

    REQUIRE(queue.pop(out2));
    REQUIRE(out2.order_id == 2);
    REQUIRE(out2.price == 20005);

    REQUIRE(queue.empty());
}

TEST_CASE("TickReplayer: Chronological ordering and pump (Req 8.6, 11.7)", "[replayer][chronological]") {
    mm::TickReplayer replayer;

    std::vector<mm::OrderEvent> events = {
        {300, 3, 20010, 5, mm::OrderEvent::Type::Add, mm::Side::Ask, {0, 0}},
        {100, 1, 20000, 10, mm::OrderEvent::Type::Add, mm::Side::Bid, {0, 0}},
        {200, 2, 20005, 20, mm::OrderEvent::Type::Add, mm::Side::Ask, {0, 0}},
    };

    replayer.load_events(std::move(events));
    REQUIRE(replayer.total_events() == 3);
    REQUIRE(replayer.has_events());

    mm::OrderEvent ev{};
    REQUIRE(replayer.next_event(ev));
    REQUIRE(ev.timestamp_ns == 100);
    REQUIRE(ev.order_id == 1);

    REQUIRE(replayer.next_event(ev));
    REQUIRE(ev.timestamp_ns == 200);
    REQUIRE(ev.order_id == 2);

    REQUIRE(replayer.next_event(ev));
    REQUIRE(ev.timestamp_ns == 300);
    REQUIRE(ev.order_id == 3);

    REQUIRE_FALSE(replayer.next_event(ev));
    REQUIRE_FALSE(replayer.has_events());
}

TEST_CASE("TickReplayer: Gap detection > 1 second (Req 9.6)", "[replayer][gap]") {
    mm::TickReplayer replayer;

    bool gap_detected = false;
    int64_t reported_prev = 0;
    int64_t reported_next = 0;

    replayer.on_gap([&](int64_t prev, int64_t next) {
        gap_detected = true;
        reported_prev = prev;
        reported_next = next;
    });

    int64_t ts1 = 1'000'000'000LL;
    int64_t ts2 = 3'000'000'000LL; // 2 seconds later (> 1s)

    std::vector<mm::OrderEvent> events = {
        {ts1, 1, 20000, 10, mm::OrderEvent::Type::Add, mm::Side::Bid, {0, 0}},
        {ts2, 2, 20005, 10, mm::OrderEvent::Type::Add, mm::Side::Ask, {0, 0}},
    };

    replayer.load_events(std::move(events));
    mm::OrderEvent ev{};
    REQUIRE(replayer.next_event(ev));
    REQUIRE(replayer.next_event(ev));

    REQUIRE(gap_detected);
    REQUIRE(reported_prev == ts1);
    REQUIRE(reported_next == ts2);
}
