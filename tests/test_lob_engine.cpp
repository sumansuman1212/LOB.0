#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

#include "mm/lob_engine.hpp"

namespace {

mm::OrderEvent event(int64_t timestamp, uint64_t id, int64_t price,
                     uint32_t qty, mm::OrderEvent::Type type, mm::Side side) {
    return mm::OrderEvent{timestamp, id, price, qty, type, side, {0, 0}};
}

mm::BookSnapshot snapshot(const mm::LobEngine& book) {
    mm::BookSnapshot result{};
    book.snapshot(result);
    return result;
}

void seed_two_sided_book(mm::LobEngine& book) {
    REQUIRE(book.process(event(1, 1, 100, 10, mm::OrderEvent::Type::Add, mm::Side::Bid)));
    REQUIRE(book.process(event(2, 2, 105, 20, mm::OrderEvent::Type::Add, mm::Side::Ask)));
}

} // namespace

TEST_CASE("duplicate Add is rejected without orphaning resting quantity",
          "[lob][task-3.1][duplicate]") {
    mm::LobEngine book{1};
    REQUIRE(book.process(event(1, 42, 100, 7, mm::OrderEvent::Type::Add, mm::Side::Bid)));
    const auto before = snapshot(book);

    REQUIRE_FALSE(book.process(event(2, 42, 99, 11, mm::OrderEvent::Type::Add, mm::Side::Bid)));

    const auto after = snapshot(book);
    REQUIRE(after.best_bid == before.best_bid);
    REQUIRE(after.bids[0].price == 100);
    REQUIRE(after.bids[0].total_qty == 7);
    REQUIRE(after.bids[0].order_count == 1);

    REQUIRE(book.process(event(3, 42, 0, 0, mm::OrderEvent::Type::Cancel, mm::Side::Bid)));
    REQUIRE(book.best_bid() == std::numeric_limits<int64_t>::min());
}

TEST_CASE("rejected crossed Modify is transaction-safe, including timestamp",
          "[lob][task-3.1][transaction]") {
    mm::LobEngine book{1};
    seed_two_sided_book(book);
    const auto before = snapshot(book);

    REQUIRE_FALSE(book.process(event(3, 1, 105, 10, mm::OrderEvent::Type::Modify, mm::Side::Bid)));

    const auto after = snapshot(book);
    REQUIRE(after.timestamp_ns == before.timestamp_ns);
    REQUIRE(after.best_bid == 100);
    REQUIRE(after.best_ask == 105);
    REQUIRE(after.bids[0].total_qty == 10);
    REQUIRE(after.bids[0].order_count == 1);
    REQUIRE(after.asks[0].total_qty == 20);
    REQUIRE(after.asks[0].order_count == 1);

    // Timestamp 2 remains valid only when the rejected modification did not
    // advance the monotonic timestamp guard.
    REQUIRE_FALSE(book.process(event(2, 3, 99, 1, mm::OrderEvent::Type::Add, mm::Side::Bid)));
    REQUIRE(book.process(event(3, 3, 99, 1, mm::OrderEvent::Type::Add, mm::Side::Bid)) == false);
    const auto final = snapshot(book);
    REQUIRE(final.bids[0].total_qty == 10);
    REQUIRE(final.bids[1].price == 99);
    REQUIRE(final.bids[1].total_qty == 1);
    REQUIRE(final.bids[1].order_count == 1);
}

TEST_CASE("order and price-level representations remain consistent across mutations",
          "[lob][task-3.1][consistency]") {
    mm::LobEngine book{1};
    seed_two_sided_book(book);
    REQUIRE_FALSE(book.process(event(3, 3, 100, 5, mm::OrderEvent::Type::Add, mm::Side::Bid)));
    REQUIRE_FALSE(book.process(event(4, 1, 100, 4, mm::OrderEvent::Type::PartialFill, mm::Side::Bid)));

    auto state = snapshot(book);
    REQUIRE(state.bids[0].price == 100);
    REQUIRE(state.bids[0].total_qty == 11);
    REQUIRE(state.bids[0].order_count == 2);

    REQUIRE_FALSE(book.process(event(5, 3, 0, 0, mm::OrderEvent::Type::Cancel, mm::Side::Bid)));
    state = snapshot(book);
    REQUIRE(state.bids[0].total_qty == 6);
    REQUIRE(state.bids[0].order_count == 1);

    REQUIRE(book.process(event(6, 1, 99, 6, mm::OrderEvent::Type::Modify, mm::Side::Bid)));
    state = snapshot(book);
    REQUIRE(state.best_bid == 99);
    REQUIRE(state.bids[0].price == 99);
    REQUIRE(state.bids[0].total_qty == 6);
    REQUIRE(state.bids[0].order_count == 1);
    REQUIRE(state.best_ask == 105);
}
