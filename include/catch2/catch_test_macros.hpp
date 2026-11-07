#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <exception>
#include <cmath>

namespace CatchShim {

struct TestCase {
    std::string name;
    std::string tags;
    std::function<void()> func;
};

inline std::vector<TestCase>& get_registry() {
    static std::vector<TestCase> registry;
    return registry;
}

inline bool register_test(const std::string& name, const std::string& tags, std::function<void()> func) {
    get_registry().push_back({name, tags, func});
    return true;
}

struct TestContext {
    int total_assertions{0};
    int passed_assertions{0};
    int failed_assertions{0};
    bool current_test_failed{false};
    std::string current_test_name;
};

inline TestContext& get_context() {
    static TestContext ctx;
    return ctx;
}

inline void log_assertion(bool success, const char* expr, const char* file, int line, bool is_require) {
    auto& ctx = get_context();
    ctx.total_assertions++;
    if (success) {
        ctx.passed_assertions++;
    } else {
        ctx.failed_assertions++;
        ctx.current_test_failed = true;
        std::cerr << "  FAILED: [" << expr << "] at " << file << ":" << line << "\n";
        if (is_require) {
            throw std::runtime_error("Assertion failed");
        }
    }
}

inline int run_all_tests() {
    auto& registry = get_registry();
    auto& ctx = get_context();
    int passed_tests = 0;
    int failed_tests = 0;

    std::cout << "======================================================\n";
    std::cout << "Running " << registry.size() << " test cases...\n";
    std::cout << "======================================================\n";

    for (const auto& tc : registry) {
        ctx.current_test_failed = false;
        ctx.current_test_name = tc.name;
        std::cout << "[RUN]  " << tc.name << " " << tc.tags << "\n";
        try {
            tc.func();
        } catch (const std::exception& e) {
            ctx.current_test_failed = true;
            std::cerr << "  EXCEPTION: " << e.what() << "\n";
        } catch (...) {
            ctx.current_test_failed = true;
            std::cerr << "  UNKNOWN EXCEPTION\n";
        }

        if (ctx.current_test_failed) {
            failed_tests++;
            std::cout << "[FAIL] " << tc.name << "\n\n";
        } else {
            passed_tests++;
            std::cout << "[PASS] " << tc.name << "\n";
        }
    }

    std::cout << "======================================================\n";
    std::cout << "Test Summary: " << passed_tests << " passed, " << failed_tests << " failed ("
              << registry.size() << " total test cases, "
              << ctx.total_assertions << " assertions)\n";
    std::cout << "======================================================\n";

    return (failed_tests == 0) ? 0 : 1;
}

} // namespace CatchShim

#define CATCH_CONCAT_INNER(a, b) a##b
#define CATCH_CONCAT(a, b) CATCH_CONCAT_INNER(a, b)

#define TEST_CASE(name, tags) \
    static void CATCH_CONCAT(catch_test_fn_, __LINE__)(); \
    static const bool CATCH_CONCAT(catch_test_reg_, __LINE__) = \
        ::CatchShim::register_test(name, tags, &CATCH_CONCAT(catch_test_fn_, __LINE__)); \
    static void CATCH_CONCAT(catch_test_fn_, __LINE__)()

#define SECTION(name) if (true)

#define STATIC_REQUIRE(...) static_assert(__VA_ARGS__, #__VA_ARGS__)

#define REQUIRE(expr) \
    do { \
        ::CatchShim::log_assertion((expr), #expr, __FILE__, __LINE__, true); \
    } while (false)

#define REQUIRE_FALSE(expr) \
    do { \
        ::CatchShim::log_assertion(!(expr), "!(" #expr ")", __FILE__, __LINE__, true); \
    } while (false)

#define CHECK(expr) \
    do { \
        ::CatchShim::log_assertion((expr), #expr, __FILE__, __LINE__, false); \
    } while (false)

#define CHECK_FALSE(expr) \
    do { \
        ::CatchShim::log_assertion(!(expr), "!(" #expr ")", __FILE__, __LINE__, false); \
    } while (false)
