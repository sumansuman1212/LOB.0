cmake_minimum_required(VERSION 3.22)
project(mm_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build type default
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Global compiler warnings & MSVC exception unwinding
# ---------------------------------------------------------------------------
if(MSVC)
  add_compile_options(/EHsc /W4 /wd4324)
  add_compile_definitions(NOMINMAX)
else()
  add_compile_options(-Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Find dependencies (optional with graceful fallback)
# ---------------------------------------------------------------------------

# libtorch (optional — if present, enables neural inference via USE_TORCH)
find_package(Torch QUIET)
if(NOT Torch_FOUND)
  if(EXISTS "/opt/libtorch/share/cmake/Torch/TorchConfig.cmake")
    list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
    find_package(Torch QUIET)
  endif()
endif()
if(NOT Torch_FOUND)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE PYTHON_TORCH_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(PYTHON_TORCH_PATH)
      list(APPEND CMAKE_PREFIX_PATH "${PYTHON_TORCH_PATH}")
      find_package(Torch QUIET)
    endif()
  endif()
endif()

if(Torch_FOUND)
  message(STATUS "Torch found: ${Torch_VERSION}")
  add_compile_definitions(USE_TORCH)
else()
  message(STATUS "Torch not found. Building with standalone analytical fallback.")
endif()

# pybind11 (optional — enables Python C++ extension)
find_package(pybind11 2.11 QUIET)
if(pybind11_FOUND)
  message(STATUS "pybind11 found: ${pybind11_VERSION}")
else()
  message(STATUS "pybind11 not found. Skipping Python C++ module build.")
endif()

# Catch2 (self-contained header shim generated into build directory)
set(CATCH_SHIM_CONTENT [==[
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
]==])

file(WRITE "${CMAKE_BINARY_DIR}/generated/catch2/catch_test_macros.hpp" "${CATCH_SHIM_CONTENT}")

# ---------------------------------------------------------------------------
# mm_core static library
# ---------------------------------------------------------------------------
add_library(mm_core STATIC
    src/cpp/feature_pipeline.cpp
    src/cpp/lob_engine.cpp
    src/cpp/policy_inference.cpp
    src/cpp/quote_generator.cpp
    src/cpp/tick_replayer.cpp
    src/cpp/strategy_framework.cpp
)

target_include_directories(mm_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(mm_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

if(Torch_FOUND)
  target_link_libraries(mm_core PUBLIC ${TORCH_LIBRARIES})
  target_include_directories(mm_core PUBLIC ${TORCH_INCLUDE_DIRS})
endif()

# ---------------------------------------------------------------------------
# mm_backtest standalone runner
# ---------------------------------------------------------------------------
add_executable(mm_backtest
    src/cpp/mm_backtest.cpp
)
target_link_libraries(mm_backtest PRIVATE mm_core)
target_include_directories(mm_backtest PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(mm_backtest PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# ---------------------------------------------------------------------------
# verify_native_inference standalone integration runner
# ---------------------------------------------------------------------------
add_executable(verify_native_inference
    src/cpp/verify_native_inference.cpp
)
target_link_libraries(verify_native_inference PRIVATE mm_core)
target_include_directories(verify_native_inference PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(verify_native_inference PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# ---------------------------------------------------------------------------
# mm_tests unit and property test runner
# ---------------------------------------------------------------------------
add_executable(mm_tests
    tests/cpp/test_data_models.cpp
    tests/cpp/test_lob_engine.cpp
    tests/cpp/test_feature_pipeline.cpp
    tests/cpp/test_policy_inference.cpp
    tests/cpp/test_quote_generator.cpp
    tests/cpp/test_tick_replayer.cpp
    tests/cpp/test_strategy_framework.cpp
)

target_link_libraries(mm_tests PRIVATE mm_core)
target_include_directories(mm_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
)
set_target_properties(mm_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

include(CTest)
enable_testing()
add_test(NAME mm_core_tests COMMAND mm_tests)

# ---------------------------------------------------------------------------
# Python extension module (pybind11)
# ---------------------------------------------------------------------------
if(pybind11_FOUND)
  pybind11_add_module(mm MODULE
      src/cpp/mm_bindings.cpp
  )

  target_link_libraries(mm PRIVATE mm_core)
  target_include_directories(mm PRIVATE ${CMAKE_SOURCE_DIR}/include)
  set_target_properties(mm PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
  )
endif()
