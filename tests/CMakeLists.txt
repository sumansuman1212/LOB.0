cmake_minimum_required(VERSION 3.22)

# ---------------------------------------------------------------------------
# Test executable (mm_tests)
# Uses self-contained header CatchShim in include/catch2/catch_test_macros.hpp
# ---------------------------------------------------------------------------
add_executable(mm_tests
    test_placeholder.cpp
    test_data_models.cpp
    test_lob_engine.cpp
    test_feature_pipeline.cpp
    test_policy_inference.cpp
    test_quote_generator.cpp
    test_tick_replayer.cpp
    test_strategy_framework.cpp
)

target_link_libraries(mm_tests
    PRIVATE
        mm_core
)

target_include_directories(mm_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(mm_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if(MSVC)
  target_compile_options(mm_tests PRIVATE /EHsc)
endif()

# Register tests with CTest
include(CTest)
add_test(NAME mm_core_tests COMMAND mm_tests)
