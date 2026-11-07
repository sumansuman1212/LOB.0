cmake_minimum_required(VERSION 3.22)

# ---------------------------------------------------------------------------
# Core library — deliberately excludes pybind11 entry points.  This keeps the
# engine independently linkable by native tests and executables.
# ---------------------------------------------------------------------------
add_library(mm_core STATIC
    feature_pipeline.cpp
    lob_engine.cpp
    policy_inference.cpp
    quote_generator.cpp
    tick_replayer.cpp
    strategy_framework.cpp
)

target_include_directories(mm_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# Torch libraries are linked in the top-level CMakeLists.txt after the
# add_subdirectory(src/cpp) call so that find_package(Torch) is already run.
# C++17 required for std::pmr, std::atomic, etc.
set_target_properties(mm_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# ---------------------------------------------------------------------------
# mm_backtest standalone runner
# ---------------------------------------------------------------------------
add_executable(mm_backtest
    mm_backtest.cpp
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
    verify_native_inference.cpp
)
target_link_libraries(verify_native_inference PRIVATE mm_core)
target_include_directories(verify_native_inference PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(verify_native_inference PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# ---------------------------------------------------------------------------
# Python extension — built when pybind11 is available
# ---------------------------------------------------------------------------
if(pybind11_FOUND)
  pybind11_add_module(mm MODULE
      mm_bindings.cpp
  )

  target_link_libraries(mm PRIVATE mm_core)
  target_include_directories(mm PRIVATE ${CMAKE_SOURCE_DIR}/include)
  set_target_properties(mm PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
  )
endif()
