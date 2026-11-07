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

# Catch2 (optional — vendored header shim in include/catch2 is used by default)
find_package(Catch2 3 QUIET)
if(Catch2_FOUND)
  message(STATUS "Catch2 v3 found: ${Catch2_VERSION}")
else()
  message(STATUS "Catch2 not found on system. Using self-contained header shim in include/catch2.")
endif()

# ---------------------------------------------------------------------------
# Global include directories
# ---------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src/cpp)
add_subdirectory(tests)

# ---------------------------------------------------------------------------
# mm_core static library configuration
# ---------------------------------------------------------------------------
if(Torch_FOUND)
  target_link_libraries(mm_core PUBLIC ${TORCH_LIBRARIES})
  target_include_directories(mm_core PUBLIC ${TORCH_INCLUDE_DIRS})
endif()

# On MSVC set /EHsc required by C++ exceptions
if(MSVC)
  target_compile_options(mm_core PRIVATE /EHsc /W4)
else()
  target_compile_options(mm_core PRIVATE -Wall -Wextra)
endif()
