##################### ROOT CMAKE ###################
cmake_minimum_required(VERSION 3.16)
project(arboria LANGUAGES CXX)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Python (needed for pybind11 module build) ----
set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# ---- Core lib ----
add_library(arboria_lib
    src/dataset/dataset.cpp
    src/node/node.cpp
    src/split_strategy/splitter.cpp
    src/tree/DecisionTree/DecisionTree.cpp
    src/split_strategy/feature_selection/randomK/randomK.cpp
    src/split_strategy/sampling/sampling.cpp
    src/tree/RandomForest/randomforest.cpp
)

target_include_directories(arboria_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

set_target_properties(arboria_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Main exe ----
add_executable(arboria_cli
    src/main.cpp
)
target_link_libraries(arboria_cli PRIVATE arboria_lib)

# ---- Dependencies ----
include(FetchContent)

# Catch2
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# pybind11 (vendorized for CI portability)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# ---- Pybind11 module ----
pybind11_add_module(arboria MODULE src/bindings/bindings.cpp)
set_target_properties(arboria PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/arboria"
)
target_link_libraries(arboria PRIVATE arboria_lib)
target_compile_features(arboria PRIVATE cxx_std_20)

# ---- Tests (optional) ----
option(ARBORIA_BUILD_TESTS "Build tests" ON)
if (ARBORIA_BUILD_TESTS)
  add_subdirectory(tests)
endif()
