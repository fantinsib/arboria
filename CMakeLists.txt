##################### ROOT CMAKE ###################
cmake_minimum_required(VERSION 3.16)
project(arboria LANGUAGES CXX)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# ---- Catch2 ----
include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

# --- Tests ---
add_subdirectory(tests)


find_package(pybind11 REQUIRED)

pybind11_add_module(arboria MODULE src/bindings/bindings.cpp)

target_link_libraries(arboria PRIVATE arboria_lib)
target_compile_features(arboria PRIVATE cxx_std_20)
