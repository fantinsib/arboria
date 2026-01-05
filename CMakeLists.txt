##################### ROOT CMAKE ###################
cmake_minimum_required(VERSION 3.16)
project(arboria LANGUAGES CXX)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Core lib ----
add_library(arboria_lib
    src/dataset/dataset.cpp
    src/node/node.cpp
    src/split_strategy/splitter.cpp
    src/tree/DecisionTree/DecisionTree.cpp
)

target_include_directories(arboria_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

# ---- Main exe ----
add_executable(arboria
    src/main.cpp
)

target_link_libraries(arboria PRIVATE arboria_lib)

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
