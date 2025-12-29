##################### ROOT CMAKE ###################
cmake_minimum_required(VERSION 3.16)
project(arboria LANGUAGES CXX)
include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Main executable ----
add_executable(arboria
    src/main.cpp
    src/dataset/dataset.cpp
)

# On ajoute uniquement des DOSSIERS ici
target_include_directories(arboria PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

# ---- Catch2 ----
include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2
)

FetchContent_MakeAvailable(Catch2)

add_subdirectory(tests)
