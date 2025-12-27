cmake_minimum_required(VERSION 3.16)
project(arboria LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(arboria 
src/main.cpp
src/dataset/dataset.cpp)

target_include_directories(arboria PRIVATE src
src/dataset/dataset.h
src/split_criterion/gini.h)
