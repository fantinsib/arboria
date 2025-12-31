/*
                                              TESTS FOR COUNTS
*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>  
#include <stdexcept>
#include <vector>
#include <cmath>
#include <utility>
#include "helpers/helpers.h"


using namespace arboria::helpers;

TEST_CASE("count_classes : Basic Usage") {

SECTION("Count from float vector"){
    std::vector<float> y{0,0,1,1,0};
    std::pair<int, int> count = count_classes(y);
    REQUIRE(count.first == 2);
    REQUIRE(count.second == 3);
    }

SECTION("Count from int vector"){
    std::vector<int> y{1,0,1,1,0};
    std::pair<int, int> count = count_classes(y);
    REQUIRE(count.first == 3);
    REQUIRE(count.second == 2);
    }

SECTION("Count from span"){
    std::vector<int> y{1,0,1,1,0, 0};
    std::vector<int> rows{0,2, 3, 5};
    std::span<const int> s(rows);
    std::pair<int, int> count = count_classes(s, y);
    REQUIRE(count.first == 3);
    REQUIRE(count.second == 1);
    }

}



TEST_CASE("count_classes : Errors") {

SECTION("Non binary label - int overload"){
    std::vector<int> y{1,0,3,1,0};
    REQUIRE_THROWS_AS(count_classes(y), std::invalid_argument);
    }

SECTION("Non binary label - span overload"){
    std::vector<int> y{1,3,1,1,0, 0};
    std::vector<int> rows{0,-1, 3, 6};
    std::span<const int> s(rows);
    REQUIRE_THROWS_AS(count_classes(s, y), std::out_of_range);
    }

SECTION("Index out of range"){
    std::vector<int> y{1,0,1,1,0, 0};
    std::vector<int> rows{0,2, 3, 6};
    std::span<const int> s(rows);
    REQUIRE_THROWS_AS(count_classes(s, y), std::out_of_range);
    }

SECTION("Negative index"){
    std::vector<int> y{1,0,1,1,0, 0};
    std::vector<int> rows{0,-1, 3, 6};
    std::span<const int> s(rows);
    REQUIRE_THROWS_AS(count_classes(s, y), std::out_of_range);
    }

}