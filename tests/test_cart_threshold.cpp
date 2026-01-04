/*

                TESTS CART THRESHOLDS 

*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>  
#include <vector>

#include "split_strategy/threshold/cart_threshold.h"
#include "dataset/dataset.h"

using arboria::split_strategy::cart_threshold;
using arboria::DataSet;

TEST_CASE("Basic Usage") {

    std::vector<float> x{1,2,3,
                        4,5,6,
                        7, 8 ,9,
                        10, 11,12};
    std::vector<float> y{0,0,1,1};
    
    DataSet data(x, y, 4, 3);

    std::vector<int> idx {3,1};
    std::span<int> s(idx);
    std::vector<float> t = cart_threshold(s, 0, data);

    REQUIRE(t.size() == 1);
    REQUIRE(t[0] == Catch::Approx(7.f));
}

TEST_CASE("cart_threshold - throws if idx has less than 2 elements") {

    std::vector<float> x{
        1,2,3,
        4,5,6
    };
    std::vector<float> y{0,1};

    DataSet data(x, y, 2, 3);

    std::vector<int> idx1{0};          
    std::span<const int> s1(idx1);
    REQUIRE_THROWS_AS(cart_threshold(s1, 0, data), std::invalid_argument);
    
    std::vector<int> idx0{};           
    std::span<const int> s0(idx0);
    REQUIRE_THROWS_AS(cart_threshold(s0, 0, data), std::invalid_argument);
}

TEST_CASE("cart_threshold - throws if col is negative") {

    std::vector<float> x{
        1,2,3,
        4,5,6
    };
    std::vector<float> y{0,1};

    DataSet data(x, y, 2, 3);

    std::vector<int> idx{0,1};
    std::span<const int> s(idx);

    REQUIRE_THROWS_AS(cart_threshold(s, -1, data), std::invalid_argument);
}

TEST_CASE("cart_threshold - throws if col >= n_cols") {

    std::vector<float> x{
        1,2,3,
        4,5,6
    };
    std::vector<float> y{0,1};

    DataSet data(x, y, 2, 3);

    std::vector<int> idx{0,1};
    std::span<const int> s(idx);

    REQUIRE_THROWS_AS(cart_threshold(s, 3, data), std::invalid_argument); // valid cols: 0,1,2
    REQUIRE_THROWS_AS(cart_threshold(s, 999, data), std::invalid_argument);
}

TEST_CASE("cart_threshold - outputs size is idx.size() - 1") {

    std::vector<float> x{
        1,2,3,
        4,5,6,
        7,8,9
    };
    std::vector<float> y{0,1,0};

    DataSet data(x, y, 3, 3);

    std::vector<int> idx{2,0,1}; 
    std::span<const int> s(idx);

    auto t = cart_threshold(s, 1, data);
    REQUIRE(t.size() == 2);
}

TEST_CASE("cart_threshold - thresholds are computed on sorted values even if idx is unordered") {

    std::vector<float> x{
        10, 0, 0,
         2, 0, 0,
         7, 0, 0
    };
    std::vector<float> y{0,1,0};

    DataSet data(x, y, 3, 3);

    std::vector<int> idx{0,2,1};   
    std::span<const int> s(idx);


    auto t = cart_threshold(s, 0, data);

    REQUIRE(t.size() == 2);
    REQUIRE(t[0] == Catch::Approx(4.5f));
    REQUIRE(t[1] == Catch::Approx(8.5f));
}

TEST_CASE("cart_threshold - handles duplicate feature values") {

    std::vector<float> x{
        1, 0, 0,
        1, 0, 0,
        3, 0, 0
    };
    std::vector<float> y{0,1,0};

    DataSet data(x, y, 3, 3);

    std::vector<int> idx{2,0,1};
    std::span<const int> s(idx);

    auto t = cart_threshold(s, 0, data);

    REQUIRE(t.size() == 2);
    REQUIRE(t[0] == Catch::Approx(1.0f));
    REQUIRE(t[1] == Catch::Approx(2.0f));
}


