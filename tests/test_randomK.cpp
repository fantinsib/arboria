/*
                                              TESTS FOR RANDOMK
*/

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>
#include <vector>
#include <random>


#include "split_strategy/feature_selection/randomK/randomK.h"

using arboria::feature_selection::randomK;

TEST_CASE("RandomK : basic usage") {

    std::vector<int> features {0,1,2,3,4};
    int mtry = 3;
    std::mt19937 rng(0);

    std::vector<int> sub_features = randomK(features, mtry, rng);

    REQUIRE(sub_features.size() == size_t(3)); //check for correct number of indices returned

    std::sort(sub_features.begin(), sub_features.end());
    REQUIRE(sub_features[0]>= 0); //no negative indices returned
    REQUIRE(sub_features[sub_features.size()-1]<= 4); //no indices outside the initial scope returned

    auto i = std::adjacent_find(sub_features.begin(), sub_features.end());
    REQUIRE(i == sub_features.end()); //only unique values
}


TEST_CASE("RandomK : mtry == features.size()") {

    std::vector<int> features {0,1,2,3,4};
    int mtry = 5;
    std::mt19937 rng(0);

    std::vector<int> sub_features = randomK(features, mtry, rng);

    REQUIRE(sub_features.size() == size_t(5)); //check for correct number of indices returned

    std::sort(sub_features.begin(), sub_features.end());
    REQUIRE(sub_features[0]>= 0); //no negative indices returned
    REQUIRE(sub_features[sub_features.size()-1]<= 4); //no indices outside the initial scope returned

    auto i = std::adjacent_find(sub_features.begin(), sub_features.end());
    REQUIRE(i == sub_features.end()); //only unique values
}


TEST_CASE("RandomK : errors - no features"){

    std::vector<int> features {};
    int mtry = 1;
    std::mt19937 rng(0);

    REQUIRE_THROWS_AS(randomK(features, mtry, rng), std::invalid_argument);
}

TEST_CASE("RandomK : errors - mtry>n_features"){

    std::vector<int> features {0,1,2,3,4,5};
    int mtry = 7;
    std::mt19937 rng(0);

    REQUIRE_THROWS_AS(randomK(features, mtry, rng), std::invalid_argument);
}

TEST_CASE("RandomK : errors - mtry <= 0"){

    std::vector<int> features {0,1,2,3,4,5};
    std::mt19937 rng(0);

    SECTION("mtry == 0"){
    int mtry = 0;
    REQUIRE_THROWS_AS(randomK(features, mtry, rng), std::invalid_argument);
    }

    SECTION("mtry < 0"){
    int mtry = -1;
    REQUIRE_THROWS_AS(randomK(features, mtry, rng), std::invalid_argument);
    }
}