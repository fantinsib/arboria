
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>


#include "split_strategy/threshold/random_threshold.h"
#include "dataset/dataset.h"

using arboria::split_strategy::random_threshold;
using arboria::DataSet;

DataSet make_sep_dataset() {
    std::vector<float> X{
        0, 0, 0,
        2, 0, 1,
        4, 1, 0,
        5, 10, 10,
        11, 10, 10,
        10, 11, 9
    };
    std::vector<float> y{0, 0, 0, 1, 1, 1};
    return DataSet(X, y, 6, 3);
}

TEST_CASE("Random Threshold : basic usage"){

    DataSet data = make_sep_dataset();
    std::mt19937 rng;
    size_t col = 1;
    std::vector<int> idx{0,1,2,3,4,5};
    int n_random_split = 4;
    std::vector<float> t = random_threshold(idx, col, data, n_random_split,rng);

    REQUIRE(t.size() == 4);

    for (float x : t){
        REQUIRE(0< x);
        REQUIRE(11> x);
    }

}

TEST_CASE("Random Threshold : basic usage with selected index"){

    DataSet data = make_sep_dataset();
    std::mt19937 rng;
    size_t col = 0;
    std::vector<int> idx{1,2,3};
    int n_random_split = 4;
    std::vector<float> t = random_threshold(idx, col, data, n_random_split,rng);

    REQUIRE(t.size() == 4);

    for (float x : t){
        REQUIRE(2< x);
        REQUIRE(5> x);
    }

}

TEST_CASE("Random Threshold : randomness"){

    DataSet data = make_sep_dataset();
    std::mt19937 rng(1);
    size_t col = 1;
    std::vector<int> idx{0,1,2,3,4,5};
    int n_random_split = 5;
    std::vector<float> t1 = random_threshold(idx, col, data, n_random_split,rng);
    std::vector<float> t2 = random_threshold(idx, col, data, n_random_split,rng);

    REQUIRE(t1.size() == 5);
    REQUIRE(t2.size() == 5);

    for (size_t i = 0; i < t1.size(); i++){
        REQUIRE(t1[i] != t2[i]);

    }
}

TEST_CASE("Random Threshold : determinism"){

    DataSet data = make_sep_dataset();
    std::mt19937 rng1(1);
    std::mt19937 rng2(1);
    size_t col = 1;
    std::vector<int> idx{0,1,2,3,4,5};
    int n_random_split = 5;
    std::vector<float> t1 = random_threshold(idx, col, data, n_random_split,rng1);
    std::vector<float> t2 = random_threshold(idx, col, data, n_random_split,rng2);

    REQUIRE(t1.size() == 5);
    REQUIRE(t2.size() == 5);

    for (size_t i = 0; i < t1.size(); i++){
        REQUIRE(t1[i] == t2[i]);

    }
}