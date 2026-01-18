/*

                                    TESTS FOR RANDOM FOREST

*/

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <vector>

#include "dataset/dataset.h"
#include "tree/RandomForest/randomforest.h"

using arboria::DataSet;
using arboria::RandomForest;

namespace {

DataSet make_separable_dataset() {
    std::vector<float> X{
        0, 0, 0,
        1, 0, 1,
        0, 1, 0,
        10, 10, 10,
        11, 10, 10,
        10, 11, 9
    };
    std::vector<float> y{0, 0, 0, 1, 1, 1};
    return DataSet(X, y, 6, 3);
}

}

TEST_CASE("RandomForest : constructor validation") {
    REQUIRE_THROWS_AS(RandomForest(0, 1, 1, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(RandomForest(1, 0, 1, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(RandomForest(1, 1, 0, 1), std::invalid_argument);
}

TEST_CASE("RandomForest : fit then predict basic usage") {
    DataSet data = make_separable_dataset();

    RandomForest forest(25, 2, 4, 123);
    forest.fit(data);

    REQUIRE(forest.is_fitted() == true);
    REQUIRE(forest.get_estimators() == 25);
    REQUIRE(forest.get_mtry() == 2);
    REQUIRE(forest.get_max_depth() == 4);

    std::vector<float> samples{
        0, 0, 0,
        10, 10, 10
    };

    std::vector<float> probas = forest.predict_proba(samples);
    REQUIRE(probas.size() == size_t(2));
    for (float p : probas) {
        REQUIRE(p >= 0.0f);
        REQUIRE(p <= 1.0f);
    }

    std::vector<int> preds = forest.predict(samples);
    REQUIRE(preds.size() == size_t(2));
    REQUIRE(preds[0] == 0);
    REQUIRE(preds[1] == 1);
}

TEST_CASE("RandomForest : predict error before fit") {
    RandomForest forest(5, 2, 3, 7);
    std::vector<float> samples{0, 0, 0};

    REQUIRE_THROWS_AS(forest.predict(samples), std::invalid_argument);
}

TEST_CASE("RandomForest : predict_proba error on wrong dimension") {
    DataSet data = make_separable_dataset();
    RandomForest forest(5, 2, 3, 7);
    forest.fit(data);

    std::vector<float> bad_samples{0, 0, 0, 1};
    REQUIRE_THROWS_AS(forest.predict_proba(bad_samples), std::invalid_argument);
}

TEST_CASE("RandomForest : mtry larger than feature count") {
    DataSet data = make_separable_dataset();
    RandomForest forest(5, 10, 3, 7);

    REQUIRE_THROWS_AS(forest.fit(data), std::invalid_argument);
}

TEST_CASE("RandomForest : out_of_bag basic range") {
    DataSet data = make_separable_dataset();
    RandomForest forest(20, 2, 4, 123);
    forest.fit(data);

    float score = forest.out_of_bag(data);
    REQUIRE(score >= 0.0f);
    REQUIRE(score <= 1.0f);
}

TEST_CASE("RandomForest : out_of_bag error on empty data") {
    DataSet data = make_separable_dataset();
    RandomForest forest(5, 2, 3, 7);
    forest.fit(data);

    std::vector<float> empty_x;
    std::vector<float> empty_y;
    DataSet empty(empty_x, empty_y, 0, 0);

    REQUIRE_THROWS_AS(forest.out_of_bag(empty), std::invalid_argument);
}
