/*

                                    TESTS FOR RANDOM FOREST

*/

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <vector>

#include "dataset/dataset.h"
#include "split_strategy/types/split_param.h"
#include "tree/RandomForest/randomforest.h"
#include "split_strategy/types/ParamBuilder/ParamBuilder.h"
#include "tree/TreeModel.h"

using arboria::DataSet;
using arboria::RandomForest;
using arboria::ParamBuilder;

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

    std::vector<HyperParam> bad_params = {
        HyperParam{.mtry = 2,.n_estimators = 0,  .max_depth = 3},   
        HyperParam{.mtry = 0,.n_estimators = 10,    .max_depth = 3},   
        HyperParam{.mtry = -97,.n_estimators = 10,  .max_depth = 3},   
        HyperParam{.mtry = 2,.n_estimators = 10,   .max_depth = 0},   
    };
    for (const auto& h_param : bad_params) {
        REQUIRE_THROWS_AS(RandomForest(h_param, 1), std::invalid_argument);
    }
}


TEST_CASE("RandomForest : fit then predict basic usage") {
    DataSet data = make_separable_dataset();

    SplitParam param = ParamBuilder(TreeModel::RandomForest, Gini{}, CART{}, RandomK{2});
    HyperParam h_param{2, 25, 4};
    RandomForest forest(h_param, 123);
    forest.fit(data, param);

    REQUIRE(forest.is_fitted() == true);
    REQUIRE(forest.get_estimators() == 25);
    REQUIRE(forest.get_max_features() == 2);
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
    HyperParam h_param{2, 25, 4};
    RandomForest forest(h_param, 123);
    std::vector<float> samples{0, 0, 0};

    REQUIRE_THROWS_AS(forest.predict(samples), std::invalid_argument);
}

TEST_CASE("RandomForest : predict_proba error on wrong dimension") {
    DataSet data = make_separable_dataset();
    HyperParam h_param{2, 25, 4};
    RandomForest forest(h_param, 123);
    SplitParam param = ParamBuilder(TreeModel::RandomForest, Gini{}, CART{}, RandomK{2});
    forest.fit(data, param);

    std::vector<float> bad_samples{0, 0, 0, 1};
    REQUIRE_THROWS_AS(forest.predict_proba(bad_samples), std::invalid_argument);
}

TEST_CASE("RandomForest : mtry larger than feature count") {
    DataSet data = make_separable_dataset();
    HyperParam h_param{8, 25, 4};
    RandomForest forest(h_param, 123);
    SplitParam param = ParamBuilder(TreeModel::RandomForest, Gini{}, CART{}, RandomK{10});

    REQUIRE_THROWS_AS(forest.fit(data, param), std::invalid_argument);
}

TEST_CASE("RandomForest : out_of_bag basic range") {
    DataSet data = make_separable_dataset();
    HyperParam h_param{2, 20, 4};
    RandomForest forest(h_param, 123);
    SplitParam param = ParamBuilder(TreeModel::RandomForest, Gini{}, CART{}, RandomK{2});

    forest.fit(data, param);

    float score = forest.out_of_bag(data);
    REQUIRE(score >= 0.0f);
    REQUIRE(score <= 1.0f);
}

TEST_CASE("RandomForest : out_of_bag error on empty data") {
    DataSet data = make_separable_dataset();
    HyperParam h_param{2, 20, 4};
    RandomForest forest(h_param, 123);
    SplitParam param = ParamBuilder(TreeModel::RandomForest, Gini{}, CART{}, RandomK{2});

    forest.fit(data, param);

    std::vector<float> empty_x;
    std::vector<float> empty_y;
    DataSet empty(empty_x, empty_y, 0, 0);

    REQUIRE_THROWS_AS(forest.out_of_bag(empty), std::invalid_argument);
}


