/*

                                    TESTS FOR RANDOM FOREST

*/


#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <stdexcept>
#include <vector>
#include <iostream>

#include "dataset/dataset.h"
#include "split_strategy/types/split_param.h"
#include "tree/RandomForest/randomforest.h"
#include "split_strategy/types/ParamBuilder/ParamBuilder.h"
#include "tree/TreeModel.h"

#include "test_access.h"

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


TEST_CASE("RandomForest : constructor validation - values not passed") {

    std::vector<HyperParam> incomp_params = {
        HyperParam{.mtry = 2,.n_estimators = 10},   
        HyperParam{.mtry = 2,.max_depth = 3},   

    };
    for (const auto& h_param : incomp_params) {
        
        DataSet data = make_separable_dataset();
        RandomForest forest(h_param, 1);
        FeatureSelection f_selection = RandomK{h_param.mtry};
        Criterion crit = Gini{};
        ThresholdComputation t_comp = CART{};
        SplitParam param = ParamBuilder(TreeModel::RandomForest, crit, t_comp, f_selection);

        forest.fit(data, param);

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


TEST_CASE("RandomForest : max_samples"){

    DataSet data = make_separable_dataset();
    float max_sample = 0.2;
    HyperParam h_param{2, 3, 4, max_sample};
    RandomForest forest(h_param, 123);
    SplitParam param = ParamBuilder(TreeModel::RandomForest, Gini{}, CART{}, RandomK{2});

    forest.fit(data, param);

    std::optional<float> m_samples = forest.get_max_samples();

    REQUIRE( m_samples.value() == Catch::Approx(max_sample));
}

TEST_CASE("RandomForest : min_sample_split"){

    DataSet data = make_separable_dataset();
    float min_sample_split = 2;
    HyperParam h_param{ .mtry=2, .min_sample_split = min_sample_split};
    RandomForest forest(h_param, 123);
    SplitParam param = ParamBuilder(TreeModel::RandomForest, Gini{}, CART{}, RandomK{2});

    forest.fit(data, param);

}

TEST_CASE("RandomForest : min_sample_split propagation"){

    DataSet data = make_separable_dataset();
    int min_sample_split = 2;
    int n_estimators = 10;
    HyperParam h_param{.mtry=2, .n_estimators= n_estimators, .min_sample_split = min_sample_split};
    RandomForest forest(h_param, 123);
    SplitParam param = ParamBuilder(TreeModel::RandomForest, Gini{}, CART{}, RandomK{2});

    forest.fit(data, param);

    for (int i=0; i < n_estimators; i++){
        
        const arboria::ForestTree& forest_tree = arboria::test::RandomForestAccess::access_forest_trees(forest, i);
        INFO("min_sample_split = " << forest_tree.tree->min_sample_split.value());
        REQUIRE(forest_tree.tree->min_sample_split.value() == min_sample_split);

    }


}


TEST_CASE("RandomForest : reproductibility"){

        std::vector<float> X{
        2,3,5,
        2,3,5,
        4, 6, 10,
        4, 6, 10,
        8, 12, 20,
        8, 12, 20

    };
    std::vector<float> y{0, 1, 0, 1, 0, 1};
    DataSet data(X,y, 6, 3);
    int min_sample_split = 2;
    int n_estimators = 2;
    HyperParam h_param{.mtry=1,.n_estimators= n_estimators, .min_sample_split = min_sample_split,  };
    RandomForest forest1(h_param, 123);
    RandomForest forest2(h_param, 321);
    SplitParam param = ParamBuilder(TreeModel::RandomForest, Gini{}, CART{}, RandomK{2});

    forest1.fit(data, param);
    forest2.fit(data, param);

    std::vector<float> sample{4,12,5};
    std::vector<float> pred1 = forest1.predict_proba(sample);
    std::vector<float> pred2 = forest2.predict_proba(sample);
    
    REQUIRE(pred1 != pred2);

}



TEST_CASE("RandomForest : reproductibility under multithreading"){

        std::vector<float> X{
        2,3,5,
        2,3,5,
        4, 6, 10,
        4, 6, 10,
        8, 12, 20,
        8, 12, 20

    };
    std::vector<float> y{0, 1, 0, 1, 0, 1};
    DataSet data(X,y, 6, 3);
    int min_sample_split = 2;
    int n_estimators = 2;
    HyperParam h_param{.mtry=1,.n_estimators= n_estimators, .min_sample_split = min_sample_split, .n_jobs =2 };
    RandomForest forest1(h_param, 123);
    RandomForest forest2(h_param, 321);
    SplitParam param = ParamBuilder(TreeModel::RandomForest, Gini{}, CART{}, RandomK{2});

    forest1.fit(data, param);
    forest2.fit(data, param);

    std::vector<float> sample{4,12,5};
    std::vector<float> pred1 = forest1.predict_proba(sample);
    std::vector<float> pred2 = forest2.predict_proba(sample);
    
    REQUIRE(pred1 != pred2);

}


