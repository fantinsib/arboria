/*
                                              TESTS FOR DECISIONTREE
*/

#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>

#include "tree/DecisionTree/DecisionTree.h"
#include "split_strategy/types/split_param.h"
#include "split_strategy/types/split_hyper.h"
#include "split_strategy/types/ParamBuilder/ParamBuilder.h"
#include "tree/TreeModel.h"

TEST_CASE("DecisionTree :  predict_one() basic usage - fit") {


    std::vector<float> X {0,2,1,
                        7,9,10,
                        1,1,2,
                        11, 9, 8,
                        2,0,1}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param{.max_depth = 4};
    
    arboria::DecisionTree tree(h_param, Classification{});

    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);
    
    tree.fit(data, params);

    REQUIRE(tree.is_fitted() == true);
    REQUIRE(tree.num_features == 3);

}

TEST_CASE("DecisionTree :  predict_one() basic usage - entropy") {


    std::vector<float> X {0,2,1,
                        7,9,10,
                        1,1,2,
                        11, 9, 8,
                        2,0,1}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param{.max_depth = 4};
    
    arboria::DecisionTree tree(h_param, Classification{});

    SplitParam params = arboria::ParamBuilder(
    TreeModel::DecisionTreeClassifier,
    std::nullopt,        // type
    Criterion{Entropy{}},// crit
    std::nullopt,        // threshold
    std::nullopt         // feature
);

    
    tree.fit(data, params);

    REQUIRE(tree.is_fitted() == true);
    REQUIRE(tree.num_features == 3);

}


TEST_CASE("DecisionTree :  predict_one() basic usage - simple predict") {

SECTION("Class 1 pred") {
    std::vector<float> X {0,2,1,
                        7,9,10,
                        1,1,2,
                        11, 9, 8,
                        2,0,1}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param{.max_depth = 4};
    
    arboria::DecisionTree tree(h_param,Classification{});
    
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);

    tree.fit(data, params);

    std::vector<float> sample {8,9,10};
    int pred = tree.predict_one(sample);
    REQUIRE(pred == 1);
    }

SECTION("Class 2 pred") {
    std::vector<float> X {0,2,1,
                        7,9,10,
                        1,1,2,
                        11, 9, 8,
                        2,0,1}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param{.max_depth = 4};
    arboria::DecisionTree tree(h_param,Classification{});
    
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);

    tree.fit(data, params);

    std::vector<float> sample {1,0,0};
    int pred = tree.predict_one(sample);
    REQUIRE(pred == 0);
    }

}

TEST_CASE("DecisionTree : instanciation without max_depth"){
    std::vector<float> X {0,2,1,
                        11, 9, 8,
                        0,2,1,
                        11, 9, 8,
                        0,2,1,}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param;
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);
    arboria::DecisionTree tree(h_param, Classification{});
    tree.fit(data, params);

    REQUIRE(tree.is_fitted() == true);

}

TEST_CASE("DecisionTree :  predict_one() :predict from duplicate row") {

    std::vector<float> X {0,2,1,
                        11, 9, 8,
                        0,2,1,
                        11, 9, 8,
                        0,2,1,}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param{.max_depth = 4};
    arboria::DecisionTree tree(h_param,Classification{});
    
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);

    tree.fit(data, params);

    std::vector<float> sample1 {1,1,2};
    int pred1 = tree.predict_one(sample1);
    
    std::vector<float> sample2 {10,11,12};
    int pred2 = tree.predict_one(sample2);

    REQUIRE(pred1 == 0);
    REQUIRE(pred2 == 1);

}


TEST_CASE("DecisionTree :  predict_one() : Duplicate samples & unique class") {
    std::vector<float> X {0,2,1,
                        0,2,1,
                        0,2,1,
                        0,2,1,
                        0,2,1}; 
    std::vector<float> y {0,0,0,0,0};
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param{.max_depth = 4};
    arboria::DecisionTree tree(h_param, Classification{});
    
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);

    tree.fit(data, params);
    std::vector<float> sample {10,11,12};
    int pred = tree.predict_one(sample);
    REQUIRE(pred == 0);
}

TEST_CASE("DecisionTree :  predict_one() :unsplitable data with unbalanced classes") {
    std::vector<float> X {0,2,1,
                        0,2,1,
                        0,2,1,
                        0,2,1,
                        0,2,1}; 
    std::vector<float> y {0,1,1,1,0};
    
    arboria::DataSet data(X, y, 5, 3); 
    HyperParam h_param{.max_depth = 4};
    arboria::DecisionTree tree(h_param,Classification{});
    
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);

    tree.fit(data, params);
    std::vector<float> sample {10,11,12};
    int pred = tree.predict_one(sample);
    REQUIRE(pred == 1); //Should return the majority class - here 1 
}

TEST_CASE("DecisionTree :  predict_one() : unsplitable data with balanced classes") {
    std::vector<float> X {0,2,1,
                        0,2,1,
                        0,2,1,
                        0,2,1};
    std::vector<float> y {0,1,1,0};
    
    arboria::DataSet data(X, y, 4, 3); 
    HyperParam h_param{.max_depth = 4};
    arboria::DecisionTree tree(h_param, Classification{});
    
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);

    tree.fit(data, params);
    std::vector<float> sample { 0,2,1};
    int pred = tree.predict_one(sample);
    REQUIRE(pred == 1); //in case of ties, should always return 1 
}


TEST_CASE("DecisionTree : error - one sample") {
    std::vector<float> X {0,2,1,}; 
    std::vector<float> y {0};
    
    arboria::DataSet data(X, y, 1, 3);
    HyperParam h_param{.max_depth = 4};
    arboria::DecisionTree tree(h_param,Classification{});
    
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);

    REQUIRE_THROWS_AS(tree.fit(data, params), std::invalid_argument);

}

TEST_CASE("DecisionTree : predict_one() error - trying to predict from non fitted tree") {
    
    std::vector<float> X {0,2,1,
                        0,2,1,
                        0,2,1,
                        0,2,1};
    std::vector<float> y {0,1,1,0};
    
    arboria::DataSet data(X, y, 4, 3); 
    HyperParam h_param{.max_depth = 4};
    arboria::DecisionTree tree(h_param,Classification{});
    
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);

    std::vector<float> sample {10,11,12};
    REQUIRE_THROWS_AS(tree.predict_one(sample), std::invalid_argument);

}

TEST_CASE("DecisionTree - .predict() - basic usage") {


    std::vector<float> X {0,2,1,
                        7,9,10,
                        1,1,2,
                        11, 9, 8,
                        2,0,1}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param{.max_depth = 4};
    arboria::DecisionTree tree(h_param,Classification{});
    
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);

    tree.fit(data, params);
    
    std::vector<float> samples_to_predict{1,2,0,
                                        0,-1,-2,
                                        10,7,9,
                                        7,12,9};
    
    std::vector<int> preds = tree.predict(samples_to_predict);

    REQUIRE(tree.is_fitted() == true);
    REQUIRE(tree.num_features == 3);
    REQUIRE(preds[0]==0);
    REQUIRE(preds[1]==0);
    REQUIRE(preds[2]==1);
    REQUIRE(preds[3]==1);
}

TEST_CASE("DecisionTree - .predict() - unique sample") {


    std::vector<float> X {0,2,1,
                        7,9,10,
                        1,1,2,
                        11, 9, 8,
                        2,0,1}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param{.max_depth = 4};
    arboria::DecisionTree tree(h_param,Classification{});
    
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);

    tree.fit(data, params);
    
    std::vector<float> samples_to_predict{10,6,10};
    
    std::vector<int> preds = tree.predict(samples_to_predict);

    REQUIRE(tree.is_fitted() == true);
    REQUIRE(tree.num_features == 3);
    REQUIRE(preds[0]==1);

}


TEST_CASE("DecisionTree - error .predict() - not fitted") {

    std::vector<float> X {0,2,1,
                        7,9,10,
                        1,1,2,
                        11, 9, 8,
                        2,0,1}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param{.max_depth = 4};
    arboria::DecisionTree tree(h_param,Classification{});
    
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);
    
    std::vector<float> samples_to_predict{1,2,0,
                                        0,-1,-2,
                                        10,7,9,
                                        7,12,9};
    

    REQUIRE_THROWS_AS(tree.predict(samples_to_predict), std::invalid_argument);

}

TEST_CASE("DecisionTree - error .predict() - dimension mismatch") {

    std::vector<float> X {0,2,1,
                        7,9,10,
                        1,1,2,
                        11, 9, 8,
                        2,0,1}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param{.max_depth = 4};
    arboria::DecisionTree tree(h_param,Classification{});
    
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);

    tree.fit(data, params);
    
    std::vector<float> samples_to_predict{1,2,0,
                                        0,-1,-2,
                                        7,12};

    REQUIRE_THROWS_AS(tree.predict(samples_to_predict), std::invalid_argument);

}

TEST_CASE("DecisionTree - error .fit() - trying to call with uninitialized SplitParam"){


    std::vector<float> X {0,2,1,
                        7,9,10,
                        1,1,2,
                        11, 9, 8,
                        2,0,1}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param{.max_depth = 4};
    arboria::DecisionTree tree(h_param,Classification{});
    
    SplitParam params; // uninitialized split
    
    REQUIRE_THROWS_AS(tree.fit(data, params), std::invalid_argument);

}

TEST_CASE("DecisionTree - building with min_sample_split"){

    std::vector<float> X {0,2,1,
                        7,9,10,
                        1,1,2,
                        11, 9, 8,
                        2,0,1}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial split
    
    arboria::DataSet data(X, y, 5, 3);
    HyperParam h_param{.max_depth = 2, .min_sample_split = 3};
    arboria::DecisionTree tree(h_param,Classification{});
    SplitParam params = arboria::ParamBuilder(TreeModel::DecisionTreeClassifier);
    tree.fit(data, params);
    REQUIRE(tree.min_sample_split == 3);
    
}