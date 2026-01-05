/*
                                              TESTS FOR DECISIONTREE
*/

#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>

#include "tree/DecisionTree/DecisionTree.h"
#include "split_strategy/types/split_param.h"

TEST_CASE("DecisionTree : basic usage - fit") {


    std::vector<float> X {0,2,1,
                        7,9,10,
                        1,1,2,
                        11, 9, 8,
                        2,0,1}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    
    arboria::DecisionTree tree(4);
    
    SplitParam params;

    tree.fit(data, params);

    REQUIRE(tree.is_fitted() == true);
    REQUIRE(tree.num_features == 3);

}


TEST_CASE("DecisionTree : basic usage - simple predict") {

SECTION("Class 1 pred") {
    std::vector<float> X {0,2,1,
                        7,9,10,
                        1,1,2,
                        11, 9, 8,
                        2,0,1}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    
    arboria::DecisionTree tree(4);
    
    SplitParam params;

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
    
    arboria::DecisionTree tree(4);
    
    SplitParam params;

    tree.fit(data, params);

    std::vector<float> sample {1,0,0};
    int pred = tree.predict_one(sample);
    REQUIRE(pred == 0);
    }

}

TEST_CASE("DecisionTree : predict from duplicate row") {

    std::vector<float> X {0,2,1,
                        11, 9, 8,
                        0,2,1,
                        11, 9, 8,
                        0,2,1,}; 
    std::vector<float> y {0,1,0,1,0}; //dataset with trivial classes
    
    arboria::DataSet data(X, y, 5, 3);
    
    arboria::DecisionTree tree(4);
    
    SplitParam params;

    tree.fit(data, params);

    std::vector<float> sample1 {1,1,2};
    int pred1 = tree.predict_one(sample1);
    
    std::vector<float> sample2 {10,11,12};
    int pred2 = tree.predict_one(sample2);

    REQUIRE(pred1 == 0);
    REQUIRE(pred2 == 1);

}


TEST_CASE("DecisionTree : Duplicate samples & unique class") {
    std::vector<float> X {0,2,1,
                        0,2,1,
                        0,2,1,
                        0,2,1,
                        0,2,1}; 
    std::vector<float> y {0,0,0,0,0};
    
    arboria::DataSet data(X, y, 5, 3);
    
    arboria::DecisionTree tree(4);
    
    SplitParam params;

    tree.fit(data, params);
    std::vector<float> sample {10,11,12};
    int pred = tree.predict_one(sample);
    REQUIRE(pred == 0);
}

TEST_CASE("DecisionTree : unsplitable data with unbalanced classes") {
    std::vector<float> X {0,2,1,
                        0,2,1,
                        0,2,1,
                        0,2,1,
                        0,2,1}; 
    std::vector<float> y {0,1,1,1,0};
    
    arboria::DataSet data(X, y, 5, 3); 
    
    arboria::DecisionTree tree(4);
    
    SplitParam params;

    tree.fit(data, params);
    std::vector<float> sample {10,11,12};
    int pred = tree.predict_one(sample);
    REQUIRE(pred == 1); //Should return the majority class - here 1 
}

TEST_CASE("DecisionTree : unsplitable data with balanced classes") {
    std::vector<float> X {0,2,1,
                        0,2,1,
                        0,2,1,
                        0,2,1};
    std::vector<float> y {0,1,1,0};
    
    arboria::DataSet data(X, y, 4, 3); 
    
    arboria::DecisionTree tree(4);
    
    SplitParam params;

    tree.fit(data, params);
    std::vector<float> sample { 0,2,1};
    int pred = tree.predict_one(sample);
    REQUIRE(pred == 1); //in case of ties, should always return 1 
}


TEST_CASE("DecisionTree : error - one sample") {
    std::vector<float> X {0,2,1,}; 
    std::vector<float> y {0};
    
    arboria::DataSet data(X, y, 1, 3);
    
    arboria::DecisionTree tree(4);
    
    SplitParam params;

    REQUIRE_THROWS_AS(tree.fit(data, params), std::invalid_argument);

}

TEST_CASE("DecisionTree : error - trying to predict from non fitted tree") {
    
    std::vector<float> X {0,2,1,
                        0,2,1,
                        0,2,1,
                        0,2,1};
    std::vector<float> y {0,1,1,0};
    
    arboria::DataSet data(X, y, 4, 3); 
    
    arboria::DecisionTree tree(4);
    
    SplitParam params;

    std::vector<float> sample {10,11,12};
    REQUIRE_THROWS_AS(tree.predict_one(sample), std::invalid_argument);

}

