

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>  
#include <cmath>
#include <stdexcept>
#include <vector>
#include <span>

#include "split_criterion/entropy.h"
#include "split_criterion/gini.h"
#include "split_strategy/splitter.h"
#include "dataset/dataset.h"
#include "split_strategy/types/ParamBuilder/ParamBuilder.h"
#include "split_strategy/types/split_param.h"
#include "split_strategy/types/split_result.h"
#include "split_strategy/types/split_context.h"


using arboria::split_strategy::Splitter;
using arboria::DataSet;
using arboria::split::weighted_gini;
using arboria::split::weighted_entropy;
using arboria::ParamBuilder;

/*

----------------------------------------------------------------------------
BEST_SPLIT BASIC USAGE 
----------------------------------------------------------------------------

*/

TEST_CASE("best_split : basic usage with perfect split - Gini") {

    std::vector<float> x{1,2,12,
                        2,9,6,
                        1, 8 ,12,
                        0.5, 1,6}; //DataSet with obvious split on col 1 
    std::vector<float> y{0,1,1,0};
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0,1,2,3};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, AllFeatures{}};
    
    Splitter splitter;
    SplitResult b_split = splitter.best_split(s, data, param);

    REQUIRE(b_split.split_feature == 1);
    REQUIRE(b_split.split_threshold == Catch::Approx(5.f));
    REQUIRE(b_split.score == Catch::Approx(0.f));
    REQUIRE(b_split.has_split() == true);

}

TEST_CASE("best_split : basic usage with unperfect split - Gini") {

    std::vector<float> x{1,2,11,
                        1,2,11.1,
                        1, 2 ,10.9,
                        1, 2,6}; 
    std::vector<float> y{1,0,1,0};
    
    //The resulting DataSet can only be split on col 2 and
    //should result in a split with node left [0] and node right [0,1,1]
    //which should give a weighted Gini score of 1/3

    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0,1,2,3};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, AllFeatures{}};
    
    Splitter splitter;
    SplitResult b_split = splitter.best_split(s, data, param);

    REQUIRE(b_split.split_feature == 2);
    REQUIRE(b_split.score == Catch::Approx(0.33333f));
    REQUIRE(b_split.has_split() == true);

}

TEST_CASE("best_split : basic usage with perfect split - Entropy") {

    std::vector<float> x{1,2,12,
                        2,9,6,
                        1, 8 ,12,
                        0.5, 1,6}; //DataSet with obvious split on col 1 
    std::vector<float> y{0,1,1,0};
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0,1,2,3};
    std::span s(rows);

    SplitParam param{Entropy{}, CART{}, AllFeatures{}};
    
    Splitter splitter;
    SplitResult b_split = splitter.best_split(s, data, param);

    REQUIRE(b_split.split_feature == 1);
    REQUIRE(b_split.split_threshold == Catch::Approx(5.f));
    REQUIRE(b_split.score == Catch::Approx(0.f));
    REQUIRE(b_split.has_split() == true);

    }


TEST_CASE("best_split : basic usage with unperfect split - Entropy") {

    std::vector<float> x{1,2,11,
                        1,2,11.1,
                        1, 2 ,10.9,
                        1, 2,6}; 
    std::vector<float> y{1,0,1,0};
    
    //The resulting DataSet can only be split on col 2 and
    //should result in a split with node left [0] and node right [0,1,1]
    //which should give a weighted entropy score of around 0.6887
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0,1,2,3};
    std::span s(rows);

    SplitParam param{Entropy{}, CART{}, AllFeatures{}};
    
    Splitter splitter;
    SplitResult b_split = splitter.best_split(s, data, param);

    REQUIRE(b_split.split_feature == 2);
    REQUIRE(b_split.score == Catch::Approx(0.6887f).epsilon(1e-3));
    REQUIRE(b_split.has_split() == true);

}

TEST_CASE("best_split : basic usage with no split") {

    std::vector<float> x{1,2,11,
                        1,2,11,
                        1, 2 ,11,
                        1, 2,11}; 
    std::vector<float> y{1,0,1,0};
    
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0,1,2,3};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, AllFeatures{}};
    
    Splitter splitter;
    SplitResult b_split = splitter.best_split(s, data, param);

    REQUIRE(b_split.has_split() == false);

}

TEST_CASE("best_split : basic usage with row selection") {

    std::vector<float> x{1,2,12,
                        2,9,6,
                        1, 8 ,12,
                        0.5, 1,6}; //DataSet with obvious split on col 1 
    std::vector<float> y{0,1,1,0};
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0,2};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, AllFeatures{}};
    
    Splitter splitter;
    SplitResult b_split = splitter.best_split(s, data, param);

    REQUIRE(b_split.score == Catch::Approx(0.f));

}


TEST_CASE("best_split : basic usage with row in random order") {

    std::vector<float> x{1,2,12,
                        2,9,6,
                        1, 8 ,12,
                        0.5, 1,6}; //DataSet with obvious split on col 1 
    std::vector<float> y{0,1,1,0};
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {3,0,2,1};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, AllFeatures{}};
    
    Splitter splitter;
    SplitResult b_split = splitter.best_split(s, data, param);

    REQUIRE(b_split.split_feature == 1);
    REQUIRE(b_split.split_threshold == Catch::Approx(5.f));
    REQUIRE(b_split.score == Catch::Approx(0.f));

}

TEST_CASE("best_split : basic usage with duplicate rows") {

    std::vector<float> x{1,2,12,
                        2,9,6,
                        1, 8 ,12,
                        0.5, 1,6}; //DataSet with obvious split on col 1 
    std::vector<float> y{0,1,1,0};
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0,0,0,2};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, AllFeatures{}};
    
    Splitter splitter;
    SplitResult b_split = splitter.best_split(s, data, param);

    REQUIRE(b_split.split_feature == 1);
    REQUIRE(b_split.score == Catch::Approx(0.f));

}
TEST_CASE("best_split : no split when 1 sample") {

    std::vector<float> x{1,2,12,
                        2,9,6,
                        1, 8 ,12,
                        0.5, 1,6}; //DataSet with obvious split on col 1 
    std::vector<float> y{0,1,1,0};
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, AllFeatures{}};
    
    Splitter splitter;
    SplitResult b_split = splitter.best_split(s, data, param);

    REQUIRE(b_split.has_split() == false);

}

/*

----------------------------------------------------------------------------
BEST_SPLIT ERRORS
----------------------------------------------------------------------------

*/


TEST_CASE("best_split : errors - empty data") {
    std::vector<float> x{}; 
    std::vector<float> y{};
    
    DataSet data(x, y, 0, 0); 
    std::vector<int> rows {0,0,0,2};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, AllFeatures{}};
    
    Splitter splitter;
    REQUIRE_THROWS_AS(splitter.best_split(s, data, param), std::invalid_argument);
}

TEST_CASE("best_split : errors - empty idx") {
    std::vector<float> x{1,2,12,
                        2,9,6,
                        1, 8 ,12,
                        0.5, 1,6}; //DataSet with obvious split on col 1 
    std::vector<float> y{0,1,1,0};
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, AllFeatures{}};
    
    Splitter splitter;
    REQUIRE_THROWS_AS(splitter.best_split(s, data, param), std::invalid_argument);
}


/*

----------------------------------------------------------------------------
BEST_SPLIT + RANDOMK BASIC USAGE 
----------------------------------------------------------------------------

*/



TEST_CASE("best_split + randomk : basic usage with span overload") {

    std::vector<float> x{1,2,12,
                        2,9,6,
                        4, 8 ,11,
                        0.5, 1,5};
    std::vector<float> y{0,1,1,0};
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0,1,2,3};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, RandomK{2}};

    Splitter splitter;

    SplitContext ctx1(1);
    SplitResult split = splitter.best_split(s, data, param, ctx1);

    REQUIRE(split.has_split() == true);

}


TEST_CASE("best_split + randomk : reproductibility") {

    std::vector<float> x{1,2,12,
                        2,9,6,
                        1, 8 ,12,
                        0.5, 1,6};
    std::vector<float> y{0,1,1,0};
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0,1,2,3};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, RandomK{2}};

    Splitter splitter;

    SplitContext ctx1(1);
    SplitResult a = splitter.best_split(s, data, param, ctx1);

    SplitContext ctx2(1);
    SplitResult b = splitter.best_split(s, data, param, ctx2);

    REQUIRE(a.split_feature == b.split_feature);
    REQUIRE(a.split_threshold == Catch::Approx(b.split_threshold));

}

TEST_CASE("best_split + randomk : mtry == num_features") {

    std::vector<float> x{1,2,12,
                        2,9,6,
                        4, 8 ,11,
                        0.5, 1,5};
    std::vector<float> y{0,1,1,0};
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0,1,2,3};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, RandomK{3}};


    Splitter splitter;

    SplitContext ctx1(1);
    SplitResult split = splitter.best_split(s, data, param, ctx1);

    REQUIRE(split.has_split() == true);

}


TEST_CASE("best_split + randomk : errors - called w/o context") {

    std::vector<float> x{1,2,12,
                        2,9,6,
                        1, 8 ,12,
                        0.5, 1,6};
    std::vector<float> y{0,1,1,0};
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0,1,2,3};
    std::span s(rows);

    SplitParam param{Gini{}, CART{}, RandomK{}};
    //pick randomK but omit SplitContext with RNG
    Splitter splitter;

    REQUIRE_THROWS_AS(splitter.best_split(s, data, param), std::invalid_argument);

}


TEST_CASE("best_split + randomk : errors - mtry illegal") {

    std::vector<float> x{1,2,12,
                        2,9,6,
                        1, 8 ,12,
                        0.5, 1,6};
    std::vector<float> y{0,1,1,0};
    
    DataSet data(x, y, 4, 3); 
    std::vector<int> rows {0,1,2,3};
    std::span s(rows);

    SplitContext ctx(1);
    Splitter splitter;

    SECTION("mtry == 0"){
    SplitParam param{Gini{}, CART{}, RandomK{0}};
    REQUIRE_THROWS_AS(splitter.best_split(s, data, param, ctx), std::logic_error);
    }

    SECTION("mtry < 0"){
    SplitParam param{Gini{}, CART{}, RandomK{-2}};
    REQUIRE_THROWS_AS(splitter.best_split(s, data, param, ctx), std::logic_error);
    }

    SECTION("mtry > num_features"){
    SplitParam param{Gini{}, CART{}, RandomK{4}};
    REQUIRE_THROWS_AS(splitter.best_split(s, data, param, ctx), std::logic_error);
    }

}
