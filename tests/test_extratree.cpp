/*

                        TESTS EXTRA TREES

*/


#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cmath>

#include "dataset/dataset.h"
#include "split_strategy/types/split_param.h"
#include "tree/RandomForest/randomforest.h"
#include "tree/ExtraTree/extratree.h"
#include "split_strategy/types/ParamBuilder/ParamBuilder.h"
#include "tree/TreeModel.h"

using arboria::DataSet;
using arboria::ParamBuilder;
using arboria::ExtraTree; 

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

DataSet make_regression_dataset() {
    std::vector<float> X{
        0,
        0,
        10,
        10
    };
    std::vector<float> y{1, 3, 5, 7};
    return DataSet(X, y, 4, 1);
}


TEST_CASE("ExtraTree : constructor") {

    DataSet data = make_separable_dataset();

    SplitParam param = arboria::ParamBuilder(TreeModel::ExtraTree, 
                                            Classification{}, Gini{}, 
                                            Random{.n_random_split = 1}, RandomK{.mtry = 2});
    HyperParam h_param{.mtry = 2, .n_random_split= 1, .n_estimators = 25};

    ExtraTree et(h_param, Classification{}, 123);

    et.fit(data, param);
    
    REQUIRE(et.is_fitted() == true);

}

TEST_CASE("ExtraTree : randomness"){
    
    DataSet data = make_separable_dataset();

    SplitParam param = arboria::ParamBuilder(TreeModel::ExtraTree, 
                                            Classification{}, Gini{}, 
                                            Random{.n_random_split = 2}, RandomK{.mtry = 2});
    HyperParam h_param{.mtry = 2, .n_random_split= 2, .n_estimators = 25};

    ExtraTree et1(h_param, Classification{}, 123);
    ExtraTree et2(h_param, Classification{}, 123);
    ExtraTree et3(h_param, Classification{}, 321);

    et1.fit(data, param);
    et2.fit(data, param);
    et3.fit(data, param);
    
    std::vector<float> x_test{2,2,2};

    REQUIRE(et1.predict_proba(x_test) == et2.predict_proba(x_test));
    REQUIRE(et1.predict_proba(x_test) != et3.predict_proba(x_test));

}
