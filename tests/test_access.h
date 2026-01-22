#pragma once
#include "tree/DecisionTree/DecisionTree.h"
#include "tree/RandomForest/randomforest.h"
#include <optional>

namespace arboria::test{

//access to private arguments of DecisionTree
struct DecisionTreeAccess {
    static const std::optional<size_t> access_min_samples_split(const arboria::DecisionTree& t);
};

//access to private arguments of RandomForest
struct RandomForestAccess {
    //Allows to access the ith ForestTree of the RandomForest. 
    static const arboria::ForestTree& access_forest_trees(const arboria::RandomForest& rf, size_t i_tree);
};

}//end of namespace