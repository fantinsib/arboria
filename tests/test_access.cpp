#include "test_access.h"
#include "split_strategy/types/split_param.h"
#include "tree/RandomForest/randomforest.h"
#include <optional>


namespace arboria::test{

const std::optional<size_t> arboria::test::DecisionTreeAccess::access_min_samples_split(const arboria::DecisionTree &tree){

    return tree.min_sample_split;

}


const arboria::ForestTree& arboria::test::RandomForestAccess::access_forest_trees(const arboria::RandomForest &rf, size_t i_tree){

    return rf.trees[i_tree];

}
}//end of namespace