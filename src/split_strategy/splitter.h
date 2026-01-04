#pragma once

#include <vector>
#include <span>

#include "split_result.h"
#include "split_param.h"
#include "split_criterion/gini.h"
#include "split_criterion/entropy.h"
#include "dataset/dataset.h"
#include "split_strategy/cart_threshold.h"
#include "split_stats.h"

namespace arboria{
namespace split_strategy{

class Splitter 
{
    public:
        Splitter();
        /**
         * @brief Search the best split given a set of row 
         * from a DataSet objet and a set of parameters
         * 
         * @param idx a span on a vector of row index from the DataSet object
         * @param data a DataSet object containing the samples and the targets
         * @param params a SplitParam struct containing info on the criterion 
         * (default : Gini), the threshold method calculation (default : CART),
         * the range of features selected for the split (default : all)
         * @throws std::invalid_argument if data or idx is empty
         * @note If no split is found, will return a SplitResult with default value
         * On a SplitResult, one can test if a split has been found with SplitResult.has_split()
         * @return a SplitResult struct 
         */
        SplitResult best_split(std::span<const int> idx, const DataSet& data, const SplitParam& params);
    
    private:
        /**
         * @brief given a impurity measure and metrics on a split, returns 
         * its weighted impurity score.
         * 
         * @param params a SplitParam struct containing info on the criterion 
         * (default : Gini), the threshold method calculation (default : CART),
         * the range of features selected for the split (default : all)
         * @param stats a SplitStat struct containing counts of left and right
         * positive and negative label count for the resuling nodes 
         * @return float 
         */
        float score_function(const SplitParam& params, const SplitStats& stats);
};
}
}