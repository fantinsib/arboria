#pragma once

#include <vector>
#include <span>

#include "types/split_context.h"
#include "types/split_result.h"
#include "split_strategy/types/split_param.h"
#include "types/split_stats.h"
#include "split_criterion/gini.h"
#include "split_criterion/entropy.h"
#include "dataset/dataset.h"
#include "split_strategy/threshold/cart_threshold.h"


namespace arboria{
namespace split_strategy{

class Splitter 
{
    public:
        Splitter();
        /**
         * @brief Search the best split given a set of row 
         * from a DataSet objet, a set of logical parameters 
         * and contextual parameters and dispatch depending on 
         * regression/classification
         * 
         * @param idx a span on a vector of row index from the DataSet object
         * @param data a DataSet object containing the samples and the targets
         * @param params a SplitParam struct containing info on the criterion 
         * (default : Gini), the threshold method calculation (default : CART),
         * the range of features selected for the split (default : all)
         * @param context a SplitContext struct containing contextual arguments 
         * to be passed to the function (std::mt19937)
         * @throws std::invalid_argument if data or idx is empty
         * @note If no split is found, will return a SplitResult with default value
         * On a SplitResult, one can test if a split has been found with SplitResult.has_split()
         * Splits are calculated with the following logic : if sample feature < candidate threshold -> left node
         * if sample feature >= candidate threshold -> right node
         * @return a SplitResult struct 
         */
        SplitResult best_split(std::span<const int> idx, const DataSet& data, const SplitParam& params, SplitContext& context);

        /**
         * @brief Overload for default no context
         * Search the best split given a set of row 
         * from a DataSet objet, a set of logical parameters 
         * and contextual parameters and dispatch depending on 
         * regression/classification
         * 
         * @param idx a span on a vector of row index from the DataSet object
         * @param data a DataSet object containing the samples and the targets
         * @param params a SplitParam struct containing info on the criterion 
         * (default : Gini), the threshold method calculation (default : CART),
         * the range of features selected for the split (default : all)
         * @param context a SplitContext struct containing contextual arguments 
         * to be passed to the function (std::mt19937)
         * @throws std::invalid_argument if data or idx is empty
         * @note If no split is found, will return a SplitResult with default value
         * On a SplitResult, one can test if a split has been found with SplitResult.has_split()
         * Splits are calculated with the following logic : if sample feature < candidate threshold -> left node
         * if sample feature >= candidate threshold -> right node
         * @return a SplitResult struct 
         */
        SplitResult best_split(std::span<const int> idx, const DataSet &data, const SplitParam &params);

        /**
         * @brief Search the best split given a set of row 
         * from a DataSet objet and a set of parameters for classification
         * 
         * @param idx a span on a vector of row index from the DataSet object
         * @param data a DataSet object containing the samples and the targets
         * @param params a SplitParam struct containing info on the criterion 
         * (default : Gini), the threshold method calculation (default : CART),
         * the range of features selected for the split (default : all)
         * @throws std::invalid_argument if data or idx is empty
         * @note If no split is found, will return a SplitResult with default value
         * On a SplitResult, one can test if a split has been found with SplitResult.has_split()
         * Splits are calculated with the following logic : if sample feature < candidate threshold -> left node
         * if sample feature >= candidate threshold -> right node
         * @return a SplitResult struct 
         */
        SplitResult best_split_classification(std::span<const int> idx, const DataSet& data, const SplitParam& params, SplitContext& context);
    
           /**
         * @brief Search the best split given a set of row 
         * from a DataSet objet and a set of parameters for regression
         * 
         * @param idx a span on a vector of row index from the DataSet object
         * @param data a DataSet object containing the samples and the targets
         * @param params a SplitParam struct containing info on the criterion 
         * (default : Gini), the threshold method calculation (default : CART),
         * the range of features selected for the split (default : all)
         * @throws std::invalid_argument if data or idx is empty
         * @note If no split is found, will return a SplitResult with default value
         * On a SplitResult, one can test if a split has been found with SplitResult.has_split()
         * Splits are calculated with the following logic : if sample feature < candidate threshold -> left node
         * if sample feature >= candidate threshold -> right node
         * @return a SplitResult struct 
         */
        SplitResult best_split_regression(std::span<const int> idx, const DataSet& data, const SplitParam& params, SplitContext& context);
    
    
    private:
        /**
         * @brief given a impurity measure and metrics on a split, returns 
         * its weighted impurity score.
         * 
         * @param params a SplitParam struct containing info on the criterion 
         * (Gini, Entropy), the threshold method calculation (CART),
         * the range of features selected for the split (All, RandomK)
         * @param stats a SplitStat struct containing counts of left and right
         * positive and negative label count for the resuling nodes 
         * @return float 
         */
        float score_function(const SplitParam& params, const ClfStats& stats);

        /**
         * @brief given a loss function and metrics on a split, returns 
         * its weighted loss.
         * 
         * @param params a SplitParam struct containing info on the criterion 
         * (Loss), the threshold method calculation (CART),
         * the range of features selected for the split (All, RandomK)
         * @param stats a RegStats struct containing sums and counts 
         * of left and right target values for the resuling nodes 
         * @return float 
         */
        float score_function(const SplitParam& params, const RegStats& stats);


    };
}
}
