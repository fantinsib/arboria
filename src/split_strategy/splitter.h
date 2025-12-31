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
        SplitResult best_split(std::span<const int> idx, const DataSet& data, const SplitParam& params);
        
    private:
        float score_function(const SplitParam& params, const SplitStats& stats);
        


};




}
}