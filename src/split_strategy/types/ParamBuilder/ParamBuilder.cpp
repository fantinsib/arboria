


#include "split_strategy/types/split_param.h"
#include "tree/TreeModel.h"
#include "tree/RandomForest/randomforest.h"
#include "tree/DecisionTree/decisiontree.h"
#include "ParamBuilder.h"
#include <optional>
#include <stdexcept>


namespace arboria{
SplitParam ParamBuilder(const TreeModel model, 
                        std::optional<Criterion> crit,
                        std::optional<ThresholdComputation> threshold, 
                        std::optional<FeatureSelection> feature){

    if (model == TreeModel::DecisionTree){
        if (!crit.has_value()){
            crit = Gini{};
        }
        if (!threshold.has_value()){
            threshold = CART{};
        }
        
        if (!feature.has_value()){
            feature = AllFeatures{};
        }
        
        SplitParam param;
        param.criterion = *crit;
        param.t_comp = *threshold;
        param.f_selection = *feature;
        
        return param;
    }

    if (model == TreeModel::RandomForest){
        if (!crit.has_value()){
            crit = Gini{};
        }
        if (!threshold.has_value()){
            threshold = CART{};
        }
        
        if (!feature.has_value()){
            feature = RandomK{};
        }
        
        SplitParam param;
        param.criterion = *crit;
        param.t_comp = *threshold;
        param.f_selection = *feature;
        
        return param;
    }

    throw std::logic_error("ParamBuilder error : Tree has not yet been implemented");

}
}