


#include "split_strategy/types/split_param.h"
#include "tree/TreeModel.h"
#include "tree/RandomForest/randomforest.h"
#include "tree/DecisionTree/DecisionTree.h"
#include "ParamBuilder.h"
#include <optional>
#include <stdexcept>
#include <variant>


namespace arboria{

enum class Task {Regression, Classification};
enum class Family {DecisionTree, RandomForest};

SplitParam ParamBuilder(const TreeModel model, 
                        std::optional<TreeType> type, 
                        std::optional<Criterion> crit ,
                        std::optional<ThresholdComputation> threshold, 
                        std::optional<FeatureSelection> feature)
{



    if (model == TreeModel::DecisionTree){
        
        if (!type.has_value()){
            throw std::invalid_argument("ParamBuilder : TreeModel must be specified to avoid ambiguity");
        }
        
        if (!crit.has_value()){
            if (std::holds_alternative<Classification>(*type))
                {crit = Gini{};}
            if (std::holds_alternative<Regression>(*type))
                {crit = SSE{};}
        }
        if (!threshold.has_value()){
            threshold = CART{};
        }
        
        if (!feature.has_value()){
            feature = AllFeatures{};
        }
        
        SplitParam param;
        param.type = *type;
        param.criterion = *crit;
        param.t_comp = *threshold;
        param.f_selection = *feature;
        
        return param;
    }


    if (model == TreeModel::RandomForest){
            
        if (!type.has_value()){
            throw std::invalid_argument("ParamBuilder : TreeModel must be specified to avoid ambiguity");
        }
        
        if (!crit.has_value()){
            if (std::holds_alternative<Classification>(*type))
                {crit = Gini{};}
            if (std::holds_alternative<Regression>(*type))
                {crit = SSE{};}
        }
        if (!threshold.has_value()){
            threshold = CART{};
        }
        
        if (!feature.has_value()){
            feature = RandomK{};
        }
        
        SplitParam param;
        param.type = *type;
        param.criterion = *crit;
        param.t_comp = *threshold;
        param.f_selection = *feature;
        
        return param;
    }

    throw std::logic_error("ParamBuilder error : Tree has not yet been implemented");

}
}
