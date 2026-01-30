


#include "split_strategy/types/split_param.h"
#include "tree/TreeModel.h"
#include "tree/RandomForest/randomforest.h"
#include "tree/DecisionTree/DecisionTree.h"
#include "ParamBuilder.h"
#include <optional>
#include <stdexcept>


namespace arboria{

enum class Task {Regression, Classification};
enum class Family {DecisionTree, RandomForest};

SplitParam ParamBuilder(const TreeModel model, 
                        std::optional<TreeType> type, 
                        std::optional<Criterion> crit ,
                        std::optional<ThresholdComputation> threshold, 
                        std::optional<FeatureSelection> feature)
{

    Task task;
    Family family; 

    if (model == TreeModel::DecisionTreeClassifier || model == TreeModel::DecisionTreeRegressor){
        family = Family::DecisionTree;

    }
    else if (model == TreeModel::RandomForestClassifier || model == TreeModel::RandomForestRegressor){
        family = Family::RandomForest;
    }

    if (model == TreeModel::DecisionTreeClassifier || model == TreeModel::RandomForestClassifier){
        task = Task::Classification;
    }

    else if (model == TreeModel::DecisionTreeRegressor || model == TreeModel::RandomForestRegressor){
        task = Task::Regression;
    }


    if (family == Family::DecisionTree){
        
        if (!type.has_value()){
            if (task == Task::Regression) type = Regression{};
            if (task == Task::Classification) type = Classification{};
        }
        
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
        param.type = *type;
        param.criterion = *crit;
        param.t_comp = *threshold;
        param.f_selection = *feature;
        
        return param;
    }


    if (family == Family::RandomForest){
            
        if (!type.has_value()){
            if (task == Task::Regression) type = Regression{};
            if (task == Task::Classification) type = Classification{};
        }
        
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
        param.type = *type;
        param.criterion = *crit;
        param.t_comp = *threshold;
        param.f_selection = *feature;
        
        return param;
    }

    throw std::logic_error("ParamBuilder error : Tree has not yet been implemented");

}
}
