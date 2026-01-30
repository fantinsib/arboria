#pragma once
#include "split_strategy/types/split_hyper.h"
#include <optional>
#include <variant>
#include <cstddef>

//--------------------- Tree Type

struct Undefined{};

struct Regression{};
struct Classification{};

using TreeType = std::variant<Undefined, Regression, Classification>;


//--------------------- FeatureSelection

//Searches all features during split
struct AllFeatures {};

//Searches only mtry features at each split
struct RandomK{
    std::optional<int> mtry; 
};

using FeatureSelection = std::variant<Undefined, AllFeatures, RandomK>;

//-------------------- ThresholdComputation

//Computes the threshold according to regular CART algorithm
struct CART{};
struct Random{};
struct Quantile{};

using ThresholdComputation = std::variant<Undefined, CART, Random, Quantile>;

//------------------ Criterion

//Uses Gini as impurity parameter
struct Gini {};
//Uses Entropy as impurity parameter
struct Entropy{};
struct SSE{};

using Criterion = std::variant<Undefined, Gini, Entropy, SSE>;



/**
 * @brief struct controlling the policy of the split (split logic)
 *
 * SplitParams defines all the algorithmic choices used when searching
 * for the best split in a node:
 *  - the type of problem (classification or regression)
 *  - which impurity criterion is used (Gini, Entropy, ...)
 *  - how threshold candidates are generated
 *  - how features are selected
 *
 * @param type Tree family type {Regression, Classification}
 * @param criterion A criterion to use in {Gini, Entropy, SSE}
 * @param t_comp The threshold computation method {CART, Random, Quantile}
 * @param f_selection The feature selection method {AllFeatures, RandomK}
 * @param hparam The hyperparameters that need to be passed to the loop
 *
 * @note By default all parameters are set to Undefined. 
 * Construction must be explicit, either by specifying the
 * arguments or by calling ParamBuilder(TreeModel), which 
 * will return a SplitParam with predefined template of parameters
 *
 */
struct SplitParam {

    TreeType type; 
    Criterion criterion;
    ThresholdComputation t_comp;
    FeatureSelection f_selection;
    
};
