#pragma once
#include <optional>
#include <variant>
#include <cstddef>

//--------------------- FeatureSelection

struct Undefined{};

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

using Criterion = std::variant<Undefined, Gini, Entropy>;

/**
 * @brief struct controlling the policy of the split (split logic)
 *
 * SplitParams defines all the algorithmic choices used when searching
 * for the best split in a node:
 *  - which impurity criterion is used (Gini, Entropy, ...)
 *  - how threshold candidates are generated
 *  - how features are selected
 *
 * @param criterion A criterion to use in {Gini, Entropy}
 * @param t_comp The threshold computation method {CART, Random, Quantile}
 * @param f_selection The feature selection method {AllFeatures, RandomK}
 *
 * @note By default all parameters are set to Undefined. 
 * Construction must be explicit, either by specifying the
 * arguments or by calling ParamBuilder(TreeModel), which 
 * will return a SplitParam with predefined template of parameters
 *
 */
struct SplitParam {

    Criterion criterion;
    ThresholdComputation t_comp;
    FeatureSelection f_selection;

};