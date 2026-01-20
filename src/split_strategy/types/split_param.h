#pragma once
#include <variant>
#include <cstddef>

//--------------------- FeatureSelection

//Searches all features during split
struct AllFeatures {};

//Searches only mtry features at each split
struct RandomK{
    int mtry = -1; 
};

using FeatureSelection = std::variant<AllFeatures, RandomK>;

//-------------------- ThresholdComputation

//Computes the threshold according to regular CART algorithm
struct CART{};
struct Random{};
struct Quantile{};

using ThresholdComputation = std::variant<CART, Random, Quantile>;

//------------------ Criterion

//Uses Gini as impurity parameter
struct Gini {};
//Uses Entropy as impurity parameter
struct Entropy{};

using Criterion = std::variant<Gini, Entropy>;

/**
 * @brief struct controlling the policy of the split (split logic)
 *
 * SplitParams defines all the algorithmic choices used when searching
 * for the best split in a node:
 *  - which impurity criterion is used (Gini, Entropy, ...)
 *  - how threshold candidates are generated
 *  - how features are selected
 *
 */
struct SplitParam {

    Criterion criterion;
    ThresholdComputation t_comp;
    FeatureSelection f_selection;

};