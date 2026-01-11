#pragma once

enum class Criterion { Gini, Entropy };
enum class ThresholdComputation { CART, Random, Quantile };
enum class FeatureSelection { All, RandomK };
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

    Criterion criterion = Criterion::Gini;
    ThresholdComputation t_comp = ThresholdComputation::CART;
    FeatureSelection f_selection = FeatureSelection::All;

    int mtry = -1; // used only if f_selection is RandomK 

};