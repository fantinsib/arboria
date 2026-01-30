

#pragma once

#include "tree/TreeModel.h"
#include "split_strategy/types/split_param.h"
#include <optional>


namespace arboria{
/**
 * @brief Builds a fully specified SplitParam from user-provided options and model invariants.
 *
 * This function centralizes the construction of @ref SplitParam to avoid scattered
 * "patches" across the codebase. It combines:
 * (1) optional user inputs (criterion, threshold computation, feature selection),
 * and (2) hard model invariants (e.g., a RandomForest defaults to RandomK if no
 * feature-selection strategy is provided).
 *
 * If an optional argument is not provided, a default policy is applied. The impurity
 * criterion and threshold computation default to @ref Gini and @ref CART respectively.
 * The default feature-selection strategy depends on the model:
 * - DecisionTree: defaults to @ref AllFeatures
 * - RandomForest: defaults to @ref RandomK
 *
 * @note If @ref RandomK is constructed with an unresolved mtry (e.g. mtry == -1),
 *       its resolution is expected to be handled later during fit, once the number
 *       of features is known.
 *
 * @param model The model kind (DecisionTree or RandomForest) used to enforce invariants.
 * @param crit Optional impurity criterion. If not provided, defaults to @ref Gini.
 * @param threshold Optional threshold computation strategy. If not provided, defaults to @ref CART.
 * @param feature Optional feature-selection strategy. If not provided, defaults to
 *        @ref AllFeatures for DecisionTree and @ref RandomK for RandomForest.
 *
 * @return A @ref SplitParam with all fields explicitly set.
 *
 * @throws std::logic_error If @p model is not supported by this builder.
 */
SplitParam ParamBuilder(const TreeModel model,
                        std::optional<TreeType> type = std::nullopt,
                        std::optional<Criterion> crit = std::nullopt,
                        std::optional<ThresholdComputation> threshold= std::nullopt, 
                        std::optional<FeatureSelection> feature = std::nullopt);


}
