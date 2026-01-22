#pragma once




#include <optional>



/**
 * @brief Struct passing the hyper parameters for models
 *
 * @param mtry Optional max feature parameter 
 * @param n_estimators Optional number of trees to train in the RandomForest
 * @param max_depth Optional maximum depth allowed to be reached in training
 * @param max_samples Optional percentage of total samples to be bootstrapped in RF 
 * @param min_sample_split Optional minimum number of samples allowed in a leaf
 * 
 */
struct HyperParam{

    std::optional<int> mtry=std::nullopt;
    std::optional<int> n_estimators=std::nullopt;
    std::optional<int> max_depth=std::nullopt;
    std::optional<float> max_samples=std::nullopt;
    std::optional<float> min_sample_split=std::nullopt;
    
};