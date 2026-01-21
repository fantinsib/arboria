#pragma once




#include <optional>

/**
 * @brief Struct containing hyper parameters for models.
 * 
 * 
 */

struct HyperParam{

    std::optional<int> mtry;
    std::optional<int> n_estimators;
    std::optional<int> max_depth;
    std::optional<int> max_samples;
    
};