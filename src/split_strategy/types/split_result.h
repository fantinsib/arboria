#pragma once

#include <cmath>
#include <vector>
#include <span>
#include <limits>


/**
 * @brief struct that carries the result of a split in the dataset
 * 
 * - split_feature : the feature on which the split is made
 * - split_threshold : the value on which the split is made
 * - score : value of criterion used
 */
struct SplitResult {

    int split_feature =-1;
    float split_threshold = std::numeric_limits<float>::quiet_NaN();
    float score = std::numeric_limits<float>::infinity();

    
    bool has_split() const {
        return split_feature >= 0 && std::isfinite(score);
    }

};

