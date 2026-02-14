#pragma once

#include "dataset/dataset.h"
#include <optional>
#include <span> 
#include <algorithm>
#include <random>
#include <stdexcept>


#include "split_strategy/sampling/sampling.h"


namespace arboria {
namespace split_strategy{


/**
 * @brief Generates random threshold
 * 
 * Draws n_random_split threshold following a uniform distribution 
 * in the interval [min(feature), max(feature)].
 * 
 * @param idx The idx of the samples passed in the dataset
 * @param col The feature 
 * @param data the dataset
 * @param n_random_split the number of thresholds to compute
 * @param rng a random number generator
 * @return std::vector<float> 
 */
inline std::vector<float> random_threshold(std::span<const int> idx,
                                            int col, 
                                            const arboria::DataSet& data,
                                            int n_random_split,
                                            std::mt19937& rng)
{

    const size_t idx_size = idx.size();
    if (idx_size == 0) throw std::invalid_argument("arboria::split_strategy::random_threshold : sorted_idx size cannot be 0");

    if (n_random_split == 0) throw std::invalid_argument("arboria::split_strategy::random_threshold : n_random_split size cannot be 0");

    std::uniform_int_distribution<size_t> index_dist(0, idx_size-1);

    float val1 = data.iloc_x(idx[index_dist(rng)], col);
    float val2 = data.iloc_x(idx[index_dist(rng)], col);

    if (val1 == val2) return {val1};

    float max_val = std::max(val1, val2);
    float min_val = std::min(val1, val2);

    std::uniform_real_distribution<float> dist(min_val, max_val);

    std::vector<float> thresholds;
    thresholds.reserve(n_random_split);

    for (size_t i = 0; i<n_random_split; i++ ){
        float r_i = dist(rng);
        thresholds.push_back(r_i);
    }
    return thresholds;
}




}
}