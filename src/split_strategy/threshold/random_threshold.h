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

    auto min_max = std::minmax_element(idx.begin(), idx.end(),
    [&](int i, int j) {return data.iloc_x(i,col) < data.iloc_x(j, col);});

    float min_val = data.iloc_x(*min_max.first, col);
    float max_val = data.iloc_x(*min_max.second, col);

    if (min_val > max_val) throw std::logic_error("arboria::split_strategy::random_threshold : min_val > max_val");

    if ((max_val - min_val) < 1e-7) {
        return std::vector<float>{min_val};
    }

    std::uniform_real_distribution<float> dist(min_val, max_val);

    std::vector<float> thresholds(n_random_split);

    for (size_t i = 0; i<n_random_split; i++ ){
        float r_i = dist(rng);
        thresholds[i] = r_i;
    }
    return thresholds;
}




}
}