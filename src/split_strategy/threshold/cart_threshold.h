
// See notes_split_logic.md for details about the implementation 

#pragma once
#include <stdexcept>
#include <vector>
#include <span>
#include <algorithm>

#include "dataset/dataset.h"

namespace arboria{
namespace split_strategy{
    
/**
 * @brief Generates a vector of thresholds candidates for a 
 * specific feature of the DataSet and for selected samples
 * 
 * @param idx A span of row indices (.size() >= 2)
 * @param col The col index of the feature (0 < col < data.n_cols())
 * @param data A reference to the DataSet containing the data
 * @throws std::invalid_argument if the DataSet is empty, if the col value is illegal
 * @return threshold vector
 * @note !!! Passed idx must be already sorted
 */
inline std::vector<float> cart_threshold(const std::span<const int> sorted_idx, int col, const arboria::DataSet& data){

    if (data.is_empty()) {throw std::invalid_argument("arboria::split_strategy::cart_threshold : DataSet is empty.");}
    if (col < 0) {throw std::invalid_argument("arboria::split_strategy::cart_threshold : column index number must be non-negative.");}
    if (col >= data.n_cols()) {throw std::invalid_argument("arboria::split_strategy::cart_threshold : no such column in the dataset.");}
    if (sorted_idx.size()<2)  {throw std::invalid_argument("arboria::split_strategy::cart_threshold : the idx span must reference at least two values.");}

    //starts by sorting the col -> returns the index of the samples sorted by value along the col

    std::vector<float> output;
    output.reserve(sorted_idx.size()-1);
    for (size_t i = 0;  i < sorted_idx.size()-1; i ++){

        float a = data.iloc_x(sorted_idx[i], col);
        float b = data.iloc_x(sorted_idx[i+1], col);

       if (a ==b) continue;
       output.push_back(((a)+(b))/2.f);
    }

    return output;
}
}
}
