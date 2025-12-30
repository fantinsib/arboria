
// See notes_split_logic.md for details about the implementation 

#pragma once
#include <vector>
#include <span>

#include "dataset/dataset.h"

/**
 * @brief Generates a vector of thresholds candidates for a 
 * specific feature of the DataSet and for selected samples
 * 
 * @param idx A span of row indices
 * @param col The col index of the feature
 * @param data A reference to the DataSet 
 * @return std::vector<float> 
 */

inline std::vector<float> cart_threshold(const std::span<const int> idx, int col, const arboria::DataSet& data){

    //starts by sorting the col -> returns the index of the samples sorted by value along the col
    std::vector<int> sorted_idx(idx.begin(), idx.end());
    std::sort(
        sorted_idx.begin(),
        sorted_idx.end(),
        [&](int a, int b) {
            return data.iloc_x(a, col) < data.iloc_x(b, col);
        }
    );

    //Then computes the midpoints
    std::vector<float> output(sorted_idx.size()-1);
    for (int i = 0;  i < sorted_idx.size()-1; i ++){

        float a = data.iloc_x(sorted_idx[i], col);
        float b = data.iloc_x(sorted_idx[i+1], col);

        output[i] = ((a+b)/2.f);
        

    }

    return output;
}
