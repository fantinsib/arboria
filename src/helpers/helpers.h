#pragma once

#include <vector>
#include <utility>
#include <stdexcept>
#include <cmath>

namespace arboria {
namespace helpers {

/**
 * @brief Returns the count of positive and negative labels in a target vector
 * 
 * @param a the target vector. All labels must be {0,1}.
 * @throws std::invalid_argument if a label is not binary
 * @return Pair : {pos_count, neg_count} with
 *      - pos_count the number of positive (1) labels in the vector
 *      - neg_count the number of negative (0) labels in the vector
 */
inline std::pair<int, int> count_classes(const std::vector<float>& a){
    constexpr float EPS = 1e-6f;
    int pos_count = 0;
    int neg_count = 0;

    for (const auto& i : a) {
        if (std::abs(i-0.f)< EPS){neg_count++;}
        else if (std::abs(i-1.f)< EPS){pos_count++;}
        else {throw std::invalid_argument("arboria::helpers::count_classes -> non-binary label detected : label not in {0,1}.");}
    }
    return {pos_count, neg_count};
}



}
}
