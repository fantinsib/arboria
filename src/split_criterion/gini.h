#pragma once

#include <vector>
#include <cmath>
#include <utility>
#include <stdexcept>

#include "helpers/helpers.h"

namespace arboria{
namespace split{


/**
 * @brief Returns Gini impurity from proportions
 *
 * The Gini impurity is defined as:
 *   G = 1 - p1² - p2²
 * 
 * @param p1 the proportion of first label
 * @param p2 the proportion of second label
 * @return Gini impurity 
 */
inline float gini(float p1, float p2){
    return 1.f - p1*p1 -p2*p2;
}

/**
 * @brief Computes Gini impurity from class counts
 * 
 * @param n1 the number of samples with the first label
 * @param n2 the number of samples with the second label 
 * @throws std::runtime_error if n1 + n2 = 0
 * @return Gini impurity 
 */
inline float gini(int n1, int n2){
    
    float denom = static_cast<float> (n1+n2);
    if (denom == 0){throw std::runtime_error("arboria::split::gini -> division by zero");}
    float p1 = static_cast<float>(n1) /denom;
    float p2 = static_cast<float>(n2)/denom;
    
    return arboria::split::gini(p1,p2);
}

/**
 * @brief Computes Gini impurity from a vector of labels
 * 
 * @param a A vector of binary labels ({0,1})
 * @throws std::invalid_argument If the vector contains non-binary labels (not in {0,1})
 * @return Gini impurity 
 */
inline float gini(const std::vector<float>& a){
    std::pair<int, int> nb_of_classes = arboria::helpers::count_classes(a);
    return arboria::split::gini(nb_of_classes.first, nb_of_classes.second);
}

/**
 * @brief Returns the weighted Gini impurity for two vectors of labels
 * 
 * @param l The left vector of labels
 * @param r The right vector of labels
 * @throws std::invalid_argument if both vector do not contain any values
 * @return Weighted Gini Impurity
 */
inline float weighted_gini(const std::vector<float>& l, const std::vector<float>& r){

    float left_gini = gini(l);
    float l_size = static_cast<float>(l.size());
    float right_gini = gini(r);
    float r_size = static_cast<float>(r.size());
    float total_size = r_size+l_size;

    if (total_size == 0.f){
        throw std::invalid_argument("arboria::split::weighted_gini -> total_size of vectors is zero");
    }

    return (l_size/total_size) * left_gini + (r_size/total_size) * right_gini;

}


}

}