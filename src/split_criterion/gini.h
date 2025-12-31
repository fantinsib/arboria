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
 * @param p1 the proportion of first label (0 <= p1 <= 1)
 * @param p2 the proportion of second label (0 <= p2 <= 1)
 * @throws std::invalid_argument if p1 or p2 is negative or if (p1 + p2) != 1
 * @return Gini impurity 
 */
inline float gini(float p1, float p2){
    
    constexpr float EPS = 1e-6f;

    if (p1 < 0.f || p2 < 0.f || p1 > 1.f || p2 > 1.f) {throw std::invalid_argument("arboria::split::gini -> proportions must be in [0,1].");}
    if (std::abs((p1+p2)-1.f) > EPS) {throw std::invalid_argument("arboria::split::gini -> sum of proportions does not add up to one");}
    
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
    
    if (n1 < 0 || n2 < 0) {throw std::invalid_argument("arboria::split::gini -> number of samples must be non-negative");}
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
 * @throws std::invalid_argument If the vector contains non-binary labels (not in {0,1}) or if is empty
 * @return Gini impurity 
 */
inline float gini(const std::vector<float>& a){
    if (a.empty()) {throw std::invalid_argument("arboria::split::gini -> the passed vector is empty");}        
    std::pair<int, int> nb_of_classes = arboria::helpers::count_classes(a);
    return arboria::split::gini(nb_of_classes.first, nb_of_classes.second);
}

/**
 * @brief Computes Gini impurity from a vector of labels
 * 
 * @param a A vector of binary labels ({0,1})
 * @throws std::invalid_argument If the vector contains non-binary labels (not in {0,1}) or if is empty
 * @return Gini impurity 
 */
inline float gini(const std::vector<int>& a){
    if (a.empty()) {throw std::invalid_argument("arboria::split::gini -> the passed vector is empty");}        
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

    float l_size = static_cast<float> (l.size());
    float r_size = static_cast<float> (r.size());
    float total_num_samples = l_size + r_size;
    if (total_num_samples == 0) throw std::invalid_argument("arboria::split::weighted_gini -> passed vectors are empty");

    float left_gini = (l_size > 0.f) ? gini(l) : 0.f;
    float right_gini = (r_size > 0.f) ? gini(r) : 0.f;


    return (l_size/total_num_samples) * left_gini + (r_size/total_num_samples) * right_gini;

}

/**
 * @brief Returns the weighted Gini impurity from the number of pos and neg labels in the left and right node
 * 
 * @param l_pos : the number of positive label in the left child
 * @param l_neg : the number of negative label in the left child
 * @param r_pos : the number of positive label in the right child
 * @param r_neg : the number of negative label in the right child
 * @throws std::invalid_argument if the total sum of the parameters is zero or if a value is negative
 * @return Weighted Gini Impurity
 */
inline float weighted_gini(int l_pos, int l_neg, int r_pos, int r_neg){

    if (l_pos < 0 || l_neg < 0 || r_pos < 0 || r_neg < 0) {throw std::invalid_argument("arboria::split::weighted_gini -> values must be non-negative");}

    const float total_num_samples = l_pos + l_neg + r_pos + r_neg;
    const float l_size = static_cast<float>(l_pos) + static_cast<float>(l_neg);
    const float r_size = static_cast<float>(r_pos) + static_cast<float>(r_neg);
    if (total_num_samples == 0.f) throw std::invalid_argument("arboria::split::weighted_gini -> no values were passed");

    const float left_gini = (l_size > 0.f) ? gini(l_pos, l_neg) : 0.f;
    const float right_gini = (r_size > 0.f) ? gini(r_pos, r_neg) : 0.f;


    return (l_size/total_num_samples) * left_gini + (r_size/total_num_samples) * right_gini;

}

}

}