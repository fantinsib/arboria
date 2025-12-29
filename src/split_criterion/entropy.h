#pragma once

#include <vector>
#include <cmath>
#include <utility>
#include <stdexcept>

#include "helpers/helpers.h"

namespace arboria{
namespace split{

/**
 * @brief Computes the Shannon entropy from the proportion of the two classes
 * 
 * @param p1 The proportion of the first label (0 <= p1 <= 1)
 * @param p2 The proportion of the second label (0 <= p2 <= 1)
 * @throws std::invalid_argument if a proportion is not in bounds or if (p1 + p2) != 1
 * @return Entropy 
 */
inline float entropy(float p1, float p2){
    constexpr float EPS = 1e-6f;

    if (p1 < 0.f || p2 < 0.f || p1 > 1.f || p2 > 1.f) {throw std::invalid_argument("arboria::split::entropy -> proportions must be in [0,1].");}
    if (std::abs((p1+p2)-1.f) > EPS) {throw std::invalid_argument("arboria::split::entropy -> sum of proportions does not add up to one");}

    float H = 0.f;
    if (p1> 0.f) H -= p1*std::log2(p1);
    if (p2> 0.f) H -= p2*std::log2(p2);

    return H;
}

/**
 * @brief Computes the Shannon entropy from the numbers of samples in each class
 * 
 * @param n1 The number of samples of first label (n1 >= 0)
 * @param n2 The number of samples of second label (n2 >= 0)
 * @throws std::invalid_argument if a count is negative or if (n1 + n2) == 0 
 * @return Entropy 
 */
inline float entropy(int n1, int n2){

    if (n1<0 || n2 < 0) {throw std::invalid_argument("arboria::split::entropy -> Counts must be non-negative");}

    float total = static_cast<float> (n1+n2);

    if (total==0.f) {throw std::invalid_argument("arboria::split::entropy -> Empty node");}

    float p1 = static_cast<float>(n1) /total;
    float p2 = static_cast<float>(n2)/total;

    return entropy(p1,p2);

}

/**
 * @brief Computes the Shannon entropy from a vector of labels
 * 
 * @param a A 1D vector of labels {0,1}
 * @throws std::invalid_argument If the passed vector is empty
 * @return Entropy 
 */
inline float entropy(const std::vector<int>& a){

    if (a.empty()) {throw std::invalid_argument("arboria::split::entropy -> the passed vector is empty");}        
        auto [n1, n2] = arboria::helpers::count_classes(a);

        return entropy(n1, n2);
}

/**
 * @brief Computes the Shannon entropy from a vector of labels
 * 
 * @param a A 1D vector of labels {0,1}
 * @throws std::invalid_argument If the passed vector is empty
 * @return Entropy 
 */
inline float entropy(const std::vector<float>& a){

    if (a.empty()) {throw std::invalid_argument("arboria::split::entropy -> the passed vector is empty");}        
        auto [n1, n2] = arboria::helpers::count_classes(a);

        return entropy(n1, n2);
}

/**
 * @brief Computes the weighted entropy for two vectors passed to children nodes
 * 
 * @param l Labels passed to the left node 
 * @param r Labels passed to the right node 
 * @throws std::invalid_argument if both vectors are empty
 * @return Weighted entropy
 */
inline float weighted_entropy(const std::vector<int>& l, const std::vector<int>& r){

    float l_size = static_cast<float> (l.size());
    float r_size = static_cast<float> (r.size());
    float total_num_samples = l_size + r_size;
    if (total_num_samples == 0) throw std::invalid_argument("arboria::split::weighted_entropy -> passed vectors are empty");

    float left_entropy = (l_size > 0.f) ? entropy(l) : 0.f;
    float right_entropy = (r_size > 0.f) ? entropy(r) : 0.f;

    return (l_size/total_num_samples)*left_entropy + (r_size/total_num_samples)*right_entropy;

}

/**
 * @brief Computes the weighted entropy for two vectors passed to children nodes
 * 
 * @param l Labels passed to the left node 
 * @param r Labels passed to the right node 
 * @throws std::invalid_argument if both vectors are empty
 * @return Weighted entropy
 */
inline float weighted_entropy(const std::vector<float>& l, const std::vector<float>& r){

    float l_size = static_cast<float> (l.size());
    float r_size = static_cast<float> (r.size());
    float total_num_samples = l_size + r_size;
    if (total_num_samples == 0) throw std::invalid_argument("arboria::split::weighted_entropy -> passed vectors are empty");

    float left_entropy = (l_size > 0.f) ? entropy(l) : 0.f;
    float right_entropy = (r_size > 0.f) ? entropy(r) : 0.f;

    return (l_size/total_num_samples)*left_entropy + (r_size/total_num_samples)*right_entropy;

}


}
}