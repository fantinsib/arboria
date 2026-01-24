#pragma once

#include <vector>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <span>

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


/**
 * @brief Returns the count of positive and negative labels in a target vector
 * 
 * @param a the target vector. All labels must be {0,1}.
 * @throws std::invalid_argument if a label is not binary
 * @return Pair : {pos_count, neg_count} with
 *      - pos_count the number of positive (1) labels in the vector
 *      - neg_count the number of negative (0) labels in the vector
 */
inline std::pair<int, int> count_classes(const std::vector<int>& a){
    
    int pos_count = 0;
    int neg_count = 0;

    for (const auto& i : a) {
        if (i == 0){neg_count++;}
        else if (i==1) {pos_count++;}
        else {throw std::invalid_argument("arboria::helpers::count_classes -> non-binary label detected : label not in {0,1}.");}
    }
    return {pos_count, neg_count};
}

/**
 * @brief Returns the count of positive and negative labels in a target vector from a span of indices
 * 
 * 
 * @param idx a span of row index. All index must be 0 <= i < targets.size()
 * @param targets the target vector. All labels must be {0,1}.
 * @throws std::invalid_argument if a label is not binary
 * @throws std::out_of_range if an index is not in range [0, targets.size()) 
 * @return Pair : {pos_count, neg_count} with
 *      - pos_count the number of positive (1) labels in the vector
 *      - neg_count the number of negative (0) labels in the vector
 */
inline std::pair<int, int> count_classes(std::span<const int> idx, const std::vector<int>& targets) {
    
    int pos_count = 0;
    int neg_count = 0;
    int vec_size = targets.size();

    for (int i :idx) {
        if (i < 0 || i >= vec_size) throw std::out_of_range("arboria::helpers::count_classes -> one of the referenced index is out of bounds for target vector");
        if (targets[i] == 1) {pos_count++;}
        else if (targets[i] == 0) {neg_count++;}
        else throw std::invalid_argument("arboria::helpers::count_classes -> non-binary label detected : label not in {0,1}.");
    }
    return {pos_count, neg_count};

}

inline std::pair<int, int> count_classes(std::span<const int> idx, const std::vector<float>& targets) {
    
    int pos_count = 0;
    int neg_count = 0;
    int vec_size = targets.size();

    for (int i :idx) {
        if (i < 0 || i >= vec_size) throw std::out_of_range("arboria::helpers::count_classes -> one of the referenced index is out of bounds for target vector");
        if (targets[i] == 1.f) {pos_count++;}
        else if (targets[i] == 0.f) {neg_count++;}
        else throw std::invalid_argument("arboria::helpers::count_classes -> non-binary label detected : label not in {0,1}.");
    }
    return {pos_count, neg_count};

}

inline float accuracy(const std::span<const int> a, const std::span<const int> b){

    const size_t n = a.size();
    if (n != b.size()) throw std::invalid_argument("arboria::helpers::accuracy : passed arguments have different lenght");
    if (n == 0) throw std::invalid_argument("arboria::helpers::accuracy : passed arguments are empty");
    size_t equal_count = 0;
    for (size_t i = 0; i < a.size(); i++){
        equal_count += (a[i]==b[i]);
    }
    return static_cast<float>(equal_count)/static_cast<float>(a.size());

}

inline uint64_t derive_seed(uint64_t seed, size_t i) {
    return seed + 0x9E3779B97F4A7C15ULL * i;
}


}
}
