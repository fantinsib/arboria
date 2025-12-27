#pragma once

#include <vector>
#include <cmath>
#include <utility>
#include <stdexcept>


namespace arboria{
namespace split{

inline std::pair<int, int> count(const std::vector<float>& a){
    constexpr float EPS = 1e-6f;
    int pos_count = 0;
    int neg_count = 0;

    for (const auto& i : a) {
        if (std::abs(i-0.f)< EPS){neg_count++;}
        else if (std::abs(i-1.f)< EPS){pos_count++;}
        else {throw std::invalid_argument("arboria::split::count -> non-binary label detected : label not in {0,1}.");}
    }
    return {neg_count, pos_count};
}

inline float gini(float p1, float p2){
    // for p1 and p2 as proportions

    return 1.f - p1*p1 -p2*p2;
}

inline float gini(int n1, int n2){
    //for n1 and n2 the number of samples

    float denom = static_cast<float> (n1+n2);
    if (denom == 0){throw std::runtime_error("arboria::split::gini -> division by zero");}
    float p1 = static_cast<float>(n1) /denom;
    float p2 = static_cast<float>(n2)/denom;
    
    return 1.f - p1*p1 -p2*p2;
}

inline float gini(const std::vector<float>& a){
    //returns gini from a vector of targets 
    std::pair<int, int> nb_of_classes = arboria::split::count(a);
    return arboria::split::gini(nb_of_classes.first, nb_of_classes.second);
}

inline float weighted_gini(const std::vector<float>& l, const std::vector<float>& r){
    //takes the target vector of left node and target vector of right node and returns the weighted gini of the split

    float left_gini = gini(l);
    float l_size = static_cast<float>(l.size());
    float right_gini = gini(r);
    float r_size = static_cast<float>(r.size());
    float total_size = r_size+l_size;

    if (total_size == 0.f){
        throw std::runtime_error("arboria::split::weighted_gini -> total_size of vectors is zero");
    }

    return (l_size/total_size) * left_gini + (r_size/total_size) * right_gini;

}


}

}