/*

            RANDOMK IMPLEMENTATION

*/

#include "randomK.h"
#include <random>
#include <stdexcept>
#include <span> 
#include <numeric>

namespace arboria{
namespace feature_selection{



std::vector<int> randomK(std::span<const int> features, const int mtry, std::mt19937& rng){

    if (features.size() == 0 || features.size()< mtry) throw std::invalid_argument("arboria::feature_selection::randomK : the number of passed features is invalid");
    if (mtry <= 0) throw std::invalid_argument("arboria::feature_selection::randomK : mtry value must be greater than or equal to zero");
    std::vector<int> vec(features.begin(), features.end());
    
    //Fisher-Yates style shuffle:
    // creating a sliding interval [i, total_size) ; at 
    // each iteration, select a random feature in [i, total_size) 
    // and swap its position with the value at i.
    // at the end, keep the first mtry positions. 
    for (int i = 0; i< mtry; i++){
        std::uniform_int_distribution<int> dist(i, vec.size()-1);
        int j = dist(rng);
        int a = vec[j];
        int b = vec[i];
        vec[j] = b;
        vec[i] = a;
    }
    return std::vector<int> (vec.begin(), vec.begin()+mtry);
};
//overload to process randomK by 
// passing only the number of features 
// instead of the full vector :
std::vector<int> randomK(int n_features, const int mtry, std::mt19937& rng){
    if (n_features <= 0) throw std::invalid_argument("arboria::feature_selection::randomK : the number of passed features is invalid");
    std::vector<int> features_vector(n_features);
    std::iota(features_vector.begin(), features_vector.end(), 0);
    return randomK(std::span<int>(features_vector), mtry, rng);

};

}
}