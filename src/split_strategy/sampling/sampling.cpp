/*
                    SAMPLING IMPLEMENTATION

*/

#include "sampling.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <numeric>
#include <stdexcept>

namespace arboria{
namespace sampling{


std::vector<size_t> bootstrap(size_t s_size, size_t n_samples, std::mt19937& rng){

    if (s_size == 0) throw std::invalid_argument("arboria::sampling::bootstrap : number of samples must be superior to zero");
    if (n_samples == 0) throw std::invalid_argument("arboria::sampling::bootstrap : number of bootstrapped samples must be strictly positive");
    std::uniform_int_distribution<size_t> dist(0, s_size-1);

    std::vector<size_t> output(n_samples);

    for (size_t i = 0; i < n_samples; i++) {
        output[i] = dist(rng);
    }

    return output;
}


std::vector<size_t> subsample(size_t s_size, size_t n_samples, std::mt19937& rng){

    if (s_size == 0) throw std::invalid_argument("arboria::sampling::subsampling : number of samples must be superior to zero");
    if (n_samples == 0 || n_samples > s_size) throw std::invalid_argument("arboria::sampling::subsampling : number of drawn samples must be strictly positive and less than or equal to number of samples");

    std::vector<size_t> vec(s_size);
    std::iota(vec.begin(), vec.end(),0);

    //Implementing Fischer-Yates style algorithm :
    for (size_t i = 0; i <n_samples; i++){

        std::uniform_int_distribution<size_t> dist(i, s_size-1);
        size_t j = dist(rng);
        std::swap(vec[i], vec[j]);
    }
    vec.resize(n_samples);
    return vec;
}

}}
