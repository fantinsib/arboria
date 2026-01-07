/*
                    SAMPLING IMPLEMENTATION

*/

#include "sampling.h"


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


}}