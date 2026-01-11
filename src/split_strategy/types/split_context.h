#pragma once




#include <random>

/**
 * @brief Struct used to carry the execution context of the model
 * to apply split policy 
 * 
 * @param seed The registered seed used for the RNG
 */

struct SplitContext{
    
    std::mt19937 rng;
    explicit SplitContext(std::uint32_t seed): rng(seed) {}
};

