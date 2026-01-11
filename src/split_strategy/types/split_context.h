#pragma once




#include <random>

/**
 * @brief Struct used to pass arguments to the Splitter::best_split function depending on the 
 * type of tree 
 * 
 * 
 */

struct SplitContext{
    
    std::mt19937 rng;
    explicit SplitContext(std::uint32_t seed): rng(seed) {}
};

