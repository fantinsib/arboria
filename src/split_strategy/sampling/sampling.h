/*
                    SAMPLING HEADER

*/
#pragma once
#include <random>
#include <span>
#include <vector>



// ---------------------- Boostrapping
namespace arboria{
namespace sampling{

/**
 * @brief Returns a vector of indices with the bootstrapping method
 * 
 * @param s_size The number of indices in the data to bootstrap (must be strictly positive)
 * @param n_samples The number of draws (must be strictly positive)
 * @param rng A random number generator
 * @throws std::invalid_argument if s_size or n_samples is less than or equal to zero
 * @returns a std::vector<int> of bootstrapped indices
 * @note Indices are sampled with replacement 
 */
std::vector<size_t> bootstrap(size_t s_size, size_t n_samples, std::mt19937& rng);
}}