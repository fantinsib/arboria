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
 * @returns a std::vector<size_t> of bootstrapped indices
 * @note Indices are sampled with replacement 
 */
std::vector<size_t> bootstrap(size_t s_size, size_t n_samples, std::mt19937& rng);


/**
 * @brief Returns a vector of samples
 *
 * @param s_size the number of indices in the sampled data (must be > 0)
 * @param n_samples the number of samples to return. Must be in (0, s_size]
 * @param rng A random number generator 
 * @throws std::invalid_argument if s_size is less than or equal to zero or if n_sample not in (0, s_size]
 * @returns a std::vector<size_t> of sampled indices
 * @note Indices are sampled without replacement 
 */
std::vector<size_t> subsample(size_t s_size, size_t n_samples, std::mt19937& rng);

}}
