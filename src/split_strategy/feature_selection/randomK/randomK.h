/*

            RANDOMK HEADER

*/


#include <vector> 
#include <span>
#include <random>

namespace arboria{
namespace feature_selection{



/**
 * @brief Returns randomly selected features index
 * 
 * @param features A non-owning view of a vector of feature index (size must be in (0,mtry))
 * @param mtry The number of features to be randomly selected
 * @param rng mt19937 random number generator
 * @throws std::invalid_argument if the number of features passed is inferior to mtry or if the vector is empty
 * @return the indices of the features
 */
std::vector<int> randomK(std::span<const int> features, const int mtry, std::mt19937& rng);


}
}