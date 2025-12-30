

#include <vector>
#include <span>


/**
 * @brief struct that carries the result of a split in the dataset
 * 
 * - split_feature : the feature on which the split is made
 * - split_threshold : the value on which the split is made
 * - score : value of criterion used
 */
struct SplitResult {

    int split_feature;
    float split_threshold;
    float score = std::numeric_limits<float>::infinity();

};