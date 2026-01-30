#include "node.h"
#include <random>
#include <limits>
#include <cmath>

namespace arboria {

Node::Node():
    feature_index(-1),
    threshold(std::numeric_limits<float>::quiet_NaN()),
    leaf_value(-1)
{}

int Node::return_feature_index() const{
    return this->feature_index;
}

float Node::return_threshold() const{
    return this->threshold;
}

bool Node::is_valid(int n_features) const{

    if (feature_index < 0 || feature_index >= n_features) //check if registered feature on which 
        return false;                                     // on which the split takes place is indeed 
    if (!std::isfinite(threshold))                     // in the range of the number of features in the tree
        return false; //check that threshold was initialized
    if (!left_child || !right_child) //check that has child, given it's not a leaf 
        return false;
    return true;
}

}
