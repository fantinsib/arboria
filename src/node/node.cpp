#include "myforest/node.h"
#include <random>

namespace myforest {

Node::Node():
    feature_index(-1),
    threshold(0.0f),
    predicted_class(-1)

{}

int Node::return_feature_index() const{

    return this->feature_index;

}

float Node::return_threshold() const{

    return this->threshold;
}

}
