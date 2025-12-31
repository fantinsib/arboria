#ifndef NODE_H
#define NODE_H
#include <memory>
#include <vector>

namespace arboria{
class Node
{
public:
    /**
     * @brief Creates a new Node 
     * 
     * Node stores key informations for a split :
     * feature_index : the column index on which the split is made (default value -1)
     * threshold : the threshold that splits the targeted feature (default value 0.0)
     * predicted_class : the majority class received by the node (default value -1)
     * left_child and right_child : pointers to the next nodes
     */
    Node();
    int feature_index;
    float threshold;
    int predicted_class;

    bool is_leaf = true;

    int return_feature_index() const;
    float return_threshold() const;

    std::vector<int> rows;

    std::unique_ptr<Node> left_child;
    std::unique_ptr<Node> right_child;
};
}
#endif // NODE_H