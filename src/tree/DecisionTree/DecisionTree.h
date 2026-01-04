
#pragma once
#include <vector>
#include <span>

#include "node/node.h"
#include "dataset/dataset.h"
#include "split_strategy/splitter.h"
#include "helpers/helpers.h"


using arboria::split_strategy::Splitter;
using arboria::helpers::count_classes;

namespace arboria {


class DecisionTree

{

    public:
        /**
        * @brief Construct a DecisionTree with a maximum depth.
        *
        * @param max_depth Maximum allowed depth of the tree.
        *        A depth of 0 results in a single leaf node.
        */
        DecisionTree(int max_depth);
        
        /**
        * @brief Fit the decision tree on a dataset.
        * Builds the tree recursively starting from the root node.
        *
        * @param data DataSet object containing samples and targets
        * @throws std::invalid_argument if the dataset is empty or invalid.
        */
        void fit(const DataSet& data);
        
        /**
         * @brief Maximum depth allowed for the construction of the DecisionTree
         */
        int max_depth;
        
        /**
         * @brief Number of features seen in the DataSet during training
         */
        int num_features;

    private:
        /**
        * @brief Recursively build the decision tree
        *
        * This method partitions the provided index span according to the
        * best available split and recursively constructs child nodes
        *
        * @param data Training dataset
        * @param node Current node 
        * @param idx Span of row indices corresponding to the samples
        * reaching this node.
        * @param depth Current depth in the tree
        */
        void fit_(const DataSet& data, Node& node, std::span<int> idx, int depth);
        bool fitted = false;
        Node root_node;
        Splitter splitter;
};
//end of namespace arboria 
}