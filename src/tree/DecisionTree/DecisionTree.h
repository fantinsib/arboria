
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
        *        
        */
        DecisionTree(int max_depth);
        
        /**
        * @brief Fit the decision tree on a dataset.
        * Builds the tree recursively starting from the root node.
        *
        * @param data DataSet object containing samples and targets
        * @param params SplitParam object passing the criterion used, 
        * the threshold computation method and the feature selection policy
        * @note If no valid split is found, the node becomes a leaf ; leaf
        * prediction is the majority class. In case of a tie, class prediction is 1.
        * @throws std::invalid_argument if the dataset is empty or invalid.
        */
        void fit(const DataSet& data, const SplitParam& params);


        /**
         * @brief Predict the class of the passed sample
         * 
         * @param sample std::span view into a vector containing 
         * a unique sample. Sample must have the same number of
         * features as seen in training (sample.size() == num_features) 
         * @throw std::invalid_argument if sample number of features
         * and training feature differ 
         * @return the predicted class 
         */
        int predict_one(const std::span<const float> sample) const;
        
        /**
         * @brief Predict the class of a set of samples
         * The input is expected to be a flat, row-major buffer containing
         * consecutive samples. Each sample must have the same number of
         * features as the data used during training
         * @param sample Non owning view over a row-major representation
         * of a set of samples. The number of features of the samples must
         * be coherent with the number of features seen in training.
         * @throws std::invalid_argument if the tree has not yet been fitted or
         * if samples dimensions are incompatible with training dataset dimensions.
         * @return a vector of int of the predicted class
         */
        std::vector<int> predict(const std::span<const float> samples) const;
     
        //Maximum depth allowed for the construction of the DecisionTree
        int max_depth;
        //Number of features seen in the DataSet during training
        int num_features;
        //Getter for fitted
        inline bool is_fitted() const {return fitted;}

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
         * @param params SplitParam object passing the criterion used, 
         * the threshold computation method and the feature selection policy
         */
        void fit_(const DataSet& data, Node& node, std::span<int> idx, int depth, const SplitParam& params);
        
        /**
         * @brief Returns the predicted class of the current node or pass the sample 
         * to the left/right child 
         *
         * @param sample a sample with .size() = num_features
         * @param node the current node 
         * @return the predicted class
         */
        int predict_one_(const std::span<const float> sample, const Node& node) const;
        
        bool fitted = false;
        Node root_node;
        Splitter splitter;
};
//end of namespace arboria 
}