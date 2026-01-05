
#include "DecisionTree.h"
#include "helpers/helpers.h"


#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <cmath>



namespace arboria {

DecisionTree::DecisionTree(int max_depth_):
    max_depth(max_depth_) {}

void DecisionTree::fit(const DataSet& data, const SplitParam& params) {

    int n_rows = data.n_rows();
    int n_cols = data.n_cols();
    if (n_rows <= 1) {throw std::invalid_argument("arboria::DecisionTree::fit -> invalid fitted DataSet");}
    //index buffer creation 
    std::vector<int> buffer(n_rows);
    std::iota(buffer.begin(), buffer.end(), 0);
    std::span<int> idx(buffer);
    fit_(data, root_node, idx, 0, params);
    fitted = true; 
    num_features = n_cols;
}

int DecisionTree::predict_one(const std::span<const float> sample) const{
    if (!fitted) {throw std::invalid_argument("arboria::DecisionTree::predict_one -> tree has not been fitted");}
    if (sample.size() != num_features) throw std::invalid_argument("arboria::DecisionTree::predict_one -> the passed sample for prediction has different number of features than seen in training");
    return predict_one_(sample, root_node);
}

std::vector<int> DecisionTree::predict(const std::span<const float> samples) const {

    if (!fitted || num_features == 0) throw std::invalid_argument("arboria::DecisionTree::predict -> tree has not been fitted");
    size_t nf = static_cast<size_t>(num_features);
    if (samples.size() % nf != 0) throw std::invalid_argument("arboria::DecisionTree::predict -> passed samples do not have the correct dimension");

    size_t num_samples = samples.size()/nf;
    std::vector<int> preds(num_samples);

    for (size_t s = 0; s<num_samples; s++){
        auto sample = samples.subspan(s*nf, nf);
        preds[s] = predict_one(sample);
    }
    
    return preds;
}






//############ Private ####

int DecisionTree::predict_one_(const std::span<const float> sample, const Node& node) const{


    if (node.is_leaf) {return node.predicted_class;}

    if (!node.is_valid(num_features)) {
        throw std::logic_error("arboria::DecisionTree::predict_one_ -> Invalid node reached");
    }


    int n_col = node.return_feature_index();
    float threshold = node.return_threshold();
    float sample_feature = sample[n_col];

    if (std::isnan(sample_feature)) throw std::invalid_argument("arboria::DecisionTree::predict_one_ -> sample contains NaN.");

    if (sample[n_col] >= threshold){return predict_one_(sample, *node.right_child);}
    else {return predict_one_(sample, *node.left_child);}
}

void DecisionTree::fit_(const DataSet& data, Node& node, std::span<int> idx, int depth, const SplitParam& params){

    //lambda function to stop iteration :
    auto end_branch= [&](){
        node.is_leaf = true;
        std::pair<int,int>count = helpers::count_classes(idx, data.y());
        int pos_count = count.first;
        int neg_count = count.second;

        (pos_count >= neg_count) ? node.predicted_class= 1 : node.predicted_class=0 ; // ">=" : in case of tie break, node predicted class = 1
        return;
    };

    std::pair<int,int>count = helpers::count_classes(idx, data.y());
    int pos_count = count.first;
    int neg_count = count.second;

    //case idx refers to less than 1 sample
    if (idx.size() <= 1) {end_branch(); return;}
    //case if the current node is pure
    if ((pos_count == 0) || (neg_count ==0)) {end_branch();return;}
    //case max depth is reached:
    if (depth == max_depth) {end_branch(); return;}

    //Compute the split :
    SplitResult split = splitter.best_split(idx, data, params);

    if (split.has_split() == false) {end_branch();return;}

    if (split.has_split()){
        int feature_index= split.split_feature;
        float threshold = split.split_threshold;
        node.feature_index = feature_index;
        node.threshold = threshold;
        node.is_leaf = false;

        auto mid = std::partition(idx.begin(), idx.end(), 
            [&](int i) {return data.iloc_x(i, feature_index) < threshold;});

        std::size_t left_size = static_cast<std::size_t>(mid - idx.begin());
        std::size_t right_size =  idx.size() - left_size;

        if (left_size==0 || right_size == 0){end_branch(); return;}

        std::span<int> left_idx(idx.data(), left_size);
        std::span<int> right_idx(idx.data()+left_size, right_size);

        node.left_child  = std::make_unique<Node>();
        node.right_child = std::make_unique<Node>();
        
        fit_(data, *node.left_child, left_idx, depth+1, params);
        fit_(data, *node.right_child, right_idx, depth+1, params);
        

    }    
    
}







}