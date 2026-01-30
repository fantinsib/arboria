
#include "DecisionTree.h"
#include "helpers/helpers.h"
#include "split_strategy/types/split_context.h"
#include "split_strategy/types/split_hyper.h"
#include "split_strategy/types/split_param.h"
#include "tree/TreeModel.h"


#include <algorithm>
#include <cstddef>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <cmath>
#include <variant>



namespace arboria {

DecisionTree::DecisionTree(HyperParam h_param, TreeType type)
{
    if (h_param.max_depth.has_value()){
        if (*h_param.max_depth <= 0 && *h_param.max_depth != -99 && *h_param.max_depth != -98) throw std::invalid_argument("arboria::tree::DecisionTree : max_depth argument must be greater than or equal 0");
        max_depth = *h_param.max_depth;
    }
    if (h_param.min_sample_split.has_value()){
        if (*h_param.min_sample_split <= 0) throw std::invalid_argument("arboria::tree::DecisionTree : min_sample_split argument must be greater than or equal 0");
        min_sample_split = *h_param.min_sample_split;
    }

    if (std::holds_alternative<Classification>(type) || std::holds_alternative<Regression>(type)){
    type_ = type;
    }
    else throw std::invalid_argument("DecisionTree constructor : invalid type");

}

    //Base fitting function with 1 full DataSet and params
void DecisionTree::fit(const DataSet& data, const SplitParam& params) {

    int n_rows = data.n_rows();
    int n_cols = data.n_cols();
    if (n_rows <= 1) {throw std::invalid_argument("arboria::DecisionTree::fit -> invalid fitted DataSet");}
    //index buffer creation 
    std::vector<int> buffer(n_rows);
    std::iota(buffer.begin(), buffer.end(), 0);
    std::span<int> idx(buffer);

    if (std::holds_alternative<Undefined>(params.criterion) || std::holds_alternative<Undefined>(params.f_selection) || std::holds_alternative<Undefined>(params.t_comp)){
        throw std::invalid_argument("arboria::DecisionTree::fit : params passed to fit function contain an undefined component");
    }

    fit(data,idx, params);
    fitted = true; 
    num_features = n_cols;
}

//overload for a specific selection of rows of the dataset and context provider
void DecisionTree::fit(const DataSet& data, 
                       const std::span<int> idx, 
                       const SplitParam& params, 
                       std::optional<std::reference_wrapper<SplitContext>> context) {

    int n_rows = data.n_rows();
    int n_cols = data.n_cols();
    if (n_rows <= 1) {throw std::invalid_argument("arboria::DecisionTree::fit -> invalid fitted DataSet");}
    if (context){
        fit_(data, root_node, idx, 0, params, context);
    }
    else {
        fit_(data, root_node, idx, 0, params);
    }
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


    if (node.is_leaf) {return node.leaf_value;}

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

//fit the DecisionTree with SplitContext :
void DecisionTree::fit_(const DataSet& data, 
                        Node& node, 
                        std::span<int> idx, 
                        int depth, 
                        const SplitParam& params, 
                        std::optional<std::reference_wrapper<SplitContext>> context){

    //lambda function to stop iteration :
    auto end_branch= [&](){
        node.is_leaf = true;

        if (std::holds_alternative<Classification>(params.type)){
            std::pair<int,int>count = helpers::count_classes(idx, data.y());
            int pos_count = count.first;
            int neg_count = count.second;

            (pos_count >= neg_count) ? node.leaf_value= 1 : node.leaf_value=0 ; // ">=" : in case of tie break, node predicted class = 1
            return;
        }

        if (std::holds_alternative<Regression>(params.type)){

            float mean = helpers::calculate_mean(idx,data.y());
            node.leaf_value = mean;
            return;
        }
    };

    std::pair<int,int>count = helpers::count_classes(idx, data.y());
    int pos_count = count.first;
    int neg_count = count.second;

    // --------- Logical stop cases
    //case idx refers to less than 1 sample
    if (idx.size() <= 1) {end_branch(); return;}
    //case if the current node is pure
    if ((pos_count == 0) || (neg_count ==0)) {end_branch();return;}
    
    //--------- Hyper Parameters stop cases
    //case max depth is reached:
    if (max_depth.has_value()){
        if (depth == max_depth) {end_branch(); return;}
    }
    if (min_sample_split.has_value()){
        if (idx.size() <= min_sample_split) {end_branch(); return;}
    }

    //Compute the split :
    SplitResult split;
    if (context) {
        SplitContext& ctx = context->get();
        split = splitter.best_split(idx, data, params, ctx);
    }
    else { split = splitter.best_split(idx, data, params);};

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
        
        fit_(data, *node.left_child, left_idx, depth+1, params, context);
        fit_(data, *node.right_child, right_idx, depth+1, params, context);
        

    }    
    
}


}