/*

                    Random Forest implementation

*/

#include "randomforest.h"
#include "dataset/dataset.h"
#include "split_strategy/sampling/sampling.h"
#include "split_strategy/types/split_context.h"
#include "split_strategy/types/split_hyper.h"
#include "split_strategy/types/split_param.h"
#include "split_strategy/types/split_hyper.h"
#include "tree/DecisionTree/DecisionTree.h"


#include <cstdint>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>

using arboria::sampling::bootstrap;
using arboria::ForestTree;

namespace arboria{

RandomForest::RandomForest(HyperParam hyperParam, std::optional<std::uint32_t> user_seed)
{

    if (hyperParam.n_estimators.has_value()) {
        if (*hyperParam.n_estimators <= 0) throw std::invalid_argument("arboria::tree::RandomForest : n_estimators argument must be greater than or equal 0");
        n_estimators = *hyperParam.n_estimators;
    }
    else n_estimators = 70;

    if (hyperParam.mtry.has_value()) {
            //mtry == -99 : auto (sqrt)
            //mtry == -98 : auto (log)
    if (*hyperParam.mtry <= 0 && *hyperParam.mtry != -99 && *hyperParam.mtry != -98) {
            std::cout<< "In RF constructor : " << *hyperParam.mtry<< std::endl; 
            throw std::invalid_argument("arboria::tree::RandomForest : mtry argument must be greater than or equal 0");
    }
        mtry = *hyperParam.mtry;
    }
    else throw std::logic_error("RandomForest Init : No value received for mtry.");

    if (hyperParam.max_depth.has_value()) {
        if (*hyperParam.max_depth <= 0) throw std::invalid_argument("arboria::tree::RandomForest : max_depth argument must be greater than or equal 0");
        max_depth = *hyperParam.max_depth;
    }

    trees.reserve(static_cast<size_t>(n_estimators));
    if (!user_seed){
        std::random_device rd;
        seed_ = static_cast<std::uint32_t>(rd());
    }
    else{seed_ = user_seed.value();}
}

void RandomForest::fit(const DataSet &data, const SplitParam& params){

    SplitContext context(seed_.value());

    fit_(data, params, context);

    fitted = true;
    num_features = data.n_cols();
}
/*
void RandomForest::fit(const DataSet &data){

    SplitParam params; 
    params.f_selection = FeatureSelection::RandomK;
    params.mtry = mtry;
    int n_rows = data.n_rows();
    SplitContext context(seed_);

    fit_(data, params, context);

    fitted = true;
    num_features = data.n_cols();
}
*/
std::vector<float> RandomForest::predict_proba(std::span<const float> samples) const{
    if (!fitted || num_features == 0) throw std::invalid_argument("arboria::RandomForest::predict_proba -> RandomForest has not been fitted");
    if (trees.size() < 1) throw std::logic_error("arboria::RandomForest::predict_proba -> no trees were found in the forest");
    size_t nf = static_cast<size_t>(num_features);
    if (samples.size() % nf != 0) throw std::invalid_argument("arboria::RandomForest::predict_proba -> passed samples do not have the correct dimension");

    size_t num_samples = samples.size()/nf;
    std::vector<float> preds(num_samples);
    

    for (size_t s = 0; s<num_samples; s++){
        float sum_votes =0; 
        auto sample = samples.subspan(s*nf, nf);
        for (const auto& t : trees){
            sum_votes += t.tree->predict_one(sample);
        }
        preds[s] = sum_votes/static_cast<float>(n_estimators);
        }
    
    return preds;

}


std::vector<int> RandomForest::predict(std::span<const float> sample) const {

    std::vector<float> prob_pred = predict_proba(sample);
    std::vector<int> class_pred(prob_pred.size());

    std::transform(prob_pred.begin(), prob_pred.end(), class_pred.begin(),
                    [](float x){return (x >= 0.5) ? 1 : 0;});
    return class_pred;
}

float RandomForest::out_of_bag(const DataSet &data) const {

    if (!fitted) throw std::invalid_argument("arboria::RandomForest::out_of_bag : RandomForest was never fitted");
    if (data.is_empty()) throw std::invalid_argument("arboria::RandomForest::out_of_bag : DataSet is empty");

    const size_t n_rows = data.n_rows();
    const size_t n_cols = data.n_cols();

    if (n_rows == 0 || n_cols ==0) throw std::logic_error("arboria::RandomForest::out_of_bag : DataSet has no rows or cols");
    if (n_cols != static_cast<size_t>(num_features)) throw std::invalid_argument("arboria::RandomForest::out_of_bag : DataSet passed does not have the same dimensions as seen during training");

    std::span<const float> samples(data.X());

    int correct_pred = 0;
    int wrong_pred = 0;

    for (size_t row = 0; row < n_rows; row++){
        
        std::span<const float> s = samples.subspan(row*n_cols, n_cols);
        int num_pred = 0;
        int sum_pred = 0;
        for (const ForestTree& t : trees){

            if (!t.in_bag[row]){
                int pred = t.tree->predict_one(s);
                sum_pred = sum_pred + pred;
                num_pred++;   
            }
        }
        if (num_pred !=0) {
            int row_vote = (static_cast<float>(sum_pred)/static_cast<float>(num_pred) >= 0.5) ? 1 : 0;
            (row_vote == data.iloc_y(row)) ? correct_pred++ : wrong_pred++; 
        }

    }
    if (correct_pred+wrong_pred == 0) throw std::logic_error("arboria::RandomForest::out_of_bag : no OOB samples.");
    return static_cast<float>(correct_pred)/(static_cast<float>(correct_pred)+static_cast<float>(wrong_pred));
    

}
/*
--------------------------------------------------------------------------------------
PRIVATE METHODS 
--------------------------------------------------------------------------------------
*/

void RandomForest::fit_(const DataSet& data, const SplitParam &param, SplitContext &context){

    const size_t n_rows = static_cast<size_t>(data.n_rows());
    const size_t n_cols = static_cast<size_t>(data.n_cols());
    const auto* rk = std::get_if<RandomK>(&param.f_selection);
    if (!rk) {
        throw std::logic_error("arboria::tree::RandomForest::fit_ : f_selection is not RandomK");
    }
    if (!rk->mtry) {
        throw std::invalid_argument("arboria::tree::RandomForest::fit_ : RandomK::mtry is not defined");
    }

    const int mtry = *rk->mtry;  

    if (mtry <= 0) {
        throw std::invalid_argument("arboria::tree::RandomForest::fit_ : mtry parameter must be > 0");
    }

    if (n_cols < mtry) {
        std::cout << "Received mtry : " << mtry << std::endl;
        throw std::invalid_argument("arboria::tree::RandomForest::fit_ : mtry parameter can't be larger than the number of features in the dataset");
    }
    //to del
    if (mtry <= 0) throw std::invalid_argument("arboria::tree::RandomForest::fit_ : mtry parameter can't be negative");

    num_features = data.n_cols();
    trees.clear();
    trees.reserve(static_cast<size_t>(n_estimators));
    for (int i = 0; i < n_estimators; i++){ //per tree
        //Bootstrapping of dataset rows :
        std::vector<size_t> boostrapped_indices = bootstrap(n_rows, static_cast<size_t>(n_rows), context.rng);
        //TODO : temporary conversion size_t -> int for
        // passing from bootstrap to DecisionTree.fit(); 
        // to delete when all ref to indices will be in size_t
        std::vector<int> passed_idx(boostrapped_indices.size());
        std::vector<bool> seen_idx(n_rows, false);
        for (size_t row_idx = 0; row_idx <boostrapped_indices.size(); row_idx++ ){
            passed_idx[row_idx] = static_cast<int>(boostrapped_indices[row_idx]);
            seen_idx[boostrapped_indices[row_idx]] = true;
        }
        // then fit tree with param.f_selection = RandomK & 
        // add to the RF list 
        ForestTree forest_tree;
        HyperParam h_param{.max_depth = max_depth};
        forest_tree.tree = std::make_unique<DecisionTree>(h_param);
        forest_tree.in_bag = std::move(seen_idx);

        trees.push_back(std::move(forest_tree));
        trees.back().tree->fit(data, passed_idx, param, context);

    }


}










}