/*

                    Random Forest implementation

*/

#include "randomforest.h"
#include "dataset/dataset.h"
#include "helpers/helpers.h"
#include "split_strategy/sampling/sampling.h"
#include "split_strategy/types/split_context.h"
#include "split_strategy/types/split_hyper.h"
#include "split_strategy/types/split_param.h"
#include "split_strategy/types/split_hyper.h"
#include "tree/DecisionTree/DecisionTree.h"
#include "tree/TreeModel.h"

#include <atomic>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>

using arboria::sampling::bootstrap;
using arboria::ForestTree;
using arboria::helpers::derive_seed;

namespace arboria{

RandomForest::RandomForest(HyperParam hyperParam, TreeType type, std::optional<std::uint32_t> user_seed)
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

            throw std::invalid_argument("arboria::tree::RandomForest : mtry argument must be greater than or equal 0");
    }
        mtry = *hyperParam.mtry;
    }
    else throw std::logic_error("RandomForest Init : No value received for mtry.");

    if (hyperParam.max_depth.has_value()) {
        if (*hyperParam.max_depth <= 0) throw std::invalid_argument("arboria::tree::RandomForest : max_depth argument must be greater than 0");
        max_depth = *hyperParam.max_depth;
    }

    if (hyperParam.max_samples.has_value()){
        if (*hyperParam.max_samples <= 0) throw std::invalid_argument("arboria::tree::RandomForest : max_samples argument must be greater than 0");
        max_samples = *hyperParam.max_samples;
    }

    if (hyperParam.min_sample_split.has_value()){
        if (*hyperParam.min_sample_split <= 0) throw std::invalid_argument("arboria::tree::RandomForest : min_sample_split argument must be greater than 0");
        min_sample_split = *hyperParam.min_sample_split;
    }

    if (hyperParam.n_jobs.has_value()){
        const unsigned hw_u = std::thread::hardware_concurrency();

        const size_t hw = (hw_u == 0) ? std::numeric_limits<size_t>::max() : static_cast<size_t>(hw_u);

        if (*hyperParam.n_jobs < -1 || *hyperParam.n_jobs == 0) 
        {
            throw std::invalid_argument("arboria::tree::RandomForest : n_jobs argument must be a positive int or equals to -1");
        }

        if (*hyperParam.n_jobs == -1 ) {
            n_jobs = std::min(static_cast<size_t>(n_estimators), hw);
        }
        else{
            n_jobs = *hyperParam.n_jobs;
        }
    }
    else {n_jobs = 1;}

    trees.reserve(static_cast<size_t>(n_estimators));
    if (!user_seed){
        std::random_device rd;
        seed_ = static_cast<std::uint32_t>(rd());
    }
    else{seed_ = user_seed.value();}

    if (std::holds_alternative<Classification>(type) || std::holds_alternative<Regression>(type)){
        type_ = type;
    }
    else throw std::invalid_argument("DecisionTree constructor : invalid type");

}

void RandomForest::fit(const DataSet &data, const SplitParam& params){

    const size_t n_rows = static_cast<size_t>(data.n_rows());
    const size_t n_cols = static_cast<size_t>(data.n_cols());
    const auto* rk = std::get_if<RandomK>(&params.f_selection);
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
        throw std::invalid_argument("arboria::tree::RandomForest::fit_ : mtry parameter can't be larger than the number of features in the dataset");
    }

    num_features = data.n_cols();
    trees.clear();
    trees.resize(static_cast<size_t>(n_estimators));


    std::atomic<size_t> next{0};

    auto worker = [&](){
        for (;;) {
            
            size_t i = next.fetch_add(1);
            if (i >= n_estimators) break;
            SplitContext context(derive_seed(seed_.value(), i));
            fit_(i, data, params, context);
        }
    };

    std::vector<std::thread>pool;
    pool.reserve(n_jobs);

    for (size_t i = 0 ; i < n_jobs; i++){
        pool.emplace_back(worker);
    }

    for (auto& t : pool){
        t.join();
    }

    fitted = true;
    num_features = data.n_cols();
}

std::vector<float> RandomForest::predict_proba(std::span<const float> samples) const{
    if (!fitted || num_features == 0) throw std::invalid_argument("arboria::RandomForest::predict_proba -> RandomForest has not been fitted");
    if (trees.size() < 1) throw std::logic_error("arboria::RandomForest::predict_proba -> no trees were found in the forest");
    size_t nf = static_cast<size_t>(num_features);
    if (samples.size() % nf != 0) throw std::invalid_argument("arboria::RandomForest::predict_proba -> passed samples do not have the correct dimension");

    size_t num_samples = samples.size()/nf;
    std::vector<float> preds(num_samples);
    
    std::atomic<size_t> next{0};

    auto worker = [&]() {
        for (;;){
            size_t i = next.fetch_add(1);
            if (i>= num_samples) break; 
            float sum_votes =0; 
            auto sample = samples.subspan(i*nf, nf);
            for (const auto& t : trees){
                sum_votes += t.tree->predict_one(sample);
            }
            preds[i] = sum_votes/static_cast<float>(n_estimators);
        }

        };

        std::vector<std::thread> pool;
        pool.reserve(n_jobs);

        for (size_t i = 0; i < n_jobs; i++){
            pool.emplace_back(worker);

        }

        for (auto& t : pool){
            t.join() ;

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

void RandomForest::fit_(size_t i, const DataSet& data, const SplitParam &param, SplitContext &context){

        //Bootstrapping of dataset rows :
        const size_t n_rows = static_cast<size_t>(data.n_rows());
        size_t bootstrap_size = max_samples.has_value() ? static_cast<size_t>(
        static_cast<double>(max_samples.value()) * static_cast<double>(n_rows)) :  n_rows;

        std::vector<size_t> boostrapped_indices = bootstrap(static_cast<size_t>(n_rows), bootstrap_size, context.rng);
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
        HyperParam h_param{.max_depth = max_depth, .min_sample_split = min_sample_split};
        
        forest_tree.tree = std::make_unique<DecisionTree>(h_param, param.type);
        forest_tree.in_bag = std::move(seen_idx);

        trees[i]=(std::move(forest_tree));
        trees[i].tree->fit(data, passed_idx, param, context);

}










}
