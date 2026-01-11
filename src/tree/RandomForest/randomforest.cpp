/*

                    Random Forest implementation

*/

#include "randomforest.h"
#include "dataset/dataset.h"
#include "split_strategy/sampling/sampling.h"
#include "split_strategy/types/split_context.h"
#include "split_strategy/types/split_param.h"
#include "tree/DecisionTree/DecisionTree.h"

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>

using arboria::sampling::bootstrap;

namespace arboria{

RandomForest::RandomForest(int n_estimators_, int mtry_, int max_depth_, std::optional<uint32_t> user_seed):

    n_estimators(n_estimators_),
    mtry(mtry_),
    max_depth(max_depth_)
{

    if (mtry <= 0) throw std::invalid_argument("arboria::tree::RandomForest : mtry argument must be greater than or equal 0");
    if (n_estimators <= 0) throw std::invalid_argument("arboria::tree::RandomForest : n_estimators argument must be greater than or equal 0");
    if (max_depth_ <= 0) throw std::invalid_argument("arboria::tree::RandomForest : max_depth argument must be greater than or equal 0");

    trees.reserve(static_cast<size_t>(n_estimators_));
    if (!user_seed){
        std::random_device rd;
        seed_ = rd();
    }
    else{seed_ = user_seed.value();}
}

void RandomForest::fit(const DataSet &data, const SplitParam& params){

    //TODO for Python API : param builder function 
    int n_rows = data.n_rows();
    SplitContext context(seed_);

    fit_(data, params, context);

    fitted = true;
    num_features = data.n_cols();
}

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
            sum_votes += t->predict_one(sample);
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
/*
--------------------------------------------------------------------------------------
PRIVATE METHODS 
--------------------------------------------------------------------------------------
*/

void RandomForest::fit_(const DataSet& data, const SplitParam &param, SplitContext &context){

    const size_t n_rows = static_cast<size_t>(data.n_rows());
    const size_t n_cols = static_cast<size_t>(data.n_cols());
    if (n_cols < mtry) throw std::invalid_argument("arboria::tree::RandomForest::fit : mtry parameter can't be larger than the number of features in the dataset");

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
        for (size_t row_idx = 0; row_idx <boostrapped_indices.size(); row_idx++ ){
            passed_idx[row_idx] = static_cast<int>(boostrapped_indices[row_idx]);
        }
        // then fit tree with param.f_selection = RandomK & 
        // add to the RF list 
        trees.push_back(std::make_unique<DecisionTree>(max_depth));
        trees.back()->fit(data, passed_idx, param, context);
    }


}










}