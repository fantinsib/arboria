/*

                    Random Forest implementation

*/

#include "randomforest.h"
#include "dataset/dataset.h"
#include "split_strategy/sampling/sampling.h"
#include "split_strategy/types/split_context.h"
#include "split_strategy/types/split_param.h"
#include "tree/DecisionTree/DecisionTree.h"

#include <cstdint>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>

using arboria::sampling::bootstrap;

namespace arboria{

RandomForest::RandomForest(int n_estimators_, int mtry_, int max_depth_, std::optional<uint32_t> user_seed):

    n_estimators(n_estimators_),
    mtry(mtry_),
    max_depth(max_depth_)
{
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

void RandomForest::fit_(const DataSet& data, const SplitParam &param, SplitContext &context){

    const size_t n_rows = static_cast<size_t>(data.n_rows());
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
        for (size_t col_idx = 0; col_idx <boostrapped_indices.size(); col_idx++ ){
            passed_idx[col_idx] = static_cast<int>(boostrapped_indices[col_idx]);
        }
        // then fit tree with param.f_selection = RandomK & 
        // add to the RF list 
        trees.push_back(std::make_unique<DecisionTree>(max_depth));
        trees.back()->fit(data, passed_idx, param, context);
    }


}









}