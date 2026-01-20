/*

                    SPLITTER CLASS 

*/

#include "splitter.h"
#include "feature_selection/randomK/randomK.h"
#include "split_strategy/types/split_context.h"
#include "split_strategy/types/split_param.h"

#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <span>

using arboria::feature_selection::randomK;

namespace arboria {
namespace split_strategy{

Splitter::Splitter() {};

// POSSIBLE IMPROVEMENT : 
// add overload/modify best_split to make the split based on a set of row indices and col indices 
//--> would allow to remove the feature selection section from inside best_split and handle it on a case by case basis

SplitResult Splitter::best_split(std::span<const int> idx, const DataSet &data, const SplitParam &params){
    
    if (std::holds_alternative<RandomK>(params.f_selection)) throw std::invalid_argument("aboria::split_strategy::Splitter::best_split : incompatible parameters and context for split - RNG must be passed if RandomK used");
    SplitContext context(0u);
    return best_split(idx, data, params, context);
};

SplitResult Splitter::best_split(std::span<const int> idx, const DataSet& data, const SplitParam& params, SplitContext& context){

// --------- Initialization & validity conditions ----------

    if (data.is_empty()) {throw std::invalid_argument("aboria::split_strategy::Splitter::best_split : dataset is empty.");}
    if (idx.empty()) {throw std::invalid_argument("aboria::split_strategy::Splitter::best_split : row index span is empty.");}

    float best_score = std::numeric_limits<float>::infinity();
    SplitResult best_split;
    const int num_features = data.n_cols();
    const int num_rows = idx.size();
    if (num_features <= 0) {throw std::invalid_argument("aboria::split_strategy::Splitter::best_split : invalid number of features");}
    if (num_rows <= 1) {
        return best_split;} //returning best_split here will return a SplitResult with 
                            // default attributes

// ------------------------------------ feature selection -----------------

// -> filling col_vector with col index

    std::vector<int> features; 
    std::visit([&](const auto& feature_selec) {
        
        using T = std::decay_t<decltype(feature_selec)>;

        if constexpr ((std::is_same_v<T, AllFeatures>)){
            features.resize(num_features);
            std::iota(features.begin(), features.end(), 0);
            
            }
        
        else if constexpr ((std::is_same_v<T, RandomK>)) {
            auto* rk = std::get_if<RandomK>(&params.f_selection);
            int mtry = rk->mtry;
            if (mtry <= 0) {throw std::logic_error("arboria::split_strategy_Splitter::best_split : number of sampled features for RandomK must be positive");}
            if (mtry > num_features) {throw std::logic_error("arboria::split_strategy_Splitter::best_split : mtry parameter can't be larger than number of features");}
            std::vector<int> all_features (num_features);
            std::iota(all_features.begin(), all_features.end(), 0);
            features = randomK(all_features, mtry, context.rng);
            
            }
            
        else throw std::logic_error("aboria::split_strategy::Splitter::best_split : feature selection parameter is not recognized");
        
    }, params.f_selection);

// ------------------------------------ loop over the features -----------------

    for (auto col : features){

// ------------------------------------ threshold computation -----------------

        std::vector<float> thresholds;
        std::visit([&](const auto& t_compute) {
            using T=std::decay_t<decltype(t_compute)>;

            if constexpr ((std::is_same_v<T, CART>)) {
                thresholds = cart_threshold(idx, col, data);
            }

            else if constexpr ((std::is_same_v<T, Random>)){
                throw std::logic_error("Not yet implemented");
            // to implement
            }
            
            else if constexpr (std::is_same_v<T, Quantile>) {
                throw std::logic_error("Not yet implemented");
            // to implement
            }
            
            else throw std::logic_error("aboria::split_strategy::Splitter::best_split : threshold computation parameter is not recognized");
        }, params.t_comp);

        for (float t : thresholds) { //iteration through the thresholds 
            // Current version -> naive count for each leaf. 
            // Optimisation to implement later : start with all the 
            // samples in the right node, iterate through the ordered samples by 
            // feature and move each sample one by one from right to left
            
                int l_neg_count = 0;
                int l_pos_count = 0;
                int r_neg_count = 0;
                int r_pos_count = 0;

                for (int i : idx){

                    const float x = data.iloc_x(i, col);
                    const int y = data.iloc_y(i); 
                    
                    if (x < t) { (y == 1 ? l_pos_count : l_neg_count)++; }
                    else { (y == 1 ? r_pos_count : r_neg_count)++; } 
            }

            if ((l_pos_count + l_neg_count) == 0 || (r_pos_count + r_neg_count) == 0) {continue;} //ignoring if we have an empty leaf
                
            
            SplitStats split_stats{l_pos_count, l_neg_count, r_pos_count, r_neg_count};            
            float score = score_function(params, split_stats);

            if (score < best_score){ //possible improvement : stop the loop if perfect split is found ?
                
                best_score = score;

                best_split.split_feature = col;
                best_split.split_threshold = t;
                best_split.score = score;
            }

        }
    
    }

    return best_split;

};

float Splitter::score_function(const SplitParam& params, const SplitStats& stats) {

// ------------------------------------ score function -----------------

    //score_function depending on the param passed (Gini by default)
    return std::visit([&](const auto& crit_function) {
        using T = std::decay_t<decltype(crit_function)>;


        if constexpr ((std::is_same_v<T, Gini>)) {
            return split::weighted_gini(stats.l_pos, stats.l_neg, stats.r_pos, stats.r_neg);}
         
        else if constexpr (std::is_same_v<T, Entropy>) {
            return split::weighted_entropy(stats.l_pos, stats.l_neg, stats.r_pos, stats.r_neg);}
    
        else throw std::logic_error("arboria::split_strategy::Splitter::score_function : no scoring criterion was passed.");
        }, params.criterion);

    

};


}
}

