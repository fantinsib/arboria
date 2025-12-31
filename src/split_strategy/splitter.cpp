/*

                    SPLITTER CLASS 
                    to do -> add tests

*/



#include "splitter.h"

#include <numeric>
#include <stdexcept>
#include <vector>
#include <span>


namespace arboria {
namespace split_strategy{

SplitResult Splitter::best_split(std::span<const int> idx, const DataSet& data, const SplitParam& params){

    float best_score = std::numeric_limits<float>::infinity();
    SplitResult best_split;
    const int num_features = data.n_cols();
    std::vector<int> col_vector;

// ------------------------------------ feature selection -----------------

// -> filling col_vector with col index

    std::vector<int> features; 
    switch (params.f_selection) {
        case FeatureSelection::All :
            features.resize(num_features);
            std::iota(features.begin(), features.end(), 0);
            break;
        
        case FeatureSelection::RandomK : {
            // TO IMPLEMENT 
            };
        break;
    }


// ------------------------------------ loop over the features -----------------

    for (auto col : features){

// ------------------------------------ threshold computation -----------------

        std::vector<float> thresholds;
        switch (params.t_comp) {
            case ThresholdComputation::CART :
                thresholds = cart_threshold(idx, col, data);
                break;

            case ThresholdComputation::Random :
                break;
            
            case ThresholdComputation::Quantile :
                break;
        }

        for (float t : thresholds) { //iteration through the thresholds 

                int l_neg_count = 0;
                int l_pos_count = 0;
                int r_neg_count = 0;
                int r_pos_count = 0;

                for (int i : idx){

                    const float x = data.iloc_x(i, col);
                    const int y = data.iloc_y(i); // supposé 0/1

                    if (x < t) { (y == 1 ? l_pos_count : l_neg_count)++; }
                    else { (y == 1 ? r_pos_count : r_neg_count)++; }
            }

            if ((l_pos_count + l_neg_count) == 0 || (r_pos_count + r_neg_count) == 0) {continue;} //ignoring if we have an empty leaf
                
            
            SplitStats split_stats{l_pos_count, l_neg_count, r_pos_count, r_neg_count};            
            float score = score_function(params, split_stats);

            if (score < best_score){
                
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
    switch (params.criterion) {
        case Criterion::Gini : //default criterion
            return split::weighted_gini(stats.l_pos, stats.l_neg, stats.r_pos, stats.r_neg);
         
        case Criterion::Entropy :
            return split::weighted_entropy(stats.l_pos, stats.l_neg, stats.r_pos, stats.r_neg);
    
            
        }

    throw std::logic_error("arboria::split_strategy::Splitter::score_function : no scoring criterion was passed.");

};


}
}

