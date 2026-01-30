/*

                    SPLITTER CLASS 

*/

#include "splitter.h"
#include "dataset/dataset.h"
#include "feature_selection/randomK/randomK.h"
#include "split_criterion/sse.h"
#include "split_strategy/types/split_context.h"
#include "split_strategy/types/split_param.h"
#include "split_strategy/types/split_stats.h"

#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <variant>
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

SplitResult Splitter::best_split(std::span<const int> idx, const DataSet &data, const SplitParam &params, SplitContext& context){
    
    if (std::holds_alternative<Regression>(params.type)) {
        return best_split_regression(idx, data, params, context);
    }

    if (std::holds_alternative<Classification>(params.type)) {
        return best_split_classification(idx, data, params, context);
    }
};


SplitResult Splitter::best_split_classification(std::span<const int> idx, const DataSet& data, const SplitParam& params, SplitContext& context){

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
            int mtry = *rk->mtry;
            if (mtry <= 0) {throw std::logic_error("arboria::split_strategy_Splitter::best_split : number of sampled features for RandomK must be positive");}
            if (mtry > num_features) {throw std::logic_error("arboria::split_strategy_Splitter::best_split : mtry parameter can't be larger than number of features");}
            std::vector<int> all_features (num_features);
            std::iota(all_features.begin(), all_features.end(), 0);
            features = randomK(all_features, mtry, context.rng);
            
            }

        else if constexpr ((std::is_same_v<T, Undefined>)) {
                throw std::invalid_argument("aboria::split_strategy::Splitter::best_split : feature selection parameter is Undefined");
            }
            
            
        else throw std::logic_error("aboria::split_strategy::Splitter::best_split : feature selection parameter is not recognized");
        
    }, params.f_selection);

// ------------------------------------ loop over the features -----------------

    for (auto col : features){

// ------------------------------------ threshold computation -----------------
        std::vector<int> sorted_idx(idx.begin(), idx.end());
        std::vector<float> thresholds;
        std::visit([&](const auto& t_compute) { //returns the threshold vector
            using T=std::decay_t<decltype(t_compute)>;

            if constexpr ((std::is_same_v<T, CART>)) {


                std::sort(sorted_idx.begin(), sorted_idx.end(),
                [&](int i, int j) {
                    return data.iloc_x(i, col) < data.iloc_x(j, col);
                });
                thresholds = cart_threshold(sorted_idx, col, data);
            }

            else if constexpr ((std::is_same_v<T, Random>)){
                throw std::logic_error("Not yet implemented");
            // to implement
            }
            
            else if constexpr (std::is_same_v<T, Quantile>) {
                throw std::logic_error("Not yet implemented");
            // to implement
            }

            else if constexpr ((std::is_same_v<T, Undefined>)) {
                throw std::invalid_argument("aboria::split_strategy::Splitter::best_split : threshold computation parameter is Undefined");
            }
            
            else throw std::logic_error("aboria::split_strategy::Splitter::best_split : threshold computation parameter is not recognized");
        }, params.t_comp);
      


        //--------------------------------
        int l_neg_count = 0;
        int l_pos_count = 0;
        int r_neg_count = 0; 
        int r_pos_count = 0;

        for (int i : idx) {
            int y = data.iloc_y(i);
            if (y == 1) r_pos_count++;
            else        r_neg_count++;
        }



            size_t p = 0;

            for (float t : thresholds) {

                while (p < sorted_idx.size()) {
                    int i = sorted_idx[p];
                    float x = data.iloc_x(i, col);
                    if (!(x < t)) break; 

                    int y = data.iloc_y(i);
                    if (y == 1) { l_pos_count++; r_pos_count--; }
                    else        { l_neg_count++; r_neg_count--; }

                    ++p;
                }
            

            if ((l_pos_count + l_neg_count) == 0 || (r_pos_count + r_neg_count) == 0) {continue;} //ignoring if we have an empty leaf
                
            //--------------------------------
            ClfStats split_stats { .l_pos = l_pos_count, .l_neg = l_neg_count, .r_pos = r_pos_count, .r_neg = r_neg_count};            
            float score = score_function(params, split_stats);


            if (score < best_score){ 
                
                best_score = score;

                best_split.split_feature = col;
                best_split.split_threshold = t;
                best_split.score = score;
            }

            if (best_split.score == 0) return best_split;

        }
    
    }

    return best_split;

};
//--------------------------------------------------------------------------------------------------------------------------------------------



SplitResult Splitter::best_split_regression(std::span<const int> idx, const DataSet& data, const SplitParam& params, SplitContext& context){

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
            int mtry = *rk->mtry;
            if (mtry <= 0) {throw std::logic_error("arboria::split_strategy_Splitter::best_split : number of sampled features for RandomK must be positive");}
            if (mtry > num_features) {throw std::logic_error("arboria::split_strategy_Splitter::best_split : mtry parameter can't be larger than number of features");}
            std::vector<int> all_features (num_features);
            std::iota(all_features.begin(), all_features.end(), 0);
            features = randomK(all_features, mtry, context.rng);
            
            }

        else if constexpr ((std::is_same_v<T, Undefined>)) {
                throw std::invalid_argument("aboria::split_strategy::Splitter::best_split : feature selection parameter is Undefined");
            }
            
            
        else throw std::logic_error("aboria::split_strategy::Splitter::best_split : feature selection parameter is not recognized");
        
    }, params.f_selection);

// ------------------------------------ loop over the features -----------------

    for (auto col : features){

// ------------------------------------ threshold computation -----------------
        std::vector<int> sorted_idx(idx.begin(), idx.end());
        std::vector<float> thresholds;
        std::visit([&](const auto& t_compute) { //returns the threshold vector
            using T=std::decay_t<decltype(t_compute)>;

            if constexpr ((std::is_same_v<T, CART>)) {


                std::sort(sorted_idx.begin(), sorted_idx.end(),
                [&](int i, int j) {
                    return data.iloc_x(i, col) < data.iloc_x(j, col);
                });
                thresholds = cart_threshold(sorted_idx, col, data);
            }

            else if constexpr ((std::is_same_v<T, Random>)){
                throw std::logic_error("Not yet implemented");
            // to implement
            }
            
            else if constexpr (std::is_same_v<T, Quantile>) {
                throw std::logic_error("Not yet implemented");
            // to implement
            }

            else if constexpr ((std::is_same_v<T, Undefined>)) {
                throw std::invalid_argument("aboria::split_strategy::Splitter::best_split : threshold computation parameter is Undefined");
            }
            
            else throw std::logic_error("aboria::split_strategy::Splitter::best_split : threshold computation parameter is not recognized");
        }, params.t_comp);
      


        //--------------------------------

        int nL = 0;
        int nR = 0;
        float y_ssL = 0;
        float y_ssR = 0;
        float y_sL = 0;
        float y_sR = 0;

        for (auto i : idx) {

            float y = data.iloc_y(i);
            nR++;
            y_sR += y;
            y_ssR += y*y;
        }


        size_t p = 0;
        for (auto t : thresholds){

            while (p < sorted_idx.size()) {
                
                int i = sorted_idx[p];
                float x = data.iloc_x(i, col);
                if (!(x < t)) break; 
                float y = data.iloc_y(i);
                nL++;
                nR--;
                y_sR -= y;
                y_sL += y;
                y_ssR -= y*y;
                y_ssL += y*y;

                ++p;

                }

                if (nL == 0 || nR == 0) continue; //ignore if we have an empty leaf
            
         //--------------------------------
            RegStats split_stats{nL, nR, y_ssL, y_ssR, y_sL, y_sR};
            float score = score_function(params, split_stats);


            if (score < best_score){ 
                
                best_score = score;

                best_split.split_feature = col;
                best_split.split_threshold = t;
                best_split.score = score;
            }

            if (best_split.score == 0) return best_split;

        }
    
    }

    return best_split;

};

//------------------------------------------------------------------------------------------------------------------------------------------------


float Splitter::score_function(const SplitParam& params, const ClfStats& stats) {

// ------------------------------------ score function -----------------

    //score_function depending on the param passed (Gini by default)
    return std::visit([&](const auto& crit_function) {
        using T = std::decay_t<decltype(crit_function)>;

        if constexpr ((std::is_same_v<T, Gini>)) {
            return split::weighted_gini(stats.l_pos, stats.l_neg, stats.r_pos, stats.r_neg);}
         
        else if constexpr (std::is_same_v<T, Entropy>) {
            return split::weighted_entropy(stats.l_pos, stats.l_neg, stats.r_pos, stats.r_neg);}
    
        
        else if constexpr ((std::is_same_v<T, Undefined>)) {
                throw std::invalid_argument("aboria::split_strategy::Splitter::score_function : scoring function parameter is Undefined");
                return 1.f;
            }
        
        else throw std::logic_error("arboria::split_strategy::Splitter::score_function : no classification scoring criterion was passed.");
        return 1.f;
        }, params.criterion);

    

};

float Splitter::score_function(const SplitParam& params, const RegStats& stats) {

// ------------------------------------ score function -----------------

    //score_function depending on the param passed 
    return std::visit([&](const auto& crit_function) {
        using T = std::decay_t<decltype(crit_function)>;

        if constexpr (std::is_same_v<T, SSE>) {
            return split::weighted_sse( stats.nL,  stats.nR,  stats.y_sL,  stats.y_sR,  stats.y_ssL, stats.y_ssR);
        }
    
        else if constexpr ((std::is_same_v<T, Undefined>)) {
                throw std::invalid_argument("aboria::split_strategy::Splitter::score_function : scoring function parameter is Undefined");
                return 1.f;
            }
        
        else throw std::logic_error("arboria::split_strategy::Splitter::score_function : no regression scoring criterion was passed.");
        return 1.f;
        }, params.criterion);

    

};


}
}

