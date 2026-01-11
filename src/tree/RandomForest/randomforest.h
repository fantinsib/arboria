/*

                    Random Forest header

*/
#pragma once

#include "dataset/dataset.h"
#include "split_strategy/types/split_param.h"
#include "tree/DecisionTree/DecisionTree.h"
#include <optional>
#include <vector>
#include <span>

namespace arboria {

class RandomForest{

    public:
    RandomForest(int n_estimators, int mtry, int max_depth, std::optional<uint32_t> seed = std::nullopt);
    int mtry;
    int n_estimators; 
    int max_depth; 
    void fit(const DataSet& data, const SplitParam& param);
    bool is_fitted() const {return fitted;}
    std::vector<int> predict(std::span<const float> sample);
    std::uint32_t seed() {return seed_;}
    void set_seed(uint32_t user_seed);

    private:
    void fit_(const DataSet& data, const SplitParam& param, SplitContext &context);
    std::vector<int> predict_(std::span<const float> sample);
    bool fitted = false;
    int num_features;
    uint32_t seed_;
    std::vector<std::unique_ptr<DecisionTree>> trees;

};



}

