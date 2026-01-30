/*

                    Random Forest header

*/
#pragma once

#include "dataset/dataset.h"
#include "split_strategy/types/split_param.h"
#include "tree/DecisionTree/DecisionTree.h"
#include "tree/TreeModel.h"
#include "split_strategy/types/split_hyper.h"



#include <optional>
#include <vector>
#include <span>

namespace arboria::test { struct RandomForestAccess; }  

namespace arboria {


/**
 * @brief Struct to save each tree of the forest with extra informations
 *
 * @param tree A RandomForest DecisionTree 
 * @param in_bag A boolean vector with size equal to number of 
 * samples in training dataset with true if sample was seen 
 * during training
 *
 */
struct ForestTree {
    //Vector containing unique pointers to the fitted DecisionTree of the forest 
    std::unique_ptr<DecisionTree> tree;    
    //Saves the indices of the samples that were seen by the tree during training 
    std::vector<bool> in_bag; 

};

class RandomForest{

    public:
    //Constructor for the RandomForest
    RandomForest(HyperParam hyperParam, TreeType type, std::optional<uint32_t> seed = std::nullopt);
   
    /**
    * @brief Fits the RandomForest model on a dataset with customs
    * params
    *
    *
    * @param data DataSet containing input samples and target values.
    * @param param SplitParam defining the splitting policy
    * (criterion, threshold computation method, and feature selection strategy).
    * 
    * @note Internally, each tree is trained using the DecisionTree::fit() method.
    */
    void fit(const DataSet& data, const SplitParam& param);

    //Returns whether the RandomForest has been fitted
    bool is_fitted() const {return fitted;}
    
    /**
    * @brief Predict class labels for a batch of samples.
    *
    * Predicts the class label for each input sample by first computing
    * class probabilities using @c predict_proba(), then applying a fixed
    * decision threshold of 0.5. Samples with predicted probability greater
    * than or equal to 0.5 are assigned to class 1, class 0 otherwise.
    *
    * @param sample Non owning view over a row-major representation
    * of a set of samples. The number of features of the samples must
    * be coherent with the number of features seen in training.
    *
    * @return A vector of predicted class labels (0 or 1), one per input sample
    *
    * @throws std::invalid_argument If the model has not been fitted or if the
    * input dimensions are inconsistent with the training data.
    *
    * @note This method does not modify the state of the model
    * @note The decision threshold is currently fixed at 0.5.
    */
    std::vector<float> predict(std::span<const float> sample) const;

    /**
    * @brief Predict class probabilities for a batch of samples.
    *
    * Computes the predicted probability of the positive class for each input
    * sample by averaging the votes of all decision trees in the forest.
    * Each tree contributes a binary prediction (0 or 1), and the final
    * probability is obtained as the percentage of trees predicting class 1
    *
    * @param sample Non owning view over a row-major representation
    * of a set of samples. The number of features of the samples must
    * be coherent with the number of features seen in training
    * @return A vector of probabilities in the range [0, 1], one per input sample
    *
    * @throws std::invalid_argument If the model has not been fitted or if the
    * input dimensions are inconsistent with the training data.
    * @throws std::logic_error If the forest contains no trained trees.
    *
    */
    std::vector<float> predict_proba(std::span<const float> sample) const;

    /**
     * @brief Compute the out-of-bag score of the RandomForest.
     *
     * Once fitted, allows to validate the tree by predicting samples
     * that were not bootstrapped and seen during training.
     *
     * @param data The DataSet seen during training
     * 
     * @return float : the accuracy of OOB prediction 
     */
    float out_of_bag(const DataSet& data) const;

    //Returns current seed
    std::uint32_t seed() const {return *seed_;}

    /**
     * @brief Set the seed for the RandomForest
     * 
     * @param user_seed the seed to be used for the RNG
     * @note The algorithm used for RNG is the Mersenne Twister algorithm
     */
    void set_seed(uint32_t user_seed);

    //Returns the number of sampled feature at each node
    int get_max_features() const {return mtry;}
    
    //returns the number of trees used for fitting
    int get_estimators() const {return n_estimators;}

    //Returns the max depth of the trees of the forest
    std::optional<int> get_max_depth() const {
        if (max_depth.has_value()) return max_depth.value();
            else return std::nullopt;}

    std::optional<float> get_max_samples() const {
        if (max_samples.has_value()) return max_samples.value();
            else return std::nullopt;}

    TreeType type_;


private:
    //Number of features to be sampled at each split 
    int mtry;
    //Number of trees in the forest
    int n_estimators; 
    //Max depth of each tree
    std::optional<int> max_depth; 
    std::optional<float> max_samples;
    std::optional<int> min_sample_split;

    /**
    * @brief Private method used to fit the RandomForest.
    *
    * Builds the forest by training @c n_estimators decision trees on bootstrapped
    * samples from the dataset. Random components (bootstrap sampling and
    * any randomized split logic such as RandomK) draw randomness from the
    * provided SplitContext.
    *
    * @param t Fits the t tree in the forest
    * @param data DataSet containing input samples and target values.
    * @param param SplitParam defining the splitting policy (criterion, threshold
    * computation method, and feature selection strategy).
    * @param context SplitContext providing runtime state required for stochastic
    * training (e.g. RNG / seed).
    * @note This method clears and rebuilds the internal tree container.
    */
    void fit_(size_t t, const DataSet& data, const SplitParam& param, SplitContext &context);
    //Wheter the RF model has already been fitted
    bool fitted = false;
    //Number of features seen during training. 
    int num_features;
    //seed : can be specified by the user (at declaration or via .set_seed()). Otherwise, 
    // is set by std::random_devices
    std::optional<std::uint32_t> seed_;
    std::vector<ForestTree> trees;
    // Parallelism
    int n_jobs; 

    //Accessor for tests
    friend struct arboria::test::RandomForestAccess;

};



}

