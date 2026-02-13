#include "tree/ExtraTree/extratree.h"
#include "split_strategy/types/split_hyper.h"
#include "split_strategy/types/split_param.h"
#include "tree/RandomForest/randomforest.h"
#include <__atomic/aliases.h>
#include <optional>
#include <stdexcept>


using arboria::ExtraTree;

namespace arboria {

ExtraTree::ExtraTree(HyperParam hyperParam, TreeType type, std::optional<std::uint32_t> user_seed)
: arboria::RandomForest(hyperParam, type, user_seed)
{   
    if (hyperParam.n_random_split.has_value()) {
        if (*hyperParam.n_random_split <= 0) throw std::invalid_argument("aboria::tree::ExtraTree : n_random_split argument must be strictly positive");
        n_random_split = *hyperParam.n_random_split;
    }
};

}