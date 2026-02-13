#pragma once
#include "split_strategy/types/split_hyper.h"
#include "split_strategy/types/split_param.h"
#include "tree/RandomForest/randomforest.h"
#include <_types/_uint32_t.h>
#include <optional>

namespace arboria {

class ExtraTree : public arboria::RandomForest {

public:
    //Constructor for ExtraTree
    ExtraTree(HyperParam param, TreeType type, std::optional<uint32_t> seed= std::nullopt);

    int get_n_random_split() const {return n_random_split;}

private:
    int n_random_split; 

};

}