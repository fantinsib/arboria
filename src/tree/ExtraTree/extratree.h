#pragma once
#include "split_strategy/types/split_hyper.h"
#include "split_strategy/types/split_param.h"
#include "tree/RandomForest/randomforest.h"
#include <_types/_uint32_t.h>
#include <optional>


class ExtraTree : public arboria::RandomForest {

public :

    ExtraTree(HyperParam hyperParam, TreeType type, std::optional<uint32_t> seed = std::nullopt);

};
