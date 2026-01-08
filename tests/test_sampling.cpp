/*
                                              TESTS FOR SAMPLING
*/

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>
#include <vector>
#include <random>
#include <unordered_set>



#include "split_strategy/sampling/sampling.h"

using arboria::sampling::bootstrap;
using arboria::sampling::subsample;

// ------------------ Bootstrapping -----------


TEST_CASE("Sampling - bootstrapping - basic usage"){

    size_t data_size = 10; 
    size_t n_samples = 5;
    std::mt19937 rng(0);
    
    std::vector<size_t> indices = bootstrap(data_size, n_samples, rng);

    REQUIRE(indices.size()==5);
    for (size_t id : indices){
        REQUIRE(id < data_size);
    }
}

TEST_CASE("Sampling - bootstrapping - repeated samples"){

    size_t data_size = 10; 
    size_t n_samples = 15;
    std::mt19937 rng(0);
    
    std::vector<size_t> indices = bootstrap(data_size, n_samples, rng);

    REQUIRE(indices.size()==15);
    for (size_t id : indices){
        REQUIRE(id < data_size);
    }
    
    std::unordered_set<size_t> uniq(indices.begin(), indices.end());
    REQUIRE(uniq.size() < indices.size()); //verify repeated values


}

TEST_CASE("Sampling - bootstrapping - error - s_size == 0"){


    size_t data_size = 0; 
    size_t n_samples = 15;
    std::mt19937 rng(0);
    
    REQUIRE_THROWS_AS(bootstrap(data_size, n_samples, rng), std::invalid_argument);

}

TEST_CASE("Sampling - bootstrapping - error - n_sample == 0"){


    size_t data_size = 10; 
    size_t n_samples = 0;
    std::mt19937 rng(0);
    
    REQUIRE_THROWS_AS(bootstrap(data_size, n_samples, rng), std::invalid_argument);

}

// ------------------ Subsampling -----------

TEST_CASE("Sampling - subsampling - basic usage"){

    size_t data_size = 10; 
    size_t n_samples = 5;
    std::mt19937 rng(0);
    
    std::vector<size_t> indices = subsample(data_size, n_samples, rng);

    REQUIRE(indices.size()==5);
    for (size_t id : indices){
        REQUIRE(id < data_size);
    }

    std::unordered_set<size_t> set(indices.begin(), indices.end());
    REQUIRE(set.size() == indices.size());

}

TEST_CASE("Sampling - subsampling - n_samples == s_size"){

    size_t data_size = 10; 
    size_t n_samples = 10;
    std::mt19937 rng(0);
    
    std::vector<size_t> indices = subsample(data_size, n_samples, rng);

    REQUIRE(indices.size()==10);
    for (size_t id : indices){
        REQUIRE(id < data_size);
    }

    std::unordered_set<size_t> set(indices.begin(), indices.end());
    REQUIRE(set.size() == indices.size());

}

TEST_CASE("Sampling - subsampling - errors : size_s == 0"){

    size_t data_size = 0; 
    size_t n_samples = 10;
    std::mt19937 rng(0);
    
    REQUIRE_THROWS_AS(subsample(data_size, n_samples, rng), std::invalid_argument);

}


TEST_CASE("Sampling - subsampling - errors : n_sample == 0"){

    size_t data_size = 10; 
    size_t n_samples = 0;
    std::mt19937 rng(0);
    
    REQUIRE_THROWS_AS(subsample(data_size, n_samples, rng), std::invalid_argument);

}


TEST_CASE("Sampling - subsampling - errors : n_sample > size_s"){

    size_t data_size = 10; 
    size_t n_samples = 11;
    std::mt19937 rng(0);
    
    REQUIRE_THROWS_AS(subsample(data_size, n_samples, rng), std::invalid_argument);

}