/*

                TESTS DATASET

*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>  
#include <cmath>
#include <stdexcept>
#include <vector>

#include "dataset/dataset.h"

using arboria::DataSet;

TEST_CASE("Basic usage") {

    std::vector<float> x{1,2,3,
                        4,5,6,
                        7, 8 ,9,
                        10, 11,12};
    std::vector<float> y{0,0,1,1};
    
    DataSet data(x, y, 4, 3);

    REQUIRE(data.n_rows() == 4);
    REQUIRE(data.n_cols() == 3);
    REQUIRE(data.is_empty() == false);
    REQUIRE(data.iloc_x(0,1) == 2);
    REQUIRE(data.iloc_x(3,2) == 12);
    REQUIRE(data.iloc_y(2) == 1);

}

TEST_CASE("Constructor Errors : incoherent n_row/n_col") {

    std::vector<float> x{1,2,3,
                        4,5,6,
                        7, 8 ,9,
                        10, 11,12};
    std::vector<float> y{0,0,1,1};
    
    
    REQUIRE_THROWS_AS(DataSet(x, y, 3, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(DataSet(x, y, 4, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(DataSet(x, y, -1, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(DataSet(x, y, 4, -1), std::invalid_argument);

}

TEST_CASE("Constructor Errors : incoherent X.size/y.size") {

    std::vector<float> x{1,2,3,
                        4,5,6,
                        7, 8 ,9,
                        10, 11,12};
    std::vector<float> y{0,0,1,};
    
    
    REQUIRE_THROWS_AS(DataSet(x, y, 4, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(DataSet(x, y, 3, 3), std::invalid_argument);

}


TEST_CASE("Is empty") {

    std::vector<float> x;
    std::vector<float> y;
    DataSet data(x, y, 0, 0);
    REQUIRE(data.is_empty() == true);

}

TEST_CASE("Index split basic usage") {

    std::vector<float> X{1,2,3,
                        4,5,6,
                        7, 8 ,9,
                        10, 11,12};
    std::vector<float> y{0,0,1,1};
    DataSet data(X, y, 4,3);
    std::vector<int> idx_l{1,3};
    DataSet left_sub = data.index_split(idx_l); //shoudl result in a left_sub dataset containing
                                                       //only the 2nd and 4th sample of the original DataSet
    
    REQUIRE(left_sub.iloc_x(0,1) == 5);
    REQUIRE(left_sub.iloc_x(1,2) == 12);
    
    std::vector<int> idx_r{0,4};    
    REQUIRE_THROWS_AS(data.index_split(idx_r), std::out_of_range);


}

TEST_CASE("Index split from empty vector") {

    std::vector<float> X{1,2,3,
                        4,5,6,
                        7, 8 ,9,
                        10, 11,12};
    std::vector<float> y{0,0,1,1};
    DataSet data(X, y, 4,3);
    std::vector<int> idx_l{};
    DataSet left_sub = data.index_split(idx_l); 
                                                       
    
    REQUIRE(left_sub.is_empty() == true);

}

TEST_CASE("Index split with duplicate indices") {

    std::vector<float> X{1,2,3,
                        4,5,6,
                        7, 8 ,9,
                        10, 11,12};
    std::vector<float> y{0,0,1,1};
    DataSet data(X, y, 4,3);
    std::vector<int> idx_l{1,1};
    DataSet left_sub = data.index_split(idx_l); 
                                                       
    
    REQUIRE(left_sub.iloc_x(0,0) == left_sub.iloc_x(1,0));
    REQUIRE(left_sub.iloc_x(0,1) == left_sub.iloc_x(1,1));
    REQUIRE(left_sub.iloc_x(0,2) == left_sub.iloc_x(1,2));

}

TEST_CASE("Index split : order conservation") {

    std::vector<float> X{1,2,3,
                        4,5,6,
                        7, 8 ,9,
                        10, 11,12};
    std::vector<float> y{0,0,1,1};
    DataSet data(X, y, 4,3);
    std::vector<int> idx_l{3,1};
    DataSet left_sub = data.index_split(idx_l); 


    REQUIRE(left_sub.iloc_x(0,0) == 10);
    REQUIRE(left_sub.iloc_x(0,1) == 11);
    REQUIRE(left_sub.iloc_x(0,2) == 12);

    REQUIRE(left_sub.iloc_x(1,0) == 4);
    REQUIRE(left_sub.iloc_x(1,1) == 5);
    REQUIRE(left_sub.iloc_x(1,2) == 6);

}


TEST_CASE("iloc out of range") {

    std::vector<float> X{1,2,3,
                        4,5,6,
                        7, 8 ,9,
                        10, 11,12};
    std::vector<float> y{0,0,1,1};
    DataSet data(X, y, 4,3);

    REQUIRE_THROWS_AS(data.iloc_x(4,0), std::out_of_range);
    REQUIRE_THROWS_AS(data.iloc_x(2,4), std::out_of_range);
    REQUIRE_THROWS_AS(data.iloc_x(-1,2), std::out_of_range);
    REQUIRE_THROWS_AS(data.iloc_x(1,-2), std::out_of_range);

    REQUIRE_THROWS_AS(data.iloc_y(4), std::out_of_range);
    REQUIRE_THROWS_AS(data.iloc_y(-1), std::out_of_range);

}

