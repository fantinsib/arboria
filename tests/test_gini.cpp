/*
                                              TESTS FOR GINI CALCULATIONS
*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>  
#include <stdexcept>
#include <vector>
#include "split_criterion/gini.h"

using arboria::split::gini;
using arboria::split::weighted_gini;

TEST_CASE("Gini Calculations (float overload)", "[gini][float]") {

    //Regular cases :
    REQUIRE(gini(0.5f, 0.5f) == Catch::Approx(0.5f));
    REQUIRE(gini(0.f, 1.f) == Catch::Approx(0.f));
    REQUIRE(gini(1.f, 0.f) == Catch::Approx(0.f));
    REQUIRE(gini(0.4f, 0.6f) == Catch::Approx(1.f - 0.4f*0.4f -0.6f*0.6f));

    //Errors :
    REQUIRE_THROWS_AS(gini(0.f,0.f), std::invalid_argument);
    REQUIRE_THROWS_AS(gini(1.2f,0.3f), std::invalid_argument);
    REQUIRE_THROWS_AS(gini(0.9f,-0.1f), std::invalid_argument);
    REQUIRE_THROWS_AS(gini(1.1f,-0.1f), std::invalid_argument);
}


TEST_CASE("Gini Calculations (int overload)", "[gini][int]") {

    // Regular cases :
    REQUIRE(gini(10, 0) == Catch::Approx(0.0f));
    REQUIRE(gini(0, 10) == Catch::Approx(0.0f));
    REQUIRE(gini(5, 5) == Catch::Approx(0.5f));
    REQUIRE(gini(3, 7) == Catch::Approx(1.0f - 0.3f*0.3f - 0.7f*0.7f));

    //Errors : 
    REQUIRE_THROWS_AS(gini(0,0), std::runtime_error);
    REQUIRE_THROWS_AS(gini(-3,3), std::invalid_argument);
    REQUIRE_THROWS_AS(gini(10,-9), std::invalid_argument);

}

TEST_CASE("Gini Calculations (vector float overload)", "[gini][std::vector]") {

    // Regular cases :
    SECTION("Balanced data"){
    std::vector<float> a{0,1,0,1};
    REQUIRE(gini(a) == Catch::Approx(0.5f));
    }
    SECTION("Pure class"){
    std::vector<float> a{0,0,0,0};
    REQUIRE(gini(a) == Catch::Approx(0.f));
    }
    SECTION("Unbalanced class"){
    std::vector<float> a{0,1,0,0};
    REQUIRE(gini(a) == Catch::Approx(0.375f));
    }

    //Errors : 

    SECTION("Non-binary labels"){
    std::vector<float> a{0,1,3,0};
    REQUIRE_THROWS_AS(gini(a), std::invalid_argument);
    }

    SECTION("Empty Vector"){
    std::vector<float> a{};
    REQUIRE_THROWS_AS(gini(a), std::invalid_argument);
    }

}

TEST_CASE("Gini Calculations (vector int overload)", "[gini][std::vector]") {

    // Regular cases :
    SECTION("Balanced data"){
    std::vector<int> a{0,1,0,1};
    REQUIRE(gini(a) == Catch::Approx(0.5f));
    }
    SECTION("Pure class"){
    std::vector<int> a{0,0,0,0};
    REQUIRE(gini(a) == Catch::Approx(0.f));
    }
    SECTION("Unbalanced class"){
    std::vector<int> a{0,1,0,0};
    REQUIRE(gini(a) == Catch::Approx(0.375f));
    }

    //Errors : 

    SECTION("Non-binary labels"){
    std::vector<int> a{0,1,3,0};
    REQUIRE_THROWS_AS(gini(a), std::invalid_argument);
    }

    SECTION("Empty Vector"){
    std::vector<int> a{};
    REQUIRE_THROWS_AS(gini(a), std::invalid_argument);
    }

}

TEST_CASE("Weighted Gini Calculations (vector overload)", "[gini][std::vector]") {

    // Regular cases
    SECTION("Balanced pure children") {
    std::vector<float> a{0,0,0,0};
    std::vector<float> b{1,1,1,1};
    REQUIRE(weighted_gini(a, b) == Catch::Approx(0.f));
    }

    SECTION("Both perfectly mixed") {
    std::vector<float> a{0,1,0,1};
    std::vector<float> b{0,1,0,1};
    REQUIRE(weighted_gini(a, b) == Catch::Approx(0.5f));
    }

    SECTION("One mixed one pure") {
    std::vector<float> a{0,1,0,1};
    std::vector<float> b{0,0,0,0};
    REQUIRE(weighted_gini(a, b) == Catch::Approx(0.25f));
    }

    SECTION("Unequal size") {
    std::vector<float> a{0,0};
    std::vector<float> b{0,1,0,1,0,1,0,1};
    REQUIRE(weighted_gini(a, b) == Catch::Approx(0.4f));
    }

    // Edge case
    SECTION("One empty child") {
    std::vector<float> a{};
    std::vector<float> b{0,1,0,1}; 
    REQUIRE(weighted_gini(a, b) == Catch::Approx(0.5f));
    }

    // Errors
    SECTION("Empty inputs") {
    std::vector<float> a{};
    std::vector<float> b{};
    REQUIRE_THROWS_AS(weighted_gini(a, b), std::invalid_argument);
    }

    SECTION("Non binary labels") {
    std::vector<float> a{0,2,0};
    std::vector<float> b{1,1,1};
    REQUIRE_THROWS_AS(weighted_gini(a, b), std::invalid_argument);
    }
}


TEST_CASE("Weighted Gini Calculations (int overload)", "[gini][int]") {

    // Regular cases
    SECTION("Balanced pure children") {
    REQUIRE(weighted_gini(0, 4, 4,0) == Catch::Approx(0.f));
    }

    SECTION("Both perfectly mixed") {
    REQUIRE(weighted_gini(2,2, 2,2) == Catch::Approx(0.5f));
    }

    SECTION("One mixed one pure") {
    REQUIRE(weighted_gini(2,2,0,4) == Catch::Approx(0.25f));
    }

    SECTION("Unequal size") {
    REQUIRE(weighted_gini(0,2, 4, 4) == Catch::Approx(0.4f));
    }

    // Edge case
    SECTION("One empty child") {    
    REQUIRE(weighted_gini(0, 0, 2,2) == Catch::Approx(0.5f));
    }

    // Errors
    SECTION("Empty inputs") {
    REQUIRE_THROWS_AS(weighted_gini(0,0,0,0), std::invalid_argument);
    }

    SECTION("Negative counts") {
    REQUIRE_THROWS_AS(weighted_gini(1,2,0,-1), std::invalid_argument);
    }
}
