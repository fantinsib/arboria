/*
                                              TESTS FOR ENTROPY CALCULATIONS
*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>  
#include <stdexcept>
#include <vector>
#include <cmath>
#include "split_criterion/entropy.h"

using arboria::split::entropy;
using arboria::split::weighted_entropy;



TEST_CASE("entropy Calculations (float overload)", "[entropy][float]") {

    //Regular cases :
    REQUIRE(entropy(0.5f, 0.5f) == Catch::Approx(1.f));
    REQUIRE(entropy(0.f, 1.f) == Catch::Approx(0.f));
    REQUIRE(entropy(1.f, 0.f) == Catch::Approx(0.f));
    REQUIRE(entropy(0.4f, 0.6f) == Catch::Approx(-0.6*std::log2(0.6) - 0.4*std::log2(0.4)));

    //Errors :
    REQUIRE_THROWS_AS(entropy(0.f,0.f), std::invalid_argument);
    REQUIRE_THROWS_AS(entropy(1.2f,0.3f), std::invalid_argument);
    REQUIRE_THROWS_AS(entropy(0.9f,-0.1f), std::invalid_argument);
    REQUIRE_THROWS_AS(entropy(1.1f,-0.1f), std::invalid_argument);
}


TEST_CASE("entropy Calculations (int overload)", "[entropy][int]") {

    // Regular cases :
    REQUIRE(entropy(10, 0) == Catch::Approx(0.0f));
    REQUIRE(entropy(0, 10) == Catch::Approx(0.0f));
    REQUIRE(entropy(5, 5) == Catch::Approx(1.0f));
    REQUIRE(entropy(3, 7) == Catch::Approx(-(0.3)*std::log2(0.3) - (0.7)*std::log2(0.7)));

    //Errors : 
    REQUIRE_THROWS_AS(entropy(0,0), std::invalid_argument);
    REQUIRE_THROWS_AS(entropy(-3,3), std::invalid_argument);
    REQUIRE_THROWS_AS(entropy(10,-9), std::invalid_argument);

}

TEST_CASE("entropy Calculations (vector float overload)", "[entropy][std::vector]") {

    // Regular cases :
    SECTION("Balanced data"){
    std::vector<float> a{0,1,0,1};
    REQUIRE(entropy(a) == Catch::Approx(1.f));
    }
    SECTION("Pure class"){
    std::vector<float> a{0,0,0,0};
    REQUIRE(entropy(a) == Catch::Approx(0.f));
    }
    SECTION("Unbalanced class"){
    std::vector<float> a{0,1,0,0};
    REQUIRE(entropy(a) == Catch::Approx(-(0.75*std::log2(0.75))- 0.25*std::log2(0.25)));
    }

    //Errors : 

    SECTION("Non-binary labels"){
    std::vector<float> a{0,1,3,0};
    REQUIRE_THROWS_AS(entropy(a), std::invalid_argument);
    }

    SECTION("Empty Vector"){
    std::vector<float> a{};
    REQUIRE_THROWS_AS(entropy(a), std::invalid_argument);
    }

}

TEST_CASE("entropy Calculations (vector int)", "[entropy][std::vector]") {

    // Regular cases :
    SECTION("Balanced data"){
    std::vector<int> a{0,1,0,1};
    REQUIRE(entropy(a) == Catch::Approx(1.f));
    }
    SECTION("Pure class"){
    std::vector<int> a{0,0,0,0};
    REQUIRE(entropy(a) == Catch::Approx(0.f));
    }
    SECTION("Unbalanced class"){
    std::vector<int> a{0,1,0,0};
    REQUIRE(entropy(a) == Catch::Approx(-(0.75*std::log2(0.75))- 0.25*std::log2(0.25)));
    }

    //Errors : 

    SECTION("Non-binary labels"){
    std::vector<int> a{0,1,3,0};
    REQUIRE_THROWS_AS(entropy(a), std::invalid_argument);
    }

    SECTION("Empty Vector"){
    std::vector<int> a{};
    REQUIRE_THROWS_AS(entropy(a), std::invalid_argument);
    }

}


TEST_CASE("Weighted entropy Calculations", "[entropy][std::vector]") {

    // Regular cases
    SECTION("Balanced pure children") {
    std::vector<float> a{0,0,0,0};
    std::vector<float> b{1,1,1,1};
    REQUIRE(weighted_entropy(a, b) == Catch::Approx(0.f));
    }

    SECTION("Both perfectly mixed") {
    std::vector<float> a{0,1,0,1};
    std::vector<float> b{0,1,0,1};
    REQUIRE(weighted_entropy(a, b) == Catch::Approx(1.f));
    }

    SECTION("One mixed one pure") {
    std::vector<float> a{0,1,0,1};
    std::vector<float> b{0,0,0,0};
    REQUIRE(weighted_entropy(a, b) == Catch::Approx(0.5f));
    }

    SECTION("Unequal size") {
    std::vector<float> a{0,0};
    std::vector<float> b{0,1,0,1,0,1,0,1};
    REQUIRE(weighted_entropy(a, b) == Catch::Approx(0.8f));
    }

    // Edge case
    SECTION("One empty child") {
    std::vector<float> a{};
    std::vector<float> b{0,1,0,1}; 
    REQUIRE(weighted_entropy(a, b) == Catch::Approx(1.f));
    }

    // Errors
    SECTION("Empty inputs") {
    std::vector<float> a{};
    std::vector<float> b{};
    REQUIRE_THROWS_AS(weighted_entropy(a, b), std::invalid_argument);
    }

    SECTION("Non binary labels") {
    std::vector<float> a{0,2,0};
    std::vector<float> b{1,1,1};
    REQUIRE_THROWS_AS(weighted_entropy(a, b), std::invalid_argument);
    }
}


TEST_CASE("Weighted entropy Calculations (int overload)", "[entropy][int]") {

    // Regular cases
    SECTION("Balanced pure children") {
    REQUIRE(weighted_entropy(0, 4, 4,0) == Catch::Approx(0.f));
    }

    SECTION("Both perfectly mixed") {
    REQUIRE(weighted_entropy(2,2, 2,2) == Catch::Approx(1.f));
    }

    SECTION("One mixed one pure") {
    REQUIRE(weighted_entropy(2,2,0,4) == Catch::Approx(0.5f));
    }

    SECTION("Unequal size") {
    REQUIRE(weighted_entropy(0,2, 4, 4) == Catch::Approx(0.8f));
    }

    // Edge case
    SECTION("One empty child") {    
    REQUIRE(weighted_entropy(0, 0, 2,2) == Catch::Approx(1.f));
    }

    // Errors
    SECTION("Empty inputs") {
    REQUIRE_THROWS_AS(weighted_entropy(0,0,0,0), std::invalid_argument);
    }

    SECTION("Negative counts") {
    REQUIRE_THROWS_AS(weighted_entropy(1,2,0,-1), std::invalid_argument);
    }
}
