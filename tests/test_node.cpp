/*
                                              TESTS FOR NODES
*/

#include <catch2/catch_test_macros.hpp>
#include <limits>

#include "node/node.h"
#include <cmath>

using arboria::Node;

TEST_CASE("Node default state") {
    Node node;
    REQUIRE(node.is_leaf == true);
    REQUIRE(node.return_feature_index() == -1);
    REQUIRE(node.leaf_value == -1);
    REQUIRE(!std::isfinite(node.return_threshold()));
    REQUIRE(node.left_child == nullptr);
    REQUIRE(node.right_child == nullptr);
    REQUIRE(node.is_valid(1) == false);
}

TEST_CASE("Node split validity") {
    Node node;
    node.is_leaf = false;
    node.feature_index = 1;
    node.threshold = 2.5f;
    node.left_child = std::make_unique<Node>();
    node.right_child = std::make_unique<Node>();

    REQUIRE(node.is_valid(3) == true);
}

TEST_CASE("Node split invalid cases") {
    Node node;
    node.is_leaf = false;
    node.feature_index = 0;
    node.threshold = 1.0f;
    node.left_child = std::make_unique<Node>();
    node.right_child = std::make_unique<Node>();

    node.feature_index = -1;
    REQUIRE(node.is_valid(2) == false);

    node.feature_index = 2;
    REQUIRE(node.is_valid(2) == false);

    node.feature_index = 1;
    node.threshold = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(node.is_valid(2) == false);

    node.threshold = 1.0f;
    node.left_child.reset();
    REQUIRE(node.is_valid(2) == false);
}

