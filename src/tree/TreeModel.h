#pragma once

// Improvement : duplicate TreeType info both as Tree attribute
//and passed to Param; logic to fix 

//enum class to indicate the general class of the tree
enum class TreeModel {
    DecisionTree,
    RandomForest};