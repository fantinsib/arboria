#pragma once

/**
 * @brief struct controlling the passed arguments 
 * to the scoring function for classification
 *
 *  - l_pos, l_neg -> the positive and negative labels going to the left node 
 *  - r_pos, r_neg -> the positive and negative labels going to the right node 
 *
 */
struct ClfStats {

public:
    int l_pos = 0;
    int l_neg = 0;
    int r_pos = 0;
    int r_neg = 0;

};

/**
 * @brief struct controlling the passed arguments 
 * to the scoring function for regression
 *
 *  - nL, nR -> number of samples sent to left and right
 *  - y_sL, y_sR -> the sum of the target value sent to the left and right 
 *  - y_ssL, y_ssR -> the squared sum of the target values
 *
 */
struct RegStats {
    int nL;
    int nR;
    float y_ssL;
    float y_ssR;
    float y_sL;
    float y_sR;
};