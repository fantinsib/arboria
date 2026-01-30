#pragma once

/**
 * @brief struct controlling the passed arguments to the scoring function
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

struct RegStats {
    int nL;
    int nR;
    float y_ssL;
    float y_ssR;
    float y_sL;
    float y_sR;
};