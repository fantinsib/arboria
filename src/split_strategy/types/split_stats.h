#pragma once

/**
 * @brief struct controlling the passed arguments to the scoring function
 *
 *  - l_pos, l_neg -> the positive and negative labels going to the left node 
 *  - r_pos, r_neg -> the positive and negative labels going to the right node 
 *
 */
struct SplitStats {

public:
    int l_pos = 0;
    int l_neg = 0;
    int r_pos = 0;
    int r_neg = 0;

};