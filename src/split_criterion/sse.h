
#pragma once

#include <vector>
#include <cmath>
#include <utility>
#include <stdexcept>


namespace arboria {
namespace split{


/**
 * @brief Computes the sum of squared errors 
 * 
 * @param y_ss the total sum of the squared y values
 * @param y_s the total sum of the y values
 * @param n the number of values
 * @param c the mean of the values
 * @return float 
 */
inline float sum_of_squared_errors(float y_ss, float y_s, int n, float c) {
    return y_ss - 2*c*y_s + n*c*c;
};

inline float weighted_sse(int nL, int nR, float y_sL, float y_sR, float y_ssL, float y_ssR){

    float c_l = y_sL/static_cast<float>(nL);
    float sse_left = sum_of_squared_errors(y_ssL, y_sL, nL, c_l);

    float c_r = y_sR/static_cast<float>(nR);
    float sse_right = sum_of_squared_errors(y_ssR, y_sR, nR, c_r);
    
    return sse_left  + sse_right;
}
}
}