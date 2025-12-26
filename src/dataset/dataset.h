/*

*****************
* DataSet Class * 
*****************

Implements a class to hold and manipulate the data as 1D vectors


*/




#include <iostream>
#include <vector>
#include <span>


namespace arboria {

class DataSet
{
public:
    DataSet(std::vector<float> X, std::vector<float> Y, int n_rows, int n_cols);
    int n_rows() const {return n_rows_;}
    int n_cols() const {return n_cols_;}

    const std::vector<float>& X() const {return X_;}
    const std::vector<float>& y() const {return y_;}

    // iloc_X -> takes row & col, return the (i,j) element of the matrix
    float iloc_x(int row, int col) const {return X_[col +row*n_cols_];}

    // iloc_y -> takes row, returns the jth element of the target vector
    float iloc_y(int row) const {return y_[row];}

    std::span<const float> row_X(int row) const;
    std::span<const float> row_y(int row) const;

    DataSet index_split(std::vector<int>& index) const;

    void print() const;

    std::pair<int, int> count_classes() const;


private:
    std::vector<float> X_;
    std::vector<float> y_;
    int n_rows_;
    int n_cols_;
};


class Table
{
    Table(std::vector<float>& x, int n_rows);
    




}










}