#include <iostream>
#include "dataset.h"


namespace arboria{

DataSet::DataSet(std::vector<float> X, std::vector<float> Y, int n_rows, int n_cols):

    X_(X),
    y_(Y),
    n_rows_(n_rows),
    n_cols_(n_cols)
{
    if (n_cols_*n_rows_ != X_.size()) throw std::invalid_argument("The specified number of rows and columns does not match the number of samples.");
    if (n_rows_ != y_.size()) throw std::invalid_argument("The size of y does not match the number of samples.");
}

std::span<const float> DataSet::row_X(int row) const {

    std::span<const float> cs(X_);
    auto sub = cs.subspan(row*n_cols_, n_cols_);
    return sub;
}

std::span<const float> DataSet::row_y(int row) const {

    std::span<const float> cs(y_);
    auto sub = cs.subspan(row,1);
    return sub;
}

std::pair<int, int> DataSet::count_classes() const {
    // Returns a std::pair with
    // first -> the number of positive class in the dataset
    // second -> the number of negative class in the dataset

    int pos_count = 0;
    int neg_count = 0;

    for (auto i : y_){

        if (i==0) neg_count++;
        if (i==1) pos_count++;
    }

    return {pos_count, neg_count};
}

DataSet DataSet::index_split(std::vector<int>& index) const {
    // Returns a subsplit of the dataset object of the rows from the specified index

    //vector index references nth row of the dataset
    std::vector<float> X_results;
    std::vector<float> y_results;

    for (auto i : index){
        for (int col = 0; col < n_cols_; col++){
            X_results.push_back(iloc_x(i, col));


    }
        y_results.push_back(iloc_y(i));
    }

    DataSet output(X_results, y_results, index.size(), n_cols_);
    return output;
}

void DataSet::print() const {

    for (int r = 0; r < n_rows_; r++){
        std::cout << "Sample " << r << " | Target : " << iloc_y(r) << std::endl;
        for (int col = 0; col < n_cols_; col++){

            std::cout << iloc_x(r, col) << ", ";

        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
}

Table::Table(std::vector<float>& x, int n_cols):

    X_(x),
    n_cols_(n_cols)
    {
        n_rows_ = x.size()/n_cols_;
      
    };












}








