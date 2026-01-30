#include <iostream>
#include <stdexcept>

#include "dataset.h"


namespace arboria{
DataSet::DataSet(std::vector<float> X, std::vector<float> Y, int n_rows, int n_cols):

    X_(std::move(X)),
    y_(std::move(Y)),
    n_rows_(n_rows),
    n_cols_(n_cols)
{
    if (n_cols_*n_rows_ != X_.size()) throw std::invalid_argument("The specified number of rows and columns does not match the number of samples.");
    if (n_rows_ != y_.size()) throw std::invalid_argument("The size of y does not match the number of samples.");

}


DataSet DataSet::index_split(const std::vector<int>& index) const {
    // Returns a subsplit of the dataset object of the rows from the specified index

    //vector index references nth row of the dataset
    std::vector<float> X_results;
    std::vector<float> y_results;

    for (auto i : index){
        if (i<0 || i>= n_rows_) {throw std::out_of_range("DataSet.index_split : row index out of bounds");}
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
}
