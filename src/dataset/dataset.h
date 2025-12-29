/*

#########################################################################

*/
#include <iostream>
#include <vector>


namespace arboria {

class DataSet
/*
DataSet class -> allows manipulation of the entire dataset (X and y values) for
easier processing throughout the algorithms.
*/
{
public:
    /**
     * @brief Constructs a DataSet from features X and targets Y
     * @param X Flattened feature vector (expected size = n_rows * n_cols)
     * @param Y Target vector (expected size = n_rows)
     * @param n_rows Number of samples
     * @param n_cols Number of features per sample
     * @throws std::invalid_argument if X.size(), Y.size(), n_rows and n_cols values are incoherent 
     * @note X uses row-major order: X[col + row * n_cols]
     */

    DataSet(std::vector<float> X, std::vector<float> Y, int n_rows, int n_cols);
    
    // Returns the number of samples in the DataSet
    int n_rows() const {return n_rows_;}

    // Returns the number of features in the DataSet
    int n_cols() const {return n_cols_;}

    // Returns a 1D vector of the samples
    const std::vector<float>& X() const {return X_;}

    // Returns the target values
    const std::vector<float>& y() const {return y_;}

    /**
     * @brief Returns the value of a sample's feature
     * @param row row index of the sample (0<= row < n_rows_)
     * @param col col index of the sample (0<= col < n_cols_)
     * @throws std::out_of_range if row or col out of bounds
     * @note X uses row-major order: X[col + row * n_cols].
     */
    float iloc_x(int row, int col) const {
        if (row < 0 || row >= n_rows_ || col < 0 || col >= n_cols_){throw std::out_of_range("DataSet.iloc_x : index out of bound");}
        return X_[col +row*n_cols_];
    }

    /**
     * @brief Returns the value of a sample's class
     * @param row row index of the sample (0<= row < n_rows_)
     * @throws std::out_of_range if row out of bounds
     */
    float iloc_y(int row) const {
        if (row < 0 || row >= n_rows_) {throw std::out_of_range("DataSet.iloc_y : index out of bound");}
        return y_[row];
    }

    /**
     * @brief Returns a DataSet containing a subset of rows
     * 
     * @param index Vector of row indices (0<= indices < n_rows_)
     * @return DataSet 
     * @throws std::out_of_range if an index value is out of bounds
     * @note Rows are returned in the same order as specified in 'index'
     */
    DataSet index_split(const std::vector<int>& index) const;

    /**
     * @brief Prints the DataSet content to standard output
     * This function prints the feature matrix X and target vector y in a readable format for debugging.
     * 
     */
    void print() const;


private:
    std::vector<float> X_;
    std::vector<float> y_;
    int n_rows_;
    int n_cols_;
};

}