/*
        BINDINGS IMPLEMENTATION
*/

#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>

#include "dataset/dataset.h"
#include "tree/DecisionTree/DecisionTree.h"
#include "split_criterion/entropy.h"
#include "split_criterion/gini.h"
#include "split_strategy/types/split_param.h"
#include "tree/RandomForest/randomforest.h"

namespace py = pybind11;

PYBIND11_MODULE(arboria, m){

    py::class_<arboria::DecisionTree>(m, "DecisionTree")
    .def(py::init<int>())

    .def("fit",
    [](arboria::DecisionTree& self, 
         py::array_t<float, py::array::c_style | py::array::forcecast> X,
        py::array_t<float, py::array::c_style | py::array::forcecast> y,
        const std::string& criterion)
    {
                
        //Input checks
                auto xb = X.request();
                if (xb.ndim != 2) {
                    throw std::runtime_error("X must be a 2D numpy array.");
                }
                auto yb = y.request();
                if (yb.ndim != 1) {
                    throw std::runtime_error("y must be a 1D numpy array.");
                }
                const size_t n_rows = static_cast<size_t>(xb.shape[0]);
                const size_t n_cols = static_cast<size_t>(xb.shape[1]);
                if ((size_t)yb.shape[0] != n_rows) {
                    throw std::runtime_error("y length must match X.shape[0].");
                }
                
                Criterion crit;
                if (criterion == "gini") crit = Criterion::Gini;
                else if (criterion == "entropy") crit = Criterion::Entropy;
                else throw std::runtime_error("Unknown split criterion passed to fit.");
                
                SplitParam param;
                param.criterion = crit;

                const float* X_ptr = static_cast<const float*>(xb.ptr);
                const float* y_ptr = static_cast<const float*>(yb.ptr);

                std::vector<float> X_vec(X_ptr, X_ptr + n_rows * n_cols);
                std::vector<float> y_vec(y_ptr, y_ptr + n_rows);

                arboria::DataSet data(std::move(X_vec), std::move(y_vec), n_rows, n_cols);
                self.fit(data, param);
            },
            
            py::arg("X"), py::arg("y"), py::arg("criterion") = "gini"
        )

        .def("predict",
        [](arboria::DecisionTree& self, 
           py::array_t<float, py::array::c_style | py::array::forcecast> X
        )
        {
            auto xb = X.request();
            size_t ndim = xb.ndim;
        
            if( ndim ==1){
            const size_t n_cols = static_cast<size_t>(xb.size);
            const float* x_ptr = static_cast<const float*>(xb.ptr);
            std::vector<float> X_vec(x_ptr, x_ptr + n_cols);
            return std::vector<int>{self.predict_one(X_vec)};
            }

            else if (ndim == 2){
            const size_t n_rows = static_cast<size_t>(xb.shape[0]);
            const size_t n_cols = static_cast<size_t>(xb.shape[1]);
            const float* x_ptr = static_cast<const float*>(xb.ptr);
            std::vector<float> X_vec(x_ptr, x_ptr +n_rows*n_cols);
            return self.predict(X_vec);
            }

            else{
            throw std::runtime_error("X must be a 1D or 2D numpy array");
            }
        },
        py::arg("X")
    )

        .def_property_readonly("is_fitted", &arboria::DecisionTree::is_fitted);




    py::class_<arboria::RandomForest>(m, "RandomForest")
        .def(py::init<int, int, int, std::optional<uint32_t>>(),
            py::arg("n_estimators"), 
            py::arg("m_try"),
            py::arg("max_depth"),
            py::arg("seed") = std::nullopt
    )

        .def("fit", 
            [](arboria::RandomForest& self, 
            py::array_t<float, py::array::c_style | py::array::forcecast> X,
            py::array_t<float, py::array::c_style | py::array::forcecast> y,
            const std::string& criterion) {
                
        //Input checks
                auto xb = X.request();
                if (xb.ndim != 2) {
                    throw std::runtime_error("X must be a 2D numpy array.");
                }
                auto yb = y.request();
                if (yb.ndim != 1) {
                    throw std::runtime_error("y must be a 1D numpy array.");
                }
                const size_t n_rows = static_cast<size_t>(xb.shape[0]);
                const size_t n_cols = static_cast<size_t>(xb.shape[1]);
                if ((size_t)yb.shape[0] != n_rows) {
                    throw std::runtime_error("y length must match X.shape[0].");
                }
                
                Criterion crit;
                if (criterion == "gini") crit = Criterion::Gini;
                else if (criterion == "entropy") crit = Criterion::Entropy;
                else throw std::runtime_error("Unknown split criterion passed to fit.");
                
                SplitParam param;
                param.criterion = crit;

                const float* X_ptr = static_cast<float*>(xb.ptr);
                const float* y_ptr = static_cast<float*>(yb.ptr);

                std::vector<float> X_vec(X_ptr, X_ptr + n_rows * n_cols);
                std::vector<float> y_vec(y_ptr, y_ptr + n_rows);

                arboria::DataSet data(std::move(X_vec), std::move(y_vec), n_rows, n_cols);
                self.fit(data, param);
            },
            
            py::arg("X"), py::arg("y"), py::arg("criterion") = "gini"
        )

        .def("predict", 
        [](arboria::RandomForest& self, py::array_t<float, py::array::c_style | py::array::forcecast> X){

            auto xb = X.request();
            size_t ndim = xb.ndim;
            
            if (ndim == 1){
                const size_t n_cols = xb.size;
                const float* x_ptr = static_cast<float*>(xb.ptr);
                const std::vector<float> X_vec(x_ptr, x_ptr+n_cols);
                return self.predict(X_vec);

            }

            else if (ndim == 2){
                
                const size_t n_rows = static_cast<size_t>(xb.shape[0]);
                const size_t n_cols = static_cast<size_t>(xb.shape[1]);

                const float* x_ptr = static_cast<float*>(xb.ptr);

                const std::vector<float> X_vec(x_ptr, x_ptr+n_cols*n_rows);
                return self.predict(X_vec);
            }

            else {throw std::runtime_error("RandomForest.predict : invalid dimension of input");}

        }
        
        )

        .def("predict_proba",
            [](arboria::RandomForest& self, py::array_t<float, py::array::c_style | py::array::forcecast> X){
                
                auto xb = X.request();
                const size_t ndim = xb.ndim;

                if (ndim == 1){

                    const size_t n_cols = xb.size;
                    const float* x_ptr = static_cast<float*>(xb.ptr);
                    std::vector<const float> X_vec(x_ptr, x_ptr + n_cols);
                    return self.predict_proba(X_vec);
                }

                else if (ndim == 2){
                    const size_t n_rows = xb.shape[0];
                    const size_t n_cols = xb.shape[1];

                    const float* x_ptr = static_cast<float*>(xb.ptr);

                    std::vector<const float> X_vec(x_ptr, x_ptr+n_cols*n_rows);
                    return self.predict_proba(X_vec);

                }

                else {throw std::runtime_error("RandomForest.predict_proba : invalid dimension on inputs");}



            }
        
        
        
        )

        .def("out_of_bag",
            [](arboria::RandomForest& self, py::array_t<float, py::array::c_style | py::array::forcecast> X,
            py::array_t<float, py::array::c_style | py::array::forcecast> y){
                
                auto xb = X.request();
                if (xb.ndim != 2) {
                    throw std::runtime_error("X must be a 2D numpy array.");
                }
                auto yb = y.request();
                if (yb.ndim != 1) {
                    throw std::runtime_error("y must be a 1D numpy array.");
                }
                const size_t n_rows = static_cast<size_t>(xb.shape[0]);
                const size_t n_cols = static_cast<size_t>(xb.shape[1]);
                if ((size_t)yb.shape[0] != n_rows) {
                    throw std::runtime_error("y length must match X.shape[0].");
                }

                const float* X_ptr = static_cast<const float*>(xb.ptr);
                const float* y_ptr = static_cast<const float*>(yb.ptr);

                std::vector<float> X_vec(X_ptr, X_ptr + n_rows * n_cols);
                std::vector<float> y_vec(y_ptr, y_ptr + n_rows);

                arboria::DataSet data(std::move(X_vec), std::move(y_vec), n_rows, n_cols);
                return self.out_of_bag(data);
            }
        
        );


        

    }