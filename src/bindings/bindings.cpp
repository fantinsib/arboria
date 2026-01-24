/*
        BINDINGS IMPLEMENTATION
*/

#include <pybind11/pytypes.h>
#include <ranges>
#include <concepts>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cstdint>

#include "dataset/dataset.h"
#include "tree/DecisionTree/DecisionTree.h"
#include "split_criterion/entropy.h"
#include "split_criterion/gini.h"
#include "split_strategy/types/split_param.h"
#include "split_strategy/types/ParamBuilder/ParamBuilder.h"
#include "tree/RandomForest/randomforest.h"
#include "helpers/helpers.h"

namespace py = pybind11;
using arboria::ParamBuilder;

PYBIND11_MODULE(_arboria, m){

    py::class_<arboria::DecisionTree>(m, "DecisionTree")
        .def(py::init([](std::optional<int> max_depth,
                                 std::optional<int> min_sample_split)
                        {        
                        HyperParam hp;
                        if (max_depth.has_value()) hp.max_depth = max_depth;
                        hp.min_sample_split = min_sample_split;

                        return std::make_unique<arboria::DecisionTree>(hp);}
                    ),
            py::arg("max_depth") = std::nullopt,
            py::arg("min_sample_split") = std::nullopt
    )

    .def("_fit",
    [](arboria::DecisionTree& self, 
         py::array_t<float, py::array::c_style | py::array::forcecast> X,
        py::array_t<float, py::array::c_style | py::array::forcecast> y,
        const std::string& criterion)
    {       
    //----------------------Input checks
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
    //----------------------DataSet build
                const float* X_ptr = static_cast<const float*>(xb.ptr);
                const float* y_ptr = static_cast<const float*>(yb.ptr);

                std::vector<float> X_vec(X_ptr, X_ptr + n_rows * n_cols);
                std::vector<float> y_vec(y_ptr, y_ptr + n_rows);

                arboria::DataSet data(std::move(X_vec), std::move(y_vec), n_rows, n_cols);


    //----------------------Param Build
                //----------Criterion
                Criterion crit;
                if (criterion == "gini") crit = Gini{};
                else if (criterion == "entropy") crit = Entropy{};
                else throw std::runtime_error("Unknown split criterion passed to fit.");

                //----------Threshold
                ThresholdComputation threshold = CART{};
                //----------Feature
                FeatureSelection feature = AllFeatures{};

                SplitParam param = ParamBuilder(TreeModel::DecisionTree, 
                    crit, 
                    threshold , 
                    feature);
            

                self.fit(data, param);
            },
            
            py::arg("X"), py::arg("y"), py::arg("criterion") = "gini",
            R"doc(
                Fit the decision tree.

                Parameters
                ----------
                X : ndarray of shape (n_samples, n_features)
                    Training input samples.
                y : ndarray of shape (n_samples,)
                    Target labels.
                criterion : {"gini", "entropy"}, default="gini"
                    Splitting criterion used to evaluate candidate splits.

                Returns
                -------
                None
                )doc"
            )

        .def("_predict",
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
        .def(py::init([](std::optional<int> n_estimators,
                        std::optional<int> m_try,
                        std::optional<int> max_depth, 
                        std::optional<float> max_samples,
                        std::optional<int> min_sample_split,
                        std::optional<int> n_jobs,
                        std::optional<std::uint32_t> seed)
                        {        
                        HyperParam hp;
                        hp.n_estimators = n_estimators;
                        hp.mtry = m_try; // value always set during Python init ; must be passed
                        hp.max_samples = max_samples;
                        hp.min_sample_split = min_sample_split;
                        if (max_depth.has_value()) {
                            hp.max_depth= max_depth;}
                        if (n_jobs.has_value()){
                            hp.n_jobs = n_jobs;
                        }
                        else {
                            hp.n_jobs = 1;
                        }

                        return std::make_unique<arboria::RandomForest>(hp, seed);}
                    ),
            py::arg("n_estimators"), 
            py::arg("m_try"),
            py::arg("max_depth") = std::nullopt,
            py::arg("max_samples") = std::nullopt,
            py::arg("min_sample_split") = std::nullopt,
            py::arg("n_jobs") = std::nullopt,
            py::arg("seed") = std::nullopt
    )

        .def("_fit", 
            [](arboria::RandomForest& self, 
            py::array_t<float, py::array::c_style | py::array::forcecast> X,
            py::array_t<float, py::array::c_style | py::array::forcecast> y,
            const std::string& criterion, const int m_try) {
                
//----------------------Input Checks 
                auto xb = X.request();
                if (xb.ndim != 2) {
                    throw std::runtime_error("X must be a 2D numpy array.");
                }
                auto yb = y.request();
                if (yb.ndim != 1) {
                    throw std::runtime_error("y must be a 1D numpy array.");
                }
                
//----------------------Param Build


                //----------Criterion
                Criterion crit;
                if (criterion == "gini") crit = Gini{};
                else if (criterion == "entropy") crit = Entropy{};
                else throw std::runtime_error("Unknown split criterion passed to fit.");

                //----------Threshold
                ThresholdComputation threshold = CART{};
                //----------Feature
                FeatureSelection feature = RandomK{m_try};

                SplitParam param = ParamBuilder(TreeModel::RandomForest, 
                    crit, 
                    threshold , 
                    feature);
            
//----------------------DataSet Build
                const size_t n_rows = static_cast<size_t>(xb.shape[0]);
                const size_t n_cols = static_cast<size_t>(xb.shape[1]);
                if ((size_t)yb.shape[0] != n_rows) {
                    throw std::runtime_error("y length must match X.shape[0].");
                }

                const float* X_ptr = static_cast<float*>(xb.ptr);
                const float* y_ptr = static_cast<float*>(yb.ptr);

                std::vector<float> X_vec(X_ptr, X_ptr + n_rows * n_cols);
                std::vector<float> y_vec(y_ptr, y_ptr + n_rows);

                arboria::DataSet data(std::move(X_vec), std::move(y_vec), n_rows, n_cols);
                self.fit(data, param);
            },
            
            py::arg("X"), py::arg("y"), py::arg("criterion") = "gini", py::arg("m_try")
        )

        .def("_predict", 
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

        .def("_predict_proba",
            [](arboria::RandomForest& self, py::array_t<float, py::array::c_style | py::array::forcecast> X){
                
                auto xb = X.request();
                const size_t ndim = xb.ndim;

                if (ndim == 1){

                    const size_t n_cols = xb.size;
                    const float* x_ptr = static_cast<float*>(xb.ptr);
                    std::vector<float> X_vec(x_ptr, x_ptr + n_cols);
                    return self.predict_proba(X_vec);
                }

                else if (ndim == 2){
                    const size_t n_rows = xb.shape[0];
                    const size_t n_cols = xb.shape[1];

                    const float* x_ptr = static_cast<float*>(xb.ptr);

                    std::vector<float> X_vec(x_ptr, x_ptr+n_cols*n_rows);
                    return self.predict_proba(X_vec);

                }

                else {throw std::runtime_error("RandomForest.predict_proba : invalid dimension on inputs");}



            }
        
        
        
        )

        .def("_out_of_bag",
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
        
        )

        .def("_get_max_samples",
            [](const arboria::RandomForest& rf) -> py::object {
            auto d = rf.get_max_samples();
            if (d.has_value()) return py::float_(static_cast<double>(*d));
            return py::none();
        }
        );


        m.def("_accuracy", 
        [](py::array_t<int, py::array::c_style | py::array::forcecast> y_true,
        py::array_t<int, py::array::c_style | py::array::forcecast> y_pred){

            if (y_true.ndim() != 1 || y_pred.ndim() != 1) throw std::invalid_argument("accuracy : passed argument must be a 1D array");
            auto a = y_true.unchecked<1>();
            auto b = y_pred.unchecked<1>();

            return arboria::helpers::accuracy(
                std::span<const int>(a.data(0), a.shape(0)),
                std::span<const int>(b.data(0), b.shape(0))
            );


        }
        
        
        );


        

    }