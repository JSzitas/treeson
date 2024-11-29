#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rf_multitarget.h"
#include "rf_singletarget.h"

namespace py = pybind11;

using scalar_t = float;
using integral_t = size_t;
using MultitargetRandomForest_t = MultitargetRandomForest<scalar_t, integral_t>;
using SingletargetRandomForest_t = SingletargetRandomForest<scalar_t, integral_t>;

PYBIND11_MODULE(treeson, m) {
  py::class_<MultitargetRandomForest_t>(m, "MultitargetRandomForest")
  .def(py::init<const std::vector<size_t>&, size_t, size_t, int>(),
  py::arg("targets"), py::arg("max_depth"), py::arg("min_nodesize"), py::arg("seed") = 42)

  .def("fit", &MultitargetRandomForest_t::fit, py::arg("data"), py::arg("n_tree"),
  py::arg("resample") = true, py::arg("sample_size") = 0, py::arg("num_threads") = 0)

  .def("predict", &MultitargetRandomForest_t::predict, py::arg("samples"), py::arg("num_threads") = 0)

  .def("feature_importance", &MultitargetRandomForest_t::feature_importance, py::arg("train_data"), py::arg("n_tree"),
  py::arg("resample"), py::arg("subsample_size"))

  .def("memoryless_predict", &MultitargetRandomForest_t::memoryless_predict, py::arg("train_data"), py::arg("test_data"),
  py::arg("n_tree"), py::arg("resample") = true, py::arg("subsample_size") = 1024)

  .def("fit_to_file", &MultitargetRandomForest_t::fit_to_file, py::arg("train_data"),
  py::arg("n_tree"), py::arg("file"), py::arg("resample") = true, py::arg("subsample_size") = 1024, py::arg("num_threads") = 0)

  .def("predict_from_file", &MultitargetRandomForest_t::predict_from_file, py::arg("samples"),
  py::arg("model_file"), py::arg("num_threads") = 04);

  py::class_<SingletargetRandomForest_t>(m, "SingletargetRandomForest")
    .def(py::init<const size_t&, size_t, size_t, int>(),
  py::arg("target"), py::arg("max_depth"), py::arg("min_nodesize"), py::arg("seed") = 42)

    .def("fit", &SingletargetRandomForest_t::fit, py::arg("data"), py::arg("n_tree"),
    py::arg("resample") = true, py::arg("sample_size") = 0, py::arg("num_threads") = 0)
    .def("predict", &SingletargetRandomForest_t::predict, py::arg("samples"), py::arg("num_threads") = 0)

    .def("memoryless_predict", &SingletargetRandomForest_t::memoryless_predict, py::arg("train_data"), py::arg("test_data"),
    py::arg("n_tree"), py::arg("resample") = true, py::arg("subsample_size") = 1024)
    .def("fit_to_file", &SingletargetRandomForest_t::fit_to_file, py::arg("train_data"),
    py::arg("n_tree"), py::arg("file"), py::arg("resample") = true, py::arg("subsample_size") = 1024, py::arg("num_threads") = 0)

    .def("predict_from_file", &SingletargetRandomForest_t::predict_from_file, py::arg("samples"),
    py::arg("model_file"), py::arg("num_threads") = 04);

}
