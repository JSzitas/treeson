#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "RandomForest.h"
#include "CustomTypes.h"  // Include your custom types as needed

namespace py = pybind11;

// Assuming CustomAccumulator and CustomImportance are your custom types
using RNG = std::mt19937; // Replace with actual RNG type
using ResultF = std::function<double()>; // Replace with actual ResultF type
using SplitStrategy = SomeSplitStrategy; // Replace with your split strategy type

PYBIND11_MODULE(my_module, m) {
  py::class_<RandomForest<ResultF, RNG, SplitStrategy>>(m, "RandomForest")
      .def(py::init<size_t, size_t, RNG, ResultF, SplitStrategy>())
      .def("memoryless_predict", &RandomForest<ResultF, RNG, SplitStrategy>::template memoryless_predict<CustomAccumulator>)
      .def("feature_importance", &RandomForest<ResultF, RNG, SplitStrategy>::template feature_importance<CustomAccumulator, CustomImportance>);

  py::class_<RandomTree<ResultF, RNG, SplitStrategy>>(m, "RandomTree")
      .def(py::init<size_t, size_t, RNG, ResultF, SplitStrategy>())
      .def("fit", &RandomTree<ResultF, RNG, SplitStrategy>::fit)
      .def("predict", &RandomTree<ResultF, RNG, SplitStrategy>::template predict<true>)
      .def("used_features", &RandomTree<ResultF, RNG, SplitStrategy>::used_features);

  // Expose your custom types as needed
  py::class_<CustomAccumulator>(m, "CustomAccumulator")
      .def(py::init<>());

  py::class_<CustomImportance>(m, "CustomImportance")
      .def(py::init<>());
}