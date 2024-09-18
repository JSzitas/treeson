#include <Rcpp.h>
#include <random>
#include <vector>
#include "../../../treeson.h"

using namespace Rcpp;
using namespace treeson;

struct resultFunc{
  std::vector<size_t> operator()(
      std::vector<size_t>& indices,
      size_t start,
      size_t end,
      [[maybe_unused]] const DataType& data) {
    std::vector<size_t> result(end-start);
    for(size_t i = start; i < end; i++) {
      result[i-start] = indices[i];
    }
    return result;
  };
};
// Create a template specialization for RandomForest with the required types
using ResultFunc = resultFunc;  // Custom result function as defined previously
using RNG = std::mt19937;       // Random number generator
using Splitter = splitters::ExtremelyRandomizedStrategy<RNG, float, 32>;  // Splitter strategy

// Define a specific instantiation of RandomForest
using RandomForestSpec = RandomForest<ResultFunc, RNG, Splitter, 32, float>;

// Specialized constructor function (more parameters can be added as needed)
RandomForestSpec create_randomforest(size_t num_trees, size_t max_depth, int seed, ResultFunc result_function, Splitter splitter) {
  RNG rng(seed);
  return RandomForestSpec(num_trees, max_depth, rng, result_function, splitter);
}

// Rcpp Module for exposing the RandomForest class and methods
RCPP_MODULE(RandomForestModule) {
  class_<RandomForestSpec>("RandomForest")
  .constructor(&create_randomforest)
  .method("fit", &RandomForestSpec::fit)
  .method("predict", &RandomForestSpec::predict)
  .method("online_predict",
  [](RandomForestSpec& self, const std::vector<float>& data, size_t max_depth,
     const std::vector<size_t>& sample_weights, bool bootstrap, size_t random_state, Rcpp::Function reducer) {

    // Convert R reducer function to C++ lambda
    auto lambda_reducer = [&reducer](auto accumulated, auto next) {
      auto r_result = reducer(accumulated, next);
      return as<decltype(accumulated)>(r_result);
    };

    auto prediction = self.online_predict(data, max_depth, sample_weights, bootstrap, random_state, lambda_reducer);
    return Rcpp::wrap(prediction);
  });
}

