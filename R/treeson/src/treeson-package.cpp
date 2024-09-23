#include <Rcpp.h>
#include <random>
#include <vector>
#include <variant>
#include "../../../treeson.h"
#include "../../../reducers.h" // Include your custom reducers

using namespace Rcpp;

// Define the specific SplitStrategy
using SplitStrategy = treeson::splitters::ExtremelyRandomizedStrategy<std::mt19937, float, 32>;
using DataType = std::vector<std::variant<std::vector<size_t>, std::vector<float>>>;

// Define the specific ResultF for IntersectionSampler
// struct resultFuncIntersection {
//   std::vector<size_t> operator()(std::vector<size_t>& indices, size_t start, size_t end, const DataType& data) {
//     std::vector<size_t> result(end - start);
//     for (size_t i = start; i < end; i++) {
//       result[i - start] = indices[i];
//     }
//     return result;
//   };
// };

// Define the specific ResultF for MultitargetMeanReducer
template<typename scalar_t>
struct resultFuncMultitargetMean {
  std::vector<size_t> targets;
  explicit resultFuncMultitargetMean(const std::vector<size_t>& target_indices) : targets(target_indices) {}

  std::vector<scalar_t> operator()(std::vector<size_t>& indices, size_t start, size_t end, const DataType& data) {
    std::vector<scalar_t> result((end - start) * targets.size());
    size_t p = 0;
    for (const auto index : targets) {
      const auto & curr = std::visit([](auto&& arg) -> auto {
        return arg;
      }, data[index]);
      for (size_t i = start; i < end; i++) {
        result[p++] = curr[indices[i]];
      }
    }
    return result;
  };
};

// MultitargetMeanReducerWrapper class
// class MultitargetMeanReducerWrapper {
// public:
//   MultitargetMeanReducer<float, std::vector<float>> reducer;
//
//   MultitargetMeanReducerWrapper(size_t n_targets) : reducer(n_targets) {}
//
//   void add(const std::vector<float>& predictions) {
//     reducer(predictions);
//   }
//
//   std::vector<float> get_result() const {
//     return reducer.result();
//   }
// };

// Wrapper class for IntersectionSampler implementation
// class IntersectionSamplerWrapper {
// public:
//   IntersectionSamplerWrapper() {}
//
//   std::vector<size_t> sample_trees(const std::vector<std::vector<size_t>>& predictions_forest,
//                                    const size_t num_trees, std::mt19937& rng) {
//     return IntersectionSampler<size_t>::sample_trees(predictions_forest, num_trees, rng);
//   }
//
//   std::unordered_set<size_t> compute_intersections(const std::vector<size_t>& sampled_trees, size_t threshold) {
//     return IntersectionSampler<size_t>::compute_intersections(sampled_trees, threshold);
//   }
// };

// RandomForest and RandomTree wrapper classes
template <typename Accumulator, typename ResultF>
class RandomForestWrapper {
public:
  using RandomForestType = treeson::RandomForest<ResultF, std::mt19937, SplitStrategy>;

  RandomForestType rf;

  RandomForestWrapper(size_t max_depth, size_t n_targets, size_t min_node_size)
    : rf(max_depth, min_node_size, std::mt19937(), ResultF(n_targets), SplitStrategy()) {

  }

  void memoryless_predict(const DataType & train_data,
                          const DataType & predict_data,
                          Accumulator& acc, size_t n_tree,
                          const std::vector<size_t>& nosplit_features,
                          bool resample, size_t sample_size) {
    rf.memoryless_predict(train_data, predict_data, acc, n_tree, nosplit_features, resample, sample_size);
  }
  /*
  std::vector<std::vector<float>> feature_importance(const DataType& train_data,
                                                     const DataType& predict_data,
                                                     const size_t n_tree,
                                                    const std::vector<size_t>& nosplit_features,
                                                    bool resample, size_t sample_size,
                                                    Accumulator& acc,
                                                    ImportanceCalculator importance) {
    return rf.feature_importance(train_data, predict_data, n_tree, nosplit_features, resample, sample_size, acc, importance);
                                                    }*/
};

// template <typename ResultF>
// class RandomTreeWrapper {
// public:
//   using RandomTreeType = treeson::RandomTree<ResultF, std::mt19937, SplitStrategy>;
//
//   RandomTreeType rt;
//
//   RandomTreeWrapper(size_t max_depth, size_t min_node_size, std::mt19937 rng, ResultF result_func)
//     : rt(max_depth, min_node_size, rng, result_func, SplitStrategy()) {}
//
//   void fit(const DataType & data) {
//     rt.fit(data);
//   }
//
//   std::vector<float> predict(const DataType& data) {
//     return rt.template predict<true>(data);
//   }
//
//   std::vector<size_t> used_features() {
//     return rt.used_features();
//   }
// };

// Register module
RCPP_MODULE(treeson_module) {
  // class_<MultitargetMeanReducerWrapper>("MultitargetMeanReducer")
  // .constructor<size_t>()
  // .method("add", &MultitargetMeanReducerWrapper::add)
  // .method("get_result", &MultitargetMeanReducerWrapper::get_result);

  // class_<IntersectionSamplerWrapper>("IntersectionSampler")
  //   .constructor<>()
  //   .method("sample_trees", &IntersectionSamplerWrapper::sample_trees)
  //   .method("compute_intersections", &IntersectionSamplerWrapper::compute_intersections);

  class_<RandomForestWrapper<MultitargetMeanReducer<float>, resultFuncMultitargetMean<float>>>("RandomForest")
    .constructor<size_t, size_t, std::mt19937, resultFuncMultitargetMean<float>>()
    .method("memoryless_predict", &RandomForestWrapper<MultitargetMeanReducer<float>, resultFuncMultitargetMean<float>>::memoryless_predict);
    // .method("feature_importance", &RandomForestWrapper<MultitargetMeanReducer<float>, resultFuncMultitargetMean<float>>::feature_importance);

  // class_<RandomTreeWrapper<resultFuncMultitargetMean<float>>>("RandomTree")
  //   .constructor<size_t, size_t, std::mt19937, resultFuncMultitargetMean<float>>()
  //   .method("fit", &RandomTreeWrapper<resultFuncMultitargetMean<float>>::fit)
  //   .method("predict", &RandomTreeWrapper<resultFuncMultitargetMean<float>>::predict)
  //   .method("used_features", &RandomTreeWrapper<resultFuncMultitargetMean<float>>::used_features);
}
