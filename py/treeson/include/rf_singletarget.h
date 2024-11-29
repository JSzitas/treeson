#ifndef TREESON_RF_RANDOM_SINGLETARG_H
#define TREESON_RF_RANDOM_SINGLETARG_H

#include <iostream>
#include <vector>
#include <random>
#include <variant>

#include "treeson.h"

template<typename scalar_t, typename integral_t,
          const size_t max_categorical_size = 32> class SingletargetRandomForest {
  using DataType = std::vector<std::variant<std::vector<integral_t>, std::vector<scalar_t>>>;
  struct WelfordMean{
    scalar_t mean;
    size_t obs;
    explicit WelfordMean() : mean(0.0), obs(1){}
    void operator()(const std::vector<scalar_t>& x,
                    const std::vector<size_t>& indices,
                    const size_t start,
                    const size_t end) {
      // make sure to only update index once other means have been updated
      for (size_t k = start; k < end; k++) {
        mean += (x[indices[k]] - mean)/obs++;
      }
    }
    void operator()(const scalar_t x) {
      // make sure to only update index once other means have been updated
        mean += (x - mean)/obs++;
    }
    [[maybe_unused]] void merge(const std::array<scalar_t,2>& x) {
      const auto wt = obs;
      const auto new_wt = x[1];
      mean = ((mean * wt) + (x[0] * new_wt)) / (wt + new_wt);
      obs += x[1];
    }
    [[nodiscard]] std::array<scalar_t, 2> result() const {
      return {mean, static_cast<scalar_t>(obs)};
    }
  };
  struct [[maybe_unused]] resultFuncMean{
    const size_t target;
    explicit resultFuncMean(const size_t target_index) : target(target_index) {}
    std::array<scalar_t,2> operator()(
        std::vector<size_t>& indices,
        size_t start,
        size_t end,
        [[maybe_unused]] const DataType& data) {
      // last result actually holds the index
      WelfordMean result;
      result(std::get<std::vector<scalar_t>>(data[target]), indices, start, end);
      const auto res = result.result();
      return res;
    };
  };
  struct MeanAccumulator{
    std::vector<WelfordMean> result;
    explicit MeanAccumulator(const size_t pred_size) : result(std::vector<WelfordMean>(pred_size, WelfordMean())) {}
    void operator()(const treeson::containers::TreePredictionResult<std::array<scalar_t, 2>>& x) {
      const size_t n_results = x.result_size();
      const auto& indices = x.indices;
      // iterate over results and write them
      for(size_t k = 0; k < n_results; k++) {
        const auto [start, end, values] = x.get_result_view(k);
        for(size_t p = start; p < end; p++) {
          result[indices[p]].merge(values);
        }
      }
    }
    [[maybe_unused]] [[nodiscard]] auto flat_results() const {
      std::vector<scalar_t> res;
      const size_t n_preds = result.size();
      for(size_t k = 0; k < n_preds; k++) {
        res.push_back(result[k].mean);
      }
      return res;
    }
    [[nodiscard]] auto results() const {
      const size_t n_preds = result.size();
      std::vector<scalar_t> res(n_preds);
      for(size_t k = 0; k < n_preds; k++) {
        res[k] = result[k].mean;
      }
      return res;
    }
  };
  // this has a hack to resize itself; it is a bit sad, but it should be sufficient
  // a lambda could be an alternative solution if written correctly
  using strat = treeson::splitters::ExtremelyRandomizedStrategy<std::mt19937, scalar_t, integral_t, max_categorical_size>;
  using Forest = treeson::RandomForest<resultFuncMean, std::mt19937, strat,
                                       max_categorical_size, scalar_t, integral_t>;
  const size_t target;
  resultFuncMean res_functor;
  std::mt19937 rng;
  strat strat_obj;
  Forest forest_;
public:
  SingletargetRandomForest(const size_t target, const size_t max_depth,
                             const size_t min_nodesize,
                             const int seed = 42
                          ) : target(target),
                              res_functor(resultFuncMean(target)),
                              rng(std::mt19937(seed)),
                              strat_obj(strat{}),
                              forest_(Forest(max_depth, min_nodesize, rng, res_functor, strat_obj)) {}
  auto memoryless_predict(const DataType& train_data,
                          const DataType& test_data,
                          const size_t n_tree,
                          const bool resample = true,
                          const size_t subsample_size = 1024){
    MeanAccumulator acc(treeson::utils::size(test_data[0]));
    forest_.memoryless_predict(acc, train_data, test_data, n_tree, {target},
                               resample, subsample_size);
    return acc.results();
  }
  auto predict(const DataType& test_data,
               const size_t num_threads = 0){
    return forest_.predict(test_data, num_threads);
  }
  [[maybe_unused]] void fit(const DataType &data,
                            const size_t n_tree,
                            const bool resample = true,
                            const size_t sample_size = 0,
                            const size_t num_threads = 0) {
    forest_.fit(data, n_tree, {target}, resample, sample_size, num_threads);
  }
  [[maybe_unused]] void fit_to_file(const DataType &data,
                                    const size_t n_tree,
                                    const std::string &file,
                                    const bool resample = true,
                                    const size_t sample_size = 0,
                                    const size_t num_threads = 0) {
    forest_.fit(data, n_tree, {target}, file, resample, sample_size, num_threads);
  }
  [[maybe_unused]] auto predict_from_file(const DataType &samples,
                                          const std::string &model_file,
                                          const size_t num_threads = 0) {
    MeanAccumulator acc(treeson::utils::size(samples[0]));
    forest_.predict(acc, samples, model_file, num_threads);
    return acc.results();
  }
};

#endif // TREESON_RF_RANDOM_SINGLETARG_H
