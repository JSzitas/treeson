#ifndef TREESON_RF_MULTITARGET_H
#define TREESON_RF_MULTITARGET_H

#include <iostream>
#include <vector>
#include <random>
#include <variant>

#include "treeson.h"

template<typename scalar_t, typename integral_t,
    const size_t max_categorical_size = 32> class MultitargetRandomForest {
  using DataType = std::vector<std::variant<std::vector<integral_t>, std::vector<scalar_t>>>;
  struct WelfordMultiMean{
    std::vector<scalar_t> means;
    explicit WelfordMultiMean(const size_t n_targets, const bool init_index = false)
        : means(std::vector<scalar_t>(n_targets+1, 0.0)){
      means.back() = static_cast<scalar_t>(init_index);
    }
    void operator()(const std::vector<scalar_t>& x,
                    const std::vector<size_t>& indices,
                    const size_t start,
                    const size_t end,
                    const size_t mean_id = 0,
                    const bool update_index = false) {
      // make sure to only update index once other means have been updated
      if(!update_index) {
        auto p = means.back();
        for (size_t k = start; k < end; k++) {
          means[mean_id] += (x[indices[k]] - means[mean_id])/p;
          p = p + 1.0f;
        }
      }
      else {
        for (size_t k = start; k < end; k++) {
          means[mean_id] += (x[indices[k]] - means[mean_id])/means.back();
          means.back() += 1.0f;
        }
      }
    }
    void operator()(const std::vector<scalar_t>& x) {
      // make sure to only update index once other means have been updated
      for (size_t k = 0; k < x.size(); k++) {
        means[k] += (x[k] - means[k])/means.back();
      }
      means.back() += 1.0f;
    }
    [[maybe_unused]] void merge(const std::vector<scalar_t>& x) {
      const auto wt = means.back();
      const auto new_wt = x.back();
      for(size_t k = 0; k < x.size()-1; k++) {
        means[k] = ((means[k] * wt) + (x[k] * new_wt)) / (wt + new_wt);
      }
      means.back() = (wt + new_wt);
    }
    [[nodiscard]] const std::vector<scalar_t>& result() const {
      return means;
    }
  };
  struct [[maybe_unused]] resultFuncMultitargetMean{
    std::vector<size_t> targets;
    explicit resultFuncMultitargetMean(
        const std::vector<size_t>& target_indices) :
                                                     targets(target_indices) {}
    std::vector<scalar_t> operator()(
        std::vector<size_t>& indices,
        size_t start,
        size_t end,
        [[maybe_unused]] const DataType& data) {
      // last result actually holds the index
      WelfordMultiMean result(targets.size(), true);
      size_t target_index = 0;
      for(const auto index : targets) {
        const auto& target = std::get<std::vector<scalar_t>>(data[index]);
        result(target, indices, start, end, target_index,
               target_index == targets.size()-1);
        target_index++;
      }
      return result.means;
    };
  };
  struct [[maybe_unused]] MeanSquaredError {
    std::vector<size_t> targets;
    explicit MeanSquaredError(const std::vector<size_t>& targets) : targets(targets) {}
    std::vector<scalar_t> operator()(const std::vector<scalar_t>& vals, const DataType &samples,
                                     const std::vector<size_t>& indices,
                                     const size_t start, const size_t end) const {
      std::vector<scalar_t> results(targets.size());
      size_t k = 0;
      for(const auto& target: targets) {
        const auto& target_ = std::get<std::vector<scalar_t>>(samples[target]);
        size_t n = 1;
        scalar_t curr_err = 0.;
        const auto mean_ = vals[targets[k]];
        for(size_t i = start; i < end; i++) {
          const auto temp = std::pow(mean_ - target_[indices[i]],2.f);
          curr_err += (temp - curr_err)/static_cast<scalar_t>(n++);
        }
        results[k++] = curr_err;
      }
      return results;
    }
  };
  struct DifferenceImportance{
    std::vector<scalar_t> operator()(const std::vector<scalar_t>& baseline,
                                     const std::vector<scalar_t>& feature) {
      std::vector<scalar_t> result(baseline.size()-1);
      for(size_t i = 0; i < baseline.size()-1; i++) {
        result[i] = 1 - (std::sqrt(baseline[i])/std::sqrt(feature[i]));
      }
      return result;
    }
  };
  struct MultitargetMeanAccumulator{
    std::vector<WelfordMultiMean> result;
    explicit MultitargetMeanAccumulator(const size_t target_size,
                                        const size_t pred_size) :
                                                                  result(std::vector<WelfordMultiMean>(pred_size, WelfordMultiMean(target_size))) {}
    void operator()(const treeson::containers::TreePredictionResult<std::vector<scalar_t>>& x) {
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
      const size_t n_targets = result.front().means.size()-1;
      const size_t n_preds = result.size();
      for(size_t j = 0; j < n_targets; j++) {
        for(size_t k = 0; k < n_preds; k++) {
          res.push_back(result[k].means[j]);
        }
      }
      return res;
    }
    [[nodiscard]] auto results() const {
      const size_t n_targets = result.front().means.size()-1;
      const size_t n_preds = result.size();
      std::vector<std::vector<scalar_t>> res(n_preds, std::vector<scalar_t>(n_targets));
      for(size_t k = 0; k < n_preds; k++) {
        auto& temp = res[k];
        for(size_t j = 0; j < n_targets; j++) {
          temp[j] = result[k].means[j];
        }
      }
      return res;
    }
    [[maybe_unused]] [[nodiscard]] auto one_result(const size_t which_target) const {
      std::vector<scalar_t> res(result.size());
      const size_t n_preds = result.size();
      for(size_t k = 0; k < n_preds; k++) {
        res[k] = result[k].means[which_target];
      }
      return res;
    }
  };
  // this has a hack to resize itself; it is a bit sad, but it should be sufficient
  // a lambda could be an alternative solution if written correctly
  struct MultiMeanWelfordAcc{
    WelfordMultiMean acc;
    bool first = true;
    MultiMeanWelfordAcc() : acc(WelfordMultiMean(0, true)) {}
    void operator()(const std::vector<scalar_t>& x) {
      if(first) {
        acc = WelfordMultiMean(x.size(), true);
        acc(x);
        first = false;
        return;
      }
      acc(x);
    }
    [[nodiscard]] const std::vector<scalar_t>& result() const {
      return acc.result();
    }
  };
  using strat = treeson::splitters::ExtremelyRandomizedStrategy<std::mt19937, scalar_t, integral_t, max_categorical_size>;
  using Forest = treeson::RandomForest<resultFuncMultitargetMean, std::mt19937, strat,
                                       max_categorical_size, scalar_t, integral_t>;
  std::vector<size_t> targets_;
  resultFuncMultitargetMean res_functor;
  strat strat_obj;
  std::mt19937 rng;
  Forest forest_;
public:
  MultitargetRandomForest(const std::vector<size_t>& targets, const size_t max_depth,
                          const size_t min_nodesize, const int seed = 42
                          ) : targets_(targets),
                              res_functor(resultFuncMultitargetMean(targets)),
        strat_obj(strat{}), rng(std::mt19937(seed)),
        forest_(Forest(max_depth, min_nodesize, rng, res_functor, strat_obj)) {}
    auto feature_importance(DataType train_data, const size_t n_tree,
                            const bool resample, const size_t subsample_size) {
      MeanSquaredError metric(targets_);
      return forest_.template feature_importance<
          MultiMeanWelfordAcc, MeanSquaredError, DifferenceImportance,
          Forest::ImportanceMethod::Omit>(metric,
          train_data, n_tree, targets_, resample, subsample_size);
    }
    auto memoryless_predict(const DataType& train_data,
                            const DataType& test_data,
                            const size_t n_tree,
                            const bool resample = true,
                            const size_t subsample_size = 1024){

    MultitargetMeanAccumulator acc(
        targets_.size(), treeson::utils::size(test_data[0]));
    forest_.memoryless_predict(acc, train_data, test_data, n_tree, targets_,
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
      forest_.fit(data, n_tree, targets_, resample, sample_size, num_threads);
    }
    [[maybe_unused]] void fit_to_file(const DataType &data,
                                      const size_t n_tree,
                                      const std::string &file,
                                      const bool resample = true,
                                      const size_t sample_size = 0,
                                      const size_t num_threads = 0) {
      forest_.fit(data, n_tree, targets_, file, resample, sample_size, num_threads);
    }
    [[maybe_unused]] auto predict_from_file(const DataType &samples,
                                            const std::string &model_file,
                                            const size_t num_threads = 0) {
      MultitargetMeanAccumulator acc(
          targets_.size(), treeson::utils::size(samples[0]));
      forest_.predict(acc, samples, model_file, num_threads);
      return acc.results();
    }
};
#endif // TREESON_RF_MULTITARGET_H
