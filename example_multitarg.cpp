#include <iostream>
#include <vector>
#include <random>
#include <variant>

#include "treeson.h"
#include "stopwatch.h"
#include "utils.h"

#include "parquet.h"

int main() {
  
  using scalar_t = float;
  
  using DataType = std::vector<std::variant<std::vector<int>, std::vector<scalar_t>>>;
    /*const std::string file_path = "test_data/train.parquet";
    try {
      auto data = load_parquet_data<int, scalar_t>(file_path); // Replace 'int, scalar_t' with appropriate types as needed
      print_data(data);
    }
    catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return 1;
    }

  return 0;*/

  // RNG and Result function
  std::mt19937 rng(22); // NOLINT(*-msc51-cpp)
  std::cout << "Loading train: ";
  auto train_data =
      load_parquet_data<int, scalar_t>("../test_data/train.parquet");
      //CSVLoader<int, scalar_t>::load_data<true, false>("test_data/dr_train.csv", 10000);
  std::cout << "Done." << std::endl;
  std::cout << "Loading test: ";
  auto test_data =
      load_parquet_data<int, scalar_t>("../test_data/test.parquet");
      //CSVLoader<int, scalar_t>::load_data<true, false>("test_data/dr_test.csv", 10000);
  std::cout << "Done." << std::endl;

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
  using strat = treeson::splitters::ExtremelyRandomizedStrategy<std::mt19937, scalar_t, int, 0>;
  auto res_functor = resultFuncMultitargetMean(std::vector<size_t>{0,1,2,3});
  auto strat_obj = strat{};
  // original max depth set to 10
  treeson::RandomForest<decltype(res_functor), std::mt19937, strat,
                        0, scalar_t, int> forest(6, 12, rng, res_functor, strat_obj);

  struct
      [[maybe_unused]] MeanSquaredError {
    std::vector<size_t> targets = {0,1,2,3};
    std::vector<scalar_t> operator()(const std::vector<scalar_t>& vals, const DataType &samples,
                      const std::vector<size_t>& indices,
                      const size_t start, const size_t end) const {
      std::vector<scalar_t> results(targets.size());
      size_t k = targets[0];
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

  //Metric, typename Reducer = treeson::reducers::defaultImportanceReducer<scalar_t>>
  //                           [[maybe_unused]] auto eval_oob
  /*
  treeson::RandomTree<resultFuncMultitargetMean, std::mt19937, strat,
                       32, scalar_t, int> tree(4, 100, rng, res_functor, strat_obj);
  std::vector<size_t> indices = treeson::utils::make_index_range(std::visit(
      [](auto &&arg) -> size_t { return arg.size(); }, train_data[0]));
  auto [train, oob] = treeson::utils::bootstrap_two_samples(indices, 1000, rng);

  tree.fit(train_data, train, {0,1,2,3});
  tree.print_terminal_node_values();
  treeson::utils::print_set(tree.used_features());
  const auto importance = tree.eval_oob<MeanSquaredError>(train_data, oob);
  treeson::utils::print_vector(importance);
  for(const auto feat: tree.used_features()) {
    const auto importance_wo =
        tree.eval_oob<MeanSquaredError>(train_data, oob, feat);
    treeson::utils::print_vector(importance_wo);
  }

  return 0;
  */
  /*
  for(const auto& n_tree: {200}){//100, 500, 1'000, 2'000, 5'000, 10'000, 50'000, 100'000}){
    struct MultiMeanWelfordAcc{
      WelfordMultiMean acc;
      MultiMeanWelfordAcc() : acc(WelfordMultiMean(4, true)) {}
      void operator()(const std::vector<scalar_t>& x) {
        acc(x);
      }
      [[nodiscard]] const std::vector<scalar_t>& result() const {
        return acc.result();
      }
    };
    struct DifferenceImportance{
        std::vector<scalar_t> operator()(const std::vector<scalar_t>& baseline,
                                       const std::vector<scalar_t>& feature) {
          std::vector<scalar_t> result(baseline.size());
          for(size_t i = 0; i < 4; i++) {
            result[i] = 1 - (std::sqrt(baseline[i])/std::sqrt(feature[i]));
          }
          return result;
        }
    };
    Stopwatch sw;
    const std::vector<size_t> targets = {0,1,2,3};
    std::cout << "Running feature importances, n_tree: " << n_tree << std::endl;
    const auto importances = forest.feature_importance<MultiMeanWelfordAcc, MeanSquaredError, DifferenceImportance>(train_data, n_tree,//'000,
                              targets, true, 1024);
    std::cout << "Importances: "<< std::endl;
    for(const auto& val : importances) {
      treeson::utils::print_vector(val);
    }
    CSVWriter<int, scalar_t>::write_data<false, false>("feature_importance_max_depth_4_n_tree_" +
                                                    std::to_string(n_tree) + ".csv",
                                                    {}, {},
                                                    importances
                                                    );
  }*/


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
    [[nodiscard]] auto one_result(const size_t which_target) const {
      std::vector<scalar_t> res(result.size());
      const size_t n_preds = result.size();
        for(size_t k = 0; k < n_preds; k++) {
          res[k] = result[k].means[which_target];
      }
      return res;
    }
   };
  for(const size_t n_tree : {//100, 500, 1'000, 2'000, 5'000, 10'000, 50'000,
            10'000}){
    {
      MultitargetMeanAccumulator acc(
          4, std::get<std::vector<scalar_t>>(test_data[0]).size());
      Stopwatch sw;
      const std::vector<size_t> targets = {0,1,2,3};
      std::cout << "N trees: "<< n_tree << std::endl;
      forest.memoryless_predict(acc, train_data, test_data, n_tree//'000'000
                                , targets, true, 1024);
      std::cout << "Computing metrics: "<< std::endl;
      for(const auto target: targets) {
        const auto res = acc.one_result(target);

        std::cout << "Target: " << target+1 << std::endl;
        std::cout << " | Spearman: " <<
            treeson::utils::spearman_correlation(
                res, std::get<std::vector<scalar_t>>(test_data[target])
                    ) <<
            " Pearson: "
                  <<
            treeson::utils::pearson_correlation(
                res, std::get<std::vector<scalar_t>>(test_data[target])
                    ) <<
                    " rmse: " << treeson::utils::rmse(
                                      res, std::get<std::vector<scalar_t>>(test_data[target])
                                          ) << std::endl;
      }
      std::cout << "Writing predictions" << std::endl;
      CSVWriter<int, scalar_t>::write_data<true, false>("predictions.csv",
                                                         {"target_w", "target_r", "target_g", "target_b"}, {},
                                                         acc.results());
    }
  }
  return 0;
}
