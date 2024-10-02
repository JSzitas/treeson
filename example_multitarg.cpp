#include <iostream>
#include <vector>
#include <random>
#include <variant>

#include "treeson.h"
#include "stopwatch.h"
//#include "utils.h"

#include "parquet.h"

int main() {
  using DataType = std::vector<std::variant<std::vector<int>, std::vector<double>>>;
    /*const std::string file_path = "test_data/train.parquet";
    try {
      auto data = load_parquet_data<int, double>(file_path); // Replace 'int, double' with appropriate types as needed
      print_data(data);
    }
    catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return 1;
    }

  return 0;*/

  // RNG and Result function
  std::mt19937 rng(22); // NOLINT(*-msc51-cpp)
  std::cout << "Loading train: " << std::endl;
  auto train_data =
      load_parquet_data<int, double>("test_data/train.parquet");
      //CSVLoader<int, double>()
      //.load_data<true, false>("test_data/dr_train.csv", 50000);
  std::cout << "Done." << std::endl;
  std::cout << "Loading test: " << std::endl;
  auto test_data =
      load_parquet_data<int, double>("test_data/train.parquet");
      //CSVLoader<int, double>()
      //.load_data<true, false>("test_data/dr_test.csv", 10000);
  std::cout << "Done." << std::endl;
  struct WelfordMultiMean{
    std::vector<double> means;
    explicit WelfordMultiMean(const size_t n_targets, const bool init_index = false)
        : means(std::vector<double>(n_targets+1, 0.0)){
      means.back() = static_cast<double>(init_index);
    }
    void operator()(const std::vector<double>& x,
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
          p = p + 1.0;
        }
      }
      else {
        for (size_t k = start; k < end; k++) {
          means[mean_id] += (x[indices[k]] - means[mean_id])/means.back();
          means.back() += 1.0;
        }
      }
    }
    void merge(const std::vector<double>& x) {
      const auto wt = means.back();
      const auto new_wt = x.back();
      for(size_t k = 0; k < x.size()-1; k++) {
        means[k] = ((means[k] * wt) + (x[k] * new_wt)) / (wt + new_wt);
      }
      means.back() = (wt + new_wt);
    }
  };

  struct [[maybe_unused]] resultFuncMultitargetMean{
    std::vector<size_t> targets;
    explicit resultFuncMultitargetMean(
        const std::vector<size_t>& target_indices) :
          targets(target_indices) {}
    std::vector<double> operator()(
        std::vector<size_t>& indices,
        size_t start,
        size_t end,
        [[maybe_unused]] const DataType& data) {
      // last result actually holds the index
      WelfordMultiMean result(targets.size(), true);
      size_t target_index = 0;
      for(const auto index : targets) {
        const auto& target = std::get<std::vector<double>>(data[index]);
          result(target, indices, start, end, target_index,
                 target_index == targets.size()-1);
          target_index++;
      }
      return result.means;
    };
  };
  using strat = treeson::splitters::ExtremelyRandomizedStrategy<std::mt19937, double, int, 32>;
  auto res_functor = resultFuncMultitargetMean(std::vector<size_t>{0,1,2,3});
  auto strat_obj = strat{};
  treeson::RandomForest<decltype(res_functor), std::mt19937, strat,
                        32, double, int> forest(10, 12, rng, res_functor, strat_obj);

  struct MultitargetMeanAccumulator{
    std::vector<WelfordMultiMean> result;
    explicit MultitargetMeanAccumulator(const size_t target_size,
                                        const size_t pred_size) :
      result(std::vector<WelfordMultiMean>(pred_size, WelfordMultiMean(target_size))) {}
    void operator()(const treeson::containers::TreePredictionResult<std::vector<double>>& x) {
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
    [[nodiscard]] auto results() const {
      std::vector<double> res;
      const size_t n_targets = result.front().means.size()-1;
      const size_t n_preds = result.size();
      for(size_t j = 0; j < n_targets; j++) {
        for(size_t k = 0; k < n_preds; k++) {
          res.push_back(result[k].means[j]);
        }
      }
      return res;
    }
    [[nodiscard]] auto one_result(const size_t which_target) const {
      const size_t n_preds = result.size();
      std::vector<double> res(n_preds);
        for(size_t k = 0; k < n_preds; k++) {
          res[k] = result[k].means[which_target];
      }
      return res;
    }
  };

  {
    static const auto test_size = std::get<std::vector<double>>(test_data[0]).size();
    struct PrefixedAccumulator {
      MultitargetMeanAccumulator acc;
      PrefixedAccumulator() : acc(MultitargetMeanAccumulator(4, test_size)){}
      void operator()(const treeson::containers::TreePredictionResult<std::vector<double>>& x) {
        acc(x);
      }
      [[nodiscard]] std::vector<double> result() const {
        return acc.results();
      }
      };
    struct DifferenceImportance{

        std::vector<double> operator()(const std::vector<double>& baseline,
                                       const std::vector<double>& feature) {
          std::vector<double> result(baseline.size());
          for(size_t i = 0; i < baseline.size(); i++) {
            result[i] = baseline[i] - feature[i];
          }
          return result;
        }
    };
    Stopwatch sw;
    const std::vector<size_t> targets = {0,1,2,3};
    std::cout << "Running feature importances" << std::endl;
    const auto importances = forest.feature_importance<PrefixedAccumulator, DifferenceImportance>(train_data, test_data, 100,//'000,
                              targets, true, 1024);
    std::cout << "Importances: "<< std::endl;
    for(const auto target: {targets[0]}) {
      treeson::utils::print_vector(importances[target]);
    }
  }

  for(const size_t n_tree : {100}){//'000}) {
    {
      MultitargetMeanAccumulator acc(
          4, std::get<std::vector<double>>(test_data[0]).size());
      Stopwatch sw;
      const std::vector<size_t> targets = {0,1,2,3};
      std::cout << "N trees: "<< n_tree << std::endl;
      forest.memoryless_predict(acc, train_data, test_data, n_tree//'000'000
                                , targets, true, 1024);
      std::cout << "Computing metrics: "<< std::endl;
      for(const auto target: targets) {
        const auto res = acc.one_result(target);
        //treeson::utils::print_vector(res);
        //treeson::utils::print_vector(std::get<std::vector<double>>(test_data[target]));

        std::cout << "Target: " << target+1 << std::endl;
        std::cout << " | Spearman: " <<
            treeson::utils::spearman_correlation(
                res, std::get<std::vector<double>>(test_data[target])
                    ) <<
            " Pearson: "
                  <<
            treeson::utils::pearson_correlation(
                res, std::get<std::vector<double>>(test_data[target])
                    ) <<
                    " rmse: " << treeson::utils::rmse(
                                      res, std::get<std::vector<double>>(test_data[target])
                                          ) << std::endl;
      }
    }
  }
  return 0;
}
