#include <iostream>
#include <vector>
#include <random>
#include <variant>

#include "../utils/rf_multitarget.h"
#include "../utils/stopwatch.h"

#include "../utils/parquet.h"

template<typename scalar_t> [[nodiscard]]
auto one_result(const std::vector<std::vector<scalar_t>>& result,
                              const size_t which) {
  std::vector<scalar_t> res(result.size());
  const size_t n_preds = result.size();
  for(size_t k = 0; k < n_preds; k++) {
    res[k] = result[k][which];
  }
  return res;
}

int main() {
  using scalar_t = float;
  using integral_t = int;
  std::cout << "Loading train: ";
  auto train_data =
      load_parquet_data<integral_t, scalar_t>("../test_data/train.parquet");
      //CSVLoader<int, scalar_t>::load_data<true, false>("test_data/dr_train.csv", 10000);
  std::cout << "Done." << std::endl;
  std::cout << "Loading test: ";
  auto test_data =
      load_parquet_data<integral_t, scalar_t>("../test_data/test.parquet");
      //CSVLoader<int, scalar_t>::load_data<true, false>("test_data/dr_test.csv", 10000);
  std::cout << "Done." << std::endl;
  const std::vector<size_t> targets = {0,1,2,3};
  MultitargetRandomForest<scalar_t, integral_t, 0> forest(targets, 10, 12);
  /*
  for(const auto& n_tree: {100}) {//, 500, 1'000, 2'000, 5'000, 10'000, 50'000, 100'000}){
    Stopwatch sw;

    std::cout << "Running feature importances, n_tree: " << n_tree << std::endl;
    const auto importances = forest.feature_importance(
        train_data, n_tree, true, 1024);

    std::cout << "Importances: "<< std::endl;
    for(const auto& val : importances) {
      treeson::utils::print_vector(val);
    }
    CSVWriter<int, scalar_t>::write_data<false, false>("feature_importance_max_depth_10_n_tree_" +
                                                    std::to_string(n_tree) + "_contrast.csv",
                                                    {}, {},
                                                    importances
                                                    );
  }*/
  for(const size_t n_tree : {100, 500, 1'000, 2'000, 5'000, 10'000}){
    {
      Stopwatch sw;
      //std::cout << "N trees: "<< n_tree << std::endl;
      auto res_ = forest.memoryless_predict(train_data, test_data, n_tree, false);
      std::cout << "Computing metrics: "<< std::endl;
      for(const auto target: targets) {
        const auto res = one_result(res_, target);
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
      /*
      std::cout << "Writing predictions" << std::endl;
      CSVWriter<int, scalar_t>::write_data<true, false>("predictions.csv",
                                                         {"target_w", "target_r", "target_g", "target_b"}, {},
                                                         res_);
                                       */
    }
  }
  return 0;
}
