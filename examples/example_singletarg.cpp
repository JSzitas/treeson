#include <iostream>
#include <vector>
#include <random>
#include <variant>

#include "../utils/rf_singletarget.h"
#include "../utils/stopwatch.h"

#include "../utils/parquet.h"

int main() {
  using scalar_t = float;
  using integral_t = int;
  std::cout << "Loading train: ";
  auto train_data =
      load_parquet_data<integral_t, scalar_t>("test_data/train.parquet");
  //CSVLoader<int, scalar_t>::load_data<true, false>("test_data/dr_train.csv", 10000);
  std::cout << "Done." << std::endl;
  std::cout << "Loading test: ";
  auto test_data =
      load_parquet_data<integral_t, scalar_t>("test_data/test.parquet");
  //CSVLoader<int, scalar_t>::load_data<true, false>("test_data/dr_test.csv", 10000);
  std::cout << "Done." << std::endl;
  //const std::vector<size_t> targets = {0,1,2,3};
  SingletargetRandomForest<scalar_t, integral_t, 0> forest(0, 10, 12, 41, 1);

  for(const size_t n_tree : {100, 500}){
      Stopwatch sw;
      auto res = forest.memoryless_predict(train_data, test_data, n_tree, false);
      std::cout << "Computing metrics: "<< std::endl;
        std::cout << " | Spearman: " <<
            treeson::utils::spearman_correlation(
                res, std::get<std::vector<scalar_t>>(test_data[0])
                    ) <<
            " Pearson: "
                  <<
            treeson::utils::pearson_correlation(
                res, std::get<std::vector<scalar_t>>(test_data[0])
                    ) <<
            " rmse: " << treeson::utils::rmse(
                             res, std::get<std::vector<scalar_t>>(test_data[0])
                                 ) << std::endl;
  }
  // test normal fit/predict
  forest.fit(train_data, size_t(1000), false, size_t(1024));
  const auto pred = forest.predict(test_data);
  treeson::utils::print_vector(pred);
  std::cout << " | Spearman: " <<
      treeson::utils::spearman_correlation(
          pred, std::get<std::vector<scalar_t>>(test_data[0])
              ) <<
      " Pearson: "
            <<
      treeson::utils::pearson_correlation(
          pred, std::get<std::vector<scalar_t>>(test_data[0])
              ) <<
      " rmse: " << treeson::utils::rmse(
                       pred, std::get<std::vector<scalar_t>>(test_data[0])
                           ) << std::endl;


  return 0;
}
