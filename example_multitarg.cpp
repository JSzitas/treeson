#include <iostream>
#include <vector>
#include <random>
#include <variant>

#include "treeson.h"
#include "reducers.h"
#include "stopwatch.h"

int main() {
  using DataType = std::vector<std::variant<std::vector<size_t>, std::vector<double>>>;
  // RNG and Result function
  std::mt19937 rng(22); // NOLINT(*-msc51-cpp)

  struct resultFuncSimple{
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
      std::vector<double> result;//(end-start * targets.size());
      //size_t p = 0;
      for(const auto index : targets) {
        //std::cout << "Feature: " << index << std::endl;
        std::visit([&](auto&& arg) -> void {
          for (size_t i = start; i < end; i++) {
            result.push_back(arg[indices[i]]);
          }
        }, data[index]);
      }
      return result;
    };
  };
  using strat = treeson::splitters::ExtremelyRandomizedStrategy<std::mt19937, double, 32>;
  //auto res_functor = resultFunc();
  auto res_functor = resultFuncMultitargetMean(std::vector<size_t>{4,5});
  auto strat_obj = strat{};
  // Example data (larger)
  DataType data = {
      // Numeric features
      std::vector<double>{0.5, 0.4, 0.6, 1.5, std::nan(""), 1.4, 1.6, 2.5, 2.4, 2.6, 3.5, 3.4, 3.6, 4.5, 4.4, 4.6},
      std::vector<double>{1.5, 1.4, std::nan(""),1.6, 0.5, 0.4, 0.6, 3.5, 3.4, 3.6, 2.5, 2.4, 2.6, 4.5, 4.4, 4.6},

      // Categorical features
      std::vector<size_t>{1, 2, 3, 1, 2, 3, 1, 1, 1, 3, 1, 2, 3, 1, 2, 3},
      std::vector<size_t>{4, 5, 6, 4, 5, 6, 4, 5, 3, 6, 4, 5, 6, 4, 5, 6},
      // Target features
      std::vector<double>{-0.52, 1.03, 1.17, -0.28, 0.52, -0.43, 0.55, -1.57,
                          -1.02, 0.86, -1.65, -0.68, -1.57, -2.19, 1.82, -1.39},
      std::vector<double>{1.51, -0.42, 0.1, -0.48, -0.83, 0.16, -1.33, -0.87,
                          1.07, 1.14, 1.39, 1.69, 0.48, 2.76, 0.71, 1.01}

  };

  treeson::RandomForest<decltype(res_functor), std::mt19937, strat,
                        32, double> forest(8, 1, rng, res_functor, strat_obj);

  /*forest.fit(data, size_t(10), std::vector<size_t>{4,5}, false, size_t(0));
  const auto& predictions_forest = forest.predict(data);
  size_t tree_id = 0, pred_id = 0;
  for(const auto& pred: predictions_forest) {
    std::cout << "Tree: " << tree_id++ << std::endl;
    pred_id = 0;
    for (const auto& prediction : pred.expand_result()) {
      std::cout << "Prediction id: "<< pred_id++ << "; ";
      for (const auto index : prediction) {
        std::cout << index << " ";
      }
      std::cout << "|\n";
    }
  }*/

  MultitargetMeanReducer<double> acc(2, std::get<std::vector<double>>(data[0]).size());
  for(const auto res : acc.result()) {
    std::cout << res << ",";
  }
  for(const auto res : acc.index()) {
    std::cout << res << ",";
  }

  {
    Stopwatch sw;
    forest.memoryless_predict(acc, data, data, 500000//'000'000
                                                 , {4,5}, false);
  }
  for(const auto res : acc.result()) {
    std::cout << res << ",";
  }
  std::cout << std::endl;

  return 0;
}
