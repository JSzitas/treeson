#include <iostream>
#include <vector>
#include <random>
#include <variant>

#include "treeson.h"

int main() {
  // Simplified pseudocode for testing:

  using DataType = std::vector<std::variant<std::vector<size_t>, std::vector<double>>>;

  // RNG and Result function
  std::mt19937 rng(22); // NOLINT(*-msc51-cpp)
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
  // Mock result function
  using strat = treeson::splitters::ExtremelyRandomizedStrategy<std::mt19937, double, 32>;


  auto res_functor = resultFunc{};
  auto strat_obj = strat{};
  // Creating tree
  treeson::RandomTree<resultFunc, std::mt19937, strat,
                          32, double> tree(3, 1, rng, res_functor, strat_obj);

  // Example data (simplified)
  /*
  std::vector<std::variant<std::vector<size_t>, std::vector<double>>> data = {
      std::vector<double>{0.5, 0.3, 0.6, 0.9, 0.2},
      std::vector<size_t>{1, 2, 3, 2, 1}
  };*/

  // Example data (larger)
  DataType data = {
      // Numeric features
      std::vector<double>{0.5, 0.4, 0.6, 1.5, std::nan(""), 1.4, 1.6, 2.5, 2.4, 2.6, 3.5, 3.4, 3.6, 4.5, 4.4, 4.6},
      std::vector<double>{1.5, 1.4, std::nan(""),1.6, 0.5, 0.4, 0.6, 3.5, 3.4, 3.6, 2.5, 2.4, 2.6, 4.5, 4.4, 4.6},

      // Categorical features
      std::vector<size_t>{1, 2, 3, 1, 2, 3, 1, 1, 1, 3, 1, 2, 3, 1, 2, 3},
      std::vector<size_t>{4, 5, 6, 4, 5, 6, 4, 5, 3, 6, 4, 5, 6, 4, 5, 6}
  };
  std::vector<size_t> indices(std::get<std::vector<double>>(data[0]).size());
  std::iota(indices.begin(), indices.end(), 0);

  /*std::cout << "Fun:" << resultFunc(std::get<std::vector<double>>(data[0]).begin(),
                                    std::get<std::vector<double>>(data[0]).end(),
                                    data) << std::endl;
  */
  tree.fit(data, indices, std::vector<size_t>{});
  std::cout << "Tree fitted" << std::endl;
  //tree.print_tree_info();
  //tree.print_tree_structure();
  tree.print_terminal_node_values();


  std::cout << "Trying prediction" << std::endl;
  const auto& predictions = tree.predict(data);
  std::cout << "Prediction successful" << std::endl;


  // Print predictions
  std::cout << "Predictions: " << std::endl;
  size_t pred_id = 0;
  for (const auto& prediction : predictions.expand_result()) {
    std::cout << "Prediction id: "<< pred_id++ << "; ";
    for (const auto index : prediction) {
      std::cout << index << " ";
    }
    std::cout << "|\n";
  }
  std::cout << std::endl;

  treeson::RandomForest<resultFunc, std::mt19937, strat,
                        32, double> forest(4, 1, rng, res_functor, strat_obj);
  forest.fit(data, size_t(10), std::vector<size_t>{}, false, size_t(0));
  const auto& predictions_forest = forest.predict(data);
  forest.prune();

  size_t tree_id = 0;
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
  }


  return 0;
}