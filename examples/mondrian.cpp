#include <iostream>
#include <vector>
#include <random>
#include <variant>

#include "../treeson.h"

int main() {
  // Simplified pseudocode for testing:
  using scalar_t = double;
  using integral_t = int;
  using DataType = std::vector<std::variant<std::vector<integral_t>, std::vector<scalar_t>>>;

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

  using strat = treeson::splitters::MondrianStrategy<
      std::mt19937, scalar_t, integral_t, 32>;
  std::vector<size_t> feats = {0,1,2};
  auto strat_obj = strat(10., feats);
  auto res_functor = resultFunc{};

  // Creating tree
  treeson::RandomTree<resultFunc, std::mt19937, strat,
                      32, scalar_t, integral_t> tree(10, 1, rng, res_functor, strat_obj);
  // Example data (larger)
  DataType data = {
      // Numeric features
      std::vector<scalar_t>{0.5, 0.4, 0.6, 1.5, std::nan(""), 1.4, 1.6, 2.5, 2.4, 2.6, 3.5, 3.4, 3.6, 4.5, 4.4, 4.6},
      std::vector<scalar_t>{1.5, 1.4, std::nan(""),1.6, 0.5, 0.4, 0.6, 3.5, 3.4, 3.6, 2.5, 2.4, 2.6, 4.5, 4.4, 4.6},

      // Categorical features
      std::vector<int>{1, 2, 3, 1, 2, 3, 1, 1, 1, 3, 1, 2, 3, 1, 2, 3},
      std::vector<int>{4, 5, 6, 4, 5, 6, 4, 5, 3, 6, 4, 5, 6, 4, 5, 6}
  };
  std::vector<size_t> indices(std::get<std::vector<scalar_t>>(data[0]).size());
  std::iota(indices.begin(), indices.end(), 0);

  tree.fit(data, indices, std::vector<size_t>{});
  std::cout << "Tree fitted" << std::endl;
  tree.print_tree_info();
  tree.print_tree_structure();
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
  // can we fit a forest, using an accumulator to get predicted values
  treeson::RandomForest<resultFunc, std::mt19937, strat,
                        32, scalar_t, int> forest(4, 1, rng, res_functor, strat_obj);
  forest.fit(data, size_t(100), std::vector<size_t>{}, false, size_t(0), size_t(1));
  const auto& predictions_forest = forest.predict(data, size_t(1));

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