#include <iostream>
#include <vector>
#include <random>
#include <variant>

#include "treeson.h"

int main() {
  // Simplified pseudocode for testing:
  using scalar_t = double;
  using DataType = std::vector<std::variant<std::vector<int>, std::vector<scalar_t>>>;

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
  using strat = treeson::splitters::ExtremelyRandomizedStrategy<std::mt19937,
                                                                scalar_t, int, 32>;


  auto res_functor = resultFunc{};
  auto strat_obj = strat{};
  // Creating tree
  treeson::RandomTree<resultFunc, std::mt19937, strat,
                          32, scalar_t, int> tree(3, 1, rng, res_functor, strat_obj);

  // Example data (simplified)
  /*
  std::vector<std::variant<std::vector<size_t>, std::vector<scalar_t>>> data = {
      std::vector<scalar_t>{0.5, 0.3, 0.6, 0.9, 0.2},
      std::vector<size_t>{1, 2, 3, 2, 1}
  };*/

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

  /*std::cout << "Fun:" << resultFunc(std::get<std::vector<scalar_t>>(data[0]).begin(),
                                    std::get<std::vector<scalar_t>>(data[0]).end(),
                                    data) << std::endl;
  */
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
  rng.seed(22); // NOLINT(*-msc51-cpp)
  treeson::RandomForest<resultFunc, std::mt19937, strat,
                        32, scalar_t, int> forest(4, 1, rng, res_functor, strat_obj);
  forest.fit(data, size_t(10), std::vector<size_t>{}, false, size_t(0), size_t(1));
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
  std::cout << "Saving tree" << std::endl;
  tree.save("tree_save_test");
  std::cout << "Loading tree" << std::endl;
  tree.load("tree_save_test");
  std::cout << "Running prediction again" << std::endl;

  std::cout << "Trying prediction" << std::endl;
  const auto& predictions2 = tree.predict(data);
  std::cout << "Prediction successful" << std::endl;
  // Print predictions
  std::cout << "Predictions match?: " << std::endl;
  const auto expanded_1 =   predictions.expand_result();
  const auto expanded_2 = predictions2.expand_result();
  std::cout << "Sizes?: " << (expanded_1.size() == expanded_2.size()) << std::endl;
  size_t pred = 0;
  for(; pred < expanded_1.size(); pred++) {
    for (const auto index : expanded_1[pred]) {
      std::cout << index << " ";
    }
    std::cout << " | ";
    for (const auto index : expanded_2[pred]) {
      std::cout << index << " ";
    }
    std::cout << "\n";
  }
  std::cout << "Saving forest" << std::endl;
  forest.save("forest_save_test");
  std::cout << "Loading forest" << std::endl;
  forest.load("forest_save_test");

  const auto& pred_from_file = forest.
                               predict(data, std::string("forest_save_test"), size_t(1));
  for(tree_id = 0; tree_id < predictions_forest.size(); tree_id++) {
    std::cout << "Tree: " << tree_id << std::endl;
    pred_id = 0;
    const auto& original = predictions_forest[tree_id].expand_result();
    const auto& deserialized = pred_from_file[tree_id].expand_result();
    for(size_t k =0; k < deserialized.size(); k++) {
      std::cout << "Prediction id: "<< pred_id++ << "; ";
      for (const auto &index : original[k]) {
        std::cout << index << " ";
      }
      std::cout << " | ";
      for (const auto &index : deserialized[k]) {
        std::cout << index << " ";
      }
      std::cout << "\n";
    }
  }
  rng.seed(22); // NOLINT(*-msc51-cpp)
  // fitting a forest but only saving it, rather than materializing
  treeson::RandomForest<resultFunc, std::mt19937, strat,
                        32, scalar_t, int> forest2(4, 1, rng, res_functor, strat_obj);
  forest2.fit(data, size_t(10), std::vector<size_t>{}, std::string("forest_test"),
             false, size_t(0), size_t(1));
  const auto pred_from_file2 = forest2.predict(
      data, std::string("forest_test"), size_t(1));
  for(tree_id = 0; tree_id < predictions_forest.size(); tree_id++) {
    std::cout << "Tree: " << tree_id << std::endl;
    pred_id = 0;
    const auto& original = pred_from_file[tree_id].expand_result();
    const auto& deserialized = pred_from_file2[tree_id].expand_result();
    for(size_t k =0; k < deserialized.size(); k++) {
      std::cout << "Prediction id: "<< pred_id++ << "; ";
      for (const auto &index : original[k]) {
        std::cout << index << " ";
      }
      std::cout << " | ";
      for (const auto &index : deserialized[k]) {
        std::cout << index << " ";
      }
      std::cout << "\n";
    }
  }
  return 0;
}