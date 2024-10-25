// multitarget_random_forest_Rcpp.cpp

#include <Rcpp.h>
#include <vector>
#include "../../../rf_multitarget.h"

using namespace Rcpp;


struct MultitargetRandomForestWrap{
private:
  using scalar_t = float;
  using integral_t = size_t;
  MultitargetRandomForest<scalar_t, integral_t, 32> forest;
  using DataType = std::vector<std::variant<std::vector<integral_t>, std::vector<scalar_t>>>;

  DataType convert_dataframe(const Rcpp::DataFrame& df) {
    DataType result;
    size_t n_cols = df.size();

    for (size_t i = 0; i < n_cols; ++i) {
      Rcpp::RObject col = df[i];
      if (Rcpp::is<Rcpp::NumericVector>(col)) {
        Rcpp::NumericVector num_col = Rcpp::as<Rcpp::NumericVector>(col);
        std::vector<scalar_t> vec(num_col.begin(), num_col.end());
        result.emplace_back(vec);
      }
      // Check if the column is integer
      else if (Rcpp::is<Rcpp::IntegerVector>(col)) {
        Rcpp::IntegerVector int_col = Rcpp::as<Rcpp::IntegerVector>(col);
        std::vector<integral_t> vec(int_col.begin(), int_col.end());
        result.emplace_back(vec);
      }
      // Check if the column is logical (boolean)
      else if (Rcpp::is<Rcpp::LogicalVector>(col)) {
        Rcpp::LogicalVector log_col = Rcpp::as<Rcpp::LogicalVector>(col);
        std::vector<integral_t> vec(log_col.begin(), log_col.end());
        result.emplace_back(vec);
      }
      // Add more type checks as needed
      else {
        Rcpp::stop("Unsupported column type");
      }
    }
    return result;
  }
public:
  MultitargetRandomForestWrap(std::vector<size_t> indices, size_t max_depth, size_t min_nodesize, int seed = 42) :
    forest(MultitargetRandomForest<scalar_t, integral_t, 32>(indices, max_depth, min_nodesize, seed)) {}
  std::vector<std::vector<scalar_t>> feature_importance(const Rcpp::DataFrame& train_data, const size_t n_tree,
                          const bool resample, const size_t subsample_size) {
    auto data = convert_dataframe(train_data);
    return forest.feature_importance(data, n_tree, resample, subsample_size);
  }

  std::vector<std::vector<float>> memoryless_predict(
      const Rcpp::DataFrame& train_data,
      const Rcpp::DataFrame& test_data,
      const size_t n_tree,
      const bool resample = true,
      const size_t subsample_size = 1024) {
    auto train = convert_dataframe(train_data);
    auto test = convert_dataframe(test_data);
    return forest.memoryless_predict(
      train, test, n_tree, resample, subsample_size);
  }
};
RCPP_MODULE(multitarget_random_forest_module) {
  class_<MultitargetRandomForestWrap>("MultitargetRandomForest")
  // Constructor
  .constructor<std::vector<size_t>, size_t, size_t, int>()
  // Methods
  .method("feature_importance", &MultitargetRandomForestWrap::feature_importance, "Calculate feature importance")
  .method("memoryless_predict", &MultitargetRandomForestWrap::memoryless_predict, "Memory-less prediction");
  //.method("fit", &MultitargetRandomForest_t::fit, "Fit the model");
}
