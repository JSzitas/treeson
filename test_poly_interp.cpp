#include <iostream>
#include <vector>
#include <random>
#include <variant>

#include "treeson.h"
#include "stopwatch.h"
#include "utils.h"

int main() {

  using DataType =
      std::vector<std::variant<std::vector<size_t>, std::vector<double>>>;
  // RNG and Result function
  std::mt19937 rng(42); // NOLINT(*-msc51-cpp)

  struct resultFuncRange {
    const size_t id;
    explicit resultFuncRange(const size_t target_index) : id(target_index) {}
    std::array<double,2> operator()(std::vector<size_t> &indices, size_t start,
                                   size_t end,
                                   [[maybe_unused]] const DataType &data) {

      const auto first = std::visit([&](auto&& arg) -> double {
        return arg[indices[start]];
      }, data[id]);
      double min_ = first, max_ = first;
      std::visit([&](auto&& arg) -> void {
        for (size_t i = start+1; i < end; i++) {
          const auto curr = arg[indices[i]];
          if(curr < min_) {
            min_ = curr;
          } else if(curr > max_) {
            max_ = curr;
          }
        }
      }, data[id]);
      return {min_, max_};
    };
  };
  using strat =
      treeson::splitters::ExtremelyRandomizedStrategy<std::mt19937, double, size_t, 0>;
  auto res_functor = resultFuncRange(2);
  auto strat_obj = strat{};

  CSVLoader<size_t, double> loader;
  DataType data = loader.load_data<true, true>("test_data/polyn_A.csv");
  auto [train, test] = train_test_split(data, 0.7);
  treeson::RandomForest<decltype(res_functor), std::mt19937, strat,
                        0, double> forest(8, 8, rng, res_functor, strat_obj);

  struct RangeAccumulator{
    size_t id;
    std::vector<double> min_, max_;
    explicit RangeAccumulator(const size_t id,
                              const size_t pred_size) : id(id) {
      min_ = std::vector<double>(pred_size, -99999999999.9);
      max_ = std::vector<double>(pred_size,  99999999999.9);
    }
    void operator()(const treeson::containers::TreePredictionResult<std::array<double,2>>& x) {
      const auto& x_ = x.expand_result();
      for(size_t i = 0; i < x_.size(); i++) {
        min_[i] = std::max(min_[i], x_[i][0]);
        max_[i] = std::min(max_[i], x_[i][1]);
      }
    }
    std::vector<double> result() const {
      std::vector<double> result;
      for (size_t i = 0; i < min_.size(); ++i) {
        result.push_back((max_[i] + min_[i])/2.0);
      }
      return result;
    }
    std::vector<std::pair<double,double>> ranges() const {
      std::vector<std::pair<double,double>> result;
      for (size_t i = 0; i < min_.size(); ++i) {
        result.push_back(std::pair(min_[i], max_[i]));
      }
      return result;
    }
  };
  std::cout << "Test size: "<< std::get<std::vector<double>>(test[2]).size() << std::endl;
  for(const auto n_tree: {200,500,1000,5'000, 50'000, 100'000, 500'000, 1'000'000})
  {
    RangeAccumulator acc(2, std::get<std::vector<double>>(test[0]).size());
    Stopwatch sw;
    forest.memoryless_predict(acc, train, test, n_tree
                              ,{2}, false);
    std::cout << "N trees: "<< n_tree << " | Spearman: " <<
        treeson::utils::spearman_correlation(
            acc.result(), std::get<std::vector<double>>(test[2])
                ) <<
        " Pearson: " <<
        treeson::utils::pearson_correlation(
            acc.result(), std::get<std::vector<double>>(test[2])
                ) << " rmse: " << treeson::utils::rmse(
                                  acc.result(), std::get<std::vector<double>>(test[2])
                                      ) << std::endl;
    /*CSVWriter<size_t, double>().write_data<true, false>(
        "test_result.csv", {"pred", "true"},{},
        {acc.result(), std::get<std::vector<double>>(test[2])});
    for(size_t i = 0; i < 10; i++) {
      std::cout << "True: " << std::get<std::vector<double>>(test[2])[i] <<
          //" | Pred: " << acc.result()[i] << "\n";
          " | Range: " << acc.ranges()[i].first << " - " <<acc.ranges()[i].second <<"\n";
    }*/
  }
}
