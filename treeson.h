#include <iostream>
#include <vector>
#include <variant>
#include <array>
#include <optional>
#include <algorithm> // for std::find
#include <limits>
#include <random>
#include <unordered_set>
#include <numeric> // for std::iota
#include <cmath>
#include <thread>
#include <mutex>

namespace treeson {
namespace utils {
// Helper function to check if a type is a container
template <typename T, typename _ = void>
struct is_container : std::false_type {};
template <typename T>
constexpr bool is_container_v = is_container<T>::value;

#ifdef __CLION_IDE__  // Use a macro that the IDE defines
#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedTemplateParameterInspection"
#endif
template <typename... Ts>
struct is_container_helper {};
#ifdef __CLION_IDE__
#pragma clang diagnostic pop
#endif

template <typename T>
struct is_container<
    T, std::conditional_t<
           false,
           is_container_helper<typename T::value_type, typename T::size_type,
                               typename T::allocator_type, typename T::iterator,
                               typename T::const_iterator>,
           void>> : public std::true_type {};

// Utility function to print elements of a container
template <typename T>
void print_container(std::ostream &os, const T &container) {
  os << "[";
  for (auto it = container.cbegin(); it != container.cend(); ++it) {
    if (it != container.cbegin())
      os << ", ";
    os << *it;
  }
  os << "]";
}

// Generic print function
template <typename T>
[[maybe_unused]] void print_element(std::ostream &os, const T &element) {
  if constexpr (is_container<T>::value) {
    print_container(os, element);
  } else {
    os << element;
  }
}
template <typename Rng>
[[maybe_unused]] size_t select_from_range(const size_t range, Rng gen) noexcept {
  std::uniform_int_distribution<size_t> dist(0, range - 1);
  return dist(gen);
}
template<typename T> struct value_type_extractor;
template<template<typename...> class Container, typename Element>
struct value_type_extractor<Container<Element>> {
  using type [[maybe_unused]] = Element;
};
template <typename T>
using value_type_t = typename value_type_extractor<T>::type;
}
namespace containers {
template <const size_t MaxCategoricalSet> class FixedCategoricalSet {
public:
  std::array<size_t, MaxCategoricalSet> data{};
  size_t size = 0;
  [[nodiscard]] bool contains(const size_t value) const noexcept {
    return std::find(data.begin(), data.begin() + size, value) !=
           data.begin() + size;
  }
  // possibly just void?
  bool add(const size_t value) noexcept {
    if (size < MaxCategoricalSet) {
      data[size++] = value;
      return true;
    }
    return false;
  }
};

template <typename scalar_t, const size_t MaxCategoricalSet> class Node {
public:
  size_t featureIndex;
  std::variant<scalar_t, FixedCategoricalSet<MaxCategoricalSet>> threshold;
  size_t terminal_index = std::numeric_limits<size_t>::max();
  std::array<size_t, 3> parent_child_index = {0, 0, 0}; // Parent, Left, Right
  bool missing_goes_right;
  Node()
      : featureIndex(std::numeric_limits<size_t>::max()), terminal_index(0),
        missing_goes_right(false) {}
  [[nodiscard]] bool isCategorical() const noexcept {
    return std::holds_alternative<FixedCategoricalSet<MaxCategoricalSet>>(
        threshold);
  }
  const FixedCategoricalSet<MaxCategoricalSet> &
  getCategoricalSet() const noexcept {
    return std::get<FixedCategoricalSet<MaxCategoricalSet>>(threshold);
  }
  scalar_t getNumericThreshold() const noexcept {
    return std::get<scalar_t>(threshold);
  }
  [[nodiscard]] bool getMIADirection() const noexcept {
    return missing_goes_right;
  }
  void assign_terminal(const size_t index) { terminal_index = index; }
};
template <typename Key, typename Value, const size_t MaxSize>
class FixedSizeMap {
public:
  FixedSizeMap() : size_(0) {}

  [[maybe_unused]] void add(const Key &key, const Value &value) {
    size_t idx = find(key);
    if (idx != size_) {
      data[idx].second = value;
    } else {
      data[size_] = std::make_pair(key, value);
      ++size_;
    }
  }

  Value &operator[](const Key &key) {
    size_t idx = find(key);
    if (idx == size_) { // New key
      data[size_] = std::make_pair(key, Value{});
      ++size_;
    }
    return data[idx].second;
  }

  const Value &operator[](const Key &key) const {
    size_t idx = find(key);
    if (idx == size_) {
      throw std::out_of_range("Key not found!");
    }
    return data[idx].second;
  }

  [[nodiscard]] size_t size() const { return size_; }

private:
  std::array<std::pair<Key, Value>, MaxSize> data;
  size_t size_;

  size_t find(const Key &key) const {
    for (size_t i = 0; i < size_; ++i) {
      if (data[i].first == key) {
        return i;
      }
    }
    return size_;
  }
};
template <typename ResultType>
struct TreePredictionResult {
  std::vector<size_t> indices;
  std::vector<std::pair<std::pair<size_t, size_t>, ResultType>> results;

  // Expands the result back into the full prediction vector
  [[maybe_unused]] std::vector<ResultType> expand_result() const {
    std::vector<ResultType> expanded_results(indices.size());
    for (const auto& [range, result] : results) {
      for (size_t i = range.first; i < range.second; ++i) {
        expanded_results[indices[i]] = result;
      }
    }
    return expanded_results;
  }
  template <typename T = ResultType,
      typename std::enable_if_t<utils::is_container_v<T>, int> = 0>
  [[maybe_unused]] std::vector<typename ResultType::value_type>
      flatten() const {
    std::vector<typename ResultType::value_type> flattened_results;
    for (const auto& [range, result_container] : results) {
      for (const auto& element : result_container) {
        flattened_results.push_back(element);
      }
    }
    return flattened_results;
  }
};
}
namespace splitters {

template <typename RNG, typename scalar_t = double,
          const size_t MaxCategoricalSet = 10>
class ExtremelyRandomizedStrategy {
public:
  void select_split(
      const std::vector<
          std::variant<std::vector<size_t>, std::vector<scalar_t>>> &data,
      std::vector<size_t> &indices,
      const std::vector<size_t> &available_features, size_t start, size_t end,
      containers::Node<scalar_t, MaxCategoricalSet> &node, RNG &rng) const {
    const size_t feature =
        available_features[utils::select_from_range(available_features.size(), rng)];
    auto [mia_direction, threshold] = std::visit(
        FeatureVisitor{data, feature, indices, start, end, rng}, data[feature]);
    node.featureIndex = feature;
    node.threshold = threshold;
    node.missing_goes_right = mia_direction;
  }

private:
  struct FeatureVisitor {
    const std::vector<std::variant<std::vector<size_t>, std::vector<scalar_t>>>
        &data;
    const size_t feature;
    std::vector<size_t> &indices;
    size_t start;
    size_t end;
    RNG &rng;

    std::pair<bool, std::variant<scalar_t, containers::FixedCategoricalSet<MaxCategoricalSet>>>
    operator()(const std::vector<scalar_t> &) const noexcept {
      return {selectMIADirection(), selectRandomThreshold()};
    }

    std::pair<bool, std::variant<scalar_t, containers::FixedCategoricalSet<MaxCategoricalSet>>>
    operator()(const std::vector<size_t> &) const noexcept {
      return {selectMIADirection(), selectRandomCategoricalSet()};
    }

    scalar_t selectRandomThreshold() const noexcept {
      std::uniform_int_distribution<size_t> dist(start, end - 1);
      const auto &numericData = std::get<std::vector<scalar_t>>(data[feature]);
      //const size_t index = dist(rng);
      scalar_t val = numericData[indices[dist(rng)]];
      // basically attempt to find an actual splitting point
      while(std::isnan(val))
        val = numericData[indices[dist(rng)]];
      return val;
    }
    [[nodiscard]] bool selectMIADirection() const noexcept {
      return std::bernoulli_distribution(0.5)(rng);
    }

    containers::FixedCategoricalSet<MaxCategoricalSet> selectRandomCategoricalSet() const noexcept {
      std::unordered_set<size_t> uniqueValues;
      const auto &categoricalData =
          std::get<std::vector<size_t>>(data[feature]);
      for (size_t i = start; i < end; ++i) {
        uniqueValues.insert(categoricalData[indices[i]]);
      }
      const size_t n_selected = utils::select_from_range(uniqueValues.size(), rng);
      std::array<size_t, MaxCategoricalSet> shuffle_temp;
      size_t i = 0;
      for (const auto &val : uniqueValues) {
        shuffle_temp[i++] = val;
      }
      std::shuffle(shuffle_temp.begin(), shuffle_temp.begin() + i, rng);
      containers::FixedCategoricalSet<MaxCategoricalSet> categoricalSet;
      for (i = 0; i < n_selected; ++i) {
        categoricalSet.add(shuffle_temp[i]);
      }
      return categoricalSet;
    }
  };
};

template <typename RNG, typename scalar_t = double,
          const size_t MaxCategoricalSet = 32>
class [[maybe_unused]] CARTStrategy {
public:
  [[maybe_unused]] explicit CARTStrategy(size_t targetIndex)
      : targetIndex(targetIndex) {}
  void select_split(
      const std::vector<
          std::variant<std::vector<size_t>, std::vector<scalar_t>>> &data,
      std::vector<size_t> &indices,
      const std::vector<size_t> &available_features, size_t start, size_t end,
      containers::Node<scalar_t, MaxCategoricalSet> &node,
      [[maybe_unused]] RNG &rng) const {
    auto bestFeature = static_cast<size_t>(-1);
    scalar_t bestThreshold = 0;
    double bestImpurity = std::numeric_limits<double>::max();

    for (const size_t feature : available_features) {
      auto [featureImpurity, featureThreshold] = std::visit(
          FeatureVisitor{data, feature, start, end,indices}, data[feature]);
      if (featureImpurity < bestImpurity) {
        bestImpurity = featureImpurity;
        bestThreshold = featureThreshold;
        bestFeature = feature;
      }
    }
    node.featureIndex = bestFeature;
    node.threshold = bestThreshold;
  }

private:
  [[maybe_unused]] const size_t targetIndex;
  struct FeatureVisitor {
    const std::vector<std::variant<std::vector<size_t>, std::vector<scalar_t>>>
        &data;
    const size_t feature, start, end;
    std::vector<size_t> &indices;

    std::tuple<double, scalar_t>
    operator()(const std::vector<scalar_t> &) const noexcept {
      return findBestNumericSplit();
    }

    std::tuple<double, scalar_t>
    operator()(const std::vector<size_t> &) const noexcept {
      return findBestCategoricalSplit();
    }

    std::tuple<double, scalar_t> findBestNumericSplit() const noexcept {
      const auto &numericData = std::get<std::vector<scalar_t>>(data[feature]);
      std::vector<scalar_t> sorted_values(end - start);

      for (size_t i = start; i < end; ++i) {
        sorted_values[i - start] = numericData[indices[i]];
      }
      std::sort(sorted_values.begin(), sorted_values.end());

      double bestImpurity = std::numeric_limits<double>::max();
      scalar_t bestThreshold = 0.0;

      for (size_t i = 0; i < sorted_values.size() - 1; ++i) {
        scalar_t threshold = (sorted_values[i] + sorted_values[i + 1]) / 2.0;
        double impurity = calculateImpurity(threshold);
        if (impurity < bestImpurity) {
          bestImpurity = impurity;
          bestThreshold = threshold;
        }
      }

      return std::make_tuple(bestImpurity, bestThreshold);
    }

    std::tuple<double, scalar_t> findBestCategoricalSplit() const noexcept {
      auto [categoryMeans, categoryCounts] = computeCategoryMeans();

      const auto &categoricalData =
          std::get<std::vector<size_t>>(data[feature]);
      std::vector<std::pair<double, size_t>> indexedEncodedValues(end - start);

      for (size_t i = start; i < end; ++i) {
        indexedEncodedValues[i - start] = {
            categoryMeans[categoricalData[indices[i]]], i};
      }
      auto cmp = [](const auto &a, const auto &b) { return a.first < b.first; };
      std::sort(indexedEncodedValues.begin(), indexedEncodedValues.end(), cmp);

      double bestImpurity = std::numeric_limits<double>::max();
      scalar_t bestThreshold = 0.0;

      double left_sum = 0.0;
      size_t left_count = 0;
      double total_sum = 0.0;
      size_t total_count = 0;

      for (const auto &[key, value] : categoryMeans.data) {
        if (total_count >= categoryMeans.size())
          break;
        total_sum += value * categoryCounts[key];
        total_count += categoryCounts[key];
      }

      for (size_t i = 0; i < indexedEncodedValues.size() - 1; ++i) {
        double encodedValue = indexedEncodedValues[i].first;

        left_sum += encodedValue;
        left_count += 1;
        total_sum -= encodedValue;
        total_count -= 1;

        if (encodedValue != indexedEncodedValues[i + 1].first) {
          double threshold =
              (encodedValue + indexedEncodedValues[i + 1].first) / 2.0;
          double impurity =
              calculateImpurity(left_sum / static_cast<double>(left_count),
                                total_sum / static_cast<double>(total_count));
          if (impurity < bestImpurity) {
            bestImpurity = impurity;
            bestThreshold = static_cast<scalar_t>(threshold);
          }
        }
      }

      return std::make_tuple(bestImpurity, bestThreshold);
    }

    std::tuple<containers::FixedSizeMap<size_t, scalar_t, MaxCategoricalSet>,
               containers::FixedSizeMap<size_t, size_t, MaxCategoricalSet>>
    computeCategoryMeans() const noexcept {
      containers::FixedSizeMap<size_t, double, MaxCategoricalSet> categorySums;
      containers::FixedSizeMap<size_t, size_t, MaxCategoricalSet> categoryCounts;

      const auto &categoricalData =
          std::get<std::vector<size_t>>(data[feature]);

      for (size_t i = start; i < end; ++i) {
        size_t category = categoricalData[indices[i]];
        categorySums[category] += indices[i]; // Replace with target value
        categoryCounts[category]++;
      }

      for (size_t i = 0; i < categorySums.size(); ++i) {
        size_t category = categorySums.data[i].first;
        categorySums[category] /= categoryCounts[category];
      }

      return std::make_tuple(categorySums, categoryCounts);
    }

    [[nodiscard]] double calculateImpurity(double left_mean,
                             double right_mean) const noexcept {
      return left_mean + right_mean;
    }

    double calculateImpurity(const scalar_t threshold) const noexcept {
      return threshold;
    }
  };
};
}
template <typename ResultF, typename RNG, typename SplitStrategy,
          const size_t MaxCategoricalSet = 32, typename scalar_t = double>
class RandomTree {
  using FeatureData = std::variant<std::vector<size_t>, std::vector<scalar_t>>;
  // public:
  using ResultType = decltype(std::declval<ResultF>()(
      std::declval<std::vector<size_t> &>(), std::declval<size_t>(),
      std::declval<size_t>(),
      std::declval<const std::vector<FeatureData> &>()));

  std::vector<containers::Node<scalar_t, MaxCategoricalSet>> nodes;
  const size_t maxDepth, minNodesize;
  RNG &rng;
  ResultF &terminalNodeFunc;
  std::unordered_set<size_t> uniqueValues;
  std::vector<ResultType> terminal_values;
  SplitStrategy &split_strategy;

public:
  [[maybe_unused]] RandomTree<ResultF, RNG, SplitStrategy, MaxCategoricalSet,
                              scalar_t>(const size_t maxDepth,
                                        const size_t minNodesize,
                                        RNG &rng,
                                        ResultF &terminalNodeFunc,
                                        SplitStrategy &strategy) noexcept
      : maxDepth(maxDepth), minNodesize(minNodesize), rng(rng),
        terminalNodeFunc(terminalNodeFunc), split_strategy(strategy) {
    nodes.reserve(static_cast<size_t>(std::pow(2, maxDepth + 1) - 1));
    uniqueValues = std::unordered_set<size_t>{};
    terminal_values = std::vector<ResultType>{};
  }
  [[maybe_unused]] void
  fit(const std::vector<FeatureData> &data, std::vector<size_t> indices,
      const std::vector<size_t> &nosplit_features) noexcept {
    // Create a list of available features, excluding nosplit_features
    std::vector<size_t> available_features;
    for (size_t i = 0; i < data.size(); ++i) {
      if (std::find(nosplit_features.begin(), nosplit_features.end(), i) ==
          nosplit_features.end()) {
        available_features.push_back(i);
      }
    }
    buildTree(data, indices, available_features, 0, 0, indices.size());
  }
  [[maybe_unused]] void fit(const std::vector<FeatureData> &data,
                            std::vector<size_t> indices,
                            std::vector<size_t> &&nosplit_features) noexcept {
    // Create a list of available features, excluding nosplit_features
    std::vector<size_t> available_features;
    for (size_t i = 0; i < data.size(); ++i) {
      if (std::find(nosplit_features.begin(), nosplit_features.end(), i) ==
          nosplit_features.end()) {
        available_features.push_back(i);
      }
    }
    buildTree(data, indices, available_features, 0, 0, indices.size());
  }

  template <const bool FlattenResults = false>
  [[nodiscard]] auto predict(const std::vector<FeatureData> &samples,
                             const std::vector<size_t> &indices) const noexcept {
    if constexpr (FlattenResults) {
      std::vector<utils::value_type_t<ResultType>> flat_result;
      predictSamples<true>(0, samples, indices, 0, indices.size(), flat_result);
      return flat_result;
    } else {
      containers::TreePredictionResult<ResultType> result;
      result.indices = indices;
      predictSamples<false>(0, samples, indices, 0, indices.size(), result);
      return result;
    }
  }
  template <const bool FlattenResults = false>
  [[nodiscard]] auto predict(const std::vector<FeatureData> &samples)
      const noexcept {
    std::vector<size_t> indices(std::visit(
        [](auto &&arg) -> size_t { return arg.size(); }, samples[0]));
    std::iota(indices.begin(), indices.end(), 0);
    if constexpr (FlattenResults) {
      std::vector<utils::value_type_t<ResultType>> flat_result;
      predictSamples<true>(0, samples, indices, 0, indices.size(), flat_result);
      return flat_result;
    } else {
      containers::TreePredictionResult<ResultType> result;
      result.indices = indices;
      predictSamples<false>(0, samples, indices, 0, indices.size(), result);
      return result;
    }
  }

  [[maybe_unused]] void print_tree_info() const noexcept {
    std::cout << "Number of terminal nodes: " << terminal_values.size()
              << std::endl;
    std::cout << "Number of inner nodes: "
              << nodes.size() - terminal_values.size() << std::endl;
    std::cout << "Total number of nodes: " << nodes.size() << std::endl;
  }
  [[maybe_unused]] void print_tree_structure() const noexcept { print_node(0, 0); }
  void print_terminal_node_values() const noexcept {
    std::cout << "Terminal Node Values:" << std::endl;
    for (size_t i = 0; i < terminal_values.size(); ++i) {
      std::cout << "Terminal Node " << i << ": ";
      utils::print_element(std::cout, terminal_values[i]);
      std::cout << std::endl;
    }
  }
  bool is_uninformative() const noexcept {
    // consists of only root
    return nodes.size() == 1;
  }

private:
  void split(const std::vector<FeatureData> &data,
             const containers::Node<scalar_t, MaxCategoricalSet> &node,
             std::vector<size_t> &indices, size_t start, size_t &mid,
             size_t end) const noexcept {

    const auto feature = node.featureIndex;
    const auto &threshold = node.threshold;
    const auto missing_dir = node.missing_goes_right;

    if (std::holds_alternative<scalar_t>(threshold)) {
      const scalar_t th = std::get<scalar_t>(threshold);
      const auto &numericData = std::get<std::vector<scalar_t>>(data[feature]);
      for (size_t i = start; i < end; ++i) {
        const auto &val = numericData[indices[i]];
        if ((val <= th) || (std::isnan(val) && missing_dir)) {
          std::swap(indices[i], indices[mid]);
          ++mid;
        }
      }
    } else {
      const auto th =
          std::get<containers::FixedCategoricalSet<MaxCategoricalSet>>(
              threshold);
      const auto &categoricalData =
          std::get<std::vector<size_t>>(data[feature]);
      for (size_t i = start; i < end; ++i) {
        const auto &val = categoricalData[indices[i]];
        if (th.contains(val)) {
          std::swap(indices[i], indices[mid]);
          ++mid;
        }
      }
    }
  }
  void buildTree(const std::vector<FeatureData> &data,
                 std::vector<size_t> &indices,
                 const std::vector<size_t> &available_features,
                 size_t parentIndex, size_t start, size_t end,
                 size_t depth = 0) noexcept {
    nodes.emplace_back();
    size_t nodeIndex = nodes.size() - 1;
    if (nodes[parentIndex].parent_child_index[1] == 0) {
      nodes[parentIndex].parent_child_index[1] = nodeIndex; // Left child
    } else {
      nodes[parentIndex].parent_child_index[2] = nodeIndex; // Right child
    }
    split_strategy.select_split(data, indices, available_features, start, end,
                                nodes[nodeIndex], rng);
    size_t mid = start;
    //std::cout << "Left going: " << mid - start << " right going: " << end - mid
    //          << std::endl;
    split(data, nodes[nodeIndex], indices, start, mid, end);
    // if a child node would be too small
    if (((mid - start) <= minNodesize) || ((end - mid) <= minNodesize) ||
        depth >= maxDepth) {
      // get rid of last node since it was actually invalid
      terminal_values.push_back(terminalNodeFunc(indices, start, end, data));
      nodes[nodeIndex].assign_terminal(terminal_values.size() - 1);
    } else {
      nodes[nodeIndex].parent_child_index[0] = parentIndex;
      buildTree(data, indices, available_features, nodeIndex, start, mid,
                depth + 1);
      buildTree(data, indices, available_features, nodeIndex, mid, end,
                depth + 1);
    }
  }
  void print_node(size_t node_index, size_t depth) const noexcept {
    const auto &node = nodes[node_index];
    if (depth >= maxDepth) {
      return;
    }
    std::string indent(depth * 4, '-');
    if (node.parent_child_index[0] == 0 && node.parent_child_index[1] == 0) {
      std::cout << indent << "Terminal Node [Value: ";
      utils::print_element(std::cout, terminal_values[node.terminal_index]);
      std::cout << ", index: " << node.terminal_index << "]" << std::endl;
      return;
    }
    std::cout << indent << "Internal Node [Feature: " << node.featureIndex
              << "]";
    if (node.isCategorical()) {
      std::cout << " [Categorical Split: ";
      const auto &catSet = node.getCategoricalSet();
      for (size_t i = 0; i < catSet.size; ++i) {
        std::cout << catSet.data[i];
        if (i != catSet.size - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "] | MIA: " << node.getMIADirection();
    } else {
      std::cout << " [Numeric Threshold: " << node.getNumericThreshold()
                << "] | MIA: " << node.getMIADirection();
    }
    std::cout << std::endl;
    print_node(node.parent_child_index[1], depth + 1);
    print_node(node.parent_child_index[2], depth + 1);
  }
  template <const bool FlattenResults, typename ResultContainer>
  void predictSamples(size_t nodeIndex, const std::vector<FeatureData> &samples,
                      std::vector<size_t> &indices, size_t start, size_t end,
                      ResultContainer &predictionResult) const noexcept {
    const auto &node = nodes[nodeIndex];
    if (node.parent_child_index[1] == 0 && node.parent_child_index[2] == 0) {
      // leaf node
      if constexpr (FlattenResults) {
        const auto &result_container = terminal_values[node.terminal_index];
        predictionResult.insert(predictionResult.end(),
                                result_container.begin(),
                                result_container.end());
      } else {
        predictionResult.results.emplace_back(
            std::make_pair(start, end), terminal_values[node.terminal_index]);
      }
      return;
    }

    auto featureData = samples[node.featureIndex];
    const auto miss_dir = node.getMIADirection();
    size_t mid = start;
    if (node.isCategorical()) {
      const auto &catSet = node.getCategoricalSet();
      const auto &catData = std::get<std::vector<size_t>>(featureData);
      for (size_t i = start; i < end; ++i) {
        const auto &val = catData[indices[i]];
        if (catSet.contains(val)) {
          std::swap(indices[i], indices[mid]);
          ++mid;
        }
      }
    } else {
      const auto &numThreshold = node.getNumericThreshold();
      const auto &numData = std::get<std::vector<scalar_t>>(featureData);
      for (size_t i = start; i < end; ++i) {
        const auto &val = numData[indices[i]];
        if ((val <= numThreshold) || (std::isnan(val) && miss_dir)) {
          std::swap(indices[i], indices[mid]);
          ++mid;
        }
      }
    }
    if (mid > start) {
      predictSamples<FlattenResults>(
          node.parent_child_index[1], samples,
          indices, start, mid, predictionResult);
    }
    if (mid < end) {
      predictSamples<FlattenResults>(
          node.parent_child_index[2], samples,
          indices, mid, end, predictionResult);
    }
  }
};

template <typename ResultF, typename RNG, typename SplitStrategy,
          size_t MaxCategoricalSet = 32, typename scalar_t = double>
class [[maybe_unused]] RandomForest {
  using FeatureData = std::variant<std::vector<size_t>, std::vector<scalar_t>>;
  using TreeType = RandomTree<ResultF, RNG, SplitStrategy, MaxCategoricalSet, scalar_t>;
  using ResultType = decltype(std::declval<ResultF>()(
      std::declval<std::vector<size_t> &>(), std::declval<size_t>(),
      std::declval<size_t>(),
      std::declval<const std::vector<FeatureData> &>()));
public:
  [[maybe_unused]] RandomForest(const size_t maxDepth,
                                const size_t minNodesize,
                                RNG& rng,
                                ResultF &terminalNodeFunc,
                                SplitStrategy &strategy) :
        maxDepth(maxDepth), minNodesize(minNodesize), rng(rng),
        terminalNodeFunc(terminalNodeFunc), strategy(strategy) {}
  [[maybe_unused]] void fit(const std::vector<FeatureData> &data,
           const size_t n_tree,
           const std::vector<size_t> &nosplit_features,
           const bool resample = true,
           const size_t sample_size = 0) {
    trees.reserve(n_tree);
    size_t bootstrap_size = sample_size > 0 ? sample_size :
      std::visit([](auto&& arg) -> size_t { return arg.size(); }, data[0]);
    const bool resample_ = resample && sample_size > 0;
    if(resample != resample_) {
      std::cout << "You specified 'resample = true', " <<
          "but provided 'sample_size = 0'. Quitting, respecify." << std::endl;
      return;
    }
    std::vector<std::thread> threads;
    std::mutex mtx;

    std::vector<size_t> indices(
        std::visit([](auto&& arg) -> size_t {
          return arg.size();
        }, data[0]));
    std::iota(indices.begin(), indices.end(), 0);
    // TODO(JSzitas): revisit and validate
    // do the lazy thing following L'Ecuyer; I am not totally sure it is applicable
    // see 'Random Numbers for Parallel Computers: Requirements and Methods, With Emphasis on GPUs'
    // Pierre L’Ecuyer, David Munger, Boris Oreshkin, Richard Simard, p.15
    // 'A single RNG with a “random” seed for each stream.'
    // link: https://www.iro.umontreal.ca/~lecuyer/myftp/papers/parallel-rng-imacs.pdf
    std::vector<size_t> seeds(n_tree);
    for(auto& seed : seeds) {
      seed = rng();
    }
    for (size_t i = 0; i < n_tree; ++i) {
      threads.emplace_back([this, &data, &indices, &nosplit_features, &mtx,
                            resample_, &seeds, bootstrap_size, i] {
        // custom random number generator for this tree
        RNG rng_(seeds[i]);
        TreeType tree(maxDepth, minNodesize, rng_, terminalNodeFunc, strategy);
        if (resample_) {
          tree.fit(data, bootstrap_sample(indices, bootstrap_size), nosplit_features);
        } else {
          tree.fit(data, indices, nosplit_features);
        }
        std::lock_guard<std::mutex> lock(mtx);
        trees.emplace_back(tree);
      });
    }
    for (auto &t : threads) {
      t.join();
    }
  }

  [[nodiscard]] std::vector<containers::TreePredictionResult<ResultType>> predict(
      const std::vector<FeatureData> &samples) const noexcept {
    std::vector<containers::TreePredictionResult<ResultType>> results;
    // Collect predictions from each tree
    for (const auto &tree : trees) {
      results.push_back(tree.predict(samples));    }
    return results;
  }
  template<typename Accumulator>
  [[maybe_unused]] void memoryless_predict(
      const std::vector<FeatureData> &train_data,
      const std::vector<FeatureData> &predict_data,
      Accumulator& accumulator,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool resample = true,
      const size_t sample_size = 0) {
    size_t bootstrap_size = sample_size > 0 ? sample_size :
      std::visit([](auto&& arg) -> size_t { return arg.size(); }, train_data[0]);
    const bool resample_ = resample && sample_size > 0;
    if(resample != resample_) {
      std::cout << "You specified 'resample = true', " <<
          "but provided 'sample_size = 0'. Quitting, respecify." << std::endl;
      return;
    }
    std::vector<std::thread> threads;
    std::mutex mtx;

    std::vector<size_t> indices(
        std::visit([](auto&& arg) -> size_t {
          return arg.size();
        }, train_data[0]));
    std::iota(indices.begin(), indices.end(), 0);
    // TODO(JSzitas): revisit and validate
    // do the lazy thing following L'Ecuyer; I am not totally sure it is applicable
    // see 'Random Numbers for Parallel Computers: Requirements and Methods, With Emphasis on GPUs'
    // Pierre L’Ecuyer, David Munger, Boris Oreshkin, Richard Simard, p.15
    // 'A single RNG with a “random” seed for each stream.'
    // link: https://www.iro.umontreal.ca/~lecuyer/myftp/papers/parallel-rng-imacs.pdf
    std::vector<size_t> seeds(n_tree);
    for(auto& seed : seeds) {
      seed = rng();
    }
    for (size_t i = 0; i < n_tree; ++i) {
      threads.emplace_back([this, &train_data, &predict_data,
                            &accumulator, &indices, &nosplit_features, &mtx,
                            resample_, &seeds, bootstrap_size, &i] {
        // custom random number generator for this tree
        RNG rng_(seeds[i]);
        TreeType tree(maxDepth, minNodesize, rng_, terminalNodeFunc, strategy);
        if (resample_) {
          tree.fit(train_data, bootstrap_sample(indices, bootstrap_size), nosplit_features);
        } else {
          tree.fit(train_data, indices, nosplit_features);
        }
        if(tree.is_uninformative()) {
          // make it clear that this tree does not count
          // and return; this enables us to avoid pruning
          i--;
          return;
        }
        // this gets passed onto a vector of thread pool size and reduced in
        // another thread maybe?
        auto prediction_result = tree.predict(predict_data);
        std::lock_guard<std::mutex> lock(mtx);
        accumulator(prediction_result);
      });
    }
    for (auto &t : threads) {
      t.join();
    }
  }
  [[maybe_unused]] void prune() {
    // TODO(JSzitas): possibly add warning and print how many trees will be pruned?
    trees.erase(
        std::remove_if(trees.begin(), trees.end(), [](const TreeType &tree) {
          return tree.is_uninformative();
        }),
        trees.end()
    );
  }
private:
  const size_t maxDepth, minNodesize;
  RNG& rng;
  ResultF& terminalNodeFunc;
  SplitStrategy& strategy;
  std::vector<TreeType> trees;

  std::vector<size_t> bootstrap_sample(const std::vector<size_t>& indices,
                                       const size_t sample_size) {
    std::uniform_int_distribution<size_t> dist(0, indices.size() - 1);
    std::vector<size_t> bootstrap_indices(sample_size);
    for (auto &index : bootstrap_indices) {
      index = indices[dist(rng)];
    }
    return bootstrap_indices;
  }
};
/*
// TODO(JSzitas): first implementation of gradient boosting,
// actually should also enable https://arxiv.org/pdf/2407.02279
template <typename TreeType>
class Treeson {
public:
  using FeatureData = typename TreeType::FeatureData;
  using ResultType = typename TreeType::ResultType;

  GradientBoosting(size_t n_estimators, double learning_rate)
      : n_estimators(n_estimators), learning_rate(learning_rate) {}

  void fit(const std::vector<FeatureData>& data, const std::vector<ResultType>& targets) {
    std::vector<ResultType> predictions(targets.size(), 0.0);
    residuals = targets;  // Initialize residuals with actual targets

    for (size_t i = 0; i < n_estimators; ++i) {
      auto tree = std::make_unique<TreeType>();

      tree->fit(data, residuals);  // Fit tree on residuals
      trees.push_back(std::move(tree));

      // Update predictions and residuals
      auto current_predictions = trees.back()->predict(data);
      for (size_t j = 0; j < predictions.size(); ++j) {
        predictions[j] += learning_rate * current_predictions[j];
        residuals[j] = targets[j] - predictions[j];
      }
    }
  }

  std::vector<ResultType> predict(const std::vector<FeatureData>& data) const {
    std::vector<ResultType> predictions(data.front().size(), 0.0);

    for (const auto& tree : trees) {
      auto current_predictions = tree->predict(data);
      for (size_t i = 0; i < predictions.size(); ++i) {
        predictions[i] += learning_rate * current_predictions[i];
      }
    }

    return predictions;
  }

private:
  size_t n_estimators;
  double learning_rate;
  std::vector<std::unique_ptr<TreeType>> trees;
  std::vector<ResultType> residuals;
};
*/
}

