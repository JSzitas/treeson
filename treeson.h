#include <iostream>
#include <vector>
#include <variant>
#include <array>
#include <fstream>
#include <optional>
#include <algorithm> // for std::find
#include <limits>
#include <set>
#include <random>
#include <unordered_set>
#include <numeric> // for std::iota
#include <cmath>
#include <thread>
#include <mutex>
#include <type_traits>
#include <condition_variable>
#include <functional>
#include <queue>

namespace treeson {
namespace utils {
// Helper function to check if a type is a container
template <typename T, typename _ = void>
struct is_container : std::false_type {};
template <typename T>
constexpr bool is_container_v = is_container<T>::value;
#ifdef __CLION_IDE__
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
                               typename T::const_iterator,
                               decltype(std::declval<T>().size()),
                               decltype(std::declval<T>().begin()),
                               decltype(std::declval<T>().end())>,
           void>> : public std::true_type {};
template<typename Accumulator, typename Arg>
struct is_invocable_with {
template<typename T>
[[maybe_unused]] static auto test([[maybe_unused]] T* p) ->
    decltype((*p)(std::declval<Arg>()), std::true_type{}) {
  return std::true_type{};
};
template <typename> static auto test(...) -> std::false_type {};
[[maybe_unused]] static constexpr bool value = std::is_same_v<decltype(test<Accumulator>(0)), std::true_type>;
};
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
// Overloaded function template for `std::array`
template <typename T, std::size_t N>
[[maybe_unused]] void print_element(std::ostream &os, const std::array<T, N> &element) {
  // Forward the std::array to the primary template for containers
  print_container(os, element);
}
template <typename Rng>
[[maybe_unused]] size_t select_from_range(const size_t range, Rng gen) noexcept {
  std::uniform_int_distribution<size_t> dist(0, range - 1);
  return dist(gen);
}
template <typename T>
[[maybe_unused]] void print_vector(const std::vector<T>& vec) {
  std::cout << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    std::cout << vec[i];
    if (i < vec.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}
[[maybe_unused]] void print_vector_of_pairs(
    const std::vector<std::array<double, 2>>& vec) {
  std::cout << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    std::cout << "(" << vec[i][0] << ", " << vec[i][1] << ")";
    if (i < vec.size() - 1) {
      std::cout << ",\n";
    }
  }
  std::cout << "]" << std::endl;
}
// Function to print a set of integers
template<typename T>
[[maybe_unused]] void print_set(const std::set<T>& mySet) {
  for (const auto elem : mySet) {
    std::cout << elem << " ";
  }
  std::cout << std::endl;
}
template<typename T> size_t size(T&x) {
    return std::visit([](auto &&arg) -> size_t { return arg.size(); },
             x);
}
template<typename scalar_t> struct Welford{
  scalar_t mean_ = 0.0, sd_ = 1.0;
  size_t i = 1;
  Welford() : mean_(0.0), sd_(1.0), i(1) {}
  void operator()(const scalar_t x) {
    const auto prev = x - mean_;
    mean_ += (prev)/static_cast<scalar_t>(i++);
    const auto curr = x - mean_;
    sd_ += prev * curr;
  }
  scalar_t mean() const { return mean_;}
  scalar_t var() const  { return (i > 1) ? sd_ / (i - 1) : 0.0; }
  scalar_t sdev() const { return std::sqrt(var());}
};
template<typename scalar_t>
[[maybe_unused]] scalar_t
pearson_correlation(const std::vector<scalar_t>& x,
                    const std::vector<scalar_t>& y) {
  const size_t n = x.size();
  std::array<Welford<scalar_t>,2> s;
  scalar_t temp = 0.0;
  for(size_t i = 0; i < n; i++) {
    temp += x[i] * y[i];
    s[0](x[i]);
    s[1](y[i]);
  }
  return (temp - n * s[0].mean() * s[1].mean())/
         ((n-1)*s[0].sdev() * s[1].sdev());
}
template<typename scalar_t>
std::vector<scalar_t> rank(const std::vector<scalar_t>& values) {
  const size_t n = values.size();
  std::vector<size_t> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&values](size_t i1, size_t i2) {
    return values[i1] < values[i2];
  });
  std::vector<scalar_t> ranks(n);
  for (size_t i = 0; i < n;) {
    size_t start = i;
    while (i < n && values[indices[start]] == values[indices[i]]) i++;
    scalar_t rank_ = (static_cast<scalar_t>(start + i - 1)) / 2.0 + 1.0;
    for (size_t j = start; j < i; ++j) {
      ranks[indices[j]] = rank_;
    }
  }
  return ranks;
}
template<typename scalar_t>
[[maybe_unused]] scalar_t
spearman_correlation(const std::vector<scalar_t>& x,
                     const std::vector<scalar_t>& y) {
  const size_t n = x.size();
  std::vector<scalar_t> x_rank = rank(x);
  std::vector<scalar_t> y_rank = rank(y);
  scalar_t d2 = 0.0;
  for (size_t i = 0; i < n; ++i) {
    d2 += std::pow(x_rank[i] - y_rank[i], 2);
  }
  return 1.0 - (6.0 * d2) / (n * (n * n - 1));
}
template<typename scalar_t>
[[maybe_unused]] scalar_t
rmse(const std::vector<scalar_t>& x,
     const std::vector<scalar_t>& y) {
  const size_t n = x.size();
  scalar_t res = 0.0;
  for(size_t i = 0; i < n; i++) {
    res += std::pow(x[i]-y[i],2);
  }
  res /= static_cast<scalar_t>(n);
  return std::sqrt(res);
}
template<typename RNG>
std::vector<size_t> bootstrap_sample(const std::vector<size_t>& indices,
                                     const size_t sample_size,
                                     RNG &rng) {
  std::uniform_int_distribution<size_t> dist(0, indices.size() - 1);
  std::vector<size_t> bootstrap_indices = indices;
  std::shuffle(bootstrap_indices.begin(), bootstrap_indices.end(), rng);
  bootstrap_indices.resize(sample_size);
  return bootstrap_indices;
}
template<typename RNG>
auto bootstrap_two_samples(const std::vector<size_t>& indices,
                           const size_t sample_size,
                           RNG &rng) {
  std::uniform_int_distribution<size_t> dist(0, indices.size() - 1);
  std::vector<size_t> bootstrap_indices = indices;
  std::shuffle(bootstrap_indices.begin(), bootstrap_indices.end(), rng);
  auto second_sample = std::vector<size_t>(sample_size);
  for(size_t i = sample_size; i< 2*sample_size; i++) {
    second_sample[i-sample_size] = bootstrap_indices[i];
  }
  bootstrap_indices.resize(sample_size);
  return std::pair(bootstrap_indices, second_sample);
}
std::vector<size_t> make_index_range(const size_t size) {
  std::vector<size_t> indices(size);
  std::iota(indices.begin(), indices.end(), 0);
  return indices;
}
template<typename T> struct value_type_extractor;
template<template<typename...> class Container, typename Element>
struct value_type_extractor<Container<Element>> {
  using type [[maybe_unused]] = Element;
};
// Specialization for std::array<T, N>
template<typename T, std::size_t N>
struct value_type_extractor<std::array<T, N>> {
  using type [[maybe_unused]] = T;
};
// Specialization for std::pair
template<typename T>
struct value_type_extractor<std::pair<T, T>> {
  using type [[maybe_unused]] = std::tuple<T, T>;
};
template<typename First, typename Second>
struct value_type_extractor<std::pair<First, Second>> {
  using type [[maybe_unused]] = First;
};
// Specialization for std::tuple
template<typename... Elements>
struct value_type_extractor<std::tuple<Elements...>> {
  using type [[maybe_unused]] = std::tuple<Elements...>;
};
template <typename T>
using value_type_t = typename value_type_extractor<T>::type;
}
namespace threading {
// Class that represents a simple thread pool
class [[maybe_unused]] ThreadPool {
  std::vector<std::thread> threads_;
  std::queue<std::function<void()> > tasks_;
  std::mutex queue_mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
public:
  explicit ThreadPool(const size_t num_threads
             = std::thread::hardware_concurrency())
  {
    for (size_t i = 0; i < num_threads; ++i) {
      threads_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] {
              return !tasks_.empty() || stop_;
            });
            if (stop_ && tasks_.empty()) {
              return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }
  ~ThreadPool()
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& thread : threads_) {
      thread.join();
    }
  }
  [[maybe_unused]] void enqueue(std::function<void()> task)
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      tasks_.emplace(std::move(task));
    }
    cv_.notify_one();
  }
};
template<typename SharedResource>
struct [[maybe_unused]] SharedResourceWrapper{
  mutable std::mutex m_mutex;
  // make sure resource and mutex are actually initialized
  SharedResource resource;
  // Default constructor
  SharedResourceWrapper() = default;
  explicit SharedResourceWrapper(SharedResource resource) : resource(std::move(resource)) {}
  SharedResourceWrapper(const SharedResourceWrapper&) = delete;
  SharedResourceWrapper& operator=(const SharedResourceWrapper&) = delete;
  SharedResourceWrapper(SharedResourceWrapper&&) noexcept = default;
  SharedResourceWrapper& operator=(SharedResourceWrapper&&) noexcept = default;
  template<typename... Args>
  auto operator()(Args&&... args) -> decltype(resource(std::forward<Args>(args)...)) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return resource(std::forward<Args>(args)...);
  }
  template<typename Index>
  auto operator[](Index&& index) -> decltype(resource[std::forward<Index>(index)]) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return resource[std::forward<Index>(index)];
  }
  template<typename Index>
  auto operator[](Index&& index) const -> decltype(resource[std::forward<Index>(index)]) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return resource[std::forward<Index>(index)];
  }
  // Proxy that locks mutex and forwards calls to the resource
  class Proxy {
    std::lock_guard<std::mutex> lock;
    SharedResource& resource;
  public:
    Proxy(std::mutex& mtx, SharedResource& res)
        : lock(mtx), resource(res) {}
    SharedResource* operator->() {
      return &resource;
    }
  };
  // Operator-> to create and return the proxy
  Proxy operator->() {
    return Proxy(m_mutex, resource);
  }
  // enables serialization of objects via buffer if resource is a buffer
  // this is not entirely the cleanest, but it will suffice
  template<typename T> void serialize(T& x) {
    std::lock_guard<std::mutex> lock(m_mutex);
    x.serialize(resource);
  }
};
}
namespace containers {
template <const size_t MaxCategoricalSet,
    typename integral_t> class FixedCategoricalSet {
public:
  std::array<integral_t, MaxCategoricalSet> data{};
  size_t size = 0;
  [[nodiscard]] bool contains(const size_t value) const noexcept {
    return std::find(data.begin(), data.begin() + size, value) !=
           data.begin() + size;
  }
  void add(const integral_t value) noexcept {
    if (size < MaxCategoricalSet) {
      data[size++] = value;
    }
  }
  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(data.data()), sizeof(integral_t) * size);
  }
  void deserialize(std::istream& is) {
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    is.read(reinterpret_cast<char*>(data.data()), sizeof(integral_t) * size);
  }
};
// specialization for empty sets; this should be something the compiler can
// entirely optimize away
template <typename integral_t>
class FixedCategoricalSet<0, integral_t> {
public:
  FixedCategoricalSet() = default;
  [[nodiscard]] bool contains(const size_t value) const noexcept {return false;}
  void add(const integral_t value) noexcept {}
  void serialize(std::ostream& os) const {}
  void deserialize(std::istream& is) {}
};
template <typename integral_t>
class FixedCategoricalSet<1, integral_t> {
private:
  integral_t data{};
  bool occupied = false;
public:
  FixedCategoricalSet() = default;
  [[nodiscard]] bool contains(const size_t value) const noexcept {
    return occupied && data == value;
  }
  void add(const integral_t value) noexcept {
    if (!occupied) {
      data = value;
      occupied = true;
    }
  }
  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&occupied), sizeof(occupied));
    if (occupied) {
      os.write(reinterpret_cast<const char*>(&data), sizeof(data));
    }
  }
  void deserialize(std::istream& is) {
    is.read(reinterpret_cast<char*>(&occupied), sizeof(occupied));
    if (occupied) {
      is.read(reinterpret_cast<char*>(&data), sizeof(data));
    }
  }
};
template <typename scalar_t, typename integral_t,
          const size_t MaxCategoricalSet> struct Node {
  size_t featureIndex, terminal_index, left = 0, right = 0;
  bool missing_goes_right;
  std::variant<scalar_t, FixedCategoricalSet<MaxCategoricalSet, integral_t>> threshold;
  Node()
      : featureIndex(std::numeric_limits<size_t>::max()), terminal_index(0),
        missing_goes_right(false) {}
  [[nodiscard]] bool isCategorical() const noexcept {
    return std::holds_alternative<FixedCategoricalSet<MaxCategoricalSet, integral_t>>(
        threshold);
  }
  const FixedCategoricalSet<MaxCategoricalSet, integral_t> &
  getCategoricalSet() const noexcept {
    return std::get<FixedCategoricalSet<MaxCategoricalSet, integral_t>>(threshold);
  }
  scalar_t getNumericThreshold() const noexcept {
    return std::get<scalar_t>(threshold);
  }
  [[nodiscard]] bool getMIADirection() const noexcept {
    return missing_goes_right;
  }
  void assign_terminal(const size_t index) { terminal_index = index; }
  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&featureIndex), sizeof(featureIndex));
    os.write(reinterpret_cast<const char*>(&terminal_index), sizeof(terminal_index));
    os.write(reinterpret_cast<const char*>(&left), sizeof(left));
    os.write(reinterpret_cast<const char*>(&right), sizeof(right));
    os.write(reinterpret_cast<const char*>(&missing_goes_right), sizeof(missing_goes_right));

    bool is_categorical = isCategorical();
    os.write(reinterpret_cast<const char*>(&is_categorical), sizeof(is_categorical));
    if (is_categorical) {
      getCategoricalSet().serialize(os);
    } else {
      scalar_t threshold_value = getNumericThreshold();
      os.write(reinterpret_cast<const char*>(&threshold_value), sizeof(threshold_value));
    }
  }
  void deserialize(std::istream& is) {
    is.read(reinterpret_cast<char*>(&featureIndex), sizeof(featureIndex));
    is.read(reinterpret_cast<char*>(&terminal_index), sizeof(terminal_index));
    is.read(reinterpret_cast<char*>(&left), sizeof(left));
    is.read(reinterpret_cast<char*>(&right), sizeof(right));
    is.read(reinterpret_cast<char*>(&missing_goes_right), sizeof(missing_goes_right));

    bool is_categorical;
    is.read(reinterpret_cast<char*>(&is_categorical), sizeof(is_categorical));
    if (is_categorical) {
      FixedCategoricalSet<MaxCategoricalSet, integral_t> categorical_set;
      categorical_set.deserialize(is);
      threshold = categorical_set;
    } else {
      scalar_t threshold_value;
      is.read(reinterpret_cast<char*>(&threshold_value), sizeof(threshold_value));
      threshold = threshold_value;
    }
  }
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
template <typename Key, typename Value>
class FixedSizeMap<Key, Value, 0> {
public:
  FixedSizeMap() = default;
  FixedSizeMap<Key, Value, 0>(const FixedSizeMap<Key, Value, 0>&) = default;
  [[maybe_unused]] void add(const Key&, const Value&) {}
  [[nodiscard]] size_t size() const { return 0; }
};
template <typename ResultType>
struct TreePredictionResult {
  std::vector<size_t> indices;
  std::vector<std::pair<std::pair<size_t, size_t>, ResultType>> results;
  [[maybe_unused]] [[nodiscard]]
  size_t result_size() const {
    return results.size();
  }
  [[maybe_unused]] std::tuple<size_t, size_t, ResultType>
      get_result_view(const size_t which_result) const {
    return {results[which_result].first.first,
            results[which_result].first.second,
            results[which_result].second
           };
  }
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
namespace serializers{
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>> serialize(const T& value, std::ostream& os) {
  os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>> deserialize(T& value, std::istream& is) {
  is.read(reinterpret_cast<char*>(&value), sizeof(value));
}
template <typename T>
void serialize(const std::vector<T>& vec, std::ostream& os) {
  size_t size = vec.size();
  os.write(reinterpret_cast<const char*>(&size), sizeof(size));
  for (const auto& element : vec) {
    serialize(element, os);
  }
}
template <typename T>
void deserialize(std::vector<T>& vec, std::istream& is) {
  size_t size;
  is.read(reinterpret_cast<char*>(&size), sizeof(size));
  vec.resize(size);
  for (auto& element : vec) {
    deserialize(element, is);
  }
}
template <typename T, size_t N>
void serialize(const std::array<T, N>& arr, std::ostream& os) {
  for (const auto& element : arr) {
    serialize(element, os);
  }
}
template <typename T, size_t N>
void deserialize(std::array<T, N>& arr, std::istream& is) {
  for (auto& element : arr) {
    deserialize(element, is);
  }
}
template <typename ResultType>
void serialize(const containers::TreePredictionResult<ResultType>& result, std::ostream& os) {
  serialize(result.indices, os);
  size_t size = result.results.size();
  os.write(reinterpret_cast<const char*>(&size), sizeof(size));
  for (const auto& pair : result.results) {
    serialize(pair.first.first, os);
    serialize(pair.first.second, os);
    serialize(pair.second, os);
  }
}

template <typename ResultType>
void deserialize(containers::TreePredictionResult<ResultType>& result, std::istream& is) {
  deserialize(result.indices, is);
  size_t size;
  is.read(reinterpret_cast<char*>(&size), sizeof(size));
  result.results.resize(size);
  for (auto& pair : result.results) {
    deserialize(pair.first.first, is);
    deserialize(pair.first.second, is);
    deserialize(pair.second, is);
  }
}
}
namespace splitters {
template <typename RNG, typename scalar_t = double,
          typename integral_t = size_t,
          const size_t MaxCategoricalSet = 10>
class [[maybe_unused]] ExtremelyRandomizedStrategy {
public:
  void select_split(
      const std::vector<
          std::variant<std::vector<integral_t>, std::vector<scalar_t>>> &data,
      std::vector<size_t> &indices,
      const std::vector<size_t> &available_features, size_t start, size_t end,
      containers::Node<scalar_t, integral_t, MaxCategoricalSet> &node, RNG &rng) const {
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
    const std::vector<std::variant<std::vector<integral_t>, std::vector<scalar_t>>>
        &data;
    const size_t feature;
    std::vector<size_t> &indices;
    size_t start;
    size_t end;
    RNG &rng;

    std::pair<bool, std::variant<scalar_t, containers::FixedCategoricalSet<MaxCategoricalSet, integral_t>>>
    operator()(const std::vector<scalar_t> &) const noexcept {
      return {selectMIADirection(), selectRandomThreshold()};
    }

    std::pair<bool, std::variant<scalar_t, containers::FixedCategoricalSet<MaxCategoricalSet, integral_t>>>
    operator()(const std::vector<integral_t> &) const noexcept {
      return {selectMIADirection(), selectRandomCategoricalSet()};
    }

    scalar_t selectRandomThreshold() const noexcept {
      const auto &numericData = std::get<std::vector<scalar_t>>(data[feature]);
      // compute min and max in a single pass
      scalar_t min_val = std::numeric_limits<scalar_t>::max();
      scalar_t max_val = std::numeric_limits<scalar_t>::lowest();
      for (size_t i = start; i < end; ++i) {
        scalar_t val = numericData[indices[i]];
        if (!std::isnan(val)) {
          min_val = std::min(min_val, val);
          max_val = std::max(max_val, val);
        }
      }
      // sample uniformly between min and max
      std::uniform_real_distribution<scalar_t> dist(min_val, max_val);
      return dist(rng);
    }
    [[nodiscard]] bool selectMIADirection() const noexcept {
      return std::bernoulli_distribution(0.5)(rng);
    }

    containers::FixedCategoricalSet<MaxCategoricalSet, integral_t>
        selectRandomCategoricalSet() const noexcept {
      std::unordered_set<integral_t> uniqueValues;
      const auto &categoricalData =
          std::get<std::vector<integral_t>>(data[feature]);
      for (size_t i = start; i < end; ++i) {
        uniqueValues.insert(categoricalData[indices[i]]);
      }
      const size_t n_selected = utils::select_from_range(uniqueValues.size(), rng);
      std::array<integral_t, MaxCategoricalSet> shuffle_temp;
      size_t i = 0;
      for (const auto &val : uniqueValues) {
        shuffle_temp[i++] = val;
      }
      std::shuffle(shuffle_temp.begin(), shuffle_temp.begin() + i, rng);
      containers::FixedCategoricalSet<MaxCategoricalSet, integral_t> categoricalSet;
      for (i = 0; i < n_selected; ++i) {
        categoricalSet.add(shuffle_temp[i]);
      }
      return categoricalSet;
    }
  };
};

template <typename RNG, typename scalar_t = double,
          typename integral_t = size_t,
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
      containers::Node<scalar_t, integral_t, MaxCategoricalSet> &node,
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
    const std::vector<std::variant<std::vector<integral_t>, std::vector<scalar_t>>>
        &data;
    const size_t feature, start, end;
    std::vector<size_t> &indices;

    std::tuple<double, scalar_t>
    operator()(const std::vector<scalar_t> &) const noexcept {
      return findBestNumericSplit();
    }

    std::tuple<double, scalar_t>
    operator()(const std::vector<integral_t> &) const noexcept {
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
          std::get<std::vector<integral_t>>(data[feature]);
      std::vector<std::pair<double, integral_t>> indexedEncodedValues(end - start);

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
               containers::FixedSizeMap<size_t, integral_t, MaxCategoricalSet>>
    computeCategoryMeans() const noexcept {
      containers::FixedSizeMap<size_t, double, MaxCategoricalSet> categorySums;
      containers::FixedSizeMap<size_t, integral_t, MaxCategoricalSet> categoryCounts;

      const auto &categoricalData =
          std::get<std::vector<integral_t>>(data[feature]);

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
namespace reducers{
template<typename scalar_t> struct defaultImportanceReducer {
  scalar_t operator()(const std::vector<std::pair<scalar_t, size_t>>& x) {
    scalar_t sum = 0.;
    size_t size = 0;
    for(const auto& val : x) {
      sum += val.first * static_cast<scalar_t>(val.second);
      size += val.second;
    }
    return sum/static_cast<scalar_t>(size);
  }
  std::vector<scalar_t> operator()(
      const std::vector<std::pair<std::vector<scalar_t>, size_t>>& x) {
    std::vector<scalar_t> sums(x[0].first.size(), 0.);
    size_t sum_index = 0;
    for(auto& sum : sums) {
      size_t size = 0;
      for(const auto& val : x) {
        sum += val.first[sum_index] * static_cast<scalar_t>(val.second);
        size += val.second;
      }
      sum /= static_cast<scalar_t>(size);
      sum_index++;
    }
    return sums;
  }
};
}
template <typename ResultF, typename RNG, typename SplitStrategy,
          const size_t MaxCategoricalSet = 32, typename scalar_t = double,
          typename integral_t = size_t>
class RandomTree {
  using FeatureData = std::variant<
      std::vector<integral_t>,
      std::vector<scalar_t>>;
  using ResultType = decltype(std::declval<ResultF>()(
      std::declval<std::vector<size_t> &>(), std::declval<size_t>(),
      std::declval<size_t>(),
      std::declval<const std::vector<FeatureData> &>()));

  std::vector<containers::Node<scalar_t, integral_t, MaxCategoricalSet>> nodes;
  const size_t maxDepth, minNodesize;
  RNG &rng;
  ResultF &terminalNodeFunc;
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
                             std::vector<size_t> &indices) const noexcept {
    if constexpr (FlattenResults) {
      std::vector<utils::value_type_t<ResultType>> flat_result;
      predictSamples<true>(0, samples, indices, 0, indices.size(), flat_result);
      return flat_result;
    } else {
      containers::TreePredictionResult<ResultType> result;
      predictSamples<false>(0, samples, indices, 0, indices.size(), result);
      // N.B.: indices get shuffled, so they should only be assigned
      // once, here, AFTER being shuffled
      result.indices = indices;
      return result;
    }
  }
  template <const bool FlattenResults = false>
  [[nodiscard]] auto predict(const std::vector<FeatureData> &samples)
      const noexcept {
    std::vector<size_t> indices = treeson::utils::make_index_range(std::visit(
        [](auto &&arg) -> size_t { return arg.size(); }, samples[0]));
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
  [[maybe_unused]] std::set<size_t> used_features() const {
    std::set<size_t> features;
    for (const auto& node : nodes) {
        features.insert(node.featureIndex);
    }
    return features;
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
  template<typename Metric, typename Reducer = treeson::reducers::defaultImportanceReducer<scalar_t>>
  [[maybe_unused]] auto eval_oob(
      const std::vector<FeatureData> &oob_data,
      std::vector<size_t> &indices) {
    using MetricResultType = decltype(std::declval<Metric>()(
      std::declval<ResultType&>(), std::declval<const std::vector<FeatureData> &>(),
      std::declval<const std::vector<size_t>&>(),
      std::declval<const size_t>(), std::declval<const size_t>()));
    std::vector<std::pair<MetricResultType, size_t>> results;
    eval_oob_impl<Metric,false>(
        0, oob_data, indices, 0, indices.size(), results, 0);
    return Reducer()(results);
  }
  template<typename Metric, typename Reducer = treeson::reducers::defaultImportanceReducer<scalar_t>>
  [[maybe_unused]] auto eval_oob(
      const std::vector<FeatureData> &oob_data,
      std::vector<size_t> &indices,
      const size_t excluded_feature) {
    using MetricResultType = decltype(std::declval<Metric>()(
      std::declval<ResultType&>(), std::declval<const std::vector<FeatureData> &>(),
      std::declval<const std::vector<size_t>&>(),
      std::declval<const size_t>(), std::declval<const size_t>()));
    std::vector<std::pair<MetricResultType, size_t>> results;
    eval_oob_impl<Metric,true>(
        0, oob_data, indices, 0, indices.size(), results, excluded_feature);
    return Reducer()(results);
  }
  void from_nodes(
      std::vector<containers::Node<scalar_t, integral_t,
                                   MaxCategoricalSet>> &&nodes,
      std::vector<ResultType>&& terminal_values){
      this->nodes = nodes;
      this->terminal_values = terminal_values;
    }
    [[maybe_unused]] void save(const std::string &filename) {
      std::ofstream out(filename + ".bin", std::ios::binary);
      if (out) serialize(out);
      out.close();
    }
    [[maybe_unused]] void load(const std::string &filename) {
      std::ifstream in(filename + ".bin", std::ios::binary);
      if (in) {
        auto [deserialized_nodes, deserialized_terminal_values] =
            deserialize(in);
        from_nodes(std::move(deserialized_nodes),
                   std::move(deserialized_terminal_values));
      }
      in.close();
    }
    void serialize(std::ostream& os) const {
      size_t size = nodes.size();
      os.write(reinterpret_cast<const char*>(&size), sizeof(size));
      for (const auto& node : nodes) {
        node.serialize(os);
      }
      size = terminal_values.size();
      os.write(reinterpret_cast<const char*>(&size), sizeof(size));
      for (const auto& value : terminal_values) {
        serializers::serialize(value, os);
      }
    }
    std::pair<std::vector<containers::Node<scalar_t, integral_t, MaxCategoricalSet>>,
              std::vector<ResultType>> deserialize(std::istream& is) const {
      size_t size;
      is.read(reinterpret_cast<char*>(&size), sizeof(size));
      std::vector<containers::Node<scalar_t, integral_t, MaxCategoricalSet>> deserialized_nodes(size);
      for (auto& node : deserialized_nodes) {
        node.deserialize(is);
      }
      is.read(reinterpret_cast<char*>(&size), sizeof(size));
      std::vector<ResultType> deserialized_terminal_values(size);
      for (auto& value : deserialized_terminal_values) {
        serializers::deserialize(value, is);
      }
      return std::make_pair(std::move(deserialized_nodes), std::move(deserialized_terminal_values));
    }
  private:
  void buildTree(const std::vector<FeatureData> &data,
                 std::vector<size_t> &indices,
                 const std::vector<size_t> &available_features,
                 size_t parentIndex, size_t start, size_t end,
                 size_t depth = 0) noexcept {
    nodes.emplace_back();
    size_t nodeIndex = nodes.size() - 1;
    if (nodes[parentIndex].left == 0) {
      nodes[parentIndex].left = nodeIndex; // Left child
    } else {
      nodes[parentIndex].right = nodeIndex; // Right child
    }
    split_strategy.select_split(data, indices, available_features, start, end,
                                nodes[nodeIndex], rng);
    size_t mid = reorder_indices(data, nodes[nodeIndex], start, end, indices);
    // if a child node would be too small
    if (((mid - start) <= minNodesize) || ((end - mid) <= minNodesize) ||
        depth >= maxDepth) {
      // get rid of last node since it was actually invalid
      terminal_values.push_back(terminalNodeFunc(indices, start, end, data));
      nodes[nodeIndex].assign_terminal(terminal_values.size() - 1);
    } else {
      buildTree(data, indices, available_features, nodeIndex, start, mid,
                depth + 1);
      buildTree(data, indices, available_features, nodeIndex, mid, end,
                depth + 1);
    }
  }
  template<typename Metric, const bool exclude_feature = false> [[nodiscard]]
  auto eval_oob_impl(
      size_t nodeIndex, const std::vector<FeatureData> &oob_data,
      std::vector<size_t> &indices, size_t start, size_t end,
      std::vector<std::pair<decltype(std::declval<Metric>()(
                                    std::declval<ResultType&>(), std::declval<const std::vector<FeatureData> &>(),
                                    std::declval<const std::vector<size_t>&>(),
                                    std::declval<const size_t>(), std::declval<const size_t>())),size_t>>& results,
      const size_t excluded_feature) const noexcept {
        const auto &node = nodes[nodeIndex];
        if (node.left == 0 && node.right == 0) {
          const auto& vals = terminal_values[node.terminal_index];
          results.push_back(std::pair(Metric()(vals, oob_data, indices, start, end), end-start));
          return;
        }
        if constexpr(exclude_feature) {
          if (nodes[nodeIndex].featureIndex == excluded_feature) {
            // consider that this probably returns only per one of these branches
            eval_oob_impl<Metric, true>(
                node.left, oob_data,
                indices, start, end, results, excluded_feature);
            eval_oob_impl<Metric, true>(
                node.right, oob_data,
                indices, start, end, results, excluded_feature);
          }
        }
        auto mid = reorder_indices(
            oob_data, nodes[nodeIndex], start, end, indices);
        if (mid > start) {
          eval_oob_impl<Metric, false>(
              node.left, oob_data,
              indices, start, mid, results, excluded_feature);
        }
        if (mid < end) {
          eval_oob_impl<Metric, false>(
              node.right, oob_data,
              indices, mid, end, results, excluded_feature);
        }
    }
  void print_node(size_t node_index, size_t depth) const noexcept {
    const auto &node = nodes[node_index];
    if (depth >= maxDepth) {
      return;
    }
    std::string indent(depth * 4, '-');
    if (node.left == 0) {
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
    print_node(node.left, depth + 1);
    print_node(node.right, depth + 1);
  }
  template <const bool FlattenResults, typename ResultContainer>
  void predictSamples(size_t nodeIndex, const std::vector<FeatureData> &samples,
                      std::vector<size_t> &indices, const size_t start, const size_t end,
                      ResultContainer &predictionResult) const noexcept {
    const auto &node = nodes[nodeIndex];
    if (node.left == 0) {
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
    auto mid = reorder_indices(samples, nodes[nodeIndex], start, end, indices);
    if (mid > start) {
      predictSamples<FlattenResults>(
          node.left, samples,
          indices, start, mid, predictionResult);
    }
    if (mid < end) {
      predictSamples<FlattenResults>(
          node.right, samples,
          indices, mid, end, predictionResult);
    }
  }
  size_t reorder_indices(const std::vector<FeatureData>& data,
                         const treeson::containers::Node<scalar_t, integral_t, MaxCategoricalSet>& node,
                         const size_t start, const size_t end, std::vector<size_t>& indices) const {
    auto& featureData = data[node.featureIndex];
    const auto miss_dir = node.getMIADirection();
    size_t mid = start;
    if (node.isCategorical()) {
      const auto &catSet = node.getCategoricalSet();
      const auto &catData = std::get<std::vector<integral_t>>(featureData);
      for (size_t i = start; i < end; ++i) {
        const auto val = catData[indices[i]];
        if (catSet.contains(val)) {
          std::swap(indices[i], indices[mid]);
          ++mid;
        }
      }
    } else {
      const auto &numThreshold = node.getNumericThreshold();
      const auto &numData = std::get<std::vector<scalar_t>>(featureData);
      for (size_t i = start; i < end; ++i) {
        const auto val = numData[indices[i]];
        if ((val <= numThreshold) || (std::isnan(val) && miss_dir)) {
          std::swap(indices[i], indices[mid]);
          ++mid;
        }
      }
    }
    return mid;
  }
};
template <typename ResultF, typename RNG, typename SplitStrategy,
          const size_t MaxCategoricalSet = 32, typename scalar_t = double,
          typename integral_t = size_t>
class [[maybe_unused]] RandomForest {
  using FeatureData = std::variant<std::vector<integral_t>, std::vector<scalar_t>>;
  using TreeType = RandomTree<ResultF, RNG, SplitStrategy, MaxCategoricalSet,
                              scalar_t, integral_t>;
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
    size_t bootstrap_size =
        sample_size > 0
            ? sample_size
            : utils::size(data[0]);
    const bool resample_ = resample && sample_size > 0;
    if (resample != resample_) {
      std::cout << "You specified 'resample = true', "
                << "but provided 'sample_size = 0'. Quitting, respecify."
                << std::endl;
      return;
    }
#ifndef NO_MULTITHREAD
    threading::SharedResourceWrapper<std::vector<TreeType>> trees_;
#endif
    // N.B.: Outer scope forces thread pool to finish before we do anything
    {
#ifndef NO_MULTITHREAD
      threading::ThreadPool pool;
      // required for thread safety
#endif
      for (size_t i = 0; i < n_tree; ++i) {
#ifdef NO_MULTITHREAD
        [this, &data, &nosplit_features, resample_, bootstrap_size]() {
#else
        const size_t seed = rng();
        pool.enqueue([this, &data, &trees_, &nosplit_features, resample_, seed,
                      bootstrap_size] {
#endif
          std::vector<size_t> indices =
              utils::make_index_range(utils::size(data[0]));
          // custom random number generator for this tree
#ifndef NO_MULTITHREAD
          RNG rng_(seed);
#else
          auto& rng_ = rng;
#endif
          TreeType tree(maxDepth, minNodesize, rng_, terminalNodeFunc,
                        strategy);
          if (resample_) {
            tree.fit(
                data,
                treeson::utils::bootstrap_sample(indices, bootstrap_size, rng_),
                nosplit_features);
          } else {
            tree.fit(data, indices, nosplit_features);
          }
          /* N.B.: for thread safety this is invoked via a proxy; be not afraid,
             but do not try to modify the implementation unless you know
             what is going on */
#ifndef NO_MULTITHREAD
          trees_->push_back(tree);
#else
          trees.push_back(tree);
#endif
        }
#ifdef NO_MULTITHREAD
        ();
#else
        );
#endif
      }
    }
#ifndef NO_MULTITHREAD
    // move to actual trees
    trees = std::move(trees_.resource);
#endif
  }
  [[maybe_unused]] void fit(const std::vector<FeatureData> &data,
                            const size_t n_tree,
                            const std::vector<size_t> &nosplit_features,
                            const std::string &file,
                            const bool resample = true,
                            const size_t sample_size = 0) {
    size_t bootstrap_size =
        sample_size > 0
            ? sample_size
            : utils::size(data[0]);
    const bool resample_ = resample && sample_size > 0;
    if (resample != resample_) {
      std::cout << "You specified 'resample = true', "
                << "but provided 'sample_size = 0'. Quitting, respecify."
                << std::endl;
      return;
    }
    std::ofstream out(file + ".bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(&n_tree), sizeof(n_tree));
#ifndef NO_MULTITHREAD
    threading::SharedResourceWrapper<std::ofstream> out_(out);
#endif
    // N.B.: Outer scope forces thread pool to finish before we do anything
    {
#ifndef NO_MULTITHREAD
      threading::ThreadPool pool;
      // required for thread safety
#endif
      for (size_t i = 0; i < n_tree; ++i) {
#ifdef NO_MULTITHREAD
        [this, &data, &nosplit_features, resample_, bootstrap_size, &out]() {
#else
        const size_t seed = rng();
        pool.enqueue([this, &data, &trees_, &nosplit_features, resample_, seed,
                      bootstrap_size, &out_] {
#endif
          std::vector<size_t> indices =
              utils::make_index_range(utils::size(data[0]));
          // custom random number generator for this tree
#ifndef NO_MULTITHREAD
          RNG rng_(seed);
#else
          auto& rng_ = rng;
#endif
          TreeType tree(maxDepth, minNodesize, rng_, terminalNodeFunc,
                        strategy);
          if (resample_) {
            tree.fit(
                data,
                treeson::utils::bootstrap_sample(indices, bootstrap_size, rng_),
                nosplit_features);
          } else {
            tree.fit(data, indices, nosplit_features);
          }
#ifndef NO_MULTITHREAD
          out_.serialize(tree);
#else
          tree.serialize(out);
#endif
        }
#ifdef NO_MULTITHREAD
        ();
#else
        );
#endif
      }
    }
  }
  [[nodiscard]] std::vector<containers::TreePredictionResult<ResultType>> predict(
      const std::vector<FeatureData> &samples) const noexcept {
    const size_t n = trees.size();
    std::vector<containers::TreePredictionResult<ResultType>> results(n);
    {
#ifndef NO_MULTITHREAD
      threading::ThreadPool pool;
      threading::SharedResourceWrapper<
          std::vector<containers::TreePredictionResult<ResultType>>>
          results_(results);
#endif
      for (size_t i = 0; i < trees.size(); ++i) {
#ifdef NO_MULTITHREAD
        results[i] = trees[i].predict(samples);
#else
        pool.enqueue([this, &results_, &samples, &i] {
          results_[i] = trees[i].predict(samples);
        });
#endif
      }
    }
#ifndef NO_MULTITHREAD
    results = std:move(results_);
#endif
    return results;
  }
  [[maybe_unused]] [[nodiscard]]
  std::vector<containers::TreePredictionResult<ResultType>> predict(
    const std::vector<FeatureData> &samples,
    const std::string &model_file) const noexcept {
    std::ifstream in(model_file + ".bin", std::ios::binary);
    size_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<containers::TreePredictionResult<ResultType>> results(n);
    {
#ifndef NO_MULTITHREAD
      threading::ThreadPool pool;
      // shared resource to access the file
      threading::SharedResourceWrapper<std::ifstream> in_(in);
      threading::SharedResourceWrapper<
          std::vector<containers::TreePredictionResult<ResultType>>
          > results_(results);
#endif
      for (size_t i = 0; i < n; i++) {
#ifdef NO_MULTITHREAD
        TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc, strategy);
        auto [nodes, values] = tree.deserialize(in);
        tree.from_nodes(std::move(nodes),std::move(values));
        results[i] = tree.predict(samples);
#else
        pool.enqueue([this, &results_, &samples, &i] {
          TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc, strategy);
          auto [nodes, values] = tree.deserialize(in_);
          results_[i] = tree.from_nodes(std::move(nodes),
                                        std::move(values)).predict(samples);
        });
#endif
      }
    }
    return results;
  }
  /*
  [[nodiscard]] void predict(
      Accumulator& accumulator,
      const std::vector<FeatureData> &samples,
      const std::string &model_file) const noexcept {
    const size_t n = trees.size();
    std::vector<containers::TreePredictionResult<ResultType>> results(n);
#ifndef NO_MULTITHREAD
    threading::ThreadPool pool;
    threading::SharedResourceWrapper<Accumulator> accumulator_(accumulator);
#endif
    for (size_t i = 0; i < trees.size(); ++i) {
#ifdef NO_MULTITHREAD
      results[i] = trees[i].predict(samples);
#else
      pool.enqueue([this, &results, &samples, &i] {
        results[i] = trees[i].predict(samples);
      });
#endif
    }
  }
   */
  template<typename Accumulator>
  [[maybe_unused]] void memoryless_predict(
      Accumulator& accumulator,
      const std::vector<FeatureData> &train_data,
      const std::vector<FeatureData> &predict_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool resample = true,
      const size_t sample_size = 0) const noexcept {
    size_t bootstrap_size = sample_size > 0 ? sample_size :
      std::visit([](auto&& arg) -> size_t { return arg.size(); }, train_data[0]);
    const bool resample_ = resample && sample_size > 0;
    if(resample != resample_) {
      std::cout << "You specified 'resample = true', " <<
          "but provided 'sample_size = 0'. Quitting, respecify." << std::endl;
      return;
    }
    // do the lazy thing following L'Ecuyer
    // see 'Random Numbers for Parallel Computers: Requirements and Methods, With Emphasis on GPUs'
    // Pierre L’Ecuyer, David Munger, Boris Oreshkin, Richard Simard, p.15
    // 'A single RNG with a “random” seed for each stream.'
    // link: https://www.iro.umontreal.ca/~lecuyer/myftp/papers/parallel-rng-imacs.pdf
#ifndef NO_MULTITHREAD
    // ensure no data races
    threading::SharedResourceWrapper<Accumulator> accumulator_(accumulator);
#else
    auto& accumulator_ = accumulator;
#endif
    {
#ifndef NO_MULTITHREAD
      threading::ThreadPool pool;
#endif
      for (size_t i = 0; i < n_tree; ++i) {
#ifdef NO_MULTITHREAD
        [this, &train_data, &predict_data, &accumulator, &nosplit_features,
         resample_, bootstrap_size]() {
#else
        const size_t seed = rng();
        pool.enqueue([this, &train_data, &predict_data, &accumulator_,
                      &nosplit_features, resample_, seed, bootstrap_size] {
#endif
          std::vector<size_t> indices(std::visit(
              [](auto &&arg) -> size_t { return arg.size(); }, train_data[0]));
          std::iota(indices.begin(), indices.end(), 0);
#ifdef NO_MULTITHREAD
          auto &rng_ = rng;
#else
          RNG rng_(seed);
#endif
          TreeType tree(maxDepth, minNodesize, rng_, terminalNodeFunc,
                        strategy);
          if (resample_) {
            tree.fit(
                train_data,
                treeson::utils::bootstrap_sample(indices, bootstrap_size, rng_),
                nosplit_features);
          } else {
            tree.fit(train_data, indices, nosplit_features);
          }
          if (tree.is_uninformative()) {
            // make it clear that this tree does not count
            return;
          }
          std::vector<size_t> pred_indices =
              utils::make_index_range(utils::size(predict_data[0]));
          auto prediction_result =
              tree.template predict<utils::is_invocable_with<
                  Accumulator,
                  std::vector<utils::value_type_t<ResultType>>>::value>(
                  predict_data, pred_indices);
          accumulator_(prediction_result);
        }
#ifdef NO_MULTITHREAD
        ();
#else
        );
#endif
      }
    }
    // move to accumulator that was passed in
    accumulator = std::move(accumulator_.resource);
  }
  [[maybe_unused]] void prune() {
    trees.erase(
        std::remove_if(trees.begin(), trees.end(), [](const TreeType &tree) {
          return tree.is_uninformative();
        }),
        trees.end()
    );
  }
  // Feature importance method
  template<typename Accumulator, typename Metric, typename Importance>
  [[maybe_unused]] auto feature_importance(
      const std::vector<FeatureData> &train_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool oob = true,
      const size_t sample_size = 0) {
    size_t bootstrap_size = sample_size > 0 ? sample_size :
                                            utils::size(train_data[0]);
    // we do not allow resampling as such here because it complicates the interface
    // and leaves us in an equal 'in-bag error' situation as if sample size was just set
    // to total size.
    std::vector<size_t> indices = utils::make_index_range(utils::size(train_data[0]));
    /* N.B.: outer scope forces thread-pool to finish - important!!!
     without this we can proceed onto the importance stuff BEFORE we have
     any results, so this MUST be closed off.*/
#ifndef NO_MULTITHREAD
    std::vector<threading::SharedResourceWrapper<Accumulator>>
        accumulators(train_data.size() + 1);
#else
    std::vector<Accumulator> accumulators(train_data.size() + 1);
#endif
    /* N.B.: outer scope forces thread-pool to finish - important!!!
 without this we can proceed onto the importance stuff BEFORE we have
 any results, so this MUST be closed off.*/
    {
#ifndef NO_MULTITHREAD
      threading::ThreadPool pool;
#endif
      for (size_t i = 0; i < n_tree; ++i) {
#ifdef NO_MULTITHREAD
        [this, &train_data, &indices, &accumulators, &nosplit_features,
         bootstrap_size, oob]() {
#else
        const auto seed = rng();
        pool.enqueue([this, &train_data, &accumulators, &indices,
                      &nosplit_features, seed, bootstrap_size, oob] {
#endif
#ifdef NO_MULTITHREAD
          auto &rng_ = rng;
#else
          RNG rng_(seed);
#endif
          TreeType tree(maxDepth, minNodesize, rng_, terminalNodeFunc,
                        strategy);
          std::vector<size_t> train_indices, test_indices;
          if (oob) {
            std::tie(train_indices, test_indices) =
                treeson::utils::bootstrap_two_samples(indices, bootstrap_size,
                                                      rng_);
          } else {
            train_indices =
                treeson::utils::bootstrap_sample(indices, bootstrap_size, rng_);
            test_indices = train_indices;
          }
          tree.fit(train_data, train_indices, nosplit_features);
          if (tree.is_uninformative()) {
            // make it clear that this tree does not count
            return;
          }
          auto used_features = tree.used_features();
          // compute baseline - update first accumulator
          accumulators[0](
              tree.template eval_oob<Metric>(train_data, test_indices));
          // update all other accumulators
          for (const auto feature : used_features) {
            accumulators[feature + 1](tree.template eval_oob<Metric>(
                train_data, test_indices, feature));
          }
        }
#ifdef NO_MULTITHREAD
        ();
#else
        );
#endif
      }
    }
    // convert from accumulator values to importances
    Importance importance_;
#ifndef NO_MULTITHREAD
    // whatever the invocation over two accumulators produces :)
    std::vector<decltype(importance_(accumulators.front().resource.result(),
                                     accumulators.front().resource.result())
                             )> importances(accumulators.size()-1);
    const auto baseline_results = accumulators.front().resource.result();
    for (size_t feature_i = 0; feature_i < accumulators.size()-1; feature_i++) {
      importances[feature_i] = importance_(
          baseline_results, accumulators[feature_i+1].resource.result());
    }
#else
    // same as above, but sans .resource everywhere
    std::vector<decltype(importance_(accumulators.front().result(),
                                     accumulators.front().result())
                             )> importances(accumulators.size()-1);
    const auto baseline_results = accumulators.front().result();
    for (size_t feature_i = 0; feature_i < accumulators.size()-1; feature_i++) {
      importances[feature_i] = importance_(
          baseline_results, accumulators[feature_i+1].result());
    }
#endif
    return importances;
  }
  [[maybe_unused]] void print() {
    for(const auto& tree : trees)
      tree.print_tree_structure();
    std::cout << "\n";
  }
  void serialize(std::ostream &os) const {
    size_t size = trees.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto &tree : trees) {
      tree.serialize(os);
    }
  }
  void deserialize(std::istream &in) {
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    for(size_t i = 0; i < size; i++) {
      if(i < trees.size()) {
      auto [nodes, values] = trees[i].deserialize(in);
      trees[i].from_nodes(std::move(nodes), std::move(values));
      } // otherwise we are appending
      else{
        TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc,
                      strategy);
        auto [nodes, values] = tree.deserialize(in);
        tree.from_nodes(std::move(nodes), std::move(values));
        trees.push_back(tree);
      }
    }
  }
  [[maybe_unused]] void save(const std::string &filename) const {
    std::ofstream out(filename + ".bin", std::ios::binary);
    if (out) serialize(out);
    out.close();
  }
  [[maybe_unused]] void load(const std::string &filename) {
    std::ifstream in(filename + ".bin", std::ios::binary);
    if (in) deserialize(in);
    in.close();
  }
  [[maybe_unused]] void save(std::ofstream& out) const {
    if (out) serialize(out);
  }
  [[maybe_unused]] void load(std::ifstream& in) {
    if (in) deserialize(in);
  }
private:
  const size_t maxDepth, minNodesize;
  RNG& rng;
  ResultF& terminalNodeFunc;
  SplitStrategy& strategy;
  std::vector<TreeType> trees;
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

