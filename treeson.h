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
}
// Disable the specific warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
template <typename> static auto test(...) -> std::false_type {}
#pragma GCC diagnostic pop
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
template<typename T> [[maybe_unused]] void deduplicate(T& x) {
  size_t n = x.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; /* incremented in loop */) {
      if (x[i] == x[j]) {
        // Move the duplicate to the end
        std::swap(x[j], x[n-1]);
        // Reduce the size to exclude the duplicate
        --n;
      } else {
        ++j; // only increment if no duplicate found
      }
    }
  }
  // Resize the vector to exclude the duplicates at the end
  x.resize(n);
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
  // same for deserialization
  template<typename T> auto deserialize(T& x) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return x.deserialize(resource);
  }
};
size_t thread_heuristic(const size_t num_threads = 0) {
  const size_t max_threads = std::thread::hardware_concurrency();
  auto used_threads = (num_threads == 0 || (num_threads > max_threads)) ? max_threads : num_threads;
  return used_threads;
}
}
namespace memory{
template<typename... Ts>
class [[maybe_unused]] MemoryPool {
public:
  MemoryPool() = default;

  template<typename T>
  void emplace_back(T&& resource) {
    pool.push_back(std::forward<T>(resource));
    availabilities.push_back(true);
  }
  template<typename T>
  struct ResourceHandle {
    MemoryPool& pool;
    T& resource;
    std::size_t index;
    ResourceHandle(MemoryPool& pool, T& resource, std::size_t index)
        : pool(pool), resource(resource), index(index) {}
    ~ResourceHandle() { pool.release(index); }
    T* operator->() { return &resource; }
    T& operator*() { return resource; }
  };
  template<typename T> [[maybe_unused]] ResourceHandle<T> get() {
    std::unique_lock<std::mutex> lock(poolMutex);
    poolCondVar.wait(lock, [this]() { return has_available_resource<T>(); });
    std::size_t i = 0;
    for (; i < pool.size(); ++i) {
      if (!availabilities[i]) continue;
      if (std::holds_alternative<T>(pool[i])) {
        availabilities[i] = false;
        break;
      }
    }
    // this is outside the loop to prevent warnings about this function sometimes
    // not returning non-void
    return ResourceHandle<T>(*this, std::get<T>(pool[i]), i);
  }

private:
  template<typename T>
  [[nodiscard]] bool has_available_resource() const {
    for (std::size_t i = 0; i < pool.size(); ++i) {
      if (availabilities[i] && std::holds_alternative<T>(pool[i])) {
        return true;
      }
    }
    return false;
  }
  void release(std::size_t index) {
    {
      std::lock_guard<std::mutex> lock(poolMutex);
      availabilities[index] = true;
    }
    poolCondVar.notify_all();
  }
  std::vector<std::variant<Ts...>> pool;
  std::vector<bool> availabilities;
  std::mutex poolMutex;
  std::condition_variable poolCondVar;
};
// used for making objects with ad-hoc tags
template<typename T, const auto tag> struct [[maybe_unused]] Tagged {
  T item;
  [[maybe_unused]] explicit Tagged(T x) : item(std::move(x)){}
  [[maybe_unused]] constexpr auto get_tag() const {
    return tag;
  }
  template<typename... Args>
  auto operator()(Args&&... args) -> decltype(resource(std::forward<Args>(args)...)) {
    return item(std::forward<Args>(args)...);
  }
  template<typename Index>
  auto operator[](Index&& index) -> decltype(item[std::forward<Index>(index)]) {
    return item[std::forward<Index>(index)];
  }
  template<typename Index>
  auto operator[](Index&& index) const -> decltype(item[std::forward<Index>(index)]) {
    return item[std::forward<Index>(index)];
  }
  // Enable begin() if item has a begin()
  template<typename U = T>
  auto begin() -> decltype(std::declval<U&>().begin()) {
    return item.begin();
  }
  // Enable begin() const if item has a const begin()
  template<typename U = T>
  auto begin() const -> decltype(std::declval<const U&>().begin()) {
    return item.begin();
  }
  // Enable end() if item has an end()
  template<typename U = T>
  auto end() -> decltype(std::declval<U&>().end()) {
    return item.end();
  }
  // Enable end() const if item has a const end()
  template<typename U = T>
  auto end() const -> decltype(std::declval<const U&>().end()) {
    return item.end();
  }
  // Enable size() if item has a size()
  template<typename U = T>
  auto size() -> decltype(std::declval<U&>().size()) {
    return item.size();
  }
  // Enable size() const if item has a const size()
  template<typename U = T>
  auto size() const -> decltype(std::declval<const U&>().size()) {
    return item.size();
  }
  // Convert to underlying type to make this object behave like the wrapped type
  explicit operator T&() {
    return item;
  }
};
}
namespace containers {
template<typename T, const size_t ShortSize>
class [[maybe_unused]] ShortVector {
private:
  using arr = std::array<T, ShortSize>;
  using vec = std::vector<T>;
  std::variant<arr, vec> data{};
  size_t current_size;
  bool use_vector;

  void to_vec() {
    vec new_vector;
    new_vector.reserve(current_size);
    auto& array_data = std::get<arr>(data);
    for (size_t i = 0; i < current_size; ++i) {
      new_vector.push_back(std::move(array_data[i]));
    }
    data = std::move(new_vector);
    use_vector = true;
  }

public:
  ShortVector() : current_size(0), data(arr{}), use_vector(false) {}

  T& operator[](const size_t index) {
    if (use_vector) {
      return std::get<vec>(data)[index];
    }
    return std::get<arr>(data)[index];
  }

  const T& operator[](const size_t index) const {
    if (use_vector) {
      return std::get<vec>(data)[index];
    }
    return std::get<arr>(data)[index];
  }

  [[nodiscard]] size_t size() const { return current_size; }

  void push_back(const T& value) {
    if (current_size < ShortSize && !use_vector) {
      std::get<arr>(data)[current_size++] = value;
    } else {
      if (current_size == ShortSize && !use_vector) {
        to_vec();
      }
      std::get<vec>(data).push_back(value);
      current_size++;
    }
  }

  void push_back(T&& value) {
    if (current_size < ShortSize && !use_vector) {
      std::get<arr>(data)[current_size++] = std::move(value);
    } else {
      if (current_size == ShortSize && !use_vector) to_vec();
      std::get<vec>(data).push_back(std::move(value));
      current_size++;
    }
  }

  template<typename... Args>
  void emplace_back(Args&&... args) {
    if (current_size < ShortSize && !use_vector) {
      std::get<arr>(data)[current_size++] = T(std::forward<Args>(args)...);
    } else {
      if (current_size == ShortSize && !use_vector) {
        to_vec();
      }
      std::get<vec>(data).emplace_back(std::forward<Args>(args)...);
      current_size++;
    }
  }

  void resize(const size_t new_size) {
    if(new_size == current_size) return;
    if (new_size <= ShortSize && !use_vector) {
      if (current_size > ShortSize) {
        auto& vector_data = std::get<vec>(data);
        arr new_array;
        for (size_t i = 0; i < std::min(current_size, new_size); ++i) {
          new_array[i] = std::move(vector_data[i]);
        }
        data = std::move(new_array);
        use_vector = false;
      }
    } else {
      if (current_size <= ShortSize) {
        to_vec();
      }
      std::get<vec>(data).resize(new_size);
    }
    current_size = new_size;
  }

  using iterator = T*;
  using const_iterator = const T*;
  iterator begin() {
    if (use_vector) {
      return std::get<vec>(data).data();
    }
    return std::get<arr>(data).data();
  }
  iterator end() {
    if (use_vector) {
      return std::get<vec>(data).data() + current_size;
    }
    return std::get<arr>(data).data() + current_size;
  }
  [[nodiscard]] const_iterator begin() const {
    if (use_vector) {
      return std::get<vec>(data).data();
    }
    return std::get<arr>(data).data();
  }

  [[nodiscard]] const_iterator end() const {
    if (use_vector) {
      return std::get<vec>(data).data() + current_size;
    }
    return std::get<arr>(data).data() + current_size;
  }
  T& front() {
    if (use_vector) {
      return std::get<vec>(data).front();
    }
    return std::get<arr>(data).front();
  }
  [[maybe_unused]] [[nodiscard]] const T& front() const {
    if (use_vector) {
      return std::get<vec>(data).front();
    }
    return std::get<arr>(data).front();
  }
  T& back() {
    if (use_vector) {
      return std::get<vec>(data).back();
    }
    return std::get<arr>(data)[current_size - 1];
  }
  [[maybe_unused]] [[nodiscard]] const T& back() const {
    if (use_vector) {
      return std::get<vec>(data).back();
    }
    return std::get<arr>(data)[current_size - 1];
  }
};

template <const size_t MaxCategoricalSet,
    typename integral_t>
class FixedCategoricalSet {
public:
  std::array<integral_t, MaxCategoricalSet> data{};
  size_t size = 0;
  [[nodiscard]] bool contains(const size_t value) const noexcept {
    return std::find(data.begin(), data.begin() + size, value) !=
           data.begin() + size;
  }
  void add(const integral_t value) noexcept {
    if (size < MaxCategoricalSet && !contains(value)) {
      data[size++] = value;
    }
  }
  using iterator = integral_t*;
  using const_iterator = const integral_t*;
  iterator begin() {
    return data.data();
  }
  [[nodiscard]] const_iterator begin() const {
    return data.data();
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
  [[nodiscard]] bool contains(const size_t) const noexcept {return false;}
  void add(const integral_t) noexcept {}
  void serialize(std::ostream&) const {}
  void deserialize(std::istream&) {}
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
          const size_t MaxCategoricalSet>
struct Node {
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
  Value& operator[](const Key &key) {
    size_t idx = find(key);
    if (idx == size_) { // New key
      data[size_] = std::make_pair(key, Value{});
      ++size_;
    }
    return data[idx].second;
  }
  const Value& operator[](const Key &key) const {
    size_t idx = find(key);
    if (idx == size_) {
      throw std::out_of_range("Key not found!");
    }
    return data[idx].second;
  }
  bool contains(const Key &key) const {
    size_t idx = find(key);
    return idx != size_;
  }
  [[nodiscard]] size_t size() const { return size_; }
private:
  ShortVector<std::pair<Key, Value>, MaxSize> data;
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
namespace samplers{
template<typename T, typename scalar_t,
          const size_t MaxLandmarks>
struct [[maybe_unused]] LandmarkSampler{
  std::vector<T> vals;
  std::vector<scalar_t> weights;
  containers::ShortVector<size_t, MaxLandmarks> landmarks_id = {};
  containers::ShortVector<scalar_t, MaxLandmarks> landmarks_val = {};
  [[maybe_unused]] LandmarkSampler(
      std::vector<T>&& vals, std::vector<scalar_t>&& weights) :
        vals(std::move(vals)), weights(std::move(weights)) {
    scalar_t cumulative_weight = 0.0;
    size_t landmark_index = 0;
    const size_t n = this->weights.size();
    // select number of landmarks dynamically such that it is at most
    const size_t n_landmarks = std::floor(std::log2(static_cast<float>(n)))+1;
    if(n_landmarks > landmarks_id.size()) {
      landmarks_id.resize(n_landmarks);
      landmarks_val.resize(n_landmarks);
    }
    size_t step_size = n / n_landmarks;
    for (size_t i = 0; i < this->weights.size(); ++i) {
      cumulative_weight += this->weights[i];
      if (i % step_size == 0 && (landmark_index + 1) < n_landmarks) {
        landmarks_id[landmark_index] = i;
        landmarks_val[landmark_index++] = cumulative_weight;
      }
    }
    // Ensure the last position is included as a landmark
    landmarks_id[landmark_index] = n-1;
    landmarks_val[landmark_index] = cumulative_weight;
  }
  [[maybe_unused]] void update_weight(const size_t index, const scalar_t new_weight) {
    const auto weight_diff = new_weight - weights[index];
    weights[index] = new_weight;
    // update landmarks
    size_t i = 0;
    for(const auto& id : landmarks_id) {
      if(id >= index) {
        landmarks_val[i] += weight_diff;
      }
      i++;
    }
  }
  template<typename RNG> [[maybe_unused]] T sample(RNG& rng) {
    std::uniform_real_distribution<scalar_t> dist(0.0, landmarks_val.back());
    scalar_t smpl = dist(rng);
    // if the first landmark is greater, we can return early :)
    if(landmarks_val.front() > smpl) return vals[0];
    // Linear search on landmarks
    size_t i = 1;
    // if a landmark is greater we know we are somewhere in-between
    for(; i < landmarks_id.size(); i++) {
      // we found the top landmark
      if(landmarks_val[i] > smpl) break;
    }
    // Linear search within the sub-segment
    scalar_t cumulative_weight = landmarks_val[i-1];
    i = landmarks_id[i-1];
    while(cumulative_weight <= smpl) {
      cumulative_weight += weights[++i];
    }
    return vals[i];
  }
};
template<typename scalar_t,
          const size_t MaxLandmarks>
struct [[maybe_unused]] LandmarkSampler<size_t, scalar_t, MaxLandmarks>{
  std::vector<scalar_t> weights;
  containers::ShortVector<size_t, MaxLandmarks> landmarks_id = {};
  containers::ShortVector<scalar_t, MaxLandmarks> landmarks_val = {};
  //std::vector<scalar_t> weights;
  [[maybe_unused]] explicit LandmarkSampler(std::vector<scalar_t>&& weights) :
    weights(std::move(weights)) {
    scalar_t cumulative_weight = 0.0;
    size_t landmark_index = 0;
    const size_t n = this->weights.size();
    // select number of landmarks dynamically such that it is at most
    const size_t n_landmarks = std::floor(std::log2(static_cast<float>(n)))+1;
    if(n_landmarks > landmarks_id.size()) {
      landmarks_id.resize(n_landmarks);
      landmarks_val.resize(n_landmarks);
    }
    size_t step_size = n / n_landmarks;
    for (size_t i = 0; i < this->weights.size(); ++i) {
      cumulative_weight += this->weights[i];
      if (i % step_size == 0 && (landmark_index + 1) < n_landmarks) {
        landmarks_id[landmark_index] = i;
        landmarks_val[landmark_index++] = cumulative_weight;
      }
    }
    // Ensure the last position is included as a landmark
    landmarks_id[landmark_index] = n-1;
    landmarks_val[landmark_index] = cumulative_weight;
  }
  [[maybe_unused]] void update_weight(const size_t index, const scalar_t new_weight) {
    const auto weight_diff = new_weight - weights[index];
    weights[index] = new_weight;
    // update landmarks
    size_t i = 0;
    for(const auto& id : landmarks_id) {
      if(id >= index) {
        landmarks_val[i] += weight_diff;
      }
      i++;
    }
  }
  template<typename RNG> [[maybe_unused]] size_t sample(RNG& rng) {
    std::uniform_real_distribution<scalar_t> dist(0.0, landmarks_val.back());
    scalar_t smpl = dist(rng);
    // if the first landmark is greater, we can return early :)
    if(landmarks_val.front() > smpl) return 0;
    // Linear search on landmarks
    size_t i = 1;
    // if a landmark is greater we know we are somewhere in-between
    for(; i < landmarks_id.size(); i++) {
      // we found the top landmark
      if(landmarks_val[i] > smpl) break;
    }
    // Linear search within the sub-segment
    scalar_t cumulative_weight = landmarks_val[i-1];
    i = landmarks_id[i-1];
    while(cumulative_weight <= smpl) {
      cumulative_weight += weights[++i];
    }
    return i;
  }
};
template<typename scalar_t, const size_t MaxLandmarks, const size_t MaxHistorySize = 100>
struct [[maybe_unused]] ReversibleLandmarkSampler {
  // Current weights
  std::vector<scalar_t> weights;
  treeson::containers::ShortVector<size_t, MaxLandmarks> landmarks_id = {};

  // History buffers for weights and landmarks
  std::array<
      treeson::containers::ShortVector<scalar_t, MaxLandmarks>,
      MaxHistorySize> landmarks_history = {};
  std::array<std::pair<size_t, scalar_t>,
             MaxHistorySize> weight_changes_history = {};
  size_t current_history_index = 0;

  // Constructor
  [[maybe_unused]] explicit ReversibleLandmarkSampler(std::vector<scalar_t>&& weights)
      : weights(std::move(weights)), current_history_index(size_t(0)),
        landmarks_history(), weight_changes_history(), landmarks_id() {
    initialize_landmarks();
  }

  // Initialize landmarks based on weights
  void initialize_landmarks() {
    scalar_t cumulative_weight = 0.0;
    size_t landmark_index = 0;
    const size_t n = weights.size();
    const size_t n_landmarks = std::floor(std::log2(static_cast<float>(n))) + 1;

    if (n_landmarks > landmarks_id.size()) {
      landmarks_id.resize(n_landmarks);
      for (auto& lh : landmarks_history) {
        lh.resize(n_landmarks);
      }
    }
    size_t step_size = n / n_landmarks;
    for (size_t i = 0; i < weights.size(); ++i) {
      cumulative_weight += weights[i];
      if (i % step_size == 0 && (landmark_index + 1) < n_landmarks) {
        landmarks_id[landmark_index] = i;
        landmarks_history[current_history_index][landmark_index++] = cumulative_weight;
      }
    }

    // Ensure the last position is included as a landmark
    landmarks_id[landmark_index] = n - 1;
    landmarks_history[current_history_index][landmark_index] = cumulative_weight;
  }
  // potentially lazy updating until sampler called again?
  [[maybe_unused]] void update(const size_t index, const scalar_t new_weight) {
    const auto weight_diff = new_weight - weights[index];
    weights[index] = new_weight;
    // Record change
    weight_changes_history[current_history_index] = std::pair<size_t, scalar_t>(index, weight_diff);
    const auto prev_history_index = current_history_index;
    // update current history index
    current_history_index = (current_history_index + 1) % MaxHistorySize;
    // Update landmarks
    for (size_t i = 0; i < landmarks_id.size(); ++i) {
      if(landmarks_id[i] < index) {
        // the same as current history
        landmarks_history[current_history_index][i] = landmarks_history[prev_history_index][i];
      }
      else {
        landmarks_history[current_history_index][i] = landmarks_history[prev_history_index][i] + weight_diff;
      }
    }
  }
  template<typename RNG>
  [[maybe_unused]] size_t sample(RNG& rng) {
    auto& current_landmarks = landmarks_history[current_history_index];
    std::uniform_real_distribution<scalar_t> dist(0.0, current_landmarks.back());
    scalar_t smpl = dist(rng);

    if (current_landmarks.front() > smpl) return 0;

    size_t i = 1;
    for (; i < landmarks_id.size(); i++) {
      if (current_landmarks[i] > smpl) break;
    }

    scalar_t cumulative_weight = current_landmarks[i - 1];
    i = landmarks_id[i - 1];
    while (cumulative_weight <= smpl) {
      cumulative_weight += weights[++i];
    }

    return i;
  }
  void revert(const size_t steps_back) {
    // shouldn't overflow hopefully
    size_t target_index = (current_history_index + MaxHistorySize - steps_back) % MaxHistorySize;
    current_history_index = target_index;
    for (size_t i = 0; i < steps_back; ++i) {
      size_t index = (target_index + i) % MaxHistorySize;
      const auto [id,change] = weight_changes_history[index];
      weights[id] -= change;
    }
  }
  [[maybe_unused]] const auto& get_current_landmarks() const {
    return landmarks_history[current_history_index];
  }
  auto weight_volume() const {
    return landmarks_history[current_history_index].back();
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
namespace embedders{
/* technically, 'embeddeds', these things can run path-wise
 * calculations and accumulate stuff; they can be used e.g. to contain a node's
 * current depth, regression statistics (for hoeffding trees), ranges (for mondrians)
 * and whatever else your heart really desires.
 */
/* placeholder type that is passed if you do not really need it; note that to
 * some extent, this is coupled to splitters; the splitter populates the embedder,
 * so the goal is to have a compile error if an empty embedder is passed
 * to a splitter which wants to populate it
 */
struct [[maybe_unused]] EmbedderEmpty{};
template <typename scalar_t, typename integral_t>
struct [[maybe_unused]] HoeffdingEmbedder{
  // currently only supports single target
  [[maybe_unused]] explicit HoeffdingEmbedder(const size_t target_) :
    target(target_), values(std::vector<utils::Welford<scalar_t>>(0)) {}
  auto operator [](const size_t index) {
    // get value at specified index; this is the welford object with mean/sd methods
    return values[index];
  }
  [[maybe_unused]] void embed(
      const std::vector<
          std::variant<std::vector<integral_t>,std::vector<scalar_t>>
          > &data,
      std::vector<size_t> &indices,
      size_t start, size_t end) {
    // consumes data, embeds additional thing to values
    const auto& targ = std::get<std::vector<scalar_t>>(data[target]);
    utils::Welford<scalar_t> temp;
    for(size_t i = start; i< end; i++) {
      temp(targ[indices[i]]);
    }
    values.push_back(temp);
  }
  size_t target;
  std::vector<utils::Welford<scalar_t>> values;
};
}
namespace splitters {
template<typename scalar_t, typename integral_t, const size_t MaxCategoricalSet,
    typename RNG>
struct RandomFeatureVisitor {
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
template<typename scalar_t, typename integral_t, const size_t MaxCategoricalSet,
          typename RNG>
struct MondrianFeatureVisitor {
  using CatSet = containers::FixedCategoricalSet<MaxCategoricalSet, integral_t>;
  using Restriction = std::variant<std::pair<scalar_t, scalar_t>,
                                   std::pair<size_t, size_t>>;
  using Range = std::variant<std::pair<scalar_t, scalar_t>, CatSet>;
  //containers::Node<scalar_t, integral_t, MaxCategoricalSet>& node;
  std::vector<Range>& global_ranges;
  const size_t feature;
  RNG& rng;
  std::tuple<bool, std::variant<scalar_t, CatSet>, Restriction>
  operator()(const std::pair<scalar_t,scalar_t> &x) const noexcept {
    const auto threshold = selectRandomThreshold(
        x.first, x.second);
    return {selectMIADirection(), threshold.second, threshold};
  }
  std::tuple<bool, std::variant<scalar_t, CatSet>, Restriction>
      operator()(const std::pair<size_t,size_t> &x) const noexcept {
    const auto categoricalSet = selectRandomCategoricalSet(
        x.first, x.second);
    return {selectMIADirection(), categoricalSet,
            std::pair<size_t, size_t>(x.first, categoricalSet.size)};
  }
private:
  std::pair<scalar_t,scalar_t> selectRandomThreshold(
      const scalar_t min_, const scalar_t max_) const noexcept {
    std::uniform_real_distribution<scalar_t> dist(min_, max_);
    const scalar_t mid_ = dist(rng);
    return std::pair<scalar_t, scalar_t>(min_, mid_);
  }
  [[nodiscard]] bool selectMIADirection() const noexcept {
    return std::bernoulli_distribution(0.5)(rng);
  }
  CatSet selectRandomCategoricalSet(const size_t start, const size_t end) const noexcept {
    auto& curr_set = std::get<
        containers::FixedCategoricalSet<MaxCategoricalSet, integral_t>>(
        global_ranges[feature]);
    const size_t n_selected = utils::select_from_range(end-start, rng);
    std::shuffle(curr_set.begin() + start, curr_set.begin() + end, rng);
    containers::FixedCategoricalSet<MaxCategoricalSet, integral_t> categoricalSet;
    for (auto it = curr_set.begin() + start; it < curr_set.begin() + n_selected; ++it) {
      categoricalSet.add(*it);
    }
    return categoricalSet;
  }
};
template <typename RNG, typename scalar_t = double,
          typename integral_t = size_t,
          const size_t MaxCategoricalSet = 10>
class [[maybe_unused]] ExtremelyRandomizedStrategy {
public:
  static constexpr size_t MAX_DEPTH = 100;
  [[maybe_unused]] void select_split(
      const std::vector<
          std::variant<std::vector<integral_t>, std::vector<scalar_t>>> &data,
      std::vector<size_t> &indices,
      const std::vector<size_t> &available_features, size_t start, size_t end,
      containers::Node<scalar_t, integral_t, MaxCategoricalSet> &node, RNG &rng) const {
    const size_t feature =
        available_features[utils::select_from_range(available_features.size(), rng)];
    auto [mia_direction, threshold] = std::visit(
        RandomFeatureVisitor<scalar_t, integral_t, MaxCategoricalSet, RNG>{
            data, feature, indices, start, end, rng}, data[feature]);
    node.featureIndex = feature;
    node.threshold = threshold;
    node.missing_goes_right = mia_direction;
  }
};
template <typename RNG, typename scalar_t = double,
          typename integral_t = size_t,
          const size_t MaxCategoricalSet = 10>
class [[maybe_unused]] MondrianStrategy {
public:
  static constexpr size_t MAX_DEPTH = 100;
private:
  std::exponential_distribution<scalar_t> exp_dist = std::exponential_distribution<scalar_t>(1.);
  using CatSet = containers::FixedCategoricalSet<MaxCategoricalSet, integral_t>;
  using Range = std::variant<std::pair<scalar_t, scalar_t>, CatSet>;
  using Restriction = std::variant<std::pair<scalar_t, scalar_t>, std::pair<size_t, size_t>>;
  /* need to keep track of feature ranges for scaling windows (against total
   * feature size) - basically how much of the d-th dimension of the hypercube
   * did we already partition? otherwise the algorithm breaks down for features
   * on different scales
   * note that under updates this also needs to be updated, but that should hopefully
   * be ok
   */
  std::vector<Range> global_ranges = {};
  //std::vector<std::variant<std::pair<scalar_t, scalar_t>, CatSet>> current_ranges = {};
  scalar_t lambda;
  // keep track of first update for offline learning; only do the update once then
  bool first = true;
  containers::ShortVector<scalar_t, 1024> splitting_times;
  samplers::ReversibleLandmarkSampler<scalar_t, 100, MAX_DEPTH> sampler;
  containers::ShortVector<std::pair<size_t, Restriction>, MAX_DEPTH> range_restriction;
  std::vector<size_t> feats;
  size_t previous_depth = 0;
  struct RangeSizeVisitor{
    scalar_t operator()(const std::pair<size_t, size_t>&x) {
      return static_cast<scalar_t>(x.second - x.first);
    }
    scalar_t operator()(const std::pair<scalar_t, scalar_t>&x){
      return x.second - x.first;
    }
    scalar_t operator()(const CatSet&x){
      return static_cast<scalar_t>(x.size);
    }
  };
public:
  [[maybe_unused]] explicit MondrianStrategy(
      const scalar_t lambda, const std::vector<size_t>& available_features) :
    lambda(lambda), feats(available_features),
    // Max depth capped to 100 - third argument
    sampler(samplers::ReversibleLandmarkSampler<scalar_t, 100, MAX_DEPTH>(
      std::move(std::vector<scalar_t>(available_features.size(), 1.)))) {
    splitting_times.push_back(0.);
    range_restriction = containers::ShortVector<std::pair<size_t, Restriction>,
                                                MAX_DEPTH>();
  }
  bool select_split(
      const std::vector<
          std::variant<std::vector<integral_t>,
              std::vector<scalar_t>>> &data,
      std::vector<size_t> &indices,
      const std::vector<size_t> &available_features, size_t start, size_t end,
      containers::Node<scalar_t, integral_t, MaxCategoricalSet> &node,
      RNG &rng, const size_t parent_index, const size_t current_depth) {
    // assumes offline, or that at least a batch is ready before using this
    if(first) {
      for(const auto& feat : feats) {
        global_ranges.push_back(
            data_range_visitor(data[feat], indices, start, end));
      }
      // initialize landmark sampler
      std::vector<scalar_t> weights(global_ranges.size(), 1.);
      sampler = samplers::ReversibleLandmarkSampler<scalar_t, 100, MAX_DEPTH>(std::move(weights));
      first = false;
    }
    // sample from exponential/size
    const scalar_t E = exp_dist(rng)/sampler.weight_volume();
    // check if greater than splitting time limit, if not return false
    if(splitting_times[parent_index] + E <= lambda) {
      // if we backtracked before getting here we need to undo some of the weight changes
      if(current_depth < previous_depth) {
        // revert sampler history :)
        // TODO(JSzitas): check that the reversal is actually correct
        sampler.revert(previous_depth-current_depth);
      }
      // this is actually weighed sampling
      const size_t feature = sampler.sample(rng);
      node.featureIndex = feature;
      // total range
      scalar_t range_difference;
      splitting_times.push_back(splitting_times[parent_index] + E);
      // when overwriting restrictions push back is wrong
      // this traverses backwards over previous depths
      Restriction new_range_restriction;
      const auto& global_range = std::visit([&](auto &&arg) ->
                                     std::variant<std::pair<size_t, size_t>,
                                                  std::pair<scalar_t, scalar_t>> {
                                       using T = std::decay_t<decltype(arg)>;
                                       if constexpr (std::is_same_v<T, CatSet>) {
                                         return std::pair<size_t, size_t>(0, arg.size);
                                       } else if constexpr (std::is_same_v<T, std::pair<scalar_t, scalar_t>>) {
                                         return arg;
                                       }
                                     }, global_ranges[feature]);
      bool found = false;
      for(size_t i = current_depth; i > 0; i--) {
        // find previous range restriction, if any, from the back
        if(range_restriction[i-1].first == feature) {
          const auto prev = range_restriction[i-1].second;
          std::tuple<bool, std::variant<scalar_t, CatSet>, Restriction> res =
              std::visit(MondrianFeatureVisitor<
                  scalar_t, integral_t, MaxCategoricalSet, RNG>{
                   global_ranges, feature, rng}, prev);
          node.missing_goes_right = std::get<bool>(res);
          node.threshold = std::get<std::variant<scalar_t, CatSet>>(res);
          new_range_restriction = std::get<Restriction>(res);
          found = true;
          break;
        }
      }
      if(!found) {
        std::tuple<bool, std::variant<scalar_t, CatSet>, Restriction> res =
            std::visit(MondrianFeatureVisitor<
                scalar_t,integral_t, MaxCategoricalSet, RNG>{
                           global_ranges, feature, rng},
                   global_range);
        node.missing_goes_right = std::get<bool>(res);
        node.threshold = std::get<std::variant<scalar_t, CatSet>>(res);
        new_range_restriction = std::get<Restriction>(res);
      }
      const auto new_range_size = std::visit(RangeSizeVisitor{}, new_range_restriction);
      scalar_t new_weight = new_range_size/std::visit(RangeSizeVisitor{}, global_range);
      // add range restriction to range restrictions
      if(current_depth > range_restriction.size()) {
        range_restriction.push_back(std::pair(feature, new_range_restriction));
      } else {
        range_restriction[current_depth] =
            std::pair(feature, new_range_restriction);
      }
      // sampler requires update adjusting for current weight - this is size/total
      sampler.update(feature, new_weight);
      previous_depth = current_depth;
      return true;
    }
    return false;
  }
private:
  auto data_range_visitor(
      const std::variant<std::vector<integral_t>, std::vector<scalar_t>>& data,
      const std::vector<size_t> &indices, const size_t start, const size_t end) {
    return std::visit([&indices, start, end](auto&& arg) ->
                      std::variant<std::pair<scalar_t, scalar_t>, CatSet> {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, std::vector<integral_t>>) {
                          CatSet cat_set;
                          for (size_t i = start; i < end; i++) cat_set.add(arg[indices[i]]);
                          return cat_set;
                        } else if constexpr (std::is_same_v<T, std::vector<scalar_t>>) {
                          scalar_t min = std::numeric_limits<scalar_t>::max();
                          scalar_t max = std::numeric_limits<scalar_t>::lowest();
                          for(size_t i = start; i < end; i++) {
                            // maybe std::minmax element is faster?
                            // the issue is that arg[indices] can access all over
                            // the place (though should generally guarantee at least
                            // monotonically increasing indices)
                            const auto val = arg[indices[i]];
                            if (val < min) min = val;
                            else if (val > max) max = val;
                          }
                          return std::pair<scalar_t, scalar_t>(min,max);
                        }}, data);
  }
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
//TODO: this needs work
template<typename scalar_t = double,
         typename integral_t = size_t,
         const size_t MaxCategoricalSet = 10>
class [[maybe_unused]] HoeffdingStrategy{
  struct RegressionStatistics {
    scalar_t mean = 0.0, m2 = 0.0;
    size_t count = 1;
    void add(const scalar_t value) {
      const scalar_t delta = value - mean;
      mean += delta / count;
      const scalar_t delta2 = value - mean;
      m2 += delta * delta2;
      count++;
    }
    scalar_t variance() const {
      return count > 1 ? m2 / (count - 1) : 0.0;
    }
  };
  std::vector<RegressionStatistics> statistics_;

  [[maybe_unused]] HoeffdingStrategy(const scalar_t delta, const size_t max_samples_per_node) :
  delta_(delta), max_samples_per_node_(max_samples_per_node){}
  scalar_t compute_hoeffding_bound(const scalar_t range, const scalar_t confidence, const size_t n) {
    return std::sqrt((range * range * std::log(1.0 / confidence)) / (2.0 * static_cast<scalar_t>(n)));
  }
  [[maybe_unused]] void update_statistics(const size_t terminal_index, const scalar_t target) {
    auto& stats = statistics_[terminal_index];
    stats.add(target);
  }
  [[maybe_unused]] void attempt_split(containers::Node<scalar_t, integral_t, MaxCategoricalSet>& node,
                     const std::vector<std::vector<scalar_t>>& data_stream) {
    auto& stats = statistics_[node.terminal_index];

    // Check if we have enough samples to consider a split
    if (stats.count >= max_samples_per_node_) {
      scalar_t best_variance_reduction = 0.0;
      size_t best_feature = std::numeric_limits<size_t>::max();
      scalar_t best_split = 0.0;

      for (size_t feature = 0; feature < data_stream[0].size() - 1; ++feature) {
        // Assuming the last column is the target
        // Sort instances by the current feature
        std::vector<std::pair<scalar_t, scalar_t>> feature_target_pairs;
        for (const auto& instance : data_stream) {
          feature_target_pairs.emplace_back(instance[feature], instance.back());
        }

        std::sort(feature_target_pairs.begin(), feature_target_pairs.end());

        // Initialize left and right statistics
        RegressionStatistics left_stats, right_stats;
        for (const auto& [feature_value, target_value] : feature_target_pairs) {
          right_stats.add(target_value);
        }

        // Evaluate split points between consecutive feature values
        for (size_t i = 0; i < feature_target_pairs.size() - 1; ++i) {
          left_stats.add(feature_target_pairs[i].second);
          right_stats.count--; // Decrement right count
          right_stats.mean *= right_stats.count / (right_stats.count + 1);
          scalar_t delta = feature_target_pairs[i].second - right_stats.mean;
          right_stats.mean += delta / right_stats.count;
          right_stats.m2 -= delta * (feature_target_pairs[i].second - right_stats.mean);

          if (feature_target_pairs[i].first != feature_target_pairs[i + 1].first) {
            // Compute variance reduction
            scalar_t variance_reduction = stats.variance()
                                          - ((left_stats.count / static_cast<scalar_t>(stats.count)) * left_stats.variance()
                                             + (right_stats.count / static_cast<scalar_t>(stats.count)) * right_stats.variance());

            // Check if this split is better
            if (variance_reduction > best_variance_reduction) {
              best_variance_reduction = variance_reduction;
              best_feature = feature;
              best_split = (feature_target_pairs[i].first + feature_target_pairs[i + 1].first) / 2.0;
            }
          }
        }
      }
      // Use Hoeffding bound to decide whether to split
      scalar_t epsilon = compute_hoeffding_bound(1.0, delta_, stats.count);
      if (best_variance_reduction > epsilon) {
        // Perform the split
        containers::Node<scalar_t, integral_t, MaxCategoricalSet> left, right;
        node.split_feature = best_feature;
        node.threshold = best_split;
      }
    }
  }
  const scalar_t delta_;
  const size_t max_samples_per_node_;
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
          typename integral_t = size_t, typename Embedder = embedders::EmbedderEmpty>
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
  bool preallocated = false;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
  [[maybe_unused]] Embedder embedder;
#pragma clang diagnostic pop
public:
  [[maybe_unused]] RandomTree(const size_t maxDepth,
                              const size_t minNodesize,
                              RNG &rng,
                              ResultF &terminalNodeFunc,
                              SplitStrategy &strategy)
      : maxDepth(maxDepth), minNodesize(minNodesize), rng(rng),
        terminalNodeFunc(terminalNodeFunc), split_strategy(strategy),
        embedder(Embedder()) {
    // TODO: perhaps consider a better way to handle depth limits?
    if(maxDepth > strategy.MAX_DEPTH) {
      throw std::invalid_argument("Maximum depth specified over limit, specified: "
                                  + std::to_string(maxDepth) +
                                  " maximum permissible: " +
                                  std::to_string(strategy.MAX_DEPTH));
    }
    nodes.reserve(static_cast<size_t>(std::pow(2, maxDepth + 1) - 1));
    terminal_values = std::vector<ResultType>{};
  }
  [[maybe_unused]] void
  fit(const std::vector<FeatureData> &data, std::vector<size_t> indices,
      const std::vector<size_t> &nosplit_features) noexcept {
    if(preallocated) {
      nodes[0].left = 0;
      refit_impl(data, indices, nosplit_features);
    } else {
      fit_impl(data, indices, nosplit_features);
    }
  }
  [[maybe_unused]] void fit(const std::vector<FeatureData> &data,
                            std::vector<size_t> indices,
                            std::vector<size_t> &&nosplit_features) noexcept {
    if(preallocated) {
      nodes[0].left = 0;
      refit_impl(data, indices, nosplit_features);
    } else {
      fit_impl(data, indices, nosplit_features);
    }
  }
  template <const bool FlattenResults = false>
  [[nodiscard]] auto predict(const std::vector<FeatureData> &samples,
                             std::vector<size_t> &indices) const noexcept {
    return predict_impl<FlattenResults>(samples, indices);
  }
  template <const bool FlattenResults = false>
  [[nodiscard]] auto predict(const std::vector<FeatureData> &samples)
      const noexcept {
    std::vector<size_t> indices = treeson::utils::make_index_range(
        utils::size(samples[0]));
    return predict_impl<FlattenResults>(samples, indices);
  }
  [[maybe_unused]] [[nodiscard]] std::set<size_t> used_features() const {
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
  [[nodiscard]] bool is_uninformative() const noexcept {
    // consists of only root or has no terminal values
    return (nodes.size() == 1) || (terminal_values.empty());
  }
  template<typename Metric,
           typename Reducer = treeson::reducers::defaultImportanceReducer<scalar_t>>
  [[maybe_unused]] auto eval_oob(
      const std::vector<FeatureData> &oob_data,
      std::vector<size_t> &indices) {
    Metric metric;
    using MetricResultType = decltype(std::declval<Metric>()(
      std::declval<ResultType&>(), std::declval<const std::vector<FeatureData> &>(),
      std::declval<const std::vector<size_t>&>(),
      std::declval<const size_t>(), std::declval<const size_t>()));
    std::vector<std::pair<MetricResultType, size_t>> results;
    eval_oob_impl<Metric,false>(
        0, oob_data, indices, 0, indices.size(), results, 0, metric);
    return Reducer()(results);
  }
  template<typename Metric,
           typename Reducer = treeson::reducers::defaultImportanceReducer<scalar_t>>
  [[maybe_unused]] auto eval_oob(
      const std::vector<FeatureData> &oob_data,
      std::vector<size_t> &indices,
      const size_t excluded_feature) {
    Metric metric;
    using MetricResultType = decltype(std::declval<Metric>()(
      std::declval<ResultType&>(), std::declval<const std::vector<FeatureData> &>(),
      std::declval<const std::vector<size_t>&>(),
      std::declval<const size_t>(), std::declval<const size_t>()));
    std::vector<std::pair<MetricResultType, size_t>> results;
    eval_oob_impl<Metric,true>(
        0, oob_data, indices, 0, indices.size(), results, excluded_feature, metric);
    return Reducer()(results);
  }
  template<typename Metric,
            typename Reducer = treeson::reducers::defaultImportanceReducer<scalar_t>>
  [[maybe_unused]] auto eval_oob(
      const std::vector<FeatureData> &oob_data,
      std::vector<size_t> &indices,
      Metric& metric) {
    using MetricResultType = decltype(std::declval<Metric>()(
        std::declval<ResultType&>(), std::declval<const std::vector<FeatureData> &>(),
        std::declval<const std::vector<size_t>&>(),
        std::declval<const size_t>(), std::declval<const size_t>()));
    std::vector<std::pair<MetricResultType, size_t>> results;
    eval_oob_impl<Metric,false>(
        0, oob_data, indices, 0, indices.size(), results, 0, metric);
    return Reducer()(results);
  }
  template<typename Metric,
            typename Reducer = treeson::reducers::defaultImportanceReducer<scalar_t>>
  [[maybe_unused]] auto eval_oob(
      const std::vector<FeatureData> &oob_data,
      std::vector<size_t> &indices,
      const size_t excluded_feature,
      Metric& metric) {
    using MetricResultType = decltype(std::declval<Metric>()(
        std::declval<ResultType&>(), std::declval<const std::vector<FeatureData> &>(),
        std::declval<const std::vector<size_t>&>(),
        std::declval<const size_t>(), std::declval<const size_t>()));
    std::vector<std::pair<MetricResultType, size_t>> results;
    eval_oob_impl<Metric,true>(
        0, oob_data, indices, 0, indices.size(), results, excluded_feature, metric);
    return Reducer()(results);
  }
  void from_nodes(
      std::vector<containers::Node<scalar_t, integral_t,
                                   MaxCategoricalSet>> && internal_nodes,
      std::vector<ResultType>&& terminal_node_values){
      this->nodes = internal_nodes;
      this->terminal_values = terminal_node_values;
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
      serializers::serialize(preallocated, os);
    }
    std::pair<std::vector<containers::Node<
        scalar_t, integral_t, MaxCategoricalSet>>,
              std::vector<ResultType>> deserialize(std::istream& is) {
      size_t size;
      is.read(reinterpret_cast<char*>(&size), sizeof(size));
      std::vector<containers::Node<scalar_t, integral_t, MaxCategoricalSet>>
          deserialized_nodes(size);
      for (auto& node : deserialized_nodes) {
        node.deserialize(is);
      }
      is.read(reinterpret_cast<char*>(&size), sizeof(size));
      std::vector<ResultType> deserialized_terminal_values(size);
      for (auto& value : deserialized_terminal_values) {
        serializers::deserialize(value, is);
      }
      serializers::deserialize(preallocated, is);
      return std::make_pair(std::move(deserialized_nodes),
                            std::move(deserialized_terminal_values));
    }
    [[maybe_unused]] void preallocate() {
      const size_t size = std::pow(2, maxDepth + 1) - 1;
      nodes = std::vector<containers::Node<
          scalar_t, integral_t, MaxCategoricalSet>>(size);
      terminal_values = std::vector<ResultType>(size);
      preallocated = true;
    }
  private:
  template<typename T> auto fit_impl(
      const std::vector<FeatureData> &data,
      std::vector<size_t> indices,
      T&& nosplit_features) noexcept{
    std::vector<size_t> available_features;
    for (size_t i = 0; i < data.size(); ++i) {
      if (std::find(nosplit_features.begin(), nosplit_features.end(), i) ==
          nosplit_features.end()) {
        available_features.push_back(i);
      }
    }
    buildTree(data, indices, available_features, 0, 0, indices.size());
  }
  template<typename T> auto refit_impl(
      const std::vector<FeatureData> &data,
      std::vector<size_t> indices,
      T&& nosplit_features) noexcept{
    std::vector<size_t> available_features;
    for (size_t i = 0; i < data.size(); ++i) {
      if (std::find(nosplit_features.begin(), nosplit_features.end(), i) ==
          nosplit_features.end()) {
        available_features.push_back(i);
      }
    }
    size_t nodeIndex = 0, terminal_index = 0;
    rebuildTree(data, indices, available_features, 0, 0, indices.size(),
                0, nodeIndex, terminal_index);
  }
  template <const bool FlattenResults = false>
  [[nodiscard]] auto predict_impl(
      const std::vector<FeatureData> &samples,
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
    size_t mid;
    if constexpr (std::is_invocable_v<
                      decltype(&SplitStrategy::select_split), SplitStrategy,
                      const std::vector<FeatureData> &, std::vector<size_t> &,
                      const std::vector<size_t> &, size_t, size_t,
                      containers::Node<scalar_t, integral_t, MaxCategoricalSet>
                          &,
                      RNG &>) {
      split_strategy.select_split(data, indices, available_features, start, end,
                                  nodes[nodeIndex], rng);
    } else if constexpr (
        std::is_invocable_v<
            decltype(&SplitStrategy::select_split), SplitStrategy,
            const std::vector<FeatureData> &, std::vector<size_t> &,
            const std::vector<size_t> &, size_t, size_t,
            containers::Node<scalar_t, integral_t, MaxCategoricalSet> &, RNG &,
            size_t, size_t>) {
      const bool did_split =
          split_strategy.select_split(data, indices, available_features, start, end,
                                      nodes[nodeIndex], rng, parentIndex, depth);
      if(!did_split) goto end_fitting_tree;
    }
    mid = reorder_indices(data, nodes[nodeIndex], start, end, indices);
    // if a child node would be too small
    if (((mid - start) < minNodesize) || ((end - mid) < minNodesize) ||
        depth >= maxDepth) {
      end_fitting_tree:
      terminal_values.push_back(terminalNodeFunc(indices, start, end, data));
      nodes[nodeIndex].assign_terminal(terminal_values.size() - 1);
    } else {
      buildTree(data, indices, available_features, nodeIndex, start, mid,
                depth + 1);
      buildTree(data, indices, available_features, nodeIndex, mid, end,
                depth + 1);
    }
  }
  void rebuildTree(const std::vector<FeatureData> &data,
                 std::vector<size_t> &indices,
                 const std::vector<size_t> &available_features,
                 size_t parentIndex, size_t start, size_t end,
                 size_t depth, size_t& nodeIndex,
                 size_t& terminal_index) noexcept {
    if (nodes[parentIndex].left == 0) {
      nodes[parentIndex].left = nodeIndex; // Left child
    } else {
      nodes[parentIndex].right = nodeIndex; // Right child
    }
    size_t mid;
    if constexpr (std::is_invocable_v<
        decltype(&SplitStrategy::select_split), SplitStrategy,
        const std::vector<FeatureData> &, std::vector<size_t> &,
        const std::vector<size_t> &, size_t, size_t,
        containers::Node<scalar_t, integral_t, MaxCategoricalSet>&, RNG &>) {
      split_strategy.select_split(data, indices, available_features, start, end,
                                  nodes[nodeIndex], rng);
    } else if constexpr (
        std::is_invocable_v<
            decltype(&SplitStrategy::select_split), SplitStrategy,
            const std::vector<FeatureData> &, std::vector<size_t> &,
            const std::vector<size_t> &, size_t, size_t,
            containers::Node<scalar_t, integral_t, MaxCategoricalSet> &, RNG &,
            size_t, size_t>) {
      const bool did_split =
          split_strategy.select_split(data, indices, available_features, start, end,
                                      nodes[nodeIndex], rng, nodeIndex, parentIndex);
      // there possibly an early return if splitting is rejected i.e. in mondrians
      if(!did_split) goto end_building_tree;
    }
    mid = reorder_indices(data, nodes[nodeIndex], start, end, indices);
    // if a child node would be too small
    if (((mid - start) <= minNodesize) || ((end - mid) <= minNodesize) ||
        depth >= maxDepth) {
      end_building_tree:
      // get rid of last node since it was actually invalid
      terminal_values[terminal_index++] = terminalNodeFunc(indices, start, end, data);
      nodes[nodeIndex].assign_terminal(terminal_index - 1);
    } else {
      const size_t node_ = nodeIndex;
      nodeIndex++;
      rebuildTree(data, indices, available_features, node_, start, mid,
                  depth+1, nodeIndex, terminal_index);
      rebuildTree(data, indices, available_features, node_, mid, end,
                  depth+1, nodeIndex, terminal_index);
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
      const size_t excluded_feature, Metric& metric) const noexcept {
        const auto &node = nodes[nodeIndex];
        if (node.left == 0 && node.right == 0) {
          const auto& vals = terminal_values[node.terminal_index];
          results.push_back(std::pair(metric(vals, oob_data, indices, start, end), end-start));
          return;
        }
        if constexpr(exclude_feature) {
          if (nodes[nodeIndex].featureIndex == excluded_feature) {
            // consider that this probably returns only per one of these branches
            eval_oob_impl<Metric, true>(
                node.left, oob_data,
                indices, start, end, results, excluded_feature, metric);
            eval_oob_impl<Metric, true>(
                node.right, oob_data,
                indices, start, end, results, excluded_feature, metric);
          }
        }
        auto mid = reorder_indices(
            oob_data, nodes[nodeIndex], start, end, indices);
        if (mid > start) {
          eval_oob_impl<Metric, false>(
              node.left, oob_data,
              indices, start, mid, results, excluded_feature, metric);
        }
        if (mid < end) {
          eval_oob_impl<Metric, false>(
              node.right, oob_data,
              indices, mid, end, results, excluded_feature, metric);
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
    if (node.left == 0 || nodes[nodeIndex].featureIndex > samples.size()) {
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
  template<typename T>
  void travel(size_t nodeIndex, const std::vector<FeatureData> &data,
              std::vector<size_t> &indices, const size_t inner_id,
              T& pathway) const noexcept {
    const auto &node = nodes[nodeIndex];
    // terminal leaf; we have arrived
    if (node.left == 0) {return;}
    // record pathway
    pathway.emplace_back(nodeIndex);
    // determine where to next
    auto& featureData = data[node.featureIndex];
    const auto miss_dir = node.getMIADirection();
    bool left = false;
    if (node.isCategorical()) {
      const auto &catSet = node.getCategoricalSet();
      const auto &catData = std::get<std::vector<integral_t>>(featureData);
      left = catSet.contains(catData[indices[inner_id]]);
    }
    else {
        const auto &numThreshold = node.getNumericThreshold();
        const auto &numData = std::get<std::vector<scalar_t>>(featureData);
        const auto val = numData[indices[inner_id]];
        left = (val <= numThreshold) || (std::isnan(val) && miss_dir);
    }
    travel(left ? node.left : node.right, data, indices, inner_id, pathway);
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
        terminalNodeFunc(terminalNodeFunc), strategy(strategy) {
    if(maxDepth > strategy.MAX_DEPTH) {
      throw std::invalid_argument("Maximum depth specified over limit, specified: "
                                  + std::to_string(maxDepth) +
                                  " maximum permissible: " +
                                  std::to_string(strategy.MAX_DEPTH));
    }
  }
  [[maybe_unused]] RandomForest(const size_t maxDepth,
                                const size_t minNodesize,
                                RNG&& rng,
                                ResultF &&terminalNodeFunc,
                                SplitStrategy &strategy) :
    maxDepth(maxDepth), minNodesize(minNodesize), rng(std::move(rng)),
    terminalNodeFunc(std::move(terminalNodeFunc)), strategy(std::move(strategy)) {
    if(maxDepth > strategy.MAX_DEPTH) {
      throw std::invalid_argument("Maximum depth specified over limit, specified: "
                                  + std::to_string(maxDepth) +
                                  " maximum permissible: " +
                                  std::to_string(strategy.MAX_DEPTH));
    }
  }
  [[maybe_unused]] void fit(const std::vector<FeatureData> &data,
           const size_t n_tree,
           const std::vector<size_t> &nosplit_features,
           const bool resample = true,
           const size_t sample_size = 0,
           const size_t num_threads = 0) {
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
    const size_t actual_num_threads = threading::thread_heuristic(num_threads);
    if(actual_num_threads == 1) {
      fit_st(data, n_tree, nosplit_features, resample_, bootstrap_size);
      return;
    }
    fit_mt(data, n_tree, nosplit_features, resample_, bootstrap_size,
           actual_num_threads);
  }
  [[maybe_unused]] void fit(const std::vector<FeatureData> &data,
                            const size_t n_tree,
                            const std::vector<size_t> &nosplit_features,
                            const std::string &file,
                            const bool resample = true,
                            const size_t sample_size = 0,
                            const size_t num_threads = 0) {
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
    const size_t actual_num_threads = threading::thread_heuristic(num_threads);
    std::ofstream out(file + ".bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(&n_tree), sizeof(n_tree));
    if(actual_num_threads == 1) {
      fit_to_file_st(data, n_tree, nosplit_features, out, resample_,
                     bootstrap_size);
      return;
    }
    fit_to_file_mt(data, n_tree, nosplit_features, out, resample_,
                   bootstrap_size, actual_num_threads);
  }
  [[nodiscard]] std::vector<containers::TreePredictionResult<ResultType>>
      predict(const std::vector<FeatureData> &samples,
              const size_t num_threads) const noexcept {
    const size_t actual_num_threads = threading::thread_heuristic(num_threads);
    if(actual_num_threads == 1) {
      return predict_st(samples);
    }
    return predict_mt(samples, actual_num_threads);
  }
  [[maybe_unused]] [[nodiscard]]
  std::vector<containers::TreePredictionResult<ResultType>> predict(
    const std::vector<FeatureData> &samples,
    const std::string &model_file,
    const size_t num_threads) const noexcept {
    std::ifstream in(model_file + ".bin", std::ios::binary);
    size_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    const size_t actual_num_threads = threading::thread_heuristic(num_threads);
    if(actual_num_threads == 1) {
      return predict_from_file_st(samples, in, n);
    }
    return predict_from_file_mt(samples, in, n, actual_num_threads);
  }
  template<typename Accumulator>
  void predict(
      Accumulator& accumulator,
      const std::vector<FeatureData> &samples,
      const std::string &model_file,
      const size_t num_threads) const noexcept {
    std::ifstream in(model_file + ".bin", std::ios::binary);
    size_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    const size_t actual_num_threads = threading::thread_heuristic(num_threads);
    if(actual_num_threads == 1) {
      predict_from_file_acc_st(accumulator, samples, in, n);
      return;
    }
    predict_from_file_acc_mt(
        accumulator, samples, in, n, actual_num_threads);
  }
  template<typename Accumulator>
  [[maybe_unused]] void memoryless_predict(
      Accumulator& accumulator,
      const std::vector<FeatureData> &train_data,
      const std::vector<FeatureData> &predict_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool resample = true,
      const size_t sample_size = 0,
      const size_t num_threads = 0) const noexcept {
    size_t bootstrap_size = sample_size > 0 ? sample_size :
      std::visit([](auto&& arg) -> size_t { return arg.size(); }, train_data[0]);
    const bool resample_ = resample && sample_size > 0;
    if(resample != resample_) {
      std::cout << "You specified 'resample = true', " <<
          "but provided 'sample_size = 0'. Quitting, respecify." << std::endl;
      return;
    }
    const size_t actual_num_threads = threading::thread_heuristic(num_threads);
    if(actual_num_threads == 1) {
      memoryless_predict_st(accumulator, train_data, predict_data, n_tree,
          nosplit_features, resample_, bootstrap_size);
      return;
    }
    memoryless_predict_mt(accumulator, train_data, predict_data, n_tree,
                          nosplit_features, resample_, bootstrap_size,
                          actual_num_threads);
  }
  // TODO: test
  [[maybe_unused]] void prune() {
    size_t i = 0;
    std::vector<size_t> removable_trees, informative_trees;
    for(const auto& tree: trees) {
      if(tree.is_uninformative()) {
        removable_trees.push_back(i);
      }
      else {
        informative_trees.push_back(i);
      }
      i++;
    }
    size_t k = informative_trees.size()-1; i = 0;
    while((removable_trees[i] < informative_trees[k]) &&
           (k >= 0) && (i < removable_trees.size())) {
      // swap informative and uninformative trees
      // removable trees go from front, informative trees from the back.
      // note that this works since both are, by construction, sorted
      std::swap(trees[removable_trees[i]], trees[informative_trees[k--]]);
      i++; k--;
    }
    // resize, dropping uninformative trees
    trees.resize(informative_trees.size());
  }
  enum ImportanceMethod{
    Omit,
    Contrast
  };
  // Feature importance method
  template<typename Accumulator, typename Metric, typename Importance,
           const ImportanceMethod Method = ImportanceMethod::Omit>
  [[maybe_unused]] auto feature_importance(
      const std::vector<FeatureData> &train_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool oob = true,
      const size_t sample_size = 0,
      const size_t num_threads = 0) {
    size_t bootstrap_size = sample_size > 0 ? sample_size :
                                            utils::size(train_data[0]);
    const size_t actual_num_threads = threading::thread_heuristic(num_threads);
    if(actual_num_threads == 1) {
      if constexpr(Method == ImportanceMethod::Omit) {
        return feature_importance_omit_st<Accumulator, Metric, Importance>(
            train_data, n_tree, nosplit_features, oob, bootstrap_size);
      }
      if constexpr (Method == ImportanceMethod::Contrast) {
        return feature_importance_contrast_st<Accumulator, Metric, Importance>(
            train_data, n_tree, nosplit_features, oob, bootstrap_size);
      }
    }
    if constexpr(Method == ImportanceMethod::Omit) {
      return feature_importance_omit_mt<Accumulator, Metric, Importance>(
          train_data, n_tree, nosplit_features, oob, bootstrap_size,
          actual_num_threads);
    }
    // used as base case
    //if constexpr (Method == ImportanceMethod::Contrast) {
    return feature_importance_contrast_mt<Accumulator, Metric, Importance>(
          train_data, n_tree, nosplit_features, oob, bootstrap_size,
          actual_num_threads);
    //}
  }
  // Feature importance method
  template<typename Accumulator, typename Metric, typename Importance,
            const ImportanceMethod Method = ImportanceMethod::Omit>
  [[maybe_unused]] auto feature_importance(
      Metric& metric,
      const std::vector<FeatureData> &train_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool oob = true,
      const size_t sample_size = 0,
      const size_t num_threads = 0) {
    size_t bootstrap_size = sample_size > 0 ? sample_size :
                                            utils::size(train_data[0]);
    const size_t actual_num_threads = threading::thread_heuristic(num_threads);
    if(actual_num_threads == 1) {
      if constexpr(Method == ImportanceMethod::Omit) {
        return feature_importance_omit_st<Accumulator, Metric, Importance>(
            metric, train_data, n_tree, nosplit_features, oob, bootstrap_size);
      }
      if constexpr (Method == ImportanceMethod::Contrast) {
        return feature_importance_contrast_st<Accumulator, Metric, Importance>(
            metric, train_data, n_tree, nosplit_features, oob, bootstrap_size);
      }
    }
    if constexpr(Method == ImportanceMethod::Omit) {
      return feature_importance_omit_mt<Accumulator, Metric, Importance>(
          metric, train_data, n_tree, nosplit_features, oob, bootstrap_size,
          actual_num_threads);
    }
    // used as base case
    return feature_importance_contrast_mt<Accumulator, Metric, Importance>(
        metric, train_data, n_tree, nosplit_features, oob, bootstrap_size,
        actual_num_threads);
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
  void fit_st(const std::vector<FeatureData> &data,
              const size_t n_tree,
              const std::vector<size_t> &nosplit_features,
              const bool resample,
              const size_t bootstrap_size) {
    for (size_t i = 0; i < n_tree; ++i) {
      std::vector<size_t> indices =
          utils::make_index_range(utils::size(data[0]));
      TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc,
                    strategy);
      if (resample) {
        tree.fit(
            data,
            treeson::utils::bootstrap_sample(indices, bootstrap_size, rng),
            nosplit_features);
      } else {
        tree.fit(data, indices, nosplit_features);
      }
      trees.push_back(tree);
    }
  }
  void fit_mt(const std::vector<FeatureData> &data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool resample,
      const size_t bootstrap_size,
      const size_t actual_num_threads) {
    threading::SharedResourceWrapper<std::vector<TreeType>> trees_;
    // N.B.: Outer scope forces thread pool to finish before we do anything
    {
      threading::ThreadPool pool(actual_num_threads);
      for (size_t i = 0; i < n_tree; ++i) {
        const size_t seed = rng();
        pool.enqueue([this, &data, &trees_, &nosplit_features, resample, seed,
                      bootstrap_size] {
          std::vector<size_t> indices =
              utils::make_index_range(utils::size(data[0]));
          // custom random number generator for this tree
          RNG rng_(seed);
          TreeType tree(maxDepth, minNodesize, rng_, terminalNodeFunc,
                        strategy);
          if (resample) {
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
          trees_->push_back(tree);
        });
      }
    }
    trees = std::move(trees_.resource);
  }
  void fit_to_file_st(const std::vector<FeatureData> &data,
                      const size_t n_tree,
                      const std::vector<size_t> &nosplit_features,
                      std::ofstream& out,
                      const bool resample,
                      const size_t bootstrap_size) {
    for (size_t i = 0; i < n_tree; ++i) {
        std::vector<size_t> indices =
            utils::make_index_range(utils::size(data[0]));
        TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc,
                      strategy);
        if (resample) {
          tree.fit(
              data,
              treeson::utils::bootstrap_sample(indices, bootstrap_size, rng),
              nosplit_features);
        } else {
          tree.fit(data, indices, nosplit_features);
        }
        tree.serialize(out);
      }
  }
  void fit_to_file_mt(const std::vector<FeatureData> &data,
                      const size_t n_tree,
                      const std::vector<size_t> &nosplit_features,
                      std::ofstream& out,
                      const bool resample,
                      const size_t bootstrap_size,
                      const size_t actual_num_threads) {
    threading::SharedResourceWrapper<std::ofstream> out_(std::move(out));
    {
      threading::ThreadPool pool(actual_num_threads);
      // required for thread safety
      for (size_t i = 0; i < n_tree; ++i) {
        const size_t seed = rng();
        pool.enqueue([this, &data, &nosplit_features, resample, seed,
                      bootstrap_size, &out_] {
          std::vector<size_t> indices =
            utils::make_index_range(utils::size(data[0]));
          // custom random number generator for this tree
          RNG rng_(seed);
          TreeType tree(maxDepth, minNodesize, rng_, terminalNodeFunc,
                        strategy);
          if (resample) {
            tree.fit(
                data,
                treeson::utils::bootstrap_sample(indices, bootstrap_size, rng_),
                nosplit_features);
          } else {
            tree.fit(data, indices, nosplit_features);
          }
          out_.serialize(tree);
        });
      }
    }
  }
  [[nodiscard]] std::vector<containers::TreePredictionResult<ResultType>>
  predict_st(const std::vector<FeatureData> &samples) const noexcept {
    const size_t n = trees.size();
    std::vector<containers::TreePredictionResult<ResultType>> results;
    results.reserve(n);
    for (size_t i = 0; i < trees.size(); ++i) {
      results.push_back(trees[i].predict(samples));
    }
    return results;
  }
  [[nodiscard]] std::vector<containers::TreePredictionResult<ResultType>>
  predict_mt(const std::vector<FeatureData> &samples,
             const size_t actual_num_threads = 1) const noexcept {
    const size_t n = trees.size();
    std::vector<containers::TreePredictionResult<ResultType>> results;
    results.reserve(n);
    threading::SharedResourceWrapper<
        std::vector<containers::TreePredictionResult<ResultType>>> results_(results);
    {
      threading::ThreadPool pool(actual_num_threads);
      for (size_t i = 0; i < trees.size(); ++i) {
        pool.enqueue([this, &results_, &samples, i] {
          results_->push_back(trees[i].predict(samples));
        });
      }
    }
    return results_.resource;
  }
  [[maybe_unused]] [[nodiscard]]
std::vector<containers::TreePredictionResult<ResultType>> predict_from_file_st(
    const std::vector<FeatureData> &samples,
      std::ifstream& model_stream,
      const size_t n) const noexcept {
  std::vector<containers::TreePredictionResult<ResultType>> results;
  results.reserve(n);
  for (size_t i = 0; i < n; i++) {
    TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc, strategy);
    auto [nodes, values] = tree.deserialize(model_stream);
    tree.from_nodes(std::move(nodes),std::move(values));
    results.push_back(tree.predict(samples));
  }
  return results;
}
std::vector<containers::TreePredictionResult<ResultType>> predict_from_file_mt(
    const std::vector<FeatureData> &samples,
    std::ifstream& model_stream,
    const size_t n,
    const size_t actual_num_threads) const noexcept {
    // shared resource to access the file
    std::vector<containers::TreePredictionResult<ResultType>> results(n);
    threading::SharedResourceWrapper<
        std::vector<containers::TreePredictionResult<ResultType>>
        > results_(std::move(results));
    threading::SharedResourceWrapper<std::ifstream> in_(std::move(model_stream));
    {
      threading::ThreadPool pool(actual_num_threads);
      for (size_t i = 0; i < n; i++) {
        pool.enqueue([this, &results_, &samples, &in_, i] {
          TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc, strategy);
          auto [nodes, values] = in_.deserialize(tree);
          tree.from_nodes(std::move(nodes), std::move(values));
          results_[i] = tree.predict(samples);
        });
      }
    }
    return results_.resource;
  }
  template<typename Accumulator>
  void predict_from_file_acc_st(
      Accumulator& accumulator,
      const std::vector<FeatureData> &samples,
      std::ifstream & in,
      const size_t n) const noexcept {
    for (size_t i = 0; i < n; i++) {
      TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc, strategy);
      auto [nodes, values] = tree.deserialize(in);
      tree.from_nodes(std::move(nodes),std::move(values));
      accumulator(tree.predict(samples));
    }
  }
  template<typename Accumulator>
  void predict_from_file_acc_mt(
      Accumulator& accumulator,
      const std::vector<FeatureData> &samples,
      std::ifstream & in,
      const size_t n,
      const size_t actual_num_threads) const noexcept {
    threading::ThreadPool pool(actual_num_threads);
    // shared resource to access the file
    threading::SharedResourceWrapper<std::ifstream> in_(std::move(in));
    threading::SharedResourceWrapper<Accumulator> accumulator_(accumulator);
    for (size_t i = 0; i < n; i++) {
      pool.enqueue([this, &accumulator_, &samples, &in_] {
        TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc, strategy);
        auto [nodes, values] = in_.deserialize(tree);
        tree.from_nodes(std::move(nodes), std::move(values));
        accumulator_(tree.predict(samples));
      });
    }
  }
  template<typename Accumulator>
  [[maybe_unused]] void memoryless_predict_st(
      Accumulator& accumulator,
      const std::vector<FeatureData> &train_data,
      const std::vector<FeatureData> &predict_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool resample,
      const size_t bootstrap_size) const noexcept {
    std::vector<size_t> indices(utils::size(train_data[0]));
    std::iota(indices.begin(), indices.end(), 0);
    for (size_t i = 0; i < n_tree; ++i) {
      TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc, strategy);
      if (resample) {
        tree.fit(
            train_data,
            treeson::utils::bootstrap_sample(indices, bootstrap_size, rng),
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
      accumulator(prediction_result);
    }
  }
  template<typename Accumulator>
  [[maybe_unused]] void memoryless_predict_mt(
      Accumulator& accumulator,
      const std::vector<FeatureData> &train_data,
      const std::vector<FeatureData> &predict_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool resample,
      const size_t bootstrap_size,
      const size_t actual_num_threads) const noexcept {
    // do the lazy thing following L'Ecuyer
    // see 'Random Numbers for Parallel Computers: Requirements and Methods, With Emphasis on GPUs'
    // Pierre L’Ecuyer, David Munger, Boris Oreshkin, Richard Simard, p.15
    // 'A single RNG with a “random” seed for each stream.'
    // link: https://www.iro.umontreal.ca/~lecuyer/myftp/papers/parallel-rng-imacs.pdf
    // ensure no data races
    std::vector<size_t> indices(utils::size(train_data[0]));
    std::iota(indices.begin(), indices.end(), 0);
    threading::SharedResourceWrapper<Accumulator> accumulator_(accumulator);
    {
      threading::ThreadPool pool(actual_num_threads);
      for (size_t i = 0; i < n_tree; ++i) {
        const size_t seed = rng();
        pool.enqueue([this, &train_data, &indices, &predict_data, &accumulator_,
                      &nosplit_features, resample, seed, bootstrap_size] {
          RNG rng_(seed);
          TreeType tree(maxDepth, minNodesize, rng_, terminalNodeFunc,
                        strategy);
          if (resample) {
            tree.fit(
                train_data,
                treeson::utils::bootstrap_sample(indices, bootstrap_size, rng_),
                nosplit_features);
          } else {
            // take copy
            std::vector<size_t> indices_ = indices;
            tree.fit(train_data, indices_, nosplit_features);
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
        });
      }
    }
    // move to accumulator that was passed in
    accumulator = std::move(accumulator_.resource);
  }
  // Feature importance method - omission of variable by ignoring split
  template<typename Accumulator, typename Metric, typename Importance>
  [[maybe_unused]] auto feature_importance_omit_st(
      const std::vector<FeatureData> &train_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool oob,
      const size_t bootstrap_size) {
    // we do not allow resampling as such here because it complicates the interface
    // and leaves us in an equal 'in-bag error' situation as if sample size was just set
    // to total size.
    std::vector<size_t> indices = utils::make_index_range(utils::size(train_data[0]));
    /* N.B.: outer scope forces thread-pool to finish - important!!!
     without this we can proceed onto the importance stuff BEFORE we have
     any results, so this MUST be closed off.*/
    std::vector<Accumulator> accumulators(train_data.size() + 1);
    /* N.B.: outer scope forces thread-pool to finish - important!!!
    without this we can proceed onto the importance stuff BEFORE we have
    any results, so this MUST be closed off.*/
    {
      for (size_t i = 0; i < n_tree; ++i) {
          TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc,
                        strategy);
          std::vector<size_t> train_indices, test_indices;
          if (oob) {
            std::tie(train_indices, test_indices) =
                treeson::utils::bootstrap_two_samples(indices, bootstrap_size,
                                                      rng);
          } else {
            train_indices =
                treeson::utils::bootstrap_sample(indices, bootstrap_size, rng);
            test_indices = train_indices;
          }
          tree.fit(train_data, train_indices, nosplit_features);
          if (tree.is_uninformative()) {
            // make it clear that this tree does not count
            continue;
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
      }
    // convert from accumulator values to importances
    Importance importance_;
    std::vector<decltype(importance_(accumulators.front().result(),
                                     accumulators.front().result())
                             )> importances(accumulators.size()-1);
    const auto baseline_results = accumulators.front().result();
    for (size_t feature_i = 0; feature_i < accumulators.size()-1; feature_i++) {
      importances[feature_i] = importance_(
          baseline_results, accumulators[feature_i+1].result());
    }
    return importances;
  }
  template<typename Accumulator, typename Metric, typename Importance>
  [[maybe_unused]] auto feature_importance_omit_mt(
      const std::vector<FeatureData> &train_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool oob = true,
      const size_t bootstrap_size = 0,
      const size_t actual_num_threads = 0) {
    // we do not allow resampling as such here because it complicates the interface
    // and leaves us in an equal 'in-bag error' situation as if sample size was just set
    // to total size.
    std::vector<size_t> indices = utils::make_index_range(utils::size(train_data[0]));
    /* N.B.: outer scope forces thread-pool to finish - important!!!
     without this we can proceed onto the importance stuff BEFORE we have
     any results, so this MUST be closed off.*/
    std::vector<threading::SharedResourceWrapper<Accumulator>>
        accumulators(train_data.size() + 1);
    /* N.B.: outer scope forces thread-pool to finish - important!!!
 without this we can proceed onto the importance stuff BEFORE we have
 any results, so this MUST be closed off.*/
    {
      threading::ThreadPool pool(actual_num_threads);
      for (size_t i = 0; i < n_tree; ++i) {
        const auto seed = rng();
        pool.enqueue([this, &train_data, &accumulators, &indices,
                      &nosplit_features, seed, bootstrap_size, oob] {
          RNG rng_(seed);
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
        });
      }
    }
    // convert from accumulator values to importances
    Importance importance_;
    // whatever the invocation over two accumulators produces :)
    std::vector<decltype(importance_(accumulators.front().resource.result(),
                                     accumulators.front().resource.result())
                             )> importances(accumulators.size()-1);
    const auto baseline_results = accumulators.front().resource.result();
    for (size_t feature_i = 0; feature_i < accumulators.size()-1; feature_i++) {
      importances[feature_i] = importance_(
          baseline_results, accumulators[feature_i+1].resource.result());
    }
    return importances;
  }
  // Feature importance method - importance vs global
  template<typename Accumulator, typename Metric, typename Importance>
  [[maybe_unused]] auto feature_importance_contrast_st(
      const std::vector<FeatureData> &train_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool oob,
      const size_t bootstrap_size) {
    // we do not allow resampling as such here because it complicates the interface
    // and leaves us in an equal 'in-bag error' situation as if sample size was just set
    // to total size.
    std::vector<size_t> indices = utils::make_index_range(utils::size(train_data[0]));
    /* N.B.: outer scope forces thread-pool to finish - important!!!
     without this we can proceed onto the importance stuff BEFORE we have
     any results, so this MUST be closed off.*/
    std::vector<Accumulator> accumulators(train_data.size() + 1);
    /* N.B.: outer scope forces thread-pool to finish - important!!!
    without this we can proceed onto the importance stuff BEFORE we have
    any results, so this MUST be closed off.*/
    {
      for (size_t i = 0; i < n_tree; ++i) {
        TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc,
                      strategy);
        std::vector<size_t> train_indices, test_indices;
        if (oob) {
          std::tie(train_indices, test_indices) =
              treeson::utils::bootstrap_two_samples(indices, bootstrap_size,
                                                    rng);
        } else {
          train_indices =
              treeson::utils::bootstrap_sample(indices, bootstrap_size, rng);
          test_indices = train_indices;
        }
        tree.fit(train_data, train_indices, nosplit_features);
        if (tree.is_uninformative()) {
          // make it clear that this tree does not count
          continue;
        }
        auto used_features = tree.used_features();
        const auto& res = tree.template eval_oob<Metric>(train_data, test_indices);
        // compute baseline - update first accumulator
        accumulators[0](res);
        // update all other accumulators for which this is pertinent
        for (const auto feature : used_features) {
          accumulators[feature + 1](res);
        }
      }
    }
    // convert from accumulator values to importances
    Importance importance_;
    std::vector<decltype(importance_(accumulators.front().result(),
                                     accumulators.front().result())
                             )> importances(accumulators.size()-1);
    const auto baseline_results = accumulators.front().result();
    for (size_t feature_i = 0; feature_i < accumulators.size()-1; feature_i++) {
      importances[feature_i] = importance_(
          baseline_results, accumulators[feature_i+1].result());
    }
    return importances;
  }
  template<typename Accumulator, typename Metric, typename Importance>
  [[maybe_unused]] auto feature_importance_contrast_mt(
      const std::vector<FeatureData> &train_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool oob = true,
      const size_t bootstrap_size = 0,
      const size_t actual_num_threads = 0) {
    // we do not allow resampling as such here because it complicates the interface
    // and leaves us in an equal 'in-bag error' situation as if sample size was just set
    // to total size.
    std::vector<size_t> indices = utils::make_index_range(utils::size(train_data[0]));
    /* N.B.: outer scope forces thread-pool to finish - important!!!
     without this we can proceed onto the importance stuff BEFORE we have
     any results, so this MUST be closed off.*/
    std::vector<threading::SharedResourceWrapper<Accumulator>>
        accumulators(train_data.size() + 1);
    /* N.B.: outer scope forces thread-pool to finish - important!!!
 without this we can proceed onto the importance stuff BEFORE we have
 any results, so this MUST be closed off.*/
    {
      threading::ThreadPool pool(actual_num_threads);
      for (size_t i = 0; i < n_tree; ++i) {
        const auto seed = rng();
        pool.enqueue([this, &train_data, &accumulators, &indices,
                      &nosplit_features, seed, bootstrap_size, oob] {
          RNG rng_(seed);
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
          const auto& res = tree.template eval_oob<Metric>(train_data, test_indices);
          // compute baseline - update first accumulator
          accumulators[0](res);
          // update all other accumulators
          for (const auto feature : used_features) {
            accumulators[feature + 1](res);
          }
        });
      }
    }
    // convert from accumulator values to importances
    Importance importance_;
    // whatever the invocation over two accumulators produces :)
    std::vector<decltype(importance_(accumulators.front().resource.result(),
                                     accumulators.front().resource.result())
                             )> importances(accumulators.size()-1);
    const auto baseline_results = accumulators.front().resource.result();
    for (size_t feature_i = 0; feature_i < accumulators.size()-1; feature_i++) {
      importances[feature_i] = importance_(
          baseline_results, accumulators[feature_i+1].resource.result());
    }
    return importances;
  }
  // Feature importance method - omission of variable by ignoring split
  template<typename Accumulator, typename Metric, typename Importance>
  [[maybe_unused]] auto feature_importance_omit_st(
      Metric& metric,
      const std::vector<FeatureData> &train_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool oob,
      const size_t bootstrap_size) {
    // we do not allow resampling as such here because it complicates the interface
    // and leaves us in an equal 'in-bag error' situation as if sample size was just set
    // to total size.
    std::vector<size_t> indices = utils::make_index_range(utils::size(train_data[0]));
    /* N.B.: outer scope forces thread-pool to finish - important!!!
     without this we can proceed onto the importance stuff BEFORE we have
     any results, so this MUST be closed off.*/
    std::vector<Accumulator> accumulators(train_data.size() + 1);
    /* N.B.: outer scope forces thread-pool to finish - important!!!
    without this we can proceed onto the importance stuff BEFORE we have
    any results, so this MUST be closed off.*/
    {
      for (size_t i = 0; i < n_tree; ++i) {
        TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc,
                      strategy);
        std::vector<size_t> train_indices, test_indices;
        if (oob) {
          std::tie(train_indices, test_indices) =
              treeson::utils::bootstrap_two_samples(indices, bootstrap_size,
                                                    rng);
        } else {
          train_indices =
              treeson::utils::bootstrap_sample(indices, bootstrap_size, rng);
          test_indices = train_indices;
        }
        tree.fit(train_data, train_indices, nosplit_features);
        if (tree.is_uninformative()) {
          // make it clear that this tree does not count
          continue;
        }
        auto used_features = tree.used_features();
        // compute baseline - update first accumulator
        accumulators[0](tree.eval_oob(train_data, test_indices, metric));
        // update all other accumulators
        for (const auto feature : used_features) {
          accumulators[feature + 1](
              tree.eval_oob(train_data, test_indices, feature, metric));
        }
      }
    }
    // convert from accumulator values to importances
    Importance importance_;
    std::vector<decltype(importance_(accumulators.front().result(),
                                     accumulators.front().result())
                             )> importances(accumulators.size()-1);
    const auto baseline_results = accumulators.front().result();
    for (size_t feature_i = 0; feature_i < accumulators.size()-1; feature_i++) {
      importances[feature_i] = importance_(
          baseline_results, accumulators[feature_i+1].result());
    }
    return importances;
  }
  template<typename Accumulator, typename Metric, typename Importance>
  [[maybe_unused]] auto feature_importance_omit_mt(
      Metric& metric,
      const std::vector<FeatureData> &train_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool oob = true,
      const size_t bootstrap_size = 0,
      const size_t actual_num_threads = 0) {
    // we do not allow resampling as such here because it complicates the interface
    // and leaves us in an equal 'in-bag error' situation as if sample size was just set
    // to total size.
    std::vector<size_t> indices = utils::make_index_range(utils::size(train_data[0]));
    /* N.B.: outer scope forces thread-pool to finish - important!!!
     without this we can proceed onto the importance stuff BEFORE we have
     any results, so this MUST be closed off.*/
    std::vector<threading::SharedResourceWrapper<Accumulator>>
        accumulators(train_data.size() + 1);
    /* N.B.: outer scope forces thread-pool to finish - important!!!
 without this we can proceed onto the importance stuff BEFORE we have
 any results, so this MUST be closed off.*/
    {
      threading::ThreadPool pool(actual_num_threads);
      for (size_t i = 0; i < n_tree; ++i) {
        const auto seed = rng();
        pool.enqueue([this, &train_data, &accumulators, &metric, &indices,
                      &nosplit_features, seed, bootstrap_size, oob] {
          RNG rng_(seed);
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
          accumulators[0](tree.eval_oob(train_data, test_indices, metric));
          // update all other accumulators
          for (const auto feature : used_features) {
            accumulators[feature + 1](tree.eval_oob(
                train_data, test_indices, feature, metric));
          }
        });
      }
    }
    // convert from accumulator values to importances
    Importance importance_;
    // whatever the invocation over two accumulators produces :)
    std::vector<decltype(importance_(accumulators.front().resource.result(),
                                     accumulators.front().resource.result())
                             )> importances(accumulators.size()-1);
    const auto baseline_results = accumulators.front().resource.result();
    for (size_t feature_i = 0; feature_i < accumulators.size()-1; feature_i++) {
      importances[feature_i] = importance_(
          baseline_results, accumulators[feature_i+1].resource.result());
    }
    return importances;
  }
  // Feature importance method - importance vs global
  template<typename Accumulator, typename Metric, typename Importance>
  [[maybe_unused]] auto feature_importance_contrast_st(
      Metric& metric,
      const std::vector<FeatureData> &train_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool oob,
      const size_t bootstrap_size) {
    // we do not allow resampling as such here because it complicates the interface
    // and leaves us in an equal 'in-bag error' situation as if sample size was just set
    // to total size.
    std::vector<size_t> indices = utils::make_index_range(utils::size(train_data[0]));
    /* N.B.: outer scope forces thread-pool to finish - important!!!
     without this we can proceed onto the importance stuff BEFORE we have
     any results, so this MUST be closed off.*/
    std::vector<Accumulator> accumulators(train_data.size() + 1);
    /* N.B.: outer scope forces thread-pool to finish - important!!!
    without this we can proceed onto the importance stuff BEFORE we have
    any results, so this MUST be closed off.*/
    {
      for (size_t i = 0; i < n_tree; ++i) {
        TreeType tree(maxDepth, minNodesize, rng, terminalNodeFunc,
                      strategy);
        std::vector<size_t> train_indices, test_indices;
        if (oob) {
          std::tie(train_indices, test_indices) =
              treeson::utils::bootstrap_two_samples(indices, bootstrap_size,
                                                    rng);
        } else {
          train_indices =
              treeson::utils::bootstrap_sample(indices, bootstrap_size, rng);
          test_indices = train_indices;
        }
        tree.fit(train_data, train_indices, nosplit_features);
        if (tree.is_uninformative()) {
          // make it clear that this tree does not count
          continue;
        }
        auto used_features = tree.used_features();
        const auto& res = tree.eval_oob(train_data, test_indices, metric);
        // compute baseline - update first accumulator
        accumulators[0](res);
        // update all other accumulators for which this is pertinent
        for (const auto feature : used_features) {
          accumulators[feature + 1](res);
        }
      }
    }
    // convert from accumulator values to importances
    Importance importance_;
    std::vector<decltype(importance_(accumulators.front().result(),
                                     accumulators.front().result())
                             )> importances(accumulators.size()-1);
    const auto baseline_results = accumulators.front().result();
    for (size_t feature_i = 0; feature_i < accumulators.size()-1; feature_i++) {
      importances[feature_i] = importance_(
          baseline_results, accumulators[feature_i+1].result());
    }
    return importances;
  }
  template<typename Accumulator, typename Metric, typename Importance>
  [[maybe_unused]] auto feature_importance_contrast_mt(
      Metric& metric,
      const std::vector<FeatureData> &train_data,
      const size_t n_tree,
      const std::vector<size_t> &nosplit_features,
      const bool oob = true,
      const size_t bootstrap_size = 0,
      const size_t actual_num_threads = 0) {
    // we do not allow resampling as such here because it complicates the interface
    // and leaves us in an equal 'in-bag error' situation as if sample size was just set
    // to total size.
    std::vector<size_t> indices = utils::make_index_range(utils::size(train_data[0]));
    /* N.B.: outer scope forces thread-pool to finish - important!!!
     without this we can proceed onto the importance stuff BEFORE we have
     any results, so this MUST be closed off.*/
    std::vector<threading::SharedResourceWrapper<Accumulator>>
        accumulators(train_data.size() + 1);
    /* N.B.: outer scope forces thread-pool to finish - important!!!
 without this we can proceed onto the importance stuff BEFORE we have
 any results, so this MUST be closed off.*/
    {
      threading::ThreadPool pool(actual_num_threads);
      for (size_t i = 0; i < n_tree; ++i) {
        const auto seed = rng();
        pool.enqueue([this, &train_data, &accumulators, &metric, &indices,
                      &nosplit_features, seed, bootstrap_size, oob] {
          RNG rng_(seed);
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
          const auto& res = tree.eval_oob(train_data, test_indices, metric);
          // compute baseline - update first accumulator
          accumulators[0](res);
          // update all other accumulators
          for (const auto feature : used_features) {
            accumulators[feature + 1](res);
          }
        });
      }
    }
    // convert from accumulator values to importances
    Importance importance_;
    // whatever the invocation over two accumulators produces :)
    std::vector<decltype(importance_(accumulators.front().resource.result(),
                                     accumulators.front().resource.result())
                             )> importances(accumulators.size()-1);
    const auto baseline_results = accumulators.front().resource.result();
    for (size_t feature_i = 0; feature_i < accumulators.size()-1; feature_i++) {
      importances[feature_i] = importance_(
          baseline_results, accumulators[feature_i+1].resource.result());
    }
    return importances;
  }
  const size_t maxDepth, minNodesize;
  RNG& rng;
  ResultF& terminalNodeFunc;
  SplitStrategy& strategy;
  std::vector<TreeType> trees;
};
// TODO(JSzitas): implement gradient boosting
template <typename ResultF, typename RNG, typename SplitStrategy,
          const size_t MaxCategoricalSet = 32, typename scalar_t = double,
          typename integral_t = size_t>
class [[maybe_unused]] Treeson {
public:
  using FeatureData = std::variant<std::vector<integral_t>, std::vector<scalar_t>>;
  using TreeType = RandomTree<ResultF, RNG, SplitStrategy, MaxCategoricalSet,
                              scalar_t, integral_t>;
  using ResultType = decltype(std::declval<ResultF>()(
      std::declval<std::vector<size_t> &>(), std::declval<size_t>(),
      std::declval<size_t>(),
      std::declval<const std::vector<FeatureData> &>()));

  [[maybe_unused]] Treeson(const size_t n_estimators, const scalar_t learning_rate)
      : n_estimators(n_estimators), learning_rate(learning_rate) {}

  [[maybe_unused]] void fit(const std::vector<FeatureData>& data, std::vector<scalar_t> targets) {
    /*
    tree.fit(data, indices, std::vector<size_t>{});
    const auto& predictions = tree.predict(data);
    */


    std::vector<ResultType> predictions(targets.size(), 0.0);
    residuals = targets;  // Initialize residuals with actual targets

    for (size_t i = 0; i < n_estimators; ++i) {
      TreeType tree;

      tree.fit(data, residuals);  // Fit tree on residuals
      trees.push_back(std::move(tree));

      // Update predictions and residuals
      auto current_predictions = trees.back().predict(data);
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
  const size_t n_estimators;
  const scalar_t learning_rate;
  std::vector<TreeType> trees;
  std::vector<ResultType> residuals;
};
}

