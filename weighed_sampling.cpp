//
// Created by jsco on 10/24/24.
//
#include "map"

#include <cassert>
#include <execution>

#include "stopwatch.h"
#include "treeson.h"


template<typename T>class SampleInfo {
public:
  size_t idx, level, idx_in_level;
  T weight;
  SampleInfo(size_t idx, size_t level, size_t idx_in_level, T weight)
      : idx(idx), level(level), idx_in_level(idx_in_level), weight(weight) {}
};

template<typename WtType> class [[maybe_unused]] FastSampler {
private:

  WtType total_weight;
  size_t n_entries, top_level, bottom_level, n_levels;
  std::vector<WtType> weights, level_weights, level_max;
  std::vector<std::vector<size_t>> level_buckets;


public:
  [[maybe_unused]] explicit FastSampler(
      size_t max_entries, WtType max_value=100.0, WtType min_value=1.0)
      : total_weight(0.0), n_entries(0) {

    top_level = static_cast<size_t>(std::floor(std::log2(max_value)) + 1);
    bottom_level = static_cast<size_t>(std::floor(std::log2(min_value)) + 1);
    n_levels = 1 + top_level - bottom_level;

    weights.resize(max_entries, 0.0);
    level_weights.resize(n_levels, 0.0);
    level_max.resize(n_levels);
    level_buckets = std::vector<std::vector<size_t>>(n_levels, std::vector<size_t>());
    for (size_t i = 0; i < n_levels; ++i) {
      level_max[i] = std::pow(2, top_level - i);
    }
  }

  [[maybe_unused]] void add(const size_t idx, const WtType weight) {
    n_entries++;
    total_weight += weight;
    weights[idx] = weight;

    auto raw_level = static_cast<size_t>(std::floor(std::log2(weight)) + 1);
    size_t level = top_level - raw_level;

    level_weights[level] += weight;
    level_buckets[level].push_back(idx);
  }

private:
  template<typename RNG> WtType smpl_real_unif(RNG& rng) {
    return std::uniform_real_distribution<WtType>(0.0, 1.0)(rng);
  }

  template<typename RNG> SampleInfo<WtType> _sample(RNG& rng) {
    WtType u = smpl_real_unif(rng) * total_weight;

    WtType cumulative_weight = 0.0;
    size_t level = 0;
    for (; level < n_levels; ++level) {
      cumulative_weight += level_weights[level];
      if (u < cumulative_weight) break;
    }

    size_t level_size = level_buckets[level].size();
    WtType level_max_val = level_max[level];
    while (true) {
      WtType inner_sample = smpl_real_unif(rng) * level_size;
      auto idx_in_level = static_cast<size_t>(std::floor(inner_sample));

      size_t idx = level_buckets[level][idx_in_level];
      WtType idx_weight = weights[idx];

      WtType u_lvl = level_max_val * (inner_sample - std::floor(inner_sample));
      if (u_lvl <= idx_weight) {
        return SampleInfo(idx, level, idx_in_level, idx_weight);
      }
    }
  }

public:
  template<typename RNG> [[maybe_unused]] size_t sample(RNG& rng) {
    return _sample(rng).idx;
  }
  template<typename RNG> [[maybe_unused]] size_t sampleAndRemove(RNG& rng) {
    SampleInfo s = _sample(rng);
    total_weight -= s.weight;
    level_weights[s.level] -= s.weight;

    int swap_idx = level_buckets[s.level].back();
    level_buckets[s.level].pop_back();
    if (s.idx_in_level < level_buckets[s.level].size()) {
      level_buckets[s.level][s.idx_in_level] = swap_idx;
    }
    n_entries--;
    return s.idx;
  }
  [[maybe_unused]] WtType getWeight(size_t idx) const {
    return weights[idx];
  }
};
class AliasMethod {
private:
  std::vector<float> probabilities;
  std::vector<size_t> aliases;
  std::mt19937 rng;
  std::uniform_real_distribution<float> dist;

public:
  AliasMethod(const std::vector<float>& weights)
      : rng(std::random_device{}()), dist(0.0, 1.0) {

    std::size_t n = weights.size();
    std::vector<float> prob(n);
    probabilities.resize(n);
    aliases.resize(n);

    // Normalize the weights
    float sum = 0.0;
    for (float weight : weights) {
      sum += weight;
    }

    std::vector<float> normalized_weights(weights.size());
    for (std::size_t i = 0; i < weights.size(); ++i) {
      normalized_weights[i] = weights[i] * n / sum;
    }

    // Auxiliary lists
    std::vector<size_t> small;
    std::vector<size_t> large;

    for (std::size_t i = 0; i < n; ++i) {
      if (normalized_weights[i] < 1.0) {
        small.push_back(i);
      } else {
        large.push_back(i);
      }
    }

    // Construct the probability and alias tables
    while (!small.empty() && !large.empty()) {
      std::size_t small_idx = small.back();
      small.pop_back();
      std::size_t large_idx = large.back();
      large.pop_back();

      probabilities[small_idx] = normalized_weights[small_idx];
      aliases[small_idx] = large_idx;

      normalized_weights[large_idx] = (normalized_weights[large_idx] + normalized_weights[small_idx]) - 1.0f;

      if (normalized_weights[large_idx] < 1.0) {
        small.push_back(large_idx);
      } else {
        large.push_back(large_idx);
      }
    }

    while (!large.empty()) {
      std::size_t large_idx = large.back();
      large.pop_back();
      probabilities[large_idx] = 1.0;
    }

    while (!small.empty()) {
      std::size_t small_idx = small.back();
      small.pop_back();
      probabilities[small_idx] = 1.0;
    }
  }

  size_t sample() {
    // Generate a uniform random integer from 0 to n - 1
    std::uniform_int_distribution<size_t> uni_idx(0, probabilities.size() - 1);
    size_t idx = uni_idx(rng);

    // Generate a uniform random number in [0, 1)
    float probability = dist(rng);

    // Return either the index or its alias based on the generated probability
    return (probability < probabilities[idx]) ? idx : aliases[idx];
  }
};

template<const size_t Setting>
void sampler_test(const size_t n,
                  const size_t draws) {
  std::vector<size_t> data;
  data.reserve(n);
  std::vector<float> weights;
  weights.reserve(n);

  for(size_t i = 0; i < n; i++) {
    data.emplace_back(i+1);
    weights.emplace_back(static_cast<float>(n-i));
  }
  treeson::utils::LandmarkSampler<size_t, float, Setting> sampler(
      std::move(data), std::move(weights));
  std::mt19937 rng(22);
  sampler.sample(rng);
  for(size_t l = 0; l < draws; l++) {
    sampler.sample(rng);//counts[sampler.sample(rng)]++;
  }
}

template<const size_t Setting> struct PrereservedSampler{
  PrereservedSampler(const size_t n) :
    sampler(treeson::utils::LandmarkSampler<size_t, float, Setting> ({}, {})),
    rng(std::mt19937(22)){
    std::vector<size_t> data;
    std::vector<float> weights;
    data.reserve(n);
    weights.reserve(n);

    for(size_t i = 0; i < n; i++) {
      data.emplace_back(i+1);
      weights.emplace_back(static_cast<float>(n-i));
    }
    sampler = treeson::utils::LandmarkSampler<size_t, float, Setting> (
        std::move(data), std::move(weights));
  }
  void sampler_test(const size_t draws) {
    for(size_t l = 0; l < draws; l++) {
      sampler.sample(rng);//counts[sampler.sample(rng)]++;
    }
  }
  auto sample() {
    return sampler.sample(rng);
  }
  void update(const size_t index, const float weight) {
    sampler.update_weight(index, weight);
  }
  treeson::utils::LandmarkSampler<size_t, float, Setting> sampler;
  std::mt19937 rng;
};

void fast_sampler_test(const size_t n,
                       const size_t draws) {
    FastSampler fst_smplr(n, static_cast<double>(n), 1.);
    for(size_t i = 0; i < n; i++) {
      fst_smplr.add(i, n-i);
    }
    std::mt19937 rng(22);
    fst_smplr.sample(rng);
    for(size_t l = 0; l < draws; l++) {
      fst_smplr.sample(rng);
    }
}

struct FastPrereservedSampler{
  FastPrereservedSampler(const size_t n) :
    sampler(FastSampler<double>(n, static_cast<double>(n), 1.)),
    rng(std::mt19937(22)) {
    for(size_t i = 0; i < n; i++) {
      sampler.add(i, n-i);
    }
  }
  void sampler_test(const size_t draws) {
    for(size_t l = 0; l < draws; l++) {
      sampler.sample(rng);//counts[sampler.sample(rng)]++;
    }
  }
  FastSampler<double> sampler;
  std::mt19937 rng;
};

void alias_test(const size_t n, const size_t draws){
  std::vector<float> weights;
  weights.reserve(n);
  for(size_t i = n; i > 0; i--) {
    weights.emplace_back(i);
  }
  AliasMethod alias_method(weights);
  for(size_t i = 0; i < draws; i++) {
    alias_method.sample();
  }
}

struct AliasPrebuilt{
  AliasMethod alias;
  AliasPrebuilt(const size_t n) : alias(AliasMethod({})){
    std::vector<float> weights;
    weights.reserve(n);
    for(size_t i = n; i > 0; i--) {
      weights.emplace_back(i);
    }
    alias = AliasMethod(weights);
  }
  void sampler_test(const size_t draws) {
    for(size_t l = 0; l < draws; l++) {
      alias.sample();//counts[sampler.sample(rng)]++;
    }
  }
};
template<typename T, size_t SMALL>
class short_vector {
private:
  T* data;
  T stack_data[SMALL];
  size_t capacity;
  bool use_heap;

public:
  explicit short_vector(size_t size) {
    if (size > SMALL) {
      data = new T[size];
      use_heap = true;
    } else {
      data = stack_data;
      use_heap = false;
    }
    capacity = size;
  }

  ~short_vector() {
    if (use_heap) {
      delete[] data;
    }
  }

  T* begin() { return data; }
  T* end() { return data + capacity; }
  T& operator[](size_t index) {
    assert(index < capacity);
    return data[index];
  }
  size_t size() const { return capacity; }
};

double unif_rand() {
  static std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng);
}

auto ProbSampleReplace2(int n, std::vector<double>p)
{
  std::vector<size_t> perm(n);
  std::iota(perm.begin(), perm.end(), 1);
  double rU;
  int i, j;
  int nm1 = n - 1;

  /* sort the probabilities into descending order */
  std::sort(std::execution::par_unseq, p.begin(), p.end(), [&](size_t a, size_t b){
    return perm[a] > perm[b];
  });

  /* compute cumulative probabilities */
  std::accumulate(p.begin()+1, p.end(), *p.begin());
  //for (i = 1 ; i < n; i++)
    //p[i] += p[i - 1];

  std::vector<size_t> result(n);
  /* compute the sample */
  for (i = 0; i < n; i++) {
    rU = unif_rand();
    for (j = 0; j < nm1; j++) {
      if (rU <= p[j])
        break;
    }
    result[i] = perm[j];
  }
  return result;
}


template<size_t SMALL = 10000>
void ProbSampleReplace(size_t n, std::vector<double>& p,
                       std::vector<size_t>& a,
                       size_t nans, std::vector<size_t>& ans) {
  // Rescale probabilities to sum to 1
  double sum_p = std::accumulate(p.begin(), p.end(), 0.0);
  for (double& val : p) {
    val /= sum_p;
  }
  ShortVector<double, SMALL> q(n);
  ShortVector<size_t, SMALL> HL(n);
  double rU;
  size_t i, j, k;

  auto H = HL.begin() - 1;
  auto L = HL.begin() + n;

  for (i = 0; i < n; i++) {
    q[i] = p[i] * n;
    if (q[i] < 1.0)
      *++H = i;
    else
      *--L = i;
  }

  if (H >= HL.begin() && L < HL.begin() + n) { // Some q[i] are >= 1 and some < 1
    for (k = 0; k < n - 1; k++) {
      i = HL[k];
      j = *L;
      a[i] = j;
      q[j] += q[i] - 1.0;
      if (q[j] < 1.0)
        L++;
      if (L >= HL.begin() + n)
        break; // Now all are >= 1
    }
  }
  for (i = 0; i < n; i++)
    q[i] += i;

  // Generate sample
  for (i = 0; i < nans; i++) {
    rU = unif_rand() * n;
    k = (size_t) rU;
    ans[i] = (rU < q[k]) ? k : a[k];
  }
}

void rwalker_sampler_test(const size_t n, const size_t draws) {

  std::vector<double> probabilities;
  probabilities.reserve(n);

  for(size_t i = 0; i < n; i++) {
    probabilities.emplace_back(static_cast<float>(n-i));
  }
  std::vector<size_t> ans(draws);
  std::vector<size_t> a(n);
  ProbSampleReplace<10000>(n, probabilities, a, draws, ans);
}


int main() {
/*
  size_t n = 10000;
  std::vector<double> probabilities = {10,9,8,7,6,5,4,3,2,1};
  size_t nans = 10000;
  std::vector<size_t> ans(nans);
  std::vector<size_t> a(n);

  ProbSampleReplace<10000>(n, probabilities, a, nans, ans);

  // Print the generated samples
  for(int i = 0; i < nans; i++) {
    std::cout << ans[i] << " ";
  }
  std::cout << std::endl;

  std::map<size_t, size_t> sample_counts;
  for(size_t i = 0; i < nans; i++) {
    sample_counts[ans[i]]++;
  }
  // Print the sample counts
  for(const auto& pair : sample_counts) {
    std::cout << "Element " << pair.first << " count: " << pair.second << std::endl;
  }
*/

  const size_t n = 10;
  const size_t draws = 100000;
  const size_t replications = 1000;
  std::map<size_t, size_t> counts, counts2;
  PrereservedSampler<4> sampler(n);
  // draw 10000 samples, then change weights
  for(size_t i = 0; i< draws; i++) {
    counts[sampler.sample()]++;
  }
  for (const auto& val : counts) {
    // if(val.first > 10 && val.first < n - 10) continue;
    std::cout << "val: " << val.first << " count: " << val.second << std::endl;
  }

  for(size_t i = 0; i < n; i++) {
    sampler.update(i, static_cast<float>(i+1));
  }
  for(size_t i = 0; i< draws; i++) {
    counts2[sampler.sample()]++;
  }

  for (const auto& val : counts2) {
    // if(val.first > 10 && val.first < n - 10) continue;
    std::cout << "val: " << val.first << " count: " << val.second << std::endl;
  }




      /*
  {
    Stopwatch sw;
    for(size_t k = 0; k < replications; k++) {
      sampler_test<8>(n, draws);
    }
    std::cout << "Type: " << 8 << " time total:" << sw() << std::endl;
  }
  {
    Stopwatch sw;
    for(size_t k = 0; k < replications; k++) {
      sampler_test<16>(n, draws);
    }
    std::cout << "Type: " << 16 << " time total:" << sw() << std::endl;
  }
  {
    Stopwatch sw;
    for(size_t k = 0; k < replications; k++) {
      sampler_test<32>(n, draws);
    }
    std::cout << "Type: " << 32 << " time total:" << sw() << std::endl;
  }
  {
    Stopwatch sw;
    for(size_t k = 0; k < replications; k++) {
      sampler_test<64>(n, draws);
    }
    std::cout << "Type: " << 64 << " time total:" << sw() << std::endl;
  }
  {
    Stopwatch sw;
    for(size_t k = 0; k < replications; k++) {
      sampler_test<128>(n, draws);
    }
    std::cout << "Type: " << 128 << " time total:" << sw() << std::endl;
  }
  {
    PrereservedSampler<64> smplr(n);
    {
      Stopwatch sw;
      for(size_t k = 0; k < replications; k++) {
        smplr.sampler_test(draws);
      }
      std::cout << "Type: " << 64 << " time total:" << sw() << std::endl;
    }
  }
  {
    PrereservedSampler<128> smplr(n);
    {
      Stopwatch sw;
      for(size_t k = 0; k < replications; k++) {
        smplr.sampler_test(draws);
      }
      std::cout << "Type: " << 128 << " time total:" << sw() << std::endl;
    }
  }
  {
    PrereservedSampler<16> smplr(n);
    {
      Stopwatch sw;
      for(size_t k = 0; k < replications; k++) {
        smplr.sampler_test(draws);
      }
      std::cout << "Type: " << 16<< std::endl;//<< " time total:" << sw() << std::endl;
    }
  }
  {
    Stopwatch sw;
    for(size_t k = 0; k < replications; k++) {
      fast_sampler_test(n, draws);
    }
    std::cout << "Type: fast" << std::endl;// | time total:" << sw() << std::endl;
  }
  {
    FastPrereservedSampler smplr(n);
    {
      Stopwatch sw;
      for (size_t k = 0; k < replications; k++) {
        smplr.sampler_test(draws);
      }
      std::cout << "Type: fast, prereserved" << std::endl;//| time total:" << sw() << std::endl;
    }
  }
  {
    Stopwatch sw;
    for (size_t k = 0; k < replications; k++) {
      alias_test(n, draws);
    }
    std::cout << "Type: alias"<< std::endl;// | time total:" << sw() << std::endl;
  }
  {
    AliasPrebuilt smplr(n);
    {
      Stopwatch sw;
      for (size_t k = 0; k < replications; k++) {
        smplr.sampler_test(draws);
      }
      std::cout << "Type: alias, prereserved"<< std::endl;// | time total:" << sw() << std::endl;
    }
  }
  {
    Stopwatch sw;
    for (size_t k = 0; k < replications; k++) {
      rwalker_sampler_test(n, draws);
    }
    std::cout << "Type: rwalker"<< std::endl;// | time total:" << sw() << std::endl;
  }
  {
    Stopwatch sw;
    for(size_t k = 0; k < replications; k++) {
      std::vector<double> weights;
      weights.reserve(n);
      for(size_t i = n; i > 0; i--) {
        weights.emplace_back(i);
      }
      ProbSampleReplace2(draws, weights);
    }
    std::cout << "Type: r"<< std::endl;// | time total:" << sw() << std::endl;
  }*/


  /*
  treeson::utils::FastSampler fst_smplr(n, static_cast<double>(n), 1.);
  for(size_t i = 0; i < n; i++) {
    fst_smplr.add(i, n-i);
  }
  std::mt19937 rng(22);
  std::map<size_t, size_t> counts;

  for(size_t l = 0; l < draws; l++) {
    counts[fst_smplr.sample(rng)]++;
  }
  for (const auto& val : counts) {
    if(val.first > 10 && val.first < n - 10) continue;
    std::cout << "val: " << val.first << " count: " << val.second << std::endl;
  }*/
  return 0;
}
