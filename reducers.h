#ifndef TREESON_REDUCERS_H
#define TREESON_REDUCERS_H

#include <vector>
#include <algorithm>
#include <unordered_set>
#include <random>
#include <iostream>

template<typename ResultType>
class IntersectionSampler {
public:
  // Sample trees from prediction results
  static std::vector<ResultType> sample_trees(
      const std::vector<std::vector<ResultType>>& predictions_forest,
      const size_t num_trees, std::mt19937& rng) {
    std::uniform_int_distribution<size_t> dist(0, predictions_forest.size() - 1);
    std::vector<ResultType> sampled_trees;

    for (size_t i = 0; i < num_trees; ++i) {
      size_t idx = dist(rng);
      sampled_trees.insert(sampled_trees.end(),
                           predictions_forest[idx].begin(),
                           predictions_forest[idx].end());
    }

    return sampled_trees;
  }

  // Compute intersections until size is below the given threshold
  static std::unordered_set<ResultType> compute_intersections(
      const std::vector<ResultType>& sampled_trees, size_t threshold) {
    if (sampled_trees.empty()) {
      return {};
    }

    // Convert to sets for intersections
    std::vector<std::unordered_set<ResultType>> sets;
    for (const auto& res : sampled_trees) {
      std::unordered_set<ResultType> set(res.begin(), res.end());
      sets.push_back(std::move(set));
    }

    // Compute intersection
    std::unordered_set<ResultType> intersection = sets[0];
    for (const auto& set : sets) {
      std::unordered_set<ResultType> temp_intersection;
      for (const auto& elem : intersection) {
        if (set.find(elem) != set.end()) {
          temp_intersection.insert(elem);
        }
      }
      std::swap(intersection, temp_intersection);
      if (intersection.size() < threshold) {
        break;
      }
    }
    return intersection;
  }
};
//TODO: This should be trivial to vectorise
template<typename scalar_t> struct WelfordMean{
  scalar_t mean = 0.0;
  size_t i = 1;
  WelfordMean() : mean(0.0), i(1) {}
  void operator()(const scalar_t x) {
    mean += (x - mean)/static_cast<scalar_t>(i++);
  }
  scalar_t result() const {
    return mean;
  }
  size_t index() const {
    return i;
  }
};
template<typename scalar_t> struct MultitargetMeanReducer {
  std::vector<WelfordMean<scalar_t>> targets;
  explicit MultitargetMeanReducer(const size_t n_targets, const size_t n_preds) {
    targets = std::vector<WelfordMean<scalar_t>>(n_targets*n_preds);
  }
  void operator()(const std::vector<scalar_t>& predictions) {
    size_t i = 0;
    for(const auto & val: predictions) {
      //std::cout << i++ << ',';
      targets[i++](val);
    }
    //std::cout << "Targets size: "<< targets.size() << " i: "<< i;
  }
  std::vector<scalar_t> result() const {
    std::vector<scalar_t> res;
    for(const auto& target : targets) {
      res.push_back(target.result());
    }
    return res;
  }
  std::vector<size_t> index() const {
    std::vector<size_t> res;
    for(const auto& target : targets) {
      res.push_back(target.index());
    }
    return res;
  }
};
#endif // TREESON_REDUCERS_H
