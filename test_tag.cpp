#include <random>

#include "treeson.h"
#include "stopwatch.h"

int main() {

  using namespace treeson::memory;
  {
    Stopwatch sw;
    MemoryPool<Tagged<std::vector<float>, 1>, Tagged<std::vector<float>, 2>,
               Tagged<std::mt19937, 1>, Tagged<std::mt19937, 2>,
               Tagged<std::normal_distribution<float>, 1>,
               Tagged<std::normal_distribution<float>, 2>>
        pool;
    pool.emplace_back(Tagged<std::vector<float>, 1>({0, 1, 2, 3, 4}));
    pool.emplace_back(Tagged<std::vector<float>, 2>({0, 1, 2, 3, 4}));
    for (auto &val : pool.get<Tagged<std::vector<float>, 2>>()->item) {
      val += 1;
    }

    std::cout << "Tagged ranges: " << std::endl;
    treeson::utils::print_vector<float>(
        pool.get<Tagged<std::vector<float>, 1>>()->item);
    treeson::utils::print_vector<float>(
        pool.get<Tagged<std::vector<float>, 2>>()->item);

    pool.emplace_back(Tagged<std::mt19937, 1>(std::mt19937(22)));
    pool.emplace_back(Tagged<std::mt19937, 2>(std::mt19937(1059)));
    pool.emplace_back(Tagged<std::normal_distribution<float>, 1>(
        std::normal_distribution<float>{0.3, 1.7}));
    pool.emplace_back(Tagged<std::normal_distribution<float>, 2>(
        std::normal_distribution<float>{-1.2, 1.3}));

    {
      treeson::threading::ThreadPool thread_pool(2);
      for (size_t n = 0; n < 1'000'000; n++) {
        thread_pool.enqueue([&]() {
          {
            auto &smpl = pool.get<Tagged<std::mt19937, 1>>()->item;
            const auto res =
                pool.get<Tagged<std::normal_distribution<float>, 1>>()->item(
                    smpl);
            size_t i = 1;
            auto& vec = pool.get<Tagged<std::vector<float>, 1>>()->item;
            for (auto &val : vec) {
              val += (std::pow(res, i) / static_cast<float>(i));
              i++;
            }
          }
        });
        thread_pool.enqueue([&]() {
          {
            auto &smpl = pool.get<Tagged<std::mt19937, 2>>()->item;
            const auto res =
                pool.get<Tagged<std::normal_distribution<float>, 2>>()->item(
                    smpl);
            size_t i = 1;
            auto& vec = pool.get<Tagged<std::vector<float>, 2>>()->item;
            for (auto &val : vec) {
              val += (std::pow(res, i) / static_cast<float>(i));
              if (val > 0)
                val = std::sqrt(val);
              i++;
            }
          }
          });
      }
    }
    treeson::utils::print_vector<float>(
        pool.get<Tagged<std::vector<float>, 1>>()->item);
    treeson::utils::print_vector<float>(
        pool.get<Tagged<std::vector<float>, 2>>()->item);
  }
  {
    Stopwatch sw;
    std::vector<float> one {0, 1, 2, 3, 4};
    std::vector<float> two {0, 1, 2, 3, 4};
    for (auto &val : two) {
      val += 1;
    }

    std::cout << "Untagged ranges: " << std::endl;
    treeson::utils::print_vector<float>(one);
    treeson::utils::print_vector<float>(two);

    std::mt19937 one_rng(22);
    std::mt19937 two_rng(1059);

    std::normal_distribution<float> dist1{0.3, 1.7};
    std::normal_distribution<float> dist2{-1.2, 1.3};

    {
      for (size_t n = 0; n < 1'000'000; n++) {
        {
          const auto res = dist1(one_rng);
          size_t i = 1;
          std::vector<float> new_vec = one;
          for (auto &val : new_vec) {
            val += (std::pow(res, i) / static_cast<float>(i));
            i++;
          }
          one = new_vec;
        }
        {
          const auto res = dist2(two_rng);
          size_t i = 1;
          std::vector<float> new_vec = two;
          for (auto &val : new_vec) {
            val += (std::pow(res, i) / static_cast<float>(i));
            if (val > 0)
              val = std::sqrt(val);
            i++;
          }
          two = new_vec;
        }
      }
    }
    treeson::utils::print_vector<float>(one);
    treeson::utils::print_vector<float>(two);
  }
  return 0;
}
