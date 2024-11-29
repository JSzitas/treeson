#ifndef TREESON_MONDRIAN_DEBIASING_H
#define TREESON_MONDRIAN_DEBIASING_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <array>

#include "tinyqr.h"

namespace debiasing {
namespace internal{
template <typename T> constexpr T cpow(T base, int exp = 2) {
  if (exp == 0)
    return 1.0; // base case for zero exponent
  if (exp < 0)
    return 1.0 / cpow(base, -exp);
  T result = 1.0;
  while (exp > 0) {
    if (exp % 2 == 1) {
      result *= base;
    }
    base *= base;
    exp /= 2;
  }
  return result;
}
template<typename T> constexpr T cpow2(int n) {
  return (n == 0) ? 1.0 : (n > 0) ? 2.0 * cpow2<T>(n - 1) : 0.5 * cpow2<T>(n + 1);
}
template<typename T> constexpr T cabs(T x) {
  return (x < 0) ? -x : x;
}
template<typename T> constexpr T crichardson_log(T x) {
  return (x - 1) - ((x - 1) * (x - 1) / 2.0) + ((x - 1) * (x - 1) * (x - 1) / 3.0) - ((x - 1) * (x - 1) * (x - 1) * (x - 1) / 4.0);
}
template<typename T> constexpr T cexp(
    T x, T epsilon = 1e-10, const size_t max_iter = 100) {
  T result = 1.0;  // e^0 = 1
  T term = 1.0;    // First term is x^0 / 0! = 1
  for (size_t n = 1; n < max_iter; ++n) {
    term *= x / n;     // Compute the next term in the series
    result += term;    // Add the term to the result
    if(cabs(term) < epsilon) break;
  }
  return result;
}
template<typename T> constexpr T clog(
    T x, T epsilon = 1e-3, const size_t iter = 100) {
  T x_ = x;
  if (x <= 0.0) {
    // Log is not defined for non-positive values.
    return std::numeric_limits<T>::quiet_NaN();
  } else if (x == 1.0) {
    return 0.0;
  }
  else if (x == 2.0) {
    return 0.693147180559945309417232121458; // ln(2)
  }
  T result = 0;
  int exponent = 0;
  // Scale x to the range [1, 2) by dividing by powers of 2
  while (x < 1) {
    x *= 2;
    --exponent;
  }
  while (x >= 2) {
    x /= 2;
    ++exponent;
  }

  // Use a polynomial approximation for the logarithm in the range [1, 2)
  const T log_scale = exponent * 0.6931472;
  T log_x = crichardson_log(x);
  result = log_scale + log_x;
  for(size_t i = 0; i < iter; i++) {
    T exp_result = cexp(result);
    if(cabs(exp_result - x_) < epsilon) break;
    result -= (exp_result - x) / exp_result;
  }
  return result;
}
template <typename scalar_t, const int J>
constexpr std::pair<std::array<scalar_t, J+1>,
                    std::array<std::array<scalar_t, J + 1>, J + 1>>
get_V_omega_inner() {
  std::array<scalar_t, J+1> a = {};
  for (int r = 0; r <= J; ++r) a[r] = cpow(0.95, r);
  std::array<scalar_t, (J+1) * (J+1)> A = {};
  for (int r = 0; r <= J; ++r) {
    for (int s = 0; s <= J; ++s) {
      A[r * (J+1) + s] = cpow(a[r], 2 * s);
    }
  }
  std::array<scalar_t, J+1> e0{};
  e0[0] = 1.0;
  auto coeff = tinyqr::clm<scalar_t, J+1, J+1>(A, e0);
  std::array<std::array<scalar_t, J + 1>, J + 1> V = {};
  for (int r = 0; r <= J; ++r) {
    for (int s = 0; s <= J; ++s) {
      V[r][s] = 2.0 / (3 * a[r]) * (1 - ((a[s]/a[r]) * clog(a[r]/a[s] + 1.)));
    }
  }

  for (int r = 0; r <= J; ++r) {
    for (int s = r+1; s <= J; ++s) {
      V[r][s] += V[s][r];
    }
  }
  for(int r =0; r <= J; r++) {
    V[r][r] = V[r][r] * 2.;
  }
  for (int s = 0; s <= J; ++s) {
    for (int r = s; r <= J; ++r) {
      V[r][s] = V[s][r];
    }
  }
  return {coeff, V};
}
template <typename scalar_t, const int J>
scalar_t get_V_omega(const size_t d) {
  auto [debias_coeffs,V] = get_V_omega_inner<J>();
  scalar_t result = 0.;
  for (int r = 0; r <= J; ++r) {
    for (int s = 0; s <= J; ++s) {
      result += std::pow(V[r][s], static_cast<scalar_t>(d)) * debias_coeffs[r] * debias_coeffs[s];
    }
  }
  return result;
}
template<typename scalar_t, const size_t n>
constexpr std::array<scalar_t, n> make_e0() {
  std::array<scalar_t, n> e0{};
  e0[0] = 1.0;
  return e0;
}
constexpr unsigned long long factorial(unsigned int n) {
  unsigned long long result = 1;
  for (unsigned int i = 2; i <= n; ++i) {
    result *= i;
  }
  return result;
}
template<typename scalar_t, const int debias_order>
std::vector<scalar_t> make_design_matrix_polynomial(
    const std::vector<scalar_t>& X_data,
    const size_t n, const size_t d) {
  const size_t ncol = (2 * debias_order + 4) * d + 1; // Number of columns
  std::vector<scalar_t> design_matrix(n * ncol, 1.0);
  for (size_t j = 0; j < d; ++j) {
    for (size_t s = 1; s <= 2 * debias_order + 4; ++s) {
      for (size_t i = 0; i < n; ++i) {
        design_matrix[(j * (2 * debias_order + 4) + s) * n + i] = std::pow(X_data[j * n + i], s);
      }
    }
  }
  return design_matrix;
}
template<typename scalar_t, const int debias_order = 2>
std::pair<std::vector<scalar_t>, scalar_t>
get_derivative_estimates_polynomial(
    const std::vector<scalar_t>& X_data,
    const size_t n, const size_t d,
    const std::vector<scalar_t>& Y_data) {
  auto design_matrix = make_design_matrix_polynomial<
      scalar_t, debias_order>(X_data, n, d);
  const auto regression_vector = tinyqr::lm(design_matrix, Y_data);
  const size_t ncol = design_matrix.size()/n;
  std::vector<scalar_t> derivative_estimates(n * d, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < d; ++j) {
      scalar_t Xij = X_data[i + (j * n)];
      size_t start_idx = j * (2 * debias_order + 4) +
                         (2 * debias_order + 1) + 1;
      scalar_t derivative_estimate =
          regression_vector[start_idx] +
          regression_vector[start_idx + 1] * Xij +
          regression_vector[start_idx + 2] * std::pow(Xij, 2) / 2.0;
      constexpr int mult = factorial((2 * debias_order) + 2);
      derivative_estimate *= mult;  // factorial(2 * debias_order + 2)
      derivative_estimates[j * n + i] = derivative_estimate;
    }
  }
  // also get variance estimates here
  scalar_t sigma2_hat = 0.0;
  for (size_t i = 0; i < n; ++i) sigma2_hat += Y_data[i] * Y_data[i];

  for (size_t i = 0; i < n; ++i) {
    scalar_t design_times_regression = 0.0;
    for (size_t j = 0; j < ncol; ++j) {
      design_times_regression += design_matrix[j * n + i] * regression_vector[j];
    }
    sigma2_hat -= Y_data[i] * design_times_regression;
  }
  sigma2_hat /= static_cast<scalar_t>(n - ncol);
  return {derivative_estimates, sigma2_hat};
}
}
// these are required to be exported
template<typename scalar_t, const int debias_order>
constexpr std::array<scalar_t, debias_order+1> get_scaling() {
  std::array<scalar_t, debias_order+1> scaling{};
  for (int r = 0; r <= debias_order; ++r) {
    scaling[r] = debiasing::internal::cpow(1.5, r);
  }
  return scaling;
}
template<typename scalar_t, const int debias_order>
constexpr std::array<scalar_t, debias_order+1> get_coeffs() {
  constexpr std::array<scalar_t, debias_order+1> scaling = get_scaling<debias_order>();

  std::array<scalar_t, (debias_order+1) * (debias_order+1)> A{};
  for (int r = 0; r <= debias_order; ++r) {
    for (int s = 0; s <= debias_order; ++s) {
      A[r * (debias_order+1) + s] = cpow(scaling[r], - 2 * s);
    }
  }
  std::array<scalar_t, debias_order+1> e0{};
  e0[0] = 1.0;
  return tinyqr::clm<scalar_t, debias_order+1, debias_order+1>(A, e0);
}
// structure like this because of dependencies
namespace internal{
template <typename scalar_t, const int debias_order>
constexpr scalar_t get_omega_bar() {
  constexpr auto debias_scaling = get_scaling<scalar_t, debias_order>();
  constexpr auto debias_coeffs = get_coeffs<scalar_t, debias_order>();

  scalar_t sum = 0.0;
  for (size_t i = 0; i <= debias_order; ++i) {
    sum += debias_coeffs[i] * cpow(debias_scaling[i], -2 * debias_order - 2);
  }
  return sum;
}
template<typename scalar_t, const size_t n>
constexpr std::array<scalar_t, n*n> make_A() {
  constexpr std::array<scalar_t, n> scaling = get_scaling<n-1>();
  std::array<scalar_t, n * n> A{};
  for (int r = 0; r < n; ++r) {
    for (int s = 0; s < n; ++s) {
      A[r * n + s] = cpow(scaling[r], - 2 * s);
    }
  }
  return A;
}
}
// exported too
template <typename scalar_t, const int debias_order>
scalar_t select_lifetime_polynomial(
    const std::vector<scalar_t>& X_data,
    const std::vector<scalar_t>& Y_data) {
  const size_t nrow = Y_data.size();
  const size_t ncol = X_data.size()/nrow;
  auto [derivative_estimates, sigma2_hat] =
      debiasing::internal::get_derivative_estimates_polynomial<
          scalar_t, debias_order>(X_data, nrow, ncol, Y_data);
  constexpr scalar_t omega_bar = debiasing::internal::get_omega_bar<
      scalar_t, debias_order>();
  constexpr scalar_t numerator_ =
      (4 * debias_order + 4) * cpow(omega_bar,2) /
      debiasing::internal::cpow(debias_order + 2,2);
  scalar_t numerator = numerator_;
  scalar_t temp = 0., numerator_temp = 0.;
  for(size_t i = 0; i < nrow; i++) {
    temp = 0.;
    for(size_t j = 0; j < ncol; j++) {
      temp += derivative_estimates[j*nrow + i];
    }
    numerator_temp += std::pow(temp,2);
  }
  numerator *= numerator_temp;
  const scalar_t V_omega = debiasing::internal::get_V_omega<
      scalar_t, debias_order>(ncol);
  scalar_t denominator = ncol * sigma2_hat * V_omega;
  return std::pow(numerator / denominator, 1.0 / (4 * debias_order + 4 + ncol));
}
}

#endif // TREESON_MONDRIAN_DEBIASING_H
